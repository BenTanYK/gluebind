"""Derive the Boresch and separation window centres for a calculation.

The runner needs ``stage_centres`` — the window centres for the Boresch DoFs
(from the unrestrained-MD distribution of each angle/dihedral) and for the
separation stage. This module computes them from the prepared system, so the
facade (:meth:`gluebind.runners.calculation.Calculation.prepare`) can wire the
runner from a config alone.

:func:`boresch_centres_from_series` (the binning) is pure and unit-tested;
:func:`compute_stage_centres` reads the equilibration trajectory and is
integration-verified (Phase 7), like the rest of the trajectory analysis.
"""

from __future__ import annotations

import math


def boresch_centres_from_series(
    series, spacing: float, *, periodic: bool = False
) -> list[float]:
    """Window centres (rad) spanning a DoF's observed range at ``spacing``.

    Places a regular grid at ``spacing`` covering ``[min, max]`` of the sampled
    values, so the umbrella windows bracket the equilibrium distribution.

    When ``periodic=True`` (for dihedrals), values are unwrapped around their
    circular mean before generating the grid, so a compact distribution crossing
    ±π remains contiguous. Ordinary angular DoFs retain linear handling.
    """
    import numpy as np

    values = np.asarray(series, dtype=float)
    if periodic:
        reference = math.atan2(
            float(np.sin(values).mean()), float(np.cos(values).mean())
        )
        values = reference + np.arctan2(
            np.sin(values - reference), np.cos(values - reference)
        )
    lo, hi = float(values.min()), float(values.max())
    if not periodic:
        raw_range = hi - lo
        ordered = np.sort(values)
        if ordered.size >= 2:
            gaps = np.diff(ordered)
            wrap_gap = (ordered[0] + 2 * math.pi) - ordered[-1]
            circular_range = 2 * math.pi - max(
                float(gaps.max()), float(wrap_gap)
            )
            if raw_range - circular_range > 1e-3:
                raise ValueError(
                    "Boresch DoF distribution appears to straddle the ±π branch cut "
                    f"(raw range {raw_range:.2f} rad but circular spread only "
                    f"{circular_range:.2f} rad); the naive [min, max] window grid "
                    "would cover a large unsampled arc. Supply explicit centres for "
                    "this DoF."
                )

    start = math.floor(lo / spacing) * spacing
    n = max(1, int(math.ceil((hi - start) / spacing)) + 1)
    return [round(start + i * spacing, 4) for i in range(n)]


def compute_stage_centres(prepared, context, config) -> dict[str, list[float]]:
    """Boresch DoF centres (from the equilibration trajectory) + separation centres.

    * **Boresch** — for each of the five DoFs, bin the distribution measured over
      the equilibration trajectory (using the resolved anchors) at the Boresch
      window spacing. Requires ``prepared.complex_trajectory``.
    * **Separation** — from the configured schedule (explicit ``centres`` or
      ``window_min``/``window_max``/``window_spacing``); these are the centres the
      steered MD snapshots.

    RMSD stage centres are *not* returned — the runner derives those from the
    sampling schedule directly.
    """
    import numpy as np

    from gluebind.boresch_geometry import DOFS
    from gluebind.runners.window import enumerate_centres
    from gluebind.selection.anchors import dof_timeseries
    from gluebind.spec_builder import _collect_series
    from gluebind.system.mdanalysis import load_amber_universe

    centres: dict[str, list[float]] = {}
    configured = config.sampling.boresch.centres
    if isinstance(configured, dict):
        explicit = {
            dof: [round(float(c), 4) for c in values]
            for dof, values in configured.items()
        }
    elif configured is not None:
        explicit = {dof: [round(float(c), 4) for c in configured] for dof in DOFS}
    else:
        explicit = {}
    missing = [dof for dof in DOFS if dof not in explicit]

    if missing:
        if prepared.complex_trajectory is None:
            raise ValueError(
                "Boresch window centres need an equilibration trajectory for "
                f"{', '.join(missing)}; provide explicit centres via the config"
            )
        traj = load_amber_universe(prepared.complex_prm7, prepared.complex_trajectory)
        anchor_atoms = [context.anchors[k] for k in ("b", "c", "B", "C")]
        series = _collect_series(
            traj, context.rec_group, context.lig_group, anchor_atoms, np
        )
        points = {
            "a": series["a"],
            "A": series["A"],
            "b": series[context.anchors["b"]],
            "c": series[context.anchors["c"]],
            "B": series[context.anchors["B"]],
            "C": series[context.anchors["C"]],
        }
        spacing = config.sampling.boresch.window_spacing or 0.1
        for dof in missing:
            centres[dof] = boresch_centres_from_series(
                dof_timeseries(points, dof),
                spacing,
                periodic=dof in ("phiA", "phiB", "phiC"),
            )
    centres.update(explicit)

    centres["separation"] = enumerate_centres(
        config.sampling.for_cv("separation", "separation")
    )
    return centres
