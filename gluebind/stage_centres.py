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


def boresch_centres_from_series(series, spacing: float) -> list[float]:
    """Window centres (rad) spanning a DoF's observed range at ``spacing``.

    Places a regular grid at ``spacing`` covering ``[min, max]`` of the sampled
    values, so the umbrella windows bracket the equilibrium distribution.

    Guards against a dihedral distribution that straddles the ±π branch cut: there
    the raw ``[min, max]`` spans almost the whole circle even though the sampled
    values are tightly clustered, so the naive grid would place windows over a
    large *unsampled* arc. When detected (the true circular spread is much smaller
    than the raw range) this raises — supply explicit centres for that DoF.
    """
    import numpy as np

    values = np.asarray(series, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    raw_range = hi - lo

    # Circular spread = 2π minus the largest empty arc (interior gaps + the
    # wrap-around gap). If an *interior* gap is the largest, the data wraps the
    # branch cut and the circular spread is smaller than the raw range.
    ordered = np.sort(values)
    if ordered.size >= 2:
        gaps = np.diff(ordered)
        wrap_gap = (ordered[0] + 2 * math.pi) - ordered[-1]
        circular_range = 2 * math.pi - max(float(gaps.max()), float(wrap_gap))
        if raw_range - circular_range > 1e-3:
            raise ValueError(
                f"Boresch DoF distribution appears to straddle the ±π branch cut "
                f"(raw range {raw_range:.2f} rad but circular spread only "
                f"{circular_range:.2f} rad); the naive [min, max] window grid would "
                "cover a large unsampled arc. Supply explicit centres for this DoF."
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
    import MDAnalysis as mda
    import numpy as np

    from gluebind.boresch_geometry import DOFS
    from gluebind.runners.window import enumerate_centres
    from gluebind.selection.anchors import dof_timeseries
    from gluebind.spec_builder import _collect_series

    centres: dict[str, list[float]] = {}

    if prepared.complex_trajectory is None:
        raise ValueError(
            "Boresch window centres need an equilibration trajectory "
            "(prepared.complex_trajectory is None); supply explicit centres via "
            "the config"
        )

    traj = mda.Universe(prepared.complex_prm7, prepared.complex_trajectory)
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
    for dof in DOFS:
        centres[dof] = boresch_centres_from_series(dof_timeseries(points, dof), spacing)

    centres["separation"] = enumerate_centres(
        config.sampling.for_cv("separation", "separation")
    )
    return centres
