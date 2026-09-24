"""Derive the Boresch and separation window centres for a calculation.

The runner needs ``stage_centres`` — the window centres for the Boresch DoFs
(from the unrestrained-MD distribution of each angle/dihedral) and for the
separation stage. This module computes them from the prepared system, so the
facade (:meth:`gluebind.runners.calculation.Calculation.prepare`) can wire the
runner from a config alone.

:func:`boresch_centres_from_series` (the binning) is pure and unit-tested;
:func:`compute_stage_centres` reads the equilibration trajectory and is
integration-verified, like the rest of the trajectory analysis.
"""

from __future__ import annotations

import json
import math
import pathlib

BORESCH_DISTRIBUTION_DIRNAME = "boresch_distributions"
BORESCH_DISTRIBUTION_METADATA = "metadata.json"


def periodic_image(values):
    """Map angular values to the periodic image around their circular mean."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    reference = math.atan2(
        float(np.sin(values).mean()), float(np.cos(values).mean())
    )
    return reference + np.arctan2(
        np.sin(values - reference), np.cos(values - reference)
    )


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
        values = periodic_image(values)
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


def _load_boresch_series(prepared, context):
    """Load the equilibration trajectory and calculate all five DoF series."""
    import numpy as np

    from gluebind.boresch_geometry import DOFS
    from gluebind.selection.anchors import dof_timeseries
    from gluebind.spec_builder import _collect_series
    from gluebind.system.mdanalysis import load_amber_universe

    if prepared.complex_trajectory is None:
        raise ValueError(
            "Boresch distributions need an equilibration trajectory "
            "(prepared.complex_trajectory is None)"
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
    return {dof: dof_timeseries(points, dof) for dof in DOFS}


def write_boresch_distributions(
    series,
    output_dir: str | pathlib.Path,
    *,
    metadata: dict | None = None,
) -> dict[str, str]:
    """Write raw and periodic-analysis Boresch DoF distributions.

    Each ``<dof>.dat`` contains ``frame``, ``raw_rad`` and ``analysis_rad``
    columns. Dihedrals retain their principal ``[-pi, pi]`` value in
    ``raw_rad`` and use the same circular-mean unwrapping as centre generation
    in ``analysis_rad``.
    """
    import numpy as np

    from gluebind.boresch_geometry import DIHEDRAL_DOFS, DOFS

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}
    frame_count = None
    for dof in DOFS:
        raw = np.asarray(series[dof], dtype=float)
        if frame_count is None:
            frame_count = int(raw.size)
        elif raw.size != frame_count:
            raise ValueError("Boresch DoF distributions have different lengths")
        analysis = periodic_image(raw) if dof in DIHEDRAL_DOFS else raw
        path = output_dir / f"{dof}.dat"
        np.savetxt(
            path,
            np.column_stack((np.arange(raw.size), raw, analysis)),
            fmt=["%d", "%.10f", "%.10f"],
            header="frame raw_rad analysis_rad",
        )
        report[dof] = str(path)

    info = {
        "columns": ["frame", "raw_rad", "analysis_rad"],
        "units": "radians",
        "periodic_dofs": list(DIHEDRAL_DOFS),
        "frame_count": frame_count or 0,
        "files": report,
    }
    if metadata:
        info.update(metadata)
    (output_dir / BORESCH_DISTRIBUTION_METADATA).write_text(
        json.dumps(info, indent=2)
    )
    return report


def compute_stage_centres(
    prepared,
    context,
    config,
    *,
    distributions_dir: str | pathlib.Path | None = None,
    distribution_metadata: dict | None = None,
) -> dict[str, list[float]]:
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
    from gluebind.boresch_geometry import DOFS
    from gluebind.runners.window import enumerate_centres

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
        series = _load_boresch_series(prepared, context)
        if distributions_dir is not None:
            write_boresch_distributions(
                series, distributions_dir, metadata=distribution_metadata
            )
        spacing = config.sampling.boresch.window_spacing or 0.1
        for dof in missing:
            centres[dof] = boresch_centres_from_series(
                series[dof],
                spacing,
                periodic=dof in ("phiA", "phiB", "phiC"),
            )
    elif distributions_dir is not None:
        series = _load_boresch_series(prepared, context)
        write_boresch_distributions(
            series, distributions_dir, metadata=distribution_metadata
        )
    centres.update(explicit)

    centres["separation"] = enumerate_centres(
        config.sampling.for_cv("separation", "separation")
    )
    return centres
