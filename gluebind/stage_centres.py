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
    values, so the umbrella windows bracket the equilibrium distribution. Note:
    this spans the raw ``[min, max]`` and does not special-case dihedral
    wrap-around at ±π — verify the distributions are unimodal and away from the
    branch cut (they are for stable interface anchors); supply explicit centres
    otherwise.
    """
    import numpy as np

    values = np.asarray(series, dtype=float)
    lo, hi = float(values.min()), float(values.max())
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
            "(prepared.complex_trajectory is None); supply explicit centres via the config"
        )

    traj = mda.Universe(prepared.complex_prm7, prepared.complex_trajectory)
    anchor_atoms = [context.anchors[k] for k in ("b", "c", "B", "C")]
    series = _collect_series(traj, context.rec_group, context.lig_group, anchor_atoms, np)
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

    centres["separation"] = enumerate_centres(config.sampling.for_cv("separation", "separation"))
    return centres
