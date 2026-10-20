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

from gluebind.config.sampling import DEFAULT_WINDOW_MIN_AUTO_OFFSET_NM

BORESCH_DISTRIBUTION_DIRNAME = "boresch_distributions"
BORESCH_DISTRIBUTION_METADATA = "metadata.json"

BORESCH_MAX_RANGE_DEG = 100.0
"""A Boresch DoF sampled over a wider range than this (degrees) during the
equilibration is flagged: the interface may be weakly bound."""
BORESCH_PEAK_MIN_PROMINENCE = 0.1
"""A density peak counts only if its prominence is at least this fraction of the
highest peak."""
BORESCH_PEAK_MIN_SEPARATION_DEG = 20.0
"""Peaks closer than this (degrees) count as one mode (the taller is kept), so
shoulders within a single basin are not read as separate states."""


def auto_window_min(
    equilibrium_nm: float, offset_nm: float, spacing_nm: float
) -> float:
    """The separation ``window_min`` for ``window_min: auto``.

    ``offset_nm`` below the equilibrium separation, rounded *down* to a multiple of
    ``spacing_nm`` so the windows reach at least ``offset_nm`` below equilibrium
    and lie on the SMD snapshot grid. Raises if that is not a positive separation.
    """
    value = math.floor((equilibrium_nm - offset_nm) / spacing_nm + 1e-9) * spacing_nm
    if value <= 0:
        raise ValueError(
            f"window_min 'auto' gives {value:.3f} nm (equilibrium separation "
            f"{equilibrium_nm:.3f} nm minus {offset_nm} nm); set window_min explicitly"
        )
    return round(value, 4)


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
    """Load the equilibration trajectory and calculate all five DoF series, plus
    ``"separation"``: the interface-centroid distance (nm) in every frame."""
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
    result = {dof: dof_timeseries(points, dof) for dof in DOFS}
    # Same centroids as the separation CV (Å -> nm).
    result["separation"] = np.linalg.norm(points["A"] - points["a"], axis=1) / 10.0
    return result


def boresch_dof_diagnostics(values, *, dof: str) -> dict:
    """Peak count and sampled range of one DoF's equilibration distribution.

    ``values`` (rad) should be the analysis values: dihedrals already unwrapped
    around their circular mean (:func:`periodic_image`), so a mode crossing ±π
    stays one mode. Peaks are found in a Gaussian kernel density estimate and
    count only if their prominence is at least
    :data:`BORESCH_PEAK_MIN_PROMINENCE` of the highest peak and they are at least
    :data:`BORESCH_PEAK_MIN_SEPARATION_DEG` apart.

    Returns ``n_peaks``, ``peak_positions_rad`` (dihedrals wrapped to [-π, π]),
    ``min_rad`` and ``max_rad`` (in the analysis frame, so a dihedral's may lie
    outside ±π), ``range_deg`` and ``warnings`` (messages for a multimodal
    distribution and for a range wider than :data:`BORESCH_MAX_RANGE_DEG`).
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    range_deg = math.degrees(hi - lo)
    if values.size < 2 or float(values.std()) < 1e-9:
        positions = [lo]  # a single value: one mode, nothing to estimate
    else:
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        # Pad by 3 kernel widths (Scott's rule, gaussian_kde's default bandwidth).
        pad = 3.0 * values.size ** (-1 / 5) * float(values.std())
        grid = np.linspace(lo - pad, hi + pad, 1024)
        density = kde(grid)
        step = float(grid[1] - grid[0])
        separation = math.radians(BORESCH_PEAK_MIN_SEPARATION_DEG)
        peaks, _ = find_peaks(
            density,
            prominence=BORESCH_PEAK_MIN_PROMINENCE * float(density.max()),
            distance=max(1, round(separation / step)),
        )
        positions = [float(grid[i]) for i in peaks] or [float(grid[density.argmax()])]

    from gluebind.boresch_geometry import DIHEDRAL_DOFS

    if dof in DIHEDRAL_DOFS:  # report dihedral peaks in [-pi, pi]
        positions = [math.atan2(math.sin(p), math.cos(p)) for p in positions]

    warnings: list[str] = []
    if len(positions) > 1:
        shown = ", ".join(f"{math.degrees(p):.0f}" for p in positions)
        warnings.append(
            f"Boresch {dof} has {len(positions)} peaks in its equilibration "
            f"distribution (at {shown} deg): the interface may be weakly bound or "
            "switching between poses"
        )
    if range_deg > BORESCH_MAX_RANGE_DEG:
        warnings.append(
            f"Boresch {dof} sampled {range_deg:.0f} deg during equilibration "
            f"(> {BORESCH_MAX_RANGE_DEG:.0f} deg): the interface may be weakly bound"
        )
    return {
        "n_peaks": len(positions),
        "peak_positions_rad": [round(p, 4) for p in positions],
        "min_rad": round(lo, 4),
        "max_rad": round(hi, 4),
        "range_deg": round(range_deg, 2),
        "warnings": warnings,
    }


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
    in ``analysis_rad``. ``metadata.json`` also records each DoF's
    :func:`boresch_dof_diagnostics` (peak count, sampled range, warnings).
    """
    import numpy as np

    from gluebind.boresch_geometry import DIHEDRAL_DOFS, DOFS

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}
    diagnostics: dict[str, dict] = {}
    frame_count = None
    for dof in DOFS:
        raw = np.asarray(series[dof], dtype=float)
        if frame_count is None:
            frame_count = int(raw.size)
        elif raw.size != frame_count:
            raise ValueError("Boresch DoF distributions have different lengths")
        analysis = periodic_image(raw) if dof in DIHEDRAL_DOFS else raw
        diagnostics[dof] = boresch_dof_diagnostics(analysis, dof=dof)
        for message in diagnostics[dof]["warnings"]:
            print(f"WARNING: {message}", flush=True)
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
        "diagnostics": diagnostics,
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
    report: dict | None = None,
) -> dict[str, list[float]]:
    """Boresch DoF centres (from the equilibration trajectory) + separation centres.

    * **Boresch** — for each of the five DoFs, bin the distribution measured over
      the equilibration trajectory (using the resolved anchors) at the Boresch
      window spacing. Requires ``prepared.complex_trajectory``.
    * **Separation** — from the configured schedule (explicit ``centres`` or
      ``window_min``/``window_max``/``window_spacing``); these are the centres the
      steered MD snapshots. ``window_min: auto`` is resolved here from the mean
      interface-centroid separation over the equilibration trajectory
      (:func:`auto_window_min`); if ``report`` is given, it receives
      ``separation_equilibrium_nm`` and the resolved ``separation_window_min_nm``.

    RMSD stage centres are *not* returned — the runner derives those from the
    sampling schedule directly.
    """
    import numpy as np

    from gluebind.boresch_geometry import DOFS
    from gluebind.runners.window import enumerate_centres

    centres: dict[str, list[float]] = {}
    series = None
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

    schedule = config.sampling.for_cv("separation", "separation")
    if schedule.window_min == "auto":
        if not schedule.window_spacing:
            raise ValueError("window_min 'auto' needs separation.window_spacing")
        if series is None:
            if prepared.complex_trajectory is None:
                raise ValueError(
                    "window_min 'auto' needs the equilibration trajectory; set "
                    "separation.window_min explicitly"
                )
            series = _load_boresch_series(prepared, context)
        equilibrium = float(np.mean(series["separation"]))
        offset = schedule.window_min_auto_offset
        if offset is None:
            offset = DEFAULT_WINDOW_MIN_AUTO_OFFSET_NM
        window_min = auto_window_min(equilibrium, offset, schedule.window_spacing)
        schedule = schedule.model_copy(update={"window_min": window_min})
        if report is not None:
            report["separation_equilibrium_nm"] = round(equilibrium, 4)
            report["separation_window_min_nm"] = window_min
    centres["separation"] = enumerate_centres(schedule)
    return centres
