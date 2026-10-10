"""Steered MD to generate separation-window starting frames.

Ports the template's ``SMD.py`` pulling scheme onto the tested restraint builders:
with all RMSD and Boresch restraints in place, the interface-CoM distance is
steered outward by a moving harmonic potential, and a frame is saved whenever the
measured distance first crosses each target window centre. Those frames seed the
separation umbrella-sampling windows.

Like the equilibration stages, the MD itself runs as a **backend job** (never on
the driver): :func:`run_smd` is the self-contained compute entry point (reads an
:class:`SmdSpec` from a working directory), and :func:`make_steered_md_runner`
returns the ``callable(boresch_eq_values)`` the runner invokes — it writes the
spec, submits one job, and waits. The window-target scheduling is pure and
tested; OpenMM/ParmEd are imported lazily inside the MD functions.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import sys
from collections.abc import Callable

import pydantic

SMD_SPEC_FILENAME = "smd.json"
SMD_RESULT_FILENAME = "result.json"
SMD_PERIODIC_IMAGES_FILENAME = "periodic_images.json"

PERIODIC_IMAGE_WARNING_ANGSTROM = 15.0
"""Warn when an SMD frame's solute comes closer than this to its own periodic
image: separation windows started from such frames may include interactions
between the partners and their periodic copies (an under-sized box)."""


def separation_window_targets(centres) -> list[float]:
    """Sorted, de-duplicated window centres (nm) to snapshot during the pull."""
    return sorted({round(float(c), 4) for c in centres})


def smd_pull_plan(
    start_nm: float,
    targets,
    *,
    compression_margin: float,
    pull_margin: float,
    n_pull_increments: int,
) -> tuple[float, float, int]:
    """The steering-centre schedule: compress, then pull the partners apart.

    The centre ``r0`` starts at the measured separation ``start_nm`` and moves
    down by ``step`` per increment, for ``n_compress`` increments, to
    ``compressed`` (the smallest target minus ``compression_margin``). It then
    moves up by ``step`` for ``n_pull_increments`` increments, ending
    ``pull_margin`` beyond the largest target, so every target is crossed on the
    way out. Compression uses the outward rate, and is skipped if the start is
    already at or below ``compressed``.

    Returns ``(compressed, step, n_compress)`` in nm.
    """
    targets = separation_window_targets(targets)
    compressed = min(start_nm, targets[0] - compression_margin)
    step = (targets[-1] + pull_margin - compressed) / n_pull_increments
    n_compress = math.ceil((start_nm - compressed) / step - 1e-9)
    return compressed, step, n_compress


def smd_frame_path(frames_dir: str | pathlib.Path, centre: float) -> pathlib.Path:
    """Path of the SMD snapshot that seeds the separation window at ``centre`` nm."""
    return pathlib.Path(frames_dir) / f"{centre:.4g}nm.rst7"


def missing_frames(frames_dir: str | pathlib.Path, centres) -> list[float]:
    """The ``centres`` (nm) whose SMD snapshot is absent or empty, sorted."""
    return [
        c
        for c in separation_window_targets(centres)
        if not smd_frame_path(frames_dir, c).is_file()
        or smd_frame_path(frames_dir, c).stat().st_size == 0
    ]


def closest_periodic_image(positions, box_vectors, atoms) -> tuple[float, int, int]:
    """Closest approach of ``atoms`` to any of their periodic images.

    ``positions`` is an ``(n_atoms, 3)`` array and ``box_vectors`` the three
    triclinic box vectors (rows), in the same length unit. Returns
    ``(distance, i, j)``: atom ``j`` comes within ``distance`` of a periodic copy
    (a non-zero lattice translation) of atom ``i``. Pairs within the same cell are
    ignored; translations of up to two cells along each box vector are searched.
    """
    import itertools

    import numpy as np
    from scipy.spatial import KDTree

    atoms = np.asarray(atoms, dtype=int)
    xyz = np.asarray(positions, dtype=float)[atoms]
    lattice = np.asarray(box_vectors, dtype=float)
    tree = KDTree(xyz)
    best = (float("inf"), -1, -1)
    for shift in itertools.product(range(-2, 3), repeat=3):
        if not any(shift):
            continue
        distances, nearest = tree.query(xyz + np.asarray(shift) @ lattice)
        k = int(np.argmin(distances))
        if distances[k] < best[0]:
            best = (float(distances[k]), int(atoms[k]), int(atoms[nearest[k]]))
    return best


def periodic_image_warning(
    periodic_images: dict,
    window_centres=(),
    threshold_A: float = PERIODIC_IMAGE_WARNING_ANGSTROM,
) -> str | None:
    """Warning text for SMD frames whose solute is within ``threshold_A`` of its
    periodic image, or ``None`` if every frame is clear.

    ``periodic_images`` is the ``{centre_nm: {distance_A, atoms}}`` report written
    by :func:`run_smd`; ``window_centres`` marks which frames seed separation
    windows in the current schedule (the rest were captured for later windows).
    """
    close = sorted(
        (float(c), float(v["distance_A"]))
        for c, v in periodic_images.items()
        if float(v["distance_A"]) < threshold_A
    )
    if not close:
        return None
    windows = {round(float(c), 4) for c in window_centres}
    used = [c for c, _ in close if round(c, 4) in windows]
    centre, distance = min(close, key=lambda item: item[1])
    message = (
        f"steered MD: the solute comes within {threshold_A:g} A of its own periodic "
        f"image in {len(close)} frame(s) between {close[0][0]:g} and "
        f"{close[-1][0]:g} nm (closest {distance:.1f} A at {centre:g} nm). "
    )
    if used:
        message += (
            f"Separation windows at {', '.join(f'{c:g}' for c in used)} nm start "
            "from these frames and may include periodic-image interactions; "
        )
    else:
        message += "No current separation window starts from these frames; "
    return message + "increase prep.box_padding_angstrom before sampling there."


def smd_snapshot_targets(schedule, window_min: float | None = None) -> list[float]:
    """Dense SMD snapshot grid (nm) from a separation :class:`WindowSampling`.

    Snapshots are saved from ``window_min`` to ``smd_capture_max`` at
    ``smd_snapshot_spacing`` — finer than, and independent of, the US window
    schedule. The US windows are a subset of this grid, so windows can be added
    later (up to ``smd_capture_max``) without re-running steered MD.
    ``window_min`` overrides the schedule's, e.g. with the value ``"auto"``
    resolved to during restraint resolution.
    """
    lo = schedule.window_min if window_min is None else window_min
    if lo == "auto":
        raise ValueError(
            "window_min 'auto' must be resolved (restraint resolution) before the "
            "SMD snapshot grid can be built"
        )
    if (
        lo is None
        or schedule.smd_snapshot_spacing is None
        or schedule.smd_capture_max is None
    ):
        raise ValueError(
            "separation schedule needs window_min, smd_snapshot_spacing and "
            "smd_capture_max to build the SMD snapshot grid"
        )
    hi, step = schedule.smd_capture_max, schedule.smd_snapshot_spacing
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


class SmdSpec(pydantic.BaseModel):
    """Everything one steered-MD run needs, self-contained (serialisable to JSON).

    Carries the restraint geometry (interface groups, Boresch anchors, the RMSD
    regions to hold rigid), the Boresch equilibrium values determined by the
    upstream Boresch stages, the target window centres to snapshot, and the MD
    parameters — so the compute node needs nothing but this file and the
    referenced structures.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    topology: str
    coordinates: str
    out_dir: str
    """Directory the per-centre ``<centre>nm.rst7`` frames are written to (this is
    the ``smd_frames_dir`` the spec builder reads for the separation windows)."""

    rec_group: list[int]
    lig_group: list[int]
    anchors: dict[str, int]
    rmsd_atoms_bound: dict[str, list[int]]
    boresch_eq_values: dict[str, float]
    window_centres: list[float]

    # MD parameters (from the sampling config)
    hmr_factor: float
    pme_cutoff_nm: float
    timestep_fs: float
    temperature_K: float

    # Steered-MD force constants / schedule (the published protocol's values;
    # stiffer than US): kcal/mol/Å² for the separation and RMSD restraints,
    # kcal/mol/rad² for the Boresch restraints.
    k_smd: float = 100.0
    k_rmsd: float = 100.0
    k_boresch: float = 200.0
    smd_compression_margin: float = 0.1
    """Distance (nm) below the smallest snapshot target to compress to, from the
    measured starting separation, before pulling outward."""
    smd_pull_margin: float = 0.5
    """Distance (nm) to steer past the furthest snapshot target so it is reached."""
    total_steps: int = 750_000
    increment_steps: int = 100
    state_data_interval_steps: int = 10000
    """Live progress-report interval, in MD steps."""
    platform: str = "CUDA"

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "SmdSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def smd_launch_command(python: str = "python") -> list[str]:
    """The command a backend runs (inside the SMD work dir) to execute it."""
    code = "from gluebind.simulation.steered_md import run_smd; run_smd('.')"
    return [python, "-c", code]


def make_steered_md_runner(
    *,
    backend,
    scheduler_factory,
    work_dir: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    topology: str,
    coordinates: str,
    rec_group: list[int],
    lig_group: list[int],
    anchors: dict[str, int],
    rmsd_atoms_bound: dict[str, list[int]],
    snapshot_centres,
    sampling,
    platform: str = "CUDA",
    handle_recorder: Callable[[str, str], None] | None = None,
    window_centres=(),
    warn: Callable[[str], None] | None = None,
):
    """Return the ``callable(boresch_eq_values)`` the runner invokes between the
    Boresch and separation stages.

    ``snapshot_centres`` is the dense SMD snapshot grid (from
    :func:`smd_snapshot_targets`) — saved independently of, and finer than, the US
    window schedule. The callable writes an :class:`SmdSpec` into ``work_dir`` and
    submits a single backend job (so the pull runs on a compute node, not the
    driver); the job writes the per-centre frames into ``out_dir``, which the spec
    builder then reads for the separation windows.

    After the job, the callable raises if it produced no result or no frame for
    any of ``window_centres`` (the current separation schedule), so a failed pull
    is never taken as complete. ``warn`` (default: this module's logger) reports
    spare snapshots that were not captured, and frames whose solute comes within
    :data:`PERIODIC_IMAGE_WARNING_ANGSTROM` of its periodic image.
    """
    from gluebind.backend.base import JobSpec, JobState

    work_dir = pathlib.Path(work_dir)
    out_dir = pathlib.Path(out_dir)

    def _generate(boresch_eq_values: dict) -> dict[float, str]:
        work_dir.mkdir(parents=True, exist_ok=True)
        spec = SmdSpec(
            topology=topology,
            coordinates=coordinates,
            out_dir=str(out_dir),
            rec_group=rec_group,
            lig_group=lig_group,
            anchors=anchors,
            rmsd_atoms_bound=rmsd_atoms_bound,
            boresch_eq_values=dict(boresch_eq_values),
            window_centres=separation_window_targets(snapshot_centres),
            hmr_factor=sampling.hmr_factor,
            pme_cutoff_nm=sampling.pme_cutoff_nm,
            timestep_fs=sampling.timestep_fs,
            temperature_K=sampling.temperature_K,
            state_data_interval_steps=sampling.state_data_interval_steps,
            smd_pull_margin=sampling.separation.smd_pull_margin or 0.5,
            smd_compression_margin=(
                sampling.separation.smd_compression_margin
                if sampling.separation.smd_compression_margin is not None
                else 0.1
            ),
            platform=platform,
        )
        spec.dump(work_dir / SMD_SPEC_FILENAME)
        result_path = work_dir / SMD_RESULT_FILENAME
        images_path = work_dir / SMD_PERIODIC_IMAGES_FILENAME
        # A re-run must not be judged by an earlier run's outputs.
        result_path.unlink(missing_ok=True)
        images_path.unlink(missing_ok=True)
        job = JobSpec(
            command=smd_launch_command(), work_dir=str(work_dir), name="steered_md"
        )
        (state,) = scheduler_factory().run(
            [job],
            on_submit=(
                (lambda _index, handle: handle_recorder("steered_md", handle))
                if handle_recorder is not None
                else None
            ),
        )
        if state is not JobState.FINISHED:
            raise RuntimeError(f"steered MD did not finish (state={state})")
        # Slurm reports a crashed job as finished too: judge by the outputs.
        if not result_path.exists():
            raise RuntimeError(
                f"steered MD produced no result; inspect its job log in {work_dir}"
            )
        missing = missing_frames(out_dir, window_centres)
        if missing:
            raise RuntimeError(
                "steered MD produced no starting frame for the separation window(s) "
                f"at {', '.join(f'{c:g}' for c in missing)} nm (the pull may not "
                f"have reached them); inspect its job log in {work_dir}"
            )
        log_warning = warn or logging.getLogger(__name__).warning
        windows = set(separation_window_targets(window_centres))
        spare = [
            c for c in missing_frames(out_dir, snapshot_centres) if c not in windows
        ]
        if spare:
            log_warning(
                "steered MD did not capture the spare snapshot(s) at "
                f"{', '.join(f'{c:g}' for c in spare)} nm; separation windows cannot "
                "be added there without re-running steered MD."
            )
        if images_path.exists():
            message = periodic_image_warning(
                json.loads(images_path.read_text()), window_centres
            )
            if message is not None:
                log_warning(message)
        return {float(k): v for k, v in json.loads(result_path.read_text()).items()}

    return _generate


def run_smd(work_dir: str | pathlib.Path) -> None:
    """Run the steered MD whose spec is at ``work_dir/smd.json`` (backend entry point).

    Writes the per-centre ``<centre>nm.rst7`` frames into ``spec.out_dir``, a
    ``result.json`` mapping centre -> path, and ``periodic_images.json`` (each
    frame's closest solute approach to its periodic image) into ``work_dir``.
    Raises on failure.
    """
    work_dir = pathlib.Path(work_dir)
    spec = SmdSpec.load(work_dir / SMD_SPEC_FILENAME)
    import openmm as mm

    periodic_images: dict = {}
    frames = run_steered_md(
        topology=spec.topology,
        coordinates=spec.coordinates,
        out_dir=spec.out_dir,
        rec_group=spec.rec_group,
        lig_group=spec.lig_group,
        anchors=spec.anchors,
        rmsd_atoms_bound=spec.rmsd_atoms_bound,
        boresch_eq_values=spec.boresch_eq_values,
        window_centres=spec.window_centres,
        hmr_factor=spec.hmr_factor,
        pme_cutoff_nm=spec.pme_cutoff_nm,
        timestep_fs=spec.timestep_fs,
        temperature_K=spec.temperature_K,
        k_smd=spec.k_smd,
        k_rmsd=spec.k_rmsd,
        k_boresch=spec.k_boresch,
        smd_compression_margin=spec.smd_compression_margin,
        smd_pull_margin=spec.smd_pull_margin,
        total_steps=spec.total_steps,
        increment_steps=spec.increment_steps,
        state_data_interval_steps=spec.state_data_interval_steps,
        platform=mm.Platform.getPlatformByName(spec.platform),
        periodic_images=periodic_images,
    )
    (work_dir / SMD_PERIODIC_IMAGES_FILENAME).write_text(
        json.dumps(periodic_images, indent=2)
    )
    (work_dir / SMD_RESULT_FILENAME).write_text(json.dumps(frames, indent=2))


def run_steered_md(
    *,
    topology,
    coordinates,
    out_dir: str | pathlib.Path,
    rec_group: list[int],
    lig_group: list[int],
    anchors: dict[str, int],
    rmsd_atoms_bound: dict[str, list[int]],
    boresch_eq_values: dict,
    window_centres,
    hmr_factor: float,
    pme_cutoff_nm: float,
    timestep_fs: float,
    temperature_K: float,
    k_smd: float = 100.0,
    k_rmsd: float = 100.0,
    k_boresch: float = 200.0,
    smd_compression_margin: float = 0.1,
    smd_pull_margin: float = 0.5,
    total_steps: int = 750_000,
    increment_steps: int = 100,
    state_data_interval_steps: int = 10000,
    platform=None,
    periodic_images: dict | None = None,
) -> dict[float, str]:
    """Steer the interface separation, saving an rst7 per window centre.

    The steering centre starts at the measured separation, first compresses the
    partners to ``smd_compression_margin`` below the smallest target, then pulls
    them apart to ``smd_pull_margin`` beyond the largest, saving each target's
    frame the first time it is reached on the way out (:func:`smd_pull_plan`);
    ``total_steps`` covers the outward pull. Returns ``{centre_nm: rst7_path}``.
    Reuses the shared system builder and restraint modules so the geometry is
    identical to sampling.

    If ``periodic_images`` is given, it is filled with ``{centre_nm: {distance_A,
    atoms}}``: each saved frame's closest approach of the solute to its own
    periodic image (:func:`closest_periodic_image`).
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from gluebind.restraints import boresch as boresch_mod
    from gluebind.restraints import rmsd as rmsd_mod
    from gluebind.restraints import separation as separation_mod
    from gluebind.restraints import system_builder as sb

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = separation_window_targets(window_centres)

    prmtop, system = sb.build_system(
        topology, hmr_factor=hmr_factor, pme_cutoff_nm=pme_cutoff_nm
    )
    solute_atoms = sorted(i for m in sb.solute_molecules(prmtop.topology) for i in m)
    positions, box = sb.load_coordinates(coordinates)
    simulation, integrator = sb.build_simulation(
        prmtop, system, timestep_fs=timestep_fs, platform=platform
    )
    simulation.context.setPeriodicBoxVectors(*box)
    simulation.context.setPositions(positions)

    # Fixed RMSD + Boresch restraints reference the equilibrated *input* structure and
    # are applied BEFORE minimisation/heating, so the regions are held rigid throughout
    # — rather than referencing a post-heat snapshot the structure has already drifted
    # into (matches run_window and the production run).
    reference = positions
    for region, atoms in rmsd_atoms_bound.items():
        rmsd_mod.add_rmsd_restraint(
            system, atoms, reference, k_rmsd, name=region, centre=None
        )
    points = boresch_mod.points_from_groups(rec_group, lig_group, anchors)
    for dof, eq_value in boresch_eq_values.items():
        boresch_mod.add_fixed_restraint(system, dof, points, eq_value, k_boresch)
    simulation.context.reinitialize(preserveState=True)
    sb.minimise_and_heat(simulation, integrator, target_temperature_K=temperature_K)

    # The moving separation bias, added after heating the restrained bound state.
    # Its centre starts at the measured separation, so steering begins smoothly.
    cv = separation_mod.make_cv(rec_group, lig_group)
    steer = mm.CustomCVForce("0.5*k_smd*(cv-r0)^2")
    steer.addGlobalParameter(
        "k_smd", k_smd * unit.kilocalories_per_mole / unit.angstrom**2
    )
    steer.addGlobalParameter("r0", 0.0)  # nm; set before the first step
    steer.addCollectiveVariable("cv", cv)
    system.addForce(steer)
    simulation.context.reinitialize(preserveState=True)
    start = steer.getCollectiveVariableValues(simulation.context)[0]

    n_pull = total_steps // increment_steps
    compressed, step, n_compress = smd_pull_plan(
        start,
        targets,
        compression_margin=smd_compression_margin,
        pull_margin=smd_pull_margin,
        n_pull_increments=n_pull,
    )
    print(
        f"steered MD: starting separation {start:.3f} nm; compressing to "
        f"{compressed:.3f} nm, then pulling to {targets[-1] + smd_pull_margin:.3f} nm",
        flush=True,
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            state_data_interval_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            progress=True,
            remainingTime=True,
            totalSteps=(n_compress + n_pull) * increment_steps,
            separator="\t",
        )
    )

    def steer_to(r0: float) -> float:
        """Move the steering centre to ``r0`` (nm), run one increment, and
        return the measured separation (nm)."""
        simulation.context.setParameter(
            "r0",
            r0 * unit.nanometers,  # ty: ignore[unresolved-attribute]
        )
        simulation.step(increment_steps)
        return steer.getCollectiveVariableValues(simulation.context)[0]

    # Compress: bring the partners together to below the smallest target.
    closest = start
    for k in range(1, n_compress + 1):
        closest = min(closest, steer_to(max(start - k * step, compressed)))
    print(
        f"steered MD: closest separation reached while compressing: {closest:.3f} nm",
        flush=True,
    )

    # Pull outward past the furthest target, snapshotting each target the first
    # time the measured distance reaches it.
    frames: dict[float, str] = {}
    remaining = list(targets)
    for k in range(1, n_pull + 1):
        if not remaining:
            break
        current = steer_to(compressed + k * step)
        while remaining and current >= remaining[0]:
            target = remaining.pop(0)
            state = simulation.context.getState(getPositions=True)
            out_path = smd_frame_path(out_dir, target)
            sb.save_rst7(
                topology, state.getPositions(), state.getPeriodicBoxVectors(), out_path
            )
            frames[target] = str(out_path)
            if periodic_images is not None:
                distance, i, j = closest_periodic_image(
                    state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                    state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.angstrom
                    ),
                    solute_atoms,
                )
                periodic_images[target] = {"distance_A": distance, "atoms": [i, j]}

    return frames
