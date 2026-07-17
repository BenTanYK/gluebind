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
import pathlib

import pydantic

SMD_SPEC_FILENAME = "smd.json"
SMD_RESULT_FILENAME = "result.json"


def separation_window_targets(centres) -> list[float]:
    """Sorted, de-duplicated window centres (nm) to snapshot during the pull."""
    return sorted({round(float(c), 4) for c in centres})


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

    # Steered-MD force constants / schedule (template defaults; stiffer than US)
    k_smd: float = 100.0
    k_rmsd: float = 50.0
    k_boresch: float = 250.0
    initial_r0_nm: float = 1.15
    total_steps: int = 750_000
    increment_steps: int = 100
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
    window_centres,
    sampling,
    platform: str = "CUDA",
):
    """Return the ``callable(boresch_eq_values)`` the runner invokes between the
    Boresch and separation stages.

    The callable writes an :class:`SmdSpec` into ``work_dir`` and submits a single
    backend job (so the pull runs on a compute node, not the driver); the job
    writes the per-centre frames into ``out_dir``, which the spec builder then
    reads for the separation windows.
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
            window_centres=separation_window_targets(window_centres),
            hmr_factor=sampling.hmr_factor,
            pme_cutoff_nm=sampling.pme_cutoff_nm,
            timestep_fs=sampling.timestep_fs,
            temperature_K=sampling.temperature_K,
            platform=platform,
        )
        spec.dump(work_dir / SMD_SPEC_FILENAME)
        job = JobSpec(command=smd_launch_command(), work_dir=str(work_dir), name="steered_md")
        (state,) = scheduler_factory().run([job])
        if state is not JobState.FINISHED:
            raise RuntimeError(f"steered MD did not finish (state={state})")
        result_path = work_dir / SMD_RESULT_FILENAME
        if result_path.exists():
            return {float(k): v for k, v in json.loads(result_path.read_text()).items()}
        return {}

    return _generate


def _save_frame_rst7(prmtop_path, positions, box_vectors, out_path) -> None:
    """Write an AMBER rst7 (positions + box) that run_window can reload."""
    import parmed

    structure = parmed.load_file(str(prmtop_path))
    structure.positions = positions
    structure.box_vectors = box_vectors
    structure.save(str(out_path), format="rst7", overwrite=True)


def run_smd(work_dir: str | pathlib.Path) -> None:
    """Run the steered MD whose spec is at ``work_dir/smd.json`` (backend entry point).

    Writes the per-centre ``<centre>nm.rst7`` frames into ``spec.out_dir`` and a
    ``result.json`` mapping centre -> path into ``work_dir``. Raises on failure.
    """
    work_dir = pathlib.Path(work_dir)
    spec = SmdSpec.load(work_dir / SMD_SPEC_FILENAME)
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
        initial_r0_nm=spec.initial_r0_nm,
        total_steps=spec.total_steps,
        increment_steps=spec.increment_steps,
        platform=spec.platform,
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
    k_rmsd: float = 50.0,
    k_boresch: float = 250.0,
    initial_r0_nm: float = 1.15,
    total_steps: int = 750_000,
    increment_steps: int = 100,
    platform=None,
) -> dict[float, str]:
    """Steer the interface separation outward, saving an rst7 per window centre.

    Returns ``{centre_nm: rst7_path}``. Force constants default to the template's
    steered-MD values (stiffer than the US windows). Reuses the shared system
    builder and restraint modules so the geometry is identical to sampling.
    """
    import openmm as mm
    import openmm.unit as unit

    from gluebind.restraints import boresch as boresch_mod
    from gluebind.restraints import rmsd as rmsd_mod
    from gluebind.restraints import separation as separation_mod
    from gluebind.restraints import system_builder as sb

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = separation_window_targets(window_centres)

    prmtop, system = sb.build_system(topology, hmr_factor=hmr_factor, pme_cutoff_nm=pme_cutoff_nm)
    positions, box = sb.load_coordinates(coordinates)
    simulation, integrator = sb.build_simulation(
        prmtop, system, timestep_fs=timestep_fs, platform=platform
    )
    simulation.context.setPeriodicBoxVectors(*box)
    simulation.context.setPositions(positions)
    sb.minimise_and_heat(simulation, integrator, target_temperature_K=temperature_K)
    reference = simulation.context.getState(getPositions=True).getPositions()

    # Fixed RMSD + Boresch restraints (rigid), then the moving separation bias.
    for region, atoms in rmsd_atoms_bound.items():
        rmsd_mod.add_rmsd_restraint(system, atoms, reference, k_rmsd, name=region, centre=None)
    points = boresch_mod.points_from_groups(rec_group, lig_group, anchors)
    for dof, eq_value in boresch_eq_values.items():
        boresch_mod.add_fixed_restraint(system, dof, points, eq_value, k_boresch)

    cv = separation_mod.make_cv(rec_group, lig_group)
    steer = mm.CustomCVForce("0.5*k_smd*(cv-r0)^2")
    steer.addGlobalParameter("k_smd", k_smd * unit.kilocalories_per_mole / unit.angstrom**2)
    steer.addGlobalParameter("r0", initial_r0_nm * unit.nanometers)
    steer.addCollectiveVariable("cv", cv)
    system.addForce(steer)
    simulation.context.reinitialize(preserveState=True)

    # Pull r0 from the initial value out past the furthest target, snapshotting
    # each target the first time the measured distance reaches it.
    r0 = initial_r0_nm
    span = max(targets) - initial_r0_nm + 0.2
    per_increment = span / (total_steps // increment_steps)
    frames: dict[float, str] = {}
    remaining = list(targets)

    for _ in range(total_steps // increment_steps):
        if not remaining:
            break
        r0 += per_increment
        simulation.context.setParameter("r0", r0 * unit.nanometers)
        simulation.step(increment_steps)
        current = steer.getCollectiveVariableValues(simulation.context)[0]
        while remaining and current >= remaining[0]:
            target = remaining.pop(0)
            state = simulation.context.getState(getPositions=True)
            out_path = out_dir / f"{target:.4g}nm.rst7"
            _save_frame_rst7(topology, state.getPositions(), state.getPeriodicBoxVectors(), out_path)
            frames[target] = str(out_path)

    return frames
