"""BioSimSpace preparation front-end.

Parameterises the glue and assembles + solvates the complex on the driver, then
dispatches the equilibration to a backend as one job per stage (minimisation ->
NVT heat -> NPT -> long NVT production; see :func:`run_equilibration_stages` and
:mod:`gluebind.simulation.prep_stage`). The isolated bulk species are likewise
equilibrated through the backend — no MD/GPU work runs on the driver at all. It
then writes a :class:`PreparedSystem` manifest — the hand-off to Phase 4
(selection) and the runner's ``spec_builder``. A single equilibration run is
used; the paper found triplicate equilibration trajectories to be essentially
identical.

The MD-running functions require BioSimSpace + a working MD engine, so they are
verified in integration (Phase 7), not the unit suite; BSS is imported lazily.
The pure helpers (force-field validation, box sizing, the manifest) are unit
tested. Equilibration runs at the paper's 300 K; the US production temperature is
a separate, config-driven value used later.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pydantic

from gluebind.config.calculation import CalculationConfig
from gluebind.config.prep import PrepConfig
from gluebind.system.inputs import (
    ComponentLayout,
    compute_layout,
    count_molecules,
    load_glue,
    load_system,
)

PREPARED_FILENAME = "prepared.json"
PRODUCTION_TEMPERATURE_K = 300.0


# ---- pure helpers ----------------------------------------------------------


def normalise_ff_name(name: str) -> str:
    """Normalise a force-field name to a BSS.Parameters attribute form."""
    return name.replace("-", "_").replace(".", "_")


def validate_forcefield(name: str, available: Sequence[str]) -> str:
    """Return the normalised name if available, else raise with the options."""
    norm = normalise_ff_name(name)
    if norm not in {normalise_ff_name(a) for a in available}:
        raise ValueError(
            f"force field {name!r} is not available; choose from {sorted(available)}"
        )
    return norm


def box_length(box_min, box_max, padding):
    """Cubic box edge = largest molecule dimension + 2 * padding.

    Works with plain floats or BSS length Quantities (any type supporting
    subtraction and ``max``).
    """
    dims = [hi - lo for lo, hi in zip(box_min, box_max, strict=False)]
    return max(dims) + 2 * padding


class PreparedSystem(pydantic.BaseModel):
    """Manifest of the prepared structures — the Phase 3 → Phase 4 hand-off."""

    schema_version: int = 1
    complex_prm7: str
    complex_rst7: str
    complex_trajectory: str | None = None
    target_bulk_prm7: str
    target_bulk_rst7: str
    receptor_bulk_prm7: str
    receptor_bulk_rst7: str
    glue_assign_to: str | None = None
    target_molecules: list[int]
    receptor_molecules: list[int]
    glue_molecule: int | None = None

    def dump(self, run_dir: str | pathlib.Path) -> pathlib.Path:
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / PREPARED_FILENAME
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, run_dir: str | pathlib.Path) -> "PreparedSystem":
        return cls.model_validate_json(
            (pathlib.Path(run_dir) / PREPARED_FILENAME).read_text()
        )


# ---- BioSimSpace operations (integration-verified) -------------------------


def available_forcefields() -> list[str]:
    import BioSimSpace as BSS

    return BSS.Parameters.forceFields()


def parameterise_glue(sdf: str | pathlib.Path, forcefield: str):
    """Parameterise the glue small molecule with the named force field."""
    import BioSimSpace as BSS

    norm = validate_forcefield(forcefield, available_forcefields())
    molecule = load_glue(sdf)
    return getattr(BSS.Parameters, norm)(molecule).getMolecule()


def assemble_and_solvate(target, receptor, glue, prep_config: PrepConfig):
    """Combine glue (MOL) + receptor + target and solvate the complex.

    Assembly order is glue first, then receptor, then target — so the glue is
    molecule 0 and each protein occupies a contiguous atom block, which the
    input->complex atom map relies on (see :mod:`gluebind.system.atom_map`)."""
    import BioSimSpace as BSS

    system = glue + receptor if glue is not None else receptor
    system = system + target

    box_min, box_max = system.getAxisAlignedBoundingBox()
    padding = prep_config.box_padding_angstrom * BSS.Units.Length.angstrom
    length = box_length(box_min, box_max, padding)
    box, angles = BSS.Box.generateBoxParameters(prep_config.box_type, length)
    return BSS.Solvent.solvate(
        prep_config.water_model,
        molecule=system,
        box=box,
        angles=angles,
        is_neutral=prep_config.neutralise,
        ion_conc=prep_config.ion_concentration_M,
    )


def equilibration_stage_plan(prep_config: PrepConfig) -> list[dict]:
    """The ordered equilibration stages as plain dicts — no BioSimSpace.

    Four stages, each dispatched as its own backend job by
    :func:`run_equilibration_stages`:

    1. **minimisation**
    2. **nvt_heat** — NVT ramp 0 K -> production T, backbone-restrained
    3. **npt** — NPT equilibration at production T, backbone-restrained (relaxes
       the box volume)
    4. **equilibration** — long NVT production equilibration at production T,
       unrestrained (the trajectory used for RMSF/anchor selection and Boresch
       distributions, and the source of the bound-state structure)

    Pure and unit-testable; the keys match :class:`PrepStageSpec`'s fields.
    """
    t = PRODUCTION_TEMPERATURE_K
    return [
        {
            "stage": "minimisation",
            "kind": "minimisation",
            "minimisation_steps": prep_config.minimisation_steps,
        },
        {
            "stage": "nvt_heat",
            "kind": "equilibration",
            "runtime_ns": prep_config.nvt_heat_ns,
            "temperature_start_K": 10.0,
            "temperature_end_K": t,
            "pressure": False,
            "restraint": "backbone",
        },
        {
            "stage": "npt",
            "kind": "equilibration",
            "runtime_ns": prep_config.npt_ns,
            "temperature_start_K": t,
            "temperature_end_K": t,
            "pressure": True,
            "restraint": "backbone",
        },
        {
            "stage": "equilibration",
            "kind": "equilibration",
            "runtime_ns": prep_config.equilibration_ns,
            "temperature_start_K": t,
            "temperature_end_K": t,
            "pressure": False,
            "restraint": "none",
        },
    ]


def run_equilibration_stages(
    solvated_prm7: str | pathlib.Path,
    solvated_rst7: str | pathlib.Path,
    plan: list[dict],
    work_dir: str | pathlib.Path,
    backend,
    *,
    platform: str = "CUDA",
    poll_interval: float = 30.0,
    save_trajectory: bool = True,
) -> tuple[str, str, str | None]:
    """Run the equilibration ``plan`` as one backend job per stage.

    Each stage runs in its own ``NN_<stage>/`` subdirectory (own SLURM log) and
    writes ``output.prm7`` / ``.rst7`` (and ``.dcd`` for MD stages), which chain
    into the next stage's input. Returns ``(final_prm7, final_rst7, trajectory)``
    where ``trajectory`` is the last MD stage's ``.dcd`` (or ``None`` if not
    produced). Blocks on each stage before submitting the next — the stages are a
    strict sequential dependency chain.
    """
    from gluebind.backend.base import JobSpec, JobState
    from gluebind.backend.scheduler import Scheduler
    from gluebind.simulation.prep_stage import (
        PREP_STAGE_OUTPUT_PREFIX,
        PREP_STAGE_SPEC_FILENAME,
        PrepStageSpec,
        prep_stage_launch_command,
    )

    work_dir = pathlib.Path(work_dir)
    scheduler = Scheduler(backend, poll_interval=poll_interval)
    input_prm7, input_rst7 = str(solvated_prm7), str(solvated_rst7)
    trajectory: str | None = None

    for i, entry in enumerate(plan, start=1):
        stage_dir = work_dir / f"{i:02d}_{entry['stage']}"
        prefix = stage_dir / PREP_STAGE_OUTPUT_PREFIX
        out_prm7, out_rst7 = f"{prefix}.prm7", f"{prefix}.rst7"

        # Resume: a stage whose output structures already exist is skipped, so an
        # interrupted prep (or an auto-prepare on a resumed run) never re-runs
        # completed equilibration stages.
        already_done = (
            pathlib.Path(out_prm7).exists() and pathlib.Path(out_rst7).exists()
        )
        if not already_done:
            stage_dir.mkdir(parents=True, exist_ok=True)
            spec = PrepStageSpec(
                input_prm7=input_prm7,
                input_rst7=input_rst7,
                platform=platform,
                save_trajectory=save_trajectory,
                **entry,
            )
            spec.dump(stage_dir / PREP_STAGE_SPEC_FILENAME)
            job = JobSpec(
                command=prep_stage_launch_command(),
                work_dir=str(stage_dir),
                name=f"prep_{entry['stage']}",
            )
            (state,) = scheduler.run([job])
            if state is not JobState.FINISHED:
                raise RuntimeError(
                    f"prep stage {entry['stage']!r} did not finish (state={state})"
                )

        input_prm7, input_rst7 = out_prm7, out_rst7
        if entry["kind"] == "equilibration" and prefix.with_suffix(".dcd").exists():
            trajectory = str(prefix.with_suffix(".dcd"))

    return input_prm7, input_rst7, trajectory


def _save(system, prefix: pathlib.Path) -> tuple[str, str]:
    import BioSimSpace as BSS

    BSS.IO.saveMolecules(str(prefix), system, ["prm7", "rst7"])
    return f"{prefix}.prm7", f"{prefix}.rst7"


def _bulk_indices(
    layout: ComponentLayout, component: str, assign_to: str | None
) -> list[int]:
    indices = list(getattr(layout, component))
    if layout.glue is not None and assign_to == component:
        indices.append(layout.glue)
    return indices


def _extract_bulk(
    system,
    indices,
    prep_config,
    out_dir,
    backend,
    *,
    platform: str,
    poll_interval: float,
) -> tuple[str, str]:
    """Isolate the given molecules and re-solvate them on the driver (cheap), then
    equilibrate through ``backend`` — like the complex, no MD runs on the driver.

    Uses the short pre-equilibration (minimisation -> NVT heat -> NPT, the first
    three stages) — not the long production run — and keeps no trajectory.
    """
    import BioSimSpace as BSS

    out_dir = pathlib.Path(out_dir)
    isolated = system[indices[0]]
    for i in indices[1:]:
        isolated = isolated + system[i]
    solvated = BSS.Solvent.solvate(
        prep_config.water_model,
        molecule=isolated.toSystem() if hasattr(isolated, "toSystem") else isolated,
        box=BSS.Box.generateBoxParameters(
            prep_config.box_type,
            box_length(
                *_bounding_box(isolated),
                prep_config.box_padding_angstrom * BSS.Units.Length.angstrom,
            ),
        )[0],
        is_neutral=prep_config.neutralise,
        ion_conc=prep_config.ion_concentration_M,
    )
    solvated_prm7, solvated_rst7 = _save(solvated, out_dir / "solvated")

    bulk_plan = equilibration_stage_plan(prep_config)[:3]  # min -> NVT heat -> NPT
    final_prm7, final_rst7, _ = run_equilibration_stages(
        solvated_prm7,
        solvated_rst7,
        bulk_plan,
        out_dir / "equilibration",
        backend,
        platform=platform,
        poll_interval=poll_interval,
        save_trajectory=False,
    )
    return final_prm7, final_rst7


def _bounding_box(molecule):
    box_min, box_max = molecule.getAxisAlignedBoundingBox()
    return box_min, box_max


def prepare(
    config: CalculationConfig,
    work_dir: str | pathlib.Path,
    backend,
    *,
    platform: str = "CUDA",
    poll_interval: float = 30.0,
) -> PreparedSystem:
    """Full preparation: parameterise, assemble, solvate, equilibrate, extract bulk.

    The cheap, CPU-bound setup (glue parameterisation, assembly, solvation, and
    bulk isolation) runs here on the driver; every MD stage — the complex
    equilibration and the two bulk-species equilibrations — is dispatched to
    ``backend`` as one job per stage (see :func:`run_equilibration_stages`), so it
    runs on compute nodes with per-stage logs and intermediate snapshots and no
    MD/GPU work runs on the driver. A single equilibration run is used (no
    ensemble).

    Writes ``solvated.*``, ``equilibration/NN_<stage>/output.*``,
    ``{target,receptor}_bulk/{solvated.*,equilibration/NN_<stage>/output.*}`` and
    the ``prepared.json`` manifest into ``work_dir``, and returns the manifest.
    """
    import BioSimSpace as BSS

    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = config.inputs

    target = load_system(inputs.target.prm7, inputs.target.rst7)
    receptor = load_system(inputs.receptor.prm7, inputs.receptor.rst7)
    glue = None
    assign_to = None
    if inputs.glue is not None:
        glue = parameterise_glue(inputs.glue.sdf, config.prep.glue_forcefield)
        assign_to = inputs.glue.assign_to

    layout = compute_layout(
        count_molecules(target), count_molecules(receptor), glue is not None
    )

    # Driver (fast, CPU): assemble + solvate, then hand the MD stages to the backend.
    solvated = assemble_and_solvate(target, receptor, glue, config.prep)
    solvated_prm7, solvated_rst7 = _save(solvated, work_dir / "solvated")

    complex_prm7, complex_rst7, trajectory = run_equilibration_stages(
        solvated_prm7,
        solvated_rst7,
        equilibration_stage_plan(config.prep),
        work_dir / "equilibration",
        backend,
        platform=platform,
        poll_interval=poll_interval,
    )

    # Bulk reference species: isolate + re-solvate on the driver (cheap), then
    # equilibrate through the backend — no MD runs on the driver.
    equilibrated = BSS.IO.readMolecules([complex_prm7, complex_rst7])
    target_bulk = _extract_bulk(
        equilibrated,
        _bulk_indices(layout, "target", assign_to),
        config.prep,
        work_dir / "target_bulk",
        backend,
        platform=platform,
        poll_interval=poll_interval,
    )
    receptor_bulk = _extract_bulk(
        equilibrated,
        _bulk_indices(layout, "receptor", assign_to),
        config.prep,
        work_dir / "receptor_bulk",
        backend,
        platform=platform,
        poll_interval=poll_interval,
    )

    prepared = PreparedSystem(
        complex_prm7=complex_prm7,
        complex_rst7=complex_rst7,
        complex_trajectory=trajectory,
        target_bulk_prm7=target_bulk[0],
        target_bulk_rst7=target_bulk[1],
        receptor_bulk_prm7=receptor_bulk[0],
        receptor_bulk_rst7=receptor_bulk[1],
        glue_assign_to=assign_to,
        target_molecules=layout.target,
        receptor_molecules=layout.receptor,
        glue_molecule=layout.glue,
    )
    prepared.dump(work_dir)
    return prepared
