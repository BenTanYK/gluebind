"""BioSimSpace preparation front-end.

Parameterises the glue, assembles + solvates the complex, runs the
pre-equilibration/equilibration chain, extracts the isolated bulk species, and
writes a :class:`PreparedSystem` manifest — the hand-off to Phase 4 (selection)
and the runner's ``spec_builder``.

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
    dims = [hi - lo for lo, hi in zip(box_min, box_max)]
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
        return cls.model_validate_json((pathlib.Path(run_dir) / PREPARED_FILENAME).read_text())


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
    """Combine target + receptor + optional glue and solvate the complex."""
    import BioSimSpace as BSS

    system = target + receptor
    if glue is not None:
        system = system + glue

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


def build_equilibration_protocols(prep_config: PrepConfig):
    """The staged pre-equilibration + equilibration protocols (named)."""
    import BioSimSpace as BSS

    ns = BSS.Units.Time.nanosecond
    kelvin = BSS.Units.Temperature.kelvin
    atm = BSS.Units.Pressure.atm
    t = PRODUCTION_TEMPERATURE_K
    return [
        ("minimisation", BSS.Protocol.Minimisation(steps=prep_config.minimisation_steps)),
        (
            "nvt_heat",
            BSS.Protocol.Equilibration(
                runtime=prep_config.nvt_heat_ns * ns,
                temperature_start=0 * kelvin,
                temperature_end=t * kelvin,
                restraint="backbone",
            ),
        ),
        (
            "nvt",
            BSS.Protocol.Equilibration(
                runtime=prep_config.nvt_ns * ns,
                temperature_start=t * kelvin,
                temperature_end=t * kelvin,
                restraint="backbone",
            ),
        ),
        (
            "npt",
            BSS.Protocol.Equilibration(
                runtime=prep_config.npt_ns * ns,
                temperature_start=t * kelvin,
                temperature_end=t * kelvin,
                pressure=atm,
                restraint="backbone",
            ),
        ),
        (
            "equilibration",
            BSS.Protocol.Equilibration(
                runtime=prep_config.equilibration_ns * ns,
                temperature_start=t * kelvin,
                temperature_end=t * kelvin,
                restraint="none",
            ),
        ),
    ]


def run_equilibration(system, protocols, *, platform: str = "CPU"):
    """Run each protocol in turn with OpenMM; return (final system, last process)."""
    import BioSimSpace as BSS

    process = None
    for name, protocol in protocols:
        process = BSS.Process.OpenMM(system, protocol, platform=platform)
        process.start()
        process.wait()
        if process.isError():
            raise RuntimeError(f"equilibration stage {name!r} failed:\n{process.stdout(20)}")
        system = process.getSystem()
    return system, process


def _save(system, prefix: pathlib.Path) -> tuple[str, str]:
    import BioSimSpace as BSS

    BSS.IO.saveMolecules(str(prefix), system, ["prm7", "rst7"])
    return f"{prefix}.prm7", f"{prefix}.rst7"


def _bulk_indices(layout: ComponentLayout, component: str, assign_to: str | None) -> list[int]:
    indices = list(getattr(layout, component))
    if layout.glue is not None and assign_to == component:
        indices.append(layout.glue)
    return indices


def _extract_bulk(system, indices, prep_config, prefix, platform) -> tuple[str, str]:
    """Isolate the given molecules, re-solvate, briefly equilibrate, and save."""
    import BioSimSpace as BSS

    isolated = system[indices[0]]
    for i in indices[1:]:
        isolated = isolated + system[i]
    solvated = BSS.Solvent.solvate(
        prep_config.water_model,
        molecule=isolated.toSystem() if hasattr(isolated, "toSystem") else isolated,
        box=BSS.Box.generateBoxParameters(
            prep_config.box_type,
            box_length(*_bounding_box(isolated), prep_config.box_padding_angstrom * BSS.Units.Length.angstrom),
        )[0],
        is_neutral=prep_config.neutralise,
        ion_conc=prep_config.ion_concentration_M,
    )
    # Short NVT + NPT for the isolated species (reuse the first protocols).
    protocols = build_equilibration_protocols(prep_config)[:4]
    equilibrated, _ = run_equilibration(solvated, protocols, platform=platform)
    return _save(equilibrated, prefix)


def _bounding_box(molecule):
    box_min, box_max = molecule.getAxisAlignedBoundingBox()
    return box_min, box_max


def prepare(
    config: CalculationConfig, work_dir: str | pathlib.Path, *, platform: str = "CPU"
) -> PreparedSystem:
    """Full preparation: parameterise, assemble, solvate, equilibrate, extract bulk.

    Writes ``complex_equil.*``, ``target_bulk.*``, ``receptor_bulk.*`` and the
    ``prepared.json`` manifest into ``work_dir``, and returns the manifest.
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

    layout = compute_layout(count_molecules(target), count_molecules(receptor), glue is not None)

    solvated = assemble_and_solvate(target, receptor, glue, config.prep)
    equilibrated, last_process = run_equilibration(
        solvated, build_equilibration_protocols(config.prep), platform=platform
    )

    complex_prm7, complex_rst7 = _save(equilibrated, work_dir / "complex_equil")
    trajectory = None
    try:
        traj_path = work_dir / "complex_equil.dcd"
        last_process.getTrajectory().getTrajectory(format="mdtraj").save(str(traj_path))
        trajectory = str(traj_path)
    except Exception:  # noqa: BLE001 - trajectory is best-effort; prep still succeeds
        trajectory = None

    target_bulk = _extract_bulk(
        equilibrated,
        _bulk_indices(layout, "target", assign_to),
        config.prep,
        work_dir / "target_bulk",
        platform,
    )
    receptor_bulk = _extract_bulk(
        equilibrated,
        _bulk_indices(layout, "receptor", assign_to),
        config.prep,
        work_dir / "receptor_bulk",
        platform,
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
