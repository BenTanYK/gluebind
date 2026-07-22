"""OpenMM production-run driver for the final equilibration stage.

The long production run that seeds the US windows (and feeds RMSF/anchor
selection) is run here in OpenMM — not BioSimSpace — so that any constant
restraints (e.g. the DDB1-scaffold surrogate for DCAF16) can be applied during
it, on atom indices resolved exactly as the US windows resolve them. Running it
in OpenMM (``AmberPrmtopFile`` reads the ``.prm7`` in file order) keeps the
restraint indexing identical to the US windows and immune to any re-indexing
BioSimSpace applies (see :mod:`gluebind.system.atom_map`).

The run reuses the shared system builder and RMSD restraint module, so the
geometry is identical to sampling. It is NVT (canonical ensemble, matching the
paper's production), starts from the NPT-equilibrated structure (no re-heating),
and holds each constant restraint to that equilibrated structure. OpenMM/ParmEd
are imported lazily inside the run function.
"""

from __future__ import annotations

import pathlib

import pydantic

PRODUCTION_SPEC_FILENAME = "production.json"
PRODUCTION_RESULT_FILENAME = "result.json"
PRODUCTION_OUTPUT_PREFIX = "output"


class ProductionSpec(pydantic.BaseModel):
    """Everything the OpenMM production run needs, self-contained for a backend job."""

    model_config = pydantic.ConfigDict(extra="forbid")

    topology: str
    coordinates: str
    restraints: list[dict] = pydantic.Field(default_factory=list)
    """Constant RMSD-to-reference restraints, each ``{name, atoms, force_constant}``
    (held about the equilibrated structure, i.e. ``centre=None``)."""
    runtime_ns: float
    timestep_fs: float = 4.0
    hmr_factor: float = 1.5
    pme_cutoff_nm: float = 1.0
    temperature_K: float = 300.0
    sample_interval_steps: int = 2500
    """Trajectory (DCD) write interval, in MD steps."""
    platform: str = "CUDA"

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ProductionSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def production_launch_command(python: str = "python") -> list[str]:
    """The command a backend runs (inside the production work dir) to execute it."""
    code = (
        "from gluebind.simulation.production import run_production; run_production('.')"
    )
    return [python, "-c", code]


def run_production(work_dir: str | pathlib.Path) -> None:
    """Run the production stage whose spec is at ``work_dir/production.json``.

    Writes ``output.prm7`` / ``output.rst7`` (the seed for the US windows) and
    ``output.dcd`` into ``work_dir``, plus a ``result.json``. Raises on failure.
    """
    import json
    import shutil

    import openmm.app as app
    import openmm.unit as unit

    from gluebind.restraints import rmsd
    from gluebind.restraints import system_builder as sb

    work_dir = pathlib.Path(work_dir)
    spec = ProductionSpec.load(work_dir / PRODUCTION_SPEC_FILENAME)
    prefix = work_dir / PRODUCTION_OUTPUT_PREFIX

    prmtop, system = sb.build_system(
        spec.topology, hmr_factor=spec.hmr_factor, pme_cutoff_nm=spec.pme_cutoff_nm
    )
    positions, box_vectors = sb.load_coordinates(spec.coordinates)
    simulation, _integrator = sb.build_simulation(
        prmtop, system, timestep_fs=spec.timestep_fs, platform=_platform(spec.platform)
    )
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)

    # Constant restraints hold each region to the NPT-equilibrated structure (the
    # reference is the input coordinates — no re-heating in production).
    for entry in spec.restraints:
        rmsd.add_rmsd_restraint(
            system,
            entry["atoms"],
            positions,
            entry["force_constant"],
            name=entry["name"],
            centre=None,
        )
        simulation.context.reinitialize(preserveState=True)

    simulation.context.setVelocitiesToTemperature(spec.temperature_K * unit.kelvin)
    simulation.reporters.append(
        app.DCDReporter(f"{prefix}.dcd", spec.sample_interval_steps)
    )

    n_steps = int(round(spec.runtime_ns / (spec.timestep_fs * 1e-6)))
    simulation.step(n_steps)

    final = simulation.context.getState(getPositions=True)
    sb.save_rst7(
        spec.topology,
        final.getPositions(),
        final.getPeriodicBoxVectors(),
        f"{prefix}.rst7",
    )
    shutil.copyfile(spec.topology, f"{prefix}.prm7")  # topology is unchanged by MD
    (work_dir / PRODUCTION_RESULT_FILENAME).write_text(
        json.dumps({"prm7": f"{prefix}.prm7", "rst7": f"{prefix}.rst7"})
    )


def _platform(name: str):
    import openmm as mm

    return mm.Platform.getPlatformByName(name)
