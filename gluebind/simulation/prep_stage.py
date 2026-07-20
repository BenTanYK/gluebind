"""The single equilibration-stage compute entry point.

Preparation is broken into one job per stage — minimisation, NVT heating, NPT
equilibration and the long NVT production equilibration — so each stage gets its
own SLURM job (and hence its own log) and writes an intermediate ``.prm7`` /
``.rst7`` snapshot the user can inspect or restart from.

:func:`run_prep_stage` is the unit of work, mirroring
:func:`gluebind.simulation.window.run_window`: it reads a :class:`PrepStageSpec`
(plus the input structures it references) from a working directory, runs that one
protocol with OpenMM via BioSimSpace, and writes the stage's final-frame
structures (and, for MD stages, the trajectory) back into that directory. It is
deliberately submission-agnostic — no scheduler, SLURM or S3 knowledge — so the
same function backs the local and SLURM backends (via
:func:`prep_stage_launch_command`) and a future AWS Batch runner. BioSimSpace is
imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import json
import pathlib
from typing import Literal

import pydantic

PREP_STAGE_SPEC_FILENAME = "prep_stage.json"
PREP_STAGE_RESULT_FILENAME = "result.json"
PREP_STAGE_OUTPUT_PREFIX = "output"
"""Basename (within the stage work dir) of the stage's final-frame structures."""

PrepStageKind = Literal["minimisation", "equilibration"]


class PrepStageSpec(pydantic.BaseModel):
    """Everything one equilibration stage needs to run, self-contained.

    Serialisable to ``prep_stage.json``; the worker needs nothing but this file
    and the input structures it points at. The protocol parameters are carried in
    the spec so the BioSimSpace protocol is rebuilt on the compute node — the
    driver never needs BioSimSpace to plan the stages.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    stage: str
    """Human-readable stage name (e.g. ``"nvt_heat"``); names the outputs/logs."""
    kind: PrepStageKind

    input_prm7: str
    input_rst7: str
    output_prefix: str = PREP_STAGE_OUTPUT_PREFIX
    """Basename, relative to the work dir, for the final-frame ``.prm7``/``.rst7``
    (and ``.dcd`` for MD stages)."""
    platform: str = "CPU"
    save_trajectory: bool = True

    # Minimisation
    minimisation_steps: int | None = None
    # Equilibration
    runtime_ns: float | None = None
    temperature_start_K: float | None = None
    temperature_end_K: float | None = None
    pressure: bool = False
    """NPT (barostat at 1 atm) when True, otherwise NVT."""
    restraint: str | None = None
    """BioSimSpace restraint keyword, e.g. ``"backbone"`` or ``"none"``."""

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PrepStageSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def prep_stage_launch_command(python: str = "python") -> list[str]:
    """The command a backend runs (inside the stage work dir) to execute it.

    Reads ``prep_stage.json`` from the current directory and runs the stage there,
    so a backend only has to set the job's working directory — the same ``python
    -c`` seam used for windows, keeping v1 free of any CLI.
    """
    code = (
        "from gluebind.simulation.prep_stage import run_prep_stage; run_prep_stage('.')"
    )
    return [python, "-c", code]


def build_protocol(
    *,
    kind: PrepStageKind,
    minimisation_steps: int | None = None,
    runtime_ns: float | None = None,
    temperature_start_K: float | None = None,
    temperature_end_K: float | None = None,
    pressure: bool = False,
    restraint: str | None = None,
):
    """Build the BioSimSpace protocol for one stage (BSS imported lazily).

    Accepts the stage-plan parameters directly so it is reused both by
    :func:`run_prep_stage` (from a :class:`PrepStageSpec`) and by the in-process
    bulk-species equilibration in :mod:`gluebind.system.prep`.
    """
    import BioSimSpace as BSS

    if kind == "minimisation":
        return BSS.Protocol.Minimisation(steps=minimisation_steps)

    ns = BSS.Units.Time.nanosecond
    kelvin = BSS.Units.Temperature.kelvin
    kwargs = {
        "runtime": runtime_ns * ns,
        "temperature_start": temperature_start_K * kelvin,
        "temperature_end": temperature_end_K * kelvin,
        "restraint": restraint,
    }
    if pressure:
        kwargs["pressure"] = BSS.Units.Pressure.atm
    return BSS.Protocol.Equilibration(**kwargs)


def run_prep_stage(work_dir: str | pathlib.Path) -> None:
    """Run the equilibration stage whose spec is at ``work_dir/prep_stage.json``.

    Loads the input structures, runs the single protocol with OpenMM, and writes
    the stage's final-frame ``<output_prefix>.prm7`` / ``.rst7`` (and, for MD
    stages, ``<output_prefix>.dcd``) plus a ``result.json`` into ``work_dir``.
    Raises on failure so the job exits non-zero.
    """
    import BioSimSpace as BSS

    work_dir = pathlib.Path(work_dir)
    spec = PrepStageSpec.load(work_dir / PREP_STAGE_SPEC_FILENAME)

    system = BSS.IO.readMolecules([spec.input_prm7, spec.input_rst7])
    protocol = build_protocol(
        kind=spec.kind,
        minimisation_steps=spec.minimisation_steps,
        runtime_ns=spec.runtime_ns,
        temperature_start_K=spec.temperature_start_K,
        temperature_end_K=spec.temperature_end_K,
        pressure=spec.pressure,
        restraint=spec.restraint,
    )
    process = BSS.Process.OpenMM(system, protocol, platform=spec.platform)
    process.start()
    process.wait()
    if process.isError():
        raise RuntimeError(f"prep stage {spec.stage!r} failed:\n{process.stdout(20)}")
    final = process.getSystem()

    prefix = work_dir / spec.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    BSS.IO.saveMolecules(str(prefix), final, ["prm7", "rst7"])

    trajectory = None
    if spec.save_trajectory and spec.kind == "equilibration":
        try:
            traj_path = prefix.with_suffix(".dcd")
            process.getTrajectory().getTrajectory(format="mdtraj").save(str(traj_path))
            trajectory = str(traj_path)
        except Exception:  # noqa: BLE001 - trajectory is best-effort; the stage still succeeds
            trajectory = None

    (work_dir / PREP_STAGE_RESULT_FILENAME).write_text(
        json.dumps(
            {
                "stage": spec.stage,
                "kind": spec.kind,
                "prm7": f"{prefix}.prm7",
                "rst7": f"{prefix}.rst7",
                "trajectory": trajectory,
            },
            indent=2,
        )
    )
