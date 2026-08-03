"""Backend worker for raw-input system construction.

This is the one compute unit which parameterises the glue, assembles the
glue-receptor-target complex, solvates it, and writes AMBER inputs. It is
submission-agnostic: LocalBackend and SlurmBackend invoke exactly this worker.
"""

from __future__ import annotations

import pathlib

import pydantic

from gluebind.config.calculation import CalculationConfig

SYSTEM_BUILD_SPEC_FILENAME = "system_build.json"
SYSTEM_BUILD_RESULT_FILENAME = "result.json"


class SystemBuildSpec(pydantic.BaseModel):
    """Everything required to build one solvated ternary-complex system."""

    model_config = pydantic.ConfigDict(extra="forbid")

    config: CalculationConfig
    output_dir: str

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "SystemBuildSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def system_build_launch_command(python: str = "python") -> list[str]:
    """Return the command a backend executes in the build work directory."""
    code = (
        "from gluebind.simulation.system_build import run_system_build; "
        "run_system_build('.')"
    )
    return [python, "-c", code]


def run_system_build(work_dir: str | pathlib.Path) -> None:
    """Build the solvated complex described by ``system_build.json``."""
    from gluebind.system.prep import build_solvated_system

    work_dir = pathlib.Path(work_dir)
    spec = SystemBuildSpec.load(work_dir / SYSTEM_BUILD_SPEC_FILENAME)
    result = build_solvated_system(spec.config, spec.output_dir)
    result.dump(work_dir / SYSTEM_BUILD_RESULT_FILENAME)
