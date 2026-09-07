"""Cluster-scoped Grid Engine submission configuration."""

from __future__ import annotations

import pathlib

import pydantic
import yaml

from gluebind.config.scheduler import SchedulerConfig

GRID_ENGINE_CONFIG_FILENAME = "grid_engine_config.yaml"


def _without_newlines(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("Grid Engine directive values must not contain newlines")
    return value


class GridEngineConfig(SchedulerConfig):
    """Parameters controlling how jobs are submitted to Grid Engine."""

    queue: str | None = pydantic.Field("gpu", description="Queue to submit to.")
    time: str | None = pydantic.Field(
        "24:00:00", description="Wall-clock limit rendered as ``-l h_rt``."
    )
    memory: str | None = pydantic.Field(
        "4G", description="Memory request rendered as ``-l h_vmem``."
    )
    resources: dict[str, str] = pydantic.Field(
        default_factory=lambda: {"gpu": "1"},
        description="Additional resource requests rendered as ``-l key=value``.",
    )
    output: str | None = pydantic.Field(
        None, description="Optional Grid Engine output-file destination."
    )
    join_output: bool = pydantic.Field(
        True, description="Whether to merge standard error into standard output."
    )
    shell: str | None = pydantic.Field(
        "/bin/bash", description="Shell path rendered as ``-S``."
    )
    preamble: list[str] = pydantic.Field(
        default_factory=list, description="Shell commands before the GlueBind command."
    )
    extra_directives: list[str] = pydantic.Field(
        default_factory=list,
        description=(
            "Additional Grid Engine directive bodies, for example ``-P project``."
        ),
    )

    @pydantic.field_validator("queue", "time", "memory", "output", "shell")
    @classmethod
    def _validate_directive_value(cls, value: str | None) -> str | None:
        return None if value is None else _without_newlines(value)

    @pydantic.field_validator("resources")
    @classmethod
    def _validate_resources(cls, value: dict[str, str]) -> dict[str, str]:
        return {_without_newlines(k): _without_newlines(v) for k, v in value.items()}

    @pydantic.field_validator("extra_directives")
    @classmethod
    def _validate_extra_directives(cls, value: list[str]) -> list[str]:
        return [_without_newlines(v) for v in value]

    def render_script(self, cmd: str, *, job_name: str = "gluebind") -> str:
        """Render a qsub script body for ``cmd`` and ``job_name``."""
        _without_newlines(job_name)
        lines = ["#!/bin/bash", "#$ -cwd", f"#$ -N {job_name}"]
        if self.queue:
            lines.append(f"#$ -q {self.queue}")
        if self.time:
            lines.append(f"#$ -l h_rt={self.time}")
        if self.memory:
            lines.append(f"#$ -l h_vmem={self.memory}")
        lines.extend(f"#$ -l {key}={value}" for key, value in self.resources.items())
        if self.output:
            lines.append(f"#$ -o {self.output}")
        if self.join_output:
            lines.append("#$ -j y")
        if self.shell:
            lines.append(f"#$ -S {self.shell}")
        lines.extend(f"#$ {directive}" for directive in self.extra_directives)
        lines.extend(["", *self.preamble, cmd, ""])
        return "\n".join(lines)

    def write_submission_script(
        self, cmd: str, run_dir: str | pathlib.Path, script_name: str = "gluebind"
    ) -> pathlib.Path:
        """Write a qsub script into ``run_dir`` and return its path."""
        _without_newlines(script_name)
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / f"{script_name}.sh"
        script_path.write_text(self.render_script(cmd, job_name=script_name))
        return script_path

    def get_submission_cmds(
        self, cmd: str, run_dir: str | pathlib.Path, script_name: str = "gluebind"
    ) -> list[str]:
        """Write the script and return the qsub command list.

        The caller must execute this command with ``cwd=run_dir`` so ``#$ -cwd``
        selects the job's self-contained workspace.
        """
        script_path = self.write_submission_script(cmd, run_dir, script_name)
        return ["qsub", str(script_path)]

    def dump(self, save_dir: str | pathlib.Path) -> pathlib.Path:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / GRID_ENGINE_CONFIG_FILENAME
        path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
        return path

    @classmethod
    def load(cls, load_path: str | pathlib.Path) -> "GridEngineConfig":
        """Load configuration from a YAML file or a directory containing it."""
        path = pathlib.Path(load_path)
        if path.is_dir():
            path /= GRID_ENGINE_CONFIG_FILENAME
        with open(path) as f:
            return cls(**(yaml.safe_load(f) or {}))
