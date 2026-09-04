"""Cluster-scoped SLURM submission configuration.

Lifted, from a3fe's ``SlurmConfig`` (the pattern is domain-
agnostic): pydantic fields for the sbatch directives, an f-string script
renderer, and YAML dump/load. Kept separate from :class:`CalculationConfig`
because it describes the *machine*, not the calculation, and is reused across
every run on a given cluster.

Original a3fe source code:
https://github.com/michellab/a3fe/blob/main/a3fe/configuration/slurm_config.py

``queue_len_lim`` is exposed here (a3fe hard-codes it) because the scheduler
throttles how many jobs sit in the real SLURM queue at once.
"""

from __future__ import annotations

import pathlib

import pydantic
import yaml

SLURM_CONFIG_FILENAME = "slurm_config.yaml"


class SlurmConfig(pydantic.BaseModel):
    """Parameters controlling how jobs are submitted to SLURM."""

    model_config = pydantic.ConfigDict(validate_assignment=True)

    partition: str = pydantic.Field("main", description="SLURM partition to submit to.")
    time: str = pydantic.Field("24:00:00", description="Time limit for each SLURM job.")
    memory: str | None = pydantic.Field(
        "4G",
        description=(
            "Memory to request for each SLURM job; use None to accept the "
            "partition default."
        ),
    )
    gres: str = pydantic.Field(
        "gpu:1", description="Generic resources to request, normally one GPU."
    )
    nodes: int = pydantic.Field(
        1, ge=1, description="Number of nodes to request for each SLURM job."
    )
    ntasks_per_node: int = pydantic.Field(
        1, ge=1, description="Number of tasks to run on each allocated node."
    )
    output: str = pydantic.Field(
        "slurm-%A.%a.out", description="Output file pattern for each SLURM job."
    )
    extra_options: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="Additional sbatch options rendered as key-value directives.",
    )
    queue_check_interval: int = pydantic.Field(
        30, ge=1, description="Seconds between SLURM queue polls."
    )
    job_submission_wait: int = pydantic.Field(
        300,
        ge=1,
        description="Seconds to wait for a submitted job to appear in the queue.",
    )
    queue_len_lim: int = pydantic.Field(
        2000,
        ge=1,
        description="Maximum number of jobs allowed in the SLURM queue at once.",
    )

    def render_script(self, cmd: str) -> str:
        """Render an sbatch script body for ``cmd``."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --time={self.time}",
            *([f"#SBATCH --mem={self.memory}"] if self.memory else []),
            f"#SBATCH --gres={self.gres}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --output={self.output}",
        ]
        lines += [f"#SBATCH --{k}={v}" for k, v in self.extra_options.items()]
        lines += ["", cmd, ""]
        return "\n".join(lines)

    def write_submission_script(
        self, cmd: str, run_dir: str | pathlib.Path, script_name: str = "gluebind"
    ) -> pathlib.Path:
        """Write the sbatch script into ``run_dir`` and return its path."""
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / f"{script_name}.sh"
        script_path.write_text(self.render_script(cmd))
        return script_path

    def get_submission_cmds(
        self, cmd: str, run_dir: str | pathlib.Path, script_name: str = "gluebind"
    ) -> list[str]:
        """Write the script and return the ``sbatch`` command list."""
        script_path = self.write_submission_script(cmd, run_dir, script_name)
        return ["sbatch", f"--chdir={run_dir}", str(script_path)]

    def slurm_output_glob(self, run_dir: str | pathlib.Path) -> str:
        """Glob matching a job's SLURM ``.out`` file(s) in ``run_dir``."""
        base = self.output.split("%")[0]
        return str(pathlib.Path(run_dir) / f"{base}*")

    def dump(self, save_dir: str | pathlib.Path) -> pathlib.Path:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / SLURM_CONFIG_FILENAME
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
        return path

    @classmethod
    def load(cls, load_dir: str | pathlib.Path) -> "SlurmConfig":
        path = pathlib.Path(load_dir) / SLURM_CONFIG_FILENAME
        with open(path) as f:
            return cls(**yaml.safe_load(f))
