"""Cluster-scoped SLURM submission configuration.

Lifted, near-verbatim, from a3fe's ``SlurmConfig`` (the pattern is domain-
agnostic): pydantic fields for the sbatch directives, an f-string script
renderer, and YAML dump/load. Kept separate from :class:`CalculationConfig`
because it describes the *machine*, not the calculation, and is reused across
every run on a given cluster.

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

    partition: str = "default"
    time: str = "24:00:00"
    gres: str = "gpu:1"
    nodes: int = pydantic.Field(1, ge=1)
    ntasks_per_node: int = pydantic.Field(1, ge=1)
    output: str = "slurm-%A.%a.out"
    extra_options: dict[str, str] = pydantic.Field(default_factory=dict)
    queue_check_interval: int = pydantic.Field(30, ge=1)
    """Seconds between ``squeue`` polls."""
    job_submission_wait: int = pydantic.Field(300, ge=1)
    """Seconds to wait for a submitted job to appear in the real queue."""
    queue_len_lim: int = pydantic.Field(2000, ge=1)
    """Max jobs the scheduler keeps in the real SLURM queue at once."""

    def render_script(self, cmd: str) -> str:
        """Render an sbatch script body for ``cmd``."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --time={self.time}",
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
