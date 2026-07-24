"""The single, unified calculation configuration.

:class:`CalculationConfig` is what the user authors (one YAML file) and what
gets dumped, fully resolved, into every run directory for provenance. It bundles
the inputs, the system-prep parameters, the sampling protocol and the restraint
definitions. The SLURM configuration is deliberately *not* part of this object
(see :class:`gluebind.config.slurm.SlurmConfig`) because it is cluster-scoped.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Literal

import pydantic
import yaml

from gluebind.config.prep import PrepConfig
from gluebind.config.restraints import RestraintsConfig
from gluebind.config.sampling import SamplingConfig

_CONFIG = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

RESOLVED_CONFIG_FILENAME = "config_resolved.yaml"


class MoleculeInput(pydantic.BaseModel):
    """A pre-parameterised protein, as an AMBER topology/coordinate pair."""

    model_config = _CONFIG

    prm7: str
    rst7: str


class GlueInput(pydantic.BaseModel):
    """The molecular glue, supplied as an SDF, and the protein it belongs to.

    ``assign_to`` records which protein's RMSD CV the glue heavy atoms join
    (chosen by whichever protein the glue binds more strongly). It also drives
    the all-Cα default: the assigned protein's default CV includes the glue.

    Requirement: the glue residue must be named ``MOL`` in the SDF. gluebind
    resolves the glue's heavy atoms by that residue name throughout (matching the
    template convention), so any other name will silently miss the glue.
    """

    model_config = _CONFIG

    sdf: str
    assign_to: Literal["target", "receptor"]


class Inputs(pydantic.BaseModel):
    """The two proteins and the optional glue."""

    model_config = _CONFIG

    target: MoleculeInput
    receptor: MoleculeInput
    glue: GlueInput | None = None


class CalculationConfig(pydantic.BaseModel):
    """A complete gluebind calculation definition."""

    model_config = _CONFIG

    inputs: Inputs
    prep: PrepConfig = pydantic.Field(default_factory=PrepConfig)
    sampling: SamplingConfig = pydantic.Field(default_factory=SamplingConfig)
    restraints: RestraintsConfig = pydantic.Field(default_factory=RestraintsConfig)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CalculationConfig":
        """Load and validate a calculation config from a YAML file."""
        path = pathlib.Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"{path} is empty")
        return cls.model_validate(raw)

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write this config to ``path`` as YAML."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        return path

    def dump_resolved(self, run_dir: str | pathlib.Path) -> pathlib.Path:
        """Dump the fully-resolved config into a run directory for provenance."""
        return self.dump(pathlib.Path(run_dir) / RESOLVED_CONFIG_FILENAME)

    def with_resolved_input_paths(
        self, base: str | pathlib.Path
    ) -> "CalculationConfig":
        """Return a copy with relative input file paths resolved against ``base``.

        Lets a ``config.yaml`` live alongside its AMBER inputs in a self-contained
        directory (paths relative to that directory) — how CalcSet treats each
        per-system subdirectory, and how a single calc can be run from its own
        input directory. Absolute paths are left unchanged.
        """
        base = pathlib.Path(base)

        def _abs(path: str) -> str:
            p = pathlib.Path(path)
            return str(p if p.is_absolute() else (base / p).resolve())

        data = self.model_dump()
        inputs = data["inputs"]
        for molecule in ("target", "receptor"):
            inputs[molecule]["prm7"] = _abs(inputs[molecule]["prm7"])
            inputs[molecule]["rst7"] = _abs(inputs[molecule]["rst7"])
        if inputs.get("glue"):
            inputs["glue"]["sdf"] = _abs(inputs["glue"]["sdf"])
        return CalculationConfig.model_validate(data)

    @property
    def config_hash(self) -> str:
        """A stable SHA-256 over the canonical config, for drift detection.

        Persisted in the run state; a resume against a mutated config is caught
        by comparing this hash.

        ``sampling.run_rmsd_us`` is deliberately excluded: it is a scope flag (it
        only controls whether the RMSD US *stages* are built), not a physics
        parameter — no already-sampled window's physics depends on it. Excluding it
        lets a separation-PMF-only run (``run_rmsd_us=False``) be *upgraded* to the
        full cycle by flipping the flag and re-running, which resumes and simply
        adds the RMSD stages rather than aborting as config drift.
        """
        canonical = json.dumps(
            self.model_dump(mode="json", exclude={"sampling": {"run_rmsd_us"}}),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
