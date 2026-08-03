"""Typed configuration for a gluebind calculation.

Two independent config families:

- :class:`~gluebind.config.calculation.CalculationConfig` — the single calculation
  config: inputs + prep + sampling + restraints. Portable across clusters.
- :class:`~gluebind.config.slurm.SlurmConfig` — cluster-scoped submission parameters,
  deliberately separate because they are reused across every run on a machine.

The calculation config splits into method parameters (paper-derived defaults, rarely
touched) and system-specific structural input (restraints), but both serialise to a
single file. Every force constant lives in exactly one place
(:class:`~gluebind.config.sampling.WindowSampling`) so simulation and analysis cannot
silently disagree.
"""

from __future__ import annotations

from gluebind.config.calculation import (
    CalculationConfig,
    GlueInput,
    Inputs,
    MoleculeInput,
)
from gluebind.config.prep import PrepConfig
from gluebind.config.restraints import (
    AlwaysOnRestraint,
    BoreschSpec,
    RestraintsConfig,
    RmsdCVSpec,
)
from gluebind.config.sampling import SamplingConfig, WindowSampling
from gluebind.config.slurm import SlurmConfig

__all__ = [
    "CalculationConfig",
    "Inputs",
    "MoleculeInput",
    "GlueInput",
    "PrepConfig",
    "SamplingConfig",
    "WindowSampling",
    "RestraintsConfig",
    "RmsdCVSpec",
    "AlwaysOnRestraint",
    "BoreschSpec",
    "SlurmConfig",
]
