"""gluebind — geometric-route umbrella-sampling ternary-complex binding free energies.

Package layout mirrors the development plan: :mod:`gluebind.config` (typed
configuration), :mod:`gluebind.state` (on-disk run state), :mod:`gluebind.backend`
(job submission), :mod:`gluebind.restraints` (OpenMM force builders),
:mod:`gluebind.selection` (anchor/equilibration analysis),
:mod:`gluebind.simulation` (window/steered-MD runners), :mod:`gluebind.runners`
(the nested orchestration hierarchy) and :mod:`gluebind.analysis` (WHAM +
free-energy assembly).

Only lightweight, dependency-safe objects are re-exported here. Heavy optional
dependencies (OpenMM, BioSimSpace, MDAnalysis) are imported lazily inside their own
subpackages so that ``import gluebind`` stays cheap and works in a bare environment.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gluebind")
except PackageNotFoundError:  # not installed (e.g. imported from a bare source tree)
    __version__ = "0.0.0+unknown"

from gluebind.config.calculation import CalculationConfig
from gluebind.config.slurm import SlurmConfig
from gluebind.runners.calc_set import CalcSet
from gluebind.runners.calculation import Calculation
from gluebind.spec_builder import RestraintContext, SpecBuilder
from gluebind.state import RunState

__all__ = [
    "CalculationConfig",
    "SlurmConfig",
    "RunState",
    "SpecBuilder",
    "RestraintContext",
    "Calculation",
    "CalcSet",
    "__version__",
]
