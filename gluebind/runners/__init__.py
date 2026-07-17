"""The nested runner hierarchy that orchestrates a calculation.

``Calculation → Group → Stage → Window``: :class:`Calculation` builds the tree
from a :class:`~gluebind.config.calculation.CalculationConfig`, submits each
window replicate through a :class:`~gluebind.backend.base.Backend`, records opaque
handles in ``.gluebind-state.json``, resumes from on-disk completion, and
aggregates per-stage PMFs into the standard-state binding free energy.
"""

from __future__ import annotations

from gluebind.runners.base import SimulationRunner
from gluebind.runners.calculation import Calculation
from gluebind.runners.group import Group
from gluebind.runners.stage import Stage
from gluebind.runners.window import Window, enumerate_centres, format_label

__all__ = [
    "SimulationRunner",
    "Calculation",
    "Group",
    "Stage",
    "Window",
    "enumerate_centres",
    "format_label",
]
