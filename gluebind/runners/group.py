"""Group runner: one CV type, the free-energy aggregation node.

A :class:`Group` collects the stages of one CV type (``boresch`` / ``rmsd`` /
``separation``) — the level at which that CV's contribution to the standard-state
binding free energy is assembled (the structural analogue of a3fe's ``Leg``).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from gluebind.runners.base import SimulationRunner
from gluebind.runners.stage import Stage


class Group(SimulationRunner):
    """The stages of one CV type."""

    def __init__(
        self, base_dir: str | pathlib.Path, *, cv_type: str, stages: Sequence[Stage]
    ) -> None:
        super().__init__(base_dir)
        self.cv_type = cv_type
        self.stages = list(stages)
        self.sub_runners = list(self.stages)
