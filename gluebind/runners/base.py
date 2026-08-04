"""Base class for the nested runner hierarchy.

``Calculation → Group → Stage → Window`` all subclass :class:`SimulationRunner`,
which provides a working directory and recursive ``setup``. Kept deliberately
light (a3fe's equivalent is much heavier): the orchestration logic — submission,
state, analysis — lives on :class:`~gluebind.runners.calculation.Calculation`,
while ``Group``/``Stage``/``Window`` mostly encode the tree and on-disk layout.
"""

from __future__ import annotations

import pathlib


class SimulationRunner:
    """A node in the runner tree with a working directory and children."""

    def __init__(self, base_dir: str | pathlib.Path) -> None:
        # Every backend worker runs from its own work directory, so persisted
        # paths must be absolute rather than relative to the driver's cwd.
        self.base_dir = pathlib.Path(base_dir).expanduser().resolve()
        self.sub_runners: list[SimulationRunner] = []

    def setup(self) -> None:
        """Create this node's directory, then recurse into children."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for runner in self.sub_runners:
            runner.setup()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base_dir.name!r})"
