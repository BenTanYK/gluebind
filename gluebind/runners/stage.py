"""Stage runner: one collective variable → one PMF → one WHAM run.

A :class:`Stage` owns the set of windows spanning its CV and knows its
contribution's role in the free-energy sum (via ``cv_type`` and, for RMSD,
whether it is a bound or bulk stage).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from gluebind.runners.base import SimulationRunner
from gluebind.runners.window import CommandFactory, SpecBuilder, Window, format_label
from gluebind.simulation.window import window_launch_command


class Stage(SimulationRunner):
    """One CV's umbrella-sampling windows."""

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        *,
        cv_type: str,
        name: str,
        dof: str | None,
        centres: Sequence[float],
        ensemble_size: int,
        spec_builder: SpecBuilder,
        command_factory: CommandFactory = window_launch_command,
    ) -> None:
        super().__init__(base_dir)
        self.cv_type = cv_type
        self.name = name
        self.dof = dof
        self.windows = [
            Window(
                self.base_dir / format_label(cv_type, centre),
                cv_type=cv_type,
                stage_name=name,
                dof=dof,
                centre=centre,
                ensemble_size=ensemble_size,
                spec_builder=spec_builder,
                command_factory=command_factory,
            )
            for centre in centres
        ]
        self.sub_runners = list(self.windows)

    def write_specs(self, boresch_eq_values: dict | None = None) -> None:
        """Write every window's replicate specs with the given Boresch eq values."""
        for window in self.windows:
            window.write_specs(boresch_eq_values)

    @property
    def is_bulk(self) -> bool:
        """True for a released/bulk RMSD stage (enters the sum with ``+`` sign)."""
        return self.name.endswith("_bulk")
