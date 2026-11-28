"""Stage runner: one collective variable → one PMF per replicate → one WHAM run.

A :class:`Stage` owns the windows spanning its CV and knows its contribution's
role in the free-energy sum (via ``cv_type`` and, for RMSD, whether it is a bound
or bulk stage).
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator, Mapping, Sequence

from gluebind.replicate import replicate_dirname
from gluebind.runners.base import SimulationRunner
from gluebind.runners.window import CommandFactory, SpecBuilder, Window, format_label
from gluebind.simulation.window import window_launch_command


class Stage(SimulationRunner):
    """One CV's umbrella-sampling windows, per replicate."""

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        *,
        cv_type: str,
        name: str,
        dof: str | None,
        centres: Mapping[int, Sequence[float]],
        spec_builders: Mapping[int, SpecBuilder],
        command_factory: CommandFactory = window_launch_command,
    ) -> None:
        super().__init__(base_dir)
        self.cv_type = cv_type
        self.name = name
        self.dof = dof
        self.spec_builders = dict(spec_builders)
        self.command_factory = command_factory
        self.windows: dict[int, list[Window]] = {}
        for replicate, replicate_centres in centres.items():
            self.add_windows(replicate_centres, replicate=replicate)

    @property
    def replicates(self) -> list[int]:
        return sorted(self.windows)

    def windows_for(self, replicate: int) -> list[Window]:
        """This replicate's complete window set, sorted by centre."""
        return self.windows[replicate]

    def all_windows(self) -> Iterator[Window]:
        """Every window of every replicate."""
        for replicate in self.replicates:
            yield from self.windows[replicate]

    def _make_window(self, centre: float, replicate: int) -> Window:
        spec_builder = self.spec_builders.get(replicate)
        if spec_builder is None:
            raise KeyError(
                f"stage {self.name!r} has no spec builder for replicate {replicate}"
            )
        return Window(
            self.base_dir
            / replicate_dirname(replicate)
            / format_label(self.cv_type, centre),
            cv_type=self.cv_type,
            stage_name=self.name,
            dof=self.dof,
            centre=centre,
            replicate=replicate,
            spec_builder=spec_builder,
            command_factory=self.command_factory,
        )

    def add_windows(
        self, centres: Sequence[float], *, replicate: int | None = None
    ) -> list[Window]:
        """Add windows at ``centres`` (skipping any already present), keeping each
        replicate's set sorted by centre. Returns the windows actually added.

        With ``replicate=None`` the centres are added to every replicate, which is
        the umbrella-sampling extensibility workflow: add intermediate or extended
        windows to a stage, then ``run`` (which resumes, submitting only the new
        windows) and re-``analyse`` (which includes them).
        """
        targets = self.replicates if replicate is None else [replicate]
        added = []
        for target in targets:
            windows = self.windows.setdefault(target, [])
            existing = {w.label for w in windows}
            for centre in centres:
                window = self._make_window(centre, target)
                if window.label in existing:
                    continue
                windows.append(window)
                existing.add(window.label)
                added.append(window)
            windows.sort(key=lambda w: w.centre)
        self.sub_runners = list(self.all_windows())
        return added

    def write_specs(self, boresch_eq_values: Mapping[int, dict] | None = None) -> None:
        """Write every window's spec, giving each replicate its own Boresch
        equilibrium values (each DoF's minimum comes from that replicate's PMF)."""
        boresch_eq_values = boresch_eq_values or {}
        for replicate in self.replicates:
            per_replicate = boresch_eq_values.get(replicate, {})
            for window in self.windows[replicate]:
                window.write_spec(per_replicate)

    @property
    def is_bulk(self) -> bool:
        """True for a released/bulk RMSD stage (enters the sum with ``+`` sign)."""
        return self.name.endswith("_bulk")
