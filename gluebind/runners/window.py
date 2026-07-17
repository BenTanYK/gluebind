"""Window runner: one bias-potential centre, sampled in ``ensemble_size`` replicates.

A :class:`Window` owns one directory per replicate (``run_01`` …). At setup it
writes each replicate's ``window.json`` via the injected ``spec_builder`` (which
Phase 3/4 supplies with resolved structures + restraint context), and it emits
one :class:`~gluebind.backend.base.JobSpec` per replicate whose command runs
:func:`gluebind.simulation.window.run_window` in that directory.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from gluebind.backend.base import JobSpec, Resources
from gluebind.runners.base import SimulationRunner
from gluebind.simulation.window import (
    RESULT_FILENAME,
    WINDOW_SPEC_FILENAME,
    WindowSpec,
    window_launch_command,
)

# spec_builder(*, cv_type, stage_name, dof, cv_centre, replicate, boresch_eq_values) -> WindowSpec
SpecBuilder = Callable[..., WindowSpec]
CommandFactory = Callable[[], list[str]]

_UNIT_SUFFIX = {"boresch": "rad", "rmsd": "A", "separation": "nm"}


def format_label(cv_type: str, centre: float) -> str:
    """Deterministic window-directory label, e.g. ``0.85rad`` / ``9A`` / ``1.5nm``."""
    return f"{centre:.4g}{_UNIT_SUFFIX[cv_type]}"


def enumerate_centres(schedule) -> list[float]:
    """Window centres from a :class:`WindowSampling` schedule.

    Uses explicit ``centres`` if given, else ``[window_min, window_max]`` at
    ``window_spacing``. Boresch and separation stages, whose ranges come from the
    MD distribution / SMD frames, must supply explicit centres.
    """
    if schedule.centres is not None:
        return [round(c, 4) for c in schedule.centres]
    if (
        schedule.window_spacing
        and schedule.window_min is not None
        and schedule.window_max is not None
    ):
        n = int(round((schedule.window_max - schedule.window_min) / schedule.window_spacing))
        return [round(schedule.window_min + i * schedule.window_spacing, 4) for i in range(n + 1)]
    raise ValueError("cannot enumerate windows: provide centres or (window_min, window_max, window_spacing)")


class Window(SimulationRunner):
    """One umbrella-sampling window (a single bias centre) and its replicates."""

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        *,
        cv_type: str,
        stage_name: str,
        dof: str | None,
        centre: float,
        ensemble_size: int,
        spec_builder: SpecBuilder,
        command_factory: CommandFactory = window_launch_command,
        resources: Resources | None = None,
    ) -> None:
        super().__init__(base_dir)
        self.cv_type = cv_type
        self.stage_name = stage_name
        self.dof = dof
        self.centre = centre
        self.ensemble_size = ensemble_size
        self.spec_builder = spec_builder
        self.command_factory = command_factory
        self.resources = resources or Resources()

    @property
    def label(self) -> str:
        return format_label(self.cv_type, self.centre)

    def replicates(self) -> range:
        return range(1, self.ensemble_size + 1)

    def replicate_dir(self, replicate: int) -> pathlib.Path:
        return self.base_dir / f"run_{replicate:02d}"

    def write_specs(self, boresch_eq_values: dict | None = None) -> None:
        """Write each replicate's ``window.json`` via the ``spec_builder``.

        ``boresch_eq_values`` carries the equilibrium values of the Boresch DoFs
        already determined; it is threaded to the ``spec_builder`` (which resolves
        it into the window's fixed restraints), enabling the sequential Boresch
        feedback. Called at run time, not in ``setup``, because a DoF's value is
        only known after the previous DoF's PMF has been analysed.
        """
        boresch_eq_values = dict(boresch_eq_values or {})
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for replicate in self.replicates():
            run_dir = self.replicate_dir(replicate)
            run_dir.mkdir(parents=True, exist_ok=True)
            spec = self.spec_builder(
                cv_type=self.cv_type,
                stage_name=self.stage_name,
                dof=self.dof,
                cv_centre=self.centre,
                replicate=replicate,
                boresch_eq_values=boresch_eq_values,
            )
            spec.dump(run_dir / WINDOW_SPEC_FILENAME)

    def is_replicate_complete(self, replicate: int) -> bool:
        return (self.replicate_dir(replicate) / RESULT_FILENAME).exists()

    def job_spec(self, replicate: int) -> JobSpec:
        return JobSpec(
            command=self.command_factory(),
            work_dir=str(self.replicate_dir(replicate)),
            name=f"{self.stage_name}_{self.label}_r{replicate}",
            resources=self.resources,
        )
