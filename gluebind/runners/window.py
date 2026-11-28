"""Window runner: one bias-potential centre, sampled once, for one replicate.

A :class:`Window` owns one directory, ``<stage>/run_NN/<centre>``. The replicate
level sits above it, on the :class:`~gluebind.runners.stage.Stage`, because a
replicate may have its own equilibrated structure, anchors and therefore its own
window centres — so each replicate owns a complete window set rather than a
column of a shared grid. The window writes its ``window.json`` via the injected
``spec_builder`` and emits the :class:`~gluebind.backend.base.JobSpec` whose
command runs :func:`gluebind.simulation.window.run_window` in that directory.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from gluebind.backend.base import JobSpec
from gluebind.runners.base import SimulationRunner
from gluebind.simulation.window import (
    RESULT_FILENAME,
    WINDOW_SPEC_FILENAME,
    WindowSpec,
    window_launch_command,
)

# spec_builder(*, cv_type, stage_name, dof, cv_centre, replicate, boresch_eq_values)
#   -> WindowSpec
SpecBuilder = Callable[..., WindowSpec]
CommandFactory = Callable[[], list[str]]

_UNIT_SUFFIX = {"boresch": "rad", "rmsd": "A", "separation": "nm"}


def format_label(cv_type: str, centre: float) -> str:
    """Deterministic window-directory label, e.g. ``0.85rad`` / ``9A`` / ``1.5nm``."""
    return f"{centre:.4g}{_UNIT_SUFFIX[cv_type]}"


def enumerate_centres(schedule) -> list[float]:
    """Window centres from a :class:`WindowSampling` schedule.

    Uses explicit ``centres`` if given, else a grid over
    ``[window_min, window_max]`` at ``window_spacing``. When ``coarse_from`` /
    ``coarse_spacing`` are set (the separation CV's two-phase schedule), the grid
    is fine (``window_spacing``) up to ``coarse_from`` and coarse
    (``coarse_spacing``) beyond it.
    """
    if isinstance(schedule.centres, dict):
        raise ValueError(
            "per-DoF centres mappings are only valid for Boresch schedules"
        )
    if schedule.centres is not None:
        return [round(c, 4) for c in schedule.centres]
    if schedule.window_min == "auto":
        raise ValueError(
            "window_min 'auto' is resolved from the equilibration trajectory during "
            "restraint resolution; resolve it before enumerating windows"
        )
    if (
        not schedule.window_spacing
        or schedule.window_min is None
        or schedule.window_max is None
    ):
        raise ValueError(
            "cannot enumerate windows: provide centres or "
            "(window_min, window_max, window_spacing)"
        )

    two_phase = schedule.coarse_from is not None and schedule.coarse_spacing
    fine_end = schedule.coarse_from if two_phase else schedule.window_max
    n_fine = int(round((fine_end - schedule.window_min) / schedule.window_spacing))
    centres = [
        round(schedule.window_min + i * schedule.window_spacing, 4)
        for i in range(n_fine + 1)
    ]

    if two_phase:
        c = fine_end + schedule.coarse_spacing
        while c <= schedule.window_max + 1e-9:
            centres.append(round(c, 4))
            c += schedule.coarse_spacing
    return centres


class Window(SimulationRunner):
    """One umbrella-sampling window: a single bias centre, for one replicate."""

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        *,
        cv_type: str,
        stage_name: str,
        dof: str | None,
        centre: float,
        replicate: int,
        spec_builder: SpecBuilder,
        command_factory: CommandFactory = window_launch_command,
    ) -> None:
        super().__init__(base_dir)
        self.cv_type = cv_type
        self.stage_name = stage_name
        self.dof = dof
        self.centre = centre
        self.replicate = replicate
        self.spec_builder = spec_builder
        self.command_factory = command_factory

    @property
    def label(self) -> str:
        return format_label(self.cv_type, self.centre)

    def write_spec(self, boresch_eq_values: dict | None = None) -> None:
        """Write this window's ``window.json`` via the ``spec_builder``.

        ``boresch_eq_values`` carries this replicate's already-determined Boresch
        equilibrium values; it is threaded to the ``spec_builder`` (which resolves
        it into the window's fixed restraints), enabling the sequential Boresch
        feedback. Called at run time, not in ``setup``, because a DoF's value is
        only known after the previous DoF's PMF has been analysed.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        spec = self.spec_builder(
            cv_type=self.cv_type,
            stage_name=self.stage_name,
            dof=self.dof,
            cv_centre=self.centre,
            replicate=self.replicate,
            boresch_eq_values=dict(boresch_eq_values or {}),
        )
        spec.dump(self.base_dir / WINDOW_SPEC_FILENAME)

    def is_complete(self) -> bool:
        return (self.base_dir / RESULT_FILENAME).exists()

    def job_spec(self) -> JobSpec:
        return JobSpec(
            command=self.command_factory(),
            work_dir=str(self.base_dir),
            name=f"{self.stage_name}_{self.label}_r{self.replicate}",
        )
