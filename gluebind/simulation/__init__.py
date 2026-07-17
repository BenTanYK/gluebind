"""Per-window compute: the single-window entry point and steered MD.

:func:`gluebind.simulation.window.run_window` is *the* compute unit — it runs one
umbrella-sampling window from a self-contained :class:`WindowSpec` in a working
directory, knowing nothing about the scheduler, SLURM, S3 or environment
variables. A SLURM/local backend invokes it via a ``python -c`` command; a
downstream AWS Batch runner imports and calls it directly. OpenMM is imported
lazily inside the functions, so importing this subpackage does not require it.
"""

from __future__ import annotations

from gluebind.simulation.prep_stage import (
    PrepStageSpec,
    prep_stage_launch_command,
    run_prep_stage,
)
from gluebind.simulation.steered_md import (
    make_frame_generator,
    run_steered_md,
    separation_window_targets,
)
from gluebind.simulation.window import WindowSpec, run_window, window_launch_command

__all__ = [
    "WindowSpec",
    "run_window",
    "window_launch_command",
    "PrepStageSpec",
    "run_prep_stage",
    "prep_stage_launch_command",
    "run_steered_md",
    "separation_window_targets",
    "make_frame_generator",
]
