"""Job-submission backends.

The :class:`Backend` seam places one window's job on compute and reports its
state via opaque handles. Two implementations ship: :class:`LocalBackend`
(testing/CI, and the reference "run one window" operation a downstream runner
wraps) and :class:`SlurmBackend` (the v1 execution path). :class:`Scheduler`
submits many window jobs through a backend and waits for them.
"""

from __future__ import annotations

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState, Resources
from gluebind.backend.local import LocalBackend
from gluebind.backend.scheduler import Scheduler
from gluebind.backend.slurm import SlurmBackend

__all__ = [
    "Backend",
    "JobSpec",
    "JobState",
    "JobHandle",
    "Resources",
    "LocalBackend",
    "SlurmBackend",
    "Scheduler",
]
