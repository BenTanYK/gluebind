"""Job-submission backends.

The :class:`Backend` seam places one window's job on compute and reports its
state via opaque handles. Built-in implementations are :class:`LocalBackend`
(testing/CI), :class:`SlurmBackend`, and :class:`GridEngineBackend`.
:class:`Scheduler` submits many window jobs through a backend and waits for them.
"""

from __future__ import annotations

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState
from gluebind.backend.grid_engine import GridEngineBackend
from gluebind.backend.local import LocalBackend
from gluebind.backend.scheduler import Scheduler, SlotPool
from gluebind.backend.slurm import SlurmBackend

__all__ = [
    "Backend",
    "JobSpec",
    "JobState",
    "JobHandle",
    "LocalBackend",
    "GridEngineBackend",
    "SlurmBackend",
    "Scheduler",
    "SlotPool",
]
