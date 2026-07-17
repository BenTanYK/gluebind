"""The submission backend seam.

A :class:`Backend` places a :class:`JobSpec` on compute and lets the caller poll
and cancel it. It is the same-process analogue of the openfe client->runner
boundary: the caller hands over a command (typically a ``python -c`` invocation
of :func:`gluebind.simulation.window.run_window`) plus a working directory, and
gets back an *opaque* handle it later polls — never inspecting the handle's
internals, so SLURM job ids, local tokens and (future) AWS Batch job ids are all
interchangeable.

Two implementations ship with gluebind — :class:`~gluebind.backend.local.LocalBackend`
(testing/CI) and :class:`~gluebind.backend.slurm.SlurmBackend` (the benchmarked
default). A third, an ``AWSBatchBackend``, is intended to be written downstream
(e.g. within Aqemia) by implementing these same three methods on top of a Batch
*client* + *runner* pair, exactly as aqemia-abfe drives openfe-client. To make
that a drop-in, two things in this module are deliberately Batch-forward:

* the handle is opaque (a plain ``str``), and
* :class:`JobSpec` carries ``inputs``/``outputs`` staging manifests that are
  no-ops on a shared filesystem (local/SLURM) but tell a Batch backend which
  files to push to / pull from S3.
"""

from __future__ import annotations

import abc
import dataclasses
import enum


class JobState(enum.Enum):
    """Backend-neutral job lifecycle state."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.FINISHED, JobState.FAILED)


@dataclasses.dataclass(frozen=True)
class Resources:
    """Backend-neutral resource request.

    Maps to a local device selection, a SLURM ``--gres``/partition, or an AWS
    Batch job-definition override, depending on the backend.
    """

    n_gpus: int = 1
    n_cpus: int = 1
    memory_gb: float | None = None
    walltime: str | None = None


@dataclasses.dataclass
class JobSpec:
    """A single unit of work to place on compute."""

    command: list[str]
    """The command to run, e.g. ``["python", "-c", "...run_window(...)"]``."""
    work_dir: str
    """Directory the command runs in and reads/writes its files under."""
    resources: Resources = dataclasses.field(default_factory=Resources)
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    name: str = "gluebind"
    inputs: list[str] = dataclasses.field(default_factory=list)
    """Files the job needs present in ``work_dir``. No-op for shared-filesystem
    backends; an AWS Batch backend stages these to S3 on submit."""
    outputs: list[str] = dataclasses.field(default_factory=list)
    """Files to retrieve after the job. No-op for shared-filesystem backends; an
    AWS Batch backend syncs these from S3 back into ``work_dir`` on completion,
    so the rest of gluebind reads results from the filesystem uniformly."""


JobHandle = str
"""Opaque, backend-specific token identifying a submitted job."""


class Backend(abc.ABC):
    """Places jobs on compute and reports their state."""

    detached: bool = False
    """Whether submitted jobs survive the driver process exiting. True for SLURM
    (and Batch); False for local. Determines whether a run can be resumed by a
    fresh process reconciling against the backend's live queue."""

    @abc.abstractmethod
    def submit(self, spec: JobSpec) -> JobHandle:
        """Submit ``spec`` and return an opaque handle."""

    @abc.abstractmethod
    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        """Return the current state of each handle."""

    @abc.abstractmethod
    def cancel(self, handle: JobHandle) -> None:
        """Best-effort cancellation of a submitted job."""
