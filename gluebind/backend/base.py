"""The submission backend seam.

A :class:`Backend` places a :class:`JobSpec` on compute and lets the caller poll
and cancel it. It is the same-process analogue of the openfe client->runner
boundary: the caller hands over a command (typically a ``python -c`` invocation
of :func:`gluebind.simulation.window.run_window`) plus a working directory, and
gets back an *opaque* handle it later polls — never inspecting the handle's
internals, so scheduler job ids, local tokens and (future) AWS Batch job ids are all
interchangeable.

Three implementations ship with gluebind — :class:`~gluebind.backend.local.LocalBackend`
(testing/CI), :class:`~gluebind.backend.slurm.SlurmBackend`, and
:class:`~gluebind.backend.grid_engine.GridEngineBackend`. An ``AWSBatchBackend``
is intended to be written downstream
by implementing these same three methods on top of a Batch *client* + *runner*
pair. To make that a drop-in, two things in this module are deliberately
Batch-forward:

* the handle is opaque (a plain ``str``), and
* :class:`JobSpec` carries ``inputs``/``outputs`` staging manifests that are
  no-ops on a shared filesystem (local/SLURM) but tell a Batch backend which
  files to push to / pull from S3. Resource allocation is deliberately owned
  by each backend's cluster configuration, so all GlueBind jobs on a backend
  use a uniform resource policy.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import subprocess
import time
from collections.abc import Callable

QUERY_TIMEOUT_S = 60.0
"""Seconds before a single queue-status query (squeue/qstat) is abandoned."""


def run_status_query(
    cmd: list[str],
    *,
    retries: int,
    wait_s: float,
    sleep: Callable[[float], None] = time.sleep,
    timeout_s: float = QUERY_TIMEOUT_S,
) -> str:
    """Run a read-only scheduler query and return its stdout, retrying failures.

    A busy or restarting controller makes ``squeue``/``qstat`` fail or hang for a
    while even though the jobs are fine, so a failed or timed-out query is retried
    up to ``retries`` attempts in total, waiting ``wait_s`` seconds and doubling the
    wait each time. Only after the last attempt is the error raised. Not for job
    submission: a submit that appeared to fail may still have been accepted, so
    retrying it could duplicate the job.
    """
    for attempt in range(1, retries + 1):
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=timeout_s
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt == retries:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise RuntimeError(
                    f"{cmd[0]} failed {retries} time(s) in a row; the scheduler "
                    f"appears unreachable. Last error: {detail}"
                ) from exc
            sleep(wait_s * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # pragma: no cover


class JobState(enum.Enum):
    """Backend-neutral job lifecycle state."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.FINISHED, JobState.FAILED)


@dataclasses.dataclass
class JobSpec:
    """A single unit of work to place on compute."""

    command: list[str]
    """The command to run, e.g. ``["python", "-c", "...run_window(...)"]``."""
    work_dir: str
    """Directory the command runs in and reads/writes its files under."""
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
    """Whether submitted jobs survive the driver process exiting. True for cluster
    schedulers (and Batch); False for local. Determines whether a run can be
    resumed by a fresh process reconciling against the backend's live queue."""

    @abc.abstractmethod
    def submit(self, spec: JobSpec) -> JobHandle:
        """Submit ``spec`` and return an opaque handle."""

    @abc.abstractmethod
    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        """Return the current state of each handle."""

    @abc.abstractmethod
    def cancel(self, handle: JobHandle) -> None:
        """Best-effort cancellation of a submitted job."""
