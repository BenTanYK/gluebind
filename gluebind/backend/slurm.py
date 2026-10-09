"""SLURM backend — the v1 execution path.

Scaffolds one umbrella-sampling window into one sbatch job and submits it. Many
such jobs are submitted independently and SLURM spreads them across the nodes of
the configured partition (targeting specific nodes is done with a ``nodelist``
entry in :attr:`SlurmConfig.extra_options`). Submitted jobs are *detached* — they
outlive the driver. A resumed driver polls the recorded handles against
``squeue`` and refuses to start while any are still queued or running.
"""

from __future__ import annotations

import getpass
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable

from gluebind.backend.base import (
    Backend,
    JobHandle,
    JobSpec,
    JobState,
    run_status_query,
)
from gluebind.config.slurm import SlurmConfig

_ENDED_STATES = {
    "COMPLETED": JobState.FINISHED,
    "FAILED": JobState.FAILED,
    "CANCELLED": JobState.FAILED,
    "TIMEOUT": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "PREEMPTED": JobState.FAILED,
    "BOOT_FAIL": JobState.FAILED,
    "DEADLINE": JobState.FAILED,
}
"""Slurm states in which a job has ended. Every other state (PENDING, RUNNING,
CONFIGURING, COMPLETING, REQUEUED, SUSPENDED, ... and any unknown future state) is
treated as live, so the driver waits rather than mistaking it for finished."""


class SlurmBackend(Backend):
    """Submit each job as a single-window sbatch job."""

    detached = True

    def __init__(
        self,
        config: SlurmConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._clock = clock
        self._sleep = sleep  # waits between retries of a failed squeue
        # A just-submitted job may not yet be visible in squeue. Track when each
        # job (submitted by *this* process) was submitted, and whether it has ever
        # been seen in the queue, so poll() does not mistake "not yet appeared" for
        # "finished" during the job_submission_wait grace window.
        self._submitted_at: dict[JobHandle, float] = {}
        self._seen: set[JobHandle] = set()
        self._lock = threading.Lock()  # a parallel CalcSet shares one backend

    def submit(self, spec: JobSpec) -> JobHandle:
        cmd = shlex.join(spec.command)
        submission = self.config.get_submission_cmds(cmd, spec.work_dir, spec.name)
        try:
            proc = subprocess.run(
                submission, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"SLURM submission failed for job {spec.name!r}.\n"
                f"stdout:\n{exc.stdout or exc.output or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ) from exc
        handle = self._parse_job_id(proc.stdout)
        with self._lock:
            self._submitted_at[handle] = self._clock()
        return handle

    @staticmethod
    def _parse_job_id(sbatch_stdout: str) -> JobHandle:
        """Extract the job id from ``sbatch --parsable`` output.

        That is ``<jobid>``, or ``<jobid>;<cluster>`` on multi-cluster sites. Any
        other output raises, rather than recording a wrong id that would never
        appear in the queue (and so read as a finished job).
        """
        lines = sbatch_stdout.strip().splitlines()
        match = re.fullmatch(r"(\d+)(;\S+)?", lines[0].strip()) if lines else None
        if match is None:
            raise RuntimeError(
                f"could not parse job id from sbatch output: {sbatch_stdout!r}"
            )
        return match.group(1)

    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        """Report each handle's state from ``squeue``.

        A job listed in an ended state (:data:`_ENDED_STATES`) is FINISHED or
        FAILED; PENDING is PENDING; every other listed state is RUNNING. Whether a
        job actually succeeded is decided by the caller from its output files.

        Ended jobs drop out of ``squeue`` after a while, so a job no longer listed
        is FINISHED — except one submitted by this process that has never been
        seen, which is held RUNNING for ``job_submission_wait`` seconds so a
        not-yet-visible job is not mistaken for finished. Handles not submitted by
        this process (a resumed run) have no grace basis, so absence means FINISHED.
        """
        queue = self._queue_states()
        now = self._clock()
        grace = self.config.job_submission_wait
        result: dict[JobHandle, JobState] = {}
        with self._lock:
            for h in handles:
                if h in queue:
                    self._seen.add(h)
                    slurm_state = queue[h]
                    if slurm_state in _ENDED_STATES:
                        result[h] = _ENDED_STATES[slurm_state]
                    elif slurm_state == "PENDING":
                        result[h] = JobState.PENDING
                    else:
                        result[h] = JobState.RUNNING
                elif h in self._seen:
                    result[h] = JobState.FINISHED  # appeared, then left the queue
                elif h in self._submitted_at and now - self._submitted_at[h] < grace:
                    result[h] = JobState.RUNNING  # submitted here, not yet visible
                else:
                    result[h] = JobState.FINISHED  # grace elapsed, or a resumed handle
        return result

    def _queue_states(self) -> dict[JobHandle, str]:
        """This user's jobs in ``squeue``, in every state: ``{job id: state}``."""
        stdout = run_status_query(
            ["squeue", "-h", "-t", "all", "-u", getpass.getuser(), "-o", "%i %T"],
            retries=self.config.poll_retries,
            wait_s=self.config.poll_retry_wait_s,
            sleep=self._sleep,
        )
        states: dict[JobHandle, str] = {}
        for line in stdout.splitlines():
            fields = line.split()
            if len(fields) >= 2:
                states[fields[0]] = fields[1]
        return states

    def cancel(self, handle: JobHandle) -> None:
        subprocess.run(["scancel", handle], check=False)
