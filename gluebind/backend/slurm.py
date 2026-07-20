"""SLURM backend — the v1 execution path.

Scaffolds one umbrella-sampling window into one sbatch job and submits it. Many
such jobs are submitted independently and SLURM spreads them across the nodes of
the configured partition (targeting specific nodes is done with a ``nodelist``
entry in :attr:`SlurmConfig.extra_options`). Submitted jobs are *detached* — they
outlive the driver — so a run can be resumed by reconciling handles against
``squeue``.
"""

from __future__ import annotations

import getpass
import subprocess
import threading
import time
from collections.abc import Callable

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState
from gluebind.config.slurm import SlurmConfig


class SlurmBackend(Backend):
    """Submit each job as a single-window sbatch job."""

    detached = True

    def __init__(
        self, config: SlurmConfig, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self._clock = clock
        # A just-submitted job may not yet be visible in squeue. Track when each
        # job (submitted by *this* process) was submitted, and whether it has ever
        # been seen in the queue, so poll() does not mistake "not yet appeared" for
        # "finished" during the job_submission_wait grace window.
        self._submitted_at: dict[JobHandle, float] = {}
        self._seen: set[JobHandle] = set()
        self._lock = threading.Lock()  # a parallel CalcSet shares one backend

    def submit(self, spec: JobSpec) -> JobHandle:
        cmd = " ".join(spec.command)
        submission = self.config.get_submission_cmds(
            cmd, spec.work_dir, script_name=spec.name
        )
        proc = subprocess.run(submission, capture_output=True, text=True, check=True)
        handle = self._parse_job_id(proc.stdout)
        with self._lock:
            self._submitted_at[handle] = self._clock()
        return handle

    @staticmethod
    def _parse_job_id(sbatch_stdout: str) -> JobHandle:
        """Extract the job id from ``sbatch`` output (``Submitted batch job N``)."""
        tokens = sbatch_stdout.split()
        if not tokens:
            raise RuntimeError(
                f"could not parse job id from sbatch output: {sbatch_stdout!r}"
            )
        return tokens[-1]

    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        """Report each handle as RUNNING (still in the queue) or FINISHED (left it).

        ``squeue`` reports *presence*, not exit status, so a handle that has left
        the queue is reported FINISHED; whether it actually succeeded is
        reconciled by the caller from the job's output files. This mirrors the
        filesystem-as-truth resume model.

        A job submitted by this process is held RUNNING until it either appears in
        the queue or ``job_submission_wait`` seconds elapse — so the scheduler does
        not read a not-yet-visible job as finished (which the caller's result-file
        gate would then flag as a false failure). Handles not submitted by this
        process (a resumed run) have no grace basis, so absence means FINISHED.
        """
        running = self._running_job_ids()
        now = self._clock()
        grace = self.config.job_submission_wait
        result: dict[JobHandle, JobState] = {}
        with self._lock:
            for h in handles:
                if h in running:
                    self._seen.add(h)
                    result[h] = JobState.RUNNING
                elif h in self._seen:
                    result[h] = JobState.FINISHED  # appeared, then left the queue
                elif h in self._submitted_at and now - self._submitted_at[h] < grace:
                    result[h] = JobState.RUNNING  # submitted here, not yet visible
                else:
                    result[h] = JobState.FINISHED  # grace elapsed, or a resumed handle
        return result

    def _running_job_ids(self) -> set[str]:
        proc = subprocess.run(
            ["squeue", "-h", "-o", "%i", "-u", getpass.getuser(), "-t", "R,PD,S,CG"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}

    def cancel(self, handle: JobHandle) -> None:
        subprocess.run(["scancel", handle], check=False)
