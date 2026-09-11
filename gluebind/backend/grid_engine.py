"""Grid Engine execution backend."""

from __future__ import annotations

import getpass
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState
from gluebind.config.grid_engine import GridEngineConfig


class GridEngineBackend(Backend):
    """Submit detached GlueBind jobs with Grid Engine's ``qsub`` command."""

    detached = True

    def __init__(
        self, config: GridEngineConfig, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self._clock = clock
        self._submitted_at: dict[JobHandle, float] = {}
        self._seen: set[JobHandle] = set()
        self._lock = threading.Lock()

    def submit(self, spec: JobSpec) -> JobHandle:
        cmd = shlex.join(spec.command)
        submission = self.config.get_submission_cmds(
            cmd, spec.work_dir, job_name=spec.name
        )
        try:
            proc = subprocess.run(
                submission,
                cwd=spec.work_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Grid Engine submission failed for job {spec.name!r}.\n"
                f"stdout:\n{exc.stdout or exc.output or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ) from exc
        handle = self._parse_job_id(proc.stdout)
        with self._lock:
            self._submitted_at[handle] = self._clock()
        return handle

    @staticmethod
    def _parse_job_id(qsub_stdout: str) -> JobHandle:
        """Extract a job ID from normal or ``-terse`` qsub output."""
        terse = re.fullmatch(r"\s*(\d+)\s*", qsub_stdout)
        normal = re.search(r"\bYour job\s+(\d+)\b", qsub_stdout)
        match = terse or normal
        if match is None:
            raise RuntimeError(
                f"could not parse job id from qsub output: {qsub_stdout!r}"
            )
        return match.group(1)

    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        """Report visible Grid Engine states, with a submission visibility grace."""
        visible = self._visible_job_states()
        now = self._clock()
        result: dict[JobHandle, JobState] = {}
        with self._lock:
            for handle in handles:
                if handle in visible:
                    self._seen.add(handle)
                    result[handle] = visible[handle]
                elif handle in self._seen:
                    result[handle] = JobState.FINISHED
                elif (
                    handle in self._submitted_at
                    and now - self._submitted_at[handle]
                    < self.config.job_submission_wait
                ):
                    result[handle] = JobState.RUNNING
                else:
                    result[handle] = JobState.FINISHED
        return result

    def _visible_job_states(self) -> dict[JobHandle, JobState]:
        proc = subprocess.run(
            ["qstat", "-u", getpass.getuser()],
            capture_output=True,
            text=True,
            check=True,
        )
        return self._parse_qstat(proc.stdout)

    @staticmethod
    def _parse_qstat(qstat_stdout: str) -> dict[JobHandle, JobState]:
        """Parse ordinary ``qstat -u USER`` output into backend states."""
        states: dict[JobHandle, JobState] = {}
        for line in qstat_stdout.splitlines():
            fields = line.split()
            if len(fields) < 5 or not re.fullmatch(r"\d+", fields[0]):
                continue
            handle, ge_state = fields[0], fields[4]
            # Eqw is the usual explicit error state. Treat any state containing
            # an error flag conservatively as FAILED as well.
            if "E" in ge_state:
                states[handle] = JobState.FAILED
            elif "q" in ge_state or "h" in ge_state or "w" in ge_state:
                states[handle] = JobState.PENDING
            else:
                # r, t, s, d, dr and site-specific visible active states all
                # remain live until they leave qstat.
                states[handle] = JobState.RUNNING
        return states

    def cancel(self, handle: JobHandle) -> None:
        subprocess.run(["qdel", handle], check=False)
