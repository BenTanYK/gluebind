"""Local (subprocess) backend — for testing and CI.

Runs each job as a child process on the local machine. Because the jobs are
children of the driver, they do *not* survive it (``detached = False``); this
backend is a convenience for exercising the orchestration and analysis layers on
a laptop without a cluster, not a production execution path.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import subprocess

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState


class LocalBackend(Backend):
    """Execute jobs as local subprocesses."""

    detached = False

    def __init__(self) -> None:
        self._jobs: dict[str, tuple[subprocess.Popen, "os.PathLike | object"]] = {}
        self._counter = itertools.count(1)

    def submit(self, spec: JobSpec) -> JobHandle:
        work_dir = pathlib.Path(spec.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        token = f"local-{next(self._counter)}"
        env = {**os.environ, **spec.env}
        out = open(work_dir / f"{spec.name}.out", "w")
        proc = subprocess.Popen(
            spec.command,
            cwd=str(work_dir),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        self._jobs[token] = (proc, out)
        return token

    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        result: dict[JobHandle, JobState] = {}
        for handle in handles:
            job = self._jobs.get(handle)
            if job is None:
                result[handle] = JobState.FAILED
                continue
            proc, out = job
            code = proc.poll()
            if code is None:
                result[handle] = JobState.RUNNING
            else:
                if not out.closed:
                    out.close()
                result[handle] = JobState.FINISHED if code == 0 else JobState.FAILED
        return result

    def cancel(self, handle: JobHandle) -> None:
        job = self._jobs.get(handle)
        if job is not None and job[0].poll() is None:
            job[0].terminate()
