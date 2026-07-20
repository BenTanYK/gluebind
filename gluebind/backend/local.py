"""Local (subprocess) backend — for testing, CI, and small local runs.

Runs each job as a child process on the local machine. Because the jobs are
children of the driver, they do *not* survive it (``detached = False``); this
backend is a convenience for exercising the orchestration and analysis layers on
a laptop or dev box without a cluster, not a production execution path.

By default jobs launch immediately with no concurrency limit. For running more
than one MD window at once on a multi-GPU box, pass ``gpu_ids`` to pin each job
to a GPU (via ``CUDA_VISIBLE_DEVICES``) and cap concurrency at the number of
GPUs, or ``max_concurrent`` to cap concurrency without pinning. When capped,
excess submissions queue and start (reusing a GPU as it frees) on each
:meth:`poll`, so no more than the cap run at once.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import subprocess
from collections.abc import Sequence

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState


class LocalBackend(Backend):
    """Execute jobs as local subprocesses, optionally throttled and GPU-pinned."""

    detached = False

    def __init__(
        self,
        *,
        gpu_ids: Sequence[int] | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self._gpu_ids = [int(g) for g in gpu_ids] if gpu_ids is not None else None
        if max_concurrent is not None:
            if max_concurrent < 1:
                raise ValueError("max_concurrent must be >= 1")
            self._max_concurrent: int | None = max_concurrent
        elif self._gpu_ids is not None:
            self._max_concurrent = len(self._gpu_ids)
        else:
            self._max_concurrent = None  # unlimited (the default testing behaviour)
        self._counter = itertools.count(1)
        self._pending: list[tuple[str, JobSpec]] = []
        self._running: dict[str, tuple[subprocess.Popen, object, int | None]] = {}
        self._terminal: dict[str, JobState] = {}
        self._free_gpus: list[int] | None = (
            list(self._gpu_ids) if self._gpu_ids else None
        )

    def _has_capacity(self) -> bool:
        return self._max_concurrent is None or len(self._running) < self._max_concurrent

    def _start(self, token: str, spec: JobSpec) -> None:
        work_dir = pathlib.Path(spec.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **spec.env}
        gpu = None
        if self._free_gpus is not None:
            gpu = self._free_gpus.pop(0)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        out = open(work_dir / f"{spec.name}.out", "w")
        proc = subprocess.Popen(
            spec.command,
            cwd=str(work_dir),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        self._running[token] = (proc, out, gpu)

    def submit(self, spec: JobSpec) -> JobHandle:
        token = f"local-{next(self._counter)}"
        if self._has_capacity():
            self._start(token, spec)
        else:
            self._pending.append((token, spec))
        return token

    def _reap(self) -> None:
        """Move finished processes to terminal state, freeing their GPU."""
        for token, (proc, out, gpu) in list(self._running.items()):
            code = proc.poll()
            if code is None:
                continue
            if not out.closed:
                out.close()
            if gpu is not None and self._free_gpus is not None:
                self._free_gpus.append(gpu)
            self._terminal[token] = JobState.FINISHED if code == 0 else JobState.FAILED
            del self._running[token]

    def _pump(self) -> None:
        """Start queued jobs while capacity (and a GPU) is available."""
        while self._pending and self._has_capacity():
            token, spec = self._pending.pop(0)
            self._start(token, spec)

    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]:
        self._reap()
        self._pump()
        pending_tokens = {token for token, _ in self._pending}
        result: dict[JobHandle, JobState] = {}
        for handle in handles:
            if handle in self._terminal:
                result[handle] = self._terminal[handle]
            elif handle in self._running:
                result[handle] = JobState.RUNNING
            elif handle in pending_tokens:
                result[handle] = JobState.PENDING
            else:
                result[handle] = JobState.FAILED
        return result

    def cancel(self, handle: JobHandle) -> None:
        job = self._running.get(handle)
        if job is not None and job[0].poll() is None:
            job[0].terminate()
        self._pending = [(t, s) for t, s in self._pending if t != handle]
