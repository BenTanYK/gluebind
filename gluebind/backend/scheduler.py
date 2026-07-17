"""Submit-many-and-wait helper, sitting above the :class:`Backend` seam.

This is the "scaffold multiple jobs" piece: given a list of per-window
:class:`JobSpec` objects, submit them (throttled to ``queue_len_lim`` live jobs
at a time) and block until all reach a terminal state. It is backend-agnostic —
the same loop drives the local and SLURM backends — so it is the exact analogue
of the multi-job scaffolding an AWS Batch *client* would perform.

The throttle exists because SLURM caps how many jobs a user may queue; on a
cluster without such a cap, leave ``queue_len_lim`` at its default.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState


class Scheduler:
    """Submit many jobs through a backend and wait for completion."""

    def __init__(
        self,
        backend: Backend,
        *,
        queue_len_lim: int = 2000,
        poll_interval: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.queue_len_lim = queue_len_lim
        self.poll_interval = poll_interval
        self._sleep = sleep

    def run(
        self,
        specs: Iterable[JobSpec],
        *,
        on_submit: Callable[[int, JobHandle], None] | None = None,
    ) -> list[JobState]:
        """Submit all ``specs`` and return their terminal states, in input order.

        ``on_submit(index, handle)`` is called as each spec is submitted, so the
        caller can persist the opaque handle (e.g. into ``RunState``) for resume.
        """
        specs = list(specs)
        pending = list(range(len(specs)))
        state: list[JobState | None] = [None] * len(specs)
        live: dict[JobHandle, int] = {}

        while pending or live:
            while pending and len(live) < self.queue_len_lim:
                index = pending.pop(0)
                handle = self.backend.submit(specs[index])
                live[handle] = index
                if on_submit is not None:
                    on_submit(index, handle)
            if not live:
                break
            for handle, job_state in self.backend.poll(list(live)).items():
                if job_state.is_terminal:
                    state[live.pop(handle)] = job_state
            if live:
                self._sleep(self.poll_interval)

        return [s if s is not None else JobState.FAILED for s in state]
