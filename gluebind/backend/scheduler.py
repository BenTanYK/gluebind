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

import threading
import time
from collections.abc import Callable, Iterable

from gluebind.backend.base import Backend, JobHandle, JobSpec, JobState


class SlotPool:
    """A shared, thread-safe cap on the total number of in-flight jobs.

    Passed to several :class:`Scheduler` instances driven concurrently (e.g. a
    ``CalcSet`` running systems in parallel) so that, however many systems submit
    at once, no more than ``size`` jobs are ever in flight together — keeping the
    driver within the cluster's per-user submission limit while letting any system
    grab a slot the moment one frees.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("SlotPool size must be >= 1")
        self.size = size
        self._sem = threading.BoundedSemaphore(size)

    def acquire(self) -> bool:
        """Take a slot without blocking; return whether one was available."""
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


class Scheduler:
    """Submit many jobs through a backend and wait for completion."""

    def __init__(
        self,
        backend: Backend,
        *,
        queue_len_lim: int = 2000,
        poll_interval: float = 30.0,
        slots: SlotPool | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.queue_len_lim = queue_len_lim
        self.poll_interval = poll_interval
        self.slots = slots
        self._sleep = sleep

    def _acquire_slot(self) -> bool:
        return True if self.slots is None else self.slots.acquire()

    def _release_slot(self) -> None:
        if self.slots is not None:
            self.slots.release()

    def run(
        self,
        specs: Iterable[JobSpec],
        *,
        on_submit: Callable[[int, JobHandle], None] | None = None,
    ) -> list[JobState]:
        """Submit all ``specs`` and return their terminal states, in input order.

        ``on_submit(index, handle)`` is called as each spec is submitted, so the
        caller can persist the opaque handle (e.g. into ``RunState``) for resume.

        With a shared :class:`SlotPool`, submission is additionally gated on a free
        global slot; a scheduler with pending work but no free slot (all taken by
        other systems) waits and retries rather than exiting.
        """
        specs = list(specs)
        pending = list(range(len(specs)))
        state: list[JobState | None] = [None] * len(specs)
        live: dict[JobHandle, int] = {}  # every entry holds exactly one acquired slot

        try:
            while pending or live:
                while (
                    pending and len(live) < self.queue_len_lim and self._acquire_slot()
                ):
                    index = pending.pop(0)
                    try:
                        handle = self.backend.submit(specs[index])
                    except BaseException:
                        self._release_slot()  # submit failed — hand the slot back
                        raise
                    live[handle] = index
                    if on_submit is not None:
                        on_submit(index, handle)
                if not live:
                    if not pending:
                        break
                    self._sleep(self.poll_interval)  # waiting for a global slot to free
                    continue
                for handle, job_state in self.backend.poll(list(live)).items():
                    if job_state.is_terminal:
                        state[live.pop(handle)] = job_state
                        self._release_slot()
                if pending or live:
                    self._sleep(self.poll_interval)
        finally:
            for _ in range(len(live)):  # release slots held by still-live jobs on error
                self._release_slot()

        return [s if s is not None else JobState.FAILED for s in state]
