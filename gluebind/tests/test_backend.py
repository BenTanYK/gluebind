"""Tests for the backend seam: LocalBackend, Scheduler, SlurmBackend helpers."""

import sys
import threading
import time

import pytest

from gluebind.backend import (
    Backend,
    JobSpec,
    JobState,
    LocalBackend,
    Scheduler,
    SlotPool,
    SlurmBackend,
)


def _spec(work_dir, code, name="job"):
    return JobSpec(
        command=[sys.executable, "-c", code], work_dir=str(work_dir), name=name
    )


def _wait(backend, handle, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = backend.poll([handle])[handle]
        if state.is_terminal:
            return state
        time.sleep(0.02)
    raise AssertionError("job did not reach a terminal state in time")


def test_jobstate_is_terminal():
    assert JobState.FINISHED.is_terminal
    assert JobState.FAILED.is_terminal
    assert not JobState.RUNNING.is_terminal
    assert not JobState.PENDING.is_terminal


def test_local_success(tmp_path):
    backend = LocalBackend()
    handle = backend.submit(_spec(tmp_path, "pass"))
    assert _wait(backend, handle) is JobState.FINISHED


def test_local_failure(tmp_path):
    backend = LocalBackend()
    handle = backend.submit(_spec(tmp_path, "import sys; sys.exit(3)"))
    assert _wait(backend, handle) is JobState.FAILED


def test_local_writes_output_file(tmp_path):
    backend = LocalBackend()
    handle = backend.submit(_spec(tmp_path, "print('hello')", name="win"))
    _wait(backend, handle)
    assert "hello" in (tmp_path / "win.out").read_text()


def test_local_unknown_handle_is_failed(tmp_path):
    assert LocalBackend().poll(["nope"])["nope"] is JobState.FAILED


def test_scheduler_runs_all(tmp_path):
    backend = LocalBackend()
    specs = [_spec(tmp_path / f"w{i}", "pass", name=f"w{i}") for i in range(5)]
    states = Scheduler(backend, poll_interval=0.01).run(specs)
    assert states == [JobState.FINISHED] * 5


def test_scheduler_throttle_still_completes(tmp_path):
    backend = LocalBackend()
    specs = [_spec(tmp_path / f"w{i}", "pass", name=f"w{i}") for i in range(4)]
    states = Scheduler(backend, queue_len_lim=1, poll_interval=0.01).run(specs)
    assert states == [JobState.FINISHED] * 4


def test_scheduler_reports_mixed_outcomes(tmp_path):
    backend = LocalBackend()
    specs = [
        _spec(tmp_path / "ok", "pass", name="ok"),
        _spec(tmp_path / "bad", "import sys; sys.exit(1)", name="bad"),
    ]
    states = Scheduler(backend, poll_interval=0.01).run(specs)
    assert states[0] is JobState.FINISHED
    assert states[1] is JobState.FAILED


def test_local_max_concurrent_caps_running(tmp_path):
    # With a cap of 1, only one job runs at a time; the rest queue (PENDING) and
    # start as slots free — but the run still completes every spec.
    backend = LocalBackend(max_concurrent=1)
    specs = [_spec(tmp_path / f"w{i}", "pass", name=f"w{i}") for i in range(4)]
    h0 = backend.submit(specs[0])
    handles = [h0] + [backend.submit(s) for s in specs[1:]]
    # immediately after submit: one running, three queued
    states = backend.poll(handles)
    assert sum(s is JobState.RUNNING for s in states.values()) <= 1
    assert any(s is JobState.PENDING for s in states.values())
    # draining to completion still finishes all four
    for h in handles:
        assert _wait(backend, h) is JobState.FINISHED


def test_local_invalid_max_concurrent():
    with pytest.raises(ValueError, match="max_concurrent"):
        LocalBackend(max_concurrent=0)


def test_local_max_concurrent_cannot_exceed_gpu_count():
    # More concurrent jobs than GPUs would exhaust the GPU pool and IndexError in
    # _start; reject it up front instead.
    with pytest.raises(ValueError, match="cannot exceed"):
        LocalBackend(gpu_ids=[0, 1], max_concurrent=4)


def test_local_gpu_pinning_round_robin(tmp_path):
    # Each job records the CUDA_VISIBLE_DEVICES it was pinned to; with two GPUs
    # the two concurrent jobs land on different devices.
    code = (
        "import os; "
        "open('gpu.txt','w').write(os.environ.get('CUDA_VISIBLE_DEVICES','none'))"
    )
    backend = LocalBackend(gpu_ids=[0, 1])
    assert backend._max_concurrent == 2  # cap defaults to the GPU count
    specs = [_spec(tmp_path / f"w{i}", code, name=f"w{i}") for i in range(2)]
    handles = [backend.submit(s) for s in specs]
    for h in handles:
        _wait(backend, h)
    pinned = {(tmp_path / f"w{i}" / "gpu.txt").read_text() for i in range(2)}
    assert pinned == {"0", "1"}


def test_slot_pool_caps_and_releases():
    pool = SlotPool(2)
    assert pool.acquire() and pool.acquire()
    assert not pool.acquire()  # exhausted
    pool.release()
    assert pool.acquire()  # a freed slot is reusable


def test_slot_pool_rejects_bad_size():
    with pytest.raises(ValueError, match="SlotPool"):
        SlotPool(0)


class _CountingBackend(Backend):
    """Fake backend that tracks peak in-flight jobs; jobs finish after one poll."""

    detached = False

    def __init__(self):
        self.live = 0
        self.max_live = 0
        self._polls: dict[str, int] = {}
        self._n = 0
        self._lock = threading.Lock()

    def submit(self, spec):
        with self._lock:
            self._n += 1
            handle = f"j{self._n}"
            self._polls[handle] = 0
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            return handle

    def poll(self, handles):
        with self._lock:
            out = {}
            for h in handles:
                self._polls[h] += 1
                if self._polls[h] >= 2:  # stay live across one poll, then finish
                    out[h] = JobState.FINISHED
                    self.live -= 1
                else:
                    out[h] = JobState.RUNNING
            return out

    def cancel(self, handle):  # pragma: no cover
        pass


def test_scheduler_respects_slot_pool(tmp_path):
    backend = _CountingBackend()
    pool = SlotPool(2)
    specs = [JobSpec(command=["x"], work_dir=str(tmp_path)) for _ in range(6)]
    states = Scheduler(backend, poll_interval=0.0, slots=pool).run(specs)
    assert states == [JobState.FINISHED] * 6
    assert backend.max_live <= 2  # the shared cap was never exceeded


def test_two_schedulers_share_one_slot_pool(tmp_path):
    # Two schedulers on separate threads sharing a pool of 2 must never, between
    # them, have more than 2 jobs in flight — the CalcSet-parallel invariant.
    backend = _CountingBackend()
    pool = SlotPool(2)

    def drive(n):
        specs = [JobSpec(command=["x"], work_dir=str(tmp_path)) for _ in range(n)]
        Scheduler(backend, poll_interval=0.0, slots=pool).run(specs)

    threads = [threading.Thread(target=drive, args=(5,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert backend.max_live <= 2


def test_detached_flags():
    assert SlurmBackend.detached is True
    assert LocalBackend.detached is False


def test_slurm_poll_grace_period_and_resume(monkeypatch):
    from gluebind.config.slurm import SlurmConfig

    cfg = SlurmConfig(job_submission_wait=100)
    clock = [1000.0]
    backend = SlurmBackend(cfg, clock=lambda: clock[0])
    monkeypatch.setattr(backend, "_running_job_ids", lambda: set())

    # freshly submitted, not yet visible in squeue, within the grace window -> RUNNING
    backend._submitted_at["j1"] = 1000.0
    assert backend.poll(["j1"])["j1"] is JobState.RUNNING
    # grace elapsed without ever appearing -> FINISHED (caller's file gate decides)
    clock[0] = 1000.0 + cfg.job_submission_wait + 1
    assert backend.poll(["j1"])["j1"] is JobState.FINISHED

    # a job seen in the queue, then gone, is FINISHED (normal completion)
    monkeypatch.setattr(backend, "_running_job_ids", lambda: {"j2"})
    backend._submitted_at["j2"] = clock[0]
    assert backend.poll(["j2"])["j2"] is JobState.RUNNING  # seen now
    monkeypatch.setattr(backend, "_running_job_ids", lambda: set())
    assert backend.poll(["j2"])["j2"] is JobState.FINISHED  # left the queue

    # a handle from a prior process (resume) has no grace basis -> FINISHED, not stuck
    assert backend.poll(["old"])["old"] is JobState.FINISHED


def test_slurm_parse_job_id():
    assert SlurmBackend._parse_job_id("Submitted batch job 12345\n") == "12345"


def test_slurm_parse_job_id_empty_raises():
    with pytest.raises(RuntimeError):
        SlurmBackend._parse_job_id("")
