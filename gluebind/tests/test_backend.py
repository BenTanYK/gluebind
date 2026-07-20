"""Tests for the backend seam: LocalBackend, Scheduler, SlurmBackend helpers."""

import sys
import time

import pytest

from gluebind.backend import JobSpec, JobState, LocalBackend, Scheduler, SlurmBackend


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


def test_detached_flags():
    assert SlurmBackend.detached is True
    assert LocalBackend.detached is False


def test_slurm_parse_job_id():
    assert SlurmBackend._parse_job_id("Submitted batch job 12345\n") == "12345"


def test_slurm_parse_job_id_empty_raises():
    with pytest.raises(RuntimeError):
        SlurmBackend._parse_job_id("")
