"""Tests for the backend seam: LocalBackend, Scheduler, SlurmBackend helpers."""

import shlex
import subprocess
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


def test_stop_request_blocks_later_submission_and_waits_for_handle_record(tmp_path):
    """A kill marker cannot race between submit() and persisted-handle recording."""
    from gluebind.stop import StopController, StopRequested

    class BackendThatFinishes(Backend):
        def __init__(self):
            self.submitted = []

        def submit(self, spec):
            self.submitted.append(spec)
            return f"job-{len(self.submitted)}"

        def poll(self, handles):
            return dict.fromkeys(handles, JobState.FINISHED)

        def cancel(self, handle):
            pass

    marker = StopController.marker_path(tmp_path)
    controller = StopController((marker,))
    backend = BackendThatFinishes()
    recorded = []
    in_callback = threading.Event()
    release_callback = threading.Event()
    result = []

    def on_submit(_index, handle):
        recorded.append(handle)
        in_callback.set()
        assert release_callback.wait(timeout=2)

    def run_scheduler():
        try:
            Scheduler(
                backend,
                poll_interval=0.0,
                submission_guard=controller.submission_permit,
            ).run(
                [_spec(tmp_path, "pass", "one"), _spec(tmp_path, "pass", "two")],
                on_submit=on_submit,
            )
        except StopRequested:
            result.append("stopped")

    driver = threading.Thread(target=run_scheduler)
    driver.start()
    assert in_callback.wait(timeout=2)

    killer = threading.Thread(target=controller.request_stop)
    killer.start()
    time.sleep(0.02)
    assert not marker.exists()  # killer waits for submit + handle persistence
    release_callback.set()
    killer.join(timeout=2)
    driver.join(timeout=2)

    assert marker.exists()
    assert recorded == ["job-1"]
    assert len(backend.submitted) == 1
    assert result == ["stopped"]


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


def test_slurm_submit_shell_quotes_command(tmp_path, monkeypatch):
    from gluebind.config.slurm import SlurmConfig

    class _CompletedProcess:
        stdout = "12345\n"  # sbatch --parsable

    monkeypatch.setattr(
        "gluebind.backend.slurm.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(),
    )
    spec = JobSpec(
        command=["python", "-c", "print('hello world')"],
        work_dir=str(tmp_path),
        name="quoted",
    )

    assert SlurmBackend(SlurmConfig()).submit(spec) == "12345"
    script = (tmp_path / "gluebind.sh").read_text()
    assert "#SBATCH --job-name=quoted" in script
    command_line = script.strip().splitlines()[-1]
    assert shlex.split(command_line) == spec.command


def test_slurm_submit_reports_sbatch_stderr(tmp_path, monkeypatch):
    from gluebind.config.slurm import SlurmConfig

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["sbatch"], output="submission rejected", stderr="Invalid partition"
        )

    monkeypatch.setattr("gluebind.backend.slurm.subprocess.run", _fail)
    spec = JobSpec(command=["echo", "hello"], work_dir=str(tmp_path), name="broken")

    with pytest.raises(RuntimeError, match="Invalid partition"):
        SlurmBackend(SlurmConfig()).submit(spec)


def test_slurm_poll_grace_period_and_resume(monkeypatch):
    from gluebind.config.slurm import SlurmConfig

    cfg = SlurmConfig(job_submission_wait=100)
    clock = [1000.0]
    backend = SlurmBackend(cfg, clock=lambda: clock[0])
    monkeypatch.setattr(backend, "_queue_states", lambda: {})

    # freshly submitted, not yet visible in squeue, within the grace window -> RUNNING
    backend._submitted_at["j1"] = 1000.0
    assert backend.poll(["j1"])["j1"] is JobState.RUNNING
    # grace elapsed without ever appearing -> FINISHED (caller's file gate decides)
    clock[0] = 1000.0 + cfg.job_submission_wait + 1
    assert backend.poll(["j1"])["j1"] is JobState.FINISHED

    # a job seen in the queue, then gone, is FINISHED (normal completion)
    monkeypatch.setattr(backend, "_queue_states", lambda: {"j2": "RUNNING"})
    backend._submitted_at["j2"] = clock[0]
    assert backend.poll(["j2"])["j2"] is JobState.RUNNING  # seen now
    monkeypatch.setattr(backend, "_queue_states", lambda: {})
    assert backend.poll(["j2"])["j2"] is JobState.FINISHED  # left the queue

    # a handle from a prior process (resume) has no grace basis -> FINISHED, not stuck
    assert backend.poll(["old"])["old"] is JobState.FINISHED


def test_slurm_parse_job_id():
    assert SlurmBackend._parse_job_id("12345\n") == "12345"
    assert SlurmBackend._parse_job_id("12345;cluster2\n") == "12345"  # multi-cluster


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        # Regression: the last token was taken as the id, giving "foo" here, an id
        # that never appears in squeue, so the running job read as finished.
        "Submitted batch job 12345 on cluster foo\n",
        "sbatch: error: something odd\n",
    ],
)
def test_slurm_parse_job_id_rejects_anything_but_a_numeric_id(stdout):
    with pytest.raises(RuntimeError, match="could not parse job id"):
        SlurmBackend._parse_job_id(stdout)


@pytest.mark.parametrize(
    "slurm_state, expected",
    [
        ("PENDING", JobState.PENDING),
        ("RUNNING", JobState.RUNNING),
        # Regression: states outside R,PD,S,CG were invisible, so a job that was
        # still CONFIGURING or REQUEUED read as finished.
        ("CONFIGURING", JobState.RUNNING),
        ("REQUEUED", JobState.RUNNING),
        ("COMPLETING", JobState.RUNNING),
        ("SOME_FUTURE_STATE", JobState.RUNNING),  # unknown -> wait, never "done"
        ("COMPLETED", JobState.FINISHED),
        ("TIMEOUT", JobState.FAILED),
        ("CANCELLED", JobState.FAILED),
        ("OUT_OF_MEMORY", JobState.FAILED),
    ],
)
def test_slurm_poll_maps_every_squeue_state(monkeypatch, slurm_state, expected):
    from gluebind.config.slurm import SlurmConfig

    backend = SlurmBackend(SlurmConfig())
    monkeypatch.setattr(backend, "_queue_states", lambda: {"7": slurm_state})
    assert backend.poll(["7"])["7"] is expected


def test_slurm_queue_states_lists_all_states_for_the_user(monkeypatch):
    from gluebind.config.slurm import SlurmConfig

    calls = []

    class _Completed:
        stdout = "11 RUNNING\n12 CONFIGURING\n13 COMPLETED\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Completed()

    monkeypatch.setattr("gluebind.backend.slurm.getpass.getuser", lambda: "me")
    monkeypatch.setattr("gluebind.backend.base.subprocess.run", fake_run)
    states = SlurmBackend(SlurmConfig())._queue_states()
    assert states == {"11": "RUNNING", "12": "CONFIGURING", "13": "COMPLETED"}
    assert calls == [["squeue", "-h", "-t", "all", "-u", "me", "-o", "%i %T"]]


# ---- retrying transient scheduler-query failures -----------------------------


def _flaky_run(failures, stdout="ok\n", exc=None):
    """A subprocess.run stand-in failing ``failures`` times, then succeeding."""
    calls = []

    class _Completed:
        pass

    def run(cmd, **kwargs):
        calls.append(kwargs)
        if len(calls) <= failures:
            raise exc or subprocess.CalledProcessError(
                1, cmd, stderr="slurm_load_jobs error: Socket timed out"
            )
        done = _Completed()
        done.stdout = stdout
        return done

    return run, calls


def test_status_query_retries_with_doubling_waits_then_succeeds(monkeypatch):
    from gluebind.backend.base import run_status_query

    run, calls = _flaky_run(failures=2)
    monkeypatch.setattr("gluebind.backend.base.subprocess.run", run)
    waits = []
    out = run_status_query(["squeue"], retries=5, wait_s=30.0, sleep=waits.append)
    assert out == "ok\n"
    assert waits == [30.0, 60.0]
    assert all(kwargs["timeout"] > 0 for kwargs in calls)  # never hangs forever


def test_status_query_retries_a_timeout(monkeypatch):
    from gluebind.backend.base import run_status_query

    run, _ = _flaky_run(failures=1, exc=subprocess.TimeoutExpired(["squeue"], 60))
    monkeypatch.setattr("gluebind.backend.base.subprocess.run", run)
    assert run_status_query(["squeue"], retries=3, wait_s=1.0, sleep=lambda s: None)


def test_status_query_gives_up_after_the_last_attempt(monkeypatch):
    # Regression: a single failed squeue (a busy controller) aborted the driver.
    # Now only persistent failure does, after every retry, with the error shown.
    from gluebind.backend.base import run_status_query

    run, calls = _flaky_run(failures=10)
    monkeypatch.setattr("gluebind.backend.base.subprocess.run", run)
    waits = []
    with pytest.raises(RuntimeError, match="failed 4 time.*Socket timed out"):
        run_status_query(["squeue"], retries=4, wait_s=30.0, sleep=waits.append)
    assert len(calls) == 4
    assert waits == [30.0, 60.0, 120.0]


def test_slurm_poll_uses_the_configured_retries(monkeypatch):
    from gluebind.config.slurm import SlurmConfig

    run, calls = _flaky_run(failures=2, stdout="5 RUNNING\n")
    monkeypatch.setattr("gluebind.backend.base.subprocess.run", run)
    waits = []
    cfg = SlurmConfig(poll_retries=3, poll_retry_wait_s=10.0)
    backend = SlurmBackend(cfg, sleep=waits.append)
    assert backend.poll(["5"])["5"] is JobState.RUNNING
    assert waits == [10.0, 20.0]
