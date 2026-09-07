"""Tests for the Grid Engine configuration and backend."""

from __future__ import annotations

import threading

import pytest

from gluebind.backend import GridEngineBackend, JobSpec, JobState
from gluebind.config import GridEngineConfig


def test_grid_engine_config_roundtrip_file_and_directory(tmp_path):
    config = GridEngineConfig(queue="accelerated", resources={"gpu": "2"})
    path = config.dump(tmp_path)
    assert GridEngineConfig.load(tmp_path) == config
    assert GridEngineConfig.load(path) == config


def test_grid_engine_render_script(tmp_path):
    config = GridEngineConfig(
        queue="gpu",
        time="02:00:00",
        memory="24G",
        resources={"gpu": "1", "cuda": "12"},
        output="logs/job.out",
        shell="/bin/bash",
        preamble=[". /etc/profile.d/modules.sh", "module load cuda"],
        extra_directives=["-P chemistry", "-pe sharedmem 4"],
    )
    path = config.write_submission_script("python -c pass", tmp_path, "window")
    script = path.read_text()
    for directive in (
        "#$ -cwd",
        "#$ -N window",
        "#$ -q gpu",
        "#$ -l h_rt=02:00:00",
        "#$ -l h_vmem=24G",
        "#$ -l gpu=1",
        "#$ -l cuda=12",
        "#$ -o logs/job.out",
        "#$ -j y",
        "#$ -S /bin/bash",
        "#$ -P chemistry",
        "#$ -pe sharedmem 4",
        ". /etc/profile.d/modules.sh",
        "module load cuda",
        "python -c pass",
    ):
        assert directive in script


@pytest.mark.parametrize("field,value", [("queue", "gpu\n#$ -l x=1")])
def test_grid_engine_rejects_newline_directives(field, value):
    with pytest.raises(ValueError, match="must not contain newlines"):
        GridEngineConfig(**{field: value})
    with pytest.raises(ValueError, match="must not contain newlines"):
        GridEngineConfig(extra_directives=["-P project\n#$ -j n"])


def test_grid_engine_submit_quotes_and_uses_work_dir(tmp_path, monkeypatch):
    calls = []

    class Completed:
        stdout = 'Your job 12345 ("quoted") has been submitted\n'

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return Completed()

    monkeypatch.setattr("gluebind.backend.grid_engine.subprocess.run", fake_run)
    spec = JobSpec(
        command=["python", "-c", "print('hello world')"],
        work_dir=str(tmp_path),
        name="quoted",
    )
    assert GridEngineBackend(GridEngineConfig()).submit(spec) == "12345"
    assert calls[0][0][0][0] == "qsub"
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert "'hello world'" in (tmp_path / "quoted.sh").read_text()


@pytest.mark.parametrize(
    "output,handle",
    [("12345\n", "12345"), ('Your job 12345 ("name") has been submitted\n', "12345")],
)
def test_grid_engine_parse_job_id(output, handle):
    assert GridEngineBackend._parse_job_id(output) == handle


def test_grid_engine_parse_job_id_failure():
    with pytest.raises(RuntimeError, match="could not parse job id"):
        GridEngineBackend._parse_job_id("submission accepted")


def test_grid_engine_qstat_states_and_grace(monkeypatch):
    backend = GridEngineBackend(
        GridEngineConfig(job_submission_wait=10), clock=lambda: 100
    )
    output = """job-ID prior name user state submit/start at queue slots ja-task-ID
-------------------------------------------------------------------------------
1 0.1 queued user qw 09/07/2026 10:00:00 1
2 0.1 held user hqw 09/07/2026 10:00:00 1
3 0.1 running user r 09/07/2026 10:00:00 queue@host 1
4 0.1 suspended user s 09/07/2026 10:00:00 queue@host 1
5 0.1 transferring user t 09/07/2026 10:00:00 queue@host 1
6 0.1 deleting user dr 09/07/2026 10:00:00 queue@host 1
7 0.1 error user Eqw 09/07/2026 10:00:00 1
"""
    states = GridEngineBackend._parse_qstat(output)
    assert states == {
        "1": JobState.PENDING,
        "2": JobState.PENDING,
        "3": JobState.RUNNING,
        "4": JobState.RUNNING,
        "5": JobState.RUNNING,
        "6": JobState.RUNNING,
        "7": JobState.FAILED,
    }
    monkeypatch.setattr(backend, "_visible_job_states", lambda: {})
    backend._submitted_at["fresh"] = 95
    assert backend.poll(["fresh", "resumed"]) == {
        "fresh": JobState.RUNNING,
        "resumed": JobState.FINISHED,
    }


def test_grid_engine_seen_job_finishes_after_departure(monkeypatch):
    backend = GridEngineBackend(GridEngineConfig(), clock=lambda: 0)
    monkeypatch.setattr(backend, "_visible_job_states", lambda: {"1": JobState.RUNNING})
    assert backend.poll(["1"])["1"] is JobState.RUNNING
    monkeypatch.setattr(backend, "_visible_job_states", lambda: {})
    assert backend.poll(["1"])["1"] is JobState.FINISHED


def test_grid_engine_cancel_forwards_opaque_handle(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "gluebind.backend.grid_engine.subprocess.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    GridEngineBackend(GridEngineConfig()).cancel("12345.server")
    assert calls == [((["qdel", "12345.server"],), {"check": False})]


def test_grid_engine_submission_bookkeeping_is_thread_safe(tmp_path, monkeypatch):
    counter = iter(range(1000))

    class Completed:
        @property
        def stdout(self):
            return f"{next(counter)}\n"

    monkeypatch.setattr(
        "gluebind.backend.grid_engine.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    backend = GridEngineBackend(GridEngineConfig())
    handles = []

    def submit(i):
        handles.append(
            backend.submit(JobSpec(["echo", str(i)], str(tmp_path), name=f"j{i}"))
        )

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(handles) == len(set(handles)) == len(backend._submitted_at) == 20
