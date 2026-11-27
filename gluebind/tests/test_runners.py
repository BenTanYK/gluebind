"""Tests for the runner hierarchy: tree construction, window enumeration,
end-to-end orchestration (LocalBackend), resume, and analysis aggregation.

These exercise the orchestration with a trivial ``spec_builder`` and a trivial
job command (no OpenMM MD), so they run fast and don't need real structures.
"""

import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from gluebind.backend import LocalBackend, Scheduler
from gluebind.backend.base import Backend, JobState
from gluebind.config import (
    CalculationConfig,
    SchedulerConfig,
    WindowSampling,
)
from gluebind.runners import Calculation, enumerate_centres, format_label
from gluebind.simulation import WindowSpec

INPUTS = {
    "target": {"prm7": "t.prm7", "rst7": "t.rst7"},
    "receptor": {"prm7": "r.prm7", "rst7": "r.rst7"},
    "glue": {"sdf": "g.sdf", "assign_to": "receptor"},
}
CENTRES = {1: {"thetaA": [1.0], "separation": [1.5]}}


def _config():
    cfg = CalculationConfig.model_validate({"inputs": INPUTS})
    cfg.sampling.ensemble_size = 1
    cfg.sampling.rmsd.window_min = 0.0
    cfg.sampling.rmsd.window_max = 0.2
    cfg.sampling.rmsd.window_spacing = 0.2
    return cfg


def _spec_builder(*, cv_type, stage_name, dof, cv_centre, replicate, boresch_eq_values):
    return WindowSpec(
        cv_type=cv_type,
        stage_name=stage_name,
        cv_centre=cv_centre,
        replicate=replicate,
        dof=dof,
        topology="t.prm7",
        coordinates="c.rst7",
        reference_coordinates="c.rst7",
        force_constant=5.0,
        sampling_time_ns=1.0,
        restraints={"boresch_eq_values": dict(boresch_eq_values)},
    )


def _trivial_command():
    # Runs in the replicate's work dir; writing result.json marks it complete.
    return [sys.executable, "-c", "open('result.json', 'w').write('{}')"]


def _fake_pmf(stage):
    # A PMF whose minimum is at cv=1.0, standing in for WHAM output in tests.
    x = np.linspace(0.0, 2.0, 21)
    return x, (x - 1.0) ** 2


def _calc(tmp_path, config=None, command_factory=_trivial_command):
    return Calculation(
        tmp_path,
        config or _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=command_factory,
        stage_centres=CENTRES,
    )


def test_scheduler_config_controls_default_scheduler(tmp_path):
    config = SchedulerConfig(queue_check_interval=7, queue_len_lim=3)
    calc = Calculation(
        tmp_path, _config(), LocalBackend(), _spec_builder, scheduler_config=config
    )
    scheduler = calc._default_scheduler()
    assert scheduler.poll_interval == 7 and scheduler.queue_len_lim == 3


class _CancellableBackend(Backend):
    """Minimal backend used to verify a persisted run can be cancelled."""

    def __init__(self):
        self.cancelled: list[str] = []

    def submit(self, spec):
        raise AssertionError("submission is not needed for this test")

    def poll(self, handles):
        return dict.fromkeys(handles, JobState.PENDING)

    def cancel(self, handle):
        self.cancelled.append(handle)


# ---- enumeration / labels --------------------------------------------------


def test_enumerate_from_range():
    s = WindowSampling(
        force_constant=5.0,
        sampling_time_ns=1.0,
        window_min=0.0,
        window_max=1.0,
        window_spacing=0.5,
    )
    assert enumerate_centres(s) == [0.0, 0.5, 1.0]


def test_simulation_runner_resolves_relative_base_dir(tmp_path, monkeypatch):
    from gluebind.runners.base import SimulationRunner

    monkeypatch.chdir(tmp_path)
    assert SimulationRunner("outputs").base_dir == tmp_path / "outputs"


def test_enumerate_explicit_centres():
    s = WindowSampling(force_constant=5.0, sampling_time_ns=1.0, centres=[1.0, 2.0])
    assert enumerate_centres(s) == [1.0, 2.0]


def test_enumerate_requires_info():
    with pytest.raises(ValueError):
        enumerate_centres(WindowSampling(force_constant=5.0, sampling_time_ns=1.0))


def test_enumerate_two_phase_separation():
    # fine 0.9->2.10 @0.05, then coarse @0.11 up to window_max (the separation CV)
    s = WindowSampling(
        force_constant=10.0,
        sampling_time_ns=30.0,
        window_min=0.9,
        window_max=3.0,
        window_spacing=0.05,
        coarse_from=2.10,
        coarse_spacing=0.10,
    )
    centres = enumerate_centres(s)
    assert centres[0] == 0.9
    assert 2.1 in centres  # transition point included
    # fine region is 0.05-spaced up to 2.10
    assert centres[1] == pytest.approx(0.95)
    # coarse region beyond 2.10 is 0.10-spaced and stays within window_max
    assert centres[centres.index(2.1) + 1] == pytest.approx(2.2)
    assert max(centres) <= 3.0


def test_separation_schedule_enumerates_with_an_explicit_window_min():
    from gluebind.config.sampling import SamplingConfig

    s = SamplingConfig()
    s.separation.window_min = 0.9
    centres = enumerate_centres(s.for_cv("separation", "separation"))
    assert centres[0] == 0.9 and max(centres) <= 3.0


def test_separation_default_auto_window_min_needs_resolution():
    # The default window_min is "auto", resolved from the equilibration
    # trajectory during restraint resolution, so it cannot be enumerated before.
    from gluebind.config.sampling import SamplingConfig

    with pytest.raises(ValueError, match="restraint resolution"):
        enumerate_centres(SamplingConfig().for_cv("separation", "separation"))


def test_format_label():
    assert format_label("boresch", 0.85) == "0.85rad"
    assert format_label("separation", 1.5) == "1.5nm"
    assert format_label("rmsd", 0.2) == "0.2A"


# ---- tree construction -----------------------------------------------------


def test_builds_all_cv_groups(tmp_path):
    calc = _calc(tmp_path)
    assert {g.cv_type for g in calc.groups} == {"boresch", "rmsd", "separation"}


def test_default_all_ca_rmsd_stages(tmp_path):
    calc = _calc(tmp_path)
    rmsd_group = next(g for g in calc.groups if g.cv_type == "rmsd")
    assert {s.name for s in rmsd_group.stages} == {
        "receptor_bound",
        "receptor_bulk",
        "target_bound",
        "target_bulk",
    }


def test_write_specs_threads_boresch_eq(tmp_path):
    calc = _calc(tmp_path)
    theta_a = calc._group("boresch").stages[0]
    assert theta_a.dof == "thetaA"
    theta_a.write_specs({1: {"thetaX": 0.5}})
    spec = WindowSpec.load(
        tmp_path / "boresch" / "thetaA" / "run_01" / "1rad" / "window.json"
    )
    assert spec.cv_type == "boresch" and spec.dof == "thetaA"
    assert spec.restraints["boresch_eq_values"] == {"thetaX": 0.5}


# ---- run / resume ----------------------------------------------------------


def test_run_completes_and_records_state(tmp_path):
    calc = _calc(tmp_path)
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    for window in calc._iter_windows():
        assert window.is_complete()
    assert (tmp_path / ".gluebind-state.json").exists()
    assert state.handles  # handles recorded via on_submit
    assert state.stage_status.get("thetaA") == "done"


def test_kill_cancels_each_persisted_handle_once(tmp_path):
    from gluebind import RunState
    from gluebind.state import now_utc_iso

    backend = _CancellableBackend()
    calc = Calculation(tmp_path, _config(), backend, _spec_builder)
    state = RunState(
        calc_id="test",
        submitted_at=now_utc_iso(),
        config_hash=calc.config.config_hash,
        config_path=str(tmp_path),
        handles={"thetaA": {1: {"1rad": "100"}, 2: {"1rad": "101"}}},
        auxiliary_handles={"system_build": ["101"], "steered_md": ["102"]},
    )
    state.save(tmp_path)

    assert calc.kill() == ["100", "101", "102"]
    assert backend.cancelled == ["100", "101", "102"]
    assert calc._stop_marker.exists()


def test_calculation_kill_does_not_stop_its_parent_set(tmp_path):
    backend = _CancellableBackend()
    set_marker = tmp_path / "set" / ".gluebind-stop"
    calc = Calculation(
        tmp_path / "set" / "system",
        _config(),
        backend,
        _spec_builder,
        stop_paths=(set_marker,),
    )

    assert calc.kill() == []
    assert calc._stop_marker.exists()
    assert not set_marker.exists()
    calc.clear_stop_request()
    assert not calc._stop_marker.exists()


def test_calculation_kill_stops_an_active_driver_before_later_submissions(tmp_path):
    """The persisted stop request reaches a live Calculation.run() driver."""

    class BlockingBackend(Backend):
        def __init__(self):
            self.submitted = []
            self.cancelled = []
            self.entered_submit = threading.Event()
            self.release_submit = threading.Event()

        def submit(self, spec):
            self.submitted.append(spec)
            self.entered_submit.set()
            assert self.release_submit.wait(timeout=2)
            return "job-1"

        def poll(self, handles):
            return dict.fromkeys(handles, JobState.FINISHED)

        def cancel(self, handle):
            self.cancelled.append(handle)

    backend = BlockingBackend()
    calc = Calculation(
        tmp_path, _config(), backend, _spec_builder, stage_centres=CENTRES
    )
    result = []

    driver = threading.Thread(
        target=lambda: result.append(
            calc.run(
                scheduler=Scheduler(backend, poll_interval=0.0), pmf_provider=_fake_pmf
            )
        )
    )
    driver.start()
    assert backend.entered_submit.wait(timeout=2)

    killer = threading.Thread(target=calc.kill)
    killer.start()
    backend.release_submit.set()
    for _ in range(100):
        if calc._stop_marker.exists():
            break
        time.sleep(0.01)
    assert calc._stop_marker.exists()
    killer.join(timeout=2)
    driver.join(timeout=2)

    assert len(backend.submitted) == 1
    assert backend.cancelled == ["job-1"]
    assert result[0].stage_status["_control"] == "stopped"


def test_kill_without_state_is_a_noop(tmp_path):
    backend = _CancellableBackend()
    calc = Calculation(tmp_path, _config(), backend, _spec_builder)

    assert calc.kill() == []
    assert backend.cancelled == []


# ---- one driver per calculation, no duplicate jobs -------------------------


def _save_state_with_handles(calc, handles=None, auxiliary_handles=None):
    from gluebind import RunState
    from gluebind.state import now_utc_iso

    RunState(
        calc_id="test",
        submitted_at=now_utc_iso(),
        config_hash=calc.config.config_hash,
        config_path=str(calc.base_dir),
        handles=handles or {},
        auxiliary_handles=auxiliary_handles or {},
    ).save(calc.base_dir)


def test_second_driver_on_the_same_directory_is_refused(tmp_path):
    first, second = _calc(tmp_path), _calc(tmp_path)
    with first._driver_session():
        with pytest.raises(RuntimeError, match="already driving"):
            second.run(
                scheduler=Scheduler(second.backend, poll_interval=0.01),
                pmf_provider=_fake_pmf,
            )
        with pytest.raises(RuntimeError, match="already driving"):
            second.prepare()


def test_driver_lock_is_released_when_a_run_fails(tmp_path):
    calc = _calc(tmp_path)
    with pytest.raises(ValueError), calc._driver_session():
        raise ValueError("driver crashed")
    with calc._driver_session():  # a later driver can take over
        pass


def test_run_refuses_to_resubmit_while_recorded_jobs_are_live(tmp_path):
    # Regression: a resumed run resubmitted every replicate without a result,
    # including ones whose jobs were still queued or running, so two jobs wrote
    # the same directory.
    backend = _CancellableBackend()  # reports every handle PENDING; submit fails
    calc = Calculation(tmp_path, _config(), backend, _spec_builder)
    _save_state_with_handles(calc, {"thetaA": {1: {"1rad": "100"}}})

    with pytest.raises(RuntimeError, match=r"1 job\(s\) .* still queued or running"):
        calc.run(
            scheduler=Scheduler(backend, poll_interval=0.0), pmf_provider=_fake_pmf
        )


def test_finished_recorded_jobs_do_not_block_a_resume(tmp_path):
    calc = _calc(tmp_path)  # LocalBackend reports unknown (earlier) handles finished
    _save_state_with_handles(calc, auxiliary_handles={"system_build": ["old-job"]})

    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert state.stage_status.get("thetaA") == "done"


def test_resubmission_keeps_the_superseded_handle_for_kill(tmp_path):
    from gluebind import RunState

    first = _calc(tmp_path)
    first.run(
        scheduler=Scheduler(first.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    stage = "receptor_bound"  # RMSD stages are resubmitted; determined DoFs are not
    window = next(w for w in first._iter_windows() if w.stage_name == stage)
    old = RunState.load(tmp_path).handles[stage][window.replicate][window.label]
    (window.base_dir / "result.json").unlink()  # force a resubmission

    second = _calc(tmp_path)
    # LocalBackend numbers jobs from 1 in every process; give the resumed driver
    # distinct ids, as a real scheduler would.
    second.backend._counter = itertools.count(1000)
    second.run(
        scheduler=Scheduler(second.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    state = RunState.load(tmp_path)
    assert state.handles[stage][window.replicate][window.label] != old
    assert state.superseded_handles == [old]
    assert old in second.kill()


def test_run_surfaces_failed_windows(tmp_path):
    # A job that exits without writing result.json is a failure; it must be
    # surfaced here (naming the window) rather than downstream as a WHAM crash.
    from gluebind import RunState

    calc = _calc(
        tmp_path,
        command_factory=lambda: [sys.executable, "-c", "raise SystemExit(1)"],
    )
    with pytest.raises(RuntimeError, match="produced no result"):
        calc.run(
            scheduler=Scheduler(calc.backend, poll_interval=0.01),
            pmf_provider=_fake_pmf,
        )
    assert "failed" in RunState.load(tmp_path).stage_status.values()


def test_run_is_idempotent(tmp_path):
    calc = _calc(tmp_path)
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    calc2 = _calc(tmp_path)
    pending = [
        (w.stage_name, w.replicate, w.label)
        for w in calc2._iter_windows()
        if not w.is_complete()
    ]
    assert pending == []


def test_resume_config_hash_mismatch_aborts(tmp_path):
    _calc(tmp_path).run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )
    changed = _config()
    changed.sampling.rmsd.force_constant = 99.0
    calc2 = _calc(tmp_path, config=changed)
    with pytest.raises(ValueError, match="config_hash"):
        calc2.run(
            scheduler=Scheduler(calc2.backend, poll_interval=0.01),
            pmf_provider=_fake_pmf,
        )


def _ensemble_calc(tmp_path, centres, **kwargs):
    """A calculation whose replicates have their own window centres."""
    config = _config()
    config.sampling.ensemble_size = len(centres)
    config.sampling.ensemble_from = "equilibration"
    return Calculation(
        tmp_path,
        config,
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres=centres,
        **kwargs,
    )


def test_each_replicate_owns_a_complete_window_set(tmp_path):
    # Replicates that equilibrated independently have different Boresch centres,
    # so the replicate sits above the centre on disk and each replicate's set is
    # contiguous rather than a column of a grid with holes.
    calc = _ensemble_calc(
        tmp_path,
        {
            1: {"thetaA": [1.0, 1.1], "separation": [1.5]},
            2: {"thetaA": [1.2, 1.3], "separation": [1.6]},
        },
    )
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )

    stage = calc._group("boresch").stages[0]
    assert [w.centre for w in stage.windows_for(1)] == [1.0, 1.1]
    assert [w.centre for w in stage.windows_for(2)] == [1.2, 1.3]
    thetaA = tmp_path / "boresch" / "thetaA"
    assert (thetaA / "run_02" / "1.3rad" / "result.json").exists()
    assert not (thetaA / "run_01" / "1.3rad").exists()
    assert all(window.is_complete() for window in calc._iter_windows())


def test_each_replicate_feeds_forward_its_own_boresch_minimum(tmp_path):
    # The DoF loop stays outermost and the replicates fan out within it: every
    # replicate's windows go in as one batch, then each replicate's own PMF
    # minimum is fed forward into the next DoF independently.
    calc = _ensemble_calc(
        tmp_path,
        {
            1: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]},
            2: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]},
        },
    )
    minima = {1: 0.8, 2: 1.4}

    def per_replicate_pmf(stage):
        cv = np.linspace(0.0, 2.0, 21)
        pmfs = [(cv - minima[r]) ** 2 for r in stage.replicates]
        return cv, (pmfs[0] + pmfs[1]) / 2, pmfs

    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01),
        pmf_provider=per_replicate_pmf,
    )

    for replicate, minimum in minima.items():
        assert state.boresch_eq_values[replicate]["thetaA"] == pytest.approx(
            minimum, abs=0.1
        )
        spec = WindowSpec.load(
            tmp_path
            / "boresch"
            / "thetaB"
            / f"run_{replicate:02d}"
            / "1rad"
            / "window.json"
        )
        assert spec.restraints["boresch_eq_values"]["thetaA"] == pytest.approx(
            minimum, abs=0.1
        )


def test_boresch_sequential_feedback(tmp_path):
    # Two Boresch DoFs: thetaA runs first, its PMF minimum (=1.0) is fed forward
    # as a fixed restraint into thetaB's windows and into separation.
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]}},
    )
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )

    assert state.boresch_eq_values[1]["thetaA"] == pytest.approx(1.0)
    assert state.boresch_eq_values[1]["thetaB"] == pytest.approx(1.0)

    theta_b_spec = WindowSpec.load(
        tmp_path / "boresch" / "thetaB" / "run_01" / "1rad" / "window.json"
    )
    assert theta_b_spec.restraints["boresch_eq_values"]["thetaA"] == pytest.approx(1.0)

    # group dir and stage name are both "separation" (uniform tree depth).
    sep_spec = WindowSpec.load(
        tmp_path / "separation" / "separation" / "run_01" / "1.5nm" / "window.json"
    )
    assert set(sep_spec.restraints["boresch_eq_values"]) == {"thetaA", "thetaB"}


# ---- steered-MD hook -------------------------------------------------------


def test_run_invokes_steered_md_before_separation(tmp_path):
    calls = []
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "separation": [1.5]}},
        steered_md_runner=lambda eq, replicates: calls.append((dict(eq), replicates)),
    )
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert len(calls) == 1
    eq_values, replicates = calls[0]
    assert "thetaA" in eq_values[1]  # invoked with the resolved Boresch eq values
    assert replicates == [1]
    assert state.stage_status.get("steered_md_run_01") == "done"


def test_run_without_separation_skips_steered_md(tmp_path):
    calls = []
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0]}},  # no separation group
        steered_md_runner=lambda eq, replicates: calls.append(eq),
    )
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert calls == []


def _counting_smd(tmp_path, counter):
    """A steered-MD stand-in that writes each pull's 1.5 nm separation frame and
    counts the pulls it ran."""
    from gluebind.simulation.steered_md import smd_frame_path

    def run(eq, replicates):
        for replicate in replicates:
            counter["n"] += 1
            frame = smd_frame_path(
                tmp_path / "smd_frames" / f"run_{replicate:02d}", 1.5
            )
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_text("frame")

    return run


def _smd_calc(tmp_path, counter):
    return Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "separation": [1.5]}},
        steered_md_runner=_counting_smd(tmp_path, counter),
    )


def test_each_replicate_pulls_its_own_separation_frames(tmp_path):
    # Ensemble SMD: one pull per replicate, each into its own frame directory,
    # and a resume repeats only the pull whose frames went missing.
    from gluebind import RunState
    from gluebind.simulation.steered_md import smd_frame_path

    counter = {"n": 0}
    centres = {
        1: {"thetaA": [1.0], "separation": [1.5]},
        2: {"thetaA": [1.0], "separation": [1.5]},
    }
    _ensemble_calc(
        tmp_path, centres, steered_md_runner=_counting_smd(tmp_path, counter)
    ).run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )

    assert counter["n"] == 2  # both replicates pulled
    state = RunState.load(tmp_path)
    assert state.stage_status["steered_md_run_01"] == "done"
    assert state.stage_status["steered_md_run_02"] == "done"

    smd_frame_path(tmp_path / "smd_frames" / "run_02", 1.5).unlink()
    calc = _ensemble_calc(
        tmp_path, centres, steered_md_runner=_counting_smd(tmp_path, counter)
    )
    for window in calc._iter_windows():
        if window.stage_name == "separation" and window.replicate == 2:
            (window.base_dir / "result.json").unlink()
    calc.run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert counter["n"] == 3  # only replicate 2 was pulled again


def test_steered_md_runs_once_across_resume(tmp_path):
    counter = {"n": 0}
    for _ in range(2):
        _smd_calc(tmp_path, counter).run(
            scheduler=Scheduler(LocalBackend(), poll_interval=0.01),
            pmf_provider=_fake_pmf,
        )
    assert counter["n"] == 1  # not repeated on the resumed run


def test_steered_md_reruns_when_a_window_frame_is_missing(tmp_path):
    # Regression: steered MD recorded as done with a window's frame missing (a
    # crashed or truncated pull) was never re-run, so the separation windows
    # failed on every resume.
    from gluebind import RunState
    from gluebind.simulation.steered_md import smd_frame_path

    counter = {"n": 0}
    _smd_calc(tmp_path, counter).run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )
    frame = smd_frame_path(tmp_path / "smd_frames" / "run_01", 1.5)
    frame.unlink()
    for window in _smd_calc(tmp_path, counter)._iter_windows():
        if window.stage_name == "separation":
            (window.base_dir / "result.json").unlink()

    _smd_calc(tmp_path, counter).run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert counter["n"] == 2  # steered MD ran again
    assert frame.exists()
    assert RunState.load(tmp_path).stage_status["steered_md_run_01"] == "done"


# ---- analysis --------------------------------------------------------------


def test_analyse_aggregates(tmp_path):
    calc = _calc(tmp_path)

    def fake_pmf(stage):
        x = np.linspace(0.0, 1.0, 11)
        return x, np.zeros_like(x)

    result = calc.analyse(fake_pmf, r_star_nm=0.8, theta_a_min=1.2, theta_b_min=1.4)
    assert set(result) == {
        "dg_bind",
        "dg_bind_sem",
        "dg_rmsd",
        "dg_boresch",
        "dg_sep",
        "dg_corr",
        "r_star_nm",
        "dg_per_replicate",
        "stage_sems",
        "rmsd_included",
    }
    assert math.isfinite(result["dg_bind"])
    assert result["dg_bind"] == pytest.approx(
        result["dg_rmsd"] + result["dg_boresch"] + result["dg_sep"] + result["dg_corr"]
    )
    assert json.loads((tmp_path / "analysis.json").read_text()) == result


def test_analyse_replaces_analysis_json(tmp_path):
    calc = _calc(tmp_path)
    first = calc.analyse(
        _fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0
    )

    def shifted_pmf(stage):
        x = np.linspace(0.0, 2.0, 21)
        return x, 2.0 * (x - 1.0) ** 2

    second = calc.analyse(
        shifted_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0
    )
    assert second != first
    assert json.loads((tmp_path / "analysis.json").read_text()) == second


def test_analyse_does_not_write_analysis_json_on_provider_failure(tmp_path):
    calc = _calc(tmp_path)

    def failing_provider(stage):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        calc.analyse(
            failing_provider, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0
        )
    assert not (tmp_path / "analysis.json").exists()


def test_analyse_reports_sem_over_repeats(tmp_path):
    # A provider that exposes per-replicate PMFs yields an overall SEM (over the
    # repeats' complete ΔG°) plus per-stage SEM diagnostics.
    calc = _ensemble_calc(
        tmp_path,
        {r: {"thetaA": [1.0], "separation": [1.5]} for r in (1, 2)},
    )

    def provider_with_repeats(stage):
        x = np.linspace(0.0, 2.0, 21)
        mean = (x - 1.0) ** 2
        # Repeats must differ in *shape*, not by a constant offset — a constant
        # cancels in the free-energy ratio, so it would leave the SEM at zero.
        return x, mean, [0.8 * mean, 1.2 * mean]

    result = calc.analyse(
        provider_with_repeats, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0
    )
    assert result["dg_bind_sem"] is not None and result["dg_bind_sem"] > 0
    assert result["stage_sems"]  # per-CV diagnostics present
    assert all(v >= 0 for v in result["stage_sems"].values())
    # the point estimate is the mean of the replicates' complete ΔG°
    per_replicate = result["dg_per_replicate"]
    assert set(per_replicate) == {"run_01", "run_02"}
    assert result["dg_bind"] == pytest.approx(
        sum(v["dg_bind"] for v in per_replicate.values()) / 2
    )


def test_analyse_gives_each_replicate_its_own_r_star_and_correction(tmp_path):
    # Replicates that equilibrated independently resolve their own window_min and
    # their own anchors, so r* and the analytical correction differ between them;
    # each replicate's ΔG° must be built from its own.
    calc = _ensemble_calc(
        tmp_path,
        {
            1: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]},
            2: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.9]},
        },
    )
    minima = {1: 0.8, 2: 1.4}

    def per_replicate_pmf(stage):
        cv = np.linspace(0.0, 2.0, 21)
        pmfs = [(cv - minima[r]) ** 2 for r in stage.replicates]
        return cv, (pmfs[0] + pmfs[1]) / 2, pmfs

    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01),
        pmf_provider=per_replicate_pmf,
    )
    result = calc.analyse(per_replicate_pmf)  # no overrides: resolved per replicate

    from gluebind.runners import calculation as calc_mod

    first, second = result["dg_per_replicate"].values()
    assert first["r_star_nm"] == pytest.approx(1.5 - calc_mod.R_STAR_BUFFER_NM)
    assert second["r_star_nm"] == pytest.approx(1.9 - calc_mod.R_STAR_BUFFER_NM)
    assert first["dg_corr"] != pytest.approx(second["dg_corr"])
    assert result["dg_corr"] == pytest.approx(
        (first["dg_corr"] + second["dg_corr"]) / 2
    )
    for values in (first, second):
        assert values["dg_bind"] == pytest.approx(
            values["rmsd"] + values["boresch"] + values["separation"]
            + values["dg_corr"]
        )


def test_analyse_sem_none_without_repeats(tmp_path):
    # A 2-tuple provider describes one calculation: no spread to report.
    calc = _calc(tmp_path)
    result = calc.analyse(_fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0)
    assert result["dg_bind_sem"] is None
    assert set(result["dg_per_replicate"]) == {"run_01"}
    assert set(result["stage_sems"].values()) == {0.0}


def test_analyse_rejects_a_provider_that_disagrees_on_replicates(tmp_path):
    # Fail loudly rather than quietly scoring a subset of the replicates.
    calc = _calc(tmp_path)  # one replicate

    def two_pmfs(stage):
        x = np.linspace(0.0, 2.0, 21)
        mean = (x - 1.0) ** 2
        return x, mean, [0.8 * mean, 1.2 * mean]

    with pytest.raises(RuntimeError, match="correspond one to one"):
        calc.analyse(two_pmfs, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0)


def test_analyse_threads_sampling_temperature(tmp_path):
    # The FE integrals must use sampling.temperature_K (the temperature WHAM ran
    # at), not a fixed 300 K — otherwise a non-300 K run silently integrates at the
    # wrong beta and returns a quantitatively wrong dG.
    cfg300 = _config()
    cfg300.sampling.temperature_K = 300.0
    cfg350 = _config()
    cfg350.sampling.temperature_K = 350.0
    kw = {"r_star_nm": 1.5, "theta_a_min": 1.0, "theta_b_min": 1.0}
    r300 = _calc(tmp_path / "t300", config=cfg300).analyse(_fake_pmf, **kw)
    r350 = _calc(tmp_path / "t350", config=cfg350).analyse(_fake_pmf, **kw)
    assert r300["dg_bind"] != pytest.approx(r350["dg_bind"])


# ---- facade (from_config / deferred wiring / analyse-from-state) -----------


def test_from_config_defers_wiring(tmp_path):
    # Built from a config object: cheap, no tree yet (wiring happens in prepare()).
    calc = Calculation.from_config(_config(), LocalBackend(), base_dir=tmp_path)
    assert calc.spec_builders == {}
    assert calc.groups == []


def test_run_auto_prepares_when_not_wired(tmp_path):
    # run() on a from_config calculation prepares itself (end to end from a single
    # call), via the private _prepare() since run() already holds the driver
    # session. We stub it to wire trivially, avoiding real MD/BSS.
    calc = Calculation.from_config(_config(), LocalBackend(), base_dir=tmp_path)
    calls = {"prepare": 0}

    def fake_prepare(**_):
        calls["prepare"] += 1
        calc.spec_builders = {1: _spec_builder}
        calc.command_factory = _trivial_command
        calc.stage_centres = {1: {"thetaA": [1.0], "separation": [1.5]}}
        calc.groups = calc._build_groups()
        calc.sub_runners = list(calc.groups)

    calc._prepare = fake_prepare
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert calls["prepare"] == 1
    # a second run() does not re-prepare (already wired)
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert calls["prepare"] == 1


def test_analyse_derives_theta_and_r_star_from_state(tmp_path):
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]}},
    )
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )

    # No r_star/theta passed: theta minima from the run state, r_star just inside
    # the outermost separation centre (1.5 - 0.01 nm).
    import gluebind.runners.calculation as calc_mod

    seen = {}
    real_correction = calc_mod.standard_state_correction

    def spy(r_star, *args, **kwargs):
        seen["r_star"] = r_star
        return real_correction(r_star, *args, **kwargs)

    calc_mod.standard_state_correction = spy
    try:
        result = calc.analyse(_fake_pmf)
    finally:
        calc_mod.standard_state_correction = real_correction
    assert seen["r_star"] == pytest.approx(1.5 - calc_mod.R_STAR_BUFFER_NM)
    assert set(result) == {
        "dg_bind",
        "dg_bind_sem",
        "dg_rmsd",
        "dg_boresch",
        "dg_sep",
        "dg_corr",
        "r_star_nm",
        "dg_per_replicate",
        "stage_sems",
        "rmsd_included",
    }
    assert math.isfinite(result["dg_bind"])


class _FakeProvider:
    """Stands in for WhamPmfProvider(config) -> callable(stage) -> (cv, pmf)."""

    def __init__(self, config):
        self.config = config

    def __call__(self, stage):
        return _fake_pmf(stage)


def test_run_self_defaults_pmf_provider_for_boresch(tmp_path, monkeypatch):
    # run() with no pmf_provider must self-default (like analyse()) for Boresch
    # stages, not raise — this is what makes from_config(...).run() / CalcSet.run()
    # work end to end. Monkeypatch the provider so no real wham binary is needed.
    import gluebind.analysis.provider as provider_mod

    monkeypatch.setattr(provider_mod, "WhamPmfProvider", _FakeProvider)
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "separation": [1.5]}},
    )
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01)
    )  # no pmf_provider
    assert state.boresch_eq_values[1]["thetaA"] == pytest.approx(1.0)


def _dump_prepared(base_dir, replicates=(1,)):
    from gluebind.system.prep import PreparedReplicate, PreparedSystem

    PreparedSystem(
        replicates={
            replicate: PreparedReplicate(
                complex_prm7="c.prm7",
                complex_rst7="c.rst7",
                complex_trajectory="c.dcd",
                target_bulk_prm7="tb.prm7",
                target_bulk_rst7="tb.rst7",
                receptor_bulk_prm7="rb.prm7",
                receptor_bulk_rst7="rb.rst7",
                target_molecules=[0],
                receptor_molecules=[1],
                glue_molecule=2,
            )
            for replicate in replicates
        }
    ).dump(base_dir / "prep")


def _restraint_context():
    from gluebind.spec_builder import RestraintContext

    return RestraintContext(
        complex_topology="c.prm7",
        complex_coordinates="c.rst7",
        rec_group=[1, 2],
        lig_group=[3, 4],
        anchors={"b": 10, "c": 11, "B": 12, "C": 13},
        rmsd_order=[],
        rmsd_atoms_bound={},
        rmsd_bulk={},
        always_on=[],
    )


def test_each_equilibration_replicate_resolves_its_own_restraints(
    tmp_path, monkeypatch
):
    # Replicates equilibrate independently, so anchors and equilibrium values are
    # resolved per replicate, from that replicate's own trajectory.
    from gluebind.backend import scheduler as scheduler_mod
    from gluebind.restraint_context import RESTRAINT_CONTEXT_FILENAME
    from gluebind.simulation.restraint_resolution import (
        RESTRAINT_RESOLUTION_RESULT_FILENAME,
        RESTRAINT_RESOLUTION_SPEC_FILENAME,
        RestraintResolutionSpec,
    )

    config = _config()
    config.sampling.ensemble_size = 3
    config.sampling.ensemble_from = "equilibration"
    _dump_prepared(tmp_path, replicates=(1, 2, 3))
    calc = Calculation.from_config(config, LocalBackend(), base_dir=tmp_path)
    prepared = calc._load_prepared()

    submitted = []

    class _FakeScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, specs, on_submit=None):
            for spec in specs:
                work_dir = pathlib.Path(spec.work_dir)
                submitted.append(spec.name)
                # stand in for the worker: a context the driver can validate
                _write_context(calc, prepared, work_dir)
                (work_dir / RESTRAINT_RESOLUTION_RESULT_FILENAME).write_text("{}")
            return [JobState.FINISHED] * len(specs)

    def _write_context(calc, prepared, work_dir):
        from gluebind.restraint_context import (
            ResolvedRestraintContext,
            context_to_data,
            prepared_hash,
        )

        spec = RestraintResolutionSpec.load(
            work_dir / RESTRAINT_RESOLUTION_SPEC_FILENAME
        )
        ResolvedRestraintContext(
            config_hash=spec.config_hash,
            prepared_hash=prepared_hash(prepared.for_replicate(spec.replicate)),
            context=context_to_data(_restraint_context()),
            stage_centres={"thetaA": [1.0], "separation": [1.5]},
        ).dump(pathlib.Path(spec.replicate_dir) / RESTRAINT_CONTEXT_FILENAME)

    monkeypatch.setattr(scheduler_mod, "Scheduler", _FakeScheduler)

    calc._resolve_restraint_context(prepared)

    assert submitted == [f"restraint_resolution_run_{n:02d}" for n in (1, 2, 3)]
    for replicate in (1, 2, 3):
        assert calc._load_resolved_restraint_context(prepared, replicate) is not None
    # resolved contexts are reused, so a second call submits nothing
    submitted.clear()
    calc._submit_rmsf_report = lambda prepared, replicate=1: {}
    calc._resolve_restraint_context(prepared)
    assert submitted == []


def test_prepare_without_restraints_reuses_prep_and_does_not_wire(tmp_path):
    # prepare(resolve_restraints=False) runs prep + writes the RMSF report but must
    # NOT resolve anchors or build the sampling tree (spec_builder stays None) — the
    # manual-anchor workflow. Prep is stubbed via the on-disk manifest; RMSF write
    # stubbed (MDA).
    _dump_prepared(tmp_path)
    calc = Calculation.from_config(_config(), LocalBackend(), base_dir=tmp_path)
    written = {}
    calc._submit_rmsf_report = lambda prepared, replicate=1: written.setdefault(
        "paths", {"receptor": "rmsf_receptor.dat", "target": "rmsf_target.dat"}
    )

    prepared = calc.prepare(resolve_restraints=False)

    # loaded the existing prep (no re-run)
    assert prepared.for_replicate(1).complex_prm7 == "c.prm7"
    assert calc.spec_builders == {}  # NOT wired — anchors not resolved
    assert calc.groups == []
    assert written["paths"]  # RMSF report requested for inspection


def test_analyse_auto_wires_from_prepared_in_fresh_process(tmp_path):
    # Fresh process: an unwired from_config calc analysing an already-prepared run
    # must re-wire from disk (rebuild the stage tree) so stages are iterated — not
    # silently return zero contributions. _wire is stubbed to avoid MDA.
    _dump_prepared(tmp_path)
    calc = Calculation.from_config(_config(), LocalBackend(), base_dir=tmp_path)

    def fake_wire(prepared):
        calc.spec_builders = {1: _spec_builder}
        calc.command_factory = _trivial_command
        calc.stage_centres = {
            1: {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]}
        }
        calc.groups = calc._build_groups()
        calc.sub_runners = list(calc.groups)

    calc._wire = fake_wire
    result = calc.analyse(_fake_pmf, theta_a_min=1.0, theta_b_min=1.0)

    assert calc.spec_builders and len(calc.groups) > 0  # re-wired, tree rebuilt
    assert set(result) == {
        "dg_bind",
        "dg_bind_sem",
        "dg_rmsd",
        "dg_boresch",
        "dg_sep",
        "dg_corr",
        "r_star_nm",
        "dg_per_replicate",
        "stage_sems",
        "rmsd_included",
    }
    assert math.isfinite(result["dg_bind"])


def test_analyse_raises_when_not_prepared(tmp_path):
    calc = Calculation.from_config(_config(), LocalBackend(), base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not prepared"):
        calc.analyse(_fake_pmf, theta_a_min=1.0, theta_b_min=1.0)


def test_from_config_defaults_base_dir_to_cwd_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = Calculation.from_config(_config(), LocalBackend())
    assert calc.base_dir == tmp_path / "outputs"


def test_analyse_r_star_falls_back_to_config(tmp_path):
    # When stage_centres lacks separation (e.g. a fresh/degenerate wiring), r_star
    # falls back to the config schedule instead of raising.
    cfg = _config()
    # Within the fake PMF's 0-2 nm grid, so the derived r* (1.65 - 0.01) is valid.
    cfg.sampling.separation.window_min = 1.15
    cfg.sampling.separation.window_max = 1.65
    cfg.sampling.separation.window_spacing = 0.5
    cfg.sampling.separation.coarse_from = None  # single-phase grid up to window_max
    calc = Calculation(
        tmp_path,
        cfg,
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "separation": [1.5]}},
    )
    calc.stage_centres = {}  # groups already built at construction; centres now gone
    result = calc.analyse(_fake_pmf, theta_a_min=1.0, theta_b_min=1.0)
    assert math.isfinite(result["dg_bind"])


def test_separation_only_mode_skips_rmsd_stages(tmp_path):
    # run_rmsd_us=False: no RMSD group is built, but Boresch + separation still run,
    # and analyse returns a partial ΔG (sep + boresch + corr) flagged rmsd_included.
    cfg = _config()
    cfg.sampling.run_rmsd_us = False
    calc = _calc(tmp_path, config=cfg)

    assert calc._rmsd_stage_names() == []
    assert calc._group("rmsd") is None
    assert {g.cv_type for g in calc.groups} == {"boresch", "separation"}

    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    result = calc.analyse(_fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0)

    assert result["rmsd_included"] is False
    assert result["dg_rmsd"] == pytest.approx(0.0)
    assert result["dg_bind"] == pytest.approx(
        result["dg_boresch"] + result["dg_sep"] + result["dg_corr"]
    )


def test_boresch_only_mode_skips_rmsd_smd_and_separation(tmp_path):
    cfg = _config()
    cfg.sampling.run_rmsd_us = False
    cfg.sampling.run_separation_us = False
    calc = _calc(tmp_path, config=cfg)

    assert {g.cv_type for g in calc.groups} == {"boresch"}
    assert calc._group("rmsd") is None
    assert calc._group("separation") is None

    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert state.boresch_eq_values[1]["thetaA"] == pytest.approx(1.0)
    assert state.stage_status.get("steered_md") is None
    assert not (tmp_path / "smd").exists()

    with pytest.raises(ValueError, match="separation umbrella sampling is disabled"):
        calc.analyse(_fake_pmf)


def test_rmsd_included_true_by_default(tmp_path):
    result = _calc(tmp_path).analyse(
        _fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0
    )
    assert result["rmsd_included"] is True


def test_upgrade_separation_only_to_full_rmsd(tmp_path):
    # The N-then-M workflow: run separation-only, then upgrade the same run to the
    # full cycle by flipping run_rmsd_us and re-running — it must resume (no
    # config_hash abort) and add only the RMSD stages.
    cfg_off = _config()
    cfg_off.sampling.run_rmsd_us = False
    _calc(tmp_path, config=cfg_off).run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )

    cfg_on = _config()
    cfg_on.sampling.run_rmsd_us = True
    calc = _calc(tmp_path, config=cfg_on)
    assert calc._group("rmsd") is not None  # RMSD stages now built
    calc.run(  # must NOT raise config_hash mismatch; runs only the new RMSD stages
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
    for window in calc._iter_windows():
        assert window.is_complete()  # RMSD windows completed on the upgrade
    result = calc.analyse(_fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0)
    assert result["rmsd_included"] is True


def test_rmsd_stage_names_bound_and_bulk_per_cv(tmp_path):
    # Each custom CV (states validated symmetric) spawns a bound and a bulk stage,
    # in CV order — the telescoping confinement legs.
    cfg = CalculationConfig.model_validate(
        {
            "inputs": INPUTS,
            "restraints": {
                "rmsd_cvs": [
                    {"name": "domainA", "protein": "target", "selection": "resid 1-10"},
                    {
                        "name": "domainB",
                        "protein": "target",
                        "selection": "resid 11-20",
                    },
                ]
            },
        }
    )
    cfg.sampling.ensemble_size = 1
    cfg.sampling.rmsd.window_min = 0.0
    cfg.sampling.rmsd.window_max = 0.2
    cfg.sampling.rmsd.window_spacing = 0.2
    calc = Calculation(
        tmp_path,
        cfg,
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "separation": [1.5]}},
    )
    names = calc._rmsd_stage_names()
    assert names == [
        "domainA_bound",
        "domainA_bulk",
        "domainB_bound",
        "domainB_bulk",
    ]


def test_stage_add_windows_dedup_and_sort(tmp_path):
    from gluebind.runners.stage import Stage

    stage = Stage(
        tmp_path / "rmsd" / "s",
        cv_type="rmsd",
        name="s",
        dof=None,
        centres={1: [0.2, 0.0], 2: [0.2, 0.0]},
        spec_builders={1: _spec_builder, 2: _spec_builder},
        command_factory=_trivial_command,
    )
    added = stage.add_windows([0.1, 0.2])  # 0.2 is a duplicate, 0.1 is new
    assert len(added) == 2  # the new centre, in each replicate
    for replicate in (1, 2):
        windows = stage.windows_for(replicate)
        assert [w.centre for w in windows] == [0.0, 0.1, 0.2]  # kept sorted
        assert all(w.base_dir.parent.name == f"run_{replicate:02d}" for w in windows)


def _wired_calc(tmp_path, **stage_centres):
    return Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={1: {"thetaA": [1.0], "separation": [1.5], **stage_centres}},
    )


def test_add_windows_extends_rmsd_stage(tmp_path):
    calc = _wired_calc(tmp_path)
    stage = calc._group("rmsd").stages[0]
    before = len(stage.windows_for(1))
    calc.add_windows(
        "rmsd", stage.name, [stage.windows_for(1)[0].centre, 99.0]
    )  # one dup, one new
    assert len(stage.windows_for(1)) == before + 1
    assert any(w.centre == 99.0 for w in stage.windows_for(1))


def test_add_windows_separation_requires_snapshot(tmp_path):
    calc = _wired_calc(tmp_path)
    with pytest.raises(ValueError, match="no SMD snapshot"):
        calc.add_windows("separation", "separation", [2.17])  # off the 0.05 nm grid


def test_add_windows_separation_with_snapshot(tmp_path):
    calc = _wired_calc(tmp_path)
    frames = tmp_path / "smd_frames" / "run_01"
    frames.mkdir(parents=True)
    (frames / "2nm.rst7").write_text("frame")  # {2.0:.4g}nm.rst7
    stage = calc.add_windows("separation", "separation", [2.0])
    assert any(w.centre == 2.0 for w in stage.all_windows())


# ---- Boresch distribution warnings -------------------------------------------


def _write_boresch_metadata(base_dir, diagnostics):
    path = base_dir / "prep" / "run_01" / "boresch_distributions" / "metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"frame_count": 10, "diagnostics": diagnostics}))


def test_boresch_distribution_warnings_reach_the_driver(tmp_path, caplog):
    # Restraint resolution records them on a compute node; the driver repeats
    # them, since the job's own output only reaches the scheduler log.
    calc = _calc(tmp_path)
    message = "Boresch thetaA has 2 peaks in its equilibration distribution"
    _write_boresch_metadata(
        tmp_path,
        {
            "thetaA": {"n_peaks": 2, "warnings": [message]},
            "thetaB": {"n_peaks": 1, "warnings": []},
        },
    )
    with pytest.warns(UserWarning, match="thetaA has 2 peaks"):
        calc._warn_boresch_diagnostics()
    assert message in caplog.text


def test_no_boresch_warnings_without_recorded_diagnostics(tmp_path, recwarn):
    calc = _calc(tmp_path)
    calc._warn_boresch_diagnostics()  # no distributions written (explicit centres)
    _write_boresch_metadata(tmp_path, {})
    calc._warn_boresch_diagnostics()
    assert not [w for w in recwarn if "Boresch" in str(w.message)]


# ---- design invariant ------------------------------------------------------


def test_orchestration_layer_is_openmm_free():
    # The driver (config/state/backend/runners) must not import OpenMM — only the
    # compute workers running run_window need it. Checked in a fresh interpreter.
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    code = (
        "import sys; import gluebind.runners, gluebind.backend, gluebind.state, "
        "gluebind.config; "
        "assert 'openmm' not in sys.modules, "
        "[m for m in sys.modules if 'openmm' in m]"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )
    assert result.returncode == 0, result.stderr
