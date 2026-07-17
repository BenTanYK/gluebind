"""Tests for the runner hierarchy: tree construction, window enumeration,
end-to-end orchestration (LocalBackend), resume, and analysis aggregation.

These exercise the orchestration with a trivial ``spec_builder`` and a trivial
job command (no OpenMM MD), so they run fast and don't need real structures.
"""

import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from gluebind.backend import LocalBackend, Scheduler
from gluebind.config import CalculationConfig, WindowSampling
from gluebind.runners import Calculation, enumerate_centres, format_label
from gluebind.simulation import WindowSpec

INPUTS = {
    "target": {"prm7": "t.prm7", "rst7": "t.rst7"},
    "receptor": {"prm7": "r.prm7", "rst7": "r.rst7"},
    "glue": {"sdf": "g.sdf", "assign_to": "receptor"},
}
CENTRES = {"thetaA": [1.0], "separation": [1.5]}


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


# ---- enumeration / labels --------------------------------------------------


def test_enumerate_from_range():
    s = WindowSampling(
        force_constant=5.0, sampling_time_ns=1.0, window_min=0.0, window_max=1.0, window_spacing=0.5
    )
    assert enumerate_centres(s) == [0.0, 0.5, 1.0]


def test_enumerate_explicit_centres():
    s = WindowSampling(force_constant=5.0, sampling_time_ns=1.0, centres=[1.0, 2.0])
    assert enumerate_centres(s) == [1.0, 2.0]


def test_enumerate_requires_info():
    with pytest.raises(ValueError):
        enumerate_centres(WindowSampling(force_constant=5.0, sampling_time_ns=1.0))


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
    theta_a.write_specs({"thetaX": 0.5})
    spec = WindowSpec.load(tmp_path / "boresch" / "thetaA" / "1rad" / "run_01" / "window.json")
    assert spec.cv_type == "boresch" and spec.dof == "thetaA"
    assert spec.restraints["boresch_eq_values"] == {"thetaX": 0.5}


# ---- run / resume ----------------------------------------------------------


def test_run_completes_and_records_state(tmp_path):
    calc = _calc(tmp_path)
    state = calc.run(scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf)
    for window in calc._iter_windows():
        assert window.is_replicate_complete(1)
    assert (tmp_path / ".gluebind-state.json").exists()
    assert state.handles  # handles recorded via on_submit
    assert state.stage_status.get("thetaA") == "done"


def test_run_is_idempotent(tmp_path):
    calc = _calc(tmp_path)
    calc.run(scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf)
    calc2 = _calc(tmp_path)
    pending = [
        (w.stage_name, w.label, r)
        for w in calc2._iter_windows()
        for r in w.replicates()
        if not w.is_replicate_complete(r)
    ]
    assert pending == []


def test_resume_config_hash_mismatch_aborts(tmp_path):
    _calc(tmp_path).run(scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf)
    changed = _config()
    changed.sampling.rmsd.force_constant = 99.0
    calc2 = _calc(tmp_path, config=changed)
    with pytest.raises(ValueError, match="config_hash"):
        calc2.run(scheduler=Scheduler(calc2.backend, poll_interval=0.01), pmf_provider=_fake_pmf)


def test_run_requires_pmf_provider_for_boresch(tmp_path):
    calc = _calc(tmp_path)  # has a thetaA Boresch stage
    with pytest.raises(ValueError, match="pmf_provider"):
        calc.run(scheduler=Scheduler(calc.backend, poll_interval=0.01))


def test_boresch_sequential_feedback(tmp_path):
    # Two Boresch DoFs: thetaA runs first, its PMF minimum (=1.0) is fed forward
    # as a fixed restraint into thetaB's windows and into separation.
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]},
    )
    state = calc.run(scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf)

    assert state.boresch_eq_values["thetaA"] == pytest.approx(1.0)
    assert state.boresch_eq_values["thetaB"] == pytest.approx(1.0)

    theta_b_spec = WindowSpec.load(tmp_path / "boresch" / "thetaB" / "1rad" / "run_01" / "window.json")
    assert theta_b_spec.restraints["boresch_eq_values"]["thetaA"] == pytest.approx(1.0)

    # group dir and stage name are both "separation" (uniform tree depth).
    sep_spec = WindowSpec.load(
        tmp_path / "separation" / "separation" / "1.5nm" / "run_01" / "window.json"
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
        stage_centres={"thetaA": [1.0], "separation": [1.5]},
        steered_md_runner=lambda eq: calls.append(dict(eq)),
    )
    state = calc.run(scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf)
    assert len(calls) == 1
    assert "thetaA" in calls[0]  # invoked with the resolved Boresch eq values
    assert state.stage_status.get("steered_md") == "done"


def test_run_without_separation_skips_steered_md(tmp_path):
    calls = []
    calc = Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={"thetaA": [1.0]},  # no separation group
        steered_md_runner=lambda eq: calls.append(eq),
    )
    calc.run(scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf)
    assert calls == []


def test_steered_md_runs_once_across_resume(tmp_path):
    counter = {"n": 0}

    def _mk():
        return Calculation(
            tmp_path,
            _config(),
            LocalBackend(),
            _spec_builder,
            command_factory=_trivial_command,
            stage_centres={"thetaA": [1.0], "separation": [1.5]},
            steered_md_runner=lambda eq: counter.__setitem__("n", counter["n"] + 1),
        )

    _mk().run(scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf)
    _mk().run(scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf)
    assert counter["n"] == 1  # not repeated on the resumed run


# ---- analysis --------------------------------------------------------------


def test_analyse_aggregates(tmp_path):
    calc = _calc(tmp_path)

    def fake_pmf(stage):
        x = np.linspace(0.0, 1.0, 11)
        return x, np.zeros_like(x)

    result = calc.analyse(fake_pmf, r_star_nm=0.8, theta_a_min=1.2, theta_b_min=1.4)
    assert set(result) == {"dg_bind", "dg_rmsd", "dg_boresch", "dg_sep", "dg_corr"}
    assert math.isfinite(result["dg_bind"])
    assert result["dg_bind"] == pytest.approx(
        result["dg_rmsd"] + result["dg_boresch"] + result["dg_sep"] + result["dg_corr"]
    )


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
