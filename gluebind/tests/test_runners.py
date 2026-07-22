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
        force_constant=5.0,
        sampling_time_ns=1.0,
        window_min=0.0,
        window_max=1.0,
        window_spacing=0.5,
    )
    assert enumerate_centres(s) == [0.0, 0.5, 1.0]


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


def test_separation_default_enumerates():
    # regression: the separation default must be enumerable (previously raised)
    from gluebind.config.sampling import SamplingConfig

    centres = enumerate_centres(SamplingConfig().for_cv("separation", "separation"))
    assert centres[0] == 0.9 and max(centres) <= 3.0


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
    spec = WindowSpec.load(
        tmp_path / "boresch" / "thetaA" / "1rad" / "run_01" / "window.json"
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
        assert window.is_replicate_complete(1)
    assert (tmp_path / ".gluebind-state.json").exists()
    assert state.handles  # handles recorded via on_submit
    assert state.stage_status.get("thetaA") == "done"


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
        (w.stage_name, w.label, r)
        for w in calc2._iter_windows()
        for r in w.replicates()
        if not w.is_replicate_complete(r)
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
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )

    assert state.boresch_eq_values["thetaA"] == pytest.approx(1.0)
    assert state.boresch_eq_values["thetaB"] == pytest.approx(1.0)

    theta_b_spec = WindowSpec.load(
        tmp_path / "boresch" / "thetaB" / "1rad" / "run_01" / "window.json"
    )
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
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
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
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )
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

    _mk().run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )
    _mk().run(
        scheduler=Scheduler(LocalBackend(), poll_interval=0.01), pmf_provider=_fake_pmf
    )
    assert counter["n"] == 1  # not repeated on the resumed run


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
        "stage_sems",
        "rmsd_included",
    }
    assert math.isfinite(result["dg_bind"])
    assert result["dg_bind"] == pytest.approx(
        result["dg_rmsd"] + result["dg_boresch"] + result["dg_sep"] + result["dg_corr"]
    )


def test_analyse_reports_sem_over_repeats(tmp_path):
    # A provider that exposes per-replicate PMFs yields an overall SEM (over the
    # repeats' total ΔG) plus per-stage SEM diagnostics.
    calc = _calc(tmp_path)

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


def test_analyse_sem_none_without_repeats(tmp_path):
    # A 2-tuple provider (no per-replicate PMFs) cannot report an uncertainty.
    calc = _calc(tmp_path)
    result = calc.analyse(_fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0)
    assert result["dg_bind_sem"] is None
    assert result["stage_sems"] == {}


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
    calc = Calculation.from_config(_config(), tmp_path, LocalBackend())
    assert calc.spec_builder is None
    assert calc.groups == []


def test_run_auto_prepares_when_not_wired(tmp_path):
    # run() on a from_config calculation calls prepare() itself (end to end from
    # a single call). We stub prepare() to wire trivially, avoiding real MD/BSS.
    calc = Calculation.from_config(_config(), tmp_path, LocalBackend())
    calls = {"prepare": 0}

    def fake_prepare():
        calls["prepare"] += 1
        calc.spec_builder = _spec_builder
        calc.command_factory = _trivial_command
        calc.stage_centres = {"thetaA": [1.0], "separation": [1.5]}
        calc.groups = calc._build_groups()
        calc.sub_runners = list(calc.groups)

    calc.prepare = fake_prepare
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
        stage_centres={"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]},
    )
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01), pmf_provider=_fake_pmf
    )

    # No r_star/theta passed: theta minima from the run state, r_star from the
    # outermost separation centre (1.5).
    result = calc.analyse(_fake_pmf)
    assert set(result) == {
        "dg_bind",
        "dg_bind_sem",
        "dg_rmsd",
        "dg_boresch",
        "dg_sep",
        "dg_corr",
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
        stage_centres={"thetaA": [1.0], "separation": [1.5]},
    )
    state = calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01)
    )  # no pmf_provider
    assert state.boresch_eq_values["thetaA"] == pytest.approx(1.0)


def _dump_prepared(base_dir):
    from gluebind.system.prep import PreparedSystem

    PreparedSystem(
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
    ).dump(base_dir / "prep")


def test_analyse_auto_wires_from_prepared_in_fresh_process(tmp_path):
    # Fresh process: an unwired from_config calc analysing an already-prepared run
    # must re-wire from disk (rebuild the stage tree) so stages are iterated — not
    # silently return zero contributions. _wire is stubbed to avoid MDA.
    _dump_prepared(tmp_path)
    calc = Calculation.from_config(_config(), tmp_path, LocalBackend())

    def fake_wire(prepared):
        calc.spec_builder = _spec_builder
        calc.command_factory = _trivial_command
        calc.stage_centres = {"thetaA": [1.0], "thetaB": [1.0], "separation": [1.5]}
        calc.groups = calc._build_groups()
        calc.sub_runners = list(calc.groups)

    calc._wire = fake_wire
    result = calc.analyse(_fake_pmf, theta_a_min=1.0, theta_b_min=1.0)

    assert (
        calc.spec_builder is not None and len(calc.groups) > 0
    )  # re-wired, tree rebuilt
    assert set(result) == {
        "dg_bind",
        "dg_bind_sem",
        "dg_rmsd",
        "dg_boresch",
        "dg_sep",
        "dg_corr",
        "stage_sems",
        "rmsd_included",
    }
    assert math.isfinite(result["dg_bind"])


def test_analyse_raises_when_not_prepared(tmp_path):
    calc = Calculation.from_config(_config(), tmp_path, LocalBackend())
    with pytest.raises(RuntimeError, match="not prepared"):
        calc.analyse(_fake_pmf, theta_a_min=1.0, theta_b_min=1.0)


def test_analyse_r_star_falls_back_to_config(tmp_path):
    # When stage_centres lacks separation (e.g. a fresh/degenerate wiring), r_star
    # falls back to the config schedule instead of raising.
    cfg = _config()
    cfg.sampling.separation.window_min = 1.15
    cfg.sampling.separation.window_max = 3.0
    cfg.sampling.separation.window_spacing = 0.5
    calc = Calculation(
        tmp_path,
        cfg,
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={"thetaA": [1.0], "separation": [1.5]},
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


def test_rmsd_included_true_by_default(tmp_path):
    result = _calc(tmp_path).analyse(
        _fake_pmf, r_star_nm=1.5, theta_a_min=1.0, theta_b_min=1.0
    )
    assert result["rmsd_included"] is True


def test_rmsd_stage_names_respect_states(tmp_path):
    # A custom CV sampled only in the bound state must not spawn a _bulk stage.
    cfg = CalculationConfig.model_validate(
        {
            "inputs": INPUTS,
            "restraints": {
                "rmsd_cvs": [
                    {
                        "name": "domainA",
                        "selection": "resid 1-10",
                        "states": ["bound", "bulk"],
                    },
                    {
                        "name": "domainB",
                        "selection": "resid 11-20",
                        "states": ["bound"],
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
        stage_centres={"thetaA": [1.0], "separation": [1.5]},
    )
    names = calc._rmsd_stage_names()
    assert names == ["domainA_bound", "domainA_bulk", "domainB_bound"]


def test_stage_add_windows_dedup_and_sort(tmp_path):
    from gluebind.runners.stage import Stage

    stage = Stage(
        tmp_path / "rmsd" / "s",
        cv_type="rmsd",
        name="s",
        dof=None,
        centres=[0.2, 0.0],
        ensemble_size=1,
        spec_builder=_spec_builder,
        command_factory=_trivial_command,
    )
    added = stage.add_windows([0.1, 0.2])  # 0.2 is a duplicate, 0.1 is new
    assert len(added) == 1
    assert [w.centre for w in stage.windows] == [0.0, 0.1, 0.2]  # kept sorted


def _wired_calc(tmp_path, **stage_centres):
    return Calculation(
        tmp_path,
        _config(),
        LocalBackend(),
        _spec_builder,
        command_factory=_trivial_command,
        stage_centres={"thetaA": [1.0], "separation": [1.5], **stage_centres},
    )


def test_add_windows_extends_rmsd_stage(tmp_path):
    calc = _wired_calc(tmp_path)
    stage = calc._group("rmsd").stages[0]
    before = len(stage.windows)
    calc.add_windows(
        "rmsd", stage.name, [stage.windows[0].centre, 99.0]
    )  # one dup, one new
    assert len(stage.windows) == before + 1
    assert any(w.centre == 99.0 for w in stage.windows)


def test_add_windows_separation_requires_snapshot(tmp_path):
    calc = _wired_calc(tmp_path)
    with pytest.raises(ValueError, match="no SMD snapshot"):
        calc.add_windows("separation", "separation", [2.17])  # off the 0.05 nm grid


def test_add_windows_separation_with_snapshot(tmp_path):
    calc = _wired_calc(tmp_path)
    frames = tmp_path / "smd_frames"
    frames.mkdir(parents=True)
    (frames / "2nm.rst7").write_text("frame")  # {2.0:.4g}nm.rst7
    stage = calc.add_windows("separation", "separation", [2.0])
    assert any(w.centre == 2.0 for w in stage.windows)


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
