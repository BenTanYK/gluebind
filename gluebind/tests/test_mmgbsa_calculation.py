"""Tests for Calculation.mmgbsa(): submission, reuse and error handling (the job is
replaced by a command that writes a canned result)."""

import json
import sys

import pytest

import gluebind.simulation.mmgbsa as mmgbsa_job
from gluebind import CalculationConfig
from gluebind.analysis.mmgbsa import MmgbsaSettings
from gluebind.backend import LocalBackend
from gluebind.runners import Calculation
from gluebind.simulation.production import ProductionSpec
from gluebind.state import RunState
from gluebind.system.prep import PreparedSystem

INPUTS = {
    "target": {"prm7": "target.prm7", "rst7": "target.rst7"},
    "receptor": {"prm7": "receptor.prm7", "rst7": "receptor.rst7"},
}
RESULT = {"dg_mmgbsa": -21.5, "dg_mmgbsa_sem": 1.2, "n_frames": 200}


ELSEWHERE = "/cluster-a/runs/system/prep/equilibration/04_production"
"""Where prepared.json says the production files are: the machine the run was
prepared on, as for a run copied to another machine for analysis."""


def _prepare(base_dir, *, glue=False, production_files=True):
    """A minimal prepared system: the manifest, the production spec and (empty)
    production topology and trajectory files."""
    production = base_dir / "prep" / "equilibration" / "04_production"
    production.mkdir(parents=True)
    ProductionSpec(
        topology="output.prm7",
        coordinates="output.rst7",
        runtime_ns=50.0,
        timestep_fs=4.0,
        sample_interval_steps=2500,
    ).dump(production / "production.json")
    if production_files:
        for name in ("output.prm7", "output.dcd"):
            (production / name).write_text("")
    PreparedSystem(
        complex_prm7=f"{ELSEWHERE}/output.prm7",
        complex_rst7=f"{ELSEWHERE}/output.rst7",
        complex_trajectory=f"{ELSEWHERE}/output.dcd",
        target_bulk_prm7="t.prm7",
        target_bulk_rst7="t.rst7",
        receptor_bulk_prm7="r.prm7",
        receptor_bulk_rst7="r.rst7",
        glue_assign_to="target" if glue else None,
        receptor_molecules=[1, 2] if glue else [0, 1],
        target_molecules=[3] if glue else [2],
        glue_molecule=0 if glue else None,
    ).dump(base_dir / "prep")


@pytest.fixture
def fake_job(monkeypatch, tmp_path):
    """Replace the MM/GBSA worker; count submissions in ``<tmp>/submissions``."""
    counter = tmp_path / "submissions"
    code = (
        "import json; "
        f"open({str(counter)!r}, 'a').write('x'); "
        f"json.dump({RESULT!r}, open('result.json', 'w'))"
    )
    monkeypatch.setattr(
        mmgbsa_job, "mmgbsa_launch_command", lambda: [sys.executable, "-c", code]
    )
    return counter


def _calc(base_dir):
    config = CalculationConfig.model_validate({"inputs": INPUTS})
    return Calculation(base_dir, config, LocalBackend(), poll_interval=0.01)


def _spec(base_dir):
    (spec,) = (base_dir / "mmgbsa").glob("*/mmgbsa.json")
    return json.loads(spec.read_text())


def test_mmgbsa_submits_the_job_and_returns_its_result(tmp_path, fake_job):
    run = tmp_path / "run"
    _prepare(run, glue=True)
    result = _calc(run).mmgbsa()
    assert result == RESULT
    assert fake_job.read_text() == "x"

    spec = _spec(run)
    assert spec["ps_per_frame"] == pytest.approx(10.0)  # 2500 steps x 4 fs
    # found in this run directory, not at prepared.json's paths on another machine
    production = (run / "prep" / "equilibration" / "04_production").resolve()
    assert spec["topology"] == str(production / "output.prm7")
    assert spec["trajectory"] == str(production / "output.dcd")
    assert (spec["receptor_molecules"], spec["target_molecules"]) == ([1, 2], [3])
    assert (spec["glue_molecule"], spec["glue_assign_to"]) == (0, "target")
    assert spec["saltcon_M"] == pytest.approx(0.15)  # the system's ion concentration
    resolved = MmgbsaSettings(saltcon_M=0.15)  # the default, salt resolved
    assert spec["settings"] == resolved.model_dump(mode="json")

    state = RunState.load(run)
    assert state.handles["_auxiliary"]["mmgbsa"]  # recorded for kill()


def test_mmgbsa_reuses_a_stored_result(tmp_path, fake_job):
    run = tmp_path / "run"
    _prepare(run)
    calc = _calc(run)
    assert calc.mmgbsa() == calc.mmgbsa() == RESULT
    assert fake_job.read_text() == "x"  # submitted once


def test_mmgbsa_keeps_results_per_settings(tmp_path, fake_job):
    run = tmp_path / "run"
    _prepare(run)
    calc = _calc(run)
    calc.mmgbsa()
    calc.mmgbsa(MmgbsaSettings(discard_ns=20.0))
    calc.mmgbsa(MmgbsaSettings(saltcon_M=0.15))  # same as the default: reused
    assert fake_job.read_text() == "xx"
    assert len(list((run / "mmgbsa").iterdir())) == 2


def test_mmgbsa_salt_can_be_set_explicitly(tmp_path, fake_job):
    run = tmp_path / "run"
    _prepare(run)
    _calc(run).mmgbsa(MmgbsaSettings(saltcon_M=0.0))
    assert _spec(run)["saltcon_M"] == 0.0


def test_mmgbsa_ion_concentration_overrides_the_system(tmp_path, fake_job):
    # e.g. sampled with no added salt, scored at 0.15 M
    run = tmp_path / "run"
    _prepare(run)
    calc = _calc(run)
    calc.config.prep.ion_concentration_M = 0.0
    calc.mmgbsa(ion_concentration_M=0.15)
    assert _spec(run)["saltcon_M"] == pytest.approx(0.15)
    # the same request through settings reuses that result
    calc.mmgbsa(MmgbsaSettings(saltcon_M=0.15))
    assert fake_job.read_text() == "x"


@pytest.mark.parametrize(
    "settings, ion, match",
    [
        (MmgbsaSettings(saltcon_M=0.1), 0.15, "conflicts"),
        (None, -0.15, ">= 0"),
    ],
)
def test_mmgbsa_rejects_a_bad_ion_concentration(
    tmp_path, fake_job, settings, ion, match
):
    run = tmp_path / "run"
    _prepare(run)
    with pytest.raises(ValueError, match=match):
        _calc(run).mmgbsa(settings, ion_concentration_M=ion)
    assert not fake_job.exists()  # nothing submitted


def test_mmgbsa_needs_a_prepared_system(tmp_path, fake_job):
    with pytest.raises(RuntimeError, match="run prepare\\(\\) first"):
        _calc(tmp_path).mmgbsa()


def test_mmgbsa_needs_the_production_files_in_the_run_directory(tmp_path, fake_job):
    run = tmp_path / "run"
    _prepare(run, production_files=False)  # manifest present, trajectory not copied
    with pytest.raises(RuntimeError, match="04_production"):
        _calc(run).mmgbsa()
    assert not fake_job.exists()


def test_mmgbsa_raises_when_the_job_writes_no_result(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _prepare(run)
    monkeypatch.setattr(
        mmgbsa_job, "mmgbsa_launch_command", lambda: [sys.executable, "-c", "pass"]
    )
    with pytest.raises(RuntimeError, match="MM/GBSA job did not produce its result"):
        _calc(run).mmgbsa()
