"""Tests for the MM/GBSA backend job, with AmberTools replaced by stub scripts."""

import json
import pathlib
import stat

import pytest

from gluebind.analysis.mmgbsa import MmgbsaSettings, Partition
from gluebind.simulation import mmgbsa as job

DELTA_TOTALS = [-10.0, -12.0, -14.0, -16.0]


def _frames_csv(totals):
    rows = "\n".join(
        f"{i},-1.0,-2.0,3.0,-0.5,-3.0,2.5,{t}" for i, t in enumerate(totals)
    )
    return (
        "GENERALIZED BORN:\n"
        "DELTA Energy Terms\n"
        "Frame #,VDWAALS,EEL,EGB,ESURF,DELTA G gas,DELTA G solv,DELTA TOTAL\n"
        f"{rows}\n\n"
    )


def _stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def amber(tmp_path, monkeypatch):
    """Stub AmberTools in ``<tmp>/amber/bin`` that write the files the job expects
    and record their arguments and AMBERHOME."""
    bin_dir = tmp_path / "amber" / "bin"
    bin_dir.mkdir(parents=True)
    (tmp_path / "frames_template.csv").write_text(_frames_csv(DELTA_TOTALS))
    _stub(
        bin_dir,
        "ante-MMPBSA.py",
        'echo "$@" > ante.args; echo "$AMBERHOME" > amberhome.txt\n'
        "for f in complex.prm7 receptor.prm7 target.prm7; do echo x > $f; done",
    )
    _stub(bin_dir, "cpptraj", "echo x > traj.nc")
    _stub(
        bin_dir,
        "MMPBSA.py",
        f'echo "$@" > mmpbsa.args; cp {tmp_path / "frames_template.csv"} frames.csv\n'
        "echo done > FINAL_RESULTS_MMPBSA.dat",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr(job, "_partition", lambda spec: Partition([1, 2, 3], [4, 5]))
    monkeypatch.setattr(job, "_count_frames", lambda trajectory: 100)
    monkeypatch.setattr(job, "_check_contact", lambda work_dir, partition: None)
    return bin_dir


def _spec(work_dir: pathlib.Path, **settings) -> None:
    job.MmgbsaSpec(
        topology="/data/complex.prm7",
        trajectory="/data/output.dcd",
        ps_per_frame=10.0,
        receptor_molecules=[0],
        target_molecules=[1],
        saltcon_M=0.15,
        settings=MmgbsaSettings(discard_ns=0.0, n_frames=4, n_blocks=2, **settings),
    ).dump(work_dir / job.MMGBSA_SPEC_FILENAME)


def test_spec_round_trips(tmp_path):
    _spec(tmp_path)
    spec = job.MmgbsaSpec.load(tmp_path / job.MMGBSA_SPEC_FILENAME)
    assert spec.settings.n_frames == 4 and spec.saltcon_M == 0.15


def test_launch_command_runs_the_worker_in_place():
    cmd = job.mmgbsa_launch_command("python")
    assert cmd[:2] == ["python", "-c"] and "run_mmgbsa('.')" in cmd[2]


def test_run_mmgbsa_writes_the_result(tmp_path, amber):
    work = tmp_path / "work"
    work.mkdir()
    _spec(work)
    job.run_mmgbsa(work)

    result = json.loads((work / job.MMGBSA_RESULT_FILENAME).read_text())
    assert result["dg_mmgbsa"] == pytest.approx(-13.0)
    assert result["dg_mmgbsa_sem"] > 0
    assert result["dg_per_frame"] == DELTA_TOTALS
    assert result["n_frames"] == 4
    assert (result["receptor_mask"], result["target_mask"]) == (":1-3", ":4-5")
    assert set(result["terms"]) >= {"VDWAALS", "EEL", "EGB", "ESURF", "DELTA TOTAL"}

    # the target is AmberTools' "ligand"; everything that is not solute is stripped
    ante = (work / "ante.args").read_text().split()
    assert ante[ante.index("-n") + 1] == ":4-5"
    assert ante[ante.index("-s") + 1] == "!(:1-5)"
    assert ante[ante.index("-l") + 1] == "target.prm7"
    mmpbsa = (work / "mmpbsa.args").read_text().split()
    assert mmpbsa[mmpbsa.index("-lp") + 1] == "target.prm7"
    # AMBERHOME comes from where the programs are, not from the environment
    assert (work / "amberhome.txt").read_text().strip() == str(amber.parent)
    assert "endframe=4" in (work / "mmgbsa.in").read_text()
    assert not (work / "traj.nc").exists()  # removed unless keep_trajectory


def test_run_mmgbsa_can_keep_the_trajectory(tmp_path, amber):
    _spec(tmp_path, keep_trajectory=True)
    job.run_mmgbsa(tmp_path)
    assert (tmp_path / "traj.nc").exists()


def test_run_mmgbsa_needs_ambertools(tmp_path, amber):
    (amber / "cpptraj").unlink()
    _spec(tmp_path)
    with pytest.raises(FileNotFoundError, match="cpptraj"):
        job.run_mmgbsa(tmp_path)


def test_a_failing_program_stops_the_job(tmp_path, amber):
    _stub(amber, "MMPBSA.py", "echo boom; exit 3")
    _spec(tmp_path)
    failed = r"MMPBSA\.py failed \(exit 3\).*MMPBSA\.log"
    with pytest.raises(RuntimeError, match=failed):
        job.run_mmgbsa(tmp_path)
    assert "boom" in (tmp_path / "MMPBSA.log").read_text()
    assert not (tmp_path / job.MMGBSA_RESULT_FILENAME).exists()


def test_a_missing_output_stops_the_job(tmp_path, amber):
    _stub(amber, "cpptraj", "true")  # exits cleanly but writes no traj.nc
    _spec(tmp_path)
    with pytest.raises(RuntimeError, match="traj.nc"):
        job.run_mmgbsa(tmp_path)


def test_scoring_fewer_frames_than_requested_fails(tmp_path, amber):
    (tmp_path / "frames_template.csv").write_text(_frames_csv([-1.0, -2.0]))
    _spec(tmp_path)
    with pytest.raises(RuntimeError, match="scored 2 frames, expected 4"):
        job.run_mmgbsa(tmp_path)


def test_resid_ranges_for_mdanalysis():
    assert job._resid_ranges([1, 2, 3, 5, 7, 8]) == "1-3 5 7-8"
