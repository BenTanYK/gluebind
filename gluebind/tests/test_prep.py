"""Tests for the Phase 3 prep layer's pure helpers and the prepared manifest.

The BioSimSpace/MD driver itself is verified in integration (Phase 7); here we
cover the force-field validation, box sizing, multi-molecule layout bookkeeping,
and the PreparedSystem manifest — none of which need BSS.
"""

import pathlib

import pytest

from gluebind.backend.base import Backend, JobState
from gluebind.config.prep import PrepConfig
from gluebind.simulation.prep_stage import (
    PREP_STAGE_SPEC_FILENAME,
    PrepStageSpec,
    prep_stage_launch_command,
)
from gluebind.system import compute_layout
from gluebind.system.prep import (
    PreparedSystem,
    box_length,
    equilibration_stage_plan,
    normalise_ff_name,
    run_equilibration_stages,
    validate_forcefield,
)


def test_box_length():
    assert box_length([0, 0, 0], [1, 2, 3], 1.5) == 6.0  # max dim 3 + 2*1.5


def test_normalise_ff_name():
    assert (
        normalise_ff_name("openff_unconstrained-2.2.1") == "openff_unconstrained_2_2_1"
    )
    assert normalise_ff_name("gaff2") == "gaff2"


def test_validate_forcefield_ok():
    assert validate_forcefield("gaff2", ["gaff2", "ff14SB"]) == "gaff2"


def test_validate_forcefield_normalises_dash_dot():
    assert (
        validate_forcefield(
            "openff_unconstrained_2.2.1", ["openff_unconstrained-2.2.1"]
        )
        == "openff_unconstrained_2_2_1"
    )


def test_validate_forcefield_unknown_raises():
    # the real env has only the -rc1 variant, not plain 2.2.1
    with pytest.raises(ValueError):
        validate_forcefield(
            "openff_unconstrained_2.2.1", ["gaff2", "openff_unconstrained-2.2.1-rc1"]
        )


def test_validate_glue_resname_accepts_mol():
    from gluebind.system.inputs import validate_glue_resname

    validate_glue_resname(["MOL"])  # single MOL residue: fine


def test_validate_glue_resname_rejects_other():
    from gluebind.system.inputs import validate_glue_resname

    with pytest.raises(ValueError, match="must be named 'MOL'"):
        validate_glue_resname(["LIG"])
    with pytest.raises(ValueError, match="must be named 'MOL'"):
        validate_glue_resname([])  # no residues


def test_compute_layout_single_chain():
    # assembly order: glue (MOL) first, then receptor, then target
    layout = compute_layout(1, 1, has_glue=True)
    assert layout.glue == 0
    assert layout.receptor == [1]
    assert layout.target == [2]
    assert layout.n_molecules == 3


def test_compute_layout_multichain_target():
    # a chain-split target (e.g. BRD4 tandem bromodomains -> 2 molecules)
    layout = compute_layout(2, 1, has_glue=True)
    assert layout.glue == 0
    assert layout.receptor == [1]
    assert layout.target == [2, 3]


def test_compute_layout_no_glue():
    layout = compute_layout(1, 1, has_glue=False)
    assert layout.glue is None
    assert layout.n_molecules == 2


def test_compute_layout_requires_molecules():
    with pytest.raises(ValueError):
        compute_layout(0, 1, has_glue=True)


def test_prepared_system_roundtrip(tmp_path):
    prepared = PreparedSystem(
        complex_prm7="complex_equil.prm7",
        complex_rst7="complex_equil.rst7",
        complex_trajectory="complex_equil.dcd",
        target_bulk_prm7="target_bulk.prm7",
        target_bulk_rst7="target_bulk.rst7",
        receptor_bulk_prm7="receptor_bulk.prm7",
        receptor_bulk_rst7="receptor_bulk.rst7",
        glue_assign_to="receptor",
        target_molecules=[0],
        receptor_molecules=[1],
        glue_molecule=2,
    )
    prepared.dump(tmp_path)
    assert PreparedSystem.load(tmp_path) == prepared


# ---- equilibration staging (per-stage jobs) --------------------------------


def test_equilibration_stage_plan_structure():
    plan = equilibration_stage_plan(PrepConfig())
    stages = [s["stage"] for s in plan]
    # four stages, in order; the old intermediate restrained-NVT stage is gone
    assert stages == ["minimisation", "nvt_heat", "npt", "equilibration"]
    assert "nvt" not in stages

    by = {s["stage"]: s for s in plan}
    assert by["minimisation"]["kind"] == "minimisation"
    assert all(
        by[s]["kind"] == "equilibration" for s in ["nvt_heat", "npt", "equilibration"]
    )
    # NVT heating: ramp 10 K -> production T, backbone-restrained, no barostat
    assert by["nvt_heat"]["temperature_start_K"] == 10.0
    assert by["nvt_heat"]["restraint"] == "backbone"
    assert by["nvt_heat"]["pressure"] is False
    # NPT: barostat on, backbone-restrained
    assert by["npt"]["pressure"] is True
    assert by["npt"]["restraint"] == "backbone"
    # production: unrestrained NVT
    assert by["equilibration"]["pressure"] is False
    assert by["equilibration"]["restraint"] == "none"


def test_prep_stage_spec_roundtrip(tmp_path):
    spec = PrepStageSpec(
        stage="npt",
        kind="equilibration",
        input_prm7="in.prm7",
        input_rst7="in.rst7",
        runtime_ns=0.4,
        temperature_start_K=300.0,
        temperature_end_K=300.0,
        pressure=True,
        restraint="backbone",
    )
    path = spec.dump(tmp_path / PREP_STAGE_SPEC_FILENAME)
    assert PrepStageSpec.load(path) == spec


def test_prep_stage_launch_command():
    cmd = prep_stage_launch_command()
    assert cmd[:2] == ["python", "-c"]
    assert "run_prep_stage" in cmd[2]


class _FakeStageBackend(Backend):
    """Simulates each prep stage without BioSimSpace: writes the output structures
    the next stage (and the orchestrator) expect, then reports the job finished."""

    def __init__(self):
        self.submitted: list[PrepStageSpec] = []
        self._counter = 0

    def submit(self, spec):
        wd = pathlib.Path(spec.work_dir)
        stage_spec = PrepStageSpec.load(wd / PREP_STAGE_SPEC_FILENAME)
        self.submitted.append(stage_spec)
        prefix = wd / stage_spec.output_prefix
        prefix.with_suffix(".prm7").write_text("prm7")
        prefix.with_suffix(".rst7").write_text("rst7")
        if stage_spec.kind == "equilibration":
            prefix.with_suffix(".dcd").write_text("dcd")
        self._counter += 1
        return f"fake-{self._counter}"

    def poll(self, handles):
        return dict.fromkeys(handles, JobState.FINISHED)

    def cancel(self, handle):  # pragma: no cover - not exercised
        pass


def test_run_equilibration_stages_chains_outputs(tmp_path):
    backend = _FakeStageBackend()
    plan = equilibration_stage_plan(PrepConfig())
    (tmp_path / "solvated.prm7").write_text("s")
    (tmp_path / "solvated.rst7").write_text("s")

    final_prm7, final_rst7, traj = run_equilibration_stages(
        tmp_path / "solvated.prm7",
        tmp_path / "solvated.rst7",
        plan,
        tmp_path / "equilibration",
        backend,
        platform="CPU",
        poll_interval=0.0,
    )

    # one job per stage, in order, each in its own numbered subdir
    assert [s.stage for s in backend.submitted] == [
        "minimisation",
        "nvt_heat",
        "npt",
        "equilibration",
    ]
    subdirs = sorted(p.name for p in (tmp_path / "equilibration").iterdir())
    assert subdirs == ["01_minimisation", "02_nvt_heat", "03_npt", "04_equilibration"]

    # each stage's input is the previous stage's output (dependency chain)
    assert backend.submitted[0].input_prm7.endswith("solvated.prm7")
    assert backend.submitted[1].input_prm7.endswith("01_minimisation/output.prm7")
    assert backend.submitted[2].input_prm7.endswith("02_nvt_heat/output.prm7")
    assert backend.submitted[3].input_prm7.endswith("03_npt/output.prm7")

    # final structures come from the last stage; trajectory from the production run
    assert final_prm7.endswith("04_equilibration/output.prm7")
    assert final_rst7.endswith("04_equilibration/output.rst7")
    assert traj is not None and traj.endswith("04_equilibration/output.dcd")


def test_run_equilibration_stages_skips_completed(tmp_path):
    backend = _FakeStageBackend()
    plan = equilibration_stage_plan(PrepConfig())
    (tmp_path / "solvated.prm7").write_text("s")
    (tmp_path / "solvated.rst7").write_text("s")
    # pre-create stage 1 (minimisation) output as if a previous run finished it
    stage1 = tmp_path / "equilibration" / "01_minimisation"
    stage1.mkdir(parents=True)
    (stage1 / "output.prm7").write_text("done")
    (stage1 / "output.rst7").write_text("done")

    final_prm7, _, _ = run_equilibration_stages(
        tmp_path / "solvated.prm7",
        tmp_path / "solvated.rst7",
        plan,
        tmp_path / "equilibration",
        backend,
        platform="CPU",
        poll_interval=0.0,
    )

    # minimisation was skipped (resume); only the remaining stages submitted
    assert [s.stage for s in backend.submitted] == ["nvt_heat", "npt", "equilibration"]
    # and the next stage's input chains from the pre-existing stage-1 output
    assert backend.submitted[0].input_prm7.endswith("01_minimisation/output.prm7")
    assert final_prm7.endswith("04_equilibration/output.prm7")
