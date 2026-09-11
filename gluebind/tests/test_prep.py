"""Tests for the Phase 3 prep layer's pure helpers and the prepared manifest.

The BioSimSpace/MD driver itself is verified in integration (Phase 7); here we
cover the force-field validation, box sizing, multi-molecule layout bookkeeping,
and the PreparedSystem manifest — none of which need BSS.
"""

import json
import pathlib
import sys
import types

import pytest

from gluebind.backend.base import Backend, JobState
from gluebind.config.prep import PrepConfig
from gluebind.simulation.bulk_build import (
    BULK_BUILD_SPEC_FILENAME,
    BulkBuildResult,
    BulkBuildSpec,
    bulk_build_launch_command,
)
from gluebind.simulation.prep_stage import (
    PREP_STAGE_SPEC_FILENAME,
    PREP_STAGE_RESULT_FILENAME,
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


def test_bulk_padding_default_is_larger_than_complex_padding():
    config = PrepConfig()
    assert config.box_padding_angstrom == 15.0
    assert config.bulk_box_padding_angstrom == 20.0


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


def test_parameterise_glue_charge_is_optional(monkeypatch):
    from gluebind.system.prep import parameterise_glue

    calls = []

    class _Parameterised:
        def getMolecule(self):
            return "parameterised"

    def fake_gaff2(molecule, **kwargs):
        calls.append(kwargs)
        return _Parameterised()

    monkeypatch.setattr("gluebind.system.prep.available_forcefields", lambda: ["gaff2"])
    monkeypatch.setattr("gluebind.system.prep.load_glue", lambda path: "molecule")
    monkeypatch.setitem(
        sys.modules,
        "BioSimSpace",
        types.SimpleNamespace(Parameters=types.SimpleNamespace(gaff2=fake_gaff2)),
    )

    parameterise_glue("glue.mol2", "gaff2")
    parameterise_glue("glue.mol2", "gaff2", ligand_charge=-1)

    assert calls == [{}, {"net_charge": -1}]


def test_validate_glue_resname_accepts_mol():
    from gluebind.system.inputs import validate_glue_resname

    validate_glue_resname(["MOL"])  # single MOL residue: fine


def test_validate_glue_resname_rejects_other():
    from gluebind.system.inputs import validate_glue_resname

    with pytest.raises(ValueError, match="must be named 'MOL'"):
        validate_glue_resname(["LIG"])
    with pytest.raises(ValueError, match="must be named 'MOL'"):
        validate_glue_resname([])  # no residues


def test_load_glue_rejects_unknown_extension():
    from gluebind.system.inputs import load_glue

    with pytest.raises(ValueError, match=".sdf or .mol2"):
        load_glue("glue.pdb")


def test_validate_waters_resnames_accepts_water_names():
    from gluebind.system.inputs import validate_waters_resnames

    validate_waters_resnames(["WAT"])
    validate_waters_resnames(["HOH", "WAT"])  # mixed common conventions: fine
    validate_waters_resnames(["hoh"])  # case-insensitive


def test_validate_waters_resnames_rejects_non_water_and_empty():
    from gluebind.system.inputs import validate_waters_resnames

    with pytest.raises(ValueError, match="only crystal-water"):
        validate_waters_resnames(["WAT", "ALA"])  # a protein residue slipped in
    with pytest.raises(ValueError, match="only crystal-water"):
        validate_waters_resnames(["NA"])  # an ion belongs in its own input
    with pytest.raises(ValueError, match="no residues"):
        validate_waters_resnames([])


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


@pytest.mark.parametrize("schema_version", [None, 0, 2])
def test_prepared_system_requires_explicit_current_schema(tmp_path, schema_version):
    prepared = PreparedSystem(
        complex_prm7="complex.prm7",
        complex_rst7="complex.rst7",
        target_bulk_prm7="target_bulk.prm7",
        target_bulk_rst7="target_bulk.rst7",
        receptor_bulk_prm7="receptor_bulk.prm7",
        receptor_bulk_rst7="receptor_bulk.rst7",
        target_molecules=[0],
        receptor_molecules=[1],
    )
    path = prepared.dump(tmp_path)
    data = json.loads(path.read_text())
    if schema_version is None:
        del data["schema_version"]
    else:
        data["schema_version"] = schema_version
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="requires v1"):
        PreparedSystem.load(tmp_path)


def test_bulk_build_spec_roundtrip_and_command(tmp_path):
    spec = BulkBuildSpec(
        complex_prm7="complex.prm7",
        complex_rst7="complex.rst7",
        molecule_indices=[1, 2],
        prep=PrepConfig(),
        output_dir="target_bulk",
    )
    path = spec.dump(tmp_path / BULK_BUILD_SPEC_FILENAME)
    assert BulkBuildSpec.load(path) == spec
    command = bulk_build_launch_command()
    assert command[:2] == ["python", "-c"]
    assert "run_bulk_build" in command[2]


def test_run_bulk_build_extracts_and_solvates_component(tmp_path, monkeypatch):
    """Exercise the BSS worker contract with a tiny in-memory BSS double."""
    from gluebind.simulation import bulk_build

    class Molecule:
        def __init__(self, indices):
            self.indices = indices

        def __add__(self, other):
            return Molecule(self.indices + other.indices)

        def getAxisAlignedBoundingBox(self):
            return ([0, 0, 0], [2, 3, 1])

        def toSystem(self):
            return self

    class System:
        def __getitem__(self, index):
            return Molecule([index])

    saved = []
    solvent = []
    fake_bss = types.SimpleNamespace(
        IO=types.SimpleNamespace(
            readMolecules=lambda files: System(),
            saveMolecules=lambda prefix, system, formats: saved.append(
                (prefix, system, formats)
            ),
        ),
        Units=types.SimpleNamespace(Length=types.SimpleNamespace(angstrom=1)),
        Box=types.SimpleNamespace(
            generateBoxParameters=lambda box_type, edge: ([edge],)
        ),
        Solvent=types.SimpleNamespace(
            solvate=lambda model, **kwargs: solvent.append((model, kwargs))
            or "solvated"
        ),
    )
    monkeypatch.setitem(sys.modules, "BioSimSpace", fake_bss)
    spec = BulkBuildSpec(
        complex_prm7="complex.prm7",
        complex_rst7="complex.rst7",
        molecule_indices=[1, 2],
        prep=PrepConfig(bulk_box_padding_angstrom=2),
        output_dir=str(tmp_path / "bulk"),
    )
    spec.dump(tmp_path / BULK_BUILD_SPEC_FILENAME)
    bulk_build.run_bulk_build(tmp_path)
    assert solvent[0][1]["molecule"].indices == [1, 2]
    assert solvent[0][1]["box"] == [7]
    assert saved[0][0] == str(tmp_path / "bulk" / "solvated")
    assert BulkBuildResult.load(tmp_path / "result.json").solvated_prm7.endswith(
        "bulk/solvated.prm7"
    )


def test_run_bulk_build_rejects_empty_component_list(tmp_path, monkeypatch):
    from gluebind.simulation import bulk_build

    monkeypatch.setitem(sys.modules, "BioSimSpace", types.SimpleNamespace())
    BulkBuildSpec(
        complex_prm7="complex.prm7",
        complex_rst7="complex.rst7",
        molecule_indices=[],
        prep=PrepConfig(),
        output_dir=str(tmp_path / "bulk"),
    ).dump(tmp_path / BULK_BUILD_SPEC_FILENAME)
    with pytest.raises(ValueError, match="at least one molecule"):
        bulk_build.run_bulk_build(tmp_path)


def test_run_prep_stage_writes_final_frame_and_trajectory(tmp_path, monkeypatch):
    """The worker can be tested as orchestration without an OpenMM run."""
    from gluebind.simulation import prep_stage

    class Process:
        def __init__(self, system, protocol, platform):
            self.system = system
            self.protocol = protocol
            self.platform = platform

        def start(self):
            pass

        def wait(self):
            pass

        def isError(self):
            return False

        def getSystem(self):
            return "final-system"

        def getTrajectory(self):
            return types.SimpleNamespace(
                getTrajectory=lambda format: types.SimpleNamespace(
                    save=lambda path: pathlib.Path(path).touch()
                )
            )

    saved = []
    fake_bss = types.SimpleNamespace(
        IO=types.SimpleNamespace(
            readMolecules=lambda files: "input-system",
            saveMolecules=lambda prefix, system, formats: saved.append(
                (prefix, system, formats)
            ),
        ),
        Process=types.SimpleNamespace(OpenMM=Process),
    )
    monkeypatch.setitem(sys.modules, "BioSimSpace", fake_bss)
    monkeypatch.setattr(prep_stage, "build_protocol", lambda **kwargs: "protocol")
    PrepStageSpec(
        stage="nvt",
        kind="equilibration",
        input_prm7="input.prm7",
        input_rst7="input.rst7",
    ).dump(tmp_path / PREP_STAGE_SPEC_FILENAME)

    prep_stage.run_prep_stage(tmp_path)

    assert saved == [(str(tmp_path / "output"), "final-system", ["prm7", "rst7"])]
    result = json.loads((tmp_path / PREP_STAGE_RESULT_FILENAME).read_text())
    assert result["trajectory"] == str(tmp_path / "output.dcd")


def test_run_prep_stage_includes_worker_logs_in_error(tmp_path, monkeypatch):
    from gluebind.simulation import prep_stage

    class FailedProcess:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def wait(self):
            pass

        def isError(self):
            return True

        def getStdout(self):
            return ["standard output"]

        def getStderr(self):
            return ["detailed error"]

    fake_bss = types.SimpleNamespace(
        IO=types.SimpleNamespace(readMolecules=lambda files: "input-system"),
        Process=types.SimpleNamespace(OpenMM=FailedProcess),
    )
    monkeypatch.setitem(sys.modules, "BioSimSpace", fake_bss)
    monkeypatch.setattr(prep_stage, "build_protocol", lambda **kwargs: "protocol")
    PrepStageSpec(
        stage="bad-stage",
        kind="minimisation",
        input_prm7="input.prm7",
        input_rst7="input.rst7",
    ).dump(tmp_path / PREP_STAGE_SPEC_FILENAME)

    with pytest.raises(RuntimeError, match="detailed error"):
        prep_stage.run_prep_stage(tmp_path)


def test_bulk_build_resume_reuses_manifest_without_backend_submission(
    tmp_path, monkeypatch
):
    import gluebind.system.prep as prep

    out_dir = tmp_path / "target_bulk"
    build_dir = out_dir / "build"
    prm7 = out_dir / "solvated.prm7"
    rst7 = out_dir / "solvated.rst7"
    prm7.parent.mkdir(parents=True)
    prm7.touch()
    rst7.touch()
    BulkBuildResult(solvated_prm7=str(prm7), solvated_rst7=str(rst7)).dump(
        build_dir / "result.json"
    )
    calls = []

    def fake_equil(prm, rst, plan, work, backend, **kwargs):
        calls.append((prm, rst, kwargs["handle_label_prefix"]))
        return "final.prm7", "final.rst7", None

    monkeypatch.setattr(prep, "run_equilibration_stages", fake_equil)

    class NoSubmitBackend(Backend):
        def submit(self, spec):
            raise AssertionError("bulk build should have been reused")

        def poll(self, handles):
            return {}

        def cancel(self, handle):
            pass

    result = prep._build_and_equilibrate_bulk(
        component="target",
        complex_prm7="complex.prm7",
        complex_rst7="complex.rst7",
        indices=[1],
        prep_config=PrepConfig(),
        out_dir=out_dir,
        backend=NoSubmitBackend(),
        platform="CUDA",
        poll_interval=1.0,
    )

    assert result == ("final.prm7", "final.rst7")
    assert calls == [(str(prm7), str(rst7), "target_bulk_")]


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
