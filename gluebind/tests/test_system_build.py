from gluebind.config.calculation import CalculationConfig
from gluebind.simulation.system_build import (
    SYSTEM_BUILD_RESULT_FILENAME,
    SYSTEM_BUILD_SPEC_FILENAME,
    SystemBuildSpec,
    run_system_build,
    system_build_launch_command,
)
from gluebind.system.prep import SolvatedSystem


def _config():
    return CalculationConfig.model_validate(
        {
            "inputs": {
                "receptor": {"prm7": "receptor.prm7", "rst7": "receptor.rst7"},
                "target": {"prm7": "target.prm7", "rst7": "target.rst7"},
                "glue": {"sdf": "glue.sdf", "assign_to": "receptor"},
            }
        }
    )


def test_system_build_spec_roundtrip(tmp_path):
    spec = SystemBuildSpec(config=_config(), output_dir=str(tmp_path / "prep"))
    path = spec.dump(tmp_path / SYSTEM_BUILD_SPEC_FILENAME)
    assert SystemBuildSpec.load(path) == spec


def test_system_build_launch_command():
    cmd = system_build_launch_command()
    assert cmd[:2] == ["python", "-c"]
    assert "run_system_build" in cmd[2]


def test_run_system_build_writes_result(tmp_path, monkeypatch):
    expected = SolvatedSystem(
        solvated_prm7="/run/prep/solvated.prm7",
        solvated_rst7="/run/prep/solvated.rst7",
        glue_assign_to="receptor",
        target_molecules=[2],
        receptor_molecules=[1],
        glue_molecule=0,
    )
    SystemBuildSpec(config=_config(), output_dir="/run/prep").dump(
        tmp_path / SYSTEM_BUILD_SPEC_FILENAME
    )

    monkeypatch.setattr(
        "gluebind.system.prep.build_solvated_system",
        lambda config, output_dir: expected,
    )
    run_system_build(tmp_path)
    assert SolvatedSystem.load(tmp_path / SYSTEM_BUILD_RESULT_FILENAME) == expected
