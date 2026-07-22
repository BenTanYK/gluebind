"""Tests for the OpenMM production-run spec + launch command (the MD run itself is
integration-verified)."""

from gluebind.simulation.production import (
    PRODUCTION_SPEC_FILENAME,
    ProductionSpec,
    production_launch_command,
)


def _spec():
    return ProductionSpec(
        topology="complex.prm7",
        coordinates="complex.rst7",
        restraints=[
            {"name": "always_on_0", "atoms": [1, 2, 3], "force_constant": 100.0}
        ],
        runtime_ns=50.0,
        timestep_fs=4.0,
        temperature_K=300.0,
        platform="CUDA",
    )


def test_production_spec_roundtrip(tmp_path):
    spec = _spec()
    path = spec.dump(tmp_path / PRODUCTION_SPEC_FILENAME)
    assert ProductionSpec.load(path) == spec


def test_production_spec_defaults():
    spec = ProductionSpec(topology="t.prm7", coordinates="t.rst7", runtime_ns=10.0)
    assert spec.restraints == []  # no constant restraints by default
    assert spec.sample_interval_steps == 2500  # coarse trajectory interval


def test_production_launch_command():
    cmd = production_launch_command()
    assert cmd[:2] == ["python", "-c"]
    assert "run_production" in cmd[2]
