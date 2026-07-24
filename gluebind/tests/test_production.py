"""Tests for the OpenMM production-run spec + launch command (the MD run itself is
integration-verified)."""

import pytest

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


class _FakeIntegrator:
    def __init__(self):
        self.temperature = None

    def setTemperature(self, t):
        self.temperature = t


class _FakeState:
    def getPositions(self):
        return []

    def getPeriodicBoxVectors(self):
        return None


class _FakeContext:
    def __init__(self):
        self.velocity_temperature = None

    def setPeriodicBoxVectors(self, *v):
        pass

    def setPositions(self, p):
        pass

    def reinitialize(self, preserveState=False):
        pass

    def setVelocitiesToTemperature(self, t):
        self.velocity_temperature = t

    def getState(self, getPositions=False):
        return _FakeState()


class _FakeSimulation:
    def __init__(self):
        self.context = _FakeContext()
        self.reporters = []

    def step(self, n):
        pass


def test_run_production_sets_integrator_bath_to_target(tmp_path, monkeypatch):
    """Regression: production must set the Langevin thermostat bath to the sampling
    temperature, not leave it at build_simulation's cold INITIAL_TEMPERATURE_K —
    otherwise the whole trajectory silently cools to ~6 K."""
    unit = pytest.importorskip("openmm.unit")

    from gluebind.restraints import rmsd
    from gluebind.restraints import system_builder as sb
    from gluebind.simulation import production as prod

    integrator = _FakeIntegrator()
    simulation = _FakeSimulation()

    monkeypatch.setattr(sb, "build_system", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(sb, "load_coordinates", lambda *a, **k: ([], (1, 2, 3)))
    monkeypatch.setattr(
        sb, "build_simulation", lambda *a, **k: (simulation, integrator)
    )
    monkeypatch.setattr(sb, "save_rst7", lambda *a, **k: None)
    monkeypatch.setattr(rmsd, "add_rmsd_restraint", lambda *a, **k: None)
    monkeypatch.setattr(prod, "_platform", lambda name: None)
    monkeypatch.setattr("openmm.app.DCDReporter", lambda *a, **k: object())

    topology = tmp_path / "complex.prm7"
    topology.write_text("prm7")  # for the final shutil.copyfile
    spec = ProductionSpec(
        topology=str(topology),
        coordinates=str(tmp_path / "complex.rst7"),
        restraints=[
            {"name": "always_on_0", "atoms": [1, 2, 3], "force_constant": 100.0}
        ],
        runtime_ns=0.01,
        temperature_K=310.0,
        platform="CPU",
    )
    spec.dump(tmp_path / PRODUCTION_SPEC_FILENAME)

    prod.run_production(tmp_path)

    assert integrator.temperature is not None, "integrator bath temperature never set"
    assert integrator.temperature.value_in_unit(unit.kelvin) == pytest.approx(310.0)
    # velocities are seeded at the target too
    got = simulation.context.velocity_temperature.value_in_unit(unit.kelvin)
    assert got == pytest.approx(310.0)
