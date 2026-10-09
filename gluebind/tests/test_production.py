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
    assert spec.state_data_interval_steps == 10000


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
        self.get_state_kwargs = []

    def setPeriodicBoxVectors(self, *v):
        pass

    def setPositions(self, p):
        pass

    def reinitialize(self, preserveState=False):
        pass

    def setVelocitiesToTemperature(self, t):
        self.velocity_temperature = t

    def getState(self, **kwargs):
        self.get_state_kwargs.append(kwargs)
        return _FakeState()


class _FakeSimulation:
    def __init__(self):
        self.context = _FakeContext()
        self.reporters = []

    def step(self, n):
        pass


def _run_with_fakes(tmp_path, monkeypatch, *, temperature_K=300.0):
    """Run ``run_production`` with OpenMM mocked out; return the recorded calls."""
    pytest.importorskip("openmm")

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
    dcd_reporters = []
    monkeypatch.setattr(
        "openmm.app.DCDReporter",
        lambda *a, **k: dcd_reporters.append((a, k)) or object(),
    )
    reporters = []
    monkeypatch.setattr(
        "openmm.app.StateDataReporter",
        lambda *a, **k: reporters.append((a, k)) or object(),
    )

    topology = tmp_path / "complex.prm7"
    topology.write_text("prm7")  # for the final shutil.copyfile
    spec = ProductionSpec(
        topology=str(topology),
        coordinates=str(tmp_path / "complex.rst7"),
        restraints=[
            {"name": "always_on_0", "atoms": [1, 2, 3], "force_constant": 100.0}
        ],
        runtime_ns=0.01,
        temperature_K=temperature_K,
        platform="CPU",
    )
    spec.dump(tmp_path / PRODUCTION_SPEC_FILENAME)

    prod.run_production(tmp_path)
    return integrator, simulation, reporters, dcd_reporters


def test_run_production_periodic_output(tmp_path, monkeypatch):
    """Regression: the trajectory is unwrapped and the final rst7 frame wrapped.

    A per-molecule-wrapped trajectory wrote the two proteins a lattice vector
    apart once a cell face fell between them, producing spurious jumps in every
    Boresch DoF. An unwrapped final frame, however, let waters drift >1000 A out
    of the box over 100 ns, overflowing rst7's fixed-width fields; wrapping it is
    safe because build_system joins the solute into one molecule.
    """
    _, simulation, _, dcd_reporters = _run_with_fakes(tmp_path, monkeypatch)

    assert len(dcd_reporters) == 1
    assert dcd_reporters[0][1].get("enforcePeriodicBox") is False
    final_frame = [
        k for k in simulation.context.get_state_kwargs if k.get("getPositions")
    ]
    assert final_frame, "final frame never requested"
    # The final rst7 frame is wrapped (the joined solute stays whole); unwrapped
    # waters overflow rst7's fixed-width fields after a long run.
    assert all(k.get("enforcePeriodicBox") is True for k in final_frame)


def test_run_production_sets_integrator_bath_to_target(tmp_path, monkeypatch):
    """Regression: production must set the Langevin thermostat bath to the sampling
    temperature, not leave it at build_simulation's cold INITIAL_TEMPERATURE_K —
    otherwise the whole trajectory silently cools to ~6 K."""
    unit = pytest.importorskip("openmm.unit")

    integrator, simulation, reporters, _ = _run_with_fakes(
        tmp_path, monkeypatch, temperature_K=310.0
    )

    assert integrator.temperature is not None, "integrator bath temperature never set"
    assert integrator.temperature.value_in_unit(unit.kelvin) == pytest.approx(310.0)
    # velocities are seeded at the target too
    got = simulation.context.velocity_temperature.value_in_unit(unit.kelvin)
    assert got == pytest.approx(310.0)
    assert len(reporters) == 1
    reporter_args, reporter_kwargs = reporters[0]
    assert reporter_args[1] == 10000
    assert reporter_kwargs["potentialEnergy"] is True
    assert reporter_kwargs["density"] is True
    assert reporters[0][1]["totalSteps"] == 2500
