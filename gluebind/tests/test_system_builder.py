"""Unit tests for the per-window heating helper."""

from unittest.mock import Mock

from gluebind.restraints import system_builder as sb


def test_minimise_and_heat_minimises_before_exact_heating_steps():
    events = []

    simulation = Mock()
    simulation.minimizeEnergy.side_effect = lambda: events.append("minimise")
    simulation.context.setVelocitiesToTemperature.side_effect = (
        lambda *_: events.append("velocities")
    )
    simulation.step.side_effect = lambda steps: events.append(("step", steps))

    integrator = Mock()
    integrator.setTemperature.side_effect = lambda *_: events.append("temperature")

    sb.minimise_and_heat(
        simulation,
        integrator,
        target_temperature_K=300.0,
        heating_steps=103,
    )

    assert events[0] == "minimise"
    assert events[1] == "velocities"
    step_events = [event for event in events if isinstance(event, tuple)]
    assert len(step_events) == sb.HEATING_INCREMENTS
    assert sum(event[1] for event in step_events) == 103
    assert step_events[-1] == ("step", 5)
