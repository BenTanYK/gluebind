"""Tests for the WindowSpec model and the run_window entry-point contract."""

import pytest

from gluebind.simulation import WindowSpec, run_window, window_launch_command
from gluebind.simulation.window import WINDOW_SPEC_FILENAME


def _spec() -> WindowSpec:
    return WindowSpec(
        cv_type="rmsd",
        stage_name="receptor_bound",
        cv_centre=0.4,
        replicate=1,
        topology="system.prm7",
        coordinates="system.rst7",
        force_constant=5.0,
        sampling_time_ns=20.0,
    )


def test_windowspec_roundtrip(tmp_path):
    spec = _spec()
    path = spec.dump(tmp_path / WINDOW_SPEC_FILENAME)
    assert WindowSpec.load(path) == spec


def test_launch_command_shape():
    cmd = window_launch_command("python3.11")
    assert cmd[0] == "python3.11"
    assert cmd[1] == "-c"
    assert "run_window('.')" in cmd[2]


def test_run_window_missing_spec_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_window(tmp_path)


def test_run_window_bad_topology_raises(tmp_path):
    # A valid spec pointing at a nonexistent topology should fail while building
    # the system (i.e. run_window is implemented, not a NotImplementedError stub).
    _spec().dump(tmp_path / WINDOW_SPEC_FILENAME)
    with pytest.raises(Exception) as excinfo:
        run_window(tmp_path)
    assert not isinstance(excinfo.value, NotImplementedError)


def test_windowspec_extra_field_forbidden():
    data = {**_spec().model_dump(), "bogus": 1}
    with pytest.raises(ValueError):
        WindowSpec.model_validate(data)
