"""Tests for the .gluebind-state.json run-state persistence."""

import json

import pytest

from gluebind.state import SCHEMA_VERSION, STATE_FILENAME, RunState, now_utc_iso


def _state(tmp_path) -> RunState:
    return RunState(
        calc_id="c1",
        submitted_at=now_utc_iso(),
        config_hash="abc123",
        config_path=str(tmp_path),
    )


def test_save_load_roundtrip(tmp_path):
    rs = _state(tmp_path)
    rs.boresch_eq_values["thetaA"] = 0.85
    rs.handles = {"thetaA": {"0.85rad": ["1", "2", "3"]}}
    rs.backend_extra = {"batch_id": "b1", "s3_prefix": "gluebind-batch/x"}
    rs.save(tmp_path)
    assert (tmp_path / STATE_FILENAME).exists()
    assert RunState.load(tmp_path) == rs


def test_save_is_atomic_no_tmp_left(tmp_path):
    _state(tmp_path).save(tmp_path)
    assert list(tmp_path.glob(".gluebind-state.*.tmp")) == []


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunState.load(tmp_path)


def test_newer_schema_rejected(tmp_path):
    _state(tmp_path).save(tmp_path)
    path = tmp_path / STATE_FILENAME
    data = json.loads(path.read_text())
    data["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        RunState.load(tmp_path)


def test_older_schema_migrates(tmp_path):
    _state(tmp_path).save(tmp_path)
    path = tmp_path / STATE_FILENAME
    data = json.loads(path.read_text())
    data["schema_version"] = 0  # pre-versioning file
    path.write_text(json.dumps(data))
    loaded = RunState.load(tmp_path)  # _migrate is a no-op today, but must not raise
    assert loaded.calc_id == "c1"


def test_backend_extra_defaults_empty(tmp_path):
    assert _state(tmp_path).backend_extra == {}
