"""Tests for the pure Boresch window-centre binning (compute_stage_centres itself
reads a trajectory and is integration-verified)."""

import json

import numpy as np
import pytest

from gluebind import CalculationConfig
from gluebind.stage_centres import (
    boresch_centres_from_series,
    compute_stage_centres,
    write_boresch_distributions,
)


def _config(boresch_centres=None):
    return CalculationConfig.model_validate(
        {
            "inputs": {
                "target": {"prm7": "target.prm7", "rst7": "target.rst7"},
                "receptor": {"prm7": "receptor.prm7", "rst7": "receptor.rst7"},
            },
            "sampling": {
                "boresch": {
                    "force_constant": 100.0,
                    "sampling_time_ns": 5.0,
                    "centres": boresch_centres,
                }
            },
        }
    )


def test_boresch_centres_spans_range_on_regular_grid():
    series = np.array([0.83, 0.95, 1.12, 1.0])
    centres = boresch_centres_from_series(series, 0.1)
    assert centres[0] == pytest.approx(0.8)  # floor(0.83/0.1)*0.1
    assert centres[-1] >= 1.12  # brackets the max
    assert np.allclose(np.diff(centres), 0.1)  # regular spacing


def test_boresch_centres_single_value_still_brackets():
    centres = boresch_centres_from_series(np.array([1.05, 1.05]), 0.1)
    assert len(centres) >= 1
    assert centres[0] == pytest.approx(1.0)


def test_boresch_centres_respects_spacing():
    centres = boresch_centres_from_series(np.array([0.0, 0.5]), 0.25)
    assert np.allclose(np.diff(centres), 0.25)
    assert centres[-1] >= 0.5


def test_boresch_centres_periodically_unwrap_dihedral_branch_cut():
    # A dihedral clustered near both -pi and +pi is compact on the circle.
    series = np.array([-3.10, -3.08, -3.05, 3.05, 3.10, 3.12])
    centres = boresch_centres_from_series(series, 0.1, periodic=True)
    assert max(centres) - min(centres) < 0.4


def test_boresch_centres_linear_mode_rejects_branch_cut_straddle():
    series = np.array([-3.10, -3.05, 3.05, 3.10])
    with pytest.raises(ValueError, match="straddle"):
        boresch_centres_from_series(series, 0.1)


def test_boresch_centres_broad_contiguous_is_fine():
    # Broad but contiguous (no wrap): the largest gap is the wrap gap, so no raise.
    centres = boresch_centres_from_series(np.linspace(-2.0, 2.0, 20), 0.5)
    assert centres[0] <= -2.0 and centres[-1] >= 2.0



def test_write_boresch_distributions_preserves_raw_and_analysis_values(tmp_path):
    series = {
        "thetaA": np.array([0.1, 0.2]),
        "thetaB": np.array([1.0, 1.1]),
        "phiA": np.array([-3.1, 3.1]),
        "phiB": np.array([-0.4, -0.3]),
        "phiC": np.array([0.5, 0.6]),
    }
    report = write_boresch_distributions(
        series, tmp_path / "boresch_distributions", metadata={"config_hash": "abc"}
    )

    assert set(report) == {"thetaA", "thetaB", "phiA", "phiB", "phiC"}
    theta = np.loadtxt(report["thetaA"], comments="#")
    phi = np.loadtxt(report["phiA"], comments="#")
    assert theta.shape == (2, 3)
    assert np.allclose(theta[:, 1], theta[:, 2])
    assert np.ptp(phi[:, 2]) < 0.2  # periodic image is compact across the branch cut
    assert np.ptp(phi[:, 1]) > 6.0  # raw principal values retain the branch cut
    metadata = json.loads(
        (tmp_path / "boresch_distributions" / "metadata.json").read_text()
    )
    assert metadata["config_hash"] == "abc"
    assert metadata["frame_count"] == 2

def test_compute_stage_centres_uses_explicit_boresch_centres_without_trajectory():
    config = _config(
        {
            "thetaA": [0.91, 1.03],
            "thetaB": [0.81, 0.93],
            "phiA": [-0.2, -0.1],
            "phiB": [0.4, 0.5],
            "phiC": [0.7, 0.8],
        }
    )
    prepared = type("Prepared", (), {"complex_trajectory": None})()
    centres = compute_stage_centres(prepared, None, config)
    assert centres["thetaA"] == [0.91, 1.03]
    assert set(centres) == {
        "thetaA", "thetaB", "phiA", "phiB", "phiC", "separation"
    }


def test_compute_stage_centres_requires_trajectory_for_missing_boresch_centres():
    prepared = type("Prepared", (), {"complex_trajectory": None})()
    with pytest.raises(ValueError, match="need an equilibration trajectory"):
        compute_stage_centres(prepared, None, _config())
