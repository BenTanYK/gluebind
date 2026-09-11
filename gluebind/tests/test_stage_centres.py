"""Tests for the pure Boresch window-centre binning (compute_stage_centres itself
reads a trajectory and is integration-verified)."""

import numpy as np
import pytest

from gluebind import CalculationConfig
from gluebind.stage_centres import boresch_centres_from_series, compute_stage_centres


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
