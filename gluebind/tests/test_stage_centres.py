"""Tests for the pure Boresch window-centre binning (compute_stage_centres itself
reads a trajectory and is integration-verified)."""

import numpy as np
import pytest

from gluebind.stage_centres import boresch_centres_from_series


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
