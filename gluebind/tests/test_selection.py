"""Tests for the Phase 4 selection logic (pure parts; MDAnalysis/RED wrappers
are integration-verified)."""

import math

import numpy as np
import pytest

from gluebind.selection import (
    interface_residues,
    is_structured,
    is_valid_anchor_set,
    local_minima,
    select_anchors,
    total_dof_variance,
    validate_manual_anchors,
)
from gluebind.selection.geometry import angle, circular_variance, dihedral, is_collinear


# ---- geometry --------------------------------------------------------------


def test_angle_right_angle():
    assert angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(math.pi / 2)


def test_dihedral_range_and_value():
    # points forming a +90 degree dihedral
    val = dihedral((1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1))
    assert -math.pi <= val <= math.pi
    assert abs(val) == pytest.approx(math.pi / 2, abs=1e-6)


def test_is_collinear():
    assert is_collinear((0, 0, 0), (1, 0, 0), (2, 0, 0))  # straight line -> 180 deg
    assert not is_collinear((1, 0, 0), (0, 0, 0), (0, 1, 0))  # right angle


def test_circular_variance():
    assert circular_variance([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert circular_variance([0.0, math.pi]) > 0.5


# ---- rmsf candidate detection ----------------------------------------------


def test_local_minima():
    # minima at indices 2 (0.1) and 5 (0.2)
    assert local_minima([1.0, 0.5, 0.1, 0.4, 0.6, 0.2, 0.7]) == [2, 5]


def test_is_structured():
    assert is_structured("H") and is_structured("E")
    assert not is_structured("-") and not is_structured("T")


# ---- interface detection ---------------------------------------------------


def test_interface_residues_within_cutoff():
    rec = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])  # res 0 near, res 1 far
    lig = np.array([[5.0, 0.0, 0.0], [50.0, 0.0, 0.0]])  # res 0 near rec[0]
    rec_idx, lig_idx = interface_residues(rec, lig, cutoff=12.0)
    assert rec_idx == [0]
    assert lig_idx == [0]


def test_interface_residues_none_within_cutoff():
    rec = np.array([[0.0, 0.0, 0.0]])
    lig = np.array([[100.0, 0.0, 0.0]])
    assert interface_residues(rec, lig, cutoff=12.0) == ([], [])


# ---- anchor set validity ---------------------------------------------------


def _mean_coords(spread=1.0):
    # a non-degenerate arrangement of the six points
    return {
        "a": np.array([0.0, 0.0, 0.0]),
        "A": np.array([spread, 0.0, 0.0]),
        "b": np.array([0.0, spread, 0.0]),
        "c": np.array([0.0, 0.0, spread]),
        "B": np.array([spread, spread, 0.0]),
        "C": np.array([spread, 0.0, spread]),
    }


def test_valid_anchor_set_true():
    assert is_valid_anchor_set(_mean_coords())


def test_valid_anchor_set_collinear_false():
    # all points on the x-axis -> every triple collinear
    coords = {label: np.array([float(i), 0.0, 0.0]) for i, label in enumerate("abcABC")}
    # remap to the expected keys
    coords = {k: np.array([float(i), 0.0, 0.0]) for i, k in enumerate(["c", "b", "a", "A", "B", "C"])}
    assert not is_valid_anchor_set(coords)


def test_validate_manual_anchors_raises_on_collinear():
    coords = {k: np.array([float(i), 0.0, 0.0]) for i, k in enumerate(["c", "b", "a", "A", "B", "C"])}
    with pytest.raises(ValueError, match="collinear"):
        validate_manual_anchors(coords)


# ---- dof variance + selection ----------------------------------------------


def _const_series(point, n_frames=5):
    return np.tile(np.asarray(point, float), (n_frames, 1))


def test_total_dof_variance_zero_for_static_geometry():
    points = {label: _const_series(coord) for label, coord in _mean_coords().items()}
    assert total_dof_variance(points) == pytest.approx(0.0, abs=1e-9)


def test_select_anchors_picks_lowest_variance_set():
    # receptor candidates 10 (static) and 11 (jittering); ligand 20, 21 static.
    rng = np.random.default_rng(0)
    static = {
        10: _const_series([0.0, 1.0, 0.0]),
        20: _const_series([1.0, 1.0, 0.0]),
        21: _const_series([1.0, 0.0, 1.0]),
        12: _const_series([0.0, 0.0, 1.0]),
    }
    jitter = _const_series([0.0, 1.0, 0.0]) + rng.normal(scale=0.5, size=(5, 3))
    coords = {**static, 11: jitter}

    result = select_anchors(
        receptor_candidates=[10, 11, 12],
        ligand_candidates=[20, 21],
        a_coords=_const_series([0.0, 0.0, 0.0]),
        A_coords=_const_series([1.0, 0.0, 0.0]),
        coords_of=lambda i: coords[i],
        collinearity_tol_deg=1.0,
    )
    assert set(result) == {"b", "c", "B", "C"}
    assert result["b"] != result["c"]
    assert 11 not in (result["b"], result["c"])  # the jittering atom is avoided


def test_select_anchors_raises_when_all_collinear():
    line = {i: _const_series([float(i), 0.0, 0.0]) for i in (10, 11, 20, 21)}
    with pytest.raises(ValueError, match="no non-collinear"):
        select_anchors(
            receptor_candidates=[10, 11],
            ligand_candidates=[20, 21],
            a_coords=_const_series([5.0, 0.0, 0.0]),
            A_coords=_const_series([6.0, 0.0, 0.0]),
            coords_of=lambda i: line[i],
        )
