"""Automated Boresch anchor selection (new code — the template did this by eye).

Given candidate receptor/ligand Cα atoms (RMSF minima, secondary-structure
filtered) and their coordinates over the equilibration trajectory, enumerate the
non-bonded anchor sets (b, c from the receptor; B, C from the ligand), reject any
with a near-collinear triple, and pick the set whose five Boresch DoFs have the
smallest combined circular variance over the trajectory — i.e. the tightest,
most well-defined restraint geometry.

The bonded anchors ``a``/``A`` (interface centroids) are supplied by the caller;
only the four non-bonded anchors are selected here. A user may override the
selection entirely (:func:`validate_manual_anchors`), which still enforces the
collinearity constraint.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import permutations

import numpy as np

from gluebind.boresch_geometry import ANCHOR_CHAIN, ANGLE_DOFS, DOF_POINTS, DOFS
from gluebind.selection.geometry import angle, circular_variance, dihedral, is_collinear

# A "points over time" mapping: label -> (n_frames, 3) coordinate array.
PointsSeries = Mapping[str, np.ndarray]
CoordsOf = Callable[[int], np.ndarray]


def dof_timeseries(points: PointsSeries, dof: str) -> np.ndarray:
    """Value of one Boresch DoF at each frame, from the per-point coord series."""
    labels = DOF_POINTS[dof]
    series = [np.asarray(points[label], float) for label in labels]
    n_frames = series[0].shape[0]
    if dof in ANGLE_DOFS:
        return np.array([angle(series[0][t], series[1][t], series[2][t]) for t in range(n_frames)])
    return np.array(
        [dihedral(series[0][t], series[1][t], series[2][t], series[3][t]) for t in range(n_frames)]
    )


def total_dof_variance(points: PointsSeries) -> float:
    """Summed circular variance of the five Boresch DoFs over the trajectory."""
    return sum(circular_variance(dof_timeseries(points, dof)) for dof in DOFS)


def is_valid_anchor_set(mean_coords: Mapping[str, np.ndarray], *, tol_deg: float = 15.0) -> bool:
    """True if no consecutive triple along the c-b-a-A-B-C chain is collinear."""
    for i in range(len(ANCHOR_CHAIN) - 2):
        p1 = mean_coords[ANCHOR_CHAIN[i]]
        p2 = mean_coords[ANCHOR_CHAIN[i + 1]]
        p3 = mean_coords[ANCHOR_CHAIN[i + 2]]
        if is_collinear(p1, p2, p3, tol_deg=tol_deg):
            return False
    return True


def validate_manual_anchors(
    mean_coords: Mapping[str, np.ndarray], *, tol_deg: float = 15.0
) -> None:
    """Raise if user-supplied anchors have a near-collinear triple."""
    if not is_valid_anchor_set(mean_coords, tol_deg=tol_deg):
        raise ValueError(
            "manual Boresch anchors have a near-collinear triple along c-b-a-A-B-C; "
            "choose non-collinear anchor points"
        )


def select_anchors(
    *,
    receptor_candidates: Sequence[int],
    ligand_candidates: Sequence[int],
    a_coords: np.ndarray,
    A_coords: np.ndarray,
    coords_of: CoordsOf,
    collinearity_tol_deg: float = 15.0,
) -> dict[str, int]:
    """Choose the non-bonded anchors {b, c, B, C} minimising total DoF variance.

    ``a_coords``/``A_coords`` are the interface centroids over the trajectory
    (shape ``(n_frames, 3)``); ``coords_of(atom_index)`` returns an atom's
    coordinate series. Returns the chosen atom indices. Raises if no non-collinear
    set exists (the caller should then request manual anchors).
    """
    best: dict[str, int] | None = None
    best_score = math.inf

    for b, c in permutations(receptor_candidates, 2):
        for big_b, big_c in permutations(ligand_candidates, 2):
            points = {
                "a": a_coords,
                "A": A_coords,
                "b": coords_of(b),
                "c": coords_of(c),
                "B": coords_of(big_b),
                "C": coords_of(big_c),
            }
            mean_coords = {label: series.mean(axis=0) for label, series in points.items()}
            if not is_valid_anchor_set(mean_coords, tol_deg=collinearity_tol_deg):
                continue
            score = total_dof_variance(points)
            if score < best_score:
                best_score = score
                best = {"b": b, "c": c, "B": big_b, "C": big_c}

    if best is None:
        raise ValueError(
            "no non-collinear anchor set found among the candidates; "
            "supply anchors manually via BoreschSpec.anchors"
        )
    return best
