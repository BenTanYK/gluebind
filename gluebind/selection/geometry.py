"""Pure numpy geometry for anchor selection.

Angle/dihedral measurement, collinearity testing, and circular variance — the
building blocks for scoring candidate Boresch anchor sets. (The OpenMM force
builders measure the same DoFs at simulation time; these are the analysis-side
equivalents used to *choose* the anchors from the equilibration trajectory.)
"""

from __future__ import annotations

import numpy as np


def angle(a, b, c) -> float:
    """Angle at vertex ``b`` (radians)."""
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def dihedral(a, b, c, d) -> float:
    """Dihedral angle a-b-c-d (radians, in [-pi, pi])."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    b0, b1, b2 = a - b, c - b, d - c
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def is_collinear(p1, p2, p3, *, tol_deg: float = 15.0) -> bool:
    """True if the angle at ``p2`` is within ``tol_deg`` of 0 or 180 degrees."""
    deg = np.degrees(angle(p1, p2, p3))
    return deg <= tol_deg or deg >= 180.0 - tol_deg


def circular_variance(angles) -> float:
    """Circular variance of a set of angles (0 = identical, →1 = dispersed)."""
    angles = np.asarray(angles, float)
    return float(1.0 - np.abs(np.mean(np.exp(1j * angles))))


def centroid(coords) -> np.ndarray:
    """Mean position of a set of coordinates (shape (..., 3))."""
    return np.asarray(coords, float).mean(axis=0)
