"""Anchor and equilibration selection (Phase 4).

Unlike the other layers, this has no template to port — the original workflow
picked anchors by eye from RMSF plots with a manual collinearity check. Here that
is automated:

* :mod:`gluebind.selection.rmsf` — Cα RMSF over the equilibration trajectory and
  the local-minimum candidate pool;
* :mod:`gluebind.selection.dssp` — MDAnalysis DSSP filter (keep α-helix/β-sheet);
* :mod:`gluebind.selection.geometry` — angle/dihedral/collinearity/circular
  variance;
* :mod:`gluebind.selection.anchors` — enumerate non-bonded anchor sets, reject
  collinear, pick the set with the smallest total Boresch-DoF variance (and
  validate manual overrides);
* :mod:`gluebind.selection.equilibration` — RED interface-RMSD equilibration check.

MDAnalysis / RED are imported lazily inside the functions that need them.
"""

from __future__ import annotations

from gluebind.selection.anchors import (
    is_valid_anchor_set,
    select_anchors,
    total_dof_variance,
    validate_manual_anchors,
)
from gluebind.selection.dssp import is_structured, structured_residues
from gluebind.selection.interface import interface_residues
from gluebind.selection.rmsf import local_minima, stablest_candidates

__all__ = [
    "select_anchors",
    "validate_manual_anchors",
    "is_valid_anchor_set",
    "total_dof_variance",
    "local_minima",
    "stablest_candidates",
    "is_structured",
    "structured_residues",
    "interface_residues",
]
