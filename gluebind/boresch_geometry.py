"""The Boresch degree-of-freedom geometry — the single source of truth.

The five Boresch DoFs as ordered anchor-point sequences, where ``a``/``A`` are the
receptor/ligand interface centroids and ``b``,``c``/``B``,``C`` are the
receptor/ligand Cα anchors:

    thetaA = angle(b, a, A)        phiA = dihedral(c, b, a, A)
    thetaB = angle(a, A, B)        phiB = dihedral(b, a, A, B)
                                   phiC = dihedral(a, A, B, C)

This module has no heavy dependencies, so it is shared by both the OpenMM force
builders (:mod:`gluebind.restraints.boresch`) and the numpy-based anchor
selection (:mod:`gluebind.selection`) without either importing the other's deps.
"""

from __future__ import annotations

DOF_POINTS: dict[str, tuple[str, ...]] = {
    "thetaA": ("b", "a", "A"),
    "thetaB": ("a", "A", "B"),
    "phiA": ("c", "b", "a", "A"),
    "phiB": ("b", "a", "A", "B"),
    "phiC": ("a", "A", "B", "C"),
}
ANGLE_DOFS = ("thetaA", "thetaB")
DIHEDRAL_DOFS = ("phiA", "phiB", "phiC")
DOFS = ("thetaA", "thetaB", "phiA", "phiB", "phiC")

# The anchor chain c-b-a-A-B-C; the DoFs are its consecutive angle/dihedral
# windows, so a set is non-degenerate iff no consecutive triple is collinear.
ANCHOR_CHAIN = ("c", "b", "a", "A", "B", "C")
