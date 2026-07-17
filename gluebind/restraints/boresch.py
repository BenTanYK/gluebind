"""Boresch orientational restraints (the geometry defined in one place).

The six anchor points are:

* ``a`` — receptor interface centroid (a multi-atom group), ``A`` — ligand
  interface centroid (multi-atom group); these are the bonded pair also used as
  the separation CV endpoints.
* ``b``, ``c`` — receptor Cα anchors; ``B``, ``C`` — ligand Cα anchors.

The five restrained degrees of freedom, as ordered anchor sequences, are the
single source of truth (the template repeats these across three scripts):

    thetaA = angle(b, a, A)          phiA = dihedral(c, b, a, A)
    thetaB = angle(a, A, B)          phiB = dihedral(b, a, A, B)
                                     phiC = dihedral(a, A, B, C)

Force constants are in kcal·mol⁻¹·rad⁻²; equilibrium values / bias centres in
radians. Dihedral restraints use the periodic ``min(dθ, 2π−dθ)`` wrap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import openmm as mm
import openmm.unit as unit

from gluebind.boresch_geometry import ANGLE_DOFS, DIHEDRAL_DOFS, DOF_POINTS, DOFS

__all__ = [
    "DOF_POINTS",
    "ANGLE_DOFS",
    "DIHEDRAL_DOFS",
    "DOFS",
    "make_cv",
    "add_fixed_restraint",
    "add_bias",
    "points_from_groups",
]

_PI = "3.14159265358979"

Points = Mapping[str, Sequence[int]]


def _measure(dof: str) -> str:
    return "angle(g1,g2,g3)" if dof in ANGLE_DOFS else "dihedral(g1,g2,g3,g4)"


def _k(force_constant: float):
    return force_constant * unit.kilocalories_per_mole / unit.radians**2


def _add_groups(force, dof: str, points: Points) -> None:
    for label in DOF_POINTS[dof]:
        force.addGroup(list(points[label]))
    force.addBond(list(range(len(DOF_POINTS[dof]))))


def make_cv(dof: str, points: Points) -> mm.CustomCentroidBondForce:
    """A bare ``CustomCentroidBondForce`` measuring ``dof`` (for use as a CV)."""
    cv = mm.CustomCentroidBondForce(len(DOF_POINTS[dof]), _measure(dof))
    _add_groups(cv, dof, points)
    return cv


def add_fixed_restraint(
    system, dof: str, points: Points, eq_value: float, force_constant: float
) -> mm.CustomCentroidBondForce:
    """Add a harmonic restraint holding ``dof`` at ``eq_value`` (radians)."""
    eq_name = f"{dof}_0"
    measure = _measure(dof)
    if dof in ANGLE_DOFS:
        expr = f"0.5*k_boresch*({measure}-{eq_name})^2"
    else:
        expr = (
            f"0.5*k_boresch*min(dtheta, 2*pi-dtheta)^2; "
            f"dtheta=abs({measure}-{eq_name}); pi={_PI}"
        )
    force = mm.CustomCentroidBondForce(len(DOF_POINTS[dof]), expr)
    force.addGlobalParameter(eq_name, eq_value)
    force.addGlobalParameter("k_boresch", _k(force_constant))
    _add_groups(force, dof, points)
    system.addForce(force)
    return force


def add_bias(
    system, dof: str, points: Points, bias_centre: float, force_constant: float
) -> mm.CustomCVForce:
    """Add an umbrella bias on ``dof`` about ``bias_centre`` (radians).

    Returns the ``CustomCVForce``; ``getCollectiveVariableValues`` on it reads
    the current value of the DoF during sampling.
    """
    if dof in ANGLE_DOFS:
        expr = "0.5*k_boresch*(cv-bias_centre)^2"
    else:
        expr = f"0.5*k_boresch*min(dtheta, 2*pi-dtheta)^2; dtheta=abs(cv-bias_centre); pi={_PI}"
    bias = mm.CustomCVForce(expr)
    bias.addGlobalParameter("k_boresch", _k(force_constant))
    bias.addGlobalParameter("bias_centre", bias_centre)
    bias.addCollectiveVariable("cv", make_cv(dof, points))
    system.addForce(bias)
    return bias


def points_from_groups(
    rec_group: Sequence[int],
    lig_group: Sequence[int],
    anchors: Mapping[str, int],
) -> dict[str, list[int]]:
    """Build the anchor-point mapping from resolved atom groups/indices."""
    return {
        "a": list(rec_group),
        "A": list(lig_group),
        "b": [anchors["b"]],
        "c": [anchors["c"]],
        "B": [anchors["B"]],
        "C": [anchors["C"]],
    }
