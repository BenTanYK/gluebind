"""Interface centre-of-mass separation CV and its umbrella bias.

The CV is the distance between the receptor and ligand interface groups (the
bonded Boresch anchors ``a`` and ``A``). The bias force constant is in
kcal·mol⁻¹·Å⁻²; the window centre ``r0`` is in nm (matching the template and the
WHAM PMF units).
"""

from __future__ import annotations

from collections.abc import Sequence

import openmm as mm
import openmm.unit as unit


def make_cv(
    rec_group: Sequence[int], lig_group: Sequence[int]
) -> mm.CustomCentroidBondForce:
    """Bare distance CV between the two interface groups' centroids."""
    cv = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
    cv.addGroup(list(rec_group))
    cv.addGroup(list(lig_group))
    cv.addBond([0, 1])
    return cv


def add_bias(
    system,
    rec_group: Sequence[int],
    lig_group: Sequence[int],
    r0_nm: float,
    force_constant: float,
) -> mm.CustomCVForce:
    """Add an umbrella bias on the separation CV about ``r0_nm`` (nm).

    Returns the ``CustomCVForce``; ``getCollectiveVariableValues`` reads the
    current separation (nm) during sampling.
    """
    bias = mm.CustomCVForce("0.5*k_r*(cv-r0)^2")
    bias.addGlobalParameter(
        "k_r", force_constant * unit.kilocalories_per_mole / unit.angstrom**2
    )
    bias.addGlobalParameter("r0", r0_nm * unit.nanometers)
    bias.addCollectiveVariable("cv", make_cv(rec_group, lig_group))
    system.addForce(bias)
    return bias
