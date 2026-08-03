"""RMSD restraints (harmonic on an ``RMSDForce`` collective variable).

A single builder covers all the template's cases: a *fixed* restraint about zero
(``0.5 k rmsd²``, for holding a partner rigid) and a *moving/umbrella* restraint
about a centre (``0.5 k (rmsd − centre)²``, the sampled RMSD CV). Force constants
are in kcal·mol⁻¹·Å⁻²; the centre is in Å; the returned force's
``getCollectiveVariableValues`` reports the RMSD in nm.
"""

from __future__ import annotations

from collections.abc import Sequence

import openmm as mm
import openmm.unit as unit


def add_rmsd_restraint(
    system,
    atoms: Sequence[int],
    reference_positions,
    force_constant: float,
    *,
    name: str,
    centre: float | None = None,
) -> mm.CustomCVForce:
    """Add a harmonic RMSD restraint on ``atoms`` relative to ``reference_positions``.

    ``name`` disambiguates this restraint's global parameters from others in the
    same system (e.g. ``"rec"``, ``"lig"``, ``"BD1"``). ``centre=None`` gives a
    fixed restraint about zero; a value gives a moving/umbrella restraint about
    that RMSD (in Å).
    """
    k_name = f"k_{name}"
    if centre is None:
        expr = f"0.5*{k_name}*rmsd^2"
    else:
        centre_name = f"{name}_centre"
        expr = f"0.5*{k_name}*(rmsd-{centre_name})^2"
    force = mm.CustomCVForce(expr)
    force.addGlobalParameter(
        k_name, force_constant * unit.kilocalories_per_mole / unit.angstrom**2
    )
    if centre is not None:
        force.addGlobalParameter(centre_name, centre * unit.angstrom)
    force.addCollectiveVariable("rmsd", mm.RMSDForce(reference_positions, list(atoms)))
    system.addForce(force)
    return force
