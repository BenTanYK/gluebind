"""Geometric-route free-energy contributions from umbrella-sampling PMFs.

Ported from the ``US_protocol_template`` analysis — *not* a3fe's alchemical
MBAR/TI. gluebind is a geometric-route + umbrella-sampling method, so each stage
yields a PMF (via WHAM) which is turned into a free-energy contribution by the
integrals below. The standard-state binding free energy is their sum:

    dG_bind = dG_RMSD + dG_Boresch + dG_sep + dG_corr

with the sign convention baked into the functions so that a plain sum is correct:

* ``rmsd_contribution`` returns ``+`` for a bulk (released) stage and ``-`` for a
  bound (applied) stage, so summing the RMSD stages gives
  ``dG_c^bulk - dG_c^bound``;
* ``boresch_contribution`` returns the negative cost of a bound-state
  orientational restraint (``-dG_o^bound``);
* ``separation_contribution`` integrates the separation PMF;
* ``standard_state_correction`` is the analytical release term.

Unit convention (matching the template): spatial CVs (RMSD, separation) and
their force constants are in **nm** / kcal·mol⁻¹·nm⁻²; angular CVs (Boresch) in
**rad** / kcal·mol⁻¹·rad⁻². The force constant passed here must equal the one
written to the WHAM metafile — the calculation config is the single source of
truth for it, which structurally prevents the sim/analysis mismatch present in
the original template scripts.
"""

from __future__ import annotations

import math

import numpy as np

BOLTZMANN = 0.0019872041  # kcal/mol/K
TEMPERATURE = 298.15  # K
STANDARD_VOLUME_NM3 = 1660.0 * 0.001  # 1660 Angstrom^3 expressed in nm^3
RADIUS_SPHERE_NM = (3.0 * STANDARD_VOLUME_NM3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _beta(temperature: float) -> float:
    return 1.0 / (BOLTZMANN * temperature)


def rmsd_contribution(
    x, pmf, force_constant: float, *, unbound: bool = False, temperature: float = TEMPERATURE
) -> float:
    """Free-energy contribution of an RMSD restraint (centre 0).

    Returns ``+`` when ``unbound`` (a released/bulk restraint) and ``-`` for a
    bound (applied) restraint, so summing over stages yields
    ``dG_c^bulk - dG_c^bound``.
    """
    beta = _beta(temperature)
    x = np.asarray(x, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    numerator = float(np.sum(np.exp(-beta * pmf)))
    denominator = float(np.sum(np.exp(-beta * (pmf + 0.5 * force_constant * x**2))))
    contribution = math.log(numerator / denominator) / beta
    return contribution if unbound else -contribution


def boresch_contribution(
    x, pmf, theta_0: float, force_constant: float, *, temperature: float = TEMPERATURE
) -> float:
    """Negative free-energy cost of applying a Boresch restraint about ``theta_0``."""
    beta = _beta(temperature)
    x = np.asarray(x, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    numerator = float(np.sum(np.exp(-beta * pmf)))
    denominator = float(np.sum(np.exp(-beta * (pmf + 0.5 * force_constant * (x - theta_0) ** 2))))
    return -math.log(numerator / denominator) / beta


def separation_contribution(x, pmf, r_star: float, *, temperature: float = TEMPERATURE) -> float:
    """Integrate the separation PMF out to ``r_star`` (all lengths in nm)."""
    beta = _beta(temperature)
    x = np.asarray(x, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    if x.size < 2:
        raise ValueError("separation PMF needs at least two points")

    # W(r*): the PMF at the first point beyond r_star.
    w_r_star = pmf[0]
    for xi, yi in zip(x, pmf):
        if xi >= r_star:
            w_r_star = yi
            break

    width = float(x[1] - x[0])
    integral = 0.0
    for xi, yi in zip(x, pmf):
        integral += width * math.exp(-beta * (yi - w_r_star))
        if xi >= r_star:
            break
    return -1.0 / beta * math.log(3.0 * integral / RADIUS_SPHERE_NM)


def standard_state_correction(
    r_star: float,
    theta_a_min: float,
    theta_b_min: float,
    force_constant: float,
    *,
    temperature: float = TEMPERATURE,
) -> float:
    """Analytical standard-state correction for releasing the separated proteins.

    Assumes a common Boresch force constant across the angular DoFs (the template
    convention); ``r_star`` in nm, angles in rad.
    """
    beta = _beta(temperature)
    corr = (
        (r_star**2)
        * math.sin(theta_a_min)
        * math.sin(theta_b_min)
        * (2 * math.pi / beta) ** 2.5
        / (8 * (math.pi**2) * (4 * math.pi * RADIUS_SPHERE_NM**2) * (force_constant**2.5))
    )
    return -1.0 / beta * math.log(corr)


def integrands(x, pmf, force_constant: float, *, temperature: float = TEMPERATURE):
    """Numerator/denominator integrands used to check convergence of a contribution.

    Returns ``(x, exp(-b W), exp(-b (W + 0.5 k x^2)))`` with non-finite PMF points
    removed. Both should decay to a small fraction of their max at the CV extremes.
    """
    beta = _beta(temperature)
    x = np.asarray(x, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    mask = np.isfinite(pmf)
    x, pmf = x[mask], pmf[mask]
    numerator = np.exp(-beta * pmf)
    denominator = np.exp(-beta * (pmf + 0.5 * force_constant * x**2))
    return x, numerator, denominator


def binding_free_energy(
    dg_rmsd: float, dg_boresch: float, dg_sep: float, dg_corr: float
) -> float:
    """Assemble the standard-state binding free energy from its contributions."""
    return dg_rmsd + dg_boresch + dg_sep + dg_corr


def combine_errors(*errors: float) -> float:
    """Combine independent standard errors in quadrature."""
    return math.sqrt(sum(e**2 for e in errors))
