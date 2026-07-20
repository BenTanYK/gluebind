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
TEMPERATURE = 300.0  # K — uniform production temperature (matches config.sampling)
STANDARD_VOLUME_NM3 = 1660.0 * 0.001  # 1660 Angstrom^3 expressed in nm^3
RADIUS_SPHERE_NM = (3.0 * STANDARD_VOLUME_NM3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _beta(temperature: float) -> float:
    return 1.0 / (BOLTZMANN * temperature)


def _finite_pmf(pmf: np.ndarray) -> np.ndarray:
    """Replace non-finite PMF bins with ``+inf`` so they contribute
    ``exp(-beta*inf) = 0`` to the integrals, without disturbing the CV grid.

    WHAM emits ``inf`` for unsampled bins (which already self-zero) and can emit
    ``nan`` on failures (which would otherwise poison the sum); this maps both,
    and ``-inf``, to a zero-weight bin.
    """
    return np.where(np.isfinite(pmf), pmf, np.inf)


def rmsd_contribution(
    x,
    pmf,
    force_constant: float,
    *,
    unbound: bool = False,
    temperature: float = TEMPERATURE,
) -> float:
    """Free-energy contribution of an RMSD restraint (centre 0).

    Returns ``+`` when ``unbound`` (a released/bulk restraint) and ``-`` for a
    bound (applied) restraint, so summing over stages yields
    ``dG_c^bulk - dG_c^bound``.
    """
    beta = _beta(temperature)
    x = np.asarray(x, dtype=float)
    pmf = _finite_pmf(np.asarray(pmf, dtype=float))
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
    pmf = _finite_pmf(np.asarray(pmf, dtype=float))
    numerator = float(np.sum(np.exp(-beta * pmf)))
    denominator = float(
        np.sum(np.exp(-beta * (pmf + 0.5 * force_constant * (x - theta_0) ** 2)))
    )
    return -math.log(numerator / denominator) / beta


def separation_contribution(
    x, pmf, r_star: float, *, temperature: float = TEMPERATURE
) -> float:
    """Integrate the separation PMF out to ``r_star`` (all lengths in nm)."""
    beta = _beta(temperature)
    x = np.asarray(x, dtype=float)
    pmf = _finite_pmf(np.asarray(pmf, dtype=float))
    if x.size < 2:
        raise ValueError("separation PMF needs at least two points")

    # W(r*): the PMF at the first point beyond r_star.
    w_r_star = pmf[0]
    for xi, yi in zip(x, pmf, strict=False):
        if xi >= r_star:
            w_r_star = yi
            break

    width = float(x[1] - x[0])
    integral = 0.0
    for xi, yi in zip(x, pmf, strict=False):
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
        / (
            8
            * (math.pi**2)
            * (4 * math.pi * RADIUS_SPHERE_NM**2)
            * (force_constant**2.5)
        )
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


def separation_plateau_reached(
    cv, pmf, *, window_nm: float = 0.4, tol: float = 0.1
) -> tuple[bool, float]:
    """Whether the separation PMF has flattened at large separation.

    Returns ``(reached, gradient)`` where ``gradient`` (kcal/mol/nm) is the
    least-squares slope of the PMF over the final ``window_nm`` of finite data;
    ``reached`` is ``|gradient| <= tol``. A non-flat tail means the unbound state
    was not reached — run windows to larger separation.
    """
    cv = np.asarray(cv, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    mask = np.isfinite(pmf)
    cv, pmf = cv[mask], pmf[mask]
    if cv.size < 2:
        return False, float("nan")
    tail = cv >= (cv[-1] - window_nm)
    if int(tail.sum()) >= 2:
        gradient = float(np.polyfit(cv[tail], pmf[tail], 1)[0])
    else:  # window too narrow for the grid — use the last two points
        gradient = float((pmf[-1] - pmf[-2]) / (cv[-1] - cv[-2]))
    return abs(gradient) <= tol, gradient


def _ends_decayed(values: np.ndarray, tol: float) -> bool:
    """Whether a normalised integrand has decayed to ``<= tol`` of its peak at both
    ends (finite entries only)."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return False
    peak = float(finite.max())
    if peak <= 0.0:
        return False
    return bool(finite[0] <= tol * peak and finite[-1] <= tol * peak)


def contribution_converged(
    cv,
    pmf,
    *,
    cv_type: str,
    force_constant: float,
    theta_0: float = 0.0,
    r_star: float | None = None,
    temperature: float = TEMPERATURE,
    tol: float = 0.01,
) -> bool:
    """Whether a stage's contribution integrand is bracketed by its windows.

    Applies the paper's convergence criterion (SI): the integrand(s) must decay to
    ``< tol`` (1%) of their maximum at both CV extremes, so ``> 98 %`` of the
    contribution is captured. For RMSD/Boresch this checks the numerator
    ``exp(-b W)`` and denominator ``exp(-b (W + ½k(η-η₀)²))``; for separation, the
    ``exp(-b (W(r)-W(r*)))`` integrand up to ``r_star``. Poor decay means windows
    should be added at the offending extreme.
    """
    beta = _beta(temperature)
    cv = np.asarray(cv, dtype=float)
    pmf = _finite_pmf(np.asarray(pmf, dtype=float))

    if cv_type == "separation":
        if r_star is None:
            r_star = float(cv[-1])
        w_star = float(pmf[0])
        for xi, yi in zip(cv, pmf, strict=False):
            if xi >= r_star:
                w_star = float(yi)
                break
        selection = cv <= r_star
        checks = [np.exp(-beta * (pmf[selection] - w_star))]
    else:
        numerator = np.exp(-beta * pmf)
        denominator = np.exp(-beta * (pmf + 0.5 * force_constant * (cv - theta_0) ** 2))
        checks = [numerator, denominator]

    return all(_ends_decayed(arr, tol) for arr in checks)


def binding_free_energy(
    dg_rmsd: float, dg_boresch: float, dg_sep: float, dg_corr: float
) -> float:
    """Assemble the standard-state binding free energy from its contributions."""
    return dg_rmsd + dg_boresch + dg_sep + dg_corr


def combine_errors(*errors: float) -> float:
    """Combine independent standard errors in quadrature."""
    return math.sqrt(sum(e**2 for e in errors))
