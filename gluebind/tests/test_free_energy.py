"""Tests for the geometric-route free-energy contributions.

These check invariants and sign conventions (not validated absolute numbers —
scientific validation against the paper is a later development phase).
"""

import math

import numpy as np
import pytest

from gluebind.analysis import free_energy as fe


def test_flat_pmf_rmsd_sign_convention():
    x = np.linspace(0.0, 0.3, 61)  # nm
    pmf = np.zeros_like(x)
    bound = fe.rmsd_contribution(x, pmf, force_constant=3000.0, unbound=False)
    bulk = fe.rmsd_contribution(x, pmf, force_constant=3000.0, unbound=True)
    # applying a restraint has a positive free-energy cost; bound enters negative,
    # bulk positive, and they are exact negatives of each other.
    assert bulk > 0
    assert bound == pytest.approx(-bulk)


def test_rmsd_zero_force_constant_is_zero():
    x = np.linspace(0.0, 0.3, 61)
    pmf = np.zeros_like(x)
    assert fe.rmsd_contribution(
        x, pmf, force_constant=0.0, unbound=True
    ) == pytest.approx(0.0)


def test_boresch_contribution_is_negative_cost():
    x = np.linspace(0.5, 1.5, 101)  # rad
    pmf = np.zeros_like(x)
    dg = fe.boresch_contribution(x, pmf, theta_0=1.0, force_constant=100.0)
    assert dg < 0  # returns the negative cost of applying the restraint


def test_separation_contribution_finite():
    x = np.linspace(0.9, 3.0, 211)  # nm
    pmf = np.zeros_like(x)
    dg = fe.separation_contribution(x, pmf, r_star=2.5)
    assert math.isfinite(dg)


def test_separation_requires_two_points():
    with pytest.raises(ValueError):
        fe.separation_contribution([1.0], [0.0], r_star=1.0)


def _pmf_with_unsampled_margins():
    """WHAM-like output: sampled 0.9-3.0 nm, empty (inf) margin bins either side."""
    x = np.round(np.arange(0.80, 3.1001, 0.01), 4)
    pmf = np.where((x >= 0.9) & (x <= 3.0), 10.0 * (x - 1.2) ** 2, np.inf)
    return x, pmf


@pytest.mark.parametrize("r_star", [3.5, 3.05, 0.85])
def test_separation_contribution_rejects_r_star_outside_the_sampled_pmf(r_star):
    # Regression: r* beyond the grid silently took W(r*) from pmf[0] and
    # integrated everything (a plausible-looking but wrong ΔG); r* in an unsampled
    # margin bin gave W(r*) = inf and a NaN result.
    x, pmf = _pmf_with_unsampled_margins()
    with pytest.raises(ValueError, match="outside the sampled separation PMF"):
        fe.separation_contribution(x, pmf, r_star=r_star)


def test_separation_contribution_rejects_an_unsampled_w_r_star():
    x, pmf = _pmf_with_unsampled_margins()
    pmf[np.argmax(x >= 2.5)] = np.inf  # a gap inside the sampled range
    with pytest.raises(ValueError, match="was not sampled"):
        fe.separation_contribution(x, pmf, r_star=2.5)


def test_separation_contribution_unchanged_for_a_valid_r_star():
    # W(r*) is the first grid point at or beyond r*, and the integral includes it.
    x, pmf = _pmf_with_unsampled_margins()
    r_star = 2.99
    i = int(np.argmax(x >= r_star))
    beta = 1.0 / (fe.BOLTZMANN * fe.TEMPERATURE)
    width = float(x[1] - x[0])
    expected_integral = width * sum(
        math.exp(-beta * (p - pmf[i])) for p in pmf[: i + 1] if np.isfinite(p)
    )
    expected = -math.log(3.0 * expected_integral / fe.RADIUS_SPHERE_NM) / beta
    assert fe.separation_contribution(x, pmf, r_star) == pytest.approx(expected)


def test_separation_convergence_check_rejects_r_star_outside_the_sampled_pmf():
    x, pmf = _pmf_with_unsampled_margins()
    with pytest.raises(ValueError, match="outside the sampled separation PMF"):
        fe.contribution_converged(
            x, pmf, cv_type="separation", force_constant=0.0, r_star=3.5
        )
    # without r*, it uses the last sampled point rather than an empty margin bin
    fe.contribution_converged(x, pmf, cv_type="separation", force_constant=0.0)


def test_standard_state_correction_finite():
    dg = fe.standard_state_correction(
        r_star=3.0, theta_a_min=1.2, theta_b_min=1.4, force_constant=100.0
    )
    assert math.isfinite(dg)


def test_integrands_flat_pmf_numerator_unity():
    x = np.linspace(0.0, 0.3, 31)
    pmf = np.zeros_like(x)
    xs, num, den = fe.integrands(x, pmf, force_constant=3000.0)
    assert np.allclose(num, 1.0)
    assert np.all(den <= num + 1e-12)  # denominator has the extra restraint term
    assert xs.shape == num.shape == den.shape


def test_integrands_drops_nonfinite():
    x = np.array([0.0, 0.1, 0.2])
    pmf = np.array([0.0, np.inf, 0.0])
    xs, num, den = fe.integrands(x, pmf, force_constant=1.0)
    assert xs.size == 2


def test_rmsd_contribution_drops_nonfinite_bins():
    # A nan/inf bin (WHAM emits these) must contribute 0, not poison the sum; for
    # the sum-based integral this equals removing those bins entirely.
    x = np.linspace(0.0, 0.3, 61)
    pmf = np.linspace(0.0, 2.0, 61)
    bad = pmf.copy()
    bad[10] = np.nan
    bad[40] = np.inf
    keep = np.ones(61, dtype=bool)
    keep[[10, 40]] = False
    got = fe.rmsd_contribution(x, bad, 3000.0, unbound=True)
    ref = fe.rmsd_contribution(x[keep], pmf[keep], 3000.0, unbound=True)
    assert math.isfinite(got)
    assert got == pytest.approx(ref)


def test_boresch_contribution_drops_nonfinite_bins():
    x = np.linspace(0.5, 1.5, 101)
    pmf = np.linspace(0.0, 1.0, 101)
    bad = pmf.copy()
    bad[5] = np.nan
    bad[50] = np.inf
    keep = np.ones(101, dtype=bool)
    keep[[5, 50]] = False
    got = fe.boresch_contribution(x, bad, theta_0=1.0, force_constant=100.0)
    ref = fe.boresch_contribution(x[keep], pmf[keep], theta_0=1.0, force_constant=100.0)
    assert math.isfinite(got)
    assert got == pytest.approx(ref)


def test_separation_contribution_finite_with_nan():
    x = np.linspace(0.9, 3.0, 211)
    pmf = np.zeros_like(x)
    pmf[5] = np.nan  # a failed bin must not poison the integral
    assert math.isfinite(fe.separation_contribution(x, pmf, r_star=2.5))


def test_separation_plateau_reached():
    x = np.linspace(0.9, 3.0, 43)
    flat_tail = -5.0 * np.maximum(0.0, 2.6 - x)  # rises to 0 by 2.6 nm, then flat
    reached, grad = fe.separation_plateau_reached(x, flat_tail)
    assert reached and abs(grad) < 0.1

    sloped = -2.0 * x  # never flattens
    reached2, grad2 = fe.separation_plateau_reached(x, sloped)
    assert not reached2 and abs(grad2) > 0.5

    accepted = -0.4 * x
    reached3, grad3 = fe.separation_plateau_reached(x, accepted)
    assert reached3 and abs(grad3) == pytest.approx(0.4)

    rejected = -0.6 * x
    reached4, grad4 = fe.separation_plateau_reached(x, rejected)
    assert not reached4 and abs(grad4) == pytest.approx(0.6)


def test_contribution_converged_bracketed_well():
    # integrand exp(-bW) peaks mid-range and decays to <1% at both ends -> converged
    x = np.linspace(-1.0, 1.0, 101)
    pmf = 100.0 * x**2
    assert fe.contribution_converged(x, pmf, cv_type="rmsd", force_constant=0.0)


def test_contribution_unconverged_edge_peak():
    # integrand rises monotonically -> peaks at the high edge -> not bracketed
    x = np.linspace(0.0, 1.0, 101)
    pmf = -50.0 * x
    assert not fe.contribution_converged(x, pmf, cv_type="rmsd", force_constant=0.0)


def test_contribution_converged_separation():
    x = np.linspace(0.9, 3.0, 43)
    pmf = 20.0 * (x - 1.8) ** 2 - 20.0  # well with rising edges within the range
    assert fe.contribution_converged(
        x, pmf, cv_type="separation", force_constant=0.0, r_star=3.0
    )


def test_binding_free_energy_is_sum():
    assert fe.binding_free_energy(-8.7, -0.5, -19.4, 7.9) == pytest.approx(
        -8.7 - 0.5 - 19.4 + 7.9
    )


def test_combine_errors_quadrature():
    assert fe.combine_errors(3.0, 4.0) == pytest.approx(5.0)


def test_temperature_changes_result():
    x = np.linspace(0.0, 0.3, 61)
    pmf = np.zeros_like(x)
    a = fe.rmsd_contribution(x, pmf, 3000.0, unbound=True, temperature=298.15)
    b = fe.rmsd_contribution(x, pmf, 3000.0, unbound=True, temperature=310.0)
    assert a != b
