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
    assert fe.rmsd_contribution(x, pmf, force_constant=0.0, unbound=True) == pytest.approx(0.0)


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
    assert not reached2 and abs(grad2) > 0.1


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
    assert fe.contribution_converged(x, pmf, cv_type="separation", force_constant=0.0, r_star=3.0)


def test_binding_free_energy_is_sum():
    assert fe.binding_free_energy(-8.7, -0.5, -19.4, 7.9) == pytest.approx(-8.7 - 0.5 - 19.4 + 7.9)


def test_combine_errors_quadrature():
    assert fe.combine_errors(3.0, 4.0) == pytest.approx(5.0)


def test_temperature_changes_result():
    x = np.linspace(0.0, 0.3, 61)
    pmf = np.zeros_like(x)
    a = fe.rmsd_contribution(x, pmf, 3000.0, unbound=True, temperature=298.15)
    b = fe.rmsd_contribution(x, pmf, 3000.0, unbound=True, temperature=310.0)
    assert a != b
