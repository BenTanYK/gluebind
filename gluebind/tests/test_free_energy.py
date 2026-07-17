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
