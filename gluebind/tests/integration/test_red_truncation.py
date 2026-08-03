"""Integration tier: RED equilibration detection on a synthetic timeseries.

Self-contained — needs only ``red`` (red-molsim), no BSS/GPU. A series that drifts
then plateaus must be truncated inside the drift so the retained portion sits on the
plateau — the check that gluebind's ``detect_equilibration`` wrapper drives RED and
maps its index back correctly.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_detect_equilibration_drops_the_drift(red_mod, tmp_path):
    from gluebind.analysis.pmf import detect_equilibration

    rng = np.random.default_rng(1)
    ramp = np.linspace(5.0, 0.0, 400)  # initial non-equilibrium drift
    plateau = 0.0 + rng.normal(0.0, 0.3, size=1600)  # equilibrated fluctuations
    series = np.concatenate([ramp, plateau])

    idx = detect_equilibration(series, subsample=1)

    assert isinstance(idx, int)
    assert 0 < idx < series.size  # detected a non-trivial equilibration point
    # the retained (post-truncation) portion should sit on the plateau (~0), i.e.
    # RED discarded the drift rather than keeping it
    assert abs(float(series[idx:].mean())) < 1.0
