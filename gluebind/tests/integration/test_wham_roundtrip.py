"""Integration tier: a WHAM round-trip on synthetic umbrella samples.

Self-contained — needs only the Grossfield ``wham`` binary (no BSS/GPU). Generates
biased samples for a *flat* underlying free-energy surface, so WHAM must recover a
~flat PMF. This is the end-to-end check that gluebind's metafile format, ``wham``
invocation, and PMF parsing are wired together correctly.

Convention: gluebind passes force constants straight through in WHAM units, and the
Grossfield bias is ``½·k·(x−x₀)²`` — so the biased ensemble of a flat PMF is Gaussian
about the window centre with variance ``kT/k``. Samples are drawn to match.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

KB_KCAL = 0.0019872041  # Boltzmann constant, kcal/mol/K


def test_wham_recovers_flat_pmf(wham_binary, tmp_path):
    from gluebind.analysis.wham import load_pmf, run_wham, write_metafile

    rng = np.random.default_rng(0)
    temperature = 300.0
    kT = KB_KCAL * temperature
    k = 10.0  # kcal/mol/nm^2 (WHAM units)
    std = np.sqrt(kT / k)  # biased-ensemble width for a flat PMF
    centres = np.round(np.arange(0.0, 2.0001, 0.2), 4)

    entries = []
    for c in centres:
        samples = rng.normal(c, std, size=2000)
        ts_path = tmp_path / f"win_{c:.4g}.dat"
        # two columns [time_index, cv_value], as collect_cv_samples writes
        np.savetxt(ts_path, np.column_stack([np.arange(samples.size), samples]))
        entries.append((str(ts_path), float(c), k))

    metafile = write_metafile(entries, tmp_path / "metafile.txt")
    pmf_out = tmp_path / "pmf.txt"
    run_wham(
        wham_binary,
        [0.0, 2.0, 50, 1e-6, temperature, 0],  # min max bins tol temp numpad
        metafile,
        pmf_out,
        log=tmp_path / "wham.log",
    )

    cv, fe = load_pmf(pmf_out)
    assert cv.size == 50
    assert np.isfinite(fe).sum() > 30  # most bins populated
    # the well-sampled interior should be ~flat for a flat underlying PMF
    interior = fe[10:40]
    interior = interior[np.isfinite(interior)]
    assert np.ptp(interior) < 1.0  # kcal/mol: WHAM + finite-sampling noise only
