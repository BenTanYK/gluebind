"""PMF post-processing: replicate averaging and RED equilibration detection.

Ports the template's ``obtain_av_PMF`` (mean + standard error across replicate
PMFs) and its RED-based truncation of CV timeseries. ``red`` is imported lazily,
so this module is importable without it; it is only needed when detecting
equilibration.
"""

from __future__ import annotations

import math

import numpy as np


def average_pmfs(
    pmfs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average replicate PMFs sharing a common CV grid.

    ``pmfs`` is a list of ``(cv, free_energy)`` pairs. Returns
    ``(cv, mean, standard_error_of_mean)``.
    """
    if not pmfs:
        raise ValueError("no PMFs to average")
    cv = np.asarray(pmfs[0][0], dtype=float)
    stack = np.vstack([np.asarray(w, dtype=float) for _, w in pmfs])
    if stack.shape[1] != cv.size:
        raise ValueError("replicate PMFs have inconsistent lengths")
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    # ddof=1 (sample standard deviation) matches the reference us_analysis.py; for
    # a single replicate the SEM is undefined, so report 0.
    sem = stack.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
    return cv, mean, sem


def pmf_minimum(cv, pmf) -> float:
    """Return the CV value at the PMF minimum.

    Used to set a Boresch DoF's equilibrium value from its umbrella-sampling PMF
    (the value fed forward to the next sequential Boresch stage).
    """
    cv = np.asarray(cv, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    # Map non-finite bins to +inf so a nan can't hijack argmin (inf never wins).
    pmf = np.where(np.isfinite(pmf), pmf, np.inf)
    return float(cv[int(np.argmin(pmf))])


def detect_equilibration(
    timeseries, *, subsample: int = 2, max_samples: int = 20000, method: str = "min_sse"
) -> int:
    """Return the index at which ``timeseries`` becomes equilibrated, via RED.

    Wraps ``red.detect_equilibration_window`` (fjclark/red), subsampling to keep
    the input manageable as the template does. Raises ``ImportError`` if ``red``
    is not installed, and propagates RED's own failure (no equilibration found)
    to the caller, which should apply a fallback truncation.
    """
    import red  # lazy: red is an optional dependency

    data = np.asarray(timeseries, dtype=float)
    idx_start, _g, _ess = red.detect_equilibration_window(
        data[:max_samples:subsample], method=method
    )
    return int(idx_start) * subsample
