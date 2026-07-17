"""Interface-RMSD equilibration detection (RED).

Wraps ``red.detect_equilibration_window`` (fjclark/red) on the interface RMSD
timeseries. If equilibration is not detected over the whole trajectory it warns
and returns ``None`` (the caller applies a fallback), rather than silently
truncating. ``red`` is imported lazily (optional dependency).
"""

from __future__ import annotations

import warnings

import numpy as np


def detect_interface_equilibration(
    interface_rmsd, *, method: str = "min_sse", warn: bool = True
) -> int | None:
    """Return the equilibration start index for the interface RMSD, or ``None``.

    ``None`` means RED found no equilibrated window over the trajectory — a signal
    to extend sampling or apply a default truncation, and a warning is emitted.
    """
    import red

    data = np.asarray(interface_rmsd, float)
    try:
        idx_start, _g, _ess = red.detect_equilibration_window(data, method=method)
    except Exception:  # noqa: BLE001 - RED raises when no equilibration is found
        if warn:
            warnings.warn(
                "interface RMSD equilibration not detected over the trajectory; "
                "consider extending the equilibration run",
                stacklevel=2,
            )
        return None
    return int(idx_start)
