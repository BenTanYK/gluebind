"""WHAM wrapper — build the metafile, run Grossfield ``wham``, load the PMF.

Ports the template's WHAM plumbing, with the hardcoded binary path replaced by a
parameter (supplied from config) and no hidden unit conversions: the caller
writes window centres and force constants already in the WHAM's units (nm / rad),
so the config remains the single source of truth. The Grossfield ``wham`` binary
must be installed separately (https://github.com/agrossfield/wham).
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Sequence

import numpy as np

METAFILE_NAME = "metafile.txt"
PMF_NAME = "pmf.txt"


def write_metafile(
    entries: Sequence[tuple[str, float, float]], path: str | pathlib.Path
) -> pathlib.Path:
    """Write a WHAM metadata file.

    ``entries`` is a sequence of ``(timeseries_path, window_centre, force_constant)``
    with centre and force constant already in consistent WHAM units.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{ts} {centre} {k}\n" for ts, centre, k in entries]
    path.write_text("".join(lines))
    return path


def build_wham_command(
    wham_binary: str | pathlib.Path,
    wham_params: Sequence,
    metafile: str | pathlib.Path,
    pmf_out: str | pathlib.Path,
) -> list[str]:
    """Assemble the ``wham`` argv.

    ``wham_params`` is ``[hist_min, hist_max, num_bins, tol, temperature, numpad]``.
    """
    if len(wham_params) != 6:
        raise ValueError(
            "wham_params must be [hist_min, hist_max, num_bins, tol, temperature, numpad]"
        )
    return [
        str(wham_binary),
        *[str(p) for p in wham_params],
        str(metafile),
        str(pmf_out),
    ]


def run_wham(
    wham_binary: str | pathlib.Path,
    wham_params: Sequence,
    metafile: str | pathlib.Path,
    pmf_out: str | pathlib.Path,
    *,
    log: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Run ``wham`` to produce ``pmf_out`` from ``metafile``."""
    cmd = build_wham_command(wham_binary, wham_params, metafile, pmf_out)
    log_handle = open(log, "w") if log is not None else subprocess.DEVNULL
    try:
        subprocess.run(cmd, check=True, stdout=log_handle, stderr=subprocess.STDOUT)
    finally:
        if log is not None:
            log_handle.close()
    return pathlib.Path(pmf_out)


def load_pmf(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a WHAM ``pmf.txt``; return ``(cv_values, free_energy)``."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]
