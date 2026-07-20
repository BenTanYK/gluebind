"""The real WHAM PMF provider — ``(cv, pmf)`` for a stage.

Given a stage (its windows, each with per-replicate CV timeseries on disk), this
writes a WHAM metadata file per replicate, runs the Grossfield ``wham`` binary
(locally on the driver, or as a submitted SLURM job — resolved question 1), loads
the per-replicate PMFs, and averages them. The returned callable is exactly the
``pmf_provider`` the runner uses for the Boresch equilibrium-value feedback and
that :meth:`Calculation.analyse` uses for the final contributions.

Unit handling is centralised in :func:`wham_units`: CV timeseries are recorded in
nm (RMSD/separation) or rad (Boresch), so window centres (Å for RMSD, nm for
separation, rad for Boresch) and force constants (kcal/mol/Å² or /rad²) are
converted to match — the config remains the single source of truth for ``k``.
"""

from __future__ import annotations

import pathlib
import shutil

from gluebind.analysis.pmf import average_pmfs
from gluebind.analysis.wham import (
    build_wham_command,
    load_pmf,
    run_wham,
    write_metafile,
)
from gluebind.backend.base import JobSpec
from gluebind.backend.scheduler import Scheduler
from gluebind.config.calculation import CalculationConfig
from gluebind.simulation.window import CV_TIMESERIES_FILENAME


def wham_units(
    cv_type: str, centre: float, force_constant: float
) -> tuple[float, float]:
    """Convert a window centre + force constant to WHAM units (nm / rad).

    * Boresch — rad and kcal/mol/rad² (no conversion).
    * RMSD — centre Å→nm (×0.1), k Å⁻²→nm⁻² (×100).
    * Separation — centre already nm, k Å⁻²→nm⁻² (×100).
    """
    if cv_type == "boresch":
        return centre, force_constant
    if cv_type == "rmsd":
        return centre * 0.1, force_constant * 100.0
    if cv_type == "separation":
        return centre, force_constant * 100.0
    raise ValueError(f"unknown cv_type {cv_type!r}")


class WhamPmfProvider:
    """Callable ``stage -> (cv, pmf)`` via WHAM over the stage's windows."""

    # Per-stage WHAM histogram bins (matching the reference us_analysis.py):
    # Boresch PMFs use far less data (fewer, tighter windows) so need fewer bins;
    # separation spans the widest range and uses the most.
    _DEFAULT_NUM_BINS = {"boresch": 50, "rmsd": 100, "separation": 200}

    def __init__(
        self,
        config: CalculationConfig,
        *,
        wham_binary: str | pathlib.Path = "wham",
        location: str = "local",
        backend=None,
        num_bins: int | dict | None = None,
        tol: float = 1e-6,
        numpad: int = 0,
        hist_margin: float = 0.1,
        apply_red: bool = True,
        red_fallback_ns: float = 3.5,
    ) -> None:
        if location not in ("local", "slurm"):
            raise ValueError("location must be 'local' or 'slurm'")
        if location == "slurm" and backend is None:
            raise ValueError("a backend is required for location='slurm'")
        resolved = shutil.which(str(wham_binary))
        if resolved is None:
            raise FileNotFoundError(
                f"WHAM binary {str(wham_binary)!r} was not found on PATH. Install it "
                "with `make wham` from the gluebind repo (this compiles Grossfield "
                "WHAM into the active conda environment), or compile it yourself and "
                "put the `wham` binary on your PATH. See the README for the fallback "
                "options if the download URL has changed."
            )
        self.config = config
        self.wham_binary = resolved
        self.location = location
        self.backend = backend
        self.num_bins = num_bins
        self.tol = tol
        self.numpad = numpad
        self.hist_margin = hist_margin
        self.apply_red = apply_red
        self.red_fallback_ns = red_fallback_ns

    def __call__(self, stage):
        schedule = self.config.sampling.for_cv(stage.cv_type, stage.name)
        k = schedule.force_constant
        ensemble_size = self.config.sampling.ensemble_size

        centres = [wham_units(stage.cv_type, w.centre, k)[0] for w in stage.windows]
        params = [
            min(centres) - self.hist_margin,
            max(centres) + self.hist_margin,
            self._num_bins(stage.cv_type),
            self.tol,
            self.config.sampling.temperature_K,
            self.numpad,
        ]

        replicate_pmfs = []
        for replicate in range(1, ensemble_size + 1):
            entries = []
            for window in stage.windows:
                timeseries = self._resolve_timeseries(stage, window, replicate)
                centre, k_wham = wham_units(stage.cv_type, window.centre, k)
                entries.append((timeseries, centre, k_wham))
            metafile = write_metafile(
                entries, stage.base_dir / f"metafile_run{replicate:02d}.txt"
            )
            pmf_out = stage.base_dir / f"pmf_run{replicate:02d}.txt"
            self._run_wham(metafile, pmf_out, params)
            replicate_pmfs.append(load_pmf(pmf_out))

        cv, mean, _sem = average_pmfs(replicate_pmfs)
        # Return the per-replicate free energies too, so analyse() can propagate the
        # spread across independent repeats into a ΔG uncertainty.
        return cv, mean, [fe for _, fe in replicate_pmfs]

    def _num_bins(self, cv_type: str) -> int:
        """WHAM histogram bins for a stage's CV type — per-stage by default, or a
        uniform int / explicit per-type dict if the caller supplied one."""
        if self.num_bins is None:
            return self._DEFAULT_NUM_BINS[cv_type]
        if isinstance(self.num_bins, dict):
            return self.num_bins.get(cv_type, self._DEFAULT_NUM_BINS[cv_type])
        return self.num_bins

    def _resolve_timeseries(self, stage, window, replicate) -> str:
        """Path to the CV timeseries WHAM should read for this window.

        RMSD and separation timeseries are RED-truncated (equilibration removed)
        into a ``RED/`` subdirectory; Boresch uses the raw file (its fixed 1 ns
        equilibration is already discarded during sampling). If RED finds no
        equilibration (or is unavailable), a fixed fraction — the first
        ``red_fallback_ns`` of the window's sampling time — is discarded instead,
        matching the reference us_analysis.py. RED runs on the driver (cheap,
        CPU-only text analysis; no MD).
        """
        import numpy as np

        raw = window.replicate_dir(replicate) / CV_TIMESERIES_FILENAME
        if stage.cv_type == "boresch" or not self.apply_red:
            return str(raw)

        data = np.loadtxt(raw)
        if data.ndim != 2 or data.shape[0] < 2:
            return str(raw)  # too short to truncate

        try:
            from gluebind.analysis.pmf import detect_equilibration

            idx = detect_equilibration(data[:, 1])
        except Exception:  # noqa: BLE001 - RED found no equilibration or is unavailable
            schedule = self.config.sampling.for_cv(stage.cv_type, stage.name)
            frac = min(self.red_fallback_ns / schedule.sampling_time_ns, 0.9)
            idx = int(frac * data.shape[0])
        idx = max(0, min(idx, data.shape[0] - 1))

        out = raw.parent / "RED" / raw.name
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, data[idx:])
        return str(out)

    def _run_wham(self, metafile, pmf_out, params) -> None:
        if self.location == "local":
            run_wham(self.wham_binary, params, metafile, pmf_out, log=f"{pmf_out}.log")
            return
        # SLURM: submit the wham invocation as a single job and wait for it.
        cmd = build_wham_command(self.wham_binary, params, metafile, pmf_out)
        spec = JobSpec(
            command=cmd, work_dir=str(pathlib.Path(pmf_out).parent), name="wham"
        )
        Scheduler(self.backend, poll_interval=5.0).run([spec])
