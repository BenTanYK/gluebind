"""CalcSet — run a whole set of calculations from one place.

The top of the runner hierarchy: ``CalcSet → Calculation → Group → Stage →
Window``. A :class:`CalcSet` is deliberately *nothing more than a directory of
standard calculation basedirs*: its ``base_dir`` contains one subdirectory per
system, each holding a ``config.yaml`` (with its AMBER inputs, paths relative to
the subdirectory). Running scaffolds the usual per-calculation layout
(``.gluebind-state.json``, ``prep/``, ``boresch/`` …) *inside each subdirectory*,
exactly as a single calculation would — CalcSet adds no new per-calculation
structure. The only set-level artifact is a ``results.csv`` at ``base_dir``.

Because each system carries its own full config, systems can differ freely —
ternary complexes with a glue, binary PPIs with none, different targets/mutants —
without any shared-base machinery.

Optional experimental ΔG° values (for correlation statistics) live in a
``benchmark.yaml`` at ``base_dir`` (``experimental_dg: {name: value}``) — the one
genuinely set-level input. Aggregation stays dependency-light (numpy + stdlib
``csv``); plotting is intentionally left out of the core.
"""

from __future__ import annotations

import csv
import pathlib

import yaml

from gluebind.backend.base import Backend
from gluebind.backend.scheduler import SlotPool
from gluebind.logutil import add_file_handler, get_logger
from gluebind.runners.base import SimulationRunner
from gluebind.runners.calculation import Calculation

logger = get_logger("calc_set")


class CalcSet(SimulationRunner):
    """A directory of per-system calculation basedirs, run and analysed together."""

    CONFIG_FILENAME = "config.yaml"
    MANIFEST_FILENAME = "benchmark.yaml"
    RESULTS_FILENAME = "results.csv"

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        backend: Backend,
        *,
        platform: str = "CUDA",
        poll_interval: float = 30.0,
    ) -> None:
        super().__init__(base_dir)
        self.backend = backend
        self.experimental = self._load_experimental(
            self.base_dir / self.MANIFEST_FILENAME
        )
        self.calcs: dict[str, Calculation] = {}
        for system_dir in self._system_dirs(self.base_dir):
            self.calcs[system_dir.name] = Calculation.from_config(
                system_dir / self.CONFIG_FILENAME,
                system_dir,
                backend,
                platform=platform,
                poll_interval=poll_interval,
            )
        self.sub_runners = list(self.calcs.values())

    @classmethod
    def _system_dirs(cls, base_dir: pathlib.Path) -> list[pathlib.Path]:
        """Subdirectories of ``base_dir`` that contain a ``config.yaml`` (sorted)."""
        if not base_dir.is_dir():
            return []
        return sorted(
            p
            for p in base_dir.iterdir()
            if p.is_dir() and (p / cls.CONFIG_FILENAME).exists()
        )

    @staticmethod
    def _load_experimental(manifest_path: pathlib.Path) -> dict[str, float]:
        """Optional ``{name: experimental_dg}`` from a ``benchmark.yaml``."""
        if not manifest_path.exists():
            return {}
        data = yaml.safe_load(manifest_path.read_text()) or {}
        return {
            str(k): float(v) for k, v in (data.get("experimental_dg") or {}).items()
        }

    def prepare(self) -> None:
        """Prepare every system (each through the shared backend)."""
        for calc in self.calcs.values():
            calc.prepare()

    def run(
        self,
        *,
        max_parallel_systems: int = 1,
        max_concurrent_jobs: int | None = None,
    ) -> None:
        """Run every system, sequentially or several at once.

        ``max_parallel_systems=1`` (default) runs each system's full pipeline (prep
        → RMSD → sequential Boresch → SMD → separation) to completion before the
        next begins. ``>1`` runs that many systems concurrently, each on its own
        thread — so one system's serial phases (the Boresch chain, prep/SMD waits)
        overlap with another's active windows, and the cluster stays fuller. The
        backend does the real work distribution across nodes; the threads just keep
        the queue fed.

        ``max_concurrent_jobs`` caps the *total* in-flight jobs across all
        concurrent systems (via a shared :class:`SlotPool`) so parallel submission
        stays within the cluster's per-user limit; leave it ``None`` to let each
        system use its own scheduler limit. Ignored when running sequentially.

        Each calc is independently resumable, so re-running skips completed work.
        One system's failure does not abort the benchmark: errors are collected and
        re-raised in a summary once every system has been attempted.
        """
        add_file_handler(self.base_dir)
        if max_parallel_systems < 1:
            raise ValueError("max_parallel_systems must be >= 1")

        if max_parallel_systems == 1:
            failures = self._run_sequential()
        else:
            failures = self._run_parallel(max_parallel_systems, max_concurrent_jobs)

        if failures:
            summary = "; ".join(f"{name}: {exc}" for name, exc in failures.items())
            raise RuntimeError(
                f"{len(failures)}/{len(self.calcs)} system(s) failed to run: {summary}"
            )
        logger.info("all %d system(s) complete", len(self.calcs))

    def _run_one(
        self, name: str, calc: Calculation, job_slots=None
    ) -> Exception | None:
        """Run one system, returning its exception (surfaced, not raised) or None."""
        try:
            calc.run(job_slots=job_slots)
            return None
        except Exception as exc:  # noqa: BLE001 - surface per system, keep going
            logger.error("system %s failed: %s", name, exc)
            return exc

    def _run_sequential(self) -> dict[str, Exception]:
        failures: dict[str, Exception] = {}
        total = len(self.calcs)
        for i, (name, calc) in enumerate(self.calcs.items(), start=1):
            logger.info("system %d/%d: running %s", i, total, name)
            exc = self._run_one(name, calc)
            if exc is not None:
                failures[name] = exc
        return failures

    def _run_parallel(
        self, max_parallel_systems: int, max_concurrent_jobs: int | None
    ) -> dict[str, Exception]:
        from concurrent.futures import ThreadPoolExecutor

        slots = SlotPool(max_concurrent_jobs) if max_concurrent_jobs else None
        workers = min(max_parallel_systems, len(self.calcs))
        logger.info(
            "running %d system(s), up to %d at once%s",
            len(self.calcs),
            workers,
            f" (<= {max_concurrent_jobs} jobs in flight)" if slots else "",
        )
        failures: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._run_one, name, calc, slots): name
                for name, calc in self.calcs.items()
            }
            for future in futures:
                name = futures[future]
                exc = future.result()  # _run_one never raises; it returns the error
                if exc is not None:
                    failures[name] = exc
        return failures

    def analyse(self, *, save_csv: bool = True) -> dict:
        """Aggregate every system's ΔG° into a table (+ ``results.csv``) and stats.

        Returns ``{"results": [...per-system rows...], "stats": {...}}``. Each row
        carries the system name, its ΔG° and components, and (where present) its
        experimental value. ``stats`` (Pearson r/R², MAE, Kendall τ) is computed
        over the systems with an experimental value, and is empty below two.
        """
        rows: list[dict] = []
        for name, calc in self.calcs.items():
            row = {"system": name, **calc.analyse()}
            if name in self.experimental:
                row["experimental_dg"] = self.experimental[name]
            rows.append(row)

        stats = correlation_stats(rows)
        if save_csv:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            write_results_csv(self.base_dir / self.RESULTS_FILENAME, rows)
        return {"results": rows, "stats": stats}


# ---- aggregation helpers (pure; unit-tested) -------------------------------


def pearson_r(x, y) -> float:
    """Pearson correlation coefficient (nan for fewer than two points)."""
    import numpy as np

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mae(x, y) -> float:
    """Mean absolute error between paired values."""
    import numpy as np

    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


def kendall_tau(x, y) -> float:
    """Kendall's τ rank correlation (nan for fewer than two points)."""
    import numpy as np

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return float((concordant - discordant) / total) if total else float("nan")


def correlation_stats(rows: list[dict]) -> dict:
    """Correlation of calculated vs experimental ΔG over rows that have both."""
    calc = [r["dg_bind"] for r in rows if r.get("experimental_dg") is not None]
    exp = [r["experimental_dg"] for r in rows if r.get("experimental_dg") is not None]
    if len(calc) < 2:
        return {}
    r = pearson_r(calc, exp)
    return {
        "n": len(calc),
        "pearson_r": r,
        "r2": r * r,
        "mae": mae(calc, exp),
        "kendall_tau": kendall_tau(calc, exp),
    }


def write_results_csv(path: str | pathlib.Path, rows: list[dict]) -> pathlib.Path:
    """Write the per-system results table to CSV (columns = union of row keys)."""
    path = pathlib.Path(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
