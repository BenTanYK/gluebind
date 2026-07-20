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
from gluebind.runners.base import SimulationRunner
from gluebind.runners.calculation import Calculation


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
        self.experimental = self._load_experimental(self.base_dir / self.MANIFEST_FILENAME)
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
            p for p in base_dir.iterdir() if p.is_dir() and (p / cls.CONFIG_FILENAME).exists()
        )

    @staticmethod
    def _load_experimental(manifest_path: pathlib.Path) -> dict[str, float]:
        """Optional ``{name: experimental_dg}`` from a ``benchmark.yaml``."""
        if not manifest_path.exists():
            return {}
        data = yaml.safe_load(manifest_path.read_text()) or {}
        return {str(k): float(v) for k, v in (data.get("experimental_dg") or {}).items()}

    def prepare(self) -> None:
        """Prepare every system (each through the shared backend)."""
        for calc in self.calcs.values():
            calc.prepare()

    def run(self) -> None:
        """Run every system to completion, one after another.

        **Sequential across systems:** each system's full pipeline (prep → RMSD →
        sequential Boresch → SMD → separation) completes before the next begins, so
        on SLURM the queue is filled by only one system at a time. Each calc is
        independently resumable, so re-running skips completed work. (Cross-system
        concurrency — submitting/polling all systems at once to fill the cluster —
        is a deliberate future enhancement, not yet implemented.)

        One system's failure does not abort the benchmark: its error is collected
        and re-raised in a summary once every system has been attempted, so a single
        bad input does not discard the runs that did complete (all are resumable).
        """
        failures: dict[str, Exception] = {}
        for name, calc in self.calcs.items():
            try:
                calc.run()
            except Exception as exc:  # noqa: BLE001 - surface per system, keep going
                failures[name] = exc
        if failures:
            summary = "; ".join(f"{name}: {exc}" for name, exc in failures.items())
            raise RuntimeError(
                f"{len(failures)}/{len(self.calcs)} system(s) failed to run: {summary}"
            )

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
