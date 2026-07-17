"""Tests for CalcSet: subdirectory discovery, path resolution, result aggregation,
correlation stats, CSV.

The per-calculation ``analyse`` (which needs WHAM) is stubbed, so these exercise
the set-level orchestration and the dependency-light aggregation without MD.
"""

import pathlib

import pytest
import yaml

from gluebind.backend import LocalBackend
from gluebind.config import CalculationConfig
from gluebind.runners.calc_set import (
    CalcSet,
    correlation_stats,
    kendall_tau,
    mae,
    pearson_r,
    write_results_csv,
)

# A ternary config (with glue) and a binary-PPI config (no glue) — proving
# systems in one set can differ structurally.
TERNARY_YAML = """
inputs:
  target: {prm7: ck1a.prm7, rst7: ck1a.rst7}
  receptor: {prm7: crbn.prm7, rst7: crbn.rst7}
  glue: {sdf: lenalidomide.sdf, assign_to: receptor}
"""
BINARY_YAML = """
inputs:
  target: {prm7: bd1.prm7, rst7: bd1.rst7}
  receptor: {prm7: dcaf16.prm7, rst7: dcaf16.rst7}
"""


def _system(base, name, config_yaml):
    d = base / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(config_yaml)
    return d


# ---- config path resolution (the self-contained-dir enabler) ---------------


def test_with_resolved_input_paths_makes_absolute(tmp_path):
    cfg = CalculationConfig.model_validate(yaml.safe_load(TERNARY_YAML))
    resolved = cfg.with_resolved_input_paths(tmp_path)
    assert resolved.inputs.target.prm7 == str((tmp_path / "ck1a.prm7").resolve())
    assert resolved.inputs.glue.sdf == str((tmp_path / "lenalidomide.sdf").resolve())
    assert pathlib.Path(resolved.inputs.receptor.rst7).is_absolute()


def test_with_resolved_input_paths_leaves_absolute_untouched():
    cfg = CalculationConfig.model_validate(
        {
            "inputs": {
                "target": {"prm7": "/abs/t.prm7", "rst7": "/abs/t.rst7"},
                "receptor": {"prm7": "/abs/r.prm7", "rst7": "/abs/r.rst7"},
            }
        }
    )
    resolved = cfg.with_resolved_input_paths("/somewhere/else")
    assert resolved.inputs.target.prm7 == "/abs/t.prm7"


# ---- pure aggregation helpers ----------------------------------------------


def test_pearson_r_perfect():
    assert pearson_r([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_mae():
    assert mae([1.0, 2.0], [1.5, 2.5]) == pytest.approx(0.5)


def test_kendall_tau_monotonic():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_correlation_stats_needs_two_points():
    assert correlation_stats([{"dg_bind": -9.0, "experimental_dg": -10.0}]) == {}


def test_correlation_stats_ignores_rows_without_experimental():
    rows = [
        {"dg_bind": -9.0, "experimental_dg": -10.0},
        {"dg_bind": -7.0, "experimental_dg": -8.0},
        {"dg_bind": -5.0},  # no experimental value -> excluded
    ]
    stats = correlation_stats(rows)
    assert stats["n"] == 2
    assert set(stats) == {"n", "pearson_r", "r2", "mae", "kendall_tau"}


def test_write_results_csv_union_columns(tmp_path):
    rows = [
        {"system": "A", "dg_bind": -9.0, "experimental_dg": -10.0},
        {"system": "B", "dg_bind": -7.0},  # missing experimental -> blank cell
    ]
    path = write_results_csv(tmp_path / "results.csv", rows)
    assert path.read_text().splitlines()[0] == "system,dg_bind,experimental_dg"


# ---- CalcSet discovery / structure -----------------------------------------


def test_scan_discovers_system_subdirs(tmp_path):
    _system(tmp_path, "CK1a-WT", TERNARY_YAML)
    _system(tmp_path, "BD1-DCAF16", BINARY_YAML)  # binary PPI, no glue
    (tmp_path / "notes").mkdir()  # no config.yaml -> ignored

    cset = CalcSet(tmp_path, LocalBackend())

    assert set(cset.calcs) == {"CK1a-WT", "BD1-DCAF16"}
    # each calc runs *in place* in its own subdir (the standard basedir)
    assert cset.calcs["CK1a-WT"].base_dir == tmp_path / "CK1a-WT"
    assert all(c.spec_builder is None for c in cset.calcs.values())  # deferred
    # structurally different configs coexist: one has glue, the other doesn't
    assert cset.calcs["CK1a-WT"].config.inputs.glue is not None
    assert cset.calcs["BD1-DCAF16"].config.inputs.glue is None
    # input paths resolved absolute, relative to each subdir
    prm7 = cset.calcs["CK1a-WT"].config.inputs.target.prm7
    assert pathlib.Path(prm7).is_absolute() and prm7.endswith("CK1a-WT/ck1a.prm7")


def test_experimental_values_from_benchmark_yaml(tmp_path):
    _system(tmp_path, "A", TERNARY_YAML)
    _system(tmp_path, "B", BINARY_YAML)
    (tmp_path / "benchmark.yaml").write_text(
        yaml.safe_dump({"experimental_dg": {"A": -10.0, "B": -8.0}})
    )
    cset = CalcSet(tmp_path, LocalBackend())
    assert cset.experimental == {"A": -10.0, "B": -8.0}


def test_analyse_aggregates_and_writes_csv(tmp_path):
    _system(tmp_path, "A", TERNARY_YAML)
    _system(tmp_path, "B", BINARY_YAML)
    (tmp_path / "benchmark.yaml").write_text(
        yaml.safe_dump({"experimental_dg": {"A": -10.0, "B": -8.0}})
    )
    cset = CalcSet(tmp_path, LocalBackend())

    fake = {
        "A": {"dg_bind": -9.5, "dg_rmsd": 0.0, "dg_boresch": 0.0, "dg_sep": -9.5, "dg_corr": 0.0},
        "B": {"dg_bind": -7.0, "dg_rmsd": 0.0, "dg_boresch": 0.0, "dg_sep": -7.0, "dg_corr": 0.0},
    }
    for name, calc in cset.calcs.items():
        calc.analyse = (lambda result: (lambda *a, **k: result))(fake[name])

    out = cset.analyse()

    assert [r["system"] for r in out["results"]] == ["A", "B"]
    assert out["results"][0]["experimental_dg"] == -10.0
    assert out["stats"]["n"] == 2
    assert out["stats"]["r2"] == pytest.approx(1.0)  # two points -> perfectly correlated
    # the ONLY set-level artifact
    assert (tmp_path / "results.csv").exists()
