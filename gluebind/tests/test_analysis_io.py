"""Tests for the WHAM metafile/PMF I/O and replicate PMF averaging."""

import numpy as np
import pytest

from gluebind.analysis import average_pmfs, build_wham_command, load_pmf, write_metafile
from gluebind.analysis.wham import METAFILE_NAME


def test_write_metafile(tmp_path):
    entries = [
        (f"{tmp_path}/0.9.txt", 0.9, 1000.0),
        (f"{tmp_path}/1.0.txt", 1.0, 1000.0),
    ]
    path = write_metafile(entries, tmp_path / METAFILE_NAME)
    lines = path.read_text().splitlines()
    assert lines[0] == f"{tmp_path}/0.9.txt 0.9 1000.0"
    assert len(lines) == 2


def test_build_wham_command_shape():
    cmd = build_wham_command("wham", [0.9, 3.0, 500, 1e-6, 298.15, 0], "meta.txt", "pmf.txt")
    assert cmd[0] == "wham"
    assert cmd[-2:] == ["meta.txt", "pmf.txt"]
    assert len(cmd) == 9


def test_build_wham_command_bad_params():
    with pytest.raises(ValueError):
        build_wham_command("wham", [0.9, 3.0, 500], "meta.txt", "pmf.txt")


def test_load_pmf(tmp_path):
    p = tmp_path / "pmf.txt"
    p.write_text("0.0 1.0\n0.1 0.5\n0.2 0.0\n")
    x, w = load_pmf(p)
    assert np.allclose(x, [0.0, 0.1, 0.2])
    assert np.allclose(w, [1.0, 0.5, 0.0])


def test_average_pmfs_identical():
    x = np.array([0.0, 0.1, 0.2])
    w = np.array([1.0, 0.5, 0.0])
    cv, mean, sem = average_pmfs([(x, w), (x, w), (x, w)])
    assert np.allclose(mean, w)
    assert np.allclose(sem, 0.0)


def test_average_pmfs_mean_and_sem():
    x = np.array([0.0, 1.0])
    cv, mean, sem = average_pmfs([(x, [0.0, 0.0]), (x, [2.0, 4.0])])
    assert np.allclose(mean, [1.0, 2.0])
    assert np.all(sem > 0)


def test_average_pmfs_sem_uses_ddof1():
    # replicates [0,0] and [2,4]: mean [1,2]; sample-std (ddof=1)/sqrt(2) = [1,2]
    x = np.array([0.0, 1.0])
    _, _, sem = average_pmfs([(x, [0.0, 0.0]), (x, [2.0, 4.0])])
    assert np.allclose(sem, [1.0, 2.0])


def test_average_pmfs_single_replicate_zero_sem():
    x = np.array([0.0, 1.0])
    _, mean, sem = average_pmfs([(x, [1.0, 2.0])])
    assert np.allclose(mean, [1.0, 2.0])
    assert np.allclose(sem, 0.0)


def test_pmf_minimum_ignores_nonfinite():
    from gluebind.analysis.pmf import pmf_minimum

    cv = np.array([0.0, 0.5, 1.0, 1.5])
    pmf = np.array([np.nan, 2.0, -1.0, np.inf])  # true minimum at cv = 1.0
    assert pmf_minimum(cv, pmf) == 1.0


def test_average_pmfs_inconsistent_length():
    x = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError):
        average_pmfs([(x, [0.0, 0.0, 0.0]), (x, [0.0, 0.0])])


def test_average_pmfs_empty():
    with pytest.raises(ValueError):
        average_pmfs([])
