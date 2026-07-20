"""Tests for the WHAM PMF provider — unit conversion and a full local run driven
by a fake ``wham`` binary (so no real WHAM install is needed)."""

import numpy as np
import pytest

from gluebind.analysis.provider import WhamPmfProvider, wham_units
from gluebind.config import CalculationConfig
from gluebind.runners.stage import Stage

INPUTS = {
    "target": {"prm7": "t.prm7", "rst7": "t.rst7"},
    "receptor": {"prm7": "r.prm7", "rst7": "r.rst7"},
}


def test_wham_units_boresch_unchanged():
    assert wham_units("boresch", 1.05, 100.0) == (1.05, 100.0)


def test_wham_units_rmsd_converts_angstrom_to_nm():
    centre, k = wham_units("rmsd", 2.0, 5.0)
    assert centre == pytest.approx(0.2)  # Å -> nm
    assert k == pytest.approx(500.0)  # Å^-2 -> nm^-2


def test_wham_units_separation_centre_nm_k_converted():
    centre, k = wham_units("separation", 1.5, 10.0)
    assert centre == pytest.approx(1.5)  # already nm
    assert k == pytest.approx(1000.0)


def test_wham_units_unknown_raises():
    with pytest.raises(ValueError):
        wham_units("bogus", 1.0, 1.0)


def test_provider_requires_backend_for_slurm():
    cfg = CalculationConfig.model_validate({"inputs": INPUTS})
    with pytest.raises(ValueError, match="backend"):
        WhamPmfProvider(cfg, wham_binary="wham", location="slurm")


def _fake_wham(tmp_path):
    """An executable stand-in for Grossfield wham: writes a fixed pmf.txt."""
    path = tmp_path / "wham"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "open(sys.argv[8], 'w').write('0.0 1.0\\n0.5 0.25\\n1.0 0.0\\n')\n"
    )
    path.chmod(0o755)
    return path


def test_provider_local_run_with_fake_binary(tmp_path):
    cfg = CalculationConfig.model_validate({"inputs": INPUTS})
    cfg.sampling.ensemble_size = 2

    stage = Stage(
        tmp_path / "rmsd" / "receptor_bound",
        cv_type="rmsd",
        name="receptor_bound",
        dof=None,
        centres=[0.0, 0.2],
        ensemble_size=2,
        spec_builder=lambda **kwargs: None,  # not used; timeseries are written directly
    )
    # write per-window, per-replicate CV timeseries
    for window in stage.windows:
        for replicate in (1, 2):
            run_dir = window.replicate_dir(replicate)
            run_dir.mkdir(parents=True)
            np.savetxt(
                run_dir / "cv_timeseries.dat",
                np.array([[0, 0.1], [1, 0.12], [2, 0.11]]),
            )

    provider = WhamPmfProvider(
        cfg, wham_binary=str(_fake_wham(tmp_path)), location="local"
    )
    cv, pmf = provider(stage)

    assert len(cv) == 3 and len(pmf) == 3
    assert np.allclose(cv, [0.0, 0.5, 1.0])
    # two replicate metafiles + PMFs were produced
    assert (stage.base_dir / "metafile_run01.txt").exists()
    assert (stage.base_dir / "metafile_run02.txt").exists()


def test_resolve_timeseries_red_truncates_rmsd(tmp_path):
    # RMSD timeseries are RED-truncated into a RED/ subdir before WHAM.
    cfg = CalculationConfig.model_validate({"inputs": INPUTS})
    stage = Stage(
        tmp_path / "rmsd" / "receptor_bound",
        cv_type="rmsd",
        name="receptor_bound",
        dof=None,
        centres=[0.0],
        ensemble_size=1,
        spec_builder=lambda **kwargs: None,
    )
    window = stage.windows[0]
    run_dir = window.replicate_dir(1)
    run_dir.mkdir(parents=True)
    np.savetxt(
        run_dir / "cv_timeseries.dat",
        np.column_stack([np.arange(100), np.linspace(0, 1, 100)]),
    )

    provider = WhamPmfProvider(
        cfg, wham_binary=str(_fake_wham(tmp_path)), apply_red=True
    )
    path = provider._resolve_timeseries(stage, window, 1)

    assert path.endswith("RED/cv_timeseries.dat")
    assert (run_dir / "RED" / "cv_timeseries.dat").exists()
    assert np.loadtxt(path).shape[0] <= 100  # truncated (RED or the fixed fallback)


def test_resolve_timeseries_boresch_uses_raw(tmp_path):
    # Boresch keeps the raw timeseries (its 1 ns equilibration is already dropped).
    cfg = CalculationConfig.model_validate({"inputs": INPUTS})
    stage = Stage(
        tmp_path / "boresch" / "thetaA",
        cv_type="boresch",
        name="thetaA",
        dof="thetaA",
        centres=[1.0],
        ensemble_size=1,
        spec_builder=lambda **kwargs: None,
    )
    window = stage.windows[0]
    run_dir = window.replicate_dir(1)
    run_dir.mkdir(parents=True)
    ts = run_dir / "cv_timeseries.dat"
    np.savetxt(ts, np.column_stack([np.arange(10), np.linspace(1.0, 1.1, 10)]))

    provider = WhamPmfProvider(
        cfg, wham_binary=str(_fake_wham(tmp_path)), apply_red=True
    )
    path = provider._resolve_timeseries(stage, window, 1)

    assert path == str(ts)
    assert not (run_dir / "RED").exists()


def test_provider_metafile_has_converted_units(tmp_path):
    cfg = CalculationConfig.model_validate({"inputs": INPUTS})
    cfg.sampling.ensemble_size = 1
    stage = Stage(
        tmp_path / "rmsd" / "receptor_bound",
        cv_type="rmsd",
        name="receptor_bound",
        dof=None,
        centres=[2.0],  # Å
        ensemble_size=1,
        spec_builder=lambda **kwargs: None,
    )
    run_dir = stage.windows[0].replicate_dir(1)
    run_dir.mkdir(parents=True)
    np.savetxt(run_dir / "cv_timeseries.dat", np.array([[0, 0.2]]))

    provider = WhamPmfProvider(
        cfg, wham_binary=str(_fake_wham(tmp_path)), location="local"
    )
    provider(stage)

    line = (stage.base_dir / "metafile_run01.txt").read_text().split()
    # centre 2.0 Å -> 0.2 nm, k 5.0 Å^-2 -> 500 nm^-2
    assert float(line[1]) == pytest.approx(0.2)
    assert float(line[2]) == pytest.approx(500.0)
