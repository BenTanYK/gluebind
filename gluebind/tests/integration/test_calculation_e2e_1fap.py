"""Slurm end-to-end validation of the full geometric-route pipeline on 1FAP.

The pytest driver runs on the login node. ``Calculation`` dispatches raw system
construction, equilibration, steered MD, and every umbrella window through
``SlurmBackend``. It therefore validates the real installed stack, including
shared filesystem paths, scheduler submission, CUDA/OpenMM, and resume.

The short sampling schedule is a machinery check, not a converged free-energy
calculation: WHAM may legitimately produce ``NaN`` when its windows lack overlap.
"""

import os
import pathlib
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slurm]


def _e2e_config(fap_inputs):
    from gluebind.config.calculation import CalculationConfig

    cfg = CalculationConfig.model_validate(
        {
            "inputs": {
                "receptor": fap_inputs["receptor"],
                "target": fap_inputs["target"],
                "glue": fap_inputs["glue"],
            },
            "prep": {
                "minimisation_steps": 1000,
                "nvt_heat_ns": 0.1,
                "npt_ns": 0.1,
                "equilibration_ns": 0.1,
            },
        }
    )
    s = cfg.sampling
    s.ensemble_size = 1
    for sub in (s.rmsd, s.boresch, s.separation):
        sub.sampling_time_ns = 0.01
        sub.equil_discard_ns = 0.0
    return cfg


def _calc(cfg, base_dir, slurm):
    from gluebind.backend import SlurmBackend
    from gluebind.runners import Calculation

    return Calculation.from_config(
        cfg,
        SlurmBackend(slurm),
        base_dir=base_dir,
        slurm_config=slurm,
        platform="CUDA",
        poll_interval=slurm.queue_check_interval,
    )


def _run_dir() -> pathlib.Path:
    """Return a shared-filesystem directory, preserving Slurm logs for diagnosis."""
    explicit = os.environ.get("GLUEBIND_TEST_RUN_DIR")
    if explicit:
        return pathlib.Path(explicit).resolve()
    root = pathlib.Path(
        os.environ.get("GLUEBIND_TEST_RUN_ROOT", ".pytest-slurm")
    ).resolve()
    return root / f"1fap-e2e-{uuid.uuid4().hex}"


def test_full_pipeline_and_resume(bss, wham_binary, fap_inputs, slurm_config):
    cfg = _e2e_config(fap_inputs)
    run_dir = _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    # run() self-prepares and self-defaults the WHAM provider for Boresch
    # feedback. All computational work is submitted through Slurm.
    calc = _calc(cfg, run_dir, slurm_config)
    calc.run()
    result = calc.analyse()
    assert result["rmsd_included"] is True
    assert "dg_bind" in result

    # Resume must re-use all completed work. A Slurm submission writes one .sh
    # file in the relevant job directory, so this detects accidental resubmission.
    scripts_before_resume = sorted(run_dir.rglob("*.sh"))
    assert scripts_before_resume
    resumed = _calc(cfg, run_dir, slurm_config)
    resumed.run()
    result2 = resumed.analyse()
    assert result2["rmsd_included"] is True
    assert sorted(run_dir.rglob("*.sh")) == scripts_before_resume
