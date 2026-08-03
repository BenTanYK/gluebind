"""Integration tier: the full geometric-route pipeline on 1FAP (the heaviest test).

Needs BioSimSpace + a GPU + the ``wham`` binary (and ``red`` for PMF truncation, else
it falls back). Runs ``Calculation.from_config(...).run().analyse()`` on the 1FAP
fixture end to end — prep, RMSD US, the sequential Boresch chain, steered MD, and
separation, all via ``LocalBackend`` on the GPU — then a ΔG° from WHAM, and checks a
re-run is idempotent (resume reuses completed windows rather than re-sampling).

The sampling/window settings below are minimal placeholders to keep the run short.
When first run against the real env, expect to tune them — especially the separation
SMD range/spacing and per-stage window counts — for a sane wall-clock; the point of
this test is that the *pipeline* runs end to end and returns a finite, resumable ΔG°.
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


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
                "minimisation_steps": 20,
                "nvt_heat_ns": 0.002,
                "npt_ns": 0.002,
                "equilibration_ns": 0.01,
            },
        }
    )
    s = cfg.sampling
    s.ensemble_size = 1
    for sub in (s.rmsd, s.boresch, s.separation):
        sub.sampling_time_ns = 0.01
        sub.equil_discard_ns = 0.0
    return cfg


def _calc(cfg, base_dir):
    from gluebind.backend import LocalBackend
    from gluebind.runners import Calculation

    return Calculation.from_config(
        cfg, base_dir, LocalBackend(), platform="CUDA", poll_interval=1.0
    )


def test_full_pipeline_and_resume(bss, wham_binary, fap_inputs, tmp_path):
    cfg = _e2e_config(fap_inputs)

    # run() self-prepares and self-defaults the WHAM provider for Boresch feedback.
    calc = _calc(cfg, tmp_path)
    calc.run()
    result = calc.analyse()
    assert math.isfinite(result["dg_bind"])
    assert result["rmsd_included"] is True

    # resume: a fresh calculation over the same dir re-runs nothing (every window is
    # complete on disk), so re-analysing the same timeseries reproduces the ΔG°.
    resumed = _calc(cfg, tmp_path)
    resumed.run()
    result2 = resumed.analyse()
    assert result2["dg_bind"] == pytest.approx(result["dg_bind"], abs=1e-6)
