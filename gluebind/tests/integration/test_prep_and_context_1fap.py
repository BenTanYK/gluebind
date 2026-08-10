"""Integration tier: BioSimSpace prep + restraint-context resolution on 1FAP.

Needs BioSimSpace + a GPU (real parameterise/solvate/equilibrate + bulk extraction).
Runs ``prepare()`` with tiny runtimes on the 1FAP fixture (all-Cα default; dry and
wet), then resolves ``build_restraint_context`` — the real exercise of the BSS /
MDAnalysis layer the unit suite can only mock, including the input->complex atom map
on the real ``TER`` split and (for the wet case) crystal waters via ``inputs.waters``.

The prep runtimes are deliberately minimal placeholders; when first run against the
real env, expect to lengthen the equilibration if auto anchor selection needs more
trajectory frames.
"""

import os
import pathlib
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.gpu]


def _run_dir(*, with_waters: bool) -> pathlib.Path:
    """Return a shared-filesystem directory, preserving Slurm logs for diagnosis."""
    root = pathlib.Path(
        os.environ.get("GLUEBIND_TEST_RUN_ROOT", ".pytest-slurm")
    ).resolve()
    case = "wet" if with_waters else "dry"
    return root / f"1fap-prep-{case}-{uuid.uuid4().hex}"


def _tiny_config(fap_inputs, *, with_waters):
    from gluebind.config.calculation import CalculationConfig

    inputs = {
        "receptor": fap_inputs["receptor"],
        "target": fap_inputs["target"],
        "glue": fap_inputs["glue"],
    }
    if with_waters:
        inputs["waters"] = fap_inputs["waters"]
    cfg = CalculationConfig.model_validate(
        {
            "inputs": inputs,
            "prep": {
                "glue_forcefield": "gaff2",
                "minimisation_steps": 1000,
                "nvt_heat_ns": 0.1,
                "npt_ns": 0.1,
                "equilibration_ns": 0.1,
            },
        }
    )
    cfg.sampling.ensemble_size = 1
    return cfg


@pytest.mark.parametrize("with_waters", [False, True], ids=["dry", "wet"])
def test_prepare_produces_manifest_and_context(
    bss, fap_inputs, slurm_config, with_waters
):
    from gluebind.backend import SlurmBackend
    from gluebind.spec_builder import build_restraint_context
    from gluebind.system.prep import prepare

    cfg = _tiny_config(fap_inputs, with_waters=with_waters)
    run_dir = _run_dir(with_waters=with_waters)
    prepared = prepare(
        cfg,
        run_dir,
        SlurmBackend(slurm_config),
        platform="CUDA",
        poll_interval=slurm_config.queue_check_interval,
    )

    # prep produced the assembled complex + both isolated bulk species
    for path in (
        prepared.complex_prm7,
        prepared.complex_rst7,
        prepared.receptor_bulk_prm7,
        prepared.target_bulk_prm7,
    ):
        assert pathlib.Path(path).exists()

    # context resolves against the real topologies via the verified atom map
    ctx = build_restraint_context(prepared, cfg)
    assert ctx.rec_group and ctx.lig_group  # interface Cα groups detected
    assert set(ctx.anchors) == {"b", "c", "B", "C"}  # four Boresch anchors selected
    assert set(ctx.rmsd_order) == {"receptor", "target"}  # all-Cα default regions
    assert set(ctx.rmsd_bulk) == {"receptor", "target"}  # bulk targets for both
