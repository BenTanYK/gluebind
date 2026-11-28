"""Integration tier: BioSimSpace prep + restraint-context resolution on 1FAP.

Needs BioSimSpace + a GPU (real parameterise/solvate/equilibrate + bulk extraction).
Runs ``prepare()`` with tiny runtimes on the 1FAP fixture (all-Cα default; dry and
wet), then resolves ``build_restraint_context`` — the real exercise of the BSS /
MDAnalysis layer the unit suite can only mock, including the input->complex atom map
on the real ``TER`` split and (for the wet case) crystal waters via ``inputs.waters``.
A third case runs ``ensemble_from: equilibration`` with two replicates, which is
the only place the per-replicate fan-out meets real trajectories.

The prep runtimes are deliberately minimal placeholders; when first run against the
real env, expect to lengthen the equilibration if auto anchor selection needs more
trajectory frames.
"""

import os
import pathlib
import uuid

import pytest

from gluebind.runners import Calculation

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.gpu]


def _run_dir(case: str) -> pathlib.Path:
    """Return a shared-filesystem directory, preserving Slurm logs for diagnosis."""
    root = pathlib.Path(
        os.environ.get("GLUEBIND_TEST_RUN_ROOT", ".pytest-slurm")
    ).resolve()
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
    run_dir = _run_dir("wet" if with_waters else "dry")
    prepared = prepare(
        cfg,
        run_dir,
        SlurmBackend(slurm_config),
        platform="CUDA",
        poll_interval=slurm_config.queue_check_interval,
    ).for_replicate(1)

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


def test_ensemble_equilibration_prepares_each_replicate_independently(
    bss, fap_inputs, slurm_config
):
    # The fan-out against the real BSS/MDAnalysis layer: two production runs from
    # the shared NPT state, each with its own bulk pair, and two contexts resolved
    # from two different trajectories.
    from gluebind.backend import SlurmBackend
    from gluebind.spec_builder import build_restraint_context

    cfg = _tiny_config(fap_inputs, with_waters=False)
    cfg.sampling.ensemble_size = 2
    cfg.sampling.ensemble_from = "equilibration"
    run_dir = _run_dir("ensemble")
    calc = Calculation.from_config(
        cfg,
        SlurmBackend(slurm_config),
        base_dir=run_dir,
        scheduler_config=slurm_config,
    )
    prepared = calc.prepare()

    assert set(prepared.replicates) == {1, 2}
    contexts = {}
    for replicate in (1, 2):
        one = prepared.for_replicate(replicate)
        # each replicate equilibrated into its own directory, bulk pair included
        for path in (
            one.complex_rst7,
            one.complex_trajectory,
            one.receptor_bulk_rst7,
            one.target_bulk_rst7,
        ):
            assert pathlib.Path(path).exists()
            assert f"run_{replicate:02d}" in str(path)
        # and resolved its own restraint context, persisted beside it
        assert calc._restraint_context_path(replicate).exists()
        contexts[replicate] = build_restraint_context(one, cfg)
        assert set(contexts[replicate].anchors) == {"b", "c", "B", "C"}

    # the RMSD regions come from input-topology selections, so they must agree
    # even though the structures differ — if they don't, the atom map is wrong
    assert contexts[1].rmsd_atoms_bound == contexts[2].rmsd_atoms_bound
    assert contexts[1].rmsd_order == contexts[2].rmsd_order
