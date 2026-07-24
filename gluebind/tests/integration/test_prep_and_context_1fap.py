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

import pathlib

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


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
                "minimisation_steps": 20,
                "nvt_heat_ns": 0.002,
                "npt_ns": 0.002,
                "equilibration_ns": 0.01,
            },
        }
    )
    cfg.sampling.ensemble_size = 1
    return cfg


@pytest.mark.parametrize("with_waters", [False, True], ids=["dry", "wet"])
def test_prepare_produces_manifest_and_context(bss, fap_inputs, tmp_path, with_waters):
    from gluebind.backend import LocalBackend
    from gluebind.spec_builder import build_restraint_context
    from gluebind.system.prep import prepare

    cfg = _tiny_config(fap_inputs, with_waters=with_waters)
    prepared = prepare(
        cfg, tmp_path, LocalBackend(), platform="CUDA", poll_interval=1.0
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
