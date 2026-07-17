"""Tests for the Phase 3 prep layer's pure helpers and the prepared manifest.

The BioSimSpace/MD driver itself is verified in integration (Phase 7); here we
cover the force-field validation, box sizing, multi-molecule layout bookkeeping,
and the PreparedSystem manifest — none of which need BSS.
"""

import pytest

from gluebind.system import compute_layout
from gluebind.system.prep import (
    PreparedSystem,
    box_length,
    normalise_ff_name,
    validate_forcefield,
)


def test_box_length():
    assert box_length([0, 0, 0], [1, 2, 3], 1.5) == 6.0  # max dim 3 + 2*1.5


def test_normalise_ff_name():
    assert normalise_ff_name("openff_unconstrained-2.2.1") == "openff_unconstrained_2_2_1"
    assert normalise_ff_name("gaff2") == "gaff2"


def test_validate_forcefield_ok():
    assert validate_forcefield("gaff2", ["gaff2", "ff14SB"]) == "gaff2"


def test_validate_forcefield_normalises_dash_dot():
    assert (
        validate_forcefield("openff_unconstrained_2.2.1", ["openff_unconstrained-2.2.1"])
        == "openff_unconstrained_2_2_1"
    )


def test_validate_forcefield_unknown_raises():
    # the real env has only the -rc1 variant, not plain 2.2.1
    with pytest.raises(ValueError):
        validate_forcefield(
            "openff_unconstrained_2.2.1", ["gaff2", "openff_unconstrained-2.2.1-rc1"]
        )


def test_compute_layout_single_chain():
    layout = compute_layout(1, 1, has_glue=True)
    assert layout.target == [0]
    assert layout.receptor == [1]
    assert layout.glue == 2
    assert layout.n_molecules == 3


def test_compute_layout_multichain_target():
    # a chain-split target (e.g. BRD4 tandem bromodomains -> 2 molecules)
    layout = compute_layout(2, 1, has_glue=True)
    assert layout.target == [0, 1]
    assert layout.receptor == [2]
    assert layout.glue == 3


def test_compute_layout_no_glue():
    layout = compute_layout(1, 1, has_glue=False)
    assert layout.glue is None
    assert layout.n_molecules == 2


def test_compute_layout_requires_molecules():
    with pytest.raises(ValueError):
        compute_layout(0, 1, has_glue=True)


def test_prepared_system_roundtrip(tmp_path):
    prepared = PreparedSystem(
        complex_prm7="complex_equil.prm7",
        complex_rst7="complex_equil.rst7",
        complex_trajectory="complex_equil.dcd",
        target_bulk_prm7="target_bulk.prm7",
        target_bulk_rst7="target_bulk.rst7",
        receptor_bulk_prm7="receptor_bulk.prm7",
        receptor_bulk_rst7="receptor_bulk.rst7",
        glue_assign_to="receptor",
        target_molecules=[0],
        receptor_molecules=[1],
        glue_molecule=2,
    )
    prepared.dump(tmp_path)
    assert PreparedSystem.load(tmp_path) == prepared
