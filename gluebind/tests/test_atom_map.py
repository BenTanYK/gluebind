"""Tests for the verified input->complex atom mapping (pure)."""

import pytest

from gluebind.system.atom_map import component_offsets, map_indices, verify_block


def test_component_offsets_cumulative_in_order():
    # gluebind assembly order: glue, then receptor, then target.
    offsets = component_offsets([("glue", 90), ("receptor", 2700), ("target", 6000)])
    assert offsets == {"glue": 0, "receptor": 90, "target": 2790}


def test_component_offsets_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        component_offsets([("glue", -1)])


def _keys(*names):
    # helper: build (atom_name, element) keys with a uniform element
    return [(n, "C") for n in names]


def test_verify_block_matches_ignoring_residue_numbers():
    # The block sits at an offset; identity is atom name + element only, so the
    # verification passes even though the complex here would have renumbered resids.
    inp = _keys("CA", "CB", "CG")
    complx = _keys("X", "Y") + inp + _keys("Z")  # block at offset 2
    verify_block("target", inp, complx, offset=2)  # no raise


def test_verify_block_raises_on_reorder():
    inp = _keys("CA", "CB", "CG")
    complx = _keys("CA", "CG", "CB")  # CB/CG swapped -> reordered within molecule
    with pytest.raises(ValueError, match="verification failed .*atom 1"):
        verify_block("target", inp, complx, offset=0)


def test_verify_block_raises_when_block_exceeds_complex():
    inp = _keys("CA", "CB", "CG")
    complx = _keys("CA", "CB")  # complex smaller than the input block
    with pytest.raises(ValueError, match="runs past the complex"):
        verify_block("target", inp, complx, offset=0)


def test_map_indices_applies_offset():
    assert map_indices([0, 5, 10], offset=2790) == [2790, 2795, 2800]


def test_end_to_end_offset_then_map():
    # A target block placed after glue+receptor; a selection resolved in the input
    # frame maps to the correct complex indices via the verified offset.
    offsets = component_offsets([("glue", 2), ("receptor", 3), ("target", 4)])
    target_in = _keys("N", "CA", "C", "O")
    complx = _keys("g1", "g2", "r1", "r2", "r3") + target_in
    verify_block("target", target_in, complx, offsets["target"])
    # user selected target input atoms [1, 2] (CA, C) -> complex [6, 7]
    assert map_indices([1, 2], offsets["target"]) == [6, 7]
