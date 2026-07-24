"""Integration tier: real BioSimSpace assembly + the input->complex atom map.

1FAP has FKBP12 and FRB on separate chains (a ``TER`` split) — the case the §14
atom map guards against. These tests assemble the real complex with BioSimSpace
and check that:

* the verified map (:class:`gluebind.spec_builder._ComplexMap`) lands on the
  intended atoms across the ``TER`` split (its construction *verifies* each block
  atom-for-atom, so a wrong BSS re-ordering would raise here), and
* appending crystal waters as a separate last component (``inputs.waters``) does
  **not** shift the protein atom blocks — the crystal-water robustness property.

Glue parameterisation (AM1-BCC) is slow, so the map is exercised on the two
proteins (``has_glue=False``); the glue-inclusive path is covered by the full-prep
integration test.
"""

import pytest

from gluebind.config.prep import PrepConfig

pytestmark = pytest.mark.integration

RECEPTOR_ATOMS = 1663  # FKBP12 (first protein block)


def _assemble(bss, fap_inputs, *, with_waters: bool, out_prefix) -> str:
    """Assemble receptor+target (+ optional crystal waters), solvate, save; return
    the complex prm7 path."""
    from gluebind.system.inputs import load_system, load_waters
    from gluebind.system.prep import assemble_and_solvate

    target = load_system(fap_inputs["target"]["prm7"], fap_inputs["target"]["rst7"])
    receptor = load_system(
        fap_inputs["receptor"]["prm7"], fap_inputs["receptor"]["rst7"]
    )
    waters = (
        load_waters(fap_inputs["waters"]["prm7"], fap_inputs["waters"]["rst7"])
        if with_waters
        else None
    )
    solvated = assemble_and_solvate(target, receptor, None, waters, PrepConfig())
    bss.IO.saveMolecules(str(out_prefix), solvated, ["prm7", "rst7"])
    return f"{out_prefix}.prm7"


def _complex_map(complex_prm7: str, fap_inputs):
    import MDAnalysis as mda

    from gluebind.spec_builder import _ComplexMap

    return _ComplexMap(
        mda.Universe(complex_prm7),
        mda.Universe(fap_inputs["target"]["prm7"]),
        mda.Universe(fap_inputs["receptor"]["prm7"]),
        has_glue=False,
    )


def test_atom_map_on_ter_split_dry(bss, fap_inputs, tmp_path):
    complex_prm7 = _assemble(
        bss, fap_inputs, with_waters=False, out_prefix=tmp_path / "dry"
    )
    # construction verifies both protein blocks appear verbatim in the complex
    cmap = _complex_map(complex_prm7, fap_inputs)
    assert len(cmap.resolve("receptor", "name CA")) == 107
    assert len(cmap.resolve("target", "name CA")) == 95
    # receptor is the first block; the target follows it contiguously
    assert cmap.offset("receptor") == 0
    assert cmap.offset("target") == RECEPTOR_ATOMS


def test_crystal_waters_do_not_shift_protein_blocks(bss, fap_inputs, tmp_path):
    dry = _complex_map(
        _assemble(bss, fap_inputs, with_waters=False, out_prefix=tmp_path / "dry"),
        fap_inputs,
    )
    wet = _complex_map(
        _assemble(bss, fap_inputs, with_waters=True, out_prefix=tmp_path / "wet"),
        fap_inputs,
    )
    # appending crystal waters last leaves every protein offset + mapping identical
    assert wet.offset("receptor") == dry.offset("receptor")
    assert wet.offset("target") == dry.offset("target")
    assert wet.resolve("receptor", "name CA") == dry.resolve("receptor", "name CA")
    assert wet.resolve("target", "name CA") == dry.resolve("target", "name CA")
