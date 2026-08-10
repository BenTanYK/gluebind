"""Integration tier: the vendored 1FAP fixture is well-formed and loadable.

Needs only MDAnalysis/RDKit (no BioSimSpace/GPU), so it is the tier's canary — it
goes green as soon as the tier is wired and pins the fixture shape the heavier
assembly tests rely on.
"""

import pytest

pytestmark = pytest.mark.integration


def test_protein_inputs_load_and_are_dry(fap_inputs):
    from gluebind.system.mdanalysis import load_amber_universe

    for role, expected_ca in (("receptor", 107), ("target", 95)):
        p = fap_inputs[role]
        u = load_amber_universe(p["prm7"], p["rst7"])
        assert len(u.select_atoms("name CA")) == expected_ca
        # crystal waters live in their own input, not the protein topologies
        assert len(u.select_atoms("resname WAT HOH")) == 0


def test_glue_is_named_mol_rapamycin(fap_inputs):
    # rdkit isn't a declared dep (it arrives transitively via BSS), so guard it —
    # the protein/water canary tests use MDAnalysis only and still run without it.
    pytest.importorskip("rdkit.Chem")
    from rdkit import Chem

    m = Chem.MolFromMol2File(fap_inputs["glue"]["mol2"], removeHs=False)
    assert m is not None
    assert m.GetProp("_Name") == "MOL"
    assert m.GetNumAtoms() == 144
    assert all(
        atom.HasProp("_TriposPartialCharge")
        for atom in m.GetAtoms()
    )


def test_waters_are_water_only(fap_inputs):
    from gluebind.system.mdanalysis import load_amber_universe

    w = fap_inputs["waters"]
    u = load_amber_universe(w["prm7"], w["rst7"])
    assert set(u.residues.resnames) == {"WAT"}
    assert len(u.residues) == 23
    assert len(u.atoms) == 69  # 23 TIP3P waters
