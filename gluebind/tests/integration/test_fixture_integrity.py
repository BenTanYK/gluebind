"""Integration tier: the vendored 1FAP fixture is well-formed and loadable.

Needs only MDAnalysis/RDKit (no BioSimSpace/GPU), so it is the tier's canary — it
goes green as soon as the tier is wired and pins the fixture shape the heavier
assembly tests rely on.
"""

import pytest

pytestmark = pytest.mark.integration


def test_protein_inputs_load_and_are_dry(fap_inputs):
    import MDAnalysis as mda

    for role, expected_ca in (("receptor", 107), ("target", 95)):
        p = fap_inputs[role]
        u = mda.Universe(
            p["prm7"], p["rst7"], format="RESTRT", topology_format="PRMTOP"
        )
        assert len(u.select_atoms("name CA")) == expected_ca
        # crystal waters live in their own input, not the protein topologies
        assert len(u.select_atoms("resname WAT HOH")) == 0


def test_glue_is_named_mol_rapamycin(fap_inputs):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    m = Chem.MolFromMolFile(fap_inputs["glue"]["sdf"])
    assert m.GetProp("_Name") == "MOL"
    assert rdMolDescriptors.CalcMolFormula(m) == "C51H79NO13"


def test_waters_are_water_only(fap_inputs):
    import MDAnalysis as mda

    w = fap_inputs["waters"]
    u = mda.Universe(w["prm7"], w["rst7"], format="RESTRT", topology_format="PRMTOP")
    assert set(u.residues.resnames) == {"WAT"}
    assert len(u.residues) == 23
    assert len(u.atoms) == 69  # 23 TIP3P waters
