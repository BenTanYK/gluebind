"""Integration check for BioSimSpace/AmberTools ligand parameterisation."""

import pytest

pytestmark = pytest.mark.integration


def test_biosimspace_parameterises_lenalidomide_sdf(
    bss, lenalidomide_sdf, tmp_path
):
    ligand = bss.IO.readMolecules(str(lenalidomide_sdf))[0]
    process = bss.Parameters.gaff2(ligand, work_dir=str(tmp_path))
    parameterised = process.getMolecule()

    assert parameterised.nAtoms() == 32
    assert parameterised.nResidues() == 1
    assert parameterised.getResidues()[0].name() == "MOL"
