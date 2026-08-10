"""
Generate the mol2 file from sdf using BioSimSpace to avoid
running the lengthy partial charge calculation during tests.
"""

import BioSimSpace as BSS

if __name__ == "__main__":
    ligand = BSS.IO.readMolecules("glue.sdf")[0]
    process = BSS.Parameters.gaff2(ligand)
    ligand = process.getMolecule()
    BSS.IO.saveMolecules("glue", ligand, "mol2")
