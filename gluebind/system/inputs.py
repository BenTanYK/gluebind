"""Input loading and multi-molecule bookkeeping.

Proteins are supplied already parameterised (prm7/rst7). A chain-split protein
(e.g. BRD4's tandem bromodomains) is loaded by BioSimSpace as *several*
molecules, so the assembled complex must track which molecule indices belong to
each logical component. :func:`compute_layout` does that bookkeeping (pure), and
the loaders wrap the BioSimSpace calls (imported lazily).
"""

from __future__ import annotations

import dataclasses
import pathlib

GLUE_RESNAME = "MOL"
"""The glue/ligand residue must carry this name. gluebind resolves the glue by
residue name throughout (e.g. ``resname MOL``), so a different name would silently
miss it — hence :func:`validate_glue_resname` rejects anything else."""


def validate_glue_resname(resnames) -> None:
    """Raise unless every glue residue is named :data:`GLUE_RESNAME` (``MOL``)."""
    names = {str(r) for r in resnames}
    if names != {GLUE_RESNAME}:
        raise ValueError(
            f"the glue residue must be named {GLUE_RESNAME!r}, but found "
            f"{sorted(names) or 'no residues'}. Rename the residue to "
            f"{GLUE_RESNAME!r} in the input SDF."
        )


WATER_RESNAMES = frozenset(
    {"WAT", "HOH", "H2O", "SOL", "TIP3", "TIP4", "TIP5", "T3P", "T4P", "SPC", "OPC"}
)
"""Residue names accepted in the optional crystal-water input. Kept permissive to
cover common AMBER/PDB/GROMACS conventions; :func:`validate_waters_resnames`
rejects anything else so a mis-supplied protein/ion file fails loudly."""


def validate_waters_resnames(resnames) -> None:
    """Raise unless every residue in the crystal-water input is a recognised water.

    The waters input must contain *only* water — proteins and ions belong in their
    own inputs — so that appending it as the last complex component cannot perturb
    the protein/glue atom blocks the restraint atom map anchors on.
    """
    names = {str(r) for r in resnames}
    if not names:
        raise ValueError("the waters input contains no residues")
    non_water = sorted(n for n in names if n.upper() not in WATER_RESNAMES)
    if non_water:
        raise ValueError(
            f"the waters input must contain only crystal-water residues, but found "
            f"{non_water}. Keep proteins and ions in their own inputs; water may be "
            f"named any of {sorted(WATER_RESNAMES)}."
        )


@dataclasses.dataclass(frozen=True)
class ComponentLayout:
    """Which molecule indices in the assembled system belong to each component.

    Assembly order is **glue (if present), then receptor, then target** — so the
    glue (residue ``MOL``) is always molecule 0. ``target``/``receptor`` are lists
    because a chain-split protein contributes multiple molecules.
    """

    target: list[int]
    receptor: list[int]
    glue: int | None

    @property
    def n_molecules(self) -> int:
        return len(self.target) + len(self.receptor) + (0 if self.glue is None else 1)


def compute_layout(n_target: int, n_receptor: int, has_glue: bool) -> ComponentLayout:
    """Molecule-index layout for ``[glue] + receptor + target`` assembly order."""
    if n_target < 1 or n_receptor < 1:
        raise ValueError("target and receptor must each contribute >= 1 molecule")
    glue_count = 1 if has_glue else 0
    glue = 0 if has_glue else None
    receptor = list(range(glue_count, glue_count + n_receptor))
    target = list(range(glue_count + n_receptor, glue_count + n_receptor + n_target))
    return ComponentLayout(target=target, receptor=receptor, glue=glue)


def load_system(prm7: str | pathlib.Path, rst7: str | pathlib.Path):
    """Load a pre-parameterised protein from an AMBER prm7/rst7 pair (BSS System)."""
    import BioSimSpace as BSS

    return BSS.IO.readMolecules([str(prm7), str(rst7)])


def load_glue(sdf: str | pathlib.Path):
    """Load the glue small molecule from an SDF (unparameterised BSS molecule).

    Enforces the :data:`GLUE_RESNAME` (``MOL``) convention up front, so a
    mis-named glue residue fails at load rather than silently disappearing from
    every ``resname MOL`` selection downstream.
    """
    import BioSimSpace as BSS

    molecule = BSS.IO.readMolecules(str(sdf))[0]
    validate_glue_resname([res.name() for res in molecule.getResidues()])
    return molecule


def load_waters(prm7: str | pathlib.Path, rst7: str | pathlib.Path):
    """Load the optional crystal-water topology, validating it is water-only.

    Returns a BioSimSpace system of water molecules, to be appended as the last
    complex component (see :func:`gluebind.system.prep.assemble_and_solvate`).
    Enforces :func:`validate_waters_resnames` up front so a mis-supplied
    protein/ion file fails at load rather than silently shifting atom indices.
    """
    system = load_system(prm7, rst7)
    validate_waters_resnames(
        res.name() for mol in system.getMolecules() for res in mol.getResidues()
    )
    return system


def count_molecules(system) -> int:
    """Number of molecules in a BioSimSpace system (>= 2 for a chain-split protein)."""
    return system.nMolecules()
