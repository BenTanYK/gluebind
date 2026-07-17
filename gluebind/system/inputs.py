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


@dataclasses.dataclass(frozen=True)
class ComponentLayout:
    """Which molecule indices in the assembled system belong to each component.

    Assembly order is target molecules, then receptor molecules, then the glue
    (if present). ``target``/``receptor`` are lists because a chain-split protein
    contributes multiple molecules.
    """

    target: list[int]
    receptor: list[int]
    glue: int | None

    @property
    def n_molecules(self) -> int:
        return len(self.target) + len(self.receptor) + (0 if self.glue is None else 1)


def compute_layout(n_target: int, n_receptor: int, has_glue: bool) -> ComponentLayout:
    """Molecule-index layout for ``target + receptor + [glue]`` assembly order."""
    if n_target < 1 or n_receptor < 1:
        raise ValueError("target and receptor must each contribute >= 1 molecule")
    target = list(range(0, n_target))
    receptor = list(range(n_target, n_target + n_receptor))
    glue = (n_target + n_receptor) if has_glue else None
    return ComponentLayout(target=target, receptor=receptor, glue=glue)


def load_system(prm7: str | pathlib.Path, rst7: str | pathlib.Path):
    """Load a pre-parameterised protein from an AMBER prm7/rst7 pair (BSS System)."""
    import BioSimSpace as BSS

    return BSS.IO.readMolecules([str(prm7), str(rst7)])


def load_glue(sdf: str | pathlib.Path):
    """Load the glue small molecule from an SDF (unparameterised BSS molecule)."""
    import BioSimSpace as BSS

    return BSS.IO.readMolecules(str(sdf))[0]


def count_molecules(system) -> int:
    """Number of molecules in a BioSimSpace system (>= 2 for a chain-split protein)."""
    return system.nMolecules()
