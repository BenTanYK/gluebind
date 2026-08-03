"""Secondary-structure assignment for anchor filtering (MDAnalysis DSSP).

Boresch anchors should sit in structured regions (α-helix / β-sheet), not
flexible loops. ``secondary_structure`` wraps MDAnalysis DSSP (integration-
verified); ``is_structured`` (pure) classifies a DSSP code.
"""

from __future__ import annotations

# DSSP one-letter codes considered "structured": helices (H/G/I) and sheets (E/B).
STRUCTURED_CODES = frozenset({"H", "G", "I", "E", "B"})


def is_structured(code: str) -> bool:
    """True if a DSSP code denotes an α-helix or β-sheet residue."""
    return code in STRUCTURED_CODES


def secondary_structure(universe) -> dict[int, str]:
    """Map residue id → DSSP code (first frame), for protein residues."""
    from MDAnalysis.analysis.dssp import DSSP

    result = DSSP(universe).run()
    codes = result.results.dssp[0]  # (n_residues,) one-letter codes, first frame
    resids = universe.select_atoms("protein and name CA").resids
    return {int(resid): str(code) for resid, code in zip(resids, codes, strict=False)}


def structured_residues(universe) -> list[int]:
    """Residue ids in α-helix / β-sheet."""
    return [
        resid
        for resid, code in secondary_structure(universe).items()
        if is_structured(code)
    ]
