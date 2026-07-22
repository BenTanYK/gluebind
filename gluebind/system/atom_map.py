"""Robust atom-index mapping from the user's input topologies to the assembled complex.

The user writes restraint selection strings against their input ``.prm7``/``.rst7``
topologies. BioSimSpace, when it assembles and writes the complex, may split a
``TER``-containing protein into several molecules and/or renumber residues — so the
complex's atom/residue indexing can differ from the inputs. Resolving a selection
directly against the complex (and trusting its residue numbers) can therefore apply
a restraint to the *wrong* atoms.

This module maps input-topology atom indices to complex atom indices by anchoring
on the one thing BSS preserves — the **per-molecule atom order** — and *verifying*
it atom-by-atom. Residue numbers are never compared (BSS may renumber; that's fine),
and atom **order** within a molecule is checked, so a selection is either mapped to
exactly the atoms the user meant or the mapping fails loudly.

The functions here are pure (they operate on lists of ``(atom_name, element)``
keys); the thin MDAnalysis extraction that produces those keys lives with the
resolver in :mod:`gluebind.spec_builder`.
"""

from __future__ import annotations

from collections.abc import Sequence

AtomKey = tuple[str, str]
"""A per-atom identity used for verification: ``(atom_name, element)``. Deliberately
excludes residue number (BSS may renumber) and residue name (kept simple; atom name
+ element is a strong per-position fingerprint within an ordered molecule)."""


def component_offsets(sizes: Sequence[tuple[str, int]]) -> dict[str, int]:
    """First-atom index of each component in a contiguous assembly, in order.

    ``sizes`` is the ordered ``(component_name, n_atoms)`` sequence of the assembly
    (for gluebind: glue, then receptor, then target). Returns
    ``{component_name: offset}``.
    """
    offsets: dict[str, int] = {}
    cursor = 0
    for name, n_atoms in sizes:
        if n_atoms < 0:
            raise ValueError(f"component {name!r} has negative atom count {n_atoms}")
        offsets[name] = cursor
        cursor += n_atoms
    return offsets


def verify_block(
    component: str,
    input_keys: Sequence[AtomKey],
    complex_keys: Sequence[AtomKey],
    offset: int,
) -> None:
    """Verify the input molecule appears verbatim at ``offset`` in the complex.

    Checks that ``complex_keys[offset : offset + len(input_keys)]`` equals
    ``input_keys`` identity-for-identity (atom name + element, *not* residue number).
    Raises :class:`ValueError` naming the first divergent atom otherwise — meaning
    BSS reordered atoms within the molecule and selections cannot be mapped safely.
    """
    n = len(input_keys)
    end = offset + n
    if end > len(complex_keys):
        raise ValueError(
            f"atom map: {component!r} block [{offset}, {end}) runs past the complex "
            f"({len(complex_keys)} atoms) — the assembled complex is smaller than the "
            "inputs; check the assembly order/contents."
        )
    for i in range(n):
        if input_keys[i] != complex_keys[offset + i]:
            raise ValueError(
                f"atom-map verification failed for {component!r} at input atom {i} "
                f"(input {tuple(input_keys[i])}, complex atom {offset + i} "
                f"{tuple(complex_keys[offset + i])}). BioSimSpace changed the "
                "per-molecule atom order, so restraint selections cannot be mapped to "
                "the complex safely."
            )


def map_indices(input_indices: Sequence[int], offset: int) -> list[int]:
    """Translate input-topology atom indices to complex indices for a verified block."""
    return [offset + int(i) for i in input_indices]
