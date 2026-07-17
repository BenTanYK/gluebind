"""Interface detection between two proteins (pure numpy).

Given the Cα coordinates of the two proteins, find the residues that form the
interface (any Cα–Cα pair within a cutoff). Used to build the Boresch bonded
anchors (interface centroids) and, by default, to focus restraints on the
interface. Kept pure so it is unit-tested; the MDAnalysis coordinate/index
extraction lives in the resolver.
"""

from __future__ import annotations

import numpy as np


def interface_residues(rec_ca_coords, lig_ca_coords, *, cutoff: float = 12.0):
    """Indices (into each input array) of residues with a Cα–Cα pair within ``cutoff``.

    Returns ``(rec_indices, lig_indices)`` — sorted lists of positions in
    ``rec_ca_coords`` / ``lig_ca_coords`` respectively (Å).
    """
    rec = np.asarray(rec_ca_coords, float)
    lig = np.asarray(lig_ca_coords, float)
    if rec.ndim != 2 or lig.ndim != 2:
        raise ValueError("coordinates must be (n_residues, 3) arrays")
    dist = np.linalg.norm(rec[:, None, :] - lig[None, :, :], axis=-1)
    close = dist <= cutoff
    rec_idx = sorted({int(i) for i in np.where(close.any(axis=1))[0]})
    lig_idx = sorted({int(j) for j in np.where(close.any(axis=0))[0]})
    return rec_idx, lig_idx
