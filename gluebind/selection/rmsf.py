"""RMSF over the equilibration trajectory + local-minimum candidate detection.

``compute_rmsf`` wraps the MDAnalysis align+RMSF workflow (as in the template's
``analyse_RMSF.py``) and is integration-verified; ``local_minima`` (pure) turns
an RMSF profile into the candidate pool of stable residues.
"""

from __future__ import annotations

import numpy as np


def compute_rmsf(universe, selection: str = "name CA"):
    """Cα RMSF over a trajectory. Returns ``(resids, rmsf_values)``.

    Aligns the trajectory to its average structure, then computes per-atom RMSF
    for the selection (MDAnalysis).
    """
    from MDAnalysis.analysis import align, rms

    average = align.AverageStructure(
        universe, universe, select=selection, ref_frame=0
    ).run()
    align.AlignTraj(
        universe, average.results.universe, select=selection, in_memory=True
    ).run()
    atoms = universe.select_atoms(selection)
    rmsf = rms.RMSF(atoms).run()
    return atoms.resids, np.asarray(rmsf.results.rmsf)


def local_minima(values, *, order: int = 1) -> list[int]:
    """Indices of local minima of ``values`` (strictly below neighbours within
    ``order``)."""
    values = np.asarray(values, float)
    minima: list[int] = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if (
            values[i] == window.min()
            and values[i] < values[i - 1]
            and values[i] < values[i + 1]
        ):
            minima.append(i)
    return minima


def stablest_candidates(resids, rmsf_values, *, top_n: int = 8) -> list[int]:
    """Residue ids of the ``top_n`` lowest-RMSF local minima (most stable)."""
    resids = np.asarray(resids)
    rmsf_values = np.asarray(rmsf_values, float)
    minima = local_minima(rmsf_values)
    minima.sort(key=lambda i: rmsf_values[i])
    return [int(resids[i]) for i in minima[:top_n]]
