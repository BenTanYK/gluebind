"""Analysis: WHAM PMFs and geometric-route free-energy assembly.

Mirrors the ``US_protocol_template`` analysis (geometric route + umbrella
sampling + WHAM), deliberately not a3fe's alchemical MBAR. WHAM turns each
stage's window histograms into a PMF (:mod:`gluebind.analysis.wham`); the PMFs
are averaged across replicates (:mod:`gluebind.analysis.pmf`); and the
geometric-route integrals (:mod:`gluebind.analysis.free_energy`) turn each PMF
into a contribution, summed into the standard-state binding free energy.
"""

from __future__ import annotations

from gluebind.analysis.free_energy import (
    binding_free_energy,
    boresch_contribution,
    combine_errors,
    integrands,
    rmsd_contribution,
    separation_contribution,
    standard_state_correction,
)
from gluebind.analysis.pmf import average_pmfs, detect_equilibration, pmf_minimum
from gluebind.analysis.provider import WhamPmfProvider, wham_units
from gluebind.analysis.wham import (
    build_wham_command,
    load_pmf,
    run_wham,
    write_metafile,
)

__all__ = [
    "rmsd_contribution",
    "boresch_contribution",
    "separation_contribution",
    "standard_state_correction",
    "integrands",
    "binding_free_energy",
    "combine_errors",
    "average_pmfs",
    "detect_equilibration",
    "pmf_minimum",
    "write_metafile",
    "build_wham_command",
    "run_wham",
    "load_pmf",
    "WhamPmfProvider",
    "wham_units",
]
