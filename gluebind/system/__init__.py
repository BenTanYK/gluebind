"""System preparation (BioSimSpace front-end) and input handling.

:mod:`gluebind.system.inputs` loads the user's pre-parameterised proteins and
optional glue and tracks the multi-molecule bookkeeping (a chain-split protein
such as BRD4 arrives as several BioSimSpace molecules). :mod:`gluebind.system.prep`
parameterises the glue, assembles + solvates the complex, runs the
pre-equilibration/equilibration, extracts the isolated bulk species, and writes a
:class:`~gluebind.system.prep.PreparedSystem` manifest — the hand-off to Phase 4
(selection) and the runner's ``spec_builder``.

BioSimSpace is imported lazily inside the functions that need it, so importing
this subpackage stays cheap and works without BSS present.
"""

from __future__ import annotations

from gluebind.system.inputs import ComponentLayout, compute_layout
from gluebind.system.prep import (
    PreparedSystem,
    box_length,
    normalise_ff_name,
    prepare,
    validate_forcefield,
)

__all__ = [
    "ComponentLayout",
    "compute_layout",
    "PreparedSystem",
    "prepare",
    "validate_forcefield",
    "normalise_ff_name",
    "box_length",
]
