"""MDAnalysis compatibility helpers for gluebind's AMBER artifacts.

gluebind uses BioSimSpace's ``.prm7`` filename convention.  MDAnalysis can
parse the underlying AMBER topology, but does not infer its parser from that
suffix, so every load must explicitly select the ``PARM7`` topology format.
Keeping that decision here prevents individual call sites from depending on
MDAnalysis filename inference.
"""

from __future__ import annotations

import pathlib

_AMBER_RESTART_SUFFIXES = {".rst7", ".inpcrd", ".restrt"}


def load_amber_universe(topology, coordinates=None):
    """Return an MDAnalysis Universe for a gluebind AMBER topology.

    ``topology`` is always parsed as ``PARM7``.  AMBER restart coordinates need
    an explicit ``RESTRT`` reader; trajectory formats such as DCD are left to
    MDAnalysis' normal coordinate-format detection.

    MDAnalysis is imported lazily so importing :mod:`gluebind.system` remains
    lightweight in environments that only use configuration or analysis code.
    """
    import MDAnalysis as mda

    topology = str(topology)
    kwargs = {"topology_format": "PARM7"}
    if coordinates is None:
        return mda.Universe(topology, **kwargs)

    coordinates = str(coordinates)
    if pathlib.Path(coordinates).suffix.lower() in _AMBER_RESTART_SUFFIXES:
        kwargs["format"] = "RESTRT"
    return mda.Universe(topology, coordinates, **kwargs)
