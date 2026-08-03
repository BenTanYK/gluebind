"""OpenMM restraint/CV force builders — the single source of the geometry.

The three template ``run_window.py`` scripts each rebuilt (and copy-pasted) the
Boresch/RMSD/distance forces inline; here they live once:

* :mod:`gluebind.restraints.boresch` — the 5-DoF Boresch geometry, fixed
  restraints, and biased CVs;
* :mod:`gluebind.restraints.rmsd` — harmonic RMSD restraints (fixed or moving);
* :mod:`gluebind.restraints.separation` — the interface-CoM distance CV + bias;
* :mod:`gluebind.restraints.system_builder` — shared system setup, heating and
  the CV sampling loop.

OpenMM is imported when these modules load, so import them explicitly (they are
not pulled in by ``import gluebind``).
"""

from __future__ import annotations

from gluebind.restraints import boresch, rmsd, separation, system_builder

__all__ = ["boresch", "rmsd", "separation", "system_builder"]
