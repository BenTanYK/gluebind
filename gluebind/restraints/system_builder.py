"""Shared OpenMM system construction, equilibration and CV sampling.

The three template ``run_window.py`` scripts repeat the same setup idiom
(``createSystem`` with PME + HMR + HBonds, a Langevin-middle integrator, energy
minimisation, a stepped heating ramp) and the same sampling loop. Those live
here once. OpenMM is imported at module load, so this module (and the rest of
``gluebind.restraints``) is only imported when a window is actually run.
"""

from __future__ import annotations

import pathlib

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

INITIAL_TEMPERATURE_K = 6.0
HEATING_INCREMENTS = 50
HEATING_STEPS_PER_INCREMENT = 1000
FRICTION_PER_PS = 1.0


def build_system(prmtop_path, *, hmr_factor: float = 1.5, pme_cutoff_nm: float = 1.0):
    """Load an AMBER prmtop and create the OpenMM ``System``."""
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        hydrogenMass=hmr_factor * unit.amu,
        nonbondedCutoff=pme_cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
    )
    return prmtop, system


def build_simulation(prmtop, system, *, timestep_fs: float, platform=None):
    """Create a ``Simulation`` with a Langevin-middle integrator at 6 K."""
    integrator = mm.LangevinMiddleIntegrator(
        INITIAL_TEMPERATURE_K * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    if platform is None:
        simulation = app.Simulation(prmtop.topology, system, integrator)
    else:
        simulation = app.Simulation(prmtop.topology, system, integrator, platform)
    return simulation, integrator


def heating_schedule(
    target_temperature_K: float, increments: int = HEATING_INCREMENTS
) -> list[float]:
    """The stepped heating ramp temperatures (K), from one increment up to the
    target inclusive — ``increments`` steps of ``target/increments`` each.

    Matches the template's ramp (300 K in 50 × 6 K steps). Starts at the first
    increment (not the second), so no step of the ramp is skipped.
    """
    step = target_temperature_K / increments
    return [(i + 1) * step for i in range(increments)]


def minimise_and_heat(simulation, integrator, *, target_temperature_K: float) -> None:
    """Minimise, then ramp the temperature to ``target_temperature_K``.

    Uses :func:`heating_schedule` so the ramp is derived from the target and no
    increment is skipped.
    """
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(INITIAL_TEMPERATURE_K * unit.kelvin)
    for temperature in heating_schedule(target_temperature_K):
        integrator.setTemperature(temperature * unit.kelvin)
        simulation.step(HEATING_STEPS_PER_INCREMENT)
    integrator.setTemperature(target_temperature_K * unit.kelvin)


def glue_heavy_atoms(topology, resname: str = "MOL") -> list[int]:
    """Indices of the glue's heavy atoms (residue ``resname``, non-hydrogen)."""
    return [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name == resname and not atom.name.startswith("H")
    ]


def atoms_in_residues(topology, residue_indices, atom_names) -> list[int]:
    """Indices of atoms in the given (0-indexed) residues whose name is selected."""
    residue_indices = set(residue_indices)
    atom_names = set(atom_names)
    return [
        atom.index
        for atom in topology.atoms()
        if atom.residue.index in residue_indices and atom.name in atom_names
    ]


def collect_cv_samples(
    simulation, bias_force, *, equil_steps: int, sampling_steps: int, record_steps: int
) -> np.ndarray:
    """Equilibrate, then sample the biased CV every ``record_steps`` steps.

    Returns an ``(n_samples, 2)`` array of ``[sample_index, cv_value]`` — the
    same format the template writes and WHAM consumes.
    """
    if equil_steps > 0:
        simulation.step(equil_steps)
    n_samples = sampling_steps // record_steps
    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        simulation.step(record_steps)
        value = bias_force.getCollectiveVariableValues(simulation.context)[0]
        samples[i] = [i, value]
    return samples


def load_coordinates(path):
    """Load an AMBER rst7/inpcrd; return ``(positions, box_vectors)``."""
    inpcrd = app.AmberInpcrdFile(str(pathlib.Path(path)))
    return inpcrd.positions, inpcrd.boxVectors


def save_rst7(prmtop_path, positions, box_vectors, out_path) -> None:
    """Write an AMBER rst7 (positions + box) that a later run can reload."""
    import parmed

    structure = parmed.load_file(str(prmtop_path))
    structure.positions = positions
    structure.box_vectors = box_vectors
    structure.save(str(out_path), format="rst7", overwrite=True)
