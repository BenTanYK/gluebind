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
    """Load an AMBER prmtop and create the OpenMM ``System``.

    The solute (every receptor/target chain, the glue and any structural ions;
    see :func:`solute_molecules`) is joined into a single OpenMM molecule so that
    periodic re-imaging can never separate its parts (:func:`join_molecules`).
    """
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        hydrogenMass=hmr_factor * unit.amu,  # ty: ignore[unsupported-operator]
        nonbondedCutoff=pme_cutoff_nm * unit.nanometer,  # ty: ignore[unresolved-attribute]
        constraints=app.HBonds,
    )
    solute = solute_molecules(prmtop.topology)
    if join_molecules(system, solute) is not None:
        print(
            f"gluebind: joined {len(solute)} solute molecules "
            f"({sum(len(m) for m in solute)} atoms) into one periodic-imaging unit",
            flush=True,
        )
    return prmtop, system


def topology_molecules(topology) -> list[list[int]]:
    """Atom indices of each covalently connected molecule, in topology order.

    The same connectivity OpenMM uses to define molecules for periodic
    re-imaging (bonds; constraints are a subset of them), computed by union-find.
    Molecules are ordered by their first atom.
    """
    n_atoms = topology.getNumAtoms()
    parent = list(range(n_atoms))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for bond in topology.bonds():
        root_a, root_b = find(bond[0].index), find(bond[1].index)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    molecules: dict[int, list[int]] = {}
    for i in range(n_atoms):
        molecules.setdefault(find(i), []).append(i)
    return list(molecules.values())


def _is_water(atoms) -> bool:
    """Water by composition (one O plus hydrogens / massless virtual sites), so
    crystal and solvent waters are recognised whatever their residue name."""
    elements = [atom.element for atom in atoms]
    oxygens = sum(1 for e in elements if e is not None and e.symbol == "O")
    others = sum(1 for e in elements if e is not None and e.symbol not in ("O", "H"))
    return oxygens == 1 and others == 0 and len(elements) >= 3


def solute_molecules(topology) -> list[list[int]]:
    """The molecules that must stay together under periodic re-imaging.

    Every molecule except waters and the *solvent* ions (monatomic molecules after
    the first water). Assembly places the solute — glue, receptor and target
    blocks, each possibly several molecules (chains, ``TER`` breaks) — before any
    water, so structural ions supplied inside a protein topology are kept while
    the bulk ions added by solvation are not. Defined from connectivity and
    composition, never residue or chain names.
    """
    atoms = list(topology.atoms())
    solute: list[list[int]] = []
    seen_water = False
    for molecule in topology_molecules(topology):
        if _is_water([atoms[i] for i in molecule]):
            seen_water = True
            continue
        if len(molecule) == 1 and seen_water:
            continue  # solvent ion
        solute.append(molecule)
    return solute


def join_molecules(system, molecules) -> mm.HarmonicBondForce | None:
    """Make ``molecules`` a single OpenMM molecule with zero-energy bonds.

    OpenMM re-images each molecule into the periodic box independently, and its
    molecules are defined by bond topology alone, regardless of force constant.
    Non-periodic restraint forces (``RMSDForce``, centroid CVs) that span several
    molecules would otherwise see one part jump by a box vector when re-imaged,
    e.g. a two-chain protein's RMSD spiking by ~4 nm. A ``k = 0`` bond from the
    first molecule to each other one (n−1 bonds) prevents this without changing
    the energy, forces or nonbonded exclusions. Returns the added force, or
    ``None`` when there is nothing to join.
    """
    if len(molecules) < 2:
        return None
    force = mm.HarmonicBondForce()
    if hasattr(force, "setName"):  # OpenMM >= 8.1
        force.setName("gluebind_solute_join")
    root = molecules[0][0]
    for molecule in molecules[1:]:
        force.addBond(root, molecule[0], 0.0, 0.0)
    system.addForce(force)
    return force


def build_simulation(prmtop, system, *, timestep_fs: float, platform=None):
    """Create a ``Simulation`` with a Langevin-middle integrator at 6 K."""
    integrator = mm.LangevinMiddleIntegrator(
        INITIAL_TEMPERATURE_K * unit.kelvin,  # ty: ignore[unsupported-operator]
        FRICTION_PER_PS / unit.picosecond,  # ty: ignore[unresolved-attribute]
        timestep_fs * unit.femtoseconds,  # ty: ignore[unresolved-attribute]
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


def minimise_and_set_temperature(
    simulation, integrator, *, target_temperature_K: float
) -> None:
    """Minimise, then set the integrator and velocities straight to the target
    temperature — **no heating ramp**.

    For windows that start from an already-equilibrated structure at the target
    temperature (the prep-equilibrated complex, or an SMD frame), so re-heating
    from cold each window would be wasted MD. The integrator was created at
    ``INITIAL_TEMPERATURE_K`` by :func:`build_simulation`, so it is set to the
    target here.
    """
    # No subset is supplied: OpenMM minimizes the whole system under all
    # restraints and the window bias currently installed.
    simulation.minimizeEnergy()
    integrator.setTemperature(
        target_temperature_K * unit.kelvin  # ty: ignore[unsupported-operator]
    )
    simulation.context.setVelocitiesToTemperature(
        target_temperature_K * unit.kelvin  # ty: ignore[unsupported-operator]
    )


def minimise_and_heat(
    simulation, integrator, *, target_temperature_K: float,
    heating_steps: int = HEATING_INCREMENTS * HEATING_STEPS_PER_INCREMENT,
) -> None:
    """Minimise, then ramp the temperature to ``target_temperature_K``.

    Uses :func:`heating_schedule` so the ramp is derived from the target and no
    increment is skipped.
    """
    if heating_steps < 0:
        raise ValueError("heating_steps must be >= 0")
    # Minimize the complete system with all window-specific forces active.
    # This must precede the heating ramp so it starts from a relaxed structure.
    # No subset is supplied: OpenMM minimizes the whole system under all
    # restraints and the window bias currently installed.
    simulation.minimizeEnergy()
    if heating_steps == 0:
        integrator.setTemperature(
            target_temperature_K * unit.kelvin  # ty: ignore[unsupported-operator]
        )
        simulation.context.setVelocitiesToTemperature(
            target_temperature_K * unit.kelvin  # ty: ignore[unsupported-operator]
        )
        return
    simulation.context.setVelocitiesToTemperature(
        INITIAL_TEMPERATURE_K * unit.kelvin  # ty: ignore[unsupported-operator]
    )
    increments = heating_schedule(target_temperature_K)
    base_steps, remainder = divmod(heating_steps, len(increments))
    for index, temperature in enumerate(increments):
        integrator.setTemperature(
            temperature * unit.kelvin  # ty: ignore[unsupported-operator]
        )
        steps = base_steps + (remainder if index == len(increments) - 1 else 0)
        if steps:
            simulation.step(steps)
    integrator.setTemperature(
        target_temperature_K * unit.kelvin  # ty: ignore[unsupported-operator]
    )


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
