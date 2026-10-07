"""Unit tests for the per-window heating helper."""

from unittest.mock import Mock

import numpy as np
import pytest

from gluebind.restraints import system_builder as sb


def test_minimise_and_heat_minimises_before_exact_heating_steps():
    events = []

    simulation = Mock()
    simulation.minimizeEnergy.side_effect = lambda: events.append("minimise")
    simulation.context.setVelocitiesToTemperature.side_effect = lambda *_: (
        events.append("velocities")
    )
    simulation.step.side_effect = lambda steps: events.append(("step", steps))

    integrator = Mock()
    integrator.setTemperature.side_effect = lambda *_: events.append("temperature")

    sb.minimise_and_heat(
        simulation,
        integrator,
        target_temperature_K=300.0,
        heating_steps=103,
    )

    assert events[0] == "minimise"
    assert events[1] == "velocities"
    step_events = [event for event in events if isinstance(event, tuple)]
    assert len(step_events) == sb.HEATING_INCREMENTS
    assert sum(event[1] for event in step_events) == 103
    assert step_events[-1] == ("step", 5)


# -- RMSD reference positions ---------------------------------------------------


def _nm(rows):
    import openmm as mm
    import openmm.unit as unit

    return [mm.Vec3(*r) for r in rows] * unit.nanometer


def test_reference_positions_places_mapped_reference_coordinates():
    import openmm.unit as unit

    source = _nm([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])  # e.g. the complex
    ref = sb.reference_positions(5, [0, 4], [3, 1], source)  # e.g. a bulk system
    xyz = np.asarray(ref.value_in_unit(unit.nanometer))
    assert xyz.shape == (5, 3)
    assert xyz[0].tolist() == [3, 3, 3] and xyz[4].tolist() == [1, 1, 1]


@pytest.mark.parametrize(
    "atoms, reference_atoms, match",
    [
        ([0, 1], [0], "reference atoms"),
        ([0], [9], "outside the reference"),
        ([7], [0], "outside the system"),
    ],
)
def test_reference_positions_rejects_bad_mappings(atoms, reference_atoms, match):
    source = _nm([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match=match):
        sb.reference_positions(3, atoms, reference_atoms, source)


# -- solute joining (periodic re-imaging) --------------------------------------

BOX_NM = 3.0


def _toy_system(layout):
    """Build a toy (topology, system, positions) from ``layout``.

    Each entry is ``("chain", n_atoms)`` (a bonded carbon chain, 0.15 nm
    spacing along x), ``("water", resname)`` or ``("ion", symbol)``. Molecules
    are placed 0.5 nm apart along y. Bonds go into both the ``Topology`` (which
    gluebind reads) and a ``HarmonicBondForce`` (which OpenMM uses to define
    molecules), mirroring ``createSystem``.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    a = mm.Vec3(BOX_NM, 0, 0)
    b = mm.Vec3(0, BOX_NM, 0)
    c = mm.Vec3(0, 0, BOX_NM)
    system.setDefaultPeriodicBoxVectors(a, b, c)
    topology.setPeriodicBoxVectors((a, b, c))
    bonds = mm.HarmonicBondForce()
    positions = []

    def add(residue, name, element, xyz):
        atom = topology.addAtom(name, element, residue)
        system.addParticle(element.mass if element is not None else 0.0)
        positions.append(mm.Vec3(*xyz))
        return atom

    for m, entry in enumerate(layout):
        y = 0.2 + 0.5 * m
        if entry[0] == "chain":
            residue = topology.addResidue("ALA", chain)
            atoms = [
                add(residue, f"C{i}", app.element.carbon, (0.2 + 0.15 * i, y, 1.5))
                for i in range(entry[1])
            ]
            for x, z in zip(atoms, atoms[1:], strict=False):
                topology.addBond(x, z)
                bonds.addBond(x.index, z.index, 0.15, 1000.0)
        elif entry[0] == "water":
            residue = topology.addResidue(entry[1], chain)
            o = add(residue, "O", app.element.oxygen, (0.2, y, 1.5))
            for dx in (0.1, -0.1):
                h = add(residue, "H", app.element.hydrogen, (0.2 + dx, y + 0.05, 1.5))
                topology.addBond(o, h)
                bonds.addBond(o.index, h.index, 0.1, 1000.0)
        else:
            residue = topology.addResidue(entry[1], chain)
            element = app.element.Element.getBySymbol(entry[1])
            add(residue, entry[1], element, (0.2, y, 1.5))
    system.addForce(bonds)
    return topology, system, positions * unit.nanometer


def _context(system, positions):
    import openmm as mm

    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions)
    return context


def test_topology_molecules_follow_bonds():
    topology, _, _ = _toy_system([("chain", 3), ("chain", 2), ("ion", "Na")])
    assert sb.topology_molecules(topology) == [[0, 1, 2], [3, 4], [5]]


def test_solute_molecules_excludes_waters_and_solvent_ions_by_composition():
    # glue-like chain, receptor split by a TER break, a structural Zn in the
    # protein block, then a crystal water with an unusual residue name, a target
    # chain after it, solvent water and solvent ions.
    topology, _, _ = _toy_system(
        [
            ("chain", 4),  # 0-3   glue
            ("chain", 3),  # 4-6   receptor chain A
            ("chain", 3),  # 7-9   receptor chain B (TER break)
            ("ion", "Zn"),  # 10   structural ion (before any water)
            ("water", "XWT"),  # 11-13 crystal water, non-standard name
            ("chain", 2),  # 14-15 target
            ("water", "WAT"),  # 16-18 solvent
            ("ion", "Na"),  # 19   solvent ion
            ("ion", "Cl"),  # 20   solvent ion
        ]
    )
    assert sb.solute_molecules(topology) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10],
        [14, 15],
    ]


def test_join_molecules_makes_solute_one_openmm_molecule_without_changing_energy():
    import openmm as mm
    import openmm.unit as unit

    topology, system, positions = _toy_system(
        [("chain", 3), ("chain", 3), ("water", "WAT"), ("ion", "Na")]
    )
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.0)
    for _ in range(system.getNumParticles()):
        nonbonded.addParticle(0.1, 0.3, 0.5)
    system.addForce(nonbonded)

    def energy():
        state = _context(system, positions).getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    before = energy()
    solute = sb.solute_molecules(topology)
    force = sb.join_molecules(system, solute)

    assert force is not None and force.getNumBonds() == len(solute) - 1 == 1
    molecules = [sorted(m) for m in _context(system, positions).getMolecules()]
    assert [0, 1, 2, 3, 4, 5] in molecules  # both chains are one molecule
    assert [6, 7, 8] in molecules and [9] in molecules  # solvent untouched
    assert energy() == pytest.approx(before, abs=1e-9)


def test_joined_solute_is_reimaged_as_one_unit():
    """The failure mode itself: per-molecule wrapping splits an unjoined
    two-chain protein whose chains straddle a box face; joined, it stays whole."""
    import openmm as mm
    import openmm.unit as unit

    topology, system, positions = _toy_system([("chain", 3), ("chain", 3)])
    # Chain A just inside the +x face, chain B just outside it: one protein.
    shifted = [
        mm.Vec3(p.x + (2.6 if i < 3 else 2.75), p.y, p.z)
        for i, p in enumerate(positions.value_in_unit(unit.nanometer))
    ] * unit.nanometer

    def wrapped_gap():
        state = _context(system, shifted).getState(
            getPositions=True, enforcePeriodicBox=True
        )
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return abs(xyz[3:, 0].mean() - xyz[:3, 0].mean())

    true_gap = abs(
        np.mean([p.x for p in shifted[3:]]) - np.mean([p.x for p in shifted[:3]])
    )
    # Unjoined: chain B's centre is outside the box, so only it is wrapped back.
    assert wrapped_gap() == pytest.approx(BOX_NM - true_gap, abs=1e-6)
    sb.join_molecules(system, sb.solute_molecules(topology))
    assert wrapped_gap() == pytest.approx(true_gap, abs=1e-6)


def test_join_molecules_is_a_no_op_for_a_single_molecule():
    topology, system, _ = _toy_system([("chain", 3), ("water", "WAT")])
    n_forces = system.getNumForces()
    assert sb.join_molecules(system, sb.solute_molecules(topology)) is None
    assert system.getNumForces() == n_forces


def test_build_system_joins_the_solute(monkeypatch):
    topology, system, _ = _toy_system(
        [("chain", 3), ("chain", 3), ("water", "WAT"), ("ion", "Cl")]
    )

    class _FakePrmtop:
        def __init__(self, path):
            self.topology = topology

        def createSystem(self, **kwargs):
            return system

    monkeypatch.setattr(sb.app, "AmberPrmtopFile", _FakePrmtop)
    _, built = sb.build_system("complex.prm7")

    joins = [f for f in built.getForces() if f.getName() == "gluebind_solute_join"]
    assert len(joins) == 1 and joins[0].getNumBonds() == 1
    assert joins[0].getBondParameters(0)[:2] == [0, 3]
