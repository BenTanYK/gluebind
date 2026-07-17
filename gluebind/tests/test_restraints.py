"""Tests for the OpenMM restraint/CV force builders on synthetic systems.

These exercise the geometry and force construction without a real prmtop: a bare
``System`` of point particles on the Reference platform, with positions chosen so
the collective variable has a known value. (End-to-end ``run_window`` on real
structures is validated in a later development phase.)
"""

import math

import pytest

pytest.importorskip("openmm")

import openmm as mm  # noqa: E402
import openmm.app as app  # noqa: E402
import openmm.unit as unit  # noqa: E402

from gluebind.restraints import boresch, rmsd, separation, system_builder  # noqa: E402


def _system(n):
    system = mm.System()
    for _ in range(n):
        system.addParticle(1.0)
    return system


def _context(system, positions):
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)
    return context


def _nm(coords):
    return [mm.Vec3(*c) for c in coords] * unit.nanometer


def test_separation_distance():
    system = _system(2)
    bias = separation.add_bias(system, [0], [1], r0_nm=0.0, force_constant=10.0)
    ctx = _context(system, _nm([(0, 0, 0), (0, 0, 1.5)]))
    assert bias.getCollectiveVariableValues(ctx)[0] == pytest.approx(1.5, abs=1e-5)


def test_boresch_thetaA_is_right_angle():
    system = _system(6)
    points = {"a": [1], "A": [2], "b": [0], "c": [3], "B": [4], "C": [5]}
    bias = boresch.add_bias(system, "thetaA", points, bias_centre=math.pi / 2, force_constant=100.0)
    # thetaA = angle(b, a, A); vertex a at origin, b along x, A along y => 90 deg.
    ctx = _context(
        system,
        _nm([(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]),
    )
    assert bias.getCollectiveVariableValues(ctx)[0] == pytest.approx(math.pi / 2, abs=1e-4)


@pytest.mark.parametrize("dof", boresch.DOFS)
def test_boresch_all_dofs_build_and_read(dof):
    system = _system(6)
    points = {"a": [0], "A": [1], "b": [2], "c": [3], "B": [4], "C": [5]}
    bias = boresch.add_bias(system, dof, points, bias_centre=1.0, force_constant=100.0)
    ctx = _context(system, _nm([(0.5 * i, 0.1 * i, 0.2 * i) for i in range(6)]))
    assert math.isfinite(bias.getCollectiveVariableValues(ctx)[0])


def test_boresch_fixed_restraint_adds_one_force():
    system = _system(6)
    points = {"a": [0], "A": [1], "b": [2], "c": [3], "B": [4], "C": [5]}
    before = system.getNumForces()
    boresch.add_fixed_restraint(system, "thetaA", points, eq_value=1.0, force_constant=100.0)
    assert system.getNumForces() == before + 1


def test_boresch_points_from_groups():
    points = boresch.points_from_groups([0, 1], [2, 3], {"b": 4, "c": 5, "B": 6, "C": 7})
    assert points["a"] == [0, 1]
    assert points["A"] == [2, 3]
    assert points["b"] == [4]
    assert points["C"] == [7]


def test_rmsd_zero_at_reference():
    system = _system(3)
    positions = _nm([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    force = rmsd.add_rmsd_restraint(system, [0, 1, 2], positions, 5.0, name="rec")
    ctx = _context(system, positions)
    assert force.getCollectiveVariableValues(ctx)[0] == pytest.approx(0.0, abs=1e-6)


def test_rmsd_moving_restraint_has_centre_param():
    system = _system(3)
    positions = _nm([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    force = rmsd.add_rmsd_restraint(system, [0, 1, 2], positions, 5.0, name="lig", centre=2.0)
    names = {force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())}
    assert {"k_lig", "lig_centre"} <= names


def _topology():
    top = app.Topology()
    chain = top.addChain()
    glue = top.addResidue("MOL", chain)
    top.addAtom("C1", app.element.carbon, glue)
    top.addAtom("H1", app.element.hydrogen, glue)
    res = top.addResidue("ALA", chain)
    top.addAtom("CA", app.element.carbon, res)
    top.addAtom("CB", app.element.carbon, res)
    return top


def test_glue_heavy_atoms_excludes_hydrogen():
    idx = system_builder.glue_heavy_atoms(_topology())
    assert idx == [0]  # C1 only; H1 excluded, ALA not glue


def test_atoms_in_residues_by_name():
    idx = system_builder.atoms_in_residues(_topology(), residue_indices=[1], atom_names={"CA"})
    assert idx == [2]
