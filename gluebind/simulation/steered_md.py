"""Steered MD to generate separation-window starting frames.

Ports the template's ``SMD.py`` pulling scheme onto the tested restraint builders:
with all RMSD and Boresch restraints in place, the interface-CoM distance is
steered outward by a moving harmonic potential, and a frame is saved whenever the
measured distance first crosses each target window centre. Those frames seed the
separation umbrella-sampling windows.

The window-target scheduling is pure and tested; the MD itself needs OpenMM +
real structures and is verified in integration (Phase 7). OpenMM/ParmEd are
imported lazily.
"""

from __future__ import annotations

import pathlib

import numpy as np


def separation_window_targets(centres) -> list[float]:
    """Sorted, de-duplicated window centres (nm) to snapshot during the pull."""
    return sorted({round(float(c), 4) for c in centres})


def make_frame_generator(
    *, topology, coordinates, restraint_context, window_centres, out_dir, config, **smd_kwargs
):
    """Return a ``callable(boresch_eq_values)`` that generates the separation-window
    SMD frames — the shape :class:`~gluebind.runners.calculation.Calculation`
    expects for its ``steered_md_runner`` hook, so the whole protocol runs from a
    single ``calc.run()``.
    """

    def _generate(boresch_eq_values: dict):
        return run_steered_md(
            topology=topology,
            coordinates=coordinates,
            restraint_context=restraint_context,
            boresch_eq_values=boresch_eq_values,
            window_centres=window_centres,
            out_dir=out_dir,
            config=config,
            **smd_kwargs,
        )

    return _generate


def _save_frame_rst7(prmtop_path, positions, box_vectors, out_path) -> None:
    """Write an AMBER rst7 (positions + box) that run_window can reload."""
    import parmed

    structure = parmed.load_file(str(prmtop_path))
    structure.positions = positions
    structure.box_vectors = box_vectors
    structure.save(str(out_path), format="rst7", overwrite=True)


def run_steered_md(
    *,
    topology,
    coordinates,
    restraint_context,
    boresch_eq_values: dict,
    window_centres,
    out_dir: str | pathlib.Path,
    config,
    k_smd: float = 100.0,
    k_rmsd: float = 50.0,
    k_boresch: float = 250.0,
    initial_r0_nm: float = 1.15,
    total_steps: int = 750_000,
    increment_steps: int = 100,
    platform=None,
) -> dict[float, str]:
    """Steer the interface separation outward, saving an rst7 per window centre.

    Returns ``{centre_nm: rst7_path}``. Force constants default to the template's
    steered-MD values (stiffer than the US windows). Reuses the shared system
    builder and restraint modules so the geometry is identical to sampling.
    """
    import openmm.unit as unit

    from gluebind.restraints import boresch as boresch_mod
    from gluebind.restraints import rmsd as rmsd_mod
    from gluebind.restraints import separation as separation_mod
    from gluebind.restraints import system_builder as sb

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = separation_window_targets(window_centres)

    prmtop, system = sb.build_system(
        topology, hmr_factor=config.sampling.hmr_factor, pme_cutoff_nm=config.sampling.pme_cutoff_nm
    )
    positions, box = sb.load_coordinates(coordinates)
    simulation, integrator = sb.build_simulation(
        prmtop, system, timestep_fs=config.sampling.timestep_fs, platform=platform
    )
    simulation.context.setPeriodicBoxVectors(*box)
    simulation.context.setPositions(positions)
    sb.minimise_and_heat(simulation, integrator, target_temperature_K=config.sampling.temperature_K)
    reference = simulation.context.getState(getPositions=True).getPositions()

    # Fixed RMSD + Boresch restraints (rigid), then the moving separation bias.
    for region, atoms in restraint_context.rmsd_atoms_bound.items():
        rmsd_mod.add_rmsd_restraint(system, atoms, reference, k_rmsd, name=region, centre=None)
    points = boresch_mod.points_from_groups(
        restraint_context.rec_group, restraint_context.lig_group, restraint_context.anchors
    )
    for dof, eq_value in boresch_eq_values.items():
        boresch_mod.add_fixed_restraint(system, dof, points, eq_value, k_boresch)

    cv = separation_mod.make_cv(restraint_context.rec_group, restraint_context.lig_group)
    import openmm as mm

    steer = mm.CustomCVForce("0.5*k_smd*(cv-r0)^2")
    steer.addGlobalParameter("k_smd", k_smd * unit.kilocalories_per_mole / unit.angstrom**2)
    steer.addGlobalParameter("r0", initial_r0_nm * unit.nanometers)
    steer.addCollectiveVariable("cv", cv)
    system.addForce(steer)
    simulation.context.reinitialize(preserveState=True)

    # Pull r0 from the initial value out past the furthest target, snapshotting
    # each target the first time the measured distance reaches it.
    r0 = initial_r0_nm
    span = max(targets) - initial_r0_nm + 0.2
    per_increment = span / (total_steps // increment_steps)
    frames: dict[float, str] = {}
    remaining = list(targets)

    for _ in range(total_steps // increment_steps):
        if not remaining:
            break
        r0 += per_increment
        simulation.context.setParameter("r0", r0 * unit.nanometers)
        simulation.step(increment_steps)
        current = steer.getCollectiveVariableValues(simulation.context)[0]
        while remaining and current >= remaining[0]:
            target = remaining.pop(0)
            state = simulation.context.getState(getPositions=True)
            out_path = out_dir / f"{target:.4g}nm.rst7"
            _save_frame_rst7(topology, state.getPositions(), state.getPeriodicBoxVectors(), out_path)
            frames[target] = str(out_path)

    return frames
