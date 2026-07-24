"""The single-window compute entry point.

:func:`run_window` is the unit of work: it reads a :class:`WindowSpec` (plus the
topology/coordinates it references) from a working directory, builds the OpenMM
system with the window's restraints, samples for the configured time, and writes
the CV timeseries and a validatable ``result.json`` back into that directory. It
raises on failure (so a subprocess exits non-zero).

It is deliberately submission-agnostic — no S3, no environment variables, no
scheduler knowledge — so the same function backs every execution path:

* the **local** and **SLURM** backends run it via :func:`window_launch_command`
  (a ``python -c`` invocation) inside the window's working directory;
* a downstream **AWS Batch runner** stages ``window.json`` + structures into a
  directory, calls ``run_window(dir)``, then uploads the outputs — exactly the
  shape of openfe-runner, minus the CLI subprocess.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import pydantic

WINDOW_SPEC_FILENAME = "window.json"
CV_TIMESERIES_FILENAME = "cv_timeseries.dat"
RESULT_FILENAME = "result.json"

CVType = Literal["boresch", "rmsd", "separation"]


class WindowSpec(pydantic.BaseModel):
    """Everything one umbrella-sampling window needs to run, self-contained.

    Serialisable and shareable across the replicates of a window (only
    ``replicate`` differs), analogous to openfe's ``transformation.json``. The
    ``restraints`` mapping carries the resolved restraint context — the Boresch
    equilibrium values, the RMSD CV region atom selections, and any always-on
    restraints — so the worker needs nothing but this file and the referenced
    structures.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    cv_type: CVType
    stage_name: str
    cv_centre: float
    replicate: int
    dof: str | None = None
    """For a Boresch window, which DoF is sampled (``thetaA``…``phiC``)."""

    topology: str
    """Path or URI to the AMBER prm7 for this window's system."""
    coordinates: str
    """Path or URI to the starting coordinates (rst7 / steered-MD frame)."""

    force_constant: float
    sampling_time_ns: float
    equil_discard_ns: float = 0.0
    timestep_fs: float = 4.0
    hmr_factor: float = 1.5
    pme_cutoff_nm: float = 1.0
    temperature_K: float = 300.0
    sample_interval_steps: int = 125

    restraints: dict = pydantic.Field(default_factory=dict)
    """Resolved restraint context (atom indices already resolved against the
    topology by the runner). Recognised keys:

    * ``rmsd``: list of ``{name, atoms, force_constant, centre|None, sampled}``
      — every RMSD restraint on the system; the one with ``sampled=True`` is the
      window's biased CV (for ``cv_type == "rmsd"``).
    * ``boresch``: ``{rec_group, lig_group, anchors:{b,c,B,C}, force_constant,
      fixed:{dof: eq_value}}`` — groups/anchors and the DoFs held fixed.
    * ``separation``: ``{rec_group, lig_group}`` — the interface groups whose
      distance is biased (for ``cv_type == "separation"``).
    """

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "WindowSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def window_launch_command(python: str = "python") -> list[str]:
    """The command a backend runs (inside the window's work dir) to execute it.

    Reads ``window.json`` from the current directory and runs the window there,
    so a backend only has to set the job's working directory — no arguments to
    thread through. This is the ``python -c`` seam that keeps v1 free of any CLI.
    """
    code = "from gluebind.simulation.window import run_window; run_window('.')"
    return [python, "-c", code]


def run_window(work_dir: str | pathlib.Path) -> None:
    """Run the umbrella-sampling window whose spec is at ``work_dir/window.json``.

    Builds the OpenMM system, applies the context restraints and the biased CV
    for this window's ``cv_type``, samples, and writes ``cv_timeseries.dat`` (the
    ``[sample_index, cv_value]`` array WHAM consumes) and a validatable
    ``result.json`` into ``work_dir``. Raises on failure.
    """
    import json

    import numpy as np

    # OpenMM-dependent imports are local so importing this module stays cheap.
    from gluebind.restraints import boresch, rmsd, separation, system_builder as sb

    work_dir = pathlib.Path(work_dir)
    spec = WindowSpec.load(work_dir / WINDOW_SPEC_FILENAME)
    ctx = spec.restraints

    prmtop, system = sb.build_system(
        spec.topology, hmr_factor=spec.hmr_factor, pme_cutoff_nm=spec.pme_cutoff_nm
    )
    positions, box_vectors = sb.load_coordinates(spec.coordinates)
    simulation, integrator = sb.build_simulation(
        prmtop, system, timestep_fs=spec.timestep_fs
    )
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    # Restraints reference the equilibrated *input* structure and are applied
    # BEFORE minimisation/heating, so they hold the structure throughout — rather
    # than being added after a free heating that could let it drift (and then
    # referenced to the drifted structure). Matches the template convention.
    reference = positions

    bias = None  # the force whose collective variable we record

    # Context RMSD restraints (fixed partners + optionally the sampled RMSD CV).
    for entry in ctx.get("rmsd", []):
        force = rmsd.add_rmsd_restraint(
            system,
            entry["atoms"],
            reference,
            entry["force_constant"],
            name=entry["name"],
            centre=entry.get("centre"),
        )
        simulation.context.reinitialize(preserveState=True)
        if entry.get("sampled"):
            bias = force

    # Fixed Boresch restraints (context for Boresch and separation windows).
    bore = ctx.get("boresch")
    if bore:
        points = boresch.points_from_groups(
            bore["rec_group"], bore["lig_group"], bore["anchors"]
        )
        for dof, eq_value in bore.get("fixed", {}).items():
            boresch.add_fixed_restraint(
                system, dof, points, eq_value, bore["force_constant"]
            )
        simulation.context.reinitialize(preserveState=True)

    # The biased CV for this window.
    if spec.cv_type == "boresch":
        if spec.dof is None:
            raise ValueError("a Boresch window requires spec.dof")
        bias = boresch.add_bias(
            system, spec.dof, points, spec.cv_centre, spec.force_constant
        )
        simulation.context.reinitialize(preserveState=True)
    elif spec.cv_type == "separation":
        sep = ctx["separation"]
        bias = separation.add_bias(
            system,
            sep["rec_group"],
            sep["lig_group"],
            spec.cv_centre,
            spec.force_constant,
        )
        simulation.context.reinitialize(preserveState=True)
    elif spec.cv_type == "rmsd":
        if bias is None:
            raise ValueError(
                "an RMSD window requires a restraint entry with sampled=True"
            )

    # Minimise + heat with all restraints in place, holding the structure to the
    # equilibrated reference throughout the ramp.
    sb.minimise_and_heat(
        simulation, integrator, target_temperature_K=spec.temperature_K
    )

    ns_per_step = spec.timestep_fs * 1e-6
    samples = sb.collect_cv_samples(
        simulation,
        bias,
        equil_steps=int(spec.equil_discard_ns / ns_per_step),
        sampling_steps=int(spec.sampling_time_ns / ns_per_step),
        record_steps=spec.sample_interval_steps,
    )

    np.savetxt(work_dir / CV_TIMESERIES_FILENAME, samples)
    result = {
        "cv_type": spec.cv_type,
        "stage_name": spec.stage_name,
        "dof": spec.dof,
        "cv_centre": spec.cv_centre,
        "replicate": spec.replicate,
        "force_constant": spec.force_constant,
        "sampling_time_ns": spec.sampling_time_ns,
        "n_samples": int(samples.shape[0]),
        "mean_cv": float(samples[:, 1].mean()) if samples.size else None,
    }
    (work_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2))
