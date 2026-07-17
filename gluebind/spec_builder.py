"""The concrete spec builder — where prep + selection + config meet.

:class:`SpecBuilder` turns a window identity (``cv_type``, ``stage_name``, ``dof``,
``cv_centre``, ``replicate``, ``boresch_eq_values``) into a fully-resolved
:class:`~gluebind.simulation.window.WindowSpec` for the runner. It is the callable
the runner's ``spec_builder`` seam expects.

The heavy, structure-dependent resolution (interface detection, anchor selection,
RMSD-region atom indices) is done once by :func:`build_restraint_context`
(MDAnalysis; integration-verified) and captured in a :class:`RestraintContext`.
Given that context, :meth:`SpecBuilder.__call__` is pure assembly — it selects the
right topology/coordinates and builds the restraint dict for each CV type, exactly
matching what :func:`gluebind.simulation.window.run_window` reads back.

Restraint conventions (matching the template + the paper's thermodynamic cycle):

* **Boresch** window: all RMSD regions held fixed (rigid), the already-determined
  Boresch DoFs fixed at their equilibrium values, the sampled DoF biased.
* **RMSD bound** window: regions *before* this one in ``rmsd_order`` held fixed,
  this region sampled (the paper's order-independent sequential application).
* **RMSD bulk** window: the region sampled in its isolated (bulk) topology.
* **Separation** window: all RMSD regions and all five Boresch DoFs fixed, the
  interface-CoM distance biased; coordinates come from the steered-MD frame for
  this window centre.
"""

from __future__ import annotations

import dataclasses
import pathlib

from gluebind.config.calculation import CalculationConfig
from gluebind.simulation.window import WindowSpec


@dataclasses.dataclass(frozen=True)
class BulkTarget:
    """The isolated-species topology/coordinates + atom indices for a bulk RMSD stage."""

    topology: str
    coordinates: str
    atoms: list[int]


@dataclasses.dataclass(frozen=True)
class RestraintContext:
    """Resolved (structure-dependent) restraint data, produced once by the resolver."""

    complex_topology: str
    complex_coordinates: str
    rec_group: list[int]
    """Receptor interface Cα atoms (+ glue heavy atoms if assigned to receptor)."""
    lig_group: list[int]
    """Ligand interface Cα atoms (+ glue heavy atoms if assigned to target)."""
    anchors: dict[str, int]
    """The four non-bonded Boresch anchors: keys ``b``, ``c``, ``B``, ``C``."""
    rmsd_order: list[str]
    """RMSD region names in bound-state application order."""
    rmsd_atoms_bound: dict[str, list[int]]
    """Region name → atom indices in the complex topology (bound state)."""
    rmsd_bulk: dict[str, BulkTarget]
    """Region name → its isolated bulk topology/coords/atoms."""


class SpecBuilder:
    """Callable that builds a :class:`WindowSpec` for any window in the calculation."""

    def __init__(
        self,
        context: RestraintContext,
        config: CalculationConfig,
        *,
        smd_frames_dir: str | pathlib.Path | None = None,
    ) -> None:
        self.ctx = context
        self.config = config
        self.smd_frames_dir = pathlib.Path(smd_frames_dir) if smd_frames_dir else None

    def __call__(
        self,
        *,
        cv_type: str,
        stage_name: str,
        dof: str | None,
        cv_centre: float,
        replicate: int,
        boresch_eq_values: dict,
    ) -> WindowSpec:
        if cv_type == "boresch":
            return self._boresch(stage_name, dof, cv_centre, replicate, boresch_eq_values)
        if cv_type == "rmsd":
            return self._rmsd(stage_name, cv_centre, replicate)
        if cv_type == "separation":
            return self._separation(cv_centre, replicate, boresch_eq_values)
        raise ValueError(f"unknown cv_type {cv_type!r}")

    # -- shared bits ---------------------------------------------------------

    def _common(self, cv_type: str, stage_name: str, replicate: int) -> dict:
        s = self.config.sampling
        schedule = s.for_cv(cv_type, stage_name)
        return {
            "stage_name": stage_name,
            "replicate": replicate,
            "sampling_time_ns": schedule.sampling_time_ns,
            "equil_discard_ns": schedule.equil_discard_ns,
            "timestep_fs": s.timestep_fs,
            "hmr_factor": s.hmr_factor,
            "pme_cutoff_nm": s.pme_cutoff_nm,
            "temperature_K": s.temperature_K,
            "sample_interval_steps": s.sample_interval_steps,
        }

    def _fixed_rmsd_list(self) -> list[dict]:
        k = self.config.sampling.rmsd.force_constant
        return [
            {
                "name": region,
                "atoms": self.ctx.rmsd_atoms_bound[region],
                "force_constant": k,
                "centre": None,
                "sampled": False,
            }
            for region in self.ctx.rmsd_order
        ]

    def _boresch_block(self, boresch_eq_values: dict) -> dict:
        return {
            "rec_group": self.ctx.rec_group,
            "lig_group": self.ctx.lig_group,
            "anchors": self.ctx.anchors,
            "force_constant": self.config.sampling.boresch.force_constant,
            "fixed": dict(boresch_eq_values),
        }

    # -- per-CV assembly -----------------------------------------------------

    def _boresch(self, stage_name, dof, cv_centre, replicate, boresch_eq_values) -> WindowSpec:
        return WindowSpec(
            cv_type="boresch",
            dof=dof,
            cv_centre=cv_centre,
            force_constant=self.config.sampling.boresch.force_constant,
            topology=self.ctx.complex_topology,
            coordinates=self.ctx.complex_coordinates,
            restraints={
                "rmsd": self._fixed_rmsd_list(),
                "boresch": self._boresch_block(boresch_eq_values),
            },
            **self._common("boresch", stage_name, replicate),
        )

    def _rmsd(self, stage_name, cv_centre, replicate) -> WindowSpec:
        region, state = stage_name.rsplit("_", 1)
        k = self.config.sampling.rmsd.force_constant

        if state == "bound":
            topology = self.ctx.complex_topology
            coordinates = self.ctx.complex_coordinates
            rmsd: list[dict] = []
            for other in self.ctx.rmsd_order:
                sampled = other == region
                rmsd.append(
                    {
                        "name": other,
                        "atoms": self.ctx.rmsd_atoms_bound[other],
                        "force_constant": k,
                        "centre": cv_centre if sampled else None,
                        "sampled": sampled,
                    }
                )
                if sampled:
                    break  # regions after this one are not yet applied
        else:  # bulk
            bulk = self.ctx.rmsd_bulk[region]
            topology = bulk.topology
            coordinates = bulk.coordinates
            rmsd = [
                {
                    "name": region,
                    "atoms": bulk.atoms,
                    "force_constant": k,
                    "centre": cv_centre,
                    "sampled": True,
                }
            ]

        return WindowSpec(
            cv_type="rmsd",
            dof=None,
            cv_centre=cv_centre,
            force_constant=k,
            topology=topology,
            coordinates=coordinates,
            restraints={"rmsd": rmsd},
            **self._common("rmsd", stage_name, replicate),
        )

    def _separation(self, cv_centre, replicate, boresch_eq_values) -> WindowSpec:
        if self.smd_frames_dir is not None:
            coordinates = str(self.smd_frames_dir / f"{cv_centre:.4g}nm.rst7")
        else:
            coordinates = self.ctx.complex_coordinates
        return WindowSpec(
            cv_type="separation",
            dof=None,
            cv_centre=cv_centre,
            force_constant=self.config.sampling.separation.force_constant,
            topology=self.ctx.complex_topology,
            coordinates=coordinates,
            restraints={
                "rmsd": self._fixed_rmsd_list(),
                "boresch": self._boresch_block(boresch_eq_values),
                "separation": {"rec_group": self.ctx.rec_group, "lig_group": self.ctx.lig_group},
            },
            **self._common("separation", "separation", replicate),
        )


def build_restraint_context(
    prepared, config: CalculationConfig, *, interface_cutoff_angstrom: float = 12.0
) -> RestraintContext:
    """Resolve a :class:`RestraintContext` from a :class:`PreparedSystem` (MDAnalysis).

    Detects the interface (Cα–Cα pairs within ``interface_cutoff_angstrom``),
    selects the Boresch anchors (``BoreschSpec.anchors`` — auto via
    :mod:`gluebind.selection`, or a validated manual override), and resolves the
    RMSD-region atom indices for the bound (complex) and bulk (isolated) topologies.

    Reuses the unit-tested pure primitives (interface detection, collinearity,
    DoF-variance selection); the MDAnalysis extraction is integration-verified
    against real structures (Phase 7), so this is not exercised by the unit suite.
    The pure assembly it feeds (:class:`SpecBuilder`) is.
    """
    import MDAnalysis as mda
    import numpy as np

    from gluebind.selection.anchors import select_anchors, validate_manual_anchors
    from gluebind.selection.dssp import structured_residues
    from gluebind.selection.interface import interface_residues
    from gluebind.selection.rmsf import compute_rmsf, stablest_candidates

    universe = mda.Universe(prepared.complex_prm7, prepared.complex_rst7)
    n_target = mda.Universe(config.inputs.target.prm7).residues.n_residues
    n_receptor = mda.Universe(config.inputs.receptor.prm7).residues.n_residues
    residues = universe.residues
    target_ca = residues[:n_target].atoms.select_atoms("name CA")
    receptor_ca = residues[n_target : n_target + n_receptor].atoms.select_atoms("name CA")

    rec_i, lig_i = interface_residues(
        receptor_ca.positions, target_ca.positions, cutoff=interface_cutoff_angstrom
    )
    rec_group = [int(i) for i in receptor_ca[rec_i].indices]
    lig_group = [int(i) for i in target_ca[lig_i].indices]

    glue_indices = [int(i) for i in universe.select_atoms("resname MOL and not name H*").indices]
    assign = config.inputs.glue.assign_to if config.inputs.glue else None
    if assign == "receptor":
        rec_group += glue_indices
    elif assign == "target":
        lig_group += glue_indices

    anchors = _resolve_anchors(
        config, prepared, universe, receptor_ca, target_ca, rec_group, lig_group, np
    )

    rmsd_order, rmsd_atoms_bound, rmsd_bulk = _resolve_rmsd_regions(
        config, prepared, universe, receptor_ca, target_ca, glue_indices, assign
    )

    return RestraintContext(
        complex_topology=prepared.complex_prm7,
        complex_coordinates=prepared.complex_rst7,
        rec_group=rec_group,
        lig_group=lig_group,
        anchors=anchors,
        rmsd_order=rmsd_order,
        rmsd_atoms_bound=rmsd_atoms_bound,
        rmsd_bulk=rmsd_bulk,
    )


def _resolve_anchors(config, prepared, universe, receptor_ca, target_ca, rec_group, lig_group, np):
    """Manual anchors (validated) or automatic selection over the trajectory."""
    from gluebind.selection.anchors import select_anchors, validate_manual_anchors
    from gluebind.selection.dssp import structured_residues
    from gluebind.selection.rmsf import compute_rmsf, stablest_candidates

    spec = config.restraints.boresch.anchors
    if spec != "auto":
        mean = {
            "a": universe.atoms[rec_group].positions.mean(axis=0),
            "A": universe.atoms[lig_group].positions.mean(axis=0),
            **{key: universe.atoms[spec[key]].position for key in ("b", "c", "B", "C")},
        }
        validate_manual_anchors(mean)
        return {key: int(spec[key]) for key in ("b", "c", "B", "C")}

    if prepared.complex_trajectory is None:
        raise ValueError(
            "automatic anchor selection needs an equilibration trajectory; "
            "provide anchors manually via BoreschSpec.anchors"
        )

    traj = mda.Universe(prepared.complex_prm7, prepared.complex_trajectory)  # noqa: F821
    structured = set(structured_residues(traj))

    def _candidates(ca_atoms):
        resids, rmsf = compute_rmsf(traj, selection=f"index {' '.join(map(str, ca_atoms.indices))}")
        pool = stablest_candidates(resids, rmsf)
        # keep only structured residues, map back to a representative CA atom index
        keep = [r for r in pool if r in structured]
        by_resid = {int(a.resid): int(a.index) for a in ca_atoms}
        return [by_resid[r] for r in keep if r in by_resid]

    rec_candidates = _candidates(receptor_ca)
    lig_candidates = _candidates(target_ca)

    series = _collect_series(traj, rec_group, lig_group, rec_candidates + lig_candidates, np)
    return select_anchors(
        receptor_candidates=rec_candidates,
        ligand_candidates=lig_candidates,
        a_coords=series["a"],
        A_coords=series["A"],
        coords_of=lambda i: series[i],
    )


def _collect_series(traj, rec_group, lig_group, atom_indices, np):
    """Per-frame coordinate series for the interface centroids and candidate atoms."""
    a_series, A_series = [], []
    atom_series = {i: [] for i in atom_indices}
    for _ in traj.trajectory:
        a_series.append(traj.atoms[rec_group].positions.mean(axis=0).copy())
        A_series.append(traj.atoms[lig_group].positions.mean(axis=0).copy())
        for i in atom_indices:
            atom_series[i].append(traj.atoms[i].position.copy())
    result = {i: np.asarray(v) for i, v in atom_series.items()}
    result["a"] = np.asarray(a_series)
    result["A"] = np.asarray(A_series)
    return result


def _resolve_rmsd_regions(config, prepared, universe, receptor_ca, target_ca, glue_indices, assign):
    """RMSD region atom indices for bound (complex) and bulk (isolated) topologies."""
    import MDAnalysis as mda

    restraints = config.restraints
    order: list[str] = []
    bound: dict[str, list[int]] = {}
    bulk: dict[str, BulkTarget] = {}

    receptor_bulk = mda.Universe(prepared.receptor_bulk_prm7, prepared.receptor_bulk_rst7)
    target_bulk = mda.Universe(prepared.target_bulk_prm7, prepared.target_bulk_rst7)

    if restraints.uses_default_all_ca:
        rec_bound = [int(i) for i in receptor_ca.indices]
        lig_bound = [int(i) for i in target_ca.indices]
        if assign == "receptor":
            rec_bound += glue_indices
        elif assign == "target":
            lig_bound += glue_indices
        order = ["receptor", "target"]
        bound = {"receptor": rec_bound, "target": lig_bound}
        bulk = {
            "receptor": _bulk_target(prepared.receptor_bulk_prm7, prepared.receptor_bulk_rst7, receptor_bulk, assign == "receptor"),
            "target": _bulk_target(prepared.target_bulk_prm7, prepared.target_bulk_rst7, target_bulk, assign == "target"),
        }
        return order, bound, bulk

    # Custom RMSD CVs from selection strings.
    for cv in restraints.rmsd_cvs:
        order.append(cv.name)
        bound[cv.name] = [int(i) for i in universe.select_atoms(cv.selection).indices]
    if restraints.rmsd_order:
        order = list(restraints.rmsd_order)
    return order, bound, bulk


def _bulk_target(prm7, rst7, bulk_universe, include_glue) -> BulkTarget:
    selection = "name CA"
    if include_glue:
        selection = f"({selection}) or (resname MOL and not name H*)"
    atoms = [int(i) for i in bulk_universe.select_atoms(selection).indices]
    return BulkTarget(topology=prm7, coordinates=rst7, atoms=atoms)
