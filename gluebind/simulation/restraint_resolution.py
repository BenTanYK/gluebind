"""Compute-node worker for MDAnalysis-heavy restraint resolution."""

from __future__ import annotations

import pathlib

import pydantic

from gluebind.config.calculation import CalculationConfig
from gluebind.restraint_context import (
    ResolvedRestraintContext,
    context_to_data,
    prepared_hash,
)
from gluebind.system.prep import PreparedSystem

RESTRAINT_RESOLUTION_SPEC_FILENAME = "restraint_resolution.json"
RESTRAINT_RESOLUTION_RESULT_FILENAME = "result.json"
RMSF_REPORT_SPEC_FILENAME = "rmsf_report.json"
RMSF_REPORT_RESULT_FILENAME = "result.json"


class RestraintResolutionSpec(pydantic.BaseModel):
    """Everything the compute-node restraint-resolution worker needs."""

    model_config = pydantic.ConfigDict(extra="forbid")

    config: CalculationConfig
    prep_dir: str
    output_path: str
    config_hash: str
    anchors_override: dict[str, int] | None = None

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "RestraintResolutionSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def restraint_resolution_launch_command(python: str = "python") -> list[str]:
    code = (
        "from gluebind.simulation.restraint_resolution import "
        "run_restraint_resolution; run_restraint_resolution('.')"
    )
    return [python, "-c", code]


def run_restraint_resolution(work_dir: str | pathlib.Path) -> None:
    """Resolve restraint geometry and persist the RMSF diagnostics."""
    import json

    from gluebind.spec_builder import build_restraint_context
    from gluebind.stage_centres import (
        BORESCH_DISTRIBUTION_DIRNAME,
        compute_stage_centres,
    )

    work_dir = pathlib.Path(work_dir)
    spec = RestraintResolutionSpec.load(work_dir / RESTRAINT_RESOLUTION_SPEC_FILENAME)
    prepared = PreparedSystem.load(spec.prep_dir)
    context = build_restraint_context(
        prepared, spec.config, anchors_override=spec.anchors_override
    )
    # Keep per-protein RMSF profiles available for post-hoc stability and
    # anchor-selection inspection, including when anchors were configured
    # manually. This runs in the same backend job as trajectory-dependent
    # restraint resolution, so the driver never needs MDAnalysis.
    write_rmsf_report(prepared, spec.config, spec.prep_dir, anchors=context.anchors)
    prepared_identity = prepared_hash(prepared)
    report: dict = {}
    stage_centres = compute_stage_centres(
        prepared,
        context,
        spec.config,
        distributions_dir=pathlib.Path(spec.prep_dir) / BORESCH_DISTRIBUTION_DIRNAME,
        distribution_metadata={
            "config_hash": spec.config_hash,
            "prepared_hash": prepared_identity,
            "anchors": context.anchors,
        },
        report=report,
    )
    if report:
        print(
            "restraint resolution: equilibrium separation "
            f"{report['separation_equilibrium_nm']:.3f} nm; window_min auto -> "
            f"{report['separation_window_min_nm']:g} nm",
            flush=True,
        )
    resolved = ResolvedRestraintContext(
        config_hash=spec.config_hash,
        prepared_hash=prepared_identity,
        context=context_to_data(context),
        stage_centres=stage_centres,
        **report,
    )
    resolved.dump(spec.output_path)
    (work_dir / RESTRAINT_RESOLUTION_RESULT_FILENAME).write_text(
        json.dumps({"output_path": spec.output_path}, indent=2)
    )


class RmsfReportSpec(pydantic.BaseModel):
    """Compute-node specification for the manual-anchor RMSF report."""

    model_config = pydantic.ConfigDict(extra="forbid")

    config: CalculationConfig
    prep_dir: str
    output_dir: str

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "RmsfReportSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def rmsf_report_launch_command(python: str = "python") -> list[str]:
    code = (
        "from gluebind.simulation.restraint_resolution import "
        "run_rmsf_report; run_rmsf_report('.')"
    )
    return [python, "-c", code]


def write_rmsf_report(
    prepared: PreparedSystem,
    config: CalculationConfig,
    output_dir: str | pathlib.Path,
    *,
    anchors: dict[str, int] | None = None,
) -> dict[str, str]:
    """Write the per-protein RMSF reports from an equilibration DCD.

    For each protein, ``rmsf_<protein>.dat`` lists the Cα RMSF per residue and
    ``rmsf_<protein>.png`` plots it. Residues are given in both numberings: the
    protein's input topology (what ``restraints.boresch.anchors`` and the RMSD
    selections use) and the assembled complex (the equilibrated ``.prm7``), and so
    are the Cα atom indices (0-based).

    ``anchors`` are the Boresch anchors as complex Cα atom indices (``b``/``c`` in
    the receptor, ``B``/``C`` in the target), marked on the plots. Without them,
    manually configured anchors are shown if they resolve. The plot is a
    convenience: if it cannot be drawn, a warning is printed and the report is
    still written.
    """
    import numpy as np

    from gluebind.selection.rmsf import (
        AnchorLabel,
        compute_rmsf,
        plot_rmsf,
        stablest_candidates,
    )
    from gluebind.spec_builder import _ComplexMap, _resolve_manual_anchors
    from gluebind.system.mdanalysis import load_amber_universe

    if prepared.complex_trajectory is None:
        raise RuntimeError(
            "cannot write an RMSF report: the equilibration produced no trajectory "
            "(prepared.complex_trajectory is None)"
        )
    universe = load_amber_universe(prepared.complex_prm7, prepared.complex_trajectory)
    cmap = _ComplexMap(
        universe,
        load_amber_universe(config.inputs.target.prm7),
        load_amber_universe(config.inputs.receptor.prm7),
        has_glue=config.inputs.glue is not None,
    )
    if anchors is None and isinstance(config.restraints.boresch.anchors, dict):
        try:
            anchors = _resolve_manual_anchors(config.restraints.boresch.anchors, cmap)
        except ValueError as exc:  # reported again, as an error, at resolution
            print(f"RMSF plot: configured anchors not shown ({exc})", flush=True)
    roles = {"b": "receptor", "c": "receptor", "B": "target", "C": "target"}

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}
    for protein in ("receptor", "target"):
        atom_indices = cmap.resolve(protein, "name CA")
        _, rmsf = compute_rmsf(
            universe, selection="index " + " ".join(map(str, atom_indices))
        )
        # Number residues as the input topology does (the complex numbering is
        # offset by the glue and, for the target, by the receptor).
        protein_u, offset = cmap.input_universe(protein), cmap.offset(protein)
        resids = [int(protein_u.atoms[i - offset].resid) for i in atom_indices]
        complex_resids = [int(universe.atoms[i].resid) for i in atom_indices]
        # Each protein is one contiguous block of the complex, so its residues
        # are renumbered by a constant shift; if not, the plot shows input only.
        shifts = {c - r for r, c in zip(resids, complex_resids, strict=True)}
        complex_offset = shifts.pop() if len(shifts) == 1 else None
        input_atoms = [i - offset for i in atom_indices]
        candidates = stablest_candidates(resids, rmsf)
        header = (
            "suggested stable anchor candidates (low-RMSF local minima, most stable "
            f"first; input topology resids): {', '.join(map(str, candidates))}\n"
            "CA atom indices are 0-based (Amber @ masks count from 1)\n"
            "resid (input)  resid (complex)  CA index (input)  CA index (complex)  "
            "rmsf(Angstrom)"
        )
        path = output_dir / f"rmsf_{protein}.dat"
        np.savetxt(
            path,
            np.column_stack(
                [
                    np.asarray(resids),
                    np.asarray(complex_resids),
                    np.asarray(input_atoms),
                    np.asarray(atom_indices),
                    rmsf,
                ]
            ),
            fmt=["%d", "%d", "%d", "%d", "%.4f"],
            header=header,
        )
        report[protein] = str(path)

        row_of = {atom: i for i, atom in enumerate(atom_indices)}
        protein_anchors = {
            key: AnchorLabel(
                resid=resids[row_of[atom]],
                atom=input_atoms[row_of[atom]],
                complex_resid=complex_resids[row_of[atom]],
                complex_atom=atom,
            )
            for key, atom in (anchors or {}).items()
            if roles.get(key) == protein and atom in row_of
        }
        try:
            report[f"{protein}_plot"] = str(
                plot_rmsf(
                    resids,
                    rmsf,
                    output_dir / f"rmsf_{protein}.png",
                    title=(
                        f"{protein[:1].upper() + protein[1:]} Cα RMSF "
                        "over the equilibration trajectory"
                    ),
                    candidates=candidates,
                    anchors=protein_anchors,
                    zones=_rmsf_zones(config, protein, protein_u),
                    complex_offset=complex_offset,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the plot must not fail resolution
            print(f"RMSF plot for the {protein} not written: {exc!r}", flush=True)
    return report


def _rmsf_zones(config: CalculationConfig, protein: str, protein_u) -> list:
    """The RMSD regions and always-on restraints on ``protein``, as plot zones in
    input-topology residue numbers."""
    from gluebind.selection.rmsf import RmsfZone

    def resids(selection: str) -> list[int]:
        return sorted({int(r) for r in protein_u.select_atoms(selection).resids})

    restraints = config.restraints
    zones = []
    if restraints.uses_default_all_ca:
        zones.append(RmsfZone("RMSD (all Cα)", resids("name CA"), "rmsd"))
    else:
        zones += [
            RmsfZone(f"RMSD {cv.name}", resids(cv.selection), "rmsd")
            for cv in restraints.rmsd_cvs
            if cv.protein == protein
        ]
    zones += [
        RmsfZone("always-on", resids(r.selection), "always_on")
        for r in restraints.always_on
        if r.protein == protein
    ]
    return zones


def run_rmsf_report(work_dir: str | pathlib.Path) -> None:
    """Run the manual-anchor RMSF report worker and write ``result.json``."""
    import json

    work_dir = pathlib.Path(work_dir)
    spec = RmsfReportSpec.load(work_dir / RMSF_REPORT_SPEC_FILENAME)
    report = write_rmsf_report(
        PreparedSystem.load(spec.prep_dir), spec.config, spec.output_dir
    )
    (work_dir / RMSF_REPORT_RESULT_FILENAME).write_text(json.dumps(report, indent=2))
