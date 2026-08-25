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
    """Resolve and persist restraint geometry from the prepared trajectory."""
    import json

    from gluebind.spec_builder import build_restraint_context
    from gluebind.stage_centres import compute_stage_centres

    work_dir = pathlib.Path(work_dir)
    spec = RestraintResolutionSpec.load(work_dir / RESTRAINT_RESOLUTION_SPEC_FILENAME)
    prepared = PreparedSystem.load(spec.prep_dir)
    context = build_restraint_context(
        prepared, spec.config, anchors_override=spec.anchors_override
    )
    resolved = ResolvedRestraintContext(
        config_hash=spec.config_hash,
        prepared_hash=prepared_hash(prepared),
        context=context_to_data(context),
        stage_centres=compute_stage_centres(prepared, context, spec.config),
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
    prepared: PreparedSystem, config: CalculationConfig, output_dir: str | pathlib.Path
) -> dict[str, str]:
    """Write the per-protein manual-anchor reports from an equilibration DCD."""
    import numpy as np

    from gluebind.selection.rmsf import compute_rmsf, stablest_candidates
    from gluebind.spec_builder import _ComplexMap
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
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}
    for protein in ("receptor", "target"):
        ca_selection = "index " + " ".join(map(str, cmap.resolve(protein, "name CA")))
        resids, rmsf = compute_rmsf(universe, selection=ca_selection)
        atom_indices = universe.select_atoms(ca_selection).indices
        resid_to_atom = {
            int(r): int(i) for r, i in zip(resids, atom_indices, strict=True)
        }
        candidates = [(r, resid_to_atom[r]) for r in stablest_candidates(resids, rmsf)]
        pretty = ", ".join(f"resid {r}=atom {i}" for r, i in candidates)
        header = (
            "suggested stable anchor candidates (low-RMSF local minima, most stable "
            f"first): {pretty}\nresid  atom_index  rmsf(Angstrom)"
        )
        path = output_dir / f"rmsf_{protein}.dat"
        np.savetxt(
            path,
            np.column_stack([np.asarray(resids), np.asarray(atom_indices), rmsf]),
            fmt=["%d", "%d", "%.4f"],
            header=header,
        )
        report[protein] = str(path)
    return report


def run_rmsf_report(work_dir: str | pathlib.Path) -> None:
    """Run the manual-anchor RMSF report worker and write ``result.json``."""
    import json

    work_dir = pathlib.Path(work_dir)
    spec = RmsfReportSpec.load(work_dir / RMSF_REPORT_SPEC_FILENAME)
    report = write_rmsf_report(
        PreparedSystem.load(spec.prep_dir), spec.config, spec.output_dir
    )
    (work_dir / RMSF_REPORT_RESULT_FILENAME).write_text(json.dumps(report, indent=2))
