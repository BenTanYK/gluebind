"""Tests for durable compute-node restraint-resolution artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from gluebind.config import CalculationConfig
from gluebind.backend import LocalBackend
from gluebind.restraint_context import (
    RESTRAINT_CONTEXT_FILENAME,
    ResolvedRestraintContext,
    context_from_data,
    context_to_data,
    prepared_hash,
)
from gluebind.simulation.restraint_resolution import (
    RESTRAINT_RESOLUTION_RESULT_FILENAME,
    RESTRAINT_RESOLUTION_SPEC_FILENAME,
    RestraintResolutionSpec,
    run_restraint_resolution,
)
from gluebind.spec_builder import AlwaysOn, BulkTarget, RestraintContext
from gluebind.system.prep import PreparedSystem
from gluebind.runners import Calculation


def _prepared(tmp_path: Path) -> PreparedSystem:
    prepared = PreparedSystem(
        complex_prm7="complex.prm7",
        complex_rst7="complex.rst7",
        complex_trajectory="complex.dcd",
        target_bulk_prm7="target.prm7",
        target_bulk_rst7="target.rst7",
        receptor_bulk_prm7="receptor.prm7",
        receptor_bulk_rst7="receptor.rst7",
        target_molecules=[0],
        receptor_molecules=[1],
    )
    prepared.dump(tmp_path / "prep")
    return prepared


def _context() -> RestraintContext:
    always = AlwaysOn("always", [7, 8], 100.0)
    return RestraintContext(
        complex_topology="complex.prm7",
        complex_coordinates="complex.rst7",
        rec_group=[1, 2],
        lig_group=[3, 4],
        anchors={"b": 10, "c": 11, "B": 12, "C": 13},
        rmsd_order=["receptor"],
        rmsd_atoms_bound={"receptor": [1, 2]},
        rmsd_bulk={
            "receptor": BulkTarget(
                "receptor.prm7", "receptor.rst7", [1, 2], [("held", [3])], [always]
            )
        },
        always_on=[always],
    )


def test_context_round_trip_preserves_nested_restraints():
    context = _context()
    assert context_from_data(context_to_data(context)) == context


def test_resolution_worker_persists_context_and_centres(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    context = _context()
    config = CalculationConfig.model_validate(
        {
            "inputs": {
                "target": {"prm7": "target.prm7", "rst7": "target.rst7"},
                "receptor": {"prm7": "receptor.prm7", "rst7": "receptor.rst7"},
            }
        }
    )
    import gluebind.spec_builder as spec_builder
    import gluebind.stage_centres as stage_centres

    monkeypatch.setattr(
        spec_builder, "build_restraint_context", lambda *a, **k: context
    )
    monkeypatch.setattr(
        stage_centres,
        "compute_stage_centres",
        lambda *a, **k: {"thetaA": [1.0], "separation": [1.5]},
    )
    work_dir = tmp_path / "prep" / "restraint_resolution"
    output = tmp_path / "prep" / RESTRAINT_CONTEXT_FILENAME
    RestraintResolutionSpec(
        config=config,
        prep_dir=str(tmp_path / "prep"),
        output_path=str(output),
        config_hash=config.config_hash,
    ).dump(work_dir / RESTRAINT_RESOLUTION_SPEC_FILENAME)

    run_restraint_resolution(work_dir)

    result = ResolvedRestraintContext.load(output)
    assert result.config_hash == config.config_hash
    assert result.prepared_hash == prepared_hash(prepared)
    assert context_from_data(result.context) == context
    assert result.stage_centres == {"thetaA": [1.0], "separation": [1.5]}
    assert json.loads(
        (work_dir / RESTRAINT_RESOLUTION_RESULT_FILENAME).read_text()
    ) == {"output_path": str(output)}


def test_wire_reuses_context_artifact_without_mdanalysis(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    context = _context()
    config = CalculationConfig.model_validate(
        {
            "inputs": {
                "target": {"prm7": "target.prm7", "rst7": "target.rst7"},
                "receptor": {"prm7": "receptor.prm7", "rst7": "receptor.rst7"},
            }
        }
    )
    ResolvedRestraintContext(
        config_hash=config.config_hash,
        prepared_hash=prepared_hash(prepared),
        context=context_to_data(context),
        stage_centres={"thetaA": [1.0], "separation": [1.5]},
    ).dump(tmp_path / "prep" / RESTRAINT_CONTEXT_FILENAME)

    import gluebind.spec_builder as spec_builder
    import gluebind.simulation.steered_md as steered_md

    monkeypatch.setattr(
        spec_builder,
        "build_restraint_context",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("must not resolve")),
    )
    monkeypatch.setattr(steered_md, "make_steered_md_runner", lambda **kwargs: object())

    calc = Calculation.from_config(config, LocalBackend(), base_dir=tmp_path)
    calc._wire(prepared)

    assert calc.spec_builder.ctx == context
    assert calc.stage_centres == {"thetaA": [1.0], "separation": [1.5]}
