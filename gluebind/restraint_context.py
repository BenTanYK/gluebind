"""Durable serialisation for resolved restraint geometry.

The resolved restraint context is expensive to obtain because automatic Boresch
anchor selection reads the equilibration trajectory through MDAnalysis.  It is
therefore written once by the restraint-resolution worker and subsequently
loaded by the lightweight orchestration process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import pydantic

from gluebind.spec_builder import AlwaysOn, BulkTarget, RestraintContext
from gluebind.system.prep import PreparedSystem

RESTRAINT_CONTEXT_FILENAME = "restraint_context.json"
SCHEMA_VERSION = 1


class ResolvedRestraintContext(pydantic.BaseModel):
    """Portable result of trajectory-dependent restraint resolution."""

    model_config = pydantic.ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    config_hash: str
    prepared_hash: str
    context: dict
    stage_centres: dict[str, list[float]]

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ResolvedRestraintContext":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def prepared_hash(prepared: PreparedSystem) -> str:
    """Stable identity for the prepared-manifest inputs to resolution."""
    payload = json.dumps(prepared.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def context_to_data(context: RestraintContext) -> dict:
    """Convert the frozen dataclass context into JSON-compatible data."""
    return dataclasses.asdict(context)


def context_from_data(data: dict) -> RestraintContext:
    """Rebuild a :class:`RestraintContext` from :func:`context_to_data` output."""
    bulk = {
        name: BulkTarget(
            topology=value["topology"],
            coordinates=value["coordinates"],
            atoms=list(value["atoms"]),
            held=[(str(n), list(a)) for n, a in value.get("held", [])],
            always_on=[AlwaysOn(**item) for item in value.get("always_on", [])],
        )
        for name, value in data["rmsd_bulk"].items()
    }
    return RestraintContext(
        complex_topology=data["complex_topology"],
        complex_coordinates=data["complex_coordinates"],
        rec_group=list(data["rec_group"]),
        lig_group=list(data["lig_group"]),
        anchors={key: int(value) for key, value in data["anchors"].items()},
        rmsd_order=list(data["rmsd_order"]),
        rmsd_atoms_bound={
            key: list(value) for key, value in data["rmsd_atoms_bound"].items()
        },
        rmsd_bulk=bulk,
        always_on=[AlwaysOn(**item) for item in data.get("always_on", [])],
    )
