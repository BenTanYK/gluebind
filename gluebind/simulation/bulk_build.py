"""Backend worker for isolated bulk-system extraction and solvation.

The worker owns every BioSimSpace/Sire operation required to turn the
equilibrated complex into one solvated receptor or target bulk reference.  This
keeps parallel :class:`CalcSet` driver threads out of BioSimSpace entirely.
"""

from __future__ import annotations

import pathlib

import pydantic

from gluebind.config.prep import PrepConfig

BULK_BUILD_SPEC_FILENAME = "bulk_build.json"
BULK_BUILD_RESULT_FILENAME = "result.json"


class BulkBuildSpec(pydantic.BaseModel):
    """Inputs required to extract and solvate one bulk reference species."""

    model_config = pydantic.ConfigDict(extra="forbid")

    complex_prm7: str
    complex_rst7: str
    molecule_indices: list[int]
    prep: PrepConfig
    output_dir: str

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "BulkBuildSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


class BulkBuildResult(pydantic.BaseModel):
    """Solvated AMBER inputs produced by :func:`run_bulk_build`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    solvated_prm7: str
    solvated_rst7: str

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "BulkBuildResult":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def bulk_build_launch_command(python: str = "python") -> list[str]:
    """Return the backend command for :func:`run_bulk_build`."""
    code = (
        "from gluebind.simulation.bulk_build import run_bulk_build; "
        "run_bulk_build('.')"
    )
    return [python, "-c", code]


def run_bulk_build(work_dir: str | pathlib.Path) -> None:
    """Extract one component from the complex, solvate it, and write a manifest."""
    import BioSimSpace as BSS

    work_dir = pathlib.Path(work_dir)
    spec = BulkBuildSpec.load(work_dir / BULK_BUILD_SPEC_FILENAME)
    if not spec.molecule_indices:
        raise ValueError("bulk build needs at least one molecule index")

    complex_system = BSS.IO.readMolecules([spec.complex_prm7, spec.complex_rst7])
    isolated = complex_system[spec.molecule_indices[0]]
    for index in spec.molecule_indices[1:]:
        isolated = isolated + complex_system[index]

    box_min, box_max = isolated.getAxisAlignedBoundingBox()
    dimensions = [hi - lo for lo, hi in zip(box_min, box_max, strict=False)]
    padding = spec.prep.bulk_box_padding_angstrom * BSS.Units.Length.angstrom
    edge = max(dimensions) + 2 * padding
    solvated = BSS.Solvent.solvate(
        spec.prep.water_model,
        molecule=isolated.toSystem() if hasattr(isolated, "toSystem") else isolated,
        box=BSS.Box.generateBoxParameters(spec.prep.box_type, edge)[0],
        is_neutral=spec.prep.neutralise,
        ion_conc=spec.prep.ion_concentration_M,
    )
    output_dir = pathlib.Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "solvated"
    BSS.IO.saveMolecules(str(prefix), solvated, ["prm7", "rst7"])
    BulkBuildResult(
        solvated_prm7=str(prefix.with_suffix(".prm7")),
        solvated_rst7=str(prefix.with_suffix(".rst7")),
    ).dump(work_dir / BULK_BUILD_RESULT_FILENAME)
