"""Restraint-definition models: the system-specific structural input.

These models describe *which* atoms are restrained for a particular complex —
the RMSD collective-variable regions, any always-on structural restraint (e.g.
the DDB1-binding helix bundle of DCAF16 that must be held in both the restrained
and released states so its contribution cancels), and the Boresch anchor
override. They carry no protocol/sampling numbers; force constants and window
schedules live in :class:`gluebind.config.sampling.SamplingConfig`.

Selections are written as MDAnalysis-style strings (e.g. ``"resid 45-98
169-216"``) rather than raw index lists: less error-prone to author and read,
and resolved + echoed back by name before any simulation runs.

If ``RestraintsConfig.rmsd_cvs`` is left empty, gluebind falls back to an
all-Cα RMSD per protein, folding the glue heavy atoms into whichever protein it
is assigned to (see :class:`gluebind.config.calculation.GlueInput`). That
default reproduces a simple single-domain complex exactly; it is *not*
appropriate for multi-domain targets (e.g. tandem bromodomains), where explicit
per-domain CVs are required.
"""

from __future__ import annotations

from typing import Literal

import pydantic

State = Literal["bound", "bulk"]
Protein = Literal["target", "receptor"]

_CONFIG = pydantic.ConfigDict(extra="forbid", validate_assignment=True)


class RmsdCVSpec(pydantic.BaseModel):
    """One RMSD collective variable applied to a region of one protein.

    Each spec becomes one or more umbrella-sampling *stages* — one per entry in
    ``states`` — named ``"{name}_{state}"`` (e.g. ``"BD1_bound"``).
    """

    model_config = _CONFIG

    name: str
    """Short identifier; also the stage directory name. Must be unique."""
    protein: Protein
    """Which input protein this CV lives on (``target`` or ``receptor``). The
    selection is resolved against *that* input topology and mapped into the
    assembled complex, so it is immune to any re-indexing BioSimSpace applies
    during assembly (see :mod:`gluebind.system.atom_map`)."""
    selection: str
    """MDAnalysis selection, resolved against the ``protein``'s **input**
    ``.prm7`` (the numbering the user authored against), then mapped to the
    complex."""
    states: list[State] = ["bound", "bulk"]
    """Which thermodynamic states this CV is sampled in."""
    include_glue: bool = False
    """Whether to add the glue heavy atoms to this CV's atom group."""

    @pydantic.field_validator("states")
    @classmethod
    def _non_empty_unique_states(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("states must not be empty")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate states: {v}")
        return v


class AlwaysOnRestraint(pydantic.BaseModel):
    """A harmonic RMSD restraint present in *every* stage's system.

    Used to substitute for a missing structural partner (e.g. the DDB1 scaffold
    for DCAF16). Because it is applied identically in the restrained and
    released states, its free-energy contribution cancels and it does not enter
    the final estimate.
    """

    model_config = _CONFIG

    protein: Protein
    """Which input protein the restrained atoms live on; the selection is
    resolved against that input topology and mapped into the complex/bulk."""
    selection: str
    force_constant: float
    """Harmonic force constant in kcal/mol/Å²."""

    @pydantic.field_validator("force_constant")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("force_constant must be > 0")
        return v


class BoreschSpec(pydantic.BaseModel):
    """Boresch orientational-restraint anchor selection.

    The two *bonded* anchors (a, A) are always the target/receptor interface
    centres of mass and are not configured here. This spec only concerns the
    four *non-bonded* anchors (b, c in the receptor; B, C in the ligand), which
    default to automatic selection from RMSF minima.
    """

    model_config = _CONFIG

    anchors: dict[str, int] | Literal["auto"] = "auto"
    """Either ``"auto"`` or an explicit mapping with keys ``b``, ``c``, ``B``,
    ``C`` to 0-indexed atom indices."""

    @pydantic.field_validator("anchors")
    @classmethod
    def _valid_anchor_keys(cls, v: object) -> object:
        if v == "auto":
            return v
        expected = {"b", "c", "B", "C"}
        if not isinstance(v, dict) or set(v) != expected:
            raise ValueError(
                f"explicit anchors must have exactly keys {sorted(expected)}"
            )
        return v


class RestraintsConfig(pydantic.BaseModel):
    """The full restraint definition for a calculation."""

    model_config = _CONFIG

    rmsd_cvs: list[RmsdCVSpec] = pydantic.Field(default_factory=list)
    """RMSD CV regions. Empty ⇒ all-Cα default (see module docstring)."""
    rmsd_order: list[str] = pydantic.Field(default_factory=list)
    """Sequential bound-state application order, by CV name. Empty ⇒ the order
    of ``rmsd_cvs``."""
    always_on: list[AlwaysOnRestraint] = pydantic.Field(default_factory=list)
    boresch: BoreschSpec = pydantic.Field(default_factory=BoreschSpec)

    @pydantic.model_validator(mode="after")
    def _check_consistency(self) -> "RestraintsConfig":
        names = [cv.name for cv in self.rmsd_cvs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate rmsd_cv names: {names}")
        unknown = set(self.rmsd_order) - set(names)
        if unknown:
            raise ValueError(
                f"rmsd_order references unknown CV names: {sorted(unknown)}"
            )
        return self

    @property
    def uses_default_all_ca(self) -> bool:
        """True when no explicit RMSD CVs are given (all-Cα fallback applies)."""
        return not self.rmsd_cvs
