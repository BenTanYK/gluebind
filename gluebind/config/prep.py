"""System-preparation configuration (BioSimSpace front-end).

Proteins are supplied to gluebind already parameterised (prm7/rst7) — commonly
because E3 ligases contain zinc that requires bespoke parameterisation (e.g.
ZAFF in tleap) which is outside gluebind's scope. Consequently only the
small-molecule glue is parameterised here, and this config governs the glue
force field, solvation, and the pre-equilibration / equilibration schedule.
"""

from __future__ import annotations

import pydantic


class PrepConfig(pydantic.BaseModel):
    """Parameters for glue parameterisation, solvation and equilibration."""

    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    glue_forcefield: str = "openff_unconstrained_2.2.1"
    """Small-molecule force field. Validated at runtime against
    ``BSS.Parameters.forceFields()`` with a clear error if unavailable (so an
    env without OpenFF 2.2.1 fails fast rather than silently). Set ``gaff2`` to
    reproduce the paper (GAFF2 + AM1-BCC)."""
    water_model: str = "tip3p"
    box_type: str = "truncatedOctahedron"
    box_padding_angstrom: float = 15.0
    ion_concentration_M: float = 0.15
    neutralise: bool = True

    minimisation_steps: int = 10000
    nvt_heat_ns: float = 0.2
    """Short NVT temperature ramp (0 K -> production temperature)."""
    npt_ns: float = 0.4
    """Short NPT equilibration to relax the box volume."""
    equilibration_ns: float = 5.0
    """Long NVT production equilibration at the production temperature. Single run
    (no ensemble): the paper found triplicate equilibration trajectories to be
    essentially identical. Source of the RMSF/anchor-selection trajectory, the
    Boresch angle distributions and the bound-state starting structure. (The
    paper uses 100 ns.)"""

    @pydantic.field_validator(
        "box_padding_angstrom",
        "nvt_heat_ns",
        "npt_ns",
        "equilibration_ns",
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @pydantic.field_validator("ion_concentration_M")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("ion_concentration_M must be >= 0")
        return v

    @pydantic.field_validator("minimisation_steps")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("minimisation_steps must be > 0")
        return v
