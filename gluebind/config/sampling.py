"""Umbrella-sampling protocol configuration.

Holds the global MD parameters and, per collective-variable type (Boresch /
RMSD / separation), a :class:`WindowSampling` schedule. Defaults reproduce the
published protocol, so a typical user overrides little or nothing.

A :class:`WindowSampling` is the *single source of truth* for its force
constant: both the OpenMM bias potential and the WHAM metadata file read ``k``
from here, which structurally prevents the simulation/analysis unit mismatch
that affected the original template scripts.

Force-constant units are per-CV-type:

* Boresch angles/dihedrals — kcal/mol/rad²
* RMSD and separation — kcal/mol/Å²
"""

from __future__ import annotations

import pydantic

_CONFIG = pydantic.ConfigDict(extra="forbid", validate_assignment=True)


class WindowSampling(pydantic.BaseModel):
    """Sampling schedule and bias strength for one collective variable."""

    model_config = _CONFIG

    force_constant: float
    sampling_time_ns: float
    equil_discard_ns: float = 0.0
    window_spacing: float | None = None
    """Spacing between window centres. ``None`` when ``centres`` is given, or
    when the range is derived at runtime (e.g. Boresch from the unrestrained-MD
    distribution, separation from steered-MD save points)."""
    window_min: float | None = None
    window_max: float | None = None
    coarse_from: float | None = None
    """Transition point beyond which ``coarse_spacing`` replaces ``window_spacing``
    (the separation CV's two-phase schedule: fine near contact, coarse further
    out)."""
    coarse_spacing: float | None = None
    """Wider window spacing applied beyond ``coarse_from``."""
    smd_snapshot_spacing: float | None = None
    """(separation) Spacing (nm) of the SMD snapshot grid. Saved densely — finer
    than the US window schedule — so windows can be added later (up to
    ``smd_capture_max``) without re-running steered MD. US window centres must fall
    on this grid."""
    smd_capture_max: float | None = None
    """(separation) Largest separation (nm) SMD captures snapshots for (US windows
    can then be extended out to here on demand)."""
    smd_pull_margin: float | None = None
    """(separation) Extra distance (nm) SMD steers past ``smd_capture_max`` to
    guarantee the final snapshot is reached."""
    centres: list[float] | None = None
    """Explicit window centres — an escape hatch overriding spacing/range."""
    auto_extend: bool = False
    """Add windows beyond ``window_max`` until the ΔG contribution converges."""
    overrides: dict[str, dict] = pydantic.Field(default_factory=dict)
    """Per-stage overrides, keyed by stage name (e.g. ``"BD1_bulk"``)."""

    @pydantic.field_validator("force_constant", "sampling_time_ns")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @pydantic.field_validator("equil_discard_ns")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("equil_discard_ns must be >= 0")
        return v

    @pydantic.field_validator(
        "window_spacing",
        "coarse_spacing",
        "smd_snapshot_spacing",
        "smd_capture_max",
        "smd_pull_margin",
    )
    @classmethod
    def _positive_optional(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("must be > 0 when set")
        return v

    def resolved(self, stage_name: str) -> "WindowSampling":
        """Return this schedule with any per-stage override merged in.

        Unknown override keys raise (the merged dict is re-validated with
        ``extra="forbid"``), so a typo in an override is an error, not a
        silent no-op.
        """
        override = self.overrides.get(stage_name)
        if not override:
            return self
        data = self.model_dump()
        data.update(override)
        data["overrides"] = {}
        return WindowSampling.model_validate(data)


def _boresch_default() -> WindowSampling:
    # 1 ns unrecorded equilibration + 5 ns recorded sampling (the paper protocol);
    # RED is not applied to Boresch — this fixed discard is used instead.
    return WindowSampling(
        force_constant=100.0,
        window_spacing=0.1,
        sampling_time_ns=5.0,
        equil_discard_ns=1.0,
    )


def _rmsd_default() -> WindowSampling:
    return WindowSampling(
        force_constant=5.0,
        window_spacing=0.2,
        window_min=0.0,
        window_max=2.2,
        sampling_time_ns=20.0,
        auto_extend=True,
    )


def _separation_default() -> WindowSampling:
    # Two-phase schedule (nm): 0.05 nm windows over 0.90-2.10 nm (fine, near
    # contact), then 0.10 nm beyond up to window_max. These fall on the 0.05 nm
    # SMD snapshot grid, so windows can be added later without re-running SMD.
    # window_max defaults to 3.0 nm (compute-saving); the plateau check flags if
    # more windows (up to the 4.0 nm SMD capture range) are needed.
    return WindowSampling(
        force_constant=10.0,
        window_min=0.90,
        window_max=3.0,
        window_spacing=0.05,
        coarse_from=2.10,
        coarse_spacing=0.10,
        smd_snapshot_spacing=0.05,
        smd_capture_max=4.0,
        smd_pull_margin=0.5,
        sampling_time_ns=30.0,
    )


class SamplingConfig(pydantic.BaseModel):
    """Global MD parameters plus the three per-CV sampling schedules."""

    model_config = _CONFIG

    timestep_fs: float = 4.0
    hmr_factor: float = 1.5
    pme_cutoff_nm: float = 1.0
    temperature_K: float = 300.0
    """Uniform production temperature (K) used for the MD, WHAM and the
    free-energy integrals — kept consistent throughout (the paper protocol)."""
    sample_interval_steps: int = 125
    ensemble_size: int = 3
    """Number of independent replicate simulations per window."""

    boresch: WindowSampling = pydantic.Field(default_factory=_boresch_default)
    rmsd: WindowSampling = pydantic.Field(default_factory=_rmsd_default)
    separation: WindowSampling = pydantic.Field(default_factory=_separation_default)

    @pydantic.field_validator(
        "timestep_fs", "hmr_factor", "pme_cutoff_nm", "temperature_K"
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @pydantic.field_validator("sample_interval_steps", "ensemble_size")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    def for_cv(self, cv_type: str, stage_name: str) -> WindowSampling:
        """Resolve the sampling schedule for a stage of a given CV type.

        ``cv_type`` is one of ``"boresch"``, ``"rmsd"``, ``"separation"``.
        """
        if cv_type not in ("boresch", "rmsd", "separation"):
            raise ValueError(f"unknown cv_type {cv_type!r}")
        base: WindowSampling = getattr(self, cv_type)
        return base.resolved(stage_name)
