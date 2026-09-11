"""Persistent run state — ``.gluebind-state.json``.

Record a single JSON file at the calculation base directory, written by the
orchestrator and read back by any fresh process so the driver can be
reconstructed and a run resumed after the terminal / IDE / SSH session closes.
The filesystem plus the live SLURM queue are the source of truth; this file
holds the *pointers* (job handles) and the mid-run *determined values* that
later stages depend on.

Completion is deliberately *not* stored as truth here; it is reconciled live
against ``squeue`` and the on-disk output files.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import pathlib
import tempfile

import pydantic

STATE_FILENAME = ".gluebind-state.json"

# Bump when adding/removing/renaming a load-bearing field. Older artifacts are
# deliberately rejected: GlueBind does not migrate run state across schemas.
SCHEMA_VERSION = 1


class RunState(pydantic.BaseModel):
    """What must survive between setup / run / analyse invocations."""

    model_config = pydantic.ConfigDict(validate_assignment=True)

    schema_version: int = SCHEMA_VERSION
    calc_id: str
    submitted_at: str
    config_hash: str
    config_path: str

    # Determined mid-run and reused by later stages.
    anchors: dict[str, int] | None = None
    boresch_eq_values: dict[str, float] = pydantic.Field(default_factory=dict)
    stage_status: dict[str, str] = pydantic.Field(default_factory=dict)

    # Opaque backend job handles: stage name -> window label -> [id per repeat].
    # ``_auxiliary`` stores non-window jobs as label -> [id].
    handles: dict[str, dict[str, list[str]]] = pydantic.Field(default_factory=dict)

    # Per-backend escape hatch: e.g. an AWS Batch backend can stash its batch ID
    # and S3 prefix here so a fresh process can reconstruct where outputs live.
    # This data is opaque to the core workflow.
    backend_extra: dict = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("anchors", mode="before")
    @classmethod
    def _valid_anchors(cls, value: object) -> object:
        """Reject malformed persisted Boresch-anchor mappings."""
        if value is None:
            return value
        expected = {"b", "c", "B", "C"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(
                f"anchors must have exactly keys {sorted(expected)} or be null"
            )
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in value.values()
        ):
            raise ValueError("all persisted anchor indices must be integers")
        return value

    def save(self, run_dir: str | pathlib.Path) -> pathlib.Path:
        """Atomically write the state file into ``run_dir``.

        Writes to a temporary file in the same directory, fsyncs, then
        ``os.replace`` — so an interrupted write can never leave a truncated
        state file.
        """
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / STATE_FILENAME
        payload = json.dumps(self.model_dump(), indent=2)

        fd, tmp = tempfile.mkstemp(
            dir=run_dir, prefix=".gluebind-state.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return path

    @classmethod
    def load(cls, run_dir: str | pathlib.Path) -> "RunState":
        """Load a state file with the exact supported schema version."""
        path = pathlib.Path(run_dir) / STATE_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No state file at {path}. Has this calculation been submitted?"
            )
        with open(path) as f:
            raw = json.load(f)

        on_disk = raw.get("schema_version")
        if on_disk != SCHEMA_VERSION:
            raise ValueError(
                f"State file at {path} has schema_version={on_disk!r}, but this "
                f"gluebind build requires v{SCHEMA_VERSION}. Start a fresh run."
            )

        try:
            return cls.model_validate(raw)
        except pydantic.ValidationError as e:
            raise ValueError(
                f"Could not load {path} as RunState v{SCHEMA_VERSION}. The file may be "
                "from an incompatible schema that was never migrated — delete it and "
                f"resubmit, or inspect it by hand.\n\nUnderlying error:\n{e}"
            ) from e
def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds resolution)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
