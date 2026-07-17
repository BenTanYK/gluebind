"""Persistent run state — ``.gluebind-state.json``.

Mirrors aqemia-abfe's ``RunState`` pattern: a single JSON file at the
calculation base directory, written by the orchestrator and read back by any
fresh process so the driver can be reconstructed and a run resumed after the
terminal / IDE / SSH session closes. The filesystem plus the live SLURM queue
are the source of truth; this file holds the *pointers* (job handles) and the
mid-run *determined values* that later stages depend on.

Differences from aqemia-abfe, by design:

* ``handles`` is nested one level deeper (``stage -> window -> [job id per
  repeat]``) for gluebind's umbrella-sampling hierarchy.
* It additionally persists ``anchors`` and ``boresch_eq_values`` — values
  computed mid-run (from RMSF and from Boresch-US PMF minima) that sequential
  downstream stages consume. Consequently the file is updated *incrementally*,
  and ``stage_status`` records progress.
* :meth:`save` is atomic (temp file + ``os.replace``), because gluebind writes
  it far more often than aqemia-abfe's write-once cadence.

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
import typing

import pydantic

STATE_FILENAME = ".gluebind-state.json"

# Bump when adding/removing/renaming a load-bearing field. Older files arrive
# without the new field (pydantic supplies the default); the first breaking
# change is the one that must grow a ``_migrate`` entry.
SCHEMA_VERSION = 1


class RunState(pydantic.BaseModel):
    """What must survive between setup / run / analyse invocations."""

    model_config = pydantic.ConfigDict(validate_assignment=True)

    schema_version: int = SCHEMA_VERSION
    calc_id: str
    submitted_at: str
    config_hash: str
    config_path: str

    # Determined mid-run and reused by later stages (no aqemia-abfe analogue).
    anchors: dict | None = None
    boresch_eq_values: dict[str, float] = pydantic.Field(default_factory=dict)
    stage_status: dict[str, str] = pydantic.Field(default_factory=dict)

    # Opaque backend job handles: stage name -> window label -> [id per repeat].
    handles: dict[str, dict[str, list[str]]] = pydantic.Field(default_factory=dict)

    # Per-backend escape hatch (mirrors aqemia-abfe's RunState.backend_extra):
    # e.g. an AWS Batch backend stashes its batch_id and S3 prefix here so a
    # fresh process can reconstruct where its outputs live. Opaque to the core.
    backend_extra: dict = pydantic.Field(default_factory=dict)

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

        fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".gluebind-state.", suffix=".tmp")
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
        """Load the state file from ``run_dir``, migrating older schemas."""
        path = pathlib.Path(run_dir) / STATE_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No state file at {path}. Has this calculation been submitted?"
            )
        with open(path) as f:
            raw = json.load(f)

        on_disk = raw.get("schema_version", 1)
        if on_disk > SCHEMA_VERSION:
            raise ValueError(
                f"State file at {path} has schema_version={on_disk}, but this "
                f"gluebind build only knows v{SCHEMA_VERSION}. It was written by a "
                "newer release — upgrade gluebind, or delete the state file and resubmit."
            )
        if on_disk < SCHEMA_VERSION:
            raw = _migrate(raw, from_version=on_disk)

        try:
            return cls.model_validate(raw)
        except pydantic.ValidationError as e:
            raise ValueError(
                f"Could not load {path} as RunState v{SCHEMA_VERSION}. The file may be "
                "from an incompatible schema that was never migrated — delete it and "
                f"resubmit, or inspect it by hand.\n\nUnderlying error:\n{e}"
            ) from e


def _migrate(raw: dict[str, typing.Any], *, from_version: int) -> dict[str, typing.Any]:
    """Upgrade a state-file dict from an older schema to ``SCHEMA_VERSION``.

    Each block is a one-step migration (v_n -> v_{n+1}); add a block whenever
    ``SCHEMA_VERSION`` is bumped. No migrations exist yet — the helper is here so
    the next field change has a home.
    """
    return raw


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds resolution)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
