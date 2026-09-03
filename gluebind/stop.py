"""Persistent, cross-process stop requests for GlueBind drivers.

The marker is deliberately separate from run state: a second process can stop a
detached Slurm driver without racing its next submission.  Advisory flock locks
make checking the marker, submitting a job, and recording its handle one
critical section.
"""

from __future__ import annotations

import contextlib
import fcntl
import pathlib
from collections.abc import Iterator, Sequence

STOP_FILENAME = ".gluebind-stop"


class StopRequested(RuntimeError):
    """Raised when a persistent stop request prevents further submission."""


class StopController:
    """Coordinate stop requests across one calculation and optional parent set."""

    def __init__(self, paths: Sequence[str | pathlib.Path]) -> None:
        self.paths = tuple(sorted({pathlib.Path(p).resolve() for p in paths}, key=str))

    @staticmethod
    def marker_path(base_dir: str | pathlib.Path) -> pathlib.Path:
        return pathlib.Path(base_dir).resolve() / STOP_FILENAME

    @staticmethod
    def _lock_path(marker: pathlib.Path) -> pathlib.Path:
        return marker.with_name(f"{marker.name}.lock")

    @contextlib.contextmanager
    def _locked(self, mode: int) -> Iterator[None]:
        files = []
        try:
            for marker in self.paths:
                marker.parent.mkdir(parents=True, exist_ok=True)
                f = self._lock_path(marker).open("a+")
                fcntl.flock(f.fileno(), mode)
                files.append(f)
            yield
        finally:
            for f in reversed(files):
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()

    def requested(self) -> bool:
        with self._locked(fcntl.LOCK_SH):
            return any(path.exists() for path in self.paths)

    def raise_if_requested(self) -> None:
        if self.requested():
            raise StopRequested("a persistent GlueBind stop request is active")

    @contextlib.contextmanager
    def submission_permit(self) -> Iterator[None]:
        """Permit one submission only while no stop marker is present.

        The caller must also persist the resulting backend handle inside this
        context, so a concurrent ``kill()`` sees and cancels it.
        """
        with self._locked(fcntl.LOCK_SH):
            if any(path.exists() for path in self.paths):
                raise StopRequested("a persistent GlueBind stop request is active")
            yield

    def request_stop(self, marker: str | pathlib.Path | None = None) -> None:
        """Persist a stop request after waiting for in-flight submissions."""
        markers = self.paths if marker is None else (pathlib.Path(marker).resolve(),)
        if not set(markers).issubset(self.paths):
            raise ValueError("stop marker is not managed by this stop controller")
        with self._locked(fcntl.LOCK_EX):
            for path in markers:
                path.touch(exist_ok=True)

    def clear_own_stop(self, own_marker: str | pathlib.Path) -> None:
        """Clear this object's marker, never an inherited parent-set marker."""
        own_marker = pathlib.Path(own_marker).resolve()
        if own_marker not in self.paths:
            raise ValueError(f"{own_marker} is not managed by this stop controller")
        with self._locked(fcntl.LOCK_EX):
            own_marker.unlink(missing_ok=True)
