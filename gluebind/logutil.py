"""Driver-side logging.

A gluebind calculation is a long-lived, lightweight driver (a3fe-style) that may
run for days while it submits and waits on cluster jobs. This module gives the
driver a place to record what it is doing so a detached run (``nohup``) leaves a
readable trail.

Library convention: gluebind only ever *emits* to the ``"gluebind"`` logger (with
a :class:`~logging.NullHandler` so importing the package is silent). It does not
configure the root logger. :func:`add_file_handler` is an opt-in convenience that
:meth:`~gluebind.runners.calculation.Calculation.run` calls to drop a
``gluebind.log`` into the run directory; an application embedding gluebind can
instead attach its own handlers to the ``"gluebind"`` logger.
"""

from __future__ import annotations

import logging
import pathlib

LOGGER_NAME = "gluebind"

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the gluebind logger (or a ``gluebind.<name>`` child)."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def add_file_handler(
    base_dir: str | pathlib.Path,
    *,
    logger_name: str = LOGGER_NAME,
    level: int = logging.INFO,
    filename: str = "gluebind.log",
) -> logging.Handler:
    """Attach a file handler under ``base_dir`` to a gluebind logger.

    ``logger_name`` defaults to the root ``"gluebind"`` logger; a per-calculation
    child (e.g. ``"gluebind.calc.<name>"``) keeps each run's log isolated while
    still propagating up to any handler on the root (so a ``CalcSet`` can hold one
    aggregate log). Idempotent: re-calling for the same file returns the existing
    handler rather than duplicating log lines (so a resumed run appends to one
    log). Returns the handler.
    """
    base_dir = pathlib.Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = (base_dir / filename).resolve()

    logger = logging.getLogger(logger_name)
    if logger.level == logging.NOTSET or logger.level > level:
        # ensure INFO records are emitted, without lowering an already-finer level
        logger.setLevel(level)
    for handler in logger.handlers:
        if getattr(handler, "_gluebind_path", None) == str(path):
            return handler

    handler = logging.FileHandler(path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler._gluebind_path = str(path)  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return handler
