"""Tests for driver-side logging setup."""

import logging

from gluebind.logutil import LOGGER_NAME, add_file_handler, get_logger


def _detach(handler):
    logging.getLogger(LOGGER_NAME).removeHandler(handler)
    handler.close()


def test_add_file_handler_writes_log(tmp_path):
    handler = add_file_handler(tmp_path)
    try:
        get_logger("test").info("hello driver")
        handler.flush()
        assert (tmp_path / "gluebind.log").exists()
        assert "hello driver" in (tmp_path / "gluebind.log").read_text()
    finally:
        _detach(handler)


def test_add_file_handler_is_idempotent(tmp_path):
    h1 = add_file_handler(tmp_path)
    h2 = add_file_handler(tmp_path)
    try:
        assert h1 is h2  # same file -> no duplicate handler (no doubled log lines)
    finally:
        _detach(h1)


def test_child_logger_propagates_to_gluebind(tmp_path):
    handler = add_file_handler(tmp_path)
    try:
        # a child ("gluebind.calculation") must reach the file handler on the parent
        get_logger("calculation").info("stage foo: complete")
        handler.flush()
        assert "stage foo: complete" in (tmp_path / "gluebind.log").read_text()
    finally:
        _detach(handler)


def test_per_calc_logs_are_isolated_but_aggregate_at_root(tmp_path):
    # Mirrors CalcSet: a root aggregate handler plus a per-calc child handler each.
    root = add_file_handler(tmp_path, logger_name=LOGGER_NAME)
    ha = add_file_handler(tmp_path / "A", logger_name="gluebind.calc.A")
    hb = add_file_handler(tmp_path / "B", logger_name="gluebind.calc.B")
    try:
        get_logger("calc.A").info("A only")
        get_logger("calc.B").info("B only")
        for h in (root, ha, hb):
            h.flush()
        log_a = (tmp_path / "A" / "gluebind.log").read_text()
        log_b = (tmp_path / "B" / "gluebind.log").read_text()
        # each per-calc log holds only its own messages (no cross-contamination)
        assert "A only" in log_a and "B only" not in log_a
        assert "B only" in log_b and "A only" not in log_b
        # the root aggregate holds both (via propagation)
        aggregate = (tmp_path / "gluebind.log").read_text()
        assert "A only" in aggregate and "B only" in aggregate
    finally:
        for h in (root, ha, hb):
            _detach(h)
