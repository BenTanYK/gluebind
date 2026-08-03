"""Tests for the package-wide MDAnalysis AMBER compatibility boundary."""

import sys
import types

from gluebind.system.mdanalysis import load_amber_universe


def _fake_mdanalysis(monkeypatch):
    calls = []

    def universe(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules, "MDAnalysis", types.SimpleNamespace(Universe=universe)
    )
    return calls


def test_loads_prm7_topology_with_explicit_parm7(monkeypatch):
    calls = _fake_mdanalysis(monkeypatch)

    load_amber_universe("system.prm7")

    assert calls == [(("system.prm7",), {"topology_format": "PARM7"})]


def test_loads_rst7_with_explicit_restart_format(monkeypatch):
    calls = _fake_mdanalysis(monkeypatch)

    load_amber_universe("system.prm7", "coordinates.rst7")

    assert calls == [
        (
            ("system.prm7", "coordinates.rst7"),
            {"topology_format": "PARM7", "format": "RESTRT"},
        )
    ]


def test_leaves_trajectory_format_to_mdanalysis(monkeypatch):
    calls = _fake_mdanalysis(monkeypatch)

    load_amber_universe("system.prm7", "trajectory.dcd")

    assert calls == [(("system.prm7", "trajectory.dcd"), {"topology_format": "PARM7"})]
