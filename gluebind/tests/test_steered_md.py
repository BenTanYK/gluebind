"""Tests for steered-MD window scheduling (pure part; the MD run is integration)."""

from gluebind.simulation import separation_window_targets


def test_separation_window_targets_sorted_unique():
    assert separation_window_targets([1.0, 0.5, 1.0, 0.9]) == [0.5, 0.9, 1.0]


def test_separation_window_targets_rounds():
    assert separation_window_targets([0.90001, 0.9]) == [0.9]
