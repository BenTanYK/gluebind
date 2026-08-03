import importlib


def test_gluebind():
    assert importlib.import_module("gluebind") is not None
