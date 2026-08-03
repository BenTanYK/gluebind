"""Package-level surface: version resolution and public re-exports."""

from importlib.metadata import version

import gluebind


def test_version_matches_installed_metadata():
    # __version__ is read from the installed distribution (setuptools_scm), not a
    # hardcoded string, so it never drifts from the built package.
    assert gluebind.__version__ == version("gluebind")
    assert gluebind.__version__ != "0.0.0+unknown"  # the not-installed fallback


def test_public_api_is_importable():
    for name in gluebind.__all__:
        assert hasattr(gluebind, name), f"{name} in __all__ but not exported"
