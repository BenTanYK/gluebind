"""Automated geometric route binding free energy calculations for ternary complexes"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("gluebind")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ["__version__"]
