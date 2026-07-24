"""Shared fixtures + tier gating for the test suite.

The fast unit suite needs nothing beyond the core deps. The ``integration`` tier
(1FAP) needs real MD/cheminformatics deps and often a GPU — neither guaranteed on
a given box. Rather than fail, tests request the dependency fixtures below
(``bss``, ``red_mod``, ``wham_binary``) and *skip* when the dependency is absent,
and ``gpu``-marked tests are auto-skipped by the collection hook. So
``make test-integration`` on a partial env runs what it can and skips the rest,
filling in as the env is completed.

Scientific validation on the real CRBN/BRD4 systems is intentionally *not* a test
tier — it is a user workflow (run the package on those inputs), so there is no
external-input gating here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest

DATA_DIR = pathlib.Path(__file__).parent / "data"


# ---- fixture data ----------------------------------------------------------


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def fap_inputs() -> dict:
    """Paths to the vendored 1FAP fixture (FKBP12·rapamycin·FRB, + crystal waters).

    ``waters`` is a separate optional input (see ``inputs.waters``); a dry-case
    test simply omits it.
    """
    d = DATA_DIR / "1fap"
    return {
        "receptor": {
            "prm7": str(d / "receptor.prm7"),
            "rst7": str(d / "receptor.rst7"),
        },
        "target": {"prm7": str(d / "target.prm7"), "rst7": str(d / "target.rst7")},
        "glue": {"sdf": str(d / "glue.sdf"), "assign_to": "receptor"},
        "waters": {"prm7": str(d / "waters.prm7"), "rst7": str(d / "waters.rst7")},
    }


# ---- dependency gating (skip when absent) ----------------------------------


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture
def bss():
    """Imported BioSimSpace, or skip if it (or its MD stack) isn't installed."""
    if not _installed("BioSimSpace"):
        pytest.skip("BioSimSpace not installed (see devtools/envs/test.yaml)")
    import BioSimSpace as BSS

    return BSS


@pytest.fixture
def red_mod():
    """Imported ``red`` (red-molsim), or skip if not installed."""
    if not _installed("red"):
        pytest.skip("red (red-molsim) not installed")
    import red

    return red


@pytest.fixture
def wham_binary() -> str:
    """Path to the Grossfield ``wham`` binary, or skip if not on PATH."""
    exe = shutil.which("wham")
    if exe is None:
        pytest.skip("wham binary not on PATH (build with `make wham`)")
    return exe


def _has_cuda() -> bool:
    try:
        from openmm import Platform

        return any(
            Platform.getPlatform(i).getName() == "CUDA"
            for i in range(Platform.getNumPlatforms())
        )
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``gpu`` tests without a CUDA GPU — the marker alone gates them,
    no per-test boilerplate."""
    if _has_cuda():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU / OpenMM CUDA platform")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
