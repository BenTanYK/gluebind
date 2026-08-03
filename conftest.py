"""Make the in-tree ``gluebind`` package importable during tests.

The package isn't pip-installed yet (pyproject/CI are added later via
cookiecutter-cms), so we put the repo root on ``sys.path`` for the test run.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
