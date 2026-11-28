"""Tests for which replicates own a stage, and the run_NN directory name."""

import pytest

from gluebind.config import SamplingConfig
from gluebind.replicate import (
    equilibration_replicates,
    replicate_dirname,
    smd_replicates,
)


def test_replicate_dirname_is_zero_padded_and_one_based():
    assert replicate_dirname(1) == "run_01"
    assert replicate_dirname(12) == "run_12"
    for bad in (0, -1):
        with pytest.raises(ValueError, match="1-based"):
            replicate_dirname(bad)


@pytest.mark.parametrize(
    ("ensemble_from", "equilibration", "smd"),
    [
        ("windows", (1,), (1,)),
        ("smd", (1,), (1, 2, 3)),
        ("equilibration", (1, 2, 3), (1, 2, 3)),
    ],
)
def test_each_stage_fans_out_from_its_point_in_the_pipeline(
    ensemble_from, equilibration, smd
):
    sampling = SamplingConfig(ensemble_from=ensemble_from, ensemble_size=3)
    assert equilibration_replicates(sampling) == equilibration
    assert smd_replicates(sampling) == smd


def test_a_shared_stage_still_lives_under_run_01():
    # one directory layout serves both modes, so nothing branches on the mode
    sampling = SamplingConfig()  # the default: nothing fanned out
    assert equilibration_replicates(sampling) == (1,)
    assert smd_replicates(sampling) == (1,)
    assert replicate_dirname(equilibration_replicates(sampling)[0]) == "run_01"
