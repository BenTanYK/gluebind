"""Tests for the calculation and SLURM configuration models."""

import pytest

from gluebind import CalculationConfig, SlurmConfig
from gluebind.config import SamplingConfig

MIN_INPUTS = {
    "target": {"prm7": "t.prm7", "rst7": "t.rst7"},
    "receptor": {"prm7": "r.prm7", "rst7": "r.rst7"},
    "glue": {"sdf": "g.sdf", "assign_to": "receptor"},
}


def _min_cfg() -> CalculationConfig:
    return CalculationConfig.model_validate({"inputs": MIN_INPUTS})


def test_zero_config_defaults_to_all_ca():
    assert _min_cfg().restraints.uses_default_all_ca is True


def test_roundtrip_and_hash_stable(tmp_path):
    cfg = _min_cfg()
    cfg.dump_resolved(tmp_path)
    loaded = CalculationConfig.load(tmp_path / "config_resolved.yaml")
    assert loaded == cfg
    assert loaded.config_hash == cfg.config_hash


def test_waters_input_optional_and_paths_resolve(tmp_path):
    # waters is absent by default ...
    assert _min_cfg().inputs.waters is None
    # ... and when present, its prm7/rst7 resolve alongside the other inputs.
    cfg = CalculationConfig.model_validate(
        {"inputs": {**MIN_INPUTS, "waters": {"prm7": "xtal.prm7", "rst7": "xtal.rst7"}}}
    )
    resolved = cfg.with_resolved_input_paths(tmp_path)
    assert resolved.inputs.waters.prm7 == str((tmp_path / "xtal.prm7").resolve())
    assert resolved.inputs.waters.rst7 == str((tmp_path / "xtal.rst7").resolve())
    # the proteins still resolve too (waters didn't disturb the others)
    assert resolved.inputs.target.prm7 == str((tmp_path / "t.prm7").resolve())


def test_config_hash_changes_with_content():
    a = _min_cfg()
    b = _min_cfg()
    b.sampling.rmsd.force_constant = 7.0
    assert a.config_hash != b.config_hash


def test_sampling_override_resolves_per_stage():
    s = SamplingConfig()
    s.rmsd.overrides = {"BD1_bulk": {"sampling_time_ns": 40.0, "window_max": 4.0}}
    resolved = s.for_cv("rmsd", "BD1_bulk")
    assert resolved.sampling_time_ns == 40.0
    assert resolved.window_max == 4.0
    # a stage with no override gets the base schedule
    assert s.for_cv("rmsd", "other").sampling_time_ns == 20.0


def test_bad_override_key_rejected():
    s = SamplingConfig()
    s.rmsd.overrides = {"BD1_bulk": {"not_a_field": 1}}
    with pytest.raises(ValueError):
        s.for_cv("rmsd", "BD1_bulk")


def test_default_force_constants():
    s = SamplingConfig()
    assert s.boresch.force_constant == 100.0
    assert s.rmsd.force_constant == 5.0
    assert s.separation.force_constant == 10.0


def test_extra_top_level_key_forbidden():
    with pytest.raises(ValueError):
        CalculationConfig.model_validate({"inputs": MIN_INPUTS, "bogus": 1})


def test_rmsd_order_must_be_full_permutation():
    from gluebind.config.restraints import RestraintsConfig, RmsdCVSpec

    cvs = [
        RmsdCVSpec(name="A", protein="target", selection="resid 1"),
        RmsdCVSpec(name="B", protein="target", selection="resid 2"),
    ]
    RestraintsConfig(rmsd_cvs=cvs, rmsd_order=["B", "A"])  # full permutation: ok
    with pytest.raises(ValueError, match="permutation"):
        RestraintsConfig(rmsd_cvs=cvs, rmsd_order=["A"])  # partial subset drops B


def test_rmsd_cv_states_must_be_symmetric():
    from gluebind.config.restraints import RestraintsConfig, RmsdCVSpec

    # both states (the default): a valid confinement cycle
    RestraintsConfig(
        rmsd_cvs=[RmsdCVSpec(name="A", protein="target", selection="resid 1")]
    )
    # asymmetric states (sampled in only one leg) break the cycle -> rejected
    for bad in (["bound"], ["bulk"]):
        with pytest.raises(ValueError, match="both 'bound' and 'bulk'"):
            RestraintsConfig(
                rmsd_cvs=[
                    RmsdCVSpec(
                        name="A", protein="target", selection="resid 1", states=bad
                    )
                ]
            )


def test_always_on_requires_explicit_rmsd_cvs():
    from gluebind.config.restraints import (
        AlwaysOnRestraint,
        RestraintsConfig,
        RmsdCVSpec,
    )

    ao = AlwaysOnRestraint(
        protein="receptor", selection="resid 116-158", force_constant=100.0
    )
    with pytest.raises(ValueError, match="always_on restraints require"):
        RestraintsConfig(always_on=[ao])  # all-Cα default + always_on unsupported
    # with explicit rmsd_cvs it is fine
    RestraintsConfig(
        rmsd_cvs=[RmsdCVSpec(name="R", protein="receptor", selection="resid 1")],
        always_on=[ao],
    )


def test_config_hash_ignores_run_rmsd_us_but_catches_real_changes():
    # run_rmsd_us is a scope flag, not physics — flipping it must NOT change the
    # hash (so a separation-only run can be upgraded to full RMSD on resume)...
    base = CalculationConfig.model_validate({"inputs": MIN_INPUTS})
    base.sampling.run_rmsd_us = False
    h_off = base.config_hash
    base.sampling.run_rmsd_us = True
    assert base.config_hash == h_off
    # ...but a genuine physics change still shifts the hash (drift guard intact).
    base.sampling.rmsd.force_constant = 99.0
    assert base.config_hash != h_off


def test_restraint_atoms_modes_default_and_independent():
    from gluebind.config.restraints import RestraintsConfig

    r = RestraintsConfig()
    assert r.rmsd_atoms == "CA" and r.always_on_atoms == "CA"  # defaults
    r2 = RestraintsConfig(rmsd_atoms="CA", always_on_atoms="backbone")
    assert r2.rmsd_atoms == "CA" and r2.always_on_atoms == "backbone"  # independent
    with pytest.raises(ValueError):
        RestraintsConfig(rmsd_atoms="sidechain")  # only CA | backbone


def test_duplicate_rmsd_cv_names_rejected():
    with pytest.raises(ValueError):
        CalculationConfig.model_validate(
            {
                "inputs": MIN_INPUTS,
                "restraints": {
                    "rmsd_cvs": [
                        {"name": "A", "selection": "resid 1-2"},
                        {"name": "A", "selection": "resid 3-4"},
                    ]
                },
            }
        )


def test_rmsd_order_unknown_name_rejected():
    with pytest.raises(ValueError):
        CalculationConfig.model_validate(
            {
                "inputs": MIN_INPUTS,
                "restraints": {
                    "rmsd_cvs": [{"name": "A", "selection": "resid 1-2"}],
                    "rmsd_order": ["A", "ghost"],
                },
            }
        )


def test_slurm_render_and_submission_cmds(tmp_path):
    slurm = SlurmConfig(partition="gpu", extra_options={"nodelist": "n1,n2"})
    cmds = slurm.get_submission_cmds("python -c pass", tmp_path)
    assert cmds[0] == "sbatch"
    assert cmds[1] == f"--chdir={tmp_path}"
    script = (tmp_path / "gluebind.sh").read_text()
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --nodelist=n1,n2" in script


def test_slurm_config_yaml_roundtrip(tmp_path):
    slurm = SlurmConfig(partition="gpu", queue_len_lim=500)
    slurm.dump(tmp_path)
    assert SlurmConfig.load(tmp_path) == slurm
