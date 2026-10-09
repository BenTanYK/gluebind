"""Tests for the SpecBuilder assembly (context -> WindowSpec) and its wiring into
the runner. The MDAnalysis resolver (build_restraint_context) is integration-only."""

import sys

import numpy as np
import pytest

from gluebind import CalculationConfig, SpecBuilder
from gluebind.backend import LocalBackend, Scheduler
from gluebind.runners import Calculation
from gluebind.simulation import WindowSpec
from gluebind.spec_builder import BulkTarget, RestraintContext

INPUTS = {
    "target": {"prm7": "t.prm7", "rst7": "t.rst7"},
    "receptor": {"prm7": "r.prm7", "rst7": "r.rst7"},
    "glue": {"sdf": "g.sdf", "assign_to": "receptor"},
}


def _context():
    return RestraintContext(
        complex_topology="complex.prm7",
        complex_coordinates="complex.rst7",
        rec_group=[1, 2, 3],
        lig_group=[4, 5, 6],
        anchors={"b": 10, "c": 11, "B": 12, "C": 13},
        rmsd_order=["receptor", "target"],
        rmsd_atoms_bound={"receptor": [1, 2, 3], "target": [4, 5, 6]},
        rmsd_bulk={
            # bulk atom i is complex atom 100 + i (receptor) / 200 + i (target)
            "receptor": BulkTarget(
                "receptor_bulk.prm7",
                "receptor_bulk.rst7",
                [0, 1, 2],
                to_complex=list(range(100, 110)),
            ),
            "target": BulkTarget(
                "target_bulk.prm7",
                "target_bulk.rst7",
                [0, 1, 2],
                to_complex=list(range(200, 210)),
            ),
        },
    )


def _config():
    return CalculationConfig.model_validate({"inputs": INPUTS})


def _builder(smd_frames_dir=None):
    return SpecBuilder(_context(), _config(), smd_frames_dir=smd_frames_dir)


# ---- Boresch ---------------------------------------------------------------


def test_boresch_spec():
    cfg = _config()
    spec = _builder()(
        cv_type="boresch",
        stage_name="thetaB",
        dof="thetaB",
        cv_centre=1.05,
        replicate=1,
        boresch_eq_values={"thetaA": 0.9},
    )
    assert spec.cv_type == "boresch" and spec.dof == "thetaB"
    assert spec.topology == "complex.prm7" and spec.coordinates == "complex.rst7"
    assert spec.force_constant == cfg.sampling.boresch.force_constant
    assert "save_state_data" not in spec.model_dump()
    assert spec.restraints["boresch"]["fixed"] == {"thetaA": 0.9}
    assert spec.restraints["boresch"]["anchors"] == {"b": 10, "c": 11, "B": 12, "C": 13}
    assert {r["name"] for r in spec.restraints["rmsd"]} == {"receptor", "target"}
    assert all(r["sampled"] is False for r in spec.restraints["rmsd"])
    assert spec.sampling_time_ns == cfg.sampling.boresch.sampling_time_ns
    assert spec.window_heating_ns == cfg.sampling.window_heating_ns


# ---- RMSD ------------------------------------------------------------------


def test_rmsd_bound_first_region_sampled_alone():
    spec = _builder()(
        cv_type="rmsd",
        stage_name="receptor_bound",
        dof=None,
        cv_centre=0.4,
        replicate=1,
        boresch_eq_values={},
    )
    assert spec.topology == "complex.prm7"
    assert [(r["name"], r["sampled"]) for r in spec.restraints["rmsd"]] == [
        ("receptor", True)
    ]
    assert spec.restraints["rmsd"][0]["centre"] == 0.4
    assert spec.restraints["rmsd"][0]["atoms"] == [1, 2, 3]


def test_rmsd_bound_later_region_fixes_earlier():
    spec = _builder()(
        cv_type="rmsd",
        stage_name="target_bound",
        dof=None,
        cv_centre=0.5,
        replicate=1,
        boresch_eq_values={},
    )
    assert [(r["name"], r["sampled"]) for r in spec.restraints["rmsd"]] == [
        ("receptor", False),
        ("target", True),
    ]


def test_rmsd_bulk_uses_isolated_topology():
    spec = _builder()(
        cv_type="rmsd",
        stage_name="target_bulk",
        dof=None,
        cv_centre=0.6,
        replicate=1,
        boresch_eq_values={},
    )
    assert spec.topology == "target_bulk.prm7"
    assert spec.coordinates == "target_bulk.rst7"
    assert len(spec.restraints["rmsd"]) == 1
    r = spec.restraints["rmsd"][0]
    assert r["name"] == "target" and r["sampled"] is True and r["atoms"] == [0, 1, 2]


# ---- always-on restraints --------------------------------------------------


def _builder_with_always_on():
    import dataclasses as dc

    from gluebind.spec_builder import AlwaysOn

    ctx = dc.replace(_context(), always_on=[AlwaysOn("ddb1", [7, 8], 100.0)])
    return SpecBuilder(ctx, _config())


_DDB1_ENTRY = {
    "name": "ddb1",
    "atoms": [7, 8],
    "reference_atoms": [7, 8],  # complex window: reference = the same atoms
    "force_constant": 100.0,
    "centre": None,  # fixed about zero
    "sampled": False,
}


def test_always_on_in_bound_rmsd_stage():
    spec = _builder_with_always_on()(
        cv_type="rmsd",
        stage_name="receptor_bound",
        dof=None,
        cv_centre=0.4,
        replicate=1,
        boresch_eq_values={},
    )
    assert _DDB1_ENTRY in spec.restraints["rmsd"]


def test_always_on_in_boresch_and_separation(tmp_path):
    b = _builder_with_always_on()
    bores = b(
        cv_type="boresch",
        stage_name="thetaA",
        dof="thetaA",
        cv_centre=1.0,
        replicate=1,
        boresch_eq_values={},
    )
    sep = b(
        cv_type="separation",
        stage_name="separation",
        dof=None,
        cv_centre=1.5,
        replicate=1,
        boresch_eq_values={"thetaA": 1.0},
    )
    assert _DDB1_ENTRY in bores.restraints["rmsd"]
    assert _DDB1_ENTRY in sep.restraints["rmsd"]


def test_always_on_absent_from_bulk_stage():
    # Complex-resolved always-on atoms do not exist in an isolated bulk topology,
    # so they must not leak into a bulk stage (bulk always-on is resolved separately).
    spec = _builder_with_always_on()(
        cv_type="rmsd",
        stage_name="target_bulk",
        dof=None,
        cv_centre=0.6,
        replicate=1,
        boresch_eq_values={},
    )
    assert all(r["name"] != "ddb1" for r in spec.restraints["rmsd"])


# ---- multi-domain bulk (held partners + always-on) -------------------------


def test_rmsd_bulk_applies_held_and_always_on():
    import dataclasses as dc

    from gluebind.spec_builder import AlwaysOn, BulkTarget

    ctx = dc.replace(
        _context(),
        rmsd_order=["BD1", "BD2"],
        rmsd_atoms_bound={"BD1": [1, 2], "BD2": [3, 4]},
        rmsd_bulk={
            "BD2": BulkTarget(
                "target_bulk.prm7",
                "target_bulk.rst7",
                atoms=[30, 31],
                to_complex=list(range(1000, 1060)),  # bulk i -> complex 1000 + i
                held=[("BD1", [10, 11])],  # earlier same-protein region held fixed
                always_on=[AlwaysOn("ddb1", [50], 100.0)],
            )
        },
    )
    spec = SpecBuilder(ctx, _config())(
        cv_type="rmsd",
        stage_name="BD2_bulk",
        dof=None,
        cv_centre=0.6,
        replicate=1,
        boresch_eq_values={},
    )
    assert spec.topology == "target_bulk.prm7"
    entries = [
        (r["name"], r["sampled"], r["centre"], r["atoms"])
        for r in spec.restraints["rmsd"]
    ]
    assert entries == [
        ("BD1", False, None, [10, 11]),  # held partner first
        ("BD2", True, 0.6, [30, 31]),  # sampled region
        ("ddb1", False, None, [50]),  # always-on last
    ]
    # every bulk restraint (held, sampled, always-on) references the bound complex
    assert spec.reference_coordinates == "complex.rst7"
    assert [r["reference_atoms"] for r in spec.restraints["rmsd"]] == [
        [1010, 1011],
        [1030, 1031],
        [1050],
    ]


# ---- RMSD reference: the bound complex for every window -------------------


def _all_window_specs(tmp_path):
    builder = _builder(smd_frames_dir=tmp_path)
    common = {"replicate": 1, "boresch_eq_values": {"thetaA": 0.9}}
    return {
        "boresch": builder(
            cv_type="boresch",
            stage_name="thetaB",
            dof="thetaB",
            cv_centre=1.0,
            **common,
        ),
        "rmsd_bound": builder(
            cv_type="rmsd", stage_name="target_bound", dof=None, cv_centre=0.4, **common
        ),
        "rmsd_bulk": builder(
            cv_type="rmsd", stage_name="target_bulk", dof=None, cv_centre=0.4, **common
        ),
        "separation": builder(
            cv_type="separation",
            stage_name="separation",
            dof=None,
            cv_centre=1.5,
            **common,
        ),
    }


def test_every_window_references_the_bound_complex(tmp_path):
    """Regression: RMSD restraints referenced each window's own starting
    frame — a different SMD snapshot per separation window, the bulk-equilibrated
    structure in bulk windows — so the restrained state differed between legs."""
    specs = _all_window_specs(tmp_path)
    # separation starts from its SMD frame but is restrained towards the complex
    assert specs["separation"].coordinates == str(tmp_path / "1.5nm.rst7")
    assert specs["rmsd_bulk"].coordinates == "target_bulk.rst7"
    for spec in specs.values():
        assert spec.reference_coordinates == "complex.rst7"


def test_complex_window_reference_atoms_are_the_restrained_atoms(tmp_path):
    specs = _all_window_specs(tmp_path)
    for name in ("boresch", "rmsd_bound", "separation"):
        for entry in specs[name].restraints["rmsd"]:
            assert entry["reference_atoms"] == entry["atoms"], name


def test_bulk_window_reference_atoms_map_into_the_complex(tmp_path):
    spec = _all_window_specs(tmp_path)["rmsd_bulk"]
    (entry,) = spec.restraints["rmsd"]
    assert entry["atoms"] == [0, 1, 2]
    assert entry["reference_atoms"] == [200, 201, 202]  # target to_complex


def test_bulk_atom_without_complex_counterpart_raises():
    target = BulkTarget("t.prm7", "t.rst7", [0, 5], to_complex=[200, 201])
    with pytest.raises(ValueError, match="not solute atoms"):
        target.reference_atoms(target.atoms)


def _universe(names, masses, resnames):
    import MDAnalysis as mda

    u = mda.Universe.empty(
        len(names), n_residues=len(names), atom_resindex=list(range(len(names)))
    )
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("masses", masses)
    u.add_TopologyAttr("resnames", resnames)
    return u


class _StubMap:
    """Minimal ``_ComplexMap``: receptor input of 3 atoms at complex offset 2."""

    def __init__(self, receptor_input):
        self._input = receptor_input

    def input_universe(self, protein):
        return self._input

    def offset(self, protein):
        return 2


def test_bulk_to_complex_maps_protein_and_glue_blocks():
    from gluebind.spec_builder import _bulk_to_complex

    # complex: glue (2) | receptor (3) | target (1);  bulk: receptor | glue | water
    complex_u = _universe(
        ["C1", "O1", "N", "CA", "C", "CA"],
        [12.0, 16.0, 14.0, 12.0, 12.0, 12.0],
        ["MOL", "MOL", "ALA", "ALA", "ALA", "GLY"],
    )
    receptor_input = _universe(["N", "CA", "C"], [14.0, 12.0, 12.0], ["ALA"] * 3)
    bulk_u = _universe(
        ["N", "CA", "C", "C1", "O1", "O"],
        [14.0, 12.0, 12.0, 12.0, 16.0, 16.0],
        ["ALA", "ALA", "ALA", "MOL", "MOL", "WAT"],
    )
    cmap = _StubMap(receptor_input)
    assert _bulk_to_complex("receptor", bulk_u, complex_u, cmap, True) == [
        2,
        3,
        4,
        0,
        1,
    ]
    assert _bulk_to_complex("receptor", bulk_u, complex_u, cmap, False) == [2, 3, 4]


def test_bulk_to_complex_rejects_reordered_atoms():
    from gluebind.spec_builder import _bulk_to_complex

    complex_u = _universe(["N", "CA", "C"], [14.0, 12.0, 12.0], ["ALA"] * 3)
    receptor_input = _universe(["N", "CA", "C"], [14.0, 12.0, 12.0], ["ALA"] * 3)
    swapped = _universe(["CA", "N", "C"], [12.0, 14.0, 12.0], ["ALA"] * 3)

    class _AtZero(_StubMap):
        def offset(self, protein):
            return 0

    with pytest.raises(ValueError, match="does not match complex atom"):
        _bulk_to_complex("receptor", swapped, complex_u, _AtZero(receptor_input), False)


def test_validate_include_glue():
    from gluebind.config.restraints import RmsdCVSpec
    from gluebind.spec_builder import _validate_include_glue

    on_target = RmsdCVSpec(
        name="BD2", protein="target", selection="resid 1", include_glue=True
    )
    # glue on the same protein as the CV: fine
    _validate_include_glue("BD2", on_target, assign="target", protein="target")
    # bound-only CV: no bulk leg to mismatch, so a cross-protein glue passes this
    # helper. (RestraintsConfig now rejects asymmetric states upstream, so this is
    # the helper's defensive branch, not a config-reachable case.)
    bound_only = RmsdCVSpec(
        name="X",
        protein="target",
        selection="resid 1",
        states=["bound"],
        include_glue=True,
    )
    _validate_include_glue("X", bound_only, assign="target", protein="receptor")
    # bulk-sampled CV on the other protein: inconsistent bound/bulk -> raise
    with pytest.raises(ValueError, match="inconsistent CV"):
        _validate_include_glue("BD2", on_target, assign="receptor", protein="target")
    # include_glue with no glue defined -> raise
    with pytest.raises(ValueError, match="no glue is defined"):
        _validate_include_glue("BD2", on_target, assign=None, protein="target")


def test_separation_keeps_fixed_rmsd_when_rmsd_us_disabled():
    # Separation-PMF-only mode skips the RMSD *US stages*, but the separation
    # window must still apply the fixed RMSD restraints (resolved from context) —
    # the flag gates stage-building in the runner, not the SpecBuilder assembly.
    cfg = _config()
    cfg.sampling.run_rmsd_us = False
    spec = SpecBuilder(_context(), cfg)(
        cv_type="separation",
        stage_name="separation",
        dof=None,
        cv_centre=1.5,
        replicate=1,
        boresch_eq_values={"thetaA": 1.0},
    )
    names = {r["name"] for r in spec.restraints["rmsd"]}
    assert names == {"receptor", "target"}  # RMSD restraints still present and fixed
    assert all(r["sampled"] is False for r in spec.restraints["rmsd"])


# ---- ComplexMap: verified input->complex atom mapping ----------------------


def _fake_universe(names, resids, resnames, masses=None):
    """A minimal MDAnalysis Universe with the attributes the resolver reads."""
    import numpy as np
    from MDAnalysis import Universe

    n = len(names)
    resid_arr = np.asarray(resids)
    n_res = len(set(resid_arr.tolist()))
    u = Universe.empty(
        n_atoms=n,
        n_residues=n_res,
        atom_resindex=_resindex(resid_arr),
        trajectory=False,
    )
    u.add_TopologyAttr("names", list(names))
    u.add_TopologyAttr("masses", list(masses if masses is not None else [12.0] * n))
    # residue-level attrs
    uniq = sorted(set(resid_arr.tolist()))
    u.add_TopologyAttr("resids", uniq)
    rn = {r: resnames[list(resid_arr).index(r)] for r in uniq}
    u.add_TopologyAttr("resnames", [rn[r] for r in uniq])
    return u


def _resindex(resids):
    uniq = sorted(set(resids.tolist()))
    order = {r: i for i, r in enumerate(uniq)}
    return [order[r] for r in resids.tolist()]


def test_complex_map_resolves_through_verified_offset():
    # Assembly order glue(MOL), receptor, target. The complex has RENUMBERED resids
    # relative to the inputs, but the map anchors on atom order (name+mass), so a
    # selection resolved on the input maps to the correct complex atoms.
    from gluebind.spec_builder import _ComplexMap

    target_in = _fake_universe(
        names=["N", "CA", "C"], resids=[10, 10, 11], resnames=["ALA", "ALA", "GLY"]
    )
    receptor_in = _fake_universe(
        names=["N", "CA"], resids=[5, 5], resnames=["SER", "SER"]
    )
    # complex: glue MOL (2 atoms), then receptor (2), then target (3); resids all
    # renumbered starting from 1 — deliberately different from the inputs.
    complex_u = _fake_universe(
        names=["C1", "C2", "N", "CA", "N", "CA", "C"],
        resids=[1, 1, 2, 2, 3, 3, 4],
        resnames=["MOL", "MOL", "SER", "SER", "ALA", "ALA", "GLY"],
    )
    cmap = _ComplexMap(complex_u, target_in, receptor_in, has_glue=True)
    # target block starts after glue(2)+receptor(2) = offset 4
    assert cmap.resolve("target", "name CA") == [5]  # target CA at complex index 5
    assert cmap.resolve("receptor", "name CA") == [3]  # receptor CA at complex index 3
    assert cmap.resolve("target", "resid 10") == [4, 5]  # input resid 10 -> complex 4,5


def test_atom_mode_filters_ca_vs_backbone():
    from gluebind.spec_builder import _ATOM_FILTER, _ComplexMap, _with_atoms

    assert _ATOM_FILTER == {"CA": "name CA", "backbone": "name C N CA"}
    assert _with_atoms("resid 5", "backbone") == "(resid 5) and name C N CA"

    # receptor residue 5 has full backbone; CA picks 1 atom, backbone picks N/CA/C.
    receptor_in = _fake_universe(
        names=["N", "CA", "C", "O"], resids=[5, 5, 5, 5], resnames=["ALA"] * 4
    )
    target_in = _fake_universe(names=["CA"], resids=[1], resnames=["GLY"])
    complex_u = _fake_universe(
        names=["N", "CA", "C", "O", "CA"],
        resids=[1, 1, 1, 1, 2],
        resnames=["ALA", "ALA", "ALA", "ALA", "GLY"],
    )
    cmap = _ComplexMap(complex_u, target_in, receptor_in, has_glue=False)
    assert cmap.resolve("receptor", _with_atoms("resid 5", "CA")) == [1]
    assert cmap.resolve("receptor", _with_atoms("resid 5", "backbone")) == [0, 1, 2]


def test_complex_map_raises_on_reordered_atoms():
    from gluebind.spec_builder import _ComplexMap

    target_in = _fake_universe(
        names=["N", "CA"], resids=[1, 1], resnames=["ALA", "ALA"]
    )
    receptor_in = _fake_universe(
        names=["N", "CA"], resids=[1, 1], resnames=["SER", "SER"]
    )
    # receptor block reordered (CA before N) -> verification must fail
    complex_u = _fake_universe(
        names=["CA", "N", "N", "CA"],
        resids=[1, 1, 2, 2],
        resnames=["SER", "SER", "ALA", "ALA"],
    )
    with pytest.raises(ValueError, match="verification failed"):
        _ComplexMap(complex_u, target_in, receptor_in, has_glue=False)


# ---- separation ------------------------------------------------------------


def test_separation_uses_smd_frame_and_fixes_all(tmp_path):
    cfg = _config()
    spec = _builder(smd_frames_dir=tmp_path)(
        cv_type="separation",
        stage_name="separation",
        dof=None,
        cv_centre=1.5,
        replicate=1,
        boresch_eq_values={
            "thetaA": 0.9,
            "thetaB": 1.0,
            "phiA": 0.1,
            "phiB": 0.2,
            "phiC": 0.3,
        },
    )
    assert spec.coordinates == str(tmp_path / "1.5nm.rst7")
    assert spec.restraints["separation"] == {
        "rec_group": [1, 2, 3],
        "lig_group": [4, 5, 6],
    }
    assert set(spec.restraints["boresch"]["fixed"]) == {
        "thetaA",
        "thetaB",
        "phiA",
        "phiB",
        "phiC",
    }
    assert all(r["sampled"] is False for r in spec.restraints["rmsd"])
    assert spec.force_constant == cfg.sampling.separation.force_constant


def test_separation_falls_back_to_complex_coords_without_smd():
    spec = _builder()(
        cv_type="separation",
        stage_name="separation",
        dof=None,
        cv_centre=1.5,
        replicate=1,
        boresch_eq_values={},
    )
    assert spec.coordinates == "complex.rst7"


# ---- resolver + wiring -----------------------------------------------------


def test_spec_builder_drives_runner(tmp_path):
    cfg = _config()
    cfg.sampling.ensemble_size = 1
    cfg.sampling.rmsd.window_min = 0.0
    cfg.sampling.rmsd.window_max = 0.2
    cfg.sampling.rmsd.window_spacing = 0.2

    builder = SpecBuilder(_context(), cfg, smd_frames_dir=tmp_path / "smd")
    calc = Calculation(
        tmp_path / "calc",
        cfg,
        LocalBackend(),
        builder,
        command_factory=lambda: [
            sys.executable,
            "-c",
            "open('result.json','w').write('{}')",
        ],
        stage_centres={"thetaA": [1.0], "separation": [1.5]},
    )
    calc.run(
        scheduler=Scheduler(calc.backend, poll_interval=0.01),
        pmf_provider=lambda stage: (
            np.linspace(0, 2, 21),
            (np.linspace(0, 2, 21) - 1.0) ** 2,
        ),
    )

    spec = WindowSpec.load(
        tmp_path / "calc" / "boresch" / "thetaA" / "1rad" / "run_01" / "window.json"
    )
    assert spec.topology == "complex.prm7"
    assert spec.restraints["boresch"]["anchors"] == {"b": 10, "c": 11, "B": 12, "C": 13}
    # thetaA is first, so no prior Boresch DoFs are fixed yet
    assert spec.restraints["boresch"]["fixed"] == {}


def test_persisted_anchor_validation_checks_topology_and_protein_ca_membership():
    from gluebind.spec_builder import _validate_persisted_anchors

    class Atom:
        def __init__(self, index):
            self.index = index

    universe = type("Universe", (), {"atoms": type("Atoms", (), {"n_atoms": 20})()})()
    receptor_ca = [Atom(1), Atom(2)]
    target_ca = [Atom(10), Atom(11)]
    anchors = {"b": 1, "c": 2, "B": 10, "C": 11}

    assert (
        _validate_persisted_anchors(anchors, universe, receptor_ca, target_ca)
        == anchors
    )
    with pytest.raises(ValueError, match="outside"):
        _validate_persisted_anchors(
            {**anchors, "C": 20}, universe, receptor_ca, target_ca
        )
    with pytest.raises(ValueError, match="receptor CA atom"):
        _validate_persisted_anchors(
            {**anchors, "b": 3}, universe, receptor_ca, target_ca
        )
    with pytest.raises(ValueError, match="target CA atom"):
        _validate_persisted_anchors(
            {**anchors, "B": 12}, universe, receptor_ca, target_ca
        )


def test_manual_anchors_resolve_one_based_input_residues():
    from gluebind.spec_builder import _resolve_manual_anchors

    class Map:
        def resolve(self, protein, selection):
            expected = {
                ("receptor", "resid 5 and name CA"): [101],
                ("receptor", "resid 7 and name CA"): [103],
                ("target", "resid 12 and name CA"): [205],
                ("target", "resid 14 and name CA"): [207],
            }
            return expected[(protein, selection)]

    assert _resolve_manual_anchors({"b": 5, "c": 7, "B": 12, "C": 14}, Map()) == {
        "b": 101,
        "c": 103,
        "B": 205,
        "C": 207,
    }


def test_manual_anchor_residue_ids_must_be_positive_and_unique_ca():
    from gluebind.spec_builder import _resolve_manual_anchors

    class Map:
        def resolve(self, protein, selection):
            return [] if "resid 9 " in selection else [1]

    with pytest.raises(ValueError, match="positive 1-based"):
        _resolve_manual_anchors({"b": 0, "c": 2, "B": 3, "C": 4}, Map())
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_manual_anchors({"b": 9, "c": 2, "B": 3, "C": 4}, Map())
