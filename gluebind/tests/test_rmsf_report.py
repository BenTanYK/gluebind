"""Tests for the RMSF report: input-topology numbering and the RMSF plot."""

import MDAnalysis as mda
import numpy as np
import pytest

import gluebind.selection.rmsf as rmsf_mod
import gluebind.simulation.restraint_resolution as resolution
import gluebind.spec_builder as spec_builder
import gluebind.system.mdanalysis as mdanalysis
from gluebind import CalculationConfig
from gluebind.selection.rmsf import AnchorLabel, RmsfZone, contiguous_runs, plot_rmsf
from gluebind.system.prep import PreparedSystem

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_contiguous_runs():
    assert contiguous_runs([5, 1, 2, 3, 7, 8, 2]) == [(1, 3), (5, 5), (7, 8)]
    assert contiguous_runs([]) == []


def test_plot_rmsf_writes_a_png(tmp_path):
    resids = list(range(1, 41))
    rmsf = 1.0 + 0.5 * np.sin(np.arange(40) / 3.0)
    path = plot_rmsf(
        resids,
        rmsf,
        tmp_path / "rmsf.png",
        title="receptor",
        candidates=[5, 14, 99],  # 99 is outside the profile: skipped
        anchors={
            "b": AnchorLabel(10, 1403, complex_resid=11, complex_atom=1456),
            "c": AnchorLabel(30, 4829),  # input numbering only
            "x": AnchorLabel(500, 1),  # outside the profile: skipped
        },
        zones=[
            RmsfZone("RMSD core", [*range(1, 15), *range(25, 41)], "rmsd"),
            RmsfZone("always-on", list(range(18, 23)), "always_on"),
            RmsfZone("elsewhere", [300, 301], "rmsd"),  # outside: skipped
        ],
        complex_offset=1,  # adds the complex-numbering axis and labels
    )
    assert path.read_bytes()[:8] == PNG_MAGIC


# ---- write_rmsf_report ---------------------------------------------------------


def _protein(n_res):
    """An input topology of ``n_res`` residues, each an N and a CA atom."""
    u = mda.Universe.empty(
        2 * n_res, n_residues=n_res, atom_resindex=np.repeat(np.arange(n_res), 2)
    )
    u.add_TopologyAttr("name", ["N", "CA"] * n_res)
    u.add_TopologyAttr("resid", np.arange(1, n_res + 1))
    u.add_TopologyAttr("resname", ["ALA"] * n_res)
    return u


def _complex(n_receptor, n_target):
    """The assembled complex: a one-atom glue (residue 1), then the receptor, then
    the target, numbered on from the glue."""
    n_res = n_receptor + n_target
    resindex = np.concatenate([[0], np.repeat(np.arange(1, n_res + 1), 2)])
    u = mda.Universe.empty(1 + 2 * n_res, n_residues=1 + n_res, atom_resindex=resindex)
    u.add_TopologyAttr("name", ["C1"] + ["N", "CA"] * n_res)
    u.add_TopologyAttr("resid", np.arange(1, n_res + 2))
    u.add_TopologyAttr("resname", ["MOL"] + ["ALA"] * n_res)
    return u


def _universes():
    return {
        "receptor.prm7": _protein(6),
        "target.prm7": _protein(5),
        "complex.prm7": _complex(6, 5),
    }


class _FakeComplexMap:
    """Assembly order glue (1 atom) -> receptor -> target, as in the real map."""

    def __init__(self, complex_u, target_u, receptor_u, *, has_glue):
        self._inputs = {"receptor": receptor_u, "target": target_u}
        n_glue = 1 if has_glue else 0
        self._offset = {
            "receptor": n_glue,
            "target": n_glue + receptor_u.atoms.n_atoms,
        }

    def input_universe(self, protein):
        return self._inputs[protein]

    def offset(self, protein):
        return self._offset[protein]

    def resolve(self, protein, selection):
        indices = self._inputs[protein].select_atoms(selection).indices
        return [int(i) + self._offset[protein] for i in indices]


@pytest.fixture
def report_env(tmp_path, monkeypatch):
    """A glue system (receptor 6 residues, target 5) with the MD parts stubbed;
    returns the plot calls."""
    universes = _universes()
    monkeypatch.setattr(
        mdanalysis, "load_amber_universe", lambda prm7, *traj: universes[prm7]
    )
    monkeypatch.setattr(spec_builder, "_ComplexMap", _FakeComplexMap)

    def fake_rmsf(universe, selection):
        n = len(selection.split()) - 1
        return np.arange(n), 1.0 + np.cos(np.arange(n))  # minima at 3 (and 6)

    monkeypatch.setattr(rmsf_mod, "compute_rmsf", fake_rmsf)
    plots = []
    real_plot = rmsf_mod.plot_rmsf
    monkeypatch.setattr(
        rmsf_mod,
        "plot_rmsf",
        lambda *a, **k: plots.append((a, k)) or real_plot(*a, **k),
    )
    return plots


def _config(**restraints):
    return CalculationConfig.model_validate(
        {
            "inputs": {
                "receptor": {"prm7": "receptor.prm7", "rst7": "receptor.rst7"},
                "target": {"prm7": "target.prm7", "rst7": "target.rst7"},
                "glue": {"sdf": "glue.sdf", "assign_to": "receptor"},
            },
            "restraints": restraints,
        }
    )


PREPARED = PreparedSystem(
    complex_prm7="complex.prm7",
    complex_rst7="complex.rst7",
    complex_trajectory="complex.dcd",
    target_bulk_prm7="t.prm7",
    target_bulk_rst7="t.rst7",
    receptor_bulk_prm7="r.prm7",
    receptor_bulk_rst7="r.rst7",
    glue_assign_to="receptor",
    receptor_molecules=[1],
    target_molecules=[2],
    glue_molecule=0,
)


def test_report_gives_input_and_complex_residue_numbers(tmp_path, report_env):
    # The complex numbers residues after the glue (and, for the target, after the
    # receptor); the report gives the input numbering anchors are configured in,
    # and the complex numbering of the equilibrated topology.
    report = resolution.write_rmsf_report(PREPARED, _config(), tmp_path)
    receptor = np.loadtxt(report["receptor"])
    target = np.loadtxt(report["target"])
    assert receptor[:, 0].tolist() == [1, 2, 3, 4, 5, 6]  # input resid
    assert receptor[:, 1].tolist() == [2, 3, 4, 5, 6, 7]  # complex: after the glue
    assert receptor[:, 2].tolist() == [1, 3, 5, 7, 9, 11]  # input CA index
    assert receptor[:, 3].tolist() == [2, 4, 6, 8, 10, 12]  # complex CA index
    assert target[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert target[:, 1].tolist() == [8, 9, 10, 11, 12]  # after glue + receptor
    assert target[:, 2].tolist() == [1, 3, 5, 7, 9]
    assert target[:, 3].tolist() == [14, 16, 18, 20, 22]
    header = (tmp_path / "rmsf_receptor.dat").read_text().splitlines()[0]
    assert header.endswith("input topology resids): 4")  # the candidate
    assert report["receptor_plot"].endswith("rmsf_receptor.png")
    assert (tmp_path / "rmsf_target.png").read_bytes()[:8] == PNG_MAGIC
    (_, receptor_kw), (_, target_kw) = report_env
    assert (receptor_kw["complex_offset"], target_kw["complex_offset"]) == (1, 7)


def test_report_plots_resolved_anchors_and_restraint_zones(tmp_path, report_env):
    config = _config(
        rmsd_cvs=[
            {
                "name": "core",
                "protein": "receptor",
                "selection": "resid 1-2 or resid 5-6",
            },
            {"name": "T", "protein": "target", "selection": "resid 1-5"},
        ],
        rmsd_order=["core", "T"],
        always_on=[
            {"protein": "receptor", "selection": "resid 3-4", "force_constant": 100.0}
        ],
    )
    anchors = {"b": 4, "c": 10, "B": 16, "C": 22}  # complex CA atom indices
    resolution.write_rmsf_report(PREPARED, config, tmp_path, anchors=anchors)
    (_, receptor_kw), (_, target_kw) = report_env
    # (input resid, input CA, complex resid, complex CA)
    assert receptor_kw["anchors"] == {
        "b": AnchorLabel(2, 3, complex_resid=3, complex_atom=4),
        "c": AnchorLabel(5, 9, complex_resid=6, complex_atom=10),
    }
    assert target_kw["anchors"] == {
        "B": AnchorLabel(2, 3, complex_resid=9, complex_atom=16),
        "C": AnchorLabel(5, 9, complex_resid=12, complex_atom=22),
    }
    zones = {(z.label, z.kind): z.resids for z in receptor_kw["zones"]}
    assert zones == {
        ("RMSD core", "rmsd"): [1, 2, 5, 6],
        ("always-on", "always_on"): [3, 4],
    }
    assert [z.label for z in target_kw["zones"]] == ["RMSD T"]
    assert receptor_kw["candidates"] == [4]


def test_report_shows_configured_manual_anchors(tmp_path, report_env):
    config = _config(boresch={"anchors": {"b": 1, "c": 6, "B": 3, "C": 5}})
    resolution.write_rmsf_report(PREPARED, config, tmp_path)  # no resolved anchors
    (_, receptor_kw), (_, target_kw) = report_env
    assert receptor_kw["anchors"] == {
        "b": AnchorLabel(1, 1, complex_resid=2, complex_atom=2),
        "c": AnchorLabel(6, 11, complex_resid=7, complex_atom=12),
    }
    assert target_kw["anchors"] == {
        "B": AnchorLabel(3, 5, complex_resid=10, complex_atom=18),
        "C": AnchorLabel(5, 9, complex_resid=12, complex_atom=22),
    }


def test_anchor_label_says_which_numbering_each_number_is_in():
    both = AnchorLabel(91, 1403, complex_resid=92, complex_atom=1456).text("b")
    assert both == "b: res 91 (input) / 92 (complex)\nCA 1403 (input) / 1456 (complex)"
    assert AnchorLabel(91, 1403).text("b") == "b: res 91\nCA 1403 (input)"


def test_default_rmsd_regions_cover_the_whole_protein(tmp_path, report_env):
    resolution.write_rmsf_report(PREPARED, _config(), tmp_path)
    (_, receptor_kw), _ = report_env
    (zone,) = receptor_kw["zones"]
    assert (zone.label, zone.resids) == ("RMSD (all Cα)", [1, 2, 3, 4, 5, 6])


def test_a_failed_plot_does_not_fail_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rmsf_mod, "plot_rmsf", _raise)
    universes = _universes()
    monkeypatch.setattr(
        mdanalysis, "load_amber_universe", lambda prm7, *t: universes[prm7]
    )
    monkeypatch.setattr(spec_builder, "_ComplexMap", _FakeComplexMap)
    monkeypatch.setattr(
        rmsf_mod,
        "compute_rmsf",
        lambda u, selection: (None, np.ones(len(selection.split()) - 1)),
    )
    report = resolution.write_rmsf_report(PREPARED, _config(), tmp_path)
    assert set(report) == {"receptor", "target"}  # the .dat reports, no plots
    assert "RMSF plot for the receptor not written" in capsys.readouterr().out


def _raise(*args, **kwargs):
    raise RuntimeError("no display")
