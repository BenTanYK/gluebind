"""Tests for the pure MM/GBSA helpers (no AmberTools needed)."""

import numpy as np
import pydantic
import pytest

from gluebind.analysis import mmgbsa as mm

# ---- settings ----------------------------------------------------------------


def test_settings_defaults():
    s = mm.MmgbsaSettings()
    assert (s.discard_ns, s.n_frames, s.igb, s.radii) == (10.0, 200, 5, "mbondi2")
    assert s.saltcon_M is None  # taken from the system's ion concentration


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"discard_ns": -1.0}, ">= 0"),
        ({"saltcon_M": -0.1}, ">= 0"),
        ({"n_blocks": 1}, "n_blocks"),
        ({"n_frames": 9, "n_blocks": 5}, "2 \\* n_blocks"),
        ({"igb": 3}, "igb"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_settings_reject_invalid_values(kwargs, match):
    with pytest.raises(pydantic.ValidationError, match=match):
        mm.MmgbsaSettings(**kwargs)


# ---- frame window ------------------------------------------------------------


def test_frame_window_spans_the_production_after_the_discard():
    # 50 ns production, a frame every 10 ps -> 5000 frames; frame k is at (k+1)*10 ps
    w = mm.frame_window(5000, 10.0, mm.MmgbsaSettings(discard_ns=10.0, n_frames=200))
    assert (w.first, w.stride, w.count) == (999, 20, 200)
    assert w.times_ns[0] == pytest.approx(10.0)
    assert w.times_ns[-1] == pytest.approx(49.8)
    assert w.frames[-1] < 5000


def test_frame_window_without_discard_starts_at_the_first_frame():
    w = mm.frame_window(100, 10.0, mm.MmgbsaSettings(discard_ns=0.0, n_frames=50))
    assert (w.first, w.stride) == (0, 2)


def test_frame_window_rejects_too_few_frames():
    with pytest.raises(ValueError, match="lower discard_ns or n_frames"):
        mm.frame_window(1000, 10.0, mm.MmgbsaSettings(discard_ns=9.0, n_frames=200))


# ---- masks and the receptor/target partition ---------------------------------


def test_residue_mask_compresses_runs():
    assert mm.residue_mask([7, 1, 2, 3, 5, 8, 2]) == ":1-3,5,7-8"
    with pytest.raises(ValueError):
        mm.residue_mask([])


def _topology(residues_per_molecule):
    """Molecules of 2 atoms per residue; returns (molecules, atom_residues)."""
    molecules, atom_residues, atom, residue = [], [], 0, 0
    for n_res in residues_per_molecule:
        atoms = []
        for _ in range(n_res):
            atoms += [atom, atom + 1]
            atom_residues += [residue, residue]
            atom += 2
            residue += 1
        molecules.append(atoms)
    return molecules, atom_residues


def _partition(layout, residues_per_molecule):
    molecules, atom_residues = _topology(residues_per_molecule)
    return mm.partition_residues(
        **layout, molecules=molecules, atom_residues=atom_residues
    )


BINARY = {
    "receptor_molecules": [0],
    "target_molecules": [1],
    "glue_molecule": None,
    "glue_assign_to": None,
}


def test_partition_binary_ignores_solvent_molecules():
    # receptor (3 residues), target (2), then three water molecules
    p = _partition(BINARY, [3, 2, 1, 1, 1])
    assert (p.receptor, p.target) == ([1, 2, 3], [4, 5])
    assert p.receptor_mask == ":1-3"
    assert p.target_mask == ":4-5"
    assert p.strip_mask == "!(:1-5)"


def test_partition_merges_a_chain_split_protein():
    layout = {**BINARY, "receptor_molecules": [0, 1], "target_molecules": [2]}
    p = _partition(layout, [2, 3, 2, 1])
    assert (p.receptor_mask, p.target_mask) == (":1-5", ":6-7")


@pytest.mark.parametrize(
    "assign_to, receptor, target",
    [("receptor", ":1-4", ":5-6"), ("target", ":2-4", ":1,5-6")],
)
def test_partition_puts_the_glue_with_its_assigned_protein(assign_to, receptor, target):
    # assembly order: glue (molecule 0) -> receptor -> target
    layout = {
        "receptor_molecules": [1],
        "target_molecules": [2],
        "glue_molecule": 0,
        "glue_assign_to": assign_to,
    }
    p = _partition(layout, [1, 3, 2])
    assert (p.receptor_mask, p.target_mask) == (receptor, target)
    assert p.solute_mask == ":1-6"


@pytest.mark.parametrize(
    "layout, match",
    [
        ({**BINARY, "glue_molecule": 2}, "glue_assign_to"),
        ({**BINARY, "target_molecules": [0]}, "repeats a molecule"),
        ({**BINARY, "target_molecules": []}, "at least one molecule"),
        ({**BINARY, "target_molecules": [5]}, "topology has 3"),
    ],
)
def test_partition_rejects_inconsistent_layouts(layout, match):
    with pytest.raises(ValueError, match=match):
        _partition(layout, [2, 2, 1])


def test_partition_rejects_a_residue_split_across_molecules():
    molecules, atom_residues = _topology([2, 2])
    atom_residues[4] = atom_residues[3]  # first atom of molecule 1 joins residue 1
    with pytest.raises(ValueError, match="spans several molecules"):
        mm.partition_residues(
            **BINARY, molecules=molecules, atom_residues=atom_residues
        )


# ---- input files ---------------------------------------------------------------


def test_cpptraj_input_selects_the_window_once():
    w = mm.FrameWindow(first=999, stride=20, count=200, ps_per_frame=10.0)
    text = mm.cpptraj_input("complex.prm7", "prod.dcd", w, "!(:1-509)")
    assert "trajin prod.dcd 1000 4980 20\n" in text  # 1-based start/stop
    assert "strip !(:1-509)\n" in text
    assert "trajout traj.nc netcdf nobox" in text
    assert "autoimage" not in text


def test_mmgbsa_input_scores_every_frame_with_the_given_model():
    text = mm.mmgbsa_input(200, igb=5, saltcon_M=0.15)
    assert "startframe=1, endframe=200, interval=1, keep_files=0" in text
    assert "igb=5, saltcon=0.15" in text
    assert "strip_mask" not in text  # topologies and trajectory are pre-stripped


# ---- parsing and statistics ----------------------------------------------------


def test_parse_delta_frames_reads_mmpbsa_eo_output(data_dir):
    frames = mm.parse_delta_frames((data_dir / "mmgbsa_frames_gb.csv").read_text())
    assert set(frames) == set(mm.DELTA_TERMS)
    total = frames["DELTA TOTAL"]
    assert total.shape == (3,)
    assert total[0] == pytest.approx(-40.98093816000255)
    # the binding energy is the sum of its components
    parts = sum(frames[t] for t in ("VDWAALS", "EEL", "EGB", "ESURF"))
    assert np.allclose(parts, total)


def test_parse_delta_frames_rejects_output_without_the_delta_table():
    with pytest.raises(ValueError, match="DELTA Energy Terms"):
        mm.parse_delta_frames("Complex Energy Terms\nFrame #,TOTAL\n0,1.0\n")


def test_block_sem():
    assert mm.block_sem(np.ones(50), 5) == pytest.approx(0.0)
    # block means 0, 1, 2, 3, 4 -> SEM = sd(block means) / sqrt(5)
    values = np.repeat(np.arange(5.0), 10)
    expected = np.std(np.arange(5.0), ddof=1) / np.sqrt(5)
    assert mm.block_sem(values, 5) == pytest.approx(expected)
    with pytest.raises(ValueError, match="cannot fill"):
        mm.block_sem(np.ones(3), 5)


def test_summarise_reports_mean_and_sem_per_term():
    frames = {"DELTA TOTAL": np.repeat(np.arange(5.0), 10)}
    out = mm.summarise(frames, n_blocks=5)
    assert out["DELTA TOTAL"]["mean"] == pytest.approx(2.0)
    assert out["DELTA TOTAL"]["sem"] > 0
