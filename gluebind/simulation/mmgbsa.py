"""Backend job: MM/GBSA of the production equilibration trajectory.

Runs AmberTools on a compute node, inside its own work directory: ``ante-MMPBSA.py``
writes stripped complex/receptor/target topologies, ``cpptraj`` writes the scored
frames (solute only) to ``traj.nc``, and ``MMPBSA.py`` scores them. The pure parts
(partition, frame window, input files, parsing) live in
:mod:`gluebind.analysis.mmgbsa`. ``result.json`` is written last and is the
success marker the driver checks.

AmberTools is located on ``PATH``; ``AMBERHOME`` is set for its subprocesses from
the location of ``MMPBSA.py``, which needs it, rather than read from the job's
environment.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pydantic

from gluebind.analysis.mmgbsa import MmgbsaSettings

MMGBSA_SPEC_FILENAME = "mmgbsa.json"
MMGBSA_RESULT_FILENAME = "result.json"
EXECUTABLES = ("ante-MMPBSA.py", "cpptraj", "MMPBSA.py")
CONTACT_CUTOFF_A = 5.0
"""A scored frame must have a receptor-target heavy-atom pair within this distance."""


class MmgbsaSpec(pydantic.BaseModel):
    """Everything the MM/GBSA worker needs, self-contained for a backend job."""

    model_config = pydantic.ConfigDict(extra="forbid")

    topology: str
    """Solvated complex topology of the production run."""
    trajectory: str
    """Production DCD (written unwrapped)."""
    ps_per_frame: float
    """Production time between DCD frames (ps)."""
    receptor_molecules: list[int]
    target_molecules: list[int]
    glue_molecule: int | None = None
    glue_assign_to: str | None = None
    saltcon_M: float
    """GB salt concentration, resolved from the settings or the system."""
    settings: MmgbsaSettings

    def dump(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "MmgbsaSpec":
        return cls.model_validate_json(pathlib.Path(path).read_text())


def mmgbsa_launch_command(python: str = "python") -> list[str]:
    """The command a backend runs (inside the MM/GBSA work dir) to execute it."""
    code = "from gluebind.simulation.mmgbsa import run_mmgbsa; run_mmgbsa('.')"
    return [python, "-c", code]


def run_mmgbsa(work_dir: str | pathlib.Path) -> None:
    """Run the MM/GBSA job whose spec is at ``work_dir/mmgbsa.json``.

    Writes the stripped topologies, input files, logs and MMPBSA.py output into
    ``work_dir``, then ``result.json``. Raises on failure.
    """
    from gluebind.analysis.mmgbsa import (
        cpptraj_input,
        frame_window,
        mmgbsa_input,
        parse_delta_frames,
        summarise,
    )

    work_dir = pathlib.Path(work_dir).resolve()
    spec = MmgbsaSpec.load(work_dir / MMGBSA_SPEC_FILENAME)
    settings = spec.settings
    exe = _find_executables()
    env = {**os.environ, "AMBERHOME": str(_amberhome(exe["MMPBSA.py"]))}

    partition = _partition(spec)
    window = frame_window(_count_frames(spec.trajectory), spec.ps_per_frame, settings)

    # AmberTools calls the second partner the "ligand": here it is the target.
    ante = [exe["ante-MMPBSA.py"], "-p", spec.topology, "-c", "complex.prm7"]
    ante += ["-r", "receptor.prm7", "-l", "target.prm7"]
    ante += ["-s", partition.strip_mask, "-n", partition.target_mask]
    ante += ["--radii", settings.radii]
    _run(ante, work_dir, "ante-MMPBSA.log", env)
    _require(work_dir, "complex.prm7", "receptor.prm7", "target.prm7")

    (work_dir / "cpptraj.in").write_text(
        cpptraj_input(spec.topology, spec.trajectory, window, partition.strip_mask)
    )
    _run([exe["cpptraj"], "-i", "cpptraj.in"], work_dir, "cpptraj.log", env)
    _require(work_dir, "traj.nc")
    _check_contact(work_dir, partition)

    (work_dir / "mmgbsa.in").write_text(
        mmgbsa_input(window.count, settings.igb, spec.saltcon_M)
    )
    mmpbsa = [exe["MMPBSA.py"], "-O", "-i", "mmgbsa.in"]
    mmpbsa += ["-o", "FINAL_RESULTS_MMPBSA.dat", "-eo", "frames.csv"]
    mmpbsa += ["-cp", "complex.prm7", "-rp", "receptor.prm7", "-lp", "target.prm7"]
    mmpbsa += ["-y", "traj.nc"]
    _run(mmpbsa, work_dir, "MMPBSA.log", env)
    _require(work_dir, "frames.csv", "FINAL_RESULTS_MMPBSA.dat")

    frames = parse_delta_frames((work_dir / "frames.csv").read_text())
    n_scored = len(frames["DELTA TOTAL"])
    if n_scored != window.count:
        raise RuntimeError(
            f"MMPBSA.py scored {n_scored} frames, expected {window.count}"
        )
    if not settings.keep_trajectory:
        (work_dir / "traj.nc").unlink()

    terms = summarise(frames, settings.n_blocks)
    result = {
        "dg_mmgbsa": terms["DELTA TOTAL"]["mean"],  # kcal/mol
        "dg_mmgbsa_sem": terms["DELTA TOTAL"]["sem"],  # block-averaged
        "terms": terms,
        "n_frames": window.count,
        "time_ns": window.times_ns,
        "dg_per_frame": [float(v) for v in frames["DELTA TOTAL"]],
        "receptor_mask": partition.receptor_mask,
        "target_mask": partition.target_mask,
        "saltcon_M": spec.saltcon_M,
        "settings": settings.model_dump(mode="json"),
    }
    tmp = work_dir / f".{MMGBSA_RESULT_FILENAME}.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, work_dir / MMGBSA_RESULT_FILENAME)


def _find_executables() -> dict[str, str]:
    found = {name: shutil.which(name) for name in EXECUTABLES}
    missing = [name for name, path in found.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"AmberTools program(s) not found on PATH: {', '.join(missing)}. "
            "Install ambertools in the gluebind environment."
        )
    return {name: str(path) for name, path in found.items()}


def _amberhome(mmpbsa_py: str) -> pathlib.Path:
    """AMBERHOME for the installation that provides ``MMPBSA.py`` (``<home>/bin``)."""
    return pathlib.Path(mmpbsa_py).resolve().parent.parent


def _partition(spec: MmgbsaSpec):
    """Receptor/target residues of the complex topology, from the molecule layout."""
    import openmm.app as app

    from gluebind.analysis.mmgbsa import partition_residues
    from gluebind.restraints.system_builder import topology_molecules

    topology = app.AmberPrmtopFile(spec.topology).topology
    return partition_residues(
        receptor_molecules=spec.receptor_molecules,
        target_molecules=spec.target_molecules,
        glue_molecule=spec.glue_molecule,
        glue_assign_to=spec.glue_assign_to,
        molecules=topology_molecules(topology),
        atom_residues=[atom.residue.index for atom in topology.atoms()],
    )


def _count_frames(trajectory: str) -> int:
    from MDAnalysis.coordinates.DCD import DCDReader

    with DCDReader(trajectory) as reader:
        return reader.n_frames


def _check_contact(work_dir: pathlib.Path, partition) -> None:
    """Raise unless receptor and target touch in every frame of ``traj.nc``.

    Guards against a frame where imaging separated the two (they are distinct
    molecules in the topology), which would score as an unbound pair.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    u = mda.Universe(
        str(work_dir / "complex.prm7"),
        str(work_dir / "traj.nc"),
        topology_format="PARM7",
        format="NCDF",
    )
    heavy = "not name H*"
    receptor = u.select_atoms(f"resid {_resid_ranges(partition.receptor)} and {heavy}")
    target = u.select_atoms(f"resid {_resid_ranges(partition.target)} and {heavy}")
    apart = [
        ts.frame
        for ts in u.trajectory
        if not len(
            capped_distance(
                receptor.positions,
                target.positions,
                CONTACT_CUTOFF_A,
                box=None,
                return_distances=False,
            )
        )
    ]
    if apart:
        raise RuntimeError(
            f"receptor and target have no heavy atoms within {CONTACT_CUTOFF_A} A in "
            f"{len(apart)} scored frame(s) (first: {apart[:5]}); the trajectory "
            "may be mis-imaged"
        )


def _resid_ranges(residues: list[int]) -> str:
    """MDAnalysis ``resid`` ranges for 1-based residue numbers (``1-313 320``)."""
    from gluebind.analysis.mmgbsa import residue_mask

    return residue_mask(residues)[1:].replace(",", " ")


def _run(command: list[str], cwd: pathlib.Path, log_name: str, env: dict) -> None:
    with (cwd / log_name).open("w") as log:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(
            f"{pathlib.Path(command[0]).name} failed (exit {result.returncode}); "
            f"see {cwd / log_name}"
        )


def _require(work_dir: pathlib.Path, *names: str) -> None:
    missing = [
        n
        for n in names
        if not (work_dir / n).is_file() or (work_dir / n).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"expected output(s) missing or empty: {', '.join(missing)}")
