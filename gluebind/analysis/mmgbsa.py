"""MM/GBSA end-point estimate from the production equilibration trajectory.

An optional analysis, separate from the umbrella-sampling free energy: it scores
frames of ``prep/equilibration/04_production`` with AmberTools' ``MMPBSA.py``. This
module holds the pure parts — settings, the frame window, the receptor/target
partition and the input files, plus parsing and summarising the per-frame
energies — so they are testable without AmberTools; the job that runs the
AmberTools programs is :mod:`gluebind.simulation.mmgbsa`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pydantic

DELTA_SECTION = "DELTA Energy Terms"
"""Heading of the per-frame binding-energy table in ``MMPBSA.py -eo`` output."""

DELTA_TERMS = ("VDWAALS", "EEL", "EGB", "ESURF", "DELTA TOTAL")
"""Per-frame terms reported; ``DELTA TOTAL`` is the MM/GBSA binding energy."""


class MmgbsaSettings(pydantic.BaseModel):
    """How the production trajectory is scored. Not part of the calculation
    config, so it never affects the config hash."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    discard_ns: float = 10.0
    """Initial production time (ns) left out as equilibration."""
    n_frames: int = 200
    """Frames scored, evenly spaced over the production after ``discard_ns``."""
    igb: Literal[1, 2, 5, 7, 8] = 5
    """Generalized Born model (``MMPBSA.py`` ``igb``)."""
    radii: str = "mbondi2"
    """PB/GB radius set written into the stripped topologies (``ante-MMPBSA.py``)."""
    saltcon_M: float | None = None
    """GB salt concentration (M). ``None`` uses the system's
    ``prep.ion_concentration_M``, so the implicit solvent matches the explicit one."""
    n_blocks: int = 5
    """Contiguous blocks for the block-averaged standard error."""
    keep_trajectory: bool = False
    """Keep the stripped ``traj.nc`` after a successful run."""

    @pydantic.field_validator("discard_ns", "saltcon_M")
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v

    @pydantic.model_validator(mode="after")
    def _enough_frames_per_block(self):
        if self.n_blocks < 2:
            raise ValueError("n_blocks must be >= 2")
        if self.n_frames < 2 * self.n_blocks:
            raise ValueError("n_frames must be at least 2 * n_blocks")
        return self


@dataclasses.dataclass(frozen=True)
class FrameWindow:
    """Which production frames are scored (0-based DCD frame indices)."""

    first: int
    stride: int
    count: int
    ps_per_frame: float

    @property
    def frames(self) -> list[int]:
        return [self.first + i * self.stride for i in range(self.count)]

    @property
    def times_ns(self) -> list[float]:
        # The first DCD frame is written after one save interval.
        return [(f + 1) * self.ps_per_frame / 1000.0 for f in self.frames]


def frame_window(
    n_trajectory_frames: int, ps_per_frame: float, settings: MmgbsaSettings
) -> FrameWindow:
    """``settings.n_frames`` evenly spaced frames after ``settings.discard_ns``.

    Raises if fewer frames than requested remain after the discard.
    """
    if ps_per_frame <= 0:
        raise ValueError("ps_per_frame must be > 0")
    # First frame whose time, (k + 1) * dt, is at or beyond the discard.
    first = max(0, math.ceil(settings.discard_ns * 1000.0 / ps_per_frame - 1e-9) - 1)
    available = n_trajectory_frames - first
    if available < settings.n_frames:
        raise ValueError(
            f"only {max(available, 0)} production frame(s) remain after discarding "
            f"{settings.discard_ns} ns ({n_trajectory_frames} frames at "
            f"{ps_per_frame:g} ps); lower discard_ns or n_frames"
        )
    return FrameWindow(
        first=first,
        stride=available // settings.n_frames,
        count=settings.n_frames,
        ps_per_frame=ps_per_frame,
    )


def residue_mask(residues: Sequence[int]) -> str:
    """Amber residue mask for 1-based residue numbers, e.g. ``:1-313,320``."""
    if not residues:
        raise ValueError("no residues for a mask")
    numbers = sorted(set(residues))
    ranges = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n != prev + 1:
            ranges.append((start, prev))
            start = n
        prev = n
    ranges.append((start, prev))
    return ":" + ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


@dataclasses.dataclass(frozen=True)
class Partition:
    """Receptor and target residues (1-based) of the complex topology."""

    receptor: list[int]
    target: list[int]

    @property
    def receptor_mask(self) -> str:
        return residue_mask(self.receptor)

    @property
    def target_mask(self) -> str:
        return residue_mask(self.target)

    @property
    def solute_mask(self) -> str:
        return residue_mask([*self.receptor, *self.target])

    @property
    def strip_mask(self) -> str:
        """Everything that is not solute: water, ions and crystal waters."""
        return f"!({self.solute_mask})"


def partition_residues(
    *,
    receptor_molecules: Sequence[int],
    target_molecules: Sequence[int],
    glue_molecule: int | None,
    glue_assign_to: str | None,
    molecules: Sequence[Sequence[int]],
    atom_residues: Sequence[int],
) -> Partition:
    """Split the complex's solute residues into receptor and target.

    ``receptor_molecules``/``target_molecules``/``glue_molecule``/``glue_assign_to``
    are the prepared system's molecule layout; ``molecules`` lists each molecule's
    atom indices in topology order and ``atom_residues`` maps each atom to its
    0-based residue index. The glue joins the protein it is assigned to.

    Raises unless the layout's molecules are distinct, each owns whole residues,
    and receptor plus target cover them exactly.
    """
    if glue_molecule is not None and glue_assign_to not in ("receptor", "target"):
        raise ValueError("a glue is present but glue_assign_to is not set")
    receptor_mols = list(receptor_molecules)
    target_mols = list(target_molecules)
    if glue_molecule is not None:
        (receptor_mols if glue_assign_to == "receptor" else target_mols).append(
            glue_molecule
        )
    used = receptor_mols + target_mols
    if len(set(used)) != len(used):
        raise ValueError(f"molecule layout repeats a molecule: {used}")
    if not receptor_mols or not target_mols:
        raise ValueError("receptor and target must each have at least one molecule")
    if max(used) >= len(molecules):
        raise ValueError(
            f"molecule layout refers to molecule {max(used)}, but the topology has "
            f"{len(molecules)}"
        )

    owner: dict[int, int] = {}  # residue -> molecule
    for index, atoms in enumerate(molecules):
        for atom in atoms:
            residue = atom_residues[atom]
            if owner.setdefault(residue, index) != index:
                raise ValueError(f"residue {residue + 1} spans several molecules")

    def residues_of(mols: list[int]) -> list[int]:
        return sorted({atom_residues[a] + 1 for m in mols for a in molecules[m]})

    return Partition(
        receptor=residues_of(receptor_mols), target=residues_of(target_mols)
    )


def cpptraj_input(
    topology: str, trajectory: str, window: FrameWindow, strip_mask: str
) -> str:
    """cpptraj script writing the scored frames, stripped to the solute.

    No imaging: the production DCD is written unwrapped, so the solute stays whole;
    the worker checks receptor-target contact in every scored frame regardless.
    """
    start = window.first + 1  # cpptraj frames are 1-based
    stop = start + (window.count - 1) * window.stride
    return (
        f"parm {topology}\n"
        f"trajin {trajectory} {start} {stop} {window.stride}\n"
        f"strip {strip_mask}\n"
        "trajout traj.nc netcdf nobox\n"
        "run\n"
    )


def mmgbsa_input(n_frames: int, igb: int, saltcon_M: float) -> str:
    """``MMPBSA.py`` input scoring every frame of the stripped trajectory.

    All frame selection happens once, in the cpptraj step.
    """
    return (
        "MM/GBSA (gluebind)\n"
        "&general\n"
        f"  startframe=1, endframe={n_frames}, interval=1,\n"
        "/\n"
        "&gb\n"
        f"  igb={igb}, saltcon={saltcon_M:g},\n"
        "/\n"
    )


def parse_delta_frames(eo_csv: str) -> dict[str, np.ndarray]:
    """Per-frame binding-energy terms from ``MMPBSA.py -eo`` output.

    Reads the ``DELTA Energy Terms`` table and returns :data:`DELTA_TERMS` as
    arrays (kcal/mol), in frame order.
    """
    lines = eo_csv.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == DELTA_SECTION)
    except StopIteration:
        raise ValueError(f"no {DELTA_SECTION!r} table in MMPBSA.py output") from None
    header = [h.strip() for h in lines[start + 1].split(",")]
    missing = [t for t in DELTA_TERMS if t not in header]
    if missing:
        raise ValueError(f"MMPBSA.py output lacks column(s) {missing}")
    rows = []
    for line in lines[start + 2 :]:
        if not line.strip():
            break
        rows.append([float(v) for v in line.split(",")])
    if not rows:
        raise ValueError(f"the {DELTA_SECTION!r} table has no frames")
    table = np.asarray(rows)
    return {t: table[:, header.index(t)] for t in DELTA_TERMS}


def block_sem(values, n_blocks: int) -> float:
    """Standard error of the mean from ``n_blocks`` contiguous block averages.

    Consecutive trajectory frames are correlated, so the naive standard error
    understates the uncertainty; block means are closer to independent. Trailing
    frames that do not fill a block are left out.
    """
    a = np.asarray(values, dtype=float)
    size = a.size // n_blocks
    if size < 1:
        raise ValueError(f"{a.size} values cannot fill {n_blocks} blocks")
    means = a[: size * n_blocks].reshape(n_blocks, size).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(n_blocks))


def summarise(frames: dict[str, np.ndarray], n_blocks: int) -> dict[str, dict]:
    """Mean and block-averaged SEM of each term (kcal/mol)."""
    return {
        term: {"mean": float(np.mean(v)), "sem": block_sem(v, n_blocks)}
        for term, v in frames.items()
    }
