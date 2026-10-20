"""RMSF over the equilibration trajectory + local-minimum candidate detection.

``compute_rmsf`` wraps the MDAnalysis align+RMSF workflow (as in the template's
``analyse_RMSF.py``) and is integration-verified; ``local_minima`` (pure) turns
an RMSF profile into the candidate pool of stable residues; ``plot_rmsf`` draws a
profile with the restraint zones and Boresch anchors for inspection.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np


@dataclasses.dataclass(frozen=True)
class RmsfZone:
    """A restrained residue region drawn as a shaded band on an RMSF plot."""

    label: str
    resids: list[int]
    kind: Literal["rmsd", "always_on"]


@dataclasses.dataclass(frozen=True)
class AnchorLabel:
    """A Boresch anchor's residue and CA atom, in the input topology of its protein
    and, when known, in the assembled complex. Atom indices are 0-based."""

    resid: int
    atom: int
    complex_resid: int | None = None
    complex_atom: int | None = None

    def text(self, name: str) -> str:
        if self.complex_resid is None or self.complex_atom is None:
            return f"{name}: res {self.resid}\nCA {self.atom} (input)"
        return (
            f"{name}: res {self.resid} (input) / {self.complex_resid} (complex)\n"
            f"CA {self.atom} (input) / {self.complex_atom} (complex)"
        )


def compute_rmsf(universe, selection: str = "name CA"):
    """CA RMSF over a trajectory. Returns ``(resids, rmsf_values)``.

    Aligns the trajectory to its average structure, then computes per-atom RMSF
    for the selection (MDAnalysis).
    """
    from MDAnalysis.analysis import align, rms

    average = align.AverageStructure(
        universe, universe, select=selection, ref_frame=0
    ).run()
    align.AlignTraj(
        universe,
        average.results.universe,
        # AverageStructure contains only ``selection`` and renumbers it from
        # zero, so apply the original selector only to the mobile trajectory.
        select={"mobile": selection, "reference": "all"},
        in_memory=True,
    ).run()
    atoms = universe.select_atoms(selection)
    rmsf = rms.RMSF(atoms).run()
    return atoms.resids, np.asarray(rmsf.results.rmsf)


def local_minima(values, *, order: int = 1) -> list[int]:
    """Indices of local minima of ``values`` (strictly below neighbours within
    ``order``)."""
    values = np.asarray(values, float)
    minima: list[int] = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if (
            values[i] == window.min()
            and values[i] < values[i - 1]
            and values[i] < values[i + 1]
        ):
            minima.append(i)
    return minima


def stablest_candidates(resids, rmsf_values, *, top_n: int = 8) -> list[int]:
    """Residue ids of the ``top_n`` lowest-RMSF local minima (most stable)."""
    resids = np.asarray(resids)
    rmsf_values = np.asarray(rmsf_values, float)
    minima = local_minima(rmsf_values)
    minima.sort(key=lambda i: rmsf_values[i])
    return [int(resids[i]) for i in minima[:top_n]]


def contiguous_runs(resids: Sequence[int]) -> list[tuple[int, int]]:
    """Inclusive ``(first, last)`` runs of consecutive residue numbers."""
    numbers = sorted({int(r) for r in resids})
    runs: list[tuple[int, int]] = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def plot_rmsf(
    resids: Sequence[int],
    rmsf: Sequence[float],
    path: str | pathlib.Path,
    *,
    title: str,
    candidates: Sequence[int] = (),
    anchors: Mapping[str, AnchorLabel] | None = None,
    zones: Sequence[RmsfZone] = (),
    complex_offset: int | None = None,
) -> pathlib.Path:
    """Save a CA RMSF profile as a PNG.

    ``resids`` are input-topology residue numbers. ``candidates`` (residue
    numbers) are marked subtly, without labels. ``anchors`` maps an anchor name
    (``b``, ``c``, ``B``, ``C``) to its :class:`AnchorLabel`, drawn and labelled
    with its residue and CA atom in each numbering. Each zone is shaded over its
    residues: RMSD regions in colour, always-on restraints hatched. Anchors or
    zones outside the plotted residues are left out.

    ``complex_offset`` is the shift from input to assembled-complex residue
    numbers (complex = input + offset); when given, a top axis shows the complex
    numbering.
    """
    from matplotlib.figure import Figure  # lazy: plotting is optional

    x = np.asarray(resids, dtype=int)
    y = np.asarray(rmsf, dtype=float)
    lookup = dict(zip(x.tolist(), y.tolist(), strict=True))
    lo, hi = int(x.min()), int(x.max())

    fig = Figure(figsize=(12.5, 4.2), layout="constrained")
    ax = fig.subplots()
    colours = ["tab:blue", "tab:green", "tab:purple", "tab:olive", "tab:cyan"]
    for i, zone in enumerate(zones):
        runs = [(a, b) for a, b in contiguous_runs(zone.resids) if b >= lo and a <= hi]
        for j, (a, b) in enumerate(runs):
            label = zone.label if j == 0 else None
            if zone.kind == "always_on":
                ax.axvspan(
                    a - 0.5,
                    b + 0.5,
                    facecolor="none",
                    edgecolor="0.7",
                    hatch="//",
                    lw=0,
                    label=label,
                )
            else:
                ax.axvspan(
                    a - 0.5,
                    b + 0.5,
                    facecolor=colours[i % len(colours)],
                    alpha=0.12,
                    lw=0,
                    label=label,
                )

    ax.plot(x, y, color="0.15", lw=1.0)

    shown = [r for r in candidates if r in lookup]
    if shown:
        ax.scatter(
            shown,
            [lookup[r] for r in shown],
            s=18,
            facecolors="none",
            edgecolors="0.5",
            linewidths=0.8,
            zorder=3,
            label="stable candidates",
        )

    # Labels sit in a band above the trace (see the y limit below), on two rows,
    # each joined to its anchor by a leader line, so they never cover the profile.
    label_rows = (0.97, 0.80)  # tops of the label rows, as axes fractions
    shown_anchors = sorted(
        ((n, a) for n, a in (anchors or {}).items() if a.resid in lookup),
        key=lambda item: item[1].resid,
    )
    previous: tuple[float, int] | None = None  # (x position, row) of the last label
    for name, anchor in shown_anchors:
        value = lookup[anchor.resid]
        ax.scatter([anchor.resid], [value], s=45, color="tab:red", zorder=4)
        position = (anchor.resid - lo) / max(hi - lo, 1)
        # Alternate rows when this label would collide with the previous one.
        row = 0
        if previous is not None and position - previous[0] < 0.3:
            row = 1 - previous[1]
        previous = (position, row)
        # Keep labels near either end of the axis inside the plot; the leader line
        # leaves the label corner directly above the anchor, so it drops vertically.
        align = "left" if position < 0.15 else "right" if position > 0.85 else "center"
        corner = {"left": (0.0, 0.0), "right": (1.0, 0.0), "center": (0.5, 0.0)}[align]
        ax.annotate(
            anchor.text(name),
            (anchor.resid, value),
            xytext=(anchor.resid, label_rows[row]),
            textcoords=("data", "axes fraction"),
            ha=align,
            va="top",
            fontsize=8,
            color="tab:red",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
            arrowprops={
                "arrowstyle": "-",
                "color": "tab:red",
                "lw": 0.6,
                "alpha": 0.7,
                "relpos": corner,
                "shrinkA": 0,
            },
            zorder=5,
        )

    ax.set_xlim(lo - 1, hi + 1)
    # Headroom: the trace fills the lower ~60%, leaving the top for the labels.
    headroom = 1.7 if shown_anchors else 1.25
    ax.set_ylim(0, max(float(y.max()) * headroom, 1e-3))
    ax.set_xlabel("Residue (input topology numbering)")
    if complex_offset is not None:
        shift = complex_offset
        top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda r: np.asarray(r) + shift,
                lambda r: np.asarray(r) - shift,
            ),
        )
        top.set_xlabel("Residue (assembled complex numbering)", fontsize=9)
        top.tick_params(labelsize=8)
    ax.set_ylabel("Cα RMSF (Å)")
    ax.set_title(title)
    if anchors:
        ax.text(
            0.005,
            0.015,
            "CA atom indices are 0-based",
            transform=ax.transAxes,
            fontsize=7,
            color="0.4",
        )
    if ax.get_legend_handles_labels()[0]:
        # Outside the axes, so it cannot cover a label or the trace.
        ax.legend(
            fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0
        )
    path = pathlib.Path(path)
    fig.savefig(path, dpi=150)
    return path
