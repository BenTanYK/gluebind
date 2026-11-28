"""Which replicates own a copy of a stage, and what that copy is called on disk.

``sampling.ensemble_from`` decides where the replicate axis begins: every stage
from that point onward is run once per replicate, and everything before it once.
The helpers here are the single source of truth for both halves of that — which
replicate indices a stage fans out over, and the ``run_NN`` directory name — so
prep, steered MD and the umbrella windows cannot drift apart on either.
"""

from __future__ import annotations

SHARED: tuple[int, ...] = (1,)
"""The replicate list for a stage that is not fanned out.

Numbered rather than special-cased so a shared stage still lives under
``run_01``: one directory layout serves both modes.
"""


def replicate_dirname(replicate: int) -> str:
    """Directory name for a 1-based replicate index: ``run_01``."""
    if replicate < 1:
        raise ValueError(f"replicate indices are 1-based; got {replicate}")
    return f"run_{replicate:02d}"


def equilibration_replicates(sampling) -> tuple[int, ...]:
    """Replicates with their own production run, bulk pair and restraint context.

    ``(1,)`` unless ``ensemble_from="equilibration"``, in which case every
    replicate equilibrates independently.
    """
    if sampling.ensemble_equilibration:
        return tuple(range(1, sampling.ensemble_size + 1))
    return SHARED


def window_replicates(sampling) -> tuple[int, ...]:
    """Replicates with their own umbrella windows — always every replicate.

    The windows are the one stage that is fanned out in every mode; this is the
    bottom of the pipeline, where the replicate axis has always lived.
    """
    return tuple(range(1, sampling.ensemble_size + 1))


def source_replicate(replicate: int, upstream: tuple[int, ...]) -> int:
    """Which ``upstream`` replicate a downstream ``replicate`` draws from.

    A fanned-out upstream stage gives each replicate its own copy; a shared one
    gives every replicate ``run_01``. Keeping this in one place means the
    downstream code never branches on ``ensemble_from``.
    """
    return replicate if replicate in upstream else SHARED[0]


def smd_replicates(sampling) -> tuple[int, ...]:
    """Replicates with their own steered-MD pull and snapshot set.

    ``(1,)`` unless ``ensemble_from`` is ``"smd"`` or ``"equilibration"``; the
    latter implies it, because a replicate with its own equilibrated structure
    has to pull from that structure.
    """
    if sampling.ensemble_smd:
        return tuple(range(1, sampling.ensemble_size + 1))
    return SHARED
