"""Tests for steered-MD window scheduling + backend dispatch (the MD run itself is
integration-verified)."""

import json
import pathlib

from gluebind.backend.base import Backend, JobState
from gluebind.backend.scheduler import Scheduler
from gluebind.simulation import separation_window_targets
from gluebind.simulation.steered_md import (
    SMD_RESULT_FILENAME,
    SMD_SPEC_FILENAME,
    SmdSpec,
    make_steered_md_runner,
    smd_launch_command,
)


def test_separation_window_targets_sorted_unique():
    assert separation_window_targets([1.0, 0.5, 1.0, 0.9]) == [0.5, 0.9, 1.0]


def test_separation_window_targets_rounds():
    assert separation_window_targets([0.90001, 0.9]) == [0.9]


def test_smd_snapshot_targets_dense_grid_and_windows_subset():
    import pytest

    from gluebind.config.sampling import SamplingConfig
    from gluebind.runners.window import enumerate_centres
    from gluebind.simulation.steered_md import smd_snapshot_targets

    sep = SamplingConfig().for_cv("separation", "separation")
    targets = smd_snapshot_targets(sep)
    assert targets[0] == 0.9
    assert targets[-1] == 4.0  # smd_capture_max, denser than the US schedule
    assert targets[1] == pytest.approx(0.95)  # 0.05 nm spacing
    # every US window centre must land on the snapshot grid (so it has a seed frame)
    grid = set(targets)
    assert all(round(c, 4) in grid for c in enumerate_centres(sep))


def _smd_spec(tmp_path):
    return SmdSpec(
        topology="t.prm7",
        coordinates="c.rst7",
        out_dir=str(tmp_path / "frames"),
        rec_group=[1, 2],
        lig_group=[3, 4],
        anchors={"b": 1, "c": 2, "B": 3, "C": 4},
        rmsd_atoms_bound={"receptor": [1, 2]},
        boresch_eq_values={"thetaA": 1.0},
        window_centres=[1.5, 2.0],
        hmr_factor=1.5,
        pme_cutoff_nm=1.0,
        timestep_fs=4.0,
        temperature_K=298.15,
    )


def test_smd_spec_roundtrip(tmp_path):
    spec = _smd_spec(tmp_path)
    path = spec.dump(tmp_path / SMD_SPEC_FILENAME)
    assert SmdSpec.load(path) == spec


def test_smd_launch_command():
    cmd = smd_launch_command()
    assert cmd[:2] == ["python", "-c"]
    assert "run_smd" in cmd[2]


class _FakeSmdBackend(Backend):
    """Simulates the SMD job: records the spec and writes the frames result.json."""

    def __init__(self):
        self.submitted: list[SmdSpec] = []
        self._counter = 0

    def submit(self, spec):
        wd = pathlib.Path(spec.work_dir)
        smd_spec = SmdSpec.load(wd / SMD_SPEC_FILENAME)
        self.submitted.append(smd_spec)
        frames = {str(c): f"{c}nm.rst7" for c in smd_spec.window_centres}
        (wd / SMD_RESULT_FILENAME).write_text(json.dumps(frames))
        self._counter += 1
        return f"smd-{self._counter}"

    def poll(self, handles):
        return dict.fromkeys(handles, JobState.FINISHED)

    def cancel(self, handle):  # pragma: no cover
        pass


class _Sampling:
    hmr_factor = 1.5
    pme_cutoff_nm = 1.0
    timestep_fs = 4.0
    temperature_K = 298.15

    class separation:
        smd_pull_margin = 0.5


def test_make_steered_md_runner_submits_backend_job(tmp_path):
    backend = _FakeSmdBackend()
    runner = make_steered_md_runner(
        backend=backend,
        scheduler_factory=lambda: Scheduler(backend, poll_interval=0.0),
        work_dir=tmp_path / "smd",
        out_dir=tmp_path / "frames",
        topology="t.prm7",
        coordinates="c.rst7",
        rec_group=[1, 2],
        lig_group=[3, 4],
        anchors={"b": 1, "c": 2, "B": 3, "C": 4},
        rmsd_atoms_bound={"receptor": [1, 2]},
        snapshot_centres=[2.0, 1.5, 1.5],
        sampling=_Sampling(),
    )

    frames = runner({"thetaA": 1.0})

    assert len(backend.submitted) == 1
    spec = backend.submitted[0]
    assert spec.boresch_eq_values == {"thetaA": 1.0}
    assert spec.window_centres == [1.5, 2.0]  # deduped + sorted
    assert frames == {1.5: "1.5nm.rst7", 2.0: "2.0nm.rst7"}
