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
    state_data_interval_steps = 10000

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


# ---- periodic-image check --------------------------------------------------


def test_closest_periodic_image_across_a_cubic_face():
    import numpy as np
    import pytest

    from gluebind.simulation.steered_md import closest_periodic_image

    # 3 A apart through the +x face of a 30 A box, 27 A apart within the cell
    positions = np.array([[1.0, 15.0, 15.0], [28.0, 15.0, 15.0], [50.0, 50.0, 50.0]])
    distance, i, j = closest_periodic_image(positions, 30.0 * np.eye(3), [0, 1])
    assert distance == pytest.approx(3.0)
    assert {i, j} == {0, 1}  # atom 2 is not solute and is ignored


def test_closest_periodic_image_in_a_truncated_octahedron():
    import numpy as np
    import pytest
    from MDAnalysis.lib.mdamath import triclinic_vectors

    from gluebind.simulation.steered_md import closest_periodic_image

    # A single atom's nearest image is one shortest lattice vector away: the edge.
    box = triclinic_vectors(np.array([100.0, 100.0, 100.0, 70.5288, 109.4712, 70.5288]))
    distance, i, j = closest_periodic_image(np.array([[3.0, 4.0, 5.0]]), box, [0])
    assert distance == pytest.approx(100.0, rel=1e-4)
    assert i == j == 0


def test_periodic_image_warning_is_none_when_all_frames_are_clear():
    from gluebind.simulation.steered_md import periodic_image_warning

    images = {"1.5": {"distance_A": 40.0, "atoms": [1, 2]}}
    assert periodic_image_warning(images, [1.5]) is None


def test_periodic_image_warning_names_close_frames_and_affected_windows():
    from gluebind.simulation.steered_md import periodic_image_warning

    images = {
        "2.5": {"distance_A": 20.0, "atoms": [1, 2]},
        "3.0": {"distance_A": 14.0, "atoms": [1, 2]},
        "3.5": {"distance_A": 9.5, "atoms": [3, 4]},
    }
    message = periodic_image_warning(images, window_centres=[2.5, 3.0])
    assert "2 frame(s) between 3 and 3.5 nm" in message
    assert "closest 9.5 A at 3.5 nm" in message
    assert "windows at 3 nm" in message  # 3.5 nm was only captured, not sampled


class _FakeSmdBackendWithImages(_FakeSmdBackend):
    """Also writes the periodic-image report, with one frame too close."""

    def submit(self, spec):
        handle = super().submit(spec)
        report = {"1.5": {"distance_A": 30.0, "atoms": [1, 2]}}
        report["2.0"] = {"distance_A": 12.0, "atoms": [3, 4]}
        path = pathlib.Path(spec.work_dir) / "periodic_images.json"
        path.write_text(json.dumps(report))
        return handle


def _runner(tmp_path, backend, warnings):
    return make_steered_md_runner(
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
        snapshot_centres=[1.5, 2.0],
        sampling=_Sampling(),
        window_centres=[1.5, 2.0],
        warn=warnings.append,
    )


def test_steered_md_runner_warns_once_for_periodic_image_contact(tmp_path):
    warnings: list[str] = []
    _runner(tmp_path, _FakeSmdBackendWithImages(), warnings)({"thetaA": 1.0})
    assert len(warnings) == 1
    assert "closest 12.0 A at 2 nm" in warnings[0]


def test_steered_md_runner_is_silent_without_a_report(tmp_path):
    warnings: list[str] = []
    _runner(tmp_path, _FakeSmdBackend(), warnings)({"thetaA": 1.0})
    assert warnings == []
