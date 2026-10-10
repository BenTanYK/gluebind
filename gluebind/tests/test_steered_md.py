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

    s = SamplingConfig()
    s.separation.window_min = 0.9
    sep = s.for_cv("separation", "separation")
    targets = smd_snapshot_targets(sep)
    assert targets[0] == 0.9
    assert targets[-1] == 4.0  # smd_capture_max, denser than the US schedule
    assert targets[1] == pytest.approx(0.95)  # 0.05 nm spacing
    # every US window centre must land on the snapshot grid (so it has a seed frame)
    grid = set(targets)
    assert all(round(c, 4) in grid for c in enumerate_centres(sep))


def test_smd_snapshot_targets_start_from_a_resolved_auto_window_min():
    import pytest

    from gluebind.config.sampling import SamplingConfig
    from gluebind.simulation.steered_md import smd_snapshot_targets

    sep = SamplingConfig().for_cv("separation", "separation")  # window_min "auto"
    with pytest.raises(ValueError, match="must be resolved"):
        smd_snapshot_targets(sep)
    targets = smd_snapshot_targets(sep, window_min=0.85)
    assert targets[0] == 0.85 and targets[-1] == 4.0


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


def test_smd_spec_uses_the_published_force_constants(tmp_path):
    spec = _smd_spec(tmp_path)
    assert (spec.k_smd, spec.k_rmsd, spec.k_boresch) == (100.0, 100.0, 200.0)
    assert spec.smd_compression_margin == 0.1


# ---- steering schedule: compress, then pull apart --------------------------


def test_smd_pull_plan_compresses_from_the_measured_start():
    # Regression: steering started from a hard-coded 1.15 nm whatever the real
    # separation, kicking the complex and saving every target below it from the
    # same first frame.
    import pytest

    from gluebind.simulation.steered_md import smd_pull_plan

    targets = [0.9, 1.0, 2.0, 4.0]
    compressed, step, n_compress = smd_pull_plan(
        1.17, targets, compression_margin=0.1, pull_margin=0.5, n_pull_increments=1000
    )
    assert compressed == pytest.approx(0.8)  # smallest target minus the margin
    assert step == pytest.approx((4.0 + 0.5 - 0.8) / 1000)  # outward rate
    assert n_compress == 100  # (1.17 - 0.8) / step increments at the same rate


def test_smd_pull_plan_with_zero_margin_compresses_to_the_smallest_target():
    import pytest

    from gluebind.simulation.steered_md import smd_pull_plan

    compressed, _, n_compress = smd_pull_plan(
        1.17, [0.9, 2.0], compression_margin=0.0, pull_margin=0.5, n_pull_increments=10
    )
    assert compressed == pytest.approx(0.9)
    assert n_compress > 0


def test_smd_pull_plan_skips_compression_below_the_compressed_point():
    from gluebind.simulation.steered_md import smd_pull_plan

    compressed, _, n_compress = smd_pull_plan(
        0.75, [0.9, 2.0], compression_margin=0.1, pull_margin=0.5, n_pull_increments=10
    )
    assert (compressed, n_compress) == (0.75, 0)  # pulled out from where it is


def test_smd_launch_command():
    cmd = smd_launch_command()
    assert cmd[:2] == ["python", "-c"]
    assert "run_smd" in cmd[2]


class _FakeSmdBackend(Backend):
    """Simulates the SMD job: records the spec, writes a frame per snapshot centre
    (except those in ``skip``) and the result.json mapping centre -> frame."""

    def __init__(self, skip=()):
        self.submitted: list[SmdSpec] = []
        self._counter = 0
        self.skip = set(skip)

    def submit(self, spec):
        from gluebind.simulation.steered_md import smd_frame_path

        wd = pathlib.Path(spec.work_dir)
        smd_spec = SmdSpec.load(wd / SMD_SPEC_FILENAME)
        self.submitted.append(smd_spec)
        frames = {}
        for c in smd_spec.window_centres:
            if c in self.skip:
                continue  # the pull "did not reach" this separation
            path = smd_frame_path(smd_spec.out_dir, c)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frame")
            frames[str(c)] = str(path)
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
        smd_compression_margin = 0.1


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
    assert frames == {
        1.5: str(tmp_path / "frames" / "1.5nm.rst7"),
        2.0: str(tmp_path / "frames" / "2nm.rst7"),
    }
    assert spec.smd_compression_margin == 0.1


def test_steered_md_runner_keeps_a_zero_compression_margin(tmp_path):
    # 0 is a valid margin (compress to exactly window_min), not "unset".
    class _ZeroMargin(_Sampling):
        class separation:
            smd_pull_margin = 0.5
            smd_compression_margin = 0.0

    backend = _FakeSmdBackend()
    make_steered_md_runner(
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
        snapshot_centres=[1.5],
        sampling=_ZeroMargin(),
    )({"thetaA": 1.0})
    assert backend.submitted[0].smd_compression_margin == 0.0


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


def _runner(tmp_path, backend, warnings, snapshots=(1.5, 2.0), windows=(1.5, 2.0)):
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
        snapshot_centres=list(snapshots),
        sampling=_Sampling(),
        window_centres=list(windows),
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


# ---- a failed or truncated pull is never taken as complete -----------------


class _CrashedSmdBackend(_FakeSmdBackend):
    """The job leaves the queue (reported finished) without writing anything."""

    def submit(self, spec):
        self._counter += 1
        return f"smd-{self._counter}"


def test_steered_md_runner_raises_when_the_job_produced_no_result(tmp_path):
    # Regression: a crashed SMD job (reported finished by Slurm) returned no
    # frames silently and was then recorded as done.
    import pytest

    with pytest.raises(RuntimeError, match="produced no result"):
        _runner(tmp_path, _CrashedSmdBackend(), [])({"thetaA": 1.0})


def test_steered_md_runner_ignores_a_stale_result_from_an_earlier_run(tmp_path):
    import pytest

    (tmp_path / "smd").mkdir()
    (tmp_path / "smd" / SMD_RESULT_FILENAME).write_text("{}")  # earlier run's
    with pytest.raises(RuntimeError, match="produced no result"):
        _runner(tmp_path, _CrashedSmdBackend(), [])({"thetaA": 1.0})


def test_steered_md_runner_raises_when_a_window_frame_is_missing(tmp_path):
    import pytest

    backend = _FakeSmdBackend(skip={2.0})  # the pull never reached 2.0 nm
    with pytest.raises(RuntimeError, match="separation window.* at 2 nm"):
        _runner(tmp_path, backend, [])({"thetaA": 1.0})


def test_steered_md_runner_only_warns_for_a_missing_spare_snapshot(tmp_path):
    warnings: list[str] = []
    backend = _FakeSmdBackend(skip={2.5})  # captured for later windows only
    runner = _runner(
        tmp_path, backend, warnings, snapshots=(1.5, 2.0, 2.5), windows=(1.5, 2.0)
    )
    frames = runner({"thetaA": 1.0})
    assert set(frames) == {1.5, 2.0}
    assert len(warnings) == 1 and "spare snapshot(s) at 2.5 nm" in warnings[0]


def test_missing_frames_treats_empty_files_as_missing(tmp_path):
    from gluebind.simulation.steered_md import missing_frames, smd_frame_path

    smd_frame_path(tmp_path, 1.0).write_text("frame")
    smd_frame_path(tmp_path, 1.5).write_text("")  # truncated write
    assert missing_frames(tmp_path, [2.0, 1.5, 1.0, 1.0]) == [1.5, 2.0]
