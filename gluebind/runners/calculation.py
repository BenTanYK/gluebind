"""Calculation runner: the top-level driver.

Builds the ``Group → Stage → Window`` tree from a :class:`CalculationConfig`,
submits each window replicate through a :class:`~gluebind.backend.base.Backend`
(via the :class:`~gluebind.backend.scheduler.Scheduler`), tracks opaque handles
in ``.gluebind-state.json``, resumes by skipping replicates already complete on
disk, and aggregates the per-stage PMFs into the standard-state binding free
energy.

Two seams are injected because they belong to other phases:

* ``spec_builder`` — produces a fully-resolved :class:`WindowSpec` for a window
  (topology, coordinates, resolved restraint atom indices). Phase 3 (BioSimSpace
  prep) + Phase 4 (selection) supply the real one; tests supply a trivial one.
* ``pmf_provider`` (to :meth:`analyse`) — returns ``(cv, pmf)`` for a stage,
  normally by running WHAM over the stage's windows and averaging replicates.

The Boresch stages are sequential (each DoF's equilibrium value is the previous
PMF's minimum); ``stage_centres`` supplies their window centres (from the
unrestrained-MD distribution) and separation's (from the SMD frames), while RMSD
window centres come from the sampling schedule.
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Callable, Iterator

from gluebind.analysis.free_energy import (
    binding_free_energy,
    boresch_contribution,
    contribution_converged,
    rmsd_contribution,
    separation_contribution,
    separation_plateau_reached,
    standard_state_correction,
)
from gluebind.analysis.pmf import pmf_minimum
from gluebind.boresch_geometry import DOFS as BORESCH_DOFS
from gluebind.backend.base import Backend
from gluebind.backend.scheduler import Scheduler, SlotPool
from gluebind.config.calculation import CalculationConfig
from gluebind.config.slurm import SlurmConfig
from gluebind.logutil import add_file_handler, get_logger
from gluebind.runners.base import SimulationRunner
from gluebind.runners.group import Group
from gluebind.runners.stage import Stage
from gluebind.runners.window import SpecBuilder, Window, enumerate_centres
from gluebind.simulation.window import window_launch_command
from gluebind.state import RunState, now_utc_iso

# Force constants live in the config in Å^-2, but the WHAM PMFs (and hence the
# free-energy integrals) work in nm. 1 Å^-2 = 100 nm^-2.
_A2_TO_NM2 = 100.0

PmfProvider = Callable[[Stage], "tuple"]


def _sem(values) -> float:
    """Standard error of the mean (ddof=1); 0 for fewer than two samples."""
    import numpy as np

    a = np.asarray(values, dtype=float)
    n = a.size
    return float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def _rmsf_report_header(candidates) -> str:
    """Header for the per-protein RMSF report ``.dat``.

    ``candidates`` is a list of ``(resid, atom_index)`` pairs, most-stable first;
    ``atom_index`` is the 0-indexed *complex* atom index of that residue's Cα — exactly
    what ``BoreschSpec.anchors`` expects, so a user can read a stable resid off the RMSF
    plot and paste its atom index straight into the config. RMSF is in Ångström
    (MDAnalysis works in Å; ``compute_rmsf`` passes those values through).
    """
    pretty = ", ".join(f"resid {r}=atom {i}" for r, i in candidates)
    return (
        f"suggested stable anchor candidates (low-RMSF local minima, most stable "
        f"first): {pretty}\nresid  atom_index  rmsf(Angstrom)"
    )


def _repeat_dg_sem(per_repeat: dict, dg_corr: float) -> float | None:
    """SEM of ΔG° over independent repeats.

    ``per_repeat`` maps stage name → ``(cv_type, [contribution per repeat])``. Each
    repeat's contributions are combined into a full ΔG° (same sign convention as
    the point estimate; the analytical ``dg_corr`` is constant across repeats), and
    the SEM is taken over those totals. Returns ``None`` when there are fewer than
    two repeats or the stages disagree on repeat count.
    """
    if not per_repeat:
        return None
    n = len(next(iter(per_repeat.values()))[1])
    if n < 2 or not all(len(vals) == n for _cv, vals in per_repeat.values()):
        return None
    totals = []
    for r in range(n):
        comp = {"boresch": 0.0, "rmsd": 0.0, "separation": 0.0}
        for _name, (cv_type, vals) in per_repeat.items():
            comp[cv_type] += vals[r]
        totals.append(
            binding_free_energy(
                comp["rmsd"], comp["boresch"], comp["separation"], dg_corr
            )
        )
    return _sem(totals)


class Calculation(SimulationRunner):
    """Drive one binding-free-energy calculation end to end."""

    def __init__(
        self,
        base_dir: str | pathlib.Path,
        config: CalculationConfig,
        backend: Backend,
        spec_builder: SpecBuilder | None = None,
        *,
        slurm_config: SlurmConfig | None = None,
        command_factory: Callable[[], list[str]] = window_launch_command,
        stage_centres: dict[str, list[float]] | None = None,
        steered_md_runner: Callable[[dict], object] | None = None,
        platform: str = "CUDA",
        poll_interval: float = 30.0,
    ) -> None:
        super().__init__(base_dir)
        self.config = config
        self.backend = backend
        self.spec_builder = spec_builder
        self.slurm_config = slurm_config
        self.command_factory = command_factory
        self.stage_centres = stage_centres or {}
        # Generates the separation-window SMD frames from the Boresch equilibrium
        # values; invoked automatically between the Boresch and separation stages.
        self.steered_md_runner = steered_md_runner
        self.platform = platform
        self.poll_interval = poll_interval
        self.prepared = None
        # Per-calculation child logger: keeps each run's gluebind.log isolated while
        # still propagating up to any root handler (e.g. a CalcSet aggregate log).
        self._log = get_logger(f"calc.{self.base_dir.name}")
        # When built via from_config the wiring is deferred to prepare(); with a
        # spec_builder supplied directly (tests / advanced use) the tree is built now.
        self.groups = self._build_groups() if spec_builder is not None else []
        self.sub_runners = list(self.groups)

    @classmethod
    def from_config(
        cls,
        config: "CalculationConfig | str | pathlib.Path",
        base_dir: str | pathlib.Path,
        backend: Backend,
        *,
        slurm_config: SlurmConfig | None = None,
        command_factory: Callable[[], list[str]] = window_launch_command,
        platform: str = "CUDA",
        poll_interval: float = 30.0,
    ) -> "Calculation":
        """Build a calculation from a config (path or object); prep/wiring deferred.

        Construction is cheap — the heavy work (system prep, restraint context,
        window centres, steered-MD hook) runs in :meth:`prepare`, which :meth:`run`
        calls automatically. So the whole calculation runs end to end from a single
        call: ``from_config(...).run()`` (then :meth:`analyse` for the ΔG°). Call
        :meth:`prepare` explicitly only if you want to inspect the prepared system
        before sampling.
        """
        if not isinstance(config, CalculationConfig):
            config_path = pathlib.Path(config)
            config = CalculationConfig.load(config_path).with_resolved_input_paths(
                config_path.parent
            )
        return cls(
            base_dir,
            config,
            backend,
            slurm_config=slurm_config,
            command_factory=command_factory,
            platform=platform,
            poll_interval=poll_interval,
        )

    def prepare(self):
        """Prepare the system and wire the runner from the config alone.

        Runs system prep through the backend (no MD on the driver), resolves the
        restraint context, computes the Boresch/separation window centres, and
        builds the ``spec_builder`` and the backend-dispatched steered-MD hook.
        Returns the :class:`~gluebind.system.prep.PreparedSystem`.

        Idempotent: if the system is already prepared (``prep/prepared.json``
        exists) the equilibration is not re-run — the manifest is loaded and only
        the cheap driver-side wiring (context/centres) is rebuilt. This is what
        lets :meth:`run` auto-prepare safely on a resumed run. Called
        automatically by :meth:`run` when the calculation is not yet wired.
        """
        from gluebind.system.prep import PreparedSystem
        from gluebind.system.prep import prepare as prepare_system

        prep_dir = self.base_dir / "prep"
        try:
            prepared = PreparedSystem.load(prep_dir)  # resume: prep already complete
            self._log.info(
                "prepare %s: reusing existing prepared system", self.base_dir.name
            )
        except FileNotFoundError:
            self._log.info(
                "prepare %s: building and equilibrating system", self.base_dir.name
            )
            prepared = prepare_system(
                self.config,
                prep_dir,
                self.backend,
                platform=self.platform,
                poll_interval=self.poll_interval,
            )
        self._wire(prepared)
        return prepared

    def equilibrate(self):
        """Run only the equilibration (+ bulk extraction) and write a per-protein Cα
        RMSF report for manual Boresch-anchor selection — **without** resolving
        anchors or building the sampling tree.

        The manual-anchor fallback (the workflow the paper used): call this,
        inspect ``prep/rmsf_{receptor,target}.dat`` (each ``resid  atom_index  rmsf``
        plus the auto-suggested stable candidates, listed as ``resid=atom``), set
        ``restraints.boresch.anchors = {"b": ..., "c": ..., "B": ..., "C": ...}`` to the
        chosen residues' ``atom_index`` values (0-indexed complex atoms), then call
        :meth:`run` — which reuses this equilibration (idempotent) and wires with the
        chosen anchors.

        Returns the :class:`~gluebind.system.prep.PreparedSystem`.
        """
        from gluebind.system.prep import PreparedSystem
        from gluebind.system.prep import prepare as prepare_system

        add_file_handler(self.base_dir, logger_name=self._log.name)
        prep_dir = self.base_dir / "prep"
        try:
            prepared = PreparedSystem.load(prep_dir)
            self._log.info(
                "equilibrate %s: reusing existing prepared system", self.base_dir.name
            )
        except FileNotFoundError:
            self._log.info(
                "equilibrate %s: building and equilibrating system", self.base_dir.name
            )
            prepared = prepare_system(
                self.config,
                prep_dir,
                self.backend,
                platform=self.platform,
                poll_interval=self.poll_interval,
            )
        report = self._write_rmsf_report(prepared)
        self._log.info(
            "equilibrate %s: RMSF report for anchor selection -> %s",
            self.base_dir.name,
            ", ".join(report.values()),
        )
        return prepared

    def _write_rmsf_report(self, prepared) -> dict[str, str]:
        """Write per-protein Cα RMSF (``resid  atom_index  rmsf``) + suggested stable
        candidates to ``prep/rmsf_<protein>.dat`` for manual anchor inspection.

        ``atom_index`` is each Cα's 0-indexed complex atom index — the value
        ``BoreschSpec.anchors`` takes — so the workflow is read-resid-off-the-plot →
        paste-its-atom-index."""
        import MDAnalysis as mda
        import numpy as np

        from gluebind.selection.rmsf import compute_rmsf, stablest_candidates
        from gluebind.spec_builder import _ComplexMap

        if prepared.complex_trajectory is None:
            raise RuntimeError(
                "cannot write an RMSF report: the equilibration produced no "
                "trajectory (prepared.complex_trajectory is None)"
            )
        universe = mda.Universe(prepared.complex_prm7, prepared.complex_trajectory)
        cmap = _ComplexMap(
            universe,
            mda.Universe(self.config.inputs.target.prm7),
            mda.Universe(self.config.inputs.receptor.prm7),
            has_glue=self.config.inputs.glue is not None,
        )
        prep_dir = self.base_dir / "prep"
        report: dict[str, str] = {}
        for protein in ("receptor", "target"):
            ca_selection = "index " + " ".join(
                map(str, cmap.resolve(protein, "name CA"))
            )
            resids, rmsf = compute_rmsf(universe, selection=ca_selection)
            # Complex atom indices of those Cα atoms, aligned with resids/rmsf (same
            # AtomGroup order) — this is what BoreschSpec.anchors takes.
            atom_indices = universe.select_atoms(ca_selection).indices
            resid_to_atom = {
                int(r): int(i) for r, i in zip(resids, atom_indices, strict=True)
            }
            # Preserve stablest_candidates' rank order (most stable first).
            candidates = [
                (r, resid_to_atom[r]) for r in stablest_candidates(resids, rmsf)
            ]
            path = prep_dir / f"rmsf_{protein}.dat"
            header = _rmsf_report_header(candidates)
            np.savetxt(
                path,
                np.column_stack(
                    [
                        np.asarray(resids),
                        np.asarray(atom_indices),
                        np.asarray(rmsf, float),
                    ]
                ),
                fmt=["%d", "%d", "%.4f"],
                header=header,
            )
            report[protein] = str(path)
        return report

    def _load_prepared(self):
        """Return the on-disk :class:`PreparedSystem`, or ``None`` if not prepared."""
        from gluebind.system.prep import PreparedSystem

        try:
            return PreparedSystem.load(self.base_dir / "prep")
        except FileNotFoundError:
            return None

    def _wire(self, prepared) -> None:
        """Build the restraint context, window centres, spec builder and steered-MD
        hook from a prepared system, and construct the group tree. Driver-side only
        (reads the trajectory; runs no MD) — shared by :meth:`prepare` and the
        re-wiring :meth:`analyse` does in a fresh process."""
        from gluebind.simulation.steered_md import (
            make_steered_md_runner,
            smd_snapshot_targets,
        )
        from gluebind.spec_builder import SpecBuilder, build_restraint_context
        from gluebind.stage_centres import compute_stage_centres

        context = build_restraint_context(prepared, self.config)
        self.stage_centres = compute_stage_centres(prepared, context, self.config)
        # SMD saves a dense snapshot grid (decoupled from — and finer than — the US
        # window schedule), so windows can be added later without re-running SMD.
        snapshot_centres = smd_snapshot_targets(
            self.config.sampling.for_cv("separation", "separation")
        )

        smd_frames_dir = self.base_dir / "smd_frames"
        self.spec_builder = SpecBuilder(
            context, self.config, smd_frames_dir=smd_frames_dir
        )
        self.steered_md_runner = make_steered_md_runner(
            backend=self.backend,
            scheduler_factory=self._default_scheduler,
            work_dir=self.base_dir / "smd",
            out_dir=smd_frames_dir,
            topology=context.complex_topology,
            coordinates=context.complex_coordinates,
            rec_group=context.rec_group,
            lig_group=context.lig_group,
            anchors=context.anchors,
            rmsd_atoms_bound=context.rmsd_atoms_bound,
            snapshot_centres=snapshot_centres,
            sampling=self.config.sampling,
            platform=self.platform,
        )
        self.prepared = prepared
        self.groups = self._build_groups()
        self.sub_runners = list(self.groups)

    # ---- tree construction -------------------------------------------------

    def _build_groups(self) -> list[Group]:
        groups: list[Group] = []
        for cv_type, stage_specs in self._stage_layout().items():
            stages = [
                Stage(
                    self.base_dir / cv_type / name,
                    cv_type=cv_type,
                    name=name,
                    dof=dof,
                    centres=centres,
                    ensemble_size=self.config.sampling.ensemble_size,
                    spec_builder=self.spec_builder,
                    command_factory=self.command_factory,
                )
                for name, dof, centres in stage_specs
            ]
            groups.append(
                Group(self.base_dir / cv_type, cv_type=cv_type, stages=stages)
            )
        return groups

    def _stage_layout(self) -> dict[str, list[tuple[str, str | None, list[float]]]]:
        """Return ``cv_type -> [(stage_name, dof, centres)]`` for every stage."""
        layout: dict[str, list] = {"boresch": [], "rmsd": [], "separation": []}

        # Boresch: five sequential DoFs; centres must be supplied (MD distribution).
        for dof in BORESCH_DOFS:
            if dof in self.stage_centres:
                layout["boresch"].append((dof, dof, list(self.stage_centres[dof])))

        # RMSD: one stage per (region, state); centres from the sampling schedule.
        for name in self._rmsd_stage_names():
            centres = self.stage_centres.get(name) or enumerate_centres(
                self.config.sampling.for_cv("rmsd", name)
            )
            layout["rmsd"].append((name, None, centres))

        # Separation: single stage; centres from the SMD frames.
        if "separation" in self.stage_centres:
            layout["separation"].append(
                ("separation", None, list(self.stage_centres["separation"]))
            )

        return {cv: stages for cv, stages in layout.items() if stages}

    def _rmsd_stage_names(self) -> list[str]:
        # Separation-PMF-only mode: skip the RMSD US *stages* only. The RMSD atom
        # groups are still resolved by build_restraint_context, so the fixed RMSD
        # restraints remain in the Boresch/SMD/separation windows — we just don't
        # sample (or score) ΔG_c^bound/ΔG_c^bulk.
        if not self.config.sampling.run_rmsd_us:
            return []
        restraints = self.config.restraints
        if restraints.uses_default_all_ca:
            regions = [("receptor", ("bound", "bulk")), ("target", ("bound", "bulk"))]
        else:
            regions = [(cv.name, tuple(cv.states)) for cv in restraints.rmsd_cvs]
        names: list[str] = []
        for region, states in regions:
            names += [f"{region}_{state}" for state in states]
        return names

    # ---- iteration helpers -------------------------------------------------

    def _iter_stages(self) -> Iterator[tuple[Group, Stage]]:
        for group in self.groups:
            for stage in group.stages:
                yield group, stage

    def _iter_windows(self) -> Iterator[Window]:
        for _group, stage in self._iter_stages():
            yield from stage.windows

    # ---- run ---------------------------------------------------------------

    def setup(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config.dump_resolved(self.base_dir)
        for group in self.groups:
            group.setup()

    def _load_or_init_state(self) -> RunState:
        try:
            state = RunState.load(self.base_dir)
        except FileNotFoundError:
            return RunState(
                calc_id=self.base_dir.name,
                submitted_at=now_utc_iso(),
                config_hash=self.config.config_hash,
                config_path=str(self.base_dir),
            )
        if state.config_hash != self.config.config_hash:
            raise ValueError(
                "config_hash mismatch: the config changed since this run was "
                "submitted; resume aborted. Start a fresh run directory or restore "
                "the original config."
            )
        return state

    def _default_scheduler(self, job_slots: SlotPool | None = None) -> Scheduler:
        return Scheduler(
            self.backend,
            queue_len_lim=self.slurm_config.queue_len_lim
            if self.slurm_config
            else 2000,
            poll_interval=(
                self.slurm_config.queue_check_interval
                if self.slurm_config
                else self.poll_interval
            ),
            slots=job_slots,
        )

    def _group(self, cv_type: str) -> Group | None:
        for group in self.groups:
            if group.cv_type == cv_type:
                return group
        return None

    def add_windows(self, cv_type: str, stage_name: str, centres) -> Stage:
        """Add umbrella-sampling windows to a stage, then re-``run``/re-``analyse``.

        For the extensibility workflow: after analysing, if a stage has poor
        overlap (or a separation PMF that has not plateaued), add intermediate or
        extended ``centres`` here, call :meth:`run` (which resumes — only the new
        windows are submitted), then :meth:`analyse` (which now includes them).

        For **separation**, each centre must already have an SMD snapshot frame
        (snapshots are saved on a dense grid up to ``smd_capture_max``); a centre
        off that grid raises, since it has no starting structure — rerun SMD with
        bespoke spacing/range for that.
        """
        if self.spec_builder is None:
            raise RuntimeError("wire the calculation first (call prepare()/run())")
        group = self._group(cv_type)
        stage = (
            next((s for s in group.stages if s.name == stage_name), None)
            if group
            else None
        )
        if stage is None:
            raise ValueError(f"no {cv_type!r} stage named {stage_name!r}")

        if cv_type == "separation":
            sep = self.config.sampling.for_cv("separation", "separation")
            frames_dir = self.base_dir / "smd_frames"
            for centre in centres:
                frame = frames_dir / f"{float(centre):.4g}nm.rst7"
                if not frame.exists():
                    raise ValueError(
                        f"no SMD snapshot for a separation window at {centre} nm "
                        f"({frame.name}). Snapshots are saved on a "
                        f"{sep.smd_snapshot_spacing} nm grid up to "
                        f"{sep.smd_capture_max} nm; "
                        "to sample a separation off that grid or beyond it, rerun SMD "
                        "with bespoke spacing/range."
                    )

        stage.add_windows(centres)
        return stage

    def run(
        self,
        *,
        scheduler: Scheduler | None = None,
        pmf_provider: PmfProvider | None = None,
        job_slots: SlotPool | None = None,
    ) -> RunState:
        """Run the whole calculation, honouring the stage dependencies.

        Order: the independent RMSD stages; then the **sequential** Boresch stages
        (each DoF's equilibrium value is the minimum of its PMF, fed forward as a
        fixed restraint to the next DoF and to separation); then the separation
        stage. Idempotent and resumable: replicates already complete on disk are
        skipped, and Boresch DoFs whose equilibrium value is already recorded in
        the state are not re-run — so an interrupted run continues mid-sequence.

        ``pmf_provider(stage) -> (cv, pmf)`` is required whenever there are Boresch
        stages still to analyse (their equilibrium values are their PMF minima).
        ``job_slots`` is a shared :class:`SlotPool` used when several calculations
        run concurrently (a parallel ``CalcSet``) to cap total in-flight jobs.
        """
        add_file_handler(self.base_dir, logger_name=self._log.name)
        self._log.info(
            "run %s: starting (backend=%s)",
            self.base_dir.name,
            type(self.backend).__name__,
        )
        if self.spec_builder is None:
            # Auto-prepare (idempotent): a from_config calculation runs end to end
            # from run() alone; prep is skipped if already complete on disk.
            self.prepare()
        self.setup()
        state = self._load_or_init_state()
        scheduler = scheduler or self._default_scheduler(job_slots)

        # 1. RMSD stages — independent of the Boresch equilibrium values.
        rmsd_group = self._group("rmsd")
        if rmsd_group:
            for stage in rmsd_group.stages:
                self._run_stage(stage, {}, state, scheduler)

        # 2. Boresch stages — sequential, feeding each PMF minimum forward.
        boresch_group = self._group("boresch")
        if boresch_group:
            unanalysed = [
                s for s in boresch_group.stages if s.dof not in state.boresch_eq_values
            ]
            if unanalysed and pmf_provider is None:
                # Self-default so a from_config calculation runs end to end from a
                # single run() (symmetric with analyse()); the Boresch feedback
                # needs PMFs. WHAM runs locally on the driver (fast, CPU) by
                # default; a slurm/backed provider can still be injected.
                from gluebind.analysis.provider import WhamPmfProvider

                pmf_provider = WhamPmfProvider(self.config)
            for stage in boresch_group.stages:
                if stage.dof in state.boresch_eq_values:
                    continue  # already determined on a previous run (resume)
                self._run_stage(stage, dict(state.boresch_eq_values), state, scheduler)
                cv, pmf, *_ = pmf_provider(stage)  # ignore per-replicate PMFs here
                state.boresch_eq_values[stage.dof] = pmf_minimum(cv, pmf)
                state.save(self.base_dir)

        # 3. Steered MD then separation — both need every Boresch equilibrium value.
        separation_group = self._group("separation")
        if separation_group:
            if (
                self.steered_md_runner is not None
                and state.stage_status.get("steered_md") != "done"
            ):
                # Generate the separation-window starting frames with the resolved
                # Boresch restraints in place. Recorded in state so a resumed run
                # doesn't repeat the (expensive) steering.
                self.steered_md_runner(dict(state.boresch_eq_values))
                state.stage_status["steered_md"] = "done"
                state.save(self.base_dir)
            for stage in separation_group.stages:
                self._run_stage(stage, dict(state.boresch_eq_values), state, scheduler)

        state.save(self.base_dir)
        self._log.info("run %s: all stages complete", self.base_dir.name)
        return state

    def _run_stage(
        self,
        stage: Stage,
        boresch_eq_values: dict,
        state: RunState,
        scheduler: Scheduler,
    ) -> None:
        """Write a stage's specs (with the given Boresch eq values), submit the
        not-yet-complete replicates, and record handles/status in the state."""
        stage.write_specs(boresch_eq_values)
        pending: list[tuple[Window, int]] = [
            (window, replicate)
            for window in stage.windows
            for replicate in window.replicates()
            if not window.is_replicate_complete(replicate)
        ]
        specs = [window.job_spec(replicate) for window, replicate in pending]
        if pending:
            self._log.info(
                "stage %s: submitting %d replicate(s) across %d window(s)",
                stage.name,
                len(pending),
                len(stage.windows),
            )
        else:
            self._log.info("stage %s: already complete, skipping", stage.name)

        def on_submit(index: int, handle: str) -> None:
            window, replicate = pending[index]
            per_window = state.handles.setdefault(stage.name, {}).setdefault(
                window.label, [""] * window.ensemble_size
            )
            per_window[replicate - 1] = handle
            state.stage_status[stage.name] = "running"

        states = scheduler.run(specs, on_submit=on_submit)

        # Surface failures here, with a pointer to the dead window/replicate, rather
        # than letting them resurface downstream as a cryptic missing-file crash in
        # WHAM or the PMF provider. A submitted replicate that produced no result is
        # a failure whatever the scheduler reported (a crash, or a job that exited 0
        # without writing its result) — the scheduler state is carried for context.
        failures = [
            f"{window.label}/run_{replicate:02d} (job {states[i].value})"
            for i, (window, replicate) in enumerate(pending)
            if not window.is_replicate_complete(replicate)
        ]

        if all(
            window.is_replicate_complete(r)
            for window in stage.windows
            for r in window.replicates()
        ):
            state.stage_status[stage.name] = "done"
            self._log.info("stage %s: complete", stage.name)
        elif failures:
            state.stage_status[stage.name] = "failed"
            self._log.error(
                "stage %s: %d replicate(s) failed", stage.name, len(failures)
            )
        state.save(self.base_dir)

        if failures:
            raise RuntimeError(
                f"stage {stage.name!r}: {len(failures)} window replicate(s) "
                f"produced no result: {', '.join(failures)}. Inspect the job logs "
                "(<window>/run_NN/*.out, or the SLURM job output) under the stage "
                "directory, then re-run to resume the remaining work."
            )

    # ---- analyse -----------------------------------------------------------

    def analyse(
        self,
        pmf_provider: PmfProvider | None = None,
        *,
        r_star_nm: float | None = None,
        theta_a_min: float | None = None,
        theta_b_min: float | None = None,
    ) -> dict:
        """Aggregate per-stage PMFs into the standard-state binding free energy.

        With no arguments, everything is resolved from the run: ``pmf_provider``
        defaults to a local WHAM provider, the ``theta_*`` minima come from the
        Boresch equilibrium values in the run state, and ``r_star_nm`` is the
        outermost separation window centre. Any of them may be passed explicitly
        to override.

        ``pmf_provider(stage)`` returns ``(cv, mean_pmf)`` or, to enable the
        uncertainty estimate, ``(cv, mean_pmf, [pmf_per_repeat, ...])``. The
        returned dict carries the point-estimate ΔG° (``dg_bind`` and its
        components) plus, when per-repeat PMFs are available, ``dg_bind_sem`` (the
        SEM over the independent repeats' total ΔG°) and ``stage_sems`` (per-CV
        SEMs that flag the least-converged stage). ``dg_bind_sem`` is ``None`` and
        ``stage_sems`` empty for a single repeat.

        In separation-PMF-only mode (``sampling.run_rmsd_us=False``) there are no
        RMSD stages, so ``dg_bind`` is ``ΔG_sep + ΔG_boresch + ΔG_corr`` — a ranking
        estimate that omits ΔG_c^bound/ΔG_c^bulk. This is signalled by
        ``rmsd_included=False`` in the returned dict.

        Works in a fresh process (the detached submit → come back later → analyse
        workflow): if the calculation isn't wired, it re-wires from the on-disk
        prepared system (rebuilding the stage tree + centres, no MD) so the stages
        are actually iterated. Raises if the system was never prepared.
        """
        if self.spec_builder is None:
            prepared = self._load_prepared()
            if prepared is None:
                raise RuntimeError(
                    "cannot analyse: the system is not prepared "
                    "(no prep/prepared.json); run() first"
                )
            self._wire(prepared)

        if pmf_provider is None:
            from gluebind.analysis.provider import WhamPmfProvider

            pmf_provider = WhamPmfProvider(self.config)
        if theta_a_min is None or theta_b_min is None or r_star_nm is None:
            state = self._load_or_init_state()
            if theta_a_min is None:
                theta_a_min = state.boresch_eq_values.get("thetaA")
            if theta_b_min is None:
                theta_b_min = state.boresch_eq_values.get("thetaB")
            if theta_a_min is None or theta_b_min is None:
                raise ValueError(
                    "cannot derive theta minima: run the Boresch stages first, "
                    "or pass theta_a_min/theta_b_min explicitly"
                )
            if r_star_nm is None:
                sep = self.stage_centres.get("separation")
                if not sep:  # e.g. constructed directly with empty stage_centres
                    sep = enumerate_centres(
                        self.config.sampling.for_cv("separation", "separation")
                    )
                r_star_nm = max(sep)

        k_boresch = self.config.sampling.boresch.force_constant  # kcal/mol/rad^2
        k_rmsd = (
            self.config.sampling.rmsd.force_constant * _A2_TO_NM2
        )  # -> kcal/mol/nm^2
        # The WHAM PMFs are produced at the sampling temperature, so the β in every
        # free-energy integral must use the same temperature (not the 300 K default).
        temp = self.config.sampling.temperature_K
        totals = {"boresch": 0.0, "rmsd": 0.0, "separation": 0.0}

        def _warn_unconverged(stage: Stage, converged: bool) -> None:
            if not converged:
                warnings.warn(
                    f"contribution for stage {stage.name!r} may be unconverged: the "
                    "integrand does not decay to <1% of its maximum at the CV "
                    "extremes (<98% captured). Add windows at the offending extreme "
                    "and re-analyse.",
                    stacklevel=2,
                )

        def _stage_contribution(cv_type, stage, cv, fe, theta_0):
            """One stage's contribution for a given PMF (mean or a single repeat)."""
            if cv_type == "boresch":
                return boresch_contribution(
                    cv, fe, theta_0, k_boresch, temperature=temp
                )
            if cv_type == "rmsd":
                return rmsd_contribution(
                    cv, fe, k_rmsd, unbound=stage.is_bulk, temperature=temp
                )
            return separation_contribution(cv, fe, r_star_nm, temperature=temp)

        # stage name -> (cv_type, [contribution for each independent repeat])
        per_repeat: dict[str, tuple[str, list[float]]] = {}

        for group, stage in self._iter_stages():
            cv, pmf, *rest = pmf_provider(stage)
            replicate_pmfs = rest[0] if rest else None
            # theta_0 is a fixed protocol value (the mean-PMF minimum), reused for
            # every repeat so the repeat spread reflects sampling noise alone.
            theta_0 = pmf_minimum(cv, pmf) if group.cv_type == "boresch" else 0.0

            if group.cv_type == "boresch":
                _warn_unconverged(
                    stage,
                    contribution_converged(
                        cv,
                        pmf,
                        cv_type="boresch",
                        force_constant=k_boresch,
                        theta_0=theta_0,
                        temperature=temp,
                    ),
                )
            elif group.cv_type == "rmsd":
                _warn_unconverged(
                    stage,
                    contribution_converged(
                        cv, pmf, cv_type="rmsd", force_constant=k_rmsd, temperature=temp
                    ),
                )
            elif group.cv_type == "separation":
                reached, gradient = separation_plateau_reached(cv, pmf)
                if not reached:
                    warnings.warn(
                        f"separation PMF for stage {stage.name!r} has not plateaued "
                        f"(gradient {gradient:.2f} kcal/mol/nm over the final 0.4 nm); "
                        "run windows to larger separation (the SMD snapshots extend to "
                        "the capture range) — or extend the SMD capture beyond it.",
                        stacklevel=2,
                    )
                _warn_unconverged(
                    stage,
                    contribution_converged(
                        cv,
                        pmf,
                        cv_type="separation",
                        force_constant=0.0,
                        r_star=r_star_nm,
                        temperature=temp,
                    ),
                )

            totals[group.cv_type] += _stage_contribution(
                group.cv_type, stage, cv, pmf, theta_0
            )
            if replicate_pmfs:
                per_repeat[stage.name] = (
                    group.cv_type,
                    [
                        _stage_contribution(group.cv_type, stage, cv, fe, theta_0)
                        for fe in replicate_pmfs
                    ],
                )

        dg_corr = standard_state_correction(
            r_star_nm, theta_a_min, theta_b_min, k_boresch, temperature=temp
        )
        dg_bind = binding_free_energy(
            totals["rmsd"], totals["boresch"], totals["separation"], dg_corr
        )
        # Per-stage SEM flags the least-converged CV; the overall SEM is taken over
        # the independent repeats' full ΔG° (not a quadrature of per-stage SEMs).
        stage_sems = {name: _sem(vals) for name, (_cv, vals) in per_repeat.items()}
        dg_bind_sem = _repeat_dg_sem(per_repeat, dg_corr)

        self._log.info(
            "analyse %s: dG_bind%s = %.2f%s kcal/mol",
            self.base_dir.name,
            "" if self._group("rmsd") is not None else " (partial: no RMSD US)",
            dg_bind,
            f" +/- {dg_bind_sem:.2f}" if dg_bind_sem is not None else "",
        )
        return {
            "dg_bind": dg_bind,
            "dg_bind_sem": dg_bind_sem,  # over independent repeats; None if < 2
            "dg_rmsd": totals["rmsd"],
            "dg_boresch": totals["boresch"],
            "dg_sep": totals["separation"],
            "dg_corr": dg_corr,
            "stage_sems": stage_sems,  # {stage: SEM} — flags the least-converged CV
            # False in separation-PMF-only mode: dg_bind then omits ΔG_c^bound/bulk
            # and is a ranking estimate, not a rigorous standard-state ΔG_bind°.
            "rmsd_included": self._group("rmsd") is not None,
        }
