# Grid Engine (SGE) backend — feasibility & implementation plan

**Status:** proposed, not yet implemented.
**Verdict:** **easy** — roughly **1.5–2 days** for a `GridEngineBackend` +
`GridEngineConfig`, most of which is mechanical mirroring of the existing SLURM
backend; the genuinely new logic is small and localized.

This documents adding a third execution backend — Grid Engine / SGE (as used on
clusters such as Edinburgh's "Eddie") — alongside the existing `LocalBackend` and
`SlurmBackend`. It is the outcome of a grounded feasibility analysis
(read of the SGE submission template + gluebind's `Backend` seam) with an
adversarial verification pass over the actual code.

---

## 1. Why it fits the existing seam

gluebind dispatches all compute through the narrow `Backend` ABC
(`submit(JobSpec) → handle`, `poll(handles) → {handle: JobState}`, `cancel(handle)`)
with **opaque handles**, and keeps cluster-scoped settings in a separate config
object (`SlurmConfig`). A Grid Engine backend is therefore a **near-verbatim mirror
of `SlurmBackend`**:

| SLURM | Grid Engine |
|-------|-------------|
| `sbatch` | `qsub` |
| `squeue` | `qstat` |
| `scancel` | `qdel` |
| `SlurmConfig` | `GridEngineConfig` |
| `#SBATCH` directives | `#$` directives |
| `--chdir=<dir>` | `-wd <dir>` |

Nothing above the backend changes: `Scheduler`, `JobSpec`, `JobState`, the runner
tree, and the on-disk layout are untouched. The submission-agnostic worker
(`run_window('.')`) is unchanged — it already writes `result.json` on success and
raises (non-zero exit) on failure, which is the filesystem success-marker both
backends rely on.

---

## 2. The completion-detection model (the key insight)

A finished SGE job **disappears from `qstat`** — there is no terminal `COMPLETED`
state to read the way SLURM exposes one via `sacct`/`squeue`. At first glance this
looks like a fundamental difference requiring `qacct`/exit-code sentinels.

**It is not** — because gluebind's `poll()` contract is already **presence-only**.
`SlurmBackend` never reads a `COMPLETED` status: it reports `FINISHED` purely on
**queue-absence** and lets the *caller* reconcile real success from `run_window`'s
`result.json`. SGE's disappear-on-finish is therefore **aligned** with gluebind's
model, not a break.

Consequences:

- **No `qacct` and no exit-code sentinel are needed** for the base contract.
- `poll()` must **never** try to distinguish success from failure in the backend
  (that would violate the presence-only rule and re-introduce the
  "gone-from-`qstat`-but-not-yet-in-`qacct`" race). It returns only `RUNNING` /
  `FINISHED`, exactly as `SlurmBackend` does — it never emits `PENDING` or `FAILED`.

---

## 3. The one real hazard: `Eqw`

A **launch-failed** SGE job does *not* vanish — it sits in `qstat` in the `Eqw`
(error) state indefinitely until deleted. If the "which jobs are still in the
queue?" query naively treats every `qstat` row as *present* (the intuitive port of
SLURM's presence check), an `Eqw` job is reported `RUNNING` forever — and because
`Scheduler.run()` loops `while pending or live` **with no timeout**, the driver
hangs indefinitely.

**Fix (the single piece of SGE logic you cannot copy from `SlurmBackend`):**
exclude `Eqw` (and the transient delete-running `dr`) from the in-queue set, so the
absence branch fires → `FINISHED` → the caller finds no `result.json` → `FAILED`.
Recommended hardening: when a handle classifies terminal *because* it is `Eqw`
(still physically in `qstat`), fire a best-effort `qdel` (`check=False`) so the
errored job stops occupying a real-queue slot — `SlurmBackend` never needs this
because finished jobs leave on their own.

---

## 4. Direct mappings — copy from the SLURM backend

These transfer essentially unchanged:

- **The poll classification loop** — freshly-submitted-within-grace → `RUNNING`;
  in the in-queue set → `RUNNING` and add to `_seen`; previously-`_seen` and now
  gone → `FINISHED`; resumed/unknown handle → `FINISHED`. Copy verbatim.
- **The submit-to-visible grace guard** — `qstat` has the same lag as `squeue`
  between the submit command returning and the job appearing. Keep the
  `_submitted_at` / `_seen` / `job_submission_wait` logic unchanged.
- **Constructor + thread-safety** — `detached = True`, the injected
  `clock=time.monotonic`, and `_submitted_at` / `_seen` / `threading.Lock`. Run the
  external `qstat` call **outside** the lock, then classify handles **under** it
  (one backend instance is shared by a parallel `CalcSet`, so this is load-bearing).
- **`cancel`** — `subprocess.run(["qdel", handle], check=False)`; best-effort,
  never raises, identical to `scancel`.
- **The three throttle knobs** — `queue_check_interval`, `job_submission_wait`,
  `queue_len_lim` carry over with identical semantics.

---

## 5. Genuine differences — real work, not a rename

1. **Job-id parsing.** `qsub` prints a human sentence
   `Your job 12345 ("name") has been submitted` — regex the integer
   (`re.search(r"Your job (\d+)", stdout)`) and raise on no match. (SLURM's
   `_parse_job_id` takes the *last* whitespace token, which for SGE would wrongly
   be `submitted`.)
2. **`qstat` parsing is columnar.** `squeue` offers `-o %i` for id-only output and
   `-t R,PD,…` to pre-filter server-side; `qstat` has neither. Read the job-id
   (column 1) and state (column 5) and classify by state code. Parse **defensively**
   and key on the numeric job id — the job *name* is truncated to 10 chars in
   default `qstat` output, and `qstat -u` prints **nothing** when the user has no
   jobs (so detect rows by "first token is an integer job-id", not "skip 2 header
   lines").
3. **SGE state-code classification.** Map `{qw, hqw, r, t, Rr, Rq, s, S, T}` →
   in-queue (present); `{Eqw, dr}` → **not** present (§3). This is the one place the
   completion model forces real logic.
4. **Environment is not propagated.** SLURM exports the caller's environment by
   default; SGE does not (unless `-V` is given). The submission template instead
   re-establishes the environment in-body via module-load / conda-activate lines.
   `GridEngineConfig` therefore needs a **templatable env-setup block**
   (`list[str]` of setup lines) prepended to the command in the rendered script.
   These lines are **site-specific** and must be config fields — never hard-coded.
   (Note: `JobSpec.env` is consumed only by `LocalBackend`; like `SlurmBackend`, the
   GE backend will not forward it, so all environment must come from the config
   block.)
5. **GPU request is site-named, not `--gres`.** SGE routes with `-q <gpu_queue>`
   **and** consumes with `-l <gpu_resource>=<n>`. Both the queue name and the
   resource-complex name are cluster-specific and must be config fields; do **not**
   port SLURM's `--gres=gpu:1`.
6. **Working directory is mandatory.** SGE defaults a job's cwd to `$HOME` (SLURM
   defaults to the submission dir), so the backend must pass `-wd <run_dir>`.
   Emit the working directory in **exactly one place** — either the command-line
   `qsub -wd <run_dir>` or an `#$ -wd <run_dir>` directive, **not both** (and not
   `#$ -cwd`, which points at the submission dir and silently disagrees with a
   command-line `-wd`). A missing/incorrect working dir means outputs and
   `result.json` land elsewhere and the caller's success-gate reports a false
   failure.
7. **Script rendering.** Emit `#$` directives (`-N`, the working dir, `-q`, `-l
   gpu=…`, `-l h_rt=…`, `-l h_vmem=…`) then the env-setup lines then the command —
   **no `srun` wrapper**. Memory semantics differ: `-l h_vmem` is a per-slot *hard
   virtual-memory* ceiling (job killed on breach), unlike SLURM's per-node
   `--mem`; document this on the config field.

---

## 6. Files

**Add:**

- `gluebind/config/gridengine.py` — `GridEngineConfig` (pydantic, mirroring
  `SlurmConfig`): `gpu_queue`, `gpu_resource`, `n_gpus`, `h_rt`, `h_vmem`,
  `email` (optional), `env_setup: list[str]`, `extra_directives: dict[str, str]`,
  plus the three unchanged throttle knobs. Methods: `render_script`,
  `write_submission_script`, `get_submission_cmds → ["qsub", "-wd", run_dir,
  script]`, `dump`/`load`.
- `gluebind/backend/gridengine.py` — `GridEngineBackend(Backend)`: `detached=True`;
  the copied constructor + poll loop; `submit` with the qsub-message regex;
  `_running_job_ids` parsing `qstat -u <user>` and **excluding `Eqw`/`dr`**;
  `cancel` → `qdel`; optional `Eqw` auto-`qdel`.
- Tests (a new `gluebind/tests/test_gridengine.py`, or additions to
  `test_backend.py`/`test_config.py`).

**Change:**

- `gluebind/backend/__init__.py`, `gluebind/config/__init__.py`,
  `gluebind/__init__.py` — export the new classes for parity with the SLURM ones.
- `gluebind/runners/calculation.py` — widen the `slurm_config` parameter (on
  `__init__` and `from_config`) to accept `GridEngineConfig` too (a union, or a
  small shared `Protocol` exposing `queue_len_lim` / `queue_check_interval`, which
  is what the `Scheduler` build reads). Renaming it to `submission_config` would be
  more honest, but a type-widen is the minimal change.

---

## 7. Implementation plan

1. Re-read `backend/slurm.py` and `config/slurm.py` as the templates.
2. Write `config/gridengine.py`: copy `config/slurm.py`, swap `#SBATCH` → `#$`
   directives, add the SGE fields, keep the throttle knobs verbatim. In
   `render_script` emit the `#$` directives, the `env_setup` block, then the
   command (no `srun`). `get_submission_cmds` returns the `qsub -wd …` argv.
3. Write `backend/gridengine.py`: copy `backend/slurm.py`; keep `detached`, the
   constructor, `submit`'s body, and the poll classification loop unchanged;
   replace `_parse_job_id` with the qsub-message regex; replace `_running_job_ids`
   with the defensive `qstat` parser that **excludes `Eqw`/`dr`**; replace
   `scancel` with `qdel`; optionally auto-`qdel` an `Eqw` handle.
4. Wire the `__init__.py` exports.
5. Widen `calculation.py`'s `slurm_config` type so the throttle knobs resolve.
6. Add the unit tests (§8).
7. Run the quality gates (pytest, ruff format + lint, type-check), mirroring the
   SLURM backend's coverage; confirm no real `qsub`/`qstat`/`qdel` is invoked.

Deferred (out of v1 scope): array jobs (`-t`), and any use of `qacct` / exit-code
sentinels — intentionally omitted because `poll()` is presence-only (§2); revisit
only if a future requirement wants the backend itself to distinguish success from
failure.

---

## 8. Testing

Mirror the existing `SlurmBackend` unit tests — **no real SGE cluster needed**,
exactly as the SLURM tests use no real cluster (monkeypatch `subprocess.run` and
`_running_job_ids`, inject the clock):

- **`parse_job_id`** — the regex pulls `12345` from
  `Your job 12345 ("...") has been submitted`; a paired test asserts it raises on
  unparseable stdout.
- **grace / resume** — port the SLURM grace test: inject the clock, monkeypatch
  `_running_job_ids` to return controlled id sets, and assert
  submitted-within-grace → `RUNNING`, grace-elapsed-and-gone → `FINISHED`,
  seen-then-gone → `FINISHED`, resumed-unknown-handle → `FINISHED`.
- **`Eqw`-is-not-running (the critical one)** — feed the `qstat` parser a fixture
  containing an `Eqw` job and assert its id is **not** in the in-queue set, so
  `poll()` classifies it terminal (guards against the infinite-hang bug).
- **`qstat` parse** — a captured multi-line `qstat -u` fixture (header + `qw` + `r`
  + `Eqw` + `dr` rows) returns only the `qw`/`r` ids; and an empty (no-jobs) output
  returns an empty set.
- **config render** — `render_script` contains the expected `#$` directives, the
  `env_setup` lines, and no `srun`/`#SBATCH`; `get_submission_cmds` returns the
  `qsub -wd …` argv; `dump`/`load` round-trips.

---

## Appendix — SGE command reference

**Submit.** `qsub <options> <script> <args>`. Prints
`Your job <id> ("<name>") has been submitted` (no `--parsable` equivalent — regex
the id). Key directives (as `#$` in-script or command-line flags): `-N <name>`
(also seeds the default `<name>.o<id>` / `.e<id>` output files), the working dir
(`-wd <dir>` / `-cwd`), `-q <queue>`, `-l <resource>=<n>` (GPU), `-l h_rt=HH:MM:SS`
(wall-clock), `-l h_vmem=<size>` (per-slot hard vmem), `-V` (export env), `-S
<shell>`, `-t <range>` (array).

**Poll.** `qstat -u <user>` lists the user's jobs; the **state** is column 5.
State codes: `qw` queued-waiting · `hqw` queued-held · `r` running · `t`
transferring (transient-running) · `Rr`/`Rq` restarted · `s`/`S`/`T` suspended ·
`dr` delete-running (transient) · **`Eqw` error — failed to launch, will not run,
must be treated as failure**. A successfully finished job is **absent** from
`qstat`.

**Cancel.** `qdel <id>` (queued or running); `qdel -f <id>` force-purges a stuck
job; array tasks addressed as `<id>.<taskid>`.
