# gluebind — Architecture & Design

**gluebind** computes the standard-state absolute binding free energy of a
**ternary complex** — two proteins bridged by a small "molecular glue" (a
molecular-glue degrader, a PROTAC warhead, or a natural product such as
rapamycin) — using the **geometric-route umbrella-sampling (US)** method rather
than an alchemical (FEP/TI) route.

It is an open reimplementation of the geometric-route US protocol described in the
ternary-complex binding free-energy preprint
([ChemRxiv](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006418/v1)), rebuilt
as a maintainable, testable, backend-portable Python package. It stands on
BioSimSpace (system prep), OpenMM (MD + restraints), MDAnalysis (selections /
RMSF), the Grossfield WHAM binary (PMFs), and RED (equilibration detection).

This document describes the architecture and the *why* behind the major design
decisions. For how to install and run, see the `README`; this is the design guide.

---

## 1. The scientific method

### 1.1 The geometric route

An alchemical route decouples the ligand from its environment. The **geometric
route** instead keeps all interactions physical and uses **restraints + umbrella
sampling** to build a reversible path between the *bound ternary complex* and the
*two separated binding partners*, each in its standard state. Every leg is a PMF
(potential of mean force) obtained by biasing one collective variable (CV) across
a series of overlapping windows and recombining them with WHAM.

Four kinds of leg contribute:

| Leg | CV | What it measures |
|-----|----|------------------|
| **Separation** | interface centre-of-mass distance | reversibly pulling the two partners apart |
| **RMSD confinement** | per-region RMSD to the bound structure | cost of confining flexible regions (run in **both** the bound complex and the isolated "bulk" partners) |
| **Boresch orientational** | 5 orientational DoFs (angles + dihedrals) | cost of fixing the relative orientation of the two partners in the bound state |
| **Standard-state correction** | — | analytical term releasing the restrained separated state to 1 M |

### 1.2 The thermodynamic cycle

The standard-state binding free energy is a **plain sum** of the leg
contributions, with the sign of each contribution baked into the analysis
functions (`gluebind/analysis/free_energy.py`):

```
ΔG_bind° = ΔG_RMSD + ΔG_Boresch + ΔG_sep + ΔG_corr
```

where, expanded:

- `ΔG_RMSD` = `ΔG_c^bulk − ΔG_c^bound` — the RMSD contribution function returns
  **+** for a released (bulk) stage and **−** for an applied (bound) stage, so a
  plain sum over the RMSD stages telescopes to the net confinement free energy;
- `ΔG_Boresch` = `−ΔG_o^bound` — the negative cost of applying the bound-state
  orientational restraint;
- `ΔG_sep` — integration of the separation PMF out to a cutoff `r*` (the point at
  which the partners no longer interact);
- `ΔG_corr` — the analytical standard-state release term.

Keeping the sign convention *inside* the contribution functions means the runner
never has to remember which legs add and which subtract — it sums, and the physics
is correct by construction. Uncertainties combine in quadrature across legs, and a
per-repeat uncertainty is taken over independent equilibration/sampling repeats.

### 1.3 Units convention

A deliberate, documented split (matching the reference protocol):

- **Config-facing** force constants are in `kcal·mol⁻¹·Å⁻²` (what a user naturally
  writes).
- **Analysis / WHAM** work in **nm** (spatial CVs — RMSD, separation) and **rad**
  (angular CVs — Boresch), with force constants in `kcal·mol⁻¹·nm⁻²` / `·rad⁻²`.

The single conversion point is `analysis/provider.py::wham_units`, so the force
constant written to the WHAM metafile is *always* the one the config declares —
structurally preventing the simulation/analysis unit mismatch that is easy to
introduce when the two live in separate scripts.

---

## 2. Architecture at a glance

gluebind is layered so that each concern is independently testable and the
heavyweight dependencies (BioSimSpace, OpenMM, a GPU, WHAM) are needed only by the
layers that genuinely require them.

```
        CalcSet ── a directory of systems, run + analysed together
           │
        Calculation ── one ternary complex, end to end
           │
     ┌─────┼──────────────┬───────────────┐
   Config  Runner tree   System prep     Analysis
  (pydantic) Group/Stage/  (BioSimSpace +   (WHAM + PMF +
             Window        atom map)        free-energy cycle)
           │
        Backend seam ── Local / SLURM (opaque handles)
           │
        Worker ── run_window('.') : a single US window, submission-agnostic
```

Data flows: a **config** describes the system → **prep** parameterises and
equilibrates it and writes a manifest → the **restraint context** resolves which
atoms are restrained → the **runner tree** enumerates windows and submits them
through a **backend** → each window runs a self-contained **worker** that writes a
CV timeseries → **analysis** turns per-stage PMFs into the binding free energy.

### 2.1 Package layout

```
gluebind/
  config/            the single user-authored configuration
    calculation.py     CalculationConfig (inputs, prep, sampling, restraints) + config_hash
    prep.py            PrepConfig (force fields, box, solvation, equilibration runtimes)
    sampling.py        SamplingConfig (windows, force constants, temperature, per-stage overrides)
    restraints.py      RmsdCVSpec / AlwaysOnRestraint / BoreschSpec (system-specific structure)
    slurm.py           SlurmConfig (cluster-scoped, deliberately separate)
  system/            inputs + assembly bookkeeping
    inputs.py          load proteins/glue/waters, molecule-layout, MOL/water validation
    prep.py            BioSimSpace prep: parameterise, assemble+solvate, equilibrate, extract bulk
    atom_map.py        the verified input→complex atom map (pure primitives)
  selection/         structural selection (MDAnalysis)
    interface.py       interface-residue detection
    anchors.py         automatic Boresch anchor selection + manual validation
    rmsf.py, dssp.py   RMSF + secondary-structure filtering of anchor candidates
    geometry.py        angle/dihedral/collinearity primitives
    equilibration.py   RED equilibration-window detection wrapper
  restraints/        OpenMM restraint forces + system construction
    system_builder.py  createSystem, integrator, minimise/heat, CV sampling loop
    boresch.py, rmsd.py, separation.py   the three restraint/CV force families
  simulation/        the compute entry points
    window.py          run_window(dir): one US window → CV timeseries (the worker)
    steered_md.py      steered MD to generate separation-window starting frames
    production.py      the OpenMM production run (for constant/always-on restraints)
    prep_stage.py      one equilibration stage as a backend job
  analysis/          PMFs → free energy
    wham.py            metafile + Grossfield wham invocation + PMF loading
    provider.py        WhamPmfProvider: stage → (cv, mean_pmf, [per-repeat pmfs])
    pmf.py             replicate averaging + RED truncation
    free_energy.py     the contribution integrals + the cycle + convergence checks
  backend/           execution seam
    base.py            Backend ABC + JobSpec + JobState
    local.py           LocalBackend (subprocess, GPU pinning, concurrency cap)
    slurm.py           SlurmBackend (sbatch/squeue, submission grace)
    scheduler.py       Scheduler (submit/poll throttle) + SlotPool (shared in-flight cap)
  runners/           the orchestration tree
    calculation.py     Calculation: builds the tree, runs stages, analyses ΔG°
    calc_set.py        CalcSet: a benchmark of systems + correlation stats
    group.py, stage.py, window.py, base.py   the tree nodes
  spec_builder.py    resolves the restraint context; builds a WindowSpec per window
  stage_centres.py   window centres from the equilibration distribution / SMD frames
  state.py           RunState + .gluebind-state.json (resumability)
  boresch_geometry.py  the 5 Boresch DoFs, their anchor points, angle/dihedral roles
  logutil.py         per-runner logging
```

---

## 3. Configuration

The whole calculation is described by one `CalculationConfig` (one YAML file),
which is validated up front and **echoed back fully resolved** into every run
directory for provenance.

### 3.1 Two categories, one file

The config deliberately separates:

- **System-specific structure** — the inputs and which atoms are restrained
  (`inputs`, `restraints`). This is what changes from complex to complex.
- **Protocol / sampling numbers** — force constants, window schedules,
  equilibration runtimes, temperature (`prep`, `sampling`). These are largely
  reusable across systems.

The **SLURM config is intentionally *not* part of `CalculationConfig`** — it is
cluster-scoped, not calculation-scoped, so it lives in a separate `SlurmConfig`.
The same calculation definition can therefore move between a laptop, a GPU box,
and a cluster unchanged.

### 3.2 Inputs

```yaml
inputs:
  receptor: { prm7: receptor.prm7, rst7: receptor.rst7 }   # the glue-presenting protein
  target:   { prm7: target.prm7,   rst7: target.rst7 }     # the recruited protein
  glue:     { sdf: glue.sdf, assign_to: receptor }         # optional; residue must be named MOL
  waters:   { prm7: waters.prm7, rst7: waters.rst7 }        # optional crystal waters
```

- Proteins are supplied **already parameterised** (AMBER `prm7`/`rst7`), dry —
  gluebind solvates the assembled complex itself.
- The **glue** is an SDF whose residue must be named `MOL` (validated at load,
  because the glue is resolved by residue name throughout — a different name would
  silently vanish from every selection). `assign_to` records which protein the
  glue binds more tightly; its heavy atoms fold into that protein's RMSD group.
- **Crystal waters** are an *optional, separate* input (§7.3), not embedded in the
  protein topologies.

### 3.3 Restraints

If `rmsd_cvs` is empty, gluebind falls back to an **all-Cα RMSD per protein**
(zero-config, correct for a simple single-domain complex). Multi-domain targets
(e.g. tandem bromodomains) declare explicit per-region CVs:

```yaml
restraints:
  rmsd_cvs:
    - { name: BD1, protein: target, selection: "resid 45-98 169-216" }
    - { name: BD2, protein: target, selection: "resid 350-460", include_glue: true }
  rmsd_order: [BD1, BD2]        # sequential bound-state application order
  rmsd_atoms: CA                # CA | backbone (C, N, CA) — applies to every CV
  always_on:                    # constant restraints present in every stage (cancel)
    - { protein: receptor, selection: "resid 116-158", force_constant: 100.0 }
  always_on_atoms: CA
  boresch: { anchors: auto }    # or explicit {b, c, B, C} complex atom indices
```

Key rules enforced at validation time:

- Selections are **residue-only** MDAnalysis strings; the atom filter (Cα vs
  backbone) comes from `rmsd_atoms`/`always_on_atoms`, kept separate so a selection
  reads cleanly and the two modes can differ.
- Each RMSD CV must be sampled in **both** `bound` and `bulk` states. The
  confinement free energy cancels only when a region is restrained in the bound
  state and released in the bulk state; an asymmetric `states` would leave a region
  held in a leg where it has no sampling stage, breaking the cycle — so it is
  rejected.
- `rmsd_order`, if given, must be a full permutation of the CV names (a partial
  subset would silently drop a region).
- `always_on` requires explicit `rmsd_cvs` (its bulk handling is only wired for
  the explicit-CV scheme).
- `include_glue` is validated for consistency with the glue's `assign_to` so a CV
  can't restrain the glue in the bound state but not in its bulk state.

### 3.4 Provenance & drift detection

`CalculationConfig.config_hash` is a stable SHA-256 over the canonical config,
persisted in the run state. A resume against a mutated config is caught by
comparing this hash — except `sampling.run_rmsd_us`, which is **excluded** because
it is a *scope* flag (see §9), not a physics parameter, so flipping it can *upgrade*
a run rather than abort it.

---

## 4. The runner hierarchy

```
CalcSet ─▶ Calculation ─▶ Group ─▶ Stage ─▶ Window ─▶ (replicates)
```

- **Window** — one umbrella-sampling window (one CV centre). Its replicates differ
  only by a random seed.
- **Stage** — a set of windows for one CV leg in one state (e.g. `BD1_bound`,
  `separation`, a single Boresch DoF).
- **Group** — the RMSD group, the Boresch group, the separation group. Groups
  encode the *dependency structure* between stages.
- **Calculation** — one ternary complex, end to end.
- **CalcSet** — a directory of independent calculations, run and analysed as a
  benchmark.

### 4.1 Stage dependencies

The run order encodes the method's dependencies:

1. **RMSD stages** — independent, run first (or in parallel).
2. **Boresch stages** — **sequential**: each DoF's equilibrium value is the
   minimum of its PMF, fed forward as a fixed restraint to the next DoF and to the
   separation leg. This is why a PMF provider is needed *during* the run, not only
   at analysis.
3. **Steered MD** — generates the separation-window starting frames from the
   Boresch equilibrium geometry.
4. **Separation stage** — biases the interface distance, starting each window from
   the matching SMD frame.

### 4.2 On-disk layout

A calculation base directory *is* a standard, human-readable scaffold:

```
<calc>/
  config_resolved.yaml         fully-resolved config (provenance)
  .gluebind-state.json         RunState (handles, determined values, progress)
  prep/                        solvated + equilibrated structures, RMSF reports
  <group>/<stage>/<window>/run_NN/
      window.json              the self-contained WindowSpec
      cv_timeseries.dat        [sample_index, cv_value] — what WHAM consumes
      result.json              validatable summary
```

A `CalcSet` is *nothing more* than a directory of these, plus a set-level
`benchmark.yaml` (optional experimental ΔG°) and a `results.csv`. Because every
system carries its own full config, systems can differ freely (ternary vs binary,
different targets) with no shared-base machinery.

---

## 5. The execution seam

### 5.1 Backend ABC

All compute is dispatched through a narrow `Backend` interface — `submit(JobSpec)
→ handle`, `poll(handles) → {handle: JobState}`, `cancel(handle)` — that returns
**opaque handles**. Nothing above the backend knows whether a job ran as a local
subprocess or a SLURM allocation.

- **LocalBackend** — runs each job as a subprocess, with optional GPU pinning
  (`CUDA_VISIBLE_DEVICES`) and a concurrency cap.
- **SlurmBackend** — `sbatch`/`squeue`, with a submission grace period so a
  just-submitted job isn't mistaken for failed before it appears in the queue.

This seam is the portability story: the same pipeline runs on a laptop, a GPU
workstation, or a cluster, and a future AWS Batch backend is a drop-in.

### 5.2 The submission-agnostic worker

The unit of compute is `run_window(dir)`: it reads a self-contained `window.json`,
builds the OpenMM system with that window's restraints, samples, and writes the CV
timeseries + a `result.json`. A backend runs it via a trivial launch command:

```
python -c "from gluebind.simulation.window import run_window; run_window('.')"
```

The worker knows *nothing* about S3, environment variables, or schedulers — it
just needs its working directory. That keeps the core free of any CLI and makes a
window trivially reproducible: stage the directory anywhere, call the function.

### 5.3 Scheduler & SlotPool

- **Scheduler** wraps a backend with submit/poll throttling and drives a batch of
  jobs to completion.
- **SlotPool** is a shared, thread-safe cap on total in-flight jobs, used when a
  `CalcSet` runs several systems concurrently so submission stays within a
  cluster's per-user limit while the threads keep the queue fed.

---

## 6. State & resumability

Every calculation is **idempotent and resumable**. A single `RunState`
(`.gluebind-state.json`, a pydantic model with `schema_version` + migration) is
written incrementally as the run progresses:

- **Replicates already complete on disk are skipped** — an interrupted run
  continues where it stopped.
- **Mid-run determined values are persisted** — the Boresch equilibrium values
  (each a PMF minimum) are recorded as they are computed, so a resumed run does
  not re-run already-determined DoFs.
- **`config_hash` guards against drift** — resuming against a changed config
  aborts loudly rather than mixing incompatible work.
- Writes are atomic (temp + replace) so a crash mid-write cannot corrupt the state.

`from_config(...).run()` therefore runs the whole pipeline end to end from a single
call, and re-running is safe.

---

## 7. System preparation & the atom map

### 7.1 Prep pipeline

`system/prep.py::prepare` does the cheap, CPU-bound setup on the driver
(parameterise the glue, assemble `glue + receptor + target`, solvate) and then
dispatches every MD stage — the complex equilibration and the two bulk-species
equilibrations — to the backend as one job per stage. No MD/GPU work runs on the
driver. It writes a `PreparedSystem` manifest that is the hand-off to selection
and the runner.

Equilibration is **minimise → NVT heat → NPT → long NVT production**, each stage a
resumable backend job in its own numbered subdirectory (its own log, its own
snapshot). A single equilibration run is used — the reference protocol found
triplicate equilibrations essentially identical.

### 7.2 Constant restraints & the OpenMM production run

Some systems need a **constant ("always-on") restraint** present during the
production run — e.g. a helix held in place as a surrogate for a missing structural
partner. Because it is applied identically in the restrained and released states,
its free-energy contribution cancels. When such restraints are configured, the
final production stage runs in **OpenMM** (not BioSimSpace) so the restraint is
applied on the exact same verified-mapped indices the US windows use — immune to
any re-indexing BioSimSpace applies during assembly.

The production run is NVT at the sampling temperature, started from the
NPT-equilibrated structure with no re-heating, and holds each constant restraint to
that equilibrated structure.

### 7.3 The verified input→complex atom map

**The problem.** A user writes restraint selections against their *input*
topologies. BioSimSpace, when it assembles and writes the complex, may split a
`TER`-containing protein into several molecules and/or renumber residues — so the
complex's atom/residue indexing can differ from the inputs. Resolving a selection
directly against the complex (and trusting its residue numbers) can silently apply
a restraint to the *wrong* atoms.

**The solution** (`system/atom_map.py` + `spec_builder.py::_ComplexMap`). Selections
are resolved against the **input** topologies and mapped into the complex by
anchoring on the one thing assembly preserves — the **per-molecule atom order** —
and *verifying* it atom-by-atom (atom name + mass, never residue number). Assembly
order is fixed as **glue → receptor → target**, so each component is a contiguous
block whose offset is verified against the complex. A selection is therefore either
mapped to exactly the atoms the user meant, or the mapping **fails loudly** — it
can never silently mismap. The pure verification primitives are unit-tested; the
thin MDAnalysis extraction is integration-verified on a real `TER`-split structure.

**Crystal waters** are the reason the atom map stays robust in the presence of
solvent in the inputs. They are supplied as a *separate* `inputs.waters` file and
appended as the **last** complex component, so the protein/glue blocks stay
contiguous at the front and no offset is perturbed regardless of how solvation
orders water. Embedding crystal waters inside a protein topology would instead
inflate that protein's atom/molecule counts and (because solvation commonly pools
all water together) either trip the verification or, worse, silently mis-extract
the bulk species — so the separate-file design is a deliberate correctness choice.

---

## 8. Restraint resolution & selection

`build_restraint_context` (MDAnalysis) resolves the structure-dependent restraint
data once, producing a `RestraintContext` that the pure `SpecBuilder` then turns
into a fully-resolved `WindowSpec` per window:

- **Interface detection** — Cα–Cα pairs within a cutoff define the interface
  groups whose centre-of-mass distance is the separation CV and whose centroids are
  the bonded Boresch anchors (`a`, `A`).
- **Boresch anchors** — the four non-bonded anchors (`b`, `c`, `B`, `C`) are
  chosen **automatically** from RMSF minima (secondary-structure filtered),
  selecting the set whose five Boresch DoFs have the smallest combined circular
  variance over the equilibration trajectory — i.e. the tightest, best-defined
  restraint geometry, rejecting near-collinear triples.
- **Manual anchors** — a fully supported fallback (the workflow the original study
  used): run `equilibrate()`, inspect the per-protein RMSF report
  (`prep/rmsf_<protein>.dat`, listing `resid  atom_index  rmsf`), and set the
  anchors explicitly to the chosen residues' Cα complex atom indices. A
  near-collinearity **warning** flags a risky choice without blocking it.
- **RMSD regions** — resolved for the bound (complex) topology and each isolated
  bulk topology, with earlier same-protein regions held fixed (the sequential,
  order-independent application) and any always-on restraints present in both.

The `SpecBuilder` is deliberately **pure assembly** given the context, so it is
fully unit-tested; the MDAnalysis-dependent resolution is integration-verified.

---

## 9. Analysis

### 9.1 WHAM provider

`WhamPmfProvider` turns a stage into `(cv, mean_pmf, [per-replicate pmfs])`:
per replicate it writes a WHAM metafile over the stage's windows, runs the
Grossfield `wham` binary (locally or as a submitted job), loads the PMF, and
averages across replicates. Timeseries for spatial CVs are **RED-truncated**
(equilibration removed) before WHAM; if RED finds no equilibration or is
unavailable, a fixed fraction is discarded instead (matching the reference).

### 9.2 Free energy & convergence

`free_energy.py` implements the contribution integrals and the cycle of §1.2, plus
two convergence guards drawn from the method's SI:

- **`contribution_converged`** — the contribution integrand must decay to <1% of
  its maximum at both CV extremes, so >98% of the contribution is bracketed by the
  windows; poor decay means windows should be added at the offending extreme.
- **`separation_plateau_reached`** — the separation PMF must flatten at large
  separation (a non-flat tail means the unbound state wasn't reached).

### 9.3 Uncertainty

Analysis reports both a point estimate and a standard error: a per-CV SEM (a
diagnostic for the least-converged stage) and an overall SEM obtained by combining
the contributions into a full ΔG° *per repeat* and taking the SEM over the
independent repeats.

### 9.4 CalcSet correlation statistics

For a benchmark, `CalcSet.analyse` correlates calculated vs experimental ΔG° —
but **partitions by whether the RMSD legs were included** (see §10). Full-ΔG°
systems get MAE/R²/Kendall τ; separation-only systems get a **rank correlation
only** (they are a ranking metric off the absolute scale). The two are never
pooled — pooling a ranking estimate with absolute ΔG° for an MAE would be
meaningless.

---

## 10. Separation-PMF-only mode

`sampling.run_rmsd_us = False` runs the **cheap active/inactive-discrimination**
variant: the separation, Boresch and SMD legs run as normal, but the RMSD US
*stages* are skipped, so the result is a **ranking metric** rather than an absolute
ΔG° (it omits the ΔG_c legs). Because `run_rmsd_us` is a scope flag and *not*
physics — no already-sampled window's physics depends on it — it is excluded from
`config_hash`. This lets a separation-only run be **upgraded** to the full cycle by
flipping the flag and re-running: the run resumes, the reused Boresch/separation
work is physics-identical, and only the RMSD stages are added. The common workflow
"rank N systems cheaply, then compute full ΔG° for the promising M" is therefore a
resume, not a restart.

---

## 11. Testing strategy

Two tiers (an a3fe-style split):

- **Unit (CI, no heavy deps)** — pure logic + mocked MD: config validation,
  analysis math, the runner tree, the Scheduler/SlotPool, the atom map, the
  `SpecBuilder`, selection primitives, backend parse/poll (injected clock). Runs in
  seconds with no GPU/BioSimSpace/WHAM.
- **Integration (opt-in)** — the real methods on a small vendored real system
  (the FKBP12·rapamycin·FRB ternary complex), run via the `Backend` seam on a GPU
  box or a cluster. Tests request dependency fixtures (`bss`, `red`, `wham`) and
  *skip* when a dependency (or GPU) is absent, so a partial environment runs a
  partial tier — "run what it can, skip the rest."

Scientific validation on real benchmark systems is a **user workflow** (run the
package on those inputs), deliberately *not* a test tier, so the suite carries no
external-input machinery.

The vendored system is a **machinery**-validation fixture (does the pipeline run
end to end and return a sane, resumable ΔG°), chosen partly because it has two
chains with a `TER` between them — exercising the atom map on a real split — and
because it ships crystal waters that exercise the separate `inputs.waters` path.

---

## 12. Design principles (the recurring *why*)

- **Fail loud, never silently wrong.** The atom map raises on any reordering; the
  glue must be named `MOL`; asymmetric RMSD states, non-permutation orders, and
  unit mismatches are rejected at validation time. A wrong-but-plausible number is
  the worst outcome for a free-energy method, so the design consistently trades a
  hard error for a silent risk.
- **One source of truth.** Force constants live in the config and flow unchanged to
  both the simulation and the WHAM metafile; the resolved config is echoed into
  every run directory.
- **Purity where it matters.** The load-bearing correctness logic (atom map,
  free-energy cycle, spec assembly, selection geometry) is pure and unit-tested;
  the unavoidably integration-shaped code (MDAnalysis/BioSimSpace/OpenMM) is kept
  thin and pushed to the edges.
- **Portability through a narrow seam.** Everything heavy goes through the
  `Backend` ABC and the submission-agnostic worker, so local/SLURM/future-batch are
  interchangeable and the science code never mentions a scheduler.
- **Resumable by default.** Long, expensive, partially-failing runs are the norm;
  the state file + config-hash + on-disk skipping make re-running always safe.
- **Zero-config default, explicit when needed.** An all-Cα RMSD reproduces a simple
  complex with no restraint config; multi-domain systems opt into explicit CVs.

---

## 13. Extension points

- **A new backend** — implement `submit`/`poll`/`cancel` returning opaque handles;
  nothing else changes.
- **A new CV / restraint family** — add a force builder under `restraints/` and a
  branch in `SpecBuilder`; the runner tree and analysis are CV-type generic.
- **A different equilibration-detection or WHAM tool** — both are isolated behind
  thin wrappers (`selection/equilibration.py`, `analysis/wham.py`).

---

## 14. Limitations & future work

- **No CLI** — the package is driven from Python (`from_config(...).run()`); a CLI
  is intentionally deferred to keep the core submission-agnostic.
- **AWS Batch backend** — anticipated by the seam, not yet implemented.
- **Single-glue ternary complexes** — the model is two proteins + one glue; other
  topologies are out of scope.
- The heavy MD integration tests carry placeholder runtimes and window counts that
  are tuned on first execution against a real GPU/BioSimSpace environment; the
  scientific benchmark on real systems is a separate, later campaign.
