# Porting to Airflow — What It Would Take

OCF's existing production services are orchestrated with Apache Airflow, as a fleet of
micro-services (each service in its own git repository, running in its own container). Since
Flexpectation orchestrates with Dagster instead, the question naturally arises whether this
project could — or should — converge on Airflow. This page records the assessment so the
reasoning stays auditable, in the same spirit as the
[rejected designs in Production Deployment](production-deployment.md#considered-but-rejected-designs).

The page answers three questions in turn:

1. **Why did we choose Dagster when we designed the system (August 2025)?** At the time the
   choice was clear-cut: Airflow could not support the experiment workflow we wanted.
2. **Would it be possible to migrate to Airflow today?** Yes — and it is a closer call than it
   was at design time, because Airflow 3.2 and 3.3 (April–July 2026) closed several of the
   gaps. The remaining gap is day-to-day experiment observability, not the data model.
3. **What would a port actually look like?** The mechanics, the trickiest parts, where Airflow
   would be genuinely better, and the options — ending with our recommendation to stay put
   for now.

Airflow claims on this page were verified against Airflow 3.3.0 (released 6 July 2026) on
19 July 2026, and are dated where they may drift.

## Why we chose Dagster (August 2025)

The ML R&D side of this project was designed around a specific workflow: mint an
`{experiment_name}__{fold_id}` partition key per cross-validation cell at runtime, and get
individually addressable, retryable, observable materialisations per key — with a UI that shows
at a glance which (experiment, fold) cells exist, succeeded, or failed, across every experiment
ever run. Dagster's `DynamicPartitionsDefinition` provides exactly this;
[ML Orchestration Design](ml-orchestration.md) documents the layer built on it.

When the system was designed in August 2025, the current Airflow release was 3.0.4 (released
8 August 2025; Airflow 3.1.0 did not arrive until 25 September 2025). At that point in time:

- **Airflow had no asset partitioning of any kind.** Partitions were the time axis of a DAG,
  full stop. [AIP-76 (Asset Partitions)](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=311626969)
  was an unshipped proposal; it did not begin landing until Airflow 3.2.0 in April 2026. The
  experiment×fold dynamic-partition design simply had no Airflow analogue.
- **The fallback pattern was compromised in the then-current release.** The natural Airflow
  shape for cross-validation — dynamic task mapping, one mapped task instance per fold — was
  undermined by a live regression in the 3.0.x series
  ([#54779](https://github.com/apache/airflow/issues/54779)): clearing a single mapped
  instance restarted *all* of its siblings, so one failed fold could not be retried alone. The
  fix shipped in 3.1.4, and the REST API could not target a single map index at all until
  October 2025 ([#43635](https://github.com/apache/airflow/issues/43635)).
- **Backfill run-config was fresh out of a bug.** Backfills silently failing to pass their
  `conf` to runs ([#51439](https://github.com/apache/airflow/issues/51439)) had been fixed
  only days earlier, in 3.0.4 itself — and a second variant (conf not applied to runs whose
  logical dates already existed,
  [#59043](https://github.com/apache/airflow/issues/59043)) survived until Airflow 3.2.2 in
  May 2026. Run-config correctness matters unusually much to us because replay-mode backfills
  guard against [lookahead bias](production-deployment.md#resolve-nwp-availability-asymmetrically-live-vs-replay).
- **Airflow could not run our Python.** Python 3.14 support arrived in Airflow 3.2.0
  (April 2026); this project is 3.14-only.

So the design-time claim is straightforward: in August 2025, Airflow could not deliver the
per-cell experiment workflow this project was built around, and its nearest substitute had a
broken per-fold retry story in the release we would have deployed. Choosing Dagster was not a
matter of taste.

## Could we migrate today? (assessed July 2026)

### What has changed since the design was made

Airflow 3.2.0 (April 2026) and 3.3.0 (July 2026) closed several of the gaps above:

- **Asset partitions shipped, including runtime-minted keys.** 3.2.0 introduced partitioned
  assets and partitioned DAG runs ([AIP-76](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=311626969));
  [3.3.0 added](https://airflow.apache.org/blog/airflow-3.3.0/) the `PartitionedAtRuntime`
  timetable and `outlet_events[asset].add_partitions(...)`, which lets a task mint arbitrary
  string partition keys decided in code — the direct conceptual analogue of Dagster's
  `DynamicPartitionsDefinition`. An `{experiment}__{fold}` key scheme is directly expressible.
- **The backfill-conf bug class is fixed.** [#59043](https://github.com/apache/airflow/issues/59043)
  (conf silently not applied to runs with existing logical dates) was fixed in Airflow 3.2.2,
  released 29 May 2026. No open issues in that class remain. One live caveat: AWS MWAA's
  newest supported version is 3.2.1, which predates the fix.
- **Retry policies became pluggable** (AIP-105, in 3.3.0) — "retry only on
  `NwpRunNotYetAvailable`, fail fast on genuine bugs" is now expressible directly.
- **Airflow runs Python 3.14** (since 3.2.0).

### What an Airflow-native experiment workflow would look like

A fair assessment has to judge Airflow's *native* redesign, not a mechanical translation of
our Dagster code. That redesign is well established in the Airflow world: one triggered DAG
run per experiment, carrying the experiment config as `conf` (with a meaningful `run_id` — the
trigger UI, CLI, and `TriggerDagRunOperator` all support custom run IDs), and one dynamically
mapped task instance per fold (`.expand()` over a fold list computed from `conf`). Astronomer's
[MLOps guidance](https://www.astronomer.io/docs/learn/airflow-mlops) documents exactly this
pattern, with MLflow as the system of record for experiment results.

This gets further than one might expect — in particular, **per-fold observability and retry
are not lost**:

- Each mapped instance has its own state, logs, and XCom, visible in the Grid view one click
  below the task row; a failed fold auto-retries alone.
- A single map index can be cleared and re-run from the UI (and via the REST API since late
  2025) without touching its siblings — on current Airflow versions.
- `map_index_template` (Airflow 2.9+) displays a rendered fold name (e.g. `fold_2024_q3`)
  instead of an integer index.

Note that this is *not* the fold-loop-inside-one-run design that
[ML Orchestration — rejected designs](ml-orchestration.md#rejected-designs) rules out: mapped
task instances keep per-fold state and retry, which is what that rejection was protecting.

### What is still missing

Three gaps remain, and they are affordances we use daily rather than blockers in principle:

- **No all-time experiment×fold status catalog.** Airflow's history is run-oriented: the Grid
  view is a date-ordered run×task matrix, with fold status collapsed one click deep and
  expandable only per run, and mapped instances are not searchable by rendered name
  ([#45100](https://github.com/apache/airflow/issues/45100)). The new AIP-76 partitions have
  the right data model but no partition-status UI — nothing shows "which keys exist /
  succeeded / failed, all time"; AIP-76 lists partition observability as future work, and the
  feature is weeks old versus Dagster partitions' years of production maturity. Dagster's
  at-a-glance partition grid has no Airflow surface today.
- **Extending a past experiment is structurally awkward.** A run's `conf` is immutable, so
  "add one more fold to last month's experiment" means triggering a second run under a second
  run ID, and nothing native re-aggregates the two. In Dagster, the new fold is one more key
  minted into the same partition set, with unified status.
- **Materialisation metadata is raw JSON.** Since Airflow 2.10 a task can attach an `extra`
  dict to an asset event, visible in the UI's asset-events list and readable downstream — a
  genuine partial equivalent of Dagster's `add_output_metadata`. But there is no rendered
  table or per-asset history plot; for human-facing metrics the standard Airflow-world answer
  is MLflow, which matches where our metrics already live but loses the per-materialisation
  summaries the Dagster UI renders on every run.

### Verdict

A migration is possible today, and the case against it is narrower than it was in August 2025.
The honest framing is no longer "Airflow can't do ML R&D" — it demonstrably can, and its data
model closed the dynamic-partition gap in July 2026. The framing is: **Airflow can run the
workload; what it lacks is the catalog** — the persistent per-cell status surface and in-place
experiment extension that Dagster gives us for free — and re-platforming to chase a weeks-old
partition feature whose observability UI is still on the roadmap would be premature.

## What would a port look like?

### Why a port is mechanically straightforward

The orchestration layer was built thin on purpose, and three properties make it portable:

- **Assets are shells.** Every asset in `src/nged_substation_forecast/defs/` loads settings,
  calls pure helpers from `packages/` (`ml_core`, `delta_store`, `nged_data`,
  `dynamical_data`), writes Delta, and reports metadata. The business logic — feature
  engineering, training, metrics, storage policy — has no Dagster imports and would not change
  at all.
- **No state crosses processes through the orchestrator.** Data crosses run boundaries via
  Delta tables; experiment state crosses via MLflow, resolved
  [by tag, never by handle](ml-orchestration.md#cross-process-run-resolution-discover-by-tag-never-pass-handles);
  the production model crosses as a directory baked into the image. Any orchestrator that can
  run a Python function on a schedule can drive this system.
- **Every write is an idempotent partition overwrite**, so retry semantics do not depend on
  orchestrator guarantees.

The Dagster-specific surface is roughly 1,900 lines across five files
(`defs/assets.py`, `defs/cv_assets.py`, `defs/production_assets.py`, `defs/jobs.py`,
`defs/schedules.py`) plus `definitions.py`, `dagster.yaml`, and the Docker Compose control
plane. Beyond code, a dozen docs pages — most of `docs/live_service/` and
[Running an experiment end-to-end](../ml_experimentation/dagster-workflow.md) — are written
around the Dagster UI and would need rewriting.

### What we actually use from Dagster, and how it maps

| Dagster feature in use | Where | Airflow 3 equivalent | Fit |
|---|---|---|---|
| 6-hourly `TimeWindowPartitionsDefinition` | `live_forecasts` | Cron schedule + `data_interval_end` | Good — Airflow's home turf |
| Daily partitions with `end_offset` | `ecmwf_ens` | Daily schedule + data interval | Good |
| Concurrency pool (`pool="ECMWF"`) | `ecmwf_ens` | Airflow pools | Direct equivalent |
| In-run retry-with-wait (`RetryRequested`) | `ecmwf_ens` NWP wait | Deferrable sensor | Airflow is **better** (see below) |
| Typed run config + Launchpad | `availability_mode`, `MetricsConfig`, `PromotedModelConfig` | DAG `params` with schema-validated trigger forms | Workable; less strongly typed |
| `DynamicPartitionsDefinition` (`{experiment}__{fold}`) | CV layer | `PartitionedAtRuntime` + `add_partitions` (3.3.0), or run-per-experiment + mapped folds | Data model now matches; the status-grid UI does not (see above) |
| Per-fold retry and observability | CV layer | Dynamic task mapping (per-instance state, logs, clear; `map_index_template`) | Good on ≥3.1.4 |
| Per-partition backfills with run config | `live_forecasts` replay | Backfills take `--dag-run-conf`; conf-dropped bug fixed in 3.2.2 | Good on ≥3.2.2; MWAA (3.2.1) still affected |
| `add_output_metadata` tables, asset catalog, lineage | every asset | Asset-event `extra` JSON (2.10+) in the events list | Partial — raw JSON, no rendered tables or history plots |
| `EcsRunLauncher` (laptop = subprocess, cloud = Fargate, switched by `dagster.yaml`) | control plane | ECS executor (Amazon provider, Fargate launch type) | Exists; per-*task* rather than per-run granularity |
| Sensors / run-status coordination (planned, [#324](https://github.com/openclimatefix/nged-substation-forecast/issues/324)) | ingest → forecast ordering | Asset-triggered DAGs, event-driven scheduling | Parity, arguably cleaner in Airflow |

### The three trickiest parts of a full port

#### The CV layer: right data model, missing status surface

A port of the CV layer would face a choice between two imperfect shapes. Adopting AIP-76
runtime partitions keeps the `{experiment}__{fold}` key scheme but stakes the daily workflow
on a weeks-old feature with no partition-status UI. Remodelling as run-per-experiment with
mapped folds is mature and keeps per-fold retry, but fragments the catalog: "which cells have
been materialised, ever" would have to be answered by MLflow (which becomes the system of
record for experiment progress) or a hand-rolled query over Airflow's task-instance table,
and extending a past experiment becomes a second, disconnected run. Either way, the
at-a-glance grid we use daily is gone until Airflow ships partition observability.

#### Replay backfills: fixed upstream, still deserving of explicit verification

Recovering missed `live_forecasts` slots means backfilling selected 6-hourly partitions with
`availability_mode="replay"`, which guards against
[lookahead bias](production-deployment.md#resolve-nwp-availability-asymmetrically-live-vs-replay).
The bug class where backfill conf was silently dropped is fixed as of Airflow 3.2.2 (see
above), but its history is instructive: a silently dropped `availability_mode="replay"` does
not fail — it produces subtly lookahead-contaminated forecasts. A port should still verify
this path end-to-end on the deployed version (MWAA's 3.2.1 predates the fix), or redesign it
(e.g. a dedicated replay DAG) so the mode cannot be dropped at all.

#### Materialisation metadata and lineage

Every asset attaches summary tables and row counts to its materialisation
(`context.add_output_metadata`), which the Dagster UI renders per run and plots over time; the
asset catalog gives lineage from ingest through forecasts, and `promoted_model` exists
specifically so that
[promotion is an audited materialisation](production-deployment.md#promote-the-champion-via-a-dagster-asset-not-a-script).
Airflow's asset-event extras cover the storage half of this (the metadata travels and is
queryable) but not the presentation half: no rendered tables, no per-asset metadata history.
The equivalents are structured logs, MLflow, or a hand-rolled reporting table. This loss is
diffuse rather than blocking, but it would be felt on every run.

### Where Airflow would be genuinely better

The assessment cuts both ways — Airflow would bring some genuine improvements:

- **The NWP publication wait.**
  [What the accepted design costs](production-deployment.md#running-the-data-ingest-runs-on-the-control-plane-vm)
  documents that `ecmwf_ens`'s `RetryRequested` loop can hold a 16 GB Fargate task idle for up
  to 4 hours on a late-publication day. An Airflow deferrable sensor would wait on the
  always-on triggerer for free and launch compute only once the run is published. (Dagster
  sensors, planned in
  [#324](https://github.com/openclimatefix/nged-substation-forecast/issues/324), close the same
  gap without a port.)
- **Per-task compute sizing.** Airflow's ECS executor launches one Fargate task per Airflow
  task, so the hourly telemetry poll would stop inheriting inference's 4 vCPU / 16 GB task
  definition automatically; multi-executor configuration also supports "light tasks local,
  heavy tasks on Fargate" natively — the hybrid that
  [would need a custom Dagster launcher](production-deployment.md#running-the-data-ingest-runs-on-the-control-plane-vm).
  The trade is ~1 minute of task provisioning latency per *task* instead of per *run*.
- **Pluggable retry policies** (AIP-105, Airflow 3.3) map cleanly onto "retry only on
  `NwpRunNotYetAvailable`, fail fast on genuine bugs".
- **Organisational familiarity and managed hosting.** Airflow is what OCF already runs in
  production, so a port buys shared operational knowledge, shared tooling, and a hiring pool.
  AWS offers managed Airflow (MWAA, supporting Airflow 3.2.1 as of May 2026) — though MWAA
  pins Python 3.12, so this project's 3.14-only packages could not be imported by MWAA workers
  directly; the standard escape is thin DAGs of `EcsRunTaskOperator` calls launching our own
  container image, which decouples the DAG environment from the project environment entirely.
  (Note that MWAA's 3.2.1 also predates the 3.2.2 backfill-conf fix discussed above.)

### Three options

#### Option A: full port — possible, but the hardest path

Everything above applies. The translation itself is not the cost — assets are thin shells with
their intended behaviour documented, so rewriting them as DAGs is days of work. The cost is
the redesign of the CV layer around the missing partition-status surface, the verification of
orchestration behaviour (schedules ticking, backfills replaying without lookahead leakage, ECS
dispatch — inherently wall-clock-bound), the docs rewrite, and the deployment cutover.
Realistically several weeks end-to-end, landing on an experimentation workflow with fewer of
the affordances we rely on today.

#### Option B: port only the live service — promising, with a real two-orchestrator cost

Move the production side to Airflow — `power_time_series_and_metadata`, `ecmwf_ens`,
`live_forecasts`, `h3_grid_weights`, and their schedules — and keep everything
experiment-shaped (registration, training, CV forecasts, metrics, promotion) on Dagster.

This works because **the seam already exists**: the
[R&D/production boundary](production-deployment.md#bake-the-model-into-the-image-at-build-time)
means the production hot path never touches MLflow, the model crosses as a baked image, and all
data crosses via Delta tables rather than orchestrator state. The two halves are coupled only
through storage — precisely the precondition for splitting orchestrators. The ported half is
also the half Airflow is genuinely good at (cron-driven, time-partitioned, no dynamic
partitions), and the result matches OCF's standard shape: the live service becomes one more
OCF-style Airflow-orchestrated micro-service, while Flexpectation R&D stays a Dagster concern.

In favour:

- Nearly all of the plausible benefit of a port (OCF alignment, MWAA option, deferrable NWP
  wait, per-task sizing) at a fraction of the loss — the CV layer, where a port would cost
  us most, never moves, and R&D loses nothing.
- A handover to NGED becomes an Airflow-only production stack; NGED never needs to learn
  Dagster.

Against (the cost of a second orchestrator, which is ongoing rather than one-off):

- **Two orchestrators to deploy, monitor, secure, upgrade, and document**, with two mental
  models and two sets of failure modes. Our own rejected-designs history repeatedly leans on
  "one execution path" as a guiding principle; this option steps away from it at the
  architecture level even while simplifying NGED's slice of it.
- **The control-plane VM question reopens.** Today one `t4g.medium` runs the Dagster control
  plane and dispatches both scheduled runs and UI-launched backtests to Fargate. After a split,
  the box runs Airflow's control plane (scheduler, DAG processor, triggerer, webserver,
  metadata DB — comparable weight to Dagster's). Either the box also keeps Dagster's control
  plane (it will not fit in 4 GB alongside MLflow and Marimo — a resize), or R&D Dagster
  retreats to laptops and loses Fargate-dispatched CV runs, which matters at V2 scale.
- **The shared ingest assets are the awkward joint.** `ecmwf_ens` and
  `power_time_series_and_metadata` feed both `live_forecasts` *and* the CV assets. With ingest
  in Airflow, the Dagster CV assets' `deps` on them dangle: they would be remodelled as
  external/observed assets, enforced lineage is lost, and "is the NWP archive fresh through my
  backtest window?" becomes a cross-tool manual check. Extending the NWP archive for a new CV
  fold becomes an *Airflow* backfill run by a researcher.
- There is an asymmetry worth noting: NGED would get a Dagster-free world, but OCF researchers
  would still live in both tools. The boundary is clean for the operator and leakier for the
  developer.
- The audit trail fragments: production run history in Airflow, promotion and experiment
  history in Dagster/MLflow.
- If the deployment target were MWAA at its current version (3.2.1), the backfill-conf bug
  would sit exactly on the ported surface (replay mode), the most correctness-critical
  operation being moved.

#### Option C: stay on Dagster, keep the seam documented — our recommendation for now

We have not yet found a technical problem in this project that a port would solve, and either
port lands somewhere between "equivalent" and "a little worse" for the people using the
workflow daily. Our suggestion is therefore to stay put for now, and to treat this page as the
documented seam: it records what would move, what it would cost, and what would justify paying
that cost — so the option stays genuinely open rather than theoretical.

## What would change this assessment

We would happily revisit this page if any of the following happens:

- **A concrete handover signal** — NGED (or a post-NIA operating agreement) indicating that
  they run Airflow or want MWAA-managed orchestration. This is the strongest trigger, and it
  points at Option B, not Option A.
- **An OCF platform decision** that all production services must run on the shared Airflow
  infrastructure. Also points at Option B.
- **AIP-76 partition observability shipping** — a per-partition status surface (which keys
  exist, succeeded, failed, over all time) for runtime-minted partitions. The data model
  arrived in Airflow 3.3.0; the UI is the remaining blocker to Option A, and it is explicitly
  on Airflow's roadmap, so this trigger is plausible rather than theoretical.
- **MWAA catching up** to Airflow ≥3.2.2 (the backfill-conf fix) and ideally 3.3+, which
  would remove the last correctness caveat on the managed-hosting path.
- **Dagster itself becoming harder to sustain** (maintenance burden, licensing or
  project-direction concerns).

## See also

- [Production Deployment — Design](production-deployment.md) — the accepted control-plane
  architecture and its rejected alternatives, several of which (EventBridge, Fargate control
  plane) rehearse the same portability arguments.
- [ML Orchestration Design](ml-orchestration.md) — the experiment layer whose Dagster
  capabilities are the main challenge for a full port.
- [Handover to NGED](../roadmap/handover.md) — the handover plan whose constraints (simplicity,
  no bespoke glue) weigh on both sides of this assessment.
