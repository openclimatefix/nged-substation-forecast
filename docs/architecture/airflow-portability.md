# Porting to Airflow — What It Would Take

OCF's existing production services are orchestrated with Apache Airflow, as a fleet of
micro-services (each service in its own git repository, running in its own container). Since
Flexpectation orchestrates with Dagster instead, the question naturally arises whether this
project could — or should — converge on Airflow. This page records the assessment (July 2026) so
the reasoning stays auditable, in the same spirit as the
[rejected designs in Production Deployment](production-deployment.md#considered-but-rejected-designs).

The short answer:

- **A port is technically possible, and deliberately so.** The codebase keeps orchestrator
  lock-in shallow: all real logic lives in `packages/` with no Dagster imports, and the Dagster
  surface is a handful of thin files. Nothing about the data or ML design depends on Dagster.
- **A full port would cost day-to-day ML R&D some capabilities we lean on heavily** —
  runtime-minted partitions, per-partition observability and backfill — which Airflow does not
  currently replicate.
- **Porting only the live service is the most promising variant** — the R&D/production seam
  already drawn by our architecture is exactly where an orchestrator boundary could sit —
  though it would mean running two orchestrators side by side, an ongoing cost weighed up
  below.

Airflow claims on this page were verified against Airflow 3.3.0 in July 2026 and are dated where
they may drift.

## Why a port is mechanically straightforward

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

The Dagster-specific surface is roughly 1,600 lines across five files
(`defs/assets.py`, `defs/cv_assets.py`, `defs/production_assets.py`, `defs/jobs.py`,
`defs/schedules.py`) plus `definitions.py`, `dagster.yaml`, and the Docker Compose control
plane. Beyond code, a dozen docs pages — most of `docs/live_service/` and
[Running an experiment end-to-end](../ml_experimentation/dagster-workflow.md) — are written
around the Dagster UI and would need rewriting.

## What we actually use from Dagster, and how it maps

Airflow 3 is a much closer match than Airflow 2 was: it supports Python 3.14 (since 3.2.0),
schedules around data intervals, runs event-driven asset scheduling, and ships an ECS executor.
The mapping is uneven, though:

| Dagster feature in use | Where | Airflow 3 equivalent | Fit |
|---|---|---|---|
| 6-hourly `TimeWindowPartitionsDefinition` | `live_forecasts` | Cron schedule + `data_interval_end` | Good — Airflow's home turf |
| Daily partitions with `end_offset` | `ecmwf_ens` | Daily schedule + data interval | Good |
| Concurrency pool (`pool="ECMWF"`) | `ecmwf_ens` | Airflow pools | Direct equivalent |
| In-run retry-with-wait (`RetryRequested`) | `ecmwf_ens` NWP wait | Deferrable sensor | Airflow is **better** (see below) |
| Typed run config + Launchpad | `availability_mode`, `MetricsConfig`, `PromotedModelConfig` | DAG `params` with schema-validated trigger forms | Workable; less strongly typed |
| `DynamicPartitionsDefinition` (`{experiment}__{fold}`) | CV layer | No direct equivalent yet | The main gap (see below) |
| Per-partition backfills with run config | `live_forecasts` replay | Backfills take `--dag-run-conf`, but with open correctness bugs | Needs careful verification (see below) |
| `add_output_metadata` tables, asset catalog, lineage | every asset | Task logs / XCom; Airflow assets carry no materialisation metadata | A loss we would feel day to day |
| `EcsRunLauncher` (laptop = subprocess, cloud = Fargate, switched by `dagster.yaml`) | control plane | ECS executor (Amazon provider, Fargate launch type) | Exists; per-*task* rather than per-run granularity |
| Sensors / run-status coordination (planned, [#324](https://github.com/openclimatefix/nged-substation-forecast/issues/324)) | ingest → forecast ordering | Asset-triggered DAGs, event-driven scheduling | Parity, arguably cleaner in Airflow |

A note on Airflow's asset partitioning: Airflow 3.2 introduced partitioned assets and
[3.3.0 expanded them](https://airflow.apache.org/blog/airflow-3.3.0/) (time-window partitions,
mappers, wait policies). This strengthens the *time*-partitioned story, but partitions remain
configuration-defined — there is no equivalent of minting new partition keys at runtime — and
the feature is months old versus Dagster partitions' years of production maturity. Worth a
fresh look whenever this page is revisited.

## The three trickiest parts of a full port

### Runtime-minted experiment×fold partitions have no direct equivalent yet

The CV layer's core mechanic is that `register_experiment_job` mints
`{experiment_name}__{fold_id}` partition keys at runtime, and `trained_cv_model` /
`cv_power_forecasts` become individually addressable, retryable, observable materialisations
per key — the Dagster UI shows exactly which (experiment, fold) cells exist, succeeded, or
failed. Airflow's partitioning is fundamentally the time axis of a DAG; a port would remodel
each (experiment, fold) as a triggered DAG run carrying `conf`, and "which cells have been
materialised" would have to be tracked elsewhere — realistically MLflow becomes the system of
record for experiment progress, and the at-a-glance grid is gone.

[ML Orchestration — rejected designs](ml-orchestration.md#rejected-designs) already records
that a fold-loop-inside-one-run design was rejected precisely for losing per-fold observability
and retry. A full Airflow port would steer us back toward the design we had rejected.

### Replay backfills would need especially careful verification

Recovering missed `live_forecasts` slots means backfilling selected 6-hourly partitions with
`availability_mode="replay"` — and replay exists to prevent
[lookahead bias](production-deployment.md#resolve-nwp-availability-asymmetrically-live-vs-replay).
Airflow backfills do accept a per-backfill `--dag-run-conf` (CLI and UI), but as of July 2026
there are open bugs where that conf is silently not applied to runs whose logical dates already
exist ([#52622](https://github.com/apache/airflow/issues/52622),
[#59043](https://github.com/apache/airflow/issues/59043)). A silently dropped
`availability_mode="replay"` does not fail — it produces subtly lookahead-contaminated
forecasts. Any port would need this path explicitly verified, or redesigned (e.g. a dedicated
replay DAG) so the mode cannot be dropped.

### Materialisation metadata and lineage

Every asset attaches summary tables and row counts to its materialisation
(`context.add_output_metadata`), which the Dagster UI renders per run and plots over time; the
asset catalog gives lineage from ingest through forecasts, and `promoted_model` exists
specifically so that
[promotion is an audited materialisation](production-deployment.md#promote-the-champion-via-a-dagster-asset-not-a-script).
Airflow has no per-materialisation metadata surface; the equivalents are structured logs or a
hand-rolled reporting table. This loss is diffuse rather than blocking, but it would be felt on
every run.

## Where Airflow would be genuinely better

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
- **Pluggable retry policies** (Airflow 3.3) map cleanly onto "retry only on
  `NwpRunNotYetAvailable`, fail fast on genuine bugs".
- **Organisational familiarity and managed hosting.** Airflow is what OCF already runs in
  production, so a port buys shared operational knowledge, shared tooling, and a hiring pool.
  AWS offers managed Airflow (MWAA, supporting Airflow 3.2.1 as of May 2026) — though MWAA
  pins Python 3.12, so this project's 3.14-only packages could not be imported by MWAA workers
  directly; the standard escape is thin DAGs of `EcsRunTaskOperator` calls launching our own
  container image, which decouples the DAG environment from the project environment entirely.

## Three options

### Option A: full port — possible, but the hardest path

Everything above applies. The translation itself is not the cost — assets are thin shells with
their intended behaviour documented, so rewriting them as DAGs is days of work. The cost is the
redesign of the CV layer around the missing dynamic partitions, the verification of
orchestration behaviour (schedules ticking, backfills replaying without lookahead leakage, ECS
dispatch — inherently wall-clock-bound), the docs rewrite, and the deployment cutover.
Realistically several weeks end-to-end, landing on an experimentation workflow with fewer of
the affordances we rely on today.

### Option B: port only the live service — promising, with a real two-orchestrator cost

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
- The Airflow backfill-conf bugs above sit exactly on the ported surface (replay mode), the
  most correctness-critical operation being moved.

### Option C: stay on Dagster, keep the seam documented — our recommendation for now

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
- **Airflow's asset partitioning maturing** to cover runtime-minted partition keys with
  per-partition observability — this would remove the main blocker to Option A.
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
