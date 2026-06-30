# Dagster ML Assets: Design Plan

This document specifies the architecture for the Dagster-based ML experimentation and
production inference pipeline.  It is written to brief a fresh implementation session;
it includes the requirements, rejected alternatives with reasons, and the final design
with enough detail to implement without ambiguity.

---

## 1. Context

The NGED Flexpectation project forecasts half-hourly power for ~32 time series
(V1 trial) scaling to ~2,500 (V2). In the V1 trial, 16 of those time series are primary substations,
2 are bulk supply points, 2 are GSPs, and the rest are customer meters, mostly single-site solar PV.
The platform has two distinct modes:

- **Research mode:** run thousands of ML experiments with full
  expanding-window cross-validation; compare results on a leaderboard.
- **Production mode:** run inference every 6 hours to generate 14-day probabilistic
  forecasts across all substations.

The current branch (`dagster-ML-assets`) added a single `cv_power_forecasts` Dagster
asset that trains and predicts inside one Python loop.  This document describes the
redesign needed to support research at scale and production simultaneously.

---

## 2. Requirements

### Functional
- Run 100s to 1000s of ML experiments, each with full expanding-window CV over the
  canonical folds (see §4.1). The canonical fold set is currently a **single MVP fold**
  (`mid_2025_to_mid_2026`); the multiple-yearly-fold protocol is the target once ECMWF ENS is
  backfilled (see §7.3.1 and docs/ml_experimentation/cross-validation-folds.md).
- All experiments evaluated on **identical canonical CV folds** — scientific fairness
  for the leaderboard is non-negotiable. (We use a single fold for now because the ECMWF numerical
  weather predictions available from Dynamical.org only go back to 2024-04-01; Dynamical.org are
  back-filling earlier years, and we move to multiple yearly folds once they do.)
- Per-fold observability: know which folds succeeded or failed for each experiment.
- Retry individual failed folds from the Dagster UI without re-running others.
- Kick off a **smoke test** (just the earliest fold) before committing to a
  full CV run, so bad configs fail fast and cheaply.
- Run production inference every 6 hours; track which runs failed and rerun them.
  Although the production inference will happen on one machine (with its own Dagster
  instance, probably in a Docker container running on AWS App Runner). ML R&D will happen on the
  personal workstations of the two ML researchers working on this project. But the production system
  and the ML R&D systems must run the exact same code.
- **Rerunnable metrics:** the researcher will add new metrics rapidly; recomputing
  metrics from the already-saved `PowerForecast` Delta table must not require
  retraining models. So the output from each inference run for each fold must be saved to disk in
  the `power_forecasts` Delta Table.
- **Arbitrary-period evaluation:** compute metrics over any slice of forecasts (e.g. the last
  month of live production) *without* those results entering the leaderboard (see §4.8).
- **Production forecast-time semantics:** a scheduled **live** run uses
  `power_fcst_init_time = now` and the freshest NWP actually present in Delta, applying **no**
  modelled publication delay — so we always exploit the latest available NWP even if the
  provider speeds up. A **replay** of a past slot (e.g. re-running a failed run the next day)
  reconstructs historical NWP availability by applying `nwp_publication_delay_hours` (see §4.9).
- Zero training-serving skew: CV (bulk mode) and production (single-run mode) call the *same*
  `engineer_features()`; a cross-mode equivalence test guarantees identical features for
  matching `(nwp_init_time, power_fcst_init_time, valid_time)` keys (see §4.9 and §5.7).

### Operational
- 1–2 researchers, but experiments can run in parallel on different machines, all
  writing to the same S3 buckets and MLflow tracking server.
- Support automated hyperparameter sweeps (programmatically created experiments).
- "One-click" experiment management from the Dagster UI.
- Also fully scriptable from the CLI (for LLM-driven experimentation).

---

## 3. Designs Considered and Rejected

### 3.1 Single `cv_power_forecasts` Asset (current code)

All folds are trained and predicted inside a single Python `for` loop within one
Dagster asset.

**Rejected because:**
- No per-fold observability or retry; the whole asset re-runs on any failure.
- No production inference path.
- No experiment isolation: a second researcher's run overwrites the asset's
  materialisation history.
- `cv_power_forecasts` asset is too long (>200 lines) mixing orchestration and ML logic.
- The `cross_validate.py` `train_end_dt` is computed as `val_start_dt`, silently
  ignoring `CvFoldConfig.train_end` (a latent bug).

### 3.2 Dagster DynamicOut Job (no assets for CV)

Use a Dagster _job_ with `DynamicOut` ops: one generator op yields a `DynamicOutput`
per fold, downstream ops run in parallel, a collect op gathers results.

**Rejected because:**
- Job run history is the only record of what ran — there is no stable, addressable
  artifact per fold in the asset catalog.
- Difficult to materialise "just fold 2023 for experiment X" from the UI; you have
  to find the right job run and retry the right step.
- Trained model artifacts (saved to disk) are not tracked as assets; harder to link
  lineage between training and prediction.

### 3.3 Dagster Assets with `StaticPartitionsDefinition`

Define CV folds as static partitions (e.g. `["2022", "2023", "2024", "2025", "2026"]`)
on the `trained_cv_model` and `cv_power_forecasts` assets.

**Rejected because:**
- Partition keys are shared across all experiments.  Two experiments running in
  parallel would appear to materialise the same partition, confusing Dagster's
  history and potentially overwriting results.
- Static partition keys must be enumerated upfront — incompatible with dynamically
  created experiments (especially automated hyperparameter sweeps).

### 3.4 One YAML Config File Per Experiment

Store each experiment's config as `conf/experiments/{experiment_name}.yaml` committed
to git.

**Rejected because:**
- Thousands of tiny YAML files would clutter the repo.
- Automated sweeps would require committing files programmatically, which breaks the
  git workflow.
- MLflow is a better canonical config store: it already stores params, metrics, and
  artifacts per run and is designed for exactly this use case.

### 3.5 MLflow Pyfunc / Model Registry for Model Saving

Use MLflow's **model registry** and **`pyfunc`** format to save and load trained models.

**Rejected because:**
- Adds significant complexity (environment snapshotting, custom loaders per model
  family, schema enforcement at the MLflow boundary).
- `BaseForecaster.save()` / `BaseForecaster.load()` are simple and sufficient.

**What we adopt instead (§4.5):** the *minimal* MLflow artifact integration —
`mlflow.log_artifacts` / `download_artifacts` wrapped around `BaseForecaster.save/load`, plus a
local-disk cache. This gives cross-machine model sharing and report-consistency without any of
the registry/`pyfunc` machinery above. (The registry-based champion-model promotion flow remains
deferred — see §6.)

### 3.6 MLflow Experiment ID (Integer) as Dagster Partition Key

Use the auto-generated MLflow integer experiment ID as the partition key prefix:
e.g. `"42__2022"`.

**Rejected because:**
- Not human-readable in the Dagster UI partition grid.
- You would need to cross-reference MLflow to understand what partition `42__2022`
  represents.  With hundreds of experiments this becomes painful.

---

## 4. Final Design

### 4.1 Key Concepts

| Concept | Definition |
|---|---|
| **Experiment** | One specific combination of model class + config (features, hyperparameters). Identified by a human-readable `experiment_name` string. Stamped onto every forecast row in a dedicated `experiment_name` column (see §4.7, §5.1) — **not** overloaded onto `power_fcst_model_name`, which keeps its meaning as the model *family* (`MODEL_NAME`). |
| **CV Fold** | One expanding-window training/validation window, identified by a short label (`fold_id`). The canonical fold set is currently a **single MVP fold** (`"mid_2025_to_mid_2026"`); the target multiple-yearly-fold protocol switches on once ECMWF ENS is backfilled (§7.3.1, docs/ml_experimentation/cross-validation-folds.md). Validation windows must be **complete** before they enter the leaderboard (a partial window cannot be fairly compared; see §4.1.2). Date ranges live in `conf/cv/default.yaml` and are immutable per leaderboard epoch. |
| **Dagster partition key** | `"{experiment_name}__{fold_id}"`, e.g. `"xgboost_with_solar__mid_2025_to_mid_2026"`. Encodes both the experiment and the fold; unique by construction (MLflow enforces unique experiment names). |
| **MLflow experiment** | One per `experiment_name`. Stores the resolved config as experiment tags. Holds all runs for that experiment. |
| **MLflow parent run** | One per experiment. Holds the resolved config params and the aggregate (mean-across-folds) metrics — the row the leaderboard sorts on. Tagged `cv_role="parent"`. |
| **MLflow fold (child) run** | One per fold, nested under the parent run. Holds per-fold training params and per-fold metrics. Tagged `cv_role="fold"`, `fold_id="mid_2025_to_mid_2026"`. |

### 4.1.1 MLflow Run Model and Cross-Process Run Resolution

You do **not** need a separate MLflow experiment per fold. MLflow's *run* is exactly the
"multiple folds belong to one experiment" concept. The structure is:

```
Experiment  "xgboost_with_solar"  (experiment_id = 42)
└── Parent run  "cv_summary"               tags: cv_role=parent   ← aggregate metrics (leaderboard row)
    └── Child run  "mid_2025_to_mid_2026"  tags: cv_role=fold, fold_id=mid_2025_to_mid_2026
                                           ↑ per-fold params + metrics (one child per canonical fold)
```

**The coordination problem:** `trained_cv_model`, `cv_power_forecasts`, and `metrics` are
separate Dagster assets running in **separate processes** (and, for retries, at separate
times). The run handle returned by `mlflow.start_run()` cannot be passed between them, and a
run cannot be left "open" across process boundaries. So the assets must **discover and resume
runs by tag**, never by passing a live handle or a run ID through Dagster metadata.

**The convention that makes this work** — three idempotent get-or-create helpers, used by all
three assets so they share one consistent run-resolution path (see §5.6):

- `get_or_create_experiment(experiment_name) -> experiment_id`
  Wraps `mlflow.get_experiment_by_name`; creates it if absent.
- `get_or_create_parent_run(experiment_id) -> run_id`
  `search_runs(experiment_id, filter="tags.cv_role = 'parent'")`; create with
  `run_name="cv_summary"`, tag `cv_role=parent` if none exists.
- `get_or_create_fold_run(experiment_id, parent_run_id, fold_id) -> run_id`
  `search_runs(experiment_id, filter="tags.cv_role = 'fold' and tags.fold_id = '{fold_id}'")`;
  create nested under the parent (`nested=True`, `run_name=fold_id`, tags
  `cv_role=fold, fold_id={fold_id}`) if none exists.

Each asset opens the run it needs by **run ID** (`mlflow.start_run(run_id=...)`), logs, and
closes it within its own process. Because lookup is by tag, re-running any single fold (or
re-running `metrics`) resumes the *same* run rather than creating a duplicate — which is the
behaviour we want for Dagster retries.

**Concurrency note:** two parallel processes could both find "no parent run" and each create
one, yielding duplicate parent runs. In practice the parent run is created once by
`register_experiment_job` (§4.3, step 4a) — before any fold can run — so the get-or-create in
the assets is only a self-healing fallback. Fold child runs are created per-partition and
never collide, because each fold is a distinct Dagster partition that Dagster will not run
concurrently with itself.

### 4.1.2 Complete Validation Windows Only

Fold eligibility requires a time series to have observations through the fold's `val_end`. A fold
whose validation window has not yet finished would silently validate on zero (or partial) data and
corrupt the "mean-across-folds" leaderboard number. Therefore:

- `conf/cv/default.yaml` contains **only folds whose validation window is complete** by the time
  the experiment runs. The current MVP fold validates `2025-07-01 → 2026-06-30`; this code is not
  expected to train until after that window has closed. When the multiple-yearly-fold protocol
  switches on, each yearly fold is added only once its validation year has finished and the NGED
  data has landed.
- `fold_id` is a plain `str` (`contracts.power_schemas.FoldId`) whose canonical values come from
  the YAML, so adding a fold is a config edit, not a schema change.
- Mid-window performance on an as-yet-incomplete window is obtained through **arbitrary-period
  evaluation** (§4.8, `evaluation_scope="ad_hoc"`), which never feeds the leaderboard.

### 4.2 Dagster Asset Graph

```
──────────────────────────────────────────────────────────────
 DATA LAYER  (existing assets, unchanged)
──────────────────────────────────────────────────────────────
 power_time_series_and_metadata   unpartitioned asset
 ecmwf_ens                        daily-partitioned asset

──────────────────────────────────────────────────────────────
 EXPERIMENT REGISTRATION + RETIREMENT  (new Dagster jobs)
──────────────────────────────────────────────────────────────
 register_experiment_job           RunConfig → creates MLflow experiment + parent run,
                                   adds partition keys to DynamicPartitionsDefinition
 retire_experiment_job             RunConfig → verifies results landed in MLflow + Delta,
                                   then deletes that experiment's dynamic partition keys
                                   (manual; see §4.3.1)

──────────────────────────────────────────────────────────────
 CV LAYER
──────────────────────────────────────────────────────────────
 eligible_time_series              fold-partitioned (by fold year), EXPERIMENT-INDEPENDENT;
                                   canonical per-fold population, frozen per leaderboard epoch
   └── deps: [power_time_series_and_metadata]
 trained_cv_model                  (key = "name__fold") trains on the eligible set, saves
                                   model (+ trained_time_series_ids in meta.json) to MLflow
   └── deps: [power_time_series_and_metadata, ecmwf_ens, eligible_time_series]
 cv_power_forecasts                (key = "name__fold") loads model, predicts on all 51
                                   members for the model's trained IDs, overwrites Delta partition
   └── deps: [trained_cv_model]

──────────────────────────────────────────────────────────────
 METRICS LAYER  (unpartitioned, rerunnable at any time)
──────────────────────────────────────────────────────────────
 metrics                           reads power_forecasts Delta, writes forecast_metrics
                                   Delta; RunConfig (population_filter + evaluation_scope);
                                   scope gates MLflow: leaderboard → golden experiments,
                                   production_monitoring → monitoring experiment, ad_hoc → none
 monitoring_sensor                 on each power_time_series_and_metadata update (~6h),
                                   runs metrics(scope=production_monitoring) over last 24h & 7d

──────────────────────────────────────────────────────────────
 PRODUCTION LAYER  (6-hourly time-window partitioned)
──────────────────────────────────────────────────────────────
 live_forecasts                    loads model from local cache (production_model_run_id);
                                   single-run inference on 51 members (availability_mode =
                                   live | replay) for the model's trained IDs; idempotent
                                   overwrite into power_forecasts Delta; logs nothing to MLflow
   └── deps: [ecmwf_ens, power_time_series_and_metadata]
```

### 4.3 Experiment Registration Flow

`register_experiment_job` is a Dagster job (not an asset) launched manually from the
Dagster UI or CLI.  Its `RunConfig` has the following fields:

```python
class RegisterExperimentConfig(Config):
    experiment_name: str         # human-readable, unique; becomes MLflow experiment name
                                 # and partition key prefix.  For sweeps, generate
                                 # programmatically, e.g. "xgboost_lr0p01_depth3".
    base_model_config: str       # path relative to PROJECT_ROOT, e.g.
                                 # "conf/model/xgboost.yaml"
    config_overrides: dict       # Hydra-style key-value overrides applied on top of
                                 # the base YAML, e.g. {"selected_features": ["lag_1h"]}
    run_mode: Literal["smoke_test", "full_cv", "register_only"]
                                 # smoke_test: adds only the earliest fold
                                 # full_cv:    adds one partition key per canonical fold in
                                 #             conf/cv/default.yaml (currently the single
                                 #             __mid_2025_to_mid_2026 fold)
                                 # register_only: adds all keys but triggers nothing
    description: str = ""        # stored as an MLflow experiment tag
```

The job performs these steps in order:

1. Resolve full config: load `base_model_config` YAML, apply `config_overrides`, build
   the concrete `BaseForecasterConfig` subclass via `hydra.utils.instantiate`.
2. Set `config.experiment_name = experiment_name` (see §4.7 on experiment_name field).
3. Call `mlflow.create_experiment(experiment_name, tags={"config": config.model_dump_json(), "description": description})`.
   If the experiment already exists, retrieve its ID (idempotent).
4. Create the experiment's **parent run** now, via `get_or_create_parent_run(experiment_id)`
   (§4.1.1), and log the resolved config params to it. Also set the **leaderboard grouping
   tags** on the parent run so the leaderboard can group/filter experiments (§4.8):
   `model_family` (= `MODEL_NAME`) and `weather_source` (from the config). The further tags the
   Milestone 1 report anticipates — `training_strategy`, `generator_capacity_estimation`,
   `switching_event_detection`, `pre_training` — are **deferred** (not implemented for several
   months); add them when those capabilities land. (`task` from the report is dropped in favour
   of grouping metrics by `time_series_type`, which is a metric dimension, not a tag — see §4.8.)
   Creating the parent run here — before any fold can run — means the per-fold assets never race
   to create it. (The helper is idempotent, so a re-registration is harmless.)
5. Based on `run_mode`, determine which fold IDs to add (read from `conf/cv/default.yaml`, never
   hard-coded — §4.1.2):
   - `"smoke_test"` → just the earliest fold, e.g. `["mid_2025_to_mid_2026"]`
   - `"full_cv"` or `"register_only"` → every canonical fold, currently
     `["mid_2025_to_mid_2026"]`
6. Add Dagster partition keys `[f"{experiment_name}__{fid}" for fid in fold_ids]` to the
   `cv_experiment_folds` `DynamicPartitionsDefinition`.
   Use `context.instance.add_dynamic_partitions("cv_experiment_folds", keys)`.
7. Log partition keys and MLflow experiment ID as Dagster output metadata.

After the job completes, the user materialises the new partitions from the Dagster asset
catalog (selecting `trained_cv_model` + `cv_power_forecasts` for the new experiment).
For a smoke test, only the earliest fold's partition exists, so only one fold runs. (With the
current single-fold config, `smoke_test` and `full_cv` add the same one partition.)

The `register_experiment_job` uses `context.instance.add_dynamic_partitions()` which requires Dagster's instance object inside an op. Make sure to use `OpExecutionContext` (not `AssetExecutionContext`) for that job's ops, since it's a job not an asset.

### 4.3.1 Experiment Retirement Flow

With 1000s of experiments × multiple folds, the `cv_experiment_folds`
`DynamicPartitionsDefinition` would grow without bound and degrade the Dagster UI and run
storage. Partition keys are therefore retired by a **manually triggered** Dagster job
(deliberate and auditable — never automatic, so we cannot accidentally drop partitions whose
results never persisted).

`retire_experiment_job` has a `RunConfig` with a single field `experiment_name: str` and:

1. Verifies the experiment's results are safely persisted **before** deleting anything:
   - the MLflow parent run exists and carries aggregate metrics, **and**
   - the `power_forecasts` Delta table contains rows for this `experiment_name`.
   If either check fails, the job raises and deletes nothing.
2. Deletes the experiment's dynamic partition keys via
   `context.instance.delete_dynamic_partition("cv_experiment_folds", key)` for each
   `f"{experiment_name}__{fid}"`.
3. Logs the deleted keys as output metadata.

Retiring partitions does **not** delete the MLflow runs or the Delta forecasts — those remain
the permanent record. Retirement only prunes Dagster's execution ledger. Like
`register_experiment_job`, its ops use `OpExecutionContext`.

### 4.4 Config Storage

- **Git** stores `conf/model/{model_class}.yaml` — the default config for a model
  family.  This file is what you edit when writing new model code.
- **MLflow** is the canonical store for the resolved, fully-specified config, in two
  complementary forms:
  - **Flattened params on the parent run.** `register_experiment_job` flattens the resolved
    config dict and logs each leaf as an individual MLflow **param** on the experiment's parent
    run (e.g. `n_estimators=500`, `max_depth=6`). This is what populates MLflow's compare/sort
    columns and lets you "grab the full resolved params for any experiment" — the primary
    requirement. Logging params (rather than stuffing the whole JSON into one experiment tag)
    also sidesteps the MLflow **tag-value length limit**, which a large config could otherwise
    exceed.
  - **A round-trippable `config` tag for deserialisation.** The experiment also carries the
    config as a JSON tag so assets can reconstruct the exact `BaseForecasterConfig` subclass.
    One genuinely long field — `selected_features` — is the only realistic candidate to bump
    the per-value length limit; if it does, log it as its own JSON-string param/tag rather than
    inline.
- **Assets** fetch config at runtime by calling
  `mlflow.get_experiment_by_name(experiment_name)` and deserialising the `"config"` tag
  back into the appropriate `BaseForecasterConfig` subclass.

This means: no per-experiment YAML files, full reproducibility (config is immutable once
registered), config is both human-queryable (params) and machine-reconstructable (tag), and it
travels with the MLflow experiment even if the YAML in git is later modified.

### 4.5 Model Artifact Storage (MLflow artifacts + local-disk cache)

Trained model binaries live in **MLflow's S3-backed artifact store**, with a **local-disk
cache** keyed by run ID. This is the *minimal* MLflow integration — `log_artifacts` /
`download_artifacts` wrapped around the existing `BaseForecaster.save/load`. It deliberately
does **not** use `mlflow.pyfunc`, the model registry, or custom flavors (those remain rejected;
§3.5). Two reasons drove the move off pure local disk:

1. **Cross-machine sharing.** Experiments run on multiple researcher workstations and production
   runs on a separate App Runner box, all "running the exact same code". A model trained on one
   machine must be loadable on another — only a shared (S3-backed) store delivers that.
2. **Consistency with the Milestone 1 report**, which states MLflow stores the trained model
   weights (report p.35).

**Two concrete `BaseForecaster` methods** (§5.2) give one consistent path. They sit on the base
class and delegate to each subclass's own disk `save`/`load`, so subclasses stay MLflow-free:

- `forecaster.save_to_mlflow(run_id)` — `self.save(tmp_dir)` (unchanged: `.ubj` per
  `time_series_id` + `meta.json`), then `mlflow.log_artifacts(tmp_dir, artifact_path="model")`
  onto the run resolved by tag.
- `ForecasterClass.load_from_mlflow(run_id, cache_base_path)` — if
  `{cache_base_path}/{run_id}/model` exists, `cls.load()` straight from that local cache;
  otherwise `mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")` into the
  cache, then load. The cache key is the **run ID**, which is immutable, so a cached model never
  goes stale and never needs invalidation.

**Production resilience (important):** the local cache is what lets the live service keep
running even if **MLflow is down**. Once the production model is cached on the App Runner box,
`live_forecasts` loads it from disk and never contacts MLflow on the hot path; only a cache miss
(first use after a model promotion, or a fresh container) needs MLflow. The cache should
therefore be on a persistent volume, or pre-seeded into the container image, so a restart during
an MLflow outage still finds the model.

`trained_cv_model` calls `save_to_mlflow`; `cv_power_forecasts` and `live_forecasts` call
`load_from_mlflow`. Each already resolves its MLflow run by tag (§4.1.1), so this adds no new
coordination — only an artifact upload/download around code that already runs.

### 4.5.1 Eligible Time Series (canonical per-fold population)

Eligibility — which `time_series_id`s take part in a fold — is **load-bearing for leaderboard
fairness** and must satisfy two invariants:

1. **Experiment-independent.** Eligibility depends *only* on data availability and the fold
   dates, **never** on the model or config. Otherwise two experiments would be scored on
   different populations and the leaderboard would no longer be apples-to-apples.
2. **Train and predict use the identical set.** The set a model is trained on must equal the
   set it predicts/scores on — even if power data changes between the train run and the predict
   run.

Two mechanisms enforce these:

- **`eligible_time_series` asset** (fold-partitioned by the canonical fold years from
  `conf/cv/default.yaml`; **not** keyed by experiment). It computes the eligible IDs for each
  fold (first obs ≥ `min_training_months` before `val_start`; last obs ≥ `val_end`) via a pure,
  data-only function and **persists** the list (small Delta/parquet). It is **frozen per
  leaderboard epoch**, so every experiment — whenever it runs — reads the *same* per-fold
  population. `trained_cv_model` and `cv_power_forecasts` both read this asset rather than each
  recomputing from live data (which could drift). This makes invariant (1) structural and dedups
  the computation across the thousands of experiments.
- **`trained_time_series_ids` in the model's `meta.json`.** `BaseForecaster.save()` records the
  exact set the model was trained on. `predict()` (or the calling asset) filters its input to
  that set, guaranteeing invariant (2) regardless of later data changes, and — crucially —
  giving **production** its population for free: `live_forecasts` forecasts *only* the
  `time_series_id`s the production model was trained on (read from the loaded model's
  `meta.json`), never a series the model has never seen.

### 4.6 `trained_cv_model` Asset

```
partitions_def: cv_experiment_folds (DynamicPartitionsDefinition)
deps: [power_time_series_and_metadata, ecmwf_ens, eligible_time_series]
```

Per materialisation:

1. Parse `context.partition_key` → `experiment_name`, `fold_id`.
2. Fetch resolved config from MLflow experiment tags; instantiate
   `BaseForecasterConfig` subclass.
3. Fetch `CvFoldConfig` for `fold_id` from `conf/cv/default.yaml`.
4. Load `power_lf` and `nwp_lf` (lazy), filtered to
   `[fold.train_start, fold.train_end]` (inclusive on both ends — see §5.1 bug fix).
5. Read the eligible `time_series_id`s for this `fold_id` from the `eligible_time_series`
   asset (§4.5.1) — do not recompute — and filter to them.
6. Call `engineer_features()`, then `forecaster.train()`. `save()` records these IDs as
   `trained_time_series_ids` in `meta.json` (§4.5.1).
7. Resolve the fold's MLflow run by tag (§4.1.1): `experiment_id =
   get_or_create_experiment(experiment_name)`, `parent_run_id =
   get_or_create_parent_run(experiment_id)`, `fold_run_id =
   get_or_create_fold_run(experiment_id, parent_run_id, fold_id)`.
8. `forecaster.save_to_mlflow(fold_run_id)` (§4.5: writes locally, then `log_artifacts` to the
   run). Open the run with `mlflow.start_run(run_id=fold_run_id)`, log training params, and close
   it within this process — do **not** pass the handle or run ID to `cv_power_forecasts`
   (which re-resolves the same run by tag).
9. Emit Dagster output metadata: n_time_series, train date range, `fold_run_id`.

### 4.7 `cv_power_forecasts` Asset

```
partitions_def: cv_experiment_folds (same DynamicPartitionsDefinition)
deps: [trained_cv_model]
```

Per materialisation:

1. Parse `context.partition_key` → `experiment_name`, `fold_id`.
2. Fetch config from MLflow (same as above).
3. Resolve `fold_run_id` by tag (§4.1.1), then
   `ForecasterClass.load_from_mlflow(fold_run_id, settings.model_cache_base_path)`
   (§4.5: local cache hit, or download artifacts from the fold run on a miss).
4. Load `power_lf` and `nwp_lf` filtered to `[fold.val_start, fold.val_end]` **and** to the
   model's `trained_time_series_ids` (from `meta.json`, §4.5.1) — so the scored population is
   exactly the trained population, even if power coverage changed since training.
5. Call `engineer_features()`, then `forecaster.predict()` (across all 51 NWP ensemble members
   — §6).
6. Overwrite `fold_id` column with the string `fold_id` (the fold's label, e.g. `"mid_2025_to_mid_2026"`).
   `predict()` stamps each row's identity columns: `power_fcst_model_name`/`power_fcst_model_version`
   carry the model **family** identity (`MODEL_NAME`/`MODEL_VERSION` class constants), while the
   dedicated `experiment_name` column carries the experiment (from
   `BaseForecasterConfig.experiment_name` — see §5.1).
7. Write to `power_forecasts` Delta, partitioned by `(experiment_name, fold_id)`, with an
   **idempotent partition overwrite** (`delta replaceWhere experiment_name=… AND fold_id=…`) —
   **not** an append. This is essential: re-running/retrying a fold (a core requirement) with an
   append would **duplicate** every row for that partition and silently double-count in metrics.
   Overwrite makes a fold re-materialisation idempotent. Partitioning by `experiment_name`
   (not `power_fcst_model_name`) also gives each experiment its own directory, so parallel
   experiments never touch each other's partitions (see §6).
8. Resume the **same** fold child run that `trained_cv_model` used, by tag (§4.1.1) — not by
   any run ID handed over from the training asset. Resolve it with `get_or_create_fold_run(
   experiment_id, parent_run_id, fold_id)` (idempotent: returns the existing fold run), open it
   with `mlflow.start_run(run_id=fold_run_id)`, log prediction metrics, and close it.
9. Emit Dagster metadata: n_rows, n_time_series, fold date range.

### 4.8 `metrics` Asset

```
partitions_def: none (unpartitioned)
deps: soft dependency on power_forecasts Delta table
```

Its `RunConfig` has two fields that together decide **what to score** and **where the results
go**:

```python
class PopulationFilter(Config):          # typed, NOT a free dict — a typo in a key must be a
    experiment_name: str | None = None   # validation error, not a silently-wrong population
    fold_id: str | None = None
    valid_time_min: datetime | None = None
    valid_time_max: datetime | None = None

class MetricsConfig(Config):
    population_filter: PopulationFilter = PopulationFilter()
    evaluation_scope: Literal["leaderboard", "production_monitoring", "ad_hoc"] = "leaderboard"
```

A **typed** `PopulationFilter` (rather than a free `dict`) is both a software-engineering and a
scientific-rigor safeguard: a mistyped key like `valid_time_mn` would silently match nothing and
score the wrong population — a typed model makes it a load-time validation error.

**Metrics are computed on the scaled `[−1, +1]` forecast**, *not* on physical MW/MVA. The
"capacity in MW" of a substation/generator is genuinely ambiguous (see
`docs/roadmap/forecast-building-blocks.md`), so a physical-units RMSE
would not be comparable across series. All experiments are therefore scored in the same scaled
space — which is also what keeps the leaderboard apples-to-apples.

- Reads from the `power_forecasts` Delta table, **filtered** by `population_filter`.
  **Incremental by default:** a leaderboard run normally scores just the experiment(s) just
  materialised (passed via `population_filter`), not the entire table — a full recompute across
  every experiment ever run is opt-in (empty filter), because that cost grows without bound.
- Joins to observed power and calls the existing `compute_metrics()` from `ml_core.metrics`.
- **Aggregation dimensions:** whenever metrics are aggregated across `time_series_id`s, compute
  them **once per `time_series_type`** (primary substation, GSP, BSP, solar PV, wind, BESS, …)
  **and** once across **all** `time_series_id`s. Each `forecast_metrics` row therefore carries a
  `time_series_type` column whose value is either a specific type or the sentinel `"all"`
  (§5.1). This replaces the report's "one leaderboard per *task*" with one table that the
  leaderboard filters by `time_series_type` — and lets a single experiment (which spans many
  types) be scored per type without separate experiments. (`time_series_type` comes from
  `TimeSeriesMetadata`, joined in at metric time.)
- Writes to the `forecast_metrics` Delta table (replaces current `cv_metrics.delta`),
  partitioned by `(experiment_name, fold_id)`. Every row carries `evaluation_scope`,
  `time_series_type`, the evaluation-window columns (`window_start`/`window_end`/`window_label`/
  `computed_at`), and an optional `mlflow_run_id` link (§5.1). The window columns describe the
  `valid_time` window each row covers **uniformly** for CV and monitoring — a CV fold row's
  window is just `[val_start, val_end]`; a monitoring row's is the trailing 24h/7d window — so
  the table is self-describing and never needs a join to MLflow to interpret a row. This keeps
  `fold_id` low-cardinality (`"live"` for all production-monitoring rows) so "find all live
  metrics" stays a trivial filter, while full window expressiveness lives in dedicated columns.
- **Scope decides where results go.** All three scopes write to `forecast_metrics` Delta
  (tagged with the scope); they differ only in MLflow logging:
  - `evaluation_scope="leaderboard"` (canonical complete-window folds only) → the **golden
    leaderboard** experiments. Logs **per-fold** metrics to each fold child run and
    **aggregate** (mean-across-folds) metrics to the experiment's **parent run**, both resumed
    by tag (§4.1.1) — `experiment_id = get_or_create_experiment(...)`, `parent_run_id =
    get_or_create_parent_run(...)`, fold runs via `get_or_create_fold_run(...)`. Aggregates are
    logged **per `time_series_type` and for `"all"`** (e.g. metric keys `rmse__solar_pv`,
    `rmse__all`), so the parent run carries every per-type leaderboard number plus the overall.
    The `"all"` aggregate is the default row the leaderboard sorts on. Lookup-by-tag means
    re-runs update existing runs in place.
  - `evaluation_scope="production_monitoring"` → a **dedicated `production_monitoring` MLflow
    experiment**, kept entirely separate from the golden leaderboard. Used for the scheduled
    live-service metrics (§4.8.1). Metrics are logged as a **time series** (one point per
    monitoring run, via MLflow's metric timestamp/step) so MLflow charts live performance over
    time — e.g. trailing-24h and trailing-7d RMSE per `time_series_type`. The monitoring runs
    are resolved by tag (one persistent run per `(window, …)`), mirroring §4.1.1.
  - `evaluation_scope="ad_hoc"` → **no MLflow at all**, Delta only. For one-off analysis (e.g.
    scoring an as-yet-incomplete validation window §4.1.2, or a researcher's manual slice).

Note on actuals: evaluating "the last 24h / 7d / month of production" scores forecasts whose
`valid_time` has already passed and now has observed power — not freshly issued forward
forecasts. This is satisfied naturally as power observations accumulate.

#### 4.8.1 Scheduled live-service monitoring

Whenever new power observations land — i.e. every 6 hours, after
`power_time_series_and_metadata` updates — a **sensor** (preferred, so it fires on the actual
data update) or a 6-hourly **schedule** materialises the `metrics` asset with
`evaluation_scope="production_monitoring"` over `fold_id="live"`, for two trailing windows:
**last 24 hours** and **last 7 days** (by `valid_time`). Each window writes a row to
`forecast_metrics` Delta with `window_label` (`"24h"`/`"7d"`), `window_start`/`window_end` set
to the trailing bounds, and `computed_at` = now (§5.1) — these are append-only, so successive
runs accumulate the sliding-window history. The same numbers are logged to the
`production_monitoring` MLflow experiment as time-series points (per window, per
`time_series_type`), and `mlflow_run_id` on the Delta rows links the two. This never touches the
golden leaderboard. `live_forecasts` itself logs **nothing** to MLflow (§4.9) — only this
monitoring step does.

#### 4.8.2 Two metric stores, one division of labour

Metrics live in **both** MLflow and the `forecast_metrics` Delta table, deliberately, because
they answer different questions:

| | **MLflow** | **`forecast_metrics` Delta** |
|---|---|---|
| Granularity | Coarse: per-`time_series_type` + `"all"` aggregates | Full cube: per `time_series_id` × time-slice × peak-events × fold × type × scope |
| Volume | tens of scalars per run | thousands of rows per experiment (grows hard at V2's ~2,500 series) |
| Consumers | leaderboard UI, sorting/grouping, the auto-researcher; live trend charts | marimo dashboard, drill-down analysis, live monitoring queries |
| Shape | key→value metric store (poor for multi-dimensional slicing) | queryable table (Polars/SQL) |

So MLflow is **the leaderboard + the live trend view** (curated aggregates); the Delta table is
**the queryable analysis cube** (granular, dashboard-feeding, home to all `ad_hoc`/monitoring
rows). The aggregates are intentionally duplicated (MLflow for the UI, Delta for querying); the
fine-grained per-series/per-slice metrics live only in Delta. `forecast_metrics` is
**internal-only** — it is not one of the tables delivered to NGED (report p.35).

Because this asset reads from Delta and writes to Delta with no model dependency, it
can be re-run at any time as new metrics are added to `compute_metrics()` without
re-running any training.

The existing `cv_metrics` Dagster asset is **deleted** and replaced by `metrics`.

### 4.8.3 Evaluating data sources with limited history

We will sometimes want to test a **new input data source whose history is shorter than the
canonical folds** — the motivating case is adding **ICON-EU NWP** (from Dynamical.org), whose
history starts later than the canonical leaderboard folds. Such a source
**cannot** enter the canonical CV folds (there is no overlapping history), so by construction this
evaluation lives entirely in `evaluation_scope="ad_hoc"` (§4.8) and **never** feeds the
leaderboard. Two patterns, answering two different questions:

**1. Controlled ablation (the principled comparison — "does the source add skill?").** Hold
*everything* constant except the source under test. Because the new source only exists from 2026,
the shared window must live within 2026:

- Pick a 2026 evaluation window bounded by the new source's availability; split it into
  train/validation (or an expanding-window split) within that window.
- **Baseline experiment:** `weather_source = "ecmwf"`, ECMWF-only features.
- **Treatment experiment:** `weather_source = "ecmwf_icon"`, ECMWF + ICON-EU features.
- Both train on the **identical rows** and are scored on the **identical rows** (same
  `time_series_id` population, same `power_fcst_init_time` grid), differing *only* in the feature
  set. Score both with `evaluation_scope="ad_hoc"` over the same `PopulationFilter`
  (`valid_time_min`/`valid_time_max` = the 2026 window).

To inherit the leaderboard's same-population guarantee (§4.5.1) for this off-leaderboard window,
materialise a **frozen ad-hoc eligibility set** for the 2026 window that both experiments read,
rather than letting each experiment pick its own population. (`trained_time_series_ids` in
`meta.json` already forces train==predict per model, but does not by itself force the *two*
experiments to share a population — the frozen set does.)

**2. Confound warning (do NOT read this as the ablation).** The tempting shortcut — take the
canonical **leaderboard champion**, run it on 2026, and compare against an ICON-EU model on 2026 —
is **statistically confounded** and must not be read as evidence about the new source. The two
models differ in **two** variables at once: the feature set *and* the **training window** (the
champion trained on four full years; the ICON-EU model is forced onto a 2026-only sliver, because
ICON-EU has no earlier history). A win or loss cannot be attributed to the source rather than to
the training data. This comparison is legitimate only as a **deployment** question ("which
forecast is better to ship *today*?"), where the confound is irrelevant because we only care which
is better now, not why.

**3. Epoch path (the eventual leaderboard-quality answer).** Once the new source has accumulated
enough history (≈1–2 complete years), promote it via a **new leaderboard epoch** (§6 fold
promotion): a fold set over source-era complete years (e.g. 2026, 2027, …) in which the new source
is canonically available, with every experiment re-scored against that fold set for apples-to-apples
comparison. The 2026 ad-hoc ablation is the **interim** signal obtained *before* enough history
exists to do this properly; it should never be presented with leaderboard rigour.

Note: this section concerns only *evaluation*. Actually **ingesting** a second NWP source (a second
downloader, NWP contract/schema changes, source-aware `WeatherFeature` parsing, and a dual-source
join in `engineer_features()`) is separate, unbuilt engineering work outside this plan's scope; see
`docs/roadmap/index.md` (v2.0).

### 4.9 `live_forecasts` Asset

```
partitions_def: TimeWindowPartitionsDefinition(cron="0 0,6,12,18 * * *", ...)
deps: [ecmwf_ens, power_time_series_and_metadata]
```

- Loads the production model via
  `ForecasterClass.load_from_mlflow(settings.production_model_run_id, settings.model_cache_base_path)`
  (§4.5): served from the local cache, so the live service keeps running even if MLflow is down
  (only a cache miss contacts MLflow). The class is inferred from a `model_class` field stored
  alongside the saved model (already part of `BaseForecaster.save()`'s `meta.json`).
- Forecasts **only** the `trained_time_series_ids` recorded in the production model's
  `meta.json` (§4.5.1) — never a series the production model has not seen.
- Runs **single-run inference** (`engineer_features(power_fcst_init_time=t0, …)`) across **all
  51 NWP ensemble members** (§6), where `t0` is the partition's scheduled time. This is
  deliberately *not* bulk mode: production forecasts one explicit `power_fcst_init_time`, not
  one-per-NWP-run.
- **Forecast-time / NWP-availability semantics** are controlled by a `RunConfig` field
  `availability_mode: Literal["live", "replay"]` (default `"live"`):
  - **`"live"`** — the scheduled production path. `t0 = now`; join the **freshest NWP run
    actually present** in Delta with `nwp_init_time <= t0`. **No** modelled publication delay is
    applied: reality already constrains the table to runs that have genuinely been published, so
    if the provider speeds up we automatically use the fresher data.
  - **`"replay"`** — re-running a *past* slot (e.g. yesterday's failed run, today). `t0 = the
    historical partition time`; join the freshest NWP run with `nwp_init_time <= t0 −
    nwp_publication_delay_hours`. The delay is applied here to reconstruct what NWP was actually
    *available* at the historical `t0` — without it we would leak NWP runs that only landed
    afterwards (lookahead / training-serving skew).
- Loads the NWP needed for that decision via `TimeWindowPartitionMapping` against the
  daily-partitioned `ecmwf_ens`.
- Calls `forecaster.predict()`, stamps `fold_id = "live"`, and writes to the shared
  `power_forecasts` Delta table with an **idempotent overwrite** of this run's
  `power_fcst_init_time` rows (`replaceWhere`), so re-running a 6-hourly partition (or a replay)
  does not duplicate rows.
- **Logs nothing to MLflow.** A 6-hourly forecast run is not an experiment; live performance is
  tracked separately by the scheduled monitoring step (§4.8.1), which logs trailing-window
  metrics to the `production_monitoring` experiment — never to the golden leaderboard.

The scheduled production sensor/schedule always materialises the current partition with the
default `availability_mode="live"`; a manual backfill of past partitions is launched with
`availability_mode="replay"`. (An automatic rule — "treat the partition as `live` iff its time
is within `nwp_publication_delay_hours` of now, else `replay`" — is a reasonable future
convenience, but the explicit flag is the source of truth.)

`settings.production_model_run_id` is a `str` field in `Settings`, set via `.env`.
Promoting a new model to production = updating this run ID and restarting Dagster; on first use
the model is downloaded from that MLflow run into the local cache, then served from disk.
(A richer MLflow-based champion-model promotion flow is deferred to Phase 2.)

### 4.10 Smoke-test fold for fast iteration

A `smoke_test` fold is a hard-coded, **non-leaderboard** CV fold for fast end-to-end sanity checks
on real data: train ~1 month, validate ~1 month, so the whole `register → train → predict → eyeball`
loop runs in well under a minute on a laptop. It exists to answer "is the model learning something
sane?", not to produce a comparable score.

It is a **first-class fold**, so it reuses the entire existing pipeline unchanged —
`eligible_time_series`, `trained_cv_model`, `cv_power_forecasts` — and is selectable per-run in the
Dagster UI as the `{experiment}__smoke_test` partition. What keeps it off the leaderboard is a
**flag, not a separate file**:

- `CvFoldConfig` gains `leaderboard: bool = True`; the smoke fold sets `leaderboard: false`.
  `conf/cv/default.yaml` becomes the fold **registry**: the epoch-pinned leaderboard folds (the
  apples-to-apples protocol, §4.1.2) plus optional non-leaderboard dev folds, distinguished by this
  flag. Keeping it in the one file is deliberate — the fold must be in the loaded fold set to reuse
  the fold-partitioned eligibility asset and the partition picker; a second YAML would need new
  import-time partition-loading plumbing and could only be swapped *globally* via a code-location
  reload, not per-run.
- `CvFoldConfig` gains optional `min_training_months: int | None = None`, overriding
  `CvConfig.min_training_months` for that fold. The smoke fold sets it to its train length (`1`), so
  eligibility requires only ~1 month of pre-`val_start` history (matching the 1-month train window)
  instead of the leaderboard's 6. `eligible_time_series` passes
  `fold.min_training_months or cv_config.min_training_months` into `eligible_time_series_ids`.
- `_fold_ids_for_run_mode` (`defs/jobs.py`) stops keying off the positional `fold_ids[0]` and
  selects by the flag instead: `smoke_test` → folds with `leaderboard == False`; `full_cv` /
  `register_only` → folds with `leaderboard == True`. This makes the run-mode names honest and
  removes the fragility where adding a fold silently shifts which fold `smoke_test` registers.
- Smoke forecasts carry `fold_id="smoke_test"`, excluded from the leaderboard exactly as `"live"`
  is: the Phase 6 `metrics` `leaderboard` scope scores only `leaderboard == True` folds. Smoke
  results are meant to be **eyeballed** via §5.1 `plot_power_forecast` (51-member fan vs ground
  truth) or scored under `ad_hoc`.

Speed comes from three independent knobs: the short windows (this fold), control-member-only
training (already in `trained_cv_model`), and a cheap experiment config (small `n_estimators` /
feature set, supplied via `config_overrides` at registration). The fold supplies the first; the
experiment supplies the last.

Concrete fold (the ECMWF ENS archive starts 2024-04-01, so early-2025 windows are safe):

```yaml
- fold_id: "smoke_test"
  leaderboard: false
  min_training_months: 1
  train_start: "2025-01-01"
  train_end:   "2025-01-31"
  val_start:   "2025-02-01"
  val_end:     "2025-02-28"
```

**Alternatives considered and rejected:** (a) a separate `conf/cv/smoke_test.yaml` selected by
`cv_config_path` — the partition set loads from a single import-time path, so swapping is global and
needs a code-location reload, not the per-run UI selection we want; (b) a run-config "type two dates
and train" ad-hoc job — it would bypass the fold-partitioned `eligible_time_series` and the
train==predict population machinery the fold reuses for free. Arbitrary-window *evaluation* already
lives in `ad_hoc` metrics (§4.8); arbitrary-window training-by-typed-dates stays out of scope.

---

## 5. Implementation Notes

### 5.1 Required Changes to Existing Code

#### `packages/ml_core/src/ml_core/cross_validate.py` (the `train_end` bug)
- **Fix the `train_end` bug while porting CV logic into the new assets.** In
  `cross_validate.py:106`, `train_end_dt` is computed as `val_start_dt`, silently ignoring
  `CvFoldConfig.train_end`. With the current YAML this happens to produce the right window
  (each `train_end` is exactly the day before `val_start`), so it is a **latent** bug: the
  moment a fold leaves a gap/embargo between `train_end` and `val_start`, training would wrongly
  run right up to `val_start`.
- The `trained_cv_model` asset (§4.6) must honour `fold.train_end` directly, using **inclusive
  end-of-day** semantics that mirror `val_end`: training window =
  `[train_start 00:00:00, train_end 23:59:59]`. No YAML changes needed.

#### `packages/contracts/src/contracts/power_schemas.py`
- **Add a dedicated `experiment_name` column to `PowerForecast`** (Categorical). This is the
  per-experiment key; do **not** overload `power_fcst_model_name` (which stays the model family
  identity from `MODEL_NAME`). Forecasts are partitioned in Delta by `(experiment_name, fold_id)`.
- `FoldId` is a plain `str` (`"live"` is the reserved production sentinel); canonical fold labels
  come from `conf/cv/default.yaml`, so adding a fold or epoch is a config edit, not a schema change
  (§4.1.2).
- `PowerForecast.fold_id` is unchanged.
- **Internal vs delivered schema (Milestone 1 report Table 1, p.28):** the columns
  `experiment_name`, `fold_id`, and `ml_flow_experiment_id` are **internal-only** — they exist
  on `PowerForecast` / the internal `power_forecasts` Delta table to support CV and the
  leaderboard, but they are **not** part of the `power_forecast` table delivered to NGED. The
  delivery step **projects them out** (the delivered table is essentially the `fold_id="live"`
  rows with internal columns dropped). When we touch Python, record this in the `PowerForecast`
  docstring; for now it lives here.

#### `packages/contracts/src/contracts/ml_schemas.py`
- **Add an `evaluation_scope` column to the `Metrics` schema** (`Literal["leaderboard",
  "production_monitoring", "ad_hoc"]`), so leaderboard, live-monitoring, and one-off metrics
  coexist in one `forecast_metrics` table and are separable (§4.8).
- **Add a `time_series_type` column to the `Metrics` schema** (the time-series category, or the
  sentinel `"all"` for the across-everything aggregate), so per-type and overall metrics coexist
  in one table (§4.8).
- **Add evaluation-window columns to the `Metrics` schema** describing the `valid_time` window
  each row covers — uniform across CV and monitoring (§4.8):
  - `window_start: datetime`, `window_end: datetime` — the window bounds. For a CV fold these
    are the fold's `val_start`/`val_end`; for monitoring they are the trailing-window bounds.
  - `window_label: str` — human label (`"2025"`, `"24h"`, `"7d"`, `"full_fold"`), cheap and
    readable.
  - `computed_at: datetime` — when the metric row was written (provenance; also what orders the
    append-only monitoring time series and distinguishes recomputations).
  These compress to almost nothing in Parquet (constant within a write batch → RLE), and keep
  the table self-describing without a cross-system join.
- **Add an `mlflow_run_id: str | None` column** as a convenience cross-link from a Delta metric
  row to its MLflow run (and back). `null` for `ad_hoc` rows (which have no MLflow run). It is a
  *link*, not the home, of the window metadata.
- `AllFeatures` is unchanged.

#### `packages/ml_core/src/ml_core/base_forecaster.py`
- **Add `experiment_name: str = ""` to `BaseForecasterConfig`.** Note the config already carries
  `ml_flow_experiment_id`, `weather_source`, and `training_strategy`; `experiment_name` joins
  them. It is set to the MLflow experiment name at registration and stored in the saved config.
- **Add `random_seed: int = 0` to `BaseForecasterConfig`** and thread it into each model's
  training (e.g. XGBoost `seed`). Reproducibility: re-training a fold must produce the *same*
  model, so retries and the leaderboard are stable.
- **Record `trained_time_series_ids` in `meta.json`** (`BaseForecaster.save()`): the exact set
  the model was trained on, read back at predict time to guarantee the train==predict population
  and to give production its population (§4.5.1).
- Each subclass's `predict()` stamps **two distinct identities** onto every `PowerForecast` row:
  model family (`power_fcst_model_name = MODEL_NAME`, `power_fcst_model_version = MODEL_VERSION`,
  both class constants) and experiment (`experiment_name = self.model_params.experiment_name`,
  plus `ml_flow_experiment_id`). Do **not** collapse experiment identity into
  `power_fcst_model_name`.
- No changes to the abstract interface (`train`, `predict`, `save`, `load`).

#### `packages/ml_core/src/ml_core/cross_validate.py`
- **Deleted wholesale in Phase 0** (§7.0.1) — it embodies the rejected single-loop CV (§3.1) and
  carries the `train_end` bug, so it is not salvaged.
- Its orchestration moves into the `trained_cv_model` / `cv_power_forecasts` assets; its reusable
  logic is **re-written fresh and correct** in `ml_core/_cv_helpers.py` as pure functions
  (`_date_to_utc_datetime`, `training_window`, `eligible_time_series_ids`) in Phase 1 — **never
  inlined into asset bodies**, so they stay unit-testable (§5.8). The intent of the deleted
  `test_cross_validate_excludes_ineligible_time_series` returns as a new unit test on
  `eligible_time_series_ids`.

#### `packages/ml_core/src/ml_core/metrics.py`
- No changes needed.  `compute_metrics()` remains a standalone Python function,
  called by the `metrics` Dagster asset.

#### `packages/contracts/src/contracts/settings.py`
- Add `production_model_run_id: str | None = None` field — the MLflow run whose model artifact
  the live service serves (downloaded once into the local cache; §4.5, §4.9).
- Add `model_cache_base_path: Path` field — root of the local-disk model cache, keyed by run ID
  (§4.5). Put it on a persistent volume so the production cache survives restarts during an
  MLflow outage.
- Add `forecast_metrics_data_path` (the `forecast_metrics.delta` table). The old
  `cv_metrics_data_path` is removed in Phase 0 (§7.0.1) along with the `cv_metrics` asset, so this
  is a fresh add, not a rename.
- `trained_ml_model_params_base_path` already exists — it can serve as the model cache root, or
  be superseded by `model_cache_base_path`.

#### `src/nged_substation_forecast/defs/assets.py`
- **Delete** `cv_power_forecasts` asset (the current monolithic one).
- **Delete** `cv_metrics` asset.
- **Add** `cv_experiment_folds = DynamicPartitionsDefinition(name="cv_experiment_folds")`.
- **Add** `eligible_time_series` asset (fold-partitioned, experiment-independent; §4.5.1).
- **Add** `trained_cv_model` asset (§4.6).
- **Add** `cv_power_forecasts` asset — same name, new implementation (§4.7).
- **Add** `metrics` asset (§4.8).
- **Add** `live_forecasts` asset (§4.9).
- **Add** `monitoring_sensor` — fires on each `power_time_series_and_metadata` update, runs
  `metrics(scope="production_monitoring")` over the trailing 24h/7d (§4.8.1).
- **Every asset is a thin orchestration shell** delegating to small, separately-unit-tested
  pure functions (§5.4, §5.8); none should mix orchestration with ML logic (the §3.1 mistake).
- Consider splitting into multiple files once this gets long:
  `defs/cv_assets.py`, `defs/production_assets.py`, `defs/metric_assets.py`.

### 5.2 New Files to Create

- `src/nged_substation_forecast/defs/jobs.py` — `register_experiment_job` /
  `RegisterExperimentConfig` (§4.3) and `retire_experiment_job` (§4.3.1). This keeps job
  definitions separate from asset definitions.
- `packages/ml_core/src/ml_core/_mlflow_runs.py` — the three idempotent run-resolution helpers
  (§5.6). Single concern: resolving MLflow runs by tag.
- The MLflow artifact persistence (`save_to_mlflow` / `load_from_mlflow` + local cache, §4.5)
  lives as **concrete methods on `BaseForecaster`** (`packages/ml_core/.../base_forecaster.py`),
  delegating to each subclass's disk `save`/`load`. No separate module: the methods belong with
  the model interface, and subclasses stay MLflow-free.
- `packages/ml_core/src/ml_core/_cv_helpers.py` — pure, data-only CV helpers, each
  independently unit-tested (§5.8): `training_window(fold)`, `eligible_time_series_ids(coverage,
  fold, min_training_months)`, `_parse_cv_partition_key`, `flatten_config(config)`, and the
  monitoring window-bounds calculation. Asset bodies stay thin by delegating here.
- A shared inference helper (e.g. `predict_and_stamp()` in `ml_core`) factoring the common
  tail used by both `cv_power_forecasts` and `live_forecasts` — `forecaster.predict()` →
  stamp identity/fold columns → return `PowerForecast`. The two assets differ only in how they
  *call* `engineer_features()` (bulk vs single-run; see §5.7), not in the predict/stamp logic.

### 5.3 What Stays the Same

The following are correct and require no changes:

- `packages/ml_core/src/ml_core/features.py` — `engineer_features()` and all feature
  classes. (Both operating modes already exist; §5.7 only *adds a test*, no code change.)
- `packages/xgboost_forecaster/` — `XGBoostForecaster` and `XGBoostConfig` (except
  that `XGBoostConfig` inherits the new `experiment_name` field from
  `BaseForecasterConfig`, and `predict()` stamps the new `experiment_name` column — §5.1).
- `packages/contracts/src/contracts/ml_schemas.py` — `AllFeatures` is unchanged; the `Metrics`
  schema gains `evaluation_scope` and `time_series_type` columns (§5.1).
- `conf/cv/default.yaml` — canonical fold definitions; immutable per leaderboard epoch.
  Currently holds the single MVP fold `mid_2025_to_mid_2026` (§4.1.2).
- `conf/model/xgboost.yaml` — default XGBoost config.
- All NWP, geo, and NGED data assets.

### 5.4 Partition Key Parsing Convention

Both CV assets will need to parse `context.partition_key`.  Use a shared helper:

```python
def _parse_cv_partition_key(partition_key: str) -> tuple[str, str]:
    """Return (experiment_name, fold_id) from a CV partition key.

    Partition key format: "{experiment_name}__{fold_id}"
    The separator is double-underscore to reduce collision risk with
    experiment names that contain single underscores.
    """
    experiment_name, fold_id = partition_key.rsplit("__", maxsplit=1)
    return experiment_name, fold_id
```

### 5.5 MLflow Minimal Integration Scope

Only the following MLflow calls are in scope for this implementation:

```python
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.create_experiment(name, tags={...})         # in register_experiment_job
mlflow.get_experiment_by_name(name)                # in the get_or_create helpers (§5.6)
mlflow.search_runs(experiment_ids=[...], filter_string="tags.cv_role = '...'")  # run lookup by tag
mlflow.start_run(experiment_id=..., run_name=..., nested=True, tags={...})      # create a run
mlflow.start_run(run_id=...)                        # resume an existing run by ID
mlflow.set_tag("cv_role", ...); mlflow.set_tag("fold_id", ...)
mlflow.log_params({...})
mlflow.log_metrics({...})
mlflow.log_artifacts(local_dir, artifact_path="model")        # save_to_mlflow (§4.5)
mlflow.artifacts.download_artifacts(run_id=..., artifact_path="model")  # load_from_mlflow (§4.5)
```

All run creation and lookup is funnelled through the §5.6 helpers; assets never call
`search_runs` / `create_experiment` directly. Artifact upload/download is funnelled through the
`BaseForecaster.save_to_mlflow` / `load_from_mlflow` methods (§4.5, §5.2).

**Not** in scope: `mlflow.pyfunc`, model registry, registry-based model promotion.

### 5.6 MLflow Run-Resolution Helpers

Create `packages/ml_core/src/ml_core/_mlflow_runs.py` (or a small module under the Dagster
app — wherever both the job and the assets can import it). It houses the three idempotent
get-or-create functions from §4.1.1 so there is exactly **one** run-resolution path shared by
`register_experiment_job`, `trained_cv_model`, `cv_power_forecasts`, and `metrics`:

```python
def get_or_create_experiment(experiment_name: str) -> str: ...
def get_or_create_parent_run(experiment_id: str) -> str: ...       # tag cv_role=parent
def get_or_create_fold_run(experiment_id: str, parent_run_id: str, fold_id: str) -> str: ...
                                                                   # tags cv_role=fold, fold_id=...
```

Each returns an ID (never an open run handle). Callers wrap the returned ID in
`with mlflow.start_run(run_id=...)` to log and close within their own process. Lookup is by
tag, making every helper safe to call from any asset and safe under Dagster retries.

**Use `MlflowClient.search_runs`, not `mlflow.search_runs`.** The top-level
`mlflow.search_runs(...)` returns a **pandas** `DataFrame` by default, which violates this
repo's Polars-only rule. The helpers use the lower-level client API, which returns a list of
`Run` objects (no pandas):

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()  # honours the tracking URI set via mlflow.set_tracking_uri(...)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.cv_role = 'fold' and tags.fold_id = 'mid_2025_to_mid_2026'",
    max_results=1,
)
fold_run_id = runs[0].info.run_id if runs else _create_fold_run(...)
```

### 5.7 Cross-Mode Feature Equivalence Test (the no-skew guarantee)

The "zero training-serving skew" requirement is met by both CV and production calling the
*same* `engineer_features()`, differing only in operating mode:

- **CV / backtest** → bulk mode (`power_fcst_init_time=None`): vectorised over the whole
  validation window, one forecast per NWP run, `power_fcst_init_time = nwp_init_time +
  nwp_publication_delay_hours` per row. No Python loop over NWP runs.
- **Production** → single-run mode (explicit `power_fcst_init_time`), per §4.9.

Because these are two code paths through one function, add a **regression test** asserting they
produce **identical** `AllFeatures` rows for the same keys. Concretely: take a fixture spanning
several **daily** NWP runs (real ECMWF ENS is issued **once per day, at 00 UTC** — not 6-hourly);
run bulk mode; then for each `nwp_init_time` run single-run **replay** mode with
`power_fcst_init_time = nwp_init_time + nwp_publication_delay_hours`; assert the rows match
exactly on `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)` and on every
feature column. This test is the enforceable form of the no-skew requirement — if a future
change diverges the two modes, it fails.

**Cadence mismatch (a known scientific subtlety).** We receive **one** NWP run per day (00 UTC)
but new NGED power data **every 6 hours**, and production issues a forecast every 6 hours. So
3 of the 4 daily production runs reuse the *same* (older) NWP run with **fresher power lags** —
the 6-hourly cadence buys fresher power, not fresher weather. Two consequences worth recording:
(a) bulk mode derives `power_fcst_init_time = nwp_init_time + delay`, i.e. **one** forecast per
NWP run (once daily), so CV currently backtests only the once-daily forecast and **under-samples
the intraday production runs**; and (b) at the 06/12/18 production runs the NWP is 6/12/18 h old,
a `power_fcst_init_time − nwp_init_time` regime the model rarely sees in training (where it is
always ≈ `delay`). For v1 this is acceptable (the daily forecast is the representative case), but
making bulk mode emit forecasts at the full 6-hourly `power_fcst_init_time` grid per NWP run is a
candidate future refinement (see §6).

### 5.8 Testing Strategy

**Principle:** every asset/job is a *thin orchestration shell* over small, single-purpose pure
functions. The pure functions carry the logic and get fast unit tests; the shells get a few
slower integration tests that exercise the real wiring (Dagster + MLflow + Delta). This is also
what keeps functions short (§5.2) and avoids the §3.1 "long asset mixing orchestration and ML"
mistake.

**Test infrastructure (shared fixtures):**
- A **file-based MLflow** tracking URI in a `tmp_path` (`mlflow.set_tracking_uri("file://…")`) —
  no server needed; the helpers already honour an injected URI.
- **Temp Delta** directories for `power_forecasts` / `forecast_metrics`.
- **Tiny synthetic** power + NWP fixtures (a handful of `time_series_id`s, a few daily NWP runs).
- An **injected clock**: `now` is passed in (not read from `datetime.now()` inside assets), so
  `live` `t0` and the monitoring 24h/7d windows are deterministic and testable.
- An integration marker (e.g. `@pytest.mark.integration`) so the pure-unit suite stays fast.

**Unit tests (pure, no I/O) — in `_cv_helpers.py` / forecasters:**
- `training_window(fold)` — **inclusive end-of-day** `[train_start 00:00:00, train_end
  23:59:59]`; explicitly covers the `train_end` bug fix and the train/val boundary (§5.1).
- `eligible_time_series_ids(coverage, fold, min_training_months)` — migrate
  `test_cross_validate_excludes_ineligible_time_series`; assert it is a function of **data only**
  (same result for any model/config).
- `_parse_cv_partition_key` — round-trips, **including experiment names containing `__`**.
- `flatten_config(config)` — nested config → flat MLflow params.
- monitoring **window-bounds** — given an injected `now`, returns the right 24h/7d
  `[window_start, window_end]`.
- `predict_and_stamp(...)` — stamps `power_fcst_model_name`/`_version`/`experiment_name`/
  `fold_id` correctly.
- **Already covered, keep:** feature-level no-lookahead (`test_nullify_leaky_lags`,
  `test_engineer_features_weather_lag_leakage_prevention`,
  `test_engineer_features_power_lag_nullification_end_to_end`,
  `test_parsed_features_forbids_power_target_leakage`, bulk-mode derivation);
  `compute_metrics` correctness; `XGBoostForecaster` save/load round-trip and "raises for unseen
  ts_id".

**Component/integration tests (file-based MLflow + temp Delta):**
- `_mlflow_runs` get-or-create **idempotency**: calling twice returns the *same* run id;
  parent/fold runs resolve by tag.
- `BaseForecaster` MLflow persistence: `save_to_mlflow` → `load_from_mlflow` round-trip; **cache
  hit vs miss** (download once, then served from local cache; survives a simulated MLflow outage
  when cached).
- `register_experiment_job`: creates the experiment + parent run + grouping tags + the right
  partition keys (from `conf/cv/default.yaml`); idempotent on re-registration.
- `retire_experiment_job`: **refuses** to delete when results are absent; deletes only when both
  MLflow + Delta results exist.

**End-to-end integration tests (Dagster in-process `materialize([...])`):**
- One-fold CV: `eligible_time_series` → `trained_cv_model` → `cv_power_forecasts` →
  `metrics(leaderboard)`. Assert forecasts land in Delta, and metrics land in Delta **and** on
  the MLflow parent/fold runs.
- **Idempotency** (guards §4.7/§4.9): materialise a fold **twice**; assert `power_forecasts` row
  count is unchanged (overwrite, not append).
- `live_forecasts` in **`live` and `replay`**: assert the two modes select different NWP runs
  (replay applies `nwp_publication_delay_hours`), that only `trained_time_series_ids` are
  forecast, and that all **51 ensemble members** are present.
- Monitoring: `metrics(production_monitoring)` writes trailing-window rows (with
  `window_label`/`window_start`/`window_end`/`computed_at`) to Delta and the
  `production_monitoring` MLflow experiment, and **never** touches a leaderboard run.

**Scientific-rigor tests (the "not cheating" guardrails):**
- **Cross-mode equivalence** (§5.7) — bulk == single-run features for matching keys (daily NWP).
- **CV-windowing no-lookahead** (new, complements the feature-level tests): no *training* row has
  `valid_time ≥ val_start` for its fold.
- **Leaderboard fairness:** two different experiments over the same fold are scored on the
  **identical** `(time_series_id, fold)` population (because `eligible_time_series` is
  experiment-independent).
- **Determinism:** training a fold twice with a fixed `random_seed` yields identical predictions
  (underpins idempotent retries and a stable leaderboard).

---

## 6. Known Limitations and Future Work

- **Champion model promotion:** currently `settings.production_model_run_id` is set
  manually.  A later (post-v1) phase will add a `champion_model` Dagster asset that queries
  MLflow for the experiment/run tagged `"Production"` and writes the run ID to a well-known
  location, removing the need for manual `.env` edits.

- **Ensemble NWP — train on control, infer on all 51 members (CV *and* production).** Training
  uses only the control member (member 0) for now — training on all 51 would need XGBoost's
  iterator API for out-of-core data, which is deferred. But **inference must run on all 51 NWP
  ensemble members** in both CV and production, because the probabilistic leaderboard metrics
  (PICP, CRPS, Spread-Skill Ratio) are meaningless on a single-member "ensemble". So
  `cv_power_forecasts` and `live_forecasts` both predict across all 51 members and write 51
  `ensemble_member` rows per `(time_series_id, valid_time)`; the `metrics` asset's probabilistic
  metrics consume that ensemble. (Training-on-control + inference-on-51 matches the Milestone 1
  report and `docs/roadmap/index.md` v0.1.)

- **Adding a fold (fold promotion):** when a new validation window completes and its NGED data
  lands (and, for the move to the yearly protocol, once ECMWF ENS is backfilled), promote it by
  adding the fold to `conf/cv/default.yaml`. No schema change is needed — `FoldId` is a plain `str`
  (§4.1.2). Promotion starts a *new leaderboard epoch* (every experiment must be re-scored against
  the new fold set for apples-to-apples comparison).

- **Concurrent Delta writes:** parallel experiments are safe because they write to different
  `experiment_name` partition directories, and re-runs are safe because each write is an
  idempotent partition overwrite (§4.7, §4.9) rather than an append — so retries can no longer
  duplicate rows. The only residual risk is two processes overwriting the *exact same*
  partition simultaneously (the same fold of the same experiment launched twice at once), which
  delta-rs may surface as a transaction-log conflict; Dagster's per-partition run concurrency
  normally prevents this, and a retry resolves it if it occurs.

- **Intraday forecast cadence in CV (under-sampling):** NWP arrives once daily (00 UTC) but
  production forecasts every 6 hours, reusing the day's NWP with fresher power lags. Bulk-mode CV
  derives one `power_fcst_init_time` per NWP run, so it backtests only the once-daily forecast
  and does not score the 06/12/18 intraday runs (§5.7). A future refinement is to have bulk mode
  emit forecasts at the full 6-hourly `power_fcst_init_time` grid per NWP run, so CV mirrors the
  production cadence exactly. For v1 the once-daily forecast is treated as representative.

- **Automated "auto-research" loop (future):** in a few months we plan to drive experimentation
  with an LLM agent (Karpathy-style auto-research): propose config → register → materialise →
  read leaderboard → iterate, with **no human in the loop and no Dagster UI in the path**. The
  design already supports each step (programmatic `register_experiment_job`, MLflow as a
  machine-readable leaderboard, `retire_experiment_job` for cleanup). The one thing to add when
  we get there is a thin Python/CLI surface for "fetch the parent-run aggregate metrics for
  experiment X" so the agent reads results without scraping the UI. See `docs/roadmap/index.md`. Not
  built now.

---

## 7. Implementation Sequence

This is a large change. Implement it in the **ordered phases** below. Each phase is sized to be a
single focused work session (and a single commit/PR on the `dagster-ML-assets` branch), and each
phase ends with a **green tree** and, wherever possible, **something the user can materialise /
run** to see progress. Do **not** start a phase before the previous one is merged and green.

### 7.0 Ground rules for the implementation agent

- **Follow `CLAUDE.md`**: Python 3.14+, **Polars only (pandas forbidden — in particular do not let
  `mlflow.search_runs` return a pandas frame; use `MlflowClient.search_runs`, §5.6)**, Patito for
  all schemas, ruff (100 cols, double quotes, Google docstrings), full type hints.
- **Definition of done for every phase:** `uv run ruff check . && uv run ruff format --check . &&
  uv run ty check && uv run pytest` all pass. Never relax an existing test to make it pass — if a
  test must change because the contract changed, change it deliberately and call it out.
- **Thin shells over pure helpers** (§5.2, §5.8): asset/job bodies orchestrate; logic lives in
  pure, separately-unit-tested functions. If an asset body grows past ~40 lines, extract.
- **Keep the tree green by coexistence:** the *old* `cv_power_forecasts` / `cv_metrics` assets and
  `cross_validate()` keep working until the phase that introduces their replacement deletes them
  (noted per phase). Where a pure helper is extracted early, have the old code *call* it so there
  is one implementation, not two.
- **The §-references are the spec.** Each phase points at the sections that define the behaviour;
  read them before coding.
- One phase = one commit; name the phase in the commit message.

### 7.0.1 Phase 0 — Clear the decks (do this first) - Completed in PR #182

When this branch started it went a slightly wrong direction (the monolithic single-loop CV,
§3.1). Delete that code **first**, in its own small PR, so every later phase adds the new design
against a clean baseline instead of diffing against soon-to-be-deleted code.

- **Delete** `packages/ml_core/src/ml_core/cross_validate.py` (the rejected single-`for`-loop
  `cross_validate()`, §3.1) and `packages/ml_core/tests/test_cross_validate.py`.
- **Delete** the `cv_power_forecasts` and `cv_metrics` assets **and** the
  `from ml_core.cross_validate import cross_validate` import in
  `src/nged_substation_forecast/defs/assets.py`. **Keep** the data-layer assets
  (`power_time_series_and_metadata`, `ecmwf_ens`, `h3_grid_weights`).
- **Fix** the stale docstring in `ml_core/metrics.py` that points at
  `ml_core.cross_validate.cross_validate` (the only remaining reference once the import is gone).
- **Remove** the now-orphaned `cv_metrics_data_path` field from `settings.py` — its only consumer
  (the deleted `cv_metrics` asset) is gone. (`forecast_metrics_data_path` is added later, in
  Phase 1, §5.1.)
- **(Optional hygiene)** delete superseded planning docs under `docs/temp/` (e.g.
  `detailed_plan_for_Dagster_ML_assets.md`); keep this `dagster_plan.md`.
- **Do NOT delete** — the new design builds on these: `conf/cv/default.yaml`,
  `conf/model/xgboost.yaml`, `hydra_schemas.CvFoldConfig`, the `ml_schemas` `Metrics` schema,
  `metrics.compute_metrics` (+ `test_metrics`), and the `power_schemas` / `base_forecaster` /
  `xgboost_forecaster` changes already on the branch.
- The eligibility and train/val-window logic in the deleted `cross_validate.py` is **re-created
  fresh and correct** in `_cv_helpers.py` in Phase 1 (the deleted version carried the `train_end`
  bug, so re-deriving is better than salvaging); the intent of
  `test_cross_validate_excludes_ineligible_time_series` returns as a new unit test there.
- **User can verify:** `uv run pytest` is green (and smaller); `uv run dg dev` loads with the
  monolithic CV assets gone and the data-layer assets intact.

### 7.1 Phase 1 — Contracts, settings, pure helpers, equivalence test (foundation) - Completed in PR #183

*No new Dagster asset yet — this is the typed foundation everything else builds on.*

- **Contracts** (§5.1): `PowerForecast.experiment_name`; `Metrics` schema gains
  `evaluation_scope`, `time_series_type`, `window_start/window_end/window_label/computed_at`,
  `mlflow_run_id`; `BaseForecasterConfig.experiment_name` + `random_seed`. Update affected
  contract tests (`test_power_forecast`, `test_ml_schemas`, `test_settings`).
- **Settings** (§5.1): add `production_model_run_id`, `model_cache_base_path`, and
  `forecast_metrics_data_path` (the old `cv_metrics_data_path` was removed in Phase 0).
- **`base_forecaster` / `xgboost_forecaster`** (§5.1): `predict()` stamps `experiment_name`;
  `save()` records `trained_time_series_ids` in `meta.json`; thread `random_seed` into training.
  Update `test_forecaster`.
- **`_cv_helpers.py`** (§5.2): write fresh (the old `cross_validate.py` was removed in Phase 0)
  `training_window(fold)` (the `train_end` **inclusive end-of-day** fix, §5.1),
  `eligible_time_series_ids(...)`, `_parse_cv_partition_key`, `flatten_config`. Unit tests per
  §5.8, including a new eligibility test that re-establishes the intent of the deleted
  `test_cross_validate_excludes_ineligible_time_series`.
- **Cross-mode equivalence test** (§5.7) — needs only the existing `engineer_features`, so land it
  now as an early rigor win.
- **User can verify:** `uv run pytest` green; the new helper + equivalence tests demonstrate the
  `train_end` fix and the no-skew guarantee.

### 7.1.5 Phase 1.5 — Unify power-lag source (Option B) - Completed in PR #183

*Small standalone change; must land before Phase 5 (backtesting), where the bug below surfaces.*

Phase 1's equivalence test had to **exclude power lags** because the two modes sourced them
differently: bulk mode self-joined the NWP-gridded frame (`_apply_power_lag(engineered_lf,
engineered_lf, ...)`) while single-run mode effectively read from the dense observed-power
series. That divergence is a **training/serving skew** on power lags, and the self-join also
hides a **fan-out duplication bug**: real ECMWF runs overlap, so a `valid_time` appears under
many `nwp_init_time`s, replicating `(time_series_id, valid_time)` rows and double-counting
lagged power.

- **Fix:** source power lags from the dense observed-power series (`power_lf`, one row per
  `(time_series_id, valid_time)`) in **both** modes. Thread `observed_power_lf` through
  `_apply_post_join_features` into `_apply_power_lag`. The lookup frame carries no
  `ensemble_member`, so `id_keys` collapses to `["time_series_id"]` — correct, since power
  observations don't vary by ensemble member, and fan-out-safe.
- **Naming:** the now-meaningful second parameter motivates clearer helper signatures —
  `_apply_power_lag(engineered_features_lf, observed_power_lf, ...)` and
  `_apply_weather_lag(engineered_features_lf, nwp_lf, ...)`, each with an `Args:` docstring
  (replaces the weak `target_lf` / `source_lf`).
- **Equivalence test:** drop the power-lag exclusion — add `power_lag_3h` to the compared
  features and extend the fixture's power series with pre-window history so the lag resolves to
  a real observed value.
- **Rolling-mean concern (investigated, resolved):** `_apply_rolling_mean_feature` was flagged as
  a possible parallel. Investigation found no fan-out (it joins back on the full primary key, 1:1)
  and that weather rolling means are cross-mode-consistent — single-run mode pads each
  `(ts, nwp_init_time, member)` group with null-weather rows that null-skipping aggregations
  ignore. This is now locked by the equivalence test, and the null-skipping invariant is
  documented on the function. A multi-stat weather extension
  (`rolling_{mean,min,max,std,median,sum}`) is captured as a ready TODO; power rolling remains
  forbidden (it would follow the Option-B pattern, anchored at `T_init`).
- **User can verify:** `uv run pytest` green; the equivalence test now asserts power lags **and**
  the weather rolling mean match across modes.

### 7.2 Phase 2 — `eligible_time_series` asset - Completed in PR #184

- Implement the fold-partitioned, experiment-independent asset (§4.5.1) over `_cv_helpers`.
- Tests: unit (from Phase 1) + an integration test materialising it on synthetic data.
- **User can verify:** materialise `eligible_time_series` in the Dagster UI and inspect the
  per-fold eligible `time_series_id` lists. *(First tangible Dagster deliverable.)*

### 7.3 Phase 3 — MLflow run resolution, model store, `register_experiment_job` - Completed in PR #185

- `_mlflow_runs.py` (§5.6), the `BaseForecaster.save_to_mlflow` / `load_from_mlflow` methods
  (§4.5), `register_experiment_job` / `RegisterExperimentConfig`, and the `cv_experiment_folds`
  `DynamicPartitionsDefinition` (§4.3).
- Tests: get-or-create **idempotency** against a file-based MLflow;
  `save_to_mlflow`/`load_from_mlflow` round-trip + cache hit/miss; register job creates experiment
  + parent run + grouping tags + the right partition keys (§5.8).
- **User can verify:** run `register_experiment_job` from the Dagster UI/CLI; see the MLflow
  experiment + parent run (config params + grouping tags) appear, and the new
  `__mid_2025_to_mid_2026` key shows up in the `cv_experiment_folds` partition set.

### 7.3.1 Phase 3.5 - Re-consider CV folds — Completed in PR #186

**Decision:** use a **single MVP fold** for now (ECMWF ENS only), because honest forecast-skill
validation needs real forecast NWP for both training and validation, and our ECMWF ENS archive
only reaches back to 2024-04-01.

- Fold `mid_2025_to_mid_2026`: train `2024-04-01 → 2025-06-30` (15 months, maximising training
  data), validate `2025-07-01 → 2026-06-30` (12 months, seasonally complete). This code won't be
  ready to train until after 2026-06-30, by which point the validation window has closed.
- **The plan is to move to the target multiple-yearly-fold protocol** (expanding training window,
  one validation year per fold — the original §4.1 design) **once Dynamical.org has backfilled
  ECMWF ENS** to the earlier years.
- `FoldId` becomes a plain `str` (`"live"` stays the production sentinel) so fold identity is fully
  config-driven; switching fold sets is then a `conf/cv/default.yaml` edit, not a schema change.
- Alternatives considered and why they were not chosen (monthly expanding CV — redundant,
  correlated folds; quarterly non-overlapping walk-forward — the sound multi-fold option, deferred;
  CERRA for pre-2024 validation — reanalysis ≠ forecast, reserved for pre-training) are recorded in
  **docs/ml_experimentation/cross-validation-folds.md**.
- Implementation: rewrite `conf/cv/default.yaml` to the single fold; relax `FoldId` to `str`;
  update the affected schema/asset docstrings and tests. No fold-*mechanics* code changes (the
  windowing in `ml_core._cv_helpers` is already generic over arbitrary train/val dates).

### 7.4 Phase 4 — `trained_cv_model` asset - Completed in PR #187

- Implement §4.6 (read config from MLflow, read eligible set, `engineer_features`, `train`,
  `save_to_mlflow`, log training params to the fold run).
- Tests: integration test materialising one fold (uses Phases 2 + 3).
- **User can verify:** register a `smoke_test` experiment, materialise `trained_cv_model` for the
  `__mid_2025_to_mid_2026` partition; see the model artifact + fold run with training params in
  MLflow.

### 7.4.1 Phase 4.1 - Document the Dagster workflow implemented so far - Completed in PR #188

- Document (probably in `docs/ml_experimentation/`) the Dagster flow we have implemented so far.
  Structure it as a sequence of simple steps, like a food recipe, for running the steps all the way
  up to `trained_cv_model`. Explain why `trained_cv_model` loads the config from MLflow, not from
  YAML.

### 7.5 Phase 5 — `cv_power_forecasts` asset - Completed in PR #189

- Implement §4.7 (`load_from_mlflow`, predict on **all 51 members** for the model's
  `trained_time_series_ids`, **idempotent partition overwrite** into `power_forecasts`, log
  prediction metrics to the fold run). **Delete the old monolithic `cv_power_forecasts`.**
- **Promote `trained_time_series_ids` to the `BaseForecaster` interface.** It is XGBoost-specific
  today; Phase 5 is the first model-agnostic consumer (`load_from_mlflow` returns a
  `BaseForecaster`, and predict must filter to the trained population), so without promotion the
  asset would have to downcast to `XGBoostForecaster` — defeating the abstraction. Define it on
  the base as **"the population this model will serve `predict`/score for"** (not "the per-series
  Boosters it holds"), so it stays honest for planned **multi-series Boosters** (one Booster for
  all solar sites, another for all primary substations, …), where a single Booster spans many
  `time_series_id`s. This is the model-agnostic encoding of the §4.5.1 train==predict population
  invariant. (Designed against this real second consumer rather than speculatively in Phase 4.)
- Tests: materialise predict for the smoke-test fold; **idempotency** (materialise twice →
  `power_forecasts` row count unchanged); assert 51 ensemble members present.
- **User can verify:** materialise `cv_power_forecasts` for `__mid_2025_to_mid_2026`, query the
  `power_forecasts` Delta table; re-materialise and confirm no duplicate rows.

### 7.5.1 Phase 5.1 - `plot_power_forecast` asset - Completed

- This design is very rough. It'll need refining before implementing!
- I want a Dagster asset that takes a `power_fcst_init_time`, and a set of `time_series_id`s, and
  writes a plot to disk. The plot will show the full 14-day forecast, starting at
  `power_fcst_init_time`. It will show all 51 ensemble members as thin grey lines. And, if
  ground-truth power data is available, the it'll also plot the ground truth on the same plot (as a
  thick blue line).
- Each `time_series_id` should be plotted on a separate panel. The x-axes should be aligned across
  `time_series_id` panels.
- This asset should accept between 1 and 4 `time_series_id`s.
- Please use Altair for the plot.
- I guess we'd want to output the plot as an HTML file, to maintain
  interactivity (e.g. I'd like the user to be able to zoom into the plot, and hover to see the
  ensemble_member ID, etc.).

### 7.5.2 Phase 5.2 — `smoke_test` fold (fast-iteration, non-leaderboard) - Completed

Implements §4.10. Independent of Phases 6–8 — it works with the already-built `trained_cv_model` +
`cv_power_forecasts`, so it can land whenever (e.g. before or after Phase 5.1).

- Add `leaderboard: bool = True` and `min_training_months: int | None = None` to `CvFoldConfig`
  (`packages/contracts/src/contracts/hydra_schemas.py`).
- Add the `smoke_test` fold to `conf/cv/default.yaml` (`leaderboard: false`,
  `min_training_months: 1`, 1-month train / 1-month val); update the file header to describe it as
  the fold registry (epoch-pinned leaderboard folds + optional non-leaderboard dev folds).
- Thread the per-fold `min_training_months` override through the `eligible_time_series` asset
  (`fold.min_training_months or cv_config.min_training_months`).
- Redefine `_fold_ids_for_run_mode` (`src/.../defs/jobs.py`) to select by the `leaderboard` flag
  rather than `fold_ids[0]`; update the `RegisterExperimentConfig.run_mode` field description.
- Tests: `test_load_cv_config_reads_canonical_yaml` (assert the leaderboard-fold set is
  `["mid_2025_to_mid_2026"]` and the smoke fold exists with `leaderboard=False`); the three
  `test_register_experiment_job` run-mode tests (now flag-based selection); a unit test that
  `eligible_time_series_ids` honours the per-fold override.
- When Phase 6 lands, the `metrics` `leaderboard` scope filters to `leaderboard == True` folds.
- **User can verify:** register a cheap experiment with `run_mode="smoke_test"`; materialise
  `eligible_time_series` / `trained_cv_model` / `cv_power_forecasts` for `{experiment}__smoke_test`;
  the full loop finishes in under a minute and the smoke forecasts never appear in leaderboard
  metrics.

### 7.6 Phase 6 — `metrics` asset (leaderboard + ad_hoc) → full CV loop works

- Implement §4.8 for the `leaderboard` and `ad_hoc` scopes (typed `PopulationFilter`, join
  actuals, `compute_metrics`, per-`time_series_type` + `"all"` aggregation, write
  `forecast_metrics` with window columns, log leaderboard aggregates to parent/fold runs).
  **Delete the old `cv_metrics` asset. (if it still exists?)**
- **Full-stack integration test (§5.8, §7.10):** add the **cross-process** full-stack test here —
  a real `mlflow server` subprocess + a real `DagsterInstance` running
  `register → trained_cv_model → cv_power_forecasts → metrics -> plot_power_forecast`, so the **by-tag cross-process run
  resolution (§4.1.1)** and the **artifact round-trip (§4.5)** are exercised for real.
- **User can verify:** the **entire CV leaderboard loop works for one experiment** — after
  materialising the smoke-test fold + `metrics`, the MLflow parent run shows the aggregate
  leaderboard metrics and `forecast_metrics` holds the per-type/overall rows. *(Milestone.)*

### 7.6.5 Phase 6.5 — MVP `effective_capacity` asset + NMAE normalisation upgrade

*The data contract (`EffectiveCapacity`) and the NMAE P99 change in `compute_metrics` are already
landed. This phase adds the Dagster asset that computes and persists the capacity estimate, then
wires it into the `metrics` asset so the NMAE denominator uses full-history P99 rather than the
validation-window P99 computed inline today.*

**Background.** NMAE is currently normalised by P99 of `|power_actual|` computed within the
joined forecast/actuals window. This is validation-window-dependent: an unusually calm year for a
wind farm gives a low P99, inflating its NMAE. Pre-computing P99 over the full history fixes this.
It also establishes the `effective_capacity` Delta table and schema that the v0.6 / v0.7
differentiable-physics upgrade will slot into without schema or interface changes.

**Already done:**
- `EffectiveCapacity` data contract added to `contracts.power_schemas` — fields
  `(time_series_id, time, effective_capacity_mw: Float32)`.
- `compute_metrics` NMAE denominator changed from `mean(|power|)` to `quantile(0.99)(|power|)`.

**To implement:**

- **`effective_capacity` Dagster asset** (unpartitioned; deps: `power_time_series_and_metadata`):
  read the `power_time_series` Delta table as a `LazyFrame`, compute
  `P99(abs(power))` per `time_series_id` over the full available history, set `time` to the
  latest observed `time` per series. Write one row per `time_series_id` to
  `effective_capacity.delta` (path from `Settings.effective_capacity_data_path`, a new field).
  Validate output against `EffectiveCapacity`.

- **`Settings.effective_capacity_data_path`**: new `Path` field pointing at the MVP Delta table.

- **Wire into `compute_metrics`**: add an optional third positional parameter
  `capacity: pt.DataFrame[EffectiveCapacity] | None = None`. When provided, join on
  `time_series_id` and use `effective_capacity_mw` as the NMAE denominator (replacing the inline
  `quantile(0.99)` computation, which stays as the fallback when `capacity` is `None`).
  The `metrics` Dagster asset reads `effective_capacity.delta` and passes it in.

- **Tests:** unit test that `compute_metrics` uses the joined `effective_capacity_mw` when a
  capacity frame is provided (and falls back to inline P99 when not). Integration test materialising
  `effective_capacity` on synthetic power data and confirming one row per series with a physically
  plausible P99 value.

- **User can verify:** materialise `effective_capacity`; inspect the Delta table to confirm P99
  values are physically reasonable (e.g. a 10 MW solar farm has `effective_capacity_mw ≈ 10`).
  Re-run `metrics`; confirm NMAE values change slightly (validation-window P99 → full-history P99).

**Future upgrade (v0.6 / v0.7):** replace the P99 computation in `effective_capacity` with the
differentiable-physics capacity model (see `docs/roadmap/differentiable-physics.md` §5). The
`EffectiveCapacity` schema, `compute_metrics` interface, and the `metrics` asset are all unchanged;
only the `effective_capacity` asset body changes.

### 7.7 Phase 7 — `live_forecasts` asset

- Implement §4.9 (`load_from_mlflow` from cache by `production_model_run_id`, single-run inference
  on 51 members in `live`/`replay`, forecast only `trained_time_series_ids`, idempotent overwrite,
  `fold_id="live"`).
- Tests: `live` vs `replay` select different NWP runs; trained-IDs-only; 51 members; idempotency.
- **User can verify:** set `production_model_run_id` to a model from Phase 4, materialise
  `live_forecasts` for the current partition, see live forecasts in `power_forecasts`.

### 7.8 Phase 8 — `production_monitoring` scope, `monitoring_sensor`, `retire_experiment_job`

- Add the `production_monitoring` scope to `metrics` (time-series logging to the dedicated
  experiment + trailing 24h/7d window columns; §4.8.1), the `monitoring_sensor` (§4.8.1), and
  `retire_experiment_job` (§4.3.1).
- Remove the `fold_id="live"` restriction in `compute_metrics()` (currently documented in its
  docstring); live rows use the same join logic but need the trailing-window bounds from the
  monitoring sensor rather than fold dates.
- Tests: sensor fires on a power update and runs `metrics(production_monitoring)`; monitoring rows
  land in Delta + the `production_monitoring` experiment and **never** on the leaderboard; retire
  job refuses to delete when results are absent and deletes when present.
- **User can verify:** trigger a power update (or the sensor), see trailing-24h/7d metrics in the
  `production_monitoring` MLflow experiment + `forecast_metrics`; run `retire_experiment_job` on a
  throwaway experiment and watch its partitions disappear (MLflow/Delta records retained).

### 7.9 Phase 9 — Remaining rigor tests and cleanup

- Add the remaining rigor tests (§5.8): CV-windowing no-lookahead, leaderboard fairness,
  determinism. (`cross_validate()` was already removed in Phase 0.) Remove any remaining dead
  code/imports; split `defs/assets.py` into
  `cv_assets.py`/`production_assets.py`/`metric_assets.py` if it has grown long (§5.1).
- **User can verify:** full `uv run pytest` green, **including the full-stack cross-process
  integration test**.
- **Docs:**
    - Update `docs/`.
    - Check the docs are still up-to-date with the code.
    - Add docs to (briefly) intro new users to the flow that our new code implements, and the reasoning behind it,
      and how to run the dagster pipeline. (Note that much of this content was added to the docs in
      Phase 4.1).
    - Also make sure the permanent docs (i.e. the docs that live in `docs/` but not in `docs/temp/`)
      capture any important ideas from `docs/temp/dagster_plan.md`,
      because `dagster_plan.md` will be deleted once this plan has been implemented in code. In
      particular, check if the permanent docs describe the aims and main design ideas of the ML R&D and
      MLops.
    - Move any content from `docs/roadmap/` that has been implemented in this plan into `docs/`:
      `docs/roadmap/` should _only_ contain ideas that have not yet been implemented in code.

### 7.10 A note on the full-stack integration tests

§5.8's default integration tier uses a **file-based MLflow** URI + **in-process Dagster
`materialize`** — fast, and right for most wiring. But **at least one** test (added in Phase 6,
re-run in Phase 9) must spin up the **full stack**: a real **`mlflow server`** subprocess (its own
backend store + artifact root) and a real **`DagsterInstance`** running the assets in **separate
processes**. Only that configuration proves the two things this design most depends on: the
**cross-process, resume-by-tag run model (§4.1.1)** and the **artifact upload/download +
local-cache** path (§4.5). Keep it behind the `@pytest.mark.integration` marker so the fast unit
suite is unaffected.
