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
- Run 100s to 1000s of ML experiments, each with full 5-fold expanding-window CV.
- All experiments evaluated on **identical canonical CV folds** — scientific fairness
  for the leaderboard is non-negotiable.
- Per-fold observability: know which folds succeeded or failed for each experiment.
- Retry individual failed folds from the Dagster UI without re-running others.
- Kick off a **smoke test** (just the earliest fold, 2022) before committing to a
  full 5-fold run, so bad configs fail fast and cheaply.
- Run production inference every 6 hours; track which runs failed and rerun them.
  Although the production inference will happen on one machine (with its own Dagster
  instance, probably in a Docker container running on AWS App Runner). ML R&D will happen on the
  personal workstations of the two ML researchers working on this project. But the production system
  and the ML R&D systems must run the exact same code.
- **Rerunnable metrics:** the researcher will add new metrics rapidly; recomputing
  metrics from the already-saved `PowerForecast` Delta table must not require
  retraining models. So the output from each inference run for each fold must be saved to disk in
  the `power_forecasts` Delta Table.
- Zero training-serving skew: CV and production use identical data pipelines.

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

### 3.5 MLflow Pyfunc Wrappers for Model Saving

Use MLflow's model registry and `pyfunc` format to save and load trained models.

**Rejected for now because:**
- Adds significant complexity (environment snapshotting, custom loaders per model
  family, schema enforcement at the MLflow boundary).
- `BaseForecaster.save()` / `BaseForecaster.load()` are simple and sufficient.
- May revisit if we move to a centralised MLflow model store shared across machines.

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
| **Experiment** | One specific combination of model class + config (features, hyperparameters). Identified by a human-readable `experiment_name` string. |
| **CV Fold** | One expanding-window training/validation window. Fold IDs are calendar years: `"2022"` … `"2026"`. Date ranges are defined in `conf/cv/default.yaml` and are immutable per leaderboard epoch. |
| **Dagster partition key** | `"{experiment_name}__{fold_id}"`, e.g. `"xgboost_with_solar__2022"`. Encodes both the experiment and the fold; unique by construction (MLflow enforces unique experiment names). |
| **MLflow experiment** | One per `experiment_name`. Stores the resolved config as experiment tags. Holds all runs for that experiment. |
| **MLflow parent run** | One per experiment. Holds the resolved config params and the aggregate (mean-across-folds) metrics — the row the leaderboard sorts on. Tagged `cv_role="parent"`. |
| **MLflow fold (child) run** | One per fold, nested under the parent run. Holds per-fold training params and per-fold metrics. Tagged `cv_role="fold"`, `fold_id="2022"`. |

### 4.1.1 MLflow Run Model and Cross-Process Run Resolution

You do **not** need a separate MLflow experiment per fold. MLflow's *run* is exactly the
"multiple folds belong to one experiment" concept. The structure is:

```
Experiment  "xgboost_with_solar"  (experiment_id = 42)
└── Parent run  "cv_summary"      tags: cv_role=parent      ← aggregate metrics (leaderboard row)
    ├── Child run  "2022"         tags: cv_role=fold, fold_id=2022   ← per-fold params + metrics
    ├── Child run  "2023"         tags: cv_role=fold, fold_id=2023
    ├── Child run  "2024"
    ├── Child run  "2025"
    └── Child run  "2026"
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

### 4.2 Dagster Asset Graph

```
──────────────────────────────────────────────────────────────
 DATA LAYER  (existing assets, unchanged)
──────────────────────────────────────────────────────────────
 power_time_series_and_metadata   unpartitioned asset
 ecmwf_ens                        daily-partitioned asset

──────────────────────────────────────────────────────────────
 EXPERIMENT REGISTRATION  (new Dagster job)
──────────────────────────────────────────────────────────────
 register_experiment_job           RunConfig → creates MLflow experiment,
                                   adds partition keys to DynamicPartitionsDefinition

──────────────────────────────────────────────────────────────
 CV LAYER  (DynamicPartitionsDefinition, key = "name__fold")
──────────────────────────────────────────────────────────────
 trained_cv_model                  trains + saves model to disk
 cv_power_forecasts                loads model, predicts, writes to Delta
   └── deps: [trained_cv_model]

──────────────────────────────────────────────────────────────
 METRICS LAYER  (unpartitioned, rerunnable at any time)
──────────────────────────────────────────────────────────────
 metrics                           reads power_forecasts Delta,
                                   writes metrics Delta, logs to MLflow

──────────────────────────────────────────────────────────────
 PRODUCTION LAYER  (6-hourly time-window partitioned)
──────────────────────────────────────────────────────────────
 live_forecasts                    loads model from production_model_path,
                                   predicts, appends to power_forecasts Delta
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
                                 # smoke_test: adds only the __2022 partition key
                                 # full_cv:    adds all 5 partition keys (__2022...__2026)
                                 # register_only: adds all keys but triggers nothing
    description: str = ""        # stored as an MLflow experiment tag
```

The job performs these steps in order:

1. Resolve full config: load `base_model_config` YAML, apply `config_overrides`, build
   the concrete `BaseForecasterConfig` subclass via `hydra.utils.instantiate`.
2. Set `config.experiment_name = experiment_name` (see §4.7 on experiment_name field).
3. Call `mlflow.create_experiment(experiment_name, tags={"config": config.model_dump_json(), "description": description})`.
   If the experiment already exists, retrieve its ID (idempotent).
4a. Create the experiment's **parent run** now, via `get_or_create_parent_run(experiment_id)`
   (§4.1.1), and log the resolved config params to it. Creating it here — before any fold can
   run — means the per-fold assets never race to create it. (The helper is still idempotent so
   a re-registration is harmless.)
4. Based on `run_mode`, determine which fold IDs to add:
   - `"smoke_test"` → `["2022"]`
   - `"full_cv"` or `"register_only"` → `["2022", "2023", "2024", "2025", "2026"]`
5. Add Dagster partition keys `[f"{experiment_name}__{fid}" for fid in fold_ids]` to the
   `cv_experiment_folds` `DynamicPartitionsDefinition`.
   Use `context.instance.add_dynamic_partitions("cv_experiment_folds", keys)`.
6. Log partition keys and MLflow experiment ID as Dagster output metadata.

After the job completes, the user materialises the new partitions from the Dagster asset
catalog (selecting `trained_cv_model` + `cv_power_forecasts` for the new experiment).
For a smoke test, only the `__2022` partition exists, so only one fold runs.

The `register_experiment_job` uses `context.instance.add_dynamic_partitions()` which requires Dagster's instance object inside an op. Make sure to use `OpExecutionContext` (not `AssetExecutionContext`) for that job's ops, since it's a job not an asset.

### 4.4 Config Storage

- **Git** stores `conf/model/{model_class}.yaml` — the default config for a model
  family.  This file is what you edit when writing new model code.
- **MLflow** stores the resolved, fully-specified config JSON as an experiment-level tag
  (`"config"`).  This is the canonical config for every historical experiment.
- **Assets** fetch config at runtime by calling
  `mlflow.get_experiment_by_name(experiment_name)` and deserialising the `"config"` tag
  back into the appropriate `BaseForecasterConfig` subclass.

This means: no per-experiment YAML files, full reproducibility (config is immutable once
registered), and the config travels with the MLflow experiment even if the YAML in git
is later modified.

### 4.5 Model Artifact Storage (Convention-Based Paths)

Trained model artifacts are saved to disk using a deterministic path derived from the
partition key, so no coordination between the `trained_cv_model` and
`cv_power_forecasts` assets is needed:

```
{settings.trained_ml_model_params_base_path}/{experiment_name}/{fold_id}/
```

`trained_cv_model` writes to this path; `cv_power_forecasts` reads from it.
Both assets derive the path identically from their `context.partition_key`.

**Future:** if models need to be shared across machines (e.g. a colleague trains on
machine A and you want to predict on machine B), move to `mlflow.log_artifact()` so
the model binary lives in MLflow's artifact store (S3-backed).  `BaseForecaster.save/
load` already abstracts the path, so this requires only a change to how the path is
derived in the assets, not to the model code.

### 4.6 `trained_cv_model` Asset

```
partitions_def: cv_experiment_folds (DynamicPartitionsDefinition)
deps: [power_time_series_and_metadata, ecmwf_ens]
```

Per materialisation:

1. Parse `context.partition_key` → `experiment_name`, `fold_id`.
2. Fetch resolved config from MLflow experiment tags; instantiate
   `BaseForecasterConfig` subclass.
3. Fetch `CvFoldConfig` for `fold_id` from `conf/cv/default.yaml`.
4. Load `power_lf` and `nwp_lf` (lazy), filtered to
   `[fold.train_start, fold.train_end]` (inclusive on both ends — see §5.1 bug fix).
5. Filter to eligible `time_series_ids` (first obs ≥ `min_training_months` before
   `val_start`, last obs ≥ `val_end`).
6. Call `engineer_features()`, then `forecaster.train()`.
7. Save model to `{models_base_path}/{experiment_name}/{fold_id}/`.
8. Resolve the fold's MLflow run by tag (§4.1.1): `experiment_id =
   get_or_create_experiment(experiment_name)`, `parent_run_id =
   get_or_create_parent_run(experiment_id)`, `fold_run_id =
   get_or_create_fold_run(experiment_id, parent_run_id, fold_id)`. Open it with
   `mlflow.start_run(run_id=fold_run_id)`, log training params, and close it within this
   process — do **not** pass the handle or run ID to `cv_power_forecasts`.
9. Emit Dagster output metadata: n_time_series, train date range, model path.

### 4.7 `cv_power_forecasts` Asset

```
partitions_def: cv_experiment_folds (same DynamicPartitionsDefinition)
deps: [trained_cv_model]
```

Per materialisation:

1. Parse `context.partition_key` → `experiment_name`, `fold_id`.
2. Fetch config from MLflow (same as above).
3. Load model from `{models_base_path}/{experiment_name}/{fold_id}/` via
   `ForecasterClass.load(path)`.
4. Load `power_lf` and `nwp_lf` filtered to `[fold.val_start, fold.val_end]`.
5. Call `engineer_features()`, then `forecaster.predict()`.
6. Overwrite `fold_id` column with the string `fold_id` (year string, e.g. `"2022"`).
   The `power_fcst_model_name` column carries `experiment_name` (stamped by `predict()`
   via `BaseForecasterConfig.experiment_name` — see §5.1).
7. Append to `power_forecasts` Delta table, partitioned by
   `(power_fcst_model_name, fold_id)`.
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

- Reads from the `power_forecasts` Delta table.  Accepts an optional filter via
  Dagster `RunConfig` (e.g. `experiment_name`, or `fold_id != "live"`).
  Default: all CV rows (fold_id ∈ year strings).
- Calls the existing `compute_metrics()` from `ml_core.metrics`.
- Writes to `forecast_metrics` Delta table (replaces current `cv_metrics.delta`),
  partitioned by `(power_fcst_model_name, fold_id)`.
- Logs **per-fold** metrics to each fold child run and **aggregate** (mean-across-folds)
  metrics to the experiment's **parent run** — both resumed by tag (§4.1.1): for each
  `experiment_name` it computes `experiment_id = get_or_create_experiment(...)` and
  `parent_run_id = get_or_create_parent_run(...)`, then logs aggregates to the parent and
  per-fold numbers to the fold runs resolved via `get_or_create_fold_run(...)`. The aggregate
  metrics on the parent run are what the leaderboard sorts on. Because lookup is by tag, this
  asset can be re-run for new metrics at any time and updates the existing runs in place.

Because this asset reads from Delta and writes to Delta with no model dependency, it
can be re-run at any time as new metrics are added to `compute_metrics()` without
re-running any training.

The existing `cv_metrics` Dagster asset is **deleted** and replaced by `metrics`.

### 4.9 `live_forecasts` Asset

```
partitions_def: TimeWindowPartitionsDefinition(cron="0 0,6,12,18 * * *", ...)
deps: [ecmwf_ens, power_time_series_and_metadata]
```

- Loads the production model from `settings.production_model_path` using
  `ForecasterClass.load()`.  The class is inferred from a `model_class` field stored
  alongside the saved model (already part of `BaseForecaster.save()`'s `meta.json`).
- Loads NWP for the relevant 6-hour window (via `TimeWindowPartitionMapping` against
  the daily-partitioned `ecmwf_ens`).
- Calls `engineer_features()` + `forecaster.predict()`.
- Stamps `fold_id = "live"` on all rows.
- Appends to the shared `power_forecasts` Delta table.

`settings.production_model_path` is a `Path` field in `Settings`, set via `.env`.
Promoting a new model to production = updating this path and restarting Dagster.
(A richer MLflow-based champion-model promotion flow is deferred to Phase 2.)

---

## 5. Implementation Notes

### 5.1 Required Changes to Existing Code

#### `packages/contracts/src/contracts/hydra_schemas.py`
- **Fix the `train_end` bug:** `CvFoldConfig.train_end` is currently loaded from the
  YAML but silently ignored in `cross_validate.py` — `train_end_dt` is computed as
  `val_start_dt` instead.  Fix: use `fold.train_end` directly when filtering training
  data.  The YAML values are correct (e.g. `"2021-12-31"` exclusive = `"2022-01-01"`
  exclusive), so no YAML changes needed — just wire the field up.

#### `packages/contracts/src/contracts/power_schemas.py`
- `FoldId` is currently a `Literal["live", "2022", ...]`.  This is fine for now but
  should be noted as needing extension when new CV epochs are added.
- No structural changes needed; `PowerForecast.fold_id` and
  `PowerForecast.power_fcst_model_name` already support the new design.

#### `packages/ml_core/src/ml_core/base_forecaster.py`
- Add `experiment_name: str = ""` field to `BaseForecasterConfig`.  This is set to
  the MLflow experiment name at registration time and stored in the saved config.
  The `predict()` implementation in each subclass should use
  `self.model_params.experiment_name or self.MODEL_NAME` as the value stamped into
  `PowerForecast.power_fcst_model_name`.
- No changes to the abstract interface (`train`, `predict`, `save`, `load`).

#### `packages/ml_core/src/ml_core/cross_validate.py`
- **Delete** the `cross_validate()` function.  Its logic is absorbed into the
  `trained_cv_model` and `cv_power_forecasts` Dagster assets.
- **Keep** any shared helper functions (e.g. `_date_to_utc_datetime`,
  eligibility-filtering logic) — move them to a new private module
  `ml_core._cv_helpers` or inline them into the asset implementations.

#### `packages/ml_core/src/ml_core/metrics.py`
- No changes needed.  `compute_metrics()` remains a standalone Python function,
  called by the `metrics` Dagster asset.

#### `packages/contracts/src/contracts/settings.py`
- Add `production_model_path: Path | None = None` field.
- Rename `cv_metrics_data_path` → `forecast_metrics_data_path` (or keep both during
  transition; the `cv_metrics.delta` file will be replaced by a broader `metrics.delta`).
- `trained_ml_model_params_base_path` already exists — this is the root for CV model
  artifacts.

#### `src/nged_substation_forecast/defs/assets.py`
- **Delete** `cv_power_forecasts` asset (the current monolithic one).
- **Delete** `cv_metrics` asset.
- **Add** `cv_experiment_folds = DynamicPartitionsDefinition(name="cv_experiment_folds")`.
- **Add** `trained_cv_model` asset (§4.6).
- **Add** `cv_power_forecasts` asset — same name, new implementation (§4.7).
- **Add** `metrics` asset (§4.8).
- **Add** `live_forecasts` asset (§4.9).
- Consider splitting into multiple files once this gets long:
  `defs/cv_assets.py`, `defs/production_assets.py`, `defs/metric_assets.py`.

### 5.2 New Files to Create

- `src/nged_substation_forecast/defs/jobs.py` — `register_experiment_job` and its
  `RegisterExperimentConfig`.  This keeps job definitions separate from asset definitions.
- `packages/ml_core/src/ml_core/_mlflow_helpers.py` — the three idempotent run-resolution
  helpers (§5.6), shared by the job and all CV/metrics assets.

### 5.3 What Stays the Same

The following are correct and require no changes:

- `packages/ml_core/src/ml_core/features.py` — `engineer_features()` and all feature
  classes.
- `packages/xgboost_forecaster/` — `XGBoostForecaster` and `XGBoostConfig` (except
  that `XGBoostConfig` inherits the new `experiment_name` field from
  `BaseForecasterConfig`).
- `packages/contracts/src/contracts/ml_schemas.py` — `AllFeatures`, `Metrics` schema.
- `conf/cv/default.yaml` — canonical fold definitions; immutable per leaderboard epoch.
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
```

All run creation and lookup is funnelled through the §5.6 helpers; assets never call
`search_runs` / `create_experiment` directly.

**Not** in scope: `mlflow.log_artifact()`, `mlflow.pyfunc`, model registry, model
promotion.

### 5.6 MLflow Run-Resolution Helpers

Create `packages/ml_core/src/ml_core/_mlflow_helpers.py` (or a small module under the Dagster
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
    filter_string="tags.cv_role = 'fold' and tags.fold_id = '2022'",
    max_results=1,
)
fold_run_id = runs[0].info.run_id if runs else _create_fold_run(...)
```

---

## 6. Known Limitations and Future Work

- **Model artifact sharing across machines:** currently models are saved to
  `settings.trained_ml_model_params_base_path` on local disk (or an S3-mounted
  filesystem).  If training and prediction need to run on different machines, migrate
  to `mlflow.log_artifact()` so the model binary is stored in MLflow's S3-backed
  artifact store.  The `BaseForecaster.save/load` abstraction is already in place;
  only the path derivation in the assets needs to change.

- **Champion model promotion:** currently `settings.production_model_path` is set
  manually.  Phase 2 will add a `champion_model` Dagster asset that queries MLflow for
  the experiment tagged `"Production"` and writes the model path to a well-known
  location, removing the need for manual `.env` edits.

- **Ensemble NWP in CV:** the current CV code uses only ensemble member 0 (the
  deterministic control member).  Full ensemble support requires implementing the
  XGBoost iterator API; this is out of scope here.

- **`FoldId` Literal extension:** when new CV epochs are added (e.g. adding a `"2027"`
  fold), update the `FoldId` `Literal` in `contracts/power_schemas.py` and add the new
  fold to `conf/cv/default.yaml`.

- **Concurrent Delta writes:** multiple experiments appending simultaneously to the
  shared `power_forecasts` Delta table is safe because they write to different
  partition directories (`power_fcst_model_name` differs).  However, if two runs of
  the exact same experiment are launched simultaneously, they will produce duplicate
  rows.  A guard at the start of `cv_power_forecasts` (check whether the partition
  already has rows in Delta) is a reasonable safeguard to add.


## 7. Implementation note

