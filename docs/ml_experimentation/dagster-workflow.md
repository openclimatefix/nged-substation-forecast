# Running an ML experiment end-to-end

How to go from raw data to a trained, MLflow-tracked model using the Dagster pipeline.

The pipeline has two layers. The **data layer** (steps 1–4) is built once and refreshed as new
data arrives; it is shared by all experiments. The **experiment layer** (steps 5–6) is repeated
for each new model or hyperparameter configuration — see [Model configuration](model-configuration.md)
for how to choose features and set hyperparameters.

---

## Step 1 — Materialise `power_time_series_and_metadata`

**Trigger:** Materialise in the Dagster UI (unpartitioned — no partition to select).

Pulls the latest NGED telemetry from S3, appends new rows to the `power_time_series.delta` Delta
table (partitioned by `time_series_id`), and upserts `metadata.parquet`. Re-materialise whenever
you want to bring in new NGED data.

## Step 2 — Materialise `h3_grid_weights`

**Trigger:** Materialise (unpartitioned). Only needs to be done once unless the GB boundary
geometry or H3 resolution changes.

Computes the fractional overlap of H3 cells with the GB boundary at the NWP grid resolution
(0.25°). The result (`h3_grid_weights.parquet`) is the lookup table `ecmwf_ens` uses to map
gridded NWP forecasts onto the H3 cells attached to each substation.

## Step 3 — Materialise `ecmwf_ens`

**Trigger:** Materialise the daily partitions from `2024-04-01` up to today (or up to the end
of your training window). Use "Materialise all" or select a date range in the Dagster UI.

Downloads the 00Z ECMWF ENS run for each partition date, converts it to a Polars DataFrame,
quantises it to 12-bit `Int16` storage, and appends it to `nwp_data.delta` (partitioned by
`[nwp_model_id, init_time]`). The `pool="ECMWF"` concurrency limit prevents OOM errors when
backfilling — Dagster schedules downloads one at a time if you materialise many partitions at
once.

## Step 4 — Materialise `eligible_time_series`

**Trigger:** Materialise the fold partition(s) you need, e.g. `mid_2025_to_mid_2026`.

Scans `power_time_series.delta` to determine which `time_series_id`s have enough data coverage
for the fold: at least `min_training_months` (default 6) of observations before `val_start`,
and observations through `val_end`. The result is written to `eligible_time_series.delta` as an
idempotent partition overwrite, so re-materialising replaces rather than appends.

Eligibility is a function of data coverage only — it is independent of any model or config —
so every experiment is scored on the identical population for a given fold. **Do not skip this
step before training.**

---

## Step 5 — Launch `register_experiment_job`

**Trigger:** Dagster UI → Jobs → `register_experiment_job` → "Launch run". Fill in the
`RegisterExperimentConfig` fields in the run config dialog:

| Field | Example | Notes |
|---|---|---|
| `experiment_name` | `"xgboost_smoke_test"` | Unique; becomes the MLflow experiment name and partition-key prefix |
| `base_model_config` | `"conf/model/xgboost.yaml"` | Path relative to `PROJECT_ROOT` |
| `config_overrides` | `{"n_estimators": 100}` | Merged onto `model_params` in the YAML |
| `run_mode` | `"smoke_test"` | `smoke_test` adds only the earliest fold; `full_cv` or `register_only` adds all folds |
| `description` | `"Quick sanity check"` | Stored as an MLflow tag — optional |

`smoke_test` is the right choice for a first run: it adds only one partition key, so you can
verify the full pipeline is wired up before committing to a potentially long `full_cv` training.

**What the job does:**

1. Loads `base_model_config` with OmegaConf and merges `config_overrides` onto `model_params`.
2. Instantiates the `BaseForecasterConfig` subclass via Hydra.
3. Creates the MLflow experiment (or resolves the existing one if the name already exists) and
   stamps four tags onto it: the resolved config as JSON (`config`), the fully-qualified Python
   class path of the forecaster (`forecaster_target`), the class path of the config class
   (`config_target`), and your `description`.
4. Creates the experiment's parent run (`cv_summary`) and logs the config as flattened params.
5. Adds dynamic partition keys (`"{experiment_name}__{fold_id}"`) to the `cv_experiment_folds`
   partition set, one per fold included in the `run_mode`.

The job is **idempotent**: re-running it with the same `experiment_name` resolves the existing
MLflow experiment and partition keys rather than creating duplicates.

## Step 6 — Materialise `trained_cv_model`

**Trigger:** Materialise the partition `"{experiment_name}__{fold_id}"`, e.g.
`"xgboost_smoke_test__mid_2025_to_mid_2026"`. The partition only appears after step 5 has run.

**What the asset does:**

1. Parses the partition key into `experiment_name` and `fold_id`.
2. Reads the forecaster class and resolved config from the MLflow experiment tags (see below).
3. Determines the training window: `[train_start 00:00 UTC, train_end 23:59 UTC]`.
4. Reads the eligible `time_series_id`s from the `eligible_time_series` asset for this fold.
5. Scans and filters `power_time_series.delta` and `nwp_data.delta` to the training window and
   eligible population (lazy — no unnecessary data is loaded into memory).
6. Engineers features via the forecaster's `feature_engineer.engineer()`.
7. Calls `forecaster.train(features)`.
8. Resolves the MLflow fold run by tag and uploads the trained model artifacts via
   `forecaster.save_to_mlflow(fold_run_id)`.
9. Logs training params (`fold_id`, `train_start`, `train_end`, `n_eligible_time_series`) to
   the fold run.

The MLflow run structure looks like this:

```
Experiment "xgboost_smoke_test"
└── cv_summary (parent run)   tags={cv_role: parent}
    │   params: n_estimators=100, learning_rate=0.05, …
    └── mid_2025_to_mid_2026  tags={cv_role: fold, fold_id: mid_2025_to_mid_2026}
            params: train_start, train_end, n_eligible_time_series
            artifacts: model/   ← trained model binary files
```

---

## Why `trained_cv_model` reads config from MLflow, not from YAML

When `register_experiment_job` runs, it resolves the base YAML plus any overrides into a single
concrete config and stamps it as a JSON tag on the MLflow experiment. **The YAML file is only
read at registration time.** From that point on, the MLflow experiment record is the authoritative
source for the experiment's config.

`trained_cv_model` reads the config back from those MLflow tags (via
`load_experiment_forecaster(experiment_name)`) for three reasons:

1. **Immutability.** The YAML on disk is mutable — someone could edit it between registering
   the experiment and training fold 2. Reading from MLflow guarantees every fold of the same
   experiment trains on exactly the config that was registered, no matter when it runs.

2. **Process independence.** Each `trained_cv_model` materialisation is a separate Dagster
   process. There is no live handle to pass between the job and the asset; the MLflow experiment
   name is the only shared identifier. The asset discovers everything it needs — forecaster class,
   config — by looking up the experiment by name.

3. **Safe retries.** If a training run fails and Dagster retries it, the asset re-reads the same
   MLflow tags and resumes the same MLflow fold run (which is identified by tag, not by a
   transient handle). The retry is guaranteed to use the same config and land in the same run
   as the original attempt.
