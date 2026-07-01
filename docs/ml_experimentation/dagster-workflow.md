# Running an ML experiment end-to-end

How to go from raw data to a trained, MLflow-tracked model using the Dagster pipeline.

The pipeline has two layers. The **data layer** (steps 1–4) is built once and refreshed as new
data arrives; it is shared by all experiments. The **experiment layer** (steps 5–8) is repeated
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
| `run_mode` | `"smoke_test"` | `smoke_test` adds the non-leaderboard dev folds (e.g. `smoke_test`); `full_cv` or `register_only` adds the leaderboard folds |
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
`"xgboost_smoke_test__smoke_test"`. The partition only appears after step 5 has run.

**What the asset does:**

1. Parses the partition key into `experiment_name` and `fold_id`.
2. Reads the forecaster class and resolved config from the MLflow experiment tags (see below).
3. Determines the training window: `[train_start 00:00 UTC, train_end 23:59 UTC]`.
4. Reads the eligible `time_series_id`s from the `eligible_time_series` asset for this fold. If the
   set is empty (the fold was never materialised in step 4, or no series meets the eligibility
   window) the asset **raises** rather than silently training nothing.
5. Loads inputs for the training window and eligible population. The NWP scan is pruned at the
   source — control member only, the eligible series' H3 cells, and the window's `init_time`
   partitions — and collected with the **streaming engine**, so a multi-month fold trains in a few
   GB rather than OOMing on the tens-of-GB NWP table (see the "Bounding feature-engineering memory"
   notes in `docs/architecture/overview.md`).
6. Engineers features via the forecaster's `feature_engineer.engineer()`.
7. Calls `forecaster.train(features, eligible_ids)` (the population is passed explicitly). The asset
   then **raises** if zero boosters were trained (e.g. no series had usable power in the window).
8. Resolves the MLflow fold run by tag and uploads the trained model artifacts via
   `forecaster.save_to_mlflow(fold_run_id)`.
9. Logs training params (`fold_id`, `train_start`, `train_end`, `n_eligible_time_series`,
   `n_trained_time_series`) to the fold run.

The MLflow run structure after training looks like this:

```text
Experiment "xgboost_smoke_test"
└── cv_summary (parent run)   tags={cv_role: parent}
    │   params: n_estimators=100, learning_rate=0.05, …
    └── smoke_test  tags={cv_role: fold, fold_id: smoke_test}
            params: train_start, train_end, n_eligible_time_series, n_trained_time_series
            artifacts: model/   ← trained model binary files
```

---

## Step 7 — Materialise `cv_power_forecasts`

**Trigger:** Materialise the same `"{experiment_name}__{fold_id}"` partition (it depends on
`trained_cv_model`).

**What the asset does:**

1. Loads the fold's model back from MLflow (via the local-disk cache) and reads its
   `trained_time_series_ids` — the population it scores (the train==predict invariant). Raises if
   the loaded model has no trained series.
2. Forecasts the **inclusive validation window** across **all ~51 NWP ensemble members** (the
   probabilistic leaderboard metrics are meaningless on a single member).
3. Bounds memory by predicting **one `init_time` chunk at a time** (`_PREDICT_INIT_CHUNK`, 14 days):
   the full ensemble over the whole window is tens of GB, so chunking on `init_time` (the partition
   key and the axis that fans the output out across runs) keeps each chunk's forecast frame small
   while every partition is read once. Measured ~9 GB peak for a 10-month fold.
4. Writes to the `power_forecasts` Delta table keyed by `(experiment_name, fold_id)`: the **first**
   chunk overwrites the partition (clearing any prior run), the rest **append**, so a full
   re-materialisation replaces the fold's rows without ever holding all forecasts in memory.
5. Logs `val_start`/`val_end` params and `n_forecast_rows`/`n_forecast_time_series`/
   `n_ensemble_members` metrics to the fold run.

---

## Inspecting a forecast — `plot_power_forecast_job`

Once forecasts exist in the `power_forecasts` Delta table, this job renders an interactive HTML
plot of a single forecast so you can eyeball it. It is independent of the training flow above —
launch it any time there are forecasts on disk to inspect.

**Trigger:** Dagster UI → Jobs → `plot_power_forecast_job` → "Launch run". The default run config
plots an existing smoke-test forecast, so you can launch it as-is; override the
`PlotPowerForecastConfig` fields to plot a different forecast:

| Field | Example | Notes |
|---|---|---|
| `experiment_name` | `"xgboost_smoke_test_3"` | Selects the `(experiment_name, fold_id)` Delta partition to read |
| `fold_id` | `"smoke_test"` | Use `"live"` for a production forecast |
| `power_fcst_init_time` | `"2025-01-29T06:00:00+00:00"` | ISO-8601 (Dagster config has no native datetime type); a naive value is read as UTC. The plot spans this time plus the 14-day horizon |
| `time_series_ids` | `[1, 2, 3, 4]` | Between 1 and 4 ids; each is drawn on its own panel |

**What the job does:**

1. Reads `power_forecasts` for `(experiment_name, fold_id)` at the chosen `power_fcst_init_time`,
   filtered to `time_series_ids`. Errors if no rows match — check the matching partition has been
   materialised.
2. Reads observed power over the 14-day horizon and the series metadata.
3. Builds an Altair chart with one panel per `time_series_id`: all 51 ensemble members as thin
   grey lines, ground truth (where available) as a thick blue line, and a shared, zoomable x-axis
   so panning one panel moves them all.
4. Saves it as a self-contained interactive HTML file under `plots_data_path` (`data/plots/`),
   named `{experiment_name}__{fold_id}__{init_time}__ts-{ids}.html`. Open it in a browser to zoom
   and hover (tooltips show the `ensemble_member` and value).

This is a **job, not an asset**, because the plot is a throwaway artifact for human inspection —
keyed by init time and series ids, with no lineage or durable catalog identity — rather than a
tracked data object in the pipeline.

---

## Step 8 — Materialise `metrics`

**Trigger:** Materialise (unpartitioned — not tied to a single fold). Fill in `MetricsConfig`
in the run config dialog before launching.

| Field | Example | Notes |
|---|---|---|
| `population_filter.experiment_name` | `"xgboost_smoke_test"` | Filters `power_forecasts` to one experiment; leave null to score all experiments at once |
| `population_filter.fold_id` | `"mid_2025_to_mid_2026"` | Filters to one fold; leave null to score all folds for the experiment |
| `population_filter.valid_time_min/max` | `"2025-10-01T00:00:00+00:00"` | ISO-8601 UTC; trims the valid_time window for ad_hoc scoring |
| `evaluation_scope` | `"leaderboard"` | `"leaderboard"` logs to MLflow; `"ad_hoc"` writes Delta only |

**What the asset does:**

1. Scans `power_forecasts` Delta, applying any non-null `PopulationFilter` predicates.
2. Groups the matched rows by `(experiment_name, fold_id)` and for each group:
   a. Calls `compute_metrics()` — joins observed power, averages ensemble members, computes
      MAE / NMAE / RMSE / MBE per `(time_series_id, fold_id, power_fcst_model_name)`.
   b. Enriches rows with scope (`evaluation_scope`), window bounds (`window_start`, `window_end`,
      `window_label`), `computed_at`, and the MLflow fold run id (leaderboard scope only).
   c. Writes to `forecast_metrics` Delta, partitioned by `(experiment_name, fold_id)` with an
      idempotent overwrite predicate — safe to re-run without duplicating rows.
3. For `evaluation_scope="leaderboard"`: builds a per-type + overall aggregate metric dict (e.g.
   `rmse__all`, `rmse__disaggregated_demand`) and logs it to the fold's MLflow child run, then
   averages across folds and logs the mean to the parent run.

After step 8, the MLflow run structure looks like this:

```text
Experiment "xgboost_smoke_test"
└── cv_summary (parent run)   tags={cv_role: parent}
    │   params: n_estimators=100, learning_rate=0.05, …
    │   metrics: rmse__all=4.3, rmse__disaggregated_demand=4.1, …   ← mean across folds
    └── smoke_test  tags={cv_role: fold, fold_id: smoke_test}
            params: train_start, train_end, …
            metrics: rmse__all=4.3, rmse__disaggregated_demand=4.1, …   ← per-fold aggregate
            artifacts: model/
```

The `forecast_metrics` Delta table stores one row per
`(time_series_id, fold_id, power_fcst_model_name, horizon_slice, metric_name, metric_param)`,
with `time_series_type` populated from metadata so per-type queries need only a simple filter.

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
