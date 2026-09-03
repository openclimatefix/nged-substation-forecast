# Running an ML experiment end-to-end

How to go from raw data to a trained, MLflow-tracked model using the Dagster pipeline.

The pipeline has two layers. The **data layer** (steps 1–5) is built once and refreshed as new
data arrives; it is shared by all experiments. The **experiment layer** (steps 6–9) is repeated
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

Downloads the 00Z ECMWF ENS run for each partition date, converts it to a Polars DataFrame, and
writes it to `nwp_data.delta` (partitioned by `[nwp_model_id, init_time]`) as physical-unit
`Float32` rounded to a 13-bit significand at write time by `delta_store.nwp`. Each run replaces its
own partition, so re-materialising a date range you have already ingested is safe. The
`pool="ECMWF"` concurrency limit caps how many partitions download at once, so materialising a
long date range queues the downloads rather than starting all of them: four run at a time, and
each of those runs four concurrent fetches of its own.

## Step 4 — Materialise `eligible_time_series`

**Trigger:** Materialise the fold partition(s) you need, e.g. `mid_2025_to_mid_2026`.

Scans `power_time_series.delta` to determine which `time_series_id`s have enough data coverage
for the fold: at least `min_training_months` (default 6) of observations before `val_start`,
and observations through `val_end`. The result is written to `eligible_time_series.delta` as an
idempotent partition overwrite, so re-materialising replaces rather than appends.

Eligibility is a function of data coverage only — it is independent of any model or config —
so every experiment is scored on the identical population for a given fold. **Do not skip this
step before training.**

## Step 5 — Materialise `effective_capacity`

**Trigger:** Materialise (unpartitioned). Re-materialise whenever step 1 has pulled in enough new
telemetry to shift a series' full-history P99.

Reads the full `power_time_series` Delta and writes one row per `time_series_id` to the
`effective_capacity` Delta table: the 99th percentile of `abs(power)` over the series' entire
observed history. This is the NMAE (normalised mean absolute error) denominator the `metrics`
asset (step 9) reads — materialise it before running `metrics`, or that step raises
`FileNotFoundError`.

---

## Step 6 — Launch `register_experiment_job`

**Trigger:** Dagster UI → Jobs → `register_experiment_job` → "Launch run". Fill in the
`RegisterExperimentConfig` fields in the run config dialog:

| Field | Example | Notes |
|---|---|---|
| `experiment_name` | `"xgboost_smoke_test"` | Unique; becomes the MLflow experiment name and partition-key prefix |
| `base_model_config` | `"conf/model/xgboost.yaml"` | Path relative to `PROJECT_ROOT` |
| `config_overrides` | `{"n_estimators": 100}` | Merged onto `model_params` in the YAML; every key must name a declared config field |
| `run_mode` | `"smoke_test"` | `smoke_test` adds the non-leaderboard dev folds (e.g. `smoke_test`); `full_cv` or `register_only` adds the leaderboard folds |
| `description` | `"Quick sanity check"` | Stored as an MLflow tag — optional |

`smoke_test` is the right choice for a first run: it adds only one partition key, so you can
verify the full pipeline is wired up before committing to a potentially long `full_cv` training.

**What the job does:**

1. Loads `base_model_config` as plain YAML and applies `config_overrides` to `model_params`.
2. Constructs the forecaster's `CONFIG_CLASS`, so pydantic validates every hyperparameter before
   the experiment is registered.
3. Creates the MLflow experiment (or resolves the existing experiment if the name already
   exists), and rejects the registration outright if that name is already registered under a
   *different* config — see "An experiment's identity is its config" below.
4. Creates the experiment's parent run (`cv_summary`) and logs the config as flattened params.
5. Stamps three tags onto the experiment: the resolved config as JSON (`config`), the
   fully-qualified Python class path of the forecaster (`forecaster_target`), and your
   `description`. The config class needs no tag of its own — it is the forecaster's `CONFIG_CLASS`.
6. Adds dynamic partition keys (`"{experiment_name}__{fold_id}"`) to the `cv_experiment_folds`
   partition set, one per fold included in the `run_mode`.

The job is **idempotent** for a given config: re-running it with the same `experiment_name`
resolves the existing MLflow experiment and partition keys rather than creating duplicates. You
may freely re-run it to edit the `description` or to add the other run mode's folds.

### An experiment's identity is its config

Re-running the job with the same `experiment_name` but a **changed** config is refused, with an
error naming the fields that differ. Register the new config under a new `experiment_name`.

Every fold of an experiment must be trained and scored under one config, or the experiment's
leaderboard row silently mixes two different models. Folds already materialised cannot be
un-trained, and `trained_cv_model` reads the config back from the experiment's `config` tag (see
"Why `trained_cv_model` reads config from MLflow, not from YAML" below), so re-pointing that tag
mid-flight would change what later folds train on. The refusal happens before any record is
written, so a rejected re-registration leaves the experiment exactly as it was.

How the refusal decides that a config changed — which tags it compares, and why an absent tag does
not count as a change — is in [Re-registering an experiment under a changed config is
rejected](../architecture/ml-orchestration.md#re-registering-an-experiment-under-a-changed-config-is-rejected).

## Step 7 — Materialise `trained_cv_model`

**Trigger:** Materialise the partition `"{experiment_name}__{fold_id}"`, e.g.
`"xgboost_smoke_test__smoke_test"`. The partition only appears after step 6 has run.

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
   GB rather than OOMing on the tens-of-GB NWP table (see
   [Bounding feature-engineering memory: prune the inputs, not the output](../architecture/performance.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)).
6. Engineers features via the forecaster's `feature_engineer.engineer()`.
7. Calls `forecaster.train(features, eligible_ids)` (the population is passed explicitly). The asset
   then **raises** if zero boosters were trained (e.g. no series had usable power in the window).
8. Resolves the MLflow fold run by tag and uploads the trained model artifacts via
   `forecaster.save_to_mlflow(fold_run_id, time_series_metadata=...)`, which puts the roster rows
   the model trained against into the archive alongside it.
9. Records the training run on the fold run as **tags**: the training window (`train_start`,
   `train_end`) and the populations (`n_eligible_time_series`, `n_trained_time_series`).

A fold run is reused on every re-materialisation of its partition and MLflow params are
write-once, so nothing that can legitimately change between materialisations may be a param. The
training window comes from the CV config (which is edited as the archive grows), and the counters
are outputs of the materialisation (the eligible population grows — and can shrink — with power
coverage). Both would therefore make a re-materialisation fail with "Changing param values is not
allowed" if logged as params. Tags, not metrics: MLflow resolves a metric's "latest" value as the
max over `(step, timestamp, value)` rather than the newest write, which would under-report a
shrunk count landing on the same timestamp/step as a prior larger one. Tags are last-write-wins,
which is the semantic actually wanted. `fold_id` itself is already a tag from run creation
(`get_or_create_fold_run`, which is also what resolves the run by it), so it is not logged again
here.

The MLflow run structure after training looks like this:

```text
Experiment "xgboost_smoke_test"
└── cv_summary (parent run)   tags={cv_role: parent}
    │   params: n_estimators=100, learning_rate=0.05, …
    └── smoke_test  tags={cv_role: fold, fold_id: smoke_test, train_start, train_end,
                           n_eligible_time_series, n_trained_time_series}
            artifacts: model/   ← trained model binary files
```

---

## Step 8 — Materialise `cv_power_forecasts`

**Trigger:** Materialise the same `"{experiment_name}__{fold_id}"` partition (it depends on
`trained_cv_model`).

**What the asset does:**

1. Loads the fold's model back from MLflow (a fresh download each time — no local cache, see
   [ML orchestration: model artifacts](../architecture/ml-orchestration.md#model-artifacts-one-replaceable-archive-no-local-cache))
   and reads its `trained_time_series_ids` — the population it scores (the train==predict
   invariant). Raises if the loaded model has no trained series.
2. Forecasts the **inclusive validation window** across **all ~51 NWP ensemble members** (the
   probabilistic leaderboard metrics are meaningless on a single member).
3. Bounds memory by predicting **one `init_time` chunk at a time** (`_PREDICT_INIT_CHUNK`, 14 days):
   the full ensemble over the whole window is tens of GB, so chunking on `init_time` (the partition
   key and the axis that fans the output out across runs) keeps each chunk's forecast frame small
   while every partition is read once. Measured ~9 GB peak for a 10-month fold.
4. Writes to the `power_forecasts` Delta table keyed by `(experiment_name, fold_id)`: the **first**
   chunk overwrites the partition (clearing any prior run), the rest **append**, so a full
   re-materialisation replaces the fold's rows without ever holding all forecasts in memory.
5. Tags the fold run with `val_start`/`val_end` and the `n_forecast_rows`/
   `n_forecast_time_series`/`n_ensemble_members` counters. All five are tags rather than params or
   metrics because they legitimately change — and can shrink — between materialisations of the
   same fold run. Params are write-once, and MLflow reports a metric's latest value as the max
   over all its writes, so a shrunk count would read back too high.

---

## Inspecting a forecast — the `view_forecasts` dashboard

Once forecasts exist in the `power_forecasts` Delta table, the `view_forecasts` marimo app plots
a single forecast so you can eyeball it. It is independent of the training flow above — launch it
any time there are forecasts on disk to inspect:

```bash
uv run marimo edit packages/dashboard/view_forecasts.py
```

Pick the population with the dropdowns: the **Fold** dropdown lists every `fold_id` present in
the `power_forecasts` table (a CV fold label, a smoke-test fold, or `live` for production
forecasts), and an **Experiment** dropdown appears when the chosen fold holds more than one
`experiment_name`. Then choose a **time series** (the dropdown groups the 32 series by type, so
all the PV sites or all the primaries sit together), a **forecast date**, and one of that day's
**forecast runs**.

**Reload data** re-reads the tables, so a CV job that finishes while the app is open shows up
without restarting marimo. Its new experiment appears in the **Experiment** dropdown; **Fold**,
**forecast date** and **forecast run** go back to their defaults, and the chosen time series is
kept.

The chart layers all 51 ensemble members as thin grey lines, observed power (wherever available,
including past the init time) as a thick blue line, and a vertical rule at the forecast init
time, spanning 24 hours before the init time to 14 days after it. The x-axis is labelled at
midnight (Europe/London) with the day of week and date, with unlabelled minor ticks every 3
hours. Scroll to zoom, drag to pan, hover for the `ensemble_member` and value.

The **Data source** radio switches the app between the local data tables (the root `.env`) and
the production S3 buckets without restarting marimo — see the
[dashboard README](https://github.com/openclimatefix/nged-substation-forecast/tree/main/packages/dashboard)
for the `.env.s3` setup, and
[Operating the live service: Inspecting a live forecast](../live_service/operations.md#inspecting-a-live-forecast)
for the production workflow.

---

## Step 9 — Materialise `metrics`

**Trigger:** Materialise (unpartitioned — not tied to a single fold). Fill in `MetricsConfig`
in the run config dialog before launching.

| Field | Example | Notes |
|---|---|---|
| `population_filter.experiment_name` | `"xgboost_smoke_test"` | Filters `power_forecasts` to one experiment; leave null to score all experiments at once |
| `population_filter.fold_id` | `"mid_2025_to_mid_2026"` | Filters to one fold; leave null to score all folds for the experiment |
| `population_filter.valid_time_min/max` | `"2025-10-01T00:00:00+00:00"` | ISO-8601 UTC; trims the valid_time window for ad_hoc scoring |
| `evaluation_scope` | `"leaderboard"` | `"leaderboard"` logs to MLflow; `"ad_hoc"` writes Delta only |

**What the asset does:**

1. Scans `power_forecasts` Delta, applying any non-null `PopulationFilter` predicates. The
   partition columns (`experiment_name` / `fold_id`) are `String`, matching what delta-rs stores,
   so the predicates push straight into the Delta scan: naming an experiment/fold prunes to just
   that partition rather than reading the whole (unbounded) table.
2. In `leaderboard` scope, drops any group whose `fold_id` is not a leaderboard fold in the CV
   config, naming the dropped ids in a warning and in the asset's `skipped_fold_ids` metadata. The
   live service writes to this same table under `fold_id="live"`, and a non-leaderboard dev fold
   such as `smoke_test` lands there too, so an unfiltered leaderboard run finds both; leaderboard
   scope dates its evaluation window from the CV config's leaderboard folds and has none for those
   rows. To score live output or a dev fold, run the asset with `evaluation_scope="ad_hoc"`, which
   takes the window from the forecast rows themselves.
3. Discovers the matching `(experiment_name, fold_id)` groups, then scores each group in batches of
   four `time_series_id` values at a time — peak memory is one batch, never a whole fold or the
   entire matched population. See
   [The other hard ceiling: Polars' 32-bit row index](../architecture/performance.md#the-other-hard-ceiling-polars-32-bit-row-index)
   for why this chunking also keeps the row-index cap out of reach at V2 scale.
   For each group:
   a. Calls `compute_metrics()` — joins observed power, collapses each forecast run's ensemble
      members into per-timestamp quantities, and computes the deterministic metrics
      (MAE / NMAE / RMSE / MBE on the ensemble mean) plus the probabilistic metrics (fair
      CRPS, spread-skill ratio, pinball loss at the 13 delivery quantiles, PICP and interval
      width for the 6 symmetric bands) per
      `(time_series_id, fold_id, power_fcst_model_name, horizon_slice)` — see the
      [evaluation-metrics reference](../techniques/evaluation-metrics.md) for definitions.
   b. Enriches rows with scope (`evaluation_scope`), window bounds (`window_start`, `window_end`,
      `window_label`), `computed_at`, and the MLflow fold run id (leaderboard scope only).
   c. Writes to `forecast_metrics` Delta, partitioned by `(experiment_name, fold_id)` with an
      idempotent overwrite predicate — safe to re-run without duplicating rows.
4. For `evaluation_scope="leaderboard"`: builds an aggregate metric dict and logs it to the
   fold's MLflow child run, then averages across folds and logs the mean to the parent run.
   The key token is `{metric_name}` for scalar metrics and `{metric_name}_{metric_param}` for
   parametric metrics, in three families: overall (`rmse__all`, `crps__all`), per type
   (`rmse__disaggregated_demand`), and per horizon slice (`nmae__all__day_ahead`). Parametric
   metrics are restricted to a headline subset in MLflow (`pinball_loss` at p10/p50/p90;
   `picp`/`interval_width` at p10_p90) — the full 13-quantile / 6-band detail stays in the
   `forecast_metrics` Delta table.

After step 9, the MLflow run structure looks like this:

```text
Experiment "xgboost_smoke_test"
└── cv_summary (parent run)   tags={cv_role: parent}
    │   params: n_estimators=100, learning_rate=0.05, …
    │   metrics: rmse__all=4.3, rmse__disaggregated_demand=4.1, …   ← mean across folds
    └── smoke_test  tags={cv_role: fold, fold_id: smoke_test, train_start, train_end, …}
            metrics: rmse__all=4.3, rmse__disaggregated_demand=4.1, …   ← per-fold aggregate
            artifacts: model/
```

The `forecast_metrics` Delta table stores one row per
`(time_series_id, fold_id, power_fcst_model_name, horizon_slice, metric_name, metric_param)`,
with `time_series_type` populated from metadata so per-type queries need only a simple filter.

---

## Viewing results in the MLflow UI

To browse experiments, compare the leaderboard, and inspect per-fold metrics and artifacts, launch
the MLflow web UI against the same tracking store the pipeline writes to:

```bash
uv run mlflow ui --gunicorn-opts "--workers 1"
```

Then open `http://localhost:5000`.

**The `--gunicorn-opts` flag is required on Python 3.14.** MLflow 3.14's default server
(uvicorn + FastAPI) fails to start on Python 3.14 — `mlflow.server.assistant` imports
`importlib.abc.Traversable`, which was removed in 3.14. Passing `--gunicorn-opts` selects the
Flask/gunicorn server instead, which works. Drop the flag once MLflow ships a Python 3.14-compatible
FastAPI server.

Full MLflow (which bundles the web server) is in the `dev` dependency group, so `uv sync` installs
it; production runs use `mlflow-skinny` (the client, without the server).

## Reproducibility: reconstructing a run from its stamped provenance

Every run carries the code and data provenance needed to answer "exactly which code and which data
produced this?". Each asset stamps **stage-prefixed** MLflow tags on the run it writes:

- `register_git_sha` / `register_git_dirty` on the **parent run** — the code that registered the
  experiment. (`git_dirty` is `true` when the working tree had uncommitted changes.)
- `train_*`, `predict_*`, `metrics_*` on the **fold run** — one snapshot per stage, because
  `trained_cv_model`, `cv_power_forecasts` and `metrics` all write to the same fold run and may run
  days apart on different code. Each carries that stage's `git_sha` / `git_dirty` plus a
  `{stage}_delta_version__{table}` tag holding the [Delta Lake](https://delta.io/) version of every
  table it read (e.g. `train_delta_version__power_time_series`, `train_delta_version__nwp_data`).
  The git SHA is stamped explicitly rather than via MLflow's auto-detection, which needs gitpython
  and a repo working directory — neither present in a production container (there the SHA degrades
  to `unknown`, never failing the run).

To reconstruct the training-time state of a fold: `git checkout {train_git_sha}`, then read each
table at its logged version with Delta time travel —
`pl.scan_delta(power_time_series_path, version=<train_delta_version__power_time_series>)`.

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
   experiment trains on exactly the config that was registered, no matter when it runs. The tag
   is immutable in turn: re-registering the experiment name under a changed config is refused
   rather than re-pointing it (see step 6 above).

2. **Process independence.** Each `trained_cv_model` materialisation is a separate Dagster
   process. There is no live handle to pass between the job and the asset; the MLflow experiment
   name is the only shared identifier. The asset discovers everything it needs — forecaster class,
   config — by looking up the experiment by name.

3. **Safe retries.** If a training run fails and Dagster retries it, the asset re-reads the same
   MLflow tags and resumes the same MLflow fold run (which is identified by tag, not by a
   transient handle). The retry is guaranteed to use the same config and land in the same run
   as the original attempt.
