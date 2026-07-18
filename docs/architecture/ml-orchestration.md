# ML Experiment Orchestration — Design Decisions

This page records the design decisions behind the Dagster/MLflow experiment layer
(registration → training → forecasting → metrics) and the alternatives we rejected, so the
reasoning stays auditable. The step-by-step *how-to* is
[Running an experiment end-to-end](../ml_experimentation/dagster-workflow.md).

## Key concepts

| Concept | Definition |
|---|---|
| **Experiment** | One specific combination of model class + config (features, hyperparameters), identified by a human-readable `experiment_name`. Stamped onto every forecast row in a dedicated `experiment_name` column — **not** overloaded onto `power_fcst_model_name`, which keeps its meaning as the model *family* (`MODEL_NAME`). |
| **CV fold** | One expanding-window train/validation split, identified by a short `fold_id` label. Date ranges live in `conf/cv/default.yaml` and are immutable per leaderboard epoch (see [Cross-validation folds](../ml_experimentation/cross-validation-folds.md)). |
| **Dagster partition key** | `"{experiment_name}__{fold_id}"` on a `DynamicPartitionsDefinition`. Encodes both dimensions; unique by construction because MLflow enforces unique experiment names. |
| **MLflow experiment** | One per `experiment_name`; carries the resolved config as tags and holds all of the experiment's runs. |
| **MLflow parent run** | One per experiment (`cv_summary`, tagged `cv_role=parent`). Holds the flattened config params and the mean-across-folds aggregate metrics — the row the leaderboard sorts on. |
| **MLflow fold run** | One per fold, nested under the parent (tagged `cv_role=fold`, `fold_id=…`). Holds per-fold training params, per-fold metrics, and the trained model artifacts. |

## Cross-process run resolution: discover by tag, never pass handles

`trained_cv_model`, `cv_power_forecasts`, and `metrics` are separate Dagster assets running in
**separate processes** (and, for retries, at separate times). An MLflow run handle cannot cross
a process boundary, so the assets **discover and resume runs by tag** through three idempotent
get-or-create helpers in `ml_core._mlflow_runs`:

- `get_or_create_experiment(experiment_name)` — wraps `mlflow.get_experiment_by_name`.
- `get_or_create_parent_run(experiment_id)` — `search_runs` for `tags.cv_role = 'parent'`.
- `get_or_create_fold_run(experiment_id, parent_run_id, fold_id)` — `search_runs` for
  `tags.cv_role = 'fold' and tags.fold_id = '…'`; creates nested under the parent if absent.

Each asset opens the resolved run by ID, logs, and closes it within its own process. Because
lookup is by tag, re-running any fold (or re-running `metrics`) resumes the *same* run instead
of creating a duplicate — exactly the behaviour Dagster retries need.

**Concurrency note:** two parallel processes could in principle both find "no parent run" and
each create one. In practice the parent run is created once by `register_experiment_job` —
before any fold can run — so the get-or-create in the assets is only a self-healing fallback.
Fold runs never collide because each fold is a distinct Dagster partition.

The complementary decision — the resolved config is stamped onto the MLflow experiment at
registration and read back by the assets, never re-read from YAML — is explained in
[Running an experiment end-to-end](../ml_experimentation/dagster-workflow.md).

## Model artifacts: MLflow artifact store + immutable local cache

Trained models live in MLflow's artifact store, wrapped by two concrete `BaseForecaster`
methods (`save_to_mlflow` / `load_from_mlflow`) that delegate to each subclass's own disk
`save`/`load` — subclasses stay MLflow-free. `load_from_mlflow` serves from a local cache at
`{model_cache_base_path}/{run_id}/model`; the cache key is the **immutable run ID**, so a
cached model never goes stale and never needs invalidation.

Today this pair is used by the CV/experiment pipeline: `cv_power_forecasts` loads a fold's
freshly trained model back from MLflow (a separate Dagster process) via this cache. Production
inference does **not** use it for v0.1 — the champion model is baked directly into the
container image at build time and loaded via the subclass's own `load`, with no MLflow call on
the runtime path at all. Once production instead fetches its champion model from MLflow
dynamically, `load_from_mlflow`'s cache becomes the mechanism that lets it keep serving through
an MLflow outage — but that is future work, not the v0.1 design.

## Idempotent writes and concurrency

`power_forecasts` is a Delta table partitioned by `(experiment_name, fold_id)`. Re-materialising
a fold **overwrites** its partition rather than appending — an append would silently duplicate
every row on retry and double-count in metrics. Two consequences:

- **Parallel experiments are safe**: they write to disjoint `experiment_name` partition
  directories and never touch each other.
- The only residual risk is the *same* fold of the *same* experiment launched twice
  simultaneously, which delta-rs surfaces as a transaction-log conflict; Dagster's
  per-partition run concurrency normally prevents this, and a retry resolves it.

Partitioning by `experiment_name` (not `power_fcst_model_name`) is deliberate: many experiments
share one model family, and the partition must isolate experiments, not families.

## Complete validation windows only

A fold enters `conf/cv/default.yaml` **only once its validation window is complete**. A fold
with a still-open window would silently validate on partial data and corrupt the
mean-across-folds leaderboard number. Mid-window performance on an incomplete window is
obtained through the `metrics` asset's `ad_hoc` evaluation scope, which never feeds the
leaderboard.

## Fold-design alternatives considered

We considered three other ways to slice the limited honest data and chose the
[single fold](../ml_experimentation/cross-validation-folds.md#current-state-a-single-fold). These
are recorded so the decision is auditable and so we can revisit them as more data lands.

### Monthly expanding CV — rejected (redundant folds)

Slide a 12-month validation window forward by one month per fold, expanding training by one month
each time (fold 1 validates 2025-04→2026-03, fold 2 validates 2025-05→2026-04, …). This "buys"
several folds from today's data, but consecutive folds share **11/12 of the validation window** and
>90% of the training data, so their metrics are correlated ~0.9+. The effective number of
independent folds is barely more than one; the small spread across them **understates** true
sampling variance (false confidence in stability), at N× the compute for almost-duplicate
information. One month is too small a change to make the folds meaningfully different.

### Quarterly non-overlapping walk-forward — deferred (the sound multi-fold option)

Expanding training, but validate on the **next 3 months, non-overlapping**:

| Fold | Train | Validate (3 mo, non-overlapping) |
|---|---|---|
| 2025-Q2 | 2024-04-01 → 2025-03-31 | 2025-04 → 2025-06 |
| 2025-Q3 | 2024-04-01 → 2025-06-30 | 2025-07 → 2025-09 |
| 2025-Q4 | 2024-04-01 → 2025-09-30 | 2025-10 → 2025-12 |
| 2026-Q1 | 2024-04-01 → 2025-12-31 | 2026-01 → 2026-03 |

Because the validation windows do not overlap, the folds are **genuinely independent**
measurements, and the set covers all four seasons (so you see seasonal skill variation, then report
per-season and the mean). This is the statistically sound version of what monthly CV reaches for.
We deferred it to keep the initial CV setup minimal; it is the recommended next step if we want
multiple folds
*before* the ECMWF back-fill enables the full yearly protocol.

### Yearly folds backed by ERA5 — rejected for validation

Keep the yearly 2022–2025 folds now by training on **ERA5 reanalysis** for the pre-2024-04-01
years. Rejected: ERA5 is *reanalysis* (it ingests future observations), so validating on it
measures the model's response to near-perfect weather, not **forecast** skill — systematically
misleading — and a leaderboard mixing reanalysis folds and ECMWF-forecast folds is
apples-to-oranges. Reanalysis is valuable, but for **pre-training** (see
[Cross-validation folds: Target](../ml_experimentation/cross-validation-folds.md#target-multiple-yearly-folds)),
not for validation.

## Two metric stores, one division of labour

Metrics live in **both** MLflow and the `forecast_metrics` Delta table, deliberately, because
they answer different questions:

| | **MLflow** | **`forecast_metrics` Delta** |
|---|---|---|
| Granularity | Coarse: per-`time_series_type` + `"all"` aggregates | Full cube: per `time_series_id` × slice × fold × scope |
| Volume | Tens of scalars per run | Thousands of rows per experiment |
| Consumers | Leaderboard UI, sorting/grouping, auto-research | Dashboards, drill-down analysis |
| Shape | Key→value metric store | Queryable table (Polars/SQL) |

MLflow is the leaderboard; the Delta table is the analysis cube. The aggregates are
intentionally duplicated; the fine-grained per-series metrics live only in Delta.
`forecast_metrics` is internal-only — it is not one of the tables delivered to NGED.

## Rejected designs

Recorded so we don't re-litigate them:

- **Single CV asset with an internal fold loop** — no per-fold observability or retry; mixes
  orchestration with ML logic; the whole asset re-runs on any failure.
- **Dagster `DynamicOut` job instead of assets** — job run history would be the only record;
  no stable, addressable per-fold artifact in the asset catalog; poor lineage.
- **Static fold partitions shared across experiments** — two experiments would appear to
  materialise the same partition, corrupting history; static keys can't support dynamically
  created experiments (hyperparameter sweeps).
- **One YAML config file per experiment in git** — thousands of files; programmatic sweeps
  would have to commit files. MLflow is the canonical resolved-config store instead.
- **MLflow pyfunc / model registry for persistence** — significant complexity (environment
  snapshotting, custom loaders, registry states) for no benefit over the simple
  `save`/`load` + artifact wrapper. A registry-based champion-promotion flow remains an option
  later without changing the artifact format.
- **Integer MLflow experiment ID as partition-key prefix** — unreadable in the Dagster UI;
  every partition would need cross-referencing against MLflow.

## Known limitation: forecast cadence under-sampling in CV

We receive **one** NWP run per day (00 UTC) but new NGED power data every 6 hours, and the
production service will issue a forecast every 6 hours — so three of the four daily production
forecasts reuse the day's NWP with *fresher power lags*. Bulk-mode CV derives one
`power_fcst_init_time` per NWP run, so it backtests only the once-daily forecast and does not
score the 06/12/18 intraday runs — a `power_fcst_init_time − nwp_init_time` regime the model
rarely sees in training. For V1 the once-daily forecast is treated as representative; a future
refinement is for bulk mode to emit forecasts on the full 6-hourly grid per NWP run so CV
mirrors the production cadence exactly.
