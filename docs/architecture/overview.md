# Architecture Overview

The architecture prioritises developer velocity, idempotent re-runs, and strict **Training-Serving Symmetry**.

The primary aim is to develop novel, ambitious, state-of-the-art ML approaches to forecasting. We are simultaneously building a "test-harness" production service so that ML research runs in a production-like environment from day one.

The aim is to manage the *entire* data pipeline in Dagster: download data, validate data, train ML models, run inference, perform backtests. MLflow tracks every experiment. Re-running a backtest should be as easy as clicking a button in Dagster. Swapping a new model into production should require minimal friction.

The system is designed as a modular monorepo using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), with [Dagster](https://dagster.io/) orchestrating the data pipeline and [MLflow](https://mlflow.org/) tracking experiments.

## Core Components

* **Environment & Modularity**: `uv` workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing**: **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
    * **Centralized Data Preparation**: All data entering ML models passes through a centralized preparation step to enforce strict data contracts, handle missing entities, and ensure consistency between training and inference.
* **Storage**: **Delta Lake** on cloud object storage for both power data and NWP weather data. Delta Lake provides ACID transactions, time-travel, and efficient partitioning. NWP data is quantised to 12-bit precision and stored as `Int16` (roughly halving the size versus `float32`), with `zstd` level-14 compression — compressing the full ECMWF ENS dataset for Great Britain to approximately 40 GB per year. A single day's NWP data takes about one minute to download and convert.
* **Orchestration**: **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally across all substations (if they want to). Dagster is responsible for detecting bad data.
* **Configuration Management**: **Hydra** combined with `pydantic-settings`. Dagster passes string overrides (e.g., model=xgboost_global) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
* **Experiment Tracking**: **MLflow**.
* **Visualisation**: Altair for plotting, Marimo for interactive data exploration and web apps.

## Lazy Evaluation Strategy

**Rule: everything stays lazy until the model boundary.**

The entire feature-engineering pipeline operates on Polars `LazyFrame`s. No code between reading from storage and calling `BaseForecaster.train()` or `predict()` is allowed to call `.collect()`. The pipeline is:

```
NwpOnDisk.scan_delta(path)               # lazy scan
  → NwpOnDisk.to_nwp_in_memory()        # lazy expression; no collect
  → FeatureEngineer.engineer()           # spatial join + feature pipeline;
                                         #   returns pt.LazyFrame[AllFeatures]; no collect
  → BaseForecaster.train/predict()       # subclass materialises the data here, at the boundary
```

**Why it matters:**

- **Memory**: Polars only materialises the data at the last moment, at the model boundary. Intermediate representations (individual join outputs, lag columns before filtering) are never written to RAM.
- **Query optimisation**: Polars can see the full plan end-to-end and push down filters (e.g. date ranges, `time_series_id` subsets) into the Delta Lake scan before any data crosses the wire.
- **Clean boundaries**: The materialisation site is the model boundary — the one place where a third-party library (XGBoost, PyTorch, …) needs a concrete in-memory array. Keeping it there makes the boundary explicit and easy to find.

**The one exception** is a `limit(1).collect()` guard in `_build_historical_weather` (inside `ml_core`) that cheaply checks for the NWP control member before building the lazy plan, so the pipeline fails loudly instead of silently returning an empty frame.

**Contract for `BaseForecaster` subclasses**: the lazy plan is materialised *at the model boundary*, as late as possible, and never by callers — never ask a caller to collect before passing data in. A subclass typically does a single `.collect()`; keeping that bounded is the **caller's** job, by pruning the *inputs* before feature engineering (not by filtering the engineered output — see below).

### Bounding feature-engineering memory: prune the inputs, not the output

The feature-engineering plan is dominated by the **NWP scan**. The NWP Delta is large — for the V1 trial it is **~83 GB**: 767 daily `init_time` partitions, each ~7.24M rows = **1671 H3 cells × 51 ensemble members × 85 native steps** (control member alone is 142k rows/partition). The 30-min upsample and the multi-run bulk join inflate that further. So the whole memory question is: *how little of that NWP do we touch?*

**You cannot prune the scan by filtering the engineered output.** `data.filter(time_series_id == x)` runs the cell-attach join, the 30-min upsample (`group_by` + `explode` + `sort` + `interpolate`) and the bulk join *first*, then drops rows. And NWP is keyed by `h3_index`, not `time_series_id`, so a `time_series_id` predicate can never reach the NWP scan at all. Pruning must be applied to the **raw inputs**, in `_load_engineering_inputs`, *before* `to_nwp_in_memory` rescales Int16→Float32.

What actually prunes the NWP scan — verified with `LazyFrame.explain()`:

| Predicate on the **raw NWP scan** | Effect |
|---|---|
| `init_time ∈ [start − 16d, end]` | **Partition prune** — NWP is partitioned by `init_time`, so only those partition directories are opened. A `valid_time` filter *alone* does **not** prune partitions (it scanned ~887 files). |
| `ensemble_member ∈ {…}` **before** the rescale | Only the requested members are decoded/rescaled (≈50× less for control-member training). Applied *after* the rescale (the old bug), all 51 members are materialised first → OOM. |
| `h3_index ∈ {cells}` | Restricts to the cells the requested series sit in. |
| `time_series_id == x` on the **output** | Prunes the power scan + metadata join, but **not** the NWP scan (no `time_series_id` there), and only after the upsample. Doesn't help. |
| `slice(n)` / `head(n)` (row count) on the output | **Nothing** — a row slice can't push through a join; the whole join runs first. |

**Pruning the partitions is necessary but not sufficient — you must also stream the collect.** `init_time` prunes whole partition *files*, but `ensemble_member` and `h3_index` are not partition keys, so they only filter at the *row* level. A single NWP parquet row group holds **every member × every cell** (the data is sorted `init → valid → member → h3`, so each row group spans all 1671 cells and 51 members), so their min/max stats can't skip groups for our 1 member / 9 cells. The **default (in-memory) engine therefore decodes every row in the pruned partitions before dropping 50/51 of the members and 1662/1671 of the cells** — measured at **7.2 GB to materialise a 611-row, one-cell, one-week, control-member slice**. The **streaming engine** (`collect(engine="streaming")`) applies the predicates per morsel, so peak memory is set by the morsel/row-group size, *not* the window length: the same slice peaks at ~2.4 GB, and a *full 15-month* control-member collect stays at ~2.5 GB. `XGBoostForecaster.train`/`predict` therefore collect with `engine="streaming"`.

**Many-to-one `h3_index` ↔ `time_series_id`:** one NWP cell covers several series (the 32 V1 series live in just **9 cells**, one holding 12), so `h3_index` pruning is keyed on the (few) cells the requested series occupy, and the feature engineer's spatial join replicates each cell's weather across its series.

**Resulting design** (`_load_engineering_inputs` applies all three NWP predicates — `init_time`, `ensemble_member`, and `h3_index` = the requested series' cells — to the raw scan, and every collect streams):

| | control member (train) | full 51-member ensemble (predict) |
|---|---|---|
| All eligible series at once | ~5 GB → `train` collects once, groups in memory | ~25 GB ✘ (OOMs) |
| One `init_time` chunk at a time | — | ~9 GB → `cv_power_forecasts` chunks by `init_time`, appends to Delta |

Prediction is bounded by chunking on **`init_time`** (`_PREDICT_INIT_CHUNK`, 14 days), *not* by cell. `init_time` is both the partition key and the axis that fans the output out across runs, so a chunk's forecast frame stays small while each partition is read exactly once and all series/cells/members are processed together (a per-*cell* loop instead OOMs on the busiest cell — 10 series × 51 members × the 10-month window ≈ 116M rows ≈ 25 GB). Measured end-to-end on the `mid_2025_to_mid_2026` fold: training peaks ~5 GB and the full 51-member validation prediction (~321M forecast rows) peaks ~9 GB — both well under a 24 GB laptop. The unused `xgboost_forecaster._data_iter.LazyFrameBatchIter` is kept for a future streaming path — fed a predicate-filtered frame or a parquet/Delta scan, never row slices of a join.

## The Universal Model Interface

All forecasting models subclass `BaseForecaster` (defined in `ml_core`), which provides a common `train` / `predict` / `save` / `load` interface. The model wrapper encapsulates the model weights and all translation logic, keeping Dagster assets completely agnostic to the underlying implementation. Each subclass of `BaseForecaster` is responsible for defining:

- _Feature engineering_: Each subclass carries a `feature_engineer: ClassVar[FeatureEngineer]` strategy (composition, not inheritance) that owns the full preparation pipeline — from raw inputs (observed power, gridded NWP, time-series metadata) to an `AllFeatures` frame. The default `TabularFeatureEngineer` does the nearest-cell NWP spatial join then runs the tabular feature pipeline. A future model that needs a different data view (e.g. a CNN wanting a spatial NWP crop per time series) overrides `feature_engineer` with a different `FeatureEngineer` subclass without touching `BaseForecaster` or any other model. `FeatureEngineer` and `TabularFeatureEngineer` live in `packages/ml_core/src/ml_core/features/`.
- _Input translation_: Transforms the canonical `AllFeatures` Polars LazyFrame into the required model shape.
- _Output translation_: Converts native model outputs into the strict `PowerForecast` schema.
- _Persistence_: Each subclass owns its own save/load format. `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a `meta.json` containing the full serialised `XGBoostConfig`. (This may change later. We may switch to saving models using native MLflow flavors (e.g., `mlflow.xgboost.log_model`), which serialize the raw model object directly.)
- _Identity_: Model name, version, and optional MLflow experiment ID travel with the config, so every `PowerForecast` row is self-describing.
