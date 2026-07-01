# Baseline forecasters: persistence and climatology on the leaderboard

## Finding

No naive baseline exists anywhere in the codebase (only docstring mentions, e.g.
`contracts/power_schemas.py:242`). Without a persistence row and a climatology row on the
leaderboard, XGBoost's NMAE numbers aren't interpretable: at 0–6 h persistence is famously hard
to beat, and at day 8–14 seasonal climatology often beats everything. A model could "win" the
leaderboard while adding no skill over naive methods and nobody would know.

Side benefit: a second and third `BaseForecaster` implementation pressure-tests the abstraction
(the docs promise the interface is model-agnostic; today only `XGBoostForecaster` exercises it).

## Implementation

New workspace package `packages/baseline_forecasters/` mirroring the `xgboost_forecaster`
layout (`pyproject.toml`, `src/baseline_forecasters/`, `tests/`). Add to the root
`pyproject.toml` `[tool.uv.sources]` and dependencies.

### 1. `PersistenceForecaster` (seasonal-naive)

Reuse the existing lag machinery instead of writing any new time-series logic:

- `selected_features` = `{"power_lag_24h", "power_lag_48h", "power_lag_168h",
  "power_lag_336h"}` (configurable in the config, this is the default).
- `predict()` = `pl.coalesce` of the lag columns in ascending-lag order. The existing
  `_nullify_leaky_lags` already nulls any lag ≤ lead time, so coalesce naturally selects the
  *shortest non-leaky* lag per row — same-time-yesterday for day-1 horizons, last-week for
  day 2–7, two-weeks-ago beyond that. Zero lookahead risk because it rides the audited pipeline.
- `train()` records `trained_time_series_ids` only (nothing to fit). `save`/`load` persist a
  `meta.json` with the config + ids (copy the pattern from `XGBoostForecaster`).
- `MODEL_NAME = "persistence"`, `MODEL_VERSION = 1`.
- Rows where all lags are null (long horizons + data gaps): drop those rows from the output
  (`PowerForecast.power_fcst` presumably non-nullable — check) and log the dropped count.

### 2. `ClimatologyForecaster`

- `train()`: collect `(time_series_id, valid_time, power)` from the features frame and build a
  lookup table of mean power per `(time_series_id, month, half-hour-of-day, is_weekend)` over
  the training window. Store as a small Polars frame; `save` writes it as one parquet +
  `meta.json`.
- `predict()`: join the lookup onto the prediction rows by the same keys.
- `selected_features` needs only the target join — reuse existing time features
  (`local_time_of_day_*` etc.) or derive month/half-hour directly from `valid_time` inside the
  forecaster; prefer deriving directly to keep the feature list empty.
- `MODEL_NAME = "climatology"`, `MODEL_VERSION = 1`.

### 3. Configs and registration

- `conf/model/persistence.yaml` and `conf/model/climatology.yaml` following
  `conf/model/xgboost.yaml`'s `_target_` structure, with `weather_source: "none"`.
- `PowerForecast.nwp_init_time` is already nullable for exactly this case
  (`power_schemas.py:242`); confirm `cv_power_forecasts` tolerates it end-to-end.
- **Ensemble members:** the CV predict path feeds all ~51 members; baselines produce identical
  forecasts per member. That is wasteful but harmless (the ensemble mean is unchanged). MVP:
  accept it. Do not special-case the pipeline for baselines in this PR; if the waste matters
  later, add a per-forecaster `uses_nwp_ensemble` class flag and let `cv_power_forecasts`
  restrict to member 0.

### 4. Run and publish

Register both via `register_experiment_job` (`run_mode: smoke_test` first, then `full_cv`) and
materialise `trained_cv_model` → `cv_power_forecasts` → `metrics` so both appear in MLflow as
leaderboard rows.

## Verification

1. Unit tests in `packages/baseline_forecasters/tests/`: persistence picks the shortest
   non-null lag; climatology lookup round-trips through `save`/`load`; both freeze
   `trained_time_series_ids`.
2. Smoke-test fold end-to-end via the existing integration-test pattern
   (`tests/test_trained_cv_model.py` fixtures).
3. Sanity-check the numbers: persistence NMAE should be *worse overall* than XGBoost but
   plausibly competitive at short horizons (fully visible once plan 03's horizon slices land —
   note the two plans compound).
