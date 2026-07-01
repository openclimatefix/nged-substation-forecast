# `live_forecasts` asset (production inference)

*Merged from the Dagster ML-assets plan, Phase 7 — the first of its three remaining phases
(see also plans 10 and 11). Phases 0–6.7 of that plan are complete (PRs #182–#214).*

## Context

Everything up to the CV leaderboard loop is built; the production inference path is not. This
asset is what the deployed container (plan `05_production_model_artifacts.md`) will actually
run every 6 hours: load the production model, forecast the latest NWP, write to
`power_forecasts` with `fold_id="live"`.

## Design

```text
partitions_def: TimeWindowPartitionsDefinition(cron="0 0,6,12,18 * * *", ...)
deps: [ecmwf_ens, power_time_series_and_metadata]
```

- **Model loading:** `ForecasterClass.load_from_mlflow(settings.production_model_run_id,
  settings.model_cache_base_path)` — served from the local cache, so the live service keeps
  running if MLflow is down (only a cache miss contacts MLflow). The concrete class is
  reconstructed from the `model_class` field in the saved model's `meta.json`. Both `Settings`
  fields already exist.
- **Population:** forecast **only** the `trained_time_series_ids` recorded in the production
  model's `meta.json` — never a series the model has not seen (the train==predict invariant).
- **Inference mode:** **single-run** feature engineering
  (`engineer_features(power_fcst_init_time=t0, …)`) across **all 51 NWP ensemble members**,
  where `t0` is the partition's scheduled time. Deliberately *not* bulk mode: production
  forecasts one explicit `power_fcst_init_time`, not one-per-NWP-run.
- **NWP availability semantics** via a `RunConfig` field
  `availability_mode: Literal["live", "replay"]` (default `"live"`):
  - `"live"` — the scheduled path. `t0 = now`; join the **freshest NWP run actually present**
    in Delta with `nwp_init_time <= t0`. **No** modelled publication delay: reality already
    constrains the table to genuinely published runs, so if the provider speeds up we
    automatically use fresher data.
  - `"replay"` — re-running a *past* slot (e.g. yesterday's failed run, today).
    `t0 = the historical partition time`; join the freshest run with
    `nwp_init_time <= t0 − nwp_publication_delay_hours`. The delay reconstructs what was
    actually *available* at the historical `t0` — without it we would leak NWP runs that only
    landed afterwards.
  - The scheduled sensor/schedule always uses `"live"` for the current partition; manual
    backfills of past partitions use `"replay"`. The explicit flag is the source of truth (an
    automatic live-iff-recent rule is a possible later convenience).
- Load the needed NWP via `TimeWindowPartitionMapping` against the daily-partitioned
  `ecmwf_ens`.
- **Write:** `forecaster.predict(..., fold_id="live")`, then an **idempotent overwrite** of
  this run's `power_fcst_init_time` rows in `power_forecasts` (`replaceWhere`), so re-running a
  6-hourly partition (or a replay) never duplicates rows.
- **Logs nothing to MLflow.** A 6-hourly forecast run is not an experiment; live performance
  is tracked by the monitoring scope (plan 10), never by this asset.

Implementation notes:

- Keep the asset a thin shell over pure helpers (repo convention): the freshest-NWP-run
  selection given `(t0, availability_mode, delay)` belongs in `ml_core._cv_helpers` (or a
  sibling), unit-tested with an injected clock — `now` is passed in, never read inside the
  asset.
- Reuse the identity-stamping tail shared with `cv_power_forecasts` (predict → stamp →
  validate) rather than duplicating it; extract a shared helper if one doesn't already exist.
- File placement: `defs/cv_assets.py` is already 898 lines — put this in a new
  `defs/production_assets.py` (the split plan 11 formalises).

## Tests

- `live` vs `replay` select **different NWP runs** for the same `t0` when a fresher run exists
  (replay applies `nwp_publication_delay_hours`; live does not).
- Only `trained_time_series_ids` are forecast.
- All 51 ensemble members present in the output.
- Idempotency: materialise the same partition twice → row count unchanged.
- Injected-clock unit tests for the run-selection helper.

## Verification

Set `production_model_run_id` to a smoke-test model's fold run, materialise `live_forecasts`
for the current partition, and confirm `fold_id="live"` rows in `power_forecasts` — then plot
them via `plot_power_forecast_job` (`fold_id: "live"`).
