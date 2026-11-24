# Plan: remove the XGBoost forecaster and the `live_forecasts` asset (issue #1722)

**The problem.** The XGBoost forecaster is the only model `live_forecasts` serves, and the
maintainer judged its forecast too poor to be useful even as a fallback. The code that exists to
train, promote, serve, check, and document that one model is spread across about 70 files: the
`xgboost_forecaster` package, the `live_forecasts` asset and its check, model promotion from
MLflow to disk and into the Docker image, the switching-event baseline scripts, and the tests that
use XGBoost as their trained forecaster.

**The planned solution.** Delete the `xgboost_forecaster` package, `conf/model/xgboost.yaml`, the
`live_forecasts` asset, its asset check, model promotion, and the baseline scripts. Move the two
NWP-run helpers that the step-5 asset of PR #1707 reuses to a neutral module. Move the tests that
use XGBoost as a stand-in trained forecaster to `ClimatologyForecaster`. Point the CV reference run
at a climatology run. The live service then writes no forecast until the step-5 asset exists (the
maintainer's decision, below). The change removes about 6,700 lines and adds a few hundred.

**Order.** This PR builds on `main` after PR #1709 merges. PR #1709 keeps its small XGBoost change
(`XGBoostForecaster.predict` reducing members with `ml_core.quantiles.to_stored_quantiles`, and the
13-column non-finite count in `live_forecasts_are_healthy`) so that `main` stays green, and this PR
deletes that change with the rest of the code. PR #1707 follows this PR.

## Verdict, size and departures

**Verdict: worth implementing.** No forecaster in the repository depends on XGBoost, and nothing
outside `live_forecasts` reads the promoted model.

**Size: complex.** The five triggers:

- **What gets stored:** fires. `live_forecasts` stops writing `fold_id="live"` rows to
  `power_forecasts`, and the production model directory is no longer written.
- **The production serving path:** fires. The serving path is removed.
- **A degradation rule:** fires. `live_forecasts_are_healthy` and the `live_forecasts` Sentry
  degradation report go.
- **More than one defensible design:** fires. What `live_forecasts` does afterwards (settled:
  nothing), and whether a later live service is one generic asset or one asset per forecaster
  (open).
- **Callers found only by searching:** fires. About 70 files name XGBoost, `live_forecasts`, or
  promotion.

**Reviews:** both plan reviews (simplicity, then correctness), and the three diff reviews of
`implement-issue`. The mutation pass is limited to the tests that move to `ClimatologyForecaster`,
because the rest of the diff is deletion.

**Departures from the issue body:** none.

## Maintainer decision: no live forecast until step 5 (2026-11-24)

The maintainer chose to remove `live_forecasts` together with XGBoost, so the live service writes
no forecast until the step-5 asset of PR #1707 exists. The step-5 network covers the 16 Primaries,
so the other 16 of the 32 trial series have no live forecast until a forecaster for them is added.

**Considered and not chosen: `live_forecasts` runs the manual heuristic.** It needs the baselines'
`NwpRunRowsWithoutWeatherFeatureEngineer` to support single-run mode (it raises
`NotImplementedError` today), `LIVE_POWER_HISTORY` raised from 15 days to at least 386 days for the
heuristic's 55-week lag, and a promoted heuristic run. It would keep a forecast for all 32 series.

## Solution

### Walk through

**The happy path: what runs afterwards.** The Dagster definitions in `definitions.py` load the
ingest assets (`ecmwf_ens`, the NGED power and metadata assets), `clean_nged_power_data`, and the
CV assets. Their schedules (`ecmwf_ens_schedule`, `power_time_series_and_metadata_schedule`) and
checks (`power_data_is_fresh`, `cleaned_power_keeps_up_with_raw`, `nwp_run_is_complete`) are
unchanged. No asset writes `fold_id="live"` rows. CV registers an experiment through
`register_experiment_job` against `conf/model/climatology.yaml` or
`conf/model/manual_heuristic.yaml`, trains through `trained_cv_model`, forecasts through
`cv_power_forecasts`, and scores through the metrics assets, all unchanged. `score_study.py`
compares a study with `conf/cv/default.yaml`'s `reference_experiment_name`, which now names a
climatology run.

**Branches.**

- **The dashboard** (`packages/dashboard/view_forecasts.py`) lists the `(experiment_name, fold_id)`
  partitions of `power_forecasts` and selects `"live"` if present, otherwise the first fold. In the
  gap it shows CV-fold forecasts only, and nothing until a CV run writes into the table that
  PR #1709 recreates.
- **The delivered table.** `power_forecasts` in the delivery bucket gets no new rows until the
  step-5 asset writes them. Nobody outside the project reads it yet, and PR #1709 deletes and
  recreates it.
- **The Docker image** keeps its role as the Dagster code-location server and run image. Its
  default `CMD` and the smoke test in `scripts/deploy/build_and_verify_image.sh` run
  `live_forecasts_job` today, so both switch to an ingest job that needs no credentials, or the
  smoke run is dropped and only the code location's import is checked. The image tag in
  `push_and_deploy_image.sh` comes from `promotion.json` today and becomes the git commit hash.

**Error handling.** Nothing new can fail. The Sentry `live-forecasts` cron monitor stops receiving
heartbeats once the new code deploys, so it is paused or deleted in Sentry by hand at deploy time,
or it alarms.

### What changes, file by file

**Deleted whole files (4,923 lines in 17 files):**

| File | Lines |
|---|---|
| `packages/xgboost_forecaster/` (README, `pyproject.toml`, `__init__.py`, `forecaster.py`, `tests/test_forecaster.py`) | 637 |
| `docs/api/xgboost_forecaster/index.md` | 5 |
| `conf/model/xgboost.yaml` | 56 |
| `scripts/forecasting/run_baseline_experiment.py` | 177 |
| `scripts/forecasting/export_baseline_forecasts.py` | 248 |
| `packages/notebooks/view_baseline_export.py` | 307 |
| `src/nged_substation_forecast/defs/promotion_assets.py` | 108 |
| `packages/ml_core/src/ml_core/production_helpers.py` | 403 |
| `tests/test_live_forecasts.py` | 725 |
| `tests/test_live_forecast_assets.py` | 59 |
| `tests/test_promoted_model.py` | 232 |
| `packages/ml_core/tests/test_production_helpers.py` | 422 |
| `docs/roadmap/xgboost-improvements.md` | 1,544 |

**Edited, with lines removed (about 1,800 lines):**

- **`defs/live_forecast_assets.py`** (333 lines): the asset, `LiveForecastsConfig`,
  `LIVE_POWER_HISTORY`, and the promoted-model load go. The file is deleted. `_parse_utc_init_time`
  and `_available_nwp_init_times` move unchanged to a new `defs/_nwp_runs.py`, because
  `checks.py::_nwp_init_times_on_disk` and PR #1707's asset call `_available_nwp_init_times`. The
  fate of `live_forecast_partitions` and `LIVE_FORECAST_HORIZON` follows open question 1.
- **`defs/checks.py`**: the `live_forecasts_are_healthy` section (about 580 lines: `_LIVE_FOLD_ID`,
  the NWP-run deadline constants, `count_missed_nwp_runs`, `evaluate_live_forecast_health`,
  `_read_live_forecast_rows`, `_read_promoted_model_facts`, the describe and metadata helpers, and
  the check) goes, with its import of `live_forecasts`.
- **`defs/schedules.py`**: `live_forecasts_job` and `live_forecasts_schedule` go (open question 1),
  and the docstrings of `power_time_series_and_metadata_schedule` stop citing them.
- **`definitions.py`**: drop `live_forecast_assets`, `promotion_assets`,
  `checks.live_forecasts_are_healthy`, and `schedules.live_forecasts_schedule`.
- **`_sentry.py`**: `NWP_CONTROL_MEMBER_MISSING_FINGERPRINT` goes. `send_forecast_checkin`,
  `LIVE_FORECAST_MONITOR_SLUG`, and `LIVE_FORECAST_MONITOR_CONFIG` stay, because PR #1707's plan
  names `send_forecast_checkin`, but their docstrings stop naming `live_forecasts` and
  `live_forecast_partitions`. `report_asset_degradation` and `sentry_capture_failure` are unchanged.
- **`ml_core/mlflow_runs.py`**: `PromotableRun` and `list_promotable_runs` go.
- **`ml_core/base_forecaster.py`**: docstrings that cite XGBoost, `live_forecasts`, or promotion.
  `load_trained_metadata` and the frozen `TimeSeriesMetadata` copy that `save` writes beside a model
  stay, because a later live asset that loads a saved forecaster needs them (open question 1).
- **`contracts/settings.py`**: `production_model_path` goes. `sentry_monitor_forecasts` stays with
  `send_forecast_checkin`.
- **`contracts/config_schemas.py`**, **`ml_core/features/_lags.py`**, **`_nwp.py`**,
  **`tabular_feature_engineer.py`**, **`features/__init__.py`**,
  **`baseline_forecasters/manual_heuristic.py`**, **`nwp_run_rows.py`**: comments and docstrings
  that name XGBoost, `live_forecasts`, or `select_nwp_init_time`.
- **`conf/cv/default.yaml`**: `reference_experiment_name` names a climatology run (open
  question 2). **`conf/model/climatology.yaml`**, **`manual_heuristic.yaml`**: their header comments
  stop citing `xgboost.yaml`.
- **Root `pyproject.toml`** and **`uv.lock`**: the `xgboost_forecaster` workspace member and the
  `xgboost` dependency. **`mkdocs.yml`**: the two nav entries.
- **`Dockerfile`**: the `production_model` build context, the `MODEL_RUN_ID` argument and label,
  `PRODUCTION_MODEL_PATH`, and the default `CMD`. **`scripts/deploy/*.sh`**: the `promotion.json`
  tag and the `live_forecasts_job` smoke run.
- **`.gitignore`**, **`README.md`**, **`CLAUDE.md`**, **`scripts/README.md`**,
  **`packages/notebooks/README.md`**, the package READMEs, and the skills (`release-model`,
  `update-dependencies`, `plan-issue`): every passage describing XGBoost, `live_forecasts`, or
  promotion as current. The `release-model` skill describes promotion end to end, so it is cut to
  what still applies or deleted.

**Tests moved to `ClimatologyForecaster`:** `tests/conftest.py` (the CV-registration fixture's
default `base_model_config` and overrides), `test_jobs.py`, `test_trained_cv_model.py`,
`test_register_experiment_job.py`, `test_forecaster_config_serialisation.py`,
`packages/ml_core/tests/test_load_experiment_forecaster.py`, and the `"xgboost"` model-name strings
in `test_s3_data_paths.py`, `contracts/tests/test_metrics.py`, and the `delta_store` tests. The live
tests in `tests/test_checks.py` (about 780 lines) and the check-in tests in `tests/test_sentry.py`
that drive `live_forecasts` go.

### Design-philosophy check

The serving path is removed rather than changed, so no degradation rule is weakened in code that
still runs. The gap is the cost: until PR #1707 lands, no live forecast reaches NGED and Sentry has
no live-slot check. The change follows the principle that code no caller uses goes, rather than
staying for a possible future caller. The two exceptions, `send_forecast_checkin` and
`load_trained_metadata`, stay because the next PR's plan names the first and open question 1
decides the second.

## Tests

No test is added: every behaviour this PR leaves is already tested. The tests that move to
`ClimatologyForecaster` keep their assertions on the CV, MLflow, and config machinery. Each has to
fail on a broken machinery path, not on XGBoost, so the mutation pass checks those moved tests
only. Two tests assert the absence of what goes:

- `tests/test_asset_selection_parses.py` (extended): the definitions load, and no asset, job, or
  schedule named `live_forecasts` or `promoted_model` is registered. This fails on `main`.
- A `git grep -il xgboost` guard is not added, because the CLAUDE.md prose rules use "XGBoost
  model" as an example and some docs keep XGBoost as history in research references.

The CV integration tests lose their only coverage of a weather feature (`temperature_2m`) through
CV, because climatology selects none. The weather path keeps its unit tests in
`ml_core/tests/test_features.py`.

## Docs to update

Every page that `git grep -il "xgboost\|live_forecasts\|promoted_model\|production_model"` finds
under `docs/` (about 40 pages), rewritten to describe the code as it now is. The heaviest are
`docs/live_service/aws.md` ("Redeploying a new champion model"), `model-releases.md`,
`operations.md`, `setup.md`, `docs/architecture/production-deployment.md`, `overview.md`,
`testing.md`, `docs/cross-validation/model-configuration.md`, `dagster-workflow.md`, and the
roadmap pages that describe XGBoost as the current model. `docs/roadmap/index.md` links to the
deleted XGBoost roadmap page, so that link goes too.

## What PR #1707's plan then has to add back

PR #1707's plan assumes `live_forecasts` keeps running as the fallback. After this PR:

- **No fallback.** When today's 00Z run is absent at 15:00 UTC, the step-5 asset writes nothing and
  NGED gets no forecast that day. The plan's error-handling text changes to say so.
- **The live asset checks**, re-added for `live_primaries_network_forecasts` if wanted: the slot
  wrote rows, every expected series is present, the horizon is at least the minimum fraction, no
  quantile column is non-finite, no row is a hindcast, and the count of missed NWP runs.
- **The Sentry check-in monitor.** PR #1707's plan relies on the `live-forecasts` monitor to catch
  a dead daemon. `send_forecast_checkin` and its monitor config stay, so the step-5 asset can call
  `send_forecast_checkin` with its own `monitor_slug` and a crontab matching its daily schedule.
- **The Docker image.** PR #1707's plan copies the bundle "beside the existing champion model". The
  bundle becomes the image's only model, and the image's default `CMD` and smoke test run the
  step-5 job.
- **`_available_nwp_init_times`** is imported from `defs/_nwp_runs.py`, not
  `defs/live_forecast_assets.py`.
- Unchanged and reused as named: `write_power_forecasts` with `replace_partition`,
  `ecmwf_ens_partitions`, `ecmwf_ens_schedule`, `_is_too_old_to_retry`, `report_asset_degradation`,
  and `sentry_capture_failure`.

## Verification commands

The `implement-issue` green-before-push set (`uv run ruff check .`, `uv run ruff format --check .`,
`uv run ty check`, `uv run pytest`, the markdown lint), plus `uv run mkdocs build --strict` for the
removed pages and links, `uv lock --check`, and a local `dagster definitions validate -m
nged_substation_forecast.definitions`.

## Risks and open questions

1. **One generic live asset, or one live asset per forecaster?** The maintainer is deciding.
    - *One generic asset:* keep `live_forecast_partitions`, `LIVE_FORECAST_HORIZON`,
      `select_nwp_init_time` and its live and replay modes, `load_trained_metadata`, and the
      replace-partition write pattern in a neutral module, and keep the 6-hourly cadence for a
      later asset that loads any saved forecaster and calls `predict`. PR #1707's plan would then
      plug the step-5 network in behind that asset rather than adding its own.
    - *One asset per forecaster:* delete `live_forecast_partitions`, `live_forecasts_job`,
      `live_forecasts_schedule`, and `select_nwp_init_time` now. Each forecaster asset owns its
      partitions and schedule, as PR #1707's plan does with its daily 12:00 UTC asset, and shares
      only the helpers listed above.

   No recommendation until the maintainer answers. Either way the `live_forecasts` asset itself is
   deleted, because it has no forecaster left to serve.
2. **The CV reference run.** `reference_experiment_name` moves from `xgboost_no_power_lags` to a
   climatology run. Recommendation: climatology, because it trains in seconds and needs no weather.
   The run has to be re-run into the `power_forecasts` table that PR #1709 recreates before
   `score_study.py` accepts a study.
3. **The Sentry monitor at deploy.** Pause or delete the `live-forecasts` cron monitor by hand when
   this PR deploys. Recommendation: pause it, and let PR #1707 resume it under its own slug.
4. **Follow-up, not in this PR: the weather features.** With XGBoost gone, no production
   forecaster selects a weather feature from `TabularFeatureEngineer`, and the step-5 asset builds
   its own inputs. The weather columns of `AllFeatures`, `windchill` (issue #1199), and the weather
   path of the tabular engineer could go in a follow-up issue. That issue changes a Patito
   contract, so it is not folded in here.
