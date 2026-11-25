# Plan: remove the XGBoost forecaster and the `live_forecasts` asset (issue #1722)

**The problem.** The XGBoost forecaster is the only model `live_forecasts` serves, and the
maintainer judged its forecast too poor to be useful even as a fallback. The code that exists to
train, promote, serve, check, and document that one model is spread across about 120 files: the
`xgboost_forecaster` package, the `live_forecasts` asset and its check, model promotion from
MLflow to disk and into the Docker image, the switching-event baseline scripts, and the tests that
use XGBoost as their trained forecaster.

**The planned solution.** Delete the `xgboost_forecaster` package, `conf/model/xgboost.yaml`, the
`live_forecasts` asset, its asset check, model promotion, and the baseline scripts. Move
`_available_nwp_init_times`, which the step-5 asset of PR #1707 reuses, and its parsing helper
into the neutral module `defs/assets.py`, beside `ecmwf_ens`, the asset that writes the table they
read. Move the tests that
use XGBoost as a stand-in trained forecaster to `ClimatologyForecaster`. Point the CV reference run
at a climatology run. The live service then writes no forecast until the step-5 asset
`live_primary_substation_forecasts` exists, and each later time-series type gets a live asset of
its own (the maintainer's decisions, below). The change removes about 7,700 lines and adds a few hundred.

**Order.** This PR builds on `main`, which already holds the merged PR #1709. PR #1709 kept a small
XGBoost change so that `main` stayed green: `XGBoostForecaster.predict` reduces members with
`ml_core.quantiles.to_stored_quantiles`, and `live_forecasts_are_healthy` counts non-finite values
across the 13 quantile columns. This PR deletes that change with the rest of the code. PR #1707
follows this PR.

**`to_stored_quantiles` stays.** The manual heuristic and two metrics tests call
`ml_core.quantiles.to_stored_quantiles`, so the function and `test_quantiles.py` stay. Only the
XGBoost caller goes.

**Deployment.** Once this PR merges, `main` writes no live forecast, so do not deploy `main` to AWS
until the step-5 PR (PR #1707) has merged too.

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
  nothing), and whether a later live service is one generic asset or one asset per time-series
  type (settled: one asset per time-series type).
- **Callers found only by searching:** fires. About 120 files name XGBoost, `live_forecasts`,
  promotion, or the production model.

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

## Maintainer decision: one live asset per time-series type (2026-11-24)

The maintainer chose one live Dagster asset per time-series type over one generic live asset that
loads any saved forecaster. So:

- `live_forecasts` is deleted, with `live_forecast_partitions`, `LIVE_FORECAST_HORIZON`,
  `live_forecasts_job`, `live_forecasts_schedule`, and `select_nwp_init_time`. Each live asset owns
  its own partitions and schedule.
- The helpers a live asset reuses stay in neutral modules: `_available_nwp_init_times` and
  `_parse_utc_init_time` in `defs/assets.py`, and `write_power_forecasts`, `ecmwf_ens_partitions`,
  `ecmwf_ens_schedule`, `_is_too_old_to_retry`, `report_asset_degradation`, and
  `sentry_capture_failure` where they are today.
- The step-5 asset of PR #1707 is named `live_primary_substation_forecasts`.

**Context for that choice.** Colleagues have asked for the wind and solar power forecasts to be
refactored into modules of their own, so that they can be reused outside the NGED project. One live
asset per time-series type, with the shared helpers in neutral modules, fits that refactor. The
refactor itself belongs to a later issue and is not planned here.

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
  `live_forecasts_job` today. The default `CMD` is deleted, and the smoke test runs
  `definitions validate -m nged_substation_forecast.definitions` in the image with
  `--network=none`, because no inference is left whose hermeticity the old smoke run proved. The
  image tag in `push_and_deploy_image.sh` comes from `promotion.json` today and becomes the git
  commit hash.

**Error handling.** Nothing new can fail. The Sentry `live-forecasts` cron monitor stops receiving
heartbeats once the new code deploys, so it is deleted in Sentry by hand at deploy time, or it
alarms.

### What changes, file by file

**Deleted whole files (5,464 lines in 20 files):**

| File | Lines |
|---|---|
| `packages/xgboost_forecaster/` (README, `pyproject.toml`, `__init__.py`, `forecaster.py`, `tests/test_forecaster.py`) | 699 |
| `docs/api/xgboost_forecaster/index.md` | 5 |
| `conf/model/xgboost.yaml` | 56 |
| `scripts/forecasting/run_baseline_experiment.py` | 177 |
| `scripts/forecasting/export_baseline_forecasts.py` | 209 |
| `packages/notebooks/view_baseline_export.py` | 244 |
| `src/nged_substation_forecast/defs/live_forecast_assets.py` (after moving two helpers out) | 333 |
| `src/nged_substation_forecast/defs/promotion_assets.py` | 108 |
| `packages/ml_core/src/ml_core/production_helpers.py` | 403 |
| `tests/test_live_forecasts.py` | 747 |
| `tests/test_live_forecast_assets.py` | 59 |
| `tests/test_promoted_model.py` | 232 |
| `packages/ml_core/tests/test_production_helpers.py` | 422 |
| `docs/roadmap/xgboost-improvements.md` | 1,544 |
| `docs/live_service/model-releases.md` | 46 |
| `.claude/skills/release-model/SKILL.md` | 180 |

**Edited, with lines removed (about 2,200 lines):**

- **`defs/assets.py`**: gains `_parse_utc_init_time` and `_available_nwp_init_times`, moved
  unchanged from `defs/live_forecast_assets.py` and placed beside `ecmwf_ens`. After this PR, the
  only caller is PR #1707's asset. Their test,
  `test_available_nwp_init_times_parses_both_fractional_second_renderings`, moves from
  `tests/test_live_forecast_assets.py` into `tests/test_assets.py`. The rest of
  `live_forecast_assets.py` goes: the asset, `LiveForecastsConfig`, `LIVE_POWER_HISTORY`,
  `LIVE_FORECAST_HORIZON`, `live_forecast_partitions`, and the promoted-model load.
- **`defs/checks.py`**: the `live_forecasts_are_healthy` section (about 580 lines: `_LIVE_FOLD_ID`,
  the NWP-run deadline constants, `count_missed_nwp_runs`, `evaluate_live_forecast_health`,
  `_read_live_forecast_rows`, `_read_promoted_model_facts`, the describe and metadata helpers,
  `_nwp_init_times_on_disk`, and the check) goes, with its import of `live_forecasts`.
- **`defs/schedules.py`**: `live_forecasts_job` and `live_forecasts_schedule` go,
  and the docstrings of `power_time_series_and_metadata_schedule` stop citing them.
- **`definitions.py`**: drop `live_forecast_assets`, `promotion_assets`,
  `checks.live_forecasts_are_healthy`, and `schedules.live_forecasts_schedule`.
- **`_sentry.py`**: `NWP_CONTROL_MEMBER_MISSING_FINGERPRINT`, `send_forecast_checkin`,
  `LIVE_FORECAST_MONITOR_SLUG`, and `LIVE_FORECAST_MONITOR_CONFIG` go. Their only caller is
  `live_forecasts`, and PR #1707's plan adds no check-in monitor. `report_asset_degradation` and
  `sentry_capture_failure` are unchanged.
- **`ml_core/mlflow_runs.py`**: `PromotableRun` and `list_promotable_runs` go.
- **`ml_core/base_forecaster.py`**: docstrings that cite XGBoost, `live_forecasts`, or promotion.
  `load_trained_metadata` and the frozen `TimeSeriesMetadata` copy that `save` writes beside a model
  stay, because a live asset for another time-series type that loads a saved forecaster needs
  them. PR #1707's asset does not use them: it reads its Primaries from its bundle's
  `primaries.json`. `fetch_model_artifacts` and `load_forecaster_from_dir` go with
  `production_helpers.py`. The `load_trained_metadata` tests in `test_base_forecaster.py` and
  `tests/test_trained_cv_model.py` read the directory `save` writes instead of an artifact that
  `fetch_model_artifacts` fetched, and the `fetch_model_artifacts` tests go.
- **`contracts/settings.py`**: `production_model_path` and `sentry_monitor_forecasts` go, with
  their tests in `contracts/tests/test_settings.py`. `Settings` forbids unknown fields from `.env`,
  so the two lines go from `.env.example` and from every deployed or workstation `.env` before the
  new code starts.
- **`contracts/config_schemas.py`**, **`ml_core/features/_lags.py`**, **`_nwp.py`**,
  **`tabular_feature_engineer.py`**, **`features/__init__.py`**, **`defs/_engineering_inputs.py`**,
  **`cleaning_assets.py`**, **`cv_assets.py`**, **`jobs.py`**, **`delta_store/power_forecasts.py`**,
  **`baseline_forecasters/manual_heuristic.py`**, **`nwp_run_rows.py`**: comments and docstrings
  that name XGBoost, `live_forecasts`, or `select_nwp_init_time`. The `_nwp.py` passage on
  `select_nwp_init_time`'s replay mode is rewritten around `NWP_PUBLICATION_DELAY_HOURS`, which
  survives, rather than renamed.
- **`conf/cv/default.yaml`**: `reference_experiment_name` names a climatology run (open
  question 1). **`conf/model/climatology.yaml`**, **`manual_heuristic.yaml`**: their header comments
  stop citing `xgboost.yaml`.
- **Root `pyproject.toml`**: the `xgboost_forecaster` dependency and its workspace source.
  **`uv.lock`**: the `xgboost_forecaster` and `xgboost` packages. **`mkdocs.yml`**: the two nav
  entries. **`docs/api/ml_core/index.md`**: the `::: ml_core.production_helpers` line, which otherwise
  breaks `mkdocs build --strict`.
- **`Dockerfile`**: the `production_model` build context, the `MODEL_RUN_ID` argument and label,
  `PRODUCTION_MODEL_PATH`, and the default `CMD`. **`.dockerignore`**: the comment on the
  `production_model` context. **`scripts/deploy/*.sh`**: the `promotion.json` tag becomes the git
  commit hash, and the smoke run becomes `definitions validate`.
- **`.gitignore`**, **`README.md`**, **`CLAUDE.md`**, **`scripts/README.md`**,
  **`packages/notebooks/README.md`**, the package READMEs (`packages/contracts/README.md` and
  `packages/ml_core/README.md`), and the skills (`update-dependencies`,
  `plan-issue`, `plan-wave`): every passage describing XGBoost, `live_forecasts`, or promotion as
  current. The `release-model` skill and `model-releases.md` describe only promotion, so both are
  deleted rather than rewritten. `docs/example_power_forecast.svg` charts the XGBoost
  live forecast on the docs home page, so the chart and its `docs/index.md` link are deleted until a
  step-5 forecast can replace them.

**Tests moved to `ClimatologyForecaster`.** `ClimatologyForecaster` raises on any non-empty
`selected_features`, and its config has no hyperparameters beyond `random_seed` and
`model_release`, so the move is not a rename:

- `tests/conftest.py`: the CV-registration fixture's default `base_model_config` becomes `{}`, in
  place of `["temperature_2m"]` and `n_estimators: 5`.
- `test_jobs.py` and `test_register_experiment_job.py`: the overrides of `n_estimators` and
  `max_depth` become overrides of `random_seed`, and each test checks that the override still
  changes the experiment's identity.
- `test_trained_cv_model.py` and `tests/test_cv_power_forecasts.py`: the two tests of the
  `power_lag_24h` look-back wiring use `conf/model/manual_heuristic.yaml`, which selects power lags,
  because climatology selects none.
- `tests/test_cv_power_forecasts.py` and `tests/test_metrics.py`: assertions that depend on
  XGBoost's output change. Climatology returns one row per primary key, carrying its cell's 13
  quantile columns, and drops rows in cells with no training samples.
- `packages/ml_core/tests/test_load_experiment_forecaster.py`, and the `"xgboost"` model-name
  strings in `test_s3_data_paths.py`, `contracts/tests/test_metrics.py`, and the `delta_store`
  tests `test_forecast_metrics.py`, `test_power_forecasts.py`, and `test_nwp.py`.

`test_forecaster_config_serialisation.py` is deleted rather than moved, because `XGBoostConfig` is
the only `BaseForecasterConfig` subclass it can check. `tests/test_asset_layer_tags.py` drops
`live_forecasts` from its production assets and its docstring drops the promotion assets. These
go: the live tests in `tests/test_checks.py` (about 970 lines, from line 801), the check-in tests
and the monitor-schedule drift guard in `tests/test_sentry.py`, the `live_forecasts_job` partition and
schedule tests in `tests/test_assets.py`, and the `list_promotable_runs` tests in
`ml_core/tests/test_mlflow_runs.py`. Comments naming XGBoost in `tests/_pytest_autoinject.py`,
`ml_core/tests/test_base_forecaster.py`, `ml_core/tests/test_features.py`,
`ml_core/tests/test_cross_mode_equivalence.py`, `ml_core/tests/test_feature_engineer.py`,
`tests/_nwp_test_data.py`, and `tests/test_metrics.py` are reworded, and
`tests/test_backup_workstation.py` drops `production_model` as a backup source.

### Design-philosophy check

The serving path is removed rather than changed, so no degradation rule is weakened in code that
still runs. The gap is the cost: until PR #1707 lands, no live forecast reaches NGED and Sentry has
no live-slot check. The change follows the principle that code no caller uses goes, rather than
staying for a possible future caller. The one exception, `load_trained_metadata` with
the frozen metadata copy, stays for a live asset of another time-series type.

## Tests

No test is added: every behaviour this PR leaves is already tested, and a test asserting that the
removed assets are absent would guard against a change nobody is going to make. The tests that
move to `ClimatologyForecaster` keep their assertions on the CV, MLflow, and config machinery. Each
has to fail on a broken machinery path, not on XGBoost, so the mutation pass checks those moved
tests only. `dagster definitions validate`, run locally and in the image smoke test, catches a
dangling import of a deleted module. CI does not run that command, but
`tests/test_asset_layer_tags.py` imports `defs` and so fails on the same dangling import.

The CV integration tests lose their only coverage of a weather feature (`temperature_2m`) through
CV, because climatology selects none. The weather path keeps its unit tests in
`ml_core/tests/test_features.py`.

## Docs to update

Every page that `git grep -il "xgboost\|live_forecasts\|promoted_model\|production_model"` finds
under `docs/` (47 pages), rewritten to describe the code as it now is. The heaviest are
`docs/live_service/aws.md` (whose "Redeploying a new champion model" section is deleted),
`operations.md`, `setup.md`, `docs/architecture/production-deployment.md`, `overview.md`,
`testing.md`, `docs/cross-validation/model-configuration.md`, `dagster-workflow.md`, and the
roadmap pages that describe XGBoost as the current model. `docs/roadmap/index.md` links to the
deleted XGBoost roadmap page, so that link goes too.

## What PR #1707's plan then has to add back

PR #1707's plan assumes `live_forecasts` keeps running as the fallback. After this PR:

- **No fallback.** When today's 00Z run is absent at 15:00 UTC, the step-5 asset writes nothing and
  NGED gets no forecast that day. The plan's error-handling text changes to say so.
- **The live asset checks**, re-added for `live_primary_substation_forecasts` if wanted: the slot
  wrote rows, every expected series is present, the horizon is at least the minimum fraction, no
  quantile column is non-finite, no row is a hindcast, and the count of missed NWP runs.
- **A dead-daemon alarm, if wanted.** PR #1707's plan adds no check-in monitor and counts on the
  existing `live-forecasts` monitor, which this PR removes with its only feeder. A heartbeat for
  `live_primary_substation_forecasts` is about 60 lines of `_sentry.py` to re-add, with a crontab
  matching its daily schedule.
- **The Docker image.** PR #1707's plan copies the bundle "beside the existing champion model". The
  bundle becomes the image's only model. The image has no default `CMD`, and its smoke test runs
  `definitions validate`.
- **`_available_nwp_init_times`** is imported from `defs/assets.py`, not
  `defs/live_forecast_assets.py`.
- Unchanged and reused as named: `write_power_forecasts` with `replace_partition`,
  `ecmwf_ens_partitions`, `ecmwf_ens_schedule`, `_is_too_old_to_retry`, `report_asset_degradation`,
  and `sentry_capture_failure`.

## Verification commands

The `implement-issue` green-before-push set (`uv run ruff check .`, `uv run ruff format --check .`,
`uv run ty check`, `uv run pytest`, the markdown lint), plus `uv run mkdocs build --strict` for the
removed pages and links, `uv lock --check`, and a local `dagster definitions validate -m
nged_substation_forecast.definitions`.

After the change, `git grep -n "production_helpers\|fetch_model_artifacts\|live_forecasts\|xgboost"`
over the whole repository, docs and tests included, returns only research references that cite
XGBoost as history. The implementation then does one CV run of the new reference and a
`score_study.py` dry run against it.

## Risks and open questions

1. **The CV reference run.** `reference_experiment_name` moves from `xgboost_no_power_lags` to a
   climatology run. Recommendation: climatology, because it trains in seconds and needs no weather.
   The run has to be re-run into the `power_forecasts` table that PR #1709 recreates before
   `score_study.py` accepts a study. `score_study.py` requires a study's `PowerForecast.PRIMARY_KEY`
   values to equal the reference's, and climatology drops rows in empty cells, so the
   implementation checks the climatology run's key set against one existing study's keys on one
   fold before switching. `docs/cross-validation/cross-validation-folds.md` is updated only if it
   names the XGBoost run.
2. **The Sentry monitor at deploy.** Delete the `live-forecasts` cron monitor by hand when this PR
   deploys, because no code feeds it any more.
3. **Follow-up, not in this PR: the weather features.** With XGBoost gone, no production
   forecaster selects a weather feature from `TabularFeatureEngineer`, and the step-5 asset builds
   its own inputs. The weather columns of `AllFeatures`, `windchill` (issue #1199), and the weather
   path of the tabular engineer could go in a follow-up issue. That issue changes a Patito
   contract, so it is not folded in here.

## Simplicity review: triage

Two independent simplicity reviews ran on the first draft and agreed on most findings.

- **Delete the Sentry check-in code:** accepted. PR #1707's plan adds no check-in monitor, so
  `send_forecast_checkin` and its monitor config had no caller left. The issue body listed the
  function under "What stays" and is amended.
- **Move the NWP-run helpers into `defs/assets.py`, and keep their test:** accepted. A new
  `defs/_nwp_runs.py` was one module too many, and the first draft deleted the helpers' only test.
- **Choose one Docker smoke option:** accepted. The smoke test runs `definitions validate`, and the
  default `CMD` is deleted.
- **Drop the absence test:** accepted. `definitions validate` catches the real failure, a dangling
  import.
- **Delete `load_trained_metadata` and the frozen metadata copy as dead code:** rejected. PR #1707
  does not use them, but a live asset for another time-series type that loads a saved forecaster
  does, and the plan now says so.
- **Delete promotion-only pages instead of rewriting them:** accepted for `model-releases.md`, the
  `release-model` skill, and the "Redeploying a new champion model" section of `aws.md`.
- **Count corrections:** accepted. About 130 files name the removed code, not 70, and 46 docs
  pages, not 40. `live_forecast_assets.py` moves to the deleted-files table, and
  `tests/test_assets.py`, `ml_core/tests/test_mlflow_runs.py`, `contracts/tests/test_settings.py`,
  `.dockerignore`, and the `cv_assets.py` docstring are added.

## Correctness review: triage

- **`fetch_model_artifacts` is still imported by surviving tests:** accepted. The function goes
  with `production_helpers.py`, and the `load_trained_metadata` tests read the saved directory.
- **Climatology rejects any selected feature, so the moved tests need named substitutes:**
  accepted. The test list now names each substitute.
- **`test_forecaster_config_serialisation.py` cannot move:** accepted. The file is deleted.
- **`test_asset_layer_tags.py` and `docs/api/ml_core/index.md` name deleted code:** accepted.
- **`test_cv_power_forecasts.py` and `test_metrics.py` change through the fixture:** accepted.
- **The climatology reference may not share the study's row keys:** accepted as a check in open
  question 1.
- **Removing two `Settings` fields breaks any `.env` that still sets them:** accepted.
- **CI does not run `definitions validate`:** accepted. The claim is corrected.
- **Docstrings in a dozen more files name the deleted code:** accepted.
- **The deleted-lines total double-counts the package:** rejected. The 16 rows sum to 5,482 lines,
  with the package's five files in one row, so the table's count of 20 files stands.
