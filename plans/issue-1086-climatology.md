# Plan: issue #1086 — the `climatology` baseline forecaster

**The problem: the leaderboard has no calendar-only reference, so nobody can tell whether the
weather ensemble still adds skill at long lead times.** Issue
[#1086](https://github.com/openclimatefix/nged-substation-forecast/issues/1086) asks for the
`climatology` baseline that [Persistence and climatology — diagnostic
bookends](https://openclimatefix.github.io/nged-substation-forecast/roadmap/metrics-and-leaderboard/#persistence-and-climatology-diagnostic-bookends)
designs. `manual_heuristic` is merged and is the bar to beat. Climatology is the loose bookend at
the far end of the horizon: at day 8 to day 14, does the NWP-driven forecast know more than the
distribution of past power at that calendar month, local half-hour of day, and day type? The
dashboard's climatology reference band in
[#354](https://github.com/openclimatefix/nged-substation-forecast/issues/354) is blocked on this
baseline.

**The solution: a `ClimatologyForecaster` in `packages/baseline_forecasters/` that stores 13
empirical quantiles of training power per calendar cell, and emits them as 13 `ensemble_member`
rows.** A cell is `(time_series_id, local month, local half-hour of day, local is-weekend)`, derived
inside the forecaster from `valid_time` in Europe/London time. `train()` deduplicates the training
targets on `(time_series_id, valid_time)` and takes the quantiles at the equiprobable levels
`(i − 0.5)/13` for i = 1 to 13. `predict()` joins the lookup onto the forecast rows and unpivots the
13 quantile columns into members 0 to 12, member 0 being the lowest quantile. The rows come from the
same NWP-run engineer the manual heuristic uses, so climatology is scored on the same
`(power_fcst_init_time, valid_time)` rows as XGBoost and the manual heuristic. The climatology needs
no new probabilistic format, column, or schema: the metrics layer already scores the members as an
equiprobable sample. That engineer, and the `meta.json` save/load round trip, move out of
`manual_heuristic.py` into shared modules, because climatology is their second caller. Nothing under
`src/nged_substation_forecast/defs/` changes.

## Verdict, size and departures

**Worth implementing, and now.** The maintainer has put climatology next, #354 waits on it, and
every piece it needs already exists: the NWP-run engineer, the `BaseForecaster` interface, the CV
asset chain, and a metrics layer that scores ensemble members with the fair continuous ranked
probability score (CRPS).

**Size: complex, which buys both plan reviews and both diff reviews.** The five triggers:

- **Changes what gets stored — yes.** New `experiment_name=climatology` rows in the
  `power_forecasts` and `forecast_metrics` Delta tables, and a new saved-model format: a parquet
  quantile lookup plus `meta.json`.
- **Touches the production serving path — no serving code changes.** The package is already a
  production dependency. A climatology model promoted by mistake raises in the shared engineer's
  single-run branch, exactly as the manual heuristic does. `XGBoostForecaster` is not touched.
- **Touches a degradation rule — yes, in the sense that climatology is a bookend of the skill
  range, not the T1.2 floor.** Climatology is an R&D baseline and is not served live, so the
  production degrade-not-raise rules do not govern it. `predict` still drops an unforecastable row
  and logs the drop rather than raising.
- **Admits more than one defensible design — yes.** The fallback for a sparse or unseen cell, the
  quantile interpolation, whether the ensemble size is a config field, what to extract for sharing,
  and whether to rename the engineer all had alternatives.
- **Spans code whose callers could not be named without searching — no.** The refactor's callers
  are listed under "What changes, file by file". The forecaster's callers are `trained_cv_model`,
  `cv_power_forecasts`, and `compute_metrics` (via the `metrics` asset).

**Out of bounds, because other sessions are editing them:** everything under
`src/nged_substation_forecast/defs/`, `packages/contracts/src/contracts/config_schemas.py`,
`conf/cv/`, and the new failure-scenario module in `ml_core` (#437). Reading and importing from them
is fine (`class_target` comes from `config_schemas`). Nothing in this plan needs an edit there.
`uv.lock` may collide with #958. This PR adds no dependency, so the lock file should not change. If
it does, whichever PR lands second re-runs `uv lock`.

**Departures from the roadmap's "PR C — `ClimatologyForecaster`" item:**

1. **This PR extracts the shared engineer and the shared `meta.json` helper.** The roadmap gives
   both extractions to PR B (persistence), on the assumption that PR B lands first among the two.
   PR B is deferred to v0.9 (#1087), so climatology is now the second caller of both.
2. **The engineer is renamed to `NwpRunRowsWithoutWeatherFeatureEngineer`.** The current name,
   `PowerLagsPerNwpRunFeatureEngineer`, says it engineers power lags. Climatology requests none, so
   the name would be false for the second caller. The new name says what the engineer delivers: one
   row per series, NWP run, and valid time, with no weather value read.
3. **The repetition the dedupe removes is once per covering NWP run, not once per run and member.**
   The roadmap says bulk-mode `AllFeatures` repeats each target "once per covering `(nwp_init_time,
   ensemble_member)`". The key-only engineer drops the member axis, and `trained_cv_model` loads
   member 0 only, so a training target repeats once per run covering it: up to 15 times at a 15-day
   horizon. The dedupe is still required.
4. **The sample counts per cell are slightly wider than the roadmap states.** The roadmap says a
   weekend cell holds about 9 to 17 samples and a weekday cell about 21 to 43 "over the ~14-month
   training window". The training window, 2024-04-01 to 2025-06-30, is 15 months. Computed in local
   time from the first post-hindcast valid time (2024-04-01 09:30 UTC) to the window's end, a
   series with complete data has 8 to 19 samples per weekend cell and 20 to 45 per weekday cell.
   April, May, and June are covered twice (weekday cells 41 to 45, weekend 16 to 19). July to March
   are covered once (weekday 20 to 24, weekend 8 to 10). All 1,152 cells per series are populated.
5. **The ensemble size is a module constant, not a config field.** See decision 4 below.
6. **The `Implementation details — baselines` section cannot simply be deleted at ship.** The
   section also holds the metrics-collapse item for the still-open
   [#1077](https://github.com/openclimatefix/nged-substation-forecast/issues/1077), whose body links
   to that section, and the unverified re-run recipe. See "Docs to update" for where each piece
   goes.

## Decisions, with reasons

### 1. Reusing the NWP-run engineer for a forecaster with no lag features

**The engineer works with `selected_features` empty, and gives `train()` and `predict()` everything
they need.** Verified against the code and by a scratch run on the manual heuristic's own test
fixtures (three daily runs, two members, one cell):

- `ParsedFeatures.from_strings(set())` parses to no features, `requires_weather_data()` is false,
  and `max_power_lag()` is zero, so `trained_cv_model` and `cv_power_forecasts` pass a
  `power_lookback` of zero. `_select_output_columns` with no requested feature returns the base
  columns: `valid_time`, `time_series_id`, `power`, `power_fcst_init_time`, `nwp_init_time`, and
  `nwp_lead_time_hours`. No `ensemble_member` column appears, because the key-only NWP frame has
  none.
- The scratch run returned 2,106 rows over 798 distinct valid times and 3 distinct
  `power_fcst_init_time` values, with each `(time_series_id, valid_time)` appearing up to 3 times,
  once per covering run. `power` was non-null wherever power existed.
- `train()` reads `time_series_id`, `valid_time`, and `power`. `predict()` reads
  `time_series_id`, `power_fcst_init_time`, and `valid_time`, and must not read `power` (decision 2).

**Recommendation: rename the class to `NwpRunRowsWithoutWeatherFeatureEngineer` and move it into
`baseline_forecasters/nwp_run_rows.py`.** The name avoids "grid", which this project already uses
for H3 cells and gridded weather data. The current docstring's "forecast-run grid" is rewritten for
the same reason. The move is the roadmap's own plan for the second caller. The class body does not
change. The one behaviour change is the `NotImplementedError` message text, which names the class.
The existing test matches on "bulk mode only", which survives.

**Recommendation: extract the `meta.json` round trip into `baseline_forecasters/_saved_model.py`,
used by both baselines and by nothing else.** Two functions: one that clears a directory
(`shutil.rmtree(..., ignore_errors=True)`, then `mkdir`) and writes `meta.json` with the three keys
`model_params` (`model_dump(mode="json")`), `trained_time_series_ids`, and `model_class`
(`class_target(forecaster)`); and one that reads `meta.json` back into a `TypedDict` with those
three keys. Both forecasters' `load` still build their config through `cls.CONFIG_CLASS`, so
`production_helpers._check_meta_is_servable` keeps validating against the class `load` uses.
`XGBoostForecaster.save` writes the same three keys, but it sits on the production serving path, so
it is left alone. No `StatelessForecaster` base class: climatology is not stateless.

The callers of the moved pieces, all updated in this PR:

- `ManualHeuristicForecaster.feature_engineer`, `ManualHeuristicForecaster.save`, and
  `ManualHeuristicForecaster.load` in `manual_heuristic.py`.
- `packages/baseline_forecasters/tests/test_manual_heuristic.py`, which imports the engineer and
  constructs it in two tests.
- The prose that names the class: `packages/baseline_forecasters/README.md`,
  `docs/architecture/code-style.md` (the naming example), and `docs/roadmap/metrics-and-leaderboard.md`
  (the PR B item, which moves to #1087 at ship).

**The refactor must not change the manual heuristic's behaviour.** `meta.json` keeps the same three
keys and the same `json.dumps` call. The existing `test_save_then_load_round_trips_and_replaces_the_directory`
and `test_engineer_rows_equal_the_tabular_rows_deduplicated_across_members` stay unchanged apart from
the import, and must pass before and after.

### 2. `train()`: the cells, the dedupe, the quantiles, and the leakage argument

**`train(data, time_series_ids)` builds one lookup row per populated cell, in six lazy steps and one
streamed collect:**

1. Select `time_series_id`, `valid_time`, and `power`. Keep rows whose `time_series_id` is in
   `time_series_ids` and whose `power` is not null.
2. Deduplicate on `(time_series_id, valid_time)`. `power` is the observation at `valid_time`, so
   every duplicate carries the same value and `keep="any"` is correct.
3. Add the three cell keys from one shared helper, `_local_calendar_cell_keys`, which `predict`
   calls too, so the two cannot drift: `local_month` (1 to 12), `local_half_hour_of_day` (0 to 47),
   and `local_is_weekend` (local Saturday or Sunday). Each derives from `valid_time` converted to
   `DEFAULT_LOCAL_TIMEZONE` (Europe/London, from `ml_core.features.feature_engineer`), so a
   23:30 UTC target in British Summer Time falls in the next local day's 00:30 slot. The pipeline's
   own `local_*` features are sine and cosine pairs plus a weekday enum, with no integer month or
   half-hour, so the forecaster derives the keys itself, as the roadmap says.
4. Group by `time_series_id` and the three keys. Aggregate one column per member,
   `power_quantile_member_00` to `power_quantile_member_12`, each
   `pl.col("power").quantile(level, "linear")` cast to `Float32`, plus `training_sample_count`.
5. Collect with `engine="streaming"`, and sort by the four key columns so the saved parquet is
   deterministic.
6. Set `trained_time_series_ids` to the sorted distinct `time_series_id` values in the lookup. That
   set equals "requested and has at least one non-null `power`", the manual heuristic's and
   XGBoost's rule.

**The quantile interpolation is Polars' `"linear"`, passed explicitly.** `"linear"` is
Hyndman and Fan's type 7: for n samples, level p reads position (n − 1)·p between the sorted
samples. `metrics.py` reads the delivery quantiles from members with the same method, so the
metrics layer and the forecaster share one definition. The method never extrapolates beyond the
smallest and largest sample, so a sparse cell's tail members sit at the observed extremes. Polars'
default is `"nearest"`, so an implementer who omits the argument gets a different, step-valued
answer: test 2 fails that mutation. Polars 2.0.0 rejects a list of levels inside `group_by().agg`,
so the 13 levels are 13 expressions.

**The levels are `(i − 0.5)/13` for i = 1 to 13, held as one module constant.** The level
for member k (0-based) is `(k + 0.5)/13`, so member 6 is exactly the median.

**A sparse cell is kept, and an unseen cell gets no lookup row.** Recommendation: no
minimum-sample threshold and no fallback in this PR. With complete data every cell holds at least 8
samples (departure 4), and the real-data run measures how many cells fall short (decision 6). A cell
with 1 sample yields 13 equal members, a degenerate but honest ensemble whose fair CRPS equals its
mean absolute error (MAE). Pooling the two neighbouring calendar months, the fallback rule
`packages/studies/src/studies/baselines.py` uses for its deterministic climatology, stays the
roadmap's "possible refinement if the numbers look ragged". `baseline_forecasters` may not import
`studies` code. Climatology is an R&D baseline, so the production rule to always emit a forecast
does not bind it. A dropped row is logged and counted rather than silently missing.

**`train` logs one aggregate line**: the number of series and cells, the minimum and median
`training_sample_count`, and the earliest and latest deduplicated `valid_time`. The real-data run
reads the latest `valid_time` to confirm that no validation target reached the lookup.

**The training never sees validation data.** `trained_cv_model` hands `train` features engineered
from `load_engineering_inputs` over `[train_start, train_end]` with `power_lookback` zero. The
power scan is bounded to `time <= train_end` and the NWP scan to `valid_time <= train_end`, so every
training target, and every `power` value on it, lies inside 2024-04-01 to 2025-06-30 for the
leaderboard fold. Truth is the cleaned power table (`scan_cleaned_power`), so a flagged reading is
neither a training sample nor a scoring target. Bank holidays are ordinary days, as for the manual
heuristic: Christmas Day on a weekday falls in that month's weekday cells. `predict` reads only the
lookup and the calendar keys of each validation row. The validation rows' own `power` column is
present in the engineered frame, and `predict` must not read it: test 10 pins that.

### 3. `predict()`: join, unpivot, drop, and stamp

**`predict(data, *, fold_id="live")` in five steps:**

1. Collect the input with `engine="streaming"`, selecting only `time_series_id`,
   `power_fcst_init_time`, and `valid_time`, and strip the Patito model with `.as_polars()`.
2. Add the cell keys with `_local_calendar_cell_keys`, and inner-join the lookup on the four key
   columns (both sides plain Polars frames, per `polars-patito-gotchas`).
3. Count the input rows the join dropped and, when the count is positive, log one warning naming
   the count and the `time_series_id`s affected. Nothing raises.
4. Unpivot the 13 quantile columns into `ensemble_member` and `power_fcst`, mapping each column
   name to its member index with `replace_strict(..., return_dtype=pl.Int8)`, so member 0 is the
   lowest quantile on every row.
5. Add `nwp_init_time` as a typed null, and the identity literals exactly as the manual heuristic
   builds them: `power_fcst_model_name`, `power_fcst_model_version` (`Int16`),
   `ml_flow_experiment_id` (`Int32`), `experiment_name`, and `fold_id`. Return
   `PowerForecast.validate(...)`.

**Empty input returns a valid empty `PowerForecast`.** The join, unpivot, and literals give a
correctly typed zero-row frame. Test 8 pins this rather than adding a branch.

**Nothing in `PowerForecast`, `compute_metrics`, or the dashboard needs a new column or schema for
a member that is a quantile sample.** Checked against the code:

- `PowerForecast.ensemble_member` is `Int8`, and members 0 to 12 fit. The primary key
  `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)` is unique, because the
  engineer emits one row per series, run, and valid time.
- `compute_metrics` groups each `(time_series_id, power_fcst_init_time, valid_time)` and treats the
  members as an equiprobable sample: the ensemble mean, the fair CRPS, the Fortin-corrected
  variance, and `quantile(q, "linear")` for each delivery quantile. Equiprobable quantile values are
  exactly the "equiprobable pseudo-sample" `docs/techniques/probabilistic-forecasting.md` describes
  for pooling quantile forecasts.
- `packages/dashboard/view_forecasts.py` draws each `ensemble_member` as a thin grey line, which
  works unchanged.
- The `ensemble_member` description is the one edit: it names the NWP member and the manual
  heuristic's analogue rank, and gains a sentence for climatology. The description on `main` does
  not yet mention climatology. The brief's claim that it already promises a climatology sentence
  "from PR C" is stale: that wording was in the manual-heuristic plan, not in the merged code.

### 4. The ensemble size: 13, hard-coded

**Recommendation: keep m = 13, as the roadmap argues, as a module constant and not a config
field.** At 8 to 19 samples per weekend cell, levels below about 0.06 already sit on the smallest
sample, so 51 members would be false precision and about 4 times as many `power_forecasts` rows. A
13-member climatology also matches the manual heuristic's 13 members, so the size-dependent metrics
— PICP, pinball loss, interval width, and the exceedance rate — compare the two baselines at equal
m. `BaseForecasterConfig` is `extra="forbid"`, so a config field would need a `ClimatologyConfig`
subclass. That subclass would join `tests/test_forecaster_config_serialisation.py`'s parametrised
checks through an explicit import, and every saved config would carry a value nobody varies.
`CONFIG_CLASS = BaseForecasterConfig`, so the serialisation test needs no change.

**`__init__` raises `ValueError` when `selected_features` is not empty.** Climatology reads no
feature. A non-empty list would describe, in MLflow's params, an experiment that is not the one
running, and a power lag would widen the power scan for nothing. The YAML sets
`selected_features: []`. Registration validates only `CONFIG_CLASS`, so a bad list surfaces at
`trained_cv_model`, as for the manual heuristic.

### 5. Save and load

**`save(path)` clears the directory and writes `meta.json` through the shared helper, then writes
`climatology_quantiles.parquet`.** `load(path)` reads `meta.json`, rebuilds the config through
`CONFIG_CLASS.model_validate`, and reads the parquet back. The population comes from `meta.json`,
never from the parquet's contents, matching `XGBoostForecaster.load`. The file name must differ from
`time_series_metadata.parquet`, which `save_to_mlflow` adds to the same directory.

**The lookup fits the one-archive MLflow path at V2 scale.** About 2,500 series × 12 months × 48
half-hours × 2 day types is 2.88 million rows, holding 13 `Float32` quantiles (37.4 million values),
3 small integer or Boolean keys, the `Int32` series id, and a `UInt32` sample count: about 180 MB
in memory, and less on disk as compressed parquet. The V1 lookup is 31 × 1,152 = 35,712 rows. The
archive `save_to_mlflow` writes holds one model file plus the frozen metadata, so the lookup adds
nothing to the merge problem `_MLFLOW_MODEL_ARTIFACT` documents. By comparison, the 40 XGBoost
boosters measured on that constant total 77 MB, which scales to several GB at V2.

### 6. The run on real data, after implementation

**Run climatology on the single leaderboard fold, `mid_2025_to_mid_2026`, for the 31 eligible
series, and compare it with `manual_heuristic` and `xgboost_baseline_retrain_790`.** The procedure
copies the manual-heuristic run, so the shared data folder gains only climatology partitions:

- Check CPU load first, and confirm no other session is writing `power_forecasts` or
  `forecast_metrics`.
- In the worktree, symlink `.env`, the shared `data` folder, `mlflow.db`, and `mlruns` from the
  primary checkout, and set `DAGSTER_HOME` to a fresh untracked directory inside the worktree.
- From a scratch driver in the session scratchpad, modelled on
  `scripts/forecasting/run_baseline_experiment.py` and never committed: register experiment
  `climatology` from `conf/model/climatology.yaml` with `config_overrides={}` and
  `run_mode="full_cv"`, materialise `trained_cv_model` and `cv_power_forecasts` for partition
  `climatology__mid_2025_to_mid_2026`, then `metrics` with
  `PopulationFilter(experiment_name="climatology", fold_id="mid_2025_to_mid_2026")`. Nothing
  touches the `xgboost_*` or `manual_heuristic` partitions or MLflow experiments.
- Before reading results, record per series: the number of populated cells out of 1,152, the
  minimum `training_sample_count`, and the rows `predict` dropped. Confirm that the latest training
  `valid_time` in the train log is no later than 2025-06-30 23:59:59 UTC, and that every forecast
  row has 13 members.
- Re-score all three experiments in a scratch script with the repo's `compute_metrics`, the same
  cleaned truth (`scan_cleaned_power`), and the same effective capacity, on the same series, writing
  nothing back. Report by group — substations (Primary, bulk supply point, and grid supply point),
  PV, and wind — with group means and the "beats" count per series, as PR #1057 did. The headline
  is `crps__all__extended_range`, with NMAE in each horizon slice. Report absolute values, not only
  contrasts. If climatology dropped any row, re-score all three on the intersection of rows as well.
- Record time and peak memory for training and prediction, and the saved lookup's size.

**Three caveats go wherever the comparison is read.** First, a set of quantiles at equiprobable
levels slightly out-scores an ensemble of independent draws of the same size under the fair CRPS
([Ferro (2014)](https://doi.org/10.1002/qj.2270)), so climatology has a small structural edge: read
a near-tie at extended range as "the NWP ensemble adds little out here", not as climatology winning.
Second, `xgboost_baseline_retrain_790` was trained and predicted on power that was not yet cleaned,
with code older than `main`, so the fair XGBoost comparison needs a retrain. Third, one fold of one
year, with no intervals.

## What changes, file by file

### `packages/baseline_forecasters/src/baseline_forecasters/`

- **`nwp_run_rows.py` (new).** Holds `NwpRunRowsWithoutWeatherFeatureEngineer`, moved from
  `manual_heuristic.py` with the body unchanged. The docstring says the engineer produces one row
  per series, NWP run, and valid time, carrying `power` and whatever power features were requested,
  possibly none, and that it reads no weather value. The `NotImplementedError` message names the new
  class.
- **`_saved_model.py` (new).** The two `meta.json` helpers from decision 1, plus the `TypedDict`
  describing the file.
- **`manual_heuristic.py`.** Drops the engineer class and imports it from `nwp_run_rows`. `save`
  and `load` call the shared helpers. The `predict` docstring names the new engineer.
- **`climatology.py` (new).** `CLIMATOLOGY_QUANTILE_LEVELS: Final[tuple[float, ...]]`, the
  quantile-column names, `_local_calendar_cell_keys` (one function returning the three key
  expressions from a `valid_time` expression and a time zone), a module `logger`, and
  `ClimatologyForecaster` with `MODEL_NAME = "climatology"`, `MODEL_VERSION = 1`, `CONFIG_CLASS =
  BaseForecasterConfig`, `feature_engineer = NwpRunRowsWithoutWeatherFeatureEngineer()`,
  `__init__`, `trained_time_series_ids`, `train`, `predict`, `save`, and `load` as described above.
- **`__init__.py`.** Exports `ClimatologyForecaster` beside `ManualHeuristicForecaster`, and the
  module docstring names both.

### `conf/model/climatology.yaml` (new)

**`_target_: baseline_forecasters.climatology.ClimatologyForecaster`, with `selected_features: []`,
`weather_source: "none"`, and `training_strategy: "none"`.** The header comment follows
`manual_heuristic.yaml`'s: the forecaster reads no feature, the cells and levels are fixed in code,
and a non-empty list raises when `trained_cv_model` runs.

### `packages/contracts/src/contracts/power_schemas.py`

**Description text of `PowerForecast.ensemble_member` only.** Adds: "For `climatology`, the index
is the rank of a quantile level, with 0 the lowest: member k is the empirical quantile at level
(k + 0.5)/13 of the series' training power in the valid time's calendar cell." No dtype,
nullability, or constraint changes, so plan approval meets the "ask before changing a Patito data
contract" rule.

### Tests

- **`packages/baseline_forecasters/tests/test_climatology.py` (new).** Tests 1 to 14 below, on
  hand-built `AllFeatures` frames.
- **`packages/baseline_forecasters/tests/test_manual_heuristic.py`.** The engineer import becomes
  `from baseline_forecasters.nwp_run_rows import NwpRunRowsWithoutWeatherFeatureEngineer`, and the two
  engineer tests use the new name. No assertion changes.
- **`tests/test_climatology_cv.py` (new).** Test 15, the integration test.
- `tests/conftest.py` needs no change: `register_experiment` already takes `base_model_config` and
  `config_overrides`.

## Design-philosophy check

**The code runs in the R&D asset chain only, and `predict` drops and logs rather than raising.** An
unseen cell drops its rows with one aggregate warning naming the series, as "Make the telemetry
name the fault" asks. Nothing raises except our own bugs: a non-empty `selected_features`, a
contract violation in `PowerForecast.validate`, or a single-run call to the shared engineer. No
asset check is added, and no warning path can raise.

**No hypothesis label is delivered.** Climatology is a diagnostic bookend, not the T1.2 floor.
The degradation ladder in `docs/design-philosophy/inherent-stability.md` names climatology as a
component of rungs 3 and 4. This PR builds the R&D baseline that measures where those rungs would
sit, not the production rung.

**Principles 3, 8, and 11 are kept.** The forecaster rides the same `register_experiment_job →
trained_cv_model → cv_power_forecasts → metrics` chain as XGBoost (principle 3), and is scored on
the same NWP-run rows (principle 8). The one difference in row sets is a row in an unseen cell,
which is dropped and counted. The key-only engineer and the streamed group-by keep the work in the
query engine (principle 11).

**One trade is inherited from the manual heuristic: the rows follow the NWP archive.** An
`init_time` missing from the archive removes that run's climatology rows, as for every other model,
which keeps the comparison like for like.

## Tests

On `main` the module does not exist, so every new test fails at import. The assertion listed for
each test is the behaviour it pins once the import resolves, chosen so that a plausible wrong
implementation fails it.

**`packages/baseline_forecasters/tests/test_climatology.py` (unit):**

1. **Construction.** `ClimatologyForecaster(BaseForecasterConfig(selected_features=set()))`
   succeeds. A config selecting `power_lag_168h` raises `ValueError`.
2. **Hand-computed quantiles.** One series, one cell, 5 deduplicated samples with power 0, 10, 20,
   30, and 40. The lookup's member-k column equals 40·(k + 0.5)/13 for k = 0 to 12, to `Float32`
   precision, and `training_sample_count` is 5. A `"nearest"` interpolation gives multiples of 10
   and fails. A level formula of k/12 fails at k = 0.
3. **The dedupe.** Duplicating a subset of the training rows — the two largest samples, under three
   extra `power_fcst_init_time` values each — leaves the lookup equal (`assert_frame_equal`) to the
   lookup trained on the rows without duplicates. Without the dedupe, the duplicated samples pull
   the interior quantiles upwards, so the assertion fails.
4. **The levels.** `CLIMATOLOGY_QUANTILE_LEVELS` has 13 entries equal to `(i − 0.5)/13` for i = 1
   to 13, written out in the test.
5. **Member order.** On the fixture of test 2, `predict` emits members 0 to 12 for the row, and
   member k's `power_fcst` equals the lookup's member-k column, so the values strictly increase with
   the member index. A shuffled column-to-member mapping fails.
6. **Local-time cell keys, across a clock change and at local midnight.** `_local_calendar_cell_keys`
   maps 2025-06-06 23:30 UTC (Saturday 00:30 BST) to June, half-hour 1, weekend; 2025-06-30 23:30 UTC
   (Tuesday 1 July 00:30 BST) to July, half-hour 1, weekday; 2025-01-17 23:30 UTC (Friday 23:30 GMT)
   to January, half-hour 47, weekday; and 2025-03-30 01:00 UTC (02:00 BST, just after the clock
   change) to March, half-hour 4, weekend. End to end, a forecaster trained only on 2025-06-07
   00:30 BST forecasts a row at 2025-06-14 23:30 UTC (Sunday 00:30 BST) and drops a row at
   2025-06-12 23:30 UTC (Friday 00:30 BST). A UTC-keyed implementation fails both halves.
7. **An unseen cell drops its rows and logs.** A forecast row in a cell with no training sample is
   absent from the output, the call does not raise, and `caplog` holds one warning carrying the
   dropped count and the series id. Rows in seen cells are unaffected.
8. **Empty input.** A zero-row `AllFeatures` frame returns a zero-row frame that
   `PowerForecast.validate` accepts.
9. **Stamping.** Every row carries `power_fcst_model_name == "climatology"`,
   `power_fcst_model_version == 1`, the config's `experiment_name` and `ml_flow_experiment_id`, the
   `fold_id` passed in (a non-default value such as `"fold_x"`), and a null `nwp_init_time`. The
   manual heuristic's mutation testing found that stamping goes untested otherwise.
10. **`predict` ignores the validation power.** Two `predict` calls on the same rows, one with the
    `power` column replaced by large arbitrary values, give equal output. A `predict` that reads
    `power` fails, which is the leakage guard the maintainer asked for.
11. **Train population.** Rows for series 1, 2, and 3, where series 2 has only null power and
    series 3 is not requested, give `trained_time_series_ids == [1]`, and the lookup holds no cell
    for series 2 or 3.
12. **Save and load.** `load(save(...))` returns the same `trained_time_series_ids`, an equal config,
    and an equal lookup, and `predict` after the round trip equals `predict` before it. A stale file
    placed in the directory before `save` is gone afterwards. `meta.json`'s `model_class` is
    `"baseline_forecasters.climatology.ClimatologyForecaster"`.
13. **Two series do not share cells.** Two series with different power in the same calendar cell
    get different quantiles. A group-by that omits `time_series_id` fails.
14. **The fixture is not vacuous.** The 13 expected quantiles of test 2's fixture are pairwise
    distinct, so a mislabelled member cannot pass tests 2 and 5 by coincidence.

The existing `test_manual_heuristic.py` suite is the refactor's regression test: it must pass
unchanged apart from the engineer's import and name.

**`tests/test_climatology_cv.py` (integration, `pytestmark = pytest.mark.integration`), test 15:**
modelled on `tests/test_manual_heuristic_cv.py`. Register `conf/model/climatology.yaml` with
`config_overrides={}`. Write power from 30 days before the first training day to 12 hours after the
validation day, then `write_cleaned_copy` it, so `scan_cleaned_power` sees every row. Write the
metadata and the eligible-series table as that module does. Write NWP as: member-0 runs on five
weekdays of August 2024 inside the training window (each giving the five valid times `half_hours`
yields, 11:00 to 13:00 BST); a validation run with members 0, 1, and 2 on Friday 2025-08-01; and a
validation run on Saturday 2025-08-02, whose cells training never saw. Materialise `trained_cv_model`
and `cv_power_forecasts`, then read `power_forecasts` and assert:

- Every row has a null `nwp_init_time`, `fold_id == FOLD_ID`, `experiment_name ==
  EXPERIMENT_NAME`, and `power_fcst_model_name == "climatology"`.
- The rows are exactly the five Friday valid times × 13 members: the three NWP members are not
  repeated, and the Saturday run's rows are dropped without failing the asset.
- Each row's `power_fcst` equals an oracle computed in the test from the five August 2024 training
  samples of that cell with `numpy.quantile(..., method="linear")` at level (k + 0.5)/13. The five
  training days carry distinct `power_at` values, so the 13 members of each row are distinct, and a
  mislabelled member fails. The validation-day power is never a sample, so a training step that
  read validation data would fail the oracle.

The test fails on `main` at the missing YAML target. The NWP run on a training day is needed because
`trained_cv_model` raises "Trained 0 of 1 eligible time series" without it.

## Docs to update

**Written to describe the code as it is after this PR.**

- **`packages/baseline_forecasters/README.md`.** A `ClimatologyForecaster` paragraph under "What
  each baseline emits": the cells, the 13 equiprobable levels and why not the delivery levels, the
  member order, and `nwp_init_time` null. The engineer section renamed, and saying both baselines
  use the engineer. New caveats: the small samples per cell, with the measured counts, and the tail
  members sitting at the observed extremes; unseen cells dropped and logged; bank holidays treated
  as ordinary days; the Ferro structural edge in a CRPS comparison; and a 13-member climatology
  compared with the 13-member manual heuristic on equal m. The README becomes the permanent home of
  the PR C design text, including the "why a distribution, not a mean" argument.
- **`CLAUDE.md`, Packages table.** The `baseline_forecasters` row says "currently
  `manual_heuristic`"; it becomes "`manual_heuristic` and `climatology`".
- **`packages/ml_core/README.md`.** "are the two subclasses outside the tests today" becomes three,
  naming `ClimatologyForecaster`.
- **`docs/architecture/code-style.md`.** The naming example names the renamed engineer (the new
  name is still the right shape of name), and the Machine Learning section's list of
  implementations adds `ClimatologyForecaster`.
- **`docs/ml_experimentation/model-configuration.md`.** "Two exist today" becomes three, adding
  `conf/model/climatology.yaml`.
- **`docs/roadmap/metrics-and-leaderboard.md`:**
    - The status banner adds the climatology baseline to the ✅ list.
    - "Baseline forecasters": "The climatology baseline, built next" becomes present tense.
    - "Persistence and climatology — diagnostic bookends": "today `XGBoostForecaster` and
      `ManualHeuristicForecaster` exercise it" adds `ClimatologyForecaster`.
    - "Implementation details — baselines" is deleted, with every surviving piece moved first.
      The PR B item is copied into the body of
      [#1087](https://github.com/openclimatefix/nged-substation-forecast/issues/1087), replacing
      that body's link to the deleted anchor. The metrics-collapse item moves to a new
      "Implementation details — metrics collapse (deleted when it ships)" subsection under "Which
      ensemble collapse defines the deterministic point forecast?", which keeps the item in the
      reviewed docs. The re-run recipe — the `trained_cv_model++` paragraph of "Guiding principle"
      and cross-cutting items 2 and 3 — moves into that subsection too, because the metrics-collapse
      item already says the post-collapse backfill is the drill that verifies the recipe. The
      section link at "Implemented as part of the baseline work" and the link in #1077's body point
      at the new subsection, and #1077's body stops calling the item "PR 1". The PR C item is
      promoted to the package README and the PR body. "The recipe" duplicates the background page
      it links to, and "Data check before interpreting the manual heuristic's results" duplicates
      the README's short-history caveat, so both are deleted. The holiday-aligned design text sits
      outside the section and stays.
    - Grep the page for "PR B", "PR C", "PR 1", and "below" after the deletion, so no sentence
      points at removed text. "the same audited, no-lookahead pipeline as `PersistenceForecaster`
      (below)" still points at the bookends section, which stays.
- **`docs/roadmap/index.md`, v0.3.** "Baseline forecasters (persistence + climatology)" becomes the
  manual heuristic and climatology, with persistence in v0.9 (#1087).
- **`#354` needs nothing from this PR beyond the rows.** The saved forecasts carry 13 members for
  every `(time_series_id, fold_id, power_fcst_init_time)` that the NWP runs give, and the values
  for one `valid_time` are the same across runs, so the band is invariant as #354 expects. Member 6
  is exactly the p50 line #354 asks for. No member sits at p10 or p90: member 1 is level 0.115 and
  member 11 is level 0.885. A dashboard calling Polars `quantile(0.1, "linear")` on the 13 members
  gets level 0.131, because that call treats the members as a sample. #354 should either draw the
  member-1-to-member-11 band and label it with those levels, or interpolate between adjacent
  members' levels. A comment on #354 at ship states this.
- **Ship-time triage.** The PR body uses `Closes #1086` and mentions #147 without a closing
  keyword. Once climatology merges, #147's remaining items are the deferred #1087 and #1088 (its
  body says so) and #715, so #147 can close by hand. That close is the maintainer's call (question
  6).

## Verification commands

The green-before-push set, plus the two CI steps the skill's set omits:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run --no-sync ty check
uv run --no-sync pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run --no-sync mkdocs build --strict --site-dir site   # then read the rendered roadmap page
uv run --no-sync python scripts/lint/check_docs_links.py
uv run --no-sync pytest packages/baseline_forecasters tests/test_climatology_cv.py tests/test_manual_heuristic_cv.py
uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files --show-diff-on-failure
```

Then the real-data run in decision 6, whose numbers go in the PR body.

## Risks and open questions

1. **Should a sparse or unseen cell fall back to a pooled cell?** Recommendation: no fallback in
   this PR. Drop unseen-cell rows with a logged count, keep sparse cells, and decide on
   neighbouring-month pooling once the real-data run reports cells per series and the minimum
   sample count. A fallback chosen before the measurement would be tuned on a guess. A fallback
   chosen after the measurement must be fixed before looking at scores, as the studies' climatology
   rule was.
2. **Rename the engineer?** Recommendation: yes, to `NwpRunRowsWithoutWeatherFeatureEngineer`,
   because the current name is false for climatology. The cost is one example in
   `docs/architecture/code-style.md`. Keeping the old name is defensible only if the maintainer reads
   "PowerLags" as "power and its lags".
3. **Extract the `meta.json` helper now, inside `baseline_forecasters` only?** Recommendation: yes.
   Moving it to `ml_core` for XGBoost too would touch the serving path for about 10 lines.
4. **Is `training_sample_count` in the lookup worth its column?** Recommendation: yes. The train log
   and the real-data check read it, and #354 or a later fallback rule needs it. The cost is one
   `UInt32` column.
5. **The `smoke_test` fold cannot exercise climatology.** That fold trains on January 2025 and
   validates on February 2025, so every validation row sits in an unseen cell and
   `cv_power_forecasts` writes an empty partition. `conf/cv/` is out of bounds. Recommendation:
   document the limitation in the README and the YAML comment, and register climatology with
   `run_mode="full_cv"`.
6. **How should #147 close?** Recommendation: the maintainer closes #147 by hand after this PR
   merges, with a comment pointing at #1087, #1088, and #715. The PR body names #147 without a
   closing keyword.
7. **May this PR edit the bodies of #1087 and #1077?** Both bodies link to the section this PR
   deletes. Recommendation: yes, at ship, under the `github-issue-pr-workflow` skill's review step.
8. **Power labels are period-ending, so local midnight's slot is the previous evening's last half
   hour.** Keying on `valid_time` as stored puts a Friday-night reading labelled Saturday 00:00 in
   the weekend cells. Recommendation: keep `valid_time` as stored. The same rule applies in `train`
   and `predict`, and it matches the pipeline's `local_*` features and the dashboard.
9. **Training memory at V2 scale.** The training frame is about 2,500 series × 21,900 half-hours ×
   up to 15 runs, around 820 million rows before the dedupe, under the 2³² row-count limit. The
   collect is streamed and projects three columns. Recommendation: accept now; the manual
   heuristic's training reads the same frame.
10. **`docs/design-philosophy/inherent-stability.md` says the manual heuristic's rows "follow the
    NWP run grid".** That sentence uses "grid" for NWP-run rows, which the naming rule warns
    against. The sentence is outside this issue's scope. Recommendation: fix the word in this PR
    only if the maintainer agrees, otherwise leave it.
11. **The XGBoost comparator is stale.** `xgboost_baseline_retrain_790` predates power cleaning and
    current code. Recommendation: report against it with the caveat, and leave the retrain to its
    own issue.

## Considered and rejected

- **A `ClimatologyConfig` with an ensemble-size field.** A subclass, a serialisation-test import,
  and a stored field for a value nobody varies (decision 4).
- **Members at the `DELIVERY_QUANTILES` levels.** The metrics layer weights members equally, so
  tail-heavy levels would put 7.7% of the mass at the 1st and 2nd percentiles and make the ensemble
  wider-tailed than the climatology it represents (the roadmap's argument).
- **A deterministic climatology, one member.** CRPS would collapse to MAE, so the comparison could
  test point accuracy only, not whether the weather ensemble knows more than the seasonal
  distribution.
- **Cell keys from the pipeline's `local_*` features.** The pipeline offers sine and cosine pairs
  and a weekday enum, with no integer month or half-hour, so recovering the keys would be lossy and
  indirect.
- **Training from the power table rather than from the engineered features.** `train` receives
  only the engineered frame, and reading power directly would need a change in `defs/`, which is
  out of bounds.
- **A new feature engineer for climatology.** The NWP-run engineer already gives the shared rows.

## Review log

- Plan review 1 (simplicity): not yet run.
- Plan review 2 (correctness and testability): not yet run.
