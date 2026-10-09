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

**The solution: a `ClimatologyForecaster` in `packages/baseline_forecasters/` that stores 51
empirical quantiles of training power per calendar cell, pooled with the cell's eight neighbours,
and emits them as 51 `ensemble_member` rows.** A cell is `(time_series_id, local month, local
half-hour of day, local is-weekend)`, derived inside the forecaster from `valid_time` in
Europe/London time. `train()` deduplicates the training targets on `(time_series_id, valid_time)`.
Each sample then counts towards nine cells: its own, and the cells one calendar month and one local
half-hour of day either side, with the same weekend flag. `train()` takes the quantiles of each
cell's pooled samples at the equiprobable levels `(i − 0.5)/51` for i = 1 to 51. `predict()` joins
the lookup onto the forecast rows and unpivots the 51 quantile columns into members 0 to 50, member
0 being the lowest quantile. The rows come from the same NWP-run engineer the manual heuristic uses,
so climatology is scored on the same `(power_fcst_init_time, valid_time)` rows as XGBoost and the
manual heuristic. The climatology needs no new probabilistic format, column, or schema: the metrics
layer already scores the members as an equiprobable sample. That engineer, and the `meta.json`
save/load round trip, move out of `manual_heuristic.py` into shared modules, because climatology is
their second caller. Nothing under `src/nged_substation_forecast/defs/` changes.

**The pooling and the member count were changed by the maintainer after both plan reviews, on the
evidence of a measurement on the leaderboard fold (decision 4).** A third correctness review then
checked the changed parts, and its findings are applied (review log).

## Solution

### Walk through

#### The happy path

**To train climatology for a fold, `trained_cv_model` builds the training rows with the shared
engineer, and `ClimatologyForecaster.train` reduces them to a quantile lookup.** The asset reads the
config `register_experiment_job` stored from `conf/model/climatology.yaml`, whose
`selected_features` is empty. `ParsedFeatures.from_strings(set()).max_power_lag()` is therefore
zero, so `load_engineering_inputs` gets a `power_lookback` of zero and bounds the cleaned power scan
(`scan_cleaned_power`) and the NWP scan to the fold's training window. The NWP scan is restricted to
member 0 (`ensemble_members=[0]`). The asset calls
`ClimatologyForecaster.feature_engineer.engineer(...)`, which is
`NwpRunRowsWithoutWeatherFeatureEngineer` in the new `nwp_run_rows.py`. The engineer strips the NWP
frame to its run, cell, and valid-time keys and hands the key-only frame to
`TabularFeatureEngineer`, so the rows are the same `(power_fcst_init_time, valid_time)` rows every
other forecaster gets, with `power` attached and no weather value read.

**`train(features, eligible_ids)` turns those rows into one lookup row per populated calendar
cell.** `train` keeps the requested series with non-null `power`, then deduplicates on
`(time_series_id, valid_time)`. The dedupe comes before the quantiles because the engineer repeats
each target once per NWP run covering it, up to 15 times, and an undeduplicated target would be
counted 15 times in its cell. `train` collects the deduplicated `(time_series_id, valid_time,
power)` frame once, with the streaming engine: about 640,000 rows on the leaderboard fold. The
earliest and latest training `valid_time` for the log line are read from that frame. The pooling
then runs over batches of series taken from the collected frame, so the nine copies of the samples
exist for one batch at a time (decision 2). For each batch, `_local_calendar_cell_keys` adds
`local_month` (`Int8`), `local_half_hour_of_day` (`Int8`), and `local_is_weekend` (`Boolean`) from
`valid_time` in `DEFAULT_LOCAL_TIMEZONE`. The keys are local, not UTC, because the substation load
follows the local clock (decision 2 weighs UTC keys on real data). A cross join with the nine
`Int8` offsets in `CLIMATOLOGY_POOLING_OFFSETS` copies each sample into its own cell and its eight
neighbours, wrapping December to January and half-hour 47 to half-hour 0. A `group_by` on
`time_series_id` plus the three keys computes the 51 columns `power_quantile_member_00` to
`power_quantile_member_50` at the levels in `CLIMATOLOGY_QUANTILE_LEVELS`, with `"linear"`
interpolation passed explicitly because Polars' default is `"nearest"`. The batches' lookups are
concatenated and sorted so the saved parquet is deterministic. `trained_time_series_ids` is set from
the lookup's distinct series. One aggregate log line reports the cell counts, the minimum and median
number of pooled samples per cell, and the earliest and latest training `valid_time`. The sample
counts are computed during training and never stored.

**`save` writes the lookup, and `trained_cv_model` uploads it.** `save_to_mlflow` calls
`ClimatologyForecaster.save(model_dir)`. `save` first calls the new `_saved_model.py` helper, which
clears the directory and writes `meta.json` with `model_params`, `trained_time_series_ids`, and
`model_class`. The directory is cleared first so a re-materialised fold never keeps a stale file.
`save` then writes `climatology_quantiles.parquet`, named so it cannot collide with the
`time_series_metadata.parquet` that `save_to_mlflow` adds next. `save_to_mlflow` packs the directory
into the single `model.tar.gz` archive.

**To forecast, `cv_power_forecasts` loads the model and calls `predict` once per 14-day `init_time`
chunk.** `load_from_mlflow` unpacks the archive and calls `ClimatologyForecaster.load`, which reads
`meta.json` through the shared helper, rebuilds the config through `CONFIG_CLASS.model_validate`,
and reads the parquet. The population comes from `meta.json`, not the parquet, as in
`XGBoostForecaster.load`. For each chunk (`_PREDICT_INIT_CHUNK`), the asset engineers the validation
rows with the same engineer, every NWP member included, and calls `predict(features,
fold_id=fold_id)`. The engineer's key-only NWP frame carries no member, so each row is one series,
run, and valid time.

**`predict` joins the lookup onto the rows and unpivots the 51 quantile columns into members.**
`predict` collects only `time_series_id`, `power_fcst_init_time`, and `valid_time`. `power` is left
out so the validation-period power cannot reach the forecast. `predict` strips the Patito model with
`.as_polars()` before the join, because a cross-model `pt` join raises. `predict` adds the cell keys
with the same `_local_calendar_cell_keys` that `train` uses, so the two cannot disagree on a cell,
and inner-joins the lookup on the four key columns. The lookup is keyed by the cell a forecast falls
in, so `predict` needs no pooling of its own. An unpivot maps each quantile column to its
`ensemble_member` through `replace_strict(..., return_dtype=pl.Int8)`, so member 0 is the lowest
quantile and member 25 the median on every row. A null `nwp_init_time` and the identity literals
are added, and `PowerForecast.validate` returns the frame. `write_power_forecasts` stores each chunk
under the `(climatology, fold_id)` partition of `power_forecasts`.

**`metrics` then scores the 51 members as an equiprobable sample, with no change to
`compute_metrics`.** `compute_metrics` groups each `(time_series_id, power_fcst_init_time,
valid_time)`, takes the fair CRPS over the members, and reads the delivery quantiles with
`quantile(q, "linear")`, the same interpolation the forecaster used.

#### The main branches off the happy path

- **A training target covered by several NWP runs** appears once per run in the engineered frame.
  The dedupe keeps one copy. `keep="any"` is safe because every copy carries the same `power`.
- **A series with no non-null training `power`, or not in `eligible_ids`,** gets no lookup row and
  is absent from `trained_time_series_ids`, so `cv_power_forecasts` never loads its rows. This
  matches the manual heuristic's and XGBoost's population rule.
- **A cell with no training sample of its own, but with a sample in one of its eight neighbours,**
  gets a lookup row from the neighbours' samples alone. On the leaderboard fold, the 29 series with
  a full training history populate every cell from their own samples, and only series 12 and 13
  gain cells this way.
- **A sparse cell** (few pooled samples) is kept. A cell with one pooled sample gives 51 equal
  members. No minimum-sample threshold or wider fallback is applied (decision 2).
- **A validation row in a cell none of whose nine neighbourhood cells holds a training sample**
  finds no lookup row, so the inner join drops the row. `predict` counts the dropped rows and logs
  one warning naming the count and the series. `cv_power_forecasts` calls `predict` once per 14-day
  chunk, so a fold with unseen cells logs one warning per affected chunk. On the leaderboard fold,
  two series with short histories (series 12 and 13, departure 4) leave 12,463 of the 518,294
  validation half-hours that have cleaned power (2.4%) in unseen cells, all in August to December
  2025. The warning fires in about 12 of the roughly 28 chunks: those whose runs reach August to
  December 2025. The warnings count rows per NWP run, so their sum exceeds that half-hour count.
- **An empty chunk** gives a zero-row, correctly typed `PowerForecast` with no special branch. The
  first chunk is written even when empty, because `cv_power_forecasts` uses the first write to
  overwrite the partition.
- **A re-materialised fold** reuses its MLflow run. `save` clears the model directory before
  writing, and the first chunk's `replace_partition` overwrites the earlier forecasts.

#### The main error-handling paths

**Climatology is an R&D baseline, so it raises only on our own bugs and degrades on a missing
cell.**

- **A non-empty `selected_features`** raises `ValueError` in `ClimatologyForecaster.__init__`. The
  error surfaces when `trained_cv_model` constructs the forecaster, because registration validates
  only `CONFIG_CLASS`.
- **No eligible series has training power** gives an empty lookup, and `trained_cv_model` raises its
  existing "Trained 0 of N eligible time series" error before anything is saved.
- **A loaded model with no trained series** raises in `cv_power_forecasts`' existing guard.
- **A single-run call** (`power_fcst_init_time` given, as `live_forecasts` would make for a promoted
  climatology model) raises `NotImplementedError` in `NwpRunRowsWithoutWeatherFeatureEngineer`,
  whose message names the class and says baselines are not served live.
- **A frame that breaks the `PowerForecast` contract** raises in `PowerForecast.validate`, which
  would mean a bug in `predict`.
- **An unseen cell** never raises: the rows are dropped and the warning above names the count and
  the series, so the leaderboard's row-set difference is visible in the logs.

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
- **Admits more than one defensible design — yes.** The pooling neighbourhood, the member count, the
  fallback for an unseen cell, the quantile interpolation, whether the ensemble size is a config
  field, what to extract for sharing, and whether to rename the engineer all had alternatives.
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
   both extractions to PR B (persistence). The roadmap names persistence as "the second forecaster
   to need" the engineer, because the roadmap did not expect climatology to use the engineer. For
   the helper, the roadmap counts only persistence and climatology, says "`climatology` lands
   first, so PR B extracts it", and leaves the manual heuristic's own copy out of the count. PR B is
   deferred to v0.9 (#1087), and climatology is the second caller of both pieces after the manual
   heuristic.
2. **The engineer is renamed to `NwpRunRowsWithoutWeatherFeatureEngineer`.** The current name,
   `PowerLagsPerNwpRunFeatureEngineer`, says it engineers power lags. Climatology requests none, so
   the name would be false for the second caller. The new name says what the engineer delivers: one
   row per series, NWP run, and valid time, with no weather value read.
3. **The repetition the dedupe removes is once per covering NWP run, not once per run and member.**
   The roadmap says bulk-mode `AllFeatures` repeats each target "once per covering `(nwp_init_time,
   ensemble_member)`". The key-only engineer drops the member axis, and `trained_cv_model` loads
   member 0 only, so a training target repeats once per run covering it: up to 15 times at a 15-day
   horizon. The dedupe is still required.
4. **A cell's own samples are slightly wider-ranging than the roadmap states, two series have far
   fewer cells, and pooling multiplies the typical count by about eight.** The roadmap says a weekend
   cell holds about 9 to 17 samples and a weekday cell about 21 to 43 "over the ~14-month training
   window". The training window, 2024-04-01 to 2025-06-30, is 15 months. Computed in local time from
   the first post-hindcast valid time (2024-04-01 09:30 UTC) to the window's end, a series with
   complete data has 8 to 19 samples of its own per weekend cell and 20 to 45 per weekday cell.
   April, May, and June are covered twice (weekday cells 41 to 45, weekend 16 to 19). July to March
   are covered once (weekday 20 to 24, weekend 8 to 10). Measured on the cleaned power of the 31
   eligible series, 29 series populate all 1,152 cells from their own samples. Their own minimum is
   8 samples per cell, except series 26 and 29 (7) and series 28 (6). Pooled over the nine-cell
   neighbourhood, those 29 series hold at least 66 samples per cell, with a median of 156. Series
   12, whose cleaned history starts on 2025-02-06, populates 482 cells from its own samples and 676
   once pooled, out of 1,152. Series 13, whose cleaned history starts on 2024-12-11, populates 583
   cells from its own samples and 781 once pooled. Both have a minimum of 1 pooled sample per cell:
   for example, series 12's only July samples are the readings labelled 23:00 and 23:30 UTC on
   30 June 2025, which fall on 1 July in local time, and an August cell near midnight pools only
   those.
5. **The ensemble has 51 members, held as a module constant, where the PR C item says "Keep
   `m = 13`".** The item asks for empirical justification before raising m. Decision 4 is that
   justification, and the maintainer approved the change.
6. **Each cell pools its eight neighbours, where the PR C item leaves pooling adjacent months as "a
   possible refinement if the numbers look ragged".** The same measurement (decision 4) found that
   pooling improves every tail score it measured for substations and wind.
7. **Only the PR C item of the `Implementation details — baselines` section is deleted at ship.**
   The PR C item tells its PR to delete the whole section. The section also holds the
   metrics-collapse item for the still-open
   [#1077](https://github.com/openclimatefix/nged-substation-forecast/issues/1077), the deferred
   PR B item, and the unverified re-run recipe. The heading stays, so its anchor and the inbound
   link at "Implemented as part of the baseline work" survive, and the section is deleted when #1077
   ships. See "Docs to update".

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
  (the PR B item, which stays on the page).

**The refactor must not change the manual heuristic's behaviour.** `meta.json` keeps the same three
keys and the same `json.dumps` call. The existing `test_save_then_load_round_trips_and_replaces_the_directory`
and `test_engineer_rows_equal_the_tabular_rows_deduplicated_across_members` stay unchanged apart from
the import, and must pass before and after.

### 2. `train()`: the cells, the dedupe, the pooling, the quantiles, and the leakage argument

**`train(data, time_series_ids)` builds one lookup row per populated cell, in eight steps:**

1. Select `time_series_id`, `valid_time`, and `power`. Keep rows whose `time_series_id` is in
   `time_series_ids` and whose `power` is not null.
2. Deduplicate on `(time_series_id, valid_time)`. `power` is the observation at `valid_time`, so
   every duplicate carries the same value and `keep="any"` is correct.
3. Collect the deduplicated three-column frame once, with `engine="streaming"`: 639,551 rows on the
   leaderboard fold. Read the earliest and latest `valid_time` for the log line from this frame.
   Steps 4 to 6 run on one batch of series at a time, taken from this frame.
4. Add the three cell keys from one shared helper, `_local_calendar_cell_keys`, which `predict`
   calls too, so the two cannot drift: `local_month` (`Int8`, 1 to 12), `local_half_hour_of_day`
   (`Int8`, 0 to 47), and `local_is_weekend` (`Boolean`, local Saturday or Sunday). Each derives
   from `valid_time` converted to
   `DEFAULT_LOCAL_TIMEZONE` (Europe/London, from `ml_core.features.feature_engineer`), so a
   23:30 UTC target in British Summer Time falls in the next local day's 00:30 slot. The pipeline's
   own `local_*` features are sine and cosine pairs plus a weekday enum, with no integer month or
   half-hour, so the forecaster derives the keys itself, as the roadmap says. Then project to the
   five narrow columns `time_series_id`, `power`, and the three keys, dropping `valid_time`.
5. Pool: cross-join the samples with `CLIMATOLOGY_POOLING_OFFSETS`, a nine-row frame of every
   `(month_offset, half_hour_offset)` pair with each offset in −1, 0, and +1. Both offset columns
   are `Int8`, so the shifted keys keep the `Int8` dtype `_local_calendar_cell_keys` returns and
   join `predict`'s keys without a cast. Replace the keys by
   the keys of the cell each copy feeds: `local_month` becomes `(local_month − 1 + month_offset) %
   12 + 1`, `local_half_hour_of_day` becomes `(local_half_hour_of_day + half_hour_offset) % 48`,
   and `local_is_weekend` is left unchanged. Polars' integer `%` is floor modulo (checked on Polars
   2.0.0: `-1 % 12` is 11, and `-1 % 48` is 47), so December and January are neighbours, and so
   are half-hours 47 and 0. Drop the offset columns.
6. Group by `time_series_id` and the three keys. Aggregate one column per member,
   `power_quantile_member_00` to `power_quantile_member_50`, each
   `pl.col("power").quantile(level, "linear")` cast to `Float32`, plus `pl.len()` as the pooled
   sample count for the train log line described below. Collect the batch's lookup. The count is
   dropped before the lookup is stored.
7. Concatenate the batches' lookups, and sort by the four key columns so the saved parquet is
   deterministic.
8. Set `trained_time_series_ids` to the sorted distinct `time_series_id` values in the lookup. That
   set equals "requested and has at least one non-null `power`", the manual heuristic's and
   XGBoost's rule, because pooling never removes a series.

**The neighbourhood is defined on the cell keys, not on adjacent instants, and the plan implements
exactly the neighbourhood that was measured.** Two consequences follow, and both match
`clim_members.py`. First, the half-hour wrap keeps the original sample's weekend flag and month: a
weekday 23:30 sample in January feeds the weekday half-hour-0 cells of December, January, and
February, even when the next instant is a Saturday. A weekday cell then always pools weekday
samples, which is the property the weekend flag exists for. Second, a weekend sample never feeds a
weekday cell, and the reverse. The lookup key stays the four columns `(time_series_id,
local_month, local_half_hour_of_day, local_is_weekend)`. A lookup row is keyed by the cell a
forecast falls in, and holds the quantiles of that cell's nine-cell neighbourhood, so `predict`
joins exactly as before.

**Pooling costs nine copies of the training samples, and the copies cannot be avoided for exact
quantiles.** A quantile of a union of samples cannot be assembled from per-cell summaries, so each
cell's aggregation must see the full set of its pooled samples. The quantile aggregation holds each
group's values in memory whatever engine runs the collect, so the plan does not claim the pooled
group-by streams. Measured on Polars 2.0.0, the pooled group-by peaks at about 65 bytes per pooled
row: 0.54 GB for 0.64 million samples, 4.1 GB for 6.4 million, and 11.6 GB for 20 million samples
(180 million pooled rows). On the leaderboard fold, the 639,551 deduplicated samples become 5.76
million pooled rows, a peak of about 0.5 GB. At V2 scale, about 55 million samples would become
about 490 million pooled rows and a peak of about 32 GB in one group-by, above the workstation's
29 GB.

**`train` therefore pools in batches of series, and the batches give the same lookup as one
group-by.** Every pooled group sits inside one series, because `time_series_id` is a group-by key
and the cross join never changes it. Splitting the series into batches therefore changes no
group's samples. A module constant, `_POOLING_SERIES_PER_BATCH: Final = 100`, sets the batch size.
At V2 a series holds about 22,000 samples, which pool to about 200,000 rows and about 13 MB, so a
batch of 100 series peaks at about 1.3 GB. At V1 the 31 series fit in one batch. The loop is a
`for` over `itertools.batched` of the sorted series ids, filtering the collected frame with
`is_in`, about 6 lines. Collecting the deduplicated frame once (step 3) means the engineered frame
is built once, however many batches follow. Test 10 pins that batching changes nothing.

**The quantile interpolation is Polars' `"linear"`, passed explicitly.** `"linear"` is
Hyndman and Fan's type 7: for n samples, level p reads position (n − 1)·p between the sorted
samples. `metrics.py` reads the delivery quantiles from members with the same method, so the
metrics layer and the forecaster share one definition. The method never extrapolates beyond the
smallest and largest sample, so a sparse cell's tail members sit at the observed extremes. Polars'
default is `"nearest"`, so an implementer who omits the argument gets a different, step-valued
answer: test 2 fails that mutation. Polars 2.0.0 rejects a list of levels inside `group_by().agg`,
so the 51 levels are 51 expressions.

**The levels are `(i − 0.5)/51` for i = 1 to 51, held as module constants.**
`CLIMATOLOGY_MEMBER_COUNT: Final = 51`, and `CLIMATOLOGY_QUANTILE_LEVELS` is derived from it. The
level for member k (0-based) is `(k + 0.5)/51`, so member 25 is exactly the median.

**A sparse cell is kept, and a cell with no sample anywhere in its neighbourhood gets no lookup
row.** Recommendation: no minimum-sample threshold and no wider fallback in this PR. Pooling answers
the sparse-cell problem for the 29 series with a full training history: their smallest pooled cell
holds 66 samples, against 6 before pooling. Pooling does not answer the sparse-cell problem for
series 12 and 13, whose short histories (departure 4) leave some cells populated only from a
neighbour, down to 1 pooled sample. A cell with 1 sample yields 51 equal members, a degenerate but
honest ensemble whose fair CRPS equals its mean absolute error (MAE). Climatology is an R&D
baseline, so the production rule to always emit a forecast does not bind it. A dropped row is
logged and counted rather than silently missing. A wider neighbourhood would not rescue series 12:
its training history has no sample from August to January, and a one-month neighbourhood cannot
fill September to December (risk 1).

**`train` logs one aggregate line**: the number of series and cells, the minimum and median number
of pooled samples per cell, and the earliest and latest deduplicated `valid_time`. The real-data run
reads the sample counts to judge sparse cells, and the latest `valid_time` to confirm that no
validation target reached the lookup. On the leaderboard fold the line should report a minimum of 1
(series 12 and 13) and a median of 156.

**The training never sees validation data, and pooling does not change that.** `trained_cv_model`
hands `train` features engineered from `load_engineering_inputs` over `[train_start, train_end]`
with `power_lookback` zero. The power scan is bounded to `time <= train_end` and the NWP scan to
`valid_time <= train_end`, so every training target, and every `power` value on it, lies inside
2024-04-01 to 2025-06-30 for the leaderboard fold. Pooling moves a training sample to a neighbouring
cell's key, never to a validation time: a validation-year cell such as series 12's August is filled
only from training-window samples. Truth is the cleaned power table (`scan_cleaned_power`), so a
flagged reading is neither a training sample nor a scoring target. Bank holidays are ordinary days,
as for the manual heuristic: Christmas Day on a weekday falls in that month's weekday cells (see
"Should bank holidays be weekend-type days?" below). `predict` reads only the lookup and the
calendar keys of each validation row. The validation rows' own `power` column is present in the
engineered frame, and `predict` must not read it: test 9 pins that.

#### Should the cells be keyed on local time or on UTC?

**Recommendation: keep local keys for every series in this PR.** The maintainer asked whether UTC
keys would be better, so the planning session measured both keyings on the leaderboard fold. The
comparison was measured on 13 unpooled members, before the pooling and member-count change, and has
not been repeated. Pooling blurs each cell by a half-hour either side, which may shrink the March
and October PV cost below.

**Local and UTC keys put a half-hour in different cells only in March and October, and in the
23:00 to 24:00 UTC hour during summer time.** March and October are the clock-change months, where
a local cell mixes days on GMT with days on BST. The own-sample counts per cell are identical under
the two keyings: 8 to 19 per weekend cell and 20 to 45 per weekday cell.

**UTC keys improve PV by about 0.5% and leave substations and wind slightly worse.** The table
gives the change in fair CRPS against local keys, negative being better. The CRPS is taken over 13
unpooled members, normalised by effective capacity, and scored on the 499,932 rows that every keying
forecasts:

| Keying | Substations | PV | Wind |
|---|---|---|---|
| UTC, all months | +0.11% (9 of 20 series better) | −0.51% (6 of 6 better) | +0.05% (0 of 3 better) |
| UTC, March and October rows only | +0.58% | −2.6% | +0.14% |
| UTC half-hour with the local weekend flag | +0.08% | −0.51% | +0.04% |
| UTC for PV and wind, local for the rest | 0 | −0.51% | +0.05% |
| Local keys split by summer time | +1.8% | +2.3% | +2.2% |

Under UTC keys in all months, the BESS and other series change by −0.06%. Pooled over all 31
series, UTC keys change the CRPS by about −0.03% against local keys, which is a wash. Splitting
local keys by summer time is worse everywhere, and about 11% worse in March and October, because
the summer-time half of March holds only one or two training days.

**Three reasons keep local keys.** Substations are the primary target, and the substation load
still scores better on local keys. UTC keys for PV and wind alone would need `time_series_type`
inside the forecaster, which reaches a forecaster only through `selected_features`, and so would
break the rule that climatology's `selected_features` is empty (decision 4). The manual heuristic's
UTC lags give no reason to match them: the roadmap and the package README already call those lags
a departure from the operator's local-clock method.

**The README states the PV cost.** In March and October a PV cell mixes days an hour apart in solar
time, which cost about 2.6% CRPS in those two months at 13 unpooled members. Switching PV and wind
to UTC keys is left to the maintainer as a cheap follow-up (risk 11).

The evidence is three read-only scripts in the planning session's scratchpad, not committed:
`clim_keying.py` and `clim_keying2.py` score the keyings, and `clim_counts.py` counts the samples
per cell and the dropped validation rows per series before pooling. `clim_counts_pooled.py` repeats
the counts with pooling.

#### Should bank holidays be weekend-type days?

**Decision: keep the plain local weekday and weekend day type in this PR, so bank holidays,
Christmas, and Easter are ordinary days.** The maintainer decided this after a read-only measurement
on the leaderboard fold, whose alternatives and numbers are in "Considered and rejected". Christmas
Day on a weekday therefore falls in December's weekday cells, as in the manual heuristic.

**The best alternative overall improves substations by only 0.33% over all rows, and needs a
holiday calendar.** That alternative, B in "Considered and rejected", treats bank holidays as
weekend-type days. B helps on Christmas bank holidays but worsens the upper tail on the other bank
holidays and worsens PV. Alternative C, which adds 24 December to 1 January to B, improves
substations by 0.49% over all rows but is 4.6% worse on the Christmas-week weekdays that are not
bank holidays. The calendar would be a hand-kept list of dates or the `holidays` package, which
would change `uv.lock`. A calendar-aware day type is the design question of
[issue #1088 (Implement the manual_heuristic_holiday_aligned baseline forecaster)](https://github.com/openclimatefix/nged-substation-forecast/issues/1088),
so treating bank holidays as weekend-type days in climatology belongs with that issue.

### 3. `predict()`: join, unpivot, drop, and stamp

**`predict(data, *, fold_id="live")` in five steps:**

1. Collect the input with `engine="streaming"`, selecting only `time_series_id`,
   `power_fcst_init_time`, and `valid_time`, and strip the Patito model with `.as_polars()`.
2. Add the cell keys with `_local_calendar_cell_keys`, and inner-join the lookup on the four key
   columns (both sides plain Polars frames, per `polars-patito-gotchas`).
3. Count the input rows the join dropped and, when the count is positive, log one warning naming
   the count and the `time_series_id`s affected. Nothing raises. `cv_power_forecasts` calls
   `predict` once per 14-day `init_time` chunk (`_PREDICT_INIT_CHUNK`), so the warning is one per
   affected chunk, not one per fold.
4. Unpivot the 51 quantile columns into `ensemble_member` and `power_fcst`, mapping each column
   name to its member index with `replace_strict(..., return_dtype=pl.Int8)`, so member 0 is the
   lowest quantile on every row.
5. Add `nwp_init_time` as a typed null, and the identity literals exactly as the manual heuristic
   builds them: `power_fcst_model_name`, `power_fcst_model_version` (`Int16`),
   `ml_flow_experiment_id` (`Int32`), `experiment_name`, and `fold_id`. Return
   `PowerForecast.validate(...)`.

**Empty input returns a valid empty `PowerForecast`.** The join, unpivot, and literals give a
correctly typed zero-row frame. Test 7 pins this rather than adding a branch.

**A climatology chunk is as large as an XGBoost chunk, which `_PREDICT_INIT_CHUNK` was sized for.**
On the leaderboard fold, the largest 14-day chunk of `xgboost_baseline_retrain_790` holds 15.5
million rows (31 series, 51 members), and the manual heuristic's largest holds 3.7 million (13
members). Climatology's largest chunk will hold up to about 15.5 million rows, less the dropped
rows. The `cv_power_forecasts` docstring puts the per-chunk forecast frame at 2 to 3 GB for 51
members. The joined frame before the unpivot is small: about 300,000 rows of 51 `Float32` columns,
about 60 MB. The manual heuristic's prediction for the whole fold took 39 seconds and peaked at
2.9 GB, so climatology's prediction is expected to take longer and peak higher. The real-data run
records both (decision 6).

**Nothing in `PowerForecast`, `compute_metrics`, or the dashboard needs a new column or schema for
a member that is a quantile sample.** Checked against the code:

- `PowerForecast.ensemble_member` is `Int8`, and members 0 to 50 fit. The primary key
  `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)` is unique, because the
  engineer emits one row per series, run, and valid time.
- `compute_metrics` groups each `(time_series_id, power_fcst_init_time, valid_time)` and treats the
  members as an equiprobable sample: the ensemble mean, the fair CRPS, the Fortin-corrected
  variance, and `quantile(q, "linear")` for each delivery quantile. Equiprobable quantile values are
  exactly the "equiprobable pseudo-sample" `docs/techniques/probabilistic-forecasting.md` describes
  for pooling quantile forecasts.
- `packages/dashboard/view_forecasts.py` draws each `ensemble_member` as a thin grey line, which
  works unchanged, with 51 lines per run in place of 13.
- The `ensemble_member` description is the one edit: it names the NWP member and the manual
  heuristic's analogue rank, and gains a sentence for climatology. The description on `main` does
  not yet mention climatology. The brief's claim that it already promises a climatology sentence
  "from PR C" is stale: that wording was in the manual-heuristic plan, not in the merged code.

### 4. The members: 51 equiprobable quantiles of each pooled cell

**Recommendation: 51 members at the equiprobable levels `(i − 0.5)/51`, taken over each cell's
nine-cell neighbourhood, held as module constants and not config fields.** The maintainer approved
this after a measurement on the leaderboard fold. The measurement trained on the cleaned power of
the 31 eligible series from 2024-04-01 09:30 UTC to 2025-06-30, and scored the 500,010 validation
half-hours from 2025-07-01 to 2026-06-30 that a 13-member unpooled lookup forecasts, so every
variant was scored on the same rows. The table gives the 20 substations (the disaggregated-demand
and raw-flow series). Each score is the group mean of the per-series change against 13 unpooled
members, negative being better. The last column is the share of rows above the p99 the metrics
layer reads from the members, whose ideal is 1%:

| Members | Pooled over | Fair CRPS | Plain CRPS | Pinball loss at p99 | Lower-tail twCRPS | Upper-tail twCRPS | Rows above p99 |
|---|---|---|---|---|---|---|---|
| 13 (reference) | own cell | 0 | 0 | 0 | 0 | 0 | 20.0% |
| 51 | own cell | +3.8% | −0.3% | −16.3% | −0.7% | −0.4% | 16.9% |
| 101 | own cell | +4.5% | −0.3% | −18.6% | −0.7% | −0.4% | 16.4% |
| 13 | 3 cells (months) | −4.6% | −2.7% | −35.7% | −9.0% | −2.3% | 11.5% |
| 13 | 9 cells | −5.3% | −3.1% | −41.6% | −8.8% | −2.6% | 10.0% |
| **51** | **9 cells** | **+0.0%** | **−3.4%** | **−53.3%** | **−9.8%** | **−3.3%** | **6.4%** |
| 13, normal distribution | own cell | −2.8% | −2.1% | −18.4% | +9.0% | +0.5% | 16.1% |
| 13, at the delivery levels, weighted | own cell | not defined | −0.1% | −21.0% | +0.6% | 0.0% | 15.9% |

The reference's absolute scores, as a percentage of effective capacity, are 10.04 fair CRPS, 10.50
plain CRPS, and 2.69 pinball loss at p99. Plain CRPS is the CRPS of the members taken as the
predictive distribution, with no small-ensemble correction. The threshold-weighted CRPS (twCRPS)
scores only the part of the distribution beyond a per-series threshold: the series' training p05
for the lower tail, and p95 for the upper tail. Both twCRPS columns use the plain form. The weighted
delivery-level variant weights each member by the probability gap its level represents, which the
metrics layer cannot do. The normal variant is clipped at zero for series that never go negative.

**Pooling improves the tails most, and 51 members improve them further.** At 13 members, pooling
over nine cells improves the substations' pinball loss at p99 by 42% and the lower-tail twCRPS by
9%. At 51 pooled members, the p99 pinball loss improves by 53% for substations, 59% for wind, and
45% for PV, and the substations' p99 is exceeded on 6.4% of rows, against 10.0% at 13 pooled
members and 20.0% at 13 unpooled. For substations and wind, the 51-member pooled variant has the
best plain CRPS, mean pinball loss, p99 pinball loss, and p99 exceedance of the variants measured.
For PV, pooling over the months alone at 51 members is up to 1.4 percentage points better on the
plain CRPS, the mean pinball loss, and the p99 pinball loss, up to 3.4 percentage points better on
the p95 pinball loss, and worse on the p99 exceedance. For
the two series in the other group, a battery and a biofuel generator, 13 members pooled over nine
cells score better on the plain CRPS and the mean pinball loss, and 51 members score better on the
p99 pinball loss and the p99 exceedance.

**Why 51 members.** The ECMWF ENS that XGBoost's members come from has 51 members, and Phase D's
linear pool of 51 XGBoost members ([#262](https://github.com/openclimatefix/nged-substation-forecast/issues/262),
[#263](https://github.com/openclimatefix/nged-substation-forecast/issues/263), and
[#264](https://github.com/openclimatefix/nged-substation-forecast/issues/264) in
[Phase D](https://openclimatefix.github.io/nged-substation-forecast/roadmap/metrics-and-leaderboard/#phase-d-ensemble-of-quantile-forecasts-representation-3-pooled-representation-2))
emits an equiprobable set of the same kind. Climatology and XGBoost therefore compare at equal m on
the size-dependent metrics: PICP, pinball loss, interval width, and the exceedance rate. The
13-member manual heuristic no longer compares at equal m, and the README says so. More than 51
members gained little: at 101 unpooled members, the substations' p99 pinball loss improves by 18.6%
against 16.3% at 51, and the plain CRPS barely moves, for twice the rows.

**Why equiprobable levels.** `compute_metrics` treats members as equiprobable samples: the ensemble
mean, the fair CRPS, the Fortin-corrected variance, and the delivery quantiles read with
`quantile(q, "linear")` all weight every member equally. Members at tail-heavy levels would be read
as a wider-tailed distribution than the climatology they represent. The pooling recipe in
`docs/techniques/probabilistic-forecasting.md` is also exact only for equally spaced levels.

**The fair CRPS cannot choose the member count, because the fair CRPS rewards a small set of
quantiles.** The fair CRPS corrects the spread term on the assumption that members are independent
draws. A set of evenly spaced quantiles is not a set of independent draws, so the correction
favours the set, and the favour shrinks as m grows. Measured on the leaderboard fold, the fair CRPS
of 13 unpooled members reads about 4% below the plain CRPS of 101 members for substations, 6% for
wind, and 7% for PV, and about 0.5% to 2% below at 51 members. The fair CRPS therefore gets worse
from 13 to 51 members (+3.8% for substations), while the plain CRPS improves only 0.3% from 13 to
101. The CRPS is dominated by the body of the distribution, and the member count matters most in the
tails, which are this project's priority.

**On the leaderboard's fair CRPS, the chosen climatology reads the same as 13 unpooled members.**
For substations, 51 pooled members change the fair CRPS by +0.04% against 13 unpooled members, and
read 5.6% worse than 13 pooled members, though their plain CRPS is 3.4% better than 13
unpooled. A reader comparing `crps__all__extended_range` across forecasters needs that, so the
README states the fair CRPS bias.

**Most of the remaining tail miscalibration is year-to-year variation, which no member
representation fixes.** The best variant still exceeds its p99 on 6.4% of substation rows, and
falls below its p01 on 3.1%, against an ideal of 1% each. The validation year's power differs from
the training year's in level and in extremes, and a distribution of the training year cannot know
that. The README states this too.

**A normal distribution was never the best variant.** For substations, 13 members drawn at the
equiprobable levels of a normal distribution fitted to each cell beat 13 unpooled empirical members
by 2.8% on the fair CRPS, but are worse in the lower tail by 7% to 9% on twCRPS, and 57% to 69%
worse on the fair CRPS for the battery and the biofuel generator, whose power is bimodal. The pooled
empirical members beat the normal variants on almost every score.

**Constants, not config fields.** `BaseForecasterConfig` is `extra="forbid"`, so a config field
would need a `ClimatologyConfig` subclass. That subclass would join
`tests/test_forecaster_config_serialisation.py`'s parametrised checks through an explicit import,
and every saved config would carry a value nobody varies. `CONFIG_CLASS = BaseForecasterConfig`, so
the serialisation test needs no change.

**Phase D needs no schema change from this PR.** Once #262 adds the percentile columns of
[Representation 2](https://openclimatefix.github.io/nged-substation-forecast/roadmap/delivery-tables/#representation-2-percentiles)
to `PowerForecast`, climatology's natural Representation 2 output is the same equiprobable set of 51
quantiles it emits as members today. This PR emits members only.

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
half-hours × 2 day types × 51 quantiles × 4 bytes is 2.88 million rows holding 147 million
`Float32` values, about 590 MB, plus the 3 small integer or Boolean keys and the `Int32` series id,
about 20 MB. On disk the parquet is smaller. The V1 lookup on the leaderboard fold is 34,865
populated cells × 51 quantiles, about 7 MB. The archive `save_to_mlflow` writes holds one model file
plus the frozen metadata, so the lookup adds nothing to the merge problem `_MLFLOW_MODEL_ARTIFACT`
documents. By comparison, the 40 XGBoost boosters measured on that constant total 77 MB, which
scales to several GB at V2.

**The stored forecasts are about 4 times the manual heuristic's.** On the leaderboard fold,
`manual_heuristic` stores 94.1 million rows (89 MB on disk) and `xgboost_baseline_retrain_790`
stores 404.9 million rows over 51 members (713 MB). Climatology emits 51 members on the full
XGBoost row set less its unseen cells, so it will store about 395 million rows. On disk the
partition is likely to fall between the manual heuristic's 0.95 bytes a row and XGBoost's 1.8 bytes
a row: about 0.4 to 0.7 GB. The `forecast_metrics` rows do not grow, because the metrics are per
series and horizon slice, not per member.

### 6. The run on real data, after implementation

**Run climatology on the single leaderboard fold, `mid_2025_to_mid_2026`, for the 31 eligible
series, and compare it with `manual_heuristic` and `xgboost_baseline_retrain_790`.** The procedure
copies the manual-heuristic run, so the shared data folder gains only climatology partitions:

- Check CPU load first, and confirm no other session is writing `power_forecasts` or
  `forecast_metrics`. Check free disk space for about 0.7 GB of new `power_forecasts` partition.
- In the worktree, symlink `.env`, the shared `data` folder, `mlflow.db`, and `mlruns` from the
  primary checkout, and set `DAGSTER_HOME` to a fresh untracked directory inside the worktree.
- From a scratch driver in the session scratchpad, modelled on
  `scripts/forecasting/run_baseline_experiment.py` and never committed: register experiment
  `climatology` from `conf/model/climatology.yaml` with `config_overrides={}` and
  `run_mode="full_cv"`, materialise `trained_cv_model` and `cv_power_forecasts` for partition
  `climatology__mid_2025_to_mid_2026`, then `metrics` with
  `PopulationFilter(experiment_name="climatology", fold_id="mid_2025_to_mid_2026")`. Nothing
  touches the `xgboost_*` or `manual_heuristic` partitions or MLflow experiments.
- Before reading results, record per series the number of populated cells out of 1,152, from the
  saved lookup. Expect 1,152 cells for the 29 full-history series, 676 for series 12, and 781 for
  series 13. Then count the `(time_series_id, valid_time)` pairs of the cleaned truth in the
  validation window that XGBoost forecasts and climatology does not. Expect 12,463 pairs, all in
  series 12 and 13. A mismatch in either count means the pooling differs from the measured
  neighbourhood. Record the summed counts of the per-chunk `predict` warnings separately, as rows
  per NWP run. Those counts cannot equal 12,463, because `predict` counts one dropped row per
  series, NWP run, and valid time, including valid times with no cleaned power. Record the minimum and
  median pooled samples per cell from the train log line (expected 1 and 156). Confirm that the
  latest training `valid_time` in the train log is no later than 2025-06-30 23:59:59 UTC, and that
  every forecast row has 51 members.
- Re-score all three experiments in a scratch script with the repo's `compute_metrics`, the same
  cleaned truth (`scan_cleaned_power`), and the same effective capacity, on the same series, writing
  nothing back. Report by group — substations (Primary, bulk supply point, and grid supply point),
  PV, and wind — with group means and the "beats" count per series, as PR #1057 did. The headline
  is `crps__all__extended_range`, with NMAE in each horizon slice. Report absolute values, not only
  contrasts. Report the pinball loss at p99 and the p99 exceedance rate beside the CRPS, because the
  fair CRPS understates the member count's effect on the tails (decision 4).
- Score all three experiments on the intersection of rows that all three forecast, always, not
  only if a row was dropped. Series 12 and 13 have unseen cells holding about 41% and 31% of their
  validation rows (departure 4), so climatology's own `forecast_metrics` rows score those two
  series on about seven months of the year while the other two experiments are scored on a full
  year. Series 12's and 13's "beats" counts are scored on the shared rows and kept in the count,
  because dropping them would hide that climatology cannot forecast part of their year, and the
  shared rows keep the comparison like for like. Report the 29 full-history series as a separate
  group as well, so the two short-history series cannot move the group means.
- Record time and peak memory for training and prediction, the saved lookup's size, and the
  `power_forecasts` partition's row count and size on disk.

**Four caveats go wherever the comparison is read.** First, a set of quantiles at equiprobable
levels slightly out-scores an ensemble of independent draws of the same size under the fair CRPS
([Ferro (2014)](https://doi.org/10.1002/qj.2270)), so climatology has a small structural edge: read
a near-tie at extended range as "the NWP ensemble adds little out here", not as climatology winning.
At 51 members the edge measured about 0.5% to 2% of the CRPS (decision 4). Second, the 13-member
manual heuristic and the 51-member climatology are not at equal m, so their PICP, pinball loss,
interval width, and exceedance rates differ partly because of the member count. Third,
`xgboost_baseline_retrain_790` was trained on power that was not yet cleaned, with code older than
`main`. The re-score scores its forecasts against the same truth with the same code, but a fair
XGBoost comparison still needs a retrain. Fourth, one fold of one year, with no intervals.

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
- **`climatology.py` (new).** `CLIMATOLOGY_MEMBER_COUNT: Final = 51`,
  `CLIMATOLOGY_QUANTILE_LEVELS: Final[tuple[float, ...]]` derived from it, the 51 quantile-column
  names, `CLIMATOLOGY_POOLING_OFFSETS` (the nine `(month_offset, half_hour_offset)` pairs as two
  `Int8` columns, each offset in −1, 0, and +1), `_POOLING_SERIES_PER_BATCH: Final = 100`,
  `_local_calendar_cell_keys` (one function returning the three key expressions, `Int8`, `Int8`,
  and `Boolean`, from a `valid_time` expression and a time zone), a module `logger`, and
  `ClimatologyForecaster` with `MODEL_NAME = "climatology"`, `MODEL_VERSION = 1`, `CONFIG_CLASS =
  BaseForecasterConfig`, `feature_engineer = NwpRunRowsWithoutWeatherFeatureEngineer()`,
  `__init__`, `trained_time_series_ids`, `train`, `predict`, `save`, and `load` as described above.
- **`__init__.py`.** Exports `ClimatologyForecaster` beside `ManualHeuristicForecaster`, and the
  module docstring names both.

### `conf/model/climatology.yaml` (new)

**`_target_: baseline_forecasters.climatology.ClimatologyForecaster`, with `selected_features: []`,
`weather_source: "none"`, and `training_strategy: "none"`.** The header comment follows
`manual_heuristic.yaml`'s: the forecaster reads no feature, the cells, the nine-cell pooling, and
the 51 levels are fixed in code, and a non-empty list raises when `trained_cv_model` runs.

### `packages/contracts/src/contracts/power_schemas.py`

**Description text of `PowerForecast.ensemble_member` only.** Adds: "For `climatology`, the index
is the rank of a quantile level, with 0 the lowest: member k is the empirical quantile at level
(k + 0.5)/51 of the series' training power in the valid time's calendar cell and the eight cells
one month and one half-hour away with the same day type." No dtype, nullability, or constraint
changes, so plan approval meets the "ask before changing a Patito data contract" rule.

### Tests

- **`packages/baseline_forecasters/tests/test_climatology.py` (new).** Tests 1 to 12 below, on
  hand-built `AllFeatures` frames.
- **`packages/baseline_forecasters/tests/test_manual_heuristic.py`.** The engineer import becomes
  `from baseline_forecasters.nwp_run_rows import NwpRunRowsWithoutWeatherFeatureEngineer`, and the two
  engineer tests use the new name. No assertion changes.
- **`tests/test_climatology_cv.py` (new).** Test 13, the integration test.
- `tests/conftest.py` needs no change: `register_experiment` already takes `base_model_config` and
  `config_overrides`.

## Design-philosophy check

**The code runs in the R&D asset chain only, and `predict` drops and logs rather than raising.** An
unseen cell drops its rows with one aggregate warning per `predict` call (one per `init_time`
chunk in `cv_power_forecasts`) naming the series, as "Make the telemetry
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
which is dropped and counted: 2.4% of the leaderboard fold's validation half-hours, all in series
12 and 13. Decision 6 therefore scores all three experiments on the rows they share. The key-only
engineer and the pooled group-by keep the work in the query engine (principle 11).

**One trade is inherited from the manual heuristic: the rows follow the NWP archive.** An
`init_time` missing from the archive removes that run's climatology rows, as for every other model,
which keeps the comparison like for like.

## Tests

On `main` the module does not exist, so every new test fails at import. The assertion listed for
each test is the behaviour it pins once the import resolves, chosen so that a plausible wrong
implementation fails it. The fixtures stay small: the hand-computed values below use at most five
samples, and the expected members follow from a one-line formula per cell.

**`packages/baseline_forecasters/tests/test_climatology.py` (unit):**

1. **Construction.** `ClimatologyForecaster(BaseForecasterConfig(selected_features=set()))`
   succeeds. A config selecting `power_lag_168h` raises `ValueError`.
2. **Hand-computed quantiles and the wrap.** One series, one cell, 5 deduplicated samples with
   power 0, 10, 20, 30, and 40, all at local January, half-hour 0, on weekdays. One of the five
   samples is a Monday at local 00:00. A pooling that shifts the instant (`valid_time` by ±30
   minutes or ±1 month) instead of the cell keys puts that sample's −30-minute copy at Sunday
   23:30, in a weekend half-hour-47 cell, and fails the assertion that follows. The lookup holds
   exactly 9 rows: months 12, 1, and 2 × half-hours 47, 0, and 1, all with `local_is_weekend`
   false. Every row's member-k column equals 40·(k + 0.5)/51 for k = 0 to 50, to `Float32`
   precision, so the test pins all 51 levels: member 0 is 0.392, member 25 is 20, and member 50 is
   39.608. A `"nearest"` interpolation gives multiples of 10 and fails. A level formula of k/50
   fails at k = 0. A truncating modulo puts the December or half-hour-47 rows at month 0 or
   half-hour −1 and fails. One further assert checks that the 51 expected values are pairwise
   distinct, so a mislabelled member cannot pass this test and test 4 by coincidence.
3. **The dedupe.** Duplicating a subset of the training rows — the two largest samples, under three
   extra `power_fcst_init_time` values each — leaves the lookup equal (`assert_frame_equal`) to the
   lookup trained on the rows without duplicates. Without the dedupe, the duplicated samples pull
   the interior quantiles upwards, so the assertion fails.
4. **Member order.** On the fixture of test 2, `predict` emits members 0 to 50 for a row in the
   January half-hour-0 weekday cell, and member k's `power_fcst` equals the lookup's member-k
   column, so the values strictly increase with the member index and member 25 is 20. A shuffled
   column-to-member mapping fails.
5. **Local-time cell keys, across both clock changes and at local midnight.** `_local_calendar_cell_keys`
   maps 2025-06-06 23:30 UTC (Saturday 00:30 BST) to June, half-hour 1, weekend; 2025-06-30 23:30 UTC
   (Tuesday 1 July 00:30 BST) to July, half-hour 1, weekday; 2025-01-17 23:30 UTC (Friday 23:30 GMT)
   to January, half-hour 47, weekday; and 2025-03-30 01:00 UTC (02:00 BST, just after the clock
   change) to March, half-hour 4, weekend. At the autumn clock change, 2025-10-26 00:30 UTC
   (01:30 BST) and 2025-10-26 01:30 UTC (01:30 GMT) both map to October, half-hour 3, weekend
   (Sunday), so the doubled local hour lands in one cell. A UTC-keyed implementation fails the
   first, second, and fourth cases, and puts 00:30 UTC in half-hour 1 in the autumn case.
6. **An unseen cell drops its rows and logs; a neighbour-filled cell does not.** On the fixture of
   test 2, three forecast rows: one in February, half-hour 1, weekday, which has no sample of its
   own but a sample in its neighbourhood; one in January, half-hour 0, weekend; and one in March,
   half-hour 0, weekday, two months from the samples. The February row is forecast with the same
   51 values as test 2. The other two rows are absent from the output, the call does not raise, and
   `caplog` holds one warning carrying the dropped count, 2, and the series id.
7. **Empty input.** A zero-row `AllFeatures` frame returns a zero-row frame that
   `PowerForecast.validate` accepts.
8. **Stamping.** Every row carries `power_fcst_model_name == "climatology"`,
   `power_fcst_model_version == 1`, the config's `experiment_name` and `ml_flow_experiment_id`, the
   `fold_id` passed in (a non-default value such as `"fold_x"`), and a null `nwp_init_time`. The
   manual heuristic's mutation testing found that stamping goes untested otherwise.
9. **`predict` ignores the validation power.** Two `predict` calls on the same rows, one with the
   `power` column replaced by large arbitrary values, give equal output. A `predict` that reads
   `power` fails, which is the leakage guard the maintainer asked for.
10. **Train population, and series that do not share cells.** Rows for series 1, 2, 3, and 4, where
    series 2 has only null power, series 3 is not requested, and series 1 and 4 carry different
    power in the same calendar cell, give `trained_time_series_ids == [1, 4]`. The lookup holds no
    cell for series 2 or 3, and the quantiles of series 1 and 4 in the shared cell differ. A
    group-by or a cross join that drops `time_series_id` fails. With `_POOLING_SERIES_PER_BATCH`
    monkeypatched to 1, so series 1 and 4 pool in separate batches, the lookup equals
    (`assert_frame_equal`) the lookup trained in one batch.
11. **Save and load.** `load(save(...))` returns the same `trained_time_series_ids`, an equal config,
    and an equal lookup, and `predict` after the round trip equals `predict` before it. A stale file
    placed in the directory before `save` is gone afterwards. `meta.json`'s `model_class` is
    `"baseline_forecasters.climatology.ClimatologyForecaster"`.
12. **The neighbourhood.** One series, weekday samples in three cells: power 0 and 40 in March,
    half-hour 10; power 20 in March, half-hour 11; and power 100 in May, half-hour 10. A weekend
    sample of power 1,000 sits in March, half-hour 10. Writing p for (k + 0.5)/51, the lookup holds:
    - March, half-hour 10, weekday: the pooled samples 0, 20, and 40, so member k is 40·p.
    - April, half-hour 10, weekday: the pooled samples 0, 20, 40, and 100, so member k is 60·p for
      k ≤ 33 and 180·p − 80 for k ≥ 34: member 33 is 39.412, member 34 is 41.765, and member 50 is
      98.235.
    - June, half-hour 10, weekday: 100 alone, so all 51 members are 100.
    - March, half-hour 12, weekday: 20 alone.
    - March, half-hour 10, weekend: 1,000 in every member, and no weekday member equals 1,000.
    - No row for March, half-hour 13, weekday, or for July, half-hour 10, weekday.

    The train log line reports a minimum pooled count of 1. A neighbourhood of months only fails
    the April and half-hour-12 rows. A neighbourhood of two months either side puts 100 into the
    March cell and fails. A neighbourhood that crosses the weekend flag puts 1,000 into a weekday
    cell and fails.

The existing `test_manual_heuristic.py` suite is the refactor's regression test: it must pass
unchanged apart from the engineer's import and name.

**`tests/test_climatology_cv.py` (integration, `pytestmark = pytest.mark.integration`), test 13:**
modelled on `tests/test_manual_heuristic_cv.py`. Register `conf/model/climatology.yaml` with
`config_overrides={}`. Write power from 30 days before the first training day through Saturday
2025-08-02 12:00 UTC, then `write_cleaned_copy` it, so `scan_cleaned_power` sees every row. The
Saturday run's valid times then carry power, so a `predict` that wrongly dropped rows with null
power cannot pass for the unseen-cell drop. Write the
metadata and the eligible-series table as that module does. Write NWP as: member-0 runs on five
weekdays of August 2024 inside the training window (each giving the five valid times `half_hours`
yields, 11:00 to 13:00 BST, local half-hours 22 to 26); a validation run with members 0, 1, and 2
on Friday 2025-08-01; and a validation run on Saturday 2025-08-02, whose weekend cells have no
training sample in their neighbourhood. Materialise `trained_cv_model` and `cv_power_forecasts`,
then read `power_forecasts` and assert:

- Every row has a null `nwp_init_time`, `fold_id == FOLD_ID`, `experiment_name ==
  EXPERIMENT_NAME`, and `power_fcst_model_name == "climatology"`.
- The rows are exactly the five Friday valid times × 51 members, 255 rows: the three NWP members are
  not repeated, and the Saturday run's rows are dropped without failing the asset.
- Each row's `power_fcst` equals an oracle computed in the test, after the oracle values pass
  through `delta_store.precision.round_to_significand_bits` with
  `keep_bits=POWER_FCST_SIGNIFICAND_BITS` (13, from `delta_store.power_forecasts`). The rounding is
  needed because `write_power_forecasts` rounds `power_fcst` to 13 significand bits, so `==`
  against an unrounded interpolated quantile fails. The oracle pools in plain Python:
  for a validation row in local half-hour h, the training samples are the 25 August 2024 samples
  whose local half-hour is within one of h, cyclically, on a weekday, with the month within one of
  August. Here that is the 10 samples of half-hours 22 and 23 for h = 22, 15 samples for h = 23 to
  25, and 10 for h = 26. The oracle then takes `numpy.quantile(..., method="linear")` at level
  (k + 0.5)/51. `power_at` gives each training half-hour a distinct value. The test asserts that
  each row's 51 rounded oracle values are pairwise distinct, so a mislabelled member fails. The
  validation-day power is never a sample, so a training step that read validation data would fail
  the oracle, and so would a `train` that did not pool.
- **CRPS flows over the members.** The test calls `compute_metrics` on the forecasts it read back,
  with `scan_cleaned_power` as the actuals, the metadata it wrote, and a one-row effective-capacity
  frame. The `horizon_slice="all"` row for `crps` is non-null and differs from the `mae` row, which
  scores the ensemble mean. A `predict` that wrote one member, or 51 copies of one value, gives
  `crps == mae` and fails. This is the roadmap PR C item's "CRPS flows over the members" test,
  without materialising the `metrics` asset and its MLflow logging.

The test fails on `main` at the missing YAML target. The NWP run on a training day is needed because
`trained_cv_model` raises "Trained 0 of 1 eligible time series" without it.

## Docs to update

**Written to describe the code as it is after this PR.**

- **`packages/baseline_forecasters/README.md`.** A `ClimatologyForecaster` paragraph under "What
  each baseline emits": the cells, the nine-cell pooling and its wrap rules, the 51 equiprobable
  levels and why neither the delivery levels nor 13 members, the member order, and `nwp_init_time`
  null. The engineer section renamed, and saying both baselines use the engineer. New caveats:
    - The pooled samples per cell, with the measured counts, and the tail members sitting at the
      observed extremes of the pooled samples.
    - Unseen cells dropped and logged, with the measured drop for a series whose history is shorter
      than the training window (about a third to two fifths of its validation rows), giving no
      series ID in case the series is a metered generator.
    - The fair CRPS bias: a set of quantiles reads lower under the fair CRPS the fewer members it
      has, about 4% to 7% at 13 members and 0.5% to 2% at 51 on the leaderboard fold, so the fair
      CRPS cannot compare member counts, and the 51-member climatology reads no better on the fair
      CRPS than a 13-member unpooled climatology would, though its tails are far better. This is the
      Ferro structural edge in a CRPS comparison with the NWP ensemble.
    - Most of the remaining tail miscalibration is year-to-year variation that no member
      representation fixes: on the leaderboard fold the substations' p99 is still exceeded on 6.4%
      of rows.
    - For the battery and the biofuel generator, 51 pooled members worsen the p01 and p05 pinball
      losses by 18% and 14% against 13 unpooled members, giving no series ID.
    - The 51-member climatology compares with XGBoost's 51 members at equal m, and with the
      13-member manual heuristic at unequal m on PICP, pinball loss, interval width, and the
      exceedance rate.
    - In March and October a PV cell mixes days an hour apart in solar time, which cost about 2.6%
      CRPS in those months at 13 unpooled members (decision 2).
    - Bank holidays are treated as ordinary days.

    The README becomes the permanent home of the PR C design text, including the "why a
    distribution, not a mean" argument.
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
    - "Implementation details — baselines": only the PR C item is deleted, and its design text is
      promoted to the package README and the PR body, rewritten for 51 pooled members. The item's
      "Keep `m = 13`" and "pooling adjacent months is a possible refinement" sentences are not
      carried over. The heading stays, so its anchor survives, and the section is deleted when
      [#1077](https://github.com/openclimatefix/nged-substation-forecast/issues/1077) ships. The
      intro's "One PR is next: `climatology` (PR C)" is rewritten to say that `climatology` has
      shipped, tracked in #1086. The PR B item's sentences about extracting the engineer and the
      `meta.json` helper become one sentence saying that `PersistenceForecaster` reuses
      `NwpRunRowsWithoutWeatherFeatureEngineer` and the shared `meta.json` helper, which this PR
      extracted. Cross-cutting item 1 drops its "PR C is tracked in #1086" clause. The
      metrics-collapse item, "The recipe", the data check, and the re-run recipe stay where they
      are.
    - Grep the page for "PR C", "PowerLagsPerNwpRunFeatureEngineer", "built next", and "m = 13"
      after the edit, so no sentence points at removed text, names the old engineer, or gives
      climatology 13 members.
- **`docs/roadmap/index.md`, v0.3.** "Baseline forecasters (persistence + climatology)" becomes the
  manual heuristic and climatology, with persistence in v0.9 (#1087).
- **`#354` needs nothing from this PR beyond the rows.** The saved forecasts carry 51 members for
  every `(time_series_id, fold_id, power_fcst_init_time)` that the NWP runs give, and the values
  for one `valid_time` are the same across runs, so the band is invariant as #354 expects. Member
  25 is exactly the p50 line #354 asks for. A dashboard calling Polars `quantile(0.1, "linear")` on
  the 51 members reads member 5, at level 0.108, and `quantile(0.9, "linear")` reads member 45, at
  level 0.892. Both sit within 0.01 of the nominal levels, so #354 can use the delivery-quantile
  calls directly. A comment on #354 at ship states the member count, the median member, and those
  two levels.
- **Ship-time triage.** The PR body uses `Closes #1086` and mentions #147 without a closing
  keyword. Once climatology merges, #147's remaining items are the deferred #1087 and #1088 (its
  body says so) and #715, so #147 can close by hand. That close is the maintainer's call (question
  5).

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

Then the real-data run in decision 6, whose numbers go in the PR body. Its expected cell counts,
dropped rows, and pooled sample counts check that the implemented pooling is the measured one.

## Risks and open questions

1. **How should climatology treat a series whose history is shorter than the training window?**
   Series 12 and 13 have unseen cells holding about 41% and 31% of their validation rows even after
   pooling (departure 4). Series 12 has no training sample in six consecutive months, August to
   January, so September to December have no sample in their neighbourhood, and August has only
   the two 1 July samples near midnight. Series 13's August to November are unseen or nearly so.
   The two options are to drop the unseen rows and score every experiment on the shared rows, or to
   leave both series out of the climatology "beats" count. Recommendation: no further fallback in
   this PR. Drop the unseen-cell rows with a logged count, keep sparse cells, score all three
   experiments on the shared rows, and report the 29 full-history series as a separate group
   (decision 6). Climatology's own `forecast_metrics` rows for series 12 and 13 still cover only
   the cells it could forecast, so a leaderboard read straight from `forecast_metrics` compares
   those two series on different row sets. The measurement in decision 4 scored only the rows a
   13-member unpooled lookup forecasts, so the 5,821 rows of series 12 and 13 that pooling newly
   fills, some from a single sample, were not scored.
2. **Rename the engineer?** Recommendation: yes, to `NwpRunRowsWithoutWeatherFeatureEngineer`,
   because the current name is false for climatology. The cost is one example in
   `docs/architecture/code-style.md`. Keeping the old name is defensible only if the maintainer reads
   "PowerLags" as "power and its lags".
3. **Extract the `meta.json` helper now, inside `baseline_forecasters` only?** Recommendation: yes.
   Moving it to `ml_core` for XGBoost too would touch the serving path for about 10 lines.
4. **The `smoke_test` fold now forecasts, but only from neighbouring-month samples.** That fold
   trains on January 2025 and validates on February 2025. Without pooling, every validation row
   would sit in an unseen cell. With pooling, each February cell pools the January samples of its
   half-hour and the two half-hours either side, so every February cell of a series with a complete
   January is populated, by construction and not by a run. The `smoke_test` scores therefore say
   nothing about climatology's quality, and in leaderboard scope `smoke_test` is skipped anyway as
   a non-leaderboard fold. Recommendation: register climatology with `run_mode="full_cv"` for the
   real-data run, because the run is about the leaderboard fold, and say nothing about `smoke_test`
   in the README or the YAML comment.
5. **How should #147 close?** Recommendation: the maintainer closes #147 by hand after this PR
   merges, with a comment pointing at #1087, #1088, and #715. The PR body names #147 without a
   closing keyword.
6. **Power labels are period-ending, so local midnight's slot is the previous evening's last half
   hour.** Keying on `valid_time` as stored puts a Friday-night reading labelled Saturday 00:00 in
   the weekend cells. Recommendation: keep `valid_time` as stored. The same rule applies in `train`
   and `predict`, and it matches the pipeline's `local_*` features and the dashboard.
7. **Training memory and time at V2 scale.** The training frame is about 2,500 series × 21,900
   half-hours × up to 15 runs, around 820 million rows before the dedupe, under the 2³² row-count
   limit. The dedupe leaves about 55 million samples, and pooling copies them to about 490 million
   rows. At the measured 65 bytes per pooled row, one group-by over all of them would peak at about
   32 GB, above the workstation's 29 GB. The plan therefore pools in batches of 100 series
   (decision 2), which caps the pooled group-by at about 1.3 GB at V2, and the collected
   deduplicated frame of about 55 million rows stays in memory beside it. The time at V2 is not
   measured. The pre-dedupe engineered frame of about 820 million rows is not batched, because
   `trained_cv_model` builds it before `train` runs. Question 8 is the structural fix for that
   frame.
8. **Should `train` receive one row per series and valid time, with no NWP runs?** The simplicity
   review proposed a training-specific hook on `FeatureEngineer` that gives `train` one row per
   `(time_series_id, valid_time)`. The hook would remove the dedupe from `train` and shrink a V2
   training frame by up to 15 times, from about 820 million rows to about 55 million. The hook
   needs an edit to `trained_cv_model`, which sits in the out-of-bounds `defs/`, and a change to
   the `BaseForecaster` and `FeatureEngineer` interfaces. Training would also give up "identical
   chain, identical rows": the forecasters would no longer all train on the same engineered rows.
   Recommendation: ship climatology under the current architecture, and leave the hook to
   [issue #1119 (Let a forecaster train on one row per series and valid time)](https://github.com/openclimatefix/nged-substation-forecast/issues/1119),
   gated on measuring the V2 training memory of climatology and the manual heuristic. This is the
   maintainer's call.
9. **`docs/design-philosophy/inherent-stability.md` says the manual heuristic's rows "follow the
   NWP run grid".** That sentence uses "grid" for NWP-run rows, which the naming rule warns
   against. The sentence is outside this issue's scope. Recommendation: fix the word in this PR
   only if the maintainer agrees, otherwise leave it.
10. **The XGBoost comparator is stale.** `xgboost_baseline_retrain_790` predates power cleaning and
    current code. The scratch re-score in decision 6 puts the truth, the effective capacity, and
    the scoring code on one footing for all three experiments, but cannot change what XGBoost was
    trained on. Recommendation: report against it with the narrowed caveat, and leave the retrain to
    its own issue.
11. **Should PV and wind switch to UTC cell keys?** On the leaderboard fold, at 13 unpooled members,
    UTC keys improve PV CRPS by 0.51% over the year and by 2.6% in March and October, and all 6 PV
    series improve (decision 2). Wind changes by +0.05%. The switch needs `time_series_type` inside
    the forecaster, which today reaches a forecaster only through `selected_features`.
    Recommendation: keep local keys for every series in this PR, and make the switch a cheap
    follow-up if long-range PV comparisons matter. This is the maintainer's call.

## Considered and rejected

**The member representation was chosen from one measurement on the leaderboard fold, summarised in
decision 4's table.** Pooling over nine cells at 51 members had the best plain CRPS, mean pinball
loss, p99 pinball loss, and p99 exceedance of the variants measured for substations and wind, and
decision 4 lists where other variants did better for PV and for the other group. The evidence is
read-only and uncommitted, in the planning session's scratchpad:
`clim_members.py` computes every variant, `clim_members.log` holds its output, and
`clim_members_per_series.parquet` holds the per-series scores. `clim_counts_pooled.py` and its log
give the pooled cell counts and the dropped rows per series.

- **Keep 13 members, as the PR C item says.** At 13 unpooled members the substations' p99 is
  exceeded on 20.0% of rows, and 51 pooled members cut the p99 pinball loss by 53%. The PR C item's
  "false precision" argument assumed 8 to 19 samples per cell. Pooling raises the 29 full-history
  series to at least 66 (decision 4).
- **Pool over months only (three cells).** Better than no pooling. For PV, better than nine cells
  by up to 1.4 percentage points on the plain CRPS, the mean pinball loss, and the p99 pinball
  loss, and by up to 3.4 percentage points on the p95 pinball loss. For substations, worse than
  nine cells at the same member count on every pinball loss and on the p99 exceedance.
- **More than 51 members.** At 101 unpooled members, the p99 pinball loss improves by 2.3
  percentage points more than at 51 and the plain CRPS barely moves, for twice the stored rows.
- **A normal distribution per cell.** Never the best variant: 2.8% better than 13 unpooled members
  on the substations' fair CRPS, but 7% to 9% worse on the lower-tail twCRPS, and 57% to 69% worse
  on the fair CRPS for the bimodal battery and biofuel generator. The pooled empirical members beat
  the normal variants on almost every score.
- **Members at tail-heavy, non-equiprobable levels, such as `DELIVERY_QUANTILES`.**
  `compute_metrics` assumes equiprobable members, so tail-heavy levels would put 7.7% of the mass at
  the 1st and 2nd percentiles and make the ensemble wider-tailed than the climatology it represents
  (the roadmap's argument). The pooling recipe in `docs/techniques/probabilistic-forecasting.md` is
  exact only for equally spaced levels. Even scored with the correct weights, which the metrics
  layer cannot apply, the delivery levels scored worse than pooled members.
- **A `ClimatologyConfig` with an ensemble-size or pooling field.** A subclass, a
  serialisation-test import, and stored fields for values nobody varies (decision 4).
- **A minimum-sample threshold, or a wider fallback for unseen cells.** Pooling already gives the
  29 full-history series at least 66 samples per cell, and no one-month fallback can fill series
  12's September to December (risk 1).
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
- **A stored `training_sample_count` column in the lookup.** Nothing reads the column after
  training, so the minimum and median pooled counts go in the train log line, and a later fallback
  rule can recompute them.
- **Keeping the name `PowerLagsPerNwpRunFeatureEngineer` and leaving the engineer in
  `manual_heuristic.py` (simplicity review, item 2).** The maintainer prefers descriptive names,
  and "PowerLags" is false for climatology, which requests no lags.
- **Leaving the `meta.json` round trip duplicated (simplicity review, item 3).** This PR is the
  helper's second caller after the manual heuristic. The roadmap's wording gives the extraction to
  PR B, but only because the roadmap counted persistence and climatology alone (departure 1).
- **Scoring climatology alone, without re-scoring the other two experiments (simplicity review,
  item 5).** The re-score puts the truth, the effective capacity, and the scoring code on one
  footing for all three experiments, and narrows the stale-XGBoost caveat to XGBoost's training
  inputs.
- **Dropping the `selected_features` `ValueError` (simplicity review, item 6).** The check keeps
  MLflow's params record honest about what ran, for about 8 lines.

**The day type was chosen from a second read-only measurement on the leaderboard fold.**
`clim_holidays.py` scored three alternatives against the plain local weekday and weekend day type,
all at nine-cell pooling and 51 members, and `clim_holidays_overall.py` rolls its per-series scores
up over all rows. Each figure is the change in fair CRPS against the plain day type, normalised by
effective capacity, with negative being better. The figures for a subset of days average the
per-series changes, and the all-rows figures weight each series by its rows. The validation year
holds one Christmas, with three bank holidays, and five other bank holidays, so the holiday-row
figures rest on one fold and a few days and are noisy. The scripts, their two logs
(`clim_holidays.log` and `clim_holidays_overall.log`), and `clim_holidays_per_series.parquet` are
in the planning session's scratchpad, not committed.

- **(B) Bank holidays as weekend-type days.** Over all rows, substations improve by 0.33% and PV
  worsens by 0.33%. Ordinary days change by less than 1% in every group, and by −0.04% for
  substations. On the Christmas bank holidays (25 and 26 December and 1 January), substations
  improve by 15.6%. On the other bank holidays, the substations' fair CRPS improves by 0.95% but
  their p99 pinball loss worsens by 58%, so the upper tail is worse. On all bank-holiday rows, PV
  worsens by 5.6%. The change is about 10 lines in the forecaster, plus either a hand-kept list of
  bank-holiday dates or the `holidays` package, which would change `uv.lock`. Rejected for this
  PR, and left to issue #1088 (decision 2).
- **(C) Alternative B, plus 24 December to 1 January as weekend-type days.** Substations improve by
  0.49% over all rows, but on the Christmas-week weekdays that are not bank holidays they worsen by
  4.6%.
- **(D) A third, holiday day type with cells of its own,** covering bank holidays and 24 December
  to 1 January, with no pooling across day types. The holiday cells hold a median of 12 pooled
  samples, and 48 Christmas bank-holiday rows have no cell at all. On the bank holidays outside
  Christmas, the substations' p99 pinball loss worsens by 455%.

## Review log

- Plan review 1, simplicity review (Opus), 2026-10-09: accepted 1, 4, 7 and the CRPS test; rejected
  2, 3, 5, 6; rearchitecture noted. Item 1 keeps the "Implementation details — baselines" heading
  and deletes only the PR C item. Item 4 cuts the unit tests from 14 to 11. Item 7 drops the stored
  sample count. The CRPS test joins the integration test. The rearchitecture is risk 8.
- Plan review 2, correctness and testability review (Opus), 2026-10-09: accepted 1, 2, 3, and
  the README caveat for PV in March and October; kept local cell keys; findings 4 and 5 were not
  defects. Item 1 adds the measured cell counts for series 12 and 13, makes the shared-row re-score
  unconditional, and rewrites risk 1. Item 2 corrects risk 4 after checking the code: an empty
  `smoke_test` partition gives `metrics` no group to score, so `metrics` warns and returns rather
  than raising `NoOverlappingActualsError`. Item 3 adds the autumn clock change to test 5. The
  maintainer's question on local versus UTC keys is answered in decision 2 and risk 11. Risk 8
  now links issue #1119.
- Maintainer-approved design change after review: pooling and 51 members, from a measurement,
  2026-10-09. A measurement on the leaderboard fold (`clim_members.py`) compared member counts,
  pooling, a normal distribution, and weighted delivery levels, and the maintainer approved
  nine-cell pooling and 51 equiprobable members. The plan was revised throughout: the summary, the
  walk through, departures 4 to 6, decisions 2 to 6, the file list, tests 2, 4, 6, 12 (new), and
  13, the docs list, risks 1, 4, and 7, a new bank-holiday risk (now decision 2's bank-holiday
  subsection), and "Considered and rejected". Pooling turns
  the `smoke_test` fold from empty to forecastable, so risk 4 was rewritten. The pooled counts and
  dropped rows were recomputed with `clim_counts_pooled.py`. **The changed parts have not been
  through a fresh adversarial review.**
- Maintainer decision on bank holidays, from a measurement, 2026-10-09. A read-only measurement on
  the leaderboard fold (`clim_holidays.py`) scored three holiday-aware day types, and the maintainer
  kept the plain local weekday and weekend day type. The bank-holiday risk was removed from "Risks
  and open questions" and recorded as a subsection of decision 2, and the three alternatives joined
  "Considered and rejected". **These changes have not been through a fresh adversarial review.**
- Plan review 3, correctness review of the pooling and member-count changes (Opus), 2026-10-09:
  verdict "implementation can start after edits"; all 8 findings accepted. The reviewer verified
  that the pooling specification and the hand-computed test values are exact. Finding 1 corrects
  the V2 training memory from about 5 GB to about 32 GB, at a measured 65 bytes per pooled row, and
  makes per-series batching of the pooling part of `train` (decision 2, risk 7, and test 10).
  Finding 2 rounds test 13's oracle to 13 significand bits before comparing. Finding 3 replaces
  decision 6's dropped-row check with a count of truth pairs XGBoost forecasts and climatology does
  not. Finding 4 adds a Monday 00:00 sample to test 2. Finding 5 makes `train` collect the
  deduplicated frame once and log the earliest and latest `valid_time` from it. Finding 6 states the
  `Int8` offsets and key dtypes. Finding 7 extends test 13's power through Saturday 2025-08-02.
  Finding 8 corrects the three-cell PV figures, the bank-holiday alternatives B and C, and the
  5.6% fair CRPS gap to 13 pooled members, and adds a README caveat for the battery and the
  biofuel generator.
