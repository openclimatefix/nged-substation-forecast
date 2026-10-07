# Plan: issue #147, PR A — the `baseline_forecasters` package and `manual_heuristic`

**The problem: no naive baseline exists, so no leaderboard number says whether we beat the manual
heuristic.** The manual heuristic is the analogue-ensemble method distribution network operators
use today: 13 observed powers at the same weekday and time of day, 6 from the last 6 weeks and 7
from 49 to 55 weeks back. Issue
[#147](https://github.com/openclimatefix/nged-substation-forecast/issues/147) asks for the baselines
the [Baseline forecasters](https://openclimatefix.github.io/nged-substation-forecast/roadmap/metrics-and-leaderboard/#baseline-forecasters)
section of the roadmap designs. The manual heuristic is also the floor in
[T1.2](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/#h1-a-service-that-mostly-runs-itself),
and #438, #715, #606, and #354 are blocked on it, so a wrong `manual_heuristic` would corrupt every
comparison those issues make.

**The solution: a new workspace package, `packages/baseline_forecasters/`, holding a
`ManualHeuristicForecaster` that rides the same cross-validation (CV) asset chain as
`XGBoostForecaster`.** The 13 analogues are 13 power lags, so the existing no-lookahead lag
machinery builds them. `predict()` unpivots the 13 lag columns into 13 `ensemble_member` rows. A
small power-only `FeatureEngineer` subclass, `PowerLagsPerNwpRunFeatureEngineer`, reads from the
numerical weather prediction (NWP) frame only its four key columns other than the member:
`nwp_model_id`, `init_time`, `h3_index`, and `valid_time`. The engineer deduplicates those keys
and hands that key-only frame to the public `TabularFeatureEngineer().engineer`. With no weather
column present, the tabular pipeline's upsample interpolates nothing, and the half-hourly grid, the
hindcast filter, and the cell join all come from the tabular code every other model uses. No
weather value and no ensemble member is ever read, so the manual heuristic's rows come from the
same forecast-run grid as every other model's without touching the 51 NWP members. Nothing under
`src/nged_substation_forecast/defs/` and nothing in `ml_core` changes. This plan details PR A only;
persistence (PR B) and climatology (PR C) are outlined at the end and each gets its own approval.

## Verdict, size and departures

**Worth implementing, and now.** Four issues and the T1.2 hypothesis test wait on this baseline,
and every piece it needs (power lags up to 17,520 h, `_nullify_leaky_lags`, the `BaseForecaster`
interface, the `meta.json` servability check) already exists.

**Size: complex, with both plan reviews and both diff reviews.** The five triggers:

- **Changes what gets stored — yes.** New rows in the `power_forecasts` Delta table, a new
  saved-model format (`meta.json` only), and a new package.
- **Touches the production serving path — no code on the serving path changes.** The package does
  become a production dependency, installed in the live image by the Dockerfile's `uv sync
  --no-dev`, though no serving code changes. Serving a baseline live is out of scope (decision 5).
  `predict` still degrades rather than raising on absent
  input: it drops all-null rows and never raises on missing history.
- **Touches a degradation rule — yes.** The manual heuristic is the T1.2 floor.
- **Admits more than one defensible design — yes.** How to avoid the 51-member fan-out (a
  `BaseForecaster` flag read in `defs/`, a member-0 filter, or a power-only feature engineer), what
  `nwp_init_time` holds, and how members are indexed all had alternatives.
- **Spans code whose callers could not be named without searching — no.** The callers are
  `trained_cv_model`, `cv_power_forecasts`, `live_forecasts`, and `compute_metrics`, all named
  below.

**Out of bounds for this PR, because other sessions are editing them:** everything under
`src/nged_substation_forecast/defs/`, `packages/contracts/src/contracts/config_schemas.py`,
`conf/cv/`, and the new failure-scenario module in `ml_core` (#437). Reading and importing from them
is fine (`class_target` and `import_class` come from `config_schemas`). The rest of
`ml_core/features/` is in bounds, though PR A edits none of it. `uv.lock` is also edited by #958:
whichever PR lands second
re-runs `uv lock` after rebasing rather than resolving the lock file by hand.

**Departures from the roadmap's "Implementation details — baselines":**

1. **One PR per baseline, three PRs, manual heuristic first.** PR A is the package skeleton and one
   module, `manual_heuristic.py`, holding `PowerLagsPerNwpRunFeatureEngineer` and
   `ManualHeuristicForecaster`. PR B is
   `persistence`; PR C is `climatology`. The roadmap orders persistence first as "the lowest-effort
   end-to-end probe", but persistence unblocks nothing, while `manual_heuristic` is the deliverable
   four issues wait on. `manual_heuristic_holiday_aligned` and `manual_heuristic_calibrated` (#715)
   stay separate follow-ups.
2. **No `uses_nwp_ensemble` flag on `BaseForecaster`, and no change to `cv_power_forecasts`.** The
   roadmap's "PR 2" has `cv_power_forecasts` pass `ensemble_members=[0]` when a class sets the flag
   `False`, which edits `defs/cv_assets.py`. Instead the baseline package overrides
   `BaseForecaster.feature_engineer` — the documented composition extension point — with
   `PowerLagsPerNwpRunFeatureEngineer`. Both `trained_cv_model` and `cv_power_forecasts` already hand
   the engineer the NWP frame `load_engineering_inputs` returns, so the engineer can derive the run
   grid from that frame without any change in `defs/`. The rows match every other model's rows,
   missed NWP runs included, with one exception. `XGBoostForecaster` predicts every NWP row, even
   one whose features are NaN, while the manual heuristic drops a row whose 13 lags are all null
   (test 3). A row loses all 13 lags only where the history reaches back to none of the 13
   analogues, so a series with history missing at every analogue scores fewer rows under the manual
   heuristic than under XGBoost.
3. **The roadmap's "PR 1" (median and p95 collapse in `ml_core/metrics.py`) is not a prerequisite.**
   The baselines emit members only. Today's metrics score the per-run ensemble mean, and CRPS and
   the pinball losses already flow over the 13 members. The "median headline plus p95 row" the
   roadmap promises for `manual_heuristic` needs PR 1, which becomes its own issue (decision 2).
4. **`ensemble_member` needs no `UInt8 → Int8` cast.** The roadmap's PR 3 text says the spine's
   member column is `UInt8`. `Nwp.ensemble_member` and `AllFeatures.ensemble_member` are both
   declared `Int8` today. The manual heuristic builds its own member index, as an `Int8` expression.
5. **No analogue config fields: the YAML lists the 13 power-lag features explicitly.** The roadmap
   has `n_weekly_analogues = 6` and `annual_week_span = (49, 55)` derive `selected_features`.
   Listing the 13 names in `selected_features` needs no config subclass and no validator. The
   drawback is that a variant moving the annual window overrides the whole list rather than one
   field.
6. **Per-series surviving-member counts are not recorded in asset metadata, because asset metadata
   is written in `defs/`, which is out of bounds.** The pre-read data check under Verification
   commands names the series short of history instead.

**One correction to the brief this plan was written from.** The NWP-absent path is not in
`defs/_engineering_inputs.py`. The NWP-absent path is in `ml_core/features/_nwp.py`
(`_join_nwp_bulk_mode`, the `processed_nwp is None` branch) and in
`tabular_feature_engineer._engineer_features`. That path sets `power_fcst_init_time = valid_time`,
so every row has lead time 0, and the comment at the end of `_engineer_features` records that
predict output built from that path always fails `PowerForecast.validate`. The path is
training-only. The baselines therefore never take the NWP-absent path. They take the NWP bulk path
instead, fed a key-only NWP frame, so each row's `power_fcst_init_time` is a real run's
`init_time` plus the publication delay.

## What changes, file by file

### `packages/baseline_forecasters/` (new package)

**The layout mirrors `packages/xgboost_forecaster/`.** `pyproject.toml` declares the project
`baseline-forecasters`, built by `uv_build` like its sibling, depending on `contracts`, `ml_core`,
`patito`, and `polars` only. No `pytest` entry: test tooling lives in the root dev group
(`docs/architecture/testing.md`).

- `src/baseline_forecasters/__init__.py` re-exports `ManualHeuristicForecaster` only.
- `src/baseline_forecasters/manual_heuristic.py` holds `PowerLagsPerNwpRunFeatureEngineer` and
  `ManualHeuristicForecaster`, whose `save` and `load` write and read `meta.json` inline. The
  forecaster class must sit at module level, because `class_target` names it in `meta.json` and in
  `conf/model/manual_heuristic.yaml`. PR B, as the second caller, extracts the engineer and the
  `meta.json` round trip into shared modules.
- `tests/` holds the unit tests listed under Tests.
- `README.md` is described under Docs.

### `PowerLagsPerNwpRunFeatureEngineer`

**A `FeatureEngineer` subclass whose `engineer` strips the `nwp` frame to its keys and delegates to
`TabularFeatureEngineer().engineer`.** In bulk mode (`power_fcst_init_time is None`) `engineer` does
two lazy steps:

1. Build the key-only frame `nwp.select("nwp_model_id", "init_time", "h3_index",
   "valid_time").unique()`, wrapped with `set_model(Nwp)` and never validated.
   `_attach_containing_nwp_cell` needs `h3_index`, and `_upsample_nwp_to_half_hourly` groups by
   every column that is neither `valid_time` nor a weather variable, so the upsample's groups are
   `(nwp_model_id, nwp_init_time, time_series_id)`. With no weather column, the upsample builds the
   half-hourly grid from each group's minimum to maximum `valid_time` and interpolates nothing.
2. Return `TabularFeatureEngineer().engineer(...)` on that frame, passing every other argument
   through unchanged.

**The rows equal the distinct `(time_series_id, power_fcst_init_time, valid_time)` rows the tabular
path produces from the full NWP frame, given overlapping member spans and one NWP model.** The
grid, the hindcast filter (`valid_time > power_fcst_init_time`), and the cell join are the tabular
code itself, so equality holds by construction. The two qualifiers come from the deduplication. The
tabular path builds one grid per member, from that member's minimum to maximum `valid_time`, and
the key-only frame builds one grid from the minimum to the maximum over every member. A gap inside
one member's valid times is filled by that member's own grid, so the two paths diverge only where
two members' spans of a run do not overlap: the key-only grid then covers the stretch between the
two spans, and the per-member grids do not. With two NWP models, `nwp_model_id` stays a group key
on both paths, so each model gives its own copy of every row. The manual heuristic would then emit each
forecast twice and duplicate the `PowerForecast` primary key. The NWP table holds one model today.
A feasibility script run during plan review confirmed the equality on a fixture of four runs, two
members, and two cells: the key-only frame gave 4,449 rows, identical to the full-weather frame's
8,898 rows deduplicated across the two members, `power` and all 13 lag columns included. A second
script, on two runs a day apart, two members, two cells, and three series, confirmed that the frame
needs no `ensemble_member` column: 4,212 rows, identical to the 8,424 full-weather rows
deduplicated across members, with no duplicate key.

**The output carries no `ensemble_member` column and the run's real `nwp_init_time`.**
`AllFeatures.ensemble_member` is declared `allow_missing=True`, and `_select_output_columns` keeps
the column only when the frame has one. `predict` builds `ensemble_member` from the lag rank and
stamps `nwp_init_time` null. The output also carries `nwp_lead_time_hours`, which `predict` does
not read.

**The run grid comes from the NWP frame's own `init_time` values, so chunked prediction stays
disjoint.** `cv_power_forecasts` walks `init_time` chunks from `val_start - MAX_NWP_LEAD` to
`val_end`, each starting 1 µs after the last one ends, and `load_engineering_inputs` bounds each
chunk's NWP scan to that chunk's `init_time` range. A run therefore reaches exactly one chunk's
engineer, and a run missing from the archive reaches none, as for every other model.

**Each run's valid times come from the frame's `valid_time` column rather than from a horizon
constant.** `load_engineering_inputs` bounds the NWP scan to `valid_time` in `[window_start,
window_end]`, and the engineer is not told the window. Keeping `valid_time` in the key-only frame
carries that clipping through. A run initialised before `val_start` (the chunks reach back
`MAX_NWP_LEAD`, 16 days) would gain rows before `val_start` if the engineer read `init_time` alone
and added a horizon.

**Single-run mode raises `NotImplementedError`.** When `power_fcst_init_time` is given, `engineer`
raises with a message saying that baselines are R&D-only and not served live (decision 5). The
raise is worth its two lines because the message names the fault. Delegating would fail with a
`ColumnNotFoundError` from `live_forecasts`' filter on a non-null `ensemble_member`, which names
neither the baseline nor the cause. Past that filter, the forecast would be silently degraded,
because `live_forecasts` reads only 15 days of power (`LIVE_POWER_HISTORY`), so every annual
analogue is null. The only single-run caller is
`live_forecasts`, which serves only a promoted model, so reaching this branch means a baseline was
promoted by mistake, which is our own bug. A promote-time refusal in `ml_core/production_helpers.py`
would catch the mistake earlier, and is an optional later hardening outside this PR.

The class docstring says that the forecaster consumes no weather, and that the NWP frame is read
only for its run, cell, and valid-time keys. `weather_source: "none"` is therefore literally true.

### `ManualHeuristicForecaster(BaseForecaster)`

**Class constants:** `MODEL_NAME = "manual_heuristic"`, `MODEL_VERSION = 1`, `CONFIG_CLASS =
BaseForecasterConfig`, and `feature_engineer = PowerLagsPerNwpRunFeatureEngineer()`. Instance state is
`self._trained_ids: list[int]`, exposed by `trained_time_series_ids`, and the ordered lag-column
names `predict` reads.

**`__init__(model_params)` parses `selected_features` with `ParsedFeatures.from_strings` and raises
`ValueError` if the config selects no power lag.** A power lag is a `LagFeature` whose `base_col`
is `"power"`. Without the check, `unpivot(on=[])` in `predict` would return zero rows for every
input, giving a forecaster that trains, saves, and emits nothing without an error. No other feature
needs a check. A weather feature or a weather lag already makes the delegated pipeline raise at
`trained_cv_model`, because the key-only frame has no weather column, and a time or static feature
is computed and ignored. The member order is the power-lag hours sorted ascending, so 0 to 5 are
the weekly analogues and 6 to 12 the annual ones under the shipped YAML.
Registration (`register_experiment_job`, `defs/jobs.py` around line 167) validates only
`CONFIG_CLASS` and never calls `__init__`, so a config with no power lag registers cleanly and the
`ValueError` surfaces at `trained_cv_model`, when the forecaster is first constructed.
The 17,520 h lag cap needs no check here, because the feature parser's `Hours` type already rejects
a longer lag.

**`train(data, time_series_ids)` records the sorted ids that have at least one non-null `power`
row and are in `time_series_ids`.** These semantics mirror XGBoost's "no usable rows, not
trained", and the maintainer chose to keep them (decision 7). `trained_cv_model` loads NWP with
`ensemble_members=[0]`, so the engineer's training rows are the run grid of the control member's
runs over the training window, the same `(power_fcst_init_time, valid_time)` rows XGBoost trains
on. The collect selects `time_series_id` and `power` only, drops nulls, takes unique ids, and
streams (`engine="streaming"`). Projection pushdown keeps the collect small, though the 13 lag
joins may still run (risk 7).

**`predict(data, *, fold_id="live")` builds the members in five steps:**

1. Collect with `engine="streaming"`, then strip the Patito model with `.as_polars()` so no later
   step can hit the dict-cast trap (`polars-patito-gotchas`). Both loaders already restrict the
   power input to `trained_time_series_ids`, and `PowerLagsPerNwpRunFeatureEngineer` emits one row per
   series, run, and valid time, so `predict` filters nothing.
2. Unpivot the lag columns, indexed by `time_series_id`, `valid_time`, and `power_fcst_init_time`.
   The input's `nwp_init_time` and `nwp_lead_time_hours` are not carried.
3. Map each lag column name to its member index — its rank in the sorted lag order `__init__`
   stored — with `replace_strict(..., return_dtype=pl.Int8)`. Ranking by lag means a member keeps
   its meaning as `_nullify_leaky_lags` sheds the short weekly lags with lead time: past 168 h of
   lead, member 0 is absent and members 1 to 12 keep their indices.
4. Drop null values. A null comes from `_nullify_leaky_lags` or from history that does not reach
   back far enough. Nothing raises, and nothing is logged.
5. Add `power_fcst` (the value cast to `Float32` by expression), `nwp_init_time` as a typed null
   (`pl.lit(None, dtype=UTC_DATETIME_DTYPE)`), and the identity literals exactly as
   `XGBoostForecaster._build_part` builds them (`power_fcst_model_name`, `power_fcst_model_version`
   as `Int16`, `ml_flow_experiment_id` as `Int32`, `experiment_name`, `fold_id`). Return
   `PowerForecast.validate(...)`.

**Empty input returns an empty frame that validates.** The unpivot and the literals produce a
correctly typed zero-row frame; the test below pins this rather than adding a special branch.

**`nwp_init_time` is null on every row.** The manual heuristic consumes no weather, and
`PowerForecast.nwp_init_time` already documents null for "models that do not use NWP (e.g.
persistence baselines)". Its two readers already handle null: `packages/dashboard/view_forecasts.py`
shows a "records no `nwp_init_time`" notice, and `defs/checks.py` types the field `datetime | None`.
`compute_metrics` groups by `power_fcst_init_time`, not `nwp_init_time`, so scoring is unaffected.

**`save(path)` clears `path` and writes `meta.json` alone; `load(path)` reads it back.** `save`
clears `path` with `shutil.rmtree(path, ignore_errors=True)`, recreates it, and writes three keys:
`model_class` (`class_target(self)`) and `model_params` (`model_dump(mode="json")`), which
`production_helpers._check_meta_is_servable` reads, and `trained_time_series_ids` (sorted), which
`load` reads. `load` builds the config through `CONFIG_CLASS.model_validate`. `save_to_mlflow` adds
the frozen metadata parquet as for any model. No `StatelessForecaster` base class, as the roadmap
says, until a third stateless model exists.

### `conf/model/manual_heuristic.yaml` (new)

**`_target_: baseline_forecasters.manual_heuristic.ManualHeuristicForecaster`, with `model_params`
listing the 13 power-lag names in `selected_features` (`power_lag_168h` to `power_lag_1008h`, then
`power_lag_8232h` to `power_lag_9240h`), `weather_source: "none"`, and `training_strategy:
"none"`.** The header comment follows `conf/model/xgboost.yaml`'s, and adds that a variant overrides
the whole list, and that only the power lags become members. The comment also says where a bad
list fails: registration accepts both a weather feature and a list with no power lag, and both
raise only when `trained_cv_model` runs.

### Root `pyproject.toml` and `uv.lock`

**Add `"baseline_forecasters"` to `[project] dependencies` beside `"xgboost_forecaster"`, and
`baseline_forecasters = { workspace = true }` to `[tool.uv.sources]`, then run `uv lock`.** A
production dependency rather than a dev-group entry, because `load_forecaster_from_dir` imports
whatever class a promoted `meta.json` names. The package pulls in nothing new (`contracts`,
`ml_core`, `patito`, and `polars` are all in the lock file already), so the `uv export --no-dev`
check in `docs/architecture/testing.md` is unaffected.

### `packages/contracts/src/contracts/power_schemas.py`

**Description text of `PowerForecast.ensemble_member` only; no dtype, nullability, or constraint
changes, and no other file under `contracts/` changes.** The description gains: an NWP-member index
for models that consume an NWP ensemble, a historical-analogue index (rank of the lag, 0 =
shortest) for `manual_heuristic`, and, from PR C, a quantile-sample index for `climatology`, so
`ensemble_member` does not imply NWP. The change does not alter what a frame must contain, so the
"ask before changing a Patito data contract" rule is met by this plan's approval.

### `tests/` at the root

- `tests/conftest.py`: the `register_experiment` fixture gains optional `base_model_config` and
  `config_overrides` arguments. Left unset, both keep today's XGBoost values, so existing callers
  are unchanged. The fixture picks the defaults with `config_overrides is None`, not `or`, so a
  caller passing `{}` sends no overrides at all. A caller-supplied `config_overrides` replaces the
  whole default overrides dict, its `selected_features` entry included, rather than merging into
  it. The existing `selected_features` argument only edits the default dict, so the fixture
  raises `ValueError` when a caller passes both arguments. The defaults send `selected_features: ["temperature_2m"]`
  and `n_estimators: 5`. `BaseForecasterConfig` sets `extra="forbid"`, so it would reject
  `n_estimators`, and `ManualHeuristicForecaster.__init__` would reject `["temperature_2m"]` for
  holding no power lag.
- `tests/test_forecaster_config_serialisation.py` needs no change, because PR A adds no config
  class.
- New integration test, `tests/test_manual_heuristic_cv.py`, described under Tests.

## Design-philosophy check

**The code runs in the R&D asset chain only, and `predict` still degrades rather than raising on
absent input.** Serving a baseline live is out of scope (decision 5). Missing history sheds members,
a row with no members is dropped, and empty input gives empty output. The pre-read data check
below names the series short of history. Nothing raises except a contract violation
(`PowerForecast.validate`), a config with no power lag at construction, a weather feature in the
delegated pipeline, or a single-run call to `PowerLagsPerNwpRunFeatureEngineer`, all of which are our
own bugs. No asset check is added.

**The change delivers the measuring instrument for T1.2.** T1.2 asks that every series still emits a
forecast at rungs 0 to 2 of the degradation ladder and still beats `manual_heuristic`. This PR does
not measure T1.2; #438 does, and needs this baseline first.

**Design principles 3 and 8 are kept.** The baseline rides the same `register_experiment_job →
trained_cv_model → cv_power_forecasts → metrics` chain as XGBoost (principle 3, one execution
path), and draws its rows from the same forecast-run grid (principle 8, every experiment scored
identically). The one difference in row sets is the all-null-lag rows the manual heuristic drops
(departure 2, test 3). Principle 11 (push the work down to the query engine) is why the key-only
frame is a lazy `select` and `unique` over the NWP scan: projection pushdown keeps the weather
columns and `ensemble_member` out of the read.

**One principle is traded, deliberately and visibly: the floor's row grid comes from the NWP
archive.** `docs/design-philosophy/inherent-stability.md` says the manual heuristic "consumes no NWP
data, so an NWP outage does not degrade it at all". In CV, an `init_time` missing from the NWP
archive removes that run's rows for the manual heuristic as for every other model. That keeps the
comparison like-for-like, which is what the leaderboard needs. No weather value is read, so a run
whose weather is wrong but present leaves the manual heuristic untouched.

## Tests

Each test names the assertion that fails on `main` today. On `main` the package does not exist, so
every import fails; the assertions below are the behavioural claims each test pins once the import
resolves.

**`packages/baseline_forecasters/tests/test_manual_heuristic.py` (unit):**

1. **Construction.** A forecaster built from the `selected_features` read out of
   `conf/model/manual_heuristic.yaml` stores, as its lag order, the hours of those power lags
   sorted ascending, and the test writes out the expected hours, 168·k for k in 1 to 6 and
   168·k for k in 49 to 55, so a typo in the YAML fails it. A config whose
   `selected_features` is `["local_time_of_day_sin"]` raises `ValueError` for holding no power lag.
2. **Shedding keeps member identity, through the real pipeline.** Synthetic power for one series
   over 56 weeks and an NWP fixture of one run, passed through
   `PowerLagsPerNwpRunFeatureEngineer.engineer` and `predict`. The power runs to at least
   `init_time + 9 h`, the run's `power_fcst_init_time`, because lead is measured from `power_fcst_init_time`:
   power ending at `init_time` would leave member 0 null at leads of 159 h to 168 h. Then: on every
   row `power_fcst` equals the lag column whose rank is its `ensemble_member`, and `nwp_init_time`
   is null. At lead under 168 h the members are 0 to 12; at lead in [168 h, 336 h) member 0 is
   absent and members 1 to 12 are present with the values of lags 336 h to 9240 h; at lead in
   [336 h, 351 h] members 0 and 1 are absent and the other 11 members are present. A mutation that
   indexes members by position after dropping nulls fails this test. `PowerForecast.validate`
   inside `predict` already enforces the `Int8` member dtype and the identity columns, so the test
   does not assert them.
3. **All-null rows.** A row whose 13 lags are all null is absent from the output, and the call does
   not raise.
4. **Empty input.** A zero-row `AllFeatures` frame returns a zero-row frame that
   `PowerForecast.validate` accepts.
5. **Train.** `train` on rows for ids {1, 2, 3}, where id 2 has only null power and id 3 is not
   requested, records `[1]`.
6. **Save and load.** `load(save(...))` returns the same `trained_time_series_ids` and an equal
   config; a stale file placed in the directory before `save` is gone afterwards; and `meta.json`'s
   `model_class` equals `"baseline_forecasters.manual_heuristic.ManualHeuristicForecaster"`.
7. **The engineer's rows equal the tabular path's, deduplicated across members.** This equality is
   the property `PowerLagsPerNwpRunFeatureEngineer` exists for. The NWP fixture holds two runs a day
   apart with members 0 and 1, a third run with member 1 only and no member 0, and two H3 cells, one
   holding two series. One run's earliest valid times are cut off, as `load_engineering_inputs`
   clips a run initialised before the window. An engineer that filters to member 0 loses the third
   run's rows, and an engineer that adds a 360 h horizon constant to `init_time` gains rows in the
   clipped stretch, so both fail the equality. Power
   reaches back far enough that each of a few lags, among them one annual lag, is non-null on some
   rows. `PowerLagsPerNwpRunFeatureEngineer().engineer` and `TabularFeatureEngineer().engineer` run on
   the same power, metadata, full-weather NWP, and lag features. The engineer's output has no
   duplicate `(time_series_id, power_fcst_init_time, valid_time)` row, and its rows, with `power`
   and the lag columns, equal the tabular output deduplicated across members. The equality catches
   an engineer that drops `h3_index` or `valid_time` from the key-only frame or skips `unique`. The
   grid the engineer inherits needs no test here: `packages/ml_core/tests/test_features.py` guards
   the half-hourly upsample (the tests from line 57 to line 216) and the hindcast filter
   (`test_engineer_features_bulk_mode_drops_hindcast_rows`, line 714).

**`tests/test_manual_heuristic_cv.py` (integration, `pytestmark = pytest.mark.integration`):**

- **Test 8: End to end through the real assets.** Register `conf/model/manual_heuristic.yaml`
  through the extended fixture, passing `config_overrides={}` so the YAML's own `selected_features`
  are used. Write synthetic power covering 56 weeks before a validation day to
  `power_time_series.delta`. Write NWP as `tests/test_cv_power_forecasts.py:_write_nwp` does: a
  member-0 run on a training day where power exists, plus a validation-day run with members 0, 1,
  and 2. Without the training run, `trained_cv_model` trains no series and raises "Trained 0 of 1
  eligible time series" (`defs/cv_assets.py` around line 405). Every NWP record sits in H3 cell
  `599423199024775167`, the `h3_res_5` that `write_metadata` hard-codes, so the series joins to the
  runs. Write the eligible-series table for the fold, as `_write_eligible` in the same test module
  does. `load_engineering_inputs` reads power through `scan_cleaned_power`
  from `settings.cleaned_power_time_series_data_path`, which keeps only rows whose `drop_reason` is
  null. The test therefore calls `write_cleaned_copy(...)` from `tests/_cleaned_power_test_data.py`
  on the raw table with no `flag_where`, as `tests/test_cv_power_forecasts.py` does, so that every
  synthetic reading reaches the forecaster. `write_metadata` in the same helper writes the
  metadata parquet. Then materialise `trained_cv_model` and `cv_power_forecasts`. Assert that the
  `power_forecasts` rows carry `power_fcst_model_name == "manual_heuristic"`, members 0 to 12 at
  short lead, values equal to the synthetic power at each lag, with the expected lags written out
  as 168·k rather than read from the YAML's `selected_features`, null `nwp_init_time`, and no row
  duplicated across the three NWP members. The synthetic power must give each of the 13 lags a
  distinct value, and must give a lag one half-hour out a different value again, so a lag off by
  one half-hour fails. One choice is an integer power equal to the half-hour index since a fixed
  epoch, taken modulo 1999 and shifted by −999. The 13 lags are 336 k half-hours for distinct k up
  to 55, and 1999 is prime, so the 13 values are distinct. Every value is an integer within ±999
  MW, which stays plausible and which 13-bit significand rounding stores exactly. The fixture
  registers the YAML through `register_experiment_job`, so this test also covers registration.

## Docs to update

Every page is written to describe the code as it now stands, with no "was previously" wording.

- **`docs/roadmap/metrics-and-leaderboard.md`, "Implementation details — baselines".** Reorder the
  PR list to manual heuristic, persistence, climatology. Replace the "PR 2" item's
  `uses_nwp_ensemble` text with the `PowerLagsPerNwpRunFeatureEngineer` design, and keep its
  `ensemble_member` overload bullet. The replacement text says that each baseline overrides
  `BaseForecaster.feature_engineer` with an engineer that reads from the NWP frame only the
  `nwp_model_id`, `init_time`, `h3_index`, and `valid_time` keys. The engineer deduplicates those
  keys and passes the key-only frame to `TabularFeatureEngineer`, so the
  baseline's rows come from the tabular pipeline's own run grid. No weather value and no ensemble
  member is read, so `weather_source: "none"` is literally true. The NWP archive's list of runs is
  what keeps every leaderboard row, baseline or ML, on the same forecast-run grid, so a missed run
  removes its rows for every model. Delete the `uses_nwp_ensemble = False` line and the member-0
  wording from the persistence, manual heuristic, and climatology items. Replace "PR 1" with a note
  that the metrics collapse is its own issue under the v0.3 epic. Delete the `manual_heuristic`
  item's details at ship time, with the summary moving to the PR body. Correct the `UInt8 → Int8`
  sentence in the persistence item. The section and its 🚧 banner stay until PR C ships.
- **`docs/ml_experimentation/model-configuration.md`.** The sentence "The only one today is
  `conf/model/xgboost.yaml`" becomes a list of the two YAML files, noting that `manual_heuristic`
  lists its 13 power-lag features explicitly.
- **`packages/baseline_forecasters/README.md`.** The README states what each baseline emits, links
  to `PowerForecast.ensemble_member` for what the member index means, and states why the
  forecast-run-grid feature engineer exists, that the weekly analogues shed with lead (decision 9),
  and the daylight-saving caveat (decision 1). The shedding statement says the weekly group holds 6
  members below 168 h of lead, 5 at leads in [168 h, 336 h), and 4 from 336 h, because an operator
  issuing a forecast at time T has only the weeks before T. No API page is added: `docs/api/` covers
  7 of the 12 packages today.
- **`docs/roadmap/metrics-and-leaderboard.md`, two other sentences that this PR makes false.** Line
  63, "No naive baseline exists anywhere in the codebase", and line 263, "today only
  `XGBoostForecaster` exercises it", are each rewritten in the present tense to name
  `ManualHeuristicForecaster` as well. The line-63 sentence's parenthesis, "only docstring
  mentions, e.g. `contracts/power_schemas.py:242`", is dropped with the claim it supports. The
  docstring mention it cites, "e.g. persistence baselines" in `PowerForecast.nwp_init_time`, now
  sits at line 463, so the citation is stale as well.
- **`packages/ml_core/README.md`.** Line 52, "`XGBoostForecaster` … is the only subclass outside the
  tests", becomes a present-tense sentence naming both subclasses.
- **`docs/architecture/code-style.md`, the Machine Learning section.** Line 330,
  "`XGBoostForecaster` is the only implementation so far", becomes a present-tense sentence naming
  `XGBoostForecaster` and `ManualHeuristicForecaster`.
- **`docs/roadmap/metrics-and-leaderboard.md`, line 997.** The sentence "The manual heuristic
  consumes no NWP and is indifferent to recent telemetry staleness" gains the same qualifier as the
  `inherent-stability.md` edit below: in cross-validation the manual heuristic's rows still follow
  the NWP run grid.
- **`docs/design-philosophy/inherent-stability.md`, "The manual heuristic is the floor".** The
  sentence saying the manual heuristic consumes no NWP data gains a qualifier: in cross-validation
  the manual heuristic's rows still follow the NWP run grid, so an NWP run missing from the archive
  removes that run's rows.
- **CLAUDE.md, the Packages table.** One row for `baseline_forecasters`.

## Verification commands

Run all of these green before every push:

- `uv run ruff check .` and `uv run ruff format .`
- `uv run --all-packages ty check`
- `uv run pytest`, which collects `packages/baseline_forecasters/tests/` by recursing into the
  repository's directories; the root project's dependency on the package only makes the test
  module's imports resolve
- `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md`
- `uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .`
- `uv run --no-sync python scripts/lint/check_docs_links.py`
- `uv run mkdocs build --strict`, then read the edited roadmap section as rendered
- `test "$(uv export --no-dev --format requirements-txt | grep -ciE '^(pvlib|cdsapi)')" -eq 0`
- `uv lock --check` after rebasing on whichever of this PR and #958 lands first

**Before reading any result, run the data check the roadmap asks for.** Count, per eligible series,
the unflagged observations at least 49 weeks before `val_start`, read with `scan_cleaned_power` on
the `cleaned_power_time_series` table rather than from the raw power table. Eligibility and the
analogues both come from the cleaned table, so a count over the raw table would include readings
the forecaster never sees. A series eligible under `min_training_months` can still lack the annual
analogues and degrade silently to a weekly-only ensemble of at most 6 members. The check surfaces
the count of such series (decision 4).

## Risks and maintainer decisions

The maintainer answered the open questions this plan raised; each answer is recorded as a decision
in the item it settles.

1. **Daylight saving time — decided: ship UTC lags and document the caveat.** A lag of a multiple
   of 168 h is a whole number of weeks in UTC. Across a clock change, the analogue sits an hour away
   in local clock time from the target, whereas the operator's method matches local weekday and
   time of day. By arithmetic, the weekly member at a lag of k weeks straddles a clock change for
   the k weeks after each change, and the annual member at a lag of L weeks for about |52 − L|
   weeks beside each change. Summed over the 6 weekly members and the 7 annual members, about one
   member value in ten sits an hour away from the target in local clock time. UTC lags keep the
   baseline on the audited lag machinery. Measure the share of member values affected in the PR,
   and state the caveat in the package README. Local-clock alignment is a follow-up alongside
   `manual_heuristic_holiday_aligned`, which needs a bespoke analogue picker anyway.
2. **Metrics collapse — decided: its own issue under the v0.3 epic.** The roadmap's "PR 1" (median
   headline, `mae`/`mbe` at `metric_param="p95"`) has not landed, so `manual_heuristic`'s
   deterministic metrics score the 13-member mean until that issue ships. Neither blocks the other.
3. **Sub-issues — decided: one sub-issue per PR under #147, plus one for
   `manual_heuristic_holiday_aligned`, created after this plan is approved,** following the
   `github-issue-pr-workflow` skill.
4. **Series with less than 55 weeks of history degrade silently to fewer members — decided: the
   pre-read data check surfaces their count.** A series with enough history can shed members too:
   every reader of power skips a reading the power cleaning flags, so the analogue that reading
   would have supplied is null and that member is shed. The one cleaning rule today,
   `substation_zero` in `nged_data/cleaning.py`, flags every reading of exactly 0 from a `Primary`,
   `BSP`, or `GSP` series.
5. **Serving the manual heuristic live — decided: out of scope; the baselines are R&D baselines for
   now.** The live path, `live_forecasts`, reads the cleaned power table and does not serve
   baselines.
6. **Reading the run grid is unlikely to be slow.** The key-column read is a subset of what
   `cv_power_forecasts` already decodes for XGBoost, and the first real fold run will show any
   slowness. The fallback is an engineer filtering the NWP frame to the control member
   (`ensemble_member == 0`), at the price of losing a run's rows when its control member is
   missing.
7. **Training engineers 13 lag columns only to read the population — decided: keep `train()`
   collecting the population.** `train` needs only the ids with non-null power, but
   `trained_cv_model` hands it the full engineered frame. Accept the extra time and record the
   measured `trained_cv_model` time in the PR body.
8. **Where the baselines live — decided: a new package, `baseline_forecasters`.** The `uv.lock`
   collision with #958 is handled by re-running `uv lock` after rebasing.
9. **The ensemble sheds weekly analogues with lead — decided: keep the shedding and state it in the
   package README.** `docs/background/manual-heuristic-forecast.md` says the weekly analogues come
   from "the last 6 weeks". Under `_nullify_leaky_lags` the weekly group holds 6 members below
   168 h of lead, 5 members at leads in [168 h, 336 h), and 4 from 336 h, because each weekly lag no
   longer than the lead is nulled. An operator issuing a forecast at time T has only the weeks
   before T, so for a far target the analogues 168 h and 336 h before the target fall after T and
   are unavailable to the operator too.
10. **NWP-outage scenarios remove the floor's rows — open, for #438 to decide.** T1.2 and #438 will
    score ML models under NWP-outage scenarios. The manual heuristic's CV rows follow the NWP run
    grid, so a scenario that removes NWP runs also removes the manual-heuristic floor's rows for
    the forecast slots those runs covered. #438 must therefore decide whether the floor is scored
    on the clean run grid or on the degraded grid.

## Considered and rejected

- **`ControlMemberFeatureEngineer`, a `TabularFeatureEngineer` subclass filtering NWP to member
  0:** rejected only because its rows would depend on the control member being present. The
  weather columns are not the reason: with only power lags requested, `_select_output_columns`
  (`tabular_feature_engineer.py`, lines 378 to 402) selects no weather column, so projection
  pushdown should already keep the weather columns out of the read. The same filter is the
  fallback in risk 6.
- **A hand-built run grid joined to the power lags through a new public `ml_core` entry point,
  `engineer_power_features`:** rejected because it is more code, and because its rows only equal
  the tabular path's if four steps re-implemented from `_attach_containing_nwp_cell`,
  `_upsample_nwp_to_half_hourly`, and the hindcast filter stay in step with the originals. The
  design needed a new public function in `ml_core`, its docstring and README edits, a test that it
  rejects weather features, and a weather check duplicating the one in
  `ManualHeuristicForecaster.__init__`. The key-only frame gets the same rows from the tabular code
  itself, so equality holds by construction.
- **Deriving the run grid from `init_time` alone and a 360 h horizon constant:** rejected because
  `load_engineering_inputs` clips the NWP frame to the window's valid times and the engineer is not
  told the window. A run initialised before `val_start` would gain rows before `val_start` that no
  other model has. The two extra columns read, `h3_index` and `valid_time`, are columns the scan's
  filters decode anyway.
- **A lazy `train` recording `sorted(set(time_series_ids))` without collecting:** rejected to keep
  population parity with XGBoost's "no usable rows, not trained" rule and the train==predict
  invariant (decision 7).
- **A shared `_meta_io.py` and a separate `_feature_engineer.py` in PR A:** rejected because each
  would have one caller until PR B. PR B extracts both when it becomes the second caller.
- **Baselines in `ml_core/baselines.py` instead of a new package:** rejected, because concrete
  models next to the abstract interface trade away design principle 5 ("Everything around the model
  is general-purpose"), and the roadmap mirrors the `xgboost_forecaster` layout (decision 8).

## Review log

- **Simplicity review (plan review 1).** Removed `ManualHeuristicConfig`, its validator, and its
  three analogue fields in favour of an explicit 13-name `selected_features` list and a power-lag
  check in `__init__`. Removed `predict`'s cross-member dedupe, its filter to
  `trained_time_series_ids`, and the per-series dropped-row helper and logging, keeping one
  aggregate log line. Removed the Delta `explain()` pushdown test, the one-off memory measurement,
  the projection decision, and the `performance.md` sentence. Removed the scoring test, the
  serialisation-test change, and the `AllFeatures.ensemble_member` description edit. Three findings
  were rejected, as recorded above.
- **Correctness review (plan review 2).** Found no defect in the lag arithmetic, the member
  indexing, the primary key, or the degradation behaviour. Changed the plan in eight places:
    - declared `weather_utils` directly in the package's `pyproject.toml`;
    - made `__init__` raise when no power lag is selected, and added that case to test 1;
    - made the fixture's `config_overrides` replace the XGBoost defaults wholesale;
    - gave the integration test synthetic power that catches a lag off by one half-hour, and gave
      the shedding test a lead in [336 h, 351 h] and a second NWP run a day later;
    - added four docs edits (two roadmap sentences, the `ml_core` README, and the
      inherent-stability qualifier), and softened the claim that every leaderboard row is scored on
      the same grid;
    - corrected risk 5's live lag coverage;
    - corrected which reader uses each `meta.json` key;
    - added open question 9 on shedding members with lead.
- **Main merged in (2026-10-07).** Main added power cleaning: every reader of observed power,
  `load_engineering_inputs` included, now reads the unflagged rows of the
  `cleaned_power_time_series` table through `scan_cleaned_power`. The plan changed in six places:
    - the integration test writes the cleaned table with `write_cleaned_copy` after writing the raw
      power;
    - the design-philosophy check, risk 4, and the package README's caveats say that a reading the
      cleaning flags sheds the member it would have supplied;
    - the pre-read data check counts unflagged observations from the cleaned table;
    - the line-63 roadmap rewrite drops the stale `power_schemas.py:242` citation;
    - risk 5 says the live lags read the cleaned table, and that live forecasts fail until the
      cleaning has run once;
    - departure 2 cites `ensemble_members=[0]` at `defs/cv_assets.py:390`.
- **Maintainer review (2026-10-07).** Replaced `ControlMemberFeatureEngineer` with the power-only
  `PowerLagsPerNwpRunFeatureEngineer` and a public `engineer_power_features` entry point in
  `ml_core/features/`. The engineer reads `h3_index` and `valid_time` as well as `init_time`,
  because the NWP frame is clipped to the window and the engineer is not told the window. Dropped
  the `weather_utils` dependency, the member-filter test, and the risk on the member-0 predicate
  reaching the scan. Added the engineer tests, the integration test, and the run-grid read
  measurement, with the member-0
  filter as its fallback in risk 6. Recorded the maintainer's answers to open questions 1 to 9 as
  decisions, and deleted the live-serving follow-up from risk 5.

- **Simplicity review of `PowerLagsPerNwpRunFeatureEngineer` (2026-10-07).** Replaced the hand-built
  run grid and the public `engineer_power_features` entry point with a key-only NWP frame passed to
  `TabularFeatureEngineer().engineer`. A scratch script confirmed the key-only frame gives the same
  rows, `power` and the 13 lags included, as a full-weather frame deduplicated across members.
  Deleted the `ml_core` function, its docstring and README edits, its weather-rejection test, and
  its weather check. Kept the single-run `NotImplementedError`, because delegating would serve a
  silently degraded forecast rather than fail. Qualified the row equality with "given contiguous
  member coverage and one NWP model". Gave the first engineer test an ECMWF-shaped fixture and a
  hand-built expected grid, and softened risk 6.
- **Second simplicity review (2026-10-07).** Most cuts were accepted; the docs edits, the
  degradation tests, and the single-run raise were kept. Accepted:
    - merged the three engineer tests into one small test against `TabularFeatureEngineer`,
      folded the unpivot test into the shedding test, and dropped the registration test and the
      `load_forecaster_from_dir` assertion;
    - put the engineer, the forecaster, and inline `save`/`load` in one module,
      `manual_heuristic.py`, deleting `_meta_io.py` and `_feature_engineer.py`;
    - dropped the member-0 stamp, after a scratch script showed the same rows without it;
    - cut `__init__` validation to "at least one power lag";
    - dropped `predict`'s log line counting rows that lost every member;
    - stated the cleaning caveat once, in risk 4;
    - stated the `ensemble_member` overload once, on `PowerForecast.ensemble_member`;
    - dropped the API page and its `mkdocs.yml` nav entry;
    - dropped the real-table timing measurement, cutting risk 6 to its fallback;
    - corrected why the member-0 engineer was rejected: only its dependence on the control member.

  Rejected (kept):
    - the `inherent-stability.md` qualifier that CV rows follow the NWP run grid;
    - the daylight-saving measurement;
    - the single-run `NotImplementedError`, now justified by naming the fault;
    - the `model-configuration.md` and `ml_core/README.md` rewrites;
    - the two roadmap sentence rewrites;
    - the roadmap Implementation-details rewrite;
    - the `power_schemas.py` description edit;
    - the all-null-rows and empty-input tests;
    - the `conftest.py` fixture change, over the optional alternative.
- **Second correctness review (2026-10-07).** All seven findings were verified against the code and
  accepted:
    - test 7 gained a member-1-only run and a clipped run, so a member-0 filter or a 360 h horizon
      constant fails it;
    - test 8 writes a member-0 training-day NWP run and the eligible table, uses the H3 cell
      `write_metadata` hard-codes, and, like test 1, hard-codes its expected lags (168·k) rather than
      reading them from the YAML;
    - added the `code-style.md` and roadmap line-997 docs edits;
    - added risk 10, on NWP-outage scenarios removing the floor's rows;
    - the shedding test's power runs to at least `init_time + 9 h`;
    - the fixture picks its defaults with `is None` and rejects `selected_features` alongside
      `config_overrides`;
    - corrected where a no-power-lag config fails, how pytest collects the package, departure 2's
      citation, risk 1's clock-change figure, and the member-span qualifier, and added the
      production-dependency clause to the sizing.

## Outline of PRs B and C (each gets its own plan or approval)

**PR B — `PersistenceForecaster` (`persistence`).** Follows the roadmap's "PR 3" text, first
extracting `PowerLagsPerNwpRunFeatureEngineer` and the `meta.json` round trip from
`manual_heuristic.py` into shared modules. `predict` is `pl.coalesce` of
`power_lag_24h`, `power_lag_48h`, `power_lag_168h`, and `power_lag_336h` in ascending-lag order, so
each row takes the shortest lag `_nullify_leaky_lags` left. The output is one member, index 0, and
`nwp_init_time` is null as in PR A. Carry the roadmap's coverage caveat: the longest lag is 336 h,
so leads from 336 h to about 351 h drop out, and persistence's `all` and `extended_range` aggregates
cover a shorter lead population than other models'.

**PR C — `ClimatologyForecaster` (`climatology`).** Follows the roadmap's "PR 5" text. `train`
deduplicates on `(time_series_id, valid_time)` before anything else, because bulk-mode `AllFeatures`
repeats each target row once per covering NWP run. It derives the cell keys — month, half-hour of
day, and weekday or weekend — from local Europe/London time inside the forecaster. It stores, per
cell, the empirical quantiles at the 13 equiprobable levels `(i − 0.5)/13` for `i` in 1 to 13, never
at `DELIVERY_QUANTILES`. The plan for PR C must state the quantile interpolation explicitly
(recommend Polars `quantile(q, "linear")`, matching `compute_metrics`). The lookup is saved as one
parquet file plus `meta.json`. `predict` joins on the cell keys, drops and logs rows in unseen
cells, and unpivots the 13 quantile columns into members 0 to 12. The README states the caveat on
small samples per cell: about 9 to 17 per weekend cell over a 14-month training window, so the
outer levels are close to the cell's minimum and maximum. Record, where the comparison with the NWP
ensemble is read, that a deterministic-quantile ensemble slightly out-scores an independent sample
of the same size under fair CRPS.
