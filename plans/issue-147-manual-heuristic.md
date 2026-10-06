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
small `FeatureEngineer` subclass filters the numerical weather prediction (NWP) input to the control
member, so the forecast-run grid is the same grid every other model is scored on, while the 51 NWP
members are never fanned out. Nothing under `src/nged_substation_forecast/defs/` changes. This plan
details PR A only; persistence (PR B) and climatology (PR C) are outlined at the end and each gets
its own approval.

## Verdict, size and departures

**Worth implementing, and now.** Four issues and the T1.2 hypothesis test wait on this baseline,
and every piece it needs (power lags up to 17,520 h, `_nullify_leaky_lags`, the `BaseForecaster`
interface, the `meta.json` servability check) already exists.

**Size: complex, with both plan reviews and both diff reviews.** The five triggers:

- **Changes what gets stored — yes.** New rows in the `power_forecasts` Delta table, a new
  saved-model format (`meta.json` only), and a new package.
- **Touches the production serving path — no code on the serving path changes.** A baseline could
  later be promoted, though, so `predict` must degrade: drop all-null rows and never raise on absent
  input.
- **Touches a degradation rule — yes.** The manual heuristic is the T1.2 floor.
- **Admits more than one defensible design — yes.** How to avoid the 51-member fan-out (a
  `BaseForecaster` flag read in `defs/`, or a feature-engineer subclass), what `nwp_init_time`
  holds, and how members are indexed all had alternatives.
- **Spans code whose callers could not be named without searching — no.** The callers are
  `trained_cv_model`, `cv_power_forecasts`, `live_forecasts`, and `compute_metrics`, all named
  below.

**Out of bounds for this PR, because other sessions are editing them:** everything under
`src/nged_substation_forecast/defs/`, `packages/contracts/src/contracts/config_schemas.py`,
`conf/cv/`, and the new failure-scenario module in `ml_core` (#437). Reading and importing from them
is fine (`class_target` and `import_class` come from `config_schemas`). `uv.lock` is also edited by
#958: whichever PR lands second re-runs `uv lock` after rebasing rather than resolving the lock file
by hand.

**Departures from the roadmap's "Implementation details — baselines":**

1. **One PR per baseline, three PRs, manual heuristic first.** PR A is the package skeleton, the
   shared `_meta_io` helper, the member-0 feature engineer, and `manual_heuristic`. PR B is
   `persistence`; PR C is `climatology`. The roadmap orders persistence first as "the
   lowest-effort end-to-end probe", but persistence unblocks nothing, while `manual_heuristic` is
   the deliverable four issues wait on. `manual_heuristic_holiday_aligned` and
   `manual_heuristic_calibrated` (#715) stay separate follow-ups.
2. **No `uses_nwp_ensemble` flag on `BaseForecaster`, and no change to `cv_power_forecasts`.** The
   roadmap's "PR 2" has `cv_power_forecasts` pass `ensemble_members=[0]` when a class sets the flag
   `False`, which edits `defs/cv_assets.py`. Instead the baseline package overrides
   `BaseForecaster.feature_engineer` — the documented composition extension point — with a
   `ControlMemberFeatureEngineer` that filters the NWP frame to `ensemble_member == 0` before
   delegating to `TabularFeatureEngineer`. `trained_cv_model` already passes `ensemble_members=[0]`,
   so only prediction changes behaviour. The bulk-mode NWP scan still defines the shared
   `(nwp_init_time, valid_time)` forecast-run grid, so every leaderboard row is scored on the same
   grid. Whether the added filter still prunes the Delta scan is measured, not assumed (see Tests).
3. **The roadmap's "PR 1" (median and p95 collapse in `ml_core/metrics.py`) is not a prerequisite.**
   The baselines emit members only. Today's metrics score the per-run ensemble mean, and CRPS and
   the pinball losses already flow over the 13 members. The "median headline plus p95 row" the
   roadmap promises for `manual_heuristic` needs PR 1, which is open question 2.
4. **Per-series surviving-member counts go to the logger, not to asset metadata.** Asset metadata
   is written in `defs/`, which is out of bounds. A unit-tested helper computes the counts and
   `predict` logs them (open question 3).
5. **`ensemble_member` needs no `UInt8 → Int8` cast.** The roadmap's PR 3 text says the spine's
   member column is `UInt8`. `Nwp.ensemble_member` and `AllFeatures.ensemble_member` are both
   declared `Int8` today. The manual heuristic builds its own member index, as an `Int8` expression.
6. **The roadmap's `annual_week_span = (49, 55)` tuple becomes two integer fields,
   `annual_first_week` and `annual_last_week`.** A registered experiment's `config_overrides`
   replace whole values under `model_params`, so two integers let an experiment move one end of the
   annual window without restating the other. `flatten_config` would stringify a tuple without
   trouble, so this is about overrides, not about MLflow parameters.

**One correction to the brief this plan was written from.** The NWP-absent path is not in
`defs/_engineering_inputs.py`. The NWP-absent path is in `ml_core/features/_nwp.py`
(`_join_nwp_bulk_mode`, the `processed_nwp is None` branch) and in
`tabular_feature_engineer._engineer_features`. That path sets `power_fcst_init_time = valid_time`,
so every row has lead time 0, and the comment at the end of `_engineer_features` records that
predict output built from that path always fails `PowerForecast.validate`. The path is
training-only. `load_engineering_inputs` always returns an NWP frame anyway, so the baselines use
the NWP-present path with member 0 only.

## What changes, file by file

### `packages/baseline_forecasters/` (new package)

**The layout mirrors `packages/xgboost_forecaster/`.** `pyproject.toml` declares the project
`baseline-forecasters`, built by `uv_build` like its sibling, depending on `contracts`, `ml_core`,
`patito`, and `polars` only. No `pytest` entry: test tooling lives in the root dev group
(`docs/architecture/testing.md`).

- `src/baseline_forecasters/__init__.py` re-exports `ManualHeuristicForecaster`,
  `ManualHeuristicConfig`, and `ControlMemberFeatureEngineer`, with a module docstring stating the
  `ensemble_member` overload.
- `src/baseline_forecasters/_meta_io.py` holds the shared `meta.json` round trip.
- `src/baseline_forecasters/_feature_engineer.py` holds `ControlMemberFeatureEngineer`.
- `src/baseline_forecasters/manual_heuristic.py` holds `ManualHeuristicConfig` and
  `ManualHeuristicForecaster`. The class must sit at module level, because `class_target` names it
  in `meta.json` and in `conf/model/manual_heuristic.yaml`.
- `tests/` holds the unit tests listed under Tests.
- `README.md` is described under Docs.

### `_meta_io.py`

**Two functions, `write_meta` and `read_meta`, shared by every baseline whose saved model is a
`meta.json` alone.** `write_meta(path, *, forecaster, trained_time_series_ids)` clears `path` with
`shutil.rmtree(path, ignore_errors=True)`, recreates it, and writes the three keys
`production_helpers._check_meta_is_servable` reads: `model_params` (`model_dump(mode="json")`),
`trained_time_series_ids` (sorted), and `model_class` (`class_target(forecaster)`). `read_meta(path,
*, forecaster_cls)` returns the config built through `forecaster_cls.CONFIG_CLASS.model_validate`
and the list of ids. No `StatelessForecaster` base class, as the roadmap says, until a third
stateless model exists.

### `ControlMemberFeatureEngineer`

**A `TabularFeatureEngineer` subclass whose `engineer` filters `nwp` to `pl.col("ensemble_member")
== NWP_ANALYSIS_MEMBER` (from `weather_utils`), re-wraps the result with
`pt.LazyFrame.from_existing(...).set_model(Nwp)`, and calls `super().engineer` with every other
argument unchanged.** The filter is applied to the raw scan, before `_attach_containing_nwp_cell`,
so the predicate sits directly above the Delta scan in the plan and can be pushed into it.

The class docstring says three things. First, the forecaster consumes no weather, and the control
member's runs are there only to define the `(nwp_init_time, valid_time)` grid every leaderboard row
is scored on. Second, `weather_source: "none"` therefore does not mean "no NWP input". Third, the
filter must reach the Parquet scan, and the test below proves the filter does.

**A projection to the key columns is not added unless measurement shows Polars fails to prune the
unused weather columns itself.** The baseline requests no weather feature, so the 30-minute upsample
interpolates columns nothing reads. Projection pushdown should drop them. The pruning test below
also inspects the projected columns, and a `select` of the key columns is added only if that test
shows the weather columns being decoded (design principle 12, "Measure; do not assume").

### `ManualHeuristicConfig(BaseForecasterConfig)`

**Three fields drive the 13 analogue lags: `n_weekly_analogues: int = 6`, `annual_first_week: int =
49`, and `annual_last_week: int = 55`.** A read-only property `analogue_lag_hours -> tuple[int,
...]` returns `168 × w` for `w` in `1..n_weekly_analogues` followed by `168 × w` for `w` in
`annual_first_week..annual_last_week`, which is already ascending. `selected_features` is declared
again with `default_factory=set`, so `conf/model/manual_heuristic.yaml` omits it.

**A `model_validator(mode="after")` derives `selected_features` and checks the fields.** The
validator:

- fills an empty `selected_features` with `{f"power_lag_{h}h" for h in analogue_lag_hours}`;
- accepts a non-empty `selected_features` equal to that set, which is what a stored config reloads
  with;
- raises `ValueError` on a non-empty `selected_features` that differs, naming both sets, so an
  override of `selected_features` cannot be silently ignored;
- raises when `annual_first_week > annual_last_week`, when `annual_first_week <=
  n_weekly_analogues` (overlapping weeks would collapse two members into one feature name), or when
  `annual_last_week × 168` exceeds the feature parser's 17,520 h cap; `n_weekly_analogues` takes
  `ge=1` through `Field`.

`weather_source` and `training_strategy` keep their base-class defaults; the YAML sets both to
`"none"`. `selected_features` keeps the base class's sorted serialiser.

### `ManualHeuristicForecaster(BaseForecaster)`

**Class constants:** `MODEL_NAME = "manual_heuristic"`, `MODEL_VERSION = 1`, `CONFIG_CLASS =
ManualHeuristicConfig`, and `feature_engineer = ControlMemberFeatureEngineer()`. Instance state is
`self._trained_ids: list[int]`, exposed by `trained_time_series_ids`.

**`train(data, time_series_ids)` records the sorted ids that have at least one non-null `power`
row and are in `time_series_ids`.** These semantics mirror XGBoost's "no usable rows, not
trained". The collect selects `time_series_id` and `power` only, drops nulls, takes unique ids, and
streams (`engine="streaming"`). Projection pushdown keeps the collect small, though the 13 lag joins
may still run (risk 7).

**`predict(data, *, fold_id="live")` builds the members in eight steps:**

1. Collect with `engine="streaming"`, then strip the Patito model with `.as_polars()` so no later
   step can hit the dict-cast trap (`polars-patito-gotchas`).
2. Keep only rows whose `time_series_id` is in `trained_time_series_ids` (the train==predict
   invariant).
3. Deduplicate on `(time_series_id, power_fcst_init_time, valid_time)` with `keep="any"`. Power
   lags do not vary by NWP member, so this makes `predict` correct even if a caller hands it the
   full ensemble, at no cost when the input is member 0 only.
4. Unpivot the 13 lag columns, indexed by `time_series_id`, `valid_time`, and
   `power_fcst_init_time`. The input's own `ensemble_member` and `nwp_init_time` are not carried.
5. Map each lag column name to its member index — the lag's rank in `analogue_lag_hours`, so 0 to 5
   are the weekly analogues and 6 to 12 the annual ones — with `replace_strict(...,
   return_dtype=pl.Int8)`. Ranking by lag means a member keeps its meaning as
   `_nullify_leaky_lags` sheds the short weekly lags with lead time: past 168 h of lead, member 0
   is absent and members 1 to 12 keep their indices.
6. Drop null values. A null comes from `_nullify_leaky_lags` or from history that does not reach
   back far enough.
7. Count the forecast rows (one per series, init time, and valid time) that lost every member, per
   series, with a helper `_dropped_row_counts(before, after)` returning a small frame. Log it
   through the module logger at `WARNING` when any row lost every member, naming each series and
   its count, and at `INFO` otherwise with the per-series surviving-member range. Nothing raises.
8. Add `power_fcst` (the value cast to `Float32` by expression), `nwp_init_time` as a typed null
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

**`save(path)` calls `write_meta`; `load(path)` calls `read_meta`.** The saved directory holds only
`meta.json`; `save_to_mlflow` adds the frozen metadata parquet as for any model.

### `conf/model/manual_heuristic.yaml` (new)

**`_target_: baseline_forecasters.manual_heuristic.ManualHeuristicForecaster`, with `model_params`
holding the three analogue fields, `weather_source: "none"`, and `training_strategy: "none"`.**
The header comment follows `conf/model/xgboost.yaml`'s, and adds that `selected_features` is derived
and that overriding it to anything else fails registration.

### Root `pyproject.toml` and `uv.lock`

**Add `"baseline_forecasters"` to `[project] dependencies` beside `"xgboost_forecaster"`, and
`baseline_forecasters = { workspace = true }` to `[tool.uv.sources]`, then run `uv lock`.** A
production dependency rather than a dev-group entry, because `load_forecaster_from_dir` imports
whatever class a promoted `meta.json` names. The package pulls in nothing new (contracts, ml_core,
patito, polars), so the `uv export --no-dev` check in `docs/architecture/testing.md` is unaffected.

### `packages/contracts/src/contracts/power_schemas.py` and `ml_schemas.py`

**Description text only; no dtype, nullability, or constraint changes.** `PowerForecast.ensemble_member`'s
description gains: an NWP-member index for models that consume an NWP ensemble, a historical-analogue
index (rank of the lag, 0 = shortest) for `manual_heuristic`, and, from PR C, a quantile-sample index
for `climatology`, so `ensemble_member` does not imply NWP. `AllFeatures.ensemble_member` gains a
description saying the column is the NWP member the row's weather came from, null where the NWP join
missed. Neither change alters what a frame must contain, so the "ask before changing a Patito data
contract" rule is met by this plan's approval.

### `tests/` at the root

- `tests/conftest.py`: the `register_experiment` fixture gains optional `base_model_config` and
  `config_overrides` arguments, defaulting to today's XGBoost values, so existing callers are
  unchanged.
- `tests/test_forecaster_config_serialisation.py`: import `ManualHeuristicConfig` and extend the
  coverage guard. Two parametrised tests construct each config with arbitrary `selected_features`.
  `test_every_config_class_forbids_extra_keys` passes `set()`, which the manual heuristic accepts.
  `test_the_declared_serialiser_actually_sorts` passes a fixed set of four unrelated features,
  which the manual heuristic's validator rejects by design. That test therefore takes its feature
  set from a per-class table, which gives `ManualHeuristicConfig` its own derived set. The test
  still checks sorting, because `power_lag_1008h` sorts before `power_lag_168h`.
- New integration test, `tests/test_manual_heuristic_cv.py`, described under Tests.

## Design-philosophy check

**The code runs in the R&D asset chain today, and is written to production rules because a baseline
could be promoted.** `predict` never raises on absent input: missing history sheds members, a row
with no members is dropped and logged, and empty input gives empty output. The logged warning names
the series, following the rule that telemetry names the fault. Nothing raises except a contract
violation (`PowerForecast.validate`) or a bad config at registration, both of which are our own bugs.
No asset check is added.

**The change delivers the measuring instrument for T1.2.** T1.2 asks that every series still emits a
forecast at rungs 0 to 2 of the degradation ladder and still beats `manual_heuristic`. This PR does
not measure T1.2; #438 does, and needs this baseline first.

**Design principles 3 and 8 are kept.** The baseline rides the same `register_experiment_job →
trained_cv_model → cv_power_forecasts → metrics` chain as XGBoost (principle 3, one execution
path), and is scored on the identical forecast-run grid (principle 8, every experiment scored
identically). Principle 11 (push the work down to the query engine) is what the pruning test
guards.

**One principle is traded, deliberately and visibly: the floor's row grid comes from the NWP
archive.** `docs/design-philosophy/inherent-stability.md` says the manual heuristic "consumes no NWP
data, so an NWP outage does not degrade it at all". In CV, an `init_time` missing from the NWP
archive removes that run's rows for the manual heuristic as for every other model. That keeps the
comparison like-for-like, which is what the leaderboard needs. In production it would mean no
forecast during an NWP outage — the opposite of what the floor promises (open question 6).

## Tests

Each test names the assertion that fails on `main` today. On `main` the package does not exist, so
every import fails; the assertions below are the behavioural claims each test pins once the import
resolves.

**`packages/baseline_forecasters/tests/test_manual_heuristic.py` (unit):**

1. **Default lags.** `ManualHeuristicConfig().analogue_lag_hours == (168, 336, 504, 672, 840, 1008,
   8232, 8400, 8568, 8736, 8904, 9072, 9240)` and `selected_features` is the matching set of 13
   names.
2. **Validator.** A config dumped and re-validated equals the original; a conflicting
   `selected_features` raises `ValidationError`; `annual_first_week=50, annual_last_week=49` raises;
   `n_weekly_analogues=49` raises (overlap); `annual_last_week=105` raises (beyond 17,520 h).
3. **Unpivot.** A hand-built two-row `AllFeatures` frame with 13 distinct lag values per row returns
   26 rows; for each, `power_fcst` equals the lag column whose rank is its `ensemble_member`;
   `ensemble_member` is `Int8`; `nwp_init_time` is null; the identity columns equal the config's.
4. **Shedding keeps member identity, through the real pipeline.** Synthetic power for one series
   over 56 weeks and a member-0 NWP grid of one run, passed through
   `ControlMemberFeatureEngineer.engineer` and `predict`: at lead under 168 h the members are 0 to
   12; at lead in [168 h, 336 h) member 0 is absent and members 1 to 12 are present with the values
   of lags 336 h to 9240 h. A mutation that indexes members by position after dropping nulls fails
   this test.
5. **All-null rows.** A row whose 13 lags are all null is absent from the output, the call does not
   raise, and `_dropped_row_counts` reports 1 for that series.
6. **Empty input.** A zero-row `AllFeatures` frame returns a zero-row frame that
   `PowerForecast.validate` accepts.
7. **Population.** Rows for a series not in `trained_time_series_ids` are absent from the output.
8. **Member fan-in.** The same features with `ensemble_member` 0, 1, and 2 return exactly the frame
   returned for member 0 alone, with no primary-key violation.
9. **Train.** `train` on rows for ids {1, 2, 3}, where id 2 has only null power and id 3 is not
   requested, records `[1]`.
10. **Save and load.** `load(save(...))` returns the same `trained_time_series_ids` and an equal
    config; a stale file placed in the directory before `save` is gone afterwards; `meta.json`'s
    `model_class` equals `"baseline_forecasters.manual_heuristic.ManualHeuristicForecaster"`;
    `production_helpers.load_forecaster_from_dir` reconstructs the class from the directory.
11. **Scoring.** `compute_metrics` on a hand-built 13-member output and matching actuals returns a
    fair CRPS equal to the hand-computed value, so the members are scored as an ensemble.

**`packages/baseline_forecasters/tests/test_feature_engineer.py` (unit):**

- **Test 12: Member filter.** NWP holding members 0, 1, and 2 gives engineered rows whose
  `ensemble_member` is 0 on every row, with the row count of a member-0-only input.
- **Test 13: The member-0 predicate reaches the Delta scan.** Write a small NWP table through
    `delta_store.nwp.write_nwp` into `tmp_path`, scan it with `Nwp.scan_delta` and the same
    `init_time`, `valid_time`, and `h3_index` filters `load_engineering_inputs` applies (with no
    `ensemble_members` filter, as `cv_power_forecasts` calls it), run
    `ControlMemberFeatureEngineer.engineer`, and assert that every NWP scan node in the engineered
    plan's `explain()` carries the `ensemble_member` predicate. The same test records which columns
    the scan projects, for the projection decision above. If the predicate does not reach the scan,
    the fallback is a change to `defs/cv_assets.py`, which is out of bounds: stop and ask the
    maintainer before editing `defs/`.

**A one-off measurement on the real NWP table goes in the PR body.** Engineer one 14-day
`init_time` chunk of the `mid_2025_to_mid_2026` validation window through
`ControlMemberFeatureEngineer` with no member filter at the loader, and again with
`ensemble_members=[0]` at the loader. Record wall time and peak resident memory for each: the two
should match to within noise, and both should sit far below the 51-member read.

**`tests/test_manual_heuristic_cv.py` (integration, `pytestmark = pytest.mark.integration`):**

- **Test 14: End to end through the real assets.** Register `conf/model/manual_heuristic.yaml`
  through the extended fixture, write synthetic power covering 56 weeks before a validation day and
  NWP with members 0, 1, and 2, then materialise `trained_cv_model` and `cv_power_forecasts`. Assert
  that the `power_forecasts` rows carry `power_fcst_model_name == "manual_heuristic"`, members 0 to
  12 at short lead, values equal to the synthetic power at each lag (to the stored 13-bit
  significand), null `nwp_init_time`, and no row duplicated across the three NWP members.

**`tests/test_jobs.py` (extended):**

- **Test 15: Registration.** `_resolve_forecaster_config` on `conf/model/manual_heuristic.yaml` with
  no overrides yields the 13 derived features; with `annual_last_week: 54` it yields 12; with a
  conflicting `selected_features` override it raises.

## Docs to update

Every page is written to describe the code as it now stands, with no "was previously" wording.

- **`docs/roadmap/metrics-and-leaderboard.md`, "Implementation details — baselines".** Reorder the
  PR list to manual heuristic, persistence, climatology. Replace the "PR 2" item's
  `uses_nwp_ensemble` text with the `ControlMemberFeatureEngineer` design, and keep its
  `ensemble_member` overload bullet. Fold "PR 1" into a note that the metrics collapse is its own
  piece of work. Delete the `manual_heuristic` item's details at ship time, with the summary moving
  to the PR body. Correct the `UInt8 → Int8` sentence in the persistence item, and the
  asset-metadata wording in the persistence and climatology items. The section and its 🚧 banner
  stay until PR C ships.
- **`docs/ml_experimentation/model-configuration.md`.** The sentence "The only one today is
  `conf/model/xgboost.yaml`" becomes a list of the two YAML files, noting that `manual_heuristic`
  derives `selected_features` from its three analogue fields.
- **`packages/baseline_forecasters/README.md`, `docs/api/baseline_forecasters/index.md`, and the
  `mkdocs.yml` "API reference" nav.** The README states what each baseline emits, the
  `ensemble_member` overload, why the member-0 feature engineer exists, and the daylight-saving
  caveat. The API page follows `docs/api/xgboost_forecaster/index.md`.
- **CLAUDE.md, the Packages table.** One row for `baseline_forecasters`.
- **`docs/architecture/performance.md`, "Bounding feature-engineering memory".** One sentence
  saying `ControlMemberFeatureEngineer` applies the member-0 predicate for forecasters that consume
  no weather, citing test 13.

## Verification commands

Run all of these green before every push:

- `uv run ruff check .` and `uv run ruff format .`
- `uv run --all-packages ty check`
- `uv run pytest`, which collects `packages/baseline_forecasters/tests/` automatically because the
  root project now depends on the package
- `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md`
- `uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .`
- `uv run --no-sync python scripts/lint/check_docs_links.py`
- `uv run mkdocs build --strict`, then read the rendered `site/api/baseline_forecasters/` page and the
  edited roadmap section
- `test "$(uv export --no-dev --format requirements-txt | grep -ciE '^(pvlib|cdsapi)')" -eq 0`
- `uv lock --check` after rebasing on whichever of this PR and #958 lands first

**Before reading any result, run the data check the roadmap asks for.** Count, per eligible series,
the observations at least 49 weeks before `val_start`. A series eligible under
`min_training_months` can still lack the annual analogues and degrade silently to a weekly-only
ensemble of at most 6 members.

## Risks and open questions

1. **Daylight saving time.** A lag of a multiple of 168 h is a whole number of weeks in UTC. Across
   a clock change, the analogue sits an hour away in local clock time from the target, whereas the
   operator's method matches local weekday and time of day. By arithmetic, a weekly member
   straddles a clock change for the 6 weeks after each change, which is roughly a quarter of target
   weeks; annual members straddle one only near the changes. *Recommendation:* ship UTC lags as the
   roadmap specifies, which keeps the baseline on the audited lag machinery. Measure the share of
   member values affected in the PR, and document the caveat in the README. Make local-clock
   alignment a follow-up alongside `manual_heuristic_holiday_aligned`, which needs a bespoke
   analogue picker anyway.
2. **Metrics collapse.** The roadmap's "PR 1" (median headline, `mae`/`mbe` at
   `metric_param="p95"`) has not landed, so `manual_heuristic`'s deterministic metrics currently
   score the 13-member mean. *Recommendation:* track the collapse as its own issue and PR, in
   parallel with this one; neither blocks the other. Not in this plan's scope.
3. **Per-series surviving-member counts in asset metadata** would need `defs/`. *Recommendation:*
   log them from `predict` through the tested helper now, and ask later whether `cv_power_forecasts`
   should record them as metadata once `defs/` is free.
4. **Sub-issues.** The roadmap asks for one tracked sub-issue per PR under #147, plus one for
   `manual_heuristic_holiday_aligned`. *Recommendation:* create them after this plan is approved,
   following the `github-issue-pr-workflow` skill.
5. **Series with less than 55 weeks of history** degrade silently to fewer members.
   *Recommendation:* the warning in `predict` names them, and the data check above counts them
   before any number is read.
6. **The floor is not yet servable live, and its CV rows depend on the NWP archive.** Two pieces of
   `defs/live_forecast_assets.py` stand in the way of promoting `manual_heuristic`. `LIVE_POWER_HISTORY`
   is 15 days, so the live spine has no power for the annual analogues. The live path also drops
   rows whose `ensemble_member` is null, so an NWP outage would leave the manual heuristic emitting
   nothing. *Recommendation:* no change here, because no baseline is promoted by this work; open a
   follow-up issue for both before anyone promotes a baseline, and have #438 score the manual
   heuristic on the clean grid under every failure scenario.
7. **Training engineers 13 lag columns only to read the population.** `train` needs only the ids
   with non-null power, but `trained_cv_model` hands it the full engineered frame. *Recommendation:*
   accept the cost (member 0, about the cost of an XGBoost training read), and record the measured
   `trained_cv_model` time in the PR body.
8. **`test_forecaster_config_serialisation.py` changes shape.** Its sort test hard-codes a feature
   set that a derived-features config must reject. *Recommendation:* the per-class feature table
   described above, rather than relaxing the validator to overwrite a conflicting value silently.

## Outline of PRs B and C (each gets its own plan or approval)

**PR B — `PersistenceForecaster` (`persistence`).** Follows the roadmap's "PR 3" text, reusing
`_meta_io` and `ControlMemberFeatureEngineer` from PR A. `predict` is `pl.coalesce` of
`power_lag_24h`, `power_lag_48h`, `power_lag_168h`, and `power_lag_336h` in ascending-lag order, so
each row takes the shortest lag `_nullify_leaky_lags` left. The output is one member, index 0, and
`nwp_init_time` is null as in PR A. Carry the roadmap's coverage caveat: the longest lag is 336 h, so
leads from 336 h to about 351 h drop out, and persistence's `all` and `extended_range` aggregates
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
