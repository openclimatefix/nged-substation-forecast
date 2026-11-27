# Plan: store `PowerForecast` as 13 quantile columns (issue #1708)

**The problem.** `PowerForecast` stores one row per series, forecast run, valid time, and
`ensemble_member`, with one deterministic value in `power_fcst`. The schema's docstring says the
spread across members expresses the uncertainty. The research neural networks for the Primaries (the
step-5 network of research experiment 705 and the Candidate of research experiment 708) output one
forecast of quantiles per series and half-hour, not one forecast per weather member. Today such a
forecast can only be stored by encoding quantile level k as `ensemble_member` k, as
`ClimatologyForecaster` does. The plan of PR #1707, which serves the step-5 network, depends on this
change and writes the quantile columns it adds. The member encoding makes every reader treat
quantiles as exchangeable members: the fair CRPS treats the 51 values as random draws, the
spread-skill ratio takes their variance, and the dashboard draws each level as a grey "member" line.

**The planned solution.** Replace `ensemble_member` and `power_fcst` with 13 Float32 columns,
`power_fcst_p1` to `power_fcst_p99` (named with `contracts.common.quantile_label`), at the 13 levels
of `contracts.common.DELIVERY_QUANTILES` agreed with NGED (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50,
0.65, 0.80, 0.90, 0.95, 0.98, 0.99). The primary key becomes `(time_series_id,
power_fcst_init_time, valid_time)`, and a row constraint forbids a quantile below the level beneath
it. The three forecasters that compute members today reduce their members to the 13 quantiles with
one shared function in `ml_core` before they return. `ml_core.metrics` scores the quantile columns
directly, at the same 13 levels it scores on `main`. The stored `power_forecasts` and
`forecast_metrics` Delta tables are deleted and recreated at their existing paths with the new
schema. The dashboard draws the power forecast as quantile bands, and its NWP panel keeps one line
per ECMWF ENS weather member, read from `Nwp` as now.

## Verdict, size and departures

**Verdict: worth implementing, roughly as the issue describes.** The research networks are the next
production forecasters (PR #1707), and storing their quantiles as members would leave every
member-based metric computing something its docstring does not describe.

**Size: complex.** The five triggers:

- **What gets stored:** fires. It changes a Patito contract and the layout of the `power_forecasts`
  Delta table.
- **The production serving path:** fires. `live_forecasts` and `XGBoostForecaster.predict` change.
- **A degradation rule:** fires weakly. `live_forecasts_are_healthy` describes the member count of
  the forecast rows in its message; that description goes, and `nwp_run_is_complete` keeps reporting
  missing members as now.
- **More than one defensible design:** fires. Wide or long quantile columns (long was considered
  and is not recommended: every reader would pivot it back); which quantile levels to store (the
  maintainer chose the 13 delivery levels); whether to keep, convert, or delete the stored
  forecasts (the maintainer chose to delete them).
- **Callers found only by searching:** fires. `PowerForecast` is named in about 30 source and test
  files across `contracts`, `ml_core`, `delta_store`, `baseline_forecasters`, `xgboost_forecaster`,
  `dashboard`, `scripts/forecasting`, and `src/nged_substation_forecast`.

**Reviews:** both plan reviews (simplicity, then correctness), and all three diff reviews in
`implement-issue`, including the mutation pass, because the monotone constraint, the reduction, and
the metric formulas are behaviours the tests are meant to pin down.

**Departures from the issue body:** none of substance. Issue #262 asked for percentile columns
*beside* the member representation; this plan replaces the member representation instead, because
no production reader needs both and two representations double every writer and reader.
Representation 3 (quantiles per member) is not planned. When this work is done, issue #262 is
closed as not planned, with a comment saying that Representation 2 is delivered by this change and
Representation 3 will not be built.

## Solution

### Walk through

**The happy path, live.** `live_forecasts` loads one ENS run and calls
`XGBoostForecaster.predict` as now. `predict` still builds one feature row per member and gets one
booster prediction per member. Before returning, it calls the new
`ml_core.quantiles.to_stored_quantiles(frame, value_col="power_fcst")`. That function groups by
`(time_series_id, power_fcst_init_time, valid_time)` plus the constant columns, takes the linear
empirical quantile of the member values at each of the 13 `contracts.common.DELIVERY_QUANTILES`
levels, and returns one row per group with the 13 `power_fcst_pXX` columns. A group with one member
returns 13 equal columns, which is the honest description of a deterministic forecast. `predict`
returns `PowerForecast.validate(...)` on that frame, and the monotone constraint passes by
construction because empirical quantiles never decrease. `live_forecasts` keeps its
`ensemble_member.is_not_null()` filter on the feature rows before `predict`: rows the NWP join
missed carry a null `ensemble_member` (`tabular_feature_engineer.py`), XGBoost predicts finite
values for them, and after the reduction nothing would mark them. It then calls
`write_power_forecasts`, which rounds all 13 columns to `POWER_FCST_SIGNIFICAND_BITS`, writes each
with the `BYTE_STREAM_SPLIT` encoding that `power_fcst` has today, sorts by the new three-column
`PRIMARY_KEY`, and replaces the slot's rows under the existing predicate.

**The happy path, CV and metrics.** `cv_power_forecasts` writes the same reduced frames through
the same writer. `ml_core.metrics.compute_metrics` joins the actuals onto the quantile rows. It no
longer collapses members: each row already is one forecast. Per row it computes:

- CRPS as twice the unweighted mean pinball loss over the 13 delivery levels. This is a choice.
  The 13 levels are tail-heavy (six of them sit at or beyond p10 and p90), so this CRPS weights the
  tails more than the same estimator over the 11 research levels of research experiment 704, and
  neither matches `studies.cross_validation.crps` or the probability-mass weighting of research
  experiments 5 to 20. Research forecasts stored at the 11 research levels are therefore not
  directly comparable with production CRPS until they are re-exported at the 13 levels;
- the deterministic metrics (MAE, RMSE, MBE, NMAE) on `power_fcst_p50`, as issue #1077 proposes;
- pinball loss at each of the 13 delivery levels, and PICP and interval width for the six existing
  delivery bands (`p1_p99`, `p2_p98`, `p5_p95`, `p10_p90`, `p20_p80`, `p35_p65`), read straight
  from the stored columns, with the loops over `DELIVERY_QUANTILES` as on `main`.

**The spread-skill ratio is retired.** Without members, its spread could only be estimated from the
quantiles by assuming a Gaussian forecast distribution, which skewed power forecasts break. PICP and
interval width for the six delivery bands already measure calibration without that assumption. The
`spread_skill_ratio` value leaves the `Metrics` contract's metric-name Enum, and its code, tests,
and docs go. Every other metric name, every `metric_param` label, and the `metric_param` Enum stay
as on `main`, because the stored levels are the levels the metrics already score.
`mean_pinball_loss` keeps its 13 levels.

**Four metrics change meaning under unchanged names:** `crps` (fair CRPS of members → quantile
CRPS), and `mae`, `rmse`, and `mbe` (ensemble mean → p50). The names reach the `forecast_metrics`
table and the MLflow keys. The `forecast_metrics` table is deleted and recreated with the
`power_forecasts` table, so no Delta table holds both definitions, and every new MLflow run is
tagged `metrics_version=2` to mark the new definitions among the old MLflow runs. The
`METRIC_NAMES` docstring in `ml_schemas.py` (fair CRPS, Fortin spread, ensemble mean, the PICP
reference of about 0.769 at 51 members) is rewritten for the quantile definitions.

**Branches off the happy path.**

- **Climatology** computes its 13 levels directly from the training power of each calendar cell
  and its pooled neighbours (`_pooled_quantile_lookup` in `climatology.py`), rather than computing
  51 levels and reducing them, because the cell's empirical quantile at the exact level is what the
  51 levels approximated. The lookup stores the 13 levels as columns named with `quantile_label`,
  and `predict` joins the lookup and renames those columns to `power_fcst_pXX`: there is no unpivot
  and no member index. `CLIMATOLOGY_MEMBER_COUNT`, `CLIMATOLOGY_QUANTILE_LEVELS`, and the
  column-to-member mapping go.
- **Manual heuristic** treats its surviving power lags as analogues, not members. After
  `_nullify_leaky_lags`, `predict` unpivots the lag columns into one value per lag and passes them
  to `to_stored_quantiles` with `value_col="power"`, so `_lag_column_to_member` and the member index
  go. With one surviving lag the 13 columns are equal, as for an XGBoost forecast from a
  one-member NWP run.
- **No ensemble-member concept stays on the power side.** After this change, `ensemble_member`
  appears only where it names an ECMWF ENS weather member: `Nwp`, the per-member feature rows that
  XGBoost predicts on, the live join-miss filter, and the dashboard's NWP panel. A set of coherent
  power scenarios, if one is added later, gets its own contract with a scenario index that is not a
  weather-member index (the delivery-recommendation page of nged-ml-research found scenarios built
  from the stored quantiles at least as good as tying scenario k to weather member k).
- **Delete and recreate the two tables.** The paths in `Settings.power_forecasts_data_path` and
  `Settings.forecast_metrics_data_path` do not change. Before the first run of the new code, the
  maintainer deletes both Delta tables, locally and in the delivery bucket, and the first write
  recreates each table with the new schema, as `write_power_forecasts` does today. No code
  converts, archives, or migrates the old rows: they mostly hold dummy forecasts, nobody outside
  the project reads either table, and both are already backed up on the maintainer's external
  disk. No study is re-run, and studies scored against the old rows stay as they are.

**Error-handling paths.**

- **A member value that is null** (a join miss): the live filter above drops those feature rows
  before `predict`, and `to_stored_quantiles` also drops null values, so a group keeps its other
  members. A group with no member left yields no row, which `live_forecasts_are_healthy` already
  reports as missing coverage.
- **A NaN quantile.** Measured on patito 0.8.6 and polars 2.0.0, a cross-column `b >= a` constraint
  accepts (NaN, NaN) and (1, NaN) and rejects only (NaN, 1), and polars' quantile sorts NaN last, so
  the ordering constraint alone does not catch a NaN. `PowerForecast` therefore gives every quantile
  field an explicit `is_finite()` constraint, so `validate` raises on a NaN or infinite quantile,
  as PR #1707's plan expects. The live check's non-finite count covers all 13 columns.
- **A thinned NWP feed.** `live_forecasts_are_healthy` does not warn about a thinned feed today; it
  only names the member count in its message. `nwp_run_is_complete` already reports missing
  members. The plan drops the member count from the check's message and from `live_forecasts`'s
  metadata, and adds no new report: counting members of the lazy `Nwp` frame would be a second
  collect, outside the asset's catch-all.
- **A crossed quantile** cannot come from `to_stored_quantiles`. A future forecaster that emits
  quantiles directly (PR #1707's network) must sort them per row before validation; the constraint
  then makes a crossing fail `validate` loudly rather than reach the table.

**Every production forecaster can produce the 13 levels.** XGBoost and the manual heuristic take
empirical quantiles of their members or lags at any level, and climatology computes any level from
its calendar cells. The research bundles of experiments 705 and 708 compute their quantiles by
bisection at the levels in their `config.json` `quantile_levels`, so they need only that config
change and a re-export in nged-ml-research.

### What changes, file by file

- **`packages/contracts/src/contracts/common.py`**: no new constant. `DELIVERY_QUANTILES` is the
  set of stored levels, and its docstring adds `PowerForecast`'s quantile columns to the artefacts
  it indexes. Column names are `f"power_fcst_{quantile_label(q)}"`, reusing the existing
  `quantile_label`, so no new helper.
- **`packages/contracts/src/contracts/ml_schemas.py`**: only the `METRIC_NAMES` docstring, rewritten
  for the quantile CRPS and the p50 deterministic metrics, and `spread_skill_ratio` leaves the
  metric-name Enum. The `metric_param` labels and Enum are unchanged.
- **`packages/contracts/src/contracts/power_schemas.py`**: `PowerForecast` loses `ensemble_member`
  and `power_fcst`, gains the 13 Float32 quantile fields (each carrying `power_fcst`'s current
  description of available power, units, and sign, written once in a shared string), per-field
  constraints (patito has no model-level constraint) that each column is finite and at least the
  column below it, and `PRIMARY_KEY =
  ("time_series_id", "power_fcst_init_time", "valid_time")`. The class docstring describes a
  quantile forecast. The PLANNED normalisation comment moves to the class docstring unchanged.
- **`packages/contracts/README.md`**: the `PowerForecast` bullet.
- **`packages/ml_core/src/ml_core/quantiles.py`** (new): `to_stored_quantiles`, shared by XGBoost
  (over weather members) and the manual heuristic (over lag analogues).
- **`packages/ml_core/src/ml_core/metrics.py`**: `ROW_KEY_COLUMNS` and `COMPARABLE_KEY_COLUMNS`
  become the three-column key (the member-count argument in their docstring goes); `compute_metrics`
  drops the member collapse; `_fair_crps_expr` is replaced by a quantile CRPS;
  `_corrected_variance_expr` and the spread-skill ratio go; the empirical delivery-quantile
  aggregation and `_quantile_column` go, because the metrics read the stored columns; the pinball,
  PICP, and interval-width loops and `_BAND_LOWER_QUANTILES` stay over `DELIVERY_QUANTILES`, as on `main`.
- **`packages/ml_core/src/ml_core/base_forecaster.py`**: the `predict` docstring.
- **`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`**: `predict` reduces with
  `to_stored_quantiles`.
- **`packages/baseline_forecasters/src/baseline_forecasters/climatology.py`**: compute the 13 levels
  per cell into a lookup of 13 `quantile_label` columns; drop `CLIMATOLOGY_MEMBER_COUNT`,
  `CLIMATOLOGY_QUANTILE_LEVELS`, the column-to-member mapping, and the unpivot to members. The saved
  lookup artefact stores 51 quantile columns per cell today, so saved climatology models are
  retrained.
  **`manual_heuristic.py`**: drop `_lag_column_to_member` and the member index; reduce the surviving
  lags with `to_stored_quantiles`; rewrite the class docstring around analogues.
  **`nwp_run_rows.py`**: the docstring that names
  `PowerForecast`. **`README.md`**: the member description.
- **`packages/delta_store/src/delta_store/power_forecasts.py`**: round every quantile column; give
  each the `BYTE_STREAM_SPLIT` column properties that `POWER_FORECASTS_WRITER_PROPERTIES` gives
  `power_fcst` today; sort by the new key; docstrings. `contracts/settings.py` is unchanged.
- **`packages/dashboard/src/dashboard/forecast_chart.py`**: `build_view_forecast_chart` draws the
  p1–p99, p5–p95, and p20–p80 bands, each fainter than the band inside it, and the p50 line
  instead of one grey line per member. Its orange comparison overlay draws the second forecast's
  p5–p95 band and p50 line.
  **`packages/dashboard/src/dashboard/view_forecasts.py`** (the marimo app) and its test: read the
  quantile columns. **`packages/notebooks/view_baseline_export.py`**: read the exported files' new
  columns.
  `build_nwp_ensemble_chart` is unchanged: it reads `Nwp` rows and keeps one line per weather
  member.
- **`src/nged_substation_forecast/defs/live_forecast_assets.py`**, **`cv_assets.py`**,
  **`checks.py`**: drop the `ensemble_member` filters and counts on forecast rows. The CV asset's
  `n_ensemble_members` metadata and MLflow tag go, because no code reads them (the one reader is
  the sentence in `docs/cross-validation/dagster-workflow.md` that lists the tag). `live_forecasts`
  keeps its `ensemble_member.is_not_null()` feature filter and drops `n_ensemble_members` from its
  metadata; `live_forecasts_are_healthy` drops the member count from its message. The live check's
  non-finite count covers all 13 columns.
- **`scripts/forecasting/score_study.py`**: no bespoke code. Its per-column dtype check and
  `PowerForecast.validate` already reject a submission in the member layout.
  **`export_baseline_forecasts.py`**: the smallest change that runs. The full-ensemble file goes;
  the ensemble-mean file goes too, because its p50 replacement would repeat a column of the
  quantile file; the quantile file holds the stored p10, p50, and p90 columns and observed power.

### Design-philosophy check

The serving path degrades as before: a missing or thinned ENS run still yields a forecast from the
members present, and `nwp_run_is_complete` reports the missing members as now. Validation stays
fail-fast in R&D and at the write. The change follows the principle that the schema describes what
the column means: a column of quantiles is no longer labelled a member.

## Tests

Each assertion below fails on `main` today, because the columns, the function, or the behaviour do
not exist there, except the two marked as regression guards.

- **`contracts/tests/test_power_forecast.py`**: one test that a frame with `power_fcst_p10 >
  power_fcst_p20` fails `validate`; a frame with two rows on the same three-column key fails; a NaN
  quantile fails `validate`.
- **`ml_core/tests/test_quantiles.py`** (new): 101 values evenly spaced 0 to 100 give every delivery
  level `q` the value `100 q` exactly; one member gives 13 equal columns; a null member value is
  dropped before the reduction; a group of all-null members yields no row.
- **`ml_core/tests/test_metrics.py`**: CRPS of a row equals twice its mean pinball loss over the 13
  levels; MAE uses p50, not the mean of the quantiles (a skewed row distinguishes the two); PICP and
  interval width for `p35_p65` read the stored `power_fcst_p35` and `power_fcst_p65` columns, with
  hand-computed values on a row whose columns differ from the empirical quantiles `main` would take
  across members; `compute_metrics` returns no `spread_skill_ratio`. The existing
  spread-skill tests go.
- **`xgboost_forecaster/tests/test_forecaster.py`**, **`baseline_forecasters/tests/*`**: `predict`
  returns one row per key, and climatology's p50 equals the cell's median. Climatology's saved
  lookup has exactly the 13 `quantile_label` columns (it has 51 on `main`). The manual heuristic
  with lags 1, 2, and 3 MW on one row returns the 13 empirical quantiles of those three values
  (`main` returns three member rows).
- **`delta_store/tests/test_power_forecasts.py`**: every quantile column is rounded to the stated
  significand bits on disk; `POWER_FORECASTS_WRITER_PROPERTIES` names all 13 quantile columns with
  `BYTE_STREAM_SPLIT`. Regression guard: the sort key is `PRIMARY_KEY` (true on `main` too).
- **`dashboard/tests/test_forecast_chart.py`**, **`test_view_forecasts.py`**: the power chart and
  its comparison overlay have band layers and no `ensemble_member` detail channel; the power
  chart's outermost band spans `power_fcst_p1` to `power_fcst_p99`. Regression
  guard: the NWP chart still has one line per member (true on `main` too).
- **`tests/test_live_forecasts.py`**: extending the predict spy, feature rows with a null
  `ensemble_member` never reach `predict`, so the stored rows hold no join-miss forecast.
- **`tests/test_checks.py`**: the live check's non-finite count counts a row whose only non-finite
  value is in `power_fcst_p99`.

## Docs to update

Only the pages that `git grep -n "ensemble_member\|power_fcst\b"` finds describing members on
forecast rows, the `PowerForecast` contract, or the metric formulas; a hit about the members of an
NWP run stays. The expected pages are `docs/roadmap/delivery-tables.md` (Table 1),
`docs/techniques/evaluation-metrics.md` (CRPS from quantiles, deterministic metrics on the median,
the passages that describe the PICP reference and the fair CRPS at 13 and 51 members, and the
spread-skill section, which is deleted), every other page that `git grep -n -i "spread.skill"`
finds (the roadmap pages, `dagster-workflow.md`, and the techniques pages), and the
`n_ensemble_members` tag sentence in `docs/cross-validation/dagster-workflow.md`. Each describes the
code as it now is. `docs/background/requirements.md` and `evaluation-metrics.md` already name the 13
`DELIVERY_QUANTILES`, so neither needs an edit for the level count.

## Verification commands

The `implement-issue` green-before-push set (`uv run ruff check .`, `uv run ruff format --check .`,
`uv run ty check`, `uv run pytest`, the markdown lint), plus `uv run mkdocs build --strict`.

## Risks and open questions

**No question is open.** The maintainer settled every decision on 2026-11-24 (see the decision
sections below):

1. **Stored levels:** the 13 `DELIVERY_QUANTILES`, as `power_fcst_p1` … `power_fcst_p99`.
2. **Tables:** one `PowerForecast` table at the existing `power_forecasts` path, and the
   `forecast_metrics` table at its existing path, both deleted and recreated. No archive, no new
   path, no study re-run.
3. **Spread-skill ratio:** retired.
4. **Research repository:** `nged-ml-research` pins `contracts`, so its studies move to the new
   layout when that repository bumps its pin, in a pull request of its own. The bundles of
   experiments 705 and 708 are re-exported with the 13 delivery levels in `quantile_levels`.
5. **Metrics:** every new MLflow run is tagged `metrics_version=2`, and the metric names stay.
6. **Issue #262:** closed as not planned when this work is done.

## Simplicity review: triage

- **Cut the conversion script and its test:** accepted. Checked delta-rs `schema_mode="overwrite"`
  as the alternative to a new path, and rejected it, because a schema overwrite replaces every
  partition and would lose the old results. Superseded: the maintainer later chose to delete the old
  results and keep the existing path.
- **Score only at the stored levels:** accepted, and superseded by the maintainer's decision to
  store the 13 delivery levels, which removes the union Enum and its six extra labels. As first
  triaged, with one change: `Metrics.metric_param` is a
  `pl.Enum` of the delivery labels and `metric_value` is not nullable, so null rows for the 6
  unstored levels are not possible; the Enum becomes the union of the delivery and stored labels.
- **Reuse `quantile_label` and drop `_quantile_column`:** accepted. Columns become
  `power_fcst_p1` … `power_fcst_p99`.
- **No bespoke rejection message in `score_study.py`:** accepted.
- **`n_ensemble_members`:** accepted. No code reads the CV metadata or tag, so both go. The live
  member count comes from the `Nwp` frame `live_forecasts` already loads, with no second read.
- **Minimal `export_baseline_forecasts.py`, docs from grep hits:** accepted.
- **Smaller points:** accepted (climatology computes the stored levels directly; one plain function
  with two callers; one monotone test; spread-skill kept with `(p90 − p10) / 2.5631`, later
  retired by the maintainer; NaN as a one-line test; long layout recorded as considered and not
  recommended).

## Correctness review: triage

- **Keep the live `ensemble_member.is_not_null()` filter:** accepted. Join-miss rows carry a null
  member (`tabular_feature_engineer.py`), and a test on the predict spy is added.
- **No thinned-NWP warning to move:** accepted. The check only names the count, and
  `nwp_run_is_complete` already reports missing members; the new report is dropped.
- **Five metrics change meaning under unchanged names:** accepted. A new `forecast_metrics` path and
  a `metrics_version=2` tag are planned, the `METRIC_NAMES` docstring is rewritten, and the choice
  is open question 5 (now four metrics, since `mean_pinball_loss` keeps its 13 levels).
- **NaN passes the ordering constraint:** accepted. Every quantile field gets an explicit
  `is_finite()` constraint, tested.
- **Missed callers:** accepted (the marimo app and its test, the comparison overlay, the notebook,
  the writer properties, the climatology artefact retrain, and the `nwp_run_rows.py` docstring).
- **The table path is NGED-facing:** accepted. The docs pages and tests that name the path are
  added, the effect on `score_study.py` and the dashboard is stated, and the choice is open
  question 2.
- **Stale PR #1707 risk:** accepted. PR #1707's plan already writes the quantile columns, so the
  ordering question is removed; issue #262 is open question 6.
- **Smaller points:** accepted (2.5631 computed from `NormalDist`, since dropped with the
  spread-skill ratio; per-field constraints, because
  patito has no model-level constraint; the CRPS estimator stated as a choice; the replacement for
  the ensemble-mean export file named).
- **Tests that pass on `main`:** accepted. Both are labelled regression guards, and the six missing
  tests are added.

## Maintainer decision: remove the remaining member code (2026-11-24)

The maintainer asked that the plan remove every ensemble-member concept from the power side now,
rather than keep code for a possible future set of coherent scenarios. Two leftovers were added:
climatology's 51-member lookup (`CLIMATOLOGY_MEMBER_COUNT` and its unpivot to members) and the
manual heuristic's lag members (`_lag_column_to_member`). The shared reduction is renamed
`to_stored_quantiles`, because its second caller reduces lag analogues rather than members. The
ECMWF ENS weather-member code stays, because it is how the forecasters read the weather ensemble.

## Maintainer decision: store the 13 delivery levels (2026-11-24)

The maintainer chose to store NGED's 13 delivery levels, `DELIVERY_QUANTILES`, rather than the 11
levels the research forecasts use. The stored levels are then the scored levels and the delivered
levels, so `STORED_QUANTILES`, the union `metric_param` Enum, and its six extra labels go, and
`mean_pinball_loss` keeps its meaning. Every production forecaster computes quantiles at any level,
and the research bundles of experiments 705 and 708 change only the `quantile_levels` of their
`config.json` and are re-exported.

## Maintainer decisions: dashboard bands and the metrics tag (2026-11-24)

The maintainer asked for a fainter p1–p99 band on the dashboard's power chart, outside the p5–p95
and p20–p80 bands. The maintainer also settled that new MLflow runs carry the tag
`metrics_version=2`, with the metric names unchanged.

## Maintainer decisions: one table, no spread-skill ratio, and issue #262 (2026-11-24)

The maintainer chose one `PowerForecast` table at the existing path rather than a second table. The
old `power_forecasts` and `forecast_metrics` tables are deleted, because they mostly hold dummy
forecasts, nobody outside the project reads them, and both are already backed up. Studies are not
re-run. The spread-skill ratio is retired. Issue #262 is closed as not planned once this work is
done.
