# Plan: store `PowerForecast` as 11 quantile columns (issue #1708)

**The problem.** `PowerForecast` stores one row per series, forecast run, valid time, and
`ensemble_member`, with one deterministic value in `power_fcst`. The schema's docstring says the
spread across members expresses the uncertainty. The research neural networks for the Primaries
(the step-5 network of research experiment 705 and the Candidate of research experiment 708) output
one forecast of 11 quantiles per series and half-hour, not one forecast per weather member. Today
such a forecast can only be stored by encoding quantile level k as `ensemble_member` k, as
`ClimatologyForecaster` does and as the plan of PR #1707 proposes for the step-5 network. That
encoding makes every reader treat quantiles as exchangeable members: the fair CRPS treats the 51
values as random draws, the spread-skill ratio takes their variance, and the dashboard draws each
level as a grey "member" line.

**The planned solution.** Replace `ensemble_member` and `power_fcst` with 11 Float32 columns,
`power_fcst_q01` to `power_fcst_q99`, at the levels the research forecasts store (0.01, 0.05, 0.10,
0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99). The primary key becomes `(time_series_id,
power_fcst_init_time, valid_time)`, and a row constraint forbids a quantile below the level beneath
it. The three forecasters that compute members today reduce their members to the 11 quantiles with
one shared function in `ml_core` before they return. `ml_core.metrics` scores the quantile columns
directly. The stored `power_forecasts` Delta table moves to a new URI, filled by converting the old
table's partitions with the same reduction. The dashboard draws the power forecast as quantile
bands, and its NWP panel keeps one line per ECMWF ENS weather member, read from `Nwp` as now.

## Verdict, size and departures

**Verdict: worth implementing, roughly as the issue describes.** The research networks are the next
production forecasters (PR #1707), and storing their quantiles as members would leave every
member-based metric computing something its docstring does not describe.

**Size: complex.** The five triggers:

- **What gets stored:** fires. It changes a Patito contract and the layout of the `power_forecasts`
  Delta table.
- **The production serving path:** fires. `live_forecasts` and `XGBoostForecaster.predict` change.
- **A degradation rule:** fires weakly. `live_forecasts_are_healthy` reads `n_ensemble_members` to
  report a thinned NWP feed, and that report moves to the NWP side (below).
- **More than one defensible design:** fires. Wide or long quantile columns; 11 or 13 levels;
  converting or re-running the stored forecasts.
- **Callers found only by searching:** fires. `PowerForecast` is named in about 30 source and test
  files across `contracts`, `ml_core`, `delta_store`, `baseline_forecasters`, `xgboost_forecaster`,
  `dashboard`, `scripts/forecasting`, and `src/nged_substation_forecast`.

**Reviews:** both plan reviews (simplicity, then correctness), and all three diff reviews in
`implement-issue`, including the mutation pass, because the monotone constraint, the reduction, and
the metric formulas are behaviours the tests are meant to pin down.

**Departures from the issue body:** none of substance. Issue #262 asked for percentile columns
*beside* the member representation; this plan replaces the member representation instead, because
no production reader needs both and two representations double every writer and reader.
Representation 3 (quantiles per member) stays out of scope, as the issue says.

## Solution

### Walk through

**The happy path, live.** `live_forecasts` loads one ENS run and calls
`XGBoostForecaster.predict` as now. `predict` still builds one feature row per member and gets one
booster prediction per member. Before returning, it calls the new
`ml_core.quantiles.members_to_quantiles(frame, value_col="power_fcst")`. That function groups by
`(time_series_id, power_fcst_init_time, valid_time)` plus the constant columns, takes the linear
empirical quantile of the member values at each of the 11 `contracts.common.STORED_QUANTILES`
levels, and returns one row per group with the 11 `power_fcst_qXX` columns. A group with one member
returns 11 equal columns, which is the honest description of a deterministic forecast. `predict`
returns `PowerForecast.validate(...)` on that frame, and the monotone constraint passes by
construction because empirical quantiles never decrease. `live_forecasts` drops its
`ensemble_member.is_not_null()` filter (the reduction has already removed the join-miss rows,
because `members_to_quantiles` drops null member values first) and calls `write_power_forecasts`,
which rounds all 11 columns to `POWER_FCST_SIGNIFICAND_BITS`, sorts by the new three-column
`PRIMARY_KEY`, and replaces the slot's rows under the existing predicate.

**The happy path, CV and metrics.** `cv_power_forecasts` writes the same reduced frames through
the same writer. `ml_core.metrics.compute_metrics` joins the actuals onto the quantile rows. It no
longer collapses members: each row already is one forecast. Per row it computes:

- CRPS as twice the mean pinball loss over the 11 stored levels (the definition the research
  experiments use, so production and research scores agree);
- the deterministic metrics (MAE, RMSE, MBE, NMAE) on `power_fcst_q50`, as issue #1077 proposes;
- pinball loss, PICP, and interval width at the 13 `DELIVERY_QUANTILES`, with the 6 delivery levels
  that are not stored (0.02, 0.20, 0.35, 0.65, 0.80, 0.98) read by linear interpolation of the
  stored quantile function between its two neighbouring stored levels;
- the spread for the spread-skill ratio as the Gaussian-equivalent standard deviation
  `(q90 − q10) / (2 × 1.2816)`.

The `Metrics` contract keeps every metric name and `metric_param` label, so the leaderboard tables
and MLflow keys do not change shape.

**Branches off the happy path.**

- **Climatology** computes its 11 levels directly from the training power of each calendar cell
  (`climatology.py`), rather than computing 51 levels and reducing them, because the cell's
  empirical quantile at the exact level is what the 51 levels approximated.
- **Manual heuristic** reduces however many lag members survived `_nullify_leaky_lags` (one per lag
  column in `_lag_column_to_member` at most). With one surviving lag the 11 columns are equal, as
  for a one-member XGBoost forecast.
- **A fresh table.** `write_power_forecasts` writes to the new table URI
  (`Settings.power_forecasts_data_path`, whose default path segment changes from `power_forecasts`
  to `power_forecasts_q11`). The first write creates the table with the new schema, as today.
- **Stored history.** A one-off script,
  `scripts/forecasting/convert_power_forecasts_to_quantiles.py`, reads the old table one
  `(experiment_name, fold_id)` partition at a time, applies `members_to_quantiles`, and writes each
  partition to the new table with `replace_partition`, so a crashed conversion is resumed by
  rerunning it. The old table is left in place, read-only, until the maintainer deletes it. The live
  slot that the live check compares against needs only the newest live run, so the live service does
  not wait for the conversion.

**Error-handling paths.**

- **A member value that is null** (a join miss): `members_to_quantiles` drops null values before
  the reduction, so a group keeps its other members, and a group with no member left yields no row,
  which `live_forecasts_are_healthy` already reports as missing coverage. **A NaN member value** (a
  NaN booster output) is not dropped, so the row's quantiles become NaN, as `power_fcst` would be
  today. The live check today counts rows whose `power_fcst` is null, NaN, or infinite in the stored
  table; after the change it counts rows where any of the 11 columns is. The implementer checks
  whether `PowerForecast.validate` rejects a NaN Float32 value, and records the answer in the test.
- **A thinned NWP feed.** Today the live check counts `n_ensemble_members` on the forecast rows to
  warn when the ENS run carries fewer members. After the change the forecast rows carry no member,
  so the live check reads the member count from the NWP run the slot used
  (`contracts.weather_schemas.assess_nwp_run_completeness` already computes it), and the warning
  text is unchanged. The check stays `WARN` and `blocking=False`, and its body still cannot raise.
- **A crossed quantile** cannot come from `members_to_quantiles`. A future forecaster that emits
  quantiles directly (PR #1707's network) must sort them per row before validation; the constraint
  then makes a crossing fail `validate` loudly rather than reach the table.

### What changes, file by file

- **`packages/contracts/src/contracts/common.py`**: add `STORED_QUANTILES` (the 11 levels) and a
  helper naming each level's column (`power_fcst_quantile_column(q)` → `"power_fcst_q05"`).
  `DELIVERY_QUANTILES` stays as the 13 levels agreed with NGED for the delivery tables, with its
  docstring updated to say the metrics interpolate the 6 unstored levels.
- **`packages/contracts/src/contracts/power_schemas.py`**: `PowerForecast` loses `ensemble_member`
  and `power_fcst`, gains the 11 Float32 quantile fields (each carrying `power_fcst`'s current
  description of available power, units, and sign, written once in a shared string), a
  `dataframe`-level constraint that each column is at least the one below it, and `PRIMARY_KEY =
  ("time_series_id", "power_fcst_init_time", "valid_time")`. The class docstring describes a
  quantile forecast. The PLANNED normalisation comment moves to the class docstring unchanged.
- **`packages/contracts/README.md`**: the `PowerForecast` bullet.
- **`packages/ml_core/src/ml_core/quantiles.py`** (new): `members_to_quantiles`.
- **`packages/ml_core/src/ml_core/metrics.py`**: `ROW_KEY_COLUMNS` and `COMPARABLE_KEY_COLUMNS`
  become the three-column key (the member-count argument in their docstring goes); `compute_metrics`
  drops the member collapse; `_fair_crps_expr` and `_corrected_variance_expr` are replaced by a
  quantile CRPS and a quantile spread; the empirical delivery-quantile aggregation is replaced by
  the interpolation above.
- **`packages/ml_core/src/ml_core/base_forecaster.py`**: the `predict` docstring.
- **`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`**: `predict` reduces with
  `members_to_quantiles`.
- **`packages/baseline_forecasters/src/baseline_forecasters/climatology.py`**: compute the 11
  levels per cell; drop the unpivot to members. **`manual_heuristic.py`**: reduce the lag members.
  **`README.md`**: the member description.
- **`packages/delta_store/src/delta_store/power_forecasts.py`**: round every quantile column;
  sort by the new key; docstrings. **`contracts/settings.py`**: the new default path segment of
  `power_forecasts_data_path`.
- **`packages/dashboard/src/dashboard/forecast_chart.py`**: `build_view_forecast_chart` draws the
  q05–q95 and q25–q75 bands and the q50 line instead of one grey line per member.
  `build_nwp_ensemble_chart` is unchanged: it reads `Nwp` rows and keeps one line per weather
  member.
- **`src/nged_substation_forecast/defs/live_forecast_assets.py`**, **`cv_assets.py`**,
  **`checks.py`**: drop the `ensemble_member` filters and counts on forecast rows; the CV
  `n_ensemble_members` metadata becomes the count of members in the NWP rows the fold read; the
  live check's non-finite test covers all 11 columns.
- **`scripts/forecasting/score_study.py`**: validate study submissions against the new schema; a
  submission in the member layout is rejected with a message naming `members_to_quantiles`.
  **`export_baseline_forecasts.py`**: the full-ensemble file goes; the mean file becomes the q50
  file; the quantile file reads the stored columns.
- **`scripts/forecasting/convert_power_forecasts_to_quantiles.py`** (new): the one-off conversion.

### Design-philosophy check

The serving path degrades as before: a missing or thinned ENS run still yields a forecast from the
members present, and the live check reports the thinning as a warning. Validation stays fail-fast in
R&D and at the write. The change follows the principle that the schema describes what the column
means: a column of quantiles is no longer labelled a member.

## Tests

Each assertion below fails on `main` today, because the columns, the function, or the behaviour do
not exist there.

- **`contracts/tests/test_power_forecast.py`**: a frame with `power_fcst_q10 > power_fcst_q25` fails
  `validate`; a frame with two rows on the same three-column key fails; the 11 quantile dtypes are
  Float32; `ensemble_member` is not a field.
- **`ml_core/tests/test_quantiles.py`** (new): 101 members evenly spaced 0 to 100 give every stored
  level `q` the value `100 q` exactly; one member gives 11 equal columns; a null member value is
  dropped before the reduction; a group of all-null members yields no row.
- **`ml_core/tests/test_metrics.py`**: CRPS of a row equals twice its mean pinball loss over the 11
  levels; MAE uses q50, not the mean of the quantiles (a skewed row distinguishes the two); pinball
  at 0.02 equals pinball at the linearly interpolated value between q01 and q05.
- **`xgboost_forecaster/tests/test_forecaster.py`**, **`baseline_forecasters/tests/*`**: `predict`
  returns one row per key, and climatology's q50 equals the cell's median.
- **`delta_store/tests/test_power_forecasts.py`**: every quantile column is rounded to the stated
  significand bits on disk; the sort key is the new `PRIMARY_KEY`.
- **`dashboard/tests/test_forecast_chart.py`**: the power chart has band layers and no
  `ensemble_member` detail channel; the NWP chart still has one line per member.
- **`tests/test_live_forecasts.py`**, **`tests/test_checks.py`**: the live check warns on a thinned
  ENS run with forecast rows that carry no member column.
- **The conversion script**: converting a two-partition table twice leaves one copy of each
  partition.

## Docs to update

`docs/roadmap/delivery-tables.md` (Table 1: the internal table holds the 11 stored quantiles;
Representation 1 no longer describes it), `docs/techniques/evaluation-metrics.md` (CRPS from
quantiles, deterministic metrics on the median, interpolated delivery levels, the spread-skill
ratio's Gaussian-equivalent spread), `docs/techniques/probabilistic-forecasting.md` (the closing
section), `docs/roadmap/metrics-and-leaderboard.md`, `docs/architecture/ml-orchestration.md`,
`docs/cross-validation/dagster-workflow.md`, and `docs/live_service/operations.md` where they
describe members on forecast rows. Each describes the code as it now is.

## Verification commands

The `implement-issue` green-before-push set (`uv run ruff check .`, `uv run ruff format --check .`,
`uv run ty check`, `uv run pytest`, the markdown lint), plus `uv run mkdocs build --strict`, and the
conversion script run on a copy of one CV partition with the row count and a spot value compared by
hand.

## Risks and open questions

1. **11 or 13 levels?** `DELIVERY_QUANTILES` holds the 13 levels agreed with NGED, and the research
   networks can only fill the 11 they store. Recommendation: store 11 now, interpolate the 6 others
   in the metrics, and leave the delivered table's level set to the later delivery PR and to NGED's
   answer.
2. **Column naming.** `power_fcst_q01` … `power_fcst_q99` (recommended: the existing column name
   plus the level, sortable, and the same `qXX` form as the research bundles), against `p1` … `p99`
   as `delivery-tables.md` writes the delivered table.
3. **Convert or re-run the stored history?** Recommendation: convert, because the reduction is
   deterministic and the conversion is far cheaper than re-running every CV fold. The metric values
   of converted runs change, because CRPS is now computed from 11 quantiles rather than from the
   members' fair CRPS, so the leaderboard needs one rescore after the conversion.
4. **PR #1707 ordering.** PR #1707 plans to store the step-5 network's quantiles as 51 members. If
   this change merges first, that plan writes the 11 research quantiles straight into the new
   columns and drops its 51-level bisection. Recommendation: merge this change first, or have
   PR #1707 write its quantiles through `members_to_quantiles` until it does.
5. **Spread-skill ratio.** The Gaussian-equivalent spread from q10 and q90 keeps the metric's name
   and meaning for a near-Gaussian forecast. The alternative is to retire the metric, which changes
   the `Metrics` contract. Recommendation: keep it, with the definition stated in
   `evaluation-metrics.md`.
6. **Research studies.** `nged-ml-research` pins `contracts`. Its studies submit `PowerForecast`
   files through `score_study.py`, so they move to the new layout when that repository bumps its
   pin, in a pull request of its own.
