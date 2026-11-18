# Plan: store `PowerForecast` as 11 quantile columns (issue #1708)

**The problem.** `PowerForecast` stores one row per series, forecast run, valid time, and
`ensemble_member`, with one deterministic value in `power_fcst`. The schema's docstring says the
spread across members expresses the uncertainty. The research neural networks for the Primaries (the
step-5 network of research experiment 705 and the Candidate of research experiment 708) output one
forecast of 11 quantiles per series and half-hour, not one forecast per weather member. Today such a
forecast can only be stored by encoding quantile level k as `ensemble_member` k, as
`ClimatologyForecaster` does. The plan of PR #1707, which serves the step-5 network, depends on this
change and writes the 11 quantile columns it adds. The member encoding makes every reader treat
quantiles as exchangeable members: the fair CRPS treats the 51 values as random draws, the
spread-skill ratio takes their variance, and the dashboard draws each level as a grey "member" line.

**The planned solution.** Replace `ensemble_member` and `power_fcst` with 11 Float32 columns,
`power_fcst_p1` to `power_fcst_p99` (named with `contracts.common.quantile_label`), at the levels
the research forecasts store (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99). The
primary key becomes `(time_series_id, power_fcst_init_time, valid_time)`, and a row constraint
forbids a quantile below the level beneath it. The three forecasters that compute members today
reduce their members to the 11 quantiles with one shared function in `ml_core` before they return.
`ml_core.metrics` scores the quantile columns directly, at the 11 stored levels. The stored
`power_forecasts` Delta table moves to a new path; the old table stays read-only, and a CV fold is
re-run wherever a comparison with old results is wanted. The dashboard draws the power forecast as
quantile bands, and its NWP panel keeps one line per ECMWF ENS weather member, read from `Nwp` as
now.

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
  and is not recommended: every reader would pivot it back); 11 or 13 levels; converting or
  re-running the stored forecasts.
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
levels, and returns one row per group with the 11 `power_fcst_pXX` columns. A group with one member
returns 11 equal columns, which is the honest description of a deterministic forecast. `predict`
returns `PowerForecast.validate(...)` on that frame, and the monotone constraint passes by
construction because empirical quantiles never decrease. `live_forecasts` keeps its
`ensemble_member.is_not_null()` filter on the feature rows before `predict`: rows the NWP join
missed carry a null `ensemble_member` (`tabular_feature_engineer.py`), XGBoost predicts finite
values for them, and after the reduction nothing would mark them. It then calls
`write_power_forecasts`, which rounds all 11 columns to `POWER_FCST_SIGNIFICAND_BITS`, writes each
with the `BYTE_STREAM_SPLIT` encoding that `power_fcst` has today, sorts by the new three-column
`PRIMARY_KEY`, and replaces the slot's rows under the existing predicate.

**The happy path, CV and metrics.** `cv_power_forecasts` writes the same reduced frames through
the same writer. `ml_core.metrics.compute_metrics` joins the actuals onto the quantile rows. It no
longer collapses members: each row already is one forecast. Per row it computes:

- CRPS as twice the unweighted mean pinball loss over the 11 stored levels. This is a choice:
  research experiment 704 uses the same estimator, but `studies.cross_validation.crps` and the
  probability-mass weighting of research experiments 5 to 20 do not;
- the deterministic metrics (MAE, RMSE, MBE, NMAE) on `power_fcst_p50`, as issue #1077 proposes;
- pinball loss at each of the 11 stored levels, and PICP and interval width for the five symmetric
  stored bands (`p1_p99`, `p5_p95`, `p10_p90`, `p25_p75`, `p40_p60`), read straight from the stored
  columns;
- the spread for the spread-skill ratio as the Gaussian-equivalent standard deviation
  `(p90 − p10) / (2 × z₀.₉) = (p90 − p10) / 2.5631`, with the constant computed from
  `statistics.NormalDist().inv_cdf(0.9)`, not typed in.

The `Metrics` contract keeps every metric name. Its `metric_param` is a `pl.Enum` built from
`DELIVERY_QUANTILES`, and `metric_value` is not nullable, so the 6 unstored delivery levels cannot
be emitted as nulls. `QUANTILE_METRIC_PARAMS` and `BAND_METRIC_PARAMS` therefore become the union of
the delivery labels and the stored labels, which adds `p25`, `p40`, `p60`, `p75`, `p25_p75`, and
`p40_p60`. Stored metric rows that carry the old labels still validate, and new runs emit only the
stored labels. `mean_pinball_loss` becomes the mean over the 11 stored levels.

**Five metrics change meaning under unchanged names:** `crps` (fair CRPS of members → quantile
CRPS), `mean_pinball_loss` (13 levels → 11), and `mae`, `rmse`, and `mbe` (ensemble mean → p50).
The names reach the `forecast_metrics` table and the MLflow keys, so old and new values would mix
on the leaderboard with no marker. The plan writes the metrics of new runs to a new
`forecast_metrics` path beside the new forecasts path, so no table holds both definitions, and
tags every new MLflow run `metrics_version=2`. The `METRIC_NAMES` docstring in `ml_schemas.py`
(fair CRPS, Fortin spread, ensemble mean, the PICP reference of about 0.769 at 51 members) is
rewritten for the quantile definitions. Open question 7 asks the maintainer to choose between this
and renaming the metrics.

**Branches off the happy path.**

- **Climatology** computes its 11 levels directly from the training power of each calendar cell
  (`climatology.py`), rather than computing 51 levels and reducing them, because the cell's
  empirical quantile at the exact level is what the 51 levels approximated.
- **Manual heuristic** reduces however many lag members survived `_nullify_leaky_lags` (one per lag
  column in `_lag_column_to_member` at most). With one surviving lag the 11 columns are equal, as
  for a one-member XGBoost forecast.
- **A fresh table.** `write_power_forecasts` writes to the path in
  `Settings.power_forecasts_data_path`, whose default path segment changes from `power_forecasts`
  to `power_forecasts_q11`. The first write creates the table with the new schema, as today. The
  old table is left in place, read-only, until the maintainer deletes it. No conversion script is
  written: converted rows would not be comparable with new runs anyway, because CRPS changes from
  the members' fair CRPS to the quantile CRPS, so a CV fold is re-run wherever a comparison is
  wanted. Rewriting the old path in place with delta-rs `schema_mode="overwrite"` was rejected,
  because an overwrite with a new schema replaces every partition of the table, so the old results
  would be lost rather than kept readable.

**Error-handling paths.**

- **A member value that is null** (a join miss): the live filter above drops those feature rows
  before `predict`, and `members_to_quantiles` also drops null values, so a group keeps its other
  members. A group with no member left yields no row, which `live_forecasts_are_healthy` already
  reports as missing coverage.
- **A NaN quantile.** Measured on patito 0.8.6 and polars 2.0.0, a cross-column `b >= a` constraint
  accepts (NaN, NaN) and (1, NaN) and rejects only (NaN, 1), and polars' quantile sorts NaN last, so
  the ordering constraint alone does not catch a NaN. `PowerForecast` therefore gives every quantile
  field an explicit `is_finite()` constraint, so `validate` raises on a NaN or infinite quantile,
  as PR #1707's plan expects. The live check's non-finite count covers all 11 columns.
- **A thinned NWP feed.** `live_forecasts_are_healthy` does not warn about a thinned feed today; it
  only names the member count in its message. `nwp_run_is_complete` already reports missing
  members. The plan drops the member count from the check's message and from `live_forecasts`'s
  metadata, and adds no new report: counting members of the lazy `Nwp` frame would be a second
  collect, outside the asset's catch-all.
- **A crossed quantile** cannot come from `members_to_quantiles`. A future forecaster that emits
  quantiles directly (PR #1707's network) must sort them per row before validation; the constraint
  then makes a crossing fail `validate` loudly rather than reach the table.

### What changes, file by file

- **`packages/contracts/src/contracts/common.py`**: add `STORED_QUANTILES` (the 11 levels). Column
  names are `f"power_fcst_{quantile_label(q)}"`, reusing the existing `quantile_label`, so no new
  helper. `DELIVERY_QUANTILES` stays as the 13 levels agreed with NGED for the delivery tables.
- **`packages/contracts/src/contracts/ml_schemas.py`**: `QUANTILE_METRIC_PARAMS` and
  `BAND_METRIC_PARAMS` become the union of the delivery and stored labels; the `mean_pinball_loss`
  description names the 11 stored levels.
- **`packages/contracts/src/contracts/power_schemas.py`**: `PowerForecast` loses `ensemble_member`
  and `power_fcst`, gains the 11 Float32 quantile fields (each carrying `power_fcst`'s current
  description of available power, units, and sign, written once in a shared string), per-field
  constraints (patito has no model-level constraint) that each column is finite and at least the
  column below it, and `PRIMARY_KEY =
  ("time_series_id", "power_fcst_init_time", "valid_time")`. The class docstring describes a
  quantile forecast. The PLANNED normalisation comment moves to the class docstring unchanged.
- **`packages/contracts/README.md`**: the `PowerForecast` bullet.
- **`packages/ml_core/src/ml_core/quantiles.py`** (new): `members_to_quantiles`.
- **`packages/ml_core/src/ml_core/metrics.py`**: `ROW_KEY_COLUMNS` and `COMPARABLE_KEY_COLUMNS`
  become the three-column key (the member-count argument in their docstring goes); `compute_metrics`
  drops the member collapse; `_fair_crps_expr` and `_corrected_variance_expr` are replaced by a
  quantile CRPS and a quantile spread; the empirical delivery-quantile aggregation and
  `_quantile_column` go, because the metrics read the stored columns; the pinball, PICP, and
  interval-width loops run over `STORED_QUANTILES`, and `_BAND_LOWER_QUANTILES` is derived from it.
- **`packages/ml_core/src/ml_core/base_forecaster.py`**: the `predict` docstring.
- **`packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`**: `predict` reduces with
  `members_to_quantiles`.
- **`packages/baseline_forecasters/src/baseline_forecasters/climatology.py`**: compute the 11 levels
  per cell; drop the unpivot to members. The saved lookup artefact stores 51 quantile columns per
  cell (`CLIMATOLOGY_MEMBER_COUNT`), so saved climatology models are retrained.
  **`manual_heuristic.py`**: reduce the lag members. **`nwp_run_rows.py`**: the docstring that names
  `PowerForecast`. **`README.md`**: the member description.
- **`packages/delta_store/src/delta_store/power_forecasts.py`**: round every quantile column; give
  each the `BYTE_STREAM_SPLIT` column properties that `POWER_FORECASTS_WRITER_PROPERTIES` gives
  `power_fcst` today; sort by the new key; docstrings. **`contracts/settings.py`**: the new default
  path segment of `power_forecasts_data_path`.
- **`packages/dashboard/src/dashboard/forecast_chart.py`**: `build_view_forecast_chart` draws the
  p5–p95 and p25–p75 bands and the p50 line instead of one grey line per member, and its orange
  comparison overlay draws the second forecast's p5–p95 band and p50 line.
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
  non-finite count covers all 11 columns.
- **`scripts/forecasting/score_study.py`**: no bespoke code. Its per-column dtype check and
  `PowerForecast.validate` already reject a submission in the member layout.
  **`export_baseline_forecasts.py`**: the smallest change that runs. The full-ensemble file goes;
  the ensemble-mean file is replaced by a p50 file; the quantile file reads the stored p10, p50,
  and p90 columns.

### Design-philosophy check

The serving path degrades as before: a missing or thinned ENS run still yields a forecast from the
members present, and `nwp_run_is_complete` reports the missing members as now. Validation stays
fail-fast in R&D and at the write. The change follows the principle that the schema describes what
the column means: a column of quantiles is no longer labelled a member.

## Tests

Each assertion below fails on `main` today, because the columns, the function, or the behaviour do
not exist there, except the two marked as regression guards.

- **`contracts/tests/test_power_forecast.py`**: one test that a frame with `power_fcst_p10 >
  power_fcst_p25` fails `validate`; a frame with two rows on the same three-column key fails; a NaN
  quantile fails `validate`.
- **`contracts/tests/test_ml_schemas.py`**: a `Metrics` row with the old label `p2` and a row with
  the new label `p25_p75` both validate under the union Enum.
- **`ml_core/tests/test_quantiles.py`** (new): 101 members evenly spaced 0 to 100 give every stored
  level `q` the value `100 q` exactly; one member gives 11 equal columns; a null member value is
  dropped before the reduction; a group of all-null members yields no row.
- **`ml_core/tests/test_metrics.py`**: CRPS of a row equals twice its mean pinball loss over the 11
  levels; MAE uses p50, not the mean of the quantiles (a skewed row distinguishes the two); the
  metrics frame carries `pinball_loss` rows for `p25` and none for `p2`; PICP and interval width
  are emitted for `p25_p75` and `p40_p60` with hand-computed values; the spread equals
  `(p90 − p10) / (2 × NormalDist().inv_cdf(0.9))` on a hand-built row.
- **`xgboost_forecaster/tests/test_forecaster.py`**, **`baseline_forecasters/tests/*`**: `predict`
  returns one row per key, and climatology's p50 equals the cell's median.
- **`delta_store/tests/test_power_forecasts.py`**: every quantile column is rounded to the stated
  significand bits on disk; `POWER_FORECASTS_WRITER_PROPERTIES` names all 11 quantile columns with
  `BYTE_STREAM_SPLIT`. Regression guard: the sort key is `PRIMARY_KEY` (true on `main` too).
- **`dashboard/tests/test_forecast_chart.py`**, **`test_view_forecasts.py`**: the power chart and
  its comparison overlay have band layers and no `ensemble_member` detail channel. Regression
  guard: the NWP chart still has one line per member (true on `main` too).
- **`tests/test_live_forecasts.py`**: extending the predict spy, feature rows with a null
  `ensemble_member` never reach `predict`, so the stored rows hold no join-miss forecast.
- **`tests/test_checks.py`**: the live check's non-finite count counts a row whose only non-finite
  value is in `power_fcst_p99`.

## Docs to update

Only the pages that `git grep -n "ensemble_member\|power_fcst\b"` finds describing members on
forecast rows, the `PowerForecast` contract, or the metric formulas; a hit about the members of an
NWP run stays. The expected pages are `docs/roadmap/delivery-tables.md` (Table 1), the pages that
name the `power_forecasts` path (`docs/architecture/aws.md`, `backup.md`, `setup.md`,
`production-deployment.md`, `ml-orchestration.md`, and `docs/live_service/operations.md`),
`docs/techniques/evaluation-metrics.md` (CRPS from quantiles, deterministic metrics on the median,
the stored levels, the spread-skill ratio's Gaussian-equivalent spread), and the
`n_ensemble_members` tag sentence in `docs/cross-validation/dagster-workflow.md`. Each describes the
code as it now is.

## Verification commands

The `implement-issue` green-before-push set (`uv run ruff check .`, `uv run ruff format --check .`,
`uv run ty check`, `uv run pytest`, the markdown lint), plus `uv run mkdocs build --strict`.

## Risks and open questions

1. **11 or 13 levels?** `DELIVERY_QUANTILES` holds the 13 levels agreed with NGED, and the research
   networks can only fill the 11 they store. Recommendation: store and score the 11 now, and leave
   the delivered table's level set to the later delivery PR and to NGED's answer.
2. **Column naming.** `power_fcst_p1` … `power_fcst_p99`, from the existing `quantile_label` and
   matching the `p1` … `p99` names of `delivery-tables.md`, against `power_fcst_q01` …
   `power_fcst_q99`, the `qXX` form of the research bundles. Recommendation: the `p` form.
3. **History.** The old table stays read-only, and no conversion runs. A comparison with an old CV
   run means re-running that fold. Recommendation: accept that, because converted rows would score
   differently anyway.
4. **The NGED-facing path moves.** `power_forecasts_data_path` derives from `data_path_delivery`, so
   the table sits under the delivery bucket that NGED reads. A new path segment moves the table NGED
   reads, `score_study.py` refuses every study until its `reference_experiment_name` is re-run into
   the new table, and the dashboard shows no history until runs land there. The alternatives are to
   keep the path and overwrite the whole table (losing the history), or to move only the internal
   CV partitions. Recommendation: the new path, with the move announced to NGED, because NGED's
   readers break on the new columns whichever path is used.
5. **Spread-skill ratio.** Kept, with the Gaussian-equivalent spread `(p90 − p10) / 2.5631` stated
   in `evaluation-metrics.md`.
6. **Research studies.** `nged-ml-research` pins `contracts`. Its studies submit `PowerForecast`
   files through `score_study.py`, so they move to the new layout when that repository bumps its
   pin, in a pull request of its own.
7. **Metrics that change meaning under the same names.** A new `forecast_metrics` path plus a
   `metrics_version=2` MLflow tag, or new metric names (`crps_quantile`, `mae_median`, and so on).
   Recommendation: the new path and tag, because renaming changes the `Metrics` contract's name
   Enum and every leaderboard query.
8. **Issue #262.** Its Representation 2 is what this change delivers, and its Representation 3 stays
   out of scope. Recommendation: re-scope issue #262 to Representation 3 when this merges; the
   maintainer decides, because this plan does not edit issue #262.

## Simplicity review: triage

- **Cut the conversion script and its test:** accepted. Checked delta-rs `schema_mode="overwrite"`
  as the alternative to a new path, and rejected it, because a schema overwrite replaces every
  partition and would lose the old results.
- **Score only at the 11 stored levels:** accepted, with one change. `Metrics.metric_param` is a
  `pl.Enum` of the delivery labels and `metric_value` is not nullable, so null rows for the 6
  unstored levels are not possible; the Enum becomes the union of the delivery and stored labels.
- **Reuse `quantile_label` and drop `_quantile_column`:** accepted. Columns become
  `power_fcst_p1` … `power_fcst_p99`.
- **No bespoke rejection message in `score_study.py`:** accepted.
- **`n_ensemble_members`:** accepted. No code reads the CV metadata or tag, so both go. The live
  member count comes from the `Nwp` frame `live_forecasts` already loads, with no second read.
- **Minimal `export_baseline_forecasts.py`, docs from grep hits:** accepted.
- **Smaller points:** accepted (climatology computes the stored levels directly; one plain function
  with two callers; one monotone test; spread-skill kept with `(p90 − p10) / 2.5632`; NaN as a
  one-line test; long layout recorded as considered and not recommended).

## Correctness review: triage

- **Keep the live `ensemble_member.is_not_null()` filter:** accepted. Join-miss rows carry a null
  member (`tabular_feature_engineer.py`), and a test on the predict spy is added.
- **No thinned-NWP warning to move:** accepted. The check only names the count, and
  `nwp_run_is_complete` already reports missing members; the new report is dropped.
- **Five metrics change meaning under unchanged names:** accepted. A new `forecast_metrics` path and
  a `metrics_version=2` tag are planned, the `METRIC_NAMES` docstring is rewritten, and the choice
  is open question 7.
- **NaN passes the ordering constraint:** accepted. Every quantile field gets an explicit
  `is_finite()` constraint, tested.
- **Missed callers:** accepted (the marimo app and its test, the comparison overlay, the notebook,
  the writer properties, the climatology artefact retrain, and the `nwp_run_rows.py` docstring).
- **The table path is NGED-facing:** accepted. The docs pages and tests that name the path are
  added, the effect on `score_study.py` and the dashboard is stated, and the choice is open
  question 4.
- **Stale PR #1707 risk:** accepted. PR #1707's plan already writes the 11 quantile columns, so the
  ordering question is removed; issue #262 is open question 8.
- **Smaller points:** accepted (2.5631 computed from `NormalDist`; per-field constraints, because
  patito has no model-level constraint; the CRPS estimator stated as a choice; the replacement for
  the ensemble-mean export file named).
- **Tests that pass on `main`:** accepted. Both are labelled regression guards, and the six missing
  tests are added.
