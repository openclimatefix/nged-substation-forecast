# `production_monitoring` scope, `monitoring_sensor`, `retire_experiment_job`

*Merged from the Dagster ML-assets plan, Phase 8. Depends on plan 02 (`live_forecasts`) —
there is nothing to monitor until live forecasts exist.*

## Context

The `metrics` asset implements the `leaderboard` and `ad_hoc` scopes; `production_monitoring`
is declared in `EVALUATION_SCOPES` but unimplemented (`EvalScopeType` in
`contracts/ml_schemas.py:219` deliberately omits it). And with thousands of experiments
planned, the `cv_experiment_folds` dynamic partition set grows without bound — partition keys
need a retirement path that cannot lose results.

## Part 1 — `production_monitoring` evaluation scope

- Extend `EvalScopeType` to `Literal["leaderboard", "production_monitoring", "ad_hoc"]`,
  bringing it in sync with `EVALUATION_SCOPES` (the docstring at `ml_schemas.py:222` already
  anticipates this).
- Remove the CV-folds-only restriction in `compute_metrics()` (documented in its docstring):
  `fold_id="live"` rows use the same join logic, with window bounds supplied by the caller
  (trailing windows, not fold dates).
- Scope behaviour in the `metrics` asset: score `fold_id="live"` forecasts over two trailing
  `valid_time` windows — **last 24 hours** and **last 7 days**. Each window writes rows to
  `forecast_metrics` Delta with `window_label` (`"24h"`/`"7d"`), the trailing
  `window_start`/`window_end` bounds, and `computed_at = now` (all columns already exist in the
  `Metrics` schema). These rows are **append-only** — successive runs accumulate the
  sliding-window history (unlike the leaderboard scope's idempotent overwrite; recomputations
  are distinguished by `computed_at`).
- MLflow: log the same aggregates to a **dedicated `production_monitoring` MLflow experiment**
  — never to the golden leaderboard — as **time-series points** (MLflow metric
  timestamp/step), one persistent run per window resolved by tag (mirroring the
  `_mlflow_runs` get-or-create convention), so MLflow charts live performance over time (e.g.
  trailing-7d NMAE per `time_series_type`). Stamp `mlflow_run_id` on the Delta rows as the
  cross-link.
- The trailing window-bounds calculation is a pure helper (injected `now`), unit-tested.
- Note: evaluating "the last 24h of production" scores forecasts whose `valid_time` has already
  passed and now has observed power — satisfied naturally as observations accumulate.

## Part 2 — `monitoring_sensor`

A Dagster sensor that fires on each `power_time_series_and_metadata` materialisation (~every
6 h, when new actuals land) and requests a `metrics` run with
`evaluation_scope="production_monitoring"` over `fold_id="live"` for both trailing windows.
Sensor preferred over a schedule so it fires on the actual data update.

Note this sensor needs a running Dagster daemon — plan 04's Option B (the direction we're
leaning) provides one. If plan 04 instead ships Option A (nothing always-on), skip the sensor
and run the monitoring step as the final op of the one-shot production job (plan 04
workstream 2 already reserves that slot).

## Part 3 — `retire_experiment_job`

A **manually triggered** job (deliberate and auditable — never automatic) with a single
config field `experiment_name: str`:

1. **Verify before deleting**: the MLflow parent run exists and carries aggregate metrics,
   **and** `power_forecasts` Delta contains rows for this `experiment_name`. If either check
   fails, raise and delete nothing.
2. Delete the experiment's dynamic partition keys via
   `context.instance.delete_dynamic_partition("cv_experiment_folds", key)` for each
   `f"{experiment_name}__{fold_id}"`.
3. Log the deleted keys as output metadata.

Retirement does **not** delete MLflow runs or Delta forecasts — those remain the permanent
record; it only prunes Dagster's execution ledger. Lives beside `register_experiment_job` in
`defs/jobs.py`; ops use `OpExecutionContext` (they need `context.instance`).

## Interaction with plan 07 (probabilistic metrics)

Any metric added to `compute_metrics` flows through this scope automatically — once plan 07's
PICP/spread-skill land, production monitoring tracks ensemble calibration over time for free.
No coupling needed; just note the ordering is flexible.

## Tests

- Sensor fires on a power update and requests the monitoring run.
- Monitoring rows land in Delta (append-only, correct window bounds from an injected clock) and
  in the `production_monitoring` MLflow experiment — and **never** touch a leaderboard run.
- `retire_experiment_job` refuses when results are absent (each check independently); deletes
  keys when both are present; MLflow + Delta untouched either way.

## Verification

Trigger a power update (or the sensor manually), see trailing-24h/7d metrics appear in the
`production_monitoring` MLflow experiment and `forecast_metrics`; run `retire_experiment_job`
on a throwaway experiment and watch its partitions disappear from the Dagster UI while its
MLflow runs and Delta rows remain.
