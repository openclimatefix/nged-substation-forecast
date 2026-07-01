# Probabilistic evaluation: horizon slices, PICP, spread-skill, CRPS — then calibration

## Finding

The 51-member ensemble is very likely **underdispersed**: XGBoost trained with squared error on
the control member (`cv_assets.py:373`) learns a conditional mean, so pushing 51 members through
it yields spread from *weather uncertainty only* — no model or observation uncertainty. Such
ensembles are systematically overconfident, worst at short horizons where members haven't
diverged. Flexibility procurement is a tails problem (P90+ peaks), so this hits the use case
directly — yet nothing measures calibration today: `compute_metrics` averages members into a
deterministic mean (`packages/ml_core/src/ml_core/metrics.py:84-90`) and scores only MAE/NMAE/
RMSE/MBE on the `"all"` horizon slice (`metrics.py:51`). We pay 51× inference cost and score
only the mean.

Fix in three phases, each an independent PR. Phases A and B are pure evaluation (no model
changes) and should land before any further MAE-driven experimentation.

## Phase A — horizon-sliced metrics

`HORIZON_SLICES` already exists in `contracts/ml_schemas.py:172` and the docs
(`docs/roadmap/metrics-and-leaderboard.md`) argue skill drivers differ radically by horizon;
only `"all"` is computed.

- In `compute_metrics` (`metrics.py`), derive `lead_time = valid_time − power_fcst_init_time`
  per row (confirm `power_fcst_init_time` survives into the forecast/actuals join; it is a
  `PowerForecast` primary-key column) and map it onto the `HORIZON_SLICES` bands with
  `pl.when/then` chains or `cut`.
- Compute the existing four metrics per `(series, fold, model, horizon_slice)` via one
  `group_by` including the slice column, plus the existing `"all"` aggregate.
- `build_mlflow_aggregate_metrics` gains keys like `nmae__all__day_ahead`. Keep the existing
  key format for `"all"` slices unchanged so historical MLflow runs stay comparable.
- Schema needs no change (the `Metrics` tall format was designed for this).

## Phase B — probabilistic metrics from the existing ensemble

Compute member-aware metrics *before* the ensemble-mean collapse in `compute_metrics`:

- **Spread-skill ratio**: mean ensemble stddev ÷ RMSE of the ensemble mean, per group. Ratio
  ≪ 1 confirms underdispersion — this is the headline diagnostic for the finding above.
- **PICP**: empirical member quantiles per `(series, valid_time)` (`pl.quantile` over members),
  then coverage of the P10–P90 interval. `metric_param` (currently only `"all"`,
  `ml_schemas.py:203`) carries the interval label, e.g. `"p10_p90"`.
- **Pinball loss** at P10/P50/P90 from the same empirical quantiles; `metric_param` = quantile
  label. (Schema docstrings already anticipate pinball/PICP.)
- **CRPS** (fair/ensemble form): per timestamp, `mean|xᵢ − y| − ½·mean|xᵢ − xⱼ|`. The pairwise
  term over 51 members is computable with a self-join on member index within
  `(series, valid_time)` groups; ~51² × rows is fine at V1 scale. If it's slow, the
  sorted-member O(m log m) form is a known optimisation — don't start there.
- Extend `METRIC_NAMES` (`ml_schemas.py:188`) and `METRIC_PARAMS` accordingly. Both are
  `pl.Enum` columns written to Delta as String (documented gotcha) — additive enum growth is
  safe for existing data.
- Update `docs/roadmap/metrics-and-leaderboard.md` statuses from 🚧 to ✅ as they land.

## Phase C — cheap calibration (after B proves the diagnosis)

Decide based on Phase B's spread-skill numbers; recommended order:

1. **Post-hoc spread inflation** (EMOS-lite): per horizon slice, fit a scalar `s` on the
   *training* window so that inflating members around the ensemble mean
   (`mean + s·(member − mean)`) makes spread match error. Zero schema change, zero new model —
   implementable as an optional step in `predict` or as a wrapper forecaster. Fit on train,
   apply on validation (no tuning on the fold being scored).
2. **XGBoost native multi-quantile** (`objective: reg:quantileerror`, several
   `quantile_alpha`s) as a separate experiment/model family. This gives directly calibrated
   quantiles but requires a percentile representation in `PowerForecast` (the roadmap already
   plans percentiles) — a bigger schema/design step, so keep it as its own follow-up plan when
   Phase B/C1 results justify it.

## Verification

- Unit tests in `packages/ml_core/tests/` with hand-computable fixtures: a 3-member toy
  ensemble where PICP, spread-skill, pinball, and CRPS are worked out by hand.
- Property check: CRPS of a single-member "ensemble" equals MAE.
- Re-score an existing experiment via the `metrics` asset (`ad_hoc` scope) and eyeball that
  spread-skill ≪ 1 at short horizons — the expected signature of the finding.
