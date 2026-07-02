# Metrics & leaderboard

How OCF measures the skill of its forecasts and compares forecasting approaches.

> **Status legend** — ✅ Implemented · 🚧 Planned · 🔬 Research. The `Metrics` schema, the
> `metrics` Dagster asset, and the deterministic metrics (MAE, NMAE, RMSE, MBE) are ✅
> implemented. The interactive leaderboard visualisation and probabilistic metrics are 🚧 planned.
> The implemented [cross-validation protocol](../ml_experimentation/cross-validation-folds.md) has
> moved out of the roadmap. See the [roadmap index](index.md) for status conventions.

---

## The leaderboard concept 🚧

A key deliverable is a **leaderboard** comparing many forecasting approaches. We plan **one
leaderboard per forecasting task**, e.g. primary substations, GSPs, BSPs, solar PV sites, wind
farms, BESS, etc.

Each leaderboard has tens (maybe hundreds) of rows. Each row is one **ML experiment**: a particular
model, trained with a particular set of features, processed a particular way. Entrants must be
compared apples-to-apples — same test dataset, same metrics, same assumptions.

Per-experiment configuration, trained weights, and metrics are stored in the project's **MLflow**
database. The leaderboard will be displayed as an interactive table (inspiration: the
"Weird ML Leaderboard") showing multiple metrics at a glance.

---

## Cross-fold validation

The cross-validation protocol is **implemented**, so it has moved to its permanent home:
[ML Experimentation → Cross-validation folds](../ml_experimentation/cross-validation-folds.md).
That page covers the expanding-window protocol, the current single MVP fold (and why the available
weather data constrains us to it), the target multiple-yearly-fold protocol, and the fold-design
alternatives we considered.

---

## Evaluation metrics

| Metric | Type | Status | Purpose |
|---|---|---|---|
| Mean absolute error (MAE) | Deterministic | ✅ | Typical error magnitude (MW). |
| Normalised MAE (NMAE) | Deterministic | ✅ | MAE normalised by the series' [effective capacity](#normalising-nmae-by-effective_capacity) (full-history P99) — comparable across substations of different sizes. |
| Root mean squared error (RMSE) | Deterministic | ✅ | Heavily penalises large misses (one 100 MW error costs more than two 50 MW errors). |
| Mean bias error (MBE) | Deterministic | ✅ | Systematic over/under-prediction. |
| Histogram of errors | Deterministic | 🚧 | Visual check that errors are ~Normal. |
| Pinball loss (quantile loss) | Quantile | 🚧 | Penalises asymmetrically by target quantile. Averaged across quantiles for a single quantile-skill score. |
| PICP (Prediction Interval Coverage Probability) | Quantile | 🚧 | Of a P10–P90 band, exactly 80% of observations should fall inside. < 80% ⇒ overconfident. |
| CRPS (Continuous Ranked Probability Score) | Ensemble | 🚧 | Probabilistic equivalent of MAE; rewards both accuracy and sharpness. |
| Spread-Skill Ratio | Ensemble | 🚧 | Ensemble spread ÷ RMSE of the ensemble mean. 1.0 = well-calibrated; < 1 under-dispersed (overconfident); > 1 over-dispersed (underconfident). |

> The `Metrics` schema (`contracts.ml_schemas.Metrics`) stores results as
> `(time_series_id, power_fcst_model_name, fold_id, horizon_slice, metric_name, metric_param,
> metric_value)`. `metric_param` carries, e.g., the quantile for Pinball Loss (`p10`) or the band
> for PICP (`p10_p90`). The `metrics` Dagster asset computes the deterministic metrics ✅ and writes
> per-series rows to `forecast_metrics` Delta (partitioned by `experiment_name, fold_id`), with
> per-fold and mean-across-folds aggregates logged to MLflow — see
> [Running an ML experiment end-to-end](../ml_experimentation/dagster-workflow.md#step-8-materialise-metrics).

### Normalising NMAE by `effective_capacity`

NMAE is MAE divided by a per-series **effective capacity**, not by the mean or a per-fold P99. A
capacity-like denominator is what makes NMAE comparable across asset types: intermittent generators
(PV, wind) spend much of their time near zero output, so normalising by the *mean* would inflate
their NMAE relative to a demand substation of similar peak size. Computing the denominator over each
series' **full history** (rather than within the validation window) also keeps it stable across
folds — an unusually calm year for a wind farm would otherwise give a low in-window P99 and an
inflated NMAE.

The denominator comes from the [`effective_capacity`](delivery-tables.md#table-4-effective_capacity)
Delta table (schema `contracts.power_schemas.EffectiveCapacity`), consumed by `compute_metrics`
(`ml_core.metrics`).

**MVP representation (v0.1): one scalar row per series.** The `effective_capacity` asset writes one
row per `time_series_id` — `effective_capacity_mw` = P99 of `|power|` over the whole observation
history, `time` = the latest observed timestep. `compute_metrics` joins it onto the per-series
metrics **on `time_series_id` alone** and divides.

**Why the MVP is a single row per series, not the value repeated at every half-hour.** The
v0.6 / v0.7 upgrade below *will* store one row per `(time_series_id, time)` half-hour — but with a
genuinely *time-varying* value. In the MVP the value is a single constant per series, so repeating it
across every half-hour would just be a denormalised encoding of one number: at V2 scale (~2,500
series × ~4 years × 17,520 half-hours/yr ≈ 175M rows) that is hundreds of millions of rows to
express ~2,500 scalars, for zero extra information. It would also *not* buy forward-compatibility,
because the real MVP→DP interface change is not the data shape but **the join** (below). The
`EffectiveCapacity` schema — `(time_series_id, time, effective_capacity_mw)` — already accommodates
both the one-row-per-series MVP and the one-row-per-half-hour DP shape with no schema change; that is
the forward-compatibility we want.

**DP upgrade (v0.6 / v0.7): time-varying, and the join changes.** The
[differentiable-physics](differentiable-physics.md) capacity model produces a value that changes over
time (panel degradation, inverter trips, seasonal derating). At that point two things change, and
nothing else:

- the `effective_capacity` asset body emits one row per `(time_series_id, time)`; and
- `compute_metrics` changes its capacity join from `time_series_id`-only to a **temporal as-of join**
  on `(time_series_id, valid_time)` — matching each forecast's `valid_time` to the capacity in effect
  at that time.

The `Metrics` schema and the rest of the metrics pipeline are untouched. Note the table is
**backward-looking only** (it holds no future `valid_time`s): fine for historical CV folds (whose
validation windows lie inside the observed history), but live-forecast scoring (Phase 7 / 8) must
choose which reference time's capacity to apply rather than expecting a row at a future `valid_time`.

One related distinction to keep straight: the *metric denominator* may use the full-history
**smoothed** DP capacity estimate, but any capacity used to normalise model inputs at forecast init
time (the two-pass training scheme) must be the **causal** estimate available at that init time, or
backtests gain lookahead — see
[Differentiable Physics §3](differentiable-physics.md#3-how-dp-fits-into-the-roadmap).

### Peak events — the metric filter that matters most for flexibility

Because NGED's goal is **flexibility procurement** (entirely about peak management and congestion),
overall RMSE only tells half the story. We add a **"Peak Events"** filter:

- **Peak RMSE / Peak Pinball Loss**: score models *only* on the top 5% highest-demand half-hours
  (or hours where solar generation unexpectedly drops during peak demand).
- **Hand-picked "hard examples"**: if NGED supplies a list of historically tricky times, we compute
  performance on those alone.

---

## Time-slices for performance evaluation

We compute every metric separately per horizon slice, because the driver of model skill changes
with lead time:

| Horizon slice | Industry term | Primary driver of model skill |
|---|---|---|
| 0–6 hours | Intraday / Nowcasting | Lagged power & persistence. NWP is often too coarse to beat simple autoregressive features here. |
| 6–36 hours | Day-Ahead | Deterministic NWP. Covers the critical day-ahead market gate; relies on the diurnal cycle + high-res weather. |
| Day 2–7 | Short/Medium Range | Synoptic weather. Skill driven by mapping large weather fronts to power; ensemble spread starts to matter. |
| Day 8–14 | Extended Range | Ensemble probabilities. Deterministic weather is essentially noise; skill comes from processing ensemble uncertainty. |

### Measuring performance during switching events 🚧

We will flag each timestep for whether it contains a switching event, and compute metrics separately
for periods with switching events in the model inputs (or in the forecast's `valid_time`). This
distinguishes models that perform well *only* on clean periods from models that handle switching
events in their inputs. The flags come from the detector described in
[Switching events & latent demand](switching-events.md).

---

## Grouping the results

Each ML experiment is tagged with metadata so we can group experiments and compute average
performance per group (e.g. "does lagged power *always* help, regardless of model sophistication?",
or "how robust is each model to weather-forecast uncertainty — CERRA reanalysis vs. operational
NWP?"). Example tags:

| Tag | Example values |
|---|---|
| `time_series_type` | PV, Wind, disaggregated demand (primaries) |
| `model_family` | baseline_persistence, xgboost, pytorch_mlp, pytorch_gnn |
| `weather_source` | none, ecmwf_control, full_ecmwf_ensemble, cerra |
| `input_features` | datetime, power_lag_24h, power_lag_7d, temperature |
| `training_strategy` | direct_multistep, horizon_as_feature, end_to_end |
| `generator_capacity_estimation` | none, simple_p99, differentiable_physics |
| `switching_event_detection` | none, simple_statistical |
| `pre_training` | none, CERRA |

> Estimating **cost savings (£)** attributable to each forecasting approach, per leaderboard row, is
> a 🔬 v2 stretch goal.
