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
| Normalised MAE (NMAE) | Deterministic | ✅ | MAE normalised by mean absolute power — comparable across substations of different sizes. |
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
> [Running an ML experiment end-to-end](../ml_experimentation/dagster-workflow.md#step-8--materialise-metrics).

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
