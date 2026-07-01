# XGBoost forecast-skill quick wins (issue #145)

## Context

Issue #145 (v0.5): quick wins to make XGBoost a strong baseline before the advanced approaches
land — explicitly *not* deep ML work ("little point in spending ages on the ML model before we
have good capacity estimation"). Its nine sub-issues, ordered here by expected
skill-per-unit-effort, with implementation notes and links to the other plans in this
directory.

**Measure before optimising.** Land plan 02 (persistence/climatology baselines — without them
"improved" is unanchored) and plan 03 Phase A (horizon-sliced metrics — several wins below are
horizon-specific and invisible in the `"all"` aggregate) first. Each win below is one
registered experiment (`register_experiment_job`) scored against the current champion on the
leaderboard fold; headline metric NMAE, sliced by horizon and `time_series_type`.

## The wins, in recommended order

### 1. Feed the model the forecast lead time (no sub-issue — found in review; ~one line)

`XGBoostForecaster` trains on `sorted(selected_features)` only
(`forecaster.py:87-88`), and `conf/model/xgboost.yaml` does not select
`nwp_lead_time_hours` — so despite the config's `training_strategy: "horizon_as_feature"` tag,
**the model never sees the horizon**. It cannot learn that NWP inputs degrade with lead time;
horizon information reaches it only through the coarse pattern of nullified lags.
`nwp_lead_time_hours` is already computed and flows through `AllFeatures`
(`tabular_feature_engineer.py:159`, `:222`), so the experiment is: add it to
`selected_features` and register. If it wins, add it to the base YAML (making the
`horizon_as_feature` tag honest).

### 2. Linearised physics features for solar and wind (#168)

Trees are poor at smooth monotone functions; give them the physics directly as features:

- **Wind**: 100 m wind speed through a generic turbine power curve (cut-in ~3 m/s, rated
  ~12–14 m/s, cut-out ~25 m/s — a piecewise/logistic closed form is fine).
- **Solar**: clear-sky irradiance at the site (Haurwitz or Ineichen closed form from solar
  zenith — **not pvlib**, which depends on pandas, forbidden in this repo), plus a simplified
  PV power feature from `downward_short_wave_radiation_flux_surface` and `temperature_2m`
  (irradiance × a temperature-derate term). Clear-sky index (actual/clear-sky irradiance) is
  the single most informative derived solar feature.

Implement as new derived-feature names in the feature parser (`_parsed_features.py`) computed
from existing NWP columns + site lat/lon from metadata — same pattern as the existing
`windchill` static feature. Expect the win to concentrate in the PV and wind
`time_series_type` slices.

### 3. Per-`time_series_type` feature lists (#201)

YAML gains `selected_features_by_type: {type: [...]}` plus the existing `selected_features` as
the default for unlisted types. Since `XGBoostForecaster` trains one booster per series, each
booster can simply resolve its series' type (already in `AllFeatures`) to a feature list at
train/predict time; persist the mapping in `meta.json`. Pairs naturally with #168 (solar
features for PV sites, turbine features for wind, neither for demand substations).

### 4. Per-horizon-window models (#149)

One booster per `(time_series_id, horizon_window)` — e.g. 0–2 d, 2–7 d, 7–14 d, configurable as
`horizon_windows` in the config. Train routes rows by `nwp_lead_time_hours`; predict routes the
same way; `save`/`load` gain a window dimension in the filename. This directly attacks the
"one booster must serve all horizons" compromise. **Requires plan 03 Phase A to evaluate** —
its win is by construction horizon-sliced. Compare against win 1 (lead time as a feature):
if #1 captures most of the benefit, #149's extra model count may not pay.

### 5. Batched training via `xgb.DataIter` (#91 — enabler)

Issue #91 already contains a complete, validated implementation design
(`LazyFrameBatchIter` + `QuantileDMatrix`, grouping-agnostic, no temp disk, `train_batch_size`
config field) — treat the issue body as the plan and implement as written. No direct skill
gain; it unblocks #148 and #104.

### 6. Train on more ensemble members (#148 — after #91)

Training on all 51 members multiplies training data ~51× for correlated rows. Before paying
that: run the dose-response experiment — control-only (today) vs ~8 spread members vs all 51
(the NWP loader already takes `ensemble_members: list[int]`, `cv_assets.py:237`). Training on
members also teaches the model the member-spread input distribution it actually sees at
inference (today it is trained on member 0 and applied to all 51 — a train/serve input-skew
flagged in the review). Note ensemble *calibration* itself is plan 03's territory.

### 7. Global model per `time_series_type` (#104)

One booster for all primaries, one for all PV sites, etc. — the biggest potential win for
data-poor series (transfer across sites), and the stepping stone to v2-scale models.
**Hard prerequisite: per-series target normalisation.** Raw-MW targets only work per-series; a
global booster mixing a 200 MW GSP with a 5 MW solar farm needs `power /
effective_capacity_mw` targets (the `effective_capacity` asset already exists) with the
inverse transform at predict time, plus static per-series features (capacity, type, H3
lat/lon) so the booster can tell sites apart. Needs #91 at ensemble scale. Largest item here —
"quick" only relative to the deep-ML roadmap; consider it the boundary of this plan.

## Explicitly deferred (not quick, or not skill)

- **#167 CERRA pre-training** — needs a whole new data-source ingestion (CERRA download,
  contracts, reanalysis-vs-forecast handling) before any training trick. The right home for
  its evaluation design is `docs/ml_experimentation/evaluating-new-data-sources.md`.
- **#198 NWP row-group layout** — throughput, not skill (~1 h per 51-member fold prediction is
  decode-bound). Do it when experiment iteration speed becomes the bottleneck — likely around
  #148, which multiplies NWP reads.
- **#176 local-time power lags** — a DST edge case affecting a handful of half-hours per year;
  the issue itself says it may not be worth worrying about yet. Revisit if the metrics slices
  ever show a DST-transition artefact.

## Verification

Each win lands as its own experiment on the leaderboard: register → `full_cv` → `metrics`
(leaderboard scope) → compare NMAE (overall + per-type + per-horizon-slice once plan 03A is
in) against the current champion and the plan-02 baselines. Keep losing experiments in MLflow
(negative results are results); promote winners' settings into `conf/model/xgboost.yaml` one
at a time so attribution stays clean.
