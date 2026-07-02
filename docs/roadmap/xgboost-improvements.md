# XGBoost improvements ("quick wins")

> **Status: 🚧 Planned (v0.5) — deferred until v0.1 is live on AWS.** Epic:
> [#145](https://github.com/openclimatefix/nged-substation-forecast/issues/145); the Tier-1
> quick wins are [#230](https://github.com/openclimatefix/nged-substation-forecast/issues/230).
> Getting *any* forecast live ([Live service](live-service.md)) takes priority over forecast
> quality.

Quick wins to make XGBoost a strong baseline before the advanced approaches land — explicitly
*not* deep ML work ("little point in spending ages on the ML model before we have good capacity
estimation"). This page merges the sub-issues of
[#145](https://github.com/openclimatefix/nged-substation-forecast/issues/145)
with additional tricks from the 2026-07 codebase review, ordered **best
bang-for-the-buck first** (expected skill per unit effort), grouped into effort tiers.

**Measure before optimising.** Land the
[persistence/climatology baselines](metrics-and-leaderboard.md#baseline-forecasters) (without
them "improved" is unanchored) and
[horizon-sliced metrics](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
(several wins below are horizon-specific and invisible in the `"all"` aggregate) first. Each
win is one registered experiment (`register_experiment_job`) scored against the current
champion on the leaderboard fold; headline metric NMAE, sliced by horizon and
`time_series_type`. Several items interact (e.g. init-time-anchored features overlap short lags
at short horizons), so land winners into `conf/model/xgboost.yaml` one at a time to keep
attribution clean.

## Tier 1 — config-level changes (hours each)

### 1. Feed the model the forecast lead time (review discovery; ~one line)

`XGBoostForecaster` trains on `sorted(selected_features)` only (`forecaster.py:87-88`), and
`conf/model/xgboost.yaml` does not select `nwp_lead_time_hours` — so despite the config's
`training_strategy: "horizon_as_feature"` tag, **the model never sees the horizon**. It cannot
learn that NWP inputs degrade with lead time; horizon information reaches it only through the
coarse pattern of nullified lags. `nwp_lead_time_hours` is already computed and flows through
`AllFeatures` (`tabular_feature_engineer.py:159`, `:222`), so the experiment is: add it to
`selected_features` and register. If it wins, add it to the base YAML (making the
`horizon_as_feature` tag honest).

### 2. Raw ordinal time features alongside sin/cos

Trees split axis-parallel: isolating "evening peak" from sin/cos pairs takes multiple awkward
splits, while a raw `half_hour_of_day` integer does it in one. Keep the sin/cos (good at the
midnight wrap-around) and *add* raw ordinals (`half_hour_of_day`, `day_of_year`) as new
`TimeFeature` names. Trivial experiment, modest-but-real expected gain.

### 3. Early stopping instead of fixed `n_estimators=500`

Every series currently gets 500 trees at lr 0.05, whether it has 15 months of clean data or
7 months of noisy data. Hold out the last few weeks of each training window as an eval set
with `early_stopping_rounds`, and each booster right-sizes itself — removing a silent
per-series over/underfit and making every later experiment cleaner. Related one-line
experiment: **recency sample weights** (exponentially decaying with sample age) to track drift
from new connections.

### 4. UK holiday and calendar features

Day-of-week and time-of-year features exist, but no holiday flags. GB demand on bank holidays
looks like a Sunday, and the Christmas–New Year fortnight is its own regime; for
demand-dominated series this is one of the highest-value features in the load-forecasting
literature and costs a static lookup (the `holidays` package is pure-Python — no pandas).
Add `is_bank_holiday`, `is_day_before_holiday`, `is_day_after_holiday`, and a
Christmas-proximity feature. Fully forecastable at any horizon.

## Tier 2 — cheap feature engineering (about a day each)

### 5. Training-data hygiene, the cheap version

Full data cleaning is roadmap v0.4, but training on stuck meters and false zeros actively
teaches the model wrong targets *today* (quality issues are ~10%+ of some series). Cheap
interim: drop training rows whose target sits inside a detected stuck window (rolling std ≈ 0;
`DataQualitySettings.stuck_std_threshold` already exists) or an isolated exact-zero run.
Cleaning only the *training* target is much lower-risk than cleaning delivered data, and it
protects every subsequent experiment from learning artefacts.

### 6. Effective (smoothed) temperature and degree-day features

GB demand responds to *lagged* temperature (building thermal inertia), not instantaneous —
National Grid's demand models use an exponentially-smoothed "effective temperature". Add an
EWM of `temperature_2m` over the past ~1–3 days (computed from the NWP trajectory itself, so
horizon-safe) plus heating-degree `max(15.5 − T, 0)`. Linearises the demand–temperature
relationship the same way item 7 does for generation.

### 7. Linearised physics features for solar and wind ([#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168))

Trees are poor at smooth monotone functions; give them the physics directly:

- **Wind**: 100 m wind speed through a generic turbine power curve (cut-in ~3 m/s, rated
  ~12–14 m/s, cut-out ~25 m/s — a piecewise/logistic closed form is fine).
- **Solar**: clear-sky irradiance at the site (Haurwitz or Ineichen closed form from solar
  zenith — **not pvlib**, which depends on pandas, forbidden here), a simplified PV power
  feature from `downward_short_wave_radiation_flux_surface` and `temperature_2m`
  (irradiance × temperature-derate), and the **clear-sky index** (actual/clear-sky irradiance)
  — the single most informative derived solar feature.

Implement as derived-feature names in `_parsed_features.py` computed from existing NWP columns
plus site lat/lon from metadata — same pattern as the existing `windchill` feature. Expect the
win to concentrate in the PV and wind `time_series_type` slices.

### 8. Interpolate clear-sky *index*, not raw irradiance, in the 30-min upsample

The NWP upsample linearly interpolates 3-hourly values to half-hourly; for shortwave radiation
that smears sunrise/sunset ramps badly. Standard fix: divide irradiance by clear-sky
irradiance at the 3-hourly points, interpolate the (slowly-varying) clear-sky index, multiply
back by the *half-hourly* clear-sky curve. Sharpens exactly the ramps that matter for PV sites
— and for MVA-metered substations whose readings bounce off zero from embedded solar. Pairs
with item 7, which builds the clear-sky model anyway.

### 9. Monotone constraints for the generation models

XGBoost's `monotone_constraints`: PV power non-decreasing in irradiance, wind power monotone
in speed below rated. Mostly buys sane extrapolation in weather regimes the training year
never saw — precisely the failure mode of a single-fold training set. A config-field addition
once items 7's features exist.

## Tier 3 — new feature machinery (days)

### 10. Init-time-anchored features (fixes the "information desert" at day 2+)

All current power lags are anchored to `valid_time` and nullified when lag ≤ lead time — so at
a 7-day horizon every lag under 168 h is null, and at 14 days the model has almost no
recent-level information. Add features anchored to `power_fcst_init_time` instead: "last
observed power at forecast time", "mean/max of the 24 h before forecast time", "power at this
half-hour-of-day yesterday relative to forecast time". These are **never leaky and never null
at any horizon** (no nullification machinery — but respect the same comms-delay conservatism
as `_nullify_leaky_lags`: observations strictly *before* init time). They give the model the
series' current level — which drives skill wherever load has drifted — and largely subsume
"blend with persistence at short horizons", since XGBoost learns the blend itself once it has
the anchor. **Must ship with a leakage test** in the spirit of the existing
`_nullify_leaky_lags` tests before it's trusted. Highest expected value of the new ideas;
ranked here only because it needs a new join path plus those tests.

### 11. Neighbouring-H3-cell weather context

Each series currently gets its nearest NWP cell only. Add the mean and gradient across the
neighbouring ring (~9 extra columns) for frontal-timing and wind-ramp information. Modest
expected gain, cheap given the `geo` H3 machinery exists.

### 12. Per-`time_series_type` feature lists ([#201](https://github.com/openclimatefix/nged-substation-forecast/issues/201))

YAML gains `selected_features_by_type: {type: [...]}` with the existing `selected_features` as
the default for unlisted types. Boosters are already per-series, so each can resolve its
series' type (in `AllFeatures`) to a feature list at train/predict time; persist the mapping
in `meta.json`. Becomes genuinely valuable once items 4–9 diverge the useful feature sets
(solar features for PV, turbine features for wind, holidays for demand).

## Tier 4 — structural model changes (weeks)

### 13. Per-horizon-window models ([#149](https://github.com/openclimatefix/nged-substation-forecast/issues/149))

One booster per `(time_series_id, horizon_window)` — e.g. 0–2 d, 2–7 d, 7–14 d, configurable.
Train and predict route rows by `nwp_lead_time_hours`; `save`/`load` gain a window dimension.
**Requires [horizon-sliced metrics](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
to evaluate** — its win is by construction horizon-sliced. Compare against item 1 first: if
the lead-time feature captures most of the benefit, the extra model count may not pay.

### 14. Batched training via `xgb.DataIter` ([#91](https://github.com/openclimatefix/nged-substation-forecast/issues/91) — enabler)

Issue [#91](https://github.com/openclimatefix/nged-substation-forecast/issues/91) already contains a complete, validated implementation design (`LazyFrameBatchIter` +
`QuantileDMatrix`, grouping-agnostic, no temp disk, `train_batch_size` config field) — treat
the issue body as the plan and implement as written. No direct skill gain; unblocks 15 and 16.

### 15. Train on more ensemble members ([#148](https://github.com/openclimatefix/nged-substation-forecast/issues/148) — after 14)

Training on all 51 members multiplies training data ~51× for correlated rows. Run the
dose-response experiment first: control-only (today) vs ~8 spread members vs all 51 (the NWP
loader already takes `ensemble_members: list[int]`, `cv_assets.py:237`). Training on members
also teaches the model the member-spread input distribution it actually sees at inference —
the train/serve input-skew flagged in the review. Ensemble *calibration* itself belongs to
[probabilistic evaluation](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics).

### 16. Global model per `time_series_type` ([#104](https://github.com/openclimatefix/nged-substation-forecast/issues/104))

One booster for all primaries, one for all PV sites, etc. — the biggest potential win for
data-poor series (transfer across sites), and the stepping stone to V2 scale. **Hard
prerequisite: per-series target normalisation** — a global booster mixing a 200 MW GSP with a
5 MW solar farm needs `power / effective_capacity_mw` targets (the `effective_capacity` asset
exists) with the inverse transform at predict time, plus static per-series features (capacity,
type, lat/lon) so the booster can tell sites apart. Needs item 14 at ensemble scale. The
boundary of "quick".

## Explicitly deferred (not quick, or not skill)

- **[#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167) CERRA pre-training** — needs a whole new data-source ingestion (CERRA download,
  contracts, reanalysis-vs-forecast handling) before any training trick. The evaluation design
  belongs to
  [Evaluating new data sources](../ml_experimentation/evaluating-new-data-sources.md).
- **[#198](https://github.com/openclimatefix/nged-substation-forecast/issues/198) NWP row-group layout** — throughput, not skill (~1 h per 51-member fold prediction is
  decode-bound). Do it when experiment iteration speed becomes the bottleneck — likely around
  item 15, which multiplies NWP reads.
- **[#176](https://github.com/openclimatefix/nged-substation-forecast/issues/176) local-time power lags** — a DST edge case affecting a handful of half-hours per year;
  the issue itself says it may not be worth worrying about yet. Revisit if the metrics slices
  ever show a DST-transition artefact.

## How each win is evaluated

Each win lands as its own experiment on the leaderboard: register → `full_cv` → `metrics`
(leaderboard scope) → compare NMAE (overall + per-type + per-horizon-slice once horizon
slicing is in) against the current champion and the
[baselines](metrics-and-leaderboard.md#baseline-forecasters). Keep losing experiments in
MLflow (negative results are results); promote winners' settings into
`conf/model/xgboost.yaml` one at a time so attribution stays clean.
