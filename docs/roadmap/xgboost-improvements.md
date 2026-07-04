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
with additional tricks from the 2026-07 codebase review, grouped into **effort tiers** and
ordered by expected skill per unit effort *within* each tier — so an item late in the list can
still be high value; it just costs more to land.

**Horizon focus: days 3–10.** The product delivers a 14-day horizon
([requirements](../background/requirements.md#core-objectives)), but users mostly act on
forecasts roughly 3 to 10 days ahead. That band shapes the ordering below in three ways:
wins that are fully forecastable at any horizon (calendar features, weather physics) outrank
wins concentrated at day 0–2 (persistence-like anchors); ECMWF ENS steps drop from 3-hourly to
6-hourly beyond 144 h, so interpolation quality matters most exactly where users look; and by
day 7–10 the control member is an increasingly unrepresentative sample of the ensemble.

**Measure before optimising.** Land the
[persistence/climatology baselines](metrics-and-leaderboard.md#baseline-forecasters) (without
them "improved" is unanchored) and
[horizon-sliced metrics](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
(several wins below are horizon-specific and invisible in the `"all"` aggregate) first. Each
win is one registered experiment (`register_experiment_job`) scored against the current
champion on the leaderboard fold; headline metric NMAE, sliced by horizon and
`time_series_type`, with the 3–10 day band as the headline horizon slice. Several items
interact (e.g. init-time-anchored features overlap short lags at short horizons), so land
winners into `conf/model/xgboost.yaml` one at a time to keep attribution clean.

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

### 2. UK holiday and calendar features

Day-of-week and time-of-year features exist, but no holiday flags. GB demand on bank holidays
looks like a Sunday, and the Christmas–New Year fortnight is its own regime; for
demand-dominated series this is one of the highest-value features in the load-forecasting
literature and costs a static lookup (the `holidays` package is pure-Python — no pandas).
Add `is_bank_holiday`, `is_day_before_holiday`, `is_day_after_holiday`, and a
Christmas-proximity feature. Fully forecastable at any horizon — squarely in the 3–10 day
band, which is why it sits this high.

### 3. Raw ordinal time features alongside sin/cos

Trees split axis-parallel: isolating "evening peak" from sin/cos pairs takes multiple awkward
splits, while a raw `half_hour_of_day` integer does it in one. Keep the sin/cos (good at the
midnight wrap-around) and *add* raw ordinals (`half_hour_of_day`, `day_of_year`) as new
`TimeFeature` names. Trivial experiment, modest-but-real expected gain.

### 4. Early stopping instead of fixed `n_estimators=500`

Every series currently gets 500 trees at lr 0.05, whether it has 15 months of clean data or
7 months of noisy data. Hold out the last few weeks of each training window as an eval set
with `early_stopping_rounds`, and each booster right-sizes itself — removing a silent
per-series over/underfit and making every later experiment cleaner. Related one-line
experiment: **recency sample weights** (exponentially decaying with sample age) to track drift
from new connections. Note the overlap with item 10, whose remaining value at 3–10 days is
also mostly drift-tracking — whichever lands second should expect a smaller measured win.

## Tier 2 — cheap feature engineering (about a day each)

### 5. Per-`time_series_type` feature lists

Issues: [#201](https://github.com/openclimatefix/nged-substation-forecast/issues/201),
[#107](https://github.com/openclimatefix/nged-substation-forecast/issues/107)

Every series type currently shares one demand-oriented feature list: four power lags (all
nullified in the 3–10 day band anyway), day-of-week and time-of-day calendar features that
mean nothing to a wind farm, and a windchill feature aimed at demand. Wind forecasts are
currently poor, and the hypothesis is that feature noise is a big part of why: a wind-tailored
list (wind speeds and directions at 10 m and 100 m, pressure — dropping the calendar features
and power lags) could deliver a large win for the wind slice *immediately*, before any new
features exist. Same logic, more mildly, for PV.

Mechanics: YAML gains `selected_features_by_type: {type: [...]}` with the existing
`selected_features` as the default for unlisted types. Boosters are already per-series, so
each can resolve its series' type (in `AllFeatures`) to a feature list at train/predict time;
persist the mapping in `meta.json`. The value compounds as items 2 and 6–9 diverge the useful
per-type sets (solar features for PV, turbine features for wind, holidays for demand).

### 6. Training-data hygiene, the cheap version

Full data cleaning is roadmap v0.4, but training on stuck meters and false zeros actively
teaches the model wrong targets *today* (quality issues are ~10%+ of some series). Cheap
interim: drop training rows whose target sits inside a detected stuck window (rolling std ≈ 0;
`DataQualitySettings.stuck_std_threshold` already exists) or an isolated exact-zero run.
Cleaning only the *training* target is much lower-risk than cleaning delivered data, and it
protects every subsequent experiment from learning artefacts.

### 7. Effective (smoothed) temperature and degree-day features

GB demand responds to *lagged* temperature (building thermal inertia), not instantaneous —
National Grid's demand models use an exponentially-smoothed "effective temperature". Add an
EWM of `temperature_2m` over the past ~1–3 days (computed from the NWP trajectory itself, so
horizon-safe) plus heating-degree `max(15.5 − T, 0)`. Linearises the demand–temperature
relationship the same way item 8 does for generation.

### 8. Linearised physics features for solar and wind

Issue: [#168](https://github.com/openclimatefix/nged-substation-forecast/issues/168)

Trees are poor at smooth monotone functions; give them the physics directly. Everything here
hangs off one shared solar-position/clear-sky helper, so this lands as a three-stage stack —
and the order matters: build the clear-sky machinery and fix the upsample **first**, then
derive features from the already-upsampled columns.

**(a) Solar position and clear-sky helper.** Closed-form, from site latitude $\phi$ and
longitude $\lambda$ (metadata) and `valid_time` — **not pvlib**, which depends on pandas,
forbidden here. Declination (Cooper's equation, $d$ = day of year):

$$
\delta = -23.45^\circ \, \cos\!\left(\tfrac{360^\circ}{365}\,(d + 10)\right)
$$

Hour angle from solar time ($t_{\text{solar}} = t_{\text{UTC}} + \lambda / 15^\circ$; the
equation-of-time correction is ≤ ~16 min and can be dropped for a feature):

$$
\omega = 15^\circ\!/\mathrm{h} \times (t_{\text{solar}} - 12\,\mathrm{h})
$$

Solar zenith angle:

$$
\cos\theta_z = \sin\phi \sin\delta + \cos\phi \cos\delta \cos\omega
$$

Clear-sky irradiance via Haurwitz (GHI-only, no turbidity input — ideal for a feature):

$$
\mathrm{GHI}_{\text{cs}} = 1098 \, \cos\theta_z \, e^{-0.057/\cos\theta_z}
\quad \text{for } \cos\theta_z > 0, \text{ else } 0
$$

**(b) Interpolate clear-sky *index*, not raw irradiance, in the 30-min upsample.** The NWP
upsample (`_upsample_nwp_to_half_hourly`) linearly interpolates native steps to half-hourly;
for shortwave radiation that smears sunrise/sunset ramps badly. And the native steps are
3-hourly only out to 144 h — from day 6 to day 15 they are **6-hourly**, so the smearing is
worst precisely in the primary 3–10 day user band. Standard fix: divide irradiance by
clear-sky irradiance at the native steps, interpolate the (slowly-varying) clear-sky index,
multiply back by the *half-hourly* clear-sky curve. Sharpens exactly the ramps that matter for
PV sites — and for MVA-metered substations whose readings bounce off zero from embedded solar.

**(c) Derived features, computed from the upsampled columns:**

- **Clear-sky index** $k_c = \mathrm{GHI} / \mathrm{GHI}_{\text{cs}}$ — the single most
  informative derived solar feature. Null it when solar elevation is below ~5–10° (the ratio
  blows up near the horizon) and clip to roughly $[0, 1.2]$.
- **Simplified PV power proxy** (PVWatts-style). Cell temperature from the Ross/NOCT model,
  then a linear temperature derate:

    $$
    T_{\text{cell}} = T_{2\text{m}} + k \, \mathrm{GHI},
    \qquad k = \tfrac{\mathrm{NOCT} - 20\,°\mathrm{C}}{800\,\mathrm{W\,m^{-2}}} \approx 0.03
    $$

    $$
    P_{\text{pv}} \propto \mathrm{GHI} \, \bigl(1 + \gamma \, (T_{\text{cell}} - 25\,°\mathrm{C})\bigr),
    \qquad \gamma \approx -0.004\,/°\mathrm{C}
    $$

    i.e. one expression, `ghi * (1 - 0.004 * (t2m_celsius + 0.03 * ghi - 25))`, clipped at 0.
    Deliberately capacity-free (per-series boosters learn the scale) and with no
    tilt/orientation modelling — embedded PV behind a substation is an unknown mix of
    orientations, and the booster can bend the proxy per series.

- **Wind power curve**: 100 m wind speed through a generic *farm-level* power curve — either
  a piecewise form (zero below cut-in ~3 m/s, normalised cubic ramp
  $(v^3 - v_{ci}^3)/(v_r^3 - v_{ci}^3)$ to rated ~12–14 m/s, flat to cut-out ~25 m/s, zero
  above) or a logistic sigmoid **masked to zero above cut-out** (an unmasked logistic is
  actively wrong in storms — precisely when NGED cares). Exact shape matters less than it
  looks: farm-level curves are smoother than single-turbine ones (aggregation, wakes,
  hub-height spread), the booster monotonically re-bends the ramp anyway, and raw
  `wind_speed_100m` stays in the feature list. What the proxy must get right is the
  saturation at rated and the two dead zones — the parts trees can't build from raw speed.
  (Large errors in the steep ramp region are dominated by NWP speed error amplified by the
  physics' own $dP/dv$ — no closed form removes that; item 14 is what addresses it.)

Implement stage (c) as derived-feature names in `_parsed_features.py` — same pattern as the
existing `windchill` feature. That pattern gives the correct order of operations for free:
`StaticFeature` expressions are applied *after* `_upsample_nwp_to_half_hourly`, so derived
features are computed at half-hourly resolution from already-interpolated inputs. This is why
(b) comes before (c): the PV proxy is essentially linear in GHI, so deriving it from
linearly-interpolated irradiance (or worse, interpolating a 3/6-hourly proxy directly) smears
the sunrise/sunset ramps exactly as raw irradiance does — stage (b) is what makes the
half-hourly solar features sharp. The same principle already holds for wind: interpolate the
smooth variable (speed), then apply the nonlinear power curve.

Expect the win to concentrate in the PV and wind `time_series_type` slices; pairs with the
per-type feature lists of item 5.

### 9. Monotone constraints for the generation models

XGBoost's `monotone_constraints`: PV power non-decreasing in irradiance, wind power monotone
in speed below rated. Mostly buys sane extrapolation in weather regimes the training year
never saw — precisely the failure mode of a single-fold training set. A config-field addition
once item 8's features exist.

## Tier 3 — new feature machinery (days)

### 10. Init-time-anchored features (current-level anchor; prerequisite for item 15)

All current power lags are anchored to `valid_time` and nullified when lag ≤ lead time — so at
a 7-day horizon every lag under 168 h is null, and at 14 days the model has almost no
recent-level information. Add features anchored to `power_fcst_init_time` instead: "last
observed power at forecast time", "mean/max of the 24 h before forecast time", "power at this
half-hour-of-day yesterday relative to forecast time". These are **never leaky and never null
at any horizon** (no nullification machinery — but respect the same comms-delay conservatism
as `_nullify_leaky_lags`: observations strictly *before* init time).

Be realistic about where the win lands, though. Per-series boosters already bake each series'
typical level, weekly cycle, and seasonal shape into the trees, so in the 3–10 day band the
incremental information is mostly **drift since training** (new connections) and short-lived
level anomalies — and demand-anomaly autocorrelation at 7 days is modest. The large wins sit
at day 0–2, where these features largely subsume "blend with persistence" (XGBoost learns the
blend itself once it has the anchor) — outside the primary user band. They also overlap the
recency sample weights of item 4; whichever lands second should expect a smaller measured win.
Where they become *structurally essential* is the global model (item 15), whose booster cannot
bake in per-series level. So: a moderate expected win now, and a hard prerequisite later.
**Must ship with a leakage test** in the spirit of the existing `_nullify_leaky_lags` tests
before it's trusted.

### 11. Neighbouring-H3-cell weather context

Each series currently gets its nearest NWP cell only. Add the mean and gradient across the
neighbouring ring (~9 extra columns) for frontal-timing and wind-ramp information. Modest
expected gain, cheap given the `geo` H3 machinery exists.

## Tier 4 — structural model changes (weeks)

### 12. Per-horizon-window models

Issue: [#149](https://github.com/openclimatefix/nged-substation-forecast/issues/149)

One booster per `(time_series_id, horizon_window)` — e.g. 0–2 d, 2–7 d, 7–14 d, configurable.
Train and predict route rows by `nwp_lead_time_hours`; `save`/`load` gain a window dimension.
**Requires [horizon-sliced metrics](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
to evaluate** — its win is by construction horizon-sliced. Under the 3–10 day focus the
interesting experiment is narrower than "many windows": does a dedicated ~3–10 day model beat
the lead-time feature *in that band*? Compare against item 1 first — if the lead-time feature
captures most of the benefit, the extra model count may not pay.

### 13. Batched training via `xgb.DataIter` (enabler)

Issue: [#91](https://github.com/openclimatefix/nged-substation-forecast/issues/91)

Issue #91 already contains a complete, validated implementation design (`LazyFrameBatchIter` +
`QuantileDMatrix`, grouping-agnostic, no temp disk, `train_batch_size` config field) — treat
the issue body as the plan and implement as written. No direct skill gain; unblocks 14 and 15.

### 14. Train on more ensemble members (after 13)

Issue: [#148](https://github.com/openclimatefix/nged-substation-forecast/issues/148)

Training on all 51 members multiplies training data ~51× for correlated rows. Run the
dose-response experiment first: control-only (today) vs ~8 spread members vs all 51 (the NWP
loader already takes `ensemble_members: list[int]`, `cv_assets.py:237`). Training on members
also teaches the model the member-spread input distribution it actually sees at inference —
the train/serve input-skew flagged in the review. That skew grows with lead time: by day 7–10
the control member is an increasingly unrepresentative sample of the ensemble, so the value of
this item concentrates precisely in the primary user band — worth remembering when deciding
how soon to invest in item 13. Ensemble *calibration* itself belongs to
[probabilistic evaluation](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics).
Member training is also one of the
[double-counting mitigations](../techniques/probabilistic-forecasting.md#caveat-double-counting-weather-uncertainty)
for the Phase-D quantile-ensemble pipeline — a second reason to land it, alongside this item
and item 1 (the lead-time feature), before or with the quantile model family.

### 15. Global model per `time_series_type`

Issue: [#104](https://github.com/openclimatefix/nged-substation-forecast/issues/104)

One booster for all primaries, one for all PV sites, etc. — the biggest potential win for
data-poor series (transfer across sites), and the stepping stone to V2 scale. **Hard
prerequisite: per-series target normalisation** — a global booster mixing a 200 MW GSP with a
5 MW solar farm needs `power / effective_capacity_mw` targets (the `effective_capacity` asset
exists) with the inverse transform at predict time, plus static per-series features (capacity,
type, lat/lon) so the booster can tell sites apart — plus the init-time-anchored features of
item 10, which supply the current-level signal a global booster cannot bake in per series.
Needs item 13 at ensemble scale. The boundary of "quick".

## Explicitly deferred (not quick, or not skill)

- **[#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167) CERRA pre-training** — needs a whole new data-source ingestion (CERRA download,
  contracts, reanalysis-vs-forecast handling) before any training trick. The evaluation design
  belongs to
  [Evaluating new data sources](../ml_experimentation/evaluating-new-data-sources.md).
- **[#176](https://github.com/openclimatefix/nged-substation-forecast/issues/176) local-time power lags** — a DST edge case affecting a handful of half-hours per year;
  the issue itself says it may not be worth worrying about yet. Revisit if the metrics slices
  ever show a DST-transition artefact.

## How each win is evaluated

Each win lands as its own experiment on the leaderboard: register → `full_cv` → `metrics`
(leaderboard scope) → compare NMAE (overall + per-type + per-horizon-slice once horizon
slicing is in, with the 3–10 day band as the headline slice) against the current champion and
the [baselines](metrics-and-leaderboard.md#baseline-forecasters). Keep losing experiments in
MLflow (negative results are results); promote winners' settings into
`conf/model/xgboost.yaml` one at a time so attribution stays clean.
