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

### Feed the model the forecast lead time (review discovery; ~one line)

`XGBoostForecaster` trains on `sorted(selected_features)` only (`forecaster.py:87-88`), and
`conf/model/xgboost.yaml` does not select `nwp_lead_time_hours` — so despite the config's
`training_strategy: "horizon_as_feature"` tag, **the model never sees the horizon**. It cannot
learn that NWP inputs degrade with lead time; horizon information reaches it only through the
coarse pattern of nullified lags. `nwp_lead_time_hours` is already computed and flows through
`AllFeatures` (`tabular_feature_engineer.py:159`, `:222`), so the experiment is: add it to
`selected_features` and register. If it wins, add it to the base YAML (making the
`horizon_as_feature` tag honest).

### UK holiday and calendar features

Day-of-week and time-of-year features exist, but no holiday flags. GB demand on bank holidays
looks like a Sunday, and the Christmas–New Year fortnight is its own regime; for
demand-dominated series this is one of the highest-value features in the load-forecasting
literature and costs a static lookup (the `holidays` package is pure-Python — no pandas).
Add `is_bank_holiday`, `is_day_before_holiday`, `is_day_after_holiday`, and a
Christmas-proximity feature. Fully forecastable at any horizon — squarely in the 3–10 day
band, which is why it sits this high.

This item also double-serves as the covariate set of the v0.6 switching detector's
[stage-1 baseline](switching-events.md#the-baseline-shared-foundation),
which raises the bar on encoding: the detector consumes the baseline's residuals raw, so an
unmodelled behavioural day becomes a phantom event candidate. Prefer encodings that generalise
across sparse examples — a days-to-nearest-holiday feature and a holiday-name categorical
rather than a lone `is_bank_holiday` flag — and cover the days a day-of-year feature
structurally cannot represent: Easter (which wanders across roughly five weeks of the
calendar), regional school half-terms, and major broadcast events such as England playing in
the later stages of a Football World Cup. Sporting fixtures carry a forecastability asymmetry
the bank holidays do not: they are known perfectly in hindsight (fine for the detector's
hindcast baseline), but at a 3–10 day horizon whether England will still be in the tournament
may be unknown at forecast time, so the forward-forecast version needs either a
"possible England match" encoding or an acceptance of that uncertainty.

### Raw ordinal time features alongside sin/cos

Trees split axis-parallel: isolating "evening peak" from sin/cos pairs takes multiple awkward
splits, while a raw `half_hour_of_day` integer does it in one. Keep the sin/cos (good at the
midnight wrap-around) and *add* raw ordinals (`half_hour_of_day`, `day_of_year`) as new
`TimeFeature` names. Trivial experiment, modest-but-real expected gain.

### Early stopping instead of fixed `n_estimators=500`

Every series currently gets 500 trees at lr 0.05, whether it has 15 months of clean data or
7 months of noisy data. Hold out the last few weeks of each training window as an eval set
with `early_stopping_rounds`, and each booster right-sizes itself — removing a silent
per-series over/underfit and making every later experiment cleaner. Related one-line
experiment: **recency sample weights** (exponentially decaying with sample age) to track drift
from new connections. Note the overlap with the init-time-anchored features, whose remaining value at 3–10 days is
also mostly drift-tracking — whichever lands second should expect a smaller measured win.

### Aligned lagged weather — the single-stage ablation control

For each power lag the model already receives, also feed it the *weather at that same lagged
time* (e.g. `temperature_2m_lag_48h` beside the 48 h power lag). This is pure config —
`LagFeature.base_col` already accepts weather variables — and lagged *datetime* adds nothing
new (it is deterministic given the target's datetime features and the fixed lag offset; the
same holds for holiday flags at the lag time once the holiday and calendar features land), so
aligned weather is the only genuinely new information. In principle the booster can then judge
how *normal* each lagged power value is — power at the lagged time relative to what the weather
then would predict — which is exactly the anomaly signal that
[the residual-lag features](#residual-lag-features-from-the-switching-detector-baseline) engineers explicitly
with a two-stage residual pipeline (the
[full design](switching-events.md#approach-1-the-two-stage-forecaster)
lives on the switching-events page). Feature engineering never adds information, only inductive
bias — so whether the explicit two-stage version is worth its machinery is an empirical
question, and this config-only variant is the cheap way to start answering it.

**Pros, relative to the two-stage residual pipeline:**

- **Almost free.** A config change: no two-pass pipeline, no per-fold baseline training. (One
  caveat is shared rather than avoided: lagged weather at past target times rides the same
  freshest-NWP-run join as the residual-lag hindcasts — a join with no publication-time cut,
  leak-free today only as a side effect of daily run cadence — so the availability cut planned
  in [#356](https://github.com/openclimatefix/nged-substation-forecast/issues/356) hardens this
  item and the residual-lag features alike.)
- **No fold-hygiene leakage risk.** The two-stage design's subtlest failure mode — a baseline
  trained on data that overlaps the evaluation fold — cannot occur, because there is no
  baseline model.
- **End-to-end optimisation.** The booster extracts whatever notion of "anomaly" actually helps
  the forecast objective, rather than the one a residual definition pre-commits to (median
  residual, a particular normalisation); no stage-1 bias is frozen into a feature.

**Cons — why we expect it to learn a much cruder "how normal is this lag" signal:**

- **Trees are structurally bad at subtraction.** XGBoost splits axis-parallel on single
  features, and differences or ratios of continuous inputs are notoriously hard for trees to
  represent. Using "power relative to expectation" here means approximating the whole
  weather → power baseline *inside interactions with* the lagged power — the expensive kind of
  structure for a gradient-boosted tree — whereas the two-stage residual hands the model that
  comparison precomputed as a single number.
- **The per-series data regime is small.** One booster per series sees on the order of
  10⁴–10⁵ training rows — not the regime in which a tree ensemble reliably discovers a
  multi-variable implicit baseline within interactions.
- **The training signal for the implicit baseline is weak.** The two-stage design learns
  weather → power as a direct regression — every training row's target is the power concurrent
  with the weather input. Here the same function must be learned only through its indirect
  effect on predicting *future* power, where the anomaly signal matters strongly on only the
  ~10% of switching-affected rows.
- **Feature-count explosion — worst for neighbours.** Each power lag brings roughly a dozen
  aligned weather columns; that is tolerable for the self-series, but the conservation
  fingerprint that the residual-lag features' neighbour variant targets would need each *neighbour's* lagged
  power plus each neighbour's lagged weather (a different H3 cell per neighbour), per
  neighbour — an order of magnitude more columns diluting split gain on small per-series
  datasets. The two-stage design collapses every neighbour to one normalised residual, or the
  whole neighbourhood to a single sum.
- **No normalisation for free.** The two-stage baseline's quantile spread expresses each
  residual in units of that series' usual wobble at that kind of moment; the single-stage model
  must learn that heteroscedasticity implicitly as well.
- **No reusable artifact.** The two-stage baseline *is* the v0.6 switching-detector baseline —
  the same fitted model feeds the changepoint detector, the sensitivity floor, and the residual
  features. This variant produces nothing inspectable: no residual series to plot, changepoint,
  or hand to the detector; its notion of normality is buried in split structure.

**Sequencing: run this before the residual-lag features, as their ablation control.** Its measured result bounds
how much anomaly signal a tabular learner extracts *unaided*, so the residual-lag features' later comparison —
"residual features beat aligned raw features by X" — cleanly isolates the value of the explicit
baseline instead of conflating it with "the model finally saw lagged weather at all". And if
this variant already captures most of the gain, that is a cheap and important discovery to make
*before* anyone builds the two-pass machinery. Expect the direct win to be modest and
concentrated at short horizons, though: the anomaly reading only exists where the power lag
itself is non-null (lead time < lag), the same nullification limit the residual-lag features note for their
valid-time-anchored variant.

### Weather-delta compensation for power lags — an implicit handle on unmetered generation

Aligned lagged weather hands the booster the lagged power value *and* the weather at that lagged
time as two separate columns, then asks it to work out that the two belong together. This item
precomputes the comparison the booster struggles to make: for each power lag, add a single
**delta** column — the change in a weather (or weather-*proxy*) variable between the lagged time
and the target `valid_time` — so the feature literally says "the power lag is from a much sunnier
moment than the one you are forecasting; compensate." It sits between aligned lagged weather and
the [residual-lag features](#residual-lag-features-from-the-switching-detector-baseline) on the
same ablation ladder: aligned weather leaves the subtraction to the tree, the two-stage residual
does the subtraction against a full fitted baseline, and this does the one subtraction that the
unmetered-generation case needs, with no baseline model.

**Why this is the strongest case for a delta feature.** A substation with embedded (unmetered) PV
or wind meters roughly `demand − C · cf(weather)`, where `C` is the unknown behind-the-meter
capacity and `cf` the capacity factor. The correction that maps the lagged observation onto the
target conditions is then `≈ −C · (cf_valid − cf_lag)` — **linear in the capacity-factor delta**,
with a per-series constant scale. A per-series booster does not need to know `C`: it discovers the
slope as an ordinary split relationship on the one delta column. That is a far easier learning
problem than the general "how anomalous was this lag" question, and it is exactly the structure
trees are otherwise bad at — a difference of two continuous inputs — served precomputed.

**Compute the delta on the right variable — this is why it depends on the physics proxies.** The
delta is only meaningful on a variable that is roughly linear in the generation it stands for:

- **PV.** A raw GHI delta is a serviceable start (the PV proxy is nearly linear in GHI), so a PV
  variant can run using irradiance directly. The
  [simplified PV power proxy](#linearised-physics-features-for-solar-and-wind) sharpens it —
  especially across the clear-sky-index-interpolated sunrise/sunset ramps and the cell-temperature
  derate — so the proxy delta is the better feature once that item lands.
- **Wind.** A raw wind-*speed* delta is actively misleading, because the power curve is cubic then
  flat: equal speed deltas at 5 m/s and at 15 m/s mean wildly different power deltas. The wind
  variant must take its delta on the
  [farm-level power-curve proxy](#linearised-physics-features-for-solar-and-wind), not on speed,
  and therefore waits for that proxy.

**Where it is weaker than the two-stage residual** — worth stating plainly so the ablation stays
honest. It only compensates for the *weather-linear* component: nonlinear, asymmetric
temperature-driven demand response and all calendar effects are outside it, whereas the residual
baseline captures both because it is a full fitted model. It also arrives unnormalised (no
per-series spread units), and it carries a confounder — a big GHI delta *should* be nearly ignored
at a substation with little embedded PV, so the model must learn per series how much of its
metered signal responds. That is fine for today's per-series boosters, but a
[global model](#global-model-per-time_series_type) would need an embedded-capacity estimate as an
interacting feature, which is the bridge to the V2
[disaggregation](disaggregation.md) work.

**Caveats carried over unchanged.** The weather/proxy value *at the lagged time* rides the same
freshest-NWP-run join with no publication-time cut that aligned lagged weather and the residual-lag
features both flag — leak-free today only as a side effect of daily run cadence — so this item
shares the availability cut planned in
[#356](https://github.com/openclimatefix/nged-substation-forecast/issues/356). Null the delta
wherever its paired power lag is nulled (`_nullify_leaky_lags`): a "conditions changed" signal with
no surviving anchor for what they changed *from* is noise. And plot every delta against observed
power before trusting it — a flipped sign convention (valid − lag vs lag − valid) is invisible to
the leaderboard but obvious in the
[feature-visualisation tool](https://github.com/openclimatefix/nged-substation-forecast/issues/359).

**Sequencing.** This is a new arm in the aligned-weather → residual-lag ablation ladder: (a)
aligned raw lagged weather, (b) these proxy deltas, (c) full two-stage residuals. Arm (b) minus
arm (a) measures what the precomputed subtraction is worth; if it captures most of arm (c)'s gain
on non-switching rows, that is a cheap and important discovery to make before the two-pass
baseline machinery is built. It is slightly more than the pure config of aligned weather — a
derived delta column in `_parsed_features.py`, in the mould of the existing derived features — so
it sits at the bottom of Tier 1. The PV variant can run now on a raw GHI delta; the wind variant
waits on the [solar/wind physics proxies](#linearised-physics-features-for-solar-and-wind).

## Tier 2 — cheap feature engineering (about a day each)

### Per-`time_series_type` feature lists

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
persist the mapping in `meta.json`. The value compounds as the later feature-engineering items (holidays, effective temperature, the solar/wind physics proxies) diverge the useful
per-type sets (solar features for PV, turbine features for wind, holidays for demand).

### Training-data hygiene, the cheap version

Full data cleaning is roadmap v0.4, but training on stuck meters and false zeros actively
teaches the model wrong targets *today* (quality issues are ~10%+ of some series). Cheap
interim: drop training rows whose target sits inside a detected stuck window (rolling std ≈ 0;
`DataQualitySettings.stuck_std_threshold` already exists) or an isolated exact-zero run.
Cleaning only the *training* target is much lower-risk than cleaning delivered data, and it
protects every subsequent experiment from learning artefacts.

### Effective (smoothed) temperature and degree-day features

GB demand responds to *lagged* temperature (building thermal inertia), not instantaneous —
National Grid's demand models use an exponentially-smoothed "effective temperature". Add an
EWM of `temperature_2m` over the past ~1–3 days (computed from the NWP trajectory itself, so
horizon-safe) plus heating-degree `max(15.5 − T, 0)`. Linearises the demand–temperature
relationship the same way the solar/wind physics proxies do for generation.

### Linearised physics features for solar and wind

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
  physics' own $dP/dv$ — no closed form removes that; training on more ensemble members is what addresses it.)

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
per-`time_series_type` feature lists.

### Monotone constraints for the generation models

XGBoost's `monotone_constraints`: PV power non-decreasing in irradiance, wind power monotone
in speed below rated. Mostly buys sane extrapolation in weather regimes the training year
never saw — precisely the failure mode of a single-fold training set. A config-field addition
once the solar/wind physics features exist.

## Tier 3 — new feature machinery (days)

### Init-time-anchored features (current-level anchor; prerequisite for the global model)

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
recency sample weights of the early-stopping item; whichever lands second should expect a smaller measured win.
Where they become *structurally essential* is the global model, whose booster cannot
bake in per-series level. So: a moderate expected win now, and a hard prerequisite later.
**Must ship with a leakage test** in the spirit of the existing `_nullify_leaky_lags` tests
before it's trusted.

### Neighbouring-H3-cell weather context

Each series currently gets its nearest NWP cell only. Add the mean and gradient across the
neighbouring ring (~9 extra columns) for frontal-timing and wind-ramp information. Modest
expected gain, cheap given the `geo` H3 machinery exists.

### Weather-abnormality (climatology z-score) features

Give the booster a sense of whether the *forecast* weather is abnormal — a heatwave, an
unusually warm spring, a storm — by feeding it, per weather variable, a standardised anomaly
`z = (x − μ) / σ` against a climatological norm for that calendar time. This promotes the
[deferred feature-grammar note](#explicitly-deferred-not-quick-or-not-skill)'s
weather-abnormality idea to a concrete first experiment. The inductive-bias case is the one the
[weather-delta features](#weather-delta-compensation-for-power-lags-an-implicit-handle-on-unmetered-generation)
already make: a z-score is a difference of two continuous inputs, exactly the structure trees
are otherwise bad at. A per-series booster could in principle learn "hot for June" from a
`day_of_year × temperature` interaction, but on the 10⁴–10⁵ rows one series provides it mostly
will not, so handing it the precomputed anomaly is legitimate inductive bias rather than
information it already holds.

**Is the anomaly the signal, or is the raw value?** For GB demand the first-order response is to
*actual* (effective) temperature, which the Tier-2
[effective-temperature and degree-day features](#effective-smoothed-temperature-and-degree-day-features)
already capture. Where the anomaly carries genuinely new information is second-order:
acclimatisation (25 °C in May prompts different behaviour than 25 °C in August), heatwave
cooling load that is nonlinear in *how* abnormal the temperature is, and behavioural shifts on
unseasonably nice days. Those effects are real but modest, and partly aliased with features the
model will already have — a raw `day_of_year` ordinal beside temperature lets a tree approximate
crude seasonally-conditional splits. So sequence this *after* effective temperature lands and
treat "anomaly beats raw + calendar" as an explicit ablation, in the same spirit as the
aligned-weather → delta → residual ladder. The one place it may punch above its weight is the
v0.6 stage-1 [switching baseline](switching-events.md#the-baseline-shared-foundation): an
unmodelled heatwave becomes exactly the phantom-event residual that baseline must avoid, and an
anomaly feature gives it a way to explain the excursion away.

**Source: ERA5, not CERRA.** The ingestion-cost objection that defers
[#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167) (CERRA
pre-training) does not bite here: a climatology is a one-off *offline* computation with only a
static lookup at inference, so no live CERRA pipeline is needed. CERRA is still the wrong source,
though, for two reasons of its own. It ends in mid-2021, so the climatology cannot reflect the
recent warm years and every z-score skews warm-positive; and, more importantly, z-scoring ECMWF
ENS *forecasts* against a *different* model's climatology lets model-pair biases contaminate the
anomaly. The clean choice is **ERA5** — same IFS lineage as the ENS forecasts, so those biases
largely cancel; running to near-present; and at 31 km ample resolution, because weather anomalies
are synoptic-scale and a heatwave does not vary meaningfully across an H3 cell. The most
self-consistent source imaginable would be a climatology computed from our own archived ENS, but
a robust day-of-year climatology wants 10+ years and the archive is nowhere near that yet, so
ERA5 wins in practice.

**Design.** Per grid cell, fit μ and σ as smooth functions of day-of-year and hour-of-day — a
low-order harmonic fit or a ±15-day rolling window, because a raw per-calendar-day climatology is
noisy even from 30 years of data — stash the coefficients as a small static Zarr, and emit
`z = (x − μ) / σ` per weather variable. This slots into the `_parsed_features.py` derived-feature
pattern today, and becomes the anomaly-vs-climatology combinator if the composable grammar ever
materialises. Storms mostly do not need it — the
[wind power-curve proxy](#linearised-physics-features-for-solar-and-wind)'s cut-out masking
already encodes "storm" for wind, and raw pressure and wind speed cover the rest — so scope the
first experiment to *temperature* anomaly only.

### Residual lag features from the switching-detector baseline

The full design and caveats live in the switching-events roadmap:
[Approach 1 — the two-stage forecaster](switching-events.md#approach-1-the-two-stage-forecaster).
In brief: fit the v0.6 stage-1 baseline (this same forecaster, configured with weather/calendar
features only and a quantile objective), then feed the production model normalised
"actual − expected" residuals at lag times instead of (or alongside) raw power lags — telling
the model how *normal* each recent observation is, so it can carry a sustained switching-event
offset forward instead of blending it into weather-driven variation. Beyond its expected metric
win, this experiment's result gates the
[decision point](switching-events.md#the-decision-point-a-feature-based-mainline-vs-the-staged-detector)
between a feature-based switching mainline and the staged detector — extra reason to schedule
it.

Four scheduling notes specific to this page:

- **Run [aligned lagged weather](#aligned-lagged-weather-the-single-stage-ablation-control)
  first.** The config-only single-stage variant — aligned lagged-weather features, letting the
  booster judge each lag's normality without an explicit baseline — is this item's ablation
  control: its measured result is the bar the residual features must clear for the two-stage
  machinery to be worth building.
- **The highest-value variant pairs with the init-time-anchored features.** Valid-time-anchored residual lags obey the
  same nullification as raw power lags (any lag ≤ lead time is null), so in the 3–10 day band
  only residuals several days old survive — while the freshest, most informative residual is
  the one from just before forecast time. The strongest form is therefore *init-time-anchored*
  residual features ("normalised residual just before forecast time", "mean residual over the
  24 h before forecast time") — never null at any horizon, and carrying exactly the anomaly
  signal that the init-time-anchored features' raw anchors mix in with ordinary weather-driven level variation. The
  same anchoring extends to the threshold-free *event-age* accumulators from the full design
  (residual EWMAs at a few half-lives, or a self-resetting CUSUM statistic): "how long has this
  series been abnormal" with no hand-coded normality threshold, because trees learn their own
  cutpoints from continuous accumulators.
- **Inspect every feature visually before it enters an experiment.** Residuals, event-age
  accumulators, and neighbour pools are easy to build subtly wrong (sign conventions,
  normalisation, availability cuts) in ways the leaderboard will not surface; plot each one
  against observed power — and the v1 switching-event labels — first. The planned
  feature-visualisation tool
  ([#359](https://github.com/openclimatefix/nged-substation-forecast/issues/359)) is the
  vehicle.
- **It costs more than a config change.** The two-pass pipeline (fit the baseline per CV fold
  on that fold's training period only; hindcast residuals over history, generating the
  booster's *training-row* residuals out-of-sample for the baseline via rolling-origin refits,
  so the booster never calibrates on in-sample residuals it will not see live; join them in as
  features) is new machinery — the hindcast leg should consume the central NWP analysis-proxy
  function planned in
  [#356](https://github.com/openclimatefix/nged-substation-forecast/issues/356), which owns the
  publication-time availability cut the no-lookahead caveat requires. And while the baseline's
  *feature list* and robust median objective are both just config, residual *normalisation*
  needs a per-series spread estimate — from the
  [quantile-objective model family](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
  once it lands, or an interim rolling MAD of the residuals. The
  neighbour-residual variant additionally needs the trial-area adjacency list
  ([switching-events open items](switching-events.md#open-items-dependencies)) and
  cross-series feature engineering — entering as a fixed set of permutation-invariant pooled
  columns (the signed neighbourhood sum and the signed most-anomalous neighbour; see the full
  design), never one column per neighbour; the self-residual version needs neither and should
  run first. Finally, adopting a winner is not free either: the live service must then run the
  baseline model too — a second deployed model plus a hindcast-residual step in the predict
  path.

## Tier 4 — structural model changes (weeks)

### Per-horizon-window models

Issue: [#149](https://github.com/openclimatefix/nged-substation-forecast/issues/149)

One booster per `(time_series_id, horizon_window)` — e.g. 0–2 d, 2–7 d, 7–14 d, configurable.
Train and predict route rows by `nwp_lead_time_hours`; `save`/`load` gain a window dimension.
**Requires [horizon-sliced metrics](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
to evaluate** — its win is by construction horizon-sliced. Under the 3–10 day focus the
interesting experiment is narrower than "many windows": does a dedicated ~3–10 day model beat
the lead-time feature *in that band*? Compare against the lead-time feature first — if the lead-time feature
captures most of the benefit, the extra model count may not pay.

### Batched training via `xgb.DataIter` (enabler)

Issue: [#91](https://github.com/openclimatefix/nged-substation-forecast/issues/91)

Issue #91 already contains a complete, validated implementation design (`LazyFrameBatchIter` +
`QuantileDMatrix`, grouping-agnostic, no temp disk, `train_batch_size` config field) — treat
the issue body as the plan and implement as written. No direct skill gain; unblocks ensemble-member training and the global model.

### Train on more ensemble members (after batched training)

Issue: [#148](https://github.com/openclimatefix/nged-substation-forecast/issues/148)

Training on all 51 members multiplies training data ~51× for correlated rows. Run the
dose-response experiment first: control-only (today) vs ~8 spread members vs all 51 (the NWP
loader already takes `ensemble_members: list[int]`, `cv_assets.py:237`). Training on members
also teaches the model the member-spread input distribution it actually sees at inference —
the train/serve input-skew flagged in the review. That skew grows with lead time: by day 7–10
the control member is an increasingly unrepresentative sample of the ensemble, so the value of
this item concentrates precisely in the primary user band — worth remembering when deciding
how soon to invest in batched training. Ensemble *calibration* itself belongs to
[probabilistic evaluation](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics).
Member training is also one of the
[double-counting mitigations](../techniques/probabilistic-forecasting.md#caveat-double-counting-weather-uncertainty)
for the Phase-D quantile-ensemble pipeline — a second reason to land it, alongside this item
and the lead-time feature, before or with the quantile model family.

### Global model per `time_series_type`

Issue: [#104](https://github.com/openclimatefix/nged-substation-forecast/issues/104)

One booster for all primaries, one for all PV sites, etc. — the biggest potential win for
data-poor series (transfer across sites), and the stepping stone to V2 scale. **Hard
prerequisite: per-series target normalisation** — a global booster mixing a 200 MW GSP with a
5 MW solar farm needs `power / effective_capacity_mw` targets (the `effective_capacity` asset
exists) with the inverse transform at predict time, plus static per-series features (capacity,
type, lat/lon) so the booster can tell sites apart — plus the init-time-anchored features, which supply the current-level signal a global booster cannot bake in per series.
Needs batched training at ensemble scale. The boundary of "quick".

## Explicitly deferred (not quick, or not skill)

- **[#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167) CERRA pre-training** — needs a whole new data-source ingestion (CERRA download,
  contracts, reanalysis-vs-forecast handling) before any training trick. The evaluation design
  belongs to
  [Evaluating new data sources](../ml_experimentation/evaluating-new-data-sources.md).
- **[#176](https://github.com/openclimatefix/nged-substation-forecast/issues/176) local-time power lags** — a DST edge case affecting a handful of half-hours per year;
  the issue itself says it may not be worth worrying about yet. Revisit if the metrics slices
  ever show a DST-transition artefact.
- **A composable feature-expression grammar (consider designing later, deliberately not now).**
  The accumulator machinery of the residual-lag features generalises well beyond power residuals: EWMAs of *any*
  base column at chosen half-lives (an EWMA of temperature *is* the effective-temperature feature), each either lagged in valid time or locked to `power_fcst_init_time` (the init-time-anchored features' anchoring), plus the [weather-abnormality features](#weather-abnormality-climatology-z-score-features) (now a Tier-3 item of their own) — how unusual the forecast weather is against
  the climatological norm for that calendar time ("is this a heat wave?"). Feature names are
  already a tiny parsed language (`ParsedFeatures.from_strings()` turns strings into typed
  `LagFeature`/`RollingFeature`/`WeatherFeature`/... objects), so the natural end state is a
  modestly richer grammar of composable combinators — base column → transform (EWMA,
  anomaly-vs-climatology) → anchoring (valid-time lag vs init-time lock) — still expressed as
  concise strings, e.g. something like `temperature_2m->ewma(3d)@init_time` (illustrative, not
  a design). The payoff couples to
  [#359](https://github.com/openclimatefix/nged-substation-forecast/issues/359): any feature a
  string can express could be tried interactively in the visualisation, then pasted into any
  model config unchanged. Deferred because grammar design done speculatively becomes an inner
  platform — grow combinators only as experiments demand them, and revisit once aligned lagged weather, effective temperature, the weather-abnormality features, the init-time-anchored features, and the residual-lag features have shown which transforms actually earn their keep.

## How each win is evaluated

Each win lands as its own experiment on the leaderboard: register → `full_cv` → `metrics`
(leaderboard scope) → compare NMAE (overall + per-type + per-horizon-slice once horizon
slicing is in, with the 3–10 day band as the headline slice) against the current champion and
the [baselines](metrics-and-leaderboard.md#baseline-forecasters). Keep losing experiments in
MLflow (negative results are results); promote winners' settings into
`conf/model/xgboost.yaml` one at a time so attribution stays clean.
