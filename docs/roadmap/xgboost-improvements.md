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

**A limit worth knowing before you rely on NaN handling.** XGBoost's NaN routing only covers the
missingness patterns present in the training data. Two consequences for the wins below: a model
trained with NWP features does **not** behave like a weather-blind model when NWP vanishes (beating
the incumbent during an outage needs outage-shaped training data, not NaN routing), and the nulls
the de-accumulated ECMWF variables carry are the one case the guarantee genuinely covers. Full
argument:
[Inherent Stability → Default directions, and their limit](../design-philosophy/inherent-stability.md#default-directions-and-their-limit).
Note that the second consequence is narrower than it sounds. Only *leading* nulls reach the model
as nulls: `_upsample_nwp_to_half_hourly` already interpolates interior ones away (see
[the null-filling item](#make-the-existing-nwp-null-filling-deliberate-bounded-and-visible)), and
the scattered per-pixel corruption mostly never becomes a null in the first place, because the
ingest renormalises each H3 cell over the grid points that arrived.

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

**Enter these as features, not as an XGBoost `base_margin`.** It is tempting to treat a physics
proxy as a baseline prediction the booster only has to *correct* — XGBoost's `base_margin` does
exactly that, continuing the boosting on top of a supplied per-row starting score. It is the wrong
tool here, for one structural reason: `base_margin` needs the baseline on the *target's* scale (an
actual MW prediction), but these proxies are deliberately capacity-free, and for most series the
target is *net demand*, where the physics explains only the embedded-generation slice. A
capacity-free, generation-only margin is a poor starting score for the whole signal, and the trees
would spend their capacity undoing its wrong scale. As a *feature*, by contrast, the per-series
booster learns the proxy's slope (implicitly discovering the behind-the-meter capacity) and how it
interacts with the demand component; the [monotone constraints](#monotone-constraints-for-the-generation-models)
below then supply the "trust the physics when extrapolating" property a margin would otherwise
provide. `base_margin` only becomes the right tool once the target is put on the proxy's scale —
which is exactly what the [global model](#global-model-per-time_series_type)'s capacity-factor
normalisation does.

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

Five scheduling notes specific to this page:

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
- **A related variant reuses the same machinery to *correct a draft*, not only to supply residual
  lags.** The stage-1 baseline can be evaluated at the target time to make a first-draft forecast
  that stage 2 then corrects — supplied either as an ordinary feature or as an XGBoost
  `base_margin` — a design axis orthogonal to the residual lags (the lags concern what stage 2
  receives; the draft, what it predicts and starts from) and cheap to add once their per-fold
  out-of-sample hindcast machinery exists. The
  [full treatment](switching-events.md#approach-1-the-two-stage-forecaster) — soft-vs-hard
  corrector, the quantile subtlety, and why the low-variance correction target is a plausible route
  to a *global* model — is on the switching-events page.

### Weather-abnormality (climatology z-score) features

Give the booster a sense of whether the *forecast* weather is abnormal — a heatwave, an
unusually warm spring, a storm, and (in
[the long-window variant](#the-long-window-variant-drought-and-sustained-heat-state) below) a
drought — by feeding it, per weather variable, a standardised anomaly
`z = (x − μ) / σ` against a climatological norm for that calendar time. This promotes the
[deferred feature-grammar note](#explicitly-deferred-not-quick-or-not-skill)'s
weather-abnormality idea to a concrete experiment. The inductive-bias case is the one the
[weather-delta features](#weather-delta-compensation-for-power-lags-an-implicit-handle-on-unmetered-generation)
already make: a z-score is a difference of two continuous inputs, exactly the structure trees
are otherwise bad at. A per-series booster could in principle learn "hot for June" from a
`day_of_year × temperature` interaction, but on the 10⁴–10⁵ rows one series provides it mostly
will not, so handing it the precomputed anomaly is legitimate inductive bias rather than
information it already holds. It sits at the end of Tier 3 because its expected win is modest
(see below) yet it carries a new data-ingestion dependency — more effort per unit skill than the
residual-lag features above it, which are expected to offer far more bang for the buck.

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

**Source: ERA5, for model-consistency with the forecasts.** Choose the climatology source by
asking which one makes the z-score a *clean* anomaly. The input `x` is an ECMWF ENS *forecast*,
so the cleanest baseline shares the ENS's own systematic biases — they then cancel in `(x − μ)`.
That is **ERA5**: same IFS lineage as the ENS, running to near-present, and at 31 km ample
resolution because weather anomalies are synoptic-scale — a heatwave does not vary meaningfully
across an H3 cell. CERRA is the tempting alternative — higher-resolution, and since its 2025
timely-update extension no longer stuck at 2021 but running to within a few months of present —
but it is a *different* model (a HARMONIE-based regional system), so z-scoring ECMWF forecasts
against a CERRA climatology folds a model-pair bias into every anomaly. (This same
model-consistency argument, together with ERA5T's near-real-time latency, is why the project now
[standardises on ERA5](data-sources.md#weather-data) as its single reanalysis for every use —
pre-training, capacity estimation, and this climatology alike; CERRA stays documented as a
higher-resolution option but is deprioritised.) The most self-consistent source imaginable would be
a climatology from our own archived ENS, but a robust day-of-year climatology wants 10+ years and
the archive is nowhere near that yet, so ERA5 wins in practice.

**Storage and ingestion — an H3-indexed Delta table built by a Dagster asset.** Store the
climatology the way the rest of the project stores gridded weather: an **H3-indexed Delta table**
keyed by `(h3_index, day_of_year, half_hour_of_day)` with a mean and standard-deviation column
per weather variable — not a bespoke Zarr. The **mean** grid need not be built from scratch:
Google's **WeatherBench2** publishes a precomputed ERA5 climatology
(`gs://weatherbench2/datasets/era5-hourly-climatology/`) — the smoothed mean by day-of-year and
6-hour, over 1990–2019 with a 61-day window, at ERA5's native 0.25° — so μ can start from that
(regridded to our H3 cells and interpolated from 6-hourly to half-hourly), leaving only **σ** for
us to compute over the same window (WeatherBench2 stores means only, no standard deviation). The
alternative is a single **Dagster asset** that ingests ERA5 (the same shape as the `ecmwf_ens` NWP
ingest, now on the [data-sources roadmap](data-sources.md#weather-data)) and reduces it to both μ
and σ in one pass, fitting each as a smooth function of day-of-year and half-hour-of-day (a
low-order harmonic fit or a ±15-day rolling window, because a raw per-calendar-day climatology is
noisy even from 30 years of data). Either way the ERA5 ingest is the dependency that places this
item late in the tier.

Be precise about the update cadence, though — it is *not* near-real-time. A 30-year climatology
is slowly varying, so the reducing asset recomputes only when a fresh chunk of ERA5 lands
(monthly at most, and even yearly would barely move μ,σ). Nothing in the feature needs live data:
μ and σ for a forecast's valid times — up to 14 days out — are fully determined in advance by the
calendar, and the z-score itself is computed at feature-engineering time in `_parsed_features.py`
from the forecast NWP value minus the climatology lookup. That derived-feature slot is also what
lets this become the anomaly-vs-climatology combinator if the composable grammar ever
materialises. (A *trailing-window* "how unusual versus the last few weeks" anomaly would need
near-real-time ingestion — but that is a different, and weaker, feature than the climatological
norm, and is not what this item builds.)

**Scope the first experiment to temperature.** Storms mostly do not need this — the
[wind power-curve proxy](#linearised-physics-features-for-solar-and-wind)'s cut-out masking
already encodes "storm" for wind, and raw pressure and wind speed cover the rest — so the first
experiment carries a *temperature* anomaly only.

#### The long-window variant: drought and sustained-heat state

Everything above z-scores an *instantaneous* forecast value against its norm for that calendar
moment. A second family of features asks a different question: how abnormal has the recent
*accumulated* weather been? Total precipitation over the past 30 or 90 days as a fraction of the
climatological norm for that window, and the equivalent accumulated-temperature (or degree-day)
anomaly, carry state that no instantaneous feature can hold. Great Britain's summer of 2026 is the
regime in which the two diverge sharply: a single 25 °C afternoon tells the model nothing about
whether the preceding quarter contained
[the driest July England and Wales have recorded since their series began in 1836](../design-philosophy/design-principles.md#input-drift-detection).

This is also the corner of the feature space where the GB brief and the
[India assessment](../architecture/adapting-to-another-geography.md#the-two-pilot-discoms-delhi-and-jaipur)
converge, which is worth knowing before deciding how much to invest here. Everything below is a
mild refinement in a rainy maritime climate and a first-order effect in an arid one — Rajasthan
sits on the margin of the Thar desert, where soiling between monsoons is severe *and* unusually
observable, because a sharp washing signal is what identifies a reversible cleanliness factor at
all. Building these features for Great Britain is therefore cheaper than it looks on the GB
business case alone.

**Where the signal plausibly is**, in roughly descending order of confidence:

- **PV soiling** — the mechanism this project has already worked out in most detail, which is why
  it leads. Dust, pollen and bird droppings accumulate on panel glass and a decent fall of rain
  washes most of it off.
  [Differentiable physics → Soiling](../techniques/differentiable-physics.md#soiling) makes the
  central point for us: Britain's *long-run average* effect is small, but the loss tracks **time
  since the last washing rainfall** rather than any climate mean, so a multi-month dry spell is
  exactly the regime in which it stops being small — and that page says the correction is worth
  adding for Great Britain, not only for dustier climates. The tabular feature is the state
  variable of that model, `d_t`, taken directly: time since precipitation last exceeded a washing
  threshold. Two things follow. It needs **no new data source** — `precipitation_surface` is
  already among the ECMWF ENS variables we download, so the rainfall history sits in the archive
  (though reconstructing a dry spell longer than one 15-day run still means stitching across
  archived runs, and so inherits the availability-cut caveat below) — and it is the one member of
  this family that needs no climatological normalisation at all, because "37 days since washing
  rain" is already interpretable in absolute terms. The
  [assessment of running this codebase over India](../architecture/adapting-to-another-geography.md#the-short-answer)
  reaches the same conclusion from the opposite direction, and is worth reading alongside this
  bullet: it argues that a reversible cleanliness factor is something "we should probably add for
  Britain anyway", precisely because Britain's rainy *average* hides real dry-spell episodes, and
  concludes that work done here would pay off in both countries. Note the division of labour with
  capacity estimation, which absorbs the long-run
  *average* soiling bias into the effective-capacity estimate
  ([honest caveats of the convex route](capacity-estimation.md#honest-caveats-of-the-convex-route)):
  that leaves precisely the time-varying part for a feature to explain, and this is the cheap
  XGBoost-era stand-in for the differentiable-physics treatment.
- **Sustained-heat demand** — the largest case by *magnitude* for the v1 population, because most
  of that population is demand. The Tier-2
  [effective temperature](#effective-smoothed-temperature-and-degree-day-features) smooths over
  roughly 1–3 days, which is building thermal inertia. A multi-week heat regime is a different
  thing: acclimatisation, cooling equipment bought partway through a hot summer and then kept, and
  ground and building-fabric temperatures that a three-day EWM cannot represent.
- **Agricultural irrigation pumping.** Drought raises it, and the trial area sits in the EMids
  licence area, which includes arable Lincolnshire. Treat that second clause as an assumption
  rather than a finding: nothing in the metadata carries a land-use or customer-mix field, so it
  needs confirming against NGED's own customer mix before anyone leans on it. Note how much easier
  the same load is to model elsewhere: the
  [India assessment](../architecture/adapting-to-another-geography.md#the-short-answer) calls
  agricultural pumping "the happier case" there, because Indian agricultural feeders are largely
  segregated and run to a published supply schedule, so a large unmetered load is partly known in
  advance. GB offers no such segregation, which is exactly why this stays an inference from
  weather rather than a measured quantity.
- **Hydro** — physically the cleanest mechanism of the four, and the only one where a 90-day
  rainfall total approaches being a *primary* driver rather than a correction. It is listed last
  anyway, because NGED's network barely has any. It has **no v1 exposure**: `Hydro` is a valid
  `time_series_type` in the contract, but the [32-series trial area](../index.md#scope) contains no
  hydro series. Nor does v2 rescue it. NGED's own
  [Embedded Capacity Register](https://connecteddata.nationalgrid.co.uk/dataset/embedded-capacity-register)
  (May 2026) lists **41 connected hydro sites totalling 25.7 MW** across all four licence areas —
  South Wales 13.7 MW over 10 sites, South West 6.2 MW over 18, East Midlands 5.3 MW over 8, West
  Midlands 0.5 MW over 5. For scale, the same register shows **5,669 MW of connected solar** and
  1,416 MW of wind on that network, so hydro is under half a percent of the embedded solar
  capacity.

    Two details from the register matter more than the headline total. First, it confirms the
    physics is the *right* physics: 39 of the 44 hydro entries are `Hydro - Run of river` and 29
    of the 41 connected sites join at 0.4 kV, so this is overwhelmingly small run-of-river with no
    storage — the most rainfall-sensitive kind there is, output tracking catchment flow almost
    directly. Second, it kills the feature's usefulness for hydro *specifically*: those 41 sites
    are spread across **32 distinct primary substations**, so no primary is hydro-dominated and
    every one of these schemes arrives diluted into a much larger net-demand signal rather than as
    its own series. The largest connected schemes are Llyn Brianne (5.45 MW, Dyfed), Elan Valley
    (4.0 MW, Powys), Chatsworth (3.7 MW, Derbyshire), Mary Tavy (2.6 MW, Devon) and Ystradffin
    (1.99 MW, Dyfed). One entry is much larger — a 58.5 MW Cwm Rheidol scheme accepted to connect
    in the South Wales area — but its target energisation date is 2037, well beyond any horizon
    this roadmap plans for.

**Why this sits behind the instantaneous z-scores, and what the leaderboard will actually tell
you.** The obstacle is not the feature, it is the effective sample size *on the training side*. A
90-day accumulator moves slowly, so the
[leaderboard fold](../ml_experimentation/cross-validation-folds.md#current-state-a-single-fold)'s
training window — 2024-04-01 to 2025-06-30, bounded below by the start of the ENS archive —
contains a *handful* of independent observations of it per series: one summer and one winter, not a
distribution. A per-series booster will still happily split on it, and what it fits will largely be
that particular year's idiosyncrasies, aliased with the seasonal `local_time_of_year_sin`/`_cos`
features and with any level drift over the same months. The instantaneous z-scores do not have this
problem, because a heatwave anomaly recurs many times within a single year.

The *validation* side is the opposite, and it is what makes this worth running rather than merely
worth describing. The same fold validates on 2025-07-01 to 2026-06-30, which contains summer 2025
(the UK's warmest on record) and spring 2026 (the warmest on record for England and Wales). This is
a forward-chained split holding out genuinely extreme regimes, so the leaderboard *can* register a
result here. What it cannot do is separate "the idea is wrong" from "one summer of training data
was not enough to fit it" — so a loss is weak evidence rather than a verdict, and should be
recorded as such in MLflow rather than closing the question.

Two things follow for how to judge it. Expect a small or negative NMAE move, and pair the
leaderboard number with an out-of-band check of whether the model still behaves sensibly when the
accumulator is pushed past the values the training window held — the same "sane extrapolation in
weather regimes the training year never saw" criterion the
[monotone constraints](#monotone-constraints-for-the-generation-models) item is judged on. Note
that this does not trade away [principle 8 ("*every experiment is scored
identically*")](../design-philosophy/design-principles.md#8-every-experiment-is-scored-identically): the
leaderboard measurement is unchanged and stays comparable, and the extrapolation check is an
*additional* acceptance criterion rather than a substitute score. The feature becomes cleanly
measurable only once [ERA5 pre-training](training-history.md) extends the
training history from one summer to several.

**Anchor it to init time, and source it from ERA5.** Compute the accumulator once at
`power_fcst_init_time` and broadcast it across every horizon in the run, exactly like the
[init-time-anchored features](#init-time-anchored-features-current-level-anchor-prerequisite-for-the-global-model)
and for the same reason: an init-time anchor is never leaky and never null at any horizon. (A
target-time-anchored variant, whose window slides with `valid_time`, is a genuinely different
feature — over a 14-day horizon it moves a 30-day window by nearly half its length — and can be
tried separately if the init-time version earns it.) The anchoring also settles where the data
comes from: a window reaching 90 days into the past cannot come from the forecast NWP trajectory at
all, and stitching it together from archived ENS runs would ride the same uncut freshest-run join
the residual-lag features flag. The clean source is the **ERA5 ingest** this item already depends
on, which supplies the accumulation and its climatological norm from one table and one
publication-time availability cut. Nothing here needs live data at forecast time beyond the ~5-day
ERA5T latency, which is immaterial to a 90-day total.

**Scope the first long-window experiment to precipitation.** The accumulated-*heat* variant is
tempting to reach for first as a cheap control, on the grounds that it is just a longer
effective-temperature EWM — but that is wrong, and for the reason this section has already given.
The Tier-2 effective temperature is computed from the NWP trajectory itself, and
`_apply_rolling_mean_feature` groups by `nwp_init_time` precisely so that a window cannot span
runs, so with a 15-day ENS run there is no multi-week EWM to configure: the heat accumulator
carries the same ERA5 dependency as the precipitation one. The genuinely cheap control is to
lengthen the EWM only as far as the trajectory allows (~7–10 days), which is a weaker experiment
and should be labelled as one.

### Make the existing NWP null-filling deliberate, bounded and visible

**Start from what the pipeline already does, which is not what the docs imply.** The three
de-accumulated ECMWF variables carry nulls beyond lead-0 — usually scattered per-pixel,
occasionally a whole `(ensemble_member, valid_time)` slice — and the ingest deliberately lands them
([known issues](../architecture/ecmwf-ens-known-issues.md#nulls-in-the-de-accumulated-variables-tolerated)).
The stated position is to leave them un-imputed:
[Missingness in learned models](../design-philosophy/inherent-stability.md#missingness-in-learned-models)
says of this exact pattern that the main risk is *someone later "fixes" it by imputing*.

But `_upsample_nwp_to_half_hourly` resamples NWP from its native 3- and 6-hourly steps to the
half-hourly grid with `interpolate()`, and Polars' `interpolate()` fills **interior** nulls as a
side effect. So an interior null is *already* filled today, from its temporal neighbours within
the same `(member, cell)` group, silently and unflagged. Only *leading* nulls survive to the model
as nulls, which is why the lead-0 convention holds and why that function's docstring is careful
about it. The "leave them un-imputed" position is therefore true of leading nulls and false of
interior ones, and nobody chose that split — it fell out of an upsampling implementation.

**How much is left to fill is smaller than it used to be, and the remainder is the blocky half.**
The ingest aggregates the 0.25° grid onto H3 cells and renormalises each cell over the grid points
that arrived, so the scattered per-pixel corruption mostly never becomes a null that this
interpolation could reach. What still arrives as an interior null is a whole
`(ensemble_member, valid_time)` slice, or a cell whose every grid point went missing at once. That
narrows this item in two ways worth knowing before starting it: the fill it is bounding is the
long-span, blocky one, which is the case where an unbounded bridge is *least* defensible; and the
population the experiment can measure on is smaller, so a null result will be harder to
distinguish from no effect.

The experiment is therefore not "should we start interpolating?" but **"the interpolation already
happening should be deliberate, bounded and visible"**:

- **Bounded.** `interpolate()` will span an arbitrarily long interior gap. Note that even the
  cheapest case bridges further than it sounds: the fill runs between the steps *either side* of
  the missing one, so losing a single native step is a 6-hour bridge in the 3-hourly part of the
  horizon and a **12-hour** one in the 6-hourly part — and the 6-hourly part is the 3–10 day band
  users act on. Several consecutive missing steps are bridged just as confidently. Cap the span
  and leave anything longer as null.
- **Visible.** Carry an `is_imputed` flag per filled variable so the model can condition on it, and
  so a forecast built on a bridged gap is distinguishable after the fact.
- **Measured.** With the flag in place, compare against today's silent behaviour, and against
  leaving interior nulls unfilled entirely. A null result is worth having: it turns the standing
  position from an assertion into a measured one — and, either way, this replaces an accident with
  a decision.

**Filling within a run is the only fill worth having, which is at least one thing today's accident
gets right.** The obvious alternative is worse. Filling from the *previous* NWP run looks
attractive —
a 24-hour-older forecast of the same target time is a decent estimate — but ECMWF regenerates its
perturbations every cycle, so ensemble member 34 of today's run and member 34 of yesterday's are
not the same trajectory continued; they are two unrelated draws sharing a label. Only member 0,
the unperturbed control, is comparable run to run. A cross-run fill would therefore graft one
weather scenario's irradiance into another scenario's row, breaking the internal coherence that
is the whole point of a per-member row. We do not ingest cloud cover, but it is latent behind
several variables we do carry — short-wave and long-wave radiation respond to the same cloud field
in opposite directions, and precipitation with them — so a cross-run fill of one of them leaves a
row describing two different skies at once. Same-run interpolation has no such problem: the
neighbouring steps belong to the same trajectory, so coherence survives, and no row ends up
mixing NWP lead times either.

Mechanics: the change lands in `_upsample_nwp_to_half_hourly`, which is where the interpolation
already happens — not as a new pass elsewhere, which would leave two fills to reason about. It
must stay lazy, per
[principle 11](../design-philosophy/design-principles.md#11-push-the-work-down-to-the-engine-materialise-once-as-late-as-possible),
and it must stay out of the ingest: imputed values in the NWP Delta table would destroy the
provenance
[principle 9](../design-philosophy/design-principles.md#9-provenance-travels-with-the-forecast-data)
guarantees. The gap bound is the substantive part — `interpolate()` has no notion of how far it is
reaching, so the bound has to be imposed around it: identify runs of consecutive nulls (a run-length
id over `valid_time` within each group), interpolate, then restore the nulls wherever the run was
too long. **Express the bound in hours, not in rows or steps**, and measure it as the span between
the bracketing non-null values: a "3 steps" bound means 9 hours early in the horizon and 18 hours
late in it, and counting *rows* is worse still because the interpolation runs after upsampling, so
a row is half an hour rather than a native step. Hours are the only unit that means the same thing
across the horizon. The `is_imputed` flags are new `AllFeatures` columns, and that schema change —
plus a bounded, still-lazy interpolation — is why this sits in this tier rather than among the
day-scale items.

Two caveats shape what a win would mean. The de-accumulated variables are *rates over the
preceding step*, so interpolating them across a 6-hourly gap is a coarser approximation than it
would be for an instantaneous field — and the 6-hourly steps are exactly the 3–10 day band users
act on. Second, whatever is decided must apply identically in training and at inference, or the
change buys a train/serve skew — the failure mode the NaN-handling limit at the top of this page
warns about — in exchange for the one it fixes. Read the horizon slices either way.

## Tier 4 — structural model changes (weeks)

### Pre-train on the ERA5-backed history

Issues: [#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143) (ingest),
[#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167) (experiments)

Our power data reaches back to late 2019 but our ECMWF ENS archive starts 2024-04-01, so today's
fold trains on 15 months and one winter. Ingesting ERA5 and pre-training on 2020–2023 takes that to
roughly 5.5 years, which is what makes the seasonal items on this page cleanly measurable — the
[long-window accumulators](#the-long-window-variant-drought-and-sustained-heat-state) above all,
and secondarily the holiday, monotone-constraint and global-model items, whose value all turns on
seasonal or regime coverage the current window does not have. The design, the era-confounding
hazard that dictates the ingest's scope, and the COVID covariate are on
[Extending the training history](training-history.md).

Two sequencing notes. The [lead-time feature](#feed-the-model-the-forecast-lead-time-review-discovery-one-line)
is a prerequisite, because the cheapest reconciliation arm leans on it to discount reanalysis
weather. And the data-hungry items below — batched training, ensemble-member training, the global
model — are worth running *after* the history lands, since that is where four extra years change
the answer most.

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

### Ensemble *statistics* as features, instead of member-by-member rows

The fork in the road that the item above assumes away. Today every ensemble member is its own
row: `AllFeatures`' primary key is
`(time_series_id, power_fcst_init_time, valid_time, ensemble_member)`, and a member is pushed
through the model one at a time. The alternative is to collapse the member axis at feature-build
time, so each `(time_series_id, valid_time)` gets **one** row whose weather columns are
*statistics over the members* — and then predict once.

**Use quantiles, not mean and standard deviation.** Per variable, something like p10/p25/p50/p75/p90,
as *columns* on that one row. Mean-and-spread implicitly assumes a near-Gaussian member
distribution, which the cloud field driving irradiance badly violates: "half the members are
overcast and half are clear" is a common and highly consequential state whose mean describes no
member at all. Quantiles keep the marginal shape, including skew and some of the bimodality, for
the price of a few more columns.

**A smaller bet, worth pricing separately: shrink the member axis instead of removing it.** The
collapse above is 51× fewer rows *however many* statistics ride on each one — quantile levels are
columns, not rows, so preferring quantiles to mean-and-spread does not make the bet any smaller.
What does make it smaller is keeping one row per scenario and using fewer scenarios: subsample a
handful of representative members, or cluster the ensemble and take one member per cluster. That
is a ~5–10× row reduction rather than 51×, and it is *qualitatively* different from the collapse,
because a subsampled member is still a physically coherent trajectory — so the mixture pooling
discussed below still works, and none of the coherence disadvantages apply. If the aim is mostly
cost, try this first; the full collapse is only worth its disadvantages if what you want is
spread-as-a-feature or the resilience property.

**Advantages.**

- **Resilience, by construction.** This is the strongest practical argument. A statistic is
  computed over whatever members actually arrived, so a missing ensemble member shifts a quantile
  slightly instead of removing rows, and a variable that is null for a few members degrades the
  estimate instead of nulling whole rows. The
  [2026-08-09 incident](../architecture/ecmwf-ens-known-issues.md#nulls-in-the-de-accumulated-variables-tolerated)
  — one member's radiation missing at two lead times — would not have been an event at all. It
  also makes the [incomplete-run](../architecture/ecmwf-ens-known-issues.md#an-incomplete-run-tolerated-and-reported)
  warning much less consequential.
- **Cost and scale.** Up to 51× fewer feature rows: the ~321M-row validation prediction, the
  memory ceiling that shapes input pruning and `init_time` chunking, and the
  [32-bit row-index ceiling](../architecture/performance.md#the-other-hard-ceiling-polars-32-bit-row-index)
  all get dramatically easier at once, and inference gets ~51× cheaper. That is direct support for
  [principle 6](../design-philosophy/design-principles.md#6-the-whole-system-must-be-exercisable-on-one-laptop)
  at V2 scale, where it is the claim most at risk.
- **Spread becomes an explicit input.** The model can learn that a wide member spread means a
  less certain forecast, which is the natural feed for the
  [band-widening](../design-philosophy/inherent-stability.md#widening-bands-the-in-band-signal)
  that is designed but not built.
- **No ensemble-member identity problem.** A quantile is identity-free, so it is comparable
  across runs in a way a member index is not — which is what would make a cross-run fill
  defensible (see
  [the null-filling item](#make-the-existing-nwp-null-filling-deliberate-bounded-and-visible)).

**Disadvantages.**

- **It gives up the weather-versus-model uncertainty decomposition.** This is the serious one, and
  it is a decision about the *product*, not just the model. The planned probabilistic design is a
  [mixture of conditional distributions](../techniques/probabilistic-forecasting.md#the-fix-formally-a-mixture-of-conditional-distributions):
  each member yields a conditional distribution $F_m$ ("how uncertain is power *given* this
  weather story"), and the disagreement *between* members carries the weather uncertainty. Pool
  them and both survive. Collapse the member axis and there is no $F_m$, so there is no linear
  pool and no source for
  [Representation 3](delivery-tables.md#representation-3-ensemble-of-percentile-forecasts) — the
  ensemble-of-percentiles delivery table. A quantile-regression model fed weather quantiles still
  produces a predictive distribution, but it is a single conflated one: it cannot answer "is this
  forecast uncertain because the weather is uncertain, or because the model is?" — the question
  behind a control-room user asking whether to wait for tomorrow's run.
    - The honest counterweight: that conflation removes the
      [double-counting](../techniques/probabilistic-forecasting.md#caveat-double-counting-weather-uncertainty)
      risk by construction, so the collapsed model might be *better calibrated* while being less
      informative. Calibration and attribution are genuinely different goods here.
- **Aggregating in weather space rather than power space.** Substation net load is a strongly
  non-linear function of irradiance and wind speed (PV clipping, the cubic turbine ramp), so
  $\mathbb{E}[f(x)] \neq f(\mathbb{E}[x])$. Pushing each member through the model and combining in
  *power* space is the correct order of operations, and it is what the current design does.
- **Per-variable quantiles destroy cross-variable coherence.** A sharper form of the point above:
  the p90 of irradiance and the p90 of temperature need not co-occur in any single member, so a
  quantile row can describe a physically impossible joint state. A member row cannot. Mitigations
  exist (member-rank-based statistics, or quantiles of a derived physics proxy rather than of each
  raw variable), and testing one is part of the experiment.
- **51× less training data.** The mirror image of the cost win, and in direct tension with
  [#148](https://github.com/openclimatefix/nged-substation-forecast/issues/148) above, whose whole
  argument is that member rows multiply the training set. The two items are alternatives at the
  same fork, so decide them together rather than in sequence.
- **It moves the `AllFeatures` primary key**, and therefore touches cross-validation, metrics and
  the leaderboard.
  [Principle 8](../design-philosophy/design-principles.md#8-every-experiment-is-scored-identically)
  means it must be scored against the existing board by a controlled comparison, not swapped in.

**How to evaluate.** One registered experiment against the member-by-member champion on the same
folds and population, with four arms: quantile features, representative-member subsampling,
control-member only (today), and all-member training. Running the subsampling arm alongside the
collapse is what separates "the cost saving was the win" from "the member axis was carrying
information", which the two-arm version cannot distinguish.
Headline NMAE sliced by horizon and `time_series_type` as usual, but this item also needs the
[probabilistic metrics](../techniques/evaluation-metrics.md#probabilistic-metrics) — spread-skill
ratio and PICP — because the thing being traded away is uncertainty structure, which NMAE cannot
see. A result where quantile features match on NMAE and lose on spread-skill is the outcome that
tells you the decomposition was doing real work.

### Global model per `time_series_type`

Issue: [#104](https://github.com/openclimatefix/nged-substation-forecast/issues/104)

One booster for all primaries, one for all PV sites, etc. — the biggest potential win for
data-poor series (transfer across sites), and the stepping stone to V2 scale. **Hard
prerequisite: per-series target normalisation** — a global booster mixing a 200 MW GSP with a
5 MW solar farm needs `power / effective_capacity_mw` targets (the `effective_capacity` asset
exists) with the inverse transform at predict time, plus static per-series features (capacity,
type, lat/lon) so the booster can tell sites apart — plus the init-time-anchored features, which supply the current-level signal a global booster cannot bake in per series.
Needs batched training at ensemble scale. The boundary of "quick".

Normalisation also unlocks `base_margin` for the generation types. Once a series' target is a
capacity factor in $[0, 1]$, its
[wind/PV physics proxy](#linearised-physics-features-for-solar-and-wind) can be put on that same
scale — the [wind proxy](#linearised-physics-features-for-solar-and-wind) already runs 0→1 (the
normalised cubic ramp), while the PV proxy is in irradiance units ($\mathrm{W/m^2}$) and needs one
extra division by a reference irradiance (≈ 1000 $\mathrm{W/m^2}$ at STC) to become a capacity
factor. On a matched scale a single-`time_series_type` generation booster can take the proxy as its
`base_margin` and learn only the site-specific deviation — physics carrying the cross-site shape,
trees the correction. This is the scale match the per-series net-demand
models lack — there the capacity-free proxy is the wrong tool as a margin (the
[physics-features section](#linearised-physics-features-for-solar-and-wind) explains why), and
capacity-factor normalisation is what supplies it. It is also the same base-margin move the
[two-stage forecaster](switching-events.md#approach-1-the-two-stage-forecaster) makes with its
stage-1 draft; there, the correction target's low variance and cross-series stationarity are part
of what makes a *global* corrector tractable at all — the same property that helps here. (Under a
log-link generation objective the margin would be `log(proxy)`, which needs a floor to handle the
PV proxy's exact zeros at night.)

## Explicitly deferred (not quick, or not skill)

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
