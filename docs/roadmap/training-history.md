# Extending the training history

> **Status: 🚧 Planned (v0.5).** Epic:
> [#145](https://github.com/openclimatefix/nged-substation-forecast/issues/145). Ingest:
> [#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143); pre-training
> experiments: [#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167); a
> weather product with a few months of history: 🔬 study
> [#999](https://github.com/openclimatefix/nged-substation-forecast/issues/999).

Our ECMWF ENS archive starts 2024-04-01; most trial-area power series go back to late 2019. An
estimate of past weather covers the gap. [ERA5](data-sources.md#weather-data) is the estimate
planned for ingest, and shares the ENS's IFS lineage, so the reanalysis-to-forecast domain shift is
smaller than a different-model reanalysis would give. Which estimate of past weather to train on,
per weather variable, is an [open question](#open-questions). Most of this page names ERA5 because
ERA5 is the planned ingest.

Almost all the cost is in the data layer. Once ERA5 and the paired residual statistics exist, moving
between the variants below is mostly configuration — so they are leaderboard arms, not decisions to
make up front.

## Scope the ingest to include the 2024+ overlap

**Fetch ERA5 across the 2024+ ENS overlap as well as the 2020–2023 gap.** This is the one design
decision that changes what the ingest builds, and it exists to avoid *era confounding*: a feature
value that occurs in only one time period is learned as a proxy for that period, not for the
underlying quantity we mean by it. If NaN-NWP appears only before 2024, "weather is missing" becomes
a perfect proxy for "2020–2023" — a demand regime carrying COVID distortions, less embedded PV, and
lower EV and heat-pump penetration. So a 2027 feed failure would be forecast from a 2021 regime. The
same trap applies to a source flag, to a lead-time-zero encoding, and to the ENS-only spread and
quantile columns, which have no ERA5 equivalent.

The general rule: **an era covariate is safe exactly when the value production will see is
well-represented in the modern era.** So, alongside the overlap fetch, randomly mask NWP features
and spread/quantile columns on a subset of 2024+ rows, as a configurable augmentation step.

The overlap has a second payoff: paired ERA5 and ENS on identical target times, which turns the
reconciliation question below into estimation rather than guesswork.

## Reconciling ERA5 with ENS

- **Lead-time-zero framing.** Do not degrade; treat ERA5 as a forecast at lead zero and let the
  lead-time feature carry the discounting. That framing separates the physical weather-to-power
  response (genuinely lead-time-invariant) from how far to trust it as forecast error grows.
  Cheapest arm and the right first run, but note the tension: pre-2024 rows then carry only lead
  zero, so every split on lead time partitions the modern rows off beneath it, and the extra history
  reaches the 3–10 day band only through the structure above such splits and in the trees that never
  make one. Boosting shares more across trees than that phrasing might suggest, so this is a
  weakening rather than a wall — but expect the win at short leads unless the invariance assumption
  holds strongly.

- **Degrade ERA5 towards ENS error statistics.** Fit the `ENS − ERA5` residual distribution per
  variable, per lead time, per season on the overlap, then sample from it when synthesising pre-2024
  features. Quantile mapping per horizon is the cheap version and is probably enough for
  temperature. Degrading towards ENS error statistics is not merely an alternative to lead-zero
  framing: it is what makes the extra history populate the long leads at all.

- **ENS reforecasts — considered and rejected.** Under Cycle 49r1 the medium-range reforecasts run
  over the past 20 years with an 11-member ensemble, so they are real forecasts with real lead-time
  error and the mismatch would largely disappear rather than needing correction. **We are not going
  to do this**: the only access is MARS, and the download would take far too long. Recorded so it is
  not re-litigated.

- **A frozen weather-response model plus a small calibrator per weather product.** Train a lag-free
  XGBoost model on the whole power history and an estimate of past weather, then fit a small
  calibrator on each weather product's overlap with power. Of the four reconciliations here, the
  frozen weather-response model is the design built to let a weather product with a few months of
  history into the forecast. The design is under research: see [A new weather product
  with a few months of history](#a-new-weather-product-with-a-few-months-of-history).

## ERA5 splits one horizon into two

Today `nwp_lead_time_hours` (how old the weather is) and the forecast horizon (which power lags are
available) differ only by the constant `NWP_PUBLICATION_DELAY_HOURS`, so one column carries both.
ERA5 decouples them: weather age is zero. But power-lag availability must still mirror production,
or pre-training teaches the model to lean on lags that vanish at serve time. Pre-training rows
therefore need a *sampled* pseudo-horizon driving `_nullify_leaky_lags`, carried separately from
weather age.

## Single pool vs two-phase warm start

- **Single pool, with per-era sample weights.** All rows in one training set, and the weight on
  pre-2024 rows becomes a tunable hyperparameter rather than a yes/no decision. Run this first: it
  is the cheap form of the mixed warm start below, and recency sample weights are already a Tier-1
  item on the [XGBoost
  improvements](xgboost-improvements.md#early-stopping-instead-of-fixed-n_estimators500) page.

- **Two-phase warm start.** Train on the ERA5 history, then continue boosting on 2024+ ENS data.
  Warm start only *adds* trees, so the correcting trees see roughly 2 years and very few examples of
  each season. And if phase one over-trusts weather, shrinking an over-confident component
  additively is harder than never building it. **Mixed phase two** — keeping down-weighted (and
  possibly degraded) ERA5 rows in phase two — is the middle path.

- **The source flag follows from that choice, not the other way round.** In a *pure* phase two the
  flag has zero variance and XGBoost can never split on it, so drop it there. Under a single pool or
  a mixed phase two it has variance and earns its place — but only because the overlap fetch
  decorrelates it from date.

## A new weather product with a few months of history

> **Status: 🔬 Research.** Study:
> [#999](https://github.com/openclimatefix/nged-substation-forecast/issues/999), under the
> Studies epic [#896](https://github.com/openclimatefix/nged-substation-forecast/issues/896). A
> design that wins the study is ported to production only if its results show promise.

**A new weather product usually arrives with months of archive, while the power history runs from
late 2019, so an XGBoost model trained only on the new product's archive discards years of power
data.** ECMWF ENS's archive starts 2024-04-01, AIFS-ENS's 2025-07-02, WeatherNext 3's 2026-01-01,
and Dynamical.org's ICON-EU's in early 2026 (dates from the [weather products
survey](../background/weather-products-survey.md#forecast-models)). The traditional approach needs
years of overlap between every product and the power data. The published comparisons of
post-processing methods we have read also assume a fixed product with ample history.

**Most of this roadmap was written before autonomous forecasting experiments proved practical, which
is why the older pages are cautious about adding weather products.** Claude Code runs whole
forecasting experiments autonomously, including downloading new datasets. That research writes its
code under `studies/` and `packages/studies/`, which no human reviews line by line, so it can test
many ideas quickly. The autonomous sweep waits on the protected scorer
([#958](https://github.com/openclimatefix/nged-substation-forecast/issues/958)). Production code
never imports from either directory, and production code is carefully reviewed. An idea here
therefore moves in three steps: a broad autonomous sweep to narrow the search space, then supervised
narrow sweeps if the results look promising, then a reviewed port of the study code to production if
the results still look promising.

**The objective is to use as many weather forecasts as possible, on the belief that more forecasts
make better wind and solar power forecasts.** Three criteria score a design against that objective:

- **Learning curve:** the forecast's skill against the number of months of the new product's
  history, and the month at which the new product first adds skill.
- **Marginal effort of adding a product:** fit one small component, with no retrain of the
  weather-response model below and no change to the other products.
- **Safe entry:** a product with no history gets zero weight, a product with a little history gets
  a small weight, and the forecast never degrades when a product is missing, as the
  [inherent-stability rules](../design-philosophy/inherent-stability.md) require.

**The project's own evidence supports the belief for wind and not yet for solar.** In the
[matched-lead
study](../studies/forecasts/matched-lead.md#a-blend-lowers-the-wind-error-at-both-leads-tested-and-the-solar-error-only-at-an-optimistic-lead),
adding ICON-EU and IFS 0.25° forecasts to ECMWF ENS lowered the wind error by 0.184 points of
capacity [0.087, 0.282] at the conservative lead, and gave no detectable solar gain (−0.033 points
[−0.106, +0.033]). That study trained an XGBoost model on 21 months of every product, which is the
traditional approach. A study of the learning curve has to allow a null result for solar.

### A weather-response model and a product calibrator

**The design trains one weather-response model on the long power history, then fits a small product
calibrator for each weather product on that product's short overlap with power.** The
weather-response model is a lag-free XGBoost model that maps weather, solar geometry, and calendar
features to power. The weather-response model trains on power from late 2019 onwards, with an
estimate of past weather as its input rather than any forecast. The product calibrator adapts the
weather-response model's output to one weather product and that product's lead times. The
weather-response model's output at the target time plays the role the draft plays in the
[switching-events two-stage
forecaster](switching-events.md#a-second-way-to-use-the-stage-1-baseline-correct-a-draft).

**"Estimates of past weather" is this page's umbrella term for every source describing weather that
has already happened.** The sources are ERA5, CAMS, the first time steps of NWP runs (the early
leads of CEDA's UKV archive, for example), weather-station observations, weather satellites, and
weather inferred from the differentiable-physics (DP) models of wind and solar farms. Each source
has errors of its own, so none of them is exact.

**The split is the weather-forecasting literature's "perfect prognosis" approach, and Canada's
Updateable MOS system was built for the new-model case.** Perfect prognosis fits a statistical
model on observed or analysed weather and applies that model to forecast weather; model output
statistics (MOS) fits on forecast weather directly. [Marzban, Sandgathe and Kalnay
(2006)](https://doi.org/10.1175/MWR3088.1) show from a formal analysis that MOS should beat perfect
prognosis on mean squared error, which is the case for fitting a calibrator on forecast weather
rather than serving the weather-response model alone. [Wilson and Vallée
(2002)](https://doi.org/10.1175/1520-0434%282002%29017%3C0206%3ATCUMOS%3E2.0.CO%3B2) built
Updateable MOS because each change of the Canadian weather model left too little archive for stable
MOS equations. Updateable MOS develops its equations from a weighted blend of old-model and
new-model data, weighted towards the new model while keeping enough old-model data for stable
equations. That weighting is the safe-entry criterion above, applied to a weather model's upgrade
rather than to a new product.

### The calibrator supplies the hedge that the weather-response model lacks

**The weather-response model predicts power given the weather that happened, but a forecast needs
power given the weather forecast, and the calibrator supplies the difference.** An XGBoost model
trained on forecast weather learns to hedge, damping its sensitivity to weather where the forecast
is often wrong ([why a model trained on past weather does not
hedge](../techniques/probabilistic-forecasting.md#a-model-trained-on-past-weather-does-not-hedge)).
The weather-response model never sees forecast error, so fed forecast weather it is over-sensitive,
more so at long lead. Averaging the weather-response model's output over calibrated ensemble members
approximates the hedge, and so does a recalibration fitted on the product's overlap with power.

**The calibrator's parameter count is limited by the number of independent weather episodes in the
overlap, not by the number of rows.** One weather system covers every site at once, so hundreds of
sites during one storm give roughly one storm's worth of evidence. As a judgement rather than a
measurement, one independent episode every two to three days gives about 150 to 230 episodes in 15
months and 30 to 45 in a three-month winter. The calibrator therefore keeps few free parameters:

- parameters that vary smoothly with lead time, rather than one set per lead;
- per-series effects shrunk towards a pool per technology;
- a new product's parameters written as departures from the ECMWF ENS calibrator's parameters; and
- no seasonal term until the overlap spans two winters.

### The calibrator can act on power or on weather

**The power-space calibrator recalibrates the power ensemble's distribution, and the study runs it
first.** The power-space calibrator follows ensemble model output statistics (EMOS), works in
capacity-factor units, pools series of one technology, and censors its predictive distribution at 0
and 1. [Phipps et al. (2022)](https://doi.org/10.1002/we.2736) compared EMOS applied to weather
ensembles, to wind power ensembles, and to both. Phipps et al. found that post-processing the power
ensemble improved calibration and sharpness, while post-processing only the weather ensemble did not
necessarily help. Phipps et al. is the one study we have found, and covers wind only. An affine map
of power cannot reproduce the hedge where the power curve bends — wind near cut-in and rated speed,
and PV at the export cap and the inverter clipping limit — which is the case for the weather-space
arm below.

**The power-space calibrator plugs into one modular component, the wrapper forecaster, shared by
three consumers.** [Phase
C](metrics-and-leaderboard.md#phase-c-low-effort-calibration-after-b-proves-the-diagnosis) of the
probabilistic metrics plan builds an EMOS-style calibration as a wrapper forecaster. The [calibrated
manual
heuristic](metrics-and-leaderboard.md#calibrating-the-manual-heuristic-aims-at-the-95th-percentile)
([#715](https://github.com/openclimatefix/nged-substation-forecast/issues/715)) and the
degradation-conditional conformal calibration
([#443](https://github.com/openclimatefix/nged-substation-forecast/issues/443)) are planned to share
that wrapper, and the product calibrator would be the third consumer. The wrapper would live in
`ml_core` as a `BaseForecaster` subclass that wraps another `BaseForecaster`, so the serving path
stays "load a model, call `predict`".

**The weather-space arm runs in two stages: a quick arm with XGBoost in the study, and an arm built
on DP models after v2.**

- **In the study:** fit a handful of weather-space parameters per variable — for example a wind
  speed scale and offset, each smooth in lead time — directly on power error through the frozen
  XGBoost weather-response model. A grid search or Nelder-Mead fits them without gradients, which
  matters because an XGBoost model's output is piecewise constant in its inputs, so its gradient
  is zero almost everywhere. A map with an offset, a scale on the ensemble mean, and a positive
  scale on each member's departure from that mean keeps each member's rank and joint
  structure across sites, lead times, and variables, so an ensemble product needs no copula step.
  A deterministic product such as ICON-EU has no members, so its calibrator supplies the whole
  weather-uncertainty term and borrows the spread's lead-time shape from ECMWF ENS.
- **After v2:** repeat the weather-space experiments through the [DP
  models](../techniques/differentiable-physics.md) of wind and solar farms. The DP models are
  differentiable, so the weather-space parameters can be fitted by gradient descent, and there can
  be many more of them. The [weather encoder's multi-source
  blending](../techniques/encoders.md#blending-several-nwp-sources) is the fullest form of this
  arm.

### Which estimate of past weather to train on

**Which estimate of past weather the weather-response model trains on is an open question, and the
past-weather studies find ERA5 is not the best estimate for sunshine or for wind at the trial area's
farms.** Given ERA5, an XGBoost model's error on past sunshine is 9.08% of capacity on the solar
study's main row set, and given CAMS it is 5.09%
([solar](../studies/past-weather/solar.md#cams-describes-past-sunshine-best-of-the-eight-main-row-set-products-by-a-wide-margin)).
At three wind farms, the error given UKV's T+0 wind as Open-Meteo serves it is 0.44 points [0.24,
0.63] lower than given ERA5's wind
([wind](../studies/past-weather/wind.md#ukv-and-icon-d2-describe-past-wind-best-of-the-five-products-tested)).
The working guess is CAMS for irradiance and the early time steps of CEDA's UKV archive for the
other variables, each tested against ERA5.

**The best estimate of past weather may be a blend of several products rather than any one
product.** In the [blending study](../studies/past-weather/blending.md), an XGBoost model given all
six solar products beats an XGBoost model given CAMS, with CAMS's neighbouring hours and its
beam/diffuse split, by 0.13 points [0.10, 0.17]; CAMS with ICON-EU gives most of that gain, 0.10
points [0.07, 0.14]. On rows from January 2021, adding SARAH-3's satellite irradiance to CAMS's
split beats the same CAMS model without SARAH-3 by 0.18 points [0.16, 0.21]. For wind, an XGBoost
model given all five wind products beats an XGBoost model given UKV with its neighbouring hours by
0.48 points [0.40, 0.56], and UKV with ICON-EU beats the same UKV model by 0.26 points [0.21, 0.31].
A control that keeps the extra columns but shuffles their weather does no better than the single
product, so the gain comes from the other products' hour-by-hour weather. Weather stations were not
part of the blending study; in the [wind
study](../studies/past-weather/wind.md#one-nearby-10-m-weather-station-trails-era5s-10-m-wind-on-its-own-and-lowers-ukvs-error-when-added-to-it),
one nearby 10 m station trails ERA5's 10 m wind on its own and lowers UKV's error when added to UKV.
Three caveats limit what the blending results say about training the weather-response model:

- **The blends are learned in power space.** Each blend is an XGBoost model given several products'
  columns and predicting power, not a blended weather field. A blend can become the
  weather-response model's input in two ways: feed every product's columns into the
  weather-response model, so every product the blend uses must be present at serve time or be
  replaced by the product calibrator's output; or build a weather-space blend fitted against some
  other estimate of past weather.
- **ICON-EU's history is too short for the whole power history.** The study's ICON-EU came from
  Open-Meteo and starts in November 2022, so a blend with ICON-EU cannot cover the power history
  back to late 2019. CAMS with SARAH-3, and CAMS with ERA5, can.
- **All four headline comparisons are post hoc**, chosen after a first run's results, as the study
  page says.

**CEDA's UKV archive carries three caveats as a training input.**

- CEDA's UKV is statistically different from the UKV the Met Office serves live. A calibrator can
  absorb that difference only if the live service does not also serve UKV.
- The Met Office's PS47 upgrade on 2026-01-21 changed UKV's microphysics and cloud scheme, so a
  UKV series spanning that date is not homogeneous.
- The UKV fields fetched from CEDA so far hold 10 m wind and winds at 1000 hPa and 925 hPa, with
  no 100 m wind and no orography. Estimating hub-height wind from those levels needs orography,
  and the 1000 hPa level lies below ground in deep lows, which are the windy days.

**Weather inferred from the DP models stays an estimate of past weather, with a circularity that
bites in training and not at test time.**

- In the training set, the circularity is real: the DP models are trained on the same estimates of
  past weather that the inversion would then estimate from power.
- At test time, a wind or solar farm is a genuine weather observation. A farm's recent output,
  inverted through the farm's DP model, observes the weather at that site now and in the recent
  past, seen through the DP model and inheriting that model's biases. The inverted weather observes
  no future lead time, so it acts like data assimilation of the forecast's starting state.
- Curtailment, outages, and capacity drift look like weather unless they are cleaned out first.

**Capacity drift and curtailment have to be cleaned out of the power target, and that cleaning
belongs to the effective-capacity work.** The weather-response model spans 2019 onwards, across
build-out and repowering, so an uncleaned target teaches both components to explain capacity changes
as weather. The [effective-capacity estimation](capacity-estimation.md) work
([#141](https://github.com/openclimatefix/nged-substation-forecast/issues/141)) owns the estimate,
and will ultimately estimate capacity inside the same DP models of wind and solar farms. This page
does not design the cleaning.

### Traps the study has to avoid

- **Stacking leakage.** The weather-response model's predictions on the calibrator's training rows
  must be out-of-fold over month blocks, or the calibrator under-corrects.
- **The calibrator absorbing the weather-response model's errors**, such as demand drift. Keep date
  and calendar features out of the calibrator, and measure the weather-response model's own error
  by fitting the same calibrator on estimate-of-past-weather inputs.
- **An optimistic stand-in.** ECMWF ENS shares ERA5's IFS lineage, so a calibrator learns ENS from
  an ERA5-trained weather-response model faster than it would learn a product from another
  family.
- **Season confounded with history length.** A calibrator fitted on the last three months before
  July sees only spring and early summer. The learning curve must be averaged over several start months.
- **A change of feed.** CEDA's UKV differs from live UKV, and Open-Meteo's ICON-EU may differ from
  Dynamical.org's ICON-EU.

### Pre-training then fine-tuning becomes a comparison arm for wind and solar

**For metered wind and solar generators, pre-training on estimates of past weather then fine-tuning
on ECMWF ENS ([#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167)) answers
the same question as the split, so the study runs #167's variants as comparison arms.** Warm start
can only add trees, so an over-confident weather sensitivity learned in the first phase is hard to
shrink. The split replaces that second phase with a calibrator small enough to fit on a few months.
The [single-pool](#single-pool-vs-two-phase-warm-start), [lead-time-zero, and
degraded-past-weather](#reconciling-era5-with-ens) arms are the fairer comparisons, because each
trains one model on every row. For demand substations the split does not apply, because their
forecasts lean on power lags, which the lag-free weather-response model does not use. So #167 still
owns the longer training history for the champion model at demand substations.

### The design moves a calibration step into the serving path

**A calibrate-and-blend step at serve time trades against [principle 2, complexity
offline](../design-philosophy/design-principles.md#2-complexity-belongs-offline-not-in-the-serving-path).**
The calibrator's coefficients are computed offline, as the [inherent-stability
rules](../design-philosophy/inherent-stability.md#where-complexity-should-live) accept for
regime-conditional calibration. But production would also run the frozen weather-response model on
every member of every product and blend the results, inside a `BaseForecaster` wrapper. A missing
product gets weight zero, so every product-absent case has to be a scored failure scenario before
the step is safe to ship.

### How the study evaluates the design

**The study measures a learning curve on stand-in products with long archives, against the same
product trained natively on the same months, on identical rows.** At each month, the calibrator is
fitted on the N months before that month and scored on that month, for N in 1, 2, 3, 6, and 12.
GEFS, from 2020-10-01 and a different model family from ERA5, is the primary stand-in, so the curve
can be averaged over start months in every season. ECMWF ENS, from 2024-04-01, is the secondary and
optimistic stand-in. AIFS-ENS, from 2025-07-02, is the real short-history case, entering as a blend
with ECMWF ENS. The controls are the weather-response model fed raw forecast members with no
calibrator, an XGBoost model trained natively on the same N months, and the ENS champion trained on
its full fold, which is a [confounded
comparison](../ml_experimentation/cross-validation-folds.md#the-confounded-comparison-which-must-not-be-read-as-the-ablation)
read only as the deployment reference.

**The split may improve calibration more than headline skill, so the study sets its kill criterion
before the first fit.** For one fixed product with ample history, a review of this design judged the
likely gain to be calibration rather than headline skill. The study therefore compares against
strong single-model baselines, scores every arm on identical rows by lead, and puts a paired
month-block bootstrap interval on every planned contrast. The study's numbers are study-page numbers
on the 9 metered generators: nothing becomes a leaderboard row until the design is built into the
pipeline.

### Open questions

- **One common set of weather variables.** Is one weather-response model on one common set of
  weather variables acceptable, given that products supply different variables? Dynamical.org's
  ICON-EU carries wind at 10 m only, so a common set would leave a richer product's extra variables unused.
- **The whole forecast, or a feature.** Is the calibrated forecast the whole forecast for wind and
  solar, beyond day 1 for example, or an input feature to the champion XGBoost model? A feature
  would need the champion retrained each time a product is added, which loses the "no retrain"
  benefit.
- **The serving path.** Is a calibration and blend step in the serving path acceptable against
  principle 2, complexity offline?
- **The failure-scenario vocabulary.** Should the [failure-scenario
  vocabulary](metrics-and-leaderboard.md#scoring-under-failure-scenarios)
  ([#437](https://github.com/openclimatefix/nged-substation-forecast/issues/437)) gain "one weather
  source absent" before that contract freezes?
- **Which estimate of past weather, per variable.** CAMS for irradiance and CEDA's UKV early time
  steps for the other variables is the working guess; ERA5, weather stations, weather satellites,
  and weather inferred from DP models are the alternatives. The best estimate may be a blend of
  several products, which raises a further question: feed the weather-response model every
  product's columns, or build a weather-space blend (see [the blending
  caveats](#which-estimate-of-past-weather-to-train-on)).

## Era covariates

The 2020–2026 span contains regime changes the weather cannot explain, in two shapes.

**Smooth trends** — EV and heat-pump uptake, embedded PV build-out, the 2022 price shock and the
Demand Flexibility Service. Handle these with recency sample weights, not a date ordinal: trees
extrapolate flat, so a date feature always sits beyond its training range at inference. The
[init-time-anchored
features](xgboost-improvements.md#init-time-anchored-features-current-level-anchor-prerequisite-for-the-global-model)
absorb level drift for the same reason.

**COVID lockdowns** are a pulse, and the case for a dedicated feature:

- A lockdown scalar in $[0, 1]$ passes the era-covariate safety rule by construction: production
  always sees 0, and 0 is abundant in the modern era. That safety pattern is the opposite of the
  NaN-NWP case, and the confounding is benign — the feature is the mechanism by which the model
  quarantines the anomalous period.

- **Source it rather than hand-coding dates, and prefer mobility to stringency.** The Oxford
  COVID-19 Government Response Tracker publishes a daily UK stringency index (0–100); Google's
  COVID-19 Community Mobility Reports sit closer to the causal driver of substation demand, and
  capture both the voluntary March-2020 withdrawal and the slow return through 2021–22. Both series
  ended in 2022, so check they are still downloadable — though a scalar that reads 0 for every
  future forecast makes a dead source a back-fill problem, not a serving problem. The measured
  evidence favours mobility, thinly: [Chen et al. (2020)](https://arxiv.org/abs/2006.08826) take UK
  national mean absolute percentage error from 10.11% to 8.74% by feeding mobility data into a
  day-ahead neural network, on a two-week test window, in an arXiv preprint. Retraining on pandemic
  data *without* mobility, though, made the UK figure worse, at 13.78% — so retraining across the
  lockdown is not safe on its own. The stringency index turned up in our search only in explanatory
  econometrics, and there [Berezvai et al. (2022)](https://doi.org/10.1016/j.segan.2022.100930)
  needed a *quadratic* specification, which one linear scalar cannot express.

- **Check whether simply adapting faster does the same job, before building the covariate.** [de
  Vilmarest and Goude (2021)](https://arxiv.org/abs/2110.00334) compare a Kalman filter handed the
  break date against one merely allowed to adapt faster everywhere with no break date at all. The
  no-break version won on three of four model families, and learning the variances rather than
  fixing them matched it. The recency sample weights above are the same idea. Run that arm first:
  the covariate has to beat faster adaptation, not merely beat doing nothing.

- **The pre-lockdown regime may never come back, and a scalar that returns to 0 says it does.**
  [Prabowo et al. (2023)](https://doi.org/10.1145/3600100.3623726), on 13 building complexes in
  Melbourne, report distribution shifts during lockdown "which do not fully revert to their
  pre-lockdown state even after restrictions are lifted". If GB substation demand behaved the same
  way — and permanent home-working makes that plausible — the post-lockdown era is a third regime
  rather than a return to the first. A lockdown scalar cannot say so, because it reads 0 both before
  2020 and after 2021, for two different worlds. Recency sample weights can, which is a second
  reason to run them as the control arm.

- **One published result supports the plan above: treat the lockdown as a labelled example rather
  than as data to discard.** [Abélès et al. (2024)](https://arxiv.org/abs/2402.14684) calibrate a
  slow process-noise variance on pre-COVID data and a fast one on 2020, then let a Markov switch
  choose between them at run time; tested on French national demand *after* the lockdowns, the
  switching version beat both fixed-variance filters. The lockdown label justifies itself by
  calibrating the fast regime, and nothing about a stringency or mobility series is needed at
  inference.

- **All of this evidence is national or building-level, none of it a distribution substation.** A de
  Vilmarest, Abélès, and Berezvai results are national transmission demand, and the only UK-level
  figure anywhere in this set is Chen et al.'s national mean absolute percentage error. A primary
  serves a few thousand customers with a strongly non-average mix, so its lockdown response could be
  far larger or far smaller than the national one, depending on whether it feeds a city centre or a
  dormitory estate. A search of OpenAlex for COVID-19 load forecasting at distribution substations
  returned nothing, so the magnitude is a per-substation question we will have to answer from NGED's
  own history.

- **Keep exclusion as an ablation arm.** Dropping 2020-03 to 2021-07 costs roughly 1.3 of about 5.5
  winters. Probably the wrong trade — lockdown distorts the occupancy and calendar response far more
  than the weather-to-power response, which is what the extra history is for — but it is one config
  flag, so measure it rather than assuming.

- **It breaks under the global model.** A per-series booster learns its own sign and magnitude for a
  national scalar, which handles customer mix for free: an industrial-estate primary and a
  residential one moved in opposite directions. A single [global booster per
  `time_series_type`](xgboost-improvements.md#global-model-per-time_series_type) cannot, without a
  customer-mix covariate we do not have.

- **It is a v0.6 requirement, and a v0.6 test case.** An unmodelled 16-month regime is the largest
  phantom event the [stage-1 switching baseline](switching-events.md#the-baseline-shared-foundation)
  could face, so the covariate is a requirement there rather than a nicety. Conversely, COVID is a
  free labelled test case for that milestone's self-resetting residual accumulators, which detect
  regime shifts with no hand-coded dates.

**Two things to check in the pre-2024 power data before trusting it.** NGED's switching logs go back
to at least 2019, so the gap years contain real switching events and need whatever masking the
modern data gets. And the primaries' "Disaggregated Demand" depends on which embedded generators
were metered at the time, so a meter coming online mid-history silently redefines that series — the
[Embedded Capacity Register](https://github.com/openclimatefix/nged-substation-forecast/issues/159)
and [MPAN-to-substation](https://github.com/openclimatefix/nged-substation-forecast/issues/241)
ingests carry the connection dates needed to check.

## Evaluation

- **Scoring against ERA5 is a diagnostic, never the promotion criterion.** It decomposes total error
  into the weather-to-power response — the part we can actually improve, since NWP error is
  exogenous to us — and the implicit hedging against forecast error. Expect the two rankings to
  disagree: under perfect weather the best model leans hard on weather features, so a large
  divergence is information about how much hedging a model does, not a bug. The same scope carries
  the [perfect-weather
  ceiling](metrics-and-leaderboard.md#the-perfect-weather-ceiling-what-it-gates), which sizes how
  much of our error is the weather *forecast's* fault and so gates how much to invest in the weather
  input at all. It lands as a new `evaluation_scope`, not as a new fold, so leaderboard folds stay
  ENS-only and both [principle
  8](../design-philosophy/design-principles.md#8-every-experiment-is-scored-identically) and the
  [rejection of reanalysis-backed validation
  folds](../architecture/ml-orchestration.md#yearly-folds-backed-by-era5-rejected-for-validation)
  stand.

- **Validate the no-NWP fallback on held-out 2024+ rows with NWP artificially removed**, never on
  pre-2024 rows — otherwise we measure fallback skill in a demand regime we will never forecast
  again.

- **Decide the evaluation protocol before sweeping the variant grid.** The cells are not
  independent, and a dozen runs against a single held-out period produce a winner whether or not
  there is a real difference — particularly since these questions hinge on seasonal behaviour and we
  have only two ENS winters to evaluate against. We need a stated test separating a genuine
  improvement from run-to-run variance.

- ERA5 is a single frozen IFS cycle across the whole archive, so year-over-year comparison within it
  is not contaminated by NWP system upgrades. The flipside: our ENS archive spans cycle changes, so
  some apparent drift there is the weather model changing rather than the electricity network.

## A staged-GRIB route fills three of the missing years without waiting for the Zarr backfill

Dynamical.org are backfilling the **operational** IFS ENS archive as a queryable Zarr store — the
real forecasts as they were issued, not reforecasts — from ECMWF's MARS tape archive: 2016-03-08 to
2024-04-01, 51 members, 0.25°, 00Z initialisations only
([dynamical-org/reformatters#446](https://github.com/dynamical-org/reformatters/issues/446)). Honest
multi-year folds from that would be strictly better than pre-training and would make most of the
variant grid unnecessary, but as of 2026-05 the Zarr estimate was **~November 2027**, MARS-bound at
roughly 0.8 TB/day against ~446 TB remaining — well after v1.0.

**Dynamical.org also stage the same MARS files as raw GRIB1 on Source Cooperative, ahead of turning
them into Zarr, and a pilot proved we can decode those files ourselves.** The staged bucket carries
complete dates from 2021-03-21 to 2024-03-31 today — about three of the missing years — readable
anonymously. The pilot ([#951](https://github.com/openclimatefix/nged-substation-forecast/issues/951),
merged) fetched the control member for 23 dates across that range and checked the result six ways:
317 of 317 sampled messages decoded bit-exact against ecCodes, the idx offset chain had no gaps, 20
of 20 re-fetched byte ranges hashed identically to the first fetch, `2t` and `2d` confirmed in
kelvin, and de-accumulation matched Dynamical.org's own clipping rule. A follow-up listing check
confirmed every one of the 23 pilot dates also has the full 51-member surface and pressure-level
files, not just the control member, so control-first fetching is not a separate question to raise
with Dynamical.org — the staged bucket already orders nothing, and we can fetch either the control
member alone or all 51 members as needed.

**The wider fetch is on hold.** Dynamical.org has indicated they may be able to materialise their
Zarr backfill over this same range sooner than the ~November 2027 estimate above, which would
replace the hand-rolled GRIB decode with a plain Zarr read. Issue
[#959](https://github.com/openclimatefix/nged-substation-forecast/issues/959) tracks the wider
staged-GRIB fetch and is paused pending their reply, rather than committing to a fetch effort
estimated at ~4-5 hours and 530 GB for the control member alone, or up to 6-14 days on the
workstation (or an estimated $10-30 on a cloud machine, in a few hours) for all 51 members, if
Dynamical.org's own Zarr route lands first.

Two details still worth tracking regardless of which route lands:

- **00Z only**, which runs against
  [#350](https://github.com/openclimatefix/nged-substation-forecast/issues/350)'s move to the live
  service's four daily inits.

- **The backfilled span crosses further ENS resolution upgrades**: 41r2 in 2016-03 (32→18 km), 48r1
  in 2023-06 (18→9 km, within the staged bucket's complete-date range), and 49r1 in 2024-11. Each is
  an era boundary under the `study` skill.

The **ERA5 ingest is unconditional** either way: capacity estimation, the [weather-abnormality
climatology](xgboost-improvements.md#weather-abnormality-climatology-z-score-features), and the ERA5
diagnostic scope all need it regardless of which ENS backfill route lands.

## Implementation details (deleted when this ships)

Ordered, and deliberately not one PR. Steps 1–2 are the data layer; the rest are experiments.

1. Ingest ERA5 for 2020 to present, gap **and** overlap
   ([#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143)).
2. Compute paired `ENS − ERA5` residual statistics on the overlap, per variable, per lead time, per
   season.
3. Build the masking augmentation (NWP features, and spread/quantile columns separately) as a
   configurable step, plus the sampled pseudo-horizon for lag nullification.
4. Add the lockdown covariate and the per-era sample weights.
5. Define the evaluation protocol and the variance-versus-improvement test, and add the ERA5
   diagnostic `evaluation_scope`.
6. Sweep the variant grid
   ([#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167)): reconciliation
   method × single-pool/two-phase × pure/mixed phase two × flag on/off × spread columns
   present/masked.

The study of [a new weather product with a few months of
history](#a-new-weather-product-with-a-few-months-of-history)
([#999](https://github.com/openclimatefix/nged-substation-forecast/issues/999)) runs separately,
under `studies/`:

1. Build the weather-response model at the 9 metered generators, lag-free and out-of-fold over
   month blocks, once per estimate of past weather under test (ERA5, CAMS for irradiance, CEDA's
   UKV early time steps for the other variables).
2. Fit the power-space calibrator per technology, smooth in lead time, in capacity-factor units,
   censored at 0 and 1, and shrunk towards the ECMWF ENS calibrator, refitted on a rolling origin.
3. Measure the learning curves on GEFS and ECMWF ENS, and on the blend of ECMWF ENS with AIFS-ENS,
   with #167's pre-training variants as comparison arms for wind and solar.
4. Run the derivative-free weather-space arm on wind and on ICON-EU.
5. Decide: promote the section to 🚧 with a milestone, or move the result to the study page and
   delete the section.

**Ordering against the rest of v0.5.** The Tier-1 and Tier-2 config wins on [XGBoost
improvements](xgboost-improvements.md) do not wait for any of this, and one of them, [the lead-time
feature](xgboost-improvements.md#feed-the-model-the-forecast-lead-time-review-discovery-one-line),
is a prerequisite for the lead-time-zero framing. The data-hungry structural items (batched
training, ensemble-member training, the global model) are worth running *after* the history lands,
since that is where four extra years change the answer most.
