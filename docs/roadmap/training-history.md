# Extending the training history with ERA5

> **Status: 🚧 Planned (v0.5).** Epic:
> [#145](https://github.com/openclimatefix/nged-substation-forecast/issues/145). Ingest:
> [#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143); pre-training
> experiments: [#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167).

Our ECMWF ENS archive starts 2024-04-01; most trial-area power series go back to late 2019.
[ERA5](data-sources.md#weather-data) covers the gap, and shares the ENS's IFS lineage, so the
reanalysis-to-forecast domain shift is smaller than a different-model reanalysis would give.

Almost all the cost is in the data layer. Once ERA5 and the paired residual statistics exist,
moving between the variants below is mostly configuration — so they are leaderboard arms, not
decisions to make up front.

## Scope the ingest to include the 2024+ overlap

**Fetch ERA5 across the 2024+ ENS overlap as well as the 2020–2023 gap.** This is the one thing
that changes what the ingest builds, and it exists to avoid *era confounding*: a feature value that
occurs in only one time period is learned as a proxy for that period, not for the thing we mean by
it. If NaN-NWP appears only before 2024, "weather is missing" becomes a perfect proxy for
"2020–2023" — a demand regime carrying COVID distortions, less embedded PV, and lower EV and
heat-pump penetration — so a 2027 feed failure would be forecast from a 2021 regime. The same trap
applies to a source flag, to a lead-time-zero encoding, and to the ENS-only spread and quantile
columns, which have no ERA5 equivalent.

The general rule: **an era covariate is safe exactly when the value production will see is
well-represented in the modern era.** So, alongside the overlap fetch, randomly mask NWP features
and spread/quantile columns on a subset of 2024+ rows, as a configurable augmentation step.

The overlap has a second payoff: paired ERA5 and ENS on identical target times, which turns the
reconciliation question below into estimation rather than guesswork.

## Reconciling ERA5 with ENS

- **Lead-time-zero framing.** Do not degrade; treat ERA5 as a forecast at lead zero and let the
  lead-time feature carry the discounting. This separates the physical weather-to-power response
  (genuinely lead-time-invariant) from how far to trust it as forecast error grows. Cheapest arm
  and the right first run, but note the tension: pre-2024 rows then carry only lead zero, so every
  split on lead time partitions the modern rows off beneath it, and the extra history reaches the
  3–10 day band only through the structure above such splits and in the trees that never make one.
  Boosting shares more across trees than that phrasing might suggest, so this is a weakening rather
  than a wall — but expect the win at short leads unless the invariance assumption holds strongly.

- **Degrade ERA5 towards ENS error statistics.** Fit the `ENS − ERA5` residual distribution per
  variable, per lead time, per season on the overlap, then sample from it when synthesising
  pre-2024 features. Quantile mapping per horizon is the cheap version and is probably enough for
  temperature. This is not merely an alternative to lead-zero framing: it is what makes the extra
  history populate the long leads at all.

- **ENS reforecasts — considered and rejected.** Under Cycle 49r1 the medium-range reforecasts run
  over the past 20 years with an 11-member ensemble, so they are real forecasts with real
  lead-time error and the mismatch would largely disappear rather than needing correction. **We are
  not going to do this**: the only access is MARS, and the download would take far too long.
  Recorded so it is not re-litigated.

## ERA5 splits one horizon into two

Today `nwp_lead_time_hours` (how old the weather is) and the forecast horizon (which power lags are
available) differ only by the constant `NWP_PUBLICATION_DELAY_HOURS`, so one column carries both.
ERA5 decouples them: weather age is zero, but power-lag availability must still mirror production,
or pre-training teaches the model to lean on lags that vanish at serve time. Pre-training rows
therefore need a *sampled* pseudo-horizon driving `_nullify_leaky_lags`, carried separately from
weather age.

## Single pool vs two-phase warm start

- **Single pool, with per-era sample weights.** All rows in one training set, and the weight on
  pre-2024 rows becomes a tunable hyperparameter rather than a yes/no decision. Run this first: it
  is the cheap form of the mixed warm start below, and recency sample weights are already a Tier-1
  item on the
  [XGBoost improvements](xgboost-improvements.md#early-stopping-instead-of-fixed-n_estimators500)
  page.

- **Two-phase warm start.** Train on the ERA5 history, then continue boosting on 2024+ ENS data.
  Warm start only *adds* trees, so the correcting trees see roughly two years and very few examples
  of each season; and if phase one over-trusts weather, shrinking an over-confident component
  additively is harder than never building it. **Mixed phase two** — keeping down-weighted (and
  possibly degraded) ERA5 rows in phase two — is the middle path.

- **The source flag follows from that choice, not the other way round.** In a *pure* phase two the
  flag has zero variance and XGBoost can never split on it, so drop it there. Under a single pool
  or a mixed phase two it has variance and earns its place — but only because the overlap fetch
  decorrelates it from date.

## Era covariates

The 2020–2026 span contains regime changes the weather cannot explain, in two shapes.

**Smooth trends** — EV and heat-pump uptake, embedded PV build-out, the 2022 price shock and the
Demand Flexibility Service. Handle these with recency sample weights, not a date ordinal: trees
extrapolate flat, so a date feature always sits beyond its training range at inference. The
[init-time-anchored features](xgboost-improvements.md#init-time-anchored-features-current-level-anchor-prerequisite-for-the-global-model)
absorb level drift for the same reason.

**COVID lockdowns** are a pulse, and the case for a dedicated feature:

- A lockdown scalar in $[0, 1]$ passes the era-covariate safety rule by construction: production
  always sees 0, and 0 is abundant in the modern era. This is the opposite of the NaN-NWP case, and
  the confounding is benign — the feature is the mechanism by which the model quarantines the
  anomalous period.

- **Source it rather than hand-coding dates.** The Oxford COVID-19 Government Response Tracker
  publishes a daily UK stringency index (0–100) — the equivalent of the `holidays` package for this
  problem. Google's COVID-19 Community Mobility Reports are the richer alternative, because
  workplace and residential mobility sits closer to the causal driver of substation demand than the
  legal state does, and it captures both the voluntary March-2020 withdrawal and the slow return
  through 2021–22. Both series ended in 2022, so check they are still downloadable — but note that a lockdown scalar is 0 for every future forecast, so a dead source is a back-fill problem, not a serving problem.

- **The measured evidence favours mobility over stringency, though it is thin.** [Chen et al.
  (2020)](https://arxiv.org/abs/2006.08826) feed Apple Mobility Trends driving data and Google
  Community Mobility Reports transit data straight into a day-ahead neural network across 12
  regions. On UK national demand they take mean absolute percentage error from 10.11% for a pre-pandemic model to 8.74% with mobility, and to 4.46% with mobility plus multi-task learning
  across similar-sized regions — while retraining on pandemic data *without* mobility made the UK figure worse, at 13.78%. Two caveats before leaning on those figures: the test window is 1 to 15 May 2020, two
  weeks, and the paper is an arXiv preprint. The Oxford stringency index turned up in our search
  only in explanatory econometrics, never as a forecasting input — and there ([Berezvai et al.
  (2022)](https://doi.org/10.1016/j.segan.2022.100930), 23 European Union member states) a
  *quadratic* specification was needed, with "the partial effect of an increase in the stringency
  index depend[ing] on the type of day (weekday or weekend), hour of the day, and initial stringency
  level", which one linear scalar cannot express.

- **Check whether simply adapting faster does the same job, before building the covariate.** [de
  Vilmarest and Goude (2021)](https://arxiv.org/abs/2110.00334) compare, on one dataset and horizon,
  a Kalman filter handed the break date against one merely allowed to adapt faster everywhere with
  no break date at all. The no-break version won on three of four model families: mean absolute
  error 11.2 against 13.6 MW for a linear model, 12.4 against 14.3 for a generalised additive model,
  and 14.3 against 16.2 for an autoregressive model, losing only on a neural network at 12.4 against
  12.3. Learning the variances rather than fixing them matched the no-break version. The recency
  sample weights above are the same idea, so run that arm first: the covariate has to beat faster
  adaptation, not merely beat doing nothing.

- **The pre-lockdown regime may never come back, and a scalar that returns to 0 says it does.**
  [Prabowo et al. (2023)](https://doi.org/10.1145/3600100.3623726), on 13 building complexes in
  Melbourne, report "significant shifts in distributions during the lockdown, which do not fully
  revert to their pre-lockdown state even after restrictions are lifted". If GB substation demand
  behaved the same way — and permanent home-working makes that plausible — then the post-lockdown
  era is a third regime rather than a return to the first, and a lockdown scalar cannot say so: it
  reads 0 both before 2020 and after 2021, for two different worlds. Recency sample weights can
  express that difference, which is a second reason to run them as the control arm.

- **One published result supports the plan above: treat the lockdown as a labelled example rather
  than as data to discard.** [Abélès et al. (2024)](https://arxiv.org/abs/2402.14684) calibrate two
  process-noise variances — a slow one on pre-COVID data (2012 to 2019) and a fast one on 2020 —
  then let a Markov switch choose between them at run time. Tested on French national half-hourly
  demand over 2021 and 2022, *after* the lockdowns, the switching version beat both fixed-variance
  filters. Note what that buys: the lockdown pays for itself by calibrating the fast regime, and
  nothing about a stringency or mobility series is needed at inference.

- **All of this evidence is national or building-level, none of it a distribution substation.** The
  de Vilmarest, Abélès, and Berezvai results are national transmission demand; the only UK-level figure anywhere in this set is Chen et al.'s national mean absolute percentage error. A primary
  substation serves a few thousand customers with a strongly non-average mix, so its lockdown
  response could be far larger or far smaller than the national one depending on whether it feeds a
  city centre or a dormitory estate. A search of OpenAlex for COVID-19 load forecasting at
  distribution substations returned nothing, so this is a per-substation question we will have to
  answer from NGED's own history.

- **Keep exclusion as an ablation arm.** Dropping 2020-03 to 2021-07 costs roughly 1.3 of about
  5.5 winters. Probably the wrong trade — lockdown distorts the occupancy and calendar response far
  more than the weather-to-power response, which is what the extra history is for — but it is one
  config flag, so measure it rather than assuming.

- **It breaks under the global model.** A per-series booster learns its own sign and magnitude for
  a national scalar, which handles customer mix for free: an industrial-estate primary and a
  residential one moved in opposite directions. A single
  [global booster per `time_series_type`](xgboost-improvements.md#global-model-per-time_series_type)
  cannot, without a customer-mix covariate we do not have.

- **It is a v0.6 requirement, and a v0.6 test case.** An unmodelled 16-month regime is the largest
  phantom event the
  [stage-1 switching baseline](switching-events.md#the-baseline-shared-foundation) could face, so
  the covariate is a requirement there rather than a nicety. Conversely, COVID is a free labelled
  test case for that milestone's self-resetting residual accumulators, which detect regime shifts
  with no hand-coded dates.

**Two things to check in the pre-2024 power data before trusting it.** NGED's switching logs go
back to at least 2019, so the gap years contain real switching events and need whatever masking the
modern data gets. And the primaries' "Disaggregated Demand" depends on which embedded generators
were metered at the time, so a meter coming online mid-history silently redefines that series — the
[ECR](https://github.com/openclimatefix/nged-substation-forecast/issues/159) and
[MPAN-to-substation](https://github.com/openclimatefix/nged-substation-forecast/issues/241) ingests
carry the connection dates needed to check.

## Evaluation

- **Scoring against ERA5 is a diagnostic, never the promotion criterion.** It decomposes total
  error into the weather-to-power response — the part we can actually improve, since NWP error is
  exogenous to us — and the implicit hedging against forecast error. Expect the two rankings to
  disagree: under perfect weather the best model leans hard on weather features, so a large
  divergence is information about how much hedging a model does, not a bug. The same scope carries
  the [perfect-weather ceiling](metrics-and-leaderboard.md#the-perfect-weather-ceiling-what-it-gates),
  which sizes how much of our error is the weather *forecast's* fault and so gates how much to
  invest in the weather input at all. It
  lands as a new `evaluation_scope`, not as a new fold, so leaderboard folds stay ENS-only and both
  [principle 8](../design-philosophy/design-principles.md#8-every-experiment-is-scored-identically)
  and the
  [rejection of reanalysis-backed validation folds](../architecture/ml-orchestration.md#yearly-folds-backed-by-era5-rejected-for-validation)
  stand.

- **Validate the no-NWP fallback on held-out 2024+ rows with NWP artificially removed**, never on
  pre-2024 rows — otherwise we measure fallback skill in a demand regime we will never forecast
  again.

- **Decide the evaluation protocol before sweeping the variant grid.** The cells are not
  independent, and a dozen runs against a single held-out period produce a winner whether or not
  there is a real difference — particularly since these questions hinge on seasonal behaviour and
  we have only two ENS winters to evaluate against. We need a stated test separating a genuine
  improvement from run-to-run variance.

- ERA5 is a single frozen IFS cycle across the whole archive, so year-over-year comparison within
  it is not contaminated by NWP system upgrades. The flipside: our ENS archive spans cycle changes,
  so some apparent drift there is the weather model changing rather than the network.

## The ECMWF ENS backfill will not arrive in time

Dynamical.org are backfilling the **operational** IFS ENS archive — the real forecasts as they were
issued, not reforecasts — from ECMWF's MARS tape archive: 2016-03-08 to 2024-04-01, 51 members,
0.25°, 00Z initialisations only
([dynamical-org/reformatters#446](https://github.com/dynamical-org/reformatters/issues/446)).
Honest multi-year folds from that would be strictly better than pre-training and would make most of
the variant grid unnecessary, but as of 2026-05 the estimate was **~November 2027**, MARS-bound at
roughly 0.8 TB/day against ~446 TB remaining. That is well after v1.0, so we plan as though it will
not arrive. Three details worth tracking:

- **The control member may land far sooner than the full ensemble.** Almost all the remaining
  volume is the 50 perturbed members; the control files are a few TB of the ~446 TB. Our CV trains
  on the control member today, so control-only completion would already be enough to found
  multi-year folds — worth asking Dynamical.org whether control-first ordering is possible.

- **00Z only**, which runs against
  [#350](https://github.com/openclimatefix/nged-substation-forecast/issues/350)'s move to the live
  service's four daily inits.

- **The backfilled span crosses two more ENS resolution upgrades** (41r2 in 2016-03, 32→18 km;
  48r1 in 2023-06, 18→9 km), compounding the cycle-change caveat above.

The **ERA5 ingest is unconditional** either way: capacity estimation, the
[weather-abnormality climatology](xgboost-improvements.md#weather-abnormality-climatology-z-score-features),
and the ERA5 diagnostic scope all need it regardless of the backfill.

## Implementation details (deleted when this ships)

Ordered, and deliberately not one PR. Steps 1–2 are the data layer; the rest are experiments.

1. Ingest ERA5 for 2020 to present, gap **and** overlap
   ([#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143)).
2. Compute paired `ENS − ERA5` residual statistics on the overlap, per variable, per lead time,
   per season.
3. Build the masking augmentation (NWP features, and spread/quantile columns separately) as a
   configurable step, plus the sampled pseudo-horizon for lag nullification.
4. Add the lockdown covariate and the per-era sample weights.
5. Define the evaluation protocol and the variance-versus-improvement test, and add the ERA5
   diagnostic `evaluation_scope`.
6. Sweep the variant grid
   ([#167](https://github.com/openclimatefix/nged-substation-forecast/issues/167)): reconciliation
   method × single-pool/two-phase × pure/mixed phase two × flag on/off × spread columns
   present/masked.

**Ordering against the rest of v0.5.** The Tier-1 and Tier-2 config wins on
[XGBoost improvements](xgboost-improvements.md) do not wait for any of this, and one of them,
[the lead-time feature](xgboost-improvements.md#feed-the-model-the-forecast-lead-time-review-discovery-one-line),
is a prerequisite for the lead-time-zero framing. The data-hungry structural items (batched
training, ensemble-member training, the global model) are worth running *after* the history lands,
since that is where four extra years change the answer most.
