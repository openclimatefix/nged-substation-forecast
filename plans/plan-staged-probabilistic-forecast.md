# Plan: document a staged probabilistic forecast, and schedule the studies that test it

**Status:** draft for review, updated after experiments 00001 to 00006. This plan changes nothing
yet. It describes a documentation pull request (PR) and a set of GitHub issues that follow once the
plan is approved. A fresh reader is
asked to critique both the design and the way we propose to record it.

## 1. What this plan asks for

**This plan proposes one documentation PR and one batch of GitHub issues.** The documentation PR
adds a roadmap page that records a candidate design for the probabilistic power forecast, with the
studies that will test it. The batch of issues is a tracking issue and five child issues for the
infrastructure and the studies, plus edits to two existing issues. Sections 6 and 7 list the exact
changes. No production code changes: the stage 1 and stage 2 questions run as ml-experiments under
`ml-experiments/`, which the research log tracks, and production code changes only through a
numbered model release after a human has reviewed the port.

**The first deliverable is an experiment, not a build.** It rebuilds the lagged-power comparison of
experiment 00005 with lag anomalies from the sources serving will have. Error-correlation coupling
(ECC), the beta-transformed linear pool, the ratio target, and the lead-bucket arm are later arms,
and the plan removes any of them that does not earn its place.

**The plan asks the maintainer to decide three things:** whether to approve the staged chain as the
best guess under test, whether to approve the documentation PR and the issues in sections 6 and 7,
and the open questions in section 10.

**What a reviewer is asked to judge:**

- **The design** (sections 3 and 4): is it more complicated than its aims require, and will the
  studies in section 5 give valid answers?
- **The documentation plan** (section 6): is the proposed home, shape, and cross-linking right for a
  design that is a best guess under test?
- **The issue plan** (section 7): are the studies ordered and bounded so that each stops or
  continues the work on a stated condition?

## 2. Context for a reader with none

**The project forecasts electricity demand and generation at the substations of a distribution
network operator** (NGED, National Grid Electricity Distribution). The forecasts are probabilistic:
the service publishes quantiles of power, and the score is the continuous ranked probability score
(CRPS, lower is better) with mean absolute error (MAE) as the second metric. The repository is a
`uv` workspace. Production code lives in `src/` and `packages/`, humans review it, and it follows
the [inherent-stability
rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/):
production never raises because an input is absent or stale; it degrades and records the
degradation. Study code lives in `studies/` and `packages/studies/`, is written and run autonomously
by agents, and is never imported by production code.

**Terms used below:**

- **NWP:** numerical weather prediction, a physics-based weather forecast. **ENS:** the ECMWF
  (European Centre for Medium-Range Weather Forecasts) ensemble, 51 members per run, served by
  Dynamical.org from 2024-04-01. **UKV:** the Met Office's deterministic regional model.
  **MOGREPS-UK:** the Met Office's regional ensemble, with a few months of history.
- **ERA5:** ECMWF's reanalysis, an estimate of past weather. **CAMS:** the Copernicus Atmosphere
  Monitoring Service's satellite-derived solar irradiance. The docs call these and similar sources
  "estimates of past weather".
- **Series:** one half-hourly power time series (a substation, a solar farm, or a wind farm). The
  quick-wins study covered 31 series: 6 solar farms in one weather region, 3 wind farms, and 22
  other series. Version 2 targets 2,500 series.
- **Quantile model:** a model trained with pinball loss, which outputs quantiles of power. **PIT:**
  probability integral transform, a histogram that shows calibration. **EMOS:** ensemble model
  output statistics, a parametric recalibration of an ensemble. **ECC:** ensemble copula coupling,
  a way to turn a calibrated marginal distribution back into ensemble members that keep their
  dependence across time and series. **ECC-Q:** ECC by sorting quantiles against member ranks.
- **Clear-sky power:** the power a site would produce under a cloudless sky at that time.

**The evidence so far is the quick-wins study** ([study
page](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/quick-wins/),
[issue #1150 (Study: quick wins for the production XGBoost
forecast)](https://github.com/openclimatefix/nged-substation-forecast/issues/1150)). The production
forecast scores a CRPS of 11.3% of capacity. The 10% to 90% band holds the outcome in about 10% of
half-hours against a nominal 80%. The best arm (quantile training, 7.85% on 12 validation months) is
still 0.6 points above the `manual_heuristic` baseline (7.2%). Part 1 of [issue #1156 (Study: the
production feature faults, and calibrating the probabilistic power
forecast)](https://github.com/openclimatefix/nged-substation-forecast/issues/1156), which audited
the production feature pipeline, shipped in [PR #1159 (Study: production feature faults and forecast
effects)](https://github.com/openclimatefix/nged-substation-forecast/pull/1159). Part 2 of #1156 is
what this plan replaces.

## 3. Aims

**The aim is not only a skilful forecast on day one.** A design that wins today's leaderboard but
cannot take a second weather model is the wrong design. The aims, in priority order:

1. **A skilful, calibrated probabilistic power forecast** at 2,500 series, judged by CRPS and MAE.
2. **Easy addition of new NWP products, including products with very short histories.** Blending
   many NWP products is the maintainer's strongest hunch for improving the forecast. Some of the
   best products have too little history to train a full forecasting model on (MOGREPS-UK, a few
   months). Some are deterministic (UKV). Some have different step widths (hourly beyond day 5,
   against ENS's 3-hourly then 6-hourly) or different variable conventions (a mean over the previous
   period, an instantaneous value, or a change between the two inside one product's life).
3. **Reuse of the weather-to-power model ("stage 1") elsewhere in the system:** switching-event
   detection, effective-capacity estimation, data cleaning (metering-fault detection), and the
   lag-anomaly features (how unusual was each lagged power value, given the weather then).
4. **Coherent ensemble members.** The grid physics must be coherent within each ensemble member (a
   member that is very windy for one wind farm is windy for all of them). Coherence matters most
   beyond day 7, where ENS members disagree sharply about wind speed and irradiance.
5. **Explainability.** An engineer should be able to say which stage produced an error.
6. **Inherent stability.** Each stage always runs. No conditional skips a stage. A missing input is
   a null that XGBoost routes, never a raised exception.
7. **Research first.** Studies test every idea before production code changes.

Aims 2 and 5 are the reason the forecast chain has more than one learned stage. Each stage has one
job: stage 1 turns weather into power, stage 2 calibrates and blends the products, and the coupling
step restores coherent members. One model doing several of those jobs is harder to explain, to test,
and to extend with a new product. Aim 3 needs stage 1 only as an offline artifact that runs on
analysis weather, so aim 3 does not by itself justify serving the chain. Stage 1 still runs at
serving time to compute the lag anomalies, but on analysis weather and not on forecast members.

**Design principles traded away.** The staged design departs from three of the [design
principles](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/).
Principle 2 (complexity belongs offline, not in the serving path) is met for training but not for
serving, where the adapter, stage 1, stage 2, and coupling all run on every forecast. Principle 3
(one execution path from research to production) is strained because stage 1 has no lead time and so
does not fit the `BaseForecaster` interface. Principle 6 (the whole system runs on one laptop)
requires three chained trained artifacts (stage 1, the stage 2 calibrator, and the blend) to run on
one machine. In return the design buys aims 2 and 5. The studies keep the stage count to what they
prove: a stage that does not earn its place is removed.

## 4. The candidate design

**The best-guess design has three parts with fixed roles: stage 1 turns weather into power, stage 2
calibrates, uses lagged power, and blends the products, and coupling restores coherent members.**
Every part always runs, and each has one job. The experiments so far support a stage 1 trained on
analysis weather, and support lagged power in stage 2. They do not support stage 2 for a single
product without lags, so stage 2 must earn its place under the two conditions it exists for: several
weather products and lagged power. The main challenger is a single-stage quantile XGBoost model on
ENS statistics with a small correction per additional product. A learned weather-correction stage
("stage 0") and a learned persistence stage ("stage 1b") are not in the first design; see the
alternatives.

```text
NWP n, member m ─▶ adapter ─▶ canonical weather ─▶ Stage 1 ─▶ power(n,m,s,t)
                                                                          │
        per-product statistics across members, lead, lag anomalies ◀──────┘
                                    │
                                    ▼
             Stage 2a: shared per-product calibrator ─▶ calibrated quantiles per product
                                    │
                                    ▼
             Stage 2b: learned blend over the products present ─▶ calibrated quantiles(s,t)
                                    │
                                    ▼
                                   ECC ─▶ 51 coherent members
```

Notation: n is the NWP product, m the ensemble member, s the series, t the valid time.

| Stage | Input | Output | Per NWP | Per member | Lead input |
|---|---|---|---|---|---|
| Adapter (code, not learned) | One member's raw fields; a per-NWP, per-variable, per-date convention table | Fields on the common half-hourly grid in the canonical convention | Yes | Yes | No |
| Stage 1 (learned) | Canonical weather, calendar, site static fields | Conditional quantiles of power for one series (9 levels), with the median used as the member's point value | No | Yes | No |
| Stage 2a, calibrator (learned) | One product's stage 1 statistics across its members (mean, spread, percentiles), lead, lagged power, lag anomalies | Calibrated quantiles of power for that product | One shared function, with a spread factor per product | No | Yes |
| Stage 2b, blend (learned) | The calibrated quantiles of the products that are present, lead | Calibrated quantiles of power per series and valid time | Combines the products | No | Yes |
| ECC (code) | Calibrated quantiles, stage 1 member medians | 51 coherent members (the ENS member count) | No | Output | No |

**Five experiments and one closed study give the evidence so far.** The [research
log](https://openclimatefix.github.io/nged-substation-forecast/ml-experiments/research-log/) holds
each experiment's question, verdict, and figures; the bullets give the headline of each. Pull
request numbers for 00005 and 00006 are [PR #1195 (Experiment 5: where lagged power enters a
two-stage demand forecast)](https://github.com/openclimatefix/nged-substation-forecast/pull/1195)
and [PR #1202 (Experiment 6: which input features work best for the ERA5-trained demand
model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1202).

- **00001, solar and wind, ENS only:** stage 1 trained on ERA5 and applied to the 51 ENS members and
  averaged had a lower mean absolute error than the same columns trained on ENS weather, by 0.57
  points of capacity for solar (interval −0.83 to −0.32) and 1.70 for wind (−2.33 to −1.08), up to
  day 6 and not beyond. The gain follows the training weather and not the training length.
- **00002, solar and wind, ENS only, quantiles:** pooling the ERA5-trained quantiles over the
  members beat an ENS-trained quantile model (CRPS −0.62 for wind, −0.24 for solar). A stage 2
  calibrator with a learned blend, given one product and no lags, did not improve on the pooled
  mixture (+0.26 wind, +0.40 solar).
- **00003, 22 series that are not wind or solar farms (16 demand series, 4 grid flows, and 2
  others):** with weather only, the ERA5-trained model was worse than the ENS-trained model by 1.37
  points (+0.68 to +2.17), and with calendar columns and lags the two were equal (+0.10, −0.26 to
  +0.50). The error fell steadily as the training start moved later, which points to drift in the
  customer mix and embedded generation.
- **00005, 16 Primary substations, ceiling arms (verdict: refuted, because the plan predicted that a
  separate persistence stage would win):** lagged power in the stage 2 calibrator beat a lag-free
  calibrated stage 1 by 1.80 points of quantile score (12 of 12 months, 16 of 16 series), and beat
  lagged power through a separate persistence stage ("stage 1b") plus a calibrator by 0.84 (99%
  interval +0.28 to +1.50). Adding the persistence stage on top of lags in stage 2 gained 0.06
  (−0.15 to +0.03). The advantage of stage 2 persisted at 3 to 15 days, and 9 of 12 months favoured
  lags in stage 2.
- **00006, Primary substations, confirmation year:** hour and weekday were the largest non-lag input
  (−0.46, 99.4% interval −0.61 to −0.30). A COVID indicator, a 14-day Christmas window, and recency
  weights helped, and a 2023-09 training start helped most (confirmation-year evidence only). Time
  of year, trailing temperature means, and windchill added nothing, because the weather block
  already carries the season.
- **Indicative inputs from a closed study ([PR #1175 (Study XGBoost pipeline fixes, lean inputs and
  calibration
  diagnostics)](https://github.com/openclimatefix/nged-substation-forecast/pull/1175)):** the
  radiation correction lowers solar pinball loss by about 0.34 points, and the pooled quantile
  forecasts under-cover for wind. No human reviewed those numbers.

All lagged arms of 00005 are ceilings: the lag anomalies use ERA5 weather at past times, which
serving does not have. The ceiling is the main open risk of the lag design, and the next experiments
build the lag anomalies from the sources serving will have.

**The adapter handles known facts as code.** Each NWP has a table of step width, accumulation or
instantaneous convention, and the date each convention applies from. The adapter resamples every NWP
(and ERA5 and CAMS) to the same half-hourly grid in the same convention. Two parts have shipped:
short-wave radiation conserves each native window's mean and long-wave radiation uses a
shape-preserving cubic interpolant ([PR #1166 (Resample short-wave and long-wave radiation over
native intervals)](https://github.com/openclimatefix/nged-substation-forecast/pull/1166)), and wind
travels as eastward and northward components, with speed and bearing derived afterwards ([PR #1167
(Retain wind components through
interpolation)](https://github.com/openclimatefix/nged-substation-forecast/pull/1167)). Learning
conventions from data would waste a short history and could not be unit-tested. A validation check
(diurnal profile against solar geometry) catches an undeclared convention change and raises a WARN
asset check, not an exception.

**Stage 1 is NWP-agnostic and lead-agnostic.** It trains on analysis weather (ERA5 and CAMS) to
power. Most trial-area power series go back to late 2019 and the ERA5 ingest starts in 2020, so it
could use up to roughly seven years against the roughly two years of ENS. Experiments 00001 and
00003 found that the extra years do not help generators and hurt demand, so the training start is a
setting for each technology. It outputs conditional quantiles of power given the weather, which
cover what remains uncertain when the weather is known (cloud flicker, unmetered load, site
variation). The forecast chain uses the per-member median as the member's point value and, as a
baseline, the mixture of the quantiles over members. **CAMS on disk covers the six solar farms and
the 22 demand series of the experiments:** hourly point downloads from 2019-09-01 in
`data/studies/downloads/reanalysis/CAMS`, fetched for [issue #784 (Throwaway experiment: does the
beam/diffuse split help an XGBoost PV
forecast?)](https://github.com/openclimatefix/nged-substation-forecast/issues/784) and [PR #1201
(Download and validate CAMS irradiance at the demand series'
locations)](https://github.com/openclimatefix/nged-substation-forecast/pull/1201). Stage 1 for 2,500
series needs CAMS at every series' location, and the service is a point API (see [the data-sources
note on using the point
API](https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/#cams-use-the-point-api-not-the-gridded-product)),
so that is a fetch of its own. The experiments so far run stage 1 on the 31-series row set.

**Stage 1 is a perfect-prognosis model, and that is a risk.** Trained on analysis weather, it learns
the sharp relationship that satellite irradiance supports (CAMS 5.09% MAE against ERA5's 9.08% on
the solar studies' row set), and analysis cloud carries about twice the gain forecast cloud does. It
will lean on variables that are much less informative in a forecast. A variance-preserving stage 0
does not close that gap. The repo's explainer on [why a power model trained on past weather does not
hedge](https://openclimatefix.github.io/nged-substation-forecast/techniques/probabilistic-forecasting/#a-power-forecasting-model-trained-on-past-weather-does-not-hedge)
names the same trade-off as perfect prognosis against model output statistics, and cites [Marzban,
Sandgathe and Kalnay (2006)](https://doi.org/10.1175/MWR3088.1) for model output statistics beating
perfect prognosis on mean squared error. Experiment 00001 found a gain over an ENS-trained model up
to day 6 and none beyond, which is consistent with that risk. The deciding contrast stays the full
chain against a single-stage model trained directly on ENS, by lead band, and a stage 1 trained on
ENS early leads remains an arm.

**Lagged power enters stage 2, as lagged power and as lag anomalies.** A lag anomaly is the observed
power at a lag time minus stage 1's expectation at that time, given the weather then. It tells stage
2 how far recent power sat from what the weather explains, which is the "delta" that stage 2 needs
to read the lags. Experiment 00005 supports this placement over a separate persistence stage that
predicts the residual from the lags, as the first column of the evidence shows. The roadmap already
has a related idea ([weather-delta compensation for power
lags](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#weather-delta-compensation-for-power-lags-an-implicit-handle-on-unmetered-generation)).
Three risks apply. The lag anomalies of 00005 are ceilings, because serving has ERA5T about 5 days
late and CAMS about 1 day late. Flagged or stuck-meter lags enter as nulls. A lag no longer than the
lead is nullified, so the lead decides which lags exist. Stage 1 alone stays lag-free and lead-free,
and reusable for aim 3.

**Stage 2 is one shared calibrator per product, followed by a learned blend.** The calibrator is one
function with identical feature names for every product: that product's mean, spread, and
percentiles of stage 1 power across its members, the lead, and the product-independent lagged power
and lag anomalies. It is fitted on the pooled rows of all products, with a multiplicative spread
factor per product, and it outputs a location plus a positive scale times a fixed quantile shape.
The blend combines the calibrated distributions of the products that are present. A product that is
absent gets no weight and the other weights renormalise, so the stage always runs and a missing
product is a degradation, not an exception. A deterministic product has zero spread, so its scale
term needs a lead-dependent intercept. A product with a short horizon fades its weight to zero
towards its last lead. ENS alone is the one-component case. Stage 1's values on the ENS-era months
are forward-chained out-of-fold, or stage 2 would see stage 1 predictions that are too good.
Experiment 00005 trained the calibrator as one multi-quantile XGBoost model per series at 9 levels,
on every third daily ENS run.

**A new product joins stage 2 through a spread factor and a blend weight, not through new columns.**
An XGBoost model given an NWP identifier splits on it freely, so the identifier cannot act as a
scale adjustment only. Separate feature columns per product also do not transfer what the model
learned: splits learned on the ENS spread column do not apply to a UKV spread column. Null versus
non-null also acts as a date flag, because the new product's columns are non-null only in the most
recent months, so the trees can split on "recent month" and absorb any drift in that period. A
3-month overlap is millions of rows but only about 30 to 45 independent weather episodes (the
estimate in the training-history roadmap). The studies keep one model with per-product columns as a
control arm (the "per-product-columns control"), which a new product retrains in full. The shared
calibrator above is the alternative the studies test against it.

**Challenger: a single-stage ENS champion with a per-product correction.** The champion is a
quantile XGBoost model trained on ENS only, with stage 1 statistics, lead,
lagged power, and lag anomalies as inputs. Each additional product adds a correction and the
champion is never retrained. The correction shifts the champion's location by `a × Δ` and its
log-scale by `b × |Δ| + c × (the product's spread)`, where `Δ` is the product's stage 1 median minus
the ENS stage 1 median, and `a`, `b`, and `c` vary smoothly with lead, are pooled per technology,
and shrink towards zero. A missing product gives zero correction, so a product enters safely by
construction. The correction cannot learn interactions between a product and the lags, which is
probably a small loss (the lead column gained only 0.05 points in quick-wins) but is unmeasured. The
long ENS record supports this design: in the study of [Lang et al.
(2020)](https://doi.org/10.5194/npg-27-23-2020), on 2 m temperature at central-European stations,
training on several years beat a sliding window of recent days, including across an ECMWF model
change. This design is also the baseline that stage 2 must beat or tie before stage 2 earns its
place.

**The blend is learned, with few parameters, because a fixed blend is not calibrated by
construction.** An equal-weight mixture of individually calibrated distributions is generally
over-dispersed ([Ranjan and Gneiting (2010)](https://doi.org/10.1111/j.1467-9868.2009.00726.x) show
that a weighted average of distinct calibrated forecasts is uncalibrated, in a case study of binary
probability-of-precipitation forecasts), and averaging quantiles re-creates under-dispersion (see
the
[explainer](https://openclimatefix.github.io/nged-substation-forecast/techniques/probabilistic-forecasting/#the-tempting-shortcut-that-doesnt-work-averaging-the-quantiles)).
The beta-transformed linear pool fixes this with two shape parameters and nests the plain linear
pool (when both equal 1); [Gneiting and Ranjan (2013)](https://doi.org/10.1214/13-EJS823) extend it
to full predictive distributions. We have read abstracts and secondary summaries of both papers, not
the full texts. Stage 2b has two learned parts, fitted on held-out months: one weight per product,
smooth in lead and shrunk towards equal weights when the overlap is short, and the two shape
parameters, smooth in lead. A mixture already widens when the product medians disagree, so a
separate disagreement input is added only if a study shows the beta-transformed pool is still
under-dispersed when products disagree. The control arm is a fixed equal-weight mixture. The studies
score both with the PIT histogram and coverage by lead band.

**Width must be monotone in spread, and XGBoost does not guarantee it.** XGBoost monotone
constraints bound a prediction against a feature, not the difference between two quantiles (both
bounds can rise while the gap shrinks). The studies therefore parameterise the calibrated
distribution as a location plus a positive scale times a fixed quantile shape, with the scale
monotone in spread, and test the width property directly on held-out data.

**Pooling stage 1 quantiles over members is a valid baseline, not double counting.** If stage 1
estimates the distribution of power given the weather and the members are draws of the weather, the
mixture over members follows from the law of total probability. The quick-wins study's 57% coverage
is more likely in-sample fitting, which out-of-fold stage 1 removes. Stage 2 must beat the mixture
baseline (with and without a simple spread rescaling) to earn its place. The repo's [double-counting
caveat](https://openclimatefix.github.io/nged-substation-forecast/techniques/probabilistic-forecasting/#caveat-double-counting-weather-uncertainty)
applies to models trained on forecast weather, whose learned spread already includes some forecast
error. Stage 1 trains on analysis weather, so its spread does not. The mixture itself is the
[pooling
recipe](https://openclimatefix.github.io/nged-substation-forecast/techniques/probabilistic-forecasting/#the-fix-formally-a-mixture-of-conditional-distributions)
in that explainer.

**One coherent distribution comes out of stage 2 and ECC.** The marginal distribution per series and
valid time is stage 2's quantiles. ECC-Q needs one quantile per member, so the coupling step
interpolates the 19 stage 2 quantile levels to the levels i/52 for i = 1 to 51, with a stated tail
rule beyond the outermost levels, and assigns them to members by the rank of each member's stage 1
median. Ties at night-time solar, where every member is zero, need no rule because every assigned
quantile is also zero. Other ties (for example when weather is missing and the medians coincide)
would, if broken by member identifier, give perfect rank correlation across the affected series and
times and distort aggregate uncertainty, so they break by a dependence template instead: the ranks
of the members' stage 1 medians at neighbouring times, or a template taken from history (the
Schaake shuffle). The studies test the template with the coherence scores. At short leads members
are nearly identical and ranks are mostly noise, so the studies score coupling by lead band with
the variogram score and include the Schaake shuffle there. The output is always the 51 ENS
members. Other NWPs enter stage 2 as columns and do not add members, which removes the question
of how to pool members of different NWPs.

**Targets for solar and wind power.** Stage 1 could predict a ratio for solar: power divided by
clear-sky PV power, multiplied back by the known half-hourly clear-sky curve, so the output stays in
megawatts ([roadmap
section](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#predict-pv-as-a-fraction-of-clear-sky-pv)).
Quantile loss in ratio space needs weights equal to the clear-sky power so that it equals the
megawatt loss. Rows with zero clear-sky power are dropped. The per-site clear-sky scale is unknown
for embedded PV, so it comes from a trailing (not centred) rolling quantile of clear-sky-normalised
output, or from a physics model for solar farms. For wind, the analogue is a `base_margin` from a
power curve of forecast wind speed, with stage 1 learning the residual. The ratio target is tested
once, as a stage 1 arm. Multiplying quantiles by a positive clear-sky value preserves their order,
so the coupling step is unaffected.

**Lead time.** Stage 0 (if built) sees all leads and takes the lead as an input. Stage 1 takes no
lead. Stage 2 takes the lead, and the lead also decides which lag features exist: a lag shorter than
or equal to the lead is nullified. Lead buckets (for example day 0 to 1, 1 to 6, 6 to 15) are one
arm in a later experiment. In quick-wins the lead column gained 0.05 points, so a bucketed design
must clear a low bar. The risks are thin data per bucket and quantile jumps at bucket boundaries.

**Training scenarios and cadence.**

- **From scratch:** take the cleaned power history from `clean_nged_power_data`. Fit stage 1 on ERA5
  and CAMS, from the training start chosen for the technology, with forward-chained out-of-fold
  predictions: each block is predicted by a model trained only on earlier blocks, with a purge gap
  of at least one week. Run the adapter then stage 1 per ENS member over the NWP era, using the
  out-of-fold stage 1. Fit stage 2 on the per-product statistics. Score end-to-end CRPS on held-out
  months.
- **Monthly retrain:** keep stage 1 frozen, because its predictions for the new month are out of
  sample. Choose between the candidate and the incumbent stage 2 by rolling-origin evaluation: fit
  each on months up to a cut-off and score both on the following months, which neither saw. Only
  after the promotion decision, refit the winner on all months. Scoring the newest month after
  fitting on it would favour the candidate. Stage 1 retrains yearly, or when cleaning rules change
  or sites are added.
- **A new NWP with short history:** add an adapter table. Stage 1 is untouched. Stage 2's calibrator
  keeps its form: the new product's rows join the pooled fit with one new spread factor, and the
  blend gains the product's weight, shrunk towards equal weights. Study 3 measures how few months of
  history this needs. The single-stage challenger instead adds a correction (a few numbers per
  technology) to a frozen champion, and the per-product-columns control retrains in full.

**Leakage rules.** Stage 1 must not train on months that stage 2 or the end-to-end score uses for
validation, and ERA5 overlaps the ENS era. Forward-chained out-of-fold predictions replace a
two-sided cross-fit, because a two-sided fit would let stage 1 see later capacity, later embedded
PV, and later cleaning outcomes, giving residuals smaller than those it meets when serving. Lag
anomalies are built from the source and latency serving will have: ERA5T lands about 5 days late and
CAMS about 1 day late, so short lags use the analysis proxy (the early leads of the latest ENS run),
whose errors exceed ERA5's. Experiment 00005 did not do this, so its lag results are ceilings.

**Regime changes in the long history, including the COVID lockdowns.** Stage 1 trains on power from
late 2019 to 2026, which includes the COVID lockdowns (about 2020-03 to 2021-07, roughly 1.3 of the
5.5 winters). The lockdowns distort the occupancy and calendar response of demand far more than the
weather-to-power response of wind and solar farms, so the question applies to demand series and
net-load substations, not to generation. **The docs already contain ideas for how to represent the
lockdowns and for which datasets to ingest, but nothing is decided and nothing is ingested.** The
[era covariates section of the training-history
roadmap](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#era-covariates)
lists the candidate ways to represent the lockdown: a scalar in [0, 1] sourced from the Oxford
COVID-19 Government Response Tracker (a daily UK stringency index) or from Google's COVID-19
Community Mobility Reports (closer to the causal driver, thinly supported by one national preprint);
recency sample weights; and excluding the period. Both candidate series ended in 2022, so whether
they are still downloadable is unchecked, and nothing has been downloaded. The section also records
three cautions: the evidence is national or building-level and none of it is a distribution
substation; a scalar that returns to 0 cannot say that the post-lockdown world differs from the
pre-lockdown world; and the scalar breaks under a global model unless a customer-mix covariate
exists. The design consequences are:

- **Stage 2 is unaffected by construction.** It trains on the ENS era (2024-04 onward), which has no
  lockdown.
- **Stage 1 is where the choice lives.** The forward-chained out-of-fold run predicts the 2020-21
  blocks with models that never saw a lockdown, so those blocks carry large residuals. Those
  residuals matter only for the reuse cases (the roadmap already names the lockdown as the largest
  phantom event for the switching baseline, and as a free labelled test case) and not for stage 2.
- **The cheap arms have run for Primary substations (experiment 00006).** A 0/1 indicator for
  2020-03-23 to 2021-07-19, set to 0 in all forecast data, lowered the selection-fold error by 0.26
  points, and it helped grid supply points most. Recency weights helped more, and a 2023-09 training
  start helped most on the confirmation year only. No Oxford or Google series has been downloaded,
  and the results give no reason to download one yet.

**Alternatives considered.**

- **A learned weather-correction stage ("stage 0")** that maps each NWP towards ERA5 and CAMS,
  member by member, before stage 1. It is a further learned artifact, a variance-preserving
  correction cannot close the perfect-prognosis gap, and a climatological quantile map has known
  defects (it does not fade with skill, it injects incoherent variance from coarse steps, and it
  breaks consistency between variables). Built only if the second-NWP study shows stage 2 failing.
  If built: fit at the NWP's own step width, correct wind speed and clear-sky index and not their
  components, smooth over lead, exclude the validation months, and include an identity arm.
- **Merge stages 0 and 1** (pretrain on ERA5 and CAMS, fine-tune per NWP with power): needs power
  data per NWP, which is the short-history problem, and suits a neural network better than XGBoost
  (an XGBoost warm start can only add trees). Studied only if stage 2 fails on a second NWP.
- **Pool the products as exchangeable members of one ensemble:** a deterministic UKV would get a
  weight of 1/52. [Hagedorn et al.
  (2012)](https://www.ecmwf.int/en/elibrary/74701-comparing-tigge-multimodel-forecasts-reforecast-calibrated-ecmwf-ensemble-forecasts)
  found that a nine-system multi-model ensemble did not beat the ECMWF ensemble alone on
  extratropical temperature, and that calibrated ECMWF often matched the multi-model ensemble. The
  gain from blending products may therefore be smaller than hoped for solar, while the project's
  matched-lead study found a wind gain of 0.184 points of capacity.
- **Weather-space correction fitted on power error** through the frozen stage 1, as the
  [training-history
  roadmap](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#the-calibrator-can-act-on-power-or-on-weather)
  proposes, instead of towards ERA5 and CAMS. The correction keeps each member's rank. The
  weather-space correction is an arm if stage 0 is built.
- **A persistence stage ("stage 1b")** that predicts the residual of power against stage 1 from the
  residuals at the forecast origin and earlier, trained on ERA5 and CAMS with simulated origins and
  leads. Experiment 00005 found that lagged power in stage 2 beat it by 0.84 points, and that adding
  it on top of stage 2 gained 0.06 (both ceiling results). Revisit only if serving-latency lag
  anomalies weaken stage 2.
- **The single-stage ENS champion with a per-product correction:** the challenger above. It drops a
  stage and is the baseline that stage 2 must beat or tie.
- **A physics stage 1** (differentiable physics, version 2): drops into the same slot with the same
  interface and has few parameters, so it suits new sites. Not studied now.
- **An NWP better than ERA5** (UKV at short leads, for local wind and cloud): the canonical
  past-weather estimate becomes a per-variable choice or blend, settled by the ceiling test.
- **Station observations** (MIDAS Open): an offline referee and a candidate component of the
  past-weather blend. No live feed exists.
- **Direct CRPS training:** a guess, not a sourced figure, puts its gain below 0.2 points over
  pinball loss at many levels. Runs only if the PIT histogram stays U-shaped.

**Interaction with switching detection, effective capacity, and metering faults.** All three want
"expected power from analysis weather and calendar", and all three share one risk, a loop between
detection and training. The guard is the same each time: a robust fit, out-of-fold expectations, one
versioned stage 1 artifact, and a rule-based detector first.

- **Switching-event detection:** the roadmap's [approach 1, the two-stage
  forecaster](https://openclimatefix.github.io/nged-substation-forecast/roadmap/switching-events/#approach-1-the-two-stage-forecaster),
  is nearly this design. Stage 1 is the baseline and its residual is the switching signal. The
  residuals feed stage 2 as lag-anomaly features (the existing idea of [residual lag features from
  the switching-detector
  baseline](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#residual-lag-features-from-the-switching-detector-baseline)),
  and a recent detected event could widen stage 2's interval. Live detection runs on the analysis
  proxy.
- **Effective capacity:** the PV ratio target needs a per-site clear-sky scale, which is an
  effective-capacity estimate, so capacity estimation feeds stage 1 and stage 1 feeds the cheap
  capacity baselines. The scale must be causal. A physics stage 1 would make capacity a fitted
  parameter.
- **Metering faults:** large residuals against stage 1 could flag stuck meters and spikes, as a
  layer after the existing rule-based cleaning. Stage 1 would otherwise learn the faults it should
  detect, and multi-month faults span the blocks, so cleaning recall is tested on the known faults
  from [PR #1159](https://github.com/openclimatefix/nged-substation-forecast/pull/1159) before any
  reuse.

## 5. The studies and experiments

**Work is split three ways.** Infrastructure (studies 1 and 2) builds the adapter and the scoring
harness. Studies (3, 4, and 5) each answer several questions or need new data, and run under the
`study` skill. The questions about stage 1 and stage 2 are ml-experiments, run under the
`ml-experiment` skill, tracked in the [research
log](https://openclimatefix.github.io/nged-substation-forecast/ml-experiments/research-log/), and
managed by the ML experiment manager, who allots the GPU and CPU slots. Each study and experiment
pre-registers a few planned contrasts. Every other comparison is exploratory and is reported as
such.

### Infrastructure and studies

- **Study 1: complete the conventions adapter.** The radiation and wind rules have shipped.
  Remaining: step widths for hourly-stepped NWPs, the same adapter for ERA5 and CAMS, the per-NWP
  convention table, and the validation check, all with unit tests on synthetic input.
- **Study 2: scoring harness and coverage test.** One scorer for every output: pinball loss at 19
  midpoint levels plus a stated treatment of the tails beyond 0.025 and 0.975, and the same
  estimator for ECC-coupled members (the fair-CRPS factor of 9/8 can flip a comparison). PIT;
  coverage by lead band and series type; a synthetic spread-factor positive control; the in-sample
  against out-of-sample coverage test on the saved `e1_quantile` samples; one coherence score (CRPS
  of summed solar farms in a weather cell), with the energy and variogram scores added once coupled
  members exist. Experiments 00002 and 00005 use their own scorers; this study replaces them with
  one.
- **Study 3: how few months of history does a new product need in stage 2?** Fit the shared
  calibrator and blend with products genuinely different from ENS: ENS before and after the 12
  November 2024 upgrade treated as two products, the ENS control member as a deterministic product,
  and a truncated long-history ensemble such as GEFS. The first two test adaptation inside one model
  family. They do not test transfer to a different family such as UKV or MOGREPS-UK, so the
  short-history claim is accepted only after a different family passes (GEFS here, or the second NWP
  in study 4). Until then the claim stays unresolved. Score a learning curve at 1, 2, 3, 6, and 12
  months of history, rotating the truncation window across at least two starting seasons, scoring by
  season, and reporting in-season and out-of-season gaps separately. **Safe entry:** at 1 and 2
  months, the forecast with the new product must be equivalent to ENS alone (the bootstrap interval
  inside ±0.15 points). Include an hourly-step transfer test, and stage 2 against the single-stage
  challenger. The `tcc` (total cloud cover) natural experiment (served from 21 November 2025; [issue
  #1157 (Ingest total cloud cover (tcc) from Dynamical.org's ECMWF IFS
  ensemble)](https://github.com/openclimatefix/nged-substation-forecast/issues/1157)) needs its own
  split, for example training November 2025 to March 2026 and scoring later, or it is dropped.
- **Study 4: a real second NWP** (UKV or MOGREPS-UK; needs an ingest issue first). Includes column
  dropout in stage 1 and withheld variables for the missing-variable path, and the fade-to-zero rule
  near a short-horizon product's last lead. A learned weather-correction stage is built only if this
  study shows stage 2 failing.
- **Study 5: longer ENS history for stage 2** (optional, parallel). Uses the ENS backfill from
  Source Cooperative; needs approval against the £30 egress cap.

### Experiments on stage 1 and stage 2

**Five experiments are done, and the research log is the record.** 00001 compared stage 1 with an
ENS-trained model for solar and wind, 00002 compared stage 2 with the pooled mixture for solar and
wind, 00003 repeated the first question for demand, 00005 asked where lagged power enters, and 00006
asked which inputs stage 1 needs for demand. (00004 was a reservation that no experiment used.) The
remaining experiments, in the order the evidence calls for, are:

- **Lag anomalies from serving sources.** Rebuild the 00005 comparison with lag anomalies built from
  ERA5T and CAMS at their serving latency, and from the analysis proxy, so the ceiling is replaced
  by a number serving can reach. This decides how much of the 1.80 points survives. This is the
  first deliverable.
- **Summaries of lagged power** (the minimum, mean, and maximum of the latest 24 hours and of the
  latest day on the same weekday; research-log item P1), placed as 00005 finds best.
- **The solar and wind single-stage baseline.** The best single-stage arm must beat
  `manual_heuristic` (7.2% CRPS) before any multi-product machinery is built. Arms: control member,
  ensemble mean, mean plus spread and percentiles, spread rescaling, and ensemble model output
  statistics (EMOS).
- **Calibration of the pooled mixture,** with wind first, because the closed Codex study found the
  pooled wind quantiles under-covering (solar 80% intervals covered above the label and 90%
  intervals below it). Arms: spread rescaling against the learned calibrator, a location-and-scale
  parameterisation against a free quantile model, and the PV ratio and wind `base_margin` targets.
- **Perfect-prognosis check,** a stage 1 trained on ENS early leads, and a ceiling test with ERA5,
  CAMS, and each NWP's short-lead fields. The 00001 result that the gain ends at day 6 makes this
  the check on the long-lead risk.
- **Coupling:** ECC-Q (with the interpolation, tail, and tie rules above) against the Schaake
  shuffle at short leads, scored by lead band. Later arm.
- **The shared calibrator against the per-product-columns control, and the learned blend against a
  fixed equal-weight mixture,** which need a second product and so wait for study 3 or 4. Later arm.
- **Regime and drift arms** (recency weights, the lockdown indicator, and training starts) for
  demand, which 00006 started.

**Deciding contrasts.** The pooled stage 1 mixture is the baseline for the probabilistic questions.
Stage 2 stays only if it beats the pooled mixture with lags or with several products, or ties it and
helps a short-history NWP in study 3 or 4, and it must also beat or tie the single-stage model with
the same inputs. If stage 1 adds nothing on the ENS era, stage 1 leaves the forecast chain and
remains an offline artifact for aim 3.

**Decision rule.** A planned contrast passes when the mean paired difference is at most −0.15 CRPS
points, the month-resampled bootstrap interval is wholly below zero (at a Bonferroni-adjusted level
across the study's planned contrasts), at least 9 of 12 months agree on fold A, and the sign matches
on fold B. Fold A trains on April 2024 to June 2025, as in quick-wins. Solar-specific contrasts
count daylight-weighted months, or March to October only, stated in advance, because winter solar
differences are near zero. Fold B is weak confirmation because the folds share series and weather
regimes, so each finalist is also scored once on the untouched holdout from July 2026 onward.
**Equivalence** means the bootstrap interval lies inside ±0.15 points. The intervals come from
`studies.bootstrap`, which resamples whole months and returns intervals but no p-values, so Holm's
procedure (which needs p-values) is replaced by the more conservative Bonferroni level. **A stage
stays** only if it beats the previous design by the rule, or is equivalent to it and helps a
short-history NWP in study 3 or study 4.

**Constraints.** Production-system data plus ERA5, CAMS, and (if approved) the ENS backfill. No
production code changes before a model release. NGED generators are anonymised in every published
figure. Validation months are held out of every stage. Evidence is 31 series, so any claim about
solar rests on 6 farms in one weather region and any claim about wind rests on 3 farms.

## 6. Documentation plan

**Recommendation: one roadmap page per candidate design, with the design's evolution kept in git.**
The prose rules say docs describe the present and that history lives in git, so a page per revision
would turn the docs into a changelog. A page per live alternative does not. The documentation PR
that follows approval of this plan contains:

1. **A new page, `docs/roadmap/staged-probabilistic-forecast.md`,** status 🔬 Research. It owns the
   production architecture of the candidate design: the aims, the stage table, the training
   scenarios, and the study order. It links the **research log** for the evidence and does not carry
   an evidence table of its own, so that the verdicts live in one place. It has a **"Considered but
   rejected"** section, as the switching-events roadmap does, where each removed stage keeps the
   reason and the study that removed it. It ends with an "Implementation details (deleted when this
   ships)" section holding the study sequence. A verdict updates the design text in place.
2. **Test entries in `docs/design-philosophy/engineering-hypotheses.md`** for the falsifiable
   claims, for example that a shared spread-skill relationship plus a per-NWP spread factor is
   equivalent to a full per-NWP fit with 3 months of history. Labels are append-only and never
   renumbered.
3. **Pointers, not rewrites,** in the pages that discuss overlapping ideas, each saying that the new
   page owns the architecture: `docs/roadmap/training-history.md` (keeps its own study, evaluation,
   and era covariates), the sections of `docs/roadmap/xgboost-improvements.md` on ensemble
   statistics, several NWP sources, and the PV ratio target, Phases C and D of
   `docs/roadmap/metrics-and-leaderboard.md`, and the switching-events, capacity-estimation, and
   data-cleaning roadmap pages.
4. **The roadmap index and navigation** list the page, and `docs/documentation-guide.md` gains one
   row for "a design under study" explaining the one-page-per-candidate-design convention.
5. **Lifecycle.** When the studies finish, the roadmap page records the verdicts and the surviving
   design. When a reviewed production implementation ships, the design moves to a
   `docs/architecture/` page describing what is built, and the roadmap page is deleted. A refuted
   candidate shrinks to a short rejected note linked from the index.

**Shared components move out of design pages** so that two candidate designs would not duplicate
them: the conventions adapter stays on the [NWP variable conventions
page](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/),
and the scoring harness and the ratio target stay with the techniques and the XGBoost roadmap.

## 7. GitHub issue plan

**Issues are created only after this plan is approved.** Each new issue gets labels, the org issue
Type, project 33 membership with Status Todo, Project NGED, and Area ML, the `JackKelly` assignee,
and a position in its parent's sub-issue order, per the repository's issue checklist. A new issue
body may be a link to the rendered roadmap section when the docs already hold the plan.

1. **Tracking issue, "Tracking: a staged probabilistic power forecast",** a sub-issue of [issue #896
   (Studies: weather products, forecast comparisons, and study
   infrastructure)](https://github.com/openclimatefix/nged-substation-forecast/issues/896). The body
   links to the roadmap page and the research log. It lists the five children in order, and it says
   that the ml-experiments are tracked in the research log and not as issues.
2. **Five child issues,** in this order, with these blocking links:
    - Study 1: "Study: complete the NWP conventions adapter and per-NWP convention table."
    - Study 2: "Study: scoring harness and coverage test for probabilistic power forecasts."
    - Study 3: "Study: how few months of history does a new product need in stage 2?" (blocked by
      study 2; the lag experiment with serving-latency lag anomalies should finish first).
    - Study 4: "Study: a second NWP with stage 2" (blocked by study 3 and by an ingest issue for the
      chosen NWP that does not yet exist).
    - Study 5: "Study: longer ECMWF ENS history for stage 2" (optional; check whether [issue #959
      (Extend ECMWF ENS training
      history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959) already
      covers it, in which case link it instead of creating a duplicate).
3. **Edit [issue #1156 (Study: the production feature faults, and calibrating the probabilistic
   power forecast)](https://github.com/openclimatefix/nged-substation-forecast/issues/1156):**
   remove Part 2, its experiments table, its gates, and its maintainer decisions, and add a pointer
   to the tracking issue. Parts 1b and 3 stay. Part 1 is already done.
4. **Resolve ownership with [issue #999 (Study: how many months of a new weather product's history
   before it adds skill to a wind or solar
   forecast?)](https://github.com/openclimatefix/nged-substation-forecast/issues/999).** The
   training-history roadmap page already designs much of the short-history question, so the tracking
   issue and #999 must say which one owns it. Study 3 either replaces #999 or becomes its child.
5. **Before editing any existing issue,** check its assignee and leave it alone if anyone other than
   `JackKelly` is assigned.

## 8. Reviews already run, and how they were triaged

**Two fresh Opus reviewers read an earlier draft: one for simplicity, one for scientific validity.**
This section is a record of what changed; it is not part of the design. "Study 2a" and "study 2b"
in the records below are experiments 00001 and 00002.

- **Accepted from the simplicity review:** redo the compute estimate from measured per-series rates;
  drop an NWP identifier as a free split variable; trim the arms (member-crossing, Schaake shuffle
  outside short leads, isotonic and conformal calibration, stage 2 in ratio space); merge the
  ceiling test into study 2; add a design-principles paragraph; defer a learned stage 0; shorten the
  critical path to four studies; fix the output at 51 members.
- **Accepted from the scientific review:** the pooling-is-double-counting claim was wrong;
  forward-chained out-of-fold predictions; lag anomalies built from the source serving has; the
  perfect-prognosis risk and the early-lead stage 1 arm; the ECC-Q specification; genuinely
  different products for the short-history test with a rotated truncation window; pre-registered
  contrasts with Bonferroni-adjusted intervals and an untouched holdout; daylight-weighted solar
  months; a separate `tcc` split; the ratio-target conditions; a cleaning-recall test before reuse.
- **Accepted from a later external review (GPT-6 Astra) of the pushed plan:** stage 2 now says how
  knowledge transfers between products (Designs 1 and 2) and the short-history claim stays
  unresolved until a different model family passes; the monthly promotion uses rolling-origin
  evaluation; the tie rule uses a dependence template; width monotonicity is tested directly and the
  scale is parameterised; the architecture page waits for a shipped production implementation.
- **Accepted from a later Opus review of how the plan adds short-history products:** stage 0 moves
  to the alternatives; aim 3 no longer justifies the serving structure; the disagreement input and
  the lag inputs leave the blend; study 3 gains the safe-entry test and a learning curve; the
  single-stage model must beat `manual_heuristic` before multi-product machinery is built. The
  reviewer read abstracts and summaries, not full texts, of the cited papers. Its recommendation to
  make the single-stage model the lead design was reversed by the maintainer, as the next bullet
  records.
- **Decided by the maintainer as the results arrived:** the staged chain is the best-guess design,
  because stages with well-defined roles are easier to explain, test, and extend than one model that
  does several jobs. Stage 2 stays as a role that must be shown under several products or lags,
  because it added nothing for ENS-only solar and wind. Experiment 00005 put lagged power in stage 2
  and dropped the separate persistence stage ("stage 1b"). Experiments 00001 and 00003 made the
  training start a setting for each technology. The ENS-trained control memorised calendar position
  in the first run of 00001, so every experiment now includes a climatology arm and a de-memorised
  control. The research log and the `ml-experiment` and `release-model` skills replace the evidence
  table, and the work splits into infrastructure, studies, and ml-experiments.
- **Review of this version:** a fresh Sonnet reviewer checked every quoted number against the
  write-ups and the cited pull requests. It found a wind under-coverage figure attributed to the
  wrong experiment, a skill that had not merged, numbering gaps in section 5, and leftovers of the
  earlier design. All were corrected. An Opus reviewer has not read this version.

## 9. Compute at version 2 scale (rough, unmeasured)

**Basis.** The quick-wins study measured, for 31 series on one GPU, 37 s (squared error) and 5 min
(9 quantile levels) for one model per series, 9.76 million training rows (production trains on the
control member, about 315,000 rows per series), 8 GB peak memory, and a feature frame of 5 min 19 s
and 5.3 GB, extrapolating to about 7 hours and 430 GB for 2,500 series in one process. That is about
9.7 s per series at 9 levels. The rates below assume cost grows linearly with series, rows, and
quantile levels. The last assumption is unverified. Read everything as an order of magnitude.

| Stage | Training at 2,500 series | Inference per daily run |
|---|---|---|
| Adapter | Not trained | Seconds to minutes, dominated by reading the NWP |
| Stage 0 (if built) | Not yet sized; depends on the form chosen | About 1,000 cells (unverified) × 51 members × 720 half-hours × about 10 variables, about 370 million values |
| Stage 1 | About 105,000 rows per series at 9 levels: about 3 s per series, so about 2 GPU-hours per fit; about 5 fits for the forward-chained run plus the final fit, so about 10 GPU-hours, yearly | About 92 million rows (51 members × 2,500 series × 720 half-hours): minutes on GPU, tens of minutes on 8 CPU cores |
| Stage 2 | One row per (init time, valid time), the production layout, with 2 to 3 years of ENS: about 15 to 25 s per series at 9 levels, so about 10 to 17 GPU-hours monthly if fitted per series; a per-type model would be cheaper but does not fit GPU memory at 2,500 series | About 1.8 million rows: seconds |
| ECC | None | Seconds |

**Stage 2 is the stage whose cost scales worst.** A monthly refit of 10 to 17 GPU-hours is a design
constraint. Options for the studies: fit stage 2 per series type with row subsampling, or on fewer
lead steps. Each NWP added widens stage 2's feature template and raises the cost. The feature frame
and input/output still probably dominate wall-clock time and need chunking by series.

## 10. Open questions for the maintainer

- Which uses need coherence across time, across series, or both, and does feeder-to-substation
  hierarchy data exist for scoring it?
- Is the target overall CRPS or exceedance at the 95th percentile? The answer decides between
  parametric and non-parametric calibration.
- The final 51-member output fits `PowerForecast`, whose `ensemble_member` already has a different
  meaning for each forecaster (for `climatology` it is the rank of a quantile level). Does the
  maintainer want a docstring line that says what the index means for the staged forecaster? The
  19-level quantile table and the stage 1 tables need their own schema in the experiment store, and
  the contract is not changed without asking.
- Stage 1 for the demand series now has ERA5 and CAMS at their locations. For the 2,500-series
  scale, is CAMS fetched at every series' location, given that the experiments found CAMS explains
  more of a Primary substation's residual than ERA5 shortwave (a post hoc result)?
- The closed Codex study found that pooled wind quantiles under-cover, and no merged experiment has
  measured wind coverage. Does the maintainer want a wind-specific calibration experiment ahead of
  the multi-product work?
- Which second NWP first, UKV or MOGREPS-UK, and which UKV product has usable history? (The CEDA
  archive differs statistically from the live feed.) Is a long-history ensemble such as GEFS
  obtainable for study 3?

## 11. Related docs and issues

Docs links are to the rendered site, under
<https://openclimatefix.github.io/nged-substation-forecast/>.

**The training-history roadmap page already designs much of this plan.** Its section [A new weather
product with a few months of
history](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#a-new-weather-product-with-a-few-months-of-history)
(study [issue #999](https://github.com/openclimatefix/nged-substation-forecast/issues/999)) proposes
[training one weather-response model on the long power history, then calibrating per weather
product](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#train-on-the-long-power-history-then-calibrate-per-weather-product).
That is stage 1 plus a per-product calibrator. This plan differs in four ways:

| Topic | The training-history page | This plan |
|---|---|---|
| Weather-space correction | A few smooth-in-lead parameters per variable fitted on **power error** through the frozen weather-response model, which keeps each member's rank, so no copula step | Stage 0 fits towards **ERA5 and CAMS**, without power data, and is built only if needed. The power-error version is an alternative arm |
| Power-space calibration | An EMOS wrapper forecaster, `BaseForecaster`-shaped, shared by three consumers | Stage 2, one model with lags and cross-NWP inputs, plus ECC |
| Which estimate of past weather | [A section on choosing](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#which-estimate-of-past-weather-to-train-on) | A per-variable choice or blend, settled by the ceiling test |
| Evaluation | [How the study evaluates the design](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#how-the-study-evaluates-the-design) and [traps to avoid](https://openclimatefix.github.io/nged-substation-forecast/roadmap/training-history/#traps-the-study-has-to-avoid) | Section 5's decision rule; read the traps list before the next stage 1 or stage 2 experiment |

**Related sections, by topic:**

- **Calibration and probabilistic forecasting:**
    - [Phase C, low-effort
      calibration](https://openclimatefix.github.io/nged-substation-forecast/roadmap/metrics-and-leaderboard/#phase-c-low-effort-calibration-after-b-proves-the-diagnosis)
      and the [probabilistic-forecasting
      explainer](https://openclimatefix.github.io/nged-substation-forecast/techniques/probabilistic-forecasting/).
    - [Issue #262 (Add percentile representations (Representations 2 & 3) to `PowerForecast` and the
      Delta write/read
      paths)](https://github.com/openclimatefix/nged-substation-forecast/issues/262), [issue #263
      (Quantile XGBoost model family (`reg:quantileerror`) emitting per-member percentile forecasts
      (Representation 3))](https://github.com/openclimatefix/nged-substation-forecast/issues/263),
      and [issue #264 (Linear-pool per-member quantiles into delivered percentiles (Representation
      2))](https://github.com/openclimatefix/nged-substation-forecast/issues/264).
    - [Issue #443 (Degradation-conditional interval calibration: conformal prediction per
      regime)](https://github.com/openclimatefix/nged-substation-forecast/issues/443) and [issue
      #715 (`manual_heuristic_calibrated`: EMOS-style calibration of the manual heuristic's 13
      analogues)](https://github.com/openclimatefix/nged-substation-forecast/issues/715).
    - [Issue #730 (Rebuild coherent scenarios from the pooled percentiles with ensemble copula
      coupling (post-v1))](https://github.com/openclimatefix/nged-substation-forecast/issues/730).
- **Ensemble statistics, several NWPs, and lead:**
    - [Ensemble statistics as
      features](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#ensemble-statistics-as-features-instead-of-member-by-member-rows),
      [several NWP sources as
      features](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#several-nwp-sources-as-features-v21),
      and [per-horizon-window
      models](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#per-horizon-window-models).
    - The encoders page on [blending several NWP
      sources](https://openclimatefix.github.io/nged-substation-forecast/techniques/encoders/#blending-several-nwp-sources).
    - [Issue #148 (Train on all ECMWF ensemble members (one by
      one))](https://github.com/openclimatefix/nged-substation-forecast/issues/148), [issue #91
      (Implement `XGBoost Iterator` (to allow us to train efficiently on large
      datasets))](https://github.com/openclimatefix/nged-substation-forecast/issues/91), and [issue
      #230 (XGBoost Tier-1 quick wins: lead-time feature, ordinal time, early stopping,
      holidays)](https://github.com/openclimatefix/nged-substation-forecast/issues/230).
- **Pre-training on estimates of past weather and longer history:**
    - [Pre-train on estimates of past
      weather](https://openclimatefix.github.io/nged-substation-forecast/roadmap/xgboost-improvements/#pre-train-on-estimates-of-past-weather).
    - [Issue #143 (Ingest ERA5 for gap filling, comparison arms, and the weather-abnormality
      climatology)](https://github.com/openclimatefix/nged-substation-forecast/issues/143), [issue
      #167 (Train XGBoost on estimates of past weather (CAMS, UKV, ERA5) to use the power history
      before 2024-04)](https://github.com/openclimatefix/nged-substation-forecast/issues/167), and
      [issue #959 (Extend ECMWF ENS training
      history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959).
- **Conventions and the adapter:**
    - [The NWP variable conventions
      page](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/),
      including [radiation
      resampling](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#radiation-resampling)
      and [interpolating components before deriving speed and
      bearing](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#interpolate-components-before-deriving-speed-and-bearing).
    - [Issue #526 (Resample short-wave and long-wave radiation over native
      intervals)](https://github.com/openclimatefix/nged-substation-forecast/issues/526) and [issue
      #525 (Store wind as u/v components rather than speed and
      direction)](https://github.com/openclimatefix/nged-substation-forecast/issues/525), both
      closed.
- **Targets and physics:** the [differentiable
  physics](https://openclimatefix.github.io/nged-substation-forecast/techniques/differentiable-physics/)
  technique page.
- **Reuse of stage 1:** the
  [capacity-estimation](https://openclimatefix.github.io/nged-substation-forecast/roadmap/capacity-estimation/)
  and
  [data-cleaning](https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-cleaning/)
  roadmap pages.
- **Evidence and process:** the [research
  log](https://openclimatefix.github.io/nged-substation-forecast/ml-experiments/research-log/),
  which holds the experiments' results, and the `release-model` skill (added by [PR #1200 (Add the
  release-model skill for cutting a numbered model
  release)](https://github.com/openclimatefix/nged-substation-forecast/pull/1200)), which describes
  how winners reach a numbered model release. Earlier evidence: the [lag-features
  study](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/lag-features/)
  and the [production feature faults
  study](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/production-feature-faults/).
- **Principles:** [inherent
  stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
  and the [design
  principles](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/).
