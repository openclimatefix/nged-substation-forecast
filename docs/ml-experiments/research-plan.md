# Research plan of the ML experiments

**This page lists the ml-experiments that are planned or proposed, sorted by priority.** The
[research log](research-log.md) holds the finished experiments, their verdicts, and the lessons so
far. The design under test is on the [staged probabilistic
forecast](../roadmap/staged-probabilistic-forecast.md) page. In this plan, CRPS is the continuous
ranked probability score, MAE is the mean absolute error, ENS is the ECMWF ensemble, ERA5 and CAMS
are estimates of past weather, stage 1a is stage 1 as the experiments train it, and ECC-Q is
ensemble copula coupling by sorting quantiles. The maintainer adds proposals to this page, and the
ML experiment manager orders them, merges proposals that test the same hypothesis, and discards the
ones an earlier result has answered.

Each planned experiment has a `result.md` with status `planned`, committed before any code runs. An
idea that has no folder yet is marked "proposed" and carries a tentative id, P1, P2, and so on,
which the sections below use as headings. A proposal receives its numeric id when its plan is
committed, and the tentative id then disappears. A lower number in the "Priority" column runs
sooner.

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [SW2-1](#sw2-1-a-fixed-scoreboard-and-an-injection-harness) | What event-level F1 do experiment 00037's detectors D0 and D1 reach on the labelled set and on injected transfers, by duration and size band? | 2 | SW2-0b (done: experiment 41, part 2) | Proposed |
| [SW2-2](#sw2-2-how-much-the-visible-events-hurt-the-best-guess) | What share of the best guess's quantile score and tail score falls in labelled switching half-hours, and in half-hours whose lagged power falls inside one? | 2 | SW2-0b (done: experiment 41, part 2) | Proposed |
| [SW2-3](#sw2-3-the-label-exclusion-ceiling) | Does excluding labelled switching from the lagless XGBoost model's training target lower its error outside switching, against a random exclusion of equal volume? | 2 | SW2-0b (done: experiment 41, part 2) | Proposed |
| [SW2-4 to SW2-9](#sw2-4-to-sw2-9-detector-iterations) | Which changes to the detector raise its event-level F1: a better residual, a retrain loop, paired-step intervals, a shape-aware step, node balance, or whitening? | 3 | SW2-1 | Proposed; one hypothesis per experiment, run in order |
| [SW2-10](#sw2-10-v2-in-miniature-a-linear-joint-fit) | Does one linear fit per trial Primary, with a weather baseline, embedded solar capacity times a PV shape, and a term per labelled transfer, beat the frozen v1 baseline outside switching and recover injected solar and transfers? | 3 | SW2-0b and SW2-3 | Proposed; the first test of a v2 piece against v1, with inputs from the PV-disaggregation line |
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |
| [PVDIS-3](#pvdis-3-how-much-forecast-error-comes-from-embedded-solar) | How much of the demand forecast's error does embedded solar explain, and does the XGBoost model already capture the solar it is given? | 2 | The maintainer's review of this plan | Proposed; part 3 of the PVDIS series |
| [PVDIS-4](#pvdis-4-a-capacity-series-that-only-rises-with-error-bars) | Can a weekly capacity series with non-negative increments recover injected capacity ramps and steps, with calibrated error bars, on the 8 MW Primaries? | 2 | The maintainer's review of this plan | Proposed; part 4 of the PVDIS series; can run beside PVDIS-3 |
| [PVDIS-5](#pvdis-5-forecasting-net-demand-as-demand-minus-forecast-solar) | Does a solar feature, or training on gross demand and subtracting forecast solar, lower or at least match the best guess's error while explaining the forecast? | 3 | PVDIS-3 and PVDIS-4 | Proposed |
| [PVDIS-6](#pvdis-6-the-mix-of-panel-orientations) | Can the share of east-, south-, and west-facing solar behind each Primary be identified? | 4 | PVDIS-4 | Proposed |
| [PVDIS-T](#pvdis-t-the-capacity-table-a-study) | What is the unmetered solar capacity behind each MW Primary over time, with error bars and outside checks? | 3 | PVDIS-4 | Proposed; a study under `docs/studies/`, not an experiment |

## Deferred until the first results are in

**These proposals wait for a decision on the shape of the production
model.** Each keeps its section below, and a result may promote it.

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P11](#p11-ensemble-statistics-as-inputs) | Do ensemble statistics (the member mean, spread, and percentiles of temperature and irradiance) beat control-member rows? | 5 | Nothing | Deferred; an ERA5-trained stage 1 cannot take ensemble spread as an input, and experiment 00018 found the staged chain level with N1 for demand |
| [P13](#p13-perfect-prognosis-and-ceiling-checks) | Does a stage 1 trained on ENS early leads beat the ERA5-trained stage 1 at long leads, and how much skill remains with ERA5, CAMS, or short-lead fields as inputs? | 5 | Nothing | Deferred; 00001 found no gain beyond day 6, and 9 generator series cannot resolve smaller gains |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |
| [P15](#p15-a-second-weather-product-already-on-disk) | For solar and wind, does a shared stage 2 calibrator with a learned blend let a second weather product already on disk (CEDA UKV or Open-Meteo) enter without retraining? | 5 | A decision on the shape of the production model | Deferred; run for solar and wind only, ENS against ENS plus one product, because demand depends less on weather than generation does |
| [P16](#p16-a-longer-ens-history) | Does the ENS backfill (2021-03 to 2024-03) give an extra validation fold, or improve the single-stage or stage 2 models? | 5 | The maintainer, who runs the fetch on a cloud machine | Deferred; tracked in [issue #959 (Extend ECMWF ENS training history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959), and the 51-member fetch cannot run autonomously because it needs a cloud machine in `us-west-2` |

## Proposals in detail

**Resolution limits.** The sections below flag the experiments whose planned contrasts the available
sample cannot resolve. The sample is 16 Primary substations, 9 generator series, and one validation
year with 12 months, and it resolves gains of roughly 0.3 points of capacity or more.

### P2: embedded-solar capacity from lagged residuals

**The embedded-solar proposal rests on one follow-up check of experiment 00006's residuals, which
motivated two different questions. Experiment 00007 then ran a first test and was inconclusive.**
The check was run by the 00006 session on CAMS irradiance at each demand series' own location, and
its figures and script are in the write-up of [PR #1202 (Experiment 6: which input features work
best for the ERA5-trained demand
model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1202). It was not
pre-registered. At Primary substation 12, the median-error Primary substation, the residual of the
chosen-feature stage 1a XGBoost model averaged -5.3 points of capacity from 08:00 to 16:59 in the
June, July, and August of the confirmation year (July 2025 to June 2026), against -1.0 at night. In
July 2025, a linear fit on 431 daylight hours explained 25.5% of the hourly residual's variance with
irradiance from the Copernicus Atmosphere Monitoring Service (CAMS), 3.8% with ERA5 shortwave, and
43.3% with both. Across the 14 Primary series that have CAMS values, CAMS alone explained more than
ERA5 alone in all 14: a median of 23.7% against 2.9%, and 39.8% with both. In one series the error
is a level shift that irradiance does not explain, and in another the residual correlates positively
with CAMS irradiance.

**The check cannot yet say whether the cause is embedded solar capacity or an error in ERA5's
irradiance.** At Primary substation 12 the joint fit gives coefficients of opposite sign and similar
size for ERA5 and CAMS, and CAMS minus ERA5 alone explains 33.9%, so much of the residual tracks the
difference between the two irradiance products. Stage 1a has already learned an irradiance response
from ERA5, so a fixed error in ERA5's cloud field would produce the same pattern without any change
in solar capacity. Daytime lighting load and behaviour on sunny days could also contribute. At
Primary substation 12, CAMS explains 25% of the variance on weekdays and 10% on weekends, so the
weekday mix matters too. The check has no time dimension, because it covers one month, one seed, and
one fold, and with hourly autocorrelation the effective sample is far below 431 hours. Because stage
1a already uses ERA5 shortwave, the comparison of 25.5% with 3.8% is not like for like, and the
partial R² of CAMS given ERA5 would be a fairer figure.

**Experiment 00007 was the cheap first test of an irradiance error against a capacity change.** It
added CAMS irradiance to ERA5 shortwave and fitted a static correction on one year of residuals. The
correction removed 46.6% of the next year's daylight squared residual against the 50% planned, and
one substation sets the pooled figure, so the verdict is inconclusive. A further test could compute
the three candidate statistics on residuals from walk-forward (out-of-fold) stage 1a fits, because
residuals from a model fitted on the same period are shrunk and would look different in operation.

**The three candidate statistics each compare what stage 1a predicted with what the substation
metered.** Each uses the residual, the metered power over capacity minus stage 1a's prediction at
the observed weather.

- **The solar slope:** regress the residual on irradiance over the trailing 14 to 28 days, in
  daylight hours only. A more negative slope means more embedded solar than stage 1a learned. The
  slope mixes capacity change, irradiance error, and other demand that varies with irradiance, so
  the regression could use the clear-sky index with day-of-week and temperature controls. Noise in
  the estimation irradiance biases the slope towards zero.
- **The sunny-minus-cloudy daytime residual:** the mean residual on sunny days minus the mean on
  cloudy days at matched hours, with a named clear-sky-index threshold. A day-minus-night residual
  is not a substitute. Commercial load is daytime-heavy, electric-vehicle charging and storage are
  night-heavy, and heat pumps are daytime and seasonal, so their trends would appear as false solar
  change.
- **The change from the same calendar window a year earlier:** this year's 28-day slope minus the
  slope in the same 28 days of the previous year. A ratio of the 28-day to the 365-day slope is
  unstable, because the slope is seasonal and the denominator can be near zero or change sign.

**The correction could be given to stage 2 as a linear offset, not as a tree feature.**
Gradient-boosted trees cannot extrapolate beyond their training range, and embedded solar capacity
grows steadily, so a product of the slope and the forecast shortwave would still be held flat by a
tree. A linear term, in stage 1b or in stage 2, would extrapolate correctly, and the tree feature
would be the comparison arm. The slope is estimated against observed irradiance and applied to
forecast irradiance, which differ in bias and scale, so the conversion has to be stated. The
past-weather source used at lag times has to be available at forecast time, because ERA5 arrives
about 5 days late, and training has to use that same source. CAMS is observed after the fact and has
no forecast, so the CAMS satellite-derived irradiance can feed the statistics only at lag times. The
CAMS irradiance at each demand series' own location comes from [PR #1201 (Download and validate CAMS
irradiance at the demand series'
locations)](https://github.com/openclimatefix/nged-substation-forecast/pull/1201), which covers the
22 series of experiment 00003 (16 Primary demand series, 4 bulk-supply and grid-supply point flows,
and 2 non-demand series, a battery and a biofuel generator). The experiment could compute each
statistic from ERA5 alone and from CAMS and ERA5 together.

**A decision rule could be fixed before any fit.** One candidate: the trend in the monthly change
from a year earlier is negative at 70% or more of the substations in the top quartile of registered
embedded solar capacity per unit of substation capacity, with a block-bootstrap 95% interval that
excludes zero, and it is absent in the night-time residuals. A second candidate: the linear-offset
arm lowers the confirmation-year daylight quantile score at Primary substations by at least 0.2
points of capacity, with a month-block bootstrap interval that excludes zero. Otherwise the
hypothesis is refuted. The thresholds are proposals for the planning step to confirm.

**Resolution limit.** The top quartile of 16 Primary substations by embedded solar capacity is 4
substations, so a rule of "70% of them" means 3 of 4, which no interval can support. The slope
statistics need more substations or a continuous measure of capacity.

### SW2: switching-event detection

**The SW2 line improves the switching-event detector first, scored by event-level F1 against a
labelled set of visible events, and leaves the use of the detector until the detector works.** The
detector serves two purposes later. The first is a lower forecast error, if switching periods act as
unexplained noise for the XGBoost model and corrupt its lagged power inputs. The second, and the
more important to NGED, is a forecast under the normal running arrangement (NRA), into which NGED
can add its own switching. The [switching events](../roadmap/switching-events.md) roadmap page
proposes building that forecast by excluding switching periods from the training target. An event
table for NGED is a third purpose, at lower priority. The production detector has to run on power
alone, because the switching logs exist only for the trial area, so the logs are used here for
evaluation only.

**The line reopens a question that experiment 00038 closed, because experiment 00038 tested a
different use of the detector.** Experiment 00038 masked detected switching from the history of the
error correction that runs after the XGBoost model, and the mask made the forecast worse. No
experiment has yet masked switching from an XGBoost training target or from a lagged input.
Experiment 00037 confirmed 37 of 612 logged closures of 2 hours to 60 days as visible in its own
residual, and scored its detectors on the 18 of them in the validation months. That reference set is
small, about 6 of the 37 are expected by chance, and the label was built from the residual the
detectors use, so a detector that shares the residual's blind spots is flattered by the label. The
line therefore starts by building a reference set independent of the detectors.

The full labeller design is in the private file `data/ml-experiments/briefs/sw2_labeller_design.md`.

#### SW2-1: a fixed scoreboard and an injection harness

**Score D0 and D1 from experiment 00037 on a scoreboard that every later detector experiment
reuses.** An event counts as found only when both its onset and its end match, each at the
detector's causal `known_from` stamp, because experiment 00038 found the end of a detected interval
to be where the damage happens. The tolerances and bands are proposals for the plan to confirm: an
onset within 2 hours and an end within 6 hours, duration bands of 6 to 24 hours, 1 to 3 days, and
over 3 days, and size bands of moved load under 10%, 10% to 25%, and 25% or more of the receiving
load. The injection harness follows experiment 00038's, with 1 to 3 receivers and durations drawn
from the confirmed closures, and adds a moved share that drifts during the event. Tuning uses odd
months and injections; even months score.

#### SW2-2: how much the visible events hurt the best guess

**Measure the share of experiment 00033's best guess's quantile score and tail score that falls in
labelled switching half-hours, and in half-hours whose lagged power falls inside a labelled event.**
The measurement needs no new fit, and it sets how much a detector could lower the forecast error.

#### SW2-3: the label-exclusion ceiling

**Train the lagless XGBoost model with labelled switching excluded from the target, without
exclusion, and with a random exclusion of equal volume, and score all three outside switching.**
SW2-3 is the label-exclusion family on the roadmap page. The labelled-against-random gap is the most
a detector could gain by cleaning the training target. The same gap is the first evidence for the
NRA forecast.

#### SW2-4 to SW2-9: detector iterations

**Each iteration tests one change to the detector against the SW2-1 scoreboard, tuned on injections
and odd months, scored on even months.** The order below is a proposal, by expected gain per hour of
work, for the plan of each iteration to confirm:

- **SW2-4:** a better residual, from the best lagless model available live, refitted with a robust
  loss, in place of `H0.5`.
- **SW2-5:** the train, detect, retrain loop, with two or three rounds of refitting the lagless
  XGBoost model without the detected intervals.
- **SW2-6:** paired-step intervals, which find the onset and its opposite return step as one
  interval with an end, with a prior on duration from the labelled set.
- **SW2-7:** a shape-aware step, which models the moved load as a share of the donor's expected
  profile, not as a constant MW shift. Experiment 00037 found the step changing by more than a
  factor of 2 within 8 of 26 confirmed closures of 6 hours or more.
- **SW2-8:** node balance, which tests whether the sum of the neighbourhood's residuals stays flat
  while single substations step, with a subset-sum over neighbours for a split.
- **SW2-9:** whitening, experiment 00037's proposed F4, an autoregressive model of order 1 per
  series and hour of day fitted to the residual.

#### SW2-10: v2 in miniature, a linear joint fit

**Fit one linear model per trial Primary that holds a weather baseline, embedded solar, and the
labelled transfers together, and score it against the frozen v1 baseline.** The model is net demand
equals the baseline, minus a capacity times a PV shape, plus, for each labelled transfer k, a share
times the donor's net power over transfer k's interval. With the intervals given, the fit is
linear. The result tells the project whether to start the v2 differentiable-physics work sooner.

- **The baseline term** is SW2-3's lagless XGBoost model, trained with labelled switching excluded.
- **The transfer intervals** come from SW2-0b, or from the logged closures padded by 24 hours if
  the labels are not ready.
- **The capacity and the PV shape** come from the PV-disaggregation line: experiment 00040's
  six-arm capacity with its bound, experiment 00039's capacity as a cross-check, and experiment
  00039's CAMS unit PV shape. A weekly capacity series replaces the experiment 00040 capacity if
  the PV line's weekly-capacity experiment runs. The 8 Primaries metered only in MVA are excluded.
- **Scoring** is on held-out months outside switching against the v1 baseline, and on experiment
  00039's injection harness extended with injected transfers that carry the donor's PV, where the
  true capacity and the true shares are known.

### PVDIS: separating embedded solar from Primary demand

**The PVDIS line estimates how much unmetered solar capacity sits behind each Primary substation, how
that capacity changes over time, and whether an explicit solar term improves or explains the net
demand forecast.** The line serves three purposes, and the second matters about as much as the
first:

- **Skill:** a lower error in the net demand forecast at each substation.
- **Explanation:** a forecast that NGED can read as demand minus solar. A solar term that leaves the
  error level but splits the forecast into parts an engineer recognises is still worth keeping, so
  the forecast experiments below test non-inferiority as well as gain.
- **Capacity:** a time series of unmetered solar capacity per Primary. NGED holds almost no
  information on unmetered solar today, so an estimate with honest error bars is useful even when
  the estimate is far from perfect.

**The capacity estimate runs offline, weekly or monthly, and the production forecast only reads the
latest capacity table.** Nothing in this line adds a fitting step to the serving path. A later step
may also estimate the mix of panel tilts, panel azimuths, and ratios of DC to AC capacity behind each
Primary, as the `UniversalSolarFleetNode` sketched in [differentiable
physics](../techniques/differentiable-physics.md) does, so that the explicit solar forecast takes
the right daily shape.

**Experiments 00039 and 00040 are parts 1 and 2 of the series; the parts below follow the
exploratory protocol in the `ml-experiment` skill.** Opus plans each part after reading the result
of the part before, so the designs below are proposals that the planning step of each part may
change. Experiment 00039 fitted one capacity per Primary per season-year, from a calendar and
temperature demand term minus one CAMS-driven solar shape. On real solar-farm output injected into
the real demand, the half-hourly fit recovered a median 0.773 (0.740 to 0.818) of the injected
capacity, and a fit on daily sums recovered 0.854 (0.815 to 0.866). Experiment 00040 attached a
bound on the capacity error of 0.18 to 0.23 at five Primaries, and found that a demand load which
rises with sunshine cannot be told apart from solar.

**The Opus review of experiment 00040 recommended stopping the series and publishing a static
capacity table; the parts below continue the series because they answer questions that review did
not consider.** The review's stop rule asked whether a further experiment could narrow the capacity
bound, and found none that could. The parts below ask different questions: how capacity changes week
by week, how much forecast error the solar causes, and whether an explicit solar term improves or
explains the forecast. The same review found that experiment 00040's table paired the daily-fit
capacity with a bound computed for the six-arm capacity C, and that the daily fit is about 2.6
times more exposed to an undetected day-long load. The parts below therefore report C as the
capacity, with the daily fit as a cross-check, unless the injections show otherwise.

**Rules shared by every part:**

- **Substations.** Only the 8 Primaries metered in MW. The 8 Primaries metered only in MVA fold
  solar into the magnitude of the flow, and experiment 00039 fitted a negative capacity at Stickney in
  every year. Wrangle T1 and Wrangle T2 share one service-area polygon in `TimeSeriesMetadata`, so any
  comparison against an outside count of solar treats them as one area.
- **Irradiance.** CAMS for past weather, because the [past-weather solar
  study](../studies/past-weather/solar.md) found the CAMS solar estimate the most accurate of the
  products it compared. ENS for forecasts, with the ENS irradiance calibrated against CAMS before a
  capacity fitted on CAMS is applied to it, because experiment 00031 suspected an ENS bias of the
  kind that cancelled its sunny-day gain under heavy cloud.
- **Capacity used at forecast time is causal.** A part that feeds a capacity to a forecast uses the
  capacity estimated from data before the forecast's training end, never the capacity fitted on the
  scored period.
- **Switching is masked.** The logged closures, padded by 24 hours, are removed from every capacity
  fit until SW2-0b's labels exist. A transfer moves a share of the donor's net load, including the
  donor's solar, so an unmasked transfer looks like a step in capacity.
- **Known answers first.** Every capacity method is scored on experiment 00039's injection harness
  before it touches the real series, and the harness gains injected capacity ramps, injected
  capacity steps, and injected transfers that carry the donor's solar.
- **Loads that track sunshine are a named confound.** Seaside visitor load at the coastal Primaries
  (Skegness, Ingoldmells, Chapel St Leonards, and Trusthorpe), irrigation pumping, cooling of crop
  stores, and air conditioning all rise on sunny days and bias the fitted capacity low. Daytime
  domestic load falls on sunny days, as people go outside, and biases the fitted capacity high. Each
  write-up reports the coastal Primaries separately and compares school-holiday weeks with term-time
  weeks.

#### PVDIS-3: how much forecast error comes from embedded solar

**Measure, with no new forecasting method, how much of the demand forecast's error the embedded
solar explains, and whether the XGBoost model already captures the solar it is given.** Experiment
00009 found that CAMS improved the demand forecast by only 0.035 points of capacity. That small gain
has two readings, which this part separates: either solar explains little of the error, or the
XGBoost model does not use irradiance well enough for a better irradiance input to show.

- **Arm D0** is experiment 00033's best guess, read from its stored predictions.
- **Arm D1** retrains the same recipe with every irradiance input removed, so the model has no
  direct route to solar. Lighting load is small now that most lighting is LED, so removing
  irradiance should remove little true demand signal. Temperature stays in, because temperature is
  demand; temperature correlates with sunshine in summer, so D1 still learns some solar indirectly,
  and the write-up reports how much.
- **The measurement** regresses each arm's residual, in daylight, on the causal capacity times the
  CAMS unit solar shape at the valid time (the solar that actually happened), and D0's residual also
  on the causal capacity times the gap between the CAMS shape and the ENS shape (the solar forecast
  error). It reports the share of residual variance and of the quantile score explained, by lead
  band, clear-sky-index bin, and Primary.
- **Two add-back previews** cost one line each: D1's forecast minus the causal capacity times the
  CAMS shape (a ceiling: a perfect solar forecast at Primaries, comparable with experiment 00031's
  oracle of 0.82 points at the 2 GSPs), and D1's forecast minus the causal capacity times the
  calibrated ENS shape (a realistic version).
- **Proposed prediction**, for the plan to confirm: the solar term explains at least 30% of D1's
  daylight residual variance in April to September, and the solar forecast error explains at most
  10% of D0's. The plan sets the thresholds and their reasons.

#### PVDIS-4: a capacity series that only rises, with error bars

**Estimate capacity on a weekly grid, with non-negative weekly increments under an L1 penalty, so
that weeks share information and the series steps only where the data demand a step.** Fitting a
free capacity per week would be too noisy: the season-year standard error of about 1.2% of capacity
implies roughly 7 to 8% per week. With the penalty, the weekly grid sets only where a step may fall,
and the penalty, chosen on injected ramps and steps, sets how small a step the fit can detect. The
part reports that detection limit per Primary. The plan chooses the observation model on the
injections, including injected day-long loads: the six-arm fit behind C, or experiment 00039's fit
on daily sums, which recovered more of the injected solar but is more exposed to a day-long load.

- **Genuine falls are modelled as a separate, reversible deficit,** a non-negative weekly term with
  its own, stronger L1 penalty, for inverter faults and solar sites switched off for weeks or months.
  The capacity itself never falls. Active network management curtailment is not modelled, because
  unmetered solar in GB is very rarely under active network management.
- **The unconstrained weekly path is reported beside the constrained one,** so that a ratchet on an
  unmasked transfer shows as a gap between the two.
- **A probabilistic capacity comes from two sources, reported separately.** Sampling uncertainty
  comes from a block bootstrap over days, refitting the whole series each time, which gives an
  ensemble of capacity paths. Systematic uncertainty comes from a multiplicative factor drawn once
  per path: the under-recovery of experiment 00039, about 10% for a rooftop fleet whose shape differs
  from a solar farm's, and experiment 00040's bound on undetected demand loads. On the injections,
  the 90% interval must contain the true capacity in about 90% of cases.
- **Validation against outside counts of solar:**
    - Steps in the series should fall near the connection dates that the Embedded Capacity Register
      gives for new sites of 50 kW or more inside each Primary's polygon. The register code lives in
      experiment 00031's folder, so it has to move to `packages/studies` first, as a reviewed
      change.
    - The estimate should be at or above the registered solar inside each polygon, because the
      register omits sites under 50 kW.
    - The count and area of solar mapped in OpenStreetMap inside each polygon, pulled fresh because a
      lot of solar has been installed since the 2020 open dataset was published.
    - The sum over the Primaries and metered solar farms under each GSP should sit at or below PV_Live's
      installed capacity for that GSP.
- **Proposed prediction**, for the plan to confirm: on the injections, the constrained series
  recovers an injected step of 10% of capacity within 4 weeks of its true date, and its error at
  each week is no larger than the season-year fit's.

#### PVDIS-5: forecasting net demand as demand minus forecast solar

**Compare three ways of using the estimated capacity in the demand forecast, against the best
guess, and adopt a way that explains the forecast even where it does not improve the score.**

- **Arm A, a solar feature:** the best guess plus a column of causal capacity times the calibrated
  ENS solar shape. Experiment 00031 found its nearest arm level; the difference here is a capacity
  that changes over time and a calibrated ENS.
- **Arm B, add solar back:** train the XGBoost model on gross demand, the metered net demand plus the
  causal capacity times the CAMS solar shape, with no irradiance inputs. At inference, subtract the
  causal capacity times the solar forecast from each ENS member, so the forecast spread includes the
  solar forecast's spread. Lagged gross demand needs CAMS at the lag time, and CAMS arrives with a
  one-day delay, so the shortest lags use ENS instead. Run Arm B with the capacity divided by the
  chosen estimator's recovery on the injections, and without that correction.
- **Arm C, the simple baseline:** the best guess plus experiment 00039's trailing midday clear-minus-
  overcast mean as a feature. Experiment 00031's capacity statistics, of the same kind, lost 0.167
  points, so Arm C is the comparison arm, not a candidate.
- **The decision rule tests both gain and non-inferiority.** An arm is a candidate when its interval
  excludes zero in the right direction, and Arm B is also adoptable for explanation when its interval
  lies inside a non-inferiority margin that the plan sets. Climatology and `manual_heuristic` rows
  give the absolute scale.
- **Waits for** PVDIS-3, whose result says how large a gain is possible, and PVDIS-4, which supplies
  the capacity series.

#### PVDIS-6: the mix of panel orientations

**Estimate the share of east-, south-, and west-facing solar behind each Primary, and drop the step
if the shares are not identified.** Experiment 00040 found that capacities on an east-south-west
menu swapped between years in a season-year fit. Pooling the shapes across Primaries with a shared
prior, or fitting on clear days only, may identify them. A Primary's orientation mix would sharpen
the explicit solar forecast in Arm B of PVDIS-5.

#### PVDIS-T: the capacity table, a study

**Publish the capacity series, with its error bars and its outside checks, as a study under
`docs/studies/`, once PVDIS-4 has run.** The study replaces the static capacity table that the Opus
review of experiment 00040 proposed, and it keeps three uncertainty terms separate: sampling, the
farm-to-rooftop transfer, and undetected demand loads. The SW2-10 joint fit reads the same series.

### P11: ensemble statistics as inputs

**The arms are the member mean, spread, and 10th and 90th percentiles of temperature and irradiance,
against rows from the control member only.** The planned contrast is the statistics arm against the
control-member arm.

### P13: perfect-prognosis and ceiling checks

**Experiment 00001 found no gain from ERA5 training beyond day 6.** The arms are the ERA5-trained
stage 1, a stage 1 trained on ENS early leads, and ceiling inputs (ERA5, CAMS, both, and each
product's short-lead fields).

**Resolution limit.** 9 generator series cannot resolve gains smaller than about 0.3 points of
capacity, and few solar farms share a weather cell.

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](../roadmap/staged-probabilistic-forecast.md), and the Schaake shuffle, scored by the variogram
score and the CRPS of summed solar farms in a weather cell, by lead band.

### P15: a second weather product already on disk

**Blending products is the maintainer's strongest hunch for improving the forecast, and it can be
tested before any ingest.** CEDA UKV and products downloaded from Open-Meteo are already on disk.
The CEDA UKV archive differs statistically from the live feed, so a result may not carry over. The
arms are ENS alone; ENS plus the second product with the shared calibrator and the learned blend;
the per-product-columns control; a fixed equal-weight mixture; and a learning curve at 1, 2, 3, 6,
and 12 months of the second product's history, rotated across seasons. The planned contrast is ENS
plus the second product against ENS alone, with a safe-entry check at 1 and 2 months (the interval
lies inside plus or minus 0.15 points).

**Resolution limit.** With 9 generator series, a safe-entry interval of plus or minus 0.15 points is
not reachable at 1 or 2 months of history. The experiment compares ENS alone with ENS plus one
product, and does not run the learning curve.

### P16: a longer ENS history

**The ENS backfill from Source Cooperative (2021-03 to 2024-03) could give an earlier validation
fold, and the full fetch cannot run autonomously.** [Issue #959 (Extend ECMWF ENS training
history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959) describes the
fetch. The archive is in `us-west-2`. Fetching the control member alone is about 530 GB and 4 to 5
hours, and fetching all 51 members is about 30 TB, which needs a cloud machine in that region and
the maintainer's approval of the cost. The issue also ties autonomous use of the fetch to the
scorer-protection work in issue 958. Every experiment since 00003 scores the same year on the same
16 series, so choices tuned there overfit it, and an earlier fold is the remedy. For demand, a
longer training history has hurt so far, so the backfill is valuable first as validation data.
