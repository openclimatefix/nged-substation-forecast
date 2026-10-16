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
| [SW2-0b](#sw2-0b-the-labelled-reference-set) | Which logged closures since 2022 are visible in the telemetry, with what onset, end, receivers, and MW moved? | 1 | SW2-0a (done: experiment 41), and the maintainer seeing its token cost | Proposed |
| [SW2-1](#sw2-1-a-fixed-scoreboard-and-an-injection-harness) | What event-level F1 do experiment 00037's detectors D0 and D1 reach on the labelled set and on injected transfers, by duration and size band? | 2 | SW2-0b | Proposed |
| [SW2-2](#sw2-2-how-much-the-visible-events-hurt-the-best-guess) | What share of the best guess's quantile score and tail score falls in labelled switching half-hours, and in half-hours whose lagged power falls inside one? | 2 | SW2-0b | Proposed |
| [SW2-3](#sw2-3-the-label-exclusion-ceiling) | Does excluding labelled switching from the lagless XGBoost model's training target lower its error outside switching, against a random exclusion of equal volume? | 2 | SW2-0b | Proposed |
| [SW2-4 to SW2-9](#sw2-4-to-sw2-9-detector-iterations) | Which changes to the detector raise its event-level F1: a better residual, a retrain loop, paired-step intervals, a shape-aware step, node balance, or whitening? | 3 | SW2-1 | Proposed; one hypothesis per experiment, run in order |
| [SW2-10](#sw2-10-v2-in-miniature-a-linear-joint-fit) | Does one linear fit per trial Primary, with a weather baseline, embedded solar capacity times a PV shape, and a term per labelled transfer, beat the frozen v1 baseline outside switching and recover injected solar and transfers? | 3 | SW2-0b and SW2-3 | Proposed; the first test of a v2 piece against v1, with inputs from the PV-disaggregation line |
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |

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

#### SW2-0b: the labelled reference set

**Label every logged operation of 2 hours to 60 days since 2022, and a sample of the detectors'
unexplained alarms.** There are 533 such operations, which group experiment 00037's 612 closures
where closures overlap in time and share a substation. The labeller also labels the 18 closures of
more than 60 days as two separate edge events each, 50 closures of 30 minutes to 2 hours to confirm
that they cannot be seen, and about 150 alarms from D0 and D1 that overlap no logged closure. The
alarms give precision as a bracket, from logged events alone up to logged events plus the alarms the
labeller judges real, because the logs miss real switching. Experiment 00037 suggests the set will
hold 40 to 70 visible events, so F1 by band will carry wide intervals and the injection harness will
carry most of the detector tuning.

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
