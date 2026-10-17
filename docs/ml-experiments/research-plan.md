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
| [SW2-11](#sw2-11-and-sw2-12-a-v1-forecast-under-the-normal-running-arrangement) | With the labelled switching removed from the training targets, does the best-guess architecture forecast the normal running arrangement as well outside switching as the plain best guess, and does a switching magnitude paired with each lagged power input, or a lag corrected by that magnitude, improve it? | 2 | RT2 (experiment 48), for the architecture | Proposed; picks the v1 NRA candidate |
| [SW2-12](#sw2-11-and-sw2-12-a-v1-forecast-under-the-normal-running-arrangement) | How much of SW2-11's NRA skill is lost when the switching labels come from the detector instead of NGED's logs? | 3 | SW2-11 and SW2-8 | Proposed |
| [00081](00081-r11-control-member-second-validation-year-demand/result.md) (R11) | Do the weekly-retraining gain (experiment 48) and the Christmas-column and half-life reads (experiment 49) keep their sign on the 274 control-member runs of 2023-07 to 2024-03, and on an all-18 km fold of 2022-07 to 2023-03? | 2 | The ML manager's go for GPU fits | Planned; plan under review, frames and fit scripts in progress |
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |
| [PVDIS-T](#pvdis-t-the-capacity-table-a-study) | What is the unmetered solar capacity behind each Primary metered in MW, with error bars and outside checks? | 2 | The maintainer's review of this plan | Proposed; a study under `docs/studies/`, season-year table first from experiment 00040's outputs, weekly after PVDIS-4 |

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

#### SW2-4 to SW2-9: detector iterations

**Each iteration tests one change to the detector against the [experiment
43](00043-sw2-1-detector-scoreboard/result.md) scoreboard, tuned on injections and odd months,
scored on even months, with recall weighted by the MW moved as the headline from SW2-7 onwards.**
The order below is a proposal, by expected gain per hour of work, for the plan of each iteration to
confirm:

- **SW2-5 (done, [experiment 68](00068-sw2-5-train-detect-retrain-loop/result.md)):** three rounds
  of refitting the lagless XGBoost model without the detected intervals changed the share of the
  moved MW found by +0.011 and the forecast error by -0.175%, and a random mask did as well.
- **SW2-6 (done, [experiment 61](00061-sw2-6-paired-step-intervals/result.md)):** pairing the
  onset and return steps into one alarm score finds less than D0M at the same alarm rate, but taking
  the end of D0M's interval from the strongest opposite step in the next 14 days finds 0.19 more of
  the moved MW.
- **SW2-7 (done, [experiment 60](00060-sw2-7-profile-weighted-step-detector/result.md)):** a
  step statistic that weights each half-hour by the expected power found the same transfers as the
  constant step on five injection shapes, so the step's shape does not limit the 24-hour detector.
  About a third of the transfers with a donor step of 6 noise units or more have the right onset and
  the wrong end, which SW2-6 (paired-step intervals) addresses.
- **SW2-8 (done, [experiment 55](00055-sw2-8-neighbour-veto-single-series-detector/result.md)):** a
  single-series detector first, with neighbour balance as a sense check, not a gate.
  Every single-series alarm stands. A metered neighbour's opposite step raises the alarm's
  confidence and names the receiver, a same-sign step on the neighbours vetoes the alarm as a
  common-mode weather or calendar miss, and an alarm with no metered neighbour keeps its
  single-series confidence. Experiment 00043 found the neighbour-balance detector D1, which uses
  the neighbour test as a gate, finding almost none of the injected transfers without a metered
  receiver. The sum of the neighbourhood's residuals and a subset-sum over neighbours for a split
  feed the confidence score.
- **SW2-9 (done, [experiment 62](00062-sw2-9-whitened-residual-step-detector/result.md)):**
  whitening, experiment 00037's proposed F4, an autoregressive model of order 1 per series and
  hour of day fitted to the residual. The filtered detector found less of the moved MW (-0.097 at
  24 hours), because the filter shrinks a day-long step faster than the noise of the 24-hour step.

#### SW2-11 and SW2-12: a v1 forecast under the normal running arrangement

**Retrain the best-guess architecture so that it never learns to predict an abnormal running
arrangement, and pick the v1 NRA candidate from three arms.** SW2-11 uses experiment 00041's
labelled switching, and SW2-12 repeats the experiment with the detector's labels, which is what
Flexpectation v2 will have. The gap between the two runs is the skill lost without a labelled set.

- **The architecture** is the winner of RT2 (experiment 00048). Weekly retraining without the
  error correction is the leading candidate. If the winner keeps an error correction, the
  correction's history leaves out switching half-hours and looks back far enough to hold 14
  unmasked days, and the write-up reports the unmasked days per window and how often the fallback
  to the last full correction fired.
- **The arms:** the plain architecture with nothing removed; the NRA arm, with logged switching
  removed from the training targets; and the NRA arm plus, for each lagged power input, the signed
  MW moved by any switching event at that lag. A fourth arm replaces each lag by the lag minus that
  MW, because a tree model cannot easily subtract one column from another.
- **The labels in SW2-11** are experiment 00041's visible events, conserved or one-sided, which
  two Sonnet labellers and an Opus adjudicator found in the telemetry at NGED's logged times. The
  target mask removes each event's interval from the training targets of its donor and receivers,
  and the magnitudes are the events' labelled MW moved.
- **The labels in SW2-12** come from the best detector after SW2-8.
- **Scoring** uses the same rows in every arm and in both runs: half-hours outside NGED's logged
  closures and outside the labelled events, where the metered target is the NRA target. A
  secondary metered view scores every half-hour and decides nothing.
- **Threshold.** The target mask needs a detector at training time only, so the NRA arm is adopted
  when its interval does not show it worse than the plain arm. The magnitude arms need a detector
  on the serving path, so they need a gain whose interval excludes zero. Experiment 00044 bounded a
  perfect detector's gain at 0.34% of the best guess's quantile score under the NRA view, so the
  gains are expected to be small.

### PVDIS: separating embedded solar from Primary demand

**The PVDIS line estimates how much unmetered solar capacity sits behind each Primary substation, how
that capacity changes over time, and whether an explicit solar term improves or explains the net
demand forecast.** The line serves three purposes, and the second matters about as much as the
first:

- **Skill:** a lower error in the net demand forecast at each substation.
- **Explanation:** a forecast that NGED can read as demand minus solar. A solar term that leaves the
  error level but splits the forecast into parts an engineer recognises is still worth keeping, so
  experiment 64 (PVDIS-5) tested non-inferiority as well as gain.
- **Capacity:** a time series of unmetered solar capacity per Primary. NGED holds almost no
  information on unmetered solar today, so an estimate with honest error bars is useful even when
  the estimate is far from perfect.

**The capacity estimate runs offline, weekly or monthly, and the production forecast only reads the
latest capacity table.** Nothing in this line adds a fitting step to the serving path. [Experiment
73](00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) (PVDIS-6) found that the mix of panel
orientations behind each Primary, the quantity the `UniversalSolarFleetNode` sketched in
[differentiable physics](../techniques/differentiable-physics.md) would fit, is not stable across
season-years, so the explicit solar forecast keeps one south-facing shape.

**Experiments 00039 and 00040 are parts 1 and 2 of the series, and the parts below follow the
exploratory protocol in the `ml-experiment` skill.** Opus plans each part after reading the result
of the part before, so the designs below are proposals that each part's plan may change. Experiment
00039 fitted one capacity per Primary per season-year, from a calendar and temperature demand term
minus one CAMS-driven solar shape. On real solar-farm output injected into the real demand, the
half-hourly fit recovered a median 0.773 (0.740 to 0.818) of the injected capacity, and a fit on
daily sums recovered 0.854 (0.815 to 0.866). Experiment 00040 compared six fits, which differ in
how much cloud-driven demand each allows, and took the six-arm capacity C as its estimate. C
recovered a median 0.992 (0.951 to 1.032) of the injected capacity. Experiment 00040 also attached
a bound on the error that an undetected demand load would cause, of 0.18 to 0.23 at five Primaries.
A demand load that rises with sunshine cannot be told apart from solar.

**The Opus review of experiment 00040 recommended stopping the series, publishing a season-year
capacity table, and not using capacity as a forecast feature; the parts below continue because the
goals changed.** That review judged the series against the questions it had then: whether a further
experiment could narrow the capacity bound, and whether capacity improves the forecast score. Four
experiments (00007, 00009, 00022, and 00031) found small or no gains from solar inputs. The
maintainer has since added two goals: a forecast that explains itself, and a capacity series that
changes week by week for NGED. Experiment 64 (PVDIS-5) rested on the explanation goal and on what
differs from those four experiments, not on a claim that the review missed the forecast question.
The same review found that the daily fit is about 2.6 times more exposed than C to an undetected
day-long load, so every part reports C as the capacity and keeps the daily fit as a cross-check.

**Rules shared by every part:**

- **Substations.** Only the 8 Primaries metered in MW, each carrying its status from the earlier
  review into every result: Kirton is reported only as a range (1.2 to 4.4 MW), because its
  daylight cloud load was flagged in 6 of 7 season-years; Wrangle T1 is excluded from any claim that
  a Primary is free of a confound; Trusthorpe is marked low-power. The 8 Primaries metered only in
  MVA fold solar into the magnitude of the flow, and experiment 00039 fitted a negative capacity at
  Stickney in every year. [Experiment
  00079](00079-pvdis-7-mw-at-mva-only-primaries/result.md) (PVDIS-7) treats them separately.
- **Service areas.** `TimeSeriesMetadata.area_wkt` holds one polygon per Primary. Wrangle T1 and
  Wrangle T2 are two transformers at one substation, feeding different circuits, and share one
  polygon. Each transformer keeps its own capacity fit, and any comparison against an outside count
  of solar uses the sum of the two against the shared polygon. Before any outside comparison, a
  check reports overlaps and gaps between neighbouring polygons, whether each Primary's location
  falls inside its own polygon, and how the union of the polygons compares with the two BSPs' area.
- **Irradiance.** CAMS for past weather, because the [past-weather solar
  study](../studies/past-weather/solar.md) found the CAMS solar estimate the most accurate of the
  products it compared. ENS for forecasts. Where a capacity fitted on CAMS is applied to ENS, the ENS
  irradiance is first calibrated against CAMS, by lead, on data from the fold's training window only.
- **Capacity used in a forecast is a real-time vintage.** A forecast experiment reads the capacity
  that a fit truncated at the fold's training end would give, never a path smoothed over the whole
  record. A path fitted once over all data passes information from later weeks back to earlier weeks
  through the penalty and the constraint.
- **Switching is masked.** The logged closures, padded by 24 hours, are removed from every capacity
  fit until SW2-0b's labels exist. A transfer moves a share of the donor's net load, including the
  donor's solar, so an unmasked transfer looks like a step in capacity. The logs miss some real
  switching, so unlogged transfers remain a source of false steps.
- **Known answers first.** Every capacity method is scored on experiment 00039's injection harness
  before its result on the real series is trusted.
- **Loads that track sunshine are a named confound.** Seaside visitor load at the coastal Primaries
  (Skegness, Ingoldmells, Chapel St Leonards, and Trusthorpe), irrigation pumping, cooling of crop
  stores, and air conditioning all rise on sunny days and bias the fitted capacity low. Daytime
  domestic load falls on sunny days, as people go outside, and biases the fitted capacity high. Each
  write-up reports the coastal Primaries separately and compares school-holiday weeks with term-time
  weeks.

#### PVDIS-T: the capacity table, a study

**Publish a season-year capacity table now, built from experiment 00040's outputs with no refit,
and replace it with PVDIS-4's weekly series once that part has run.** NGED's need is to improve on
almost no information, so the season-year table is useful before the weekly series exists. The
study reports C as the capacity, keeps sampling uncertainty separate from the named systematic
scenarios, and carries each Primary's status from the earlier review. The joint fit of
[experiment 00063](00063-sw2-10-linear-joint-fit-v2-in-miniature/result.md) read the same
capacities.

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
