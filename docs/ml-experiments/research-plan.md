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
| [SW2-4 to SW2-9](#sw2-4-to-sw2-9-detector-iterations) | Which changes to the detector raise its event-level F1: a better residual, a retrain loop, paired-step intervals, a shape-aware step, node balance, or whitening? | 3 | SW2-1 | Proposed; one hypothesis per experiment, run in order |
| [SW2-10](#sw2-10-v2-in-miniature-a-linear-joint-fit) | Does one linear fit per trial Primary, with a weather baseline, embedded solar capacity times a PV shape, and a term per labelled transfer, beat the frozen v1 baseline outside switching and recover injected solar and transfers? | 3 | SW2-0b | Proposed; the first test of a v2 piece against v1, with inputs from the PV-disaggregation line |
| [SW2-11](#sw2-11-and-sw2-12-a-v1-forecast-under-the-normal-running-arrangement) | With the labelled switching removed from the training targets, does the best-guess architecture forecast the normal running arrangement as well outside switching as the plain best guess, and does a switching magnitude paired with each lagged power input, or a lag corrected by that magnitude, improve it? | 2 | RT2 (experiment 48), for the architecture | Proposed; picks the v1 NRA candidate |
| [SW2-12](#sw2-11-and-sw2-12-a-v1-forecast-under-the-normal-running-arrangement) | How much of SW2-11's NRA skill is lost when the switching labels come from the detector instead of NGED's logs? | 3 | SW2-11 and SW2-8 | Proposed |
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |
| [PVDIS-3](#pvdis-3-how-much-forecast-error-comes-from-embedded-solar) | What does the irradiance input add to the best demand forecast, and how much of the remaining error does embedded solar explain? | 2 | The maintainer's review of this plan | Proposed; part 3 of the PVDIS series |
| [PVDIS-4](#pvdis-4-a-capacity-series-that-rises-easily-and-falls-reluctantly) | Can a weekly capacity series, with rises penalised lightly and falls heavily, recover injected changes in capacity, with calibrated sampling intervals, at the Primaries metered in MW? | 2 | The maintainer's review of this plan | Proposed; part 4 of the PVDIS series; can run beside PVDIS-3 |
| [PVDIS-5](#pvdis-5-forecasting-net-demand-as-demand-minus-forecast-solar) | Does a solar feature, or training on gross demand and subtracting forecast solar, lower or at least match the best guess's error while explaining the forecast? | 3 | PVDIS-3 and PVDIS-4 | Proposed |
| [PVDIS-6](#pvdis-6-the-mix-of-panel-orientations) | Can the share of east-, south-, and west-facing solar behind each Primary be identified? | 4 | PVDIS-4 | Proposed |
| [PVDIS-7](#pvdis-7-mw-at-the-primaries-metered-only-in-mva) | Can a fit of the magnitude of demand minus solar recover MW, and the direction of flow, at the Primaries metered only in MVA? | 4 | PVDIS-4 | Proposed; research, which NGED accepts may not succeed |
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

**Each iteration tests one change to the detector against the
[experiment 43](00043-sw2-1-detector-scoreboard/result.md) scoreboard, tuned on injections and odd
months, scored on even months.** The order below is a proposal, by expected gain per hour of
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
- **SW2-8:** a single-series detector first, with neighbour balance as a sense check, not a gate.
  Every single-series alarm stands. A metered neighbour's opposite step raises the alarm's
  confidence and names the receiver, a same-sign step on the neighbours vetoes the alarm as a
  common-mode weather or calendar miss, and an alarm with no metered neighbour keeps its
  single-series confidence. Experiment 00043 found the neighbour-balance detector D1, which uses
  the neighbour test as a gate, finding almost none of the injected transfers without a metered
  receiver. The sum of the neighbourhood's residuals and a subset-sum over neighbours for a split
  feed the confidence score.
- **SW2-9:** whitening, experiment 00037's proposed F4, an autoregressive model of order 1 per
  series and hour of day fitted to the residual.

#### SW2-10: v2 in miniature, a linear joint fit

**Fit one linear model per trial Primary that holds a weather baseline, embedded solar, and the
labelled transfers together, and score it against the frozen v1 baseline.** The model is net demand
equals the baseline, minus a capacity times a PV shape, plus, for each labelled transfer k, a share
times the donor's net power over transfer k's interval. With the intervals given, the fit is
linear. The result tells the project whether to start the v2 differentiable-physics work sooner.

- **The baseline term** is a lagless XGBoost model fitted inside SW2-10, trained with labelled
  switching excluded.
- **The transfer intervals** come from SW2-0b, or from the logged closures padded by 24 hours if
  the labels are not ready.
- **The capacity and the PV shape** come from the PV-disaggregation line: experiment 00040's
  six-arm capacity with its bound, experiment 00039's capacity as a cross-check, and experiment
  00039's CAMS unit PV shape. A weekly capacity series replaces the experiment 00040 capacity if
  the PV line's weekly-capacity experiment runs. The 8 Primaries metered only in MVA are excluded.
- **Scoring** is on held-out months outside switching against the v1 baseline, and on experiment
  00039's injection harness extended with injected transfers that carry the donor's PV, where the
  true capacity and the true shares are known.

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
  PVDIS-5 tests non-inferiority as well as gain.
- **Capacity:** a time series of unmetered solar capacity per Primary. NGED holds almost no
  information on unmetered solar today, so an estimate with honest error bars is useful even when
  the estimate is far from perfect.

**The capacity estimate runs offline, weekly or monthly, and the production forecast only reads the
latest capacity table.** Nothing in this line adds a fitting step to the serving path. PVDIS-6 asks
whether the mix of panel orientations behind each Primary can also be estimated, as the
`UniversalSolarFleetNode` sketched in [differentiable
physics](../techniques/differentiable-physics.md) would, so that the explicit solar forecast takes
the right daily shape.

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
changes week by week for NGED. PVDIS-5 rests on the explanation goal and on what differs from those
four experiments, not on a claim that the review missed the forecast question. The same review
found that the daily fit is about 2.6 times more exposed than C to an undetected day-long load, so
every part reports C as the capacity and keeps the daily fit as a cross-check.

**Rules shared by every part:**

- **Substations.** Only the 8 Primaries metered in MW, each carrying its status from the earlier
  review into every result: Kirton is reported only as a range (1.2 to 4.4 MW), because its
  daylight cloud load was flagged in 6 of 7 season-years; Wrangle T1 is excluded from any claim that
  a Primary is free of a confound; Trusthorpe is marked low-power. The 8 Primaries metered only in
  MVA fold solar into the magnitude of the flow, and experiment 00039 fitted a negative capacity at
  Stickney in every year. PVDIS-7 treats them separately.
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

#### PVDIS-3: how much forecast error comes from embedded solar

**Measure what the irradiance input adds to the best demand forecast, given the other routes by
which the XGBoost model can learn solar, and how much of the remaining error the solar explains.**
Two earlier results bound the answer. Experiment 00009's observed-weather diagnostic found that the
stage 1a model does use irradiance: reading CAMS at validation lowered the daylight error of the
median by 0.88 points of capacity, against 0.04 with ENS. Experiment 00022 found that ERA5 weather in
place of ENS, on experiment 00033's recipe, lowered the quantile score by 0.10 points (1.9% of 5.32),
of which ERA5 shortwave gave 0.08, so a better solar forecast has little room on that recipe. PVDIS-3
asks how that room divides between the solar itself and the forecast of the weather that drives it.

- **Arm D0** retrains experiment 00033's recipe with three matched seeds, on the Primaries metered in
  MW and the rows of `fold_a_every5d_v1`, so that refit noise does not enter the contrast.
- **Arm D1** retrains the same recipe with the shortwave irradiance columns removed. A second arm,
  D1L, also removes longwave radiation, which is a direct proxy for cloud. D1 is not blind to
  solar: lagged power, temperature, the sun's position, and the 14-day error correction all carry
  some solar, and removing irradiance also removes the route for cloud-driven demand. The write-up
  therefore reads D0 minus D1 as what the irradiance input adds, not as the solar share.
- **The primary measure has no fitted coefficient.** Capacity is in MW, so each arm's forecast is
  shifted by minus the capacity times the solar anomaly, with a coefficient fixed at 1, and
  re-scored. The anomaly is the CAMS unit shape at the valid time minus its month-by-half-hour mean
  over the training window, because the XGBoost model has already learned the mean solar for each
  hour and month. The capacity is experiment 00040's C for season-year 2023/24, the latest one that
  ends before the July 2025 validation start, and the write-up states that it is about one year of
  growth out of date.
- **The shift that uses CAMS at the valid time is an oracle,** a ceiling on what knowing the real
  solar would add. The capacity in the shift is itself an estimate.
- **Any regression of the residual on the solar anomaly is cross-fitted by month:** fitted on odd
  months and scored on even months, by lead band, clear-sky-index bin, and Primary.
- **The decision rule**, which the plan fixes with its reasons, is stated in quantile-score points
  on daylight rows from April to September, at the Bonferroni level for its contrasts. The plan
  derives its thresholds from experiment 00022's 0.10 points and experiment 00009's diagnostic.

#### PVDIS-4: a capacity series that rises easily and falls reluctantly

**Estimate capacity on a weekly grid, with weekly increments under an asymmetric L1 penalty: a light
penalty on rises and a heavy penalty on falls.** Weeks then share information, and the series steps
only where the evidence supports a step. A free capacity per week would be too noisy: the
season-year standard error of about 1.2% of capacity implies about 8.7% per week on average (1.2% ×
√52), and far more in a winter week, when the sun is low. The part reports the standard error of a
week by season and the smallest step it can detect at each Primary.

- **Why falls are allowed at all.** The fit estimates the solar that the Primary's meter sees, not
  the installed capacity. That apparent solar can fall when home batteries charge from midday solar,
  when a feeder is permanently reconfigured, or when a solar site is switched off for repairs. A
  strict constraint that never lets the series fall would lower past capacity to fit a fall instead.
  The fit with no constraint is reported beside the asymmetric fit, and the gap between them is a
  diagnostic with a threshold that the plan sets.
- **The estimand on the injections is the injected change.** A penalised, constrained fit is not
  linear, and the real series already carries unknown solar. So the score is the path fitted with an
  injection minus the path fitted without it, compared with the injected path. The plan also
  injects the same path at several baseline levels to check how far the fit departs from
  additivity, or injects into a semi-synthetic baseline whose solar is known.
- **The injections add four cases to experiment 00039's harness:** capacity ramps, capacity steps,
  temporary falls of weeks to months, and transfers that carry the donor's solar. A fifth case is a
  seasonal cycle in daytime demand, like the coastal visitor season or irrigation, which measures
  how far a seasonal load creeps into the capacity series. The penalties are tuned on the
  injections and scored on held-out injections.
- **The estimate production reads is scored as a real-time vintage.** The fit is repeated at rolling
  origins, each truncated at its origin, and the latest week of each vintage is scored against the
  truth. The part reports how far the first estimate of a week moves by the final fit. Penalised
  fits are least constrained at the end of the record, which is the week production reads.
- **False steps on the real series** are counted with experiment 00039's placebo sky, which shuffles
  days within a month, as a measure of steps that weather noise alone produces.
- **A probabilistic capacity has two separate parts.** The first part, sampling uncertainty, is an
  ensemble of capacity paths from a bootstrap over blocks of 7 days or more, refitting the whole
  path each time. A bootstrap of a penalised fit at a boundary tends to under-cover, so the part
  checks the ensemble's coverage on the injections separately at weeks with no true step and at
  weeks with a step. The second part, systematic uncertainty, is reported as named scenarios and
  never merged into the ensemble: the farm-to-rooftop difference in solar shape, an undetected
  day-long load (which pushes capacity up), and a load that rises with sunshine (which pushes it
  down).
- **Outside checks, in order of strength:**
    - The Department for Energy Security and Net Zero's sub-regional statistics of small-scale solar
      installations, by lower-layer super output area, summed over each Primary's polygon and
      converted to expected peak AC output, are the strongest check. The plan states the tolerance
      before the comparison.
    - The Embedded Capacity Register is a soft prior check, not a floor. NGED's Primary series is
      "Disaggregated Demand", from which metered generation has already been removed, and experiment
      00031 found the register's median at 0.08 of effective capacity against 0.33 from the data.
      The check uses only connected solar sites of 50 kW to under 1 MW, dated before each week, and
      first confirms what the register's date field records. A step-date check uses only sites
      larger than the Primary's detectable step, against randomly placed placebo dates.
    - Solar mapped in OpenStreetMap inside each polygon is descriptive only, because its rooftop
      coverage is uneven.
    - Experiment 00039's comparison with each BSP's fitted capacity is repeated weekly.

#### PVDIS-5: forecasting net demand as demand minus forecast solar

**Compare two ways of using the capacity series in the demand forecast against the best guess, with
a shuffled-capacity control for each, and adopt a way that explains the forecast where it matches
the best guess's score.**

- **Arm A, a solar feature:** experiment 00033's recipe plus a column of the capacity vintage times
  the calibrated ENS solar shape. Experiment 00031's nearest arm, A2 (data-only solar products, with
  a capacity refitted over a trailing 28-day window), lost 0.036 points (+0.002 to +0.074). What
  differs here is the calibration of ENS against CAMS, the CAMS shape in place of a generic shape,
  and a capacity series smoothed over many weeks.
- **Arm B, add solar back:** train the XGBoost model on gross demand, the metered net demand plus the
  capacity vintage times the CAMS solar shape, with no shortwave inputs. At inference, subtract the
  capacity times each ENS member's solar from that member's gross quantiles, then pool the members
  and apply the existing spread factor, in that order. Each lagged gross-demand input is built in
  training exactly as serving would build it at that lead: from CAMS where CAMS's one-day delay
  allows, and from ENS otherwise. The plan states whether the 14-day error correction acts on gross
  or net residuals. Arm B reports the coverage of the 0.05 and 0.95 quantiles in sunny and in cloudy
  daylight separately, because the spread of ENS leaves out the uncertainty in capacity.
- **Controls:** each arm has a twin whose capacity values are shuffled between forecast dates
  within each Primary and calendar month, as experiment 00031's A6 control did. In experiment
  00031, the shuffled control gained as much at leads under 48 hours as the real arm did. A
  climatology row and a `manual_heuristic` row give the absolute scale. Experiment 00039's trailing
  clear-minus-overcast midday mean, as a feature, is a sanity row: experiment 00031's capacity
  statistics of the same kind lost 0.167 points.
- **The decision rule tests gain and non-inferiority.** An arm is a candidate when its interval
  excludes zero in the right direction and the shuffled twin's does not. Arm B is also adoptable
  for explanation when the upper end of its interval lies below a non-inferiority margin, which the
  plan fixes with its reason (+0.1 points, about 2% of the best guess's score, is the starting
  proposal).
- **Power.** Experiment 00031's paired contrasts reached half-widths of 0.03 to 0.08 points over 16
  Primaries, so about 0.05 to 0.11 is expected over the 8 metered in MW. The plan states which
  effects that resolves.
- **Waits for** PVDIS-3, which says how large a gain is possible, and PVDIS-4, which supplies the
  capacity vintages.

#### PVDIS-6: the mix of panel orientations

**Estimate the share of east-, south-, and west-facing solar behind each Primary, and drop the step
if the shares are not identified.** Experiment 00040 found that capacities on an east-south-west
menu swapped between years in a season-year fit. Pooling the shapes across Primaries with a shared
prior, or fitting on clear days only, may identify them. A Primary's orientation mix would sharpen
the explicit solar forecast in Arm B of PVDIS-5.

#### PVDIS-7: MW at the Primaries metered only in MVA

**Fit the magnitude of demand minus solar, so that the reading bounces off zero instead of going
negative, and test on the Primaries metered in MW whether the fit recovers the direction of flow and
the solar capacity.** The 8 MVA Primaries have MVA meters only, and NGED cannot supply real and
reactive power there, so recovering MW from MVA is the only route to forecasting MW at every
Primary. The model assumes a power factor of 1, which is weakest exactly where the flow passes
through zero, because reactive power then dominates the reading. The known-answer test takes the
absolute value of the MW readings at the 8 Primaries metered in MW and scores the recovered sign
and capacity against the real ones. A fit that passes that test is then applied to the 8 MVA
Primaries. NGED would like MW at every Primary, and accepts that the result is research.

#### PVDIS-T: the capacity table, a study

**Publish a season-year capacity table now, built from experiment 00040's outputs with no refit,
and replace it with PVDIS-4's weekly series once that part has run.** NGED's need is to improve on
almost no information, so the season-year table is useful before the weekly series exists. The
study reports C as the capacity, keeps sampling uncertainty separate from the named systematic
scenarios, and carries each Primary's status from the earlier review. The SW2-10 joint fit reads the
same capacities.

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
