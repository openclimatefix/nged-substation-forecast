# Glossary of terms used in the experiment pages

**This page defines each term that the experiment pages and the theme Summary pages use without
redefining.** Every definition gives its source: the experiment that introduced the term, or the
code or contract that fixes its meaning. Terms appear in alphabetical order. The [data-processing
page](data-processing.md) has its own short list of terms for the processing steps it describes.

## Terms

### 14-day correction

**The 14-day correction adds the mean error of the last 14 days to every forecast quantile.** The
error is the observed power minus the forecast median, taken over the half-hours whose power was
already observed when the forecast was issued, and the correction is zero when fewer than 3 days of
errors exist ([experiment 19](00019-primary-demand-stack-vs-manual-heuristic/result.md) and its
scoring script).

### Abnormal running arrangement (ARA) and normal running arrangement (NRA)

**The normal running arrangement (NRA) is the electricity network as it is when no load transfer is
on, and an abnormal running arrangement (ARA) is the period while an operator has closed a normally
open switch, so that one Primary substation carries load that belongs to a neighbour.** An "NRA
forecast" predicts the demand that the substation would have drawn under the NRA ([experiment
88](00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md),
[experiment 89](00089-sw2-22-longest-history-nra-forecast-with-ara-lags-masked/result.md)).

### Base margin

**A base margin is a starting prediction that an XGBoost model receives for each row, in place of
the single base score that XGBoost otherwise estimates from the training labels.** The wind models
use the power that the turbine power curve gives at the forecast wind speed. The solar offset of the
Primary models uses the forecast output of the unmetered solar panels behind the substation
([experiment 80](00080-bm0-zero-base-margin-xgboost-quantile-models/result.md), [experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)).

### Best guess

**The best guess is the forecasting recipe assembled in [experiment
33](00033-best-guess-architecture-vs-production/result.md).** For Primary substation demand it is N1
with the 14-day correction and a spread factor of 1.15, which is the stack without its half-life
(experiment 19's `stack_minus_h` arm, rebuilt in experiment 33 as `n1_cs`). For bulk and grid supply
points it is N1 alone. For solar and wind farms it is a mixture, over the 51 ENS members, of XGBoost
models trained on ERA5 weather, with a spread factor and, for wind, a power-curve base margin.

### BG2

**BG2 is the second best-guess recipe, assembled in [experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md) from the best earlier ideas.**
For Primary substations it is a quantile blend of five members, with a solar offset at the five
Primaries that have a clean solar capacity estimate. The members are a weekly-refitted XGBoost
model, a monthly-refitted XGBoost model, an analogue model (the experiment's summary does not state
its type), a model pooled by substation type, and the stack. For bulk and grid supply points it is
the weekly-refitted XGBoost model with no correction. Experiment 97 recalibrates only the stored
weekly-refitted forecast and calls it BG2.

### Bulk supply point and grid supply point

**A grid supply point (GSP) is a substation where the transmission network feeds NGED's distribution
network, and a bulk supply point (BSP) is a substation one step further down.** NGED's GSPs step
voltage from 400 kV or 275 kV to 132 kV, and its BSPs step from 132 kV to 33 kV or 66 kV ([NGED's
network and its data](../background/network.md)). A Primary substation sits below a BSP. The trial
area holds two GSPs and two BSPs, and the experiments report them together as "supply points" or
"BSP and GSP". The [data-processing page](data-processing.md#decide-which-rows-are-demand) says not
to treat them as clean demand.

### Capacity and points of capacity

**Capacity in the experiment scores is the effective capacity of a series: the 99th percentile of
the absolute power over the series' whole observed history, in the series' own unit (MW or MVA).**
The `EffectiveCapacity` contract fixes this definition for version 0.1
(`packages/contracts/src/contracts/power_schemas.py`), and the experiment scripts divide power and
forecast by `effective_capacity_mw`. A point of capacity is 1% of that value, so a score of 5.26
points on a series of 20 MW capacity is an error of about 1.05 MW. A lower score is better.

### Climatology

**`climatology` is a baseline forecaster that predicts each half-hour from past power in the same
calendar cell.** A cell is a series, a local month, a local half-hour of the day, and a
weekday-or-weekend flag, smoothed over the neighbouring months and half-hours. The forecaster stores
51 equiprobable quantiles per cell and consumes no weather
(`packages/baseline_forecasters/src/baseline_forecasters/climatology.py`).

### Control member

**The control member is the one ENS run that starts from ECMWF's unperturbed analysis of the current
weather.** The other 50 of the 51 ENS members start from perturbed analyses
([ENS horizons](../studies/forecasts/ens-horizons.md)). N1 is trained on the control member and then
applied to all 51 members, and the 51 sets of quantiles are pooled
([experiment 13](00013-primary-demand-against-stored-baselines/result.md)).

### Coverage

**The coverage of a forecast quantile is the share of observed values at or below it.** A correct
q05 covers 0.05 and a correct q95 covers 0.95. The coverage of the 5% to 95% range is the share of
observed values between q05 and q95, which is 90% for a correct forecast ([experiment
19](00019-primary-demand-stack-vs-manual-heuristic/result.md) reports it as "5% to 95% coverage").

### CRPS

**The continuous ranked probability score (CRPS) scores a whole forecast distribution against the
observed value, and the experiments approximate it from the forecast quantiles.** The approximation
is twice the probability-mass-weighted sum of the pinball losses, which is the [quantile
score](#quantile-score-and-pinball-loss) (`studies.cross_validation.crps`). Experiments 2, 16, and
20 report it under the name CRPS, and experiments 13 and 33 say that the quantile score is its
estimate.

### D codes

**A D code (D1 to D22) is the anonymised label of one of the 22 demand-side series in fold A: 16
Primary substations, 4 bulk or grid supply points, and a battery and a biofuel generator.** The
labels are defined in [experiment 3](00003-era5-trained-stage1a-vs-ens-trained-demand/result.md) and
are used with the same meaning in the pages checked (experiments 5, 27, 46, 51, 53, 64, and 74 and
the data-processing page), for example D1 is a grid supply point and D9, D10, D11, D14, and D16 are
the five Primaries with a clean solar capacity estimate. A D code and a substation name are
different labels. D codes are not the same as the detector names D0, D0M, D1, and D2 (see [Detectors
D0, D0M, D1, and D2](#detectors-d0-d0m-d1-and-d2)), nor the labels P12 and P13 of [experiment
101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md). Generators carry
separate anonymous labels such as "Solar farm A", assigned by `studies.anonymise`.

### Detectors D0, D0M, D1, and D2

**In the switching experiments, D0, D0M, D1, and D2 are names of detectors, not series.** D0 is the
single-series detector. D0 raises an alarm when the average residual over a window steps by more
than a threshold. D0M is D0 with its threshold chosen by the same search that sets D2's threshold
and with no veto; at the 24-hour window it uses 4.0 noise units and ends each alarm interval with a
level tracker that stops after 60 days. D0 at the thresholds of experiment 43 (6.0 noise units at
the 4-hour window and 4.0 at the 24-hour window) is a separate arm, although the search returned
those same thresholds. D1 is the neighbour-balance detector of experiment 37, which also requires a
neighbouring substation to step the opposite way (thresholds 3.75 at the 4-hour window and 2.5 at
the 24-hour window). D2 is D0M plus a neighbour veto ([experiment
37](00037-switching-first-look/result.md), [experiment
43](00043-sw2-1-detector-scoreboard/result.md), [experiment
55](00055-sw2-8-neighbour-veto-single-series-detector/result.md), [experiment
58](00058-sw2-8b-donor-step-recut-and-stronger-veto/result.md), [experiment
66](00066-sw2-16-end-rule-as-baseline/result.md)). Experiment 49 also uses D0 as the label of a
spread-factor contrast, and experiment 87 uses D1 and D4 as labels of decisions.

### ENS, ERA5, and CAMS

**ENS is the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble forecast with 51
members, and ERA5 and the Copernicus Atmosphere Monitoring Service (CAMS) irradiance are estimates
of past weather.** ENS is a forecast issued before the weather happens, and the live forecast is
given ENS. ERA5, the fifth-generation ECMWF reanalysis, arrives about 5 days late ([experiment
10](00010-elexon-price-lags-stage1a-demand/result.md)), so a result that uses ERA5 as a forecast
input is a ceiling and not a usable product. CAMS is a satellite-based estimate of sunlight
([experiment 7](00007-cams-static-correction-stage1a-residual/result.md)). An XGBoost model trained
on ERA5 or CAMS learns from weather that was not forecast.

### Frozen XGBoost model

**A frozen XGBoost model is trained once, on data up to the start of the validation year, and is not
refitted during that year.** In [experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md), the frozen forecast without
the 14-day correction scores 6.927 for Primary substations (S1), and the frozen forecast with the
correction is the stack (S2, 5.543).

### Half-life

**A half-life of 0.5 years gives a training row a weight of 0.5 raised to the power of the row's age
in years divided by 0.5.** The weight therefore halves for every 0.5 years of age
(`ml-experiments/00017-training-window-and-retraining-rule-drift-demand/e00017_fit.py`, [experiment
17](00017-training-window-and-retraining-rule-drift-demand/result.md)).

### Injected and real load transfers

**A load transfer is a switching operation that moves part of one Primary substation's demand (the
donor) onto neighbouring substations for hours or days.** An injected transfer is a synthetic
transfer that the experiment adds to the real power readings, so its start, end, and size are known.
A real transfer is one of the 66 hand-labelled operations of [experiment
41](00041-sw2-0a-labeller-gate/result.md), of which 35 last over 1 day. A transfer is "large" when
it moves at least 25% of the receiving substations' load.

### Interval-level convention

**Each interval states its level, and the level depends on how many contrasts the experiment
planned.** An experiment that plans one contrast reports a 95% interval. An experiment that plans N
contrasts divides the 5% error rate across them with a Bonferroni correction, so nine contrasts give
a 99.44% interval (1 minus 0.05 divided by 9). Intervals resample whole calendar months
(`.claude/skills/ml-experiment/SKILL.md`, [experiment
19](00019-primary-demand-stack-vs-manual-heuristic/result.md)).

### `manual_heuristic`

**`manual_heuristic` is a baseline forecaster that predicts each half-hour as an ensemble of the
power observed at the same weekday and time of day in earlier weeks and years.** Under its
configuration the 13 ensemble members are the same time in each of the last 6 weeks and the same
time 49 to 55 weeks earlier; a lag that would not yet have been observed at the forecast origin is
absent. The forecaster consumes no weather
(`packages/baseline_forecasters/src/baseline_forecasters/manual_heuristic.py`).

### MW and MVA

**A series is metered in MW (signed active power, which can be negative when the substation exports)
or in MVA (apparent power, which is never negative).** The unit is the `units` column of the
metadata table. Eight of the 16 trial Primaries are metered only in MVA, so solar output behind such
a Primary appears as demand in daylight. See the [data-processing
page](data-processing.md#series-and-units) and experiments
[56](00056-mva-only-primaries-skill-against-mw-primaries/result.md) and
[79](00079-pvdis-7-mw-at-mva-only-primaries/result.md).

### N1

**N1 is the single-stage XGBoost quantile model, one per series, that is trained on the ENS control
member with recent power as inputs.** It receives power lags measured from the forecast origin and
weather columns from the ENS member, and its output quantiles are pooled across the 51 ENS members
([experiment 13](00013-primary-demand-against-stored-baselines/result.md)).

### Noise unit

**One noise unit is 1.4826 times the median absolute deviation of a substation's 24-hour residual
step on odd-numbered months.** A step of 4.0 noise units is therefore four times the usual size
([experiment 66](00066-sw2-16-end-rule-as-baseline/result.md)).

### NWP

**NWP is numerical weather prediction: a physical weather model run forward from the current
state of the atmosphere.** ENS is the NWP product that the live forecast consumes. Other NWP
products in the experiments include the Met Office UKV and NOAA's GEFS.

### PF14

**PF14 is the benchmark end rule of the switching experiments, written after seeing results.** It
ends each alarm interval at the first opposite step that is at least half the onset step (and at
least 1.5 noise units) within 14 days, takes the largest step back within one window after that, and
keeps the tracked end when no step qualifies ([experiment
66](00066-sw2-16-end-rule-as-baseline/result.md), [experiment
69](00069-sw2-17-missed-onsets-of-the-end-rule/result.md)).

### Pooled mixture

**A pooled mixture applies the XGBoost quantile model to each of the 51 ENS members and pools the
resulting quantiles as equal-weight points.** The forecast quantiles are the weighted quantiles of
the pooled values ([experiment 13](00013-primary-demand-against-stored-baselines/result.md),
[experiment 2](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md)).

### Pre-registration, decision rule, and verdict words

**An experiment pre-registers its hypothesis, prediction, and decision rule in a `result.md` that is
committed with status `planned` before any code runs.** The decision rule states which interval and
threshold make each verdict, so the verdict follows from the numbers and not from the author's
reading. The verdicts are `supported`, `partly supported`, `small gain` (the interval excludes zero
in the predicted direction but misses the threshold), `not supported`, `refuted`, `inconclusive`,
`invalid`, and `failed`. See [How an experiment is
recorded](README.md#how-an-experiment-is-recorded) and `.claude/skills/ml-experiment/SKILL.md`. A
write-up marks a verdict that its decision rule does not cover as not pre-registered, as [experiment
13](00013-primary-demand-against-stored-baselines/result.md) does.

### Primary substation

**A Primary substation steps voltage down from 33 kV or 66 kV to 11 kV.** NGED operates 1,161
Primary substations ([NGED's network and its data](../background/network.md)), and 16 of the 32
trial series are Primaries. A Primary is fed from a bulk supply point, which is a higher-voltage
substation (see [Bulk supply point and grid supply
point](#bulk-supply-point-and-grid-supply-point)).

### Quantile score and pinball loss

**The quantile score is twice the probability-mass-weighted mean of the pinball losses over the
forecast quantile levels, in points of capacity.** The pinball loss of a quantile at level τ is τ
times the shortfall of the forecast below the observation, or (1 − τ) times the excess of the
forecast above it. Each level's weight is the probability mass between the mid-points to its
neighbouring levels, so the score is not a plain mean over the levels. Experiments 33, 34, and 48
use nine levels (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, and 0.95), and experiment 87 and
later use 11 levels (those nine plus 0.01 and 0.99) (`pinball_quantile_score` in
`ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations/e00087_common.py`). Scores
from experiments with different levels, row sets, or series cannot be compared.

### Recall (MW-weighted)

**MW-weighted recall is the share of the moved megawatts that sit in found transfers.** A detector
finds a transfer when it raises an alarm starting within 2 hours of the transfer's start and places
the end within the larger of 6 hours and 10% of the transfer's duration
([experiment 66](00066-sw2-16-end-rule-as-baseline/result.md),
[experiment 101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)).

### Refit and rolling refit

**A refit retrains the XGBoost model on newer data at a scheduled interval, such as weekly or
monthly.** A rolling refit (P-roll) retrains monthly on a rolling 12-month window with 150 boosting
rounds ([experiment 34](00034-retraining-cadence-and-cheap-proxies-demand/result.md)). Weekly refits
are the cadence of BG2.

### Residual

**In the switching experiments, the residual is the measured power minus the forecast of an XGBoost
demand model trained on ERA5 weather that uses no past power.** The detectors look for steps in the
residual ([experiment 66](00066-sw2-16-end-rule-as-baseline/result.md)).

### RO-NC

**RO-NC is the refit-offset neighbour-count detector.** RO-NC is PF14 with two changes. The step
is taken from whichever of two residual models, refitted half a month apart, did not refit within
24 hours. The candidates are scored by a logistic regression that includes the count of neighbours
stepping the same way as the alarming substation ([experiment
82](00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md), [experiment
101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). X is the refit-offset
detector alone. The experiment scripts under `ml-experiments/` still name RO-NC `XL` in code, for
example in the arm names `lagmw_xl` and `mask_xl` of experiment 86, so a search of the code for
`XL` finds the RO-NC detector.

### Season-year

**A season-year is a 12-month period, and its start month depends on the series of work.** In the
solar-disaggregation experiments it runs from September to August, such as 2019/20 ([experiment
39](00039-pv-disaggregation-cams-injection/result.md)). In the differentiable-physics series it
runs from July to June (`FIRST_SEASON_MONTH` in `packages/studies/src/studies/dp_calendar.py`).

### Skill

**Skill is 100 times (1 minus the best guess's quantile score divided by `manual_heuristic`'s
quantile score), pooled over the series being compared.** A positive skill means the best guess
beats `manual_heuristic` ([experiment
56](00056-mva-only-primaries-skill-against-mw-primaries/result.md)).

### Spread factor

**The spread factor scales each forecast quantile's distance from the median.** A quantile q becomes
the median plus the factor times (q minus the median), so a factor of 1.15 widens the band by 15%
and a factor below 1 narrows it ([experiment
19](00019-primary-demand-stack-vs-manual-heuristic/result.md)). For solar and wind farms the factor
is fitted on earlier cases to bring the 5% to 95% coverage near its target
([experiment 33](00033-best-guess-architecture-vs-production/result.md)).

### Stack

**The stack is N1 with the 14-day correction and a spread factor of 1.15, and the experiments differ
on the half-life.** The stack of [experiment
19](00019-primary-demand-stack-vs-manual-heuristic/result.md) weights older training rows with a
half-life of 0.5 years. The stacks of experiments
[33](00033-best-guess-architecture-vs-production/result.md), 34, and 48 use a half-life of 2 years,
and are trained once. The experiment 87 scripts use the same 2-year weighting for its stack, which
its write-up does not state, and there the stack is the frozen forecast plus the 14-day correction.

### Stages 0, 1a, and 2

**Stage 1a is an XGBoost quantile model per series, trained on analysis weather (ERA5 and CAMS),
with no power lags and no lead time.** Stage 0 maps ENS weather toward the weather that stage 1a
trained on (a quantile mapping in experiments 16 and 20). Stage 2 is a calibrator, a second XGBoost
model that corrects the stage 1a quantiles and can use lagged power ([staged probabilistic
forecast](../roadmap/staged-probabilistic-forecast.md)).

### Tail coverage

**Tail coverage is the coverage of a quantile in the tails: the share of observed values below the
0.01 and 0.05 quantiles or above the 0.95 and 0.99 quantiles.** A correct forecast has a tail
coverage equal to the quantile's level on the lower tail, and equal to one minus the level on the
upper tail. Experiment 87 reports it at those four levels beside the pinball loss there ([experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)).

### `xgboost_baseline`

**`xgboost_baseline` is the production-style forecast: the stored forecast of the XGBoost model that
the project's production recipe produced.** The experiments read it from the `power_forecasts` Delta
table for the fold `mid_2025_to_mid_2026`. Experiment 87 did not confirm whether the local Dagster
cross-validation assets or S3 wrote that table, so none of the experiments we read confirms that the
stored forecast matches the deployed one ([experiment
13](00013-primary-demand-against-stored-baselines/result.md), [experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)).
