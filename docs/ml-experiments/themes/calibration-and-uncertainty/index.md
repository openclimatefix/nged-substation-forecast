# Calibration and uncertainty

**Correcting the spread of the forecast's ranges has gained little so far, and combining forecasts
has gained more.** Each forecast gives a range of likely values. The 5% to 95% range should contain
90% of measured values, and that share is the range's coverage. For solar farms, the range contained
94.3% to 95.3% of measured values before one correction and 89.5% to 91.8% after it, with no
detectable change in accuracy. For wind farms, the range over-covered only in the first two days
(95.3% and 94.4%). A small wind gain appeared only for a correction chosen after the results were
seen. For Primary substations, which step 33 kV or 66 kV down to 11 kV, no correction passed the
pre-registered tests. The gains that cleared their thresholds came from combining forecasts. Pooling
the quantiles from the 51 weather-forecast members, using an XGBoost model trained on estimates of
past weather, beat an XGBoost model trained on archived weather forecasts by 0.62 points of capacity
for wind and 0.24 for solar ([experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md)). Starting the wind
forecast from the turbine power curve lowered the wind error by 0.055 ([experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md)). Averaging the quantiles
of three stored forecasts lowered the Primary substation error by about 0.1 ([experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md)). One point of capacity
is 1% of a series' capacity, and a lower score is better. Every result rests on one validation year
and a few dozen series. The terms below are defined once for the whole page.

## Terms used on this page

- **Primary substation:** a substation that steps 33 kV or 66 kV down to 11 kV. The trial area has
  16.
- **Point of capacity:** 1% of a series' capacity. A lower score is better.
- **Quantile score:** twice the mean pinball loss over the quantile levels, in points of capacity.
  The pinball loss scores one quantile: it charges a measurement above the quantile at the quantile
  level and one below it at the complement. Experiment 35 and experiment 80 use 9 levels, and
  experiments 97 and 99 use 11.
- **CRPS:** the continuous ranked probability score, estimated from 9 quantile levels, in points of
  capacity. Experiments 2, 16, and 20 report it.
- **Scores across experiments:** the experiments use different row sets, series, and quantile
  levels, so a score from one experiment cannot be compared with a score from another.
- **Interval levels:** each interval is stated with its level beside it. The levels differ (95%,
  98.33%, 98.75%, 99.4%, 99.58%). A 95% interval is not split across contrasts, and the other levels
  divide a 5% error rate across the contrasts an experiment planned.
- **ENS and ERA5:** ENS is the European Centre for Medium-Range Weather Forecasts ensemble forecast,
  which has 51 members. The archived ENS forecasts are what the live forecast is given. ERA5 is an
  estimate of past weather, so a model trained on ERA5 learns from weather that was not forecast.
- **Coverage of one quantile and of a band:** the coverage of one quantile, such as q05, is the
  share of measured values at or below it, so q05 should cover 0.05 and q95 should cover 0.95. The
  90% band between q05 and q95 should contain 90% of measured values. An exceedance is a measured
  value beyond a tail quantile.
- **Stages 0, 1a, and 2:** stage 1a is an XGBoost model per series, trained on ERA5, that turns
  weather into power quantiles. Stage 0 maps the ENS weather onto the weather the stage 1a model
  trained on. The stage 2 calibrator is a second XGBoost model that corrects the stage 1a quantiles.
- **Pooled mixture:** the quantiles of stage 1a, applied to each of the 51 ENS members, pooled as
  equal-weight points.
- **Spread factor:** a number by which the quantiles are scaled about the median. A factor below 1
  narrows the band.
- **Base score and base margin:** XGBoost starts from a base score that it estimates from the
  training labels. A base margin supplies a starting prediction for each row instead, and XGBoost
  then ignores the base score. For wind, the base margin is the power the turbine power curve gives
  at the forecast wind speed.
- **Lead band:** a group of forecast lead times, such as 0 to 24 hours or 1 to 2 days.
- **BG2 and its weekly stage:** BG2 is the second best-guess recipe of [experiment
  00087](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md). For Primary
  substations it is a five-model quantile blend with an offset for embedded solar, and it scores
  5.260 on that experiment's rows. Experiments 97 and 99 use only BG2's weekly-retrained XGBoost
  stage (500 boosting rounds, 5.355 in experiment 87), which they call BG2 and the best guess. This
  page calls that stage BG2's weekly stage.
- **N1, `a7_cs`, and the stack:** the three stored forecasts that experiment 35 blends for Primary
  substations. N1 is a single-stage XGBoost quantile model trained on ENS weather. `a7_cs` is an
  XGBoost model with seven weekly-analogue input columns. The stack is N1 with a correction from the
  last 14 days of the forecast's own errors and a spread factor of 1.15. Earlier experiments (19,
  20, 30, and 33) had already scored these three Primary substation forecasts.
- **Control member:** the unperturbed forecast of the 51 ENS members.

## Does it work?

**Pooling the ERA5-trained quantiles over the ENS members beat an XGBoost quantile model trained on
ENS weather, and a stage 2 calibrator on top did not help.** In [experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md), the pooled mixture
scored 0.62 points lower in CRPS for wind (95% interval -0.99 to -0.26) and 0.24 points lower for
solar (-0.36 to -0.06). The gain sits in the first six days. The stage 2 calibrator scored worse
than the mixture by 0.40 points for solar (+0.26 to +0.61) and by 0.26 for wind with an interval
that includes zero (-0.17 to +0.64). The experiment used 6 solar farms and 3 wind farms in one
validation year and no lagged power, so it cannot say whether a calibrator that sees recent power
would help.

**One spread factor brought solar's band to about 90% coverage, and the CRPS did not change
detectably.** In [experiment 00016](../../00016-pooled-mixture-calibration-solar-wind/result.md),
the uncorrected solar band over-covered in every lead band (94.3% to 95.3%). One factor (fitted
between 0.90 and 1.00, median 0.95) gave 89.5% to 91.8%, with a CRPS change of +0.006 (99.4%
interval -0.004 to +0.024). Wind over-covered in the first two days only (95.3% and 94.4%), and no
planned correction fixed that. A spread factor on two lead groups (0 to 2 days and 3 to 15 days),
chosen after seeing the results, gave 90.7% to 92.5% in every wind lead band and changed the CRPS by
-0.021 (99.4% interval -0.042 to -0.002), a small gain.

**For wind, the power-curve base margin carries the gain, and the spread factor is a coverage fix.**
In [experiment 00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md), the base
margin lowered the wind CRPS by 0.055 (99.58% interval -0.110 to -0.007). The margin with the spread
factor lowered it by 0.071 (-0.133 to -0.014). The spread factor's extra -0.016 (-0.036 to +0.004)
is not detectable, and [experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md) advises treating the wind
spread factor as a coverage fix. For solar, none of the three changes helped detectably: a
CAMS-trained stage 1a model (-0.009), a stage 0 quantile mapping onto CAMS irradiance (+0.036), or a
spread factor (+0.006). The full solar stack was 0.035 higher (-0.075 to +0.116), and that interval
includes zero. The solar intervals are 0.19 to 0.30 points wide, so a gain under 0.1 points could
not be resolved.

**No recalibration of the stored Primary substation quantiles passed the pre-registered tests.** In
[experiment 00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md), the
stored q05 of BG2's weekly stage covered 0.035 and the stored q95 covered 0.938 at 12 to 36 hours
lead, where 0.05 and 0.95 are nominal. The stored median covered only 0.45, so the stored median
sits slightly low. Coverage of q95 swung from 0.88 to 0.97 between months, which a table fitted on
earlier months cannot follow. No arm passed both pre-registered tests, and the verdict is `refuted`.
One arm (Pareto tails, arm 6) lowered the Primary substation quantile score at 12 to 36 hours by
0.00009 (95% interval -0.00015 to -0.00001) but failed the tail test. The experiment scored 9 months
(October 2025 to June 2026) at 16 Primary substations and 4 bulk and grid supply points. Nine months
are too few to tell a real improvement from noise, so the experiment is inconclusive about
recalibration in general.

**Averaging the quantiles of three stored forecasts lowered the Primary substation quantile score by
0.097 points, and the solar and wind gains were unresolved.** In [experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md), the lead-band blend of
the monthly-retrained N1, `a7_cs`, and the stack scored 5.025 against 5.122 for the best of the
three (98.33% interval -0.187 to -0.026). One fixed weight vector scored 5.020 and was -0.102
against the best member (interval -0.201 to -0.025, level not stated in the write-up). The blend was
scored against its own members and not against BG2 or BG2's weekly stage. The lead-band blend uses a
separate set of three weights for each lead band. Solar (-0.039, 98.33% interval -0.115 to +0.031)
and wind (-0.074, -0.194 to +0.055) had intervals that include zero.

**A loss based on the CRPS of the pooled members recovered about half of the harm that extra members
do to the pinball loss.** In [experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md), the CRPS
loss beat the pinball loss (same 6 members per run, same custom optimiser) by 0.039 points (95%
interval -0.057 to -0.018), a `small gain`. That pinball arm scored 5.622, against 5.547 for the
pinball loss on the control member alone, so 6 members cost it 0.075 and the CRPS loss recovered
0.039 of that. The pinball loss on 6 members with XGBoost's built-in optimiser scored 0.052 points
worse than the control member alone (95% interval +0.029 to +0.074). Against the control-member
recipe the CRPS arm is +0.036 points (95% interval -0.022 to +0.111), so the pilot cannot say
whether the CRPS arm is better or worse. Its 12 to 36 hour tails stayed outside 0.01 of nominal (q95
covered 0.924 and q05 covered 0.061).

**Starting an XGBoost model from zero, instead of from the base score XGBoost estimates, did not
lower the quantile score of any XGBoost model that serves.** In [experiment
00080](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md), the Primary substation
difference was -0.017 points (98.75% interval -0.072 to +0.054), the bulk and grid supply point
difference was +0.006 (-0.091 to +0.092), and the solar farm difference was -0.010 (-0.036 to
+0.028). Wind XGBoost models without the power-curve base margin were worse with a zero start by
+0.049 (95% interval +0.032 to +0.065). The verdict is `inconclusive`: no gain could be shown. Four
times as many boosting rounds scored 0.211 points worse (95% interval +0.100 to +0.352) at 8 Primary
substations.

![Coverage of the 5% to 95% band by lead
band](../../00016-pooled-mixture-calibration-solar-wind/figure_1.svg)

**Figure 1: the uncorrected 5% to 95% band over-covers for solar in every lead band and for wind in
the first two days, and a spread factor corrects solar.** Source: [experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md).

![CRPS difference of each part of the wind and solar
stacks](../../00020-solar-wind-stack-cams-stage0-spread-margin/figure_1.svg)

**Figure 2: the stack of changes lowers the wind CRPS and does not lower the solar CRPS.** Negative
values are better. Source: [experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md).

![Primary substation tail coverage of each recalibration, 12 to 36 hour
lead](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/figure_1_primary_tail_coverage.svg)

**Figure 3: at 12 to 36 hours lead, none of the recalibrations of the Primary substation quantiles
of BG2's weekly stage, including the Student-t and skew-normal arms, moved q05 or q95 coverage by
more than 0.013.** Source: [experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md).

![Blends minus the best single
member](../../00035-lead-weighted-quantile-blend-of-stored-models/figure_1.svg)

**Figure 4: blending lowers the Primary substation quantile score, and the solar and wind gains are
unresolved.** Negative values are better. Source: [experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md).

## Best design so far

**These are the best of the recipes tested in this theme, and the experiments did not score them
against each other.** The Primary substation blend of experiment 35 was scored against its own three
members. Experiments 97 and 99 were scored against BG2's weekly stage, which scored 5.356 on
experiment 99's rows. No experiment scored the blend against BG2 or against BG2's weekly stage.

**For Primary substations, the blend of three stored forecasts is the best design tested.** Average
the quantiles of the monthly-retrained N1, `a7_cs`, and the stack with one fixed weight vector,
which scored 5.020 against 5.025 for a separate set of weights for each lead band ([experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md)). The stored quantiles
of BG2's weekly stage were left as stored, because no recalibration passed the pre-registered tests
([experiment 00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)).
Training stays on the control member with the pinball loss, because the CRPS loss on 6 members was
not shown to beat it ([experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md)).

**For solar and wind farms, the best design tested is the pooled mixture with small additions.**
Pool the ERA5-trained quantiles over the ENS members ([experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md)). Add a spread factor
for solar if a 90% band is wanted ([experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md)). Add a power-curve base margin
for wind ([experiment 00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md)).
Experiments 16 and 20 found no detectable solar CRPS gain over the pooled mixture.

## Candidates for production

Four experiments have `production_status: candidate`, and none has `adopted` status. The first three
would change how the solar and wind mixture is made. The fourth would add forecasts to the Primary
substation service. A model release plan decides whether any of them is ported.

- **[experiment 00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md), pooled
  mixture:** pool the ERA5-trained quantiles over the ENS members for solar and wind. The stage 2
  calibrator did not help.
- **[experiment 00016](../../00016-pooled-mixture-calibration-solar-wind/result.md), solar spread
  factor:** scale the solar mixture's quantiles about the median by one fitted factor. The CRPS did
  not change detectably. The wind two-group factor was chosen after seeing the results and has not
  been tested on another year.
- **[experiment 00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md), wind base
  margin:** add a power-curve base margin to the wind stage 1a quantile model, which lowered the
  CRPS by 0.055 points (99.58% interval -0.110 to -0.007). Nothing detectably helped solar.
- **[experiment 00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md), Primary
  substation quantile blend:** serve a fixed-weight average of the quantiles of the
  monthly-retrained N1, `a7_cs`, and the stack. One fixed weight vector scored 5.020 and was -0.102
  against the best member (interval -0.201 to -0.025, level not stated in the write-up). The
  experiment's verdict contrast, the lead-band blend, was -0.097 (98.33% interval -0.187 to -0.026).
  The blend needs three forecasts, one of them refitted monthly, and it was not scored against BG2.

## Gotchas and implementation details

**Report the pooled mixture beside any calibrator.** Without the mixture, a calibrator that is worse
than pooling can look like a gain over an XGBoost model trained on ENS weather. [experiment
00002](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) reported the mixture
beside the stage 2 calibrator and found the calibrator worse. Check again if the calibrator gets new
inputs, such as lagged power, which that experiment did not use ([experiment
00005](../../00005-lag-placement-stage1b-vs-stage2-demand/result.md) found lagged power helped for
16 Primary substations).

**Fit each correction only on valid times before the first run it corrects.** [experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md) and [experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md) did so, and
[experiment 00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)
asserted it before scoring. A fit that includes later outcomes would flatter the correction.
[experiment 00016](../../00016-pooled-mixture-calibration-solar-wind/result.md) required 5,000
earlier non-trivial cases per fit, so with the `spread_band` arm the wind 0 to 24 hour band was
fitted in only 3 months and the 1 to 2 day band in 8, and the other months stayed uncorrected. Check
again if the validation period or the series count grows, because the minimum is then reached
earlier.

**Compute solar coverage on non-trivial cases only.** [experiment
00016](../../00016-pooled-mixture-calibration-solar-wind/result.md) counted only cases whose
uncorrected 95% quantile exceeds 1% of capacity, so that night-time solar zeros do not inflate the
coverage. Check again for any series that is not zero at night.

**Refit the spread factor for each arm on that arm's own earlier cases.** [experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md) did so, because a factor
from [experiment 00016](../../00016-pooled-mixture-calibration-solar-wind/result.md) would not fit
different quantiles. Check again if the quantiles come from an XGBoost model that [experiment
00020](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md) did not stack.

**Score the coverage and pinball loss of q01, q05, q95, and q99 with the number of exceedances, and
store the quantile levels needed to do so.** [experiment
00097](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md) scored those four
levels from stored forecasts at 11 levels (0.01, 0.05 to 0.95, and 0.99), and called a cell with
fewer than 30 days holding an exceedance untestable. Nine scored months gave intervals of plus or
minus 0.02 on q95 coverage, wider than the 0.01 bar. Check again once 2 years of scored runs exist.

**Test a starting value only on XGBoost models without a power-curve base margin.** A base margin
makes XGBoost ignore its base score, so [experiment
00080](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md) found that a zero start
changed nothing on the wind XGBoost model with the margin. Check again if the XGBoost version or
objective changes.

**Run a repeat fit to measure GPU XGBoost's noise floor.** GPU fits are not bit-reproducible. In
[experiment 00080](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md), a repeat fit
of the frozen XGBoost model scored +0.004 points apart at Primary substations and -0.007 at grid
supply points. Those differences were above the planned tolerance of 0.001 and well below the
intervals, and in weekly repeat fits single quantiles moved by up to 0.06 of capacity. Check again
on a different GPU or XGBoost version.

**Hold the retraining schedule fixed within a contrast.** The CRPS-loss arms of [experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md) were
refitted monthly on a rolling 12-month window with 150 boosting rounds and a learning rate of 0.15.
They scored 0.19 to 0.23 points above BG2's weekly stage, which retrains weekly with 500 rounds, and
the schedule probably accounts for the gap, so the absolute scores cannot be set beside that
stage's. Check again if the schedule of either side changes.

**Pool pinball-trained quantiles with equal weights, not with the CRPS atom masses.** The CRPS
treats each of the 11 quantiles as a point with a probability mass, called an atom mass, and the
masses are unequal (0.02 at q01 and q99, 0.26 at q25 and q75). In [experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md), pooling
the pinball arm's quantiles with the CRPS masses scored 0.025 points worse. Check again if the
pinball arm is replaced by a CRPS-trained one.

**Apply the 14-day correction to each member before blending, not to the blend, and leave the
monthly-retrained N1 uncorrected.** The correction adds the forecast's own errors from the last 14
days to the next forecast. In [experiment
00035](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md), the blend with the
monthly-retrained N1 uncorrected beat the blend with every member corrected by 0.135 points.
Correcting the blend afterwards scored 0.048 points worse (95% interval +0.008 to +0.101), but that
contrast favours the corrected-first arm, because that arm's weights were chosen on its own scores.
Check again if the members change.

## What we have not settled

**Several questions are open.** Whether a calibrator that sees recent power helps solar and wind
farms is untested, although [experiment
00005](../../00005-lag-placement-stage1b-vs-stage2-demand/result.md) found it helped Primary
substations. The wind two-group spread factor has not been tested on another year. [experiment
00099](../../00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/result.md) suggests
trying a tail-specific spread factor, or pooled-member spread statistics as input columns, to fix
the Primary tails that its CRPS loss left in place. The blend gains for solar and wind may exist on
more than 6 solar farms and 3 wind farms. Whether the month-to-month swing in Primary tail coverage
can be tracked needs at least 2 years of scored runs. The [Research plan](plan.md) lists what is
planned next, and the [Experiments](experiments.md) page lists every experiment tried.
