---
id: "00188"
slug: switching-d10-audit-and-onset-location-in-look-back
title: >-
  Are D10's extra errors from unlogged load shifts, and does locating the onset inside the look-back
  help the mask?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00188-switching-d10-audit-and-onset-location-in-look-back/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible rule for
  "unlogged event" (part 1) and for locating an onset (part 2), so both are fixed below before any
  score is read. Callers: this folder's scripts only; they are copies of the modules of experiments
  156 and 171 and read those experiments' stored tables, cached forecasts and injected worlds.
hypothesis: >-
  Part 1. Experiment 171 found that ignoring recent power while a gated pair alarm is active scores
  0.030 points of effective capacity worse than no mask on rows outside logged transfers (95%
  interval -0.011 to +0.074), and that leaving out one substation, D10, moves the estimate to
  +0.002. If D10's gated alarms are real switching that NGED did not log, those rows are not
  ordinary days, and treating them as events leaves a no-harm interval inside +/-0.03. Part 2. The
  alarm for a transfer is known hours to days after the step, but the forecast runs every 6 hours,
  so at each run the readings since the step are in the look-back. A step-fitting search over those
  readings finds the onset more precisely than the alarm time, and masking from the found onset
  gives a smaller forecast error in the first 24 hours after the alarm than masking from the alarm
  time.
prediction: >-
  Written before any result. Part 1: with the confirmed unlogged-event rows removed as well, the
  pooled 27-month gated-mask-minus-no-mask interval lies inside -0.03 to +0.03 points of effective
  capacity, and the November to March interval lies inside -0.05 to +0.05. The prediction fails if
  the harm stays spread across ordinary days, that is, if most of D10's gated alarms are not
  confirmed by the rule below or the interval stays wider than the limit. Part 2: masking from the
  located onset cuts the extra error (the score above the score of a mask that starts at the true
  onset) in the first 24 hours after the alarm by at least one third, compared with masking from
  the alarm time. The prediction fails if the located onset is no better than the alarm time or if
  the located onset's median error is larger than the 6-hour spacing of the forecast origins
  (12 half-hours).
decision_rule: >-
  Pre-registered before any forecast score is read. PART 1 (D10 audit). Candidate events are all
  gated pair-alarm intervals (balance score 0.7 or more, experiment 171) at the six decision
  substations. A candidate is a confirmed unlogged event when it is not matched to a logged
  transfer (experiment 171's rule: step from 24 hours before the logged onset to the logged end)
  and a measure that is independent of the alarm's own statistic confirms it. The measure is the
  difference between the median residual in the 12 hours after the alarm's step stamp (starting 1
  hour after) and the median residual in the 12 hours before (ending 1 hour before), for the alarm's
  series (own step) and for each partner series named by the alarm (partner steps). The candidate is
  confirmed when |own step| is at least 0.3 MW, the sum of partner steps has the opposite sign, and
  the ratio |sum of partner steps| / |own step| lies in 0.5 to 2.0. The thresholds are fixed here;
  the gate itself uses peak tests and ratios of 0.7 to 1.43 on a different statistic. A placebo
  applies the same measure at the same stamps shifted by +/-7, 14, and 21 days and reports the
  share confirmed, which is the rate at which the rule confirms by chance. Rows: experiment 171's
  rows (every cached run, six Primary substations, 2024-04 to 2026-06, rows with a reading outside
  logged transfers) with the rows whose valid time lies from a confirmed event's step to 72 hours
  after its end also removed. Scored and bootstrapped exactly as in experiment 171 (twice the mean
  pinball loss over 11 quantiles, whole calendar months resampled, 20,000 draws, 95%). G1': pooled
  MASK_GATED minus K0 has an interval inside -0.03 to +0.03. G2': the same on November to March
  rows inside -0.05 to +0.05. Verdict for part 1 follows experiment 171: `supported` when both
  hold, `partly supported` when only G1' holds, `refuted` when the pooled point estimate is above
  +0.03, `inconclusive` otherwise. Reported beside it: how many gated alarms per substation are
  confirmed, the contrast on the rows of confirmed events alone, on the rows of unconfirmed gated
  alarms alone (the harm that remains), and D10 alone. Because the rows with a gated alarm active
  carry the whole difference, removing confirmed events can lower the contrast by construction; the
  informative results are the confirmed share, the placebo share, and the unconfirmed-alarm
  contrast. PART 2 (onset location). The alarm families are the pair alarm (every pair alarm,
  experiment 156's headline window) and RO-NC (experiment 147's intervals). Truth onset: the
  injected onset in experiment 141's 44 injected worlds (primary; exact), and the logged onset for
  real transfers (secondary; the log is coarser than the stamp). Forecast origins: every 6 hours,
  phase of experiment 171's runs, so the cached daily runs sit on the grid; an alarm is actionable
  at the first origin whose latest data stamp is at or after the alarm's known stamp. LOCATED
  onset: at that first origin, using only data up to its latest stamp, search stamps t from 24
  hours before to 24 hours after the alarm's own step stamp (at least 4 half-hours of data after t)
  for the largest |D(t)| of the right sign, where D(t) is the mean of 12 half-hours from t minus
  the mean of 12 half-hours before t (the after-window shrinks to the data available), on the
  alarm series' residual; for the pair alarm D is the own step minus the sum of the partners' steps
  over the same stamps. Location error is the located stamp minus the true onset, in half-hours;
  reported as median and 90th percentile of the absolute error, and the median signed error, for
  the alarm time (known stamp minus true onset), the detector's own step stamp, and the located
  onset. Mask arms (K0 filter with readings set missing, experiments 156 and 171): MASK_ALARM
  (readings missing from the first actionable origin's latest stamp onwards), MASK_STEP (from the
  detector's step stamp, experiment 171's arm), MASK_LOCATED (from the located onset), MASK_TRUE
  (from the true onset), and K0 (no mask). The end rule is experiment 148's, the same in every
  arm. Rows: all leads of every cached run whose forecast origin falls within 24 hours after the
  alarm's first actionable origin, from the 44 injected worlds (events at the six decision
  substations that the alarm finds) and from the real logged transfers it finds; scored against
  the true normal-running-arrangement series (injected) or the reading (real). Extra error of an
  arm = its mean quantile score minus MASK_TRUE's. Decisive contrast C2: 1 - (extra error of
  MASK_LOCATED) / (extra error of MASK_ALARM), pooled over alarms found in injected worlds, with a
  95% interval from resampling events. Part 2 is `supported` when the point estimate is at least
  1/3 and the interval excludes 0, `partly supported` when the point estimate is positive with an
  interval that excludes 0 but below 1/3, `refuted` when the point estimate is at most 0, and
  `inconclusive` otherwise. C1: the location error contrast (located minus alarm time), reported
  per family. The real transfers are reported in the same form and decide nothing: there are few
  of them. Matching, fixed before any mask score was read: in an injected world an alarm finds a
  transfer when an interval of the family on one of the substation's series has a step stamp within
  48 half-hours (24 hours) of the true onset, and the earliest-known such interval is used; for a
  real transfer experiment 171's rule applies (a step from 24 hours before the logged onset to the
  logged end). The cached forecasts are daily, so each found transfer is scored on the one run
  whose newest usable stamp lies from the first usable origin to 24 hours after it, all leads. A
  threshold of 1/3 is the maintainer's figure for porting a located-onset step, which
  adds a search to the serving path and so needs a gain this size. Post hoc and labelled: the
  located onset recomputed at each later origin; the mask from the located onset with a gated alarm
  only; the quantised origin's share of the error.
benchmark: >-
  Part 1: experiment 171's unmasked stack (K0) on the same rows, and its +0.030 (-0.011 to +0.074).
  Part 2: MASK_ALARM (mask from the alarm time) and MASK_TRUE (the best a located onset could do).
row_set_id: >-
  Part 1: experiment 171's rows (every cached run of six Primary substations, 2024-04-10 to
  2026-06-29) less confirmed unlogged events. Part 2: runs within 24 hours after each alarm, in
  experiment 141's 44 injected worlds and in real logged transfers; not comparable with the
  leaderboard.
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; onset-location error in half-hours.
code: ml-experiments/00188-switching-d10-audit-and-onset-location-in-look-back/
verdict: partly supported
best_arm: >-
  Part 2: masking from the detector's own step stamp or from the located onset, 2.815 and 2.828
  points of effective capacity on 247 injected transfers, against 3.127 from the alarm time and
  2.797 from the true onset. Part 1: none; the pooled gated-mask difference stays at +0.014 (-0.022
  to +0.051).
headline: >-
  Part 2 passes its pre-registered rule: masking from the located onset removes 91% of the extra
  error of masking from the alarm time (95% interval 74% to 98%). The detector's own step stamp
  removes 95%, so the search of the look-back adds nothing (located minus step stamp +0.013, -0.008
  to +0.037 points). Part 1 fails its prediction: with 8 confirmed unlogged events removed the
  pooled gated-mask-minus-no-mask difference is +0.014 (-0.022 to +0.051), wider than +/-0.03.
lessons: >-
  The detector already reports where the step happened, so masking back to that stamp once the alarm
  fires is the whole gain, and a second search over the look-back is not worth porting. Only 3 of
  D10's 13 unlogged gated alarm intervals are confirmed by an independent step test, so D10's
  extra error is not shown to be unlogged switching, and no-harm of the gated mask on ordinary days
  is still not shown.
take_home_message: >-
  Once a switching alarm fires, masking recent power back to the alarm's own step stamp removes
  almost all the extra error of masking only from the alarm time, and searching the look-back for a
  more precise onset adds nothing. D10's extra error under the gated mask is not shown to be
  unlogged switching, since an independent step test confirms only 3 of its 13 unlogged alarm
  intervals (its recall was not measured), so showing that the gated mask does no harm on ordinary
  days still needs more months of forecasts.
advice:
  - >-
    Mask recent power back to the detector's own step stamp once an alarm fires, not from the alarm
    time: the extra error falls by 95% (83% to 100%) (C2, figure 1).
  - >-
    Do not add a search of the look-back to locate the onset: the detector's step stamp is already
    a median 0 half-hours from the true onset, and the located onset is no closer and has a larger
    90th-percentile error (figure 3).
  - >-
    Do not assume D10's gated alarms are unlogged switching: an independent persistent-step test
    confirms 3 of 13 of its unlogged alarm intervals (its recall was not measured), and D10 alone
    still has a positive point estimate of +0.081 (interval -0.007 to +0.232) with the 3 removed
    (part 1, figure 2).
  - >-
    Do not read the real-transfer scores as evidence: 15 transfers were found by the alarms, and
    RO-NC's mask from the step stamp scored 12.9 against 8.3 with no mask on 6 of them (real rows).
production_status: not-for-production
adopted_in: none
---

**Masking back to the detector's own step stamp removes almost all the extra error of masking only
from the alarm time, and a search of the readings since the last forecast for a more precise onset
adds nothing.** Part 2 meets its pre-registered rule (`supported`), but its comparison arm, masking
from the alarm time, is an arm that experiment 171 never used. Part 1 misses its rule
(`inconclusive`): the interval of the gated mask's difference from no mask is not inside +/-0.03.
The overall verdict, `partly supported`, is a judgement, because the plan registered a verdict for
each part only.

**Background: what a mask is.** Our forecast corrects itself using recent power. A "mask" sets
recent power aside while a switching alarm is active. A pair alarm is a step in power at one
substation matched by an opposite step at a neighbouring metered substation. A gated alarm is a
pair alarm whose balance score is 0.7 or more, meaning the two steps are close in size. RO-NC is the
detector of [experiment 147](../00147-causal-live-end-rule-for-ro-nc/result.md), which fires on a
step in a refitted residual with support from neighbouring series. A step stamp is the half-hourly
timestamp at which a detector says the step happened. The alarm time is the later timestamp at
which the alarm became known: about 6 hours after the step for the pair alarm (a median of 12
half-hours), and 48 hours for RO-NC (96 half-hours). The look-back is the readings between the step
and the forecast origin, the moment a forecast run is issued (every 6 hours in production).

**Background: what experiment 171 found and what we asked.** [Experiment
171](../00171-switching-gated-mask-no-harm-pooled-months/result.md) found that the gated mask scores
0.030 points of effective capacity worse than no mask on 27 months of rows outside logged transfers
(95% interval -0.011 to +0.074), and that leaving out one substation, D10, removes the difference.
Scores are twice the mean pinball loss over 11 quantiles, in points of effective capacity (the
score in megawatts over the substation's AC capacity, times 100). A logged transfer is a movement of
load between two substations recorded in the switching log of National Grid Electricity
Distribution (NGED). The six substations scored, anonymised as D5, D9, D10, D11, D14, and D16, are
Primary substations. **Part 1** asked whether D10's gated alarms follow switching that NGED did not
log. **Part 2** asked whether the onset of a transfer can be located more precisely in the look-back
than the alarm time gives, and whether masking from the located onset beats masking from the alarm
time.

**Part 1 method.** Of the 49 gated alarms since 2024-04-10, 40 touch a scored substation, and 31 of
those match no logged transfer. We confirmed an alarm when a persistent step (the median residual
over the 12 hours after the alarm's step stamp minus the median over the 12 hours before) at its
substation had an opposite step of similar size (0.5 to 2 times) at its partner substations, using
thresholds fixed in the plan. We removed the rows from the step to 72 hours after each confirmed
event and re-scored experiment 171's rows with the same quantile score and the same
month-resampled interval. A row is one forecast for one half-hour from one forecast origin. The
forecast is a frozen generalised additive model with a [Kalman
filter](https://doi.org/10.1115/1.3662552), as in experiment 171.

**Part 2 method.** Experiment 141's 44 worlds hold 210 injected transfers, and the two alarms find
them 247 times in total (a transfer found by both alarms counts twice). We score those 247 matches
and 15 logged transfers found by an alarm. At the first 6-hourly origin that can see an alarm, we
searched the stamps within 24 hours of the detector's step stamp for the largest step with the
same sign as the alarm's step, using only readings up to that origin. The stamp found is the
"located" onset. Four masks differ only in the first masked stamp: the alarm time, the detector's
step stamp, the located onset, and the true onset. The extra error of a mask is its score minus the
score of the mask from the true onset, in points of effective capacity, on all leads of the daily
forecast issued in the first 24 hours after the alarm. C2 is the share of the alarm-time mask's
extra error that another mask removes: 1 minus the extra error of that mask over the extra error of
the alarm-time mask.

![Figure 1: Masking back to the step stamp removes the extra error; locating adds nothing](figure_1_headline.svg)

![Figure 2: Removing confirmed unlogged events leaves the pooled interval wider than 0.03](figure_2_part1_audit.svg)

![Figure 3: The detector's step stamp is already close to the true onset](figure_3_location_error.svg)

## Results

- **Part 2, decisive contrast C2 (injected, both alarms, 247 transfers).** Extra error over the
  true-onset mask: alarm time +0.330 (+0.133 to +0.529), step stamp +0.018 (0.000 to +0.038),
  located onset +0.031 (+0.005 to +0.059). Reduction from the alarm time to the located onset:
  0.907 (0.735 to 0.984), against the pre-registered 1/3. Verdict for part 2: `supported`.
- **What C2 does not show.** The step stamp gives a reduction of 0.946 (0.832 to 1.001), and the
  located onset minus the step stamp is +0.013 (-0.008 to +0.037) points. The located onset is no
  better, so the gain comes from masking back to the detector's step stamp, which experiment 171
  already does.
- **By alarm.** Pair alarm (87 transfers): reduction 0.779 (0.519 to 0.961); extra error alarm time
  +0.175, step stamp +0.042, located +0.039. RO-NC (160 transfers): reduction 0.936 (0.725 to
  1.021); extra error alarm time +0.415, step stamp +0.005, located +0.027.
- **Absolute scores, injected, both alarms.** No mask 3.328, mask from the alarm time 3.127, step
  stamp 2.815, located onset 2.828, true onset 2.797, and the stream without the injected
  megawatts 2.483. Masking from the alarm time barely beats no mask because the readings since the
  step, which carry the transfer, stay in the filter.
- **Location error, injected (half-hours; median |error| / 90th percentile).** Alarm time: pair
  12.0 / 25.8, RO-NC 96.0 / 97.0. Step stamp: pair 0.0 / 29.6, RO-NC 0.0 / 5.1. Located onset: pair
  0.0 / 33.6, RO-NC 0.0 / 16.0. The median located error is 0 half-hours, well below the
  12-half-hour (6-hour) spacing of forecast origins. The located onset is closer than the step stamp
  for 10% (pair) and 14% (RO-NC) of transfers and farther for 14% and 26%.
- **Sensitivity (post hoc).** Resampling the 39 worlds with scored transfers (of 44) gives 0.907
  (0.747 to 0.974). Leaving out any one substation leaves the reduction between 0.88 and 0.97. D10
  alone, 14 transfers: 0.780 (-0.351 to 0.965).
- **Real logged transfers (secondary, decide nothing).** 9 transfers found by the pair alarm and 6
  by RO-NC. Pair alarm: extra error over the true-onset mask alarm time -0.027 (-0.713 to +0.677),
  step stamp +0.151, located +0.095. RO-NC: alarm time -1.536, step stamp +2.991, located +2.742
  (all intervals include 0 except the RO-NC difference between alarm time and located onset, -4.279,
  -6.881 to -1.590). On 6 transfers RO-NC's mask from the step stamp scores 12.949 against 8.298
  with no mask, as in experiment 156.
- **Part 1, pre-registered contrast.** Experiment 171's rows reproduce at +0.030 (-0.012 to +0.074,
  1,579,711 rows; experiment 171 printed -0.011 for the lower end, because this experiment uses
  a different bootstrap seed). With 8 confirmed unlogged events removed (1,543,752 rows) the pooled
  difference is +0.014 (-0.022 to +0.051), half-width 0.036 against the limit of 0.03. November to
  March: +0.002 (-0.094 to +0.100), half-width 0.097 against 0.05. G1' and G2' both fail and the
  point estimate is below +0.03, so the verdict for part 1 is `inconclusive`, and the prediction
  that the interval would fall inside +/-0.03 failed.
- **Part 1, how many alarms were confirmed.** 8 of the 31 unlogged gated alarms at a decision
  substation (the plan's candidates). The same rule applied at whole-week shifts of each alarm's
  stamp confirms 5.6%, a placebo share that estimates how often the rule confirms an alarm by
  chance. Unlogged and confirmed per substation (gated alarm intervals on the substation's series,
  so one alarm counts at each substation it touches; 50 unlogged intervals from 31 alarms, and 10
  confirmed intervals from 8 alarms): D5 6 and 2, D9 0 and 0, D10 13 and 3, D11 14 and 0, D14 14 and
  4, D16 3 and 1.
- **Part 1, where the harm is.** Rows of confirmed events alone: +0.726 (-0.011 to +1.191). Rows of
  unconfirmed gated alarms alone: +0.212 (-0.002 to +0.615). D10 alone: +0.170 (+0.012 to +0.382)
  before and +0.081 (-0.007 to +0.232) after removal. So the point estimate stays positive where the
  independent test finds no matching opposite step, but its interval includes 0.
- **Tail calibration (injected rows; coverage at q01 / q05 / q95 / q99, nominal 0.01 / 0.05 / 0.95
  / 0.99).** No mask 0.032 / 0.066 / 0.875 / 0.952. Alarm time 0.024 / 0.059 / 0.888 / 0.962. Step
  stamp 0.029 / 0.080 / 0.962 / 0.990. Located onset 0.029 / 0.080 / 0.962 / 0.991 (true onset
  0.029 / 0.081 / 0.964 / 0.991). Every masked arm covers the upper tail better than no mask, and
  the step stamp, located, and true-onset arms over-cover the lower q05 more (0.080 against 0.066,
  nominal 0.05). The curtailment-cost metric was not scored, because the experiment scores single
  substations and the metric needs a node and its generators.

## Checks

- **Median minus mean forecast, per substation (located-onset arm, part 2 rows).** D10 -0.110, D11
  -0.055, D14 +0.022, D16 +0.035, D5 -0.367, D9 -0.074 points. The part 1 values are
  those of experiment 171.
- **Error of the mean forecast beside the quantile contrast.** Part 2, alarm time minus located
  onset: +0.556 (+0.253 to +0.873) in the mean absolute error, the same direction as the quantile
  score. Part 1 reuses experiment 171's score tables, and its mean-forecast check (+0.064) was not
  recomputed after the removal.
- **Audit triggers.** None fired on the injected rows. On the 15 real transfers the true-onset
  mask (7.695) scores worse than no mask (6.899) and worse than the alarm-time mask (7.065), so
  the true-onset mask is not a floor on real data; those rows decide nothing. No control failed, the
  result does not contradict an earlier finding or the prediction, the quantile and mean-forecast
  contrasts agree, leaving one substation out changes no verdict, and no nested arm beats the arm
  that contains it. The interval of part 1 (half-width 0.036) is 0.86 of experiment 171's 0.042. No
  audit was run.

## Caveats

- **The alarm-time mask is a comparison arm, not an earlier design.** Experiment 171 already masks
  from the detector's step stamp, so C2 mainly shows what masking back to the step is worth.
- **Part 1's removal lowers the contrast by construction.** Rows with a gated alarm active carry the
  whole difference, so removing confirmed events removes some of it. The informative results are
  the small confirmed share, the 5.6% placebo share, and the harm that stays on unconfirmed alarms.
- **The confirmation rule has no positive control.** Its recall was not measured on logged
  transfers or injected events, so a low confirmed share could also mean a low-power rule. The
  5.6% placebo share bounds only the chance rate.
- **The confirmation rule is one choice.** A looser step test, or one that used the metered
  neighbours' own power, could confirm more alarms.
- **Counts of transfers.** The 247 injected transfers are (world, substation, transfer, alarm)
  rows, 210 distinct (world, substation, transfer) combinations, because a transfer found by both
  alarms counts once for each; the pooled interval treats them as independent, and the
  whole-world resampling partly covers this. The location errors are over all 153 pair-alarm and
  430 RO-NC matches, of which 87 and 160 have scored forecast rows.
- **The step stamp's location error is truncated.** An alarm counts as finding a transfer only when
  its step stamp lies within 24 hours of the true onset, so the step stamp's error cannot exceed
  48 half-hours, while the located onset (searched 24 hours either side of the step stamp) can.
  That truncation favours the step stamp in the location comparison.
- **The alarm-time mask starts at the known stamp.** The plan said the first usable origin's
  newest stamp; the code masks from the known stamp, up to 11 half-hours earlier (mean 5.0 to 5.7).
  That makes the alarm-time arm slightly better than registered and biases C2 downward.
- **Overlapping alarms.** Every alarm at a substation is applied to every scored row there, so a
  row scored for one transfer can see another alarm's located onset, computed from readings up to
  11 half-hours after that row's cutoff. The share of rows affected was not counted.
- **The overall verdict is a judgement.** The plan registered a verdict for each part only.
- **The real logged transfers are few.** 15 transfers found by an alarm, and the logged onset is
  coarser than a half-hour.
- **Lab shortcuts.** The residual comes from the analysis proxy (a weather estimate for the recent
  past), the transfers are injected as constant-weight shares, and the end rule is that of
  experiments 147 and 148.
- **The forecasts are daily.** A 6-hourly cadence is simulated by taking, per found transfer, the
  one daily run whose newest stamp lies in the first 24 hours after the first origin that sees the
  alarm. The located onset is computed once, at that first origin.
- **The retraining schedule does not affect the contrasts.** The cached forecasts are the frozen
  stack of experiment 171, with no refitting, so the contrasts do not depend on a retraining
  schedule.

## What the next experiment should answer

The next question is why D10's gated alarms raise the error when most of them match no opposite
step: whether they are false alarms that a stricter gate would drop, which would show in a
gate tuned on the rows with an active alarm. Showing no harm needs about 54 months of forecasts, as
in experiment 171.
