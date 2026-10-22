---
id: "00226"
slug: lsc-why-learned-score-mask-lost-part-1
title: >-
  Why did the learned LSC score find more load shifts but give a worse masked forecast, and does a
  combined mask fix it?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00226-lsc-why-learned-score-mask-lost-part-1/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible (how the pair alarm and
  the learned score combine), so three combinations U1 to U3 are fixed below before any score is
  read. Callers: this folder's scripts only; they are copies of experiment 203's modules and read
  its stored tables and experiment 195's alarm tables.
hypothesis: >-
  Experiment 203's learned-score mask (7.31 points of effective capacity on the 72 hours after the
  7 logged load shifts, against 6.32 for the pair-alarm mask) loses because the learned alarm is
  known 2 to 4 days after the step, so its mask starts late on long load shifts. If that is the
  cause, a mask that takes the pair alarm's onset where the pair alarm fires, and the learned
  score's coverage elsewhere, keeps the learned score's extra matches without the loss.
prediction: >-
  Written before any forecast score is read. The best of U1 to U3 scores at most 6.32 on the 7
  transfers (matching the pair-alarm mask), has a pooled ordinary-day point estimate (mask minus no
  mask) at or below +0.02 on experiment 171's rows, and matches at least 20 of the 35 logged load
  shifts lasting over 1 day. The prediction fails if every combination scores above 6.32 on the 7.
  The page then says that the extra load shifts found are the wrong ones for the forecast (small,
  late, or low-value), and gives their size and timing.
decision_rule: >-
  Pre-registered. Same harness, rows, controls, and positive controls as experiment 203 (daily 09:00
  UTC cached runs, not 6-hourly, because the cached ensemble forecasts exist only for those runs;
  causal only). Combined masks, all built from experiment 195's R re-ranker alarms (known 192
  half-hours after the step, as in 203) and the pair alarm (every pair alarm, experiment 163's
  PAIR_NEW): U1, pair-alarm intervals plus learned-score intervals, merging intervals on one series
  that start within 24 hours, each interval keeping the earlier known time and the later end; U2,
  the learned-score intervals kept only where the candidate's opposite-step ratio folded to at most
  1 (min of ratio and 1/ratio, 0 where no opposite step) is at least 0.7, experiment 166's gate;
  U3, the learned-score intervals with the end known only by experiment 148's rule (the estimated
  end plus 60 half-hours, no experiment 147 stamp and no 96 half-hour shortcut) and the step stamp
  as the onset. Verdict by the brief's prediction on the best of U1 to U3 (chosen on the 7
  transfers, so the selection is reported as such): `supported` when the best combination scores at
  most 6.32 on the 7, its ordinary-day point estimate is at most +0.02, and it matches at least 20
  of 35; `partly supported` when two of the three hold; `not supported` when at most one holds.
  Intervals: transfers resampled whole for the gain (20,000 draws), rows resampled by whole calendar
  month for the harm. Descriptive, not bars: the per-transfer table, the gap decomposition, the harm
  split by alarm source (learned alarms that a pair alarm also raises within 48 half-hours, and
  learned-only alarms), by Primary substation, and by season; the same ordinary-day contrast with
  learned-only alarms removed that are balanced-opposite candidates (an opposite step of 0.4 to 2.5
  times the own step, experiment 195's definition, probable unlogged load shifts). The smallest gain
  worth porting is zero for a mask change that adds no pipeline step beyond the learned score
  already assumed; the harm limit is +/-0.05 points, from the brief.
benchmark: >-
  The pair-alarm mask (6.32 points on the 72 hours after the 7 transfers) and the unmasked forecast
  (13.69), experiment 163; the pair-alarm gated mask minus no mask on ordinary days (+0.030),
  experiment 171; the re-ranker mask (7.31, +0.031), experiment 203.
row_set_id: >-
  experiment 163's AFTER rows (7 and 11 transfers) and every cached run (2024-04-10 to 2026-06-29)
  of six Primary substations; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity.
code: ml-experiments/00226-lsc-why-learned-score-mask-lost-part-1/
verdict: not supported
best_arm: >-
  None. Of the three pre-registered combinations, U1 (pair alarm and learned score merged) scores
  6.350 points of effective capacity on the 72 hours after the 7 logged load shifts against 6.321
  for the pair-alarm mask (+0.029, 95% interval -0.103 to +0.236), finds 21 of 35, and does harm
  on ordinary days (+0.057, +0.015 to +0.100). U2 (learned score gated on balance) does no harm
  (-0.007, -0.041 to +0.013) but scores 7.336 and finds 6 of 35.
headline: >-
  No combination of the pair alarm and the learned score keeps the learned score's extra matches
  and avoids the harm. The 0.99 point gap on the 7 load shifts is two load shifts: the learned score
  has no alarm at the start of one 580-hour load shift (+2.23 points) and masks another 372-hour
  load shift more tightly than the pair alarm (-1.25). Learned alarms that a pair alarm also raises
  do no harm on ordinary days (-0.008); learned-only alarms do (+0.038), and removing the
  balanced-opposite ones leaves +0.036.
lessons: >-
  The learned score's extra matches come from alarms that a pair alarm does not raise. Those alarms
  give no gain in the 72 hours after a load shift (13.65 against 13.69 for no mask on the 7 load
  shifts) and carry the harm. The alarms both sources raise are harmless but find only 5 to 6 load
  shifts. A mask needs an early alarm on long load shifts, and early alarms on ordinary days are
  what the pair alarm gets wrong.
take_home_message: >-
  The learned load-shift score finds more real load shifts than the pair alarm, but the extra ones
  it finds do not improve the forecast and do the harm, so merging the two alarms does not give a
  mask that is both as good after load shifts and harmless on ordinary days. The alarms that both
  sources raise are harmless but too few to help. The next step is fewer false alarms per matched
  load shift, not more matches.
advice:
  - >-
    Judge an alarm source on the forecast after the load shifts it matches, not on how many it
    matches: the learned score's 24 matches against 5 for the pair alarm did not lower the
    forecast error (Results, found of 35 and the 7-load-shift scores).
  - >-
    Report the harm of a learned alarm source split into alarms a pair alarm also raises and
    learned-only alarms: the first group does no harm and the second does (figure 2).
  - >-
    Do not merge the pair alarm's mask with the learned score's by taking the later end: the
    merged mask inherits the pair alarm's ordinary-day harm (U1, U4 in figure 2).
  - >-
    Drop the 7-load-shift contrast as a decision rule: two load shifts make up the whole gap, so
    its sign says nothing about the other five (figure 3).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**The learned load-shift score (experiment 195's small neural network re-ranker) finds more real
load shifts than the pair alarm, but combining the two alarms does not give a mask that is both
as good as the pair alarm after load shifts and harmless on ordinary days.** A load shift moves
demand between Primary substations for hours or days, and the forecast's recent-power correction
misreads the moved demand as a change in the substation's own demand. A "mask" tells the
correction to ignore the readings while an alarm is open. Experiment 203 found that the learned
score's mask scored 7.31 points of effective capacity (percent of a substation's effective
capacity; lower is better) on the 72 hours after 7 logged load shifts, against 6.32 for the pair
alarm's mask, although the learned score matches 24 of 35 real load shifts against 5 for the pair
alarm. This experiment split that gap load shift by load shift, split the harm on ordinary days by
alarm source, and tested three combined masks that were fixed before any score was read. No
combination met the pre-registered prediction.

The alarms are those of [experiment 195](../00195-switching-neural-plus-detector-features/result.md),
the pair alarm of [experiment 156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md)
with the balance gate of [experiment
166](../00166-switching-gated-mask-and-soft-mask-while-alarmed/result.md), and the forecast
harness of [experiment 203](../00203-switching-replay-learned-score-in-masked-forecast/result.md),
which this experiment reuses unchanged: the same cached daily forecasts, the same rows, and the same
controls, which reproduce here (no mask 13.686, pair-alarm mask 6.321, best guess 2 10.091).

![Mask minus pair-alarm mask after real load shifts, by combination](figure_1_gain_after_load_shifts.svg)

![Mask minus no mask on ordinary days, by combination and alarm source](figure_2_harm_ordinary_days.svg)

![Contribution of each load shift to the 0.99 point gap](figure_3_gap_decomposition.svg)

![Masks of three alarm sources against the logged interval](figure_4_masks_against_logged_interval.svg)

## Results

- **Verdict: `not supported`.** The best combination on the 7 load shifts is U1, and of the three
  conditions it meets one (finds 21 of 35, at least 20). It misses 6.32 by 0.03 points (6.350) and
  its ordinary-day harm is +0.057, above the +0.02 limit. U2 meets the harm condition and misses the
  other two. The prediction fails because every combination loses to or only ties the pair-alarm mask
  on the 7.
- **Combinations (all use the re-ranker, known 192 half-hours after the step).** U1: pair alarm
  and learned score merged, one interval per 24 hours, earlier known time and later end. U2: learned
  alarms whose folded opposite-step ratio is at least 0.7 (experiment 166's gate). U3: learned alarms
  with the end known only by experiment 148's rule. U4 (post hoc, not pre-registered): pair-alarm
  onset and learned-alarm end.
- **Mean score of the 72 hours after a load shift (points).** The 7 load shifts that best guess 2
  also scores: no mask 13.686, pair 6.321, learned score 7.310, U1 6.350, U2 7.336, U3 7.310, U4
  6.419, best guess 2 10.091. The 11 load shifts of experiment 163's scored and replicate months: no
  mask 10.129, pair 6.083, learned score 5.623, U1 5.154, U2 6.186, U3 5.639, U4 5.196.
- **Gain against the pair-alarm mask (95% interval, load shifts resampled whole).** On the 7: U1
  +0.029 (-0.103 to +0.236), U2 +1.016 (-2.214 to +6.986), U3 +0.989 (-2.229 to +6.963), U4 +0.098
  (+0.018 to +0.325). On the 11: U1 -0.929 (-3.245 to +0.074), U2 +0.104 (-2.080 to +3.979), U3
  -0.444 (-3.504 to +3.734), U4 -0.887 (-3.194 to +0.125). Leaving D5 out of the 7 moves the
  learned-score gap from +0.99 to +0.09.
- **The 0.99 point gap is two load shifts.** One 580-hour load shift at D5 (card ce4b) adds +2.226
  points: the pair alarm is known 7 hours before the logged onset (its step is 13 hours before),
  and the learned score has no alarm at the step (its only alarm there is the next load shift's onset, known 95.5 hours after this load
  shift ends). One 372-hour load shift at D5 (card b2f7) adds -1.248, because the learned mask
  covers 580 hours with 218 outside the logged interval, and the pair mask covers 1,164 hours with
  794 outside. The other seven scoring windows add +0.011 in total.
- **Late onsets are the cause where the learned score loses.** The learned score's alarm for ce4b
  is known after the load shift ends, and 2.2 of the 2.3 points where the learned score loses come
  from that one load shift. Over-long masks are the pair alarm's cost where the learned score wins (b2f7).
  Which cause applies to a given load shift is in `printed_decompose.txt` (onset error, hours known
  after the logged onset, hours masked inside and outside the logged interval, and logged hours
  never masked).
- **Matches of the 35 logged load shifts lasting over 1 day (experiment 195's rule).** Pair alarm
  5, gated pair alarm 3, RO-NC 19, learned logistic regression 20, learned re-ranker 24, U1 21, U2 6,
  U3 24, U4 21.
- **Harm on ordinary days (arm minus no mask, 1,579,711 rows, no mask 2.718; 95% interval from
  whole months).** Pair alarm +0.055 (+0.009 to +0.105), learned re-ranker +0.031 (-0.020 to
  +0.086), U1 +0.057 (+0.015 to +0.100), U2 -0.007 (-0.041 to +0.013), U3 +0.025 (-0.026 to
  +0.081), U4 +0.062 (+0.021 to +0.106).
- **Split of the re-ranker's alarms by whether a pair alarm also raises them (within 24 hours).**
  Alarms both raise (23 to 29 per fold): -0.008 (-0.040 to +0.010). Learned-only alarms (81 to 94
  per fold): +0.038 (+0.002 to +0.089). Learned-only alarms without the balanced-opposite candidates
  (an opposite step of 0.4 to 2.5 times the own step, probable unlogged load shifts; 64 to 76 per
  fold): +0.036 (-0.000 to +0.086). Removing the probable unlogged load shifts therefore does not
  remove the harm.
- **Learned-only alarms give no gain after a load shift.** On the 7 load shifts a mask built from
  them scores 13.651 against 13.686 with no mask (-0.035, -0.152 to +0.000), and on the 11 load
  shifts 9.506 against 10.129 (-0.624, -2.048 to -0.009). The extra load shifts the learned score
  finds are small or late for the forecast.
- **Harm by Primary substation (arm minus no mask).** The pair alarm's harm sits at D10 (+0.221;
  +0.022 pooled without D10). The learned score's harm sits at D10 (+0.124) and D9 (+0.088, where
  the pair alarm raises none). U2 is at or below +0.014 at every substation.
- **Harm by season.** The learned score's harm is in autumn (September to November, +0.148, +0.003
  to +0.327), against -0.044 in winter (-0.189 to +0.049). U2 is at most +0.016 in every season.
- **Row sets and schedule.** Experiment 163's AFTER rows and experiment 171's ordinary rows, as in
  experiment 203. The cached forecasts are daily at 09:00 UTC, not 6-hourly (a 6-hourly replay
  needs new ensemble forecasts), and the end of an alarm that experiment 147 did not raise is known
  60 half-hours after its estimated end, a lab shortcut. Nothing is refitted.
- **Checks: median and mean forecast.** The median forecast minus the mean forecast per site is the
  same for no mask and the learned-score mask to three decimals (D5 -0.146, D10 -0.058, D11 -0.030,
  D14 +0.037, D16 +0.017, D9 -0.019), because a mask changes the correction only through the mean.
- **Checks: mean-forecast error beside each contrast.** The mean forecast's absolute error on
  ordinary days, arm minus no mask, points the same way as each quantile-score contrast: U1 +0.104
  (+0.035 to +0.176), U2 -0.007 (-0.055 to +0.026), U3 +0.033 (-0.041 to +0.114), U4 +0.103 (+0.038
  to +0.174). On the 7 load shifts the learned score's mean-forecast error minus the pair alarm's is
  +0.525 (-3.238 to +7.311), the same sign as +0.989.
- **Checks: audit triggers.** Trigger 2 fired in form (the prediction failed) and is answered by the
  load-shift decomposition above. Trigger 4 (leaving one site out) changes the 7-load-shift gain,
  which is why the page does not rest on that contrast. The controls reproduce and the quantile-score
  and mean-forecast contrasts agree. Trigger 7 applies in part, because experiment 227 takes the
  best combination from this page as one of its alarm sources, but experiment 227 sweeps its own
  thresholds and does not depend on the headline sign. No audit was run.
- **Caveats.** The learned scores are partly in-sample on ordinary days (experiment 195 trained them
  with even-month candidates as negatives). Seven and 11 load shifts are few, so every gain interval
  is wide. U1 to U3 were chosen on the 7 load shifts, and U4 is post hoc. The pair-alarm intervals
  that are merged into U1 are the same 242 alarms of experiment 163.
- **What this narrows.** The learned score's extra matches are not where the forecast gain is, and
  they are where the ordinary-day harm is. A mask that stays at or below the harm limit needs fewer
  false alarms per matched load shift, and an alarm that arrives early on long load shifts. The
  next experiment, 227, tests alarm sources and thresholds by alarms per matched load shift, and
  uses of an alarm that keep recent power.
