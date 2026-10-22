---
id: '00225'
slug: dp-network-heads-on-unlabelled-worst-days
title: >-
  Differentiable physics: do the neural network's corrections help the worst days that no label
  explains?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: done
size: >-
  simple. Stored: private tables under
  data/ml-experiments/00225-dp-network-heads-on-unlabelled-worst-days/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one is pre-registered (the brief's), and the
  choice of the headline network among the stored ones is fixed below. Callers: this folder's
  scripts, which read the stored tables of experiments 172 and 201 and the stored forecasts of
  experiment 155. No fit, no new method.
hypothesis: >-
  Experiments 172, 174, and 175 found that the worst 5% of substation-days carry 41.7% of the excess
  loss of arm D (the best-guess convex forecast of experiment 155), and that 37 of the 72 worst days
  at the five decision substations carry none of the labels (switching event, holiday, cold snap,
  flagged reading, Economy 7 step). Experiments 200 and 201 found that a neural network corrects
  arm D mainly on very hot and very cold days. The unlabelled worst days therefore sit at
  temperature extremes, and the neural network cuts their loss.
prediction: >-
  Written before any score is read. (1) The neural network of experiment 201 (arm 2) cuts the
  unlabelled
  worst days' excess loss by at least 20%. (2) At least half of the 37 unlabelled days have a
  day-mean temperature in the outer 10% of the site's range (below the 10th or above the 90th
  percentile of the site's own training days), against about 20% for random days. If both fail, the
  unlabelled days are single-substation events that a pooled neural network cannot see. If only the
  second passes, the neural network under-corrects the extremes, and a tail-focused network is the
  next step.
decision_rule: >-
  Prediction 1 holds when the share of the 37 days' excess loss that the network removes (the sum
  over the days of arm D's day score minus the network's day score, divided by the sum of arm D's
  excess) is at least 20%. Prediction 2 holds when at least 19 of the 37 days lie in the outer 10%
  of the site's temperature range. Both are compared with 1,000 draws of 37 ordinary days (the
  same substations and calendar months, not among the worst 72), reported as the percentile of the
  real value among the draws. The shares carry no formal interval beyond the draws, because the
  unit is one day and the sample is 37 days; the verdict is descriptive and the write-up says so.
  The
  control of experiment 172 recovered 89.2% of the days it injected, so each share is read as
  approximate.
benchmark: >-
  Arm D's day scores (experiment 155 as stored, D+1 band), reproduced against experiment 172's
  day table within 0.001 points, and the 1,000-draw random control
row_set_id: fold_a_daily_gap_v1
metric: >-
  Day score: the mean quantile score (twice the mean pinball loss over 11 quantiles, points of
  effective capacity) of a local day's D+1 rows (lead 15 to 39 hours) outside switching masks, over
  the scored year 2025-07-03 to 2026-06-30. Excess: the day score minus the site's median day score
  of arm
  D.
code: ml-experiments/00225-dp-network-heads-on-unlabelled-worst-days/
date: 2026-10-22
verdict: partly supported
best_arm: >-
  Arm 2 of experiment 201 (arm D plus calendar and holiday flags): removes 22.6% of the 37
  unlabelled worst days' excess loss (seeds 19.8%, 25.9%, 22.3%), against 11.8% on the 35 labelled
  worst days.
production_status: not-for-production
adopted_in: none
headline: >-
  The neural network removes 22.6% of the unlabelled worst days' excess loss, but those days are
  not hot or cold extremes: 4 of 37 lie in the outer 10% of temperature, against about 7 or 8
  expected by chance.
take_home_message: >-
  The neural network does cut the loss on the 37 worst days that no label explains, by about a
  fifth, and cuts it by more than on the 35 days that do carry a label. The unlabelled worst days
  are not at temperature extremes, so a neural network aimed at the hottest and coldest days would
  not target them.
  What helps on them is the forecast input: a neural network without one removes 4%.
advice:
  - >-
    Do not build a tail-focused neural network for the unlabelled worst days: they are not at
    temperature
    extremes (4 of 37 in the outer 10%, figure 1).
  - >-
    Keep the forecast inputs in any network aimed at the worst days: a neural network with no
    forecast input removes 3.9% of their excess, against 22.6% with it (figure 2).
  - >-
    Read the share of excess removed on worst days beside a control, because days chosen for a high
    arm D error improve a little under any change (the blind network, figure 2).
---

**The neural network removes 22.6% of the excess loss on the 37 worst days that no label explains,
but those days are not hot or cold extremes.** Only 4 of the 37 days lie in the outer 10% of their
substation's temperature range, fewer than the 7 or 8 expected for random days (the 8th percentile
of 1,000 random draws). Prediction 1 passes
and prediction 2 fails, so the result is partly supported. The neural network with forecast inputs
removes 22.6% of the excess and the one without removes 3.9%, so the gain depends on the forecast
inputs.

**This experiment re-scores stored forecasts and fits nothing.** It reads the stored tables of
[experiment 172](../00172-dp-error-budget-worst-days/result.md) (the worst days) and the stored
predictions of [experiment 201](../00201-dp-gam-residual-mlp/result.md) (the neural network), The
worst days are the 72 days with the highest excess at the five decision substations (D5, D10, D11,
D14, and D16). A day is labelled if it carries a switching event, a holiday, a cold snap, a flagged
reading, or an Economy 7 step. A day's score is the mean quantile score of its day-ahead rows (leads
15 to 39
hours), so "points" below are points of the quantile score, in percentage points of effective
capacity. A day's excess is its score minus its substation's median day score. Arm D is the
best-guess convex demand forecast of
[experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md).
Rebuilding experiment 172's day table from the stored rows gives all 1,392 days with a maximum score
difference of 0.0.

![Temperature percentile of the 37 unlabelled worst days, and the share in the outer 10% of
temperature in 1,000 draws of 37 ordinary days.](figure_1_temperature_of_the_unlabelled_days.svg)

![Share of arm D's excess loss removed on the unlabelled and labelled worst days by four stored
neural networks, with the range over three seeds.](figure_2_excess_removed_by_class.svg)

![Five unlabelled worst days: observed power, arm D, and arm D plus the neural
network.](figure_3_five_unlabelled_days.svg)

## Results

- **Prediction 1 (the network cuts the unlabelled days' excess loss by at least 20%): passed.**
  Arm 2 of experiment 201 removes 22.6% (seeds 19.8%, 25.9%, 22.3%). The day score falls from 4.93
  to 4.29 points. On the 35 labelled worst days it removes 11.8% (7.03 to 6.46 points), and on all
  1,392 days 33.4% of excess.
- **The gain is larger than on random days: the real gain lies above all 1,000 draws.** The mean
gain per day on the unlabelled worst days is
  0.64 points, against 0.21 points (5th to 95th percentile 0.03 to 0.42) over 1,000 draws of 37
  ordinary days matched on substation and calendar month.
- **Prediction 2 (at least half the days at temperature extremes): failed.** Four of the 37 days (1
below the 10th percentile and 3 above the 90th) lie in the outer 10% of the substation's days before
2025-07-01, against a
  mean of 20.1% (5th to 95th percentile 10.8% to 29.7%) over the random draws; the real share sits
  at the 8th percentile of the draws. The 35 labelled days are more extreme (8 of 35), and the
  unlabelled days are milder than average (mean 11.3 degrees Celsius against 11.5 for all days).
- **Only the neural networks with forecast inputs remove more than 4% of the excess.** Arm 0 (no
forecast input) removes 3.9% of the unlabelled days' excess, so choosing days for a high arm D error
accounts for about 4%. Arm 4 (arm 2 plus recent power) removes 26.6%, and arm 6 (arm D's outputs
only) removes 23.2%. Across all 1,392 days arm 2 removes 33.4% of the excess, a larger share than on
the unlabelled worst days.
- **The three seeds straddle the 20% threshold.** Three training runs of arm 2 that differ only in
random seed span 19.8% to 25.9% on the unlabelled days, so the 20% threshold
  is within the seed spread.

## Checks

- **The earlier control found 89.2% of its injected days, so shares are approximate.** Experiment
172's control inserted known unusual days into the data and recovered 89.2% of them.
- **The temperature definition does not change the finding.** The headline uses the ensemble-mean
temperature that arm D used. With
  ERA5 day temperature, as in experiment 172, the unlabelled days average 11.5 degrees Celsius.
- **The verdict is descriptive, because the sample is 37 days.** The unit is one day and the sample
is 37 days, so the shares carry no
  interval beyond the random draws.

## Caveats

- **Days chosen for a high arm D error tend to score better under a different forecast.** Arm 0
removes 3.9%, which indicates the size of that effect, and the 22.6% is read against it.
- **The result covers the five decision substations together, not each one.** It was not split by
substation.

## Experiments this builds on

[Experiments 172](../00172-dp-error-budget-worst-days/result.md),
[174](../00174-dp-unlabelled-worst-days-weather-shared-or-single-site/result.md),
[175](../00175-dp-unlabelled-day-shapes/result.md),
[200](../00200-dp-pooled-correction-head-on-kalman-state/result.md),
[201](../00201-dp-gam-residual-mlp/result.md), and
[209](../00209-dp-pinball-table-vs-network-head-on-arm-d/result.md).
