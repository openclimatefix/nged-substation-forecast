---
id: '00225'
slug: dp-network-heads-on-unlabelled-worst-days
title: >-
  Differentiable physics: do the neural network's corrections help the worst days that no label
  explains?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: planned
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
  temperature extremes, and the network cuts their loss.
prediction: >-
  Written before any score is read. (1) The network of experiment 201 (arm 2) cuts the unlabelled
  worst days' excess loss by at least 20%. (2) At least half of the 37 unlabelled days have a
  day-mean temperature in the outer 10% of the site's range (below the 10th or above the 90th
  percentile of the site's own training days), against about 20% for random days. If both fail, the
  unlabelled days are single-substation events that a pooled network cannot see. If only the second
  passes, the network under-corrects the extremes, and a tail-focused network is the next step.
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
---

**Status: planned. No score has been read.**

The question is whether the neural network's corrections of
[experiment 201](../00201-dp-gam-residual-mlp/result.md) reach the 37 worst days of
[experiment 172](../00172-dp-error-budget-worst-days/result.md) that carry no label, and whether
those days are hot or cold extremes. The plan reads stored tables only and fits nothing.

## Design

- **Sites and days.** The five decision substations (D5, D10, D11, D14, D16). The 72 worst days
  (worst 5% of each substation's days by excess loss) split into 37 with no label and 35 with at
  least one label, from experiment 172's `worst_days_table.parquet`. The sixth substation (D9) has
  no contrast.
- **Networks.** The headline network is arm 2 of experiment 201 (arm D's inputs plus calendar and
  holiday flags), the stored network with the largest gain. Experiment 200's heads store losses by
  month only, so they cannot be scored by day; their change in loss by month is a sign cross-check.
  Arm 0 of experiment 201 (lead, calendar, and substation only, no forecast) is the blind network,
  and arm 4 (adds lagged power) is the network with recent power. Each arm has three seeds, scored
  separately; the day score of an arm is the mean over the seeds, and the seed spread is reported.
- **Rows.** A day's rows are arm D's D+1 rows (lead 15 to 39 hours) outside the hand-filtered
  switching masks, on the keys (substation, power forecast initialisation time, valid time). The
  scored rows of experiment 201 (2025-07-02 to 2026-06-29) are checked against experiment 172's
  scored year, and the overlap is used if they differ.
- **Temperature.** Each day's temperature is the mean of the ensemble-mean 2-metre temperature that
  arm D used (the `drv_temp_c` column of experiment 201's feature files) over the day's half-hours,
  as a percentile among the same substation's days before 2025-07-01.
- **Controls.** 1,000 draws of 37 days that are not among the worst 72, matching each unlabelled
  day's substation and calendar month. Each draw gives the share in the outer 10% of temperature and
  the network's share of excess removed.
- **Reported.** For the 37 unlabelled days, the 35 labelled days, and the 72 together: arm D's and
  the networks' day scores, the share of excess removed, and the same by temperature decile.

## Figures

1. The temperature percentile of the 37 unlabelled days against the random draws.
2. The network's change in excess loss for the unlabelled, labelled, and control days.
3. Five unlabelled days: arm D, arm D plus the network, and the observed power.
