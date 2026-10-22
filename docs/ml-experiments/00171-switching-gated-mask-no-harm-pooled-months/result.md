---
id: "00171"
slug: switching-gated-mask-no-harm-pooled-months
title: >-
  Does the gated mask on recent power do no harm on ordinary days, once 27 months of rows are pooled?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  simple. Stored: private score tables under
  data/ml-experiments/00171-switching-gated-mask-no-harm-pooled-months/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible design, because the arms, the
  gate, the rows and the interval are those of experiment 166 with the scored year replaced by every
  cached run. Callers: this folder's scripts only; they are copies of experiment 166's modules and
  read experiment 166's stored balance scores and experiment 156's stored alarm tables.
hypothesis: >-
  Experiment 166 found that ignoring recent power (a "mask") only while a pair alarm with a balance
  score of at least 0.7 is active (the "gated mask") scores 0.009 points of effective capacity
  worse than no mask on the real rows outside logged transfers, with a 95% interval of -0.068 to
  +0.081 from one year of rows. That interval was too wide to show no harm. Pooling every cached
  forecast run, 27 months instead of 12, shows whether the gated mask does no harm on ordinary days.
prediction: >-
  Written before any result. On the pooled real rows outside logged transfers (six Primary
  substations, 2024-04 to 2026-06), the gated mask minus the unmasked stack has a 95% interval
  inside +/-0.03 points of effective capacity, and the same contrast on November to March rows
  has an interval inside +/-0.05. The prediction fails if the pooled point estimate is above
  +0.03, or if the pooled interval is still wider than +/-0.03; in that case no-harm cannot be
  shown from this data, and the page says so. Pooling 27 months instead of 12 shrinks a
  month-resampled interval by about 1.5 times (the square root of 27 over 12), so a half-width
  near 0.05 is the expected size and the interval inside +/-0.03 is not expected.
decision_rule: >-
  Pre-registered before any forecast score is read. Arms: K0 (the frozen generalised additive
  model with the Kalman filter and no mask), MASK_GATED (readings set missing from the moment a
  pair alarm with balance score 0.7 or more is known until the end rule of experiments 147 and 148
  fires), and, for reference only, MASK_NEW (every pair alarm) and MASK_UNION (pair or
  refit-offset neighbour-count alarm). Rows: every row of every cached run of the six Primary
  substations of experiment 166 (D5, D9, D10, D11, D14, D16; forecast origins 2024-04-10 to
  2026-06-29) with a reading, whose valid time lies outside every logged transfer from its onset
  to 72 hours after its end on that substation. Every series has more than 12 months of residual
  history before its first run, because the power panel starts in 2022-01. Scored with the same
  quantile score, the same frozen filter, spread and alarm tables as experiment 166. Interval:
  rows resampled by whole calendar month (year and month together), 20,000 draws, 95%. Support
  needs both G1 and G2. G1: pooled, MASK_GATED minus K0 has an interval inside -0.03 to +0.03. G2:
  the same on November to March rows, inside -0.05 to +0.05. The verdict is `supported` when both
  hold, `partly supported` when only G1 holds, `refuted` when the pooled point estimate is above
  +0.03, and `inconclusive` otherwise (an interval wider than the limit with the point estimate
  not above +0.03). The threshold is +/-0.03 because the mask adds alarm tables to the serving
  path, which is worth doing only if ordinary days do not get worse (experiment 166). The gain
  side stays as in experiment 166 (the 72 hours after a logged transfer ends, 7 transfers shared
  with best guess 2) and is not pooled with the no-harm rows. Reported beside the contrast: the
  number of gated alarms and masked half-hours (grid half-hours between an alarm's onset being
  known and its end being known) per substation and per calendar year, and the share of masked
  half-hours outside every logged transfer. Post hoc and labelled: the scored year alone and the
  earlier 15 months alone, to see whether the two agree; MASK_NEW and MASK_UNION minus K0;
  leaving one substation out; and the rows with a gated alarm active.
benchmark: >-
  The unmasked stack (K0) on the same rows; experiment 166's interval for the scored year alone
  (+0.009, -0.068 to +0.081).
row_set_id: >-
  every cached run (2024-04-10 to 2026-06-29) of six Primary substations, rows with a reading
  outside logged transfers; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity.
code: ml-experiments/00171-switching-gated-mask-no-harm-pooled-months/
verdict: inconclusive
best_arm: >-
  None. The gated mask scores 2.748 points of effective capacity on the pooled rows outside logged
  transfers, against 2.718 with no mask (difference +0.030, 95% interval -0.011 to +0.074).
headline: >-
  Pooling 27 months of real rows outside logged transfers gives gated mask minus no mask of +0.030
  points (95% -0.011 to +0.074). The point estimate is just inside the +0.03 limit and the interval
  is wider than +/-0.03. In November to March the difference is +0.036 (-0.060 to +0.135).
lessons: >-
  The first year's estimate of near-zero harm (+0.009) did not hold up. Over 27 months the gated mask
  adds about 0.03 points on ordinary days, almost all of it at D10 and on the 7.8% of rows where a
  gated alarm is active. The interval is still too wide to show no harm, and the cached forecasts
  cannot supply the roughly 54 months a half-width of 0.03 would need.
take_home_message: >-
  Ignoring recent power while a balanced pair alarm is active adds about 0.03 points of error on
  ordinary days over 27 months of real rows (interval -0.011 to +0.074), against 0.009 in the first
  year alone. The interval is still too wide to show that the harm is below 0.03, and the extra
  error comes mostly from one substation, D10. Showing no harm needs either many more months of
  forecasts or a look at why D10's gated alarms mask days that need no mask.
advice:
  - >-
    Do not port the gated mask on the strength of experiment 166's +0.009: over 27 months the
    estimate is +0.030 (-0.011 to +0.074), and 15 of those months alone give +0.045 (G1, figure 1).
  - >-
    Check D10's 15 gated alarms before any further no-harm test, because leaving D10 out moves the
    estimate from +0.030 to +0.002 (-0.036 to +0.035) and D10 alone shows +0.170 (figure 2).
  - >-
    Plan for about 54 months of forecasts, or a smaller per-row variance, before expecting a
    no-harm interval inside +/-0.03; the cached ENS forecasts cover 27 months only (G1).
  - >-
    Report the share of rows with a gated alarm active beside the contrast: 7.8% of rows, +0.386
    points there and -0.000 elsewhere, so the harm is confined to them (figure 2, Checks).
production_status: not-for-production
adopted_in: none
---

**Over 27 months of real rows, ignoring recent power while a balanced switching alarm is active adds
about 0.03 points of error on ordinary days, and the interval is still too wide to call that
harmless; most of the extra error comes from one substation, D10.** Our forecast corrects itself
using recent power. [Experiment
166](../00166-switching-gated-mask-and-soft-mask-while-alarmed/result.md) set recent power aside (a
"mask") only while a pair alarm whose opposite steps at a metered neighbour roughly cancel (a
balance score of 0.7 or more) was active, and found a large gain in the 72 hours after a logged
transfer of load between substations ends. It could not show that the mask does no harm on other
days, because one year of rows gave an interval of -0.068 to +0.081. Here we scored every cached
forecast run of the same six Primary substations, 27 months instead of 12, on rows outside logged
transfers. The gated mask scores 0.030 points of effective capacity worse than no mask (95% interval
from resampling calendar months, -0.011 to +0.074). The verdict is `inconclusive`: the point
estimate is just inside the +0.03 limit, and the interval is wider than +/-0.03, so no-harm cannot
be shown from this data.

**We asked whether the first year's near-zero harm was luck.** The forecast, called the stack here,
is a frozen generalised additive model with a Kalman filter ([Kalman,
1960](https://doi.org/10.1115/1.3662552)) that corrects each forecast using its recent errors, as
in [experiment 163](../00163-switching-real-event-comparison-gam-stack-against-bg2/result.md).
Nothing in the stack, the alarm tables of [experiment
156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md), or the gate is refitted;
only the rows change. Scores are twice the mean pinball loss over 11 quantiles, in points of
effective capacity; lower is better. The gain side stays as in experiment 166 (7 logged transfers,
72 hours after each ends, a gain of 7.35 points). It is not pooled with the no-harm rows here.

![Figure 1: Gated mask minus no mask on rows outside logged transfers](figure_1_headline.svg)

![Figure 2: The same contrast by substation](figure_2_by_substation.svg)

![Figure 3: The same contrast by calendar month](figure_3_by_month.svg)

- **Arms.** K0 (no mask), MASK_GATED (readings set missing from the moment a pair alarm with
  balance score 0.7 or more is known until the end rule of [experiment
  148](../00148-switching-end-rule-lookahead-ro-nc/result.md) fires), and for reference MASK_NEW
  (every pair alarm) and MASK_UNION (pair alarm or refit-offset neighbour-count alarm, as in
  [experiment 165](../00165-switching-drop-recent-power-while-alarmed-all-rows/result.md)). No refit
  schedule applies; the stack is frozen. The K0 spread of experiment 115 is shared, so arms differ
  in location only.
- **Rows.** 1,579,711 rows with a reading from 811 cached runs at D5, D9, D10, D11, D14, and D16
  (anonymised substation labels), forecast origins 2024-04-10 to 2026-06-29, outside every logged
  transfer from onset to 72 hours after its end. Each series has more than 12 months of residual
  history before its first run. Earlier years are not available: the cached member forecasts, from
  chained fits, start in 2024-04. Some filter and spread settings were tuned on the odd months of
  2024-07 to 2025-06, which are inside these rows.
- **Absolute scores, pooled.** K0 2.718, MASK_GATED 2.748, MASK_NEW 2.773, MASK_UNION 2.915.
- **G1, pooled (27 months).** MASK_GATED minus K0 +0.030 (-0.011 to +0.074), half-width 0.042
  against the limit 0.03. **G2, November to March (10 months, 579,096 rows)** +0.036 (-0.060 to
  +0.135), half-width 0.098 against the limit 0.05. April to October (17 months) +0.026 (+0.002 to
  +0.060).
- **Reference arms, pooled.** MASK_NEW minus K0 +0.055 (+0.009 to +0.105); MASK_UNION minus K0
  +0.197 (+0.078 to +0.342).
- **Post hoc, labelled: the two periods.** Valid time from 2025-07
  (experiment 166's year, here on all its rows) +0.011 (-0.066 to +0.085); April 2024 to June 2025
  +0.045 (-0.000 to +0.102).
- **Post hoc, labelled: by substation.** Alone: D5 -0.027 (-0.099 to +0.009), D9 +0.000 (no gated
  alarm after 2024-04), D10 +0.170 (+0.011 to +0.383), D11 +0.010 (+0.001 to +0.024), D14 +0.001
  (-0.004 to +0.007), D16 +0.023 (-0.105 to +0.167). Leaving out D10: +0.002 (-0.036 to +0.035).
  Leaving out any other substation leaves the estimate between +0.031 and +0.040.
- **Post hoc, labelled: rows with a gated alarm active at the forecast origin.** 122,598 rows
  (7.8%): +0.386 (-0.164 to +0.893). The other 1,457,113 rows: -0.000 (-0.001 to +0.001). The extra
  error is confined to the 7.8%.
- **Gated alarms and masked half-hours** (alarms starting from 2024-04-10; 2024 is April to
  December and 2026 is January to June; a masked half-hour lies between the alarm being known and
  its end being known):

| Substation | 2024 alarms / half-hours | 2025 alarms / half-hours | 2026 alarms / half-hours |
|---|---|---|---|
| D5 | 1 / 186 | 7 / 3,639 | 3 / 979 |
| D9 | 0 / 0 | 0 / 0 | 0 / 0 |
| D10 | 4 / 3,528 | 7 / 1,982 | 4 / 819 |
| D11 | 4 / 928 | 5 / 1,711 | 6 / 1,133 |
| D14 | 4 / 711 | 10 / 2,093 | 3 / 598 |
| D16 | 0 / 0 | 4 / 1,765 | 3 / 979 |

- **Share of masked half-hours outside every logged transfer.** Gated alarms: 65 alarms, 21,051
  masked half-hours, 17,028 outside (80.9%). Every pair alarm: 146 alarms, 39,789 half-hours, 86.2%
  outside. Either alarm: 178 alarms, 54,660 half-hours, 89.2% outside. Most masking happens on days
  with no logged transfer, so the no-harm contrast is where the mask spends its time.
- **Predictions against outcome.** Predicted: interval inside +/-0.03 (not expected, a half-width
  near 0.05); observed half-width 0.042. Predicted: winter interval inside +/-0.05; observed
  half-width 0.098, because 10 winter months are fewer than the pooled 27. The point estimate
  stayed below +0.03 (0.0297).

## Checks

- **Median minus mean forecast, per site (identical for every arm).** D5 -0.146, D9 -0.019, D10
  -0.058, D11 -0.030, D14 +0.037, D16 +0.017 points.
- **Error of the mean forecast beside the quantile contrast.** Mean absolute error of the mean
  forecast, MASK_GATED minus K0: +0.064 (-0.003 to +0.140). Same direction as the quantile score.
- **Triggers.** None fired. The wide interval was predicted. Leaving out D10 moves the estimate from
  +0.030 to +0.002 but its sign and the `inconclusive` verdict hold. The interval is 0.56 times as
  wide as experiment 166's, not under half. No audit was run.

## Caveats

- **Twenty-seven months is the limit of the data, not of the method.** No earlier forecasts exist
  to score, so a half-width of 0.03 would need about 54 months at this variance.
- **Months are the resampling unit.** With 27 months, and 10 in winter, the interval is itself
  uncertain.
- **Lab conditions.** The residual uses the analysis proxy, the alarm tables were tuned on odd
  months of the same stream, and the end of each alarm is known by the 6-hour rule.

## What the next experiment should answer

The next question is why D10's 15 gated alarms mask days that need no mask: whether they follow
real switching that no log records, or false alarms. A gate that removes them may bring the
estimate to zero. The gain side needs more than 7 logged transfers before a port can be weighed.
