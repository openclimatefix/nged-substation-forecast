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
  Over 27 months the gated mask scores 0.030 points of effective capacity worse than no mask on rows
  outside logged transfers (95% interval -0.011 to +0.074; unrounded 0.0297). Harm below 0.03 is
  therefore neither shown nor excluded. From November to March the difference is +0.036 (95%
  interval -0.060 to +0.135).
lessons: >-
  The first year's +0.009 is reproduced here (+0.011), but the earlier 15 months give +0.045, so
  near-zero harm does not generalise across periods. Leaving out D10 gives +0.002, and the rows
  with a gated alarm active carry the whole difference, as a mask changes nothing elsewhere. The
  interval is still too wide to show no harm. The cached ensemble forecasts cover only 27 months.
take_home_message: >-
  Ignoring recent power while a gated switching alarm is active scores 0.030 points of effective
  capacity worse over 27 months of rows outside logged transfers (95% interval -0.011 to +0.074),
  against +0.009 in experiment 166's 12 months. That interval is too wide to show the harm is
  below 0.03, and the difference comes mostly from one substation, D10. Showing no harm needs more
  months of forecasts, or an explanation of why D10's gated alarms raise the error.
advice:
  - >-
    Do not port the gated mask on the strength of experiment 166's +0.009: over 27 months the
    estimate is +0.030 (-0.011 to +0.074) (G1, figure 1).
  - >-
    Treat the gain from April 2024 to June 2025 alone, +0.045 (-0.000 to +0.102), as post hoc
    evidence that the harm varies between periods (figure 1).
  - >-
    Check D10's 15 gated alarms before any further no-harm test. Leaving D10 out moves the estimate
    from +0.030 to +0.002 (-0.036 to +0.035). D10 alone gives +0.170 (+0.011 to +0.383), post hoc
    (figure 2).
  - >-
    Expect to need about 54 months of forecasts for an interval inside +/-0.03, if the true harm is
    well below 0.03. The cached ensemble (ENS) forecasts cover 27 months (G1).
  - >-
    Report the share of rows with a gated alarm active beside the contrast: 7.8% of rows give
    +0.386 (-0.164 to +0.893) and the other 92.2% give 0.000 (rounded), post hoc (post hoc
    bullets).
production_status: not-for-production
adopted_in: none
---

**The gated mask scores 0.030 points of effective capacity worse than no mask on 27 months of rows
outside logged transfers (95% interval -0.011 to +0.074).** The interval includes zero and is wider
than the limit of +/-0.03 that we set in advance. The verdict is therefore `inconclusive`: harm
below 0.03 is neither shown nor excluded. Most of the difference comes from one substation, D10.

**Background: what the mask is.** Our forecast corrects itself using recent power. A "mask" sets
recent power aside while a switching alarm is active. A logged transfer is a movement of load
between two substations recorded in NGED's switching log. [Experiment
166](../00166-switching-gated-mask-and-soft-mask-while-alarmed/result.md) masked recent power only
while a "gated alarm" was active. A gated alarm is a pair alarm (a step at one substation matched by
an opposite step at a neighbouring metered substation, from [experiment
156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md)) with a balance score of
0.7 or more. The balance score measures how nearly the two steps cancel, and 1 means exactly.
Experiment 166 found a gain of 7.35 points in the 72 hours after a logged transfer ends. It could not show that the mask does no
harm on other rows, because 12 months of rows gave an interval of -0.068 to +0.081. A score in
points of effective capacity is the score in megawatts divided by the substation's effective
capacity (as in experiment 166), times 100. Primary substations are the substations that step power
down onto the local distribution network.

**The first year's near-zero harm (+0.009) is reproduced here (+0.011), but the earlier 15 months
give +0.045.** We scored every cached forecast run of the same six Primary substations, 27 months
instead of 12, on rows outside logged transfers. A row is one forecast for one half-hour from one
forecast origin. The forecast, called the stack here, is a frozen generalised additive model with a
Kalman filter ([Kalman, 1960](https://doi.org/10.1115/1.3662552)). The filter corrects each
forecast using its recent errors, as in [experiment
163](../00163-switching-real-event-comparison-gam-stack-against-bg2/result.md). Nothing is refitted;
only the rows change. Scores are twice the mean pinball loss over 11 quantiles, in points of
effective capacity; lower is better. The gain of 7.35 points is taken from experiment 166 (7 logged
transfers, 72 hours after each ends). It is not pooled with the rows outside logged transfers.

![Figure 1: The pooled interval is wider than the limit of 0.03](figure_1_headline.svg)

![Figure 2: The extra error comes mostly from D10](figure_2_by_substation.svg)

![Figure 3: The extra error comes from a few months](figure_3_by_month.svg)

## Design and results

- **Decision rule.** G1: the pooled contrast has a 95% interval inside -0.03 to +0.03. G2: the
  November to March contrast has an interval inside -0.05 to +0.05. The verdict is `supported` when
  both hold, `partly supported` when only G1 holds, `refuted` when the pooled point estimate is
  above +0.03, and `inconclusive` otherwise. The limit of 0.03 is there because the mask adds alarm
  tables to the serving path, which is worth doing only if ordinary rows do not get worse.
- **Arms.** K0 (no mask), MASK_GATED (readings set missing from the moment a gated alarm is known
  until the end rule of [experiment 148](../00148-switching-end-rule-lookahead-ro-nc/result.md)
  fires), and for reference MASK_NEW (every pair alarm) and MASK_UNION (pair alarm or refit-offset
  neighbour-count alarm, as in [experiment
  165](../00165-switching-drop-recent-power-while-alarmed-all-rows/result.md)). The stack is frozen.
  The K0 forecast spread of experiment 115 is shared, so arms differ only in where the forecast
  distribution sits.
- **Rows.** 1,579,711 rows with a reading, from 811 cached forecast runs (one per day) at D5, D9,
  D10, D11, D14, and D16 (anonymised substation labels), forecast origins 2024-04-10 to 2026-06-29.
  Each row's valid time lies outside every logged transfer from onset to 72 hours after its end.
  Each substation has more than 12 months of past forecast errors before its first run, because the
  power panel starts in 2022-01. The cached forecasts start in 2024-04, so none earlier were
  available to score.
- **Absolute scores, pooled.** K0 2.718, MASK_GATED 2.748, MASK_NEW 2.773, MASK_UNION 2.915.
- **G1 and G2.** Pooled (27 months): MASK_GATED minus K0 +0.030 (-0.011 to +0.074), half-width 0.042
  against the limit 0.03. November to March (10 months, 579,096 rows): +0.036 (-0.060 to +0.135),
  half-width 0.098 against the limit 0.05. April to October (17 months): +0.026 (+0.002 to +0.060).
- **Reference arms, pooled.** MASK_NEW minus K0 +0.055 (+0.009 to +0.105). MASK_UNION minus K0
  +0.197 (+0.078 to +0.342). Masking on more alarms, including the ones with no balanced opposite
  step, raises the error further.
- **Post hoc, labelled: the two periods.** Valid times from July 2025, the 12 months of experiment
  166's scored year (here on every row of those valid times, where experiment 166 used the
  forecasts shared with best guess 2): +0.011 (-0.066 to +0.085). Valid times before July 2025:
  +0.045 (-0.000 to +0.102). The two intervals overlap almost completely.
- **Post hoc, labelled: by substation.** Alone: D5 -0.027 (-0.099 to +0.009), D9 0.000 (no gated
  alarm from 2024-04-10 onwards), D10 +0.170 (+0.011 to +0.383), D11 +0.010 (+0.001 to +0.024), D14
  +0.001 (-0.004 to +0.007), D16 +0.023 (-0.105 to +0.167). Leaving out D10: +0.002 (-0.036 to
  +0.035). Leaving out any other substation leaves the estimate between +0.031 and +0.040.
- **Post hoc, labelled: rows with a gated alarm active at the forecast origin.** 122,598 rows
  (7.8%): +0.386 (-0.164 to +0.893). The other 1,457,113 rows: 0.000 (-0.001 to +0.001), rounded
  from a value below 0.0005 in size. A mask changes the forecast only while a gated alarm is active
  or its filter state is recovering, so the 7.8% carry the whole difference.
- **Gated alarms and masked half-hours.** The table counts gated alarms known from 2024-04-10 (2024
  is April to December and 2026 is January to June) and the half-hours between each alarm being
  known and its end being known.

| Substation | 2024 gated alarms / masked half-hours | 2025 gated alarms / masked half-hours | 2026 gated alarms / masked half-hours |
|---|---|---|---|
| D5 | 1 / 186 | 7 / 3,639 | 3 / 979 |
| D9 | 0 / 0 | 0 / 0 | 0 / 0 |
| D10 | 4 / 3,528 | 7 / 1,982 | 4 / 819 |
| D11 | 4 / 928 | 5 / 1,711 | 6 / 1,133 |
| D14 | 4 / 711 | 10 / 2,093 | 3 / 598 |
| D16 | 0 / 0 | 4 / 1,765 | 3 / 979 |

- **Share of masked half-hours outside every logged transfer.** Gated alarms: 65 alarms, 21,051
  masked half-hours, 17,028 outside (80.9%). Every pair alarm: 146 alarms, 39,789 half-hours, 86.2%
  outside. Either alarm: 178 alarms, 54,660 half-hours, 89.2% outside. Most masked half-hours
  therefore fall outside logged transfers, though the rows scored here are defined by valid time
  and the half-hours by the alarm's span, so the two sets overlap without matching.
- **Prediction against outcome.** The interval was expected to be about +/-0.05 wide, not inside
  +/-0.03, so the prediction that it would fall inside +/-0.03 failed on its own terms. The
  observed half-width is 0.042. The winter half-width is 0.098 against a limit of 0.05; the winter
  subset has 10 months against 27 pooled. The point estimate stayed below +0.03 (0.0297).

## Checks that did not change the verdict

- **Median minus mean forecast, per substation (identical for every arm).** D5 -0.146, D9 -0.019,
  D10 -0.058, D11 -0.030, D14 +0.037, D16 +0.017 points.
- **Error of the mean forecast beside the quantile contrast.** Mean absolute error of the mean
  forecast, MASK_GATED minus K0: +0.064 (-0.003 to +0.140), in the same direction as the quantile
  score.
- **Audit triggers (the seven conditions in the experiment procedure that call for an independent
  audit).** None fired. The wide interval was predicted. Leaving out D10 moves the estimate from
  +0.030 to +0.002, but the estimate stays positive and the verdict stays `inconclusive`. The
  half-width of 0.042 is 0.57 of experiment 166's 0.075, not under half. No audit was run.

## Caveats

- **The 27-month limit comes from the data, not the method.** A half-width of 0.03 would need about
  54 months at this variance, if the true harm is well below 0.03.
- **With 27 months, 10 of them in winter, the resampled interval is itself uncertain.**
- **Pair alarms appear at both ends of one transfer.** D5 and D16 share identical 2026 gated alarms,
  so the rows of neighbouring substations are not independent.
- **Three conditions favour the mask in this test.** The residual uses the analysis proxy (a
  weather estimate for the recent past), the alarm tables of experiment 156 were tuned on odd
  months of the same stream, and each alarm's end is known by the end rule of experiments 147 and
  148, which uses a 6-hour wait. The filter and spread settings were tuned on the odd months of
  2024-07 to 2025-06, which are inside the rows scored here.

## What the next experiment should answer

The next question is why D10's 15 gated alarms raise the error: whether they follow real switching
that no log records, or false alarms. Leaving D10 out gives +0.002. More than 7 logged transfers
are needed to measure the gain before the mask can be added to production.
