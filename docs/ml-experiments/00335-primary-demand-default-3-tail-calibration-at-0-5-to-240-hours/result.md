---
id: '00335'
slug: primary-demand-default-3-tail-calibration-at-0-5-to-240-hours
title: >-
  Can a per-lead-band calibration of the 0.01, 0.05, 0.95, and 0.99 quantiles bring the tails of
  Primary demand research default 3 to nominal at every lead from 0.5 to 240 hours?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
  - curtailment-cost
status: done
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00335-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the three arms are fixed
  by the question. Callers: this folder's scripts, which import the folders of experiments 317, 319,
  320, and 322 and call reviewed `studies.*` modules without changing them. No change under
  `packages/`.
hypothesis: >-
  The 0.01, 0.05, 0.95, and 0.99 quantiles of Primary demand research default 3 miss their nominal
  rates past 96 hours (experiments 317 and 320 found 2.1% to 2.9% of observations below q01 and a
  1.8% upper tail at 7 to 10 days). A calibration fitted per 24-hour lead band and per group
  (season, day or night, cold winter night) on the forecast errors of earlier origins brings the
  tails to nominal at every lead from 0.5 to 240 hours without costing quantile score.
prediction: >-
  Written before any score is read; each can fail. (Stage 0) The tail rates are off by at least 0.8
  percentage points at every lead band past 96 hours, the miss is larger in the upper tail on cold
  winter nights and in the lower tail on sunny daylight, and it varies more by Primary than by lead
  past 168 hours. (PC1, C1 per-band per-group widening) The shares below q01 and above q99 fall to
  within 0.3 points of 1% at every lead band, the 5% to 95% band covers 89% to 91%, cold-winter-
  night
  q95 coverage reaches 0.89 to 0.91 at 168 to 240 hours, q01 and q99 pinball improve by at least
  0.03 points pooled, and the pooled quantile score stays within 0.01 points of the default. (PC2,
  C2 parametric tail) C2 beats C1 on q01 and q99 pinball by at least 0.01 points at 168 to 240
  hours. (PC3, C3 adaptive conformal) Rolling coverage stays within 3 points of nominal in every
  month with a pooled score no worse than C1's, and C3 is adopted only if its worst-month tail miss
  is at least 1 point smaller than C1's.
decision_rule: >-
  Family rule of research-default.md, three pre-registered arms (C1, C2, C3), at most one adopted.
  The arms change a setting on stored forecasts (tier 1), so the minimum gain on the quantile score
  is 0 points; the case for adoption rests on tail pinball and coverage improving with no harm
  elsewhere. An arm is adopted when its tail pinball improves, the headline score is no worse than
  the tolerance, no protected group breaks its bound at 0.5 to 240 hours (pooled 0.02 points, q01
  and q99 pinball 0.05, cold winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative), and
  leaving one Primary out does not flip the sign. The reproduction gate (the default equals
  experiment 322's default 3 within 0.01 points: 2.727 at 0.5 to 240 hours, 3.077 at 168 to 240
  hours) comes first. If stage 0 shows tail rates within 1.2% and 98.8% to 99.2% at every lead band,
  the result is "already calibrated" and the arms are not run.
benchmark: >-
  Primary demand research default 3 (experiment 322's D1). The manual heuristic and climatology
  are references; BG2 is a reference at 0.5 to 168 hours only.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  at 0.5 to 240 hours, six Primaries, scored year 2025-07-02 to 2026-06-30, origin 09:00 UTC, rows
  outside the hand-filtered events. Share of observations below q01, q05 and above q95, q99, and
  coverage of the 5% to 95% band, per 24-hour lead band. Pinball loss at q01, q05, q95, q99.
  Threshold-weighted CRPS. 12-month block bootstrap intervals.
best_arm: >-
  None adopted. Default (Primary demand research default 3) stays: 2.727 points at 0.5 to 240 hours.
  C1, the best calibration, scores 2.728 (difference +0.0014, 95% interval -0.0019 to +0.0054).
headline: >-
  As-of tail calibrations move the shares below q01 and above q99 only part of the way to 1%, raise
  the q01 and q99 pinball loss slightly, and leave the quantile score unchanged to within 0.006
  points. Fitted with hindsight on the scored year, the same C1 table reaches 1.00% in every band
  and lowers the score by 0.0038 points, so the miss is year-specific and not a limit of the method.
take_home_message: >-
  Primary demand research default 3 is already close to calibrated at 1 to 10 days: its 5% to 95%
  band covers 89.3% to 90.9% and 1.1% to 1.9% of observations fall below q01. Calibrations fitted
  on earlier months move the tail shares only part of the way, because the miss changes from year
  to year, so none is worth adopting. A table fitted with hindsight on the same year would fix the
  tails and slightly improve the score, so a calibration that follows the current year is the next
  thing to try.
lessons:
  - >-
    The tail miss of Primary demand research default 3 is small (at most 0.95 points) and shifts
    between months and between years.
  - >-
    C3 (adaptive conformal) holds the shares below q01 at 1.0% to 1.3% and has the smallest
    worst-month miss (1.6 points against 3.4 for the default), but its q99 pinball loss is 0.011
    points worse.
  - >-
    Cold winter nights at 216 to 240 hours stay 10% above q95 (nominal 5%) under every arm.
advice:
  - >-
    Do not add a table calibration of q01, q05, q95, and q99 to Primary demand research default 3 at
    0.5 to 240 hours, because as-of fits change the quantile score by less than 0.006 points and
    raise the q01 and q99 pinball loss (C1 to C3).
  - >-
    Test the cold-winter-night upper tail at 216 to 240 hours with a model input, not a calibration,
    because every calibration leaves 9.6% to 9.9% above q95 there (stage 0 and C1).
  - >-
    Test a calibration that follows the year, such as a faster-adapting conformal update, only if
    NGED need the tail shares at 1.0%, because the hindsight fit shows room of 0.0038 points (post
    hoc check).
verdict: not supported
production_status: not-for-production
adopted_in: none
code: ml-experiments/00335-primary-demand-default-3-tail-calibration-at-0-5-to-240-hours/
---

## Take-home message

**Primary demand research default 3 is already close to calibrated at 1 to 10 days, and tail
calibrations fitted on earlier months do not improve it.** Primary demand research default 3 is our
research demand forecast for six Primary substations (anonymised as D5, D9, D10, D11, D14, and D16),
up to 10 days ahead. We asked whether correcting its 0.01, 0.05, 0.95, and 0.99 quantiles, with a
different correction for each 24-hour lead band and each kind of half-hour (season, day or night,
forecast cold night), brings the share of observations beyond each quantile to nominal. Three
corrections were tested, each fitted only on origins completed before each month: C1 widens or
narrows each tail by a table of factors (the recipe of experiments
[160](../00160-calibration-band-width-fix-on-stage-2-winner/result.md) and
[164](../00164-calibration-cold-night-tail-under-coverage/result.md) in every band), C2 replaces
the tails with a Student-t fitted to the errors, and C3 is adaptive conformal prediction ([Gibbs
and Candès (2021)](https://arxiv.org/abs/2106.00170)) with a 60-day window. None reaches nominal:
the share below q01 stays at 1.0% to 1.4% (the default: 1.1% to 1.9%). The quantile score, twice the
mean pinball loss in points of effective capacity, does not improve (C1 +0.0014, 95% interval
-0.0019 to +0.0054 points), and the q01 and q99 pinball loss rises by 0.003 to 0.011 points. Fitting
the C1 table with hindsight on the scored year gives 1.00% in every band and lowers the score by
0.0038 points, so the miss changes from year to year and a table fitted on earlier months cannot
follow it.

Horizon: leads 0.5 to 240 hours (NGED's 1 to 10 days). Retraining schedule: the stored forecasts of
experiment [322](../00322-primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication/result.md)
are used unchanged, so no model was refitted here. The reproduction gate passed: the default scores
2.7265 at 0.5 to 240 hours and 3.0767 at 168 to 240 hours (expected 2.727 and 3.077).

## Figures

![Share below q01 by lead](tail_below_q01.svg)

![Share above q99 by lead](tail_above_q99.svg)

![Coverage of the 5% to 95% band by lead](coverage_5_95.svg)

![q99 pinball loss against the default](q99_pinball_difference.svg)

## Results

- **Arms:** default (Primary demand research default 3); C1 (factor on the distance of q01, q05, q95,
  and q99 from the median, per band and group, bounded 0.8 to 1.6, shrunk to 1 under 500 half-hours
  per cell, interior quantiles unchanged apart from a sort); C2 (Student-t tail beyond q10 and q90);
  C3 (adaptive conformal, step 0.005, 60 days, per band, no groups). Groups: Apr-Oct day, Apr-Oct
  night, Nov-Mar day, Nov-Mar night, Nov-Mar forecast cold night (at least half of the 51 members
  below 3 C; the member share is missing below 168 hours, so those half-hours fall in "Nov-Mar
  night").
- **Rows:** `fold_a_daily_gap_v1`, six Primaries, scored year 2025-07-02 to 2026-06-30, origin
  09:00 UTC, 798,116 half-hours outside hand-filtered events. History for the fits is the replicate
  year (even months of 2024-25, 168 to 240 hours only) plus earlier scored months, so the first
  cuts have no history below 168 hours and use a factor of 1 there.
- **Quantile score at 0.5 to 240 hours:** default 2.7265; C1 2.7280; C2 2.7321; C3 2.7319; manual
  heuristic 3.4143; climatology 3.6644. BG2 is not scored past 168 hours. At 168 to 240 hours the
  default scores 3.0767 and C1, C2, and C3 score 3.0786, 3.0839, and 3.0858.
- **Contrasts against the default (0.5 to 240 hours, 95% interval, 12 origin-month blocks):** C1
  +0.0014 (-0.0019 to +0.0054); C2 +0.0056 (+0.0012 to +0.0098); C3 +0.0054 (+0.0002 to +0.0116).
  Threshold-weighted continuous ranked probability score (twCRPS): C1 +0.0004, C2 +0.0012, C3
  +0.0016 (each interval includes zero except
  C3 against C1).
- **Tail calibration (pinball difference against the default, 0.5 to 240 hours; positive is worse):**

| Arm | q01 | q05 | q95 | q99 |
|---|---|---|---|---|
| C1 | +0.0032 | +0.0059 | -0.0004 | +0.0026 |
| C2 | +0.0082 | +0.0044 | +0.0072 | +0.0107 |
| C3 | +0.0028 | +0.0093 | +0.0073 | +0.0112 |

- **Tail shares by lead (below q01 / above q99, per cent):** default 1.1 to 1.95 / 1.4 to 1.9; C1
  1.0 to 1.6 / 1.5 to 1.7; C2 0.6 to 1.4 / 0.7 to 1.3; C3 1.0 to 1.3 / 1.1 to 1.4. The 5% to 95%
  band covers 89.3% to 90.9% for the default and 89.5% to
  90.0% for C3 (nominal 90%).
- **Worst month:** the largest tail miss in one month is 3.4 points for the default, 2.9 for C1, 2.5
  for C2, and 1.6 for C3.
- **Cold winter nights:** C1 lowers the score by 0.014 points (95% interval -0.028 to 0.000) at 0.5
  to 240 hours. The share above q95 at 216 to 240 hours stays at 9.6% to 10% under every arm
  (nominal 5%), and at 168 to 216 hours C1 brings it from 7.2% to 5.2% and from 7.1% to 5.8%.
- **Sunny daylight:** no arm changes the score by more than 0.009 points (all intervals include
  zero).
- **Hindsight fit (post hoc):** the C1 table fitted on the scored year itself gives 1.00% below
  q01 and above q99 in every band, apart from 1.02% to 1.06% below q01 past 192 hours, and lowers
  the score by 0.0038 points.

## Checks

- **Median minus mean forecast, per site:** D5 +0.04, D9 -0.24, D10 -0.43, D11 -0.02, D14 +0.18,
  D16 +0.19 points; identical for every arm, because no arm changes the median.
- **Error of the median forecast beside each contrast:** mean absolute error 4.7523 points for every
  arm, so the contrasts above come from the tails only.
- **Triggers:** trigger 2 fired for the prediction that C1 brings the tail shares within 0.3
  points of 1%. The explanation was tested by the hindsight fit above, which reaches 1.00% with the
  same table, so no audit was run. Leaving one Primary out does not change the sign of any
  contrast. No other trigger fired.

## Caveats and what to try next

- Predictions: stage 0 predicted a miss of at least 0.8 points at every band past 96 hours, and the
  miss was 0.45 to 0.95 points, so that prediction failed in part. PC1 failed, because the tail
  shares stay beyond 0.3 points of 1% and the q01 and q99 pinball loss does not improve by 0.03
  points. PC2 failed, because C2 is worse than C1 on q01 and q99 pinball. PC3 failed, because
  the 5% to 95% band of C3 misses 90% by 5.5 points in its worst month (the prediction was within
  3 points), although its worst-month tail miss is 1.3 points smaller than C1's.
- The curtailment-cost metric was not scored: it needs a node and its generators, and the Primary
  demand series have none.
- The history below 168 hours is short for the first cuts of the scored year (see Rows), which
  makes C1 and C2 weaker there.
- Next: a model input for the cold-winter-night upper tail at 216 to 240 hours, and a calibration
  that adapts within the year (a faster conformal update) only if NGED need the tail shares at 1.0%.
