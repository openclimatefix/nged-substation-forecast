---
id: '00190'
slug: dp-kalman-level-adaptation-gated-boost
part: KLA
theme: differentiable-physics-forecast-engine
also_relevant_to:
- lagged-power-and-error-correction
- production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00190-dp-kalman-level-adaptation-gated-boost/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (how the boost is gated, whether it applies to the whole gap between a band's readings),
  and this plan pre-registers one and runs the always-on and shared-level variants as arms. Callers:
  this folder's scripts, which copy the filter and forecast code of experiment 136 and the
  corrections of experiment 155; they call the reviewed `studies.*` modules unchanged. No new
  data.
hypothesis: >-
  The Kalman filter of the convex demand model (GAM-K) has a level state that is a random walk with
  one shared process-noise setting. A single setting must trade speed on a real level shift against
  noise on ordinary days. A gate that multiplies the level's process noise by m only while the last k
  one-step innovations (pooled over the four hour bands) share one sign and their mean exceeds z
  times the day-before innovation spread gives speed when a shift looks real and keeps the quiet
  setting otherwise. The mechanism predicts that the gate lowers the excess loss on days where the
  residual keeps one sign for hours (the 15 smooth-offset days of experiment 175), without harming
  ordinary days.
prediction: >-
  Written before any score is read. P1: on the 15 smooth-offset days, A1 lowers the excess loss by
  at least 20% against A0, with the gain concentrated more than 6 hours after the offset starts.
  P2: on all rows A1 minus A0 lies within -0.03 to +0.01 points, and its 95% upper bound is at most
  +0.01. P3: A1 beats A2 (always-on boost) on all rows by at least 0.01 points. P4: A3 (shared
  level) beats A1 on the smooth-offset days. P5: on the 17 block days A1 helps by under 10%.
decision_rule: >-
  The known-answer gate comes first: A1 must recover at least 50% of the injected offset's excess
  loss for the 16-hour and 36-hour offsets at the 0.10 size, and A1's score on the 40 left-alone days
  must be within +0.005 of A0's at 95%. If the gate fails the verdict is `inconclusive` and no
  contrast is read as evidence about the real days. Then the headline contrast is A1 minus A0 over
  all rows (leads 0.5 to 168 hours, the five decision Primaries D5, D10, D11, D14, and D16, the
  scored year), read at the 95% level of the month-resampled paired bootstrap (20,000 resamples).
  The hypothesis is `supported` when P1 and P2 hold and P3 holds; `partly supported` when P1 and P2
  hold and P3 fails (a plain faster filter is as good); `not supported` when P1 fails with the gate
  passed; `inconclusive` otherwise. The smallest gain worth porting is 0.02 points over all rows, or
  a 20% cut of excess loss on the smooth-offset days with no harm elsewhere, because the gate adds a
  run counter and three settings to the serving path; a smaller all-row gain is reported as a
  `small gain`. m, k, and z are chosen on the odd tuning months of the replicate year only, from
  the grid m in {3, 5, 10}, k in {4, 6, 8}, z in {0.5, 1.0}; the pre-registered point (m = 5, k = 6,
  z = 1.0) is kept unless a grid cell lowers the tuning-month score by at least 0.005 points. The
  settings are frozen before the scored year is read.
benchmark: >-
  Arm D of experiment 155 as stored (2.756 points over leads 0.5 to 168 hours at the five decision
  Primaries in the scored year) and the calibrated spread of experiment 169 arm A (2.7519). A0
  rebuilt here must reproduce 2.756 within 0.005 or the experiment stops.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles) in points of effective
  capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168 hours; also on
  rows within 24 hours after a gate trigger, on the 15 smooth-offset days and 17 block days of
  experiment 175, and on cold nights; tail coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00190-dp-kalman-level-adaptation-gated-boost/
---

# Faster Kalman level state when the residual keeps one sign

**Status: planned.** The design is a note from the Discuss: Kalman session of 2026-10-22. This page
holds the plan and the pre-registered predictions; the results are added after the code has run.

## Question

Experiment 175 found that 15 of the 37 worst days that no labelled event explained are smooth
same-sign offsets in the residual of the best-guess forecast (arm D of experiment 155), and 17 are
sharp-edged blocks that no switching detector fired on. We ask whether a level state that speeds up
when the residual keeps one sign for several hours cuts the excess loss on days like those, without
harming ordinary days. The idea is a gated version of the Kalman filter on additive-model
coefficients; the filter itself is experiment 111's, and the stored inputs are experiment 155's.

## Arms

All arms use arm D's stored inputs (the T-all fits and the ensemble-mean weather of experiment 136's
arm A2, the winter night offset, and recipe R of experiment 155). Only the level state's process noise
changes. The hour bands are as in arm D.

- **A0:** arm D as built by experiment 155. The reference.
- **A1, gated boost:** the level's process noise is multiplied by m while the counter is on.
- **A2, always-on boost:** the same multiplier at every step.
- **A3, gated boost with a shared level:** during an on-run, each band's level increment is also
  added to the other three bands' levels.

## Pre-registered predictions

1. On the 15 smooth-offset days, A1 lowers the excess loss by at least 20% against A0, with gains
   concentrated more than 6 hours after the offset starts.
2. On all rows, A1 is within -0.03 to +0.01 points of A0, and the 95% upper bound of A1 minus A0 is
   at most +0.01.
3. A1 beats A2 on all rows by at least 0.01 points.
4. A3 beats A1 on the smooth-offset days.
5. On the 17 block days A1 helps by under 10%.

## Decisions made while planning (all before any score is read)

- The counter reads innovations in time order across the four hour bands and includes the current
  reading. The "day-before innovation spread" is the root mean square of the 48 innovations before
  the k under test. The counter is off until 48 earlier innovations exist.
- The multiplier applies to the whole process-noise increment since the band's previous reading,
  because the other bands' innovations have shown that the level may have moved during the gap.
- The gate and the tuning use the 3.5-hour telemetry delay of experiment 168. Runs are the daily runs
  of experiment 155 for the scored contrasts, and every 6 hours for the known-answer gate, so that
  the gate can show the recovery by hours since the offset started.
- The known-answer gate runs on the residual stream of the proxy-weather forecast (a lab shortcut),
  with a quantile table taken from the clean filtered residuals; it needs no ensemble run.
