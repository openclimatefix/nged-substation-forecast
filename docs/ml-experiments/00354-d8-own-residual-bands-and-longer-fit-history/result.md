---
id: '00354'
slug: d8-own-residual-bands-and-longer-fit-history
title: >-
  Do a forecast band fitted on D8's own residuals, and a longer fit history with a level step, close
  D8's gap to BG2?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - retraining-and-training-history
status: planned
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history/
  only. Serving path: not touched. Degradation rule: not touched (the cold-start rule that falls
  back to the donated spread is an R&D device here, and a port would need its own plan). Designs:
  more than one is defensible (band by residual quantile, by hour-and-season cell, or shrunk to a
  prior; level step per hour band or a single step), so the plan fixes all three band rules and
  both step choices as arms. Callers: the scripts import experiments 252 and 264 and call their
  fit, forecast, and scoring code, one new design column excepted. Asset graph: not touched.
hypothesis: >-
  (Fix 1) At D8 (Marsh Lane), a forecast band built from the quantiles of D8's own past residuals
  scores much better than the band the stack borrows from six other Primaries, and costs little
  at Primaries with no gas engine. (Fix 3) A GAM fitted on D8's history since 2023-09, with a free
  level step per hour band at 2025-04-01, scores better than the GAM fitted from 2025-04 only,
  and the step is needed.
prediction: >-
  Written before any new score is read. The stored-data check in the analysis (a 365-day as-of
  residual band on experiment 252's arm D, scored 8.70 against BG2's 7.58 on rows after the cold
  start) was already known. Fix 1: arm 2 (arm D median plus as-of 365-day residual quantiles)
  closes at least 60% of arm 1's gap to BG2 in every one of the 12 scored months, keeps q05 to q95
  coverage in 0.85 to 0.95 in each quarter after the first, and does not beat BG2 in more than two
  months. At D2, D3, and D4 the same rule changes the quantile score by less than 3% of the
  site's arm 1 score. Fix 3: (b) lowers the first three scored months (July to September 2025) by at
  least 2 points and the year by at least 0.5 points of quantile score against (a), and (c) is worse
  than (a) in 2026.
decision_rule: >-
  Five planned contrasts, paired, month-resampled (20,000 resamples), 99% intervals (a Bonferroni
  correction for five). C1: arm 2 minus arm 1 at D8, all rows. C2: arm 3 minus arm 2. C3: arm 4 minus
  arm 2. C4: (b) minus (a) over all rows, with each fit's best band from fix 1. C5: (c) minus (a) in
  2026. C1 and C4 bear the verdict. Threshold for C1: the interval wholly below -1.0 points,
  because a rolling residual table and a cold-start rule are a new pipeline step that every
  Primary would carry. Threshold for C4: the interval wholly below -0.5 points, because the longer
  fit and the step column are one more design term at D8 only. `supported` when both clear their
  thresholds and the pre-registered monthly, coverage, and placebo conditions hold. `partly
  supported` when one of C1 and C4 clears its threshold and the other does not. `small gain` when
  an interval lies wholly below zero without clearing its threshold and nothing clears. `not
  supported` otherwise. C2, C3, the placebo changes, and the first-three-months contrast are
  descriptive. Control gate: the script must reproduce experiment 252's stored D8 scores (arm D
  11.433, BG2 7.593) before any new arm is read.
benchmark: >-
  BG2, climatology, and manual_heuristic as stored by experiment 87; experiment 252's arm A1 (stage-2
  arm d) with its donated spread as arm 1
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series'
  effective capacity (1 point is 0.1 MW at D8); median error in MW; q05 to q95 coverage; share
  below q01; error on sunny daylight half-hours
code: ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history/
verdict: null
production_status: not-for-production
adopted_in: none
date: 2026-10-26
---

**Plan (before any score is read).** The write-up replaces this section.

- **Question.** D8 (Marsh Lane) scores 11.43 points on the stack's card against BG2's 7.59. Most of
  the gap sits in the forecast band, which the stack borrows from six other Primaries, and the rest
  may come from a fit history of only 3 to 14 months. Experiment 353 found a band fitted on D8's
  own earlier months lowered the score by 2.22 points (-2.90 to -1.56). This experiment tests that
  fix against placebo Primaries, and tests a longer fit history. The fixes were approved by the
  maintainer on 2026-10-26.
- **Fix 1 arms (stored forecasts, all at D8).** (1) Arm D with the donated spread, as stored. (2) Arm
  D's median plus the as-of quantiles of D8's own residuals over the trailing 365 days (28-day
  minimum, 6-hour reporting lag, refreshed every 7 origins; method of experiment 264). (3) The same
  by hour band and season cell. (4) Arm 2 shrunk towards BG2's own as-of residual quantiles (method
  of experiment 265, weight n / (n + 2,000)). (5) BG2. Plus arm A (no stage-2 corrections) with
  arm 2's band, because idea 2 of the analysis is folded in. Before the first 28 days of residuals
  exist, arms 2 to 4 keep the donated spread.
- **Placebos.** Arms 1 and 2 at D2, D3, and D4 (Primaries with no engine), and arm 3 for a lower-tail
  check. Report the change in score and coverage, and judge the lower tail (below q01 and q05)
  separately.
- **Fix 3 arms (monthly refits at D8 only).** (a) Fit from 2025-04, as stored. (b) Fit from 2023-09
  (after the 2023-08-15 engine change) with an unpenalised level step per hour band at 2025-04-01.
  (c) Fit from 2023-09 with no step. (d), the neighbours' profile as a prior, is not run, because
  `studies.dp_gam` has no prior hook. Every fit arm is scored with the stage-2 steps (arm D) and
  without them (arm A), each with the band rule that scores best in fix 1 on all rows. That choice
  uses the scored year, so it is post hoc, but all fit arms share the band.
- **Scoring.** The scored year 2025-07-02 to 2026-06-30 at D8, leads 0.5 to 168 hours (the stored
  ensemble forecasts stop at 168 hours, so the 240-hour range cannot be scored), month-resampled 95%
  and 99% intervals. Stored-data results and refitted results are reported separately.
- **Row set.** `fold_a_daily_gap_v1`, outside the hand-filtered events, as experiment 252 scores it.
- **Weather at forecast time.** The fits and the forecasts use the same ensemble forecast weather
  and past-weather proxy as experiment 252, so every input exists at forecast time. The as-of
  residuals use readings six hours after their valid time, which is a lab assumption about NGED's
  delivery lag.
- **Earlier advice followed.** Experiment 353: fit D8's band on D8's own earlier months, and leave
  the night offset and recipe R off at D8 until it has more history (arm A tests this). Experiment
  278: monthly refits are enough, so the fits here are monthly.
- **Compute.** One site, monthly fits (about 0.4 seconds each), one process at a time; no GPU.
