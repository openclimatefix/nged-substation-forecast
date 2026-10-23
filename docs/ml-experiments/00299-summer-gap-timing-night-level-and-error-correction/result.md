---
id: '00299'
slug: summer-gap-timing-night-level-and-error-correction
title: >-
  When does the best-guess demand model fall behind BG2 in spring, what causes D10's night offset,
  and does the error-correction layer make the spring excess worse?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-forecast-engine
  - differentiable-physics-embedded-generation
status: done
size: >-
  medium, three zero-fit parts in one experiment. Stored: private files under
  data/ml-experiments/00299-summer-gap-timing-night-level-and-error-correction/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: the timing and night-level parts have one
  defensible design (rescoring stored forecasts); the error-correction part has two (stored pairs
  versus a refit), and this plan pre-registers the stored pairs and records a refit as not done.
  Callers: this folder's scripts, which read experiment 266's master frame and the stored forecasts
  of experiments 111, 155, 200, 201, and 289 and call reviewed `studies.*` modules unchanged. No
  fits, no GPU, no downloads, 2 CPU cores, thread caps 1.
hypothesis: >-
  The April-to-June excess of arm D (the convex demand model with a Kalman filter on recent errors)
  over BG2 (the production-style XGBoost forecast) at the substations D5 and D9 is a step that
  starts at the March clock change and sits at short leads, which would point at a local-time
  mismatch or at BG2's use of recent power. D10's night-time level offset of about -2 points
  (21:00 to 07:30 UTC) is a step on one date, not a smooth drift with temperature. The Kalman
  filter and the correction head are net harmful at D5 and D9 in April to June.
prediction: >-
  Written before any stored forecast is cut by week, hour, lead, or date. Each line can fail.
  (Part 1, timing) At least half of D5's and D9's April-to-June daylight excess of arm D over BG2
  appears within 4 weeks of the last Sunday of March (2026-03-29); and at least half of it sits at
  leads under 24 hours. If neither holds, the excess is gradual and physical. (Part 3, D10 night
  level) The nightly mean residual of arm D at D10 changes in a single step inside a 2-week window,
  not smoothly with temperature; if smooth, the heating hinge or the night demand term is mis-set
  for spring, and the page names which. (Part 2, error correction) Removing the Kalman filter or
  the correction head lowers the April-to-June excess at D5 and D9 by at least 30% and changes July
  to September by under 0.02 points.
decision_rule: >-
  Intervals resample whole ISO weeks of forecast initialisation time (about 13 weeks in April to
  June), the same blocks for every contrast. Part 1: the timing prediction holds only if both
  halves hold at D5 and at D9 separately; one half holding is reported as partial. A step after the
  clock change is also read against the October 2025 change as a mirror test. Part 3: a single step
  is supported when a one-change-point fit on the nightly mean residual lowers the squared error
  by at least 50% of the total variance explained by a fit of the same residual on night
  temperature band and month, and the change point's 95% interval (block bootstrap over weeks) is
  under 2 weeks wide; a candidate date is reported as "inferred, unconfirmed". Part 2: supported
  when the lower bound of the 95% interval of the reduction is at or above 30% at both D5 and D9
  and the July-to-September change has an interval inside -0.02 to +0.02; refuted when the upper
  bound is below 30%; otherwise undecided. A filter that helps in every season is reported as a
  finding. The threshold of zero applies to porting any removal, because dropping a layer adds no
  production complexity; the 30% bar is for how much of the spring excess it must explain to be
  the cause.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2) on the 561,207 rows of experiments 228, 251, and 266 (six
  substations, 2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside
  hand-filtered load shifts). Part 2's without-filter and without-head arms are scored on the rows
  they share with arm D after an exact key-equality gate, with the row count stated.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  the signed residual of the median in points, and the share of the April-to-June excess that each
  cut accounts for.
code: ml-experiments/00299-summer-gap-timing-night-level-and-error-correction/
verdict: not supported
best_arm: >-
  Not applicable: a diagnostic. Arm D minus BG2 in April to June is +0.319 points (95% interval
  +0.198 to +0.430, 14 weeks), reproducing experiment 266.
headline: >-
  The April-to-June excess is not a step at the clock change and not a short-lead effect. At D5 it
  is a midday, clear-sky excess at every lead (+1.67 points on clear half-hours); at D9 it is broad
  across the day. D10's night offset has no single step. Neither the Kalman filter nor any of the
  three correction heads raises the spring excess: the heads lower it.
lessons:
  - >-
    The excess grows with sun elevation and sunshine, so a clock-change mismatch is not the cause
    (D5 clear half-hours +1.67, overcast -0.31 points).
  - >-
    The Kalman filter changes the April-to-June excess by +0.02 (D5) and +0.06 (D9) points, and
    helps in July to September at D9 (-0.60).
  - >-
    The correction heads lower arm D's April-to-June daylight score at D5 by 0.49 to 0.63 points,
    so they hide part of the excess instead of causing it.
  - >-
    D10's night-time residual varies by month (R squared 0.13) but a single step explains 0.01 to
    0.03 of its variance.
production_status: not-for-production
adopted_in: none
date: 2026-10-23
take_home_message: >-
  The spring excess of the convex demand model over BG2 grows with sunshine, appearing on clear
  midday half-hours at D5 and across the whole day at D9, so the cause is how the solar term
  responds to strong sun and not a clock-change error or a short-lead error. The Kalman filter and
  the correction heads are not the cause: removing the filter changes little and the heads reduce
  the excess. D10's night-time offset drifts from month to month and has no single date on which it
  changes.
advice:
  - >-
    Look for the spring excess in the solar term's response on clear, high-sun half-hours at D5 and
    D9, not in the clock-change handling or the error-correction layers (Parts 1 and 2, figures 1
    and 4).
  - >-
    Keep the correction head when testing spring fixes: it lowers the D5 April-to-June daylight
    score by 0.49 points (Part 2).
  - >-
    Treat D10's night offset as a month-by-month level error and test a night term that varies by
    month, not a metering step on one date (Part 3, figure 3).
  - >-
    Stratify spring checks by realised clear-sky index, because the excess at D5 changes sign
    between clear (+1.67) and overcast (-0.31) half-hours (Part 1).
---

**The spring excess of the convex demand model (arm D) over BG2 follows the sun, not the clock.** We
asked three questions of experiment 266's stored forecasts. When does arm D fall behind BG2, the
production-style XGBoost forecast? What is D10's night-time level error? Do the Kalman filter on
recent errors and the neural-network correction heads make the spring excess larger? Nothing is
refitted. The excess sits on clear midday half-hours at D5 (+1.67 points) and across the whole day at
D9, at every lead, so a clock-change mismatch and BG2's use of recent power do not explain it. The
Kalman filter changes it by +0.02 and +0.06 points, and the three correction heads lower it. D10's
night residual has no single step. D5, D9, and D10 are anonymised Primary substations. The scores
are quantile scores in points of effective capacity, over experiment 266's 561,207 rows (six
substations, 2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside hand-filtered
load shifts), and intervals resample whole ISO weeks of forecast initialisation time (14 weeks in
April to June).

The pre-registered predictions all failed: the excess is not concentrated within four weeks of the
clock change or at short leads (Part 1), D10's offset is not one step (Part 3), and removing the
filter or a head does not lower the excess by 30% (Part 2).

![Weekly daylight excess at D5, D9, and D10](fig1_weekly_gap.svg)

![April-to-June daylight excess by lead band at D5 and D9](fig2_lead_band.svg)

![D10 nightly residual and monthly mean](fig3_d10_night.svg)

![Share of the excess removed by each layer](fig4_layers.svg)

## Results

- **Gate:** arm D minus BG2 in April to June is +0.319 (+0.198 to +0.430), the figure of experiment
  266.
- **Part 1, timing (predicted, and failed):** the first four weeks after 29 March 2026 hold 29% of
  D5's April-to-June daylight excess (27% of its rows) and 10% of D9's (26% of rows); the required
  share was 50%. Leads under 24 hours hold 14% of D5's excess (14% of rows) and 6% of D9's (14% of
  rows); the required share was 50%. Both halves fail at both substations, so the excess is gradual.
- **Lead and sunshine, April to June daylight (points):** D5 +2.45 at 0 to 6 hours, +0.30 at 6 to
  24, +1.22 at 24 to 48, +1.10 at 48 to 168; D9 +0.25, +0.35, +0.59, +0.84. By realised clear-sky
  index at D5: clear +1.67 (+1.23 to +2.11), mixed +0.72, overcast -0.31 (-0.90 to +0.35). At D9:
  clear +1.02, mixed +0.47, overcast +0.25.
- **Hour of day, April to June, local time:** D5's excess peaks at 12:00 to 15:00 (+3.0 to +3.7
  points) and is near zero at night. D9's is +0.5 to +1.0 from 08:00 to 23:00 and +0.2 to +0.4 at
  night, so it is not shaped like solar output alone.
- **Clock-change comparison (descriptive):** D5's daylight gap is -0.03 in the four weeks before
  29 March 2026 and +1.12 in the four weeks after, but in the autumn it falls from +0.52 to +0.13
  across 26 October 2025. The jump at D5 and the fall in autumn both match sun elevation and
  sunshine, and this design cannot separate a clock-change effect from them.
- **Season at D5 and D9:** July to September daylight excess is +0.31 (D5) and +0.17 (D9), against
  +1.12 and +0.73 in April to June. D10: +0.20 daylight and +0.60 night in April to June.
- **Part 3, D10 night level (predicted, and failed):** over 635 nights (2024-07 to 2026-06, leads 12
  to 36 hours, rows outside hand-filtered load shifts, which is a subset of the rows the earlier
  experiments scored), the best single-step fit explains 0.9% of the variance of the nightly mean
  residual; a fit on month explains 12.7% and on temperature band and month 13.3%. Within the scored
  year the step explains 2.6% and temperature band and month 11.1%. The change-point date has a 95%
  interval of 623 days (whole period) and 310 days (scored year), far above the 2-week bar. The
  April-to-June mean residual is -0.55 points in 2025 and -0.95 in 2026. Day of week explains 1.2%
  (Sunday and Monday evening nights -1.0 to -1.2 points). No candidate date is inferred.
- **Part 2, error correction (predicted, and failed), April-to-June daylight:** removing the Kalman
  filter from GAM-K (experiment 111's GAM-K against its filterless GAM0) lowers the excess by
  1.7% at D5 (-18.7% to +21.5%) and 8.1% at D9 (-54.9% to +38.3%), both under the 30% bar, and
  changes July to September by -0.04 (D5) and +0.60 (D9) points, so the July-to-September condition
  also fails. Removing a correction head (experiment 201's arms 1, 5, and 6 against arm D) raises
  the D5 excess: arm 1 by 0.49 points (-0.71 to -0.28 for the head's own effect), arm 5 by 0.63,
  arm 6 by 0.58. At D9 the head's effect is -0.23 (-0.54 to +0.16), -0.14, and +0.10 points.
- **Filter by season, all six substations, GAM-K minus GAM0:** -0.14 (-0.39 to +0.06) points in
  April to June daylight and -0.30 (-0.42 to -0.19) in July to September daylight.

## Checks

- **Median minus mean forecast, arm D, per substation:** exactly 0.000 points at D5, D9, D10, D11,
  D14, and D16, because arm D's median is its mean. The mean-forecast error equals the
  median-forecast error at every substation (mean absolute error of arm D: D5 6.01, D9 4.21, D10
  5.31, D11 4.21, D14 3.69, D16 4.48 MW).
- **Triggers:** 1 (a control fails), 3 (quantile score against mean-error direction), 4
  (leave-one-out), 5 (nested arm), and 6 (interval width) do not apply, because the experiment has
  no controls, no leave-one-out contrast, and no sister experiment on these contrasts. Trigger 2
  (a result contradicts its prediction) fired for every prediction; the page records an explanation
  for each (sun elevation and sunshine, a month-scale drift, and the heads helping), and the head
  result agrees with experiments 200, 201, and 266. Trigger 7 holds for the planned
  solar-times-temperature experiment, whose design depends on the D5 and D9 excess being a
  clear-sky effect that the additive model cannot give different weights; the manager decides
  whether that needs an audit.

## Caveats

- **Part 2 is stored pairs only.** Arm D without its filter is not stored, so the filter test uses
  the older GAM-K and GAM0 of experiment 111, which differ from arm D (no night offset, no recipe R).
  A refit of arm D without the filter was not run.
- **D10's night residual here is smaller than the -2 points of experiment 281.** This page uses arm D
  at leads 12 to 36 hours over 21:00 to 07:30 UTC outside load shifts; experiment 281 used a different
  arm and lead set.
- **Intervals are wide.** April to June holds 14 weeks of forecast initialisation, and the
  nightly series are noisy; the single-step test has little power, which the page reports through
  the interval width.
- **Retraining schedule:** not applicable, because nothing is refitted.

## What to try next

- Test the solar term's response on clear, high-sun half-hours at D5 and D9 directly, for example
  with the experiment 4 interaction terms of `dp_summer_gap_next_experiments.md`, using realised
  clear-sky index strata as the first read.
- Test a night term that varies by month at D10 instead of looking for a date.

## Code

- `ml-experiments/00299-summer-gap-timing-night-level-and-error-correction/`: `e00299_timing.py`,
  `e00299_night.py`, `e00299_correction.py`, and `e00299_charts.py`.
