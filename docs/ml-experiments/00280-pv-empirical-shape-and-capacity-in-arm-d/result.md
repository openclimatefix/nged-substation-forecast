---
id: "00280"
slug: pv-empirical-shape-and-capacity-in-arm-d
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00280-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one way to put a new shape and
  capacity into arm D, but the brief fixes the arms. Callers: this folder's scripts, which copy
  experiment 180's pipeline and replace one method of `studies.dp_ens_weather.SunGeometry` inside
  the experiment's own process, so no `packages/` file changes. Asset graph: not touched.
hypothesis: >-
  The April to June 2026 gap between arm D (the monthly-refit generalised additive model of
  experiments 155 and 220, scored on experiment 178's rows) and BG2 comes from its solar term:
  the pinned solar capacity is too high and the south-facing fleet shape is not the shape the
  metered farms show. Replacing the fleet shape by the pooled empirical shape of experiment 269,
  at the capacity that shape's real-series fit supports, lowers arm D's April to June error at the
  substations where the pinned capacity looks too high (D5, D9, and D16).
prediction: >-
  Written before any forecast. (P1) The empirical shape at the pinned capacity (e0) and at the
  swapped level (e1) change the sunny-daylight mean absolute error by under 0.02 MW at each of D9,
  D10, D11, D14, and D16, because experiments 73, 77, and 180 found a different shape did not move
  the score. (P2) The empirical shape at its fitted capacity (e2) lowers the April to June
  daylight mean absolute error at D9 and D16 by at least 0.03 MW each, since e2's level is 0.66 and
  0.63 of the pinned capacity, and at D5 by under half of that with the 60% export clip on. (P3) The
  April to June factor calibrated on April to June 2025 (s1, factors 1.02 to 1.13, so a larger
  capacity) does not lower the April to June 2026 error: its point estimate is above -0.01 MW at
  D5, D9, and D16. (P4) D10's April to June night-time error does not change by more than 0.02 MW
  in any arm.
decision_rule: >-
  Three planned contrasts, each a paired difference against the control (c0) on experiment 178's
  rows (scored year, leads up to 168 hours, outside the switching events), in mean absolute error
  of the median forecast in MW, with week-block 98.33% intervals (resampling ISO weeks of forecast
  initialisation time; April to June holds about 13 blocks, so intervals are wide and the page says
  so). C1: e2 minus c0 on April to June 2026 daylight half-hours, mean of D5, D9, and D16 with
  equal site weights, in the assembly configuration (60% export clip at D5, none elsewhere). C2: e2
  minus c0 on sunny daylight half-hours (the top third of each substation's daylight half-hours by
  the fixed fleet shape), mean of the six substations with equal weights, same configuration. C3:
  s1 minus c0 on the same rows as C1. `supported` when C1's point estimate is at most -0.02 MW and
  its interval lies wholly below zero, C2's point estimate is at most zero, and no substation's
  sunny-daylight error rises by 0.02 MW or more (the maintainer's no-harm margin). `partly
  supported` when C1's interval lies wholly below zero but the -0.02 MW or the no-harm test is
  missed. `refuted` when C1's interval lies wholly above zero. `inconclusive` otherwise. The
  threshold is -0.02 MW because the change adds one stored table per substation and one scale
  factor to the production forecast, and a smaller gain does not repay a second solar path. The
  verdict is read before the secondary arms.
benchmark: >-
  The control c0: arm D with the fleet shape and the pinned capacity, re-run through this folder's
  code (it must reproduce experiment 180's control exactly), and BG2 (experiment 87, `p_s6_bg2`).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median forecast (q50) in MW and pinball loss averaged over the 11
  quantiles in MW, on experiment 178's rows and on the April to June 2026 daylight slice.
verdict: planned
production_status: not-for-production
adopted_in: none
code: ml-experiments/00280-pv-empirical-shape-and-capacity-in-arm-d/
---

# PV capacity: does the empirical solar shape lower arm D's April to June error?

**Plan only; the write-up replaces this page after the forecasts.** Experiment 269 found that a
table learned from the metered farms' output recovers an injected farm better than the physical
fleet shape does. This experiment puts that table into arm D and scores the forecasts. Experiment
266 had found that arm D trails BG2 in April to June because of its solar term at D5 and D9, and
that the pinned capacity looks too high there.

## Arms

Every arm refits arm D's monthly fits and rebuilds its forecasts from the same stored ensemble
weather, so only the solar term differs. Shape levels come from experiment 269's real-series fits
at season-year 2025 and the calibration of s1 uses only April to June 2025, so neither reads the
scored year.

- **c0, control:** the fleet shape at the pinned capacity.
- **e0, empirical shape at the pinned capacity:** the pooled empirical table, level 1.
- **e1, empirical shape at the swapped level:** level equal to the capacity fitted under the
  empirical shape over the capacity fitted under the fleet shape (1.07 to 1.10), so swapping the
  shape leaves the solar output level unchanged.
- **e2, empirical shape at its fitted capacity:** level equal to the capacity fitted under the
  empirical shape over the pinned capacity (0.63 to 0.95).
- **s1, seasonal capacity-bias check:** the fleet shape times a factor in April, May, and June,
  calibrated on the control's April to June 2025 forecasts only (1.02 to 1.13).
- **s2, oracle bound (not a forecast):** the fleet shape times the April to June factors that
  experiment 266 measured on the scored year (0.59 at D5, 0.86 at D9, 0.90 at D16, 1 elsewhere).
  It uses scored-year information and shows how much a capacity fix could gain at best.
- **Export clip:** each arm runs with no clip, with the 60% clip at D5 (the assembly of experiment
  180), and with the 60% clip at D9 as well (the next test that experiment 266 proposed).
