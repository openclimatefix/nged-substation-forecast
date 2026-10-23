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
  the score. (P2) The empirical shape at its fitted capacity (e2, level 0.762 at D9 and 0.687 at
  D16, 1.003 at D5) lowers the April to June daylight mean absolute error by at least 0.02 MW,
  averaged over D9 and D16, and may overshoot s2 (the oracle cut) because e2 also changes the
  shape. At D5 e2 moves the error by under 0.02 MW with the 60% export clip on. (P3) The April to
  June factor calibrated on April to June 2025 (s1, factors 1.02 to 1.13, so a larger capacity)
  does not lower the April to June 2026 error: its point estimate is above -0.01 MW at D9 and D16.
  (P4) D10's April to June night-time error does not change by more than 0.02 MW in any arm; if it
  does, the change is not caused by the solar term and is reported as a side effect.
decision_rule: >-
  Five planned contrasts, each a paired difference against the control (c0) on experiment 178's
  rows (scored year, leads up to 168 hours, outside the switching events), in mean absolute error
  of the median forecast in MW, with week-block 99% intervals (resampling the weeks of forecast
  initialisation time, equal weight per substation; about 13 blocks fall in April to June, so
  intervals are wide and the page says so). C1: e2 minus c0 on April to June daylight half-hours,
  mean of D9 and D16, assembly configuration (60% export clip at D5 only). C2: e2 minus c0 on
  sunny daylight half-hours, mean of D5, D10, D11, D14, and D16 (the five decision substations),
  same configuration. C3: f1 minus c0 on the C1 rows, which separates the capacity change from the
  shape change. C4: e1 minus c0 on the C1 rows, which isolates the shape. C5: s1 minus c0 on the C1
  rows. The verdict reads only the assembly configuration, and C1 first. `supported`: C1's point
  estimate is at most -0.02 MW with its interval wholly below zero, C2's point estimate is at most
  +0.01 MW, and no substation's sunny-daylight error rises by 0.02 MW or more. `partly supported`:
  C1's interval is wholly below zero but its point estimate is above -0.02 MW or the no-harm test
  fails. `small gain`: C1's interval is wholly below zero and its point estimate is above -0.02 MW
  with the no-harm test passed. `refuted`: C1's interval lies wholly above zero. `not supported`:
  C1's interval contains zero and its point estimate is above -0.01 MW. `inconclusive`: C1's
  interval contains zero and its point estimate is at or below -0.01 MW. If P4 fails, the verdict
  stands and the side effect is reported. The -0.02 MW threshold is about 4% of the control's
  April to June daylight error at D9 and D16 (0.47 and 0.59 MW) and about 30% of the control's gap
  to BG2 at D9 (0.07 MW); a smaller gain does not repay a second stored table per substation.
  A leave-one-site-out check on C1 repeats it with D9 and then D16 alone.
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
weather, so only the solar term differs. **No level, table, or factor reads data from 2025-07-01
onward**: the pooled tables use each farm's rows before that date, the capacities that set the
levels are refits that stop at that date, and the factor of s1 uses only April to June 2025. The
rows are fixed by the fleet shape on CAMS irradiance, so every arm is scored on identical rows.
The control c0 must be bit-identical to experiment 180's control; if it is not, the run aborts.

- **c0, control:** the fleet shape at the pinned capacity.
- **e0, empirical shape at the pinned capacity:** the pooled empirical table, level 1.
- **e1, empirical shape at the swapped level:** level equal to the capacity fitted under the
  empirical shape over the capacity fitted under the fleet shape (1.07 to 1.11), so swapping the
  shape leaves the solar output level unchanged.
- **e2, empirical shape at its fitted capacity:** level equal to the capacity fitted under the
  empirical shape over the pinned capacity (0.69 at D16, 0.76 at D9, 0.94 to 1.17 elsewhere).
- **f1, fleet shape at the real-fit capacity:** level equal to the capacity fitted under the fleet
  shape over the pinned capacity (0.63 at D16, 0.69 at D9, 0.87 to 1.09 elsewhere). Comparing f1
  with e2 separates the shape from the capacity.
- **s1, seasonal capacity-bias check:** the fleet shape times a factor in April, May, and June,
  calibrated on the control's April to June 2025 forecasts only (1.02 to 1.13).
- **s2, oracle bound (not a forecast):** the fleet shape times the April to June factors that
  [experiment 266](../00266-summer-gap-diagnostics-solar-term/result.md) measured on the scored
  year (0.59 at D5, 0.86 at D9, 0.90 at D16, 1 elsewhere). It uses scored-year information and
  shows how much a capacity fix could gain at best.
- **Export clip:** the assembly configuration has the 60% clip at D5 only (as in
  [experiment 180](../00180-dp-sunny-daylight-fix-d5-solar-shape/result.md)). A second
  configuration clips at D9 too, and a third has no clip. Only the assembly configuration decides
  the verdict.

D9 and D16 are the primary pair because their real-fit levels are far below the pinned capacity.
D5 is read separately, because the level there is about 1.0 and the clip dominates. D10, D11, and
D14 are the out-of-selection check: the real fit did not pick a capacity cut there.

## Sources and checks

The shape comes from [experiment 269](../00269-pv-capacity-gain-decomposition/result.md). Earlier
shape experiments ([73](../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) and
[77](../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md)) found no score gain from a
different shape, which is why P1 predicts none for e0 and e1. Earlier `advice` items followed:
score on identical rows, report absolute error beside every contrast, and fix the row set from a
shape no arm changes. No earlier item is departed from.
