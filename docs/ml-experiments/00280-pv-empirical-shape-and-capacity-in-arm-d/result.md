---
id: "00280"
slug: pv-empirical-shape-and-capacity-in-arm-d
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
  - solar-disaggregation
status: finished
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
verdict: not supported
production_status: not-for-production
adopted_in: none
date: 2026-10-23
headline: >-
  April to June daylight at D9 and D16 (mean absolute error of the median forecast, equal
  substation weights): the control 0.528 MW; the empirical shape at its fitted capacity (e2) changes
  it by -0.009 MW (99% interval -0.041 to +0.024), the oracle capacity cut (s2) by -0.003 MW
  (-0.014 to +0.008). At D5, s2 raises it by +0.081 MW (+0.053 to +0.115).
best_arm: >-
  none; the empirical shape at the pinned capacity (e0) is nearest, at -0.010 MW (-0.022 to +0.001)
  on April to June daylight at D9 and D16 and +0.006 MW (+0.002 to +0.009) on all half-hours
take_home_message: >-
  Putting the empirical solar shape into the forecast, at the pinned capacity or at the lower
  capacity the real series supports, does not lower the April to June error at the substations where
  the pinned capacity looked too high. The change is under 0.01 MW against an error of about 0.5 MW,
  and the intervals are wide. A capacity cut that is correct for the whole of April to June is not
  the fix either: at D5 it raises the error. This experiment finds no sign that the shape or the
  capacity of the solar term closes the summer gap to BG2.
lessons:
  - >-
    The shape does not move the score. e0 and e1 change April to June daylight error at D9 and D16
    by -0.010 and -0.006 MW, inside intervals that include zero, and raise the error on all
    half-hours at D9 and D16 by 0.006 and 0.007 MW.
  - >-
    The capacity does not move it either. At D9 and D16 the oracle cut (s2) changes the error by
    -0.003 MW, and at D5 it raises it by 0.081 MW, so a capacity cut that suits D9 and D16 is not
    safe at D5.
  - >-
    The seasonal factor calibrated on April to June 2025 (s1, factors above 1) changes the error by
    +0.004 MW at D9 and D16 and -0.002 MW at D5. The 2025 and 2026 April to June capacities do not
    point the same way.
advice:
  - >-
    Do not put the pooled empirical solar shape into arm D to close the April to June gap to BG2: at
    D9 and D16 it changes daylight error by under 0.01 MW and raises error on all half-hours (C1,
    C4).
  - >-
    Do not cut the solar capacity by the oracle April to June factor at D5 on top of the 60% export
    clip: it raises D5's April to June daylight error by 0.081 MW (the oracle arm s2).
code: ml-experiments/00280-pv-empirical-shape-and-capacity-in-arm-d/
---

# PV capacity: does the empirical solar shape lower arm D's April to June error?

**Take-home message: no. Neither the empirical solar shape nor a lower solar capacity changes arm D's
April to June error by more than 0.01 MW at D9 and D16.** Arm D is the monthly-refit generalised
additive model of experiments
[155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md) and
[220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md), scored on [experiment
178](../00178-dp-sunny-daylight-gap-diagnosis/result.md)'s rows. [Experiment
266](../00266-summer-gap-diagnostics-solar-term/result.md) put the April to June gap to BG2 in its
solar term. [Experiment 269](../00269-pv-capacity-gain-decomposition/result.md) found a pooled
empirical shape that recovers an injected farm better than the physical fleet shape, with real
capacities 0.2 to 0.4 MW lower. This experiment puts that shape and those capacities into arm D.

**Verdict: not supported.** C1 (e2 minus the control, April to June daylight, D9 and D16) is -0.009
MW with a 99% interval of -0.041 to +0.024, so the interval contains zero and the point estimate is
above -0.01 MW. The control's April to June daylight error at D9 and D16 is 0.528 MW, and its gap to
BG2 there is +0.044 MW (-0.018 to +0.103), so the largest gain any arm showed (0.010 MW) closes under
a quarter of the
gap. Every level, table, and factor except the oracle arm s2 uses data before 2025-07-01 only, and
all 36 stored control frames equal experiment 180's control frames (`e00280_check_control.py`).

![Change in error by arm](result_effect.svg)

**Figure 1. Almost every interval contains zero; the oracle cut at D5 stands clear of it, upward.** Week-block
intervals are wide (about 13 weeks of April to June, 4,000 resamples).

## Results

**The shape alone changes the error by about 0.01 MW, and the all-half-hours error rises.** At D9
and D16, April to June daylight, e0 (the empirical shape at the pinned capacity) is -0.010 MW (-0.022
to +0.001), e1 (swapped level) -0.006 MW (-0.014 to +0.002), and C4 is therefore small. On all
half-hours at D9 and D16, e0 is +0.006 MW (+0.002 to +0.009), e1 +0.007 MW (+0.003 to +0.010), and
e2 +0.008 MW (0.000 to +0.014). The empirical shape raises the error outside April to June daylight.

**The lower capacity helps D9 and does nothing at D16.** e2 at D9 alone is -0.022 MW (-0.060 to
+0.010) and at D16 alone +0.005 MW (-0.028 to +0.040), so the pooled C1 is a D9 result. With a 60%
export clip at D9 as well, e2 at D9 is -0.004 MW (-0.010 to +0.001): the clip already removes most
of what the capacity cut offers. f1 (the fleet shape at the real-fit capacity) is -0.002 MW at D9
and D16 (-0.035 to +0.029), so the capacity gives no more than the shape did (C3).

**The capacity bias in April to June does not generalise.** s1 (a factor of 1.02 to 1.13 from April
to June 2025) is +0.004 MW at D9 and D16 (-0.004 to +0.011), as P3 predicted. s2 (the oracle factors
of experiment 266, which read the scored year) is -0.003 MW at D9 and D16 (-0.014 to +0.008) and
+0.081 MW at D5 (+0.053 to +0.115): the capacity cut experiment 266 measured at D5 is already
supplied by the 60% clip, and cutting it again overshoots.

**The other substations and the night-time error do not move.** At D10, D11, and D14, April to June
daylight changes by at most +0.003 MW in any arm, and the April to June night-time error of the five
decision substations changes by at most +0.004 MW (P4 holds). On sunny half-hours, e2 is +0.007 MW
over the five decision substations (-0.009 to +0.030), but it raises D16's sunny-half-hour error by
+0.020 MW, which reaches the 0.02 MW no-harm limit. The verdict does not change, because C1 fails first.

## Predictions

- **P1 (e0 and e1 change sunny-daylight error by under 0.02 MW at each of D9, D10, D11, D14, D16):**
  met; the largest is e0 at D9 on sunny half-hours, -0.016 MW (the group means for D16 and the
  others are within 0.004 MW).
- **P2 (e2 lowers April to June daylight error by at least 0.02 MW over D9 and D16, and moves D5 by
  under 0.02 MW):** the first part is not met (-0.009 MW); the D5 part is met (-0.003 MW).
- **P3 (s1 is above -0.01 MW at D9 and D16):** met; +0.004 MW.
- **P4 (night-time error at D10 does not move by more than 0.02 MW in any arm):** met; D10 moves by
  at most +0.004 MW.

## Caveats

- **Pinball loss agrees.** C1 in pinball loss is -0.001 MW (-0.011 to +0.008).
- **The intervals are wide.** About 13 weeks of April to June fall in the scored year, so a gain of
  0.02 MW would not show unless it sat in most weeks.
- **s2 is a bound, not a forecast,** because it reads the scored year.
- **D9 and D16 are two substations.** C1 is a mean over two, and D9 carries it.

## What the next experiment should answer

Neither the shape nor the capacity of the solar term closes the April to June gap in this test. The
next experiment should test the parts of arm D outside the solar term, and should score D9 alone,
where e2 gains 0.022 MW (-0.060 to +0.010) but the interval includes zero.
