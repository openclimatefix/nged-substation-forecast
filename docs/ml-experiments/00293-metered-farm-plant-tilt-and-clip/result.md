---
id: '00293'
slug: metered-farm-plant-tilt-and-clip
title: >-
  Does a 20 degree panel tilt and a higher, softer clip improve the spread-binned solar-farm power
  model?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private member-response arrays and predictions under
  data/ml-experiments/00293-metered-farm-plant-tilt-and-clip/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design for a higher or softer
  clip (a higher hard cap, a soft cap, both), so the clip level and form are fixed here before any
  score is read. Callers: this folder's scripts, copied from experiments 224, 270, 282, and 287,
  with no new reviewed module. Asset graph: not touched.
hypothesis: >-
  The six metered farms fit best at a 15 to 20 degree tilt (experiment 282), while the spread-binned
  plant (P2 of experiment 270) assumes 35 degrees and caps its forecast at the 0.995 quantile of
  the tuning months' output. Replacing the fleet tilt with 20 degrees and the hard cap with a
  higher, soft cap lowers P2's April to July under-prediction and its 0.99 pinball loss.
prediction: >-
  Written before any score is read. The quantile score improves by under 1% (experiment 218's
  per-farm tilt and azimuth fit gained only 0.003 in the fraction of output explained). With both
  changes together (arm c), the under-prediction of the mean forecast on clear half-hours from
  April to July (experiment 282 measured 5 to 8% at the stack) shrinks by at least 3 percentage
  points, and the 0.99 pinball loss falls by at least 3%. This pair is the prediction that could
  fail.
decision_rule: >-
  Quantile score and 0.99 pinball loss in points of capacity, equal weight per farm, on experiment
  287's scored rows (own history, 12 to 36 hour leads, 2025-09 to 2026-08), with intervals from
  resampling the 12 scored months at 98.75% (Bonferroni across the four score contrasts a, b, c,
  d, each against P2). Supported when arm c lowers the 0.99 pinball loss by at least 3% with an
  interval wholly below zero and shrinks the April to July under-prediction by at least 3
  percentage points. A change of tilt and clip constants costs nothing to port, so any gain whose
  interval excludes zero would be worth taking; the 3% bar is the size that would show the plant
  shape limits the tail band. Stop rule: if arm c gains under 0.5% in quantile score and under 1%
  at the 0.99 quantile, report that the plant shape does not limit the band and do not carry the
  shape change to the stack.
benchmark: >-
  P2 (experiment 270's six-by-three forecast-and-spread-binned plant, rebuilt here as the control
  and checked to match experiment 287's stored P2), plus experiment 287's GAM, mixture-density head,
  and blend as references for the 0.99 tail
row_set_id: >-
  Experiment 287's rows: experiment 224's scored-year daylight half-hours at 12 to 36 hour leads
  that every arm holds (farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, 0.99, mean-forecast error, and relative bias on clear
  half-hours from April to July
code: ml-experiments/00293-metered-farm-plant-tilt-and-clip/
production_status: not-for-production
adopted_in: none
---

*Plan, written before any score is read.*

**Question.** [Experiment 282](../00282-pv-plant-curve-audit/result.md) found that all six metered
solar farms fit best at a panel tilt of 15 to 20 degrees, while the plant in our forecasts assumes
35 degrees, and that the best-guess stack's hard clip sits 7 to 18 points of capacity under the
farms' measured 99th percentile. [Experiment 287](../00287-metered-farm-spread-binned-plant-heads/result.md)
puts heads on the spread-binned plant ("P2", from [experiment
270](../00270-metered-farm-xgboost-vs-plant-diagnostic/result.md)). This experiment asks whether
changing the plant itself, not the head, improves P2's band.

**Clip premise.** P2 already caps its forecast at the 0.995 quantile of the tuning months' output
(1.04 to 1.11 of capacity), which is near the measured 99th percentile, so the 7 to 18 point gap
belongs to the stack and not to P2. The tuning-month 0.999 quantile (1.06 to 1.16) is the higher
level tested here.

## Arms (all on P2's six forecast bins by three spread bins, member responses recomputed)

- **Control (P2):** fleet shape at 35 degrees facing south, hard clip at the tuning 0.995 quantile.
- **(a):** fleet shape at 20 degrees, clip as the control.
- **(b):** fleet shape at 35 degrees, soft clip `s / (1 + (s / c)^6)^(1/6)` (a p-norm minimum with
  exponent 6, as in experiment 218) at `c` = tuning 0.999 quantile of output; the scale is fitted on
  the tuning rows with the clip included; the quantile cap is `c`.
- **(c):** both (a) and (b).
- **(d):** experiment 282's pooled leave-one-farm-out shape (a table of output against irradiance
  and sun elevation fitted to the other five farms), hard clip as the control, for completeness.
- **Zero history** (the other five farms train the offsets and the scale and cap are their
  medians) is scored for every arm, as in experiment 287.

## Contrasts and scoring

- **C1 to C4:** quantile score of (a), (b), (c), (d) minus P2, own history, 98.75% month-block
  interval. The same contrasts for zero history, the 0.99 pinball loss, and the March to June slice
  are secondary.
- **Reported beside them:** absolute scores, tail coverage and pinball loss at 0.01, 0.05, 0.95,
  0.99 next to experiment 287's blend, median minus mean forecast per farm, and the error of the
  mean forecast beside every score contrast.
- **April to July bias:** per farm, the sum of actual output over the sum of the mean forecast, on
  rows with clear-sky index 0.9 or more, from April to July 2026, minus one.
- **Weather:** ECMWF ensemble 00 UTC forecasts, available at forecast time, with no past weather.
- **Schedule:** one fit per arm on the months before 2025-09, no refits; a weekly refit could change
  the absolute scores. Sorted quantiles throughout. The scaled-limit curtailment metric is not run,
  because these are single series.

## Advice followed from earlier results

- Test the 20 degree tilt and a higher or softer clip (experiment 282, advice items 1 and 4).
- Experiment 282 advised against the pooled shape for a single farm; arm (d) repeats it only
  because it is scored here on the full band, which experiment 282 did not do.
