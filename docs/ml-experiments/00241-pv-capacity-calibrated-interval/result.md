---
id: "00241"
slug: pv-capacity-calibrated-interval
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00241-<slug>/ only (a table of
  corrected estimates). Serving path: not touched. Degradation rule: not touched. Designs: more
  than one defensible way to calibrate (the brief fixes the median recovery and the spread of
  recoveries). Callers: this folder's script, which reads experiment 129's stored known-answer
  recoveries and fits nothing. Asset graph: not touched.
hypothesis: >-
  The recoveries that experiment 129 measured when it injected real solar farms into a Primary
  substation's power can be turned into a calibrated estimate of unmetered solar capacity: dividing
  the capacity found by the Primary's median recovery over the other farms corrects its bias, and
  the spread of those recoveries gives an 80% interval that covers the truth about 80% of the time.
prediction: >-
  Written before any computation. For a held-out farm, the corrected estimate is within 10% of the
  injected capacity at 4 of the 5 decision Primaries, and the 80% interval covers the injected
  capacity in 70% to 90% of the held-out injections. Experiment 129's own range covered 23 of 30
  cells, so the prediction expects a similar coverage.
decision_rule: >-
  The calibrated interval is reported on the substation cards as an 80% interval if both
  conditions hold. If the coverage lies outside 70% to 90%, the interval stays a spread over
  scenarios and is not shown as an 80% interval. A Primary passes the 10% condition when the median
  absolute error of its corrected estimates over its held-out injections is at most 10% of the
  injected capacity. The estimate is offline only.
benchmark: >-
  The uncorrected estimate and experiment 129's range, on the same injections (23 of 30 cells
  covered)
row_set_id: >-
  Experiment 129's known-answer table for arm 2 (the convex fit with ridge 0.1 and CAMS
  irradiance): the five decision Primaries, injected farms A, B, C, D, and F at 20%, 50%, and 100%
  of a pinned capacity, season-years 2019 to 2024. No forecast is scored.
metric: >-
  Recovery: the capacity found with a real farm's output added, minus the capacity found without
  it, divided by the injected capacity, in the capacity unit of experiment 129 (the 99th percentile
  of alternating-current output over daylight half-hours, in MW). Error: corrected estimate divided
  by the injected capacity, minus 1. Coverage: the share of held-out injections whose injected
  capacity lies inside the 80% interval
code: ml-experiments/00241-pv-capacity-calibrated-interval/
---

# PV capacity: how well does a calibrated interval cover?

**Planned, not yet run.** The experiment fits nothing. It reads the recoveries in experiment 129's
known-answer table (arm 2, farms A to F are metered generators and are never named). For each
Primary substation and each held-out farm, it divides the capacity found by the median recovery
over the other four farms at the same Primary, and builds an 80% interval from the 10th and 90th
percentiles of those recoveries. The maintainer's labelled exception (data to 2026-07-31) is not
used, because the stored fits stop at 2026-07-01.

## Plan: design

- **Held-out injection.** One Primary, one farm, one fraction (20%, 50%, or 100% of the pinned
  capacity), and one season-year. The calibration set holds the same Primary's injections of the
  four other farms, without cloud-driven load. The held-out farm never enters its own calibration.
- **Corrected estimate.** The capacity found divided by the median recovery of the calibration
  set. Its ratio to the injected capacity is the held-out recovery divided by that median.
- **Interval.** The capacity found divided by the 90th percentile of the calibration recoveries
  (lower end), and by the 10th percentile (upper end).
- **Two calibration sets.** The four other farms at the same season-year as the held-out
  injection (12 recoveries), and the four other farms over all six season-years (72 recoveries).
- **Reading.** The decision reads season-year 2024, the last complete season-year, over 15 held-out
  injections per Primary (5 farms by 3 fractions). Coverage over all six season-years is reported
  beside it.
- **Shift.** The same intervals applied to injections with a cloud-driven load added (experiment
  129's `cloud_load` rows) show whether the calibration survives demand that follows cloud.
- **Limit to state.** The injections of experiment 129 set the farm's output to zero when the sun
  is lower than 15 degrees above the horizon, so the recoveries understate a farm that produces in
  winter.

## Plan: checks

- **Row counts:** every Primary and season-year holds 15 injections without load, and 15 with load.
- **Reproduction:** the stored recoveries equal the capacity found with the farm minus the capacity
  found without it, divided by the injected capacity.
