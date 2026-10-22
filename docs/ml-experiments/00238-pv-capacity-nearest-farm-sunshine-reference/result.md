---
id: "00238"
slug: pv-capacity-nearest-farm-sunshine-reference
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00238-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible way to turn a farm's
  output into a unit shape, but the brief fixes the choice (the farm's normalised output, corrected
  for the heat loss it already carries). Callers: this folder's scripts, which copy experiment 231
  and call the reviewed `studies.dp_*` modules without editing them. Asset graph: not touched.
hypothesis: >-
  Using the recorded output of a metered solar farm within 30 km of a Primary substation as the
  measure of sunshine, in place of the CAMS satellite irradiance, lets the capacity fit of
  experiment 129 find more of an injected real solar farm.
prediction: >-
  Written before any fit. The farm-only fit of an injected farm's output on the reference farm's
  output explains at least 0.97 of its squared output, against 0.953 for the CAMS shape, and the
  recovery of the injected farm rises by at least 0.05 at 4 of the 5 decision Primaries, against
  the CAMS shape on the same rows. Experiment 231 lowers this prior: replacing the irradiance input
  by clear-sky values changed the farm-only fit by 0.002, so the expected rise is smaller than
  0.05.
decision_rule: >-
  A metered farm within 30 km used as the sunshine reference is kept for further work if the
  recovery rises by at least 0.05 at 4 of 5 decision Primaries, against the CAMS shape on the same
  rows. Point estimates decide, and the month-block intervals are reported beside them. A smaller
  rise means the CAMS irradiance is not the limit and the reference is dropped. The threshold is
  0.05 because the reference needs a metered farm near every unmetered Primary, and a smaller rise
  does not justify that data dependency. A Primary's recovery is the median over its eligible
  injected farms, and D9 is shown beside the five decision Primaries. The recovery decides when it
  disagrees with the farm-only fit. The estimate is offline only, so no serving path changes.
benchmark: >-
  The CAMS shape on the same rows and injections (the row set of the reference series).
  Experiment 231's rung 4 (the full joint fit on all rows, pooled recovery 0.827 with 80% interval
  0.814 to 0.842 at season-year 2024) is context only, because the injected farms here are a subset
  of its injections
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before 2026-07-01, at D5,
  D10, D11, D14, and D16, with D9 beside them, restricted to half-hours at which the reference
  farm has a reading. No forecast is scored.
metric: >-
  Recovery: the 99th percentile of the recovered solar output over daylight half-hours of
  season-year 2024 (July 2024 to June 2025), divided by the 99th percentile of the injected farm's
  output over the same half-hours. The recovered output is the capacity found with the farm added at
  50% of the Primary's pinned capacity, minus the capacity found without it, times the fitted solar
  shape. Both percentiles are of alternating-current output, in MW, so the ratio does not depend on
  the unit of the shape. The plain capacity ratio, in the unit of experiment 231, is reported too
code: ml-experiments/00238-pv-capacity-nearest-farm-sunshine-reference/
---

# PV capacity: does a nearby metered farm work as the sunshine reference?

**Planned, not yet run.** The experiment repeats the injection test of experiment 231: the recorded
output of a real solar farm (farms A to F stand for metered generators and are never named) is added
to a Primary substation's power at 50% of the Primary's pinned capacity, and the convex capacity fit
of experiment 129 reports how much of the farm it finds again. The only change is the solar shape:
the satellite-based shape is replaced by the recorded output of a different metered farm near the
Primary. Fits stop at 2026-07-01, so the maintainer's labelled exception (data to 2026-07-31) is not
used. No forecast is scored.

## Design

- **Distances.** From the farms' and Primaries' coordinates, the Primaries' nearest farms are
  3 to 22 km away (D5, D10, D14, and D16 within 6 km; D9 and D11 at 20 to 22 km), so every Primary
  passes the brief's 30 km rule.
- **The reference farm.** For an injected farm X, the reference is the nearest farm to the Primary
  among the other farms with data from 2019-09 (A, B, C, D, F), provided it lies within 30 km of
  X. A reference at X itself would return the injected output and make the test trivial. The
  injected farm and the reference farm are different sites, as a real unmetered farm and a metered
  reference would be, but their distance (up to 30 km) is often larger than the distance from a
  real unmetered farm to its nearest metered farm. The recovery is therefore also reported against
  the distance between the injected farm and the reference farm. Farm E is neither injected nor a
  reference, because its record starts later than 2019-09. A Primary with fewer than three eligible
  farms is reported as not testable.
- **The shape.** The reference farm's output divided by its 99th percentile over daylight
  half-hours, with the temperature derating divided out, so that the fit does not apply the derating
  a second time. The fit's derating is `u (1 - 0.004 (T + 25 u - 25))` for the shape `u` and the
  Primary's temperature `T` in degrees Celsius, and `u` is found by fixed-point steps. The script
  checks that the fit's derated shape equals the farm's normalised output to within 1e-6 on daylight
  half-hours. The reference farm's own temperature differs from the Primary's, which this
  approximation ignores. Night half-hours take a shape of zero, as the farm output does. Half-hours
  at which the reference farm has a dropout or no reading drop out of both the
  base and the injected fit.
- **Controls.** The CAMS shape on the identical rows (so a change in the rows is not mistaken for a
  change of shape), and the CAMS shape on all rows. The script asserts that the CAMS and reference
  arms train on the same number of rows. The CAMS shape on all rows, fed through the same
  shape-substitution code, must reproduce experiment 231's rung 4 (the full joint fit on all rows)
  to within 1e-6.
- **The farm-only fit.** The share of the injected farm's squared output explained by one scale of
  the reference shape and of the CAMS shape, on season-year 2024 daylight half-hours, as in
  experiment 231's rungs 1 and 2.
- **Intervals.** A month-block bootstrap, 200 resamples, stratified as in experiment 231, and the
  range over the eligible farms.
- **Limits to state.** The reference farm's output also reflects its tilt, azimuth, curtailment,
  and soiling, so a null result can mean that one farm's output is not a clean sunshine index, not
  that a better reference would not help. A reference farm metered behind the Primary would sit in
  both the base and the injected series and cancel in their difference.
- **Stop rule.** If the reference does not explain more of the injected farm than the CAMS shape
  on the farm-only fit, the injection fits still run, because they are cheap, and the write-up says
  the reference carries no new information.

## Plan checks

- **Reproduction and derating:** as above.
- **Review:** one Sonnet review of this plan, one of the code, one of the write-up.
