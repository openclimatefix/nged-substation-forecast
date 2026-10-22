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
  A farm within 30 km as the sunshine reference is worth pursuing if the recovery rises by at least
  0.05 at 4 of 5 Primaries. A smaller rise means the CAMS irradiance is not the limit and the
  reference is dropped. The threshold is 0.05 because the reference needs a metered farm near every
  unmetered Primary, a data dependency that a smaller rise would not pay for. The estimate is
  offline only, so no serving path changes.
benchmark: >-
  The CAMS shape on the same rows and injections (the row set of the reference series), and
  experiment 231's rung 4 on all rows: pooled recovery 0.827 (80% interval 0.814 to 0.842) at
  season-year 2024
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before 2026-07-01, at D5,
  D10, D11, D14, and D16, with D9 beside them, restricted to half-hours at which the reference
  farm has a reading. No forecast is scored.
metric: >-
  Recovery: the capacity found with a real farm's output added at 50% of the Primary's pinned
  capacity, minus the capacity found without it, divided by the injected capacity, read at
  season-year 2024 (July 2024 to June 2025). Capacity unit: the 99th percentile of alternating-
  current output over daylight half-hours, in MW
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
  injected farm therefore stands at a distance from the reference like the distance from the
  reference to an unmetered farm near the Primary. A Primary with fewer than three eligible
  farms is reported as not testable.
- **The shape.** The reference farm's output divided by its 99th percentile over daylight
  half-hours, with the temperature derating of the fit divided out so that it is not applied
  twice. Half-hours at which the reference farm has a dropout or no reading drop out of both the
  base and the injected fit.
- **Controls.** The CAMS shape on the identical rows (so a change in the rows is not mistaken for a
  change of shape), and experiment 231's rung 4 on all rows. With the reference mask switched off
  the script must reproduce experiment 231's rung 4 to within 1e-6.
- **The farm-only fit.** The share of the injected farm's squared output explained by one scale of
  the reference shape and of the CAMS shape, on season-year 2024 daylight half-hours, as in
  experiment 231's rungs 1 and 2.
- **Intervals.** A month-block bootstrap, 200 resamples, stratified as in experiment 231, and the
  range over the eligible farms.
- **Stop rule.** If the reference does not explain more of the injected farm than the CAMS shape
  on the farm-only fit, the injection fits still run (they are cheap) but the write-up says the
  reference carries no new information.

## Plan checks

- **Reproduction:** as above.
- **Review:** one Sonnet review of this plan, one of the code, one of the write-up.
