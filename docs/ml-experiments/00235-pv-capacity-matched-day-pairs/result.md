---
id: "00235"
slug: pv-capacity-matched-day-pairs
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple to medium. Stored: private files under data/ml-experiments/00235-<slug>/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (how days
  are matched), but the matching rule is fixed by the brief. Callers: this folder's scripts, which
  copy experiments 129 and 231 and call the reviewed `studies.dp_*` modules without editing them.
  Asset graph: not touched.
hypothesis: >-
  The capacity of unmetered solar behind a Primary substation can be read from the difference in
  net power between a sunny day and a cloudy day of the same day type and the same temperature,
  because ordinary demand cancels in the difference. That estimator should find more of an injected
  real solar farm than the convex fit of experiment 129 (arm 2), which finds 0.78 to 0.88.
prediction: >-
  Written before any fit. On the injection test the matched-pair estimator recovers 0.90 to 1.10
  of the injected farm at 4 of the 5 decision Primaries, at the cost of fewer usable rows and a
  wider interval. A placebo on pairs from a sun-blind winter month recovers under 0.15. With a
  cloud-driven load of 10% of the injected capacity added (as experiment 129), the recovery stays
  within 0.10 of the recovery without the load, against a rise of about 0.2 for the convex fit.
decision_rule: >-
  The estimator is kept for a later experiment if the recovery without load is 0.90 to 1.10 at 4 of
  5 Primaries and the placebo recovers under 0.15. It is dropped if the recovery without load is
  under 0.88 at 3 or more Primaries. The cloud-load contrast is a second pre-registered contrast:
  recovery with the load within 0.10 of recovery without it at 4 of 5 Primaries is a gain over the
  convex fit; otherwise demand that follows cloud is not cancelled by pairs. No cost of porting is
  weighed, because the experiment scores no forecast and changes no production code.
benchmark: >-
  Experiment 231's rung 4 (experiment 129's arm 2) on the same injections: pooled recovery 0.827
  (80% interval 0.814 to 0.842) at season-year 2024
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before 2026-07-01, at D5,
  D10, D11, D14, and D16, with D9 beside them. No forecast is scored.
metric: >-
  Recovery: the capacity found with a real farm's output added at 50% of the Primary's pinned
  capacity, minus the capacity found without it, divided by the injected capacity, from pairs of
  days in season-year 2024 (July 2024 to June 2025). Capacity unit: the 99th percentile of
  alternating-current output over daylight half-hours, in MW
code: ml-experiments/00235-pv-capacity-matched-day-pairs/
---

# PV capacity: does matching sunny and cloudy days remove the demand?

**Planned, not yet run.** The experiment asks whether the capacity of the unmetered solar behind a
Primary substation can be read from pairs of days that have the same day type (weekday, Saturday,
or Sunday) and a mean daylight temperature within 1 K of each other, where one day is sunny and the
other cloudy. The estimator is a regression of the difference in net power between the two days, at
each daylight half-hour, on the difference of the solar shape, with no demand terms. The experiment
adds the recorded output of a real solar farm (farms A to F stand for metered generators, and are
never named) to the Primary's power and measures how much of it the estimator finds again. Fits stop
at 2026-07-01, so the maintainer's labelled exception (data to 2026-07-31) is not used. No forecast
is scored.

## Design

- **Pairs.** Within season-year 2024, a day is sunny when its daylight mean of the clear-sky index
  (CAMS global irradiance over clear-sky irradiance at the Primary) is at least 0.4 above the other
  day's. A pair needs the same day type, a mean daylight temperature within 1 K, and the two days
  within 21 days of each other. Each day takes at most three partners, the ones with the smallest
  temperature gap. Holiday periods are excluded.
- **Estimator.** The least-squares scale through the origin of the pair difference in net power on
  the pair difference in the temperature-derated unit shape of the fit in experiment 129, over the
  daylight half-hours at which both days have a training row. The recovery is the scale found with
  the farm's output added, minus the scale found without it, divided by the injected capacity.
- **Linearity caveat.** The estimator is linear in the power, so the difference of its two scales is
  the scale of the farm's own output difference on the shape difference. A farm injection alone
  therefore cannot show demand absorption, and the recovery without load is a property of the
  estimator's shape and weather error (experiment 231's rung 1 reads 0.93). The informative tests
  are the cloud-driven load of experiment 129 (added to the net power, scaled to 10% of the injected
  capacity) and the placebo.
- **Placebo.** The same pairing and estimator on December and January days only, where the sun is
  low, reported as the share of the injected capacity found. The injected farm still produces a
  little output in those months, so the placebo's expected value is small and not zero.
- **Intervals.** A month-block bootstrap of the pairs by the month of the sunny day, 200 resamples,
  stratified as in experiment 231, and the range over the five farms.

## Plan checks

- **Pair count:** report the number of pairs and the usable half-hours per Primary. A Primary with
  fewer than 30 pairs is reported and left out of the "4 of 5" count.
- **Toy check:** the script runs its pairing and scale on a synthetic series with a known capacity
  and no demand, and must return that capacity to within 1e-6 before any real fit runs.
