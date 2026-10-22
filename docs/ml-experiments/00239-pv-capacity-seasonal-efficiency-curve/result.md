---
id: "00239"
slug: pv-capacity-seasonal-efficiency-curve
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00239-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible form of the curve (12
  monthly factors, a smooth harmonic, a temperature term), but the brief fixes 12 monthly factors.
  Callers: this folder's scripts, which copy experiment 231 and call the reviewed `studies.dp_*`
  modules without editing them. Asset graph: not touched.
hypothesis: >-
  The metered solar farms' output falls 6% to 9% from spring to summer by a pattern that no clip or
  heat term explains. A seasonal efficiency curve of 12 monthly factors, shared by all farms,
  fitted on the farms' own output, and then applied to the solar shape in a Primary substation's
  capacity fit, lets that fit find more of an injected real solar farm.
prediction: >-
  Written before any fit. With each farm's factors fitted on the other farms only, the curve
  raises the share of a farm's squared output that one scale of the satellite-based shape explains
  from 0.953 to at least 0.968, which removes a third of the unexplained share, and it raises the
  recovery of an injected farm by at least 0.03 at 4 of the 5 decision Primaries. A flat curve (all
  12 factors within 3% of each other) would mean that the fall of the flat top is a clip or a
  curtailment and not an efficiency effect. Experiment 231 raises the prior: its flat-top divisor
  check found that the seasonal flat top moves the divisor, and that the plant shape explains 47% of
  the shortfall.
decision_rule: >-
  The curve is kept for further work if the farm-only fit reaches 0.968 and the recovery rises by
  at least 0.03 at 4 of 5 decision Primaries, with point estimates deciding and the month-block
  intervals reported beside them. The threshold for the recovery is 0.03, below the 0.05 of
  experiments 236 and 238, because the curve costs no new data: 12 numbers fitted on farms already
  metered. A smaller rise, or a flat curve, drops the idea. The estimate is offline only.
benchmark: >-
  The satellite shape without the curve, on the same rows and injections (experiment 231's rung 4,
  pooled recovery 0.827 with 80% interval 0.814 to 0.842 at season-year 2024, in the capacity unit
  of that experiment)
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before 2026-07-01, at D5,
  D10, D11, D14, and D16, with D9 beside them. No forecast is scored.
metric: >-
  Recovery: the 99th percentile of the recovered solar output over daylight half-hours of
  season-year 2024 (July 2024 to June 2025), divided by the 99th percentile of the injected farm's
  output over the same half-hours, both of alternating-current output in MW. The plain capacity
  ratio, in the unit of experiment 231, is reported too. Farm-only fit: the share of a farm's
  squared output that one scale of the shape explains
code: ml-experiments/00239-pv-capacity-seasonal-efficiency-curve/
---

# PV capacity: does a shared seasonal efficiency curve help?

**Planned, not yet run.** The experiment repeats the injection test of experiment 231 (the recorded
output of a real solar farm is added to a Primary substation's power at 50% of the Primary's pinned
capacity, and the convex capacity fit of experiment 129 reports how much of the farm it finds
again; farms A to F are metered generators and are never named). The change is a multiplier on the
solar shape for each calendar month. Fits stop at 2026-07-01, so the maintainer's labelled
exception (data to 2026-07-31) is not used. No forecast is scored.

## Plan: design

- **The curve.** For each farm and calendar month, the least-squares scale of the farm's output
  (divided by its 99th percentile) on the temperature-derated satellite shape at the farm's own
  site, over daylight half-hours (sun below 75 degrees of zenith) before 2026-07-01. The farms are
  A, B, C, D, and F, the farms with data from 2019-09. The curve is the mean over the farms of the
  monthly scales, divided by the mean of its own 12 values so that the curve has mean 1.
- **No leakage from the injected farm.** For an injected farm X, the curve uses the other four
  farms only. The farm-only fit of X also uses that curve.
- **Applying the curve.** The Primary's derated shape is multiplied by the factor of the
  half-hour's calendar month. The temperature derating is then divided out again (the fixed-point
  step of experiment 238), so that the fit does not apply the derating a second time.
- **Arms.** The satellite shape without the curve, and the satellite shape with the curve, on the
  same rows and the same injections at D5, D9, D10, D11, D14, and D16.
- **Intervals.** A month-block bootstrap, 200 resamples, stratified as in experiment 231, over the
  fits of the Primary's injected farms; the curve is held fixed in each resample, which understates
  the uncertainty of the curve. The spread of the factors over the farms is reported beside it.
- **Reading a flat curve.** If every factor lies within 3% of the others, the fall of the flat top
  is not a seasonal efficiency effect, and the write-up says so.

## Plan: checks

- **Reproduction:** the arm without the curve reproduces experiment 231's rung 4 to within 1e-6,
  and a curve of 12 factors equal to 1 reproduces the arm without the curve.
- **Derating:** the shape returned for the fit, derated by the fit's own formula, equals the
  satellite shape times the factor to within 1e-6.
- **Review:** one Sonnet review of this plan, one of the code, one of the write-up.
