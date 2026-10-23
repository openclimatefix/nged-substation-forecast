---
id: "00269"
slug: pv-capacity-gain-decomposition
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00269-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one way to separate sunshine from plant
  shape, but the brief fixes the four arms. Callers: this folder's scripts, which copy experiment 238
  and call the reviewed `studies.*` modules without editing them. Asset graph: not touched.
hypothesis: >-
  Most of the 0.090 gain that experiment 238 found from using a nearby farm's recorded output as the
  solar shape comes from the farm's plant shape (tilt, orientation, flat top) and from the capacity
  unit in which recovery is read, and little comes from the farm's measured sunshine.
prediction: >-
  Written before any fit. (1) The plant shape and the unit together (the pooled empirical shape's
  recovery minus the satellite shape's, plus the unit effect) are at least 60% of the 0.090 gain.
  (2) The sunshine share (the nearby farm's recorded output minus the same farm's plant response
  driven by satellite irradiance) is under 0.04 and sits mostly in the pairs of farms under 5 km
  apart. (3) In the energy unit, the pooled empirical shape recovers 0.90 to 1.10 of the injected
  farm at 4 of the 5 decision Primaries. A sunshine share of at least 0.05 beyond 5 km would show
  that the satellite irradiance is a real limit.
decision_rule: >-
  Point estimates decide, with month-block 80% intervals beside them. Arm 4 (the pooled empirical
  shape) is kept as a replacement for the physical fleet shape if it recovers 0.90 or more in some
  capacity unit at 4 of 5 decision Primaries; stage 2 then runs. If arm 4 recovers under 0.80 in
  every unit, the learned response does not transfer between farms and stage 2 does not run. The
  threshold is 0.90 because a Primary's capacity read 10% low shifts its solar term by more than
  the width of the capacity ranges of experiment 129 (1.2 to 1.6 MW).
benchmark: >-
  Arm 1, the satellite fleet shape, on the same rows and injections as experiment 238 (pooled
  recovery 0.780), and experiment 238's nearby-farm arm (0.867).
row_set_id: >-
  Experiment 238's: experiment 129's training rows (the feature cache of experiment 106) before
  2026-07-01 at D5, D10, D11, D14, and D16, with D9 beside them, restricted to half-hours at which
  the reference farm has a reading. No forecast is scored.
metric: >-
  Recovery, in four capacity units: the ratio of 99th percentiles of alternating-current output
  over daylight half-hours (experiment 238), the capacity ratio of experiment 231, the ratio of 95th
  percentiles, and the ratio of daylight energy over clear-sky half-hours. Recovered output is the
  capacity found with the farm added at 50% of the Primary's pinned capacity minus the capacity
  found without it, times the fitted solar shape. Fits stop at 2026-07-01.
verdict: planned
production_status: not-for-production
adopted_in: none
code: ml-experiments/00269-pv-capacity-gain-decomposition/
---

# PV capacity: where does the nearby-farm gain come from?

**Plan only; the write-up replaces this page after the fits.** Experiment 238 raised the pooled
recovery of an injected real solar farm from 0.780 to 0.867 by using a nearby farm's recorded output
as the solar shape. That output holds the farm's measured sunshine and also its plant shape, and the
gain shrinks to 0.052 in experiment 231's capacity unit. This experiment splits the gain into
sunshine, plant shape, and unit.

## Arms (solar shape in the convex fit of experiment 129)

1. **Satellite fleet shape** (control, 0.780 in experiment 238).
2. **Nearby farm's recorded output** (experiment 238, 0.867).
3. **Nearby farm's plant, satellite sunshine.** The reference farm's output over irradiance and sun
   elevation is learned as a smooth monotone table, as in experiment 218, from the farm's own CAMS
   irradiance, then read at the Primary's CAMS irradiance and sun elevation. Arm 2 minus arm 3 is the
   sunshine share.
4. **Pooled empirical fleet shape.** The same learned table for each of the five farms other than
   the injected one (leave-the-injected-farm-out), averaged, read at the Primary's CAMS. Arm 4 minus
   arm 1 is the share a measured plant shape gives with no nearby-farm weather.

Every arm trains on the rows where the reference farm has a reading. Arms 1 and 2 reproduce
experiment 238.

## Checks planned

- Arm 1 on all rows reproduces experiment 238's satellite control to three decimal places.
- Distance table with and without the pairs under 5 km (the same 2 farms in both directions).
- Absolute recovery beside every contrast; plots of recovery by arm and unit, gain against distance.
