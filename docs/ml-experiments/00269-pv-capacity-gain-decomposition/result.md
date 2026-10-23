---
id: "00269"
slug: pv-capacity-gain-decomposition
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
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
  that the satellite irradiance is a real limit. Stage 2: the capacities move by under 0.3 MW at 4
  of 5 Primaries.
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
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-23
headline: >-
  Pooled recovery of an injected farm, ratio of 99th percentiles: 0.780 with the satellite fleet
  shape, 0.867 with the nearby farm's output (gain 0.090, 80% interval 0.074 to 0.100), 0.886 with
  the same farm's plant response driven by satellite sunshine, and 0.892 with the mean plant
  response of the five other farms (gain 0.105, 0.095 to 0.113). The sunshine share is -0.014
  (-0.023 to -0.002).
best_arm: >-
  pooled empirical shape, pooled recovery 0.892 (ratio of 99th percentiles), 0.966 (capacity ratio)
take_home_message: >-
  The nearby farm's measured sunshine adds nothing to finding an injected solar farm: the whole gain
  of experiment 238 comes from the farms' plant shape (tilt, orientation, flat top), which a table
  learned from other farms' output and driven by the satellite irradiance reproduces. Using that
  pooled shape finds 0.966 of the injected farm in the capacity unit of experiment 231 (0.826 with
  the physical fleet shape) and 0.995 in the clear-sky energy unit, so the satellite irradiance is
  not the limit and no metered farm is needed beside each substation. Switching to the pooled shape
  lowers the fitted capacities of the five decision substations by 0.2 to 0.4 MW, and the effect on
  the sunny-half-hour forecast score has not been run.
lessons:
  - >-
    The satellite irradiance is not the limit on finding an injected farm. Driving a farm's learned
    plant response with the satellite irradiance recovers as much as the farm's own recorded output
    (0.886 against 0.867), so the sunshine share is -0.014.
  - >-
    The ratio of 99th percentiles understates what the pooled empirical shape finds: it recovers
    0.892 in that unit, 0.905 at the 95th percentile, 0.966 as a capacity ratio, and 0.995 in
    clear-sky daylight energy. The nearby farm's recorded output scores worse than the satellite
    fleet shape in the energy unit (0.859 against 0.917).
  - >-
    The 5 pairs of farms under 5 km apart (the same 2 farms in both directions) hold the only
    sunshine share above zero (median 0.075), and farms 5 km or more apart show -0.010 to -0.029.
advice:
  - >-
    Use the mean learned plant response of several metered farms, read at the substation's satellite
    irradiance, in place of the physical fleet shape when a solar capacity is fitted (figures 1 and
    2).
  - >-
    Do not add a nearby metered farm's recorded output as a sunshine reference for the capacity
    fit, because its sunshine share is -0.014 and beyond 5 km it is not above zero (figure 3).
  - >-
    State the capacity unit beside every recovery figure and read a shape comparison in the capacity
    ratio and the energy unit as well as the ratio of 99th percentiles, because the unit moves a
    pooled recovery by up to 0.1 (figure 1).
code: ml-experiments/00269-pv-capacity-gain-decomposition/
---

# PV capacity: where does the nearby-farm gain come from?

**Take-home message: the nearby farm's measured sunshine adds nothing, and the whole gain of
experiment 238 comes from the farm's plant shape.** We asked why a nearby solar farm's recorded
output helps a fitting method find the output of a second solar farm that we added to a
substation's demand. The recorded output carries the farm's sunshine and also its plant shape (the
tilt, the orientation, and the flat top of the output curve). We replaced the recorded output by a
smooth table, learned from the farms' output against the satellite irradiance and the height of the
sun, and read the table at the substation's own satellite irradiance. The table found as much as the
recorded output did, so the sunshine share is zero and the plant shape carries the gain. The method
is the convex fit of [experiment
129](../00129-differentiable-physics-pv-capacity-known-answer-test/result.md), the injections are
those of [experiment 238](../00238-pv-capacity-nearest-farm-sunshine-reference/result.md), and the
learned table follows [experiment 218](../00218-metered-solar-farm-plant-fit/result.md).

**Verdict: partly supported.** Every stage 1 prediction was met. The stage 2 prediction that the
real capacities move by under 0.3 MW at 4 of 5 substations was not met (3 of 5), and the
sunny-half-hour score of the forecast was not run (see "What was not run"). Farms A to F are
metered generators and stay anonymous. Recovery is the share of the injected farm's output that the
fit finds again, in the ratio of 99th percentiles of alternating-current output over daylight
half-hours unless a unit is named. Fits stop at 2026-07-01, and no forecast is scored.

![Recovery by arm and capacity unit](figure_01_pooled.svg)

**Figure 1. Pooled over the five decision Primaries, the learned plant shape (arms 3 and 4) finds
more of the injected farm than the nearby farm's output does, in all four units.**

![Recovery of the pooled empirical shape](figure_02_pooled_shape.svg)

**Figure 2. The pooled empirical shape finds 0.90 or more at 5 of 5 Primaries in the capacity ratio
and the energy unit, and at 2 of 5 in the ratio of 99th percentiles.**

![Sunshine share against distance](figure_03_distance.svg)

**Figure 3. The sunshine share is above zero only for the pairs of farms under 5 km apart.**

![Capacities of the real series under the two shapes](figure_04_real_capacities.svg)

**Figure 4. The pooled empirical shape lowers each decision Primary's fitted capacity by 0.2 to 0.4
MW, and every value stays inside the 80% range of experiment 129.**

## Arms

- **Arm 1, satellite fleet shape** (control): the south-facing physical shape of experiment 129.
- **Arm 2, nearby farm's recorded output:** the nearest other farm's output, as in experiment 238.
- **Arm 3, nearby farm's plant, satellite sunshine:** that farm's learned table (non-negative, smooth,
  rising with irradiance, as in experiment 218), read at the Primary's satellite irradiance and sun
  height. Arm 2 minus arm 3 is the sunshine share.
- **Arm 4, pooled empirical shape:** the mean of the learned tables of the five farms other than the
  injected one, read at the Primary's satellite irradiance and sun height.
- **Rows:** every arm trains on the half-hours at which the reference farm has a reading (75,107 to
  85,231 per Primary, 45,017 at D9).

## Results

- **Pooled recovery, ratio of 99th percentiles (median over farms, then over the five Primaries; 80%
  interval from resampling whole months):** arm 1 0.780 (0.765 to 0.790), arm 2 0.867 (0.863 to
  0.878), arm 3 0.886 (0.874 to 0.893), arm 4 0.892 (0.874 to 0.897).
- **Contrasts:** total gain 0.090 (0.074 to 0.100); sunshine share -0.014 (-0.023 to -0.002); plant
  shape of the nearby farm 0.095 (0.083 to 0.103); pooled shape over the satellite fleet shape 0.105
  (0.095 to 0.113). The plant shape explains 115% of the gain, against the predicted 60%.
- **Other units, pooled recovery for arms 1, 2, 3, and 4:** capacity ratio of experiment 231 0.826,
  0.869, 0.925, and 0.966 (arm 4: 0.944 to 0.973); ratio of 95th percentiles 0.809, 0.869, 0.890,
  and 0.905; clear-sky daylight energy 0.917, 0.859, 1.003, and 0.995 (arm 4: 0.978 to 1.004). The
  sunshine share is -0.060, -0.012, and -0.122 in those three units.
- **Arm 4 at the decision Primaries:** at least 0.90 at 5 of 5 in the capacity ratio (0.916 to 0.979)
  and in the energy unit (0.960 to 1.009), at 3 of 5 at the 95th percentile, and at 2 of 5 in the
  ratio of 99th percentiles (0.862 to 0.908). Arm 4 is under 0.80 nowhere, so stage 2 ran. D9 reads
  0.813 to 0.920.
- **By distance between the injected and reference farms (ratio of 99th percentiles, median over
  pairs):** the 5 pairs under 5 km apart gain 0.198 in total, of which 0.075 is sunshine (range
  0.040 to 0.109); the 8 pairs 5 to 12 km apart gain 0.069, sunshine -0.010; the 16 pairs 12 to 30 km
  apart gain 0.073, sunshine -0.029. One pair at 10 km reaches a sunshine share of 0.054 (figure 3).
- **Without the pairs under 5 km,** the pooled recovery for arms 1 to 4 is 0.773, 0.866, 0.884, and
  0.879 in the ratio of 99th percentiles, and arm 4 is 0.966 in the capacity ratio and 0.992 in the
  energy unit.
- **Stage 2, real capacities (season-year 2025, July 2025 to June 2026).** Each capacity is divided
  by the recovery of its own shape (the capacity ratio of this experiment's injections at
  season-year 2024). The pooled empirical shape moves the capacity by -0.24, -0.29, -0.43, -0.30,
  and -0.23 MW at D5, D10, D11, D14, and D16 (under 0.3 MW at 3 of 5; the prediction asked for 4),
  and by -0.22 MW at D9. All five lie inside experiment 129's 80% range (1.2 to 1.6 MW wide). The
  physical-shape capacities match experiment 129's published estimates to within 0.09 MW at
  season-year 2024.

## What was not run

- **The sunny-daylight score of arm D with the empirical shape was not run.** It needs the empirical
  shape in the ensemble weather path of `studies` (a reviewed change, because that path builds the
  shape from the plant parameters), and monthly refits and ensemble forecasts at six substations
  for the scored year. That is far above the slot's 4 core-hours. The first evidence for the
  maintainer's summer-underperformance diagnosis therefore stays open.
- **Bootstrap ranges for the stage 2 capacities** were not refitted; the table uses experiment 129's
  ranges as context.

## Checks

- **Reproduction:** arms 1 and 2 reproduce experiment 238's tables, and the satellite shape on all
  rows reproduces experiment 231's rung 4, with a largest difference of 0.000 at three decimal
  places.
- **Median forecast minus mean forecast, and the mean-forecast error:** not applicable, because no
  forecast is scored.
- **Audit triggers:** none fired. The result agrees with experiment 231 (the irradiance explains
  almost none of the shortfall), the injections are those of experiment 238 and give the same
  interval for the total gain, and no nested arm beats the arm that contains it. This result answers
  the question behind triggers 2 and 7 of experiment 238. One Sonnet review of the code and one of
  the page were run; no Opus audit.

## Caveats

- **The pooled shape includes plants like the injected one.** Farm C sits 2 km from farm A, so the
  mean over the five other farms holds a close twin of each injected farm in some cases. The sunshine
  share needs no twin, but the transfer of arm 4 to a farm unlike all six metered farms is untested.
- **Recovery depends on the unit.** The same fit reads 0.892 in the ratio of 99th percentiles and
  0.995 in clear-sky energy; the decision rule's "some unit" is satisfied by two of four units.
- **Operational choices made after the plan:** the plant-and-unit share is read as 1 minus the
  sunshine share over the total gain, in the ratio of 99th percentiles; the stage 2 divisors come
  from this experiment's injections and not from the known-answer grid of experiment 129; stage 2
  fits stop at 2026-07-01, not 2026-08-01.

## What the next experiment should answer

- **Does the empirical shape lower the sunny-half-hour error of the forecast?** Port the pooled
  table into the ensemble weather path in `packages/studies` and score arm D at D5, D9, D10, D11,
  D14, and D16 on the sunny-daylight rows of experiment 178. The prediction that could fail is a
  change of under 0.02 MW at the five other substations.
- **Is the plant shape of an unseen farm predictable?** Leave each farm out of the pool and also
  leave out its nearest neighbour, then repeat arm 4.
