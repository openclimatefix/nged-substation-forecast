---
id: "00231"
slug: pv-capacity-audit-where-recovery-goes
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple to medium. Stored: private files under data/ml-experiments/00231-<slug>/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible, but the
  ladder of fits is fixed by the maintainer's brief. Callers: this folder's scripts, which copy
  experiments 129 and 157 and call the reviewed `studies.dp_*` modules without editing them. Asset
  graph: not touched.
hypothesis: >-
  Of the roughly 0.2 of an injected solar farm's capacity that the capacity estimator of experiment
  129 (arm 2) does not find, most is lost because demand terms absorb part of the solar signal, a
  quarter or more is lost to the irradiance input, and neither the winter rows nor the definition
  of capacity (the divisor) explains much.
prediction: >-
  Written before any fit. Taking the shortfall as 1 minus the recovery of the full joint fit
  (rung 4), pooled over the five decision Primaries: demand absorption (rung 3 to rung 4) is at
  least 40% of it; the irradiance input (rung 1 to rung 2) is at least 25% of it; the season share
  (full fit to the summer-rows-only fit) and the divisor share are each under 15%. Any other split
  is a finding and names the next experiment.
decision_rule: >-
  The prediction holds only if all four share conditions hold on the pooled median over the five
  decision Primaries and the median share falls on the stated side at 4 of the 5 Primaries. A
  share whose month-block 80% interval spans the threshold is called inconclusive for that share.
  There is no cost of porting to weigh, because the experiment changes no code in production and
  scores no forecast. The smallest share worth acting on is 15% of the shortfall, because a cause
  below that cannot recover more than 0.03 of the injected capacity.
benchmark: >-
  Experiment 129's arm 2 recovery of a real farm at 50% of the pinned capacity at season-year 2024 (rung 4 must
  reproduce it to within 0.005 at every Primary and farm)
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before 2026-07-01, at D5,
  D10, D11, D14, and D16, with D9 beside them. Rows where the injected farm has no reading drop out
  of both the base and the injected fit. No forecast is scored.
metric: >-
  Recovery: the capacity found with a real farm's output added at 50% of the Primary's pinned
  capacity, minus the capacity found without it, divided by the injected capacity, read at
  season-year 2024 (July 2024 to June 2025, the last complete season-year in experiment 129's
  known-answer table). Capacity unit: the 99th percentile of alternating-current output over daylight half-hours, in MW. Shares are fractions of the shortfall 1 minus
  rung 4
code: ml-experiments/00231-pv-capacity-audit-where-recovery-goes/
---

# PV capacity audit: where does the recovery shortfall go?

**Planned, not yet run.** The experiment adds the recorded output of a real solar farm to a Primary
substation's power, asks the capacity estimator of experiment 129 how much of the farm it finds
again, and then repeats the question under six simpler or narrower conditions to see which change
closes the gap to 1. The Primaries are D5, D10, D11, D14, and D16, with D9 beside them, and the
farms are A, B, C, D, and F, injected one at a time at 50% of the Primary's pinned capacity. Farm
labels A to F stand for metered generators and are never named. The fits stop at 2026-07-01, so the
maintainer's labelled exception (data to 2026-07-31) is not used. No forecast is scored.

## The ladder of fits

Every rung reports the recovery at season-year 2024, as a median over the five farms, with a
month-block 80% interval (a stratified bootstrap of the training months, 200 resamples, the same
draws for every farm at a Primary). The brief's "injection start months" is read as the training
months, because the injection covers the whole history.

- **Rung 1, farm only:** the least-squares scale of the farm's output on the temperature-derated
  fleet unit shape (CAMS irradiance), over all daylight half-hours of season-year 2024. No demand.
- **Rung 2, farm only on clear half-hours:** the same scale over the half-hours where the Primary's
  CAMS global irradiance is at least 0.9 of its clear-sky value and the clear-sky value is at
  least 200 W/m².
- **Rung 3, demand frozen:** the base fit (no injection) is solved first. Then the injected series
  is fitted with every column except the capacity increments held at the base fit's values.
- **Rung 4, full joint fit:** experiment 129's arm 2 (convex, ridge 0.1) fitted to the base and the
  injected series; the recovery we report.
- **Rung 5, season rows:** rung 4 refitted on April to August rows only, and on November to
  February rows only, read at the same season-year.
- **Rung 6, divisor check:** the 99th percentile of the recovered solar output (rung 4 capacity
  difference times the derated shape) over daylight half-hours, divided by the farm's own 99th
  percentile over the same half-hours, for the whole season-year and for each season (spring March
  to May, summer June to August, autumn September to November, winter December to February).
  Point values per farm; the interval is the range over farms.

The shares of the shortfall `1 - R4` are: plant shape `1 - R2`, irradiance input `R2 - R1`,
estimator noise with frozen demand `R1 - R3`, demand absorption `R3 - R4`. They sum to the
shortfall. The season share is `R5_summer - R4` and the divisor share is `|R4_p99 - R4|`, where
`R4_p99` is the whole-season-year 99th-percentile ratio of rung 6; both are alternatives to the
ladder and not terms of the sum.

## Plan checks

- **Read-out year (changed before the real run):** the first draft of this plan read the recovery at
  season-year 2025. A one-farm pilot at D10 (10 resamples, not a result) then showed that
  experiment 129's known-answer table stops at season-year 2024, so the plan reads season-year 2024
  to make rung 4 comparable with it. The pilot's numbers played no part in the choice.

- **Reproduction:** the injected fit's capacity must equal experiment 129's `kat_arm2` matched-farm
  capacity (50% injection, no cloud load, season-year 2024) to within 1e-6 relative at every Primary
  and farm. Experiment 129 subtracts one base capacity fitted on all rows from every injected fit,
  whereas this experiment fits the base on the rows where the farm has a reading, so that the two
  fits of a recovery share one row set. The page reports rung 4 with both bases, and the two
  recoveries must agree to within 0.02.
- **Wiring:** the monthly Gram matrices must sum to the Gram matrix of `accumulate_gram` to within
  1e-8 relative, and the rung-4 base capacity must equal `fit_gam`'s.
