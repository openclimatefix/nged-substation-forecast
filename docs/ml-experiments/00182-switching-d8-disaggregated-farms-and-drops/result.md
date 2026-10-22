---
id: "00182"
slug: switching-d8-disaggregated-farms-and-drops
theme: switching-detectors
also_relevant_to:
  - solar-disaggregation
status: done
size: >-
  simple. Stored: private tables under data/ml-experiments/00182-switching-d8-disaggregated-farms-and-drops/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (a descriptive
  comparison). Callers: this folder's scripts only, which read the stored drops of experiment 159 and
  the power table.
hypothesis: >-
  D8 (Marsh Lane) is a disaggregated Primary: the metadata says the output of two solar farms was
  added back to its reading, and that both farms' analogues are not working. When an analogue fails,
  D8's reading loses that farm's output, so D8's multi-day drops (experiment 159) coincide with a
  stuck, zero, or missing reading at the farms, and the drop depth matches the farms' lost output.
prediction: >-
  At least 15 of the 29 drops coincide with a stuck, zero, or missing reading at one of the two
  farms, against at most 5 of 29 for random windows of the same lengths and time of day. Stop rule:
  under 8 of 29 means the farms do not explain the drops.
decision_rule: >-
  Descriptive. Count the drops with a farm anomaly (a window with under half its half-hours present,
  all zero, or a run of 6 or more identical positive readings) and compare with 1,000 random windows
  per drop, drawn from 2023-07-01 onwards at the same time of day. Also test whether the drop depth
  by hour of day follows clear-sky irradiance.
benchmark: Experiment 159 drops table (29 drops).
row_set_id: none (describes stored residuals and power, no forecast is scored)
metric: >-
  Drops with a farm anomaly out of 29, against random windows; correlation of depth by hour with
  clear-sky irradiance.
code: ml-experiments/00182-switching-d8-disaggregated-farms-and-drops/
verdict: inconclusive
best_arm: >-
  Not applicable (descriptive). The farm-anomaly count is 29 of 29 for the drops and 29.0 of 29 for
  random windows, so the readings do not separate them.
headline: >-
  The two farms' readings cannot test the hypothesis, because farm 1 has 1 reading in the power
  table and farm 2 has 4,643 (441 above zero, none after 2026-01-26), so both are "missing" in every
  drop and every random window. D8's shortfall is deeper at midday than at night in 24 of 29 drops
  (median +1.5 MW; bootstrap 95% +0.5 to +3.7), which fits a lost solar farm but also peaks in the
  evening.
lessons: >-
  The hypothesis is neither supported nor refuted. A test needs the farms' output from another
  source, such as the farms' installed capacity with clear-sky irradiance or weather estimates.
take_home_message: >-
  We cannot tell from the farms' own readings whether D8's multi-day drops are the two disaggregated
  farms dropping out, because the readings for the farms are almost entirely absent from the power
  table and so look the same inside and outside the drops. D8's shortfall is larger in daylight
  than at night in 24 of 29 drops, which is consistent with lost solar output, but it is also large
  in the evening, so the daylight pattern does not settle it. The next step is to ask NGED for the
  farms' output and capacity, or to model the farms from weather and compare the modelled output
  with the drop depth.
advice:
  - >-
    Do not use the farms behind D8's disaggregation as evidence about D8's drops, because their
    readings are absent in every drop and in every random window (Figure 2).
  - >-
    Treat a note in the metadata table's `information` field as a snapshot taken in July 2026, not
    a history, because the files from before then carry no `information` value (Checks).
  - >-
    To test whether D8's drops are lost farm output, model the farms from capacity and weather and
    compare with the drop depth by hour, because D8's shortfall is deeper at midday than at night in
    24 of 29 drops (Figure 3).
production_status: not-for-production
adopted_in: none
---

# Experiment 182: do D8's multi-day drops coincide with failed analogues of its two disaggregated farms?

**The two farms' own readings cannot answer the question, because they are absent from the power
table for the whole period of the drops.** D8 (Marsh Lane) is a disaggregated Primary substation:
the metadata table says the output of two solar farms was added back to its reading, and that the
analogue (the stand-in measurement that supplies each farm's output) of both farms is "not
working". [Experiment 159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md)
found 29 multi-day drops at D8 that no neighbour step or detector explains. The hypothesis was
that, when an analogue fails, D8's reading loses that farm's output, so a drop equals the lost
output. Farm 1 has one reading in the power table, and farm 2 has 4,643 readings, of which 441
are above zero and none come after 2026-01-26. The farm reading is therefore "missing" inside all 29
drops and inside every random window (Figure 2), so the pre-registered count (at least 15 of 29
against at most 5 for random windows) cannot be met on the random side. The verdict is
`inconclusive`.

D8's shortfall does lean towards daylight. In 24 of the 29 drops the shortfall (expected minus
metered power) is deeper from 10:00 to 15:00 local time than from 22:00 to 04:00, by a median of 1.5
MW (bootstrap 95% interval 0.5 to 3.7). The hourly extra shortfall inside drops correlates with
clear-sky irradiance at Spearman +0.44 (p = 0.033) across the 24 hours, against -0.12 outside
drops. The extra shortfall is also large at 20:00 to 22:00 (2.4 to 3.3 MW), which a solar farm
cannot explain, so the daylight pattern is a hint and not a signature.

![Readings per month for the two farms](figure_1.svg)

![Windows with a farm anomaly, drops against random windows](figure_2.svg)

![D8's extra shortfall in drops by hour, with clear-sky irradiance](figure_3.svg)

## Results

- **Farm readings:** farm 1 has 1 reading on the grid (a zero on 2024-09-26) and farm 2 has 4,643
  (2023-04-11 to 2026-01-26), 441 of them above zero and the last on 2026-01-23. Farm 2 is typed
  "Other (Generation)" in the metadata table, so it need not be solar.
- **Per-drop status in daylight (Figure 2):** farm 1 is missing in 29 of 29 drops, and farm 2 is
  missing in 28 and all-zero in 1. In the 7 days before a drop, farm 2 is missing in 28 and normal
  in 1. A farm anomaly that was not present in the previous 7 days appears in 1 of 29 drops,
  against a mean of 0.5 in 1,000 random draws (share of draws at or above 1: 0.37).
- **Farm output against drop depth:** the farms' combined daylight median over the 7 days before is
  0.0 MW in the 10 drops that have any reading there, against a median depth of 2.71 MW. The other
  19 drops have no reading to compare.
- **Random windows:** 1,000 draws of 29 windows of the same lengths and times of day, from 2023-07-01
  onwards, show a farm anomaly in 29.0 of 29 on average (5th to 95th percentile 29 to 29).
- **Depth by hour (Figure 3):** the extra shortfall inside drops is about 1.5 MW from 00:00 to
  06:00, 2.4 to 3.2 MW from 07:00 to 15:00, 0.7 MW at 17:00 to 18:00, and 2.4 to 3.3 MW from 20:00
  to 22:00. Per drop, the Spearman correlation of hourly depth with clear-sky irradiance has a median
  of +0.49 and is positive in 23 of 29 drops.
- **Metadata `information` field:** 5 of the 33 series have a value. They are D8's own note about the
  two disaggregated farms, "Analogues not working" for the two farms, "Analogues suspect" for one of
  the six metered solar farms of [experiment
  179](../00179-metered-solar-farm-exploration/result.md), and "Customer has not exported since June
  2026" for a biofuel generator. The last two are not connected to D8, so no other note explains a D8
  anomaly.
- **Raw JSON:** in the files read for D8 and the two farms (15.8 MB: the whole-history file and the
  first and last six-hour file of each series), every reading holds only `startTime`, `endTime`, and
  `value`, and the top-level fields are the ones the metadata table stores. The `Information` field
  is null in the whole-history file (to 2026-03-26) and set in the July and October 2026 files, so
  the notes are a snapshot and the files carry no history of when an analogue failed.

## Checks

- **Drop table and panel:** the drops are the 29 of experiment 159, unchanged. D8's expected power is
  the frozen convex demand model of experiment 130, so "shortfall" is relative to that model.
- **Daylight rule:** daylight is clear-sky irradiance above 50 W/m² at D8's location, and one rule
  (under half present, all zero, or 6 or more identical positive readings) is applied to both farms.
- **Triggers for audit:** none fired, because no forecast is scored.

## Caveats and next steps

- The test cannot discriminate because absence is the baseline for both farms, not because the
  hypothesis failed.
- The hourly pattern comes from 29 drops chosen by their depth in the residual, and the evening peak
  has no solar explanation.
- Next, either obtain the farms' output and capacity from NGED, or model them from capacity and
  weather estimates and compare with the drop depth by hour. A neural component is not suggested,
  because the limit is missing data.
