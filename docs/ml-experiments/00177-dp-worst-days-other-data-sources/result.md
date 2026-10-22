---
id: '00177'
slug: dp-worst-days-other-data-sources
part: BG-EB4
title: >-
  Can data we already hold, other than power and weather, explain the 37 unlabelled worst days of
  the best-guess forecast?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- switching-labels-in-the-forecast
- production-reference-and-best-guess
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00177-dp-worst-days-other-data-sources/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (the window around a day, what counts as a logged operation), so this plan fixes one
  of each and reports alternatives as labelled sensitivities. Callers: this folder's scripts, which
  read experiment 175's day tables, NGED's switching log, the power Delta table, and the metadata
  table, and call no modelling code. No forecast model is fitted.
hypothesis: >-
  At least a quarter of the 37 unlabelled worst substation-days of experiments 172, 174, and 175
  coincide with an event recorded in data the project already holds: a logged switching operation
  near the substation, a step in a metered generator on the same or an interconnected Primary, a
  reading-quality fault in the substation's own series, or a repeat of the substation's own bad
  weekday or date.
prediction: >-
  Written before any result is read. The raw switching log, which holds many events too small for
  the 66 hand-filtered transfers, covers 8 to 15 of the 37 days, against 3 to 7 of 37 matched
  ordinary days. Metered generators cover 0 to 4 days, since the trial area has 13 metered
  generators. Reading-quality faults cover 0 to 3 days. Recurrence of the weekday or date covers 0
  to 3 days.
decision_rule: >-
  Per source, a day is explained when the source's indicator, fixed below, is true for that
  substation and local date. A source explains a day beyond chance when its rate over the 37 days
  exceeds the mean rate over 1,000 draws of 37 matched ordinary days (same substation counts, same
  calendar months, drawn from experiment 175's ordinary days) by at least 15 points, and the
  one-sided p-value from the draws is under 0.05. The experiment is `supported` when the union of
  sources that pass explains at least 10 of the 37 days (a quarter), `partly supported` when a
  source passes but the union explains fewer than 10, `refuted` when no source passes. A source
  readable only after the fact is reported as such. Threshold reason: a finding is worth a
  follow-up experiment only if it explains a quarter of the worst days, which carry about 42% of
  the excess loss.
benchmark: >-
  37 matched ordinary days per draw, 1,000 draws, scored with the same indicators. The matched
  ordinary-day pool is experiment 175's `ordinary_day_shapes.parquet` (200 days).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Count of the 37 days each source explains, and the same count on matched ordinary days. The days
  are experiment 175's `day_shapes.parquet`: experiment 155's arm D, day-ahead band, 2025-07-03 to
  2026-06-30.
code: ml-experiments/00177-dp-worst-days-other-data-sources/
date: 2026-10-22
---

**This page is the pre-registration. No day has been compared with any source yet.**

## Sources and indicators, fixed before any result is read

- **Raw NGED telemetry fields.** The parser keeps only `endTime` and `value` from each file's
  readings, and the stored Delta table has only `power`, `time`, and `time_series_id`. The raw
  JSON files sit on S3, not on disk. Stage 1 does not read them. If the maintainer wants the
  check, the question is whether an unparsed field (voltage, current, power factor, tap position)
  exists in the raw files, and it needs one read of a few files from S3.
- **S1, raw switching log** (`Switching Logs.xlsx`, one sheet per time-series pair, "Unknown" for
  Primaries outside the trial area). Indicator: the substation's own sheet or a sheet of an
  interconnected Primary holds a row with state `Confirmed` or `Field Confirmed` and action
  `Telecontrol OPEN`, `Telecontrol CLOSE`, `Open from Earth`, or `Confirm Opened` whose `Manual
  Time` falls on the local date or the evening before. Sensitivity: any state; a window of plus
  or minus 1 day.
- **S2, metered generators.** Indicator: a metered generator with the same `substation_number`
  as the Primary, or on an interconnected Primary, has a step in its half-hourly power on the
  local date: a jump of more than 3 times the spread of its day-before changes, lasting at least
  4 hours. Sensitivity: any generator in the trial area.
- **S3, reading quality of the substation's own series.** Indicator: the local date holds at
  least 6 missing half-hours, or a run of at least 8 identical consecutive readings, or a
  reading-count shortfall against the substation's usual 48.
- **S4, own history.** Indicator: the same substation has at least two other worst-5% days on the
  same weekday within 8 weeks, or within plus or minus 3 days of the same date a year earlier.
  Only dates before the day count for the second branch.
- **Deferred to stage 2, needing a download or a decision from the ML MANAGER:** Elexon and
  NESO public data (demand turn-up, balancing actions, Demand Flexibility Service events),
  Distribution Future Energy Scenarios, published outage lists, school terms, and local events.
  Local convective cloud, fog, snow, and
  wind-chill are weather that the ENS mean misses, and experiment 174 already put an upper bound
  on that share.

## Checks to print

- The count and rate of each indicator on the 37 days, on each ordinary-day draw (mean and 95%
  range), and the union.
- Each indicator's coverage of the whole year, so that an indicator true on most days (the log
  may cover only part of the year) is reported as uninformative.
