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
status: done
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
verdict: refuted
best_arm: >-
  Not applicable (descriptive). No source explains the 37 worst days beyond chance. The best is a
  high system price on 18 of 37 days, against 14.9 on matched ordinary days (95% range 11 to 18),
  which is 3 days more, under the 15-point bar.
headline: >-
  The union of logged switching, reading quality, and the substation's own bad weekdays covers 5 of
  37 unlabelled worst days, against 5.4 on matched ordinary days. Demand Flexibility Service events
  cover 9 (ordinary 8.5), and a high national system price covers 18 (ordinary 14.9, p = 0.075). The
  raw NGED files carry no field beyond time and power.
lessons:
  - >-
    The worst unlabelled days are not explained by the logged switching, reading faults, a
    recurring bad weekday, Demand Flexibility Service events, or national price spikes.
  - >-
    The raw NGED telemetry holds only time and power per reading, so no voltage, current, or tap
    position exists to read.
  - >-
    The metered generators sit on their own substation numbers, so none maps to a Primary or its
    neighbours and a generator-step source cannot be tested without a location join.
take_home_message: >-
  The 37 worst days of the best-guess forecast that carry no label are not explained by any data we
  hold or can read free of charge: logged switching, reading faults, repeat bad weekdays, Demand
  Flexibility Service events, and national price spikes all cover them no more often than they cover
  ordinary days. The raw NGED files hold only time and power, so no extra telemetry exists. The
  remaining candidates are local: planned outage notices, local events, and a location join to the
  metered generators.
advice:
  - >-
    Do not look for the unlabelled worst days in the raw NGED files: each reading holds only
    `startTime`, `endTime`, and `value` (check of three files, one per decision Primary).
  - >-
    Do not expect national Elexon system prices or Demand Flexibility Service events to explain
    single-substation worst days: both cover them as often as ordinary days (figure 1).
  - >-
    Join metered generators to Primaries by location before testing generator steps as a cause: the
    generators sit on their own substation numbers (stage 1, source S2).
production_status: not-for-production
adopted_in: none
---

**The 37 worst unlabelled days are not explained by any data we hold or can read free of charge.**
[Experiment 172](../00172-dp-error-budget-worst-days/result.md) found the 37 worst days of five
Primary substations that carry no label.
[Experiment 174](../00174-dp-unlabelled-worst-days-weather-shared-or-single-site/result.md) found
that better weather explains under a third of their loss, and
[experiment 175](../00175-dp-unlabelled-day-shapes/result.md) found that about half are sharp-edged
blocks. We asked whether data we have not yet used marks those days: NGED's logged switching
operations, faults in the substation's own readings, its own bad weekdays, and public Demand
Flexibility Service (DFS) events and national system prices from NESO and Elexon. For each source we
counted the days it covers among the 37 and among 1,000 random sets of 37 ordinary days from the
same substations and months. No source covers the 37 days more often than ordinary days by the
15 points the plan set as the bar.

**The verdict is `refuted`.** The three sources readable on disk cover 5 of 37 days together,
against 5.4 on ordinary days. DFS events cover 9 (ordinary 8.5). A high national system price covers
18 (ordinary 14.9, one-sided p = 0.075), the closest of any source and still 3 days, not 5.5.

![Days explained by each source: the 37 worst days against 37 matched ordinary days](fig1_sources.svg)

| Source | 37 worst days | Ordinary days (mean, 95% range) | Share of all days the rule covers |
|---|---|---|---|
| Logged switching, confirmed (S1) | 0 | 0.3 (0 to 1) | 0.7% |
| Reading quality (S3) | 0 | 0.0 (0 to 0) | 0.4% |
| Own bad weekday (S4) | 5 | 5.1 (2 to 8) | 12.4% |
| Union of stage 1 | 5 | 5.4 (2 to 9) | 13.4% |
| DFS event (P1) | 9 | 8.5 (6 to 11) | 25.9% |
| High system price, top 5% (P2) | 18 | 14.9 (11 to 18) | 39.9% |
| Metered generator step (S2) | not testable | not testable | 0% |

- **Raw NGED files carry no extra field.** Three files, one for each of three decision Primaries,
  hold only `startTime`, `endTime`, and `value` in each reading, plus the top-level fields the
  metadata table already stores. No voltage, current, power factor, or tap position exists.
- **S2 could not be tested.** The 13 metered generators have their own substation numbers, so none
  shares a number with a Primary or an interconnected Primary. A step test needs a location join
  that this experiment did not build.
- **Looser versions of the rules are uninformative.** Switching-log rows of any state cover 15 of
  37 days (ordinary 11.2, p = 0.035) but the rule covers 39.5% of all days. A step in any
  generator in the area covers 26 of 37 (ordinary 20.5) and covers 62.6% of all days.
- **The switching log ends on 2026-03-25,** and the DFS archive on 2026-03-27, so about 3 of the 12
  scored months, and 14 of the 37 days for DFS, have no record to test.
- **The ordinary-day draws match by substation and month,** but one of the 37 days has no ordinary
  day of the same month at its substation, so each draw falls back to the substation's other months
  for that day.

## Checks

- **Median forecast minus mean forecast, and the error of the mean or median forecast:** not
  applicable, because the experiment fits no model and scores no forecast.
- **Audit triggers:** none fired. No control, quantile-score contrast, nested arm, or interval
  comparison exists to trigger one. Trigger 7 does not apply, because no later experiment's design
  depends on this result's sign.
- **Public data checks:** 365 days of system prices (17,520 settlement periods, 363 days of 48 plus
  one of 50 and one of 46 at the clock changes) with no duplicate keys, no nulls, prices from
  -97.92 to 800.00 pounds per megawatt-hour, and a monthly mean from 64.6 to 103.2.

## What the next experiment should answer

**The remaining candidates are local, and each needs a source we do not hold yet.** The sources
still open are planned outage notices from NGED, local events and school terms, and a location join
of the metered generators to the Primaries so that a step in a nearby generator can be tested. The
shape result of experiment 175 points the same way: half the days are blocks at one substation that
no neighbour mirrors. A source of the load itself, such as a customer-level or feeder-level record
from NGED, would test the hypothesis directly. Ask NGED whether an outage list exists for these
Primaries.

## Pre-registration

**This page began as the pre-registration. The sections below were written before any result was read.**

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

## Stage 2 addendum, fixed before any public data was read

**Stage 1 ran first and found no source that passes.** The raw NGED file check is done: three
files, one per decision Primary, carry only `startTime`, `endTime`, and `value` in each reading,
and the same top-level fields the metadata table already holds. So no unparsed telemetry field
exists, and that source is closed. Stage 2 reads public data under the same decision rule and the
same 1,000 matched ordinary-day draws.

- **P1, Demand Flexibility Service (DFS) events.** Source: the NESO DFS service-requirement
  archive for 2025/2026. Indicator: a live or test DFS event is scheduled on the local date.
- **P2, system price spikes.** Source: Elexon settlement system prices for each day. Indicator:
  the highest system sell price or system buy price in any settlement period of the local date
  lies in the top 5% of all settlement periods in 2025-07-03 to 2026-06-30.
- **Reading the result.** A national indicator can only explain a single-substation block through
  a mechanism, such as a demand-turn-down event moving load. A rate near the ordinary-day rate is
  reported as no explanation, however many days the indicator covers.
