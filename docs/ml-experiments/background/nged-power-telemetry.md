# NGED power telemetry

**The power readings are half-hourly telemetry from 32 time series in NGED's trial
area, and several of their properties change how an experiment must read them.** This page lists
the properties that have cost experiments time. The processing steps themselves are on the
[data-processing page](../data-processing.md), and the context of the network is on the [network
page](../../background/network.md).

## The series

- **There are 32 series that carry data.** They are 16 Primary substations, 6 solar farms, 3 wind
  farms, 2 grid supply points (GSPs), 2 bulk supply points (BSPs), and 3 other generators or storage
  units ([home page](../../index.md)) (disk).
- **The `TimeSeriesMetadata` table has 33 rows.** One row has no data. Count 32 series, and do not
  change 32 to 33 (disk).
- **"32" also counts something else on the network page.** The 41 connected hydro sites sit across
  "32 distinct primary substations" ([network page](../../background/network.md)). That is a count
  of Primaries across all four licence areas and has no link to the 32 trial series (unchecked).
- **The `time_series_type` column does not say what a generator is.** A type of "Other
  (Generation)" need not be solar, so do not assume a daylight profile (exp 182; the Other
  (Generation) series; provisional).
- **Label the generators "Solar farm A" to "Solar farm F" and similar outside the private data
  store.** `studies.anonymise` holds the labels. Substations may be named. The D codes (D1 to D22)
  that the write-ups use are anonymised labels of the demand-side series, defined in the
  [glossary](../glossary.md#d-codes) (code).

## Units

- **A series is metered in MW (signed active power) or in MVA (apparent power, never negative).**
  Read the unit from the `units` column. Eight of the 16 Primaries are metered only in MVA, so solar
  output behind one of them appears as demand in daylight (disk).
- **Use an MVA Primary for demand only on dark rows.** The rule, its evidence, and the row counts
  are in the [data-processing page](../data-processing.md#decide-which-rows-are-demand) (exp 103; 8
  MVA Primaries; established).
- **A negative value at an MVA series is a fault between the meter and us**, not an export
  (`PowerTimeSeries` field description) (code).
- **GSPs and BSPs are not clean demand.** The GSPs are dominated by generation, and the BSPs sum the
  trial Primaries ([data-processing page](../data-processing.md#decide-which-rows-are-demand); exp
  103; 2 GSPs, 2 BSPs; established).

## Raw and cleaned tables

- **Two Delta tables hold the power: `data/NGED/power_time_series.delta` (raw) and
  `data/NGED/cleaned_power_time_series.delta` (cleaned).** The cleaned table holds every raw row
  plus a nullable `drop_reason` column that names the cleaning rule that flagged the row (code).
- **Read power through `studies.power.scan_power`.** `scan_power` keeps only the rows with a
  null `drop_reason`. A study must not name the raw table; a test enforces this (code).
- **Read the raw table only when the question is about availability.** The raw table is what NGED
  delivered, and it holds the rows that cleaning flags (exp 182; series 32 and 33; provisional).
- **One cleaning rule exists today: `substation_zero`.** It flags a reading of exactly 0 from a
  Primary, BSP, or GSP (`nged_data.cleaning`) (code).
- **`substation_zero` drops real zeros for a switched-out transformer.** Switching experiments
  that need those rows must treat the flagged rows as kept ([zero-reading
  entry](../data-processing.md#decide-which-rows-are-demand); exp 132; switching log, few events;
  provisional).
- **For `time_series_id` 32 and 33, the cleaned and raw tables hold identical rows** (exp 182,
  `e00182_check.py`; 2 series; provisional).

## Timestamps

- **The `time` column marks the end of the half-hour that the reading covers.** Do not shift it.
  Power timestamps show no 30-minute error on either side of NGED's fix of 2026-03-26 08:30 UTC
  ([data-processing page](../data-processing.md#timestamps-and-availability); exp 103; the DP
  decision series; established).
- **Before that instant, the ingest moves NGED's stamp 30 minutes earlier**
  (`PowerTimeSeries.correct_late_timestamps`). A study must not shift the stamp a second time
  (code).
- **Join power to an hourly weather product on `(time - 1 µs).truncate("1h") + 1h`.** The
  data-processing page gives the evidence (unchecked).
- **Assume a power reading reaches the forecast 3.5 hours after the half-hour it covers.** No
  experiment has measured the delay on live telemetry (exp 87, 104, 106; assumption; provisional).

## Gaps, ramp-up, and regime changes

- **Count a series' history from its first unflagged row.** Only five Primaries hold long
  histories: D5, D10, and D11 from 2019-09, D14 from 2019-10, and D16 from 2020-12. The late
  starters begin between 2023 and 2025 (exp 103; 16 Primaries; established).
- **The start cuts and the fixed lockdown window are on the data-processing page.** See [trim the
  start of each series](../data-processing.md#trim-the-start-of-each-series) and [fixed
  windows](../data-processing.md#fixed-windows) (exp 6, 102, 103, 173; established).
- **Some series changed regime.** D8's weekday daytime demand fell from about 9 MW to about 3 MW
  between 2023-10 and 2025 (exp 103; D8; established).
- **A metered solar farm can climb to its rated output in stages over months.** One farm took eight
  months ([network page](../../background/network.md#a-new-solar-farm-reaches-full-output-in-stages-over-months)).
  The metered farms miss 7.6% to 27.6% of their half-hours, so check coverage before comparing
  farms (exp 179; 6 farms, one year; provisional).
- **A flat top in a farm's output is not proof of clipping.** A pre-registered rule found AC
  clipping at 2 of 6 farms and a post hoc rule at 4 of 6, and an export cap or panel geometry
  makes the same shape (exp 179; 6 farms; provisional).
- **The farm series behind a disaggregated Primary can be almost absent.** For D8, one farm has one
  raw reading ever, so "missing" is the baseline (exp 182; D8; provisional).
- **A metered farm's `effective_capacity_mw` may itself be estimated from output.** Power over
  capacity near 1.0 is therefore not an independent level (exp 179, 183; 6 farms; provisional).

## Capacity

- **Scores are in points of capacity, where capacity is the 99th percentile of the absolute power
  over the series' whole observed history.** One point is 1% of that value. See the
  [glossary](../glossary.md#capacity-and-points-of-capacity) (code).

Last checked against experiment 183.
