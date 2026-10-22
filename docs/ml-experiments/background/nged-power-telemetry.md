# NGED power telemetry

**The power readings are half-hourly telemetry from 32 time series in NGED's East Midlands licence
area, and several of their properties change how an experiment must read them.** This page lists
the properties that have cost experiments time. The processing steps themselves are on the
[data-processing page](../data-processing.md), and the context of the network is on the [network
page](../../background/network.md).

## The series

- **There are 32 series that carry data.** They are 16 Primary substations, 6 solar farms, 3 wind
  farms, 2 grid supply points (GSPs), 2 bulk supply points (BSPs), and 3 other generators or storage
  units ([home page](../../index.md)).
- **The `TimeSeriesMetadata` table has 33 rows.** One row has no data. Count 32 series, and do not
  change 32 to 33.
- **"32" also counts something else on the network page.** The 41 connected hydro sites sit across
  "32 distinct primary substations" ([network page](../../background/network.md)). That is a count
  of Primaries across all four licence areas and has no link to the 32 trial series.
- **The `time_series_type` column does not say what a generator is.** A type of "Other
  (Generation)" need not be solar, so do not assume a daylight profile (experiment 182).
- **Label the generators "Solar farm A" to "Solar farm F" and similar outside the private data
  store.** `studies.anonymise` holds the labels. Substations may be named. The D codes (D1 to D22)
  that the write-ups use are anonymised labels of the demand-side series, defined in the
  [glossary](../glossary.md#d-codes).

## Units

- **A series is metered in MW (signed active power) or in MVA (apparent power, never negative).**
  Read the unit from the `units` column. Eight of the 16 Primaries are metered only in MVA, so solar
  output behind one of them appears as demand in daylight.
- **Use an MVA Primary for demand only on dark rows.** The rule, its evidence, and the row counts
  are in the [data-processing page](../data-processing.md#decide-which-rows-are-demand) (experiment
  103).
- **A negative value at an MVA series is a fault between the meter and us**, not an export
  (`PowerTimeSeries` field description).
- **GSPs and BSPs are not clean demand.** The GSPs are dominated by generation, and the BSPs sum the
  trial Primaries ([data-processing page](../data-processing.md#decide-which-rows-are-demand),
  experiment 103).

## Raw and cleaned tables

- **Two Delta tables hold the power: `data/NGED/power_time_series.delta` (raw) and
  `data/NGED/cleaned_power_time_series.delta` (cleaned).** The cleaned table holds every raw row
  plus a nullable `drop_reason` column that names the cleaning rule that flagged the row.
- **Read power through `studies.power.scan_power`.** It keeps only the rows with a null
  `drop_reason`. A study must not name the raw table; a test enforces this.
- **Read the raw table only when the question is about availability.** The raw table is what NGED
  delivered, and it holds the rows that cleaning flags (experiment 182).
- **One cleaning rule exists today: `substation_zero`.** It flags a reading of exactly 0 from a
  Primary, BSP, or GSP (`nged_data.cleaning`).
- **`substation_zero` drops real zeros for a switched-out transformer.** Switching experiments
  that need those rows must treat the flagged rows as kept ([switching
  entry](../data-processing.md#decide-which-rows-are-demand), experiment 132; provisional).
- **For series 32 and 33, the cleaned and raw tables hold identical rows** (experiment 182).

## Timestamps

- **The `time` column marks the end of the half-hour that the reading covers.** Do not shift it.
  Power timestamps show no 30-minute error on either side of NGED's fix of 2026-03-26 08:30 UTC
  ([data-processing page](../data-processing.md#timestamps-and-availability), experiment 103).
- **Before that instant, the ingest moves NGED's stamp 30 minutes earlier**
  (`PowerTimeSeries.correct_late_timestamps`). A study must not shift the stamp a second time.
- **Join power to an hourly weather product on `(time - 1 µs).truncate("1h") + 1h`.** The same
  page gives the evidence.
- **Assume a power reading reaches the forecast 3.5 hours after the half-hour it covers.** This is
  an assumption: no experiment has measured the delay on live telemetry (experiments 87, 104, 106;
  provisional).

## Gaps, ramp-up, and regime changes

- **Count a series' history from its first unflagged row.** Only five Primaries hold long
  histories: D5, D10, and D11 from 2019-09, D14 from 2019-10, and D16 from 2020-12. The late
  starters begin between 2023 and 2025 (experiment 103).
- **Cut the first months of most demand series.** The cut per series (0.7 to 19.4 months at 13 of
  the 16 Primaries) is in the [trim table](../data-processing.md#trim-the-start-of-each-series).
  An uncut start hurts a model that has no recent power input far more than one that has it
  (experiment 102).
- **Use the fixed lockdown window 2020-03-23 to 2021-07-19, and do not detect it from the data.**
  Check that a fit that leaves the window out kept at least 180 days of training rows (experiments
  6, 103, 173).
- **Some series changed regime.** D8's weekday daytime demand fell from about 9 MW to about 3 MW
  between 2023-10 and 2025, and other steps are listed in the [regime
  entry](../data-processing.md#fixed-windows) (experiment 103).
- **A metered solar farm can climb to its rated output in stages over months.** One farm took eight
  months ([network page](../../background/network.md#a-new-solar-farm-reaches-full-output-in-stages-over-months)).
  The metered farms miss 7.6% to 27.6% of their half-hours, so check coverage before comparing
  farms (experiment 179).
- **A flat top in a farm's output is not proof of clipping.** A pre-registered rule found AC
  clipping at 2 of 6 farms and a post hoc rule at 4 of 6, and an export cap or panel geometry
  makes the same shape (experiment 179).
- **The farm series behind a disaggregated Primary can be almost absent.** For D8, one farm has 1
  raw reading ever, so "missing" is the baseline (experiment 182).
- **A metered farm's `effective_capacity_mw` may itself be estimated from output.** Power over
  capacity near 1.0 is therefore not an independent level (experiment 179).

## Capacity

- **Scores are in points of capacity, where capacity is the 99th percentile of the absolute power
  over the series' whole observed history.** One point is 1% of that value. See the
  [glossary](../glossary.md#capacity-and-points-of-capacity).

Last checked against experiment 183.
