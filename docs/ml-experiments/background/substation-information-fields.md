# Substation information fields

**Four sources describe a substation: the metadata table, NGED's hierarchy files, the effective-capacity
table, and the estimates of solar capacity that experiments have built.** This page lists what each
source holds, where it sits, and which fields have misled an experiment. The power readings
themselves are on the [NGED power telemetry page](nged-power-telemetry.md), and the paths are on the
[disk page](what-is-on-disk.md).

## The metadata table

- **`data/NGED/metadata.parquet` holds one row per time series, under the `TimeSeriesMetadata`
  contract.** The field descriptions in
  `packages/contracts/src/contracts/power_schemas.py` own the meaning of each column (code).
- **The trial area has 33 rows: 16 Primaries, 2 BSPs, 2 GSPs, and 13 customer connections.**
  Of the 13 connections, 7 are typed PV, 3 Wind, and 1 each BESS, Biofuel, and Other (Generation)
  (disk).
- **Only the 16 Primaries carry an area polygon (`area_wkt`)**, and every customer row has a null
  polygon (disk).
- **`information` is a free-text note, filled for 5 of the 33 rows.** The notes are a snapshot from
  July 2026, because the earlier raw files carry no value, so a note cannot say when a fault began
  (exp 182; 33 rows; provisional).

## The hierarchy files

- **Four NGED files and a README sit in `data/NGED/substation_hierarchy_2026-10-15/`.** They are
  `Substations.csv`, `Primary Substation Interconnections.csv`, `Monitor Direction.csv`, and
  `Peak Loads.xlsx`. They are private (disk).
- **`Substations.csv` gives each substation's BSP and GSP.** The trial area has two GSPs, each with
  one BSP (disk).
- **`Primary Substation Interconnections.csv` lists pairs of Primaries that can be connected.**
  Every trial-area Primary has at least one pair, so the topology looks complete
  ([data sources](../../roadmap/data-sources.md#provided-as-reference-files-mostly-static-reference-historical);
  docs).
- **`Monitor Direction.csv` gives each series' meter type and flow direction.** One solar farm's
  substation number differs from its metadata row, and the other sources agree with the metadata
  (docs).
- **`Peak Loads.xlsx` disagrees with the telemetry at several Primaries**, so experiments take the 99th
  percentile of observed power as the capacity of a series instead (docs).
- **The Primaries beneath a BSP do not sum to the BSP's reading.** Primaries that report only
  apparent power, and the generators and unmetered load beneath the BSP, stop the sum from closing.
  Adding every child Primary to a bottom-up sum did not improve the BSP forecast (exp 161, 204; 2
  BSPs, one scored year; provisional).

## The effective-capacity table

- **`data/effective_capacity/` holds one value per series: the 99th percentile of the absolute power
  over the series' whole observed history, in MW.** It is not a nameplate, firm, or connection
  rating (code).
- **Scores in points of capacity divide by this value.** See the
  [glossary](../glossary.md#capacity-and-points-of-capacity) (code).
- **The value is built from the series' own output, so a farm's power divided by it near 1.0 is not
  an independent level** (exp 179, 183; 6 farms; provisional).

## Estimates of solar capacity behind a Primary

- **No source tells us the solar capacity behind a Primary, so experiments estimate it from the
  Primary's power.** The estimates are AC capacity in MW, taken as the 99th percentile of the
  solar output over daylight half-hours (exp 129; 5 Primaries; provisional).
- **Experiment 45 built weekly capacity "vintages", one block per origin Monday, and the `frozen`
  variant uses data before the origin only.** A cut earlier than a Primary's first vintage receives
  a later vintage's level, which is hindsight. The rule is in the `studies.dp_pinned_capacity`
  docstring (code).
- **The latest week of a real-time vintage has 3.6 times the path error of the same week in the
  final fit.** The pinned-capacity code therefore holds the median of the last 13 weeks (exp 45; 8
  Primaries with injected paths; provisional).
- **Experiment 129 gives one capacity per season-year at D5, D10, D11, D14, and D16, with a
  range.** For 2025/26 the estimate is 3.3 to 4.5 MW per Primary at the five tested, and the range
  is a spread over test scenarios, not a calibrated interval (exp 129; 5 Primaries; provisional).
- **Reading the capacity from the Primary's own power did not improve the forecast.** An as-of
  capacity scored 0.110 points of capacity worse than the pinned vintage on the sunniest third of
  daylight half-hours (exp 139; 5 Primaries, one scored year; provisional).
- **A monthly capacity path recovered less of an injected farm than a yearly one**, 0.67 to 0.78
  against 0.75 to 0.87 (exp 157; 6 Primaries, 5 injected farms; provisional).
- **Refreshing the capacity table once per season-year lowered the error by 0.018 points of
  capacity, and a weekly refresh made no distinguishable difference** (exp 78; the fold A row set;
  small gain; provisional).
- **The one dated list of installations we hold, the Feed-in Tariff register, ends on 2021-03-24.**
  No capacity step after that date can be matched to an installation (exp 215; 6 Primaries;
  provisional).
- **The published estimates are on the [embedded solar capacity
  study](../../studies/embedded-pv-capacity.md)** (docs).

Last checked against experiment 218.
