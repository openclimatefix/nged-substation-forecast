# Substation information fields

**Four sources describe a substation: the metadata table, NGED's hierarchy files, the effective-capacity
table, and the estimates of solar capacity that experiments have built.** This page lists what each
source holds, where it sits, and which fields have misled an experiment. The power readings
themselves are on the [NGED power telemetry page](nged-power-telemetry.md), and the paths are on the
[disk page](what-is-on-disk.md).

## The metadata table

- **`data/NGED/metadata.parquet` holds one row per time series, under the `TimeSeriesMetadata`
  contract.** Its columns are `time_series_id`, `time_series_name`, `time_series_type`, `units`,
  `licence_area`, `substation_number`, `substation_type`, `latitude`, `longitude`, `information`,
  `area_wkt`, `area_center_lat`, `area_center_lon`, and `h3_res_5` (code).
- **`substation_type` is one of Primary, BSP, GSP, HV Customer, or EHV Customer.** The first three are
  substations. A generator or battery is a customer connection (code).
- **`time_series_type` is "Disaggregated Demand" for every Primary and "Raw Flow" for every BSP and
  GSP.** "Disaggregated Demand" means NGED has already removed the metered generation connected to
  that Primary. The remaining types in the trial area are BESS, Biofuel, PV, Wind, and Other
  (Generation) (code).
- **For a customer series, `latitude` and `longitude` give the substation, not the customer's
  site.** The H3 cell (`h3_res_5`) follows the same coordinates (code).
- **Only Primaries carry an area polygon (`area_wkt`).** A generator's polygon, where one exists,
  covers the generator itself (code).
- **`information` is a free-text note, filled for 5 of the 33 rows.** It is a snapshot from July 2026,
  because the earlier raw files carry no value, so it cannot say when a fault began (exp 182; 33
  rows; provisional).

## The hierarchy files

- **Five NGED files and a README sit in `data/NGED/substation_hierarchy_2026-10-15/`.** They are
  `Substations.csv`, `Primary Substation Interconnections.csv`, `Monitor Direction.csv`,
  `Peak Loads.xlsx`, and `Switching Logs.xlsx`. They are private (disk).
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
- **A BSP reading is a soft hint about the Primaries beneath it, never an exact sum.** Primaries
  that report only apparent power, and the generators and unmetered load beneath a BSP, stop the sum
  from closing (exp 124, 161, 204; 2 BSPs; provisional).

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
- **Experiment 45 built weekly capacity "vintages", one block per origin Monday.** The `frozen`
  variant uses data before the origin only. The first origin is 2023-08-07 (2025-04-14 at D9), and a
  cut earlier than a Primary's first vintage receives a later vintage's level, which is hindsight
  (`studies.dp_pinned_capacity`; code).
- **The latest week of a real-time vintage has 3.6 times the path error of the same week in the
  final fit.** Exp 45 measured this on 8 Primaries with injected paths (provisional), and the pinned-capacity
  code holds the median of the last 13 weeks instead (code).
- **Experiment 129 gives one capacity per season-year at D5, D10, D11, D14, and D16, with a
  range.** For 2025/26 the estimate is 3.3 to 4.5 MW per Primary at the five tested, and the range
  is a spread over test scenarios, not a calibrated interval (exp 129; 5 Primaries; provisional).
- **Reading the capacity from the Primary's own power did not improve the forecast.** An as-of
  capacity scored 0.110 points worse on the sunniest third, and a monthly path recovered less than
  a yearly one (exp 139, 157; 5 Primaries, one year; provisional).
- **Re-estimating the capacity before each run gained 0.018 points of capacity** (exp 78; 5
  Primaries; provisional).
- **The one dated list of installations we hold, the Feed-in Tariff register, ends on 2021-03-24.**
  No capacity step after that date can be matched to an installation (exp 215; 5 Primaries;
  provisional).
- **The published estimates are on the [embedded solar capacity
  study](../../studies/embedded-pv-capacity.md)** (docs).

Last checked against experiment 183.
