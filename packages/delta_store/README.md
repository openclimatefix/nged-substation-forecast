# `delta_store`

Physical storage policy for the project's Delta tables.

## Why this package exists

`contracts` owns each table's *logical* shape and meaning. This package owns its *physical* layout:
parquet writer properties (codec + per-column encodings), compression-friendly sort orders, and
significand-precision rounding, plus the write helpers that apply them. Dagster assets stay thin by
writing through this package rather than calling `write_deltalake` with ad-hoc settings — and it
becomes impossible to land rows in a table without its storage format applied.

Every lever applied by `power_forecasts` and `nwp` below is measured against real data rather than
assumed — see [Storage formats: measured, not
assumed](https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#storage-formats-measured-not-assumed)
for the comparison between those two tables, and [design principle
12](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#12-measure-do-not-assume)
for why measuring rather than assuming matters project-wide. This package covers six tables. The
other four carry no writer-properties tuning, because no measurement has yet been made to justify
any tuning.

**The flagship tuned table is the internal `power_forecasts` table, whose storage format shrank a
403.6M-row development copy of that table from 6.33 GB to 0.73 GB.** The levers are ZSTD,
`DELTA_BINARY_PACKED` timestamps, `BYTE_STREAM_SPLIT` floats, member-adjacent sorting (the ~51
ensemble members of one forecast target on adjacent rows), and rounding `power_fcst` to a 13-bit
significand. `POWER_FORECASTS_WRITER_PROPERTIES`, below on this page, gives the lever-by-lever
breakdown, measured on the table's least compressible single file rather than on the full table.

## Contents

- `precision.round_to_significand_bits()` — rounds a `Float32` expression to a chosen number of
  significand bits in pure Polars arithmetic (Veltkamp splitting), zeroing the low mantissa bits so
  `BYTE_STREAM_SPLIT` + zstd can compress them away. The trick and its preconditions are rigorously
  documented on the function.
- `power_forecasts` — the `power_forecasts` table's writer properties, sort order, precision policy,
  and `write_power_forecasts()`.
- `nwp` — the `nwp` table's writer properties, sort order, precision policy, and `write_nwp()`; its
  writer properties are deliberately *different* from `power_forecasts`'s, because the same
  encodings measured worse on numerical weather prediction (NWP) data.
- `power_time_series` — `write_power_time_series()`, an append-only write to the `power_time_series`
  table.
- `eligible_time_series` — `write_eligible_time_series()`, a per-`fold_id`-partition overwrite to
  the `eligible_time_series` table.
- `effective_capacity` — `write_effective_capacity()`, a whole-table overwrite to the
  `effective_capacity` table.
- `forecast_metrics` — `write_forecast_metrics()`, a per-`(experiment_name, fold_id)`-partition
  overwrite to the `forecast_metrics` table, including the Enum→String cast delta-rs needs before
  writing.
