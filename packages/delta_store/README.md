# `delta_store`

Physical storage policy for the project's Delta tables.

## Why this package exists

`contracts` owns each table's *logical* shape and meaning. This package owns its *physical*
layout: parquet writer properties (codec + per-column encodings), compression-friendly sort
orders, and significand-precision rounding, plus the write helpers that apply them. Dagster
assets stay thin by writing through this package rather than calling `write_deltalake` with
ad-hoc settings — and it becomes impossible to land rows in a table without its storage format
applied.

Every lever below is measured against real data rather than assumed — see [Storage formats:
measured, not
assumed](https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#storage-formats-measured-not-assumed)
for the comparison across both tables, and [design principle
12](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#12-measure-do-not-assume)
for why that discipline matters project-wide. The flagship example is the internal
`power_forecasts` table: ZSTD + `DELTA_BINARY_PACKED` timestamps + `BYTE_STREAM_SPLIT` floats +
member-adjacent sorting + rounding `power_fcst` to a 13-bit significand shrank the 403.6M-row
development table from 6.33 GB to 0.73 GB. `POWER_FORECASTS_WRITER_PROPERTIES`, below on this
page, breaks that figure down lever by lever.

## Contents

- `precision.round_to_significand_bits()` — rounds a `Float32` expression to a chosen number of
  significand bits in pure Polars arithmetic (Veltkamp splitting), zeroing the low mantissa bits
  so `BYTE_STREAM_SPLIT` + zstd can compress them away. The trick and its preconditions are
  rigorously documented on the function.
- `power_forecasts` — the `power_forecasts` table's writer properties, sort order, precision
  policy, and `write_power_forecasts()`.
- `nwp` — the `nwp` table's writer properties, sort order, precision policy, and `write_nwp()`;
  its writer properties are deliberately *different* from `power_forecasts`'s, because the same
  encodings measured worse on NWP data.
