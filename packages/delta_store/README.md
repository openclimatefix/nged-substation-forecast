# `delta_store`

Physical storage policy for the project's Delta tables.

## Why this package exists

`contracts` owns each table's *logical* shape and meaning. This package owns its *physical*
layout: parquet writer properties (codec + per-column encodings), compression-friendly sort
orders, and significand-precision rounding, plus the write helpers that apply them. Dagster
assets stay thin by writing through this package rather than calling `write_deltalake` with
ad-hoc settings — and it becomes impossible to land rows in a table without its storage format
applied.

The flagship example is the internal `power_forecasts` table: ZSTD + `DELTA_BINARY_PACKED`
timestamps + `BYTE_STREAM_SPLIT` floats + member-adjacent sorting + rounding `power_fcst` to a
13-bit significand shrank the 403.6M-row development table from 6.33 GB to 0.73 GB.

## Contents

- `precision.round_to_significand_bits()` — rounds a `Float32` expression to a chosen number of
  significand bits in pure Polars arithmetic (Veltkamp splitting), zeroing the low mantissa bits
  so `BYTE_STREAM_SPLIT` + zstd can compress them away. The trick and its preconditions are
  rigorously documented on the function.
- `power_forecasts` — the `power_forecasts` table's writer properties, sort order, precision
  policy, and `write_power_forecasts()`.
