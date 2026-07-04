# NWP storage experiment: Float32 + significand rounding instead of Int16 quantisation

> Experiment plan (requested by Jack, 2026-07-04, during PR #268 review discussion). Nothing may
> link here from code or `docs/`. When the experiment concludes, promote the results and any
> surviving design decisions to `docs/` (likely `docs/architecture/overview.md` and/or a
> `docs/roadmap/` item for the migration) and delete this file.

## Objective

Decide whether to replace the NWP table's Int16 affine quantisation (`NwpScalingParams`, the
`NwpOnDisk` / `NwpInMemory` dual-schema dance) with plain Float32 columns rounded via
`delta_store.precision.round_to_significand_bits()` — the scheme now used by `power_forecasts`.

**Why**: collapse `NwpOnDisk` + `NwpInMemory` into one `Nwp` contract; delete
`to_nwp_in_memory()` and `NwpScalingParams`; remove the rescale boundary that forces the
"filter `ensemble_member` *before* the rescale or OOM" discipline; no per-variable ranges to
maintain (and no silent clipping when a record-breaking value exceeds a configured range).

**Cost to quantify**: mantissa rounding spends its bits differently from affine quantisation.
For offset-dominated variables it is coarser per bit (temperature in Kelvin ~4.6× at 12 bits;
MSL pressure in Pa ~9×) while being *finer* near zero (precipitation, radiation). Estimated
storage impact +30–70% vs today's ~40 GB/yr — but that estimate must be measured, not trusted:
the power_forecasts extrapolation from one file was off by 2.5× (favourably) at table level.

## Step 1 — storage benchmark (no production code changes)

Scratchpad script over a handful of real `init_time` partitions (pick recent + seasonal spread):

1. Read partitions, rescale to Float32 physical units (existing `to_nwp_in_memory` path).
2. Grid: `keep_bits ∈ {10, 12, 13, 16}` × sort order ∈ {current `init → valid → member → h3`,
   member-early `init → member → valid → h3`} × writer properties (ZSTD;
   BYTE_STREAM_SPLIT on all Float32 weather columns; DELTA_BINARY_PACKED on
   `init_time`/`valid_time`).
3. Measure bytes/partition vs the current Int16 + zstd-14 files; extrapolate GB/yr.
4. Record max relative error per variable per `keep_bits` (sanity: matches 2^-keep_bits).

## Step 2 — read-path benchmark (the possible hidden win)

The current sort (`init → valid → member → h3`) is why row-group statistics cannot skip the ~50
unwanted ensemble members and control-member training reads must stream-decode every row
(documented in `docs/architecture/overview.md`). With the member-early sort from Step 1:

- Measure wall time + peak RSS of a control-member, one-month, 9-cell collect against both sort
  orders (streaming engine, same predicates as `_load_engineering_inputs`).
- Confirm row-group pruning with `.explain()` / pyarrow row-group stats.

A large read-side win here could justify the migration even at the pessimistic end of the
storage estimate.

## Step 3 — decision gate (discuss with Jack)

Adopt if: total table ≤ ~60 GB/yr at a `keep_bits` whose per-variable absolute error is
comfortably below feature relevance (proposed: temperature step ≤ 0.25 K, MSL pressure step
≤ 1 hPa), and the Step 2 numbers don't argue otherwise. Otherwise: keep Int16, close this file,
record the numbers in `docs/architecture/overview.md`.

## Step 4 — migration outline (separate PR, only if adopted)

1. `delta_store.nwp`: writer properties, sort cols, `keep_bits` constant, `write_nwp()`.
2. Single `Nwp` contract in `contracts.weather_schemas` (Float32 weather columns); delete
   `NwpOnDisk`, `NwpInMemory`, `NwpScalingParams`, `to_nwp_in_memory()`; update
   `dynamical_data` download/convert and `_load_engineering_inputs` (filters stay, the
   before-the-rescale ordering constraint disappears).
3. One-off in-place rewrite of the historical table (~83 GB, `init_time`-partitioned), then
   re-measure the training/prediction memory table in `docs/architecture/overview.md`.

## ⚠️ Gotcha for the in-place rewrite (hit this on the power_forecasts rewrite, 2026-07-04)

When rewriting a Delta table in place in chunks (first chunk `mode="overwrite"` + partition
predicate, later chunks append), **every read must be pinned to the pre-rewrite table version**
(`pl.scan_delta(path, version=N)` with `N` captured before the first write). An unpinned scan
reads the *latest* version — and after the first chunk's overwrite, the partition's old files
are gone from it, so later chunks silently read back only already-rewritten rows. On the
power_forecasts rewrite this made chunk 2 collect 0 rows and would have dropped 10.7M rows;
only a per-partition row-count assert caught it, and recovery was possible *only because vacuum
had not yet run* (time-travel back to the pre-rewrite version).

Rewrite-script checklist derived from that incident:

- [ ] Capture `DeltaTable(path).version()` up front; pin **all** reads to it.
- [ ] Assert per-partition row counts (and max relative error) against the pinned version
      **before** vacuuming.
- [ ] `vacuum(retention_hours=0, enforce_retention_duration=False)` only after every partition
      verifies — vacuum is the point of no return.
- [ ] Disk headroom: old + new files coexist until vacuum (~2× table size transiently). For the
      ~83 GB NWP table, consider vacuuming after each verified partition group instead of once
      at the end (trade-off: earlier point of no return per partition).
