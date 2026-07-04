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

## Results (measured 2026-07-04, branch `experiment/nwp-significand-rounding`)

Scratch scripts (not committed — see the checklist above): loaded real `NwpOnDisk` partitions
from the local ~88 GB table (810 daily partitions, 2024-04-01 → 2026-06-30), rescaled to
`NwpInMemory` Float32 physical units, then rounded with the existing
`delta_store.precision.round_to_significand_bits`.

### Step 1 — storage

9 partitions spread across every season + the most recent (2024-04-01, 2024-07-09, 2024-10-19,
2025-01-29, 2025-05-10, 2025-08-19, 2025-11-27, 2026-03-10, 2026-06-30), each 7,243,785 rows.
The "current" baseline was **rewritten fresh** with today's `WriterProperties(ZSTD, 14)` rather
than trusted from `du` — some on-disk partitions predate a mid-history switch from SNAPPY to
ZSTD-14 (`git log` shows the switch landed 2026-05-15; anything from before that date on disk is
SNAPPY, which would have made the baseline look artificially large).

| Config (sort=current unless noted) | avg MB/partition | extrapolated GB/yr |
|---|---:|---:|
| **Baseline**: Int16, ZSTD-14 (today's format) | 115.3 | 42.1 |
| Float32 + significand, `keep_bits=10`, ZSTD-3 | 93.9 | 34.3 |
| Float32 + significand, `keep_bits=12`, ZSTD-3 | 106.2 | 38.8 |
| **Float32 + significand, `keep_bits=13`, ZSTD-3** | **110.6** | **40.4** |
| Float32 + significand, `keep_bits=16`, ZSTD-3 | 116.4 | 42.5 |
| `keep_bits=13`, ZSTD-3, sort=member-early | 112.9 | 41.2 |
| `keep_bits=13`, ZSTD-3 + BYTE_STREAM_SPLIT (mirrors `power_forecasts`) | 133.8 | 48.8 |
| `keep_bits=13`, ZSTD-14 + BYTE_STREAM_SPLIT | 123.8 | 45.2 |

**The plan's pessimistic +30–70% estimate does not hold — like the `power_forecasts`
extrapolation, this one was also wrong, and also favourably.** At `keep_bits=13` (the same
significand budget already used by `POWER_FCST_SIGNIFICAND_BITS`), plain Float32 + significand
rounding is **~4% *smaller* than today's Int16 table**, before even touching the read-path win.

**Surprise finding: `BYTE_STREAM_SPLIT` makes it *worse* here, unlike `power_forecasts`.**
Per-column inspection (keep_bits=13) shows every continuous variable individually larger under
BSS (e.g. `temperature_2m`: ~2.0 MB with plain ZSTD-3 defaults vs ~3.9 MB with BSS, per
partition). Hypothesis: significand rounding collapses NWP variables to a small set of distinct
values (many H3 cells/ensemble members land on the same rounded value), so plain
dictionary+RLE — Parquet's default column encoding, left alone — captures that repetition
directly; BSS scatters the same repeats across 4 separate byte planes, each with its own
higher entropy, and loses more than it gains. `power_forecasts` is the opposite case: its
target values are a near-continuous ML output with no natural repetition, so BSS wins there.
**Lesson: the writer-properties choice is data-dependent and must be re-measured per table, not
copied from `power_forecasts`.** Sort order (current vs member-early) barely affects bytes
(~2% penalty for member-early) — the effect is dwarfed by the encoding choice.

Max observed error at `keep_bits=13` (bound 2⁻¹³ ≈ 1.22×10⁻⁴ relative, confirmed not exceeded
for any variable):

| Variable | max abs error | Decision-gate threshold |
|---|---:|---|
| `temperature_2m` (°C) | 0.0039 °C | ≤ 0.25 K |
| `pressure_reduced_to_mean_sea_level` (Pa) | 8 Pa (0.08 hPa) | ≤ 1 hPa |
| `pressure_surface` (Pa) | 8 Pa | — |
| all other variables | ≤ 0.5 (native units) | — |

Both gate thresholds are cleared with >10× margin — and in fact even `keep_bits=10` clears them
(temp 0.031 °C, MSL pressure 0.64 hPa). The plan's Kelvin-based pessimism about
"offset-dominated" temperature doesn't apply: `NwpInMemory.temperature_2m` is stored in
**Celsius**, not Kelvin, so its magnitude (and hence its significand-rounding error) is an order
of magnitude smaller than the plan's worst-case estimate assumed. Pressure (Pa, genuinely large
offset ~10⁵) is the real test case and still passes comfortably.

### Step 2 — read path (the hidden win the plan hoped for — confirmed, and large)

Built two real 29-partition (June 2025) Delta tables at the Step 1 winner (`keep_bits=13`,
plain ZSTD-3), partitioned by `init_time` exactly like production: one with the current sort
(`valid_time → ensemble_member → h3_index`), one member-early (`ensemble_member → valid_time →
h3_index`). Benchmarked a control-member (`ensemble_member == 0`), 9-cell (the real V1
`h3_res_5` set), whole-month streaming collect against both, each in its own subprocess (twice,
order swapped, to rule out page-cache artefacts):

| Sort order | wall time | peak RSS |
|---|---:|---:|
| Current (`valid_time` first) | 0.15 s | 882–1138 MB |
| Member-early | 0.02–0.04 s | 203–206 MB |

**~4–6× faster and ~4–6× less peak memory**, reproduced on both orderings of the two runs. This
matches the mechanism `docs/architecture/overview.md` already documents for why the *current*
sort can't prune members: with `ensemble_member` sorted first, each ~1M-row Parquet row group
spans only ~7 of the 51 members, so row-group min/max stats let the reader skip most groups
outright for a single-member predicate — instead of decoding every row in every group and
filtering after, as today's sort forces.

### Step 3 — decision gate

**Adopt.** Both gate conditions are met with room to spare:

- Storage: 40.4 GB/yr at `keep_bits=13` (`current` sort) or 41.2 GB/yr (`member-early` sort) —
  both comfortably under the ~60 GB/yr ceiling, and *below* today's 42.1 GB/yr baseline.
- Per-variable error: >10× inside both proposed thresholds (temperature, MSL pressure).
- Step 2 doesn't argue otherwise — it argues *for* switching the sort order too: paying the
  ~2% storage penalty of member-early sort buys a ~5× win on every control-member training/predict
  read, which is the hot path for `_load_engineering_inputs`.

**Recommendation to bring to Jack:** adopt Float32 + `round_to_significand_bits(keep_bits=13)`
for NWP storage, written with plain `WriterProperties(compression="ZSTD", compression_level=3)`
(no `BYTE_STREAM_SPLIT` — measured worse here) and the **member-early** sort order
(`init_time → ensemble_member → valid_time → h3_index`), accepting its ~2% storage cost for the
~5× control-member read win. Proceed to Step 4 (migration outline) once confirmed.
