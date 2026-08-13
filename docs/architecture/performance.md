# Performance and Scale

This page holds the measured performance engineering behind three of the
[design principles](../design-philosophy/design-principles.md) — [*the whole system must be
exercisable on one
laptop*](../design-philosophy/design-principles.md#6-the-whole-system-must-be-exercisable-on-one-laptop),
[*push the work down to the engine; materialise once, as late as
possible*](../design-philosophy/design-principles.md#11-push-the-work-down-to-the-engine-materialise-once-as-late-as-possible),
and [*measure; do not
assume*](../design-philosophy/design-principles.md#12-measure-do-not-assume). Everything here exists so that the full
pipeline — training and a complete 51-member backtest — runs on an ordinary laptop, which is what
keeps experimentation fast
([H2](../design-philosophy/engineering-hypotheses.md#h2-a-hundred-experiments-per-person-in-a-peak-month)) and the
running service cheap ([H4](../design-philosophy/engineering-hypotheses.md#h4-it-runs-for-pocket-money)).

## Storage formats: measured, not assumed

Both big Delta tables use the same precision lever: values are rounded to a 13-bit significand in
pure Polars arithmetic — Veltkamp splitting, rigorously documented on
`delta_store.precision.round_to_significand_bits` (max relative error 2⁻¹³ ≈ 1.2×10⁻⁴) — which
zeroes the low mantissa bits that are otherwise incompressible entropy. Beyond that shared lever,
the two tables need *different* writer properties, and each choice below was measured rather than
assumed:

* **NWP** (`delta_store.nwp`) stores physical-unit `Float32` with plain ZSTD-3 and parquet's
  default dictionary+RLE encodings — `BYTE_STREAM_SPLIT` measures *worse* here, because the
  significand rounding collapses many cells/members onto repeated values that a dictionary exploits
  better. Rows are sorted member-early so single-member reads can skip row groups. The result is
  ~40–41 GB per year for the full ECMWF ENS dataset for Great Britain, and a single day's NWP data
  takes about one minute to download and convert. The measured numbers are in
  [PR #271](https://github.com/openclimatefix/nged-substation-forecast/pull/271).

* **`power_forecasts`** (`delta_store.power_forecasts`) sorts ensemble members of the same target
  adjacent, uses `DELTA_BINARY_PACKED` timestamps and `BYTE_STREAM_SPLIT` for `power_fcst`
  (near-continuous ML output has no repeats for a dictionary to exploit), shrinking the 403.6M-row
  development table from 6.33 GB to 0.73 GB (8.7×). Details in
  [PR #268](https://github.com/openclimatefix/nged-substation-forecast/pull/268).

## Lazy evaluation strategy

**Rule: everything stays lazy until the model boundary.**

The entire feature-engineering pipeline operates on Polars `LazyFrame`s. No code between reading from storage and calling `BaseForecaster.train()` or `predict()` is allowed to call `.collect()`. The pipeline is:

```text
Nwp.scan_delta(path)                     # lazy scan, already Float32 physical units
  → FeatureEngineer.engineer()           # spatial join + feature pipeline;
                                         #   returns pt.LazyFrame[AllFeatures]; no collect
  → BaseForecaster.train/predict()       # subclass materialises the data here, at the boundary
```

**Why it matters:**

* **Memory**: Polars only materialises the data at the last moment, at the model boundary. Intermediate representations (individual join outputs, lag columns before filtering) are never written to RAM.
* **Query optimisation**: Polars can see the full plan end-to-end and push down filters (e.g. date ranges, `time_series_id` subsets) into the Delta Lake scan before any data crosses the wire.
* **Clean boundaries**: The materialisation site is the model boundary — the one place where a third-party library (XGBoost, PyTorch, …) needs a concrete in-memory array. Keeping it there makes the boundary explicit and easy to find.

**The one exception** is a `limit(1).collect()` guard in `_engineer_features` (inside `ml_core`), reached only when a weather lag feature is requested. It checks the raw NWP scan for the control member, so the pipeline fails loudly instead of silently returning an empty frame. It probes the raw frame rather than the upsampled one deliberately: `ensemble_member` is one of the upsample's group-by keys, so the two are equivalent, but `SLICE` cannot push through the upsample's window functions — probing the upsampled frame would execute the whole upsample of the control member before answering.

**Contract for `BaseForecaster` subclasses**: the lazy plan is materialised *at the model boundary*, as late as possible, and never by callers — never ask a caller to collect before passing data in. A subclass typically does a single `.collect()`; keeping that bounded is the **caller's** job, by pruning the *inputs* before feature engineering (not by filtering the engineered output — see below).

### Bounding feature-engineering memory: prune the inputs, not the output

The feature-engineering plan is dominated by the **NWP scan**. The NWP Delta is large — for the V1 trial it is **~86 GB**: 810 daily `init_time` partitions, each up to ~7.24M rows = **1671 H3 cells × 51 ensemble members × 85 native steps** (control member alone is 142k rows/partition). The 30-min upsample and the multi-run bulk join inflate that further. So the whole memory question is: *how little of that NWP do we touch?*

**You cannot prune the scan by filtering the engineered output.** `data.filter(time_series_id == x)` runs the cell-attach join, the 30-min upsample (`group_by` + `explode` + `interpolate`) and the bulk join *first*, then drops rows. And NWP is keyed by `h3_index`, not `time_series_id`, so a `time_series_id` predicate can never reach the NWP scan at all. Pruning must be applied to the **raw inputs**, in `_load_engineering_inputs`, directly on the `Nwp.scan_delta` scan.

What actually prunes the NWP scan — verified with `LazyFrame.explain()`:

| Predicate on the **raw NWP scan** | Effect |
|---|---|
| `init_time ∈ [start − 16d, end]` | **Partition prune** — NWP is partitioned by `init_time`, so only those partition directories are opened. A `valid_time` filter *alone* does **not** prune partitions (it scanned ~887 files). |
| `ensemble_member ∈ {…}` on the raw scan | Only the requested members are decoded (≈50× less for control-member training) — and because rows are sorted member-early (`delta_store.nwp.NWP_SORT_COLS`), row-group min/max stats let the reader skip most of each partition outright for a single-member predicate. |
| `h3_index ∈ {cells}` | Restricts to the cells the requested series sit in. |
| `time_series_id == x` on the **output** | Prunes the power scan + metadata join, but **not** the NWP scan (no `time_series_id` there), and only after the upsample. Doesn't help. |
| `slice(n)` / `head(n)` (row count) on the output | **Nothing** — a row slice can't push through a join; the whole join runs first. |

**Pruning the partitions is necessary but not sufficient — you must also stream the collect.** `init_time` prunes whole partition *files*, but `ensemble_member` and `h3_index` are not partition keys, so they act *within* each file. The `ensemble_member` predicate is cheap because of the on-disk sort: rows are ordered `init → member → valid → h3` (`delta_store.nwp.NWP_SORT_COLS`), so each ~1M-row row group spans only a handful of members and a single-member read skips most row groups via min/max stats — measured on a real 29-day, 9-cell, control-member collect at **~5× faster and ~5× less peak memory** than the previous `valid_time`-first sort. `h3_index` is *not* a sort-early column, though: cell filtering is still decode-then-filter within the surviving row groups, so an in-memory collect would still materialise every surviving row before dropping 1662/1671 of the cells. The **streaming engine** (`collect(engine="streaming")`) applies the predicates per morsel, so peak memory is set by the morsel/row-group size, *not* the window length: measured pre-migration, a *full 15-month* control-member collect stayed at ~2.5 GB (the member-early sort only lowers this). `XGBoostForecaster.train`/`predict` therefore collect with `engine="streaming"`.

**Many-to-one `h3_index` ↔ `time_series_id`:** one NWP cell covers several series (the 32 V1 series live in just **9 cells**, one holding 12), so `h3_index` pruning is keyed on the (few) cells the requested series occupy, and the feature engineer's spatial join replicates each cell's weather across its series.

**Resulting design** (`_load_engineering_inputs` applies all three NWP predicates — `init_time`, `ensemble_member`, and `h3_index` = the requested series' cells — to the raw scan, and every collect streams):

| | control member (train) | full 51-member ensemble (predict) |
|---|---|---|
| All eligible series at once | ~5 GB → `train` collects once, groups in memory | ~25 GB ✘ (OOMs) |
| One `init_time` chunk at a time | — | ~9 GB → `cv_power_forecasts` chunks by `init_time`, appends to Delta |

Prediction is bounded by chunking on **`init_time`** (`_PREDICT_INIT_CHUNK`, 14 days), *not* by cell. `init_time` is both the partition key and the axis that fans the output out across runs, so a chunk's forecast frame stays small while each partition is read exactly once and all series/cells/members are processed together (a per-*cell* loop instead OOMs on the busiest cell — 10 series × 51 members × the 10-month window ≈ 116M rows ≈ 25 GB). Measured end-to-end on the `mid_2025_to_mid_2026` fold: training peaks ~5 GB and the full 51-member validation prediction (~321M forecast rows) peaks ~9 GB — both well under a 24 GB laptop.

## The other hard ceiling: Polars' 32-bit row index

RAM is not the only bound a Polars query must respect. Default Polars builds use a 32-bit row
index (`IdxSize`), so any single materialised frame, row count, or row index is capped at
**2³² ≈ 4.29 billion rows** — and crossing the cap raises **no error**. Row counts wrap modulo
2³²: `pl.scan_delta` over the ~5.9-billion-row NWP dev table reports
1,652,180,189 rows via `.select(pl.len())` (= 5,947,147,485 mod 2³², exactly), and
`group_by(...).agg(pl.len())` wraps identically for any single group past the cap, streaming
engine included. This was investigated and empirically pinned down in
[issue #293](https://github.com/openclimatefix/nged-substation-forecast/issues/293): the
wraparound reproduces on plain Parquet scans (it is not a Delta Lake bug), and the
`polars-u64-idx` build (64-bit index, lags mainline releases) returns correct counts on the same
queries.

What is and isn't affected — each verified empirically on 5-billion-row scans:

| Operation on a >2³²-row scan | Outcome |
|---|---|
| `pl.len()` / `group_by(...).agg(pl.len())` where a count exceeds 2³² | **Silently wraps mod 2³²** |
| Materialising a single frame of ≥2³² rows | Unsupported (in practice RAM is exhausted first, which at least fails loudly) |
| Filtered / partition-pruned query whose *result* is < 2³² rows | Correct |
| Value aggregations (`sum`, `min`/`max`, quantiles) over > 2³² rows | Correct — values aren't indices |

The rules that follow:

* **Never row-count a table that can exceed 2³² rows with Polars.** Whole-table counts and
  data-completeness checks must come from the Delta transaction log — `DeltaTable(path).count()`,
  or summing `num_records` over `get_add_actions(flatten=True)` — which is metadata-only, exact,
  and reads no data files.
* The [lazy evaluation strategy](#lazy-evaluation-strategy) above already keeps every production
  collect far below the cap (input pruning + `init_time` chunking), so pipeline reads are
  unaffected. The cap is a second, independent reason that rule exists: an unbounded collect of a
  V2-scale table wouldn't just OOM, its row accounting would be wrong.
* Tables past the cap today: **NWP** (~5.9B rows). **`power_forecasts`** will pass it at V2 scale
  (~1 trillion rows — see [forecast delivery](forecast-delivery.md#how-big-is-flexpectations-power-forecast-data)).
  The one code path that must be re-chunked before then is the `metrics` asset, which currently
  collects a whole `(experiment_name, fold_id)` group of `power_forecasts` at once and discovers
  groups via a full-table `unique()` — fine at V1 (≤ ~414M rows/fold), but a V2 fold is tens of
  billions of rows, so scoring will need to chunk within a fold (e.g. by `valid_time` window or
  `time_series_id` batch) for RAM reasons anyway; the index cap makes it correctness-critical too.
