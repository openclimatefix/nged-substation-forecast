# Re-measure the NWP predicate-pushdown speed-ups (#659)

**The problem: five places in the repo describe what the member-early on-disk sort buys a
single-member NWP read, and none of them rests on a measurement of the code as it stands today.**
Two still assert the original figures — "~5× faster, ~5× less peak memory ... for a ~2% storage
cost" — which were measured before PR #653 fixed the `Nwp` dtype mismatch, during the period when
the cast in `Nwp.scan_delta` was silently defeating Parquet predicate pushdown. The other three had
their figures stripped by PR #653 and now say the speed-up "needs re-measuring against real data".

**The planned solution: a committed benchmark script that writes the same 29 real NWP partitions
into two scratch Delta tables, one sorted member-early and one sorted `valid_time`-first, reads the
control member over 9 H3 cells from each, and reports four numbers — wall time, peak resident
memory, on-disk bytes, and how many Parquet row groups the scan can skip.** The measured figures
then go into the five locations. The benchmark runs against the local development NWP table on this
machine, so it reads no S3.

## Verdict, size and departures

**Verdict: worth doing, and cheaper than the issue assumes.** The claims in `performance.md` and
the `dynamical_data` README are exactly the kind design principle 12, [measure; do not
assume](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#12-measure-do-not-assume),
exists to keep honest, and two of them currently state numbers we know were measured under a bug.

**Size: medium.** The change lands in docs, docstrings, and one new script; it touches no Patito
contract, no Delta table, no asset, and no degradation rule. The risk is entirely in whether the
measurement measures what it claims. That buys both plan reviews — the simplicity review because
the plan adds a new file, and the correctness review because the benchmark method is the whole of
the risk — and one diff review, the correctness-and-cut-it-down pass. The mutation pass is not
worth running: this change pins no behaviour that a mutant could break.

### Departure 1: no S3 read, and so no sign-off

**The issue opens by saying the work "requires reading real production NWP data from S3 and needs
the maintainer's explicit sign-off". It does not.** A complete, current NWP Delta table already
sits on this machine at `/home/jack/dev/nged-substation-forecast/data/NWP`, reachable from the
worktree through the repo's usual `data` symlink. Checked while planning:

- 123 GB, 898 daily `init_time` partitions, 2024-04-01 to 2026-09-16.
- Every partition file has an mtime of 2026-09-05 or later, so the whole table was written by the
  current `write_nwp` — after PR #653 merged on 2026-08-14 — and therefore carries the member-early
  sort and the corrected dtypes.
- The table's logical schema reads back as `ensemble_member: Int8`, `h3_index: Int64`,
  `nwp_model_id: String`, and `categorical_precipitation_type_surface: Int16`, matching the
  post-#653 contract.
- `Nwp.scan_delta(path).filter(pl.col("ensemble_member").is_in([0])).explain()` puts
  `SELECTION: col("ensemble_member").cast(Int64).is_in([[0]])` *inside* the Parquet scan, so
  pushdown genuinely happens on this table.

**A local read is also the better measurement, not merely the cheaper one.** The quantity under
test is how much Parquet decode work row-group skipping avoids. Reading over S3 adds network
latency that varies run to run and would swamp the difference. The one thing the local number
understates is the S3 case: there, a skipped row group also skips a network range request, so the
local figure is a floor. The write-up should say so rather than implying the number transfers.

### Departure 2: five locations carry the claim, not three, and the excluded file is gone

**PR #653 rewrote three locations but left two other copies of the discredited figures standing,
and neither is named in the issue.** A grep for the figures across the repo finds:

| Location | State today |
|---|---|
| `packages/delta_store/src/delta_store/nwp.py:20-22` (**module** docstring) | Still asserts "~5x faster and ~5x less peak memory for a ~2% storage cost" |
| `packages/dynamical_data/README.md:49-53` ("Read path") | Still asserts "**~5× faster, ~5× less peak memory** (0.15 s / ~1 GB → 0.02–0.04 s / ~205 MB), for a ~2% storage cost" |
| `packages/delta_store/src/delta_store/nwp.py` (`NWP_SORT_COLS` docstring) | Figures removed by #653; says they "need re-measuring" |
| `docs/architecture/performance.md:76` and `:81` | Figures removed by #653; says they "need re-measuring" |
| `src/nged_substation_forecast/defs/_engineering_inputs.py:78-83` | Figures removed by #653; links to `performance.md` |

The two survivors are the higher priority of the five, because a doc that states a wrong number is
worse than one that admits it does not know.

**The issue's "Not in scope" paragraph points at a file that no longer exists.**
`docs/architecture/adapting-to-another-geography.md` is not in `docs/architecture/`, and no file
under `docs/` matches `*geograph*`. Nothing to exclude, and nothing to do about it.

### Departure 3: commit the benchmark rather than running it once

**The issue asks for a measurement; this plan also asks for the measurement to stay runnable.** The
reason #659 exists at all is that the original figures could not be re-derived when their premise
changed — there was no script, only a number in a README. Committing
`scripts/benchmark_nwp_sort_order.py` beside the existing `scripts/export_baseline_forecasts.py`
means the next change to `NWP_SORT_COLS` or `NWP_WRITER_PROPERTIES` can re-run it in one command
instead of reopening this issue. The cost is one file of roughly 150 lines that nothing else
imports.

## What changes, file by file

### `scripts/benchmark_nwp_sort_order.py` (new)

Builds both arms from the same source rows, so the sort order is the only difference between them:

1. Read 29 consecutive `init_time` partitions from the local NWP table — proposed window
   **2026-08-18 to 2026-09-15 inclusive**, all written by the current `write_nwp` and all after the
   2024-11-12 boundary before which `categorical_precipitation_type_surface` is null.
2. Write each partition twice, into two scratch Delta tables under
   `~/.cache/nged-nwp-sort-benchmark/`: one through the real `write_nwp` (member-early), one
   through the same code path with `NWP_SORT_COLS` monkeypatched to the previous ordering,
   `("init_time", "valid_time", "ensemble_member", "h3_index")`. Git history confirms that is the
   ordering the member-early sort replaced (`dynamical_data`, before commit `83a40738`).
3. For each arm, run the production read — `Nwp.scan_delta(table).filter(init_time in window,
   ensemble_member == 0, h3_index.is_in(cells)).collect(engine="streaming")` — mirroring what
   `load_engineering_inputs` does. The 9 cells come from the `h3_res_5` column of
   `data/NGED/metadata.parquet`, which holds 33 series across 9 unique cells, matching the docs'
   "the 32 V1 series live in just 9 cells".
4. Report, per arm: median wall time over several warm-cache runs after a warm-up, peak resident
   memory, total Parquet bytes on disk, and the number of row groups whose `ensemble_member`
   min/max statistics contain 0.
5. Delete the scratch tables. Peak scratch usage is about 9 GB (2 × 29 × ~145 MB) against 542 GB
   free.

**Three methodological traps the script has to avoid, each of which would silently produce a wrong
number for the docs.**

- **Peak memory needs one subprocess per arm.** `resource.getrusage(...).ru_maxrss` is a
  high-water mark that never falls within a process, so measuring both arms in one process reports
  the larger arm's figure twice. `tracemalloc` is not an alternative: Polars allocates outside
  Python's allocator, so it would see almost none of the usage.
- **Wall time is a warm-cache median, and the write-up must say so.** Row-group skipping saves
  decode work. A cold page cache adds disk IO that both swamps the difference and varies between
  runs, and dropping the page cache needs root, which this script will not ask for.
- **Reading the member-early arm from the production table would confound the comparison.** That
  arm would then differ from the other in file boundaries and row-group boundaries as well as in
  sort order. Both arms are freshly written.

**The row-group count is the figure least likely to rot, and the plan treats it as the headline
mechanism number.** Read each Parquet file's row-group metadata with `pyarrow.parquet.ParquetFile`
and count the row groups whose `ensemble_member` statistics span the control member. That count is
deterministic and machine-independent, and it measures the mechanism the docs actually assert,
where the timing ratio measures one laptop on one afternoon.

### `packages/dynamical_data/README.md`

The "Read path" paragraph becomes the single home of the method and the figures: the window, the
cell count, the member, that the timings are warm-cache medians, the row groups skipped out of the
total, and the note that S3 would do at least as well. **The storage table above it is not
touched** — its rows were measured over 9 partitions spread across the seasons under a different
config sweep, so overwriting them with a 29-partition single-window number would silently change
what the table means. The new storage premium is stated in the read-path paragraph, scoped to those
29 partitions.

### `packages/delta_store/src/delta_store/nwp.py`

The module docstring's "~5x faster and ~5x less peak memory for a ~2% storage cost" and the
`NWP_SORT_COLS` docstring's "need re-measuring" sentence both become the new headline ratio plus
the existing link to the README for the method.

### `docs/architecture/performance.md`

The `ensemble_member` row of the pruning table (line 76) and the paragraph beneath it (line 81)
lose "The exact speed-up needs re-measuring against real data" and gain the measured ratio. The
`h3_index` row (line 77) needs no figure — it asserts only that `h3_index` is not a sort-early
column, which the row-group count will confirm or refute.

### `src/nged_substation_forecast/defs/_engineering_inputs.py`

The `ensemble_member` bullet in `load_engineering_inputs`'s docstring keeps its link to
`performance.md` and gains the ratio.

## Design-philosophy check

**The script is an R&D measurement tool and fails fast, which is the right half of the
inherent-stability rule.** It runs in nobody's serving path, reads the NWP table, and writes only
into a scratch directory it creates and deletes. A missing partition, an empty read, or a scratch
table that fails to write must raise rather than degrade: a benchmark that quietly measures 3
partitions instead of 29 puts a wrong number into five documents, which is worse than not running.

No Patito contract, Delta table, Dagster asset, asset check, or degradation rule changes, so the
production degradation ladder is untouched. Design principle 12, "measure; do not assume", is the
reason for committing the script rather than running it once and deleting it.

## Tests

**One new test, on the only piece of the script that can be wrong without anyone noticing.** The
timings and byte counts are read straight off the clock and the filesystem, and a wrong one shows
up as an implausible number. The row-group-skipping counter is different: a counter that always
returns zero, or that reads the wrong column's statistics, produces a plausible figure that goes
into the docs unchallenged.

`packages/delta_store/tests/test_nwp.py::test_member_early_sort_skips_more_row_groups` — build two
small tables from the existing `_make_nwp` fixture, one through `write_nwp` and one with
`NWP_SORT_COLS` patched to the `valid_time`-first ordering, then assert the counter reports strictly
fewer skippable row groups for the member-early table. **It fails on `main` today** because the
counter function does not exist there. The test needs enough rows and a small enough row-group size
to produce more than one row group, which `_make_nwp(n)` can supply.

Nothing else here is testable, and the plan does not pretend otherwise: a benchmark's output is
checked by reading it. The mechanism it depends on already has a regression test in
`test_scan_pushes_filters_into_the_parquet_scan`.

## Docs to update

The five locations listed under Departure 2. This issue completes no roadmap item, so there is no
"Implementation details" section to delete and no status banner to move.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Plus, specific to this change: run the benchmark itself and paste its output into the PR body, so
the figures in the diff can be checked against the run that produced them.

## Risks and open questions

**Should the measured figures live in one place or five?** Repeating a number in five documents is
how two of them came to be wrong. **Recommendation: the `dynamical_data` README holds the method
and every figure; the other four locations carry at most the one headline ratio and a link.** That
keeps the number where a code reader will see it while leaving one place to edit when it changes.

**Is a committed script worth a new file, or is a one-off run enough?** **Recommendation: commit
it**, for the reason in Departure 3 — the absence of one is why this issue exists. The simplicity
reviewer should attack this specifically.

**What if the new figures are much smaller than 5×?** They may well be: the original 0.15 s → 0.02 s
pair is small enough that fixed costs could dominate. **Recommendation: write in whatever the
measurement says, including "no measurable difference", and keep the row-group count as the
statement of the mechanism.** A sort order that turns out not to pay for its ~2% storage would be a
genuine finding and should be raised as its own issue, not quietly absorbed.

**Is the proposed 29-day window the right one?** Any 29 consecutive partitions are equally current,
since the whole table was rewritten in September 2026. **Recommendation: 2026-08-18 to 2026-09-15**,
named in the README so the figure is reproducible.
