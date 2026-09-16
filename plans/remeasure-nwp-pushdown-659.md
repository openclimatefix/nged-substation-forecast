# Re-measure the NWP predicate-pushdown speed-ups (#659)

**The problem: five places in the repo describe what the member-early on-disk sort buys a
single-member NWP read, and none of them rests on a measurement of the code as it stands.** Three
still assert the original figures — "~5× faster, ~5× less peak memory ... for a ~2% storage cost" —
measured before PR #653 corrected the `Nwp` dtype mismatch, during the period when the cast in
`Nwp.scan_delta` silently defeated Parquet predicate pushdown. The other two had their figures
stripped by #653 and now say the speed-up "needs re-measuring against real data".

**The planned solution: once issue #759 has landed, measure the rewritten table through the
production read path and write the figures into the five locations.** The work is a short probe run
against the local development NWP table, then an edit to five files. No S3 read, no new file, no new
test.

## Sequencing: this issue runs after #759

**Planning this issue found a one-line defect in `write_nwp` and it is being dealt with first, in
issue #759.** `write_nwp` sorts member-early so a single-member read can skip row groups, but
delta-rs splits each partition across two Parquet files and scatters the Arrow chunks, so each row
group ends up holding about 12 *non-contiguous* members. Parquet min/max cannot express a set with
holes, so a row group holding `{0, 1, 4, 5, 6, 15…22}` advertises `[0, 22]`. The measured effect is
that only members at the ends of the range prune at all: the control member reads 14.2% of rows,
while member 25 reads 95.7%. Passing a `target_file_size` large enough to hold a partition in one
file restores the order and brings every member to ~14.5%, storage-neutral. #759 makes that change
and rewrites the 898 stored partitions (about 1 hour serial, measured at 4.2 s per partition).

**Running this issue first would mean documenting a defect we can remove in one line.** The prose
explaining why min/max over-advertises a non-contiguous member set is the hardest writing in this
job, and it would be deleted days later — which CLAUDE.md's "write about the present, not the past"
rule forbids. Worse, the `NWP_SORT_COLS` docstring's existing claim that each row group "spans only
a handful of member values instead of all ~51" is *false today and true after #759*, so waiting
turns a rewrite into an insertion.

**What this issue measures therefore changes.** After #759 the pruning is uniform across members, so
the write-up states one figure that holds for any member, rather than a table showing the benefit
collapsing in the middle of the range. The comparison the issue asks for — member-early against the
`valid_time`-first sort it replaced — is unaffected and still runs.

## Verdict, size and departures

**Verdict: worth doing.** The claims in `performance.md`, `forecast-delivery.md`, and the
`dynamical_data` README are exactly what design principle 12, [measure; do not
assume](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#12-measure-do-not-assume),
exists to keep honest, and three of them state numbers measured under a bug.

**Size: medium.** The change lands in five files — two docstrings, one README paragraph, and two
docs pages. It touches no Patito contract, no Delta table, no asset, and no degradation rule. Both
plan reviews ran. One diff review at implementation, the correctness-and-cut-it-down pass; the
mutation pass is not worth running, because this change pins no behaviour a mutant could break.

### Departure 1: no S3 read, and so no sign-off

**The issue opens by saying the work "requires reading real production NWP data from S3 and needs
the maintainer's explicit sign-off". It does not.** A complete, current NWP Delta table sits on this
machine at `/home/jack/dev/nged-substation-forecast/data/NWP`, reachable from the worktree through
the repo's usual `data` symlink. Checked while planning:

- 123 GB, 898 daily `init_time` partitions, 2024-04-01 to 2026-09-16.
- Every partition file has an mtime of 2026-09-05 or later, so the whole table was written by the
  current `write_nwp`, after PR #653 merged on 2026-08-14.
- The logical schema reads back as `ensemble_member: Int8`, `h3_index: Int64`, `nwp_model_id:
  String`, and `categorical_precipitation_type_surface: Int16`, matching the post-#653 contract.
- `Nwp.scan_delta(path).filter(pl.col("ensemble_member").is_in([0])).explain()` puts
  `SELECTION: col("ensemble_member").cast(Int64).is_in([[0]])` *inside* the Parquet scan.

**A local read is also the better measurement, not merely the cheaper one.** The quantity under test
is how much Parquet decode work row-group skipping avoids. S3 adds network latency that varies run
to run and would swamp the difference. The local figure understates the S3 case, where a skipped row
group also skips a network range request, so it is a floor — the write-up should say so rather than
implying the number transfers.

### Departure 2: five locations carry the claim, not the three the issue names

**PR #653 rewrote three locations but left three other copies of the discredited figures standing,
and none is named in the issue.** Meanwhile one location the issue does name needs nothing.

| Location | State today | Action |
|---|---|---|
| `packages/delta_store/src/delta_store/nwp.py` module docstring, line 21 | Asserts "~5x faster and ~5x less peak memory for a ~2% storage cost" | **Delete** the parenthetical; the sentence already links to the README that will carry the figures |
| `packages/dynamical_data/README.md:49-53` ("Read path") | Asserts "**~5× faster, ~5× less peak memory** (0.15 s / ~1 GB → 0.02–0.04 s / ~205 MB), for a ~2% storage cost" | Rewrite as the single home of method and figures |
| `packages/delta_store/src/delta_store/nwp.py` `NWP_SORT_COLS` docstring | "need re-measuring" | Replace with the measured figure; the "handful of member values" sentence becomes true once #759 lands and stays |
| `docs/architecture/performance.md:76` and `:81` | "needs re-measuring"; a parenthetical two sentences below `:81` makes an unmeasured peak-memory claim | Replace both, and delete the parenthetical |
| `docs/architecture/forecast-delivery.md:477-479` | Asserts "0.02 seconds and uses 205 MB of RAM", then claims that speed "is what lets us run cross-validation across **every ensemble member**" | Replace; after #759 the every-member claim becomes true, which is worth stating rather than deleting |
| `src/nged_substation_forecast/defs/_engineering_inputs.py:78-83` | Carries **no** figure and no "needs re-measuring" sentence — names the mechanism and links to `performance.md` | **Leave untouched** |

**`_engineering_inputs.py` is already correct, and adding a figure there would create another copy of
a number this plan is trying to keep in one place.** The issue lists it because #653 edited it; what
#653 left behind needs nothing.

**The issue's "Not in scope" paragraph points at a file that no longer exists.**
`docs/architecture/adapting-to-another-geography.md` is not in `docs/architecture/`, and nothing
under `docs/` matches `*geograph*`. Nothing to exclude. The issue's own quotation of
`performance.md` ("≈50x less for control-member training") is likewise stale — #653 removed it.

### Departure 3: no committed benchmark script

**The first draft of this plan proposed committing `scripts/benchmark_nwp_sort_order.py`, on the
argument that #659 exists because the original figures could not be re-derived. That argument is
wrong.** The figures went wrong because pushdown silently stopped working under a dtype mismatch, and
re-running a saved script under that bug would have produced the same wrong number. What catches that
failure is a test, and one already exists:
`packages/delta_store/tests/test_nwp.py::test_scan_pushes_filters_into_the_parquet_scan`.

Repo convention agrees. `scripts/` holds lint and check scripts wired into pre-commit, two deploy
scripts named step by step in `docs/live_service/aws.md`, and two pipeline drivers. It holds no
benchmark. Every other measurement claim in this repo — the `BYTE_STREAM_SPLIT` sweep, the
nine-partition storage table, the per-variable `keep_bits` sweep — is a one-off run whose numbers
were written down beside the decision they justified. Principle 12 asks for the measurement written
down next to the decision, not for a harness nobody re-runs. The probe goes into the README as a
fenced block beside the figures it produced, so the method cannot drift from the numbers.

## What changes, file by file

### The probe (run, not committed)

**Run this after #759 has landed and the table has been rewritten**, so it measures the layout the
docs will describe. Re-confirm before starting that a sampled partition really does hold one Parquet
file and that a mid-range member prunes, since those are #759's deliverables and this issue's
premise.

*The layout census is read-only and needs no scratch table.* Read the `ensemble_member` row-group
statistics of partitions sampled across 2024, 2025, and 2026, decode the column to count the members
each row group actually holds, and report rows decoded for a control member and for a mid-range
member. Sampling across eras matters: before #759 the control-member figure was stable at 14.1–16.2%
while the mid-range figure swung from 78% to 96%, so the write-up must not quote a single era as
though it were table-wide.

*The sort-order comparison writes two arms* from one partition read back through `Nwp.scan_delta`:

1. **Member-early**, through the real `write_nwp`.
2. **`valid_time`-first**, with `delta_store.nwp.NWP_SORT_COLS` monkeypatched to `("init_time",
   "valid_time", "ensemble_member", "h3_index")` — git history confirms that is the ordering the
   member-early sort replaced (`83a40738^:.../convert_to_polars.py:60`). Patch the module global,
   not a local `from ... import` binding; `write_nwp` reads it at call time.

**Both arms must be freshly written**, and this matters more than it looks: before #759 delta-rs's
reordering was non-deterministic, so a fresh write of a partition did not reproduce the stored
copy's row-group boundaries. Comparing a fresh arm against the stored table would confound sort
order with that non-determinism.

*The timing arm uses the 29-day production read shape*, since a single partition returns 765 rows
and would measure fixed costs rather than decode. Write 29 partitions per arm and time
`Nwp.scan_delta(table).filter(...).collect(engine="streaming")` with the filter shape
`load_engineering_inputs` actually uses
(`src/nged_substation_forecast/defs/_engineering_inputs.py:137-147`): bounded `init_time`, bounded
`valid_time`, and `pl.col("ensemble_member").is_in([0])` — not `== 0`. The 9 cells come from the
`h3_res_5` column of `data/NGED/metadata.parquet`, which holds 33 series across 9 unique cells.
Report a warm-cache median after a warm-up.

Measure peak resident memory once per arm, each in **its own subprocess**:
`resource.getrusage(...).ru_maxrss` is a high-water mark that never falls within a process, so both
arms in one process would report the larger figure twice. `tracemalloc` is not an alternative —
Polars allocates outside Python's allocator.

Scratch usage peaks at about **8.4 GB** (2 arms × 29 partitions × ~145 MB), under
`~/.cache/nged-nwp-sort-benchmark/` and deleted afterwards. Not `/tmp`, which is a 31 GB tmpfs.
There is 542 GB free.

### `packages/dynamical_data/README.md`

The "Read path" paragraph becomes the single home of the figures and the method: rows decoded for a
single-member read, that the figure holds for any member rather than only the control member, the
storage premium over a `valid_time`-first sort, the window and sample it was measured on, and the
note that S3 would do at least as well. The probe follows as a fenced block.

**The nine-partition storage table above it is not touched** — its rows were measured across the
seasons under a different config sweep, so overwriting them with a single-window number would
silently change what the table means.

### `packages/delta_store/src/delta_store/nwp.py`

Delete the module docstring's figure parenthetical, keeping the link. Replace `NWP_SORT_COLS`' "need
re-measuring" sentence with the measured figure. Its "handful of member values" sentence needs no
change once #759 has landed, but the docstring should say that the claim depends on a partition
landing in one file — which is #759's `target_file_size` — so the two constants are not edited
independently later.

### `docs/architecture/performance.md`

Line 76's "The exact speed-up needs re-measuring against real data" and line 81's equivalent gain the
measured figure. Line 81's "peak-memory reduction versus a `valid_time`-first sort" clause is deleted
as unmeasured, and so is the parenthetical "(the member-early sort only lowers this)" two sentences
later, which makes the identical claim — deleting one and leaving the other would be half a fix.

### `docs/architecture/forecast-delivery.md`

The "It's fast to query, even on a laptop" bullet loses the `0.02 s` / `205 MB` pair and gains the
measured figure. Its next sentence — "That speed is what lets us run cross-validation across every
ensemble member, for every fold, on a laptop" — is false today and true after #759, so it stays and
gains the evidence it currently lacks. This page is written for an outside reader, so the correction
matters most here.

The same bullet's neighbour says the local table is "821 daily runs, ~5.9 billion rows, April 2024 to
July 2026 — **~93 GB**". It is now 898 partitions and 123 GB. Strictly outside the issue, but it is
the adjacent line in the bullet being rewritten, and CLAUDE.md requires docs to describe the present.

## Design-philosophy check

No Patito contract, Delta table, Dagster asset, asset check, or degradation rule changes, so the
production degradation ladder is untouched. The probe is an R&D measurement: it reads the NWP table
and writes only into a scratch directory it creates and deletes, and it should raise rather than
degrade on a missing partition or an empty read, because a benchmark that quietly measures the wrong
thing puts a wrong number into five documents.

## Tests

**None, and the plan does not pretend otherwise.** The deliverable is a measurement and a prose
correction; there is no new behaviour for a test to pin. The first draft proposed a test of a
row-group-skipping counter, which only existed because that draft committed the counter.
`test_scan_pushes_filters_into_the_parquet_scan` keeps guarding the half of the mechanism that
already held — that the predicate reaches the Parquet scan — and #759 owns the test changes that go
with the layout, including replacing `test_on_disk_format`'s per-file `is_sorted()` assertion, which
passes today only because `_make_nwp()` writes 6 rows into a single row group.

## Docs to update

The five locations in Departure 2. This issue completes no roadmap item, so there is no
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

The probe's own output goes in the PR body, so the figures in the diff can be checked against the run
that produced them.

## Risks and open questions

**This plan is blocked on #759 and should not start before it lands.** If #759 is descoped or its
rewrite of the stored partitions is deferred, this plan needs revisiting rather than running as
written: the figures would then differ between partitions written before and after the change, and
the docs would have to describe two layouts.

**How should the figures be quoted?** **Recommendation: name the sample.** Quoting one era's number
as a table-wide constant is the same mistake this issue exists to correct, one era later. State the
partitions sampled and the spread across them.

**What if the wall-time difference is not measurable?** The 29-day read returns about 22,000 rows, so
fixed costs may dominate and the two arms may time the same despite the decode difference.
**Recommendation: say exactly that, and lead with rows decoded rather than a wall-time ratio.** "No
measurable difference at this scale" is a finding, not a failed measurement.
