# Re-measure the NWP predicate-pushdown speed-ups (#659)

**The problem: five places in the repo describe what the member-early on-disk sort buys a
single-member NWP read, and every one of them is wrong.** Two still assert the original figures —
"~5× faster, ~5× less peak memory ... for a ~2% storage cost" — measured before PR #653 fixed the
`Nwp` dtype mismatch, during the period when the cast in `Nwp.scan_delta` silently defeated Parquet
predicate pushdown. The other two had their figures stripped by #653 and now say the speed-up
"needs re-measuring against real data". All four also state a *mechanism* that measurement shows is
false: that sorting member-early makes each row group span "only a handful of member values instead
of all ~51".

**The planned solution: measure the real table, then write down both the numbers and the corrected
mechanism.** Row-group pruning does happen — a control-member read decodes 14% of the rows — but
not for the reason the docs give, and not for any member except the two at the ends of the range.
The work is a short probe run against the local development NWP table, then an edit to four
locations. No S3 read, no new file, no new test.

## Verdict, size and departures

**Verdict: worth doing, and worth more than the issue expects.** The issue asks for a number. The
measurement supplies one, and also shows that the stated mechanism behind it is wrong and that the
benefit does not generalise to the ensemble members a future per-member training run would ask for.
That is exactly what design principle 12, [measure; do not
assume](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#12-measure-do-not-assume),
is for.

**Size: medium.** The change lands in four files — two docstrings, one README paragraph, and two
lines of a docs page. It touches no Patito contract, no Delta table, no asset, and no degradation
rule. Both plan reviews are worth running, the simplicity one because the first draft of this plan
proposed a committed harness and a new test that measurement showed were unnecessary, the
correctness one because the revised mechanism claim is now the load-bearing part. One diff review,
the correctness-and-cut-it-down pass. The mutation pass is not worth running: this change pins no
behaviour a mutant could break.

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

**A local read is also the better measurement, not merely the cheaper one.** The quantity under
test is how much Parquet decode work row-group skipping avoids. S3 adds network latency that varies
run to run and would swamp the difference. The local figure understates the S3 case, where a skipped
row group also skips a network range request, so it is a floor — the write-up should say so rather
than implying the number transfers.

### Departure 2: four locations carry the claim, not the three the issue names

**PR #653 rewrote three locations but left three other copies of the discredited figures standing,
and none is named in the issue.** Meanwhile one location the issue does name needs nothing.

| Location | State today | Action |
|---|---|---|
| `packages/delta_store/src/delta_store/nwp.py` module docstring, line 21 | Asserts "~5x faster and ~5x less peak memory for a ~2% storage cost" | **Delete** the parenthetical; the sentence already links to the README that will carry the figures |
| `packages/dynamical_data/README.md:49-53` ("Read path") | Asserts "**~5× faster, ~5× less peak memory** (0.15 s / ~1 GB → 0.02–0.04 s / ~205 MB), for a ~2% storage cost" | Rewrite as the single home of method, figures, and corrected mechanism |
| `packages/delta_store/src/delta_store/nwp.py` `NWP_SORT_COLS` docstring | "need re-measuring"; also states the false "handful of member values" mechanism | Replace both |
| `docs/architecture/performance.md:76` and `:81` | "needs re-measuring"; `:81` repeats the false mechanism, and a parenthetical two sentences later makes the same unmeasured peak-memory claim | Replace all three |
| `docs/architecture/forecast-delivery.md:477-479` | Asserts "0.02 seconds and uses 205 MB of RAM", then claims that speed "is what lets us run cross-validation across **every ensemble member**" | Replace; the every-member claim is the one the measurement refutes |
| `src/nged_substation_forecast/defs/_engineering_inputs.py:78-83` | Carries **no** figure and no "needs re-measuring" sentence — names the mechanism and links to `performance.md` | **Leave untouched** |

**`_engineering_inputs.py` is already correct, and adding a figure there would create a fourth copy
of a number this plan is trying to keep in one place.** The issue lists it because #653 edited it;
what #653 left behind needs nothing.

**The issue's "Not in scope" paragraph points at a file that no longer exists.**
`docs/architecture/adapting-to-another-geography.md` is not in `docs/architecture/`, and nothing
under `docs/` matches `*geograph*`. Nothing to exclude. The issue's own quotation of
`performance.md` ("≈50x less for control-member training") is likewise stale — #653 removed it.

### Departure 3: no committed benchmark script

**The first draft of this plan proposed committing `scripts/benchmark_nwp_sort_order.py`, on the
argument that #659 exists because the original figures could not be re-derived. That argument is
wrong.** The figures went wrong because pushdown silently stopped working under a dtype mismatch, and
re-running a saved script under that bug would have produced the same wrong number. What catches
that failure is a test, and one already exists:
`packages/delta_store/tests/test_nwp.py::test_scan_pushes_filters_into_the_parquet_scan`.

Repo convention agrees. `scripts/` holds lint and check scripts wired into pre-commit, two deploy
scripts named step by step in `docs/live_service/aws.md`, and two pipeline drivers. It holds no
benchmark. Every other measurement claim in this repo — the `BYTE_STREAM_SPLIT` sweep, the
nine-partition storage table, the per-variable `keep_bits` sweep — is a one-off run whose numbers
were written down beside the decision they justified. Principle 12 asks for the measurement written
down next to the decision, not for a harness nobody re-runs. The probe goes into the README as a
fenced block beside the figures it produced, so the method cannot drift from the numbers.

## What the measurement already shows

Run while planning, against 10 real September 2026 partitions (72 row groups, 72.4M rows), reading
each Parquet file's `ensemble_member` row-group statistics and then decoding the column to see what
those row groups actually hold:

| Requested member | Rows physically present | Rows decoded | Over-read |
|---:|---:|---:|---:|
| 0 (control) | 14.2% | **14.2%** | **1.00×** |
| 5 | 18.7% | 53.1% | 2.83× |
| 10 | 17.3% | 75.9% | 4.38× |
| 25 | 28.8% | **95.7%** | 3.33× |
| 40 | 16.9% | 68.1% | 4.03× |
| 50 | 15.6% | 15.6% | **1.00×** |

**Each row group holds about 12 of the 51 members — "a handful", as the docs say — but the 12 are a
scattered, non-contiguous set, which is the part the docs get wrong.** Row group 0 of
`init_time=2026-09-10` holds members `{0, 1, 4, 5, 6, 15…22}`. Parquet min/max statistics cannot
express a set with holes in it, so that row group advertises `[0, 22]` and admits 10 members it does
not contain. Across the 10 partitions the median row group holds 12 distinct members while
advertising a span of 35.5 (maximum span 51).

**The pruning that does happen is therefore an artifact of the control member sitting at the end of
the range.** The over-read column measures it exactly: min/max is exact at members 0 and 50 (only a
row group that contains 0 can have a minimum of 0) and wrong by 2.8 to 4.4× everywhere in between.
**A per-member training run that asks for a mid-range member gets essentially no row-group
pruning** — member 25 decodes 96% of the rows. This is the most valuable thing the measurement
produces, and no document currently says it.

**The cause is delta-rs reordering the batches, not a bad sort — and fixing it is free.** One
`(init_time, ensemble_member)` slice is 7,243,785 ÷ 51 = 142,035 rows, so a 1,048,576-row row group
spans 7.4 slices; an order-preserving write should give 7–9 contiguous members per row group and
~14.5% rows decoded *for every member*. Writing the same sorted Arrow table with
`pyarrow.parquet.write_table` instead of through delta-rs confirms it, at no storage cost:

| Write path | Rows decoded, member 0 | member 10 | member 25 | Size |
|---|---:|---:|---:|---:|
| `write_nwp` today (delta-rs) | 14.5% | 75.9% | 95.7% | 144.80 MB |
| Order-preserving (`pyarrow`) | 14.5% | **14.5%** | **14.5%** | 144.83 MB |

`write_nwp` sorts correctly — the frame is monotone in `ensemble_member` and `to_arrow()` yields 32
in-order chunks — and delta-rs then splits the rows across two files and redistributes the chunks
non-contiguously. Member 0's block lands in the *second* file, so this is genuine stream reordering
rather than a size-ordered split. It is also non-deterministic: a fresh write of the same partition
split 5,217,728/2,026,057 where the stored copy split 5,231,063/2,012,722.

**Scope note: the figures vary across the table's two and a bit years, and the plan must publish
them accordingly.** The control-member figure is stable (14.1–16.2% sampled across 2024, 2025, and
2026), so it can be stated as a table-wide range. The mid-range figures are not: member 25 ranges
from 78% to 96% and the median span from 28 to 35.5, so they are published as one 10-partition
sample, not as a table-wide constant. The qualitative conclusion holds at every era sampled.

## What changes, file by file

### The probe (run, not committed)

**The probe has two halves, measured over different amounts of data, because the two questions need
different amounts.**

*The layout census is read-only and needs no scratch table at all.* Read the `ensemble_member`
row-group statistics of real partitions sampled across 2024, 2025, and 2026, decode the column to
count the members each row group actually holds, and produce the rows-present / rows-decoded /
over-read table above. This is where every published percentage comes from.

*The sort-order comparison writes three arms* from one partition read back through `Nwp.scan_delta`:

1. **Member-early**, through the real `write_nwp`.
2. **`valid_time`-first**, with `delta_store.nwp.NWP_SORT_COLS` monkeypatched to `("init_time",
   "valid_time", "ensemble_member", "h3_index")` — git history confirms that is the ordering the
   member-early sort replaced (`83a40738^:.../convert_to_polars.py:60`). Patch the module global,
   not a local `from ... import` binding; `write_nwp` reads it at call time.
3. **Order-preserving**, the identical sorted Arrow table written with
   `pyarrow.parquet.write_table`. This is the control that separates "the sort is wrong" from
   "delta-rs reorders it", and it is what makes the follow-up issue decidable.

**Both written arms must be freshly written, and this matters more than it looks:** delta-rs's
reordering is non-deterministic, so a fresh write of a partition does not reproduce the stored
copy's row-group boundaries. Comparing a fresh arm against the production table would confound sort
order with that non-determinism.

*The timing arm uses the 29-day production read shape*, since a single partition returns 765 rows
and would measure fixed costs rather than decode. Write 29 partitions per arm and time
`Nwp.scan_delta(table).filter(...).collect(engine="streaming")` with the filter shape
`load_engineering_inputs` actually uses (`src/nged_substation_forecast/defs/_engineering_inputs.py:137-147`):
bounded `init_time`, bounded `valid_time`, and `pl.col("ensemble_member").is_in([0])` — not `== 0`.
The 9 cells come from the `h3_res_5` column of `data/NGED/metadata.parquet`, which holds 33 series
across 9 unique cells. Report a warm-cache median after a warm-up.

Measure peak resident memory once per arm, each in **its own subprocess**:
`resource.getrusage(...).ru_maxrss` is a high-water mark that never falls within a process, so both
arms in one process would report the larger figure twice. `tracemalloc` is not an alternative —
Polars allocates outside Python's allocator.

Scratch usage peaks at about **8.7 GB** (2 arms × 29 partitions × ~145 MB, plus one partition for
the third arm), under `~/.cache/nged-nwp-sort-benchmark/` and deleted afterwards. Not `/tmp`, which
is a 31 GB tmpfs. There is 542 GB free.

### `packages/dynamical_data/README.md`

The "Read path" paragraph becomes the single home of the figures, the method, and the corrected
mechanism: rows decoded for a control-member read, the storage premium, the member-survival table
above, the note that the row groups hold a scattered ~35-member subset rather than a contiguous
block, and the note that S3 would do at least as well. The probe follows as a fenced block.

**The nine-partition storage table above it is not touched** — its rows were measured across the
seasons under a different config sweep, so overwriting them with a single-window number would
silently change what the table means.

### `packages/delta_store/src/delta_store/nwp.py`

Delete the module docstring's figure parenthetical, keeping the link. Replace `NWP_SORT_COLS`'
"need re-measuring" sentence with the measured rows-decoded figure, and replace its "handful of
member values" sentence with what the row groups actually contain and why the control member still
prunes.

### `docs/architecture/performance.md`

Line 76's "The exact speed-up needs re-measuring against real data" and line 81's equivalent gain
the measured figure. Line 81's "each ~1M-row row group spans only a handful of members" is corrected
the same way as the docstring. Its "peak-memory reduction versus a `valid_time`-first sort" clause
is deleted as unmeasured, and so is the parenthetical "(the member-early sort only lowers this)" two
sentences later, which makes the identical claim — deleting one and leaving the other would be half
a fix.

### `docs/architecture/forecast-delivery.md`

The "It's fast to query, even on a laptop" bullet loses the `0.02 s` / `205 MB` pair and gains the
measured control-member figure. **Its next sentence needs more than a number swapped.** "That speed
is what lets us run cross-validation across every ensemble member, for every fold, on a laptop" is
precisely the claim the measurement refutes: every member other than 0 and 50 gets little or no
row-group pruning. This page is written for an outside reader, so the correction matters most here.

The same bullet's neighbour says the local table is "821 daily runs, ~5.9 billion rows, April 2024
to July 2026 — **~93 GB**". It is now 898 partitions and 123 GB. Strictly this is outside the
issue, but it is the adjacent line in the bullet being rewritten, and CLAUDE.md requires docs to
describe the present.

## Design-philosophy check

No Patito contract, Delta table, Dagster asset, asset check, or degradation rule changes, so the
production degradation ladder is untouched. The probe is an R&D measurement: it reads the NWP table
and writes only into a scratch directory it creates and deletes, and it should raise rather than
degrade on a missing partition or an empty read, because a benchmark that quietly measures the
wrong thing puts a wrong number into four documents.

## Tests

**None, and the plan does not pretend otherwise.** The deliverable is a measurement and a prose
correction; there is no new behaviour for a test to pin. The first draft proposed a test of a
row-group-skipping counter, which only existed because that draft committed the counter — and
`_make_nwp` cannot produce a realistic multi-row-group frame anyway, so the test would have asserted
a property the real write path does not have and gone green. `test_scan_pushes_filters_into_the_parquet_scan` keeps guarding the half of the mechanism that
still holds — that the predicate reaches the Parquet scan.

**`test_on_disk_format` must not be cited as covering the other half.** It asserts
`key["k"].is_sorted()` per Parquet file and passes only because `_make_nwp()` writes 6 rows into a
single row group; every real file fails that assertion, including one freshly written by
`write_nwp`. That is the "a green suite proves nothing" case exactly, and closing it belongs with
the layout fix below rather than here, because a test written now would pin behaviour that issue is
about to change.

## Docs to update

The four locations in Departure 2. This issue completes no roadmap item, so there is no
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

The probe's own output goes in the PR body, so the figures in the diff can be checked against the
run that produced them.

## Risks and open questions

**The measurement found a free 6× read speed-up for every non-control ensemble member, and whether
to take it is the maintainer's call.** `write_nwp` sorts correctly; delta-rs then scatters the Arrow
chunks so the row groups hold ~12 non-contiguous members instead of ~8 contiguous ones. An
order-preserving write recovers ~14.5% rows decoded for *every* member at 144.83 MB against today's
144.80 MB — storage-neutral within noise. **Recommendation: open it as its own issue and sequence it
after #659**, so this issue records what the table does today and that issue changes it. Doing both
at once would mean #659 publishes numbers its own PR invalidates. That follow-up should also carry
the `test_on_disk_format` gap, since the test and the layout have to change together.

**Should the member-survival table go in the README, or is it more than the docs need?**
**Recommendation: include it.** It is the only evidence that the benefit does not generalise beyond
the control member, and `forecast-delivery.md` currently promises the opposite to an outside reader.

**How should the varying figures be quoted?** **Recommendation: the control-member figure as a
table-wide range (14–16%, sampled across 2024, 2025, and 2026), and the mid-range figures as one
10-partition sample**, named as such. Quoting member 25's 95.7% as a table-wide constant would be
the same mistake this issue exists to fix, one era later.

**What if the wall-time difference is not measurable?** The 29-day read returns about 22,000 rows,
so fixed costs may dominate and the two arms may time the same despite the decode difference.
**Recommendation: say exactly that, and lead with rows decoded rather than a wall-time ratio.**
"No measurable difference at this scale" is a finding, not a failed measurement, and it is the
honest reading of a change that removes 85% of the decode work from a read that was already taking
hundredths of a second.
