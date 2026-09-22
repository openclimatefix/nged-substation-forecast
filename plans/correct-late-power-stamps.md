# Correct NGED's half-hour-late power stamps (#790)

**The problem.** NGED stamped every half-hourly power reading half an hour late until 08:30 UTC on
26 March 2026, and have stamped every reading correctly since, without marking the change. `PowerTimeSeries.time` is
documented as period-ending — the value stamped `T` is the mean over `(T − 30 min, T]` — and for
roughly 93% of the rows in our `power_time_series` Delta table that is false: the value stamped `T`
is the mean over `(T − 60 min, T − 30 min]`. Every model trained on that table has learnt a
half-hour timing error, and the error is not uniform across the record. The measurement, and the
three independent signals that pin the changepoint, are in the [beam/diffuse
appendix](https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/#the-power-timestamps-before-26-march-2026-are-half-an-hour-late).

**The planned solution.** Correct the stamps at the ingest boundary, in
`read_nged_json.py`, following the `PowerTimeSeries.drop_implausible_rows` precedent that already
repairs this feed in the same function. The stored table then means what the contract says it means,
and no read site can be silently wrong. The existing on-disk table is not migrated: it is dropped
and re-materialised from NGED's bucket after this merges, on the workstation and on AWS. That is the
maintainer's decision, and it is what makes the ingest the right home for the correction — a
re-materialise re-parses every historic file, so without the ingest guard the rebuild would
reintroduce the fault it is being run to remove.

The change is three edits and no new mechanism: one constant, one classmethod, one call.

## Verdict, size and departures

**Verdict: worth implementing, and the fault is real.** The issue's account of the current code is
accurate: `read_nged_json.py:85` renames `endTime` to `time` and stores it unchanged, and nothing
downstream corrects it. The changepoint instant, the direction and the magnitude are already settled
and published on `main`; this issue is the engineering that acts on them, not a re-measurement.

**Size: complex.** Answering each of the five triggers in turn:

| Trigger | Answer |
|---|---|
| Changes what gets stored | **Fires.** Every pre-changepoint `time` in the `power_time_series` table changes by 30 minutes. |
| Touches the production serving path | **Fires.** `live_forecasts` reads power through `load_engineering_inputs`, off the table this change rewrites. The change is an identity on every row that path reads (see the design-philosophy check), but the trigger fires on the surface touched, not on the measured effect. |
| Touches a degradation rule | Does not fire. No degradation ladder, asset check or warning path is added or edited. |
| More than one defensible design | **Fires.** Correct-at-ingest and correct-at-read are both defensible, and the issue asks for the choice explicitly. |
| Callers you could not name without searching | **Fires.** Eight modules touch the power table, and the list came from a grep. |

Four of five fire, so the issue gets a plan, both plan reviews, and both diff reviews.

**Departure 1 — the issue frames this as a choice between a backfill and a rule every consumer
knows. Neither cost is paid.** The issue's "What to decide" section weighs "correct on ingest, and
pay for a one-off backfill" against "correct on read, and make every consumer know the rule". The
maintainer has since decided to drop the table and re-materialise it rather than migrate it, which
removes the backfill from the ingest option altogether and leaves it strictly simpler than the read
option on every axis: two files edited instead of nine, one new public name instead of three, no
second meaning for `pt.DataFrame[PowerTimeSeries]`, and no door for a future read site to bypass.

**Departure 2 — nothing is added to `packages/delta_store/` or to the read path.** No
`scan_delta` classmethod, no correction helper, no pre-commit guard, and no edit to
`_engineering_inputs.py`, `cv_assets.py`, either dashboard notebook, or
`export_baseline_forecasts.py`. All six read sites keep reading the table with a bare scan, which is
correct once the table itself is correct.

**Departure 3 — the retrain and the NGED conversation are not in this issue's scope.** The issue
lists both under "What to decide". Retraining is a run of the existing CV assets once the table has
been rebuilt, not a code change; asking NGED whether the correction was fleet-wide is a human
action. Both are recorded under "Risks and open questions" so neither is lost.

## What changes, file by file

### `packages/contracts/src/contracts/power_schemas.py`

- **Add `POWER_STAMPS_CORRECTED_AT: Final[datetime] = datetime(2026, 3, 26, 8, 30, tzinfo=UTC)`**, a
  module-level constant whose docstring states what the fault is, which three measurements pin the
  instant, and links to the published appendix. The same instant is already declared twice on the
  unmerged `beam-diffuse-split-experiment` branch, under the name `ALIGNMENT_FIXED_AT`
  (`build_dataset.py:101`, `site_e_commissioning.py:51`); this is its permanent home, and that
  branch should import and rename rather than re-declare once both have merged.

- **Add `PowerTimeSeries.correct_late_stamps(dataframe) -> pl.DataFrame`**, a classmethod applying
  `pl.when(pl.col("time") < POWER_STAMPS_CORRECTED_AT).then(pl.col("time").dt.offset_by("-30m")).otherwise(pl.col("time"))`.

  It sits beside `drop_implausible_rows` (`power_schemas.py:113`) and is documented the same way:
  an ingest-boundary repair of a confirmed upstream fault, to be called only at the boundary that
  receives NGED's raw JSON, before `validate`. `drop_implausible_rows` is the precedent this change
  follows throughout — a repair of this feed, living as a method on this model, called from
  `read_nged_json.py`.

- **Extend the `time` field description** to say that the ingest corrects NGED's known half-hour
  offset on readings before `POWER_STAMPS_CORRECTED_AT`, so a reader of the contract can tell that
  the stored value differs from NGED's own `endTime` for those rows, and why. The description's
  existing claim — that `time` ends the 30-minute observation period — becomes true of the whole
  table rather than of 5% of it.

### `packages/nged_data/src/nged_data/read_nged_json.py`

- **Call `PowerTimeSeries.correct_late_stamps` inside `_extract_power_time_series`**, after the
  `str.to_datetime` on lines 89-91 and before `drop_implausible_rows` on line 97. Order matters,
  and the comment should name the real consequence: correcting first means `drop_implausible_rows`
  judges the stamp that will actually be stored. Reversed, a row whose corrected stamp falls outside
  the plausible range survives the drop and then fails `PowerTimeSeries.validate`
  (`power_schemas.py:87` → `common.py:116`), which raises. `download_and_parse_files` catches only
  the null-`data` `InvalidOperationError` (`storage.py:236-245`), so that `ValueError` reaches the
  asset's `BaseException` guard (`assets.py:153-165`) and costs three retries and a failed run. The
  chosen order is the fail-open one; the reversed order turns malformed external data into an
  ingest outage, which is exactly what `drop_implausible_rows` exists to prevent.

- The `:00`/`:30` alignment check is unaffected: a 30-minute shift maps the half-hour grid onto
  itself.

### Nothing else changes

Stated explicitly, because a reviewer should see these as decisions rather than omissions:

- **No read site changes.** `_engineering_inputs.py:125`, `cv_assets.py:198`, `cv_assets.py:986`,
  `view_forecasts.py:311`, `map_and_timeseries.py:121` and `export_baseline_forecasts.py:121` all
  keep their bare scans.
- **`select_new_rows` and `time_series_coverage` are untouched.** Both compare parsed rows against
  stored rows; after this change both sides carry corrected stamps, so the anti-join and the
  watermark stay consistent with no edit.
- **No migration script.** The table is dropped and re-materialised.

## Design-philosophy check

**This code path runs in production, and the change cannot degrade the live service.**
`live_forecasts` widens its window by `LIVE_POWER_HISTORY`, 15 days
(`production_assets.py:57`, applied at `:329`), so every row it reads is post-changepoint, where the
correction is an identity. It does not pass `power_lookback`, which defaults to `timedelta(0)`. The live service's behaviour is unchanged by this merge, and stays unchanged until a model
is retrained on the rebuilt table.

**The change is very nearly inert until the table is rebuilt, but not entirely, and the exception
is worth stating precisely.** `select_new_rows`' file-listing branch keeps a file when its
`end_time` falls within `_LATE_FILE_LOOKBACK` — 3 days (`storage.py:361`) — of *that series' own*
on-disk `last_time`. A series is therefore inert under this change if and only if every file inside
its own 3-day window carries post-changepoint stamps only. That holds for every series still
reporting, whose trailing 3 days sit months after the changepoint. It fails for a series whose
`last_time` is itself pre-changepoint, because then the 3-day window reaches back into 2026 or
earlier and NGED's bulk history file for that series is re-listed and re-parsed every hour — as it
already is on `main` today, where the anti-join discards every row because the keys match.

**Two dead series are in that state, and the merge appends about 54 rows to them.** Series 33 is the
out-of-service meter `checks.py:142` silences, with no readings since 2026-01-26, and series 32
holds a single 2024 reading. After the merge their re-parsed stamps move 30 minutes earlier, so the
anti-join key becomes `(time_series_id, T − 30 min)`, which matches nothing wherever the series has
a gap at `T − 30 min`. Those rows are appended, leaving two dead series holding a mixture of
corrected and uncorrected stamps until the rebuild erases them. The append is one-off: on the next
hourly run the corrected keys exist and the anti-join discards them. A new `time_series_id`, whose
null `last_time` downloads its whole history unconditionally (`storage.py:496`), would land in the
same state — no duplicates, but a corrected series inside an uncorrected table.

**None of that is a defect in the change, and no code guards against it. The rebuild is the
guard.** The harm is bounded to two series that are out of service, and it is why the rebuild should
follow the merge promptly rather than at leisure. It goes in the PR body so that whoever runs the
rebuild is not surprised by a row count that moved before they touched anything.

**Nothing raises and nothing degrades.** `correct_late_stamps` is a pure Polars expression with no
I/O, no branch on data availability, and no failure mode short of the `time` column being absent,
which the schema forbids. No asset check is added or edited.

**This trades away design principle 15, and the trade is the maintainer's explicit decision.**
Principle 15 — "Transform data in feature engineering, not in the ingest, unless it saves a lot of
storage" — makes feature engineering the default home for a transform, and this plan puts a
transform in the ingest against it. What is bought is that `PowerTimeSeries` stays a true account of
the table it governs, so no present or future read site can be silently wrong, and that the whole
change is three edits rather than nine. What is given up is that revising the constant costs a
re-materialise rather than a config change, which is the cost principle 15 exists to warn about.
Two things make it the right trade here. The maintainer has chosen re-materialise-over-migrate as
the standing maintenance route for this table, so that cost is already accepted rather than newly
incurred. And principle 15's own test is what a transform *destroys*: this one destroys nothing that
the recorded constant cannot restore, because a 30-minute shift is exactly invertible.

**Hypotheses.** No hypothesis in `engineering-hypotheses.md` is delivered by this change. It is a
data-correctness fix, not a claim about the engineering.

## Tests

New and changed tests, each named with the assertion that fails on `main` today.

In `packages/contracts/tests/test_power_time_series.py` (the existing home for this model's tests —
do not open a second file):

1. **A stamp before the changepoint moves back 30 minutes.** `correct_late_stamps` on a row stamped
   `2026-03-26 08:00Z` returns `2026-03-26 07:30Z`. Fails on `main`: the method does not exist.
2. **A stamp at the changepoint and a stamp after it are unchanged.** Rows stamped
   `2026-03-26 08:30Z` and `2026-03-26 09:00Z` come back identical. This is the boundary the whole
   change turns on, and a `<=` where the rule wants `<` passes test 1 and fails this one.
3. **A corrected multi-series frame still validates.** Apply `correct_late_stamps` to a frame
   spanning the changepoint for two series and call `PowerTimeSeries.validate()`. Kept as a cheap
   end-to-end invariant, not as proof of uniqueness: shifting a contiguous prefix by a constant
   −30 min is injective, and the shifted range ends at 08:00 while the unshifted range starts at
   08:30, so a duplicate key is unreachable by construction. What it does catch is an implementation
   that shifts the wrong side of the boundary, or shifts by the wrong sign.

In `packages/nged_data/tests/test_read_nged_json.py`:

4. **A parsed pre-changepoint reading is stored 30 minutes earlier than its `endTime`.** Write a
   new test rather than leaning on the existing fixture at lines 121-123: that test asserts only
   `n_dropped`, `height` and `power` (lines 131-134) and no `time` value at all, so it passes on
   `main` and after the change alike. The new test parses one pre-changepoint reading and one
   post-changepoint reading and asserts both stored `time` values, pinning both branches at the
   boundary the ingest actually crosses. Fails on `main`, which stores `endTime` unchanged.
5. **A reading whose corrected stamp falls out of the plausible range is dropped, not raised on.**
   There is exactly one such reading, and the test must use it: `endTime` of
   `2000-01-01T00:00:00Z`, which sits on `MIN_PLAUSIBLE_DATETIME` (`common.py:25`) while its
   corrected stamp, `1999-12-31 23:30Z`, does not. The `MAX` side is unreachable, because the shift
   only ever moves a stamp earlier. Assert the row is dropped and `n_dropped` counts it. Fails on an
   implementation that drops before correcting, where the row survives the drop and then raises out
   of `validate` — so this test discriminates the ordering bug by the exception, and it is the only
   test that would.

In `tests/test_assets.py`: the ingest-asset fixture at line 323 uses an `endTime` of
`2026-03-05 12:30:00+0000`, and line 346 asserts the stored `time` is `12:30`. That assertion
becomes `12:00`. It is the one existing assertion in the repo that changes, and it is a valuable
one — an end-to-end run of the Dagster asset, not of the parser alone. No new assertion; the
existing one is updated and gains a comment naming why.

## Docs to update

- **`docs/results/beam-diffuse-split.md:894-895`** — states "the ingest takes `endTime`
  unchanged", which this change makes false. This page does not mention `power_time_series`, so the
  grep below does not find it; it is named here because it is the one published page that documents
  the behaviour being changed.
- **`packages/contracts/README.md:124-131`** — describes `drop_implausible_rows` at length as *the*
  ingest-boundary repair of this feed. Add a sentence for `correct_late_stamps`, or the README names
  one of two boundary repairs.
- **`docs/roadmap/data-cleaning.md`** — add a section for the stamp correction alongside the
  commissioning ramp and the export cap, in the present tense: what the fault was, that the ingest
  now corrects it, and that a corrected series carries no reading at 08:00 on 26 March 2026 (see
  risk 4). This page catalogues the trial-area data faults and is currently silent on the one fault
  we have fixed.
- **`src/nged_substation_forecast/defs/assets.py`** — `power_time_series_and_metadata`'s docstring
  says "Nothing cleans this data further". That stays true, but the docstring should now say the
  ingest corrects the stamp offset, because a Dagster asset docstring is operator documentation and
  an operator rebuilding this table needs to know the stored stamps are not NGED's own.
- **`docs/architecture/` and `docs/ml_experimentation/dagster-workflow.md`** — a grep for
  `power_time_series` across `docs/` returns 13 files. The implementer reads them and edits only
  those left inconsistent, rather than working from a guessed list here.
- **No roadmap status banner moves.** This issue completes no roadmap item:
  `docs/roadmap/data-cleaning.md` stays 🚧 Planned, because the commissioning ramp and the
  shape-change detection are still open.

## Verification commands

The green-before-push set from `implement-issue`:

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict          # docs/ pages and a new cross-link are edited
```

No network-gated tests are needed: nothing here touches NWP conversion conventions or an external
feed. No marimo notebook is edited, so `check_marimo_notebooks.py` has nothing new to see (the
pre-commit hook runs it anyway).

## Risks and open questions

**1. Is the correction fleet-wide? Settled: yes, correct every series.** The measured evidence is
PV-only, and weaker than the issue body implies. The centroid measurement needs solar geometry. The
two corroborating signals do not extend it either: measured around the changepoint, only the PV
series drop from 48 rows per day to about 24 and only the PV series stop publishing exact zeros,
while the Disaggregated Demand and Raw Flow series stay at 48 rows per day and never carried exact
zeros at all. Both signals are NGED ceasing to pad a generator's overnight hours with zeros, which a
demand series has no equivalent of.

**What settles it is a mechanism rather than a measurement: NGED's data conversion code treats every
series in the trial area identically, so there is no path by which solar would be a special case.**
That is better evidence than the statistics could have been, because it explains why a fleet-wide
fault is the only kind this feed can have, rather than observing that one is consistent with six
meters. It agrees with NGED's own account of the fault — "looking at the historic data ... all the
data is 30 mins late" — which describes the feed rather than the PV feed. NGED have not tested the
claim rigorously, and the decision is taken knowing that.

*Decided:* apply `POWER_STAMPS_CORRECTED_AT` to every `time_series_id`, with no per-series
exemption and no per-series lookup. Should NGED later report that some feed was converted by a
different path, the constant becomes a per-series lookup and the table is rebuilt again — which is
cheap, because rebuilding is already the standing maintenance route for this table.

**2. If NGED republishes a corrected history, this correction must be deleted before the next
re-materialise, or the stamps are corrected twice.** Nothing this plan builds can detect that: a
corrected pre-changepoint file is indistinguishable from an uncorrected one when read on its own.
*Recommendation:* say so in `correct_late_stamps`' docstring, where whoever runs the rebuild will
meet it, and ask NGED whether a republish is coming before the rebuild is scheduled.

Detection is not impossible, only out of scope here — a corrected republish arrives as a bulk
history file whose rows suddenly fail the de-duplication anti-join for a period the series already
covers, and the ingest already holds the frames that would show it. That work, and a separate
detector for NGED re-aligning *new* readings, are
[#804](https://github.com/openclimatefix/nged-substation-forecast/issues/804), which follows this
issue rather than blocking it.

**3. The change does almost nothing until the table is dropped and re-materialised.** That is an
operational step, not a code step, and it has to happen on the workstation and on AWS. The one
exception is the roughly 54 rows the first hourly run appends to the two dead series, set out in the
design-philosophy check above. *There is no question here for the reviewer* — it is recorded so the
PR body carries it and so neither the unchanged row counts nor the small change in those two series
is mistaken for a defect.

**4. A corrected series has no reading at 08:00 on 26 March 2026.** The last late reading is stamped
08:00 and moves to 07:30; the first correct reading is stamped 08:30. The half-hour ending 08:00 was
therefore never published, and the corrected series carries a one-slot gap there. This is a fact
about the feed rather than a bug in the correction, and it is the honest representation: a gap says
"no measurement" where interpolating would invent one. *Recommendation:* document it in
`docs/roadmap/data-cleaning.md` and leave the gap. **Worth the reviewer confirming that nothing
downstream assumes a gapless half-hourly grid** — the rolling and lag features are the place to
check.

**5. Re-scoring changes every stored leaderboard number.** `cv_forecast_metrics` joins actuals to
forecasts on `valid_time`, so once the table is rebuilt, forecasts already on disk score against
different actuals — and forecasts from models trained on late stamps will look worse, not better.
That is the honest number, but leaderboard rows computed before and after the rebuild are not
comparable. *Recommendation:* state it in the PR body, re-materialise `cv_forecast_metrics` after
the rebuild, and treat the pre-rebuild rows as superseded rather than as a regression to
investigate.

**6. Retraining before v0.2.1 goes to AWS (#646).** Every promoted model has learnt the late stamps.
Retraining is a run of the existing CV assets, not a code change, so it belongs after the rebuild
rather than in this PR. *Recommendation:* the PR body links #646 and says the retrain must land
between the two. Worth its own follow-up issue rather than a note that lives only in a PR body —
**the reviewer's call.**

**7. Editing a Patito contract.** CLAUDE.md says to ask before changing one. This plan extends the
`time` field's `description` and adds a classmethod; no field is widened to `| None` and no range is
relaxed, so the rule's stated failure mode — hiding a defect to make a failing `validate()` pass —
is not in play. **Flagged for the reviewer's explicit approval anyway.**

## What the simplicity review changed, and what was rejected

The first adversarial review argued for exactly the design above — correct at ingest, build no read
door — and its case was adopted in full. Its verified findings:

- **The plan's `Nwp.scan_delta()` precedent was false.** `Nwp.scan_delta` is not the only door to the
  `nwp` table: `view_forecasts.py:164`, `:409` and `:432` all read it with a bare `pl.scan_delta`.
  The analogy the first plan leaned on does not hold, and `PowerTimeSeries.drop_implausible_rows`
  — an ingest-boundary repair of this feed, on this model — is the much closer precedent.
- **`drop_implausible_rows` also refutes "the stored table holds what NGED sent".** The ingest
  already renames `endTime`, recasts the timezone, drops malformed rows and drops columns. NGED's own
  bucket is the provenance record, not our Delta table.
- **A `when/then` over `time` inside a read-time scan would have blocked predicate pushdown** into
  the parquet row-group statistics, making the live path's few-day window scan each series' full
  history over S3 (principle 11). The ingest design never meets this.
- **Factual corrections adopted:** `_engineering_inputs.py` exports `load_engineering_inputs`, not
  `_scan_inputs`; the test file is `test_power_time_series.py`, not `test_power_schemas.py`; the
  last late reading moves to 07:30, not 08:00, leaving the gap in risk 4 that the first plan missed.
- **Cut on the reviewer's argument:** the pre-commit guard hook (nothing left to guard), the
  `correct_late_stamps`/`scan_delta` split (no in-repo caller for the second method), the
  `scan_delta`-resolves-defaults test (tests pydantic-settings plumbing already covered by
  `test_settings.py:40` and `:97`), and the four read-site edits that changed no number.

Nothing was rejected. Two findings became moot rather than wrong: the reviewer's proposed migration
script modelled on `scripts/forecasting/rewrite_nwp_row_groups.py` is unnecessary under the
drop-and-re-materialise decision, and its note that swapping `pl.scan_delta(` for a helper call in
`view_forecasts.py` would silently empty the `DELTA_READ_CALLS` guard
(`packages/dashboard/tests/test_view_forecasts.py:22`) no longer applies, since no notebook is
edited. Both are recorded because each would bite a future change that revisits this decision.


## What the correctness review changed, and what was rejected

The second adversarial review checked every line reference against the code, and checked the safety
claims against the local table and a read-only listing of NGED's bucket. Its findings, all verified
and all taken:

- **The plan's central safety claim was false.** "The hourly ingest never re-parses a historic file"
  is wrong: `select_new_rows` applies `_LATE_FILE_LOOKBACK` against *each series' own* `last_time`,
  so a series whose `last_time` is pre-changepoint re-lists its bulk history file every hour.
  Series 32 and 33 are in that state, and the merge appends about 54 corrected rows to them before
  any rebuild. The design-philosophy check now states the true condition and the consequence.
- **The ordering rationale was the weaker of the two available.** Reversing `correct_late_stamps`
  and `drop_implausible_rows` does not store a bad row — it raises out of `validate`, which costs
  three retries and a failed ingest run. That is a fail-open-versus-fail-closed boundary, and it is
  now what the plan and its comment say.
- **Test 4 would not have failed on `main`.** The fixture the plan leaned on asserts `n_dropped`,
  `height` and `power`, and no `time` value, so it passes either way. Test 4 is now a new test that
  asserts both stored stamps. Test 5's row is now named exactly — `2000-01-01T00:00:00Z` is the only
  reading whose corrected stamp leaves the plausible range.
- **The docs sweep missed the one page that documents the behaviour being changed.**
  `docs/results/beam-diffuse-split.md:894-895` says the ingest takes `endTime` unchanged, and does
  not contain the string the plan's grep searched for.
- **Risk 1's evidence was inherited from the issue body rather than checked.** Both corroborating
  signals are PV-only, so the fleet-wide case rests on NGED's own sentence alone. Risk 1 now says so,
  which makes the confirmation worth asking for rather than a formality.
- **Line-reference corrections taken:** `_LATE_FILE_LOOKBACK` at `storage.py:361`; `str.to_datetime`
  at `read_nged_json.py:89-91`; `cv_assets.py:198`/`:986`; `map_and_timeseries.py:121`;
  `export_baseline_forecasts.py:121`; `LIVE_POWER_HISTORY` rather than `power_lookback` on the live
  path; the unmerged branch's constant is named `ALIGNMENT_FIXED_AT`; the `docs/` grep returns 13
  files, not 17.

Nothing was rejected. The review independently confirmed four things the plan had right and which
the implementer should therefore not re-litigate: the 08:00 gap arithmetic on 26 March 2026 and
test 3's "duplicate key unreachable by construction" argument; that nothing downstream assumes a
gapless half-hourly grid (the lag features are a time-keyed left join, the rolling means use
`rolling_mean_by`, eligibility reads only `first_time`/`last_time`, and capacity is a quantile);
that no new fail-closed path is introduced and no warning path can raise; and that
`tests/test_assets.py:346` is the only asserted `time` value in the repo that moves.


## What the first diff review changed, and what was rejected

The first diff review found **no correctness defect**. It verified the dtype at the call site (the
`when/then` compares against a timezone-aware constant, and a naive or differently-zoned column
would raise rather than coerce silently), the exclusive boundary, that a 30-minute shift maps the
:00/:30 grid onto itself, that a null `time` passes through unchanged, and that no corrected key can
meet an uncorrected one. Its findings were all about size, and all taken:

- **One of the two new contract tests was redundant.** `correct_late_timestamps` cannot see
  `time_series_id`, so "two series at once" tested nothing, and the value assertions duplicated the
  parametrised test. That `validate` passes across the boundary is already proved on the real
  pipeline by `test_extract_power_time_series_corrects_late_timestamps`, which ends in
  `PowerTimeSeries.validate`. Deleting it also removed the reason for widening `_frame` to
  `Sequence`, and two module-level fixtures.
- **The `after_stays` parametrised case was subsumed** by `instant_stays`: any monotone threshold
  mutation that moves 09:00 also moves 08:30.
- **The contracts README restated the method docstring**, and `docs/api/contracts/index.md` renders
  both within one screen — the rule at `docs/architecture/code-style.md:207`. Cut to one sentence.
- **Prose cuts taken**: the constant docstring's three enumerated measurements (the linked page
  holds them), "The order is not cosmetic:" as an announcing clause, the gap paragraph's argument
  about what the function deliberately does not do, the `time` field description's restating trailing
  clause, the call-site comment's first sentence, and the asset docstring's restating sentence.
- **A false claim in a test docstring.** `MIN_PLAUSIBLE_DATETIME` is not the only `endTime` that is
  in range before the correction and out of range after it — `MIN + 15 min` is too. It is the only
  one that *distinguishes the two orderings*, because a misaligned reading is dropped either way.
- **An incomplete rename and a broken wrap.** "Stamp" survived in the beam/diffuse results table, in
  `capacity-estimation.md` and in `disaggregation.md`; and an earlier edit had left "two" orphaned on
  its own line.

**One finding rejected, and one referred rather than applied.** The review proposed cutting the
clause in `data-cleaning.md` explaining that the rows-per-day and exact-zero signals are
generation-only — accepted, because the docs page never makes the claim that clause rebuts, so a
first-time reader has nothing to hang it on. It also observed that the section's design rationale
sits on a `docs/roadmap/` page marked Planned, when `docs/architecture/` is where CLAUDE.md puts
rationale for shipped behaviour — and that `code-style.md` forbids code linking to a roadmap page,
which is why the docstrings carry the fleet-wide argument themselves rather than linking. That is a
real structural observation and a larger change than this issue, so it is reported rather than made.


## What the mutation review changed

The mutation pass ran 27 mutations against the full suite and found **one survivor that matters** —
and it is the test case the first diff review had just talked this plan into deleting.

**Mutating the predicate from `<` to `!=` passes all 941 tests.** That mutation shifts every reading
except the one at the correction instant, so it corrupts every reading the live service has ingested
since 26 March 2026, and nothing goes red. The cause is that every timestamp the suite pushed
through `correct_late_timestamps` sat at or before the instant: the unit test's two cases, the
ingest test's 08:00 and 08:30, both JSON fixtures (which end at exactly 08:30), and the `test_assets`
fixtures (all on 5 March). Reproduced here before acting on it, and the restored `after_stays` case
was confirmed red against the mutation and green without it.

**The first review's argument for deleting that case was wrong, and worth recording.** It reasoned
that `after_stays` was subsumed by `instant_stays`, because "any monotone threshold mutation that
moves 09:00 also moves 08:30". The reasoning holds for monotone mutations and `!=` is not one. The
case now carries a comment saying why it is not the redundancy it looks like, so the next reviewer
does not delete it again.

**The one other survivor is an equivalent mutant, correctly not a finding.** Moving the call to
after `.sort(...)`, still before `drop_implausible_rows`, changes no behaviour — a constant shift of
a contiguous prefix cannot reorder the series, which is the docstring's own argument. The ordering
constraint that does matter, before `drop_implausible_rows`, is killed by
`test_extract_power_time_series_drops_a_reading_the_correction_pushes_out_of_range`, and by that
test alone.

Every other mutation was killed: both boundary directions on the constant, five offset mutations
including `-29m` and `-30s`, dropping `.otherwise`, swapping the branches, the no-op, deleting the
call site, calling it twice, and three timezone mutations on the constant.


## What the prose sweep changed

The diff's new prose went through the `prose-review` procedure: Pass B by hand (the five new
paragraphs in `docs/roadmap/data-cleaning.md` each run to one conclusion, so none was split), then a
one-rule-at-a-time sentence sweep gated on the merge-base. It returned 29 findings, of which 28 were
taken.

The concentrations were the two rules that dominate every sweep in this repo: pronouns and "one"
standing in place of a noun (12 findings, including "a correct one", "not on the one NGED sent", and
an asset docstring carrying three "it"s with three different referents 40 words apart), and long
sentences carrying two or three claims (6 findings). The sweep also caught a category error — a test
docstring called `MIN_PLAUSIBLE_DATETIME` "the only reading", when it is a timestamp constant — and
a sentence readable two ways: "a reading NGED stamps `T` before this instant" reads first as NGED
having done the stamping before the instant, where the meaning is that the timestamp falls before
it.

**One finding rejected.** The sweep flagged "what it buys" and "what it costs" in
`docs/roadmap/data-cleaning.md` under the money-metaphor rule. That rule is about describing
*performance* in money terms, and CLAUDE.md's design-principles section explicitly asks a change
that trades away a principle to say "what is bought in return". The idiom is house style for exactly
this passage, so it stays; the dangling "get it wrong by omission" in the same sentence was fixed.

The sweep also caught two mechanical artefacts of the earlier `stamp`-to-`timestamp` rename: an
orphaned wrap in `docs/roadmap/disaggregation.md` and an over-long line in
`docs/results/beam-diffuse-split.md`.

The guards the skill requires all pass: `check_prose_only.py` proves the sweep changed the structure
of no Python file, `check_comment_wrap.py` that it stranded no comment line, `check_structure.py`
that no link, span, list item or heading was lost, and `check_render_loss.py` that no table row
loses content when rendered.


## What the final correctness review changed

A third independent review of the finished diff, briefed on correctness alone, **found no real
defect**. It ran the repair rather than reasoning about it: the dtype and timezone at the call site,
the exclusive boundary, a null `time`, an empty frame, frames wholly on either side of the
boundary, and a reading at `MIN_PLAUSIBLE_DATETIME`. A naive or non-UTC column raises `SchemaError`
on the comparison rather than coercing silently, which is the safe failure and is unreachable from
the ingest. It confirmed the de-duplication anti-join can discard but never duplicate a corrected
key in the pre-rebuild mixed state, and that no downstream consumer assumes a gapless grid.

It also re-derived the claims this plan makes rather than trusting them: the 54 appended rows across
series 32 and 33 (1 and 53 respectively), the 08:00 gap on 26 March 2026 across all 30 reporting
series, and that `live_forecast_partitions` starts on 2026-06-28, so the live path cannot reach
before 10 April 2026 even on a full backfill — a stronger guarantee than the 15-day
`LIVE_POWER_HISTORY` this plan cited.

**It independently confirmed the `after_stays` case is load-bearing**, mutating the predicate three
ways and finding `after_stays` the sole killer of `!=`. It read the comment attached to that case
and judged its argument sound, which is what the comment was written to achieve.

Three non-defect observations, all acted on:

- **"Roughly 95%" was stale.** Measured on the workstation table, the pre-correction share is 92.7%
  and falls every day NGED publish more readings. The figure came from the issue body. Corrected to
  93%.
- **The one-code-path claim was stated as bare fact.** We have not read NGED's code: NGED reported
  it, and CLAUDE.md asks for a finding to carry its scope. Both the constant docstring and
  `data-cleaning.md` now attribute it and say the fleet-wide scope rests on that report.
- **The PR body had dropped this plan's recommendation to ask NGED about a republish before
  scheduling the rebuild.** Restored, because that question is what prevents a double correction.
