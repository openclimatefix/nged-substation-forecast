# Correct NGED's half-hour-late power stamps (#790)

**The problem.** NGED stamped every half-hourly power reading half an hour late until 08:30 UTC on
26 March 2026, then corrected the feed without marking the change. `PowerTimeSeries.time` is
documented as period-ending — the value stamped `T` is the mean over `(T − 30 min, T]` — and for
roughly 95% of the rows in our `power_time_series` Delta table that is false: the value stamped `T`
is the mean over `(T − 60 min, T − 30 min]`. Every model trained on that table has learnt a
half-hour timing error, and the error is not uniform across the record. The measurement, and the
three independent signals that pin the changepoint, are in the [beam/diffuse
appendix](https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/#the-power-stamps-before-26-march-2026-are-half-an-hour-late).

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
  instant, and links to the published appendix. The same constant already exists twice on the
  unmerged `beam-diffuse-split-experiment` branch (`build_dataset.py:101`,
  `site_e_commissioning.py:51`); this is its permanent home, and that branch should import it rather
  than re-declare it once both have merged.

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
  `str.to_datetime` on line 86-88 and before `drop_implausible_rows` on line 97. Order matters and
  should carry a one-line comment: correcting first means `drop_implausible_rows` judges the stamp
  that will actually be stored, so a corrected stamp landing outside the plausible range is caught
  rather than stored.

- The `:00`/`:30` alignment check is unaffected: a 30-minute shift maps the half-hour grid onto
  itself.

### Nothing else changes

Stated explicitly, because a reviewer should see these as decisions rather than omissions:

- **No read site changes.** `_engineering_inputs.py:125`, `cv_assets.py:199`, `cv_assets.py:987`,
  `view_forecasts.py:311`, `map_and_timeseries.py:122` and `export_baseline_forecasts.py:170` all
  keep their bare scans.
- **`select_new_rows` and `time_series_coverage` are untouched.** Both compare parsed rows against
  stored rows; after this change both sides carry corrected stamps, so the anti-join and the
  watermark stay consistent with no edit.
- **No migration script.** The table is dropped and re-materialised.

## Design-philosophy check

**This code path runs in production, and the change cannot degrade the live service.**
`live_forecasts` reads power over a `power_lookback` window measured in days
(`_engineering_inputs.py:37`), so every row it reads is post-changepoint, where the correction is an
identity. The live service's behaviour is unchanged by this merge, and stays unchanged until a model
is retrained on the rebuilt table.

**The change is also inert until the table is rebuilt, which is the safety property that makes it
shippable on its own.** `select_new_rows`' file-listing filter uses a 3-day `_LATE_FILE_LOOKBACK`
(`storage.py:362`), so the hourly ingest never re-parses a historic file. Merging this change
against the existing table therefore corrects nothing and breaks nothing: the correction fires only
on a full re-materialise. There is no window in which corrected and uncorrected stamps interleave.

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

4. **A parsed pre-changepoint reading is stored 30 minutes earlier than its `endTime`.** The
   existing fixture at lines 121-123 uses `2026-01-01` stamps, which sit before the changepoint, so
   its expected `time` values move by −30 minutes. Extend the fixture with a post-changepoint
   reading whose expected `time` does not move, so one test pins both branches at the boundary the
   ingest actually crosses. Fails on `main`, which stores `endTime` unchanged.
5. **The correction runs before `drop_implausible_rows`, not after.** A reading whose `endTime` sits
   just inside the plausible range but whose corrected stamp falls outside it must be dropped, not
   stored. Fails on an implementation that corrects after dropping. This is the only ordering bug
   the change can have, and nothing else would catch it.

In `tests/test_assets.py`: the ingest-asset fixture at line 323 uses `2026-03-05 12:30:00+0000`,
which is pre-changepoint, so its expected stored `time` moves by −30 minutes. No new assertion; the
existing one is updated and gains a comment naming why.

## Docs to update

- **`packages/contracts/README.md`** — if it describes `PowerTimeSeries.time`, restate it to match
  the extended field description. Check before editing.
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
  `power_time_series` across `docs/` returns 17 files. The implementer reads them and edits only
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

**1. Is the correction fleet-wide, or only the six metered solar farms?** The measurement needs
solar geometry, so it cannot speak for a substation load series. Two of the three signals — the
rows-per-day drop from 48 to 26 and the disappearance of exact-zero rows — are properties of what
NGED publishes rather than of PV physics, and both are consistent with a fleet-wide change, but
neither establishes the direction or magnitude of the offset on a non-PV series. *Recommendation:*
correct fleet-wide, because a single feed correcting at a single instant is by far the likeliest
reading, and ask NGED in parallel. If the answer is "per feed", `POWER_STAMPS_CORRECTED_AT` becomes
a per-`time_series_id` lookup and the table is rebuilt again. **This is the reviewer's call to
confirm.**

**2. If NGED republishes a corrected history, this correction must be deleted before the next
re-materialise, or the stamps are corrected twice.** Nothing in the code can detect that, because a
corrected pre-changepoint file is indistinguishable from an uncorrected one. *Recommendation:* say
so in `correct_late_stamps`' docstring, where whoever runs the rebuild will meet it, and ask NGED
whether a republish is coming before the rebuild is scheduled.

**3. The change does nothing until the table is dropped and re-materialised.** That is an
operational step, not a code step, and it has to happen on the workstation and on AWS. *There is no
question here for the reviewer* — it is recorded so the PR body carries it and so it is not
mistaken for a defect when the merged code changes no stored row.

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
