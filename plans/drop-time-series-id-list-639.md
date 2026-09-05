# Drop the full time_series_id list from ingest summary metadata (#639)

**Problem.** `_BaseSummary.time_series_ids` (`assets.py:741`) renders every `time_series_id` in a
batch into one Dagster metadata string —
`str(v.unique().sort().to_list())` — for both `_FileListingSummary` and `_PowerTimeSeriesSummary`.
At V1 scale (32 time series) that's a short, harmless string. V2 scales to roughly 2,500 time
series, so every materialisation of `power_time_series_and_metadata` would render up to that many
IDs into a single Dagster UI field. Two identical `# TODO` comments (`assets.py:791`, `:810`) flag
this as unresolved.

**Solution.** Drop the `time_series_ids` string field entirely rather than truncating it to a
sample. Keep the count: turn `n_time_series_ids` from a field computed by parsing that string back
into a plain field populated directly from `df["time_series_id"].n_unique()`. That removes the
`ast.literal_eval` round-trip along with the field it exists to parse, and leaves exactly the
aggregate stats the issue names as the fallback (`n_files`/`n_rows`, `start_time`, `end_time`,
`n_time_series_ids`).

## Status / how to resume

**Paused for human review — no code has been written.** The `/plan-issue 639` session that wrote
this plan was interrupted (workstation shutdown) before the plan was approved, so this is exactly
where a fresh session should pick up. Everything below this line is the finished plan output of
that skill's steps 1–8; nothing is in progress or half-written.

- **Branch:** `drop-time-series-id-list-639`, pushed to `origin`. This PR is opened from it.
- **Worktree:** `.claude/worktrees/drop-time-series-id-list-639` on the workstation that ran the
  planning session (see "What changes, file by file" and "Tests" below — everything named there is
  still unimplemented on this branch, which currently contains only this plan file). A fresh
  session anywhere else should `git worktree add` (or `git clone` + `git checkout`) this branch
  rather than assume that path exists.
- **Reviews already run:** the simplicity review (`plan-issue` step 5) ran and its one accepted
  finding (cut the proposed V2-scale test) is already folded into the "Tests" section below — see
  "What each review changed". The correctness review (step 7) was deliberately **not** run; see
  "Chosen reviews" below for why, and reconsider that call if anything else about the plan changes
  first.
- **Next step once a human approves this plan as-is:** resume at `implement-issue` step 2 in the
  worktree above — the worktree, branch and plan already exist, so there's nothing left of
  `implement-issue` step 1 to redo. That means: implement the "What changes, file by file" section,
  run the "Verification commands", open a PR (replacing or updating this one) with the
  green-before-push set passing, then the diff-time adversarial reviews `implement-issue` calls for.
- **Next step if the human wants changes first:** edit this plan file directly (it's the same file
  `implement-issue` will read), or send this session's continuation a message with the requested
  change before implementation starts. The "Risks and open questions" section below flags the one
  open judgement call worth resolving before implementation: whether dropping the full ID list is
  acceptable to whoever debugs ingest issues.

## Verdict, size and departures

**Worth doing, roughly as described.** The scale problem is real and the two TODOs are the
project's own record that it's unresolved. V1 is unaffected either way, so there's no live
incident forcing a particular mechanism — matching the issue's note that this can land after other
wave-1 work.

**Size: Medium.** No Patito contract, Delta table, asset, or production degradation path changes;
this is Dagster UI observability metadata only. But it does remove a field (`time_series_ids`) and
change what `n_time_series_ids` reads from, so it's not a pure mechanical rename either — picking
the mechanism (truncate-with-count vs. drop-to-aggregate) is exactly the design choice the issue
flags as open. That gets a plan; see "Chosen reviews" below for how much of it gets adversarial
review.

**Departure from the issue body:** the issue frames both options as live alternatives. This plan
picks "drop the full list, keep only aggregate stats" and rejects "truncate to a sample with a
total count", because:

- A truncated sample answers "is ID X in this batch?" no better than dropping the field outright —
  the ID an operator is actually looking for during an incident is exactly as likely to be outside
  a truncated sample as to be missing entirely. The full list was never a practical way to check
  membership by eye once it holds hundreds of entries, so a sample doesn't preserve a debugging
  capability that already stopped being useful before hitting V2 scale.
- Truncation adds a size constant, a truncation branch, and a test for the boundary — all for a
  capability (partial visibility into which specific IDs appeared) with no identified consumer.
  Nothing in the codebase or docs reads `summary.time_series_ids` except the test file itself
  (verified: only match outside `assets.py`/`tests/test_assets.py` for
  `.time_series_ids`/`time_series_ids ==` is the unrelated `trained_time_series_ids` on
  `BaseForecaster`).
- If a future incident genuinely needs to know whether a specific ID landed in a given hourly
  batch, the Delta table itself is the source of truth and can be queried directly — the Dagster
  metadata field was never the only way to answer that question, just a shortcut for the V1-scale
  case where eyeballing a 32-entry list was fast.

## What changes, file by file

All in `src/nged_substation_forecast/defs/assets.py`.

- **`_BaseSummary`** (~line 732):
  - Delete the `time_series_ids: str = "N/A"` field and its `@field_validator("time_series_ids",
    mode="before")` / `unique_time_series_ids` method (~lines 741, 748–751).
  - Replace the `n_time_series_ids` `@computed_field` `@property` (~lines 753–756) with a plain
    field `n_time_series_ids: int = 0`, populated directly by each subclass's `from_data_frame`
    rather than derived from a string.
  - Drop the now-unused `import ast` (line 6) — nothing else in the file uses it.
- **`_FileListingSummary.from_data_frame`** (~line 781): replace
  `time_series_ids=df["time_series_id"]` with `n_time_series_ids=df["time_series_id"].n_unique()`;
  delete the TODO comment above it (line 791).
- **`_PowerTimeSeriesSummary.from_data_frame`** (~line 803): same substitution; delete the TODO
  comment (line 810).
- Empty-frame branches (`return cls(stage=stage_name, n_files=0)` /
  `return cls(stage=stage_name, n_rows=0)`) are unchanged — `n_time_series_ids` keeps its `0`
  default, so the empty case still doesn't need `.n_unique()` called on an empty column.

No change to `_ProcessedFileListing`, `PowerTimeSeries`, `make_table`, or the call sites in
`power_time_series_and_metadata` (`assets.py:128`, `:201`) — they pass dataframes through
unchanged; only what `_BaseSummary` extracts from them changes.

## Design-philosophy check

Pure Dagster UI observability metadata, not a production data path — nothing here degrades,
widens uncertainty bands, or touches what gets stored. No asset check, no `WARN`/`blocking`
question. No engineering hypothesis applies. Trades away no principle in
`design-principles.md`: this removes a field rather than adding one, so if anything it's a small
move toward "prefer no config over a config nobody reads" in spirit, though at a scale too small to
be worth citing as a real instance of that principle.

## Tests

All in `tests/test_assets.py`, in the "summary classes (pure, no Dagster)" section (~line 1265).

- **`test_file_listing_summary_non_empty`** (~line 1288): remove the
  `assert summary.time_series_ids == "[9, 11]"` line; keep
  `assert summary.n_time_series_ids == 2`, which now exercises the new direct-count path instead of
  the string-parse path. **Would fail on `main` today** in the sense that
  `summary.time_series_ids` won't exist as an attribute after the field is deleted — the test as
  written today would `AttributeError`/fail a `pydantic` extra-field check if run against the new
  class, which is exactly the signal that the field is gone.
- **`test_power_time_series_summary_non_empty`** (~line 1304): same edit — drop the
  `time_series_ids` assertion, keep `n_time_series_ids == 2`.
- **`test_summary_empty_frame_uses_na_defaults`** (~line 1331): drop
  `assert summary.time_series_ids == "N/A"`; keep `assert summary.n_time_series_ids == 0`. Rename
  the test to `test_summary_empty_frame_uses_defaults` since "N/A" no longer describes what's being
  asserted (only `start_time`/`end_time` still default to that string). Update its docstring, which
  currently says "the `\"N/A\"` defaults survive... and `n_time_series_ids` short-circuits... without
  calling `ast.literal_eval`" — the second half of that sentence describes machinery this plan
  deletes, so it needs rewriting to describe the new default (`n_time_series_ids: int = 0`) instead.
- No new test is needed for the `.n_unique()` substitution itself: the three edited tests already
  cover non-empty-with-duplicates (asserts `== 2` where 3 rows share an ID — proof the count is
  deduped, not just `len(df)`), non-empty-without-duplicates, and empty. That's the same coverage
  the old string-parsing path had, exercised the same way, so no test is being weakened by this
  change.
- No large-scale (~2,500 ID) test is added. The first draft of this plan proposed one, reasoning
  it would catch a regression back to rendering the full list at scale — but `.n_unique()` and a
  plain `int` field have no scale-dependent code path: the fix's whole point is that the *string*
  rendering (the one thing with scale behaviour) is what's gone. A 2,500-row fixture would exercise
  the identical count-and-assign path the existing 3-row dedup test already covers, just with a
  bigger loop, so it would raise the fixture-building cost without raising the odds of catching
  anything the smaller tests miss. See "What each review changed" below.

## Docs to update

None. Grepped `docs/` for `time_series_ids`: every hit is `trained_time_series_ids` on
`BaseForecaster` (an unrelated, documented concept in `docs/ml_experimentation/dagster-workflow.md`,
`docs/roadmap/metrics-and-leaderboard.md`, `docs/live_service/operations.md`,
`docs/architecture/production-deployment.md`). Nothing in `docs/` describes this Dagster UI summary
table's fields, so nothing goes stale.

This issue doesn't complete a roadmap item — no roadmap page or "Implementation details" section
to close out.

## Verification commands

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest tests/test_assets.py -k summary
uv run pytest  # full suite before push
```

No network-gated or mkdocs-affecting change here, so nothing beyond the standard
green-before-push set.

## Chosen reviews

Medium size. Choosing **one plan review**: the simplicity review (step 5), run first per the
"earlier of a pair" rule — this plan removes a field and changes what a computed value reads from,
which is exactly the "adds/removes a field" trigger for spending a simplicity review. Skipping the
correctness review: the "current behaviour" account here is a five-line diff each in two methods,
directly readable from the code excerpt above, and the tests it would check are already named
concretely with the assertion each one pins. Diff-time reviews (in `implement-issue`) get decided
at implementation time per that skill's own rules.

## What each review changed

**Simplicity review (step 5, run):** confirmed the plan's chosen mechanism (drop the field, plain
`n_time_series_ids: int`) is already the minimum-diff option — both the truncation alternative and
keeping `n_time_series_ids` as a `@computed_field` over an excluded `pl.Series` field would add
more machinery for no extra capability. One finding accepted: the proposed V2-scale test added
fixture cost without adding coverage, since nothing in the new code path is scale-dependent — cut
from the "Tests" section above. No findings rejected.

**Correctness review (step 7):** not run — see "Chosen reviews" above.

## Risks and open questions

- **Is losing the full ID list acceptable to whoever debugs ingest issues day to day?** My
  recommendation is yes, per the reasoning under "Departure from the issue body" — but this is a
  judgement call about an operational workflow I can't fully observe from the code, so it's worth
  the human reviewer's explicit sign-off rather than treating it as settled by this plan alone.
- **Should `n_time_series_ids` become a `@computed_field` again in the future** if another
  aggregate (e.g. a min/max ID, or a small fixed-size sample) gets added later? Not needed now —
  a plain field is simpler while there's only one aggregate to carry, and this project treats a
  later breaking change as cheap.
