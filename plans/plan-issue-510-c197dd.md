# Plan — #510: cap `power_data_is_fresh`'s late-series metadata table

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/510>
Branch: `claude/plan-issue-510-c197dd`

## Verdict

**Worth implementing, as described.** The issue's premise checks out against `main`:
`_late_table_metadata` ([checks.py:226-239](../src/nged_substation_forecast/defs/checks.py))
iterates the whole `late` frame with no cap, while all three sibling listings do cap —
`_MAX_MISSING_SERIES_LISTED = 20` (checks.py:379), `MAX_LATE_SERIES_IN_CONTEXT = 50` and
`MAX_LATE_SERIES_IN_MESSAGE = 20` (`_sentry.py:197`, `:204`). The uncapped one is also the only
listing written to **durable** storage as a structured table: in the AWS deployment Dagster's event
log is Postgres (`docs/live_service/aws.md:982-992`), `pg_dump`ed to S3 nightly (`:1273-1277`), so
an uncapped table inflates the backup as well as the database.

Scale of the waste, if a total feed stall persists at V2 (~2,500 series): one record serialises
through Dagster's serdes to **143 bytes** — measured, and including the
`{"__class__": "TableRecord", "data": {…}}` envelope — so ~349 KB per hourly evaluation, ~8.4
MB/day, **~3.1 GB/year**, versus ~7 KB/hour and ~63 MB/year with a 50-row cap. At V1 (32 series) the
table is trivially small, which is why this is "before V2", not urgent.

The change is small and self-contained. It is the same rule the neighbours already follow, so it
removes an inconsistency rather than adding a mechanism.

### Departures from the issue body

None of substance — the issue deliberately left two things open, and this plan decides them
(sections below). One clarification the issue's point 2 invites: the truncation goes in
`_late_table_metadata`, i.e. **inside** the check's catch-all, not outside it. Building the records
is already guarded; what is unguarded is Dagster's serialisation of the returned
`AssetCheckResult`. Capping the record count therefore cannot itself be made safer by moving —
the point of the cap is to make the *unguarded* step small, not to guard it.

## Decision 1 — the cap is 50

`_MAX_LATE_SERIES_IN_TABLE: Final[int] = 50`, matching `MAX_LATE_SERIES_IN_CONTEXT` rather than the
two 20s.

There is no clean precedent to follow, and the issue body's table is misleading on this point:
it labels `_MAX_MISSING_SERIES_LISTED` as "the sibling check's description", but that one constant
caps *both* a prose line (checks.py:762-765) **and** a durable Dagster metadata field
(checks.py:812-814). So the repo does not in fact reserve 20 for prose and 50 for structured
payloads; one of the 20s already covers a durable listing.

The argument for 50 therefore rests on role rather than on precedent. `MAX_LATE_SERIES_IN_MESSAGE`
caps a Sentry *issue title* and `_MAX_MISSING_SERIES_LISTED` caps a one-line check *description* —
both are read at a glance, and both must stay short for that reason; `missing_time_series_ids`
inherits their number only because it reuses the same constant, and it is a `str(list[int])`, which
is a fraction of a `TableRecord`'s cost per entry. The Dagster late-series table is not read at a
glance: it is a sortable, browsable drill-down, the same role the Sentry event *context* plays. 50
in both means the operator sees the same leading 50 series in each place, which is one fewer thing
to reconcile at 3pm on a bad day, and 50 × 143 B is ~7 KB — negligible either way.

Cutting to 20 would be defensible and cheap; see open question 1.

The rows are already sorted worst-first (never-reported, then most-stale first, by
`evaluate_power_freshness`'s `.sort(["status", "hours_late"], ...)`), so a head-50 is the 50 most
worth looking at, not an arbitrary 50.

## Decision 2 — where the truncation lives

Inside `_late_table_metadata`, via `late.head(_MAX_LATE_SERIES_IN_TABLE)` before the list
comprehension.

**Not** in `evaluate_power_freshness`. `PowerFreshnessResult.late` must stay complete: it is the
pure function's honest output, and `report_power_freshness` takes its own independent
`head(MAX_LATE_SERIES_IN_CONTEXT)` from it (`_sentry.py:264`). Capping at the source would silently
couple the Sentry context cap to the Dagster table cap and would make the pure function lie about
what it found. (`n_stale` / `n_never` are `stale.height` / `never.height`, read off the two
pre-`concat` frames the sort never touches, so they would survive a source-level cap — but that is
luck, not a reason.)

## Decision 3 — the house rule that keeps a truncated table honest

The true count is already carried uncapped by `n_late` (and `n_stale`, `n_never_reported`,
`n_series_total`), so the existing metadata already satisfies the "a truncated list never makes a
large stall look small" rule the sibling docstrings state.

Add one field on top: **`n_late_listed`** — how many rows the table actually holds. This mirrors
`late_series_shown` in the Sentry context (`_sentry.py:287`), and removes the "is this table
everything?" ambiguity without making the operator compare a row count against `n_late` by eye. It
is an `int`, so Dagster keeps one type per key across runs and can plot it (the convention
`_live_forecast_check_metadata` documents at checks.py:791-793). Named `n_late_listed` rather than
`late_time_series_shown` to match this check's own `n_*` metadata naming.

## What changes, file by file

### `src/nged_substation_forecast/defs/checks.py`

1. **New constant** `_MAX_LATE_SERIES_IN_TABLE: Final[int] = 50`, placed in the
   `power_data_is_fresh` section beside `_LATE_TABLE_SCHEMA` (~line 96), with a docstring in the
   style of its three siblings: what it caps, why 50 rather than 20, that the table lands in
   Dagster's event log (Postgres, nightly `pg_dump`) so it is durable, and that `n_late` carries the
   true count so a truncated table never makes a large stall look small.
2. **`_to_asset_check_result`** — truncate **once**, here, and use the same local for both metadata
   entries:

    - bind `listed = result.late.head(_MAX_LATE_SERIES_IN_TABLE)`;
    - pass `listed` to `_late_table_metadata`;
    - add `"n_late_listed": listed.height` to the metadata dict.

   One truncation point means the row count and the table can never disagree. Do **not** try to read
   the length back off `_late_table_metadata`'s return value: it is annotated `-> MetadataValue`,
   which has no `records` attribute, so `.records` is an `unresolved-attribute` error under `ty`
   (only the narrower `TableMetadataValue` has it). Deriving the count in the caller avoids both
   that and a redundant second `min`.

3. **`_late_table_metadata`** — unchanged in body; extend the docstring to say it renders an
   already-truncated frame and to name `_MAX_LATE_SERIES_IN_TABLE` as where the cap is applied, so
   the single caller relationship is documented rather than assumed.

No signature changes, no behaviour change to `evaluate_power_freshness`, `PowerFreshnessResult` or
the Sentry path.

### Docs

- **`docs/live_service/operations.md`** (~line 164): "its metadata carries a table of the late
  series with `last_seen` and `hours_late`" → say the table lists the leading `N` worst offenders
  (never-reported first, then most-stale first) and that `n_late` is the true count, so a full table
  is not a full stall.
- **`docs/architecture/production-deployment.md`** (~line 76): "The count of each, plus a table of
  the offending `time_series_id`s, lands in the check's Dagster metadata" → add that the table is
  capped for the same reason the Sentry payloads are (which that page already explains at ~line
  201), and that the counts beside it are not.

Both written in the present tense, describing how the code works now — no "previously uncapped"
framing (CLAUDE.md, "Write about the present, not the past").

Nothing else needs touching: `docs/design-philosophy/inherent-stability.md:320` says only that the
late-series table is the provider channel, which stays true.

### Not in scope (report, don't fix)

`_live_forecast_check_metadata` writes `missing_time_series_ids` capped at
`_MAX_MISSING_SERIES_LISTED` but has no "how many were listed" companion — the same small
ambiguity `n_late_listed` closes here. That is the sibling check, not this issue. Flagging it for
Jack rather than fixing it.

## Design-philosophy check

- **Which side of the line?** Production. `power_data_is_fresh` runs hourly in the live service, so
  the rule is degrade-and-record, never raise.
- **Can the warning path now raise?** No, and this change strictly *reduces* exposure.
  `pl.DataFrame.head()` on a frame with fewer rows than the cap returns the whole frame and cannot
  raise on an empty frame. The call sits inside `_check_power_data_freshness`, under
  `power_data_is_fresh`'s `except BaseException` guard, so even a bug here degrades to
  "could not evaluate" rather than failing the hourly run. Rule 7 of
  [The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
  is what is at stake, and the *unguarded* step — Dagster serialising the returned
  `AssetCheckResult` into an event — gets a strictly smaller payload. That is the whole point of the
  issue's second question.
- **Severity unchanged**: still `AssetCheckSeverity.WARN`, still `blocking=False`. No new asset
  check.
- **Hypotheses**: this serves **H1** (the service keeps answering as inputs degrade) only
  indirectly — it protects the *reporting* path at the scale where a stall is worst. It is best
  described as removing a fail-closed risk rather than delivering a hypothesis; no label needs
  citing as delivered.
- **Design principles**: nothing traded away. The change makes two code paths agree on a rule they
  were supposed to share.

## Tests (`tests/test_checks.py`)

One new test, plus one existing test left alone.

**New: `test_power_data_is_fresh_late_table_is_capped`.** Unit-level, against
`_to_asset_check_result` (not the full end-to-end check) so it needs no Delta table and no clock
freeze: build a `PowerFreshnessResult` whose `late` frame holds `_MAX_LATE_SERIES_IN_TABLE + 10`
stale rows, call `checks._to_asset_check_result(...)`, and assert

- `len(late_table.records) == checks._MAX_LATE_SERIES_IN_TABLE` — **fails on `main` today**, where
  it would be `_MAX_LATE_SERIES_IN_TABLE + 10`;
- `result.metadata["n_late"].value == _MAX_LATE_SERIES_IN_TABLE + 10` — the true count survives
  truncation (this one passes on `main`, and is the assertion that makes the first one safe to
  keep);
- `result.metadata["n_late_listed"].value == checks._MAX_LATE_SERIES_IN_TABLE` — fails on `main`
  (the key does not exist), and pins the new field;
- the retained records are the worst offenders: with `hours_late` descending in the fixture, the
  ids present are the leading slice, not an arbitrary one.

Assert `isinstance(late_table, TableMetadataValue)` before touching `.records`, exactly as
test_checks.py:209 already does — without that narrowing, `ty` rejects `.records` on the
`MetadataValue` the metadata dict is typed to hold. Calling a private helper directly from a test
has precedent at test_checks.py:683 (`checks._to_live_forecast_check_result`).

Building the frame follows the existing pattern at test_checks.py:252-267 (the sentinel
`PowerFreshnessResult` in `test_power_data_is_fresh_hands_evaluated_result_to_sentry`), which
already shows the exact dtypes: `time_series_id` `Int32`, `last_seen` cast to `UTC_DATETIME_DTYPE`,
`hours_late` `Float64`, `status` a plain string column. Note `_late_table_metadata` reads `status`
by value only, so the `pl.Enum` dtype the production path produces is not required by this test —
matching the existing sentinel is fine, and mirroring it keeps the two consistent.

**Unchanged: `test_power_data_is_fresh_end_to_end`** (test_checks.py:182-211) asserts
`late_ids == {2, 99}` with 2 late series — comfortably under any cap, so it stays green and keeps
proving the uncapped small case still lists everything.

A second test asserting the *small* case is not truncated is not needed: the end-to-end test above
already is that test.

## Verification commands

The green-before-push set:

```bash
uv run ruff format . && uv run ruff check . && uv run --all-packages ty check && uv run pytest
```

Plus, because docs change:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

```bash
uv run mkdocs build --strict
```

No network-gated tests are relevant (nothing here touches NWP conversion), and there is no Marimo
notebook in the diff.

## Risks and open questions

1. **Is 50 the right cap, or should it be 20?** *Recommendation: 50*, for the "same role as the
   Sentry context" argument in Decision 1 — but note that argument is weaker than it first looked:
   `_MAX_MISSING_SERIES_LISTED = 20` already caps a durable metadata field as well as a description,
   so there is no clean "structured payloads get 50" precedent to appeal to. If Jack would rather
   every listing in the codebase share one number, 20 is the alternative and costs only drill-down
   detail; the Sentry context still carries 50.
2. **Is `n_late_listed` worth a new metadata key?** *Recommendation: yes* — it is one int, it
   mirrors the Sentry context field, and it makes a truncated table self-describing. If Jack thinks
   `n_late` alone is enough, dropping it removes one bullet from the test and nothing else.
3. **Should the sibling gap (`missing_time_series_ids` with no "listed" count) be closed in the same
   PR?** *Recommendation: no* — different check, out of scope for #510, and CLAUDE.md says to
   discuss out-of-scope changes rather than make them. Worth a follow-up issue if Jack agrees the
   ambiguity is real.

## Adversarial review

A fresh sub-agent reviewed this plan against the code with no sight of the reasoning that produced
it. It confirmed the factual claims about current behaviour (line numbers, constants, sort order,
existing metadata keys), confirmed by running them that all three "fails on `main`" assertions
really do fail today, confirmed no consumer of `PowerFreshnessResult.late` or doc page is missed
(the four consumers are `_late_table_metadata`, `_capture_power_freshness_warning`,
`tests/test_sentry.py:34-51` and the snippet at `docs/live_service/sentry.md:68-95`; the last two go
through the Sentry path only), and confirmed the fail-open analysis and the Postgres/`pg_dump`
claim. It raised three defects.

### Accepted and fixed

1. **The byte estimate omitted Dagster's serdes envelope.** A `TableRecord` serialises to 143 bytes,
   not ~110, because of the `{"__class__": "TableRecord", "data": {…}}` wrapper. Verified
   independently by running `serialize_value` on a representative record. Every derived figure was
   ~30% low; the Verdict section now carries the measured numbers (~349 KB/hour, ~3.1 GB/year at V2,
   ~63 MB/year capped). The error was in the safe direction, but these numbers will be quoted in the
   PR body.
2. **Decision 1's justification was factually wrong.** It claimed the two caps of 20 govern
   at-a-glance prose only, but `_MAX_MISSING_SERIES_LISTED` also caps the durable
   `missing_time_series_ids` metadata field (checks.py:812-814) — the plan's own "Not in scope"
   section said as much, so the plan contradicted itself, having inherited the mislabelling in the
   issue body's table. Decision 1 is rewritten to say there is no clean precedent and to argue for
   50 on role and cost instead, and open question 1 now flags the weakened argument rather than
   presenting 50 as the obvious inheritance.
3. **"No signature changes" contradicted the preferred way of computing `n_late_listed`.**
   `_late_table_metadata` returns `MetadataValue`, which has no `.records`, so deriving the count
   from its return value is an `unresolved-attribute` error under `ty` unless the return type is
   narrowed to `TableMetadataValue`. Resolved by truncating once in `_to_asset_check_result` and
   using `listed.height` — no signature change, no drift, no redundant `min`. The two nits the
   reviewer raised alongside are also folded in: the `n_stale`/`n_never` aside is reworded (they are
   read from the two pre-`concat` frames, not "before the sort"), and the new test is now told to
   narrow with `isinstance(..., TableMetadataValue)` before touching `.records`.

### Rejected

Nothing. The reviewer manufactured no findings and explicitly declined to pad the list.
