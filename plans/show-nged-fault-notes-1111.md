# Show NGED's fault notes in the freshness check (issue 1111)

**Problem.** NGED's JSON files carry a free-text `Information` field naming known meter faults. Ingestion already stores it in the `information` column of the metadata parquet, but nothing reads it, and the column's description ("always null in the V1 trial area") is out of date.

**Solution.** `power_data_is_fresh` prints each series' note in its description and metadata, beside the silenced ids, so the operator sees why a series is silenced or odd. `drop_reason` stays value-based. Two stale description strings are corrected.

## Verdict, size and departures

**Verdict:** worth doing, in the reduced form below. The issue's four questions were settled in the comments on the issue.

**Size: medium.** The five triggers:

- **What gets stored:** no. No column, dtype or table changes. The `information` field's description string changes.
- **Production serving path:** no. The check runs hourly beside the ingest, not in `load a model, call predict`.
- **Degradation rule:** no. The check stays `WARN` and `blocking=False`, its `try`/`except BaseException` guard is untouched, and the healthy-or-late logic does not change. The new read of the notes is guarded on its own, so a failure there cannot cost the freshness report.
- **More than one defensible design:** the issue discussion settled it (series-level evidence, not a row rule).
- **Callers not nameable without searching:** no. `evaluate_power_freshness` has one production caller and the tests.

**Reviews:** one plan review (simplicity, because the plan adds a result field) and one diff review (correctness and cutting down).

**Departures from the issue's proposal comment:** none. From the first proposal that the maintainer approved in chat: the "warn when a series carries a fault note" bullet is dropped, because series 19, 26, and 29 report and carry a note today, so the check would be permanently yellow.

## What changes, file by file

- `src/nged_substation_forecast/defs/checks.py`
  - `_read_expected_ids` also returns the non-null `information` of each series, as a `dict[int, str]` beside the ids, from the same `scan_parquet` of the metadata table. It selects `information` only when `"information" in lf.collect_schema()`, because the column is `allow_missing`. It has no `except` of its own: the metadata table is already read here, and the check body's existing guard covers a corrupt file.
  - `_describe_power_freshness(result, fault_notes)` appends one sentence when `fault_notes` is non-empty, `NGED fault notes: 19: "…"; 26: "…".`, with each note in full. A note for a silenced id appears here too.
  - `_to_asset_check_result(result, fault_notes)` and `_check_power_data_freshness` pass the notes through. `evaluate_power_freshness`, `PowerFreshnessResult`, and `report_power_freshness` (Sentry) are unchanged: the notes take no part in classification.
- `packages/contracts/src/contracts/power_schemas.py`: the `information` description becomes "Free-text note from NGED about a known meter fault or a customer's status. Null for most series."
- `packages/nged_data/src/nged_data/storage.py`: rewrite the `remove_small_files_from_listing` docstring sentence that says the field is always null.
- Docs: `docs/live_service/operations.md` (one or two sentences on the new description sentence, with an invented example note) and `docs/roadmap/data-sources.md` (the "does not yet act on" bullet now says the notes are shown to the operator, and that cleaning does not read them because a note has no date and would rewrite a series' whole history).

## Design-philosophy check

The change runs in production, in an asset check. The check stays `WARN` and `blocking=False`, and nothing new can raise: the note read is wrapped on its own. No hypothesis label is delivered. The notes are shown, not acted on, so no principle is traded away.

## Tests (`tests/test_checks.py`)

Each would fail on `main` because the notes parameter does not exist:

- `_describe_power_freshness` names a note for a silenced id and for a reporting id, with the note in full.
- With no notes, the description is byte-identical to today's, so a green run says nothing new.
- `_read_expected_ids` returns the non-null notes from a parquet fixture, and returns no notes when the fixture has no `information` column.

Fixtures and the docs example use invented note text, never a real string: a real note can name a generator, and the repository is public.

## Verification

The green-before-push set from `implement-issue`, plus `uv run mkdocs build --strict` and `uv run pydoclint`, the docs-link checker, and `uv run pytest tests/test_checks.py`.

## Risks and open questions

- **Series 33 will show no note.** Its stored `information` is null because the 520-byte size filter in `remove_small_files_from_listing` drops its files (469 to 476 bytes, no readings) before download. Series 33 is the only silenced series today. Capturing the note needs a separate fetch of the newest file of each series with no readings. Recommendation: accept the gap, and add the note to the comment above `_SILENCED_TIME_SERIES_IDS` by hand.
- **Reviewer findings rejected:** none. The simplicity review's cuts are all taken: no separate `_read_fault_notes` helper or `except`, no `fault_notes` field on `PowerFreshnessResult`, no truncation or duplicate metadata entry, and a shorter docs change.
- **Sentry stays clean** only while `report_power_freshness` is unchanged, because a note can name a generator.
- **Missing-half-hour detector** is a separate follow-up issue, not part of this change.
