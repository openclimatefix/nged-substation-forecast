# Show NGED's fault notes in the freshness check (issue 1111)

**Problem.** NGED's JSON files carry a free-text `Information` field naming known meter faults. Ingestion already stores it in the `information` column of the metadata parquet, but nothing reads it, and the column's description ("always null in the V1 trial area") is out of date.

**Solution.** The ingest also reads the metadata of series whose newest file carries no readings, so the metadata table is as current as NGED's files. `power_data_is_fresh` prints each series' note in its description and metadata, beside the silenced ids, so the operator sees why a series is silenced or odd. `drop_reason` stays value-based. Two stale description strings are corrected.

## Verdict, size and departures

**Verdict:** worth doing, in the reduced form below. The issue's four questions were settled in the comments on the issue.

**Size: medium.** The five triggers:

- **What gets stored:** no. No column, dtype or table changes. The `information` field's description string changes.
- **What gets stored, continued:** the ingest now refreshes more rows of the metadata parquet (for series that carry no readings), so the stored table's content changes for those series, though its schema does not.
- **Production serving path:** no. The check runs hourly beside the ingest, not in `load a model, call predict`.
- **Degradation rule:** no. The check stays `WARN` and `blocking=False`, its `try`/`except BaseException` guard is untouched, and the healthy-or-late logic does not change. The new read of the notes is guarded on its own, so a failure there cannot cost the freshness report.
- **More than one defensible design:** the issue discussion settled it (series-level evidence, not a row rule).
- **Callers not nameable without searching:** no. `evaluate_power_freshness` has one production caller and the tests.

**Reviews:** the simplicity plan review has run. A correctness plan review runs next, because the ingest write path now changes. One diff review follows.

**Departures from the issue's proposal comment:** none. From the first proposal that the maintainer approved in chat: the "warn when a series carries a fault note" bullet is dropped, because series 19, 26, and 29 report and carry a note today, so the check would be permanently yellow.

## What changes, file by file

- `src/nged_substation_forecast/defs/checks.py`
  - `_read_expected_ids` also returns the non-null `information` of each series, as a `dict[int, str]` beside the ids, from the same `scan_parquet` of the metadata table. It selects `information` only when `"information" in lf.collect_schema()`, because the column is `allow_missing`. It has no `except` of its own: the metadata table is already read here, and the check body's existing guard covers a corrupt file.
  - `_describe_power_freshness(result, fault_notes)` appends one sentence when `fault_notes` is non-empty, `NGED fault notes: 19: "…"; 26: "…".`, with each note in full. A note for a silenced id appears here too.
  - `_to_asset_check_result(result, fault_notes)` and `_check_power_data_freshness` pass the notes through. `evaluate_power_freshness`, `PowerFreshnessResult`, and `report_power_freshness` (Sentry) are unchanged: the notes take no part in classification.
- `packages/nged_data/src/nged_data/storage.py`: new `download_metadata_of_series_without_new_files(store, all_files, downloaded_files)`. It takes the newest file per `time_series_id` among the series that have no file in this run's download list and whose newest file is at or under the size threshold (a file with no readings), downloads each, and returns the `TimeSeriesMetadata` frame that `_extract_time_series_metadata` gives. It never parses power. A series whose newest file is large is skipped, because its normal download refreshes the metadata whenever NGED publishes new readings, and downloading the newest file of every series every hour would cost one request per series per hour at V2 scale. The 520-byte threshold moves out of the `remove_small_files_from_listing` default into a module constant that both functions use.
- `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`):
  - Call the new function inside the existing NGED-bucket `try`, so a transient S3 error retries with the listing. A fault in parsing one of these small files must not stall the power write, so the call has its own `except BaseException`, with the usual cancellation re-raise, that logs, calls `report_asset_degradation`, and carries on with no extra metadata.
  - Concatenate this frame with `downloaded.metadata` (diagonal concat, then `unique` on `time_series_id` keeping the downloaded one) before `upsert_metadata`.
  - Today the `NoNewData` branch returns before any metadata upsert, and most hours have no new files. The `NoNewData` branch must therefore also upsert the extra metadata when it is non-empty. The upsert is factored into one helper that both paths call, keeping the existing guard that a metadata fault never stops the power write.
  - Add `n_metadata_only_files_downloaded` to the output metadata.
- `packages/contracts/src/contracts/power_schemas.py`: the `information` description becomes "Free-text note from NGED about a known meter fault or a customer's status. Null for most series."
- `packages/nged_data/src/nged_data/storage.py`: rewrite the `remove_small_files_from_listing` docstring sentence that says the field is always null.
- Docs: `docs/live_service/operations.md` (one or two sentences on the new description sentence, with an invented example note) and `docs/roadmap/data-sources.md` (the "does not yet act on" bullet now says the notes are shown to the operator, and that cleaning does not read them because a note has no date and would rewrite a series' whole history).

## Design-philosophy check

The change runs in production, in an asset check. The check stays `WARN` and `blocking=False`, and nothing new can raise: the note read is wrapped on its own. No hypothesis label is delivered. The notes are shown, not acted on, so no principle is traded away.

## Tests (`tests/test_checks.py`)

Each would fail on `main` because the notes parameter does not exist:

- `_describe_power_freshness` names a note for a silenced id and for a reporting id, with the note in full.
- With no notes, the description is byte-identical to today's, so a green run says nothing new.
- `download_metadata_of_series_without_new_files`, against a fake store or the moto S3 fixture the storage tests already use: returns the note of a series whose only recent file is small; skips a series that has a file in `downloaded_files`; skips a series whose newest file is large; takes the newest of several small files of one series; returns an empty frame for an empty listing.
- Asset (`tests/test_assets.py`): an hour in which NGED published no new readings still upserts the note of a series with a small file, which fails on `main` because `NoNewData` returns early; and a malformed small file does not stop the power write, and reports degradation.
- `_read_expected_ids` returns the non-null notes from a parquet fixture, and returns no notes when the fixture has no `information` column.

Fixtures and the docs example use invented note text, never a real string: a real note can name a generator, and the repository is public.

## Verification

The green-before-push set from `implement-issue`, plus `uv run mkdocs build --strict` and `uv run pydoclint`, the docs-link checker, and `uv run pytest tests/test_checks.py`.

## Risks and open questions

- **Cost at V2 scale.** A series with no readings has its newest small file (about 500 bytes) downloaded on every hourly run, because no state records that the file was already read. V1 has two such series. V2 has about 2,500 series, and the number of silent ones is unknown. Recommendation: accept now, and add a watermark if the Dagster run metadata `n_metadata_only_files_downloaded` grows.
- **A stale note stays until NGED publishes a newer file.** A silent series that NGED stops writing files for keeps its last note.
- **Series 33's note** is captured by the fetch above, so the freshness description shows it.
- **Reviewer findings rejected:** none. The simplicity review's cuts are all taken: no separate `_read_fault_notes` helper or `except`, no `fault_notes` field on `PowerFreshnessResult`, no truncation or duplicate metadata entry, and a shorter docs change.
- **Sentry stays clean** only while `report_power_freshness` is unchanged, because a note can name a generator.
- **Missing-half-hour detector** is a separate follow-up issue, not part of this change.
