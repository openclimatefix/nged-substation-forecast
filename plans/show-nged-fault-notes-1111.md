# Show NGED's fault notes in the freshness check (issue 1111)

**Problem.** NGED's JSON files carry a free-text `Information` field naming known meter faults. Ingestion stores it in the `information` column of the metadata parquet, but a series that has stopped reporting never has its note refreshed, nothing reads the column, and its description ("always null in the V1 trial area") is out of date.

**Solution.** The ingest downloads the newest file of every series, so each series' metadata comes from NGED's newest file. `power_data_is_fresh` prints each series' note in its description. `drop_reason` stays value-based. Two stale description strings are corrected.

## Verdict, size and departures

**Verdict:** worth doing, in the form below.

**Why the newest file of every series.** On the real data, `select_new_rows` offers every file newer than a series' power watermark minus 3 days (`_LATE_FILE_LOOKBACK`), so each hour re-downloads about 13 large files per reporting series. Series 32 and 33 each have one old large file (end time 2026-03-26 08:00, the backfill file, with a null `Information`) that is offered every hour, so their metadata comes from that old file and never from their newest small file, which carries the note. Taking the newest file of every series fixes this. A reporting series already has its newest file in the download list, so the extra downloads are one per silent series. An earlier design downloaded only the newest small file of series with no file in the list, and would have missed series 32 and 33 for this reason.

**Size: medium.** The five triggers:

- **What gets stored:** the metadata parquet's content for silent series becomes current. No column, dtype or table changes. The `information` field's description string changes.
- **Production serving path:** no. The ingest and the check run hourly beside the forecasts.
- **Degradation rule:** no. The check stays `WARN` and `blocking=False` with its guard untouched, and the ingest's guards are untouched.
- **More than one defensible design:** the issue discussion settled that notes are series-level evidence. The download design was compared in a review, and this one is the simplest of four.
- **Callers not nameable without searching:** no. `download_and_parse_files` has one production caller.

**Reviews:** a simplicity and a correctness plan review, a diff review, and a review of whether the 520-byte size filter should exist at all. All their findings were triaged.

**Departures from the proposal comment on the issue:** the new warning for a series with a fault note is dropped, because series 19, 26, and 29 report and carry a note today, so the check would be permanently yellow.

## What changes, file by file

- `packages/nged_data/src/nged_data/storage.py`
  - New `add_newest_file_of_each_series(all_files, new_files)`. It returns `new_files` plus the newest file of every series, without duplicates, in ascending `end_time` order. `download_and_parse_files` already processes files in that order and keeps the last metadata per series, so the newest file's metadata wins.
  - The `remove_small_files_from_listing` docstring no longer says the `information` field is always null, and points to the new function.
- `src/nged_substation_forecast/defs/assets.py` (`power_time_series_and_metadata`): call the new function after `select_new_rows`, download its result, and add a "Files downloaded" row to the file-listing summary. The `NoNewData` and metadata-upsert flow is unchanged.
- `src/nged_substation_forecast/defs/checks.py`
  - `_read_expected_ids_and_fault_notes` replaces `_read_expected_ids`. It also returns the non-null `information` of each series as a `dict[int, str]` from the same `scan_parquet`, selecting `information` only when the column exists, because it is `allow_missing`.
  - `_describe_power_freshness(result, fault_notes)` appends `NGED fault notes: 19: "…"; 33: "…".` with each note in full, a note for a silenced series included. `_to_asset_check_result` passes the notes through. `evaluate_power_freshness`, `PowerFreshnessResult`, and the Sentry report are unchanged.
- `packages/contracts/src/contracts/power_schemas.py`: the `information` description becomes "Free-text note from NGED about a known meter fault or a customer's status. Null for most series."
- Docs: `packages/nged_data/README.md` (the new function), `docs/live_service/operations.md` (the description sentence, with an invented example note), and `docs/roadmap/data-sources.md` (the "does not yet act on" bullet).

## Design-philosophy check

The change runs in production. The check stays `WARN` and `blocking=False`, and nothing new can raise out of it: the note read shares the existing guard. The ingest gains no new failure path beyond downloading and parsing one more file per silent series, which is as exposed as the files it already downloads. Sentry stays clean only while `report_power_freshness` is unchanged, because a note can name a generator.

## Tests

Each would fail on `main` because the new function and parameter do not exist.

- `add_newest_file_of_each_series` adds the newest small file after an older large file, and adds nothing for a series whose newest file is already listed.
- Asset: a series with an older large file and a newer data-less file carrying a note has that note in the metadata parquet, and the power still lands.
- `_describe_power_freshness` names a note for a silenced and for a reporting series, and is unchanged with no notes. `_read_expected_ids_and_fault_notes` returns the non-null notes, copes with no `information` column, and copes with no metadata table.

Fixtures and the docs example use invented note text, never a real string.

## Verification

The green-before-push set, plus `uv run mkdocs build --strict` and `pydoclint`.

## Risks and open questions

- **A silent series whose files all have `data: []` still raises nothing, but an hour in which every downloaded file has no readings raises `NoNewData`, and that hour does not refresh any metadata.** That cannot happen while any series reports. No change is planned.
- **A malformed small file stalls the ingest, as a malformed large file already does.** The earlier design isolated each file. Isolating every file is a separate change inside `download_and_parse_files`.
- **A file with `data: null`** (not `[]`) raises a different error from the one `download_and_parse_files` catches on the current Polars. Real files carry `data: []`, so nothing reaches this today. It is out of scope.
- **Cost at V2 scale.** One extra download per silent series per hour. The existing 3-day lookback already re-downloads about 13 files per reporting series per hour, and at 2,500 series that is the larger cost. It deserves its own issue.
- **A missing-half-hour detector** is a separate follow-up issue.
