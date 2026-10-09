# Plan: explore Elexon INDDEM, INDGEN, and GSP-group take data (issue 1108)

**Problem.** Elexon publishes two series, INDDEM and INDGEN, that sum the final Physical Notifications (PNs) of every Balancing Mechanism Unit (BMU) into a national total and 17 overlapping boundaries. Nobody on the project has looked at them. The desk study on issue 1106 found that the 17 boundaries are nested sums of 17 transmission study zones, that the sums include embedded BMUs, and that each issue reaches at most about 40 hours ahead. It recommended stopping, and the maintainer asked for a cheaper step: look at the data before deciding whether any forecasting experiment is worth running.

**Planned solution.** A descriptive study with no forecasting model, no contrast, and no hypothesis test. It downloads 13 months (2025-09-01 to 2026-09-30) of INDDEM and INDGEN, of Elexon's settled GSP-group take (AGV, CDCA-I029), and of PV_Live's solar generation for NGED's four licence areas. It then builds a docs page made mostly of figures: first two known-answer checks (the AGV unit, and the sign of the zones derived from the boundaries), then what INDDEM and INDGEN look like, then how they compare with AGV for NGED's four licence areas.

## Verdict, size and departures

**Verdict: worth doing, as the maintainer described.** The cost is about 800 small API requests, two zip downloads of about 21 MB, and no fitting. The page settles three facts the desk study could only read: the unit of AGV, how far the derived zones obey their sign rule, and how much a day-ahead issue differs from the latest issue.

**Size: complex.** The five triggers:

- **What gets stored:** fires. A published page under `docs/studies/` and parquet files under `data/studies/`. Under the `study` skill a published page counts as stored.
- **Production serving path:** does not fire. Only `studies/` and `docs/` change.
- **Degradation rule:** does not fire.
- **More than one defensible design:** fires. The zone-to-group comparison, the INDDEM issue rule, and the AGV settlement-run rule each admit several designs.
- **Callers that cannot be named without searching:** does not fire. All code is new, and the only existing code touched is `market_common.write_readme`'s hard-coded study name and one `SERIES` table.

Reviews: both plan reviews (simplicity, then correctness). After implementation: an Opus code review of each script before its first run, two Opus scientific-validity reviews, the diff reviews, and a `prose-review` of the page. Nothing changes under `packages/studies/`, so the mutation pass does not run. The `study` skill asks for two science reviews even though this study makes no contrast, because the claims about units and about which groups match which zones can be wrong in ways a reader cannot see.

**Departures from the issue body.** None in scope. One addition: a check of the AGV unit against the national demand outturn already on disk.

## What changes, file by file

### Downloads in `studies/market_downloads/`

- **`fetch_system_series.py`: add two `SeriesSpec` entries, `elexon_inddem` and `elexon_indgen`,** copied from `elexon_ndf` (same `/datasets/<X>/stream` endpoint queried by publish time, one chunk per UTC day, a `boundary` column, every issue kept, `expect_daily_publish=True`). Only the value column differs (`demand_mw` and `generation_mw`). The docstring states that INDDEM values are negative for import. A request with no `boundary` returns all 18 boundaries (verified by the simplicity reviewer: 49,824 rows for two days, 18 boundaries × 2,768 rows from 47 issues), so the download is about 790 requests.
- **`fetch_agv.py`, new, short.** Downloads `AGV_2025.zip` and `AGV_2026.zip` from `https://www.elexon.co.uk/open-data/` (the URL redirects to S3, so the client follows redirects), keeps the rows of one settlement run (SF; see "Design choices"), converts settlement date and period to UTC with `add_period_time` from `fetch_system_series`, and writes `elexon_agv/elexon_agv.parquet` and `lineage.json`. No chunk cache, because the whole download is two files.
- **`fetch_pv_live.py`, new.** Calls PV_Live's `pes/{id}` endpoint for NGED's four licence areas (PES ids 11, 14, 19, 20, mapped in order to GSP groups `_B`, `_E`, `_K`, `_L`; the implementer confirms the mapping against PV_Live's `pes_list` before the first run and the page states the source), in chunks that respect the API's limits through `market_common.get_json` and `fetch_missing_chunks`. Writes `pv_live/pv_live.parquet` with `installedcapacity_mwp`.
- **`market_common.write_readme`:** make the hard-coded "battery-versus-solar-PV study (pull/1094)" sentence a `purpose` argument, so the new downloads' README files name the right study. About 5 lines.

### The study in `studies/indgen_inddem/` (new folder, nothing under `packages/studies/`)

- **`build_tables.py`:** holds the CVA Change Circular 235 table as a dict literal (the boundary-to-zone-set map, solved as a 17×17 0/1 system; the circular's formulas are not transcribed a second time), derives zones, picks issues (the latest issue at or before a cut-off, and the 00 UTC issue), converts AGV to signed MW, joins PV_Live, and writes `report.md` printing every number the page quotes, including the transcribed table and the per-zone count of sign violations.
- **`make_charts.py`:** draws the figures. Loads the `dataviz` skill's rules and uses `plotting.ocf_theme`.
- **`README.md`:** maps each script to the page.

### Page and navigation

- **`docs/studies/indgen-inddem-and-gsp-take.md`, new.** Follows the `study` skill's structure, with the Summary opening with plain-language answers to the study's questions.
- **`mkdocs.yml`** nav entry and **`docs/studies/index.md`** entry.

## Design choices

- **AGV uses a single settlement run, SF,** so the whole window has one vintage. The latest run per period would mix RF, R3, and II across the window and show a quality step that comes from the run choice. SF covers settlement dates up to about 2026-09-20, so the AGV comparison ends there; the unit check and the INDDEM/INDGEN figures run to 2026-09-30. If a figure needs the last ten days, the page says which run fills them.
- **Two views of INDDEM per target half-hour:** the latest issue published at or before the start of the target settlement period, and the 00 UTC issue of the target's UTC day (the latest issue published at or before 00:00 UTC, reaching about 27.5 hours ahead). Both are plain filters in `build_tables.py`.
- **Signed MW:** AGV volumes are MWh per half-hour, import positive and export negative, doubled to MW. The unit is verified against INDO before the page quotes it.

## The figures, in the order the page runs them (eight)

1. **AGV against INDO (unit check).** AGV summed over the 14 groups and converted to MW, against the initial national demand outturn on disk under `downloads/market/elexon_demand_outturn/`, for one fixed week and as a scatter of all half-hours. AGV is net of embedded generation, so the figure states how far below INDO it should sit.
2. **Zone signs (transcription check).** The share of half-hours where each derived zone has the wrong sign, for INDGEN and INDDEM.
3. **National INDDEM and INDGEN** over the year and over one fixed winter week and one fixed summer week.
4. **The 17 zones' mean daily profiles** as small multiples.
5. **Issue reach** against publish hour of day.
6. **The 00 UTC issue against the latest issue** for the same target half-hour, national and for the four zones that match NGED's areas best.
7. **NGED's four groups against their matching zones:** AGV beside the zone's INDDEM, the fixed winter and summer weeks as small multiples. A table of mean level per candidate zone and group shows the match; a correlation of daily-profile anomalies (each series minus its own mean daily profile) replaces the correlation of raw series, which would sit near 0.9 everywhere because every series shares the daily and weekly cycle.
8. **AGV with and without PV_Live's solar** beside the matching zone's INDDEM.

**Every number on the page is exploratory.** The study names no planned contrasts because it tests no hypothesis. The page says so once, in "Data and methods", and states that no significance test is run.

## Design-philosophy check

All code is R&D: it fails fast and degrades nothing. It runs nowhere near production, adds no asset check, and delivers no hypothesis from `engineering-hypotheses.md`. No principle in `design-principles.md` is traded away. The data are public, so the anonymisation rule does not apply, and no NGED generator series is read.

## Tests

- **No new package module, so no new package test.** The study's scripts are checked by their own output. Each `SeriesSpec` download writes `lineage.json` with exact row counts and gap counts, and the `data-validation` checklist runs on the first chunk of each download. `build_tables.py` prints the join row counts, the transcribed zone table, and the zone sign violations into `report.md`.
- **`market_common.write_readme`'s new `purpose` argument** gets an assertion added to the existing `packages/studies/tests/test_market_downloads.py`: the README text contains the given purpose and not the string "battery-versus-solar-PV" when the purpose is another study. That assertion fails on `main`, where the sentence is hard-coded.
- **The two new `SeriesSpec` entries** get an assertion in the same test file that each spec's fetch parameters send `publishDateTimeFrom` and `publishDateTimeTo` and no `boundary` filter, and that the value column is named as planned.

## Docs to update

- The new page, the nav entry, and the `docs/studies/index.md` entry.
- **`docs/background/gb-battery-scheduling.md`** has two out-of-date statements (the Elexon API does not define the boundaries, and whether the sums include embedded BMUs is untested). The desk study on issue 1106 made them out of date, not this study, so they go in a small separate PR off `main`. Only a line about the zone mapping can wait for this study to merge.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest --run-studies -n auto packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict   # then read the rendered page and its figures
```

Plus every other CI step (`pydoclint`, the docs-link checker), run locally before the push.

## Order of work

1. Plan reviews, then triage.
2. Add the two `SeriesSpec` entries, `fetch_agv.py`, `fetch_pv_live.py`, and the `write_readme` change. Each script gets a fresh Opus code review before its first run. Run the first chunk of each download, run the `data-validation` checklist on it, then run the rest.
3. Write `build_tables.py` and `make_charts.py`, review them, run them, and read the report.
4. First Opus science review, then fixes and re-runs; draft the page; second Opus science review; diff review; `prose-review`; persona reviews of the page.

## Findings the simplicity review made, and what the plan did with them

- **Accepted:** reuse `fetch_system_series.py` for INDDEM and INDGEN (finding 1); no `packages/studies` modules (2); one AGV settlement run (3); fewer figures, no INDGEN split, fixed weeks instead of weeks chosen by a rule, and a correlation of daily-profile anomalies instead of raw series (4); one transcription of the zone table and no round-trip test (5); PV_Live for four areas only (6); move the battery-scheduling correction to its own PR (7); the `purpose` argument (8); no chunk cache for AGV (9).
- **Rejected:** none.

## Risks and open questions

- **Which GSP groups belong in which zone is not published.** Figure 7 lets the data show a plausible match, labelled exploratory, and the page claims no more than the figure shows. Recommendation: do not try to derive a definitive mapping in this study.
- **PV_Live area to GSP group correspondence.** The simplicity reviewer reports the standard mapping as IDs 10 to 23 in order to `_A` to `_P`, skipping I and O. The implementer confirms it from PV_Live's `pes_list` and a second source before the first run.
- **Should the study run one Opus science review instead of two?** The `study` skill requires two. This study makes no contrast, which lowers the risk, but the unit and the zone match can still be wrong. Recommendation: keep two.
- **Does the maintainer want a polling job for NGED's live flow data?** Out of scope here. It would only start accumulating the regional history the desk study found missing, and could be its own issue.
