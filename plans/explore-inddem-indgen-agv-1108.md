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

- **`fetch_system_series.py`: add two `SeriesSpec` entries, `elexon_inddem` and `elexon_indgen`,** copied from `elexon_ndf` (same `/datasets/<X>/stream` endpoint queried by publish time, one chunk per UTC day, a `boundary` column, every issue kept, `expect_daily_publish=True`). Only the value column differs: the raw field is `demand` for INDDEM and `generation` for INDGEN, written as `demand_mw` and `generation_mw`. The publish window starts on 2025-08-31, so that the 00 UTC issue exists for targets on 2025-09-01. The docstring states that INDDEM values are negative for import. The new keys are added to `ALL_SOURCES` and not to `ELEXON_DEFAULT_SOURCES`, so a default re-run does not download 790 more days. The gap check counts issues per day (the check in `run_series` records only days with no publication), so a day with a missing issue (2025-12-15 has 47, with 08:46 absent) is listed. A request with no `boundary` returns all 18 boundaries (verified by the simplicity reviewer: 49,824 rows for two days, 18 boundaries × 2,768 rows from 47 issues), so the download is about 790 requests.
- **`fetch_agv.py`, new, short.** Downloads `AGV_2025.zip` and `AGV_2026.zip` from `https://www.elexon.co.uk/open-data/` (the URL redirects to S3, so the client follows redirects), keeps the rows of one settlement run (SF; see "Design choices"), converts settlement date and period to UTC with `add_period_time` from `fetch_system_series`, and writes `elexon_agv/elexon_agv.parquet` and `lineage.json`, recording a sha256 for each zip because `AGV_2026.zip` is rewritten daily. No chunk cache, because the whole download is two files. The `data-validation` step also resolves what the `Estimate Indicator` field means (it is `T` on 89% of SF rows).
- **`fetch_pv_live.py`, new.** Calls PV_Live's `pes/{id}` endpoint for NGED's four licence areas: PES ids 11 (`_B`), 14 (`_E`), 21 (`_K`), and 22 (`_L`), checked against PV_Live's `pes_list`, which carries the GSP-group letter for each id (ids 19 and 20 are `_J` and `_H`, southern England, so the id order does not follow the letters). A request may not exceed 366 days, so each area takes at least two chunks, through `market_common.get_json` and `fetch_missing_chunks`. **PV_Live labels each half-hour by its end** (its README says the period 14:30 to 15:00 carries the timestamp 15:00), whereas AGV, INDDEM, and INDO are labelled by period start, so the script writes `time = datetime_gmt − 30 minutes`, and the `data-validation` step checks the result against a solar-noon profile. PV_Live covers only PV that does not take part in the Balancing Mechanism, and it revises its estimates, so the lineage records `updated_gmt`. Writes `pv_live/pv_live.parquet` with `installedcapacity_mwp`.
- **`market_common.write_readme`:** make the hard-coded "battery-versus-solar-PV study (pull/1094)" sentence a `purpose` argument with a default equal to today's sentence, so the 10 existing call sites are unchanged. `run_series` writes one README per spec, so `SeriesSpec` gains an optional `purpose` field that `run_series` passes on. The module docstrings of `fetch_system_series.py` and `market_common.py`, and the data table in `studies/README.md` (the `market/<source>/` row), list the new sources and scripts.

### The study in `studies/indgen_inddem/` (new folder, nothing under `packages/studies/`)

- **`build_tables.py`:** holds the CVA Change Circular 235 table as a dict literal (the boundary-to-zone-set map for N and B1 to B17, solved as an 18×17 0/1 system: zone Z12 belongs to no B boundary and appears only in N, so a 17×17 system would be singular; the code asserts full column rank; the circular's formulas are not transcribed a second time). The system is over-determined, which gives a sharper transcription check than any sign test: B16 − B11 must equal B9 − B17 − B8, and `report.md` prints the residual (the reviewer measured 0 ± 2 MW, rounding only, on 2,768 INDDEM pairs). The data folder comes from `study_dir_for` in `studies.sources`, derives zones, picks issues (the latest issue at or before a cut-off, and the 00 UTC issue), converts AGV to signed MW, joins PV_Live, and writes `report.md` printing every number the page quotes, including the transcribed table and the per-zone count of sign violations.
- **`make_charts.py`:** draws the figures. Loads the `dataviz` skill's rules and uses `plotting.ocf_theme`.
- **`README.md`:** maps each script to the page.

### Page and navigation

- **`docs/studies/indgen-inddem-and-gsp-take.md`, new.** Follows the `study` skill's structure, with the Summary opening with plain-language answers to the study's questions.
- **`mkdocs.yml`** nav entry and **`docs/studies/index.md`** entry.

## Design choices

- **AGV uses a single settlement run, SF,** so the whole window has one vintage. The latest run per period would mix R3, R2, R1, SF, and II across the window and show a quality step that comes from the run choice. SF covers settlement dates 2025-09-01 to 2026-09-20 (385 dates, complete), so every figure that uses AGV ends on 2026-09-20; the INDDEM and INDGEN figures run to 2026-09-30.
- **Two views of INDDEM per target half-hour:** the latest issue published at or before the start of the target settlement period, and the first issue of the target's UTC day (the latest issue published at or before 00:00 UTC). The first issue reaches 0 to about 28 hours ahead depending on the target half-hour (about 27.5 hours in summer and 28.5 hours in winter), so the page never calls it a day-ahead view, and its difference from the latest view is plotted against target hour, because the difference grows with lead by construction. Both views are plain filters in `build_tables.py`. The issue schedule follows UK local time (the long issue appears at 11:47 UTC in winter and 10:48 UTC in summer), so the figures of mean daily profile and of issue reach use UK local time, or split by season.
- **Signed MW:** AGV volumes are MWh per half-hour, import positive and export negative, doubled to MW. The unit is confirmed as MWh per half-hour (2 × the sum of AGV over the 14 groups divided by INDO has a median of 0.926, a 1st to 99th percentile range of 0.87 to 0.96, and a correlation of 0.996). Each group, date, period, and run appears once, as either an import or an export row.

## The figures, in the order the page runs them (eight)

1. **AGV against INDO (unit check).** AGV summed over the 14 groups and converted to MW, against the initial national demand outturn on disk under `downloads/market/elexon_demand_outturn/`, for one fixed week and as a scatter of all half-hours. The pass criterion is numeric: 2 × the sum of AGV divided by INDO near 0.9 to 0.95 confirms MWh per half-hour, near 0.46 would mean MW, and a midday dip would mean INDO is gross of embedded generation. INDO is also net of embedded generation, so the page does not attribute the roughly 7% gap to embedded generation (the ratio is flat through the day); the cause, perhaps transmission losses or transmission-connected demand, is stated as unexplained unless a source says otherwise.
2. **Zone signs (transcription check).** The share of half-hours where each derived zone has the wrong sign, for INDGEN and INDDEM.
3. **National INDDEM and INDGEN** over the year and over one fixed winter week and one fixed summer week.
4. **The 17 zones' mean daily profiles** as small multiples.
5. **Issue reach** against publish hour of day.
6. **The first issue of the day against the latest issue** for the same target half-hour, national only, as a difference against target hour.
7. **NGED's four groups against their matching zones:** AGV beside the zone's INDDEM, the fixed winter and summer weeks as small multiples. A table of mean level per candidate zone and group shows the match. The correlation uses anomalies: each series minus its own mean by settlement period, day type (weekday or weekend), and month. Subtracting only a mean daily profile does not work (on the 14 AGV groups the median correlation between two groups is 0.76 raw, 0.77 after subtracting a daily profile, and 0.47 after subtracting a profile by month and day type), so the anomaly definition is checked on pairs of AGV groups, the known-answer case, before it is applied to zones. The four NGED zones are chosen here, so figure 6 does not use them.
8. **AGV with and without PV_Live's solar** beside the matching zone's INDDEM.

**Every number on the page is exploratory.** The study names no planned contrasts because it tests no hypothesis. The page says so once, in "Data and methods", and states that no significance test is run.

## Design-philosophy check

All code is R&D: it fails fast and degrades nothing. It runs nowhere near production, adds no asset check, and delivers no hypothesis from `engineering-hypotheses.md`. No principle in `design-principles.md` is traded away. The data are public, so the anonymisation rule does not apply, and no NGED generator series is read.

## Tests

- **No new package module.** The study's scripts are checked by their own output. Each download writes `lineage.json` with exact row counts and counts of days and issues missing, and the `data-validation` checklist runs on the first chunk of each download. `build_tables.py` prints the join row counts, the transcribed zone table, and the zone sign violations into `report.md`.
- **`market_common.write_readme`'s new `purpose` argument** gets an assertion added to `packages/studies/tests/test_market_downloads.py`: the README text contains the given purpose and not the string "battery-versus-solar-PV" when the purpose is another study. That assertion fails on `main`, where the sentence is hard-coded.
- **The two new `SeriesSpec` entries** get a test in `packages/studies/tests/test_market_system_series.py`, in the style of `test_fetch_series_chunk_drops_rows_published_outside_the_chunk`: `get_json` is monkeypatched to return a fake INDDEM row carrying `demand` and a fake INDGEN row carrying `generation`, and the test asserts the value column is non-null. This fails when a spec copied from `elexon_ndf` keeps the wrong raw field name, because `frame_from_rows` turns a missing field into nulls without an error (the likely bug: 19.7 million null values downloaded with no warning).

## Docs to update

- The new page, the nav entry, and the `docs/studies/index.md` entry.
- **`studies/README.md`** (the data table row for `market/<source>/`).
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

## Findings the correctness review made, and what the plan did with them

- **Accepted:** the PV_Live PES ids were wrong, now 11, 14, 21, and 22 (verified against `pes_list`); the PV_Live end-of-interval timestamp, 366-day limit, scope, and revisions; the singular 17×17 zone system, replaced by the 18×17 system with the B16 identity check; the anomaly definition that removes the shared cycle; the wrong AGV-against-INDO expectation, now numeric; the "day-ahead" label, the lead confound, and local-time axes; the `SeriesSpec` test that could not catch null values; `ALL_SOURCES`, the `purpose` plumbing, the docstrings, and `studies/README.md`; the AGV run inconsistencies; the figure order; the issue-count gap check; the licence question.
- **Raised as a question, not adopted:** deriving the group-to-zone mapping from supplier PNs (see the open questions).
- **Rejected:** none.

## Risks and open questions

- **Which GSP groups belong in which zone is not published, but it may be derivable.** Supplier base BMU ids carry the GSP-group letter (`2__A…` to `2__P…`; one all-BMU PN request returned 574 supplier BMUs across all 14 groups), and INDDEM is a sum of PNs, so regressing each zone's INDDEM on per-group supplier-PN sums over a few dozen half-hours could identify the allocation nearly exactly, for a few dozen requests. A one-period check (2025-12-15 at 17:30 UTC) gave zone Z10 = −146 MW against about −1,970 MW for group `_B`'s suppliers, which sits against the desk study's guess that Z10 is the East Midlands. Question for the maintainer: derive the mapping this way, or let figure 7 show a plausible match only? Recommendation: derive it, because it turns the study's central comparison from a guess into a measurement, and the cost is small.
- **Licences.** The plan names no licence for PV_Live or for Elexon's Open Settlement Data, and the README writer needs both. Under the `study` skill, accepting a data licence is the maintainer's decision. Question for the maintainer: is attribution under Elexon's open data licence (as announced) and PV_Live's published terms acceptable? The implementer finds PV_Live's terms before the first run.
- **Should the study run one Opus science review instead of two?** The `study` skill requires two. This study makes no contrast, which lowers the risk, but the unit and the zone match can still be wrong. Recommendation: keep two.
- **Does the maintainer want a polling job for NGED's live flow data?** Out of scope here. It would only start accumulating the regional history the desk study found missing, and could be its own issue.
