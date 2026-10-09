# Plan: explore Elexon INDDEM, INDGEN, and GSP-group take data (issue 1108)

**Problem.** Elexon publishes two series, INDDEM and INDGEN, that sum the final Physical Notifications (PNs) of every Balancing Mechanism Unit (BMU) into a national total and 17 overlapping boundaries. Nobody on the project has looked at them. The desk study on issue 1106 found that the 17 boundaries are nested sums of 17 transmission study zones, that the sums include embedded BMUs, and that each issue reaches at most about 40 hours ahead. It recommended stopping, and the maintainer asked for a cheaper step: look at the data before deciding whether any forecasting experiment is worth running.

**Planned solution.** A descriptive study with no forecasting model, no contrast, and no hypothesis test. It downloads 13 months (2025-09-01 to 2026-09-30) of INDDEM and INDGEN, of Elexon's settled GSP-group take (AGV, CDCA-I029), and of PV_Live's solar generation per distribution licence area. It then builds a docs page made mostly of figures: first a known-answer check that the zone formulas and the AGV unit are right, then what INDDEM and INDGEN look like, then how they compare with AGV for NGED's four licence areas.

## Verdict, size and departures

**Verdict: worth doing, as the maintainer described.** The cost is one download of about 14,000 small API requests (or fewer, see the open questions) and 21 MB of zips, and no fitting. The page's value is that it settles three facts the desk study could only read: the unit of AGV, the zone-to-group correspondence, and how much a day-ahead issue differs from the latest issue.

**Size: complex.** The five triggers:

- **What gets stored:** a published page under `docs/studies/` and parquet files under `data/studies/`. Under the `study` skill a published page counts as stored, so this trigger fires.
- **Production serving path:** not touched. Nothing under `src/` or any package except `packages/studies` changes.
- **Degradation rule:** not touched.
- **More than one defensible design:** fires. The zone-to-group mapping, the rule for choosing an INDDEM issue per target half-hour, and the rule for choosing an AGV settlement run each admit several designs.
- **Callers that cannot be named without searching:** does not fire. All code is new, and only the new scripts call it.

Reviews: both plan reviews (simplicity, then correctness). After implementation: the `study` skill's Opus code review of each script before its first run, two Opus scientific-validity reviews, the diff reviews, and a `prose-review` of the page. The `study` skill asks for two science reviews even though this study makes no contrast, because the claims about the zone mapping and about units can be wrong in ways a reader cannot see.

**Departures from the issue body.** None in scope. Two additions the issue does not name: a check of AGV's unit against the national demand outturn already on disk, and a correction of the battery-scheduling page's two out-of-date statements (see "Docs to update").

## What changes, file by file

### Machinery in `packages/studies/src/studies/` (tested)

- **`agv.py`, new.** `latest_run_per_period(frame)` keeps, for each `(gsp_group, settlement_date, settlement_period, import_export)`, the row from the latest settlement run, ranked II < SF < R1 < R2 < R3 < RF < DF by Elexon's chronology and never by string order. `signed_take_mw(frame)` converts a take volume in MWh per half-hour into signed MW (import positive, export negative, summed to one value per group and period). The unit is checked against the national demand outturn before the page quotes it.
- **`cva_zones.py`, new.** The boundary-to-zone table from CVA Change Circular 235, Appendix 1, as data, and `zones_from_boundaries(frame)`, which applies the circular's formulas (for example Z12 = N − B9 − B12 − B14 − B15, Z10 = B9 − B17 − B8). The table is transcribed from the circular's PDF and printed into the report so a reviewer can check it against the source.
- **`indgen_issues.py`, new.** `latest_issue_at_or_before(frame, as_of)` picks the issue with the greatest publish time at or before a cut-off, for each target half-hour. `issue_reach(frame)` reports how many half-hours each issue covers.

### Downloads in `studies/market_downloads/` (scripts reviewed before first run)

- **`fetch_indgen_inddem.py`, new.** Uses `market_common` (resumable per-chunk cache, `README.md`, `lineage.json`). One request per UTC day and dataset, asking for all boundaries if the API allows, or one request per boundary if not. The first chunk is measured before committing to the rest, and the `data-validation` checklist runs on it. Writes `data/studies/downloads/market/elexon_indgen/` and `elexon_inddem/`, one tidy parquet each, with `publish_time`, `start_time`, `boundary`, and `value_mw`.
- **`fetch_agv.py`, new.** Downloads `AGV_2025.zip` and `AGV_2026.zip` from `https://www.elexon.co.uk/open-data/` (the URL redirects to S3, so the client follows redirects), reads the monthly CSVs, converts settlement date and period to UTC with `market_common.period_start_utc`, and writes `elexon_agv/elexon_agv.parquet` with every settlement run kept. Selecting the latest run happens later in `studies.agv`, so the raw download stays reproducible.
- **`fetch_pv_live.py`, new.** Calls PV_Live's `pes/{id}` endpoint for the 14 distribution licence areas plus the national series, in chunks that respect the API's limits, and writes `pv_live/pv_live.parquet` with `installedcapacity_mwp`.

### The study in `studies/indgen_inddem/` (new folder)

- **`build_tables.py`:** builds the joined half-hourly tables (zones from boundaries, latest-run AGV, gross-demand estimate = AGV plus PV_Live where the area correspondence holds) and writes a `report.md` that prints every number the page quotes.
- **`make_charts.py`:** draws the figures. It loads the `dataviz` skill's rules and uses `plotting.ocf_theme`.
- **`README.md`:** maps each script to the page.

### Page and navigation

- **`docs/studies/indgen-inddem-and-gsp-take.md`, new.** Follows the `study` skill's structure. The Summary opens with plain-language answers to the study's questions.
- **`mkdocs.yml`** nav entry and **`docs/studies/index.md`** entry.

## The figures, in the order the page runs them

1. **Known-answer checks first.**
   - **Unit of AGV:** summed over the 14 groups and converted to MW, AGV plotted against the initial national demand outturn (INDO) already on disk under `downloads/market/elexon_demand_outturn/`, for a week and as a scatter of all half-hours. AGV is net of embedded generation, so it should sit below INDO by roughly the embedded generation; the figure states what is and is not expected.
   - **Zone nesting:** the derived zones' sign by boundary, with the share of half-hours violating the sign rule.
2. **INDDEM and INDGEN themselves:** national total over a typical week per season and over the year; each boundary's daily profile; how INDGEN splits into interconnectors, embedded, and transmission-connected plant, using the one-day PN comparison from the desk study as the only decomposition.
3. **Issues:** how far ahead each issue reaches by publish time of day; how the value for one target half-hour changes between the 00 UTC issue and the latest issue before it (the day-ahead view against the latest view); the zonal erratic behaviour the desk study saw at one zone, shown for all zones.
4. **Against AGV:** a 17-zone by 14-group correlation heatmap of de-meaned half-hourly series, labelled exploratory; for NGED's four groups (`_B`, `_E`, `_K`, `_L`), time series of AGV beside the best-matching zone's INDDEM for a few weeks chosen by a stated rule (the clearest, most variable, and dullest weeks), and a scatter.
5. **Embedded solar:** PV_Live's installed capacity and generation per NGED area, and AGV with and without the solar term added, beside the matching zone's INDDEM.

**Every number on the page is exploratory.** The study names no planned contrasts because it tests no hypothesis. The page says so once, in "Data and methods", and states that no significance test is run, so no interval claims significance.

## Design-philosophy check

All code is R&D: it fails fast and degrades nothing. It runs nowhere near production, adds no asset check, and delivers no hypothesis from `engineering-hypotheses.md`. No principle in `design-principles.md` is traded away. The data are public, so the anonymisation rule does not apply, and no NGED generator series is read.

## Tests (in `packages/studies/tests/`)

- **`test_agv.py`:** a frame with all seven runs for one period returns the DF row (a string-order `max` would return SF, so the test fails on that bug). A period with only II and SF returns SF. Import and export rows for one group and period combine into one signed value. Unequal group sizes give unequal MW.
- **`test_cva_zones.py`:** zones built from known non-negative zone values and passed through the boundary sums are recovered exactly, and the table matches the transcription in the report for three spot-checked boundaries (B13 = Z17, B17 = Z11, B10 = Z16 + Z17). A zone with a wrong sign in the table fails the round trip.
- **`test_indgen_issues.py`:** with issues at 05:16, 05:46, and 06:16 the cut-off 06:00 returns the 05:46 issue and never the 06:16 one (the no-lookahead test), and a cut-off before the first issue returns no row rather than an error.
- **Scripts** are checked by their own output. Each fetch script's `lineage.json` carries exact row counts and gap counts, and `build_tables.py` prints the join row counts into `report.md`.

## Docs to update

- The new page, the nav entry, and the `docs/studies/index.md` entry.
- **`docs/background/gb-battery-scheduling.md`**, the physical notifications section: two statements are now out of date (that the Elexon API does not define the boundaries, and that the embedded-BMU question is untested). Rewrite them to describe the circular's definitions and the verified embedded-BMU result, and keep the statement that no source we found maps the boundaries to NGED's licence areas, if the study's mapping figure does not change it.
- **`studies/market_downloads/README.md`** (or the folder's equivalent) for the three new fetch scripts.

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
2. Write the three `packages/studies` modules and their tests, with the zone table transcribed from the PDF.
3. Write the three fetch scripts. Each gets a fresh Opus code review before its first run. Run the first chunk of each download, run the `data-validation` checklist on it, then run the rest.
4. Write `build_tables.py` and `make_charts.py`, review them, run them, and read the report.
5. First Opus science review, then fixes and re-runs; draft the page; second Opus science review; diff review; `prose-review`; persona reviews of the page.

## Risks and open questions

- **Does the Insights API return all boundaries for one request?** If it does not, the downloads take about 14,000 requests (395 days × 2 datasets × 18 boundaries) instead of about 800. The first chunk settles it. Recommendation: measure first and, if it is 14,000 requests, keep the download but run it at the existing 4 threads and cap the window at 13 months.
- **Which GSP groups belong in which zone is not published.** The plan lets the data show a plausible mapping, in a heatmap labelled exploratory, and the page makes no claim beyond what the heatmap shows. Recommendation: do not try to derive a definitive mapping in this study.
- **PV_Live area to GSP group correspondence.** PV_Live labels areas by distribution licence area (for example `WPD (Midlands)`), and AGV labels them `_A` to `_P`. The correspondence is a standard Elexon table. The implementer must confirm it from a source before adding the two, and the page states the source.
- **Should the study run one Opus science review instead of two?** The `study` skill requires two. This study makes no contrast, which lowers the risk, but the unit and the mapping can still be wrong. Recommendation: keep two.
- **Does the maintainer want a polling job for NGED's live flow data?** Out of scope here. It would only start accumulating the regional history the desk study found missing, and could be its own issue.
