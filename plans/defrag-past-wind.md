# Restructure the past-wind study page into one coherent whole

## The problem

`docs/studies/weather-products-for-past-wind.md` (1,606 lines, 23 figures) grew one section per new idea: five products on the main row set, ICON-DREAM-EU on its own row set, ECMWF's HRES and ENS day 0 on a shorter row set, and one nearby MIDAS weather station. Each section repeats its own planned-contrast definitions, caveats and reproduce block, and repeats methods that the past-solar page has already moved to the shared Methods page. A reader cannot find one answer to "which product should Flexpectation use for past wind". The figure numbers are baked into 8 chart scripts as scattered constants and into SVG titles. Two negative intervals in the station section print without a sign.

## The planned solution

PR A (code and charts) puts the headline results into one leaderboard and one contrast chart of four row-set blocks, each block repeating ERA5 as a lighter reference row, reusing the stacked-block helpers that the past-solar PR A adds to `studies.charts`. PR B (prose) rewrites the page to the 12-part paper structure in `.claude/skills/study/SKILL.md` by moving sentences verbatim, rewriting only connective text, deleting methods that the shared Methods page already holds and linking to it. New reanalysis arms (CERRA 100 m wind, NORA3, ICON-DREAM-EU as a slot) follow as separate plans, each on its own row set with its own refitted reference rows.

## Verdict, size and departures

**Verdict:** worth doing, as directed by the coordinator. **Size: complex.** The five triggers:

1. What gets stored: yes. A new write-once folder `data/studies/beam_diffuse_split/past_weather_v2/wind_leaderboard/` (report and `intervals.parquet`) and about 12 SVGs redrawn. No Delta table, Patito model or asset.
2. Production serving path: no. Nothing under `defs/`, `ml_core` or `xgboost_forecaster`.
3. Degradation rule: no.
4. More than one defensible design: yes. Whether blocks reuse the solar helpers or wind gets its own, how the station block (contrasts against ERA5's 10 m wind, not a product ranking) sits in a leaderboard, and which figures merge.
5. Callers not nameable without searching: yes. Figure numbers sit in 8 scripts and in SVG text, and inbound links and anchors come from the survey, sibling pages and script docstrings.

Reviews bought: both plan reviews, all diff reviews, plus the coordinator's extras (Opus outline review, fact check of every changed sentence containing a number, scope-of-claims review, two Opus science reviews on PR A's leaderboard, network-planner and ML-engineer persona reviews, one-rule prose sweeps, a Sonnet read of the rendered HTML).

**Departures from the brief:**

- PR A stacks on `defrag-past-solar` (PR #911), because the stacked-block helpers, `studies.page_numbers` and `FIGURE_NUMBERS` live there and are not on `main`. It rebases onto `main` when the solar PR A merges.
- PR A is not merged and no chart is redrawn until the coordinator says the wind numbers are final (the D1 refit of `wind_icon_dream`, #906). Until then only code that reads saved losses is written, and any number it prints from `wind_icon_dream` is provisional.
- New arms (CERRA, NORA3) are not part of these two PRs. Each is a follow-on plan after PR B merges.
- PR B does not start until the coordinator confirms the Methods page has merged.

## What changes, file by file

PR A (code; reads saved losses, no refit):

- `studies/beam_diffuse_split/figure_numbers.py` (from the solar branch): a wind map beside the solar one, or a second mapping in the same module. The 8 wind chart scripts import it in place of their `FIGURE_*` constants (`wind_product_charts.py`, `ens_hres_past_wind_charts.py`, `station_wind_arms_charts.py`, `wind_icon_dream_charts.py`, and the era5-by-year and served-wind scripts wherever they hold a number). Provisional map, 23 -> 16, for the outline review to confirm: new 1 = old 1, 11, 13, 21 (leaderboards); new 2 = old 2, 12, 16 planned panels, 21 headline (contrasts); 3 = old 3 (domain); 4 = old 4, 14; 5 = old 5, 15; 6-10 = old 6-10; 11 = old 16; 12 = old 17; 13 = old 18; 14 = old 19, 20; 15 = old 22; 16 = old 23.
- `studies/beam_diffuse_split/past_wind_leaderboard.py` (new, modelled on `past_solar_leaderboard.py`): reads saved losses of the four row sets (main; ICON-DREAM-EU's own; ECMWF from December 2024; station arms), recomputes per-arm intervals and each planned contrast, and writes `past_weather_v2/wind_leaderboard/report.md` and `intervals.parquet`. It stops unless every recomputed number matches a printed one at printed precision (`assert_matches_printed`). Each block declares its own device and setting; the script asserts that no contrast mixes devices (published wind arms are CPU fits).
- `past_wind_leaderboard_charts.py` (new): Figure 1 and Figure 2 as four-block stacked charts built with the solar branch's `RowSetBlock`, `stacked_leaderboard` and `stacked_contrasts`. The reference row is ERA5 in every block. Each block carries its own uncovered-month share in its label or caption, from the #906 measurement (main 16.2%, ICON-DREAM-EU 25.1%; the other two blocks' shares are read from the fold report, not assumed).
- Existing wind chart scripts: merged figures redrawn from the map; SVG titles carry the new numbers; time-series charts keep days 1-7, month and year in text only, % of capacity and `aria=False`.
- `check_page_numbers.py` guard (fact 4): fix the two unsigned negative intervals in the station section, found by reading that section and running the guard on it, and add the one line it needs to the wind page's Reproducing block. The fix is prose, so it lands in PR B; PR A only makes the guard's per-section calls for wind work through `studies.page_numbers`.
- `studies/beam_diffuse_split/README.md`: figure references.

PR B (prose, stacked on PR A, after Methods merges):

- Rewrite `weather-products-for-past-wind.md` (moved to `docs/studies/past-weather/wind.md` by the solar PR B, not by this plan) into the 12 parts: Title; Summary (two paragraphs, headline figures, scoped take-home bullets); AI disclaimer; Key findings; Introduction with the products table; Data and methods (study-specific: the wind hub heights and processing, the four row sets, the ECMWF and station arms, linking the shared Methods page for row-set definitions, planned contrasts, capacity normalisation, bootstrap intervals, the second setting and shared gotchas); Results (one subsection per finding, "The XGBoost models work" first); Discussion "What to use"; Limitations; Scope; Data and code availability; Reproducing the figures. Sentences move verbatim; only connective text is rewritten; duplicates of Methods content are deleted.
- New paragraph in Data and methods on the products' processing (fact 2), worded only after the coordinator confirms: Open-Meteo's ERA5 100 m and 10 m wind matches native CDS ERA5 to within Open-Meteo's 0.1 km/h rounding (RMS 0.06 km/h, ratio of means 1.0000, nearest cell, no interpolation, no 0.98 factor); W2 and W3 share one ERA5 cell, so the evidence is two independent series; Open-Meteo's ICON-EU matches Dynamical.org's DWD ICON-EU for radiation, temperature and 10 m wind to within rounding. Unchecked: the ICON-family 100 m wind (a documented factor of 0.98 on the native 120 m speed; no native ICON 100 m wind is on disk), ICON-D2 and UKV wind, and IFS. Wind is instantaneous in Open-Meteo's products and radiation is hour-ending. The device (CPU) is stated.
- Each block's limitation sentence carries its own uncovered-month share.
- Inbound links and anchors into the wind page are re-pointed (survey page, siblings, roadmap, script docstrings), as the solar PR B does for its own page.

Follow-on plans (not in these PRs): CERRA wind at 100 m (then 75, 50 and 150 m once downloaded), NORA3 (once the Download coordinator confirms it is on disk), and an ICON-DREAM-EU slot. Each new arm gets its own row set with its own refitted ERA5-style reference rows fitted on the fixed folds (`rotate_folds` over `with_eras`, `raise_on_uncovered_months`), a hub-height column and a domain-coverage row, and is fitted on one device, CPU unless the coordinator says otherwise.

## Design-philosophy check

R&D and documentation only. Nothing runs in production, so nothing degrades, and the studies fail fast: the leaderboard script stops on any mismatch. No asset check is added. No hypothesis is cited. The Discussion states which findings support the choices in the live-service roadmap and does not commit the project to work.

## Tests

Each test states what bug it catches.

- Leaderboard rows: each block holds exactly one row per arm, every bootstrap `n_rows` equals the block's row count, and a frame holding two settings or two devices under one arm name raises or is filtered explicitly. Catches the silent cross-join found in the solar losses, and a contrast that mixes CPU and GPU fits.
- `FIGURE_NUMBERS`: wind values 1 to 16 with no gap or duplicate, and every wind-page SVG has a key.
- The guard function perturbs one printed number and expects `assert_matches_printed` to stop.
- Signed intervals: a negative interval printed without its sign is caught by the page-number guard (a fixture with `[0.12, −0.03]` written `[0.12, 0.03]` fails); this test fails on `main` if the guard cannot see the case, and if it can, the failing check is run on the station section as a gate instead.
- Gates as scripts (scratch, uncommitted unless noted):
  - Number conservation: every decimal, every integer of 10 or more, and every `X [a, b]` triple in the new page appears in the old page (fetched from `origin/main` at run time) or the new report; every dropped number is a duplicate still present elsewhere and signed off by a reviewer. Blind spots stated in the PR body: it cannot see a number attached to the wrong row set, product or setting, or a deleted qualifier.
  - `check_page_numbers.py` section by section, one call per report, reusing the solar session's fix for its false failures.
  - Planned/exploratory status preserved per contrast (table in the PR body).
  - Every absolute error (a decimal followed by `%`) sits in a paragraph naming its row set.
  - SVG title number equals caption number, and SVG `Figure \d+` text matches the map.
  - `git grep` for the old page path returns only redirect keys.

## Docs to update

The wind page, `studies/beam_diffuse_split/README.md`, and every inbound link and anchor. Everything is written for the present. No roadmap item completes; this file is deleted at ship time into the PR body.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict       # then read the rendered HTML of the wind page
uv run pre-commit run --all-files
```

Compute: no refit in PR A. The leaderboard script's first run needs the runner slot from the coordinator, writes only to its own new folder, and runs only after the coordinator confirms the wind numbers are final. Every new or changed script gets an Opus review before it runs.

## Hard rules restated

Generators are W1 to W3 only. No MIDAS station names, coordinates, farm mapping or per-generator distances (pooled ranges only). Time-series charts show days 1-7, month and year in text only, % of capacity, `aria=False` on data marks. Scratch under `.claude/worktrees/scratch/`. Nothing near an issue number says close, fix or resolve.

## Risks and open questions

1. The station block is a set of contrasts against ERA5's 10 m wind rather than a product ranking. Recommendation: draw it as the fourth block of Figure 1 with 10 m in the label and hub height in the column, so the reader does not compare it with 100 m rows.
2. Figure 1 invites cross-block comparison of absolute errors from different row sets. Recommendation: block labels carry dates and row counts, and every block repeats ERA5.
3. Base branch: stacking on `defrag-past-solar` means a rebase when it merges. Recommendation: accept.
4. If the D1 refit moves wind numbers, the leaderboard is computed on the new losses and the conservation gate compares against the post-refit old page.
