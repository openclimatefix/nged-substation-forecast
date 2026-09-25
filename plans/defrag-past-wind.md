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

- `studies/beam_diffuse_split/figure_numbers.py` (from the solar branch): a wind map beside the solar one, or a second mapping in the same module. The 8 wind chart scripts import it in place of their `FIGURE_*` constants (`wind_product_charts.py`, `wind_icon_dream_charts.py`, `ens_hres_past_wind_charts.py` and `station_wind_arms_charts.py`, the four chart scripts; `check_served_wind.py` holds no figure numbers). Figure numbers also sit in `ens_hres_past_wind_charts.py` docstrings and a `ValueError` message and in `wind_product_charts.py` chart text ("Figure 9"); each is updated from the map. Figure 3 is `docs/roadmap/assets/weather_product_domains.svg`, shared with the solar and blending pages, and has no figure text. Provisional map, 23 -> 15, for the outline review to confirm: new 1 = old 1, 11, 13, and the absolute errors of 21 (leaderboards); new 2 = old 2, 12, 13's planned panels, and the contrasts of 21; 3 = old 3 (domain); 4 = old 4, 14; 5 = old 5, 15; 6 = old 6; 7 = old 7, 20, 23 (planned contrasts per farm: UKV, ECMWF, station); 8-10 = old 8-10; 11 = old 16 (ECMWF fold-design robustness); 12 = old 17; 13 = old 18; 14 = old 19 (time of day); 15 = old 22. Old 6 (April to September) and old 22 (August to December) split the year differently, so they stay apart. Only `ens_hres_past_wind_charts.py` and `station_wind_arms_charts.py` hold `FIGURE_*` constants; in `wind_product_charts.py` and `wind_icon_dream_charts.py` the numbers appear in docstrings and SVG text only.
- `studies/beam_diffuse_split/past_wind_leaderboard.py` (new; about 150 lines of `RowSet` declarations, importing the generic scoring, report and interval code from `past_solar_leaderboard.py` rather than copying it): reads saved losses of the four row sets (main; ICON-DREAM-EU's own; ECMWF from December 2024; station arms), recomputes per-arm intervals and each planned contrast, and writes `past_weather_v2/wind_leaderboard/report.md` and `intervals.parquet`. It stops unless every recomputed number matches a printed one at printed precision (`assert_matches_printed`). Each block declares its setting. The solar script hard-codes more than three constants, so PR A first generalises it, in this PR's own diff and after the solar PR A has merged (so no change lands in a PR under review): `RowSet` gains a reference arm (`era5_wind` in the main, ICON-DREAM-EU and ECMWF blocks, `era5_10m_wind` in the station block; solar has `era5_global`), the planned-section prefix (the main report's is "Deciding contrasts, named before the run", and it holds one exploratory block marked only by a trailing sentence, so the parser needs a per-contrast rule, not a per-heading one), the heading regex ("common site-hours" in the main and ICON-DREAM-EU reports, "common farm-hours" in the ECMWF report, "on N farm-hours" in the station report), and the per-arm interval-table parser (main: no per-arm intervals; ICON-DREAM-EU: 5 columns; station: 6 columns with backticked arm names). The interval check must fail loudly rather than skip when a table has a shape it does not know. The station row set has a second losses file, `losses_post_review.parquet`, which the block reads as well. Each generalisation gets a test that fails on the solar parser today. If the generalisation proves larger than a separate wind script, the implementer stops and says so before choosing.
- `past_wind_leaderboard_charts.py` (new): Figure 1 and Figure 2 as four-block stacked charts built with the solar branch's `RowSetBlock`, `stacked_leaderboard` and `stacked_contrasts`. The reference row is ERA5 in every block; in the station block it is `era5_10m_wind`. The station block shows the arms and contrasts its report holds (the station's 10 m wind, ERA5's 10 m and 100 m wind, and UKV with and without the station; its planned contrast S2 is `ukv_station_wind` minus `ukv_padded_wind`, not a contrast against ERA5), so its contrast panel states which arm each contrast is against; the implementer reads the report to fix the arm list and the outline review confirms it. The station block's label or caption carries its share of scored rows in months that occur in one year only (42.2% in its report). Each block carries its own uncovered-month share in its label or caption, from the #906 measurement (main 16.2%, ICON-DREAM-EU 25.1%; the other two blocks' shares are read from the fold report, not assumed).
- Existing wind chart scripts: merged figures redrawn from the map; SVG titles carry the new numbers; time-series charts keep days 1-7, month and year in text only, % of capacity and `aria=False`.
- `check_page_numbers.py` guard (fact 4): fix the two unsigned negative intervals in the station section, found by reading that section and running the guard on it, and add the one line it needs to the wind page's Reproducing block. The fix is prose, so it lands in PR B; PR A only makes the guard's per-section calls for wind work through `studies.page_numbers`.
- `studies/beam_diffuse_split/README.md`: figure references.

PR B (prose, stacked on PR A, after Methods merges):

- Rewrite `weather-products-for-past-wind.md` (moved to `docs/studies/past-weather/wind.md` by the solar PR B, not by this plan) into the 12 parts: Title; Summary (two paragraphs, headline figures, scoped take-home bullets); AI disclaimer; Key findings; Introduction with the products table; Data and methods (study-specific: the wind hub heights and processing, the four row sets, the ECMWF and station arms, linking the shared Methods page for row-set definitions, planned contrasts, capacity normalisation, bootstrap intervals, the second setting and shared gotchas); Results (one subsection per finding, "The XGBoost models work" first); Discussion "What to use"; Limitations; Scope; Data and code availability; Reproducing the figures. Sentences move verbatim; only connective text is rewritten; duplicates of Methods content are deleted.
- New paragraph in Data and methods on the products' processing (fact 2), worded only after the coordinator confirms: Open-Meteo's ERA5 100 m and 10 m wind matches native CDS ERA5 to within Open-Meteo's 0.1 km/h rounding (RMS 0.06 km/h, ratio of means 1.0000, nearest cell, no interpolation, no 0.98 factor); W2 and W3 share one ERA5 cell, so the evidence is two independent series; Open-Meteo's ICON-EU matches Dynamical.org's DWD ICON-EU for radiation, temperature and 10 m wind to within rounding. Unchecked: the ICON-family 100 m wind (a documented factor of 0.98 on the native 120 m speed; no native ICON 100 m wind is on disk), ICON-D2 and UKV wind, and IFS. Wind is instantaneous in Open-Meteo's products and radiation is hour-ending. The device (CPU) is stated.
- Each block's limitation sentence carries its own uncovered-month share.
- Inbound links and anchors into the wind page are re-pointed (survey page, siblings, roadmap, script docstrings), as the solar PR B does for its own page.

Follow-on plans (not in these PRs): CERRA wind at 100 m (on disk; the 75, 50 and 150 m levels are still downloading, about a day), and NORA3 (on disk in `data/studies/weather/NORA3/NORA3_wind.parquet`, 2015-01-01 to 2026-08-31). ICON-DREAM-EU is already one of PR A's four blocks. Each new arm gets its own row set with its own refitted ERA5-style reference rows fitted on the fixed folds (`rotate_folds` over `with_eras`, `raise_on_uncovered_months`), a hub-height column and a domain-coverage row, and is fitted on one device, CPU unless the coordinator says otherwise.

## Design-philosophy check

R&D and documentation only. Nothing runs in production, so nothing degrades, and the studies fail fast: the leaderboard script stops on any mismatch. No asset check is added. No hypothesis is cited. The Discussion states which findings support the choices in the live-service roadmap and does not commit the project to work.

## Tests

Each test states what bug it catches.

- Leaderboard rows: only the new parsing gets new tests (the "farm-hours" heading, the "Hub height shown" column), each failing on the solar parser today. Cross-join, mismatch and sign guards are inherited from the solar `score_row_set`, `assert_matches_printed` and `page_numbers` tests and are not copied.
- `FIGURE_NUMBERS`: `tests/test_figure_numbers.py` gains wind cases (values 1 to 15 for wind with no gap or duplicate; every wind-page SVG has a key; a wind `SUPERSEDED_SVGS` list for the SVGs the page keeps linking until PR B, as solar has). The check that each SVG's title number equals the map is new, since the existing tests have no such check; it reads the title only, because chart text cross-references other figures (`wind_leaderboard.svg` says "Figure 2"). The solar-specific order and stem tests are not parametrised.
- The two unsigned negative intervals are found by running the guard on the station section as a gate, since `page_numbers` already enforces interval signs.
- The wind losses have no device column, so no device assertion is written; the page states the fits ran on CPU.
- Gates as scripts (scratch, uncommitted unless noted):
  - Number conservation: every decimal, every integer of 10 or more, and every `X [a, b]` triple in the new page appears in the old page (fetched from `origin/main` at run time) or the new report; every dropped number is a duplicate still present elsewhere and signed off by a reviewer. Blind spots stated in the PR body: it cannot see a number attached to the wrong row set, product or setting, or a deleted qualifier.
  - `check_page_numbers.py` section by section, one call per report, reusing the solar session's fix for its false failures.
  - Planned/exploratory status preserved per contrast (table in the PR body).
  - Every absolute error (a decimal followed by `%`) sits in a paragraph naming its row set.
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

1. Number conservation and `check_page_numbers.py` overlap where a report prints a number. Both stay, because the coordinator requires the conservation gate and the guard cannot see a number that no report prints.
2. The station block is a set of contrasts against ERA5's 10 m wind rather than a product ranking. Recommendation: draw it as the fourth block of Figure 1 with 10 m in the label and hub height in the column, so the reader does not compare it with 100 m rows.
3. Figure 1 invites cross-block comparison of absolute errors from different row sets. Recommendation: block labels carry dates and row counts, and every block repeats ERA5.
4. Base branch: stacking on `defrag-past-solar` means a rebase when it merges. Recommendation: accept.
5. Planned or exploratory: the main row set's ICON arms are the 80 m arms chosen after the first run (the plan specified 100 m), ICON-DREAM-EU's planned contrasts were written after the five products were scored, the main report's "Deciding" table holds one exploratory contrast (`icon_d2` minus `ukv`), and per-farm rows sit under planned headings. Each contrast's status is fixed from the report text and listed in a table in the PR body before the merged charts are drawn.
6. Hard rule and a verbatim move: the page states each generator's distance to its ICON-DREAM-EU grid cell (1.9, 3.4 and 1.3 km) in Data and methods. The rule covers MIDAS stations and per-generator distances; whether it reaches generator-to-grid-cell distances is a question for the coordinator, and until answered PR B states the range only.
7. If the D1 refit moves wind numbers, the leaderboard is computed on the new losses and the conservation gate compares against the post-refit old page.

## Plan review 1 (simplicity): triage

Adopted: import the generic leaderboard code from the solar script instead of cloning it (any parameterisation goes to the solar session); drop the device assertion (no wind losses file has a device column); drop the cross-join, mismatch and signed-interval tests already covered on the solar branch; parametrise the existing figure-number tests over both maps; correct the figure map (old 16 was listed twice; old 13's panels belong in new Figure 2) and merge old 7, 20 and 23 into one per-farm figure, giving 15; name `era5_10m_wind` as the station block's reference; fold the SVG-title gate into the figure-number test.

Rejected: merging the leaderboard and chart scripts (the page-number guard needs a persisted `report.md`; solar uses the same split, and the reviewer agreed); dropping number conservation (the coordinator requires it, and it catches numbers no report prints); dropping the PR B processing paragraph (the coordinator asked for it, wording pending their confirmation).

## Plan review 2 (correctness): triage

All findings were checked against the saved reports and adopted above: the solar script hard-codes a reference arm, a planned-section prefix, a heading regex and a 4-column interval parser (verified against the wind reports: the main report's planned section is "Deciding contrasts, named before the run" and it carries the "Hub height shown" column too); the station block is not 10 m against 10 m (S2 is `ukv_station_wind` minus `ukv_padded_wind`); there are four chart scripts, not eight; the SVG-title test is new rather than existing, and needs a `SUPERSEDED_SVGS` list; old Figure 21 is a contrast figure, so only its absolute errors feed Figure 1; the planned-status risks are Risk 5; the missed docstring and chart-text callers are named; the generator-to-grid-cell distance question is Risk 6; and ICON-DREAM-EU is no longer a follow-on slot. The reviewer could not run `git grep` for inbound links, so that sweep is done at PR B time and is a gate. Nothing was rejected. The reviewer also created and deleted an empty `.venv` in the worktree, and `git status` was clean afterwards.
