# Restructure the past-solar study page into one coherent whole (#910)

## The problem

`docs/studies/weather-products-for-past-solar.md` (1,588 lines, 25 figures) grew one section per new idea: CAMS and ERA5, four extra Open-Meteo models, ECMWF ENS, and MIDAS weather stations. Each section carries its own leaderboard, planned-contrast definitions, caveats and reproduce block. A reader cannot find the one answer to "which product should Flexpectation use for past solar", and the three sibling pages (past wind, blending, ENS horizons) will repeat the pattern. Nothing says where methods shared by the solar, wind and blending studies live.

## The planned solution

PR A (code, charts, drift fixes) puts every headline result into one leaderboard and one contrast chart of four row-set blocks, each block repeating CAMS and ERA5 as lighter reference rows. PR B (prose, nav, links) rewrites the page around those charts by moving sentences verbatim and rewriting only connective text, adds a "take-home for Flexpectation" section near the top, and creates a new Studies > Past weather section whose shared Methods page holds the methods the solar, wind and blending pages share. The approved parent plans are `plan-defrag-past-solar.md` and `plan-study-sequencing.md` (coordinator's worktree); this file is their executable form and adds the docs structure decided since.

## Verdict, size and departures

**Verdict:** worth doing, as approved by the maintainer. **Size: complex.** The five triggers:

1. What gets stored: yes. A new write-once folder `data/studies/past_weather_v2/solar_leaderboard/` (report, `intervals.parquet`; no past-solar row set has one today, the intervals in the page come from `report.md` files), and about 17 SVGs redrawn. No Delta table, Patito model or asset.
2. Production serving path: no. Nothing under `defs/`, `ml_core` or `xgboost_forecaster`.
3. Degradation rule: no.
4. More than one defensible design: yes. Whether a shared leaderboard helper is a list of row-set blocks or one frame, whether pages move with or without redirects, and where the Methods page sits.
5. Callers not nameable without searching: yes. Every figure number is baked into SVG titles and chart scripts, and page/anchor links come from the survey, three sibling pages, roadmap pages and script docstrings (see the inventory below).

Reviews bought: both plan reviews, then all diff reviews plus the extras the coordinator specified (Opus outline review, fact check, scope-of-claims review, two Opus science reviews on PR A's leaderboard, network-planner and ML-engineer persona reviews, one-rule prose sweeps, a Sonnet read of the rendered HTML).

**Departures from the parent plans:**

- The parent plan's PR B moves no pages. This plan also moves pages (nav restructure); the parent plan's "shared methods page" decision is made now, and the page is created in PR B.
- PR A's charts are not redrawn and PR A is not merged until the coordinator says the fold-fix measurement (#868, #892, PR #906) has finished and the past-solar numbers are final. Until then only code that reads saved losses is written.
- CERRA solar and WeatherNext 3 are slots only; CERRA's numbers arrive in a follow-on plan after this restructure merges.

## Docs structure (decided, PR B)

New nav, under Studies:

```text
Studies
  Overview                          studies/index.md            (unchanged URL)
  Does a weather product's beam/diffuse split help a PV forecast?   (unchanged)
  Past weather
    Overview                        studies/past-weather/index.md
    Methods shared by the past-weather studies   studies/past-weather/methods.md
    Solar                           studies/past-weather/solar.md    (was weather-products-for-past-solar.md)
    Wind                            studies/past-weather/wind.md     (was weather-products-for-past-wind.md)
    Blending                        studies/past-weather/blending.md (was blending-weather-products.md)
  Forecasts
    ECMWF ENS at each horizon       studies/forecasts/ens-horizons.md (was ens-forecast-horizons.md)
    NWP forecasts at matched leads  studies/forecasts/matched-lead.md (was nwp-forecasts-at-matched-leads.md)
```

Assets stay in `docs/studies/assets/`; moved pages link them as `../../assets/...` (one level deeper than before). The Methods page holds, moved verbatim from the solar page: the row sets (four headline row sets plus the record row set of 115,594 generator-hours from January 2021, which feeds Figures 7 and 8 and is defined in the "Row sets" table but stays out of the headline leaderboard), the 14 planned contrasts and the planned/exploratory rule, capacity normalisation, bootstrap intervals, the second hyperparameter setting, and the gotchas that are not solar-specific. The Overview holds the take-home for all three past-weather studies in one paragraph each, linking the pages. The wind and blending pages keep their duplicated methods until their own restructures, which then only delete duplicates and add links.

Moving a page changes its slug, so the work list in PR B is: `git mv` the five pages; update `mkdocs.yml`; fix every inbound link and `#anchor` (inventory below); fix `studies/**` script docstrings and reproduce text that name pages or figure numbers; fix `.claude/skills` and `CLAUDE.md` mentions; and add a `redirect_maps` entry per moved page. The docs-link check and `mkdocs build --strict` prove no link is left dangling.

## What changes, file by file

PR A (code; reads saved `losses.parquet`, no refit; the drift fixes below go out earlier as their own PR):

- Drift fixes first, as their own prose-only PR (PR 0), which can land before the fold-fix numbers are final. Contents: product table 12 -> 14, the uses of "eight" that count products rather than models (`:16`, `:253`; about 20 uses in all, each checked), generator-hours vs site-hours, "two" vs four exploratory comparisons, the ICON-D2 edge vs `weather_product_domains.py`, and the IFS-HRES vs ERA5 result `0.58 [0.36, 0.81]` stated only in "What to use". Prose-only, so it can land in PR A ahead of the charts.
- `packages/studies/src/studies/charts.py` (extended, no new module): a `RowSetBlock` (label, dates, site-hours, arm rows) and a function that stacks blocks into one `leaderboard_panel` figure with a shared x range and hollow reference rows, and the same for the contrast chart, built on the existing `report_errors`, `bootstrap_absolute`, `leaderboard_panel` and `figure`. Each chart script's private `_leaderboard` is split into a function that returns rows, so the existing bootstrap-and-mismatch guards (`weather_product_charts.py`, `ens_past_solar_charts.py`) are reused rather than rewritten. The rows function takes `setting` explicitly, filters to it before any bootstrap, and asserts `n_rows` equals the block's site-hours (the saved losses reuse arm names across both hyperparameter settings, and the bootstrap silently cross-joins them). One `FIGURE_NUMBERS` map, in `studies/beam_diffuse_split/` beside the chart scripts so that `packages/studies` stays page-agnostic, replaces the scattered solar-page `FIGURE_*` constants and the figure numbers inside SVG text; the wind scripts' constants are untouched. Tests in `packages/studies/tests/`.
- `studies/beam_diffuse_split/past_solar_leaderboard.py` (new): reads the saved `pooled`-setting losses of the four headline row sets, recomputes per-arm intervals on main and extra rows, CAMS minus ERA5 on ENS and station rows, CAMS+station minus ERA5, and `ukv_pair_global` (two snapshots as separate inputs) minus ERA5 (the `ukv_trap_global` contrast, 0.90 [0.72, 1.09], is already printed); writes `past_weather_v2/solar_leaderboard/report.md`; stops unless recomputed numbers match every number already printed at printed precision.
- `weather_product_charts.py`, `ens_past_solar_charts.py`, `station_past_solar_charts.py` and the other chart scripts: import `FIGURE_NUMBERS`; the four leaderboards become Figure 1 and the four contrast charts Figure 2. Redraw is held until the coordinator releases it.
- Issue #904: it concerns `check_page_numbers.py` and the past-wind station report (`station_wind_arms.py`), whose rows are signed and three-decimal against unsigned page figures, and whose `intervals.parquet` sits beside it. Teach `check_page_numbers.py` that format, moving the shared parsing into `packages/studies` with tests. The throwaway `check_station_page_numbers.py` (which reads the solar page by path and only checks lines added since `origin/main`, so it cannot survive the restructure) is deleted in PR B once PR A has shown that `check_page_numbers.py` covers the solar station report; the plan's earlier version wrongly aimed #904 at that throwaway.

PR B (prose, nav, links) stacked on PR A:

- Delete `check_station_page_numbers.py` only if the counts it checks (site-hours such as 60,033) are covered by the conservation gate's integer check and by `check_page_numbers.py`; the latter checks decimals only.
- Rewrite `weather-products-for-past-solar.md` -> `past-weather/solar.md` to the approved outline (H1, one-paragraph answer, take-home for Flexpectation, key findings, introduction, products and how each was scored with a "Four row sets" table and one table of 14 planned contrasts, results by question, why products differ, robustness, implied capacity, what to use, limitations, one reproducing block), moving sentences verbatim.
- New `past-weather/index.md` and `past-weather/methods.md`; move the other four pages; update `mkdocs.yml`, `docs/studies/index.md`, `docs/background/weather-products-survey.md`, `docs/roadmap/data-sources.md`, `docs/roadmap/live-service.md`, `docs/architecture/nwp-variable-conventions.md:121`, `docs/roadmap/xgboost-improvements.md` (lines 179, 263, 1384), the sibling pages, the script docstrings and READMEs (including the repo-relative and backticked page paths in `studies/nwp_forecast_comparison/README.md:7`, `nwp_forecast_comparison.py:7`, `ens_past_solar.py:181,211,240`, `blend_satellites_charts.py:13`, `check_new_products.py:233`, which no link checker reads), and `.claude/skills/study/SKILL.md` (its "every deciding contrast" rule, for the second-setting decision). Moved pages' outbound relative links (`beam-diffuse-split.md`, `../roadmap/...`) gain `../`, and Figure 3's image link becomes `../../roadmap/assets/weather_product_domains.svg`.

Figure map (25 -> 17): new 1 = old 1, 15, 17, 21; new 2 = old 2, 16, 18, 22; 3 stays; 4 (old 20 dropped); 5-9 stay; 10 = old 10 + 11; 11 = old 19; 12 adds IFS-HRES; 13 stays; 14 = old 23 + ENS per-generator; 15 = old 24; 16 = old 25; 17 = old 14 (implied capacity). Figure 12's IFS-HRES arm exists only on the extra rows, so its label names that row set. Figure 1 design: dot plus 95% line, x axis "Mean absolute error (% of capacity; smaller is better)", one colour per product family, hollow reference rows, block labels carry dates and site-hours, palette validated with the `dataviz` skill. Slots for CERRA solar and WeatherNext 3 are one row in the row-set registry, one row in the "Four row sets" table (which becomes five, six), and one more Figure 1 block each.

## Design-philosophy check

R&D and documentation only. Nothing runs in production, so nothing degrades; the studies fail fast (the leaderboard script stops on any mismatch with printed numbers). No asset check is added. Hypotheses: none cited; the take-home section states which findings support the choices in the live-service roadmap and links, and does not commit the project to work (CLAUDE.md prose rule).

## Tests

Each test states what bug it catches.

- `studies.charts` block rows function: fed two settings under one arm name, it raises (catches the silent 2-by-2 cross-join that moved CAMS minus ERA5 from -4.089 to -4.066 on the station rows). Each block holds exactly one row per arm, and reference rows carry the hollow flag (catches pooled blocks and a lost reference marker). Every bootstrap `n_rows` equals the block's site-hours.
- `FIGURE_NUMBERS`: values are 1 to 17 with no gap or duplicate, and every solar-page SVG has a key.
- The mismatch guard lives as a function in `packages/studies` (pytest cannot import `studies/beam_diffuse_split` scripts), and a test perturbs one printed number and expects the guard to stop.
- #904: `check_page_numbers.py` takes the print precision per report rather than a constant `INTERVAL_PRINT_DECIMALS = 4`. The fixture supplies an `intervals` frame with 0.994766 [0.58524, 1.45206], the report row `+0.995 [+0.585, +1.452]` and the page text `0.99 [0.59, 1.45]`; it fails on `main` (reported missing) and would fail under a tolerance that loosens the guard, since 0.995 rounds half-up to 1.00.
- Gates as scripts (uncommitted unless noted):
  - Number conservation: every decimal, every integer of 10 or more, and every `X [a, b]` triple in the union of the new Solar, Methods and Overview pages appears in `git show origin/main:docs/studies/weather-products-for-past-solar.md` (baseline fetched at run time and re-run after every rebase) or in the new report; every number in the old page is in the new pages or on a signed-off dropped list. Blind spots, stated in the PR body: it cannot see a number attached to the wrong row set, product or setting, or a deleted qualifier.
  - `check_page_numbers.py` section by section: run once on `main`'s solar page first to learn its false-failure rate (it has only ever run on the wind page; reports print 3 decimals and the page quotes 2). Each section names its reports, one call per report, and the leaderboard report is checked with its own `intervals.parquet`.
  - Planned/exploratory: diff the Methods page's table of 14 planned contrasts against the ` (planned)` labels (`charts.NAMED_SUFFIX`) in the chart data; a table goes in the PR body.
  - Row-set naming: an absolute error is a decimal immediately followed by `%`; each must sit in a paragraph naming a row set. It cannot catch a wrong row set, so the fact-check reviewer covers that.
  - SVG titles: compare the full title text (joined across its `<text>` nodes), exempt Figure 3 (no number), and grep SVG text for `Figure \d+` against `FIGURE_NUMBERS`. Plus a repo-wide `git grep -nE 'weather-products-for-past-(solar|wind)|blending-weather-products|ens-forecast-horizons|nwp-forecasts-at-matched-leads'` that must return only `mkdocs.yml` redirect keys.

## Docs to update

Every page listed under PR B above, all written for the present. `docs/studies/index.md` gains the Past weather and Forecasts groupings. `studies/beam_diffuse_split/README.md` and the reproduce text change page names. Ship-time triage: no roadmap item completes; the plan file is deleted at ship time into the PR body.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict       # then read the rendered HTML of every moved page
uv run pre-commit run --all-files  # runs the docstring markdown hook, pydoclint and the docs-link check
```

Compute: no refit. The leaderboard script's first run needs the runner slot from the coordinator (worktrees share one `data/` folder), writes only to its own new folder, and runs only after the fold-fix session reports the past-solar numbers final.

## Hard rules restated

Labels A-F only for generators; no MIDAS station names, coordinates, farm mapping or per-generator distances (pooled ranges only); time-series charts show days 1-7 with month and year in text only, % of capacity, `aria=False` on data marks; scratch under `.claude/worktrees/scratch/`; nothing near an issue number that says close, fix or resolve.

## Risks and open questions

1. Redirects: decided (via the coordinator) to keep old page URLs working. `mkdocs-redirects` is already installed and pinned (`pyproject.toml`, `<1.2.3`) and configured with two `redirect_maps` entries in `mkdocs.yml`, so PR B adds one entry per moved page and re-points the existing `studies/weather-products-for-the-past.md` entry to the new Solar page; no dependency commit is needed. `uv run mkdocs build --strict` must accept the map. Redirects cannot cover anchors inside pages, so every inbound anchor link is still updated: the inventory found 12 distinct anchors into the solar page (7 links from the survey page), 5 into the wind and blending pages from siblings, and the rest into beam-diffuse-split and ens-forecast-horizons, of which `ens-forecast-horizons` and `nwp-forecasts-at-matched-leads` also move (PR B's file list includes their inbound links) and `beam-diffuse-split` does not.
2. Beam/diffuse split page stays at the top of Studies rather than under Past weather. Recommendation: leave it, since it is a forecast-skill study rather than a product comparison.
3. Wrong row set on a number (5.20 is the extra rows' CAMS, not main's 5.09); Figure 1 inviting cross-block comparison (ENS 8.267 vs ICON-EU 8.386); a qualifier deleted with a duplicate; planned status lost when charts merge; the take-home overclaiming; recomputed intervals drifting from printed ones. Each has a gate above.
4. If the fold fix moves past-solar numbers, the refit lands first and the leaderboard is computed on the new losses; the conservation gate then compares against the post-refit old page, and the planned-contrast diff table goes in the PR body.
5. Second XGBoost setting (maintainer decision, via the coordinator): it is shown only for planned and deciding contrasts and for any result near the 5% line, and dropped for other exploratory arms. Precedence: the near-the-line clause wins, so an exploratory result near the line shows its second setting where a saved sensitivity arm exists; where none is saved (the extra rows have none for CAMS, ERA5, SARAH-3, UKV, ICON global, ICON-DREAM-EU or ARPEGE; the main rows have none for `ukv_trap`, `ukv_pair` and the `_ctx` arms), the page says so and prints no second-setting number, since this plan refits nothing. It is one table or one marker, never a doubled chart, and the both-settings-must-agree verdict rule stays, reserved for planned and deciding contrasts (all of which have a saved second setting). "Near the line" is defined once on the Methods page as an interval bound within 20% of the interval's width from zero (roughly a two-sided p-value between 0.006 and 0.24 under a normal approximation); on today's numbers it flags one row against ERA5, UKV (Open-Meteo hourly) minus ERA5 on the main rows, -0.218 [-0.392, -0.058], which has saved sensitivity arms. The maintainer confirms the definition.

## Plan review 1 (simplicity): triage

Adopted:

- No new `studies.leaderboard` module: extend `studies.charts` and split each chart script's `_leaderboard` into a rows function (reuse of existing guards). Reason: the building blocks exist and the block registry has no second caller yet.
- #904 targets `check_page_numbers.py` and the wind station report, not `check_station_page_numbers.py` (verified against the issue body).
- Drift fixes as their own prose-only PR, ahead of PR A.

Rejected, with reasons:

- *Group the nav without moving page files.* Rejected: the maintainer asked for the new URLs, Methods and Overview pages under a Past weather section, and for every inbound link and anchor to be planned. The cost of the moves is real (five redirect entries, about 32 files of links) and is bounded by the docs-link check. Recorded as the fallback if the maintainer prefers grouping only.
- *Drop `past_solar_leaderboard.py`; CAMS minus ERA5 on the ENS and station rows are new results.* Rejected: the brief names the script, per-arm intervals and these contrasts as approved (Figure 2 is every arm against ERA5 on its own rows). The reviewer is right that they are the only new numbers, which is why the script stops on any mismatch with printed numbers and its report is a write-once folder with those numbers marked exploratory.
- *Drop the title-equals-caption, row-set-in-paragraph and planned-status gates.* Rejected: the gates are the maintainer's requirement. They stay as scratch scripts and a PR-body table, none committed except number conservation's inputs; if any turns out to be a one-line grep it is run as one.
- *Adding IFS-HRES to Figure 12 and ENS per-generator to Figure 14 is scope growth.* Rejected as part of the approved figure map, but each is dropped if it needs a new run rather than saved numbers.
- *Merge Overview and Methods into one page.* Rejected: the maintainer named both pages.

## Plan review 2 (correctness): triage

The reviewer read the saved losses (read-only) and found nine defects; all nine are real and applied above. (1) The two hyperparameter settings share arm names in every saved `losses.parquet`, so an unfiltered contrast silently cross-joins them; the rows function now takes `setting`, asserts `n_rows`, and has a test. (2) `ens-forecast-horizons` and the matched-lead page also move; the file list, `../` link fixes, Figure 3 asset path and the repo-wide `git grep` gate are added, and the inventory count is 11 distinct anchors carried by 20 links into the solar page. (3) There are five row sets; the record row set is defined in the row-set table. (4) #904 needs per-report print precision and an `intervals` frame in its fixture; retiring `check_station_page_numbers.py` is conditional on the counts being covered. (5)-(7) Gate definitions tightened (reports named per section, baseline fetched at run time, union of new pages, integers checked, full SVG title text, mechanical definition of an absolute error). (8) Tests restated, guard moved into `packages/studies`. (9) Second-setting precedence and the skill edit added. Also adopted from the smaller points: `FIGURE_NUMBERS` beside the chart scripts, `ukv_pair_global` named, IFS-HRES label naming its row set, PR 0 separate, and the row-set table regenerated from the losses because #892 may move the ENS block's start date. Nothing was rejected.
