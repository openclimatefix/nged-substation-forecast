# Restructure the past-solar study page into one coherent whole (#910)

## The problem

`docs/studies/weather-products-for-past-solar.md` (1,588 lines, 25 figures) grew one section per new idea: CAMS and ERA5, four extra Open-Meteo models, ECMWF ENS, and MIDAS weather stations. Each section carries its own leaderboard, planned-contrast definitions, caveats and reproduce block. A reader cannot find the one answer to "which product should Flexpectation use for past solar", and the three sibling pages (past wind, blending, ENS horizons) will repeat the pattern. Nothing says where methods shared by the solar, wind and blending studies live.

## The planned solution

PR A (code, charts, drift fixes) puts every headline result into one leaderboard and one contrast chart of four row-set blocks, each block repeating CAMS and ERA5 as lighter reference rows. PR B (prose, nav, links) rewrites the page around those charts by moving sentences verbatim and rewriting only connective text, adds a "take-home for Flexpectation" section near the top, and creates a new Studies > Past weather section whose shared Methods page holds the methods the solar, wind and blending pages share. The approved parent plans are `plan-defrag-past-solar.md` and `plan-study-sequencing.md` (coordinator's worktree); this file is their executable form and adds the docs structure decided since.

## Verdict, size and departures

**Verdict:** worth doing, as approved by the maintainer. **Size: complex.** The five triggers:

1. What gets stored: yes. A new write-once folder `data/studies/past_weather_v2/solar_leaderboard/` (report, `intervals.parquet`), and about 17 SVGs redrawn. No Delta table, Patito model or asset.
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

Assets stay in `docs/studies/assets/`; moved pages link them as `../assets/...`. The Methods page holds, moved verbatim from the solar page: the four row sets, the 14 planned contrasts and the planned/exploratory rule, capacity normalisation, bootstrap intervals, the second hyperparameter setting, and the gotchas that are not solar-specific. The Overview holds the take-home for all three past-weather studies in one paragraph each, linking the pages. The wind and blending pages keep their duplicated methods until their own restructures, which then only delete duplicates and add links.

Moving a page changes its slug, so the work list in PR B is: `git mv` the five pages; update `mkdocs.yml`; fix every inbound link and `#anchor` (inventory below); fix `studies/**` script docstrings and reproduce text that name pages or figure numbers; fix `.claude/skills` and `CLAUDE.md` mentions; and if no redirect plugin is installed, decide with the maintainer whether to add `mkdocs-redirects` (open question 3). The docs-link check and `mkdocs build --strict` prove no link is left dangling.

## What changes, file by file

PR A (code; reads saved `losses.parquet`, no refit):

- Drift fixes first, as their own commit: product table 12 -> 14, the three uses of "eight", generator-hours vs site-hours, "two" vs four exploratory comparisons, the ICON-D2 edge vs `weather_product_domains.py`, and the IFS-HRES vs ERA5 result `0.58 [0.36, 0.81]` stated only in "What to use". Prose-only, so it can land in PR A ahead of the charts.
- `packages/studies/src/studies/leaderboard.py` (new): a list of row-set blocks (`RowSetBlock` with label, dates, site-hours, arm rows), the shared leaderboard and contrast-chart builders, and one `FIGURE_NUMBERS` map that every chart script imports. Tests in `packages/studies/tests/`.
- `studies/beam_diffuse_split/past_solar_leaderboard.py` (new): reads the saved losses of the four row sets, recomputes per-arm intervals on main and extra rows, CAMS minus ERA5 on ENS and station rows, CAMS+station minus ERA5, and UKV-two-snapshots minus ERA5; writes `past_weather_v2/solar_leaderboard/report.md`; stops unless recomputed numbers match every number already printed at printed precision.
- `weather_product_charts.py`, `ens_past_solar_charts.py`, `station_past_solar_charts.py` and the other chart scripts: import `FIGURE_NUMBERS`; the four leaderboards become Figure 1 and the four contrast charts Figure 2. Redraw is held until the coordinator releases it.
- Issue #904: teach `check_station_page_numbers.py` (or a shared parser in `packages/studies` with tests) the station report's signed three-decimal rows, so it covers the station numbers.

PR B (prose, nav, links) stacked on PR A:

- Rewrite `weather-products-for-past-solar.md` -> `past-weather/solar.md` to the approved outline (H1, one-paragraph answer, take-home for Flexpectation, key findings, introduction, products and how each was scored with a "Four row sets" table and one table of 14 planned contrasts, results by question, why products differ, robustness, implied capacity, what to use, limitations, one reproducing block), moving sentences verbatim.
- New `past-weather/index.md` and `past-weather/methods.md`; move the other four pages; update `mkdocs.yml`, `docs/studies/index.md`, `docs/background/weather-products-survey.md`, `docs/roadmap/data-sources.md`, `docs/roadmap/live-service.md`, the sibling pages and script docstrings.

Figure map (25 -> 17): new 1 = old 1, 15, 17, 21; new 2 = old 2, 16, 18, 22; 3 stays; 4 (old 20 dropped); 5-9 stay; 10 = old 10 + 11; 11 = old 19; 12 adds IFS-HRES; 13 stays; 14 = old 23 + ENS per-generator; 15 = old 24; 16 = old 25; 17 = old 14 (implied capacity). Figure 1 design: dot plus 95% line, x axis "Mean absolute error (% of capacity; smaller is better)", one colour per product family, hollow reference rows, block labels carry dates and site-hours, palette validated with the `dataviz` skill. Slots for CERRA solar and WeatherNext 3 are one row in the row-set registry, one row in the "Four row sets" table (which becomes five, six), and one more Figure 1 block each.

## Design-philosophy check

R&D and documentation only. Nothing runs in production, so nothing degrades; the studies fail fast (the leaderboard script stops on any mismatch with printed numbers). No asset check is added. Hypotheses: none cited; the take-home section states which findings support the choices in the live-service roadmap and links, and does not commit the project to work (CLAUDE.md prose rule).

## Tests

- `studies.leaderboard`: a block with a reference row renders once; a contrast is computed on the arm's own rows (fails if computed on the pooled rows); a `FIGURE_NUMBERS` lookup of an unknown chart raises. Each fails on `main` because the module does not exist.
- `past_solar_leaderboard.py`: a test that a deliberately perturbed printed number makes the mismatch guard stop (fails on `main`: no guard).
- #904: the guard parses `+0.995 [+0.585, +1.452]` against page text `0.99 [0.59, 1.45]` (fails on `main`: reports missing).
- Gates as scripts, not pytest: number-conservation (every decimal and `X [a, b]` triple in the new page appears in the old page or the new report; each dropped number a still-present duplicate, signed off), `check_page_numbers.py` section by section, planned/exploratory status preserved per contrast, every absolute error inside a paragraph that names its row set, SVG title number equals caption number.

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

1. Redirects: decided (via the coordinator): add `mkdocs-redirects`, pinned, in its own small commit, and check `uv run mkdocs build --strict` handles the redirect map. Redirects keep old page URLs working for GitHub issues and PR bodies. Anchors inside pages cannot be redirected, so every inbound anchor link is still updated.
2. Beam/diffuse split page stays at the top of Studies rather than under Past weather. Recommendation: leave it, since it is a forecast-skill study rather than a product comparison.
3. Wrong row set on a number (5.20 is the extra rows' CAMS, not main's 5.09); Figure 1 inviting cross-block comparison (ENS 8.267 vs ICON-EU 8.386); a qualifier deleted with a duplicate; planned status lost when charts merge; the take-home overclaiming; recomputed intervals drifting from printed ones. Each has a gate above.
4. If the fold fix moves past-solar numbers, the refit lands first and the leaderboard is computed on the new losses; the conservation gate then compares against the post-refit old page, and the planned-contrast diff table goes in the PR body.
5. The maintainer is considering running the second XGBoost setting only on planned or deciding contrasts and near-threshold results. The Methods page's second-setting paragraph is moved verbatim and stays unchanged until the coordinator confirms the rule; charts are not changed for it.
