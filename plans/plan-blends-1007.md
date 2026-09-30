# Plan: which weather forecast to add to ECMWF ENS first (#1007)

**The problem.** The matched-lead page scores each weather product alone against the ENS mean, and tests one blend (ENS, ICON-EU, and IFS 0.25°) at day 1 and day 2. The page cannot say which single product to add to ENS first, because no "ENS mean plus one product" blend exists for AIFS Single, WeatherNext 3 (WN3), or UKV at lead-matched days, and the ICON-EU blend exists only inside a three-product blend.

**The plan.** Fit "ENS mean plus one product" blends, as extra columns in one XGBoost model, for ICON-EU, AIFS Single, WN3, and UKV, at lead days 1, 2, 3, 5, and 7 where each product reaches them, for solar and wind. Each blend keeps the existing harness: the same rows as its ENS-alone reference, the same XGBoost settings, folds, and seeds, a shuffled-weather negative control, and both an optimistic and a conservative lead. Draw (blend minus ENS mean alone) as dot-and-interval rows, one panel per lead day, in the style of `dot_interval_vs_ens.py`, and rank the products per technology and lead only where the paired intervals support a ranking. A stage 2 tests whether adding two products beats the best single addition. The results go on a new page under Studies > Forecasts, and the matched-lead page links to it.

## Verdict, size and departures

**Verdict: worth doing as described.** Nothing in the issue is stale. No open PR or issue covers it: searches for "blend" and "AIFS" found #836 (past-weather blending, closed), #957 (CERRA wind blend, closed), #923 and #954 (AIFS on the matched-lead page, closed), #974 (WN3 and AIFS on the matched-lead page, closed), and #999 (short product history, open, a different question). The work extends #810's family of forecast studies, so the issue is attached under #896, the parent that holds #810 and #974, rather than under #810 (which has no sub-issues).

**Size: complex.** The five triggers:

1. **What gets stored: fires.** Several new write-once folders under `data/studies/`, one new published page, new figures. No Patito model, Delta table, or asset changes.
2. **Production serving path: does not fire.** Only `studies/`, `packages/studies/`, and `docs/studies/` change.
3. **Degradation rule: does not fire.** Study code is R&D and fails fast.
4. **More than one defensible design: fires.** The day grid, two-product blends, UKV's inclusion, WN3's comparability, and how the blends enter the headline figures each have several defensible answers (see "Risks and open questions").
5. **Callers not nameable without searching: fires partly.** `fit_aifs.py` (4,157 lines) and `nwp_forecast_comparison.py` (2,482 lines) hold the blend arms, the permutation guard, and the fit loop. Whether a helper can be imported or has to move into `packages/studies/` is settled by grepping their public names.

**Reviews bought:** the brief for this plan runs no plan review. The parent runs fresh Opus plan reviews (simplicity, then correctness). After the code, the `study` skill requires two Opus scientific-validity reviews, a diff review, and a `prose-review` of the page. The mutation pass runs only if `packages/studies/` changes.

**Departures from the issue body.** None. The plan adds one decision to it: the lead days (see below).

## What exists today

**The blend harness exists and is reusable.** `nwp_forecast_comparison.py` holds `BLEND_ARMS` (P4a and P4b: ENS day-1 mean plus ICON-EU and IFS 0.25° at day 1 and day 2), `add_blend_guard_columns` (shuffles each non-ENS product among hours sharing a generator, year-month, and UTC hour of day, through `studies.blending.climatology_permutation`), and two exploratory single-product wind blends (`WIND_SINGLE_PRODUCT_BLENDS`: ENS plus ICON-EU, and ENS plus IFS 0.25°, at day 2). `fit_aifs.py --blends` fits "ENS mean plus AIFS Single" and "ENS mean plus AIFS ENS mean" at days 1, 2, 7, and 14 on the `single` row set (16 months), each with a control and a mirror control, into `data/studies/nwp_forecast_comparison_aifs_blends`. `fit_aifs.py --wn3` fits WN3 alone at days 1, 2, 7, and 14 on the 7-month `wn3` row set, with no blend. `dot_interval_vs_ens.py` draws (product minus ENS mean) dots from saved per-row losses and already supports two marks per row (the same-rows reference, and the 21-month ENS mean on shared keys) and dashed intervals under 12 months.

**The published blend results are the comparison target.** Blend P4a (optimistic lead) lowered the wind error by 0.656 points and the solar error by 0.347. Blend P4b (conservative lead) lowered the wind error by 0.184 points and left the solar error unchanged at -0.033 [-0.106, +0.033]. At day 7, ENS plus AIFS Single lowered the solar error by 0.396 points [-0.578, -0.201], with no claimable wind difference, and at day 14 no forecast had skill to compare. All of these are on the matched-lead page's "A blend lowers the wind error..." and "At day 7 a blend of ENS and AIFS Single..." sections.

**What each product reaches** (from the page and `docs/background/weather-products-survey.md`). ICON-EU: Open-Meteo Previous Runs, days 1 to 4 (120 h run), day 0 from the extra-lead build. AIFS Single: whole 00 UTC runs, fitted at days 1, 2, 7, and 14. WN3: whole 00 UTC runs to 360 h, 7 months of rows. UKV live: Previous Runs fills only `previous_day1`, so day 1 only (and day 0). UKV-CEDA: 00, 06, 12, and 18 UTC runs to 54 h, overlapping about 9 of the 22 comparison months; the 03 and 15 UTC T120 store is still downloading. CEDA UKV differs statistically from live UKV, so no blend trains on one and infers on the other.

## Design

**A blend is extra columns in one XGBoost model, never an average of forecasts.** The blend of ENS mean and product P at lead day d gives the model the ENS mean's day-d columns plus P's day-d columns. Its reference is an XGBoost model given the ENS mean's day-d columns alone, fitted on the blend's own rows. Its control has the same column count, with P's columns shuffled within generator, year-month, and UTC hour of day (`climatology_permutation`). `colsample_bytree` stays at 1, and the seeds, folds, eras, settings, and GPU device stay as in the harness.

**Each blend is scored on the row set that its product's own history fixes, with the reference refitted on those rows.** Products are only compared with each other inside one row set, because two products that differ in rows differ in months of weather.

| Row set | Months | Products blended |
|---|---|---|
| `shared` | 21 | ICON-EU, UKV live (day 1 only), IFS 0.25° (for stage 2) |
| `single` | 16 | ICON-EU (refitted), AIFS Single, UKV live day 1 if its rows cover the set, IFS 0.25° |
| `wn3` | 7 | WN3, with ICON-EU and AIFS Single refitted on the same 7 months as comparators |

The ICON-EU blend is therefore fitted on all three row sets, so that it is the common yardstick, and is refitted in the new harness as the brief requires. Existing AIFS blend losses at days 1, 2, 7, and 14 are reused only if a check shows their rows, folds, seeds, settings, and device match the new harness's, and are otherwise refitted.

**Optimistic and conservative leads follow the published definition.** For Previous Runs products (ICON-EU, UKV live, IFS 0.25°), the optimistic blend reads the product at day d (P4a) and the conservative blend reads the product's day-(d+1) offset, which is one day older (P4b). The conservative blend decides any verdict. AIFS Single and WN3 are whole-run products read from the same 00 UTC run convention as ENS, so each has one reading. Whether that reading is also conservative depends on each product's publication time against the 09:00 UTC issue time, which is unmeasured for WN3 (see risks).

**The planned contrasts are fixed here, before any fit.** For each technology:

1. **C1:** ENS plus ICON-EU minus ENS alone, at day 1 and day 2, conservative lead, on `shared`. This reproduces the published direction in the new harness.
2. **C2:** ENS plus AIFS Single minus ENS alone, at day 1 and day 2, on `single`.
3. **C3:** ENS plus UKV live minus ENS alone, at day 1, on `shared`.
4. **C4:** each of C1 to C3's blends minus its own shuffled control (the guard: the gain comes from the added weather, not the column count).
5. **C5:** ENS plus AIFS Single minus ENS plus ICON-EU, at day 1 and day 2, on `single`. This is the paired contrast that a ranking of the two rests on.

Every other number is exploratory and labelled so: the days 3, 5, and 7 rows, the WN3 rows, the two-product blends, and the UKV-CEDA rows. The second hyperparameter setting (`SENSITIVITY_HYPER_PARAMETERS`) runs on every planned contrast and on any result near the 5% line.

**The ranking rule is fixed before the run.** Product A ranks above product B at a technology and lead only if the month-resampled 95% interval of (blend A minus blend B), on rows both share, lies wholly below zero at the primary and the sensitivity setting. Otherwise the page says the two are not separable. A product whose blend minus ENS-alone interval includes zero is not recommended for that technology and lead at all. WN3's rows are never ranked against the others, because they rest on 7 months.

**Stage 2 tests two-product blends, and only after stage 1's results are read.** The pairs are fixed by a rule, not by eye: the two products with the lowest single-add error per technology at day 1 and day 2 on `single`, plus the existing ENS, ICON-EU, and IFS 0.25° pair (P4) for continuity. The one contrast is (pair blend minus the best single add), and the page labels the stage exploratory.

**Days: 1, 2, 3, 5, and 7, where a product reaches them.** ICON-EU reaches days 1 to 3 at most on Previous Runs and ends at day 4, UKV live reaches day 1, and AIFS Single and WN3 reach all five. The first step prints which (product, day) columns exist on disk (`--dry-run`), and adds only the missing ones to `build_forecast_inputs.py`. Day 0 is excluded because a day-0 value could come from a run that a live service cannot read. Days 10 and 14 are excluded, on the page's own evidence that the ENS mean is no better than climatology from day 10 and that the day-14 AIFS blend had no skill to compare. The published day-14 AIFS blend rows stay on the matched-lead page.

**Cost.** Fits per product, day, domain, blend, and control take about 20 to 60 s on the A6000. Stage 1 is about 4 products, 5 days at most, 2 domains, and 2 fits (blend and control) per row set, plus the ENS-alone references: on the order of 150 to 250 fits, or 1 to 4 GPU hours. Stage 2 adds under 100. No data is fetched, so the £30 fetch cap is not touched.

## What changes, file by file

1. **New folder `studies/nwp_forecast_blends/`** (issue #861 gives each study its own folder):
   - `fit_blends.py`: builds the job list (blend, ENS-alone reference, control) per row set, domain, product, day, and lead; fits on the GPU with `--check` first; writes write-once folders `data/studies/nwp_forecast_blends_<stage>/` holding per-row losses, out-of-fold predictions, and a stamp; never overwrites an existing folder or any folder a merged page quotes; prints every arm's column list into `report.md`.
   - `blend_charts.py`: reads the saved losses and draws the dot-and-interval figures (blend minus ENS mean alone, one panel per lead day, hollow and dashed marks for under 12 months, two marks on WN3 rows) through `studies.charts`, and the priority table.
   - `README.md`: what each output file holds.
2. **`studies/nwp_forecast_comparison/build_forecast_inputs.py`:** add the ICON-EU, IFS 0.25°, and AIFS Single day 3 and day 5 columns onto the published `(site, time)` keys if the dry run shows them missing. Keys stay identical, so pairs join across folders.
3. **`packages/studies/src/studies/blending.py` and tests, only if a helper has to move.** The blend arm naming (`blend_<product>_day<N>_<role>`), the job-list builder, and the paired (blend A minus blend B) join with a same-keys check are candidates, because `fit_aifs.py` and `nwp_forecast_comparison.py` should not be imported as modules by a third script (the skill calls that pattern out). The mutation pass runs if these change.
4. **`docs/studies/forecasts/blends-with-ens.md`:** the new page in the study structure (title, summary with headline figure, disclaimer, key findings, introduction, data and methods, results, discussion, limitations, scope, data and code availability, reproducing). `mkdocs.yml` gains the nav entry. The matched-lead page's blend section and the survey page link to it.

## Tests

The study scripts are not unit-tested, so their check is their own `report.md`, from which every page number is taken. Any helper moved into `packages/studies/` gets tests that each fail on `main` today because the function is absent or lacks the behaviour:

- **Blend arm naming round-trip:** parsing `blend_aifs_single_day7_control` returns the product, day, and role. Fails on `main` only if the new naming changes; otherwise the test is omitted (a test that passes before and after tests nothing).
- **Same-keys check on a paired difference:** a difference of two losses tables whose `(site, time, seed)` keys differ raises `ValueError` naming the extra keys. Fails on `main`, where the function does not exist. A fixture has unequal generator capacities.
- **Job list completeness:** every blend in the job list has a control of the same column count, and building a blend whose control columns are missing raises. This mirrors `nwp_forecast_comparison.py`'s existing check and fails on `main` for the new helper.

## Design-philosophy check

R&D path: fails fast, as the inherent-stability page requires for the CV, training, and metrics code. Nothing runs in production, so no degradation rule, asset check, or Sentry tag applies. The study serves hypothesis T1.x only indirectly (which inputs a future model reads), and cites none.

**Anonymisation.** Outputs carry only the anonymised `site` labels (A to F and W1 to W3). No per-generator series is charted without its label removed, and no coordinate is printed.

## Docs to update

The new page, `mkdocs.yml`, a link from the matched-lead page's "A blend lowers the wind error..." section, `docs/background/weather-products-survey.md` where it recommends which products to add to ENS, and the roadmap's "Several NWP sources as features" section with the ranking. Each is written in the present tense.

## Verification commands

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest`, `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, the pydoclint and docs-link-checker steps that CI runs, and `nvidia-smi` plus a CPU-load check before each fit.

## Risks and open questions

1. **Which lead days?** Recommendation: 1, 2, 3, 5, and 7 where reached. Exclude day 0 (run may be unreadable live), and days 10 and 14 (no skill to compare; the published day-14 rows stay on the old page).
2. **Two-product blends?** Recommendation: yes, as a small exploratory stage 2 after stage 1, with pairs chosen by the written rule and one contrast against the best single add. Without the stage, the question "is ICON-EU plus IFS 0.25° better than the best single add?" stays unanswered.
3. **UKV now, or after the T120 store is complete?** Recommendation: include live UKV at day 1 now (it is already on the shared 21 months, and needs no CEDA data), and defer UKV-CEDA (days 2 to 5, and its 9 of 22 months) to a follow-up issue once the T120 store is complete. Mixing CEDA-trained weights with live-UKV inference is ruled out by the page's own warning. UKV's era boundary (2026-01-21) and the Open-Meteo backfill boundary (12 August 2024) need era-cut folds, as in the published run.
4. **Is WN3's blend row comparable?** Recommendation: no. It rests on 7 months (February to April and June to 10 September 2026), so every WN3 contrast is descriptive. Show it as its own row with a hollow mark and dashed interval, against both references (the same-rows ENS mean, and the 21-month ENS mean on shared keys), refit ICON-EU and AIFS Single on the same 7 months as comparators, and never rank WN3 against rows on other row sets. Its publication latency and lookahead are unestablished (the matched-lead page's "What is known about WN3's lookahead"), so its optimistic and conservative leads are not separated.
5. **CRPS and percentiles?** No. They belong in the main ML experiments. This study scores the mean's point error only.
6. **How do blends enter the headline figures?** Recommendation: leave Figures 1 and 2 on the matched-lead page (single products) unchanged, and give the new page its own headline figure, a dot-and-interval chart of (blend minus ENS mean alone) per product, with the priority table under it. Adding blend rows to the old figures would mix row sets that differ in months. The page was just simplified in #1006, and stays stable.
7. **Reuse or refit the existing AIFS blend losses?** Recommendation: reuse only after a check that rows, folds, seeds, settings, and device match; otherwise refit. A mismatch here would make the ICON-EU and AIFS rows incomparable without any visible sign.
8. **Code placement.** Whether the blend helpers move into `packages/studies/` (and so trigger the mutation pass) is settled by the first grep of `fit_aifs.py`'s and `nwp_forecast_comparison.py`'s public names. Recommendation: move only what the new script imports from a script, and nothing else.
