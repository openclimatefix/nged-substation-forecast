# Plan: which weather forecast to add to ECMWF ENS first (#1007)

**The problem.** The matched-lead page scores each weather product alone against the ENS mean, and tests one blend (ENS, ICON-EU, and IFS 0.25°) at day 1 and day 2. The page cannot say which single product to add to ENS first, because "ENS mean plus one product" blends exist only for AIFS Single (already fitted, not yet charted against the other products), and do not exist for ICON-EU at the GPU harness's rows, UKV, or WeatherNext 3 (WN3).

**The plan.** Extend `fit_aifs.py`'s blend arms from AIFS to ICON-EU and UKV, fit only the missing arms at days 1, 2, and 7 on the `single` row set into one new write-once folder through a thin driver, reuse the saved AIFS Single blend losses, and fit an ENS plus WN3 blend and its control inside the existing `run_wn3` (descriptive only). Draw (blend minus ENS mean alone) as dot-and-interval rows by adding blend rows and one comparison to `dot_interval_vs_ens.py`. The results go on a new short page under Studies > Forecasts, and the matched-lead page links to it.

## Verdict, size and departures

**Verdict: worth doing, at a smaller scope than the first draft.** Nothing in the issue is stale. No open PR or issue covers it: searches for "blend" and "AIFS" found #836 (past-weather blending, closed), #957 (CERRA wind blend, closed), #923, #954, and #974 (AIFS and WN3 on the matched-lead page, closed), and #999 (short product history, open, a different question). The issue is attached under #896, the parent of #810 and #974.

**Size: complex.** The five triggers:

1. **What gets stored: fires.** One new write-once folder under `data/studies/`, a published page, new figures. No Patito model, Delta table, or asset changes.
2. **Production serving path: does not fire.** Only `studies/` and `docs/studies/` change.
3. **Degradation rule: does not fire.** Study code is R&D and fails fast.
4. **More than one defensible design: fires.** The decisions under "Risks and open questions".
5. **Callers not nameable without searching: does not fire.** The change edits `fit_aifs.py`'s blend arm table and `dot_interval_vs_ens.py`'s product table, each read by its own driver and tests. The implementer greps `BLEND_AIFS_PREFIXES`, `blend_arms`, `blend_inputs`, and `PRODUCTS` before editing.

**Reviews bought:** the simplicity review of this plan has run and its findings are triaged below. The parent runs a second, correctness review. After the code, the `study` skill requires two Opus scientific-validity reviews, a diff review, and a `prose-review` of the page. No mutation pass is planned, because `packages/studies/` does not change (if an implementer finds it must, the pass runs).

**Departures from the issue body.** The issue asks for days 0, 3, 4, 5, and 10 "as available". This plan fits days 1, 2, 7, and 14 only (day 14 from saved losses), and moves days 3 and 5 to a follow-up issue. It drops day 0 (a day-0 value can come from a run a live service cannot read) and day 10 (the page shows the ENS mean no better than climatology from day 10). It moves two-product blends and UKV-CEDA to follow-up issues. The ENS plus WN3 blend is descriptive and is not refitted beside ICON-EU or AIFS Single on the 7 WN3 months.

## What exists today, and what is reused

**AIFS Single blends are fitted already, by the same code and device.** `data/studies/nwp_forecast_comparison_aifs_blends/{solar,wind}_single_day{1,2,7,14}_losses.parquet` hold `blend_aifs_single_dayN` with its `_control` and `_mirror` arms, `ens_mean_dayN`, and `aifs_single_dayN` on the `single` row set (16 months; 85,710 solar rows per arm at day 1), at the primary setting and, for the blend and its control, the sensitivity setting. Checked: every arm's `device` column reads `cuda`, and the stamp (`*_losses.json`) records the same two hyperparameter settings, the A6000, and XGBoost 3.4.1. The new ICON-EU and UKV arms are fitted by the same `fit_aifs.py` fit loop on the same `single` row set, so rows, folds, seeds, and settings match by construction, and the reference `ens_mean_dayN` is the one already in those files. The reuse therefore needs no separate check beyond the driver's own stamp test (`fit_aifs` refuses losses from another build or device).

**The published wind ICON-EU blend cannot be reused as a comparison.** `blend_wind_icon_eu_day2` and `blend_wind_ifs025_day2` (with controls) sit in `data/studies/nwp_forecast_comparison/wind_losses.parquet`, which has no `device` column (a CPU fit) and scores the 21-month `shared` rows, not `single`. They serve as a cross-check of the new GPU ICON-EU blend's wind day-2 direction, never as an arm in a paired contrast.

**The day-14 AIFS blend losses and the day-7 IFS 0.25° reference are also reused** as they are, for day 14 and as a stage-free reference at day 7. `nwp_forecast_comparison_p4_seeds` holds the GPU refit of the three-product P4 blends and controls on the published rows.

**The harness already takes a product as a parameter.** `BLEND_AIFS_PREFIXES` maps a blend's product name to its weather-column prefix at one day, `blend_arm_name` names the arms, `blend_arms(row_set, day)` lists every arm of one frame, `blend_shuffles` lists its shuffled columns, `blend_inputs` joins the extra-lead columns, and `fit_day5_aifs_wn3.py` is the pattern for a thin driver that calls `fit_aifs` and writes one write-once folder.

**What each product reaches.** ICON-EU: Open-Meteo Previous Runs, days 1 to 4. UKV live: Previous Runs fills only `previous_day1`, so day 1. AIFS Single and WN3: whole 00 UTC runs. UKV-CEDA differs statistically from live UKV, so no blend trains on one and infers on the other.

## Design

**A blend is extra columns in one XGBoost model, never an average of forecasts.** The blend of the ENS mean and product P at lead day d gives the model the ENS mean's day-d columns plus P's. Its reference is an XGBoost model given the ENS mean's columns alone (already in the frame). Its control has the same column count, with P's columns shuffled within generator, year-month, and UTC hour of day (`climatology_permutation`). `colsample_bytree` stays at 1, and the seeds, folds, eras, settings, and GPU device stay as in `fit_aifs.py`.

**Optimistic and conservative leads follow the published definition.** For a Previous Runs product, the optimistic blend reads the product at day d and the conservative blend reads its day-(d+1) offset (published P4a and P4b). The conservative blend decides any verdict. AIFS Single and WN3 are read from the 00 UTC run, as ENS is, so each has one reading. Whether that reading is conservative depends on the product's publication time against the 09:00 UTC issue time, which is unmeasured for WN3.

**New fits, on `single` only.** Product by product:

| Product | Arms to fit | Days | Notes |
|---|---|---|---|
| ICON-EU | blend (optimistic), blend (conservative), control, at day d and d+1 offsets | 1, 2 (ICON-EU ends at day 4 and has no day-7 offset on Previous Runs) | Wind day 2 cross-checks the published CPU blend |
| UKV live | blend, control | 1 | Day 1 only; no conservative offset exists |
| AIFS Single | reused, not refitted | 1, 2, 7, 14 | Files above |
| WN3 | blend, control inside `run_wn3` on the 7-month `wn3` rows | 1, 2, 7, 14 | Descriptive; see risks |

ICON-EU is fitted on `single` only. The `shared` 21-month row set is kept only if UKV's rows fail to cover `single` (the implementer checks the coverage table in the dry run first). Fits are about 20 to 60 s each on the A6000, so the new fits are on the order of 50 to 80 and take about an hour. No data is fetched, so the £30 fetch cap is not touched.

**The planned contrasts are fixed here, before any fit.** For each technology:

1. **C1:** ENS plus ICON-EU minus ENS alone, at day 1 and day 2, conservative lead, on `single`.
2. **C2:** ENS plus AIFS Single minus ENS alone, at day 1, 2, and 7, on `single` (existing losses, read again under the plan).
3. **C3:** ENS plus UKV live minus ENS alone, at day 1, on `single` (or `shared`, per the coverage check).
4. **C4:** each of C1 to C3's blends minus its own shuffled control (the guard).
5. **C5:** ENS plus AIFS Single minus ENS plus ICON-EU, at day 1 and day 2. This is the paired contrast a ranking of the two rests on, and the one comparison added to `dot_interval_vs_ens.py` (reference = the other blend).

Every other number is exploratory and labelled so: the day-14 rows, the WN3 rows, and any row not listed. The second hyperparameter setting runs on every planned contrast and on any result near the 5% line.

**The ranking rule is fixed before the run.** Product A ranks above product B at a technology and lead only if the month-resampled 95% interval of (blend A minus blend B), on rows both share, lies wholly below zero at the primary and the sensitivity setting. Otherwise the page says the two are not separable. A product whose blend-minus-ENS-alone interval includes zero is not recommended for that technology and lead. WN3 is never ranked against the other products, because it rests on 7 months.

## What changes, file by file

1. **`studies/nwp_forecast_comparison/fit_aifs.py`:** add `icon_eu` (optimistic day d and conservative day d+1) and `ukv` (day 1) to `BLEND_AIFS_PREFIXES` and `blend_arms`, `blend_shuffles`, and the arm-name parser; have `blend_inputs` join the ICON-EU and UKV columns from the published inputs onto `single`'s `(site, time)` keys. `run_wn3` gains the ENS plus WN3 blend and its control (shuffled WN3 columns).
2. **A thin driver in `studies/nwp_forecast_comparison/`** in the pattern of `fit_day5_aifs_wn3.py` (importing `fit_aifs` and `studies.guards.refuse_to_overwrite`, which that script does already): fits only the arms whose losses the AIFS blends folder lacks, at days 1, 2, and 7 on `single`, into one new write-once folder that it refuses to overwrite, with a `--check` first and every arm's column list printed into its `report.md`.
3. **`studies/nwp_forecast_comparison/dot_interval_vs_ens.py`:** add blend `ProductRow`s (behind a flag or a second `PRODUCTS` tuple) that read the reused and new losses, and the C5 comparison, reusing `_check_same_keys`. The figure's panels, hollow marks, and dashed short intervals stay as they are.
4. **`docs/studies/forecasts/blends-with-ens.md`** in the study page structure, the `mkdocs.yml` nav entry, and a link from the matched-lead page's blend section.
5. **`studies/nwp_forecast_comparison/README.md`:** the new folder and the new flag.

## Tests

Study scripts are not unit-tested, so the check is each script's own `report.md`, from which every page number is taken. Two script-level guards each have an assertion that fails if broken: the driver raises on a folder that exists (`refuse_to_overwrite`), and `_check_same_keys` raises when a blend and its reference differ in `(site, time, seed)` keys, which the C5 comparison exercises. If an implementer adds a helper to `packages/studies/`, that helper gets a test that fails on `main` and the mutation pass runs.

## Design-philosophy check

R&D path: fails fast, as the inherent-stability page requires for CV, training, and metrics code. Nothing runs in production, so no degradation rule, asset check, or Sentry tag applies. Outputs carry only anonymised `site` labels, and no per-generator series is charted with a label that identifies it.

## Docs to update

The new page, `mkdocs.yml`, a link from the matched-lead page's blend section, `docs/background/weather-products-survey.md` where it recommends which products to add to ENS, and the roadmap's "Several NWP sources as features" section with the ranking. Each is written in the present tense.

## Verification commands

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest`, `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, the pydoclint and docs-link-checker steps that CI runs, `nvidia-smi` and a CPU-load check before each fit, and the driver's `--check`.

## Risks and open questions

1. **Which lead days?** Recommendation: 1, 2, 7, and 14 (the last from saved losses), as the issue names them. Days 3 and 5 go to a follow-up.
2. **Is WN3's blend row comparable?** Recommendation: no. It rests on 7 months (February to April and June to 10 September 2026). Show it as its own row with a hollow mark and dashed interval, against the same-rows ENS mean and the 21-month ENS mean on shared keys, and never rank it against rows on other row sets. Its publication latency and lookahead are unestablished.
3. **UKV now, or later?** Recommendation: live UKV at day 1 now, and UKV-CEDA (days 2 to 5, and the T120 store) in a follow-up. UKV's era boundary (2026-01-21) needs era-cut folds, as `fit_aifs.py` already does.
4. **CRPS and percentiles?** No. They belong in the main ML experiments.
5. **How do blends enter the headline figures?** Recommendation: a new short page with its own headline figure and ranking table, leaving Figures 1 and 2 (single products) unchanged. Adding blend rows to them would mix row sets that differ in months. An alternative that would also satisfy the issue is one new section and one figure inside the matched-lead page's existing blend section.
6. **Does the C5 comparison belong in `dot_interval_vs_ens.py`?** Recommendation: yes, through one flag, because the file already pairs a product with its reference.

**Follow-up issues to open after approval (not opened by this plan):**

- Blends of ENS plus two products (is ICON-EU plus IFS 0.25° better than the best single addition?), with pairs chosen by a written rule.
- UKV-CEDA blends, including the T120 store to 5 days once its download completes.
- Days 3 and 5 for the ICON-EU, UKV, AIFS Single, and WN3 blends.

## Findings of the simplicity review, triaged

**Accepted:**

- Reuse the saved AIFS Single blend losses, verified above as the same rows, folds, seeds, settings, and device by construction. The published wind ICON-EU blend is not reusable, because it is a CPU fit on the 21-month rows.
- No new `fit_blends.py`: extend `fit_aifs.py`'s blend table and add a thin driver.
- No move into `packages/studies/`, no helper tests, and no mutation pass; reuse `_check_same_keys` and `refuse_to_overwrite`.
- No `blend_charts.py`: extend `dot_interval_vs_ens.py` with blend rows and the C5 comparison.
- Cut stage 2 (two-product blends) to a follow-up issue.
- Cut the ICON-EU and AIFS Single refits on the 7 WN3 months: fit ENS plus WN3 and its control inside `run_wn3`, descriptive.
- Fit ICON-EU on `single` only, keeping `shared` only if UKV's rows fail to cover `single`.
- Days 1, 2, 7, and 14 (the issue's days, from the existing day-14 losses), with days 3 and 5 in a follow-up.

**Rejected:**

- Put the results in the matched-lead page's existing blend section and drop the new page: the ranking is a separate conclusion from the matched-lead page, which is already about 2,100 lines. The reviewer's point that one section and one figure there would also satisfy the issue is recorded as risk 5, for the maintainer to decide.
