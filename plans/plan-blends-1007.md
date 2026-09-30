# Plan: which weather forecast to add to ECMWF ENS first (#1007)

**The problem.** The matched-lead page scores each weather product alone against the ENS mean, and tests one blend (ENS, ICON-EU, and IFS 0.25°) at day 1 and day 2. The page cannot say which single product to add to ENS first, because "ENS mean plus one product" blends exist only for AIFS Single (already fitted, not yet charted against the other products), and do not exist for ICON-EU at the GPU harness's rows, UKV, or WeatherNext 3 (WN3).

**The plan.** Add a new arm table (a new function, leaving the published arm lists untouched) for ICON-EU and UKV blends, fit the missing arms on the `single` row set into one new write-once folder through a thin driver, reuse the saved AIFS Single blend losses, and fit an ENS plus WN3 blend and its control in the same driver (descriptive only). Draw (blend minus ENS mean alone) as dot-and-interval rows at the primary setting by adding blend rows and one comparison to `dot_interval_vs_ens.py`, and print every planned contrast at both settings in the driver's `report.md`. The results go on a new short page under Studies > Forecasts, and the matched-lead page links to it.

## Verdict, size and departures

**Verdict: worth doing, at a smaller scope than the first draft.** Nothing in the issue is stale. No open PR or issue covers it: searches for "blend" and "AIFS" found #836 (past-weather blending, closed), #957 (CERRA wind blend, closed), #923, #954, and #974 (AIFS and WN3 on the matched-lead page, closed), and #999 (short product history, open, a different question). The issue is attached under #896, the parent of #810 and #974.

**Size: complex.** The five triggers:

1. **What gets stored: fires.** One new write-once folder under `data/studies/`, a published page, new figures. No Patito model, Delta table, or asset changes.
2. **Production serving path: does not fire.** Only `studies/` and `docs/studies/` change.
3. **Degradation rule: does not fire.** Study code is R&D and fails fast.
4. **More than one defensible design: fires.** The decisions under "Risks and open questions".
5. **Callers not nameable without searching: fires.** `blend_arms`, `stage_arms_fitted`, and `wn3_arms` are called by `run_blends` and `run_wn3`, which call `check_saved_losses`, which refuses saved losses whose arms differ. Four published folders (`_aifs_blends`, `_wn3`, `_wn3_extra_days`, `_day5_aifs_wn3`) therefore depend on those lists, and three test files pin them: `packages/studies/tests/test_fit_aifs_blends.py`, `test_wn3_split.py`, and `test_dot_interval_vs_ens.py`.

Four triggers fire, so the size stays complex.

**Reviews bought:** the simplicity review of this plan has run and its findings are triaged below. The parent runs a second, correctness review. After the code, the `study` skill requires two Opus scientific-validity reviews, a diff review, and a `prose-review` of the page. `packages/studies/tests/` changes (three test files are extended), and `packages/studies/src/` does not, so the mutation pass runs on the new arm function and the new asserts only if an implementer changes `packages/studies/src/`.

**Departures from the issue body.** The issue asks for days 0, 3, 4, 5, and 10 "as available". This plan charts days 1, 2, 7, and 14, but only where a product reaches them: ICON-EU blends exist at days 1 and 2 only, UKV at day 1 only, and the AIFS Single blend (reused) and the WN3 blend (new, descriptive) at days 1, 2, 7, and 14. Days 3 and 5 move to a follow-up issue. It drops day 0 (a day-0 value can come from a run a live service cannot read) and day 10 (the page shows the ENS mean no better than climatology from day 10). It moves two-product blends and UKV-CEDA to follow-up issues. The ENS plus WN3 blend is descriptive and is not refitted beside ICON-EU or AIFS Single on the 7 WN3 months.

## What exists today, and what is reused

**AIFS Single blends are fitted already, by the same code and device, but not at every setting.** `data/studies/nwp_forecast_comparison_aifs_blends/{solar,wind}_single_day{1,2,7,14}_losses.parquet` hold `blend_aifs_single_dayN` with its `_control` and `_mirror` arms, `ens_mean_dayN`, and `aifs_single_dayN` on the `single` row set (16 months; 85,710 solar rows per arm at day 1). Every arm's `device` column reads `cuda`, and the stamp (`*_losses.json`) records the two hyperparameter settings, the A6000, and XGBoost 3.4.1. The files lack most sensitivity arms: solar day 2 holds none, and solar `ens_mean_day1` and wind `blend_aifs_single_day1_control` are primary-only. The ranking rule needs both settings, so the driver lists every (arm, setting) pair that the reused folder lacks, for the contrasts C1 to C5, and refits exactly those pairs on the same rows, folds, seeds, and device. The implementer prints that list in the dry run.

**"Same rows by construction" is checked, not assumed.** Before the driver uses any reused arm, it asserts that its own stamp's inputs SHA-256, GPU name, and XGBoost version equal the reused folder's stamp, and that the `(site, time, seed, fold)` keys of each new arm equal those of the saved `ens_mean_dayN`. Both asserts have tests that use a `tmp_path` fixture and the monkeypatched `out_of_fold_losses` fake that `test_fit_aifs_blends.py` already uses (near line 323), so no GPU is needed.

**The published wind ICON-EU blend cannot be reused as a comparison.** `blend_wind_icon_eu_day2` and `blend_wind_ifs025_day2` (with controls) sit in `data/studies/nwp_forecast_comparison/wind_losses.parquet`, which has no `device` column (a CPU fit) and scores the 21-month `shared` rows, not `single`. They serve as a cross-check of the new GPU ICON-EU blend's wind day-2 direction, never as an arm in a paired contrast.

**The day-14 AIFS blend losses and the day-7 IFS 0.25° reference are also reused** as they are, for day 14 and as a stage-free reference at day 7. `nwp_forecast_comparison_p4_seeds` holds the GPU refit of the three-product P4 blends and controls on the published rows.

**The harness already takes a product as a parameter.** `BLEND_AIFS_PREFIXES` maps a blend's product name to its weather-column prefix at one day, `blend_arm_name` names the arms, `blend_arms(row_set, day)` lists every arm of one frame, `blend_shuffles` lists its shuffled columns, `blend_inputs` joins the extra-lead columns, and `fit_day5_aifs_wn3.py` is the pattern for a thin driver that calls `fit_aifs` and writes one write-once folder.

**What each product reaches.** ICON-EU: Open-Meteo Previous Runs, days 1 to 4. UKV live: Previous Runs fills only `previous_day1`, so day 1. AIFS Single and WN3: whole 00 UTC runs. UKV-CEDA differs statistically from live UKV, so no blend trains on one and infers on the other.

## Design

**A blend is extra columns in one XGBoost model, never an average of forecasts.** The blend of the ENS mean and product P at lead day d gives the model the ENS mean's day-d columns plus P's. Its reference is an XGBoost model given the ENS mean's columns alone (already in the frame). Its control has the same column count, with P's columns shuffled within generator, year-month, and UTC hour of day (`climatology_permutation`). `colsample_bytree` stays at 1, and the seeds, folds, eras, settings, and GPU device stay as in `fit_aifs.py`.

**Optimistic and conservative leads follow the published definition.** For ICON-EU, the optimistic blend reads the product at day d and the conservative blend reads its day-(d+1) offset (published P4a and P4b). `arm_prefixes` builds a blend's prefixes from `BLEND_AIFS_PREFIXES[product].format(day=day)` beside a fixed `ens_mean_day{d}`, so the conservative blend needs a second product key that reads `icon_eu_day{d+1}` beside `ens_mean_day{d}`. AIFS Single and WN3 are read from the 00 UTC run, as ENS is, so each has one reading. Whether that reading is conservative depends on the product's publication time against the 09:00 UTC issue time, which is unmeasured for WN3. UKV live has only the optimistic day-1 value and no clear run cycle (the V1b score was 1.07 to 1.10 against the 1.3 threshold), so UKV has no conservative reading.

**New fits, on `single` only, in a new function called only by the new driver.** `blend_arms`, `stage_arms_fitted`, and `wn3_arms` are not changed (see "What changes"). The published `single` rows already carry `icon_eu_day1` to `icon_eu_day3` and `ukv_day1` with no null on any row at days 1, 2, and 7, so `blend_inputs` needs no change and no `shared` fallback is needed.

| Product | Arms to fit | Days | Notes |
|---|---|---|---|
| ICON-EU | blend (optimistic: `icon_eu_day{d}`), blend (conservative: `icon_eu_day{d+1}`), each with a control | 1, 2 | ICON-EU has no day-7 or day-14 value in the published inputs, so it has no cell there |
| UKV live | optimistic blend and control | 1 | No conservative reading; an optimistic upper bound |
| AIFS Single | reused, plus the missing sensitivity arms | 1, 2, 7, 14 | |
| WN3 | ENS plus WN3 blend and its control, on the 7-month `wn3` rows | 1, 2, 7, 14 | Descriptive. The wind blend uses `ens_mean` columns, so its reference is `ens_mean_dayN`, not `ens_meanvec_dayN` |

Fits take about 20 to 60 s each on the A6000, so the new fits, including the missing sensitivity arms, are on the order of 60 to 100 and take about an hour. No data is fetched, so the £30 fetch cap is not touched.

**The planned contrasts are fixed here, before any fit.** For each technology:

1. **C1:** ENS plus ICON-EU minus ENS alone, at day 1 and day 2, at the conservative lead (and at the optimistic lead, exploratory), on `single`.
2. **C2:** ENS plus AIFS Single minus ENS alone, at day 1, 2, and 7, on `single`.
3. **C3:** ENS plus UKV live minus ENS alone, at day 1. This is an optimistic upper bound, and UKV is never ranked.
4. **C4:** each of C1 to C3's blends minus its own shuffled control (the guard).
5. **C5:** ENS plus AIFS Single minus ENS plus ICON-EU, at day 1 and day 2, the one comparison added to `dot_interval_vs_ens.py`. The ICON-EU conservative blend reads runs at least 48 hours old where the AIFS Single blend reads the same-day 00 UTC run, so C5 is biased towards AIFS Single. C5 is therefore run against both ICON-EU blends.

Every other number is exploratory and labelled so: the day-14 rows, the WN3 rows, and the optimistic ICON-EU contrasts. The second hyperparameter setting runs on every planned contrast and on any result near the 5% line.

**The ranking rule is fixed before the run.** The rule reuses `lead_verdict`, which needs both (blend minus ENS alone) and (blend minus control) to lie below zero, because on the matched-lead page the solar controls were worse than ENS alone. A product is recommended at a technology and lead only if `lead_verdict` passes at the primary and the sensitivity setting. AIFS Single ranks above ICON-EU only if C5 excludes zero against the optimistic ICON-EU blend, and ICON-EU ranks above AIFS Single only if C5 excludes zero against the conservative one. Otherwise the page says the two are not separable. About 16 recommendation tests run at uncorrected 95% intervals, so the page states that about one in twenty would pass by chance, and labels every ranking as uncorrected for multiplicity. WN3 and UKV are never ranked.

## What changes, file by file

1. **`studies/nwp_forecast_comparison/fit_aifs.py`: add, do not change.** `blend_arms`, `stage_arms_fitted`, and `wn3_arms` keep their current lists, because `run_blends` and `run_wn3` call `check_saved_losses`, which refuses saved losses whose arms differ, and the four published folders would stop being re-runnable. A new function (for example `product_blend_arms`) lists the ICON-EU, UKV, and WN3 blend arms, and is called only by the new driver. `BLEND_AIFS_PREFIXES` gains the ICON-EU, conservative ICON-EU, and UKV keys that `arm_prefixes` reads, and `blend_shuffles` covers their controls.
2. **A thin driver in `studies/nwp_forecast_comparison/`** in the pattern of `fit_day5_aifs_wn3.py` (importing `fit_aifs` and `studies.guards.refuse_to_overwrite`): fits the new arms, the missing sensitivity arms, and the WN3 blend and control, into one new write-once folder, with a `--check` first, the two stamp and key asserts above, every arm's column list in `report.md`, and C1 to C5 at both settings in `report.md`.
3. **`studies/nwp_forecast_comparison/dot_interval_vs_ens.py`:** add blend `ProductRow`s (behind a flag or a second `PRODUCTS` tuple) and the C5 comparison, reusing `_check_same_keys`. `load_arms` reads the primary setting only, so the dots stay at the primary setting and the two-setting numbers come from the driver's `report.md`.
4. **`docs/studies/forecasts/blends-with-ens.md`** in the study page structure, the `mkdocs.yml` nav entry, `docs/studies/index.md` (which lists every study page), and a link from the matched-lead page's blend section.
5. **`studies/nwp_forecast_comparison/README.md`:** the new folder and the new flag.

## Tests

**Three test files already pin this code, and the plan extends them.** `packages/studies/tests/test_fit_aifs_blends.py` covers `blend_arms`, `arm_prefixes`, and `check_saved_losses` (its `test_each_stage_holds_the_arms_the_plan_counts` expects 30 arms), `test_wn3_split.py` covers `wn3_arms` (near line 170), and `test_dot_interval_vs_ens.py` covers the dot script. New assertions, each failing on `main` today because the function or key does not exist or the behaviour is absent:

- **The old arm lists stay pinned:** `blend_arms`, `stage_arms_fitted`, and `wn3_arms` return their current lists (a guard against the change in finding 2).
- **Conservative prefixes:** `arm_prefixes("blend_<conservative>_day2") == ("ens_mean_day2", "icon_eu_day3")`, and the control reads `icon_eu_day3_permuted`.
- **The new arm function** lists each new blend with a control of the same column count.
- **The stamp assert raises** when the inputs SHA, GPU, or XGBoost version differs from the reused folder's, and **the key assert raises** when a new arm's `(site, time, seed, fold)` differs from the saved `ens_mean_dayN`, both on a `tmp_path` fixture and the monkeypatched `out_of_fold_losses` fake, with unequal capacities across sites.
- **`test_dot_interval_vs_ens.py`:** the C5 comparison raises through `_check_same_keys` when its two blends' keys differ.

`packages/studies/src/` is not planned to change, so the mutation pass runs only if it does.

## Design-philosophy check

R&D path: fails fast, as the inherent-stability page requires for CV, training, and metrics code. Nothing runs in production, so no degradation rule, asset check, or Sentry tag applies. Outputs carry only anonymised `site` labels, and no per-generator series is charted with a label that identifies it.

## Docs to update

The new page, `mkdocs.yml`, a link from the matched-lead page's blend section, `docs/studies/index.md`, `docs/background/weather-products-survey.md` where it recommends which products to add to ENS, and the roadmap's "Several NWP sources as features" section with the ranking. Each is written in the present tense.

## Verification commands

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest`, `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, `uv run pre-commit run --all-files` (which includes the pydoclint and docs-link-checker steps), `uv run pytest --run-network -m network` if any network-marked test is touched, `nvidia-smi` and a CPU-load check before each fit, and the driver's `--check`.

## Risks and open questions

1. **Which lead days?** Recommendation: 1, 2, 7, and 14 where a product reaches them (ICON-EU days 1 and 2, UKV day 1, AIFS Single and WN3 all four). Days 3 and 5 go to a follow-up.
2. **Is WN3's blend row comparable?** Recommendation: no. It rests on 7 months (February to April and June to 10 September 2026). Show it as its own row with a hollow mark and dashed interval, against the same-rows ENS mean. The 21-month ENS mean on shared keys is a second mark only, because it compares a 7-month fit with a 21-month one. Never rank WN3 against rows on other row sets. Its publication latency and lookahead are unestablished.
3. **UKV now, or later?** Recommendation: live UKV at day 1 now, as an optimistic upper bound that is never ranked, and UKV-CEDA (days 2 to 5, and the T120 store) in a follow-up. The `single` row set drops 2026-01, the month of the UKV upgrade on 2026-01-21, rather than cutting an era at it.
4. **CRPS and percentiles?** No. They belong in the main ML experiments.
5. **How do blends enter the headline figures?** Recommendation: a new short page with its own headline figure and ranking table, leaving Figures 1 and 2 (single products) unchanged. Adding blend rows to them would mix row sets that differ in months. An alternative that would also satisfy the issue is one new section and one figure inside the matched-lead page's existing blend section.
6. **Does the C5 comparison belong in `dot_interval_vs_ens.py`?** Recommendation: yes, through one flag, because the file already pairs a product with its reference.

**Follow-up issues to open after approval (not opened by this plan):**

- Blends of ENS plus two products (is ICON-EU plus IFS 0.25° better than the best single addition?), with pairs chosen by a written rule.
- UKV-CEDA blends, including the T120 store to 5 days once its download completes.
- Days 3 and 5 for the ICON-EU, UKV, AIFS Single, and WN3 blends.

## Findings of the simplicity review, triaged

**Accepted:**

- Reuse the saved AIFS Single blend losses, with the same rows, folds, seeds, and device checked by the driver's asserts, not assumed (correctness review 3 and 8). The published wind ICON-EU blend is not reusable, because it is a CPU fit on the 21-month rows.
- No new `fit_blends.py`: add a new arm function to `fit_aifs.py` and a thin driver.
- No move into `packages/studies/src/`; reuse `_check_same_keys` and `refuse_to_overwrite`.
- No `blend_charts.py`: extend `dot_interval_vs_ens.py` with blend rows and the C5 comparison.
- Cut stage 2 (two-product blends) to a follow-up issue.
- Cut the ICON-EU and AIFS Single refits on the 7 WN3 months: fit ENS plus WN3 and its control inside `run_wn3`, descriptive.
- Fit ICON-EU on `single` only, keeping `shared` only if UKV's rows fail to cover `single`.
- Days 1, 2, 7, and 14 (the issue's days, from the existing day-14 losses), with days 3 and 5 in a follow-up.

**Rejected:**

- Put the results in the matched-lead page's existing blend section and drop the new page: the ranking is a separate conclusion from the matched-lead page, which is already about 2,100 lines. The reviewer's point that one section and one figure there would also satisfy the issue is recorded as risk 5, for the maintainer to decide.

## Findings of the correctness review, all accepted

1. The three test files that pin this code exist, and the plan extends them instead of claiming none exist.
2. `blend_arms`, `stage_arms_fitted`, and `wn3_arms` are not changed, because `check_saved_losses` would refuse every published folder. A new function serves the new driver only, and a test pins the old lists.
3. The saved AIFS losses lack most sensitivity arms, so the driver refits every missing (arm, setting) pair that C1 to C5 need.
4. `dot_interval_vs_ens.py` reads the primary setting only, so the driver's `report.md` prints C1 to C5 at both settings.
5. The conservative ICON-EU lead needs a second product key in `BLEND_AIFS_PREFIXES`, with a test on `arm_prefixes` and on the control's prefix.
6. C5 is biased by lead, so the ranking rule runs it against both ICON-EU blends.
7. UKV is an optimistic upper bound and is never ranked.
8. "Same rows by construction" is replaced by two tested asserts (stamp, and keys against the saved `ens_mean_dayN`), using a `tmp_path` fixture and the monkeypatched fit.
9. `blend_inputs` needs no change, and the `shared` fallback is dropped.
10. Neither ICON-EU nor UKV reaches day 7 or day 14, so the plan promises no such cells.
11. The WN3 wind reference is `ens_mean_dayN`, and the 21-month reference is a second mark only.
12. The ranking rule reuses `lead_verdict` and states the multiplicity treatment.
13. Risk 3 is reworded: `single` drops 2026-01 instead of cutting an era.
14. `docs/studies/index.md` joins the docs to update.
15. The verification set adds `pre-commit run --all-files` and the network-marked tests where touched.
16. Trigger 5 fires, and the size stays complex.
