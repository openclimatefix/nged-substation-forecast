# Revise the matched-lead NWP study page (#912)

## The problem

The page `docs/studies/nwp-forecasts-at-matched-leads.md` (1,145 lines, merged in PR #901 for issue #810) has four faults the maintainer named. Its headings lean on how the leads are matched, which is method, so a reader meets methods inside "Results" and the answer late. Its most useful figures (day-1 absolute error against climatology, Figures 5 and 6) sit fifth and sixth, behind the planned contrasts, so a reader sees gaps between products before seeing how good any of them is. Figures 11 and 12 draw two colours for seven forecast products (six in the wind chart), so a reader cannot tell the lines apart. The second XGBoost setting appears beside every contrast, including exploratory ones, which doubles the numbers a reader must take in.

## The planned solution

Two pull requests. **PR 1 (skill, small, own branch `study-skill-page-structure`)** writes the paper structure and the second-setting rule into `.claude/skills/study/SKILL.md`, agreed with the "Past weather: Solar" session. **PR 2 (this branch)** restructures the page to that outline, moving sentences verbatim wherever possible, puts a multi-lead absolute-error figure first, keeps the planned contrasts second, recolours the by-lead-day lines with one validated colour per product, and drops the second setting from exploratory rows unless the result is near the 5% line. The multi-lead figure needs a small refit (day 5 and day 14 arms, primary setting only), which waits for the runner slot from the coordinator; the figure is first built from the leads already fitted (day 0 to day 3), and the new leads are added when the fit lands.

## Verdict, size and departures

**Verdict:** worth doing, as the maintainer asked. **Size: complex**, from the five triggers:

1. **What gets stored:** yes. Redrawn SVGs, a published page, and (for day 5 and day 14) a new write-once results folder `data/studies/nwp_forecast_comparison_leads/` holding losses, predictions and a report. No Delta table, Patito model or asset.
2. **Production serving path:** no. Nothing under `defs/`, `ml_core` or `xgboost_forecaster`.
3. **Degradation rule:** no.
4. **More than one defensible design:** yes. How to show several lead days on one figure (see "Figure design"), and whether the day-14 arms sit on the leaderboard or in their own panel.
5. **Callers not nameable without searching:** yes. Figure numbers are baked into SVG titles, chart-script constants and page text; anchors into the page come from `docs/studies/index.md`, the survey page, roadmap pages and script docstrings.

Reviews bought: both plan reviews, both diff reviews, plus the extras the coordinator specified (scope-of-claims and fact review of every changed sentence containing a number, chart design review, one-rule prose sweeps, a network-planner persona). The maintainer expected medium; the study skill says a published page counts as what gets stored, so the size is complex.

**Departures from the request:** none in scope. The maintainer's "second setting" rule is applied as the coordinator relayed it (planned and deciding contrasts, and near the 5% line).

## Feasibility of the multi-lead figure (checked against the saved losses and inputs)

Fitted arms in `solar_losses.parquet` and `wind_losses.parquet` (both settings):

| Product | Fitted lead days | Not fitted, and whether the input exists |
|---|---|---|
| ENS mean | 0, 1, 2, 3 | 5, 7, 10 and 14 exist in the ENS store (leads 114 to 150 h and 330 to 360 h for days 5 and 14); the ENS-horizons page fits ENS at those days on rows that span the IFS 49r1 change |
| GEFS mean | 1, 2, 3 | the GEFS store holds 00 UTC runs at 3-hour steps to 240 h and 6-hour steps from 246 h to 840 h; days 5 and 14 exist |
| IFS 0.25° | 1, 2, 3 | previous_day4 to previous_day7 hold data (100% of hours from 2024-12-01, the study's span) |
| GFS | 1, 2, 3 | previous_day4 to previous_day7 hold data (100% from 2024-12-01) |
| ICON global | 1, 2, 3 | previous_day4 to previous_day6 hold data (99.5% to 100% from 2024-12-01); day 7 is empty |
| ICON-EU | 1, 2, 3 | previous_day4 holds data (99.4% from 2024-12-01); day 5 and later are empty, so ICON-EU ends at day 4 |
| ARPEGE (solar only) | 1, 2, 3 | `previous_day4` onwards is empty, so ARPEGE ends at day 3 |
| UKV, ICON-D2, AROME, DMI and KNMI HARMONIE-AROME | 1 | day 2 and later are empty in the Previous Runs archive (UKV's day 1 is filled on 96% of hours from 2024-12-01) |

So the maintainer's five leads 0, 1, 3, 5 and 14 are all fitted only for ENS at 0, 1 and 3. Day 5 needs a refit for ENS, GEFS, IFS 0.25°, GFS and ICON global. Day 14 needs one for ENS and GEFS only, because the Previous Runs archive stops at day 7. ICON-EU, UKV and ICON-D2 cannot be shown at day 5, and the figure marks that absence. Nothing is interpolated.

**Refit scope, decided after plan review 2.** New arms (all exploratory, primary setting only, on the GPU as the maintainer asked): ENS mean at day 5 and day 14; IFS 0.25°, GFS and ICON global at day 5; GEFS mean at day 5, 10 and 14; plus ENS mean at day 1 refitted on the GPU as a device noise floor, and the ECMWF AIFS arms when the download lands (AIFS goes on its own row set and off Figure 1; see "ECMWF AIFS"). GEFS mean at day 10 and day 14 are added as exploratory arms at the maintainer's request: the earlier reason for leaving them out was a code limit, not a data limit (the on-disk GEFS cache holds 3-hourly leads to 240 h and 6-hourly leads to 840 h, with no missing value at leads 114 to 366 h in 2026-02 and 2026-08).

**Changes GEFS days 5, 10 and 14 need (each reviewed by Opus before anything runs).** (1) The build's 240 h lead filter (`GEFS_STEP_MEAN_MAX_LEAD_HOURS`) is lifted to the longest lead a band reads, and `studies.resample.gefs_step_means` accepts the 6-hourly steps beyond 240 h (in `packages/studies`, so it gets tests for the new limit and a mutation pass). (2) `ens_forecast_horizons._step_width` takes the 3-hourly to 6-hourly switch as a parameter (144 h for ENS, 240 h for GEFS), so the day-5 band (114 to 150 h) and the day-10 band (which crosses 240 h) get the right step widths. (3) The complete-run check covers every lead each band reads, not 0 to 95 h. (4) A read-only verification script, `verify_extra_leads.py`, checks which window GEFS's radiation averages over at 6-hourly steps: at valid hours 00, 06, 12 and 18 UTC it compares GEFS's window mean with clear-sky irradiance averaged over the 3-hour and the 6-hour window ending at the valid time, and requires the ratio for the chosen window to match the ratio at leads below 240 h, where the window is documented. The build is written only if the check passes; if it fails, the page says GEFS is not fitted beyond day 5 and gives that data reason.

**Mechanism (plan review 2, findings 10 to 12).** The new days go in their own constant (`EXPLORATORY_LEAD_DAYS`) that reaches only the arm builders, never `ENS_DAYS`, because `ENS_DAYS` also decides which baseline columns the shared rows require, and a day-5 or day-14 requirement would move the shared row set and every published number. A new script ``build_forecast_inputs.py --extra-leads` in `studies/nwp_forecast_comparison/`` takes the published `<domain>_forecast_inputs.parquet` keys, joins the new columns onto them, and writes to a required `--output-dir` (no default) in a new write-once folder. The fit step copies nothing: it fits only the new arms, and writes their losses and predictions, and its own report, to the new folder. The published `report.md` is regenerated from the published losses only in a scratch copy, to prove the published tables are byte-identical.

**Cost.** Each new arm is 3 fitting seeds, per generator, per fold, at one setting: about a dozen arms per technology against 70 and 74 (arm, setting) jobs in the published run. The runner request gives the wall time once one arm is timed on the GPU.

## Figure design (recommended)

**Recommendation: one figure per technology, products as rows, lead days as coloured marks on each row, plus a table that gives every number.** Rows sort by day-1 error, best first. Each row carries up to six marks, one per fitted lead day (0, 1, 2, 3, 5, 14), each with its 95% interval as a short line; the marks are dodged vertically inside the row so intervals do not overprint. Lead day is the colour, so Figure 1's key is the lead days (validated for colour-blind separation), and the product names sit on the axis. A lead day a product does not have is drawn as nothing, and the row's right-hand label says "day 5: not fitted" or "day 5: no forecast that long" where the reason is known. The climatology and smart-persistence baselines sit as reference lines through the plot, not as rows.

Trade-offs against the alternatives:

- **Lead days as colour on product rows (recommended)** answers "how good is each product" first, which is what the maintainer called the headline, and shows the rise with lead in one glance. Its weakness is that colour must carry lead, so products cannot also be coloured; the product name on the axis does that job.
- **Lines with lead day on the x axis, one colour per product (today's Figures 11 and 12, recoloured)** show the slope with lead best and hold the ENS bracket shading. Its weakness is seven products of similar level at day 1, which overprint. This design stays as the second lead figure, because the ENS bracket shading is the page's evidence that bracketed products lose.
- **One panel per lead day** is easy to read and repeats every product name five times, so the page grows by four charts. Rejected.

Colours for the lead-day figure and for the recoloured lines are validated with `validate_palette.py` (all-pairs, light mode). The best six-colour set found from the brand colours and the extra colours (Data Blue, Data Sky, Data Green, Data Amber, Data Deep Teal, Data Burnt Orange) has a worst all-pairs colour-blind distance of 10.5 and a worst normal-vision distance of 19.5. Data Green fails the script's lightness band (L 0.81 against a ceiling of 0.77) and three colours have a contrast warning against the page background, so the direct labels carry those lines. No seventh chromatic colour passes. Wind's lines have six products, so each gets one colour. Solar's have seven; ARPEGE, which is in the solar chart only, is drawn black and dashed with a direct label (black passes both separation checks). The `dataviz` skill records that the maintainer once swapped Data Burnt Orange for Data Magenta on the ENS-horizons page because Burnt Orange clashed with Data Amber; this page uses both, because no other set of six passes, and the PR body says so. The five chromatic lead colours of the leaderboard (Data Blue, Sky, Deep Teal, Amber, Burnt Orange) pass every check except contrast; day 0 is black.

## Page structure (agreed with the "Past weather: Solar" session)

Title; summary (two paragraphs at most) with the headline figures directly under it; disclaimer; key findings; introduction; data and methods; results; discussion (what to use); limitations; scope; data and code availability; reproducing. The page already has scope, data and code availability, and reproducing. Moves, all verbatim where possible:

- **Summary:** shortened to two paragraphs (headline verdicts, the setting and interval convention stated once). The long definitions in today's opening paragraph (what a difference means, how the products are read) move to Data and methods and the Introduction table. Figures 1 and 2 become the products-by-lead-day figures (today's 5 and 6, extended to several leads and without the blend and control rows, whose absolute errors stay in Figures 9 and 10), Figures 3 and 4 the planned contrasts (today's 1 and 2), Figures 5 and 6 the models-work weeks (today's 3 and 4); Figures 7 to 12 keep their numbers. The Summary also gains scoped take-home bullets (one per use of the data), and keeps its sign convention and live-lead caveats beside the headline numbers.
- **Introduction:** the question, the product table, the three ways the products are read (moved from the summary), what ENS is here.
- **Data and methods:** everything under today's "How the leads are matched" (the lead rule, the bracket, the three verdicts, the planned/exploratory rule, the ENS-monotonicity test), the XGBoost model, scored hours and folds. The subsection headings are methods headings, not findings. The UKV radiation rebuild moves here from the lead section. Results sections keep only what the results show.
- **Results:** "The XGBoost models work" first (today's Figures 3 and 4, then the climatology and persistence paragraphs and the unequal-lead and overlapping-interval caveats, which travel with the leaderboard figure into its subtitle and stay in Results as a paragraph under a heading), then the planned contrasts for solar and wind, blending, ensemble averaging and timing, other products, error by lead day, the UKV eras. Every Results heading becomes the section's conclusion (its bolded lead, shortened), which renames 10 in-page anchors (page lines 77 to 99, 251, 252, 944) and no external anchor; inbound links from `docs/studies/index.md`, the survey and `xgboost-improvements.md` carry no anchor.
- **Discussion (what to use)**, **What this study cannot separate** (a Limitations subsection), **Limitations**, **Scope**, **Data and code availability**, **Reproducing**: four sentences change. Scope's "leads beyond day 3" limit (day 5 and day 14 arms exist for some products), Data and code availability's commit hash, Reproducing's commands (the new folder's build and fit), and Limitations' "the study reads leads to 95 h". The V1, V1b and V3 verification numbers stay in Data and methods as checks on the lead rule, under a heading that says so; this reads "No result appears here" (study skill, item 6) as covering results about weather products, not checks on the method.

Headings under Data and methods name a method; headings under Results state a result.

## Second XGBoost setting

The rule: keep the second setting for planned contrasts, for deciding contrasts (P4b and the guards) and for any result near the 5% line; drop it for exploratory arms; one table or a marker, never doubled charts; verdicts still need both settings to agree. "Near the 5% line" is the definition the solar session writes on the shared past-weather methods page: an interval bound within 20% of the interval's width from zero. On this page:

- Kept: the planned-contrast tables (P1 to P4b, guards) and the deciding-contrast paragraphs.
- Dropped unless near the line: the exploratory rows of the blending table (control minus ENS, IFS 0.25° minus ENS), and any other exploratory sensitivity figure. Each exploratory row is checked against the definition; a script prints which rows qualify, so the page states the rule and lists the rows that stay.
- The headline charts keep the primary setting as the dot and the second setting as a marker, since they show only planned contrasts. Figure 1 (leaderboard) is primary only.

## ECMWF AIFS (deferred to a later PR: the data had not landed, so nothing was built or fitted and the page does not mention AIFS)

The maintainer wants ECMWF's AIFS, its machine-learning weather model, both the single run (AIFS Single) and the ensemble (AIFS ENS), in the page. The download coordinator is fetching the data and will report the path, cadence, lead range and start date. AIFS is served at 6-hourly steps, so the plan settles fairness before any build.

**How 6-hourly steps map to hourly targets.** The rules are the ENS-horizons page's, reused through `ens_forecast_horizons` and `studies.resample`, and each is stated on the page. Accumulated surface solar radiation becomes a window mean over its 6-hour step. Solar radiation is interpolated to hourly targets through the clear-sky index, and temperature linearly. Wind speed and direction are interpolated as vector components. Interpolation stays within one run and never crosses runs, and each hourly target is read from the run whose step covers it, at the lead the study's row set fixes. Where AIFS serves 100 m wind, it is read as served; where it does not, wind is scored at 10 m only and the page says so.

**A like-for-like control.** A weather product served every 3 or 1 hours is favoured over a 6-hourly product by time resolution alone. The control is ENS's mean, ENS's control member and IFS 0.25° at day 1, each subsampled to the same 6-hourly valid times and interpolated with the same rules (`band_steps(six_hourly=True)` and `studies.resample.coarsen_to_six_hourly` already do this for ENS). AIFS is reported at its native steps beside these controls, and the hours that fall on an AIFS step are scored separately from the interpolated hours.

**Which rows are scored.** AIFS arms are scored on their own row set: the published shared rows restricted to the hours AIFS covers, with folds cut inside the study's eras and inside the AIFS version eras, and every reference (ENS at day 1, the controls above, IFS 0.25° at day 1) refitted on that row set on the same device. AIFS marks never share a chart axis with the shared-row marks of Figure 1, because absolute errors on different hours cannot be compared. Each AIFS number appears with its row count and month count.

**Version eras, keyed on the forecast's init time and never on its valid time.** Per ECMWF's Confluence page as relayed by the coordinator and awaiting the docs pull request that carries the era table: AIFS Single v1.0 from 2025-02-25 to 2025-07-31 00 UTC and from 2025-08-02 to 2025-08-26, with init times from 2025-07-31 06 UTC to 2025-08-01 18 UTC excluded (a v1.1 attempt that was reverted); AIFS Single v1.1 from 2025-08-27 06 UTC to 2026-05-11; AIFS Single v2 from 2026-05-12 00 UTC. AIFS ENS v1 from 2025-07-01 to 2026-05-11, v2 from 2026-05-12. The v2 cut coincides with IFS Cycle 50r1, so a v2 gain cannot be separated from a change in IFS itself, and the IFS 0.25° and ENS references are split at the same date. Each version era is reported separately, with its own absolute error; the v2 era is short.

**Labels and claims.** No AIFS contrast was written into the plan before a result existed, so every AIFS number is exploratory and post hoc, and the second setting appears only for a result near the 5% line. The Discussion answers "is AIFS improving faster than conventional weather models" only as far as the evidence reaches: the only trend evidence found so far is ECMWF's committee noting a small skill decrease over 12 months, so the page does not say AIFS is improving. The AIFS arms join the runner request only after this section is reviewed.

## ICON-D2 near-analysis arm (added at the coordinator's request)

The maintainer asked why ICON-D2 is the best product in the past-weather studies and mid-ranking at day 1 here. The hypothesis is that the past studies serve ICON-D2 at a lead of 0 to 3 hours and this study at about 24 to 26 hours. The plan adds one exploratory arm set: ICON-D2 Previous Runs at day 0, 1, 2 and 3 (where `previous_dayN` is filled on the study span) with ICON-EU day 0 and day 1 as references, on the identical shared rows, and a split of each error by lead hour modulo 3. It is checked first that `previous_day0` matches the historical-forecast series the past studies use. The page gets a short exploratory paragraph, with absolute skill at every lead, and a Discussion note that skill from a near-analysis feed does not carry to day-ahead use. The arms join the same runner request and the same GPU rules, and are exploratory (primary setting only).

## What changes, file by file

PR 1 (`study-skill-page-structure`):

- `.claude/skills/study/SKILL.md`: "Writing the page" gains the twelve-part outline (Summary carries scoped take-home bullets and is the only place a recommendation appears without its evidence; Discussion does not repeat them; Data and methods holds only what is specific to the study and links the shared methods page where one exists; new study pages go under Studies > Past weather or Studies > Forecasts); the second-setting rule in "Design rules" is replaced; the chart section says the first figure of a ranking page is the absolute-error figure and the planned contrasts follow. The line in the `CLAUDE.md` skills table for `study` is checked, and changed only if it now misdescribes the skill.

PR 2 (this branch), as built:

- `docs/studies/nwp-forecasts-at-matched-leads.md`: restructured as above.
- `studies/nwp_forecast_comparison/nwp_forecast_charts.py`: `by_lead_day` colours each product; a new leaderboard figure (products as rows, lead-day marks, absent leads left blank); figure numbers and titles in `FIGURE_NUMBERS` and `TITLES`; an `--extra-dir` option. No check of each SVG title's figure number is in the script: a scratch script compared every caption with `TITLES` instead.
- `studies/nwp_forecast_comparison/verify_extra_leads.py`, `build_forecast_inputs.py --extra-leads`, and `fit_extra_leads.py`: the extra lead days' checks, inputs, and GPU fits, written to the new write-once folder. `nwp_forecast_comparison.py` is unchanged for the refit, and `studies.resample.gefs_step_means` is unchanged (the build filters around it).
- `studies/nwp_forecast_comparison/README.md`: the new scripts and folder, and what each file holds.
- `packages/studies/`: `cross_validation.py` gains a `device` option that defaults to the CPU, with tests, and `ens_forecast_horizons.py` takes the step-width switch as a parameter.
- Inbound anchors: `docs/studies/index.md`, `docs/background/weather-products-survey.md`, `docs/roadmap/*` and the study README and docstrings were grepped for the page's anchors.

## Design-philosophy check

R&D and documentation only; nothing runs in production, so nothing degrades. The study fails fast (the build stops on missing runs). No asset check is added. No hypothesis (`H1`, `T1.2`) is claimed. The Discussion states what the evidence supports and does not commit the project to work.

## Tests

Study scripts have no unit tests; the check is their own output. The gates are:

- **Number conservation:** every decimal, and every integer of 10 or more, on the revised page appears in `git show origin/main:docs/studies/nwp-forecasts-at-matched-leads.md` or in `data/studies/nwp_forecast_comparison/report.md`; a number about a day-5 or day-14 arm must appear in a row of the new leads report that names that arm. The gate passes on the unedited page, so it guards against regressions and cannot show the restructure is right. This fails if a sentence is rewritten with a number the report does not hold. Blind spots (a number attached to the wrong row; a deleted qualifier) go in the PR body.
- **SVG title against caption:** for every figure, the figure number in the SVG title equals the number in the caption, and the grep for `Figure \d+` inside SVG text matches `FIGURE_NUMBERS`.
- **Second-setting rule:** a scratch script applies the near-the-line definition to every exploratory row on the page that shows a second setting, at either setting, and prints each row's margin, not only a yes or no. Plan review 2 found these rows near the line: wind P4a control minus ENS (at the primary setting), wind ENS plus IFS 0.25° at day 2 minus ENS (both settings), and wind IFS 0.25° day-2 guard (at the sensitivity setting only). The control-minus-ENS rows and the solar ENS-monotonicity rows decide whether the planned guard is informative and whether a bracket is voided, so they keep the second setting as deciding inputs whatever the rule gives. Places to apply it: the blending table, the solar-guard paragraph, the wind post hoc paragraph, the IFS guard sentence, the summary's "at either setting", the solar voiding paragraph, and Figures 9 and 10 (`blends()` draws the control-minus-ENS rows at both settings). The headline figure keeps the second setting as a marker.
- **Lead absence:** the figure script draws a mark only for an arm present in the saved losses, so an absent lead is blank by construction and nothing is filled.
- **Docs gates:** `uv run mkdocs build --strict` with the rendered HTML read, the docs-link check, `uv run pre-commit run --all-files`, and `pymarkdown scan`.
- **Refit checks (if it runs):** the new inputs' `(site, time)` keys equal the published inputs' keys, by an anti-join in both directions; every new arm's missing-feature share is printed and stays at or below 1.5% (the largest share among the published exploratory arms); the regenerated day 0 to 3 report from the published losses is byte-identical to the published `report.md`; a device noise floor is reported (ENS mean at day 1, fitted on the GPU, against the published CPU fit).
- **Figure-number checks:** the SVG title starts `Figure N:` where N equals the caption's N (the check matches that prefix only, because the leaderboard SVGs also cite another figure in their subtitle); every `Figure N` or `Figures N and M` in the page's prose names a figure whose caption says what the sentence says.

## Plan review 1 (simplicity): triage

Adopted:

- The page restructure keeps sentences verbatim and moves whole sections: a new "Data and methods" H2 holds the lead rule and the XGBoost model sections, a new "Results" H2 holds the rest, existing sections drop one level, and an anchor changes only where a heading is renamed. The summary is still cut to two paragraphs, because the study skill now says so.
- The second-setting check and the lead-absence check are cheaper: by hand, and by construction. The page-number guard is dropped (it has only ever run on the wind page).
- The fact check and the one-rule prose sweeps cover the sentences the diff changes, not the whole page.

Rejected, with reasons:

- *Drop the day 5 and day 14 refit; the issue asks only for fitted leads, and the ENS-horizons page has ENS at days 5 and 14.* Rejected: the maintainer asked, through the coordinator, for refits to get the multi-lead figure, and the ENS-horizons numbers span the IFS 49r1 change (issue #892) so they are not comparable with this study's rows.
- *Extend `studies.charts.leaderboard_panel` with a condition-colour option.* Rejected: one study calls it, and a change to `packages/studies` brings a mutation pass. The panel is drawn in the study's chart script.
- *Merge the skill change into this PR.* Rejected: the coordinator set the skill edit as its own small PR (#915), agreed with the Past weather: Solar session.
- *Drop the network-planner persona review.* Rejected: the coordinator specified it.
- *Drop Figures 11 and 12.* Deferred to the maintainer: the maintainer asked for them to be recoloured, so they stay, and the PR body asks whether to drop them once the rows figure exists.

## Plan review 2 (correctness): triage

The reviewer read the code, the saved losses, the stores and the skill, and found 22 defects. All 22 are real, and applied above: the feasibility percentages are over the study's span and the GEFS store holds 3-hour steps to 240 h and 6-hour steps beyond (1, 2); the arm counts and the figure list are corrected (3, 4, 14); four sentences the plan called unchanged are named (5); Results headings become conclusions, which the plan now says renames 10 in-page anchors (6); the Summary gets the take-home bullets and keeps its sign and live-lead caveats, and the verification numbers are declared a method check (7, 22); the ENS conversion sentence is corrected (8); GEFS day 14 was dropped and GEFS day 5 got a parameterised step width (9), and GEFS days 10 and 14 were reinstated later at the maintainer's request (see "Refit scope"); the new days get their own constant, the new inputs are keyed on the published inputs, and the fit step fits only the new arms into a new folder (10 to 12); every place that hard-codes days 0 to 3 is edited, or stated as left alone (13); smart persistence is drawn at day 1 only and labelled (15); the refit and conservation gates are strengthened (16, 17); the figure-number gate matches `Figure N:` and covers prose (18); the second-setting classification and the near-the-line rows are settled (19, 20); the colour claims are corrected and the Amber and Burnt Orange precedent is surfaced (21); AIFS stays off Figure 1 unless it shares the row set (22). The reviewer also noted uncommitted chart-script work in the worktree: the recolour and the products-by-lead figure were started while the plan was in review, and both are within the plan.

## Docs to update

The page itself; the study skill (PR 1); `studies/nwp_forecast_comparison/README.md`; inbound links listed above. Ship-time triage: no roadmap item completes; the plan file is deleted at ship time into the PR body. When "Past weather: Solar" moves this page to `studies/forecasts/matched-lead.md`, that session updates links; this branch avoids editing `mkdocs.yml`.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict      # then read the rendered HTML of the page
uv run pre-commit run --all-files # docstring markdown hook, pydoclint, docs-link check
```

Compute: no fit until the coordinator grants the runner slot. The refit runs one job at a time, checks `uptime` before and after (load under about 24 to start), uses a modest thread count, and writes only to the new folder.

## Hard rules restated

Generators appear only as A to F and W1 to W3; no station or farm mapping. Time-series charts show days 1 to 7, month and year in text only, % of capacity, `aria=False` on data marks. Scratch under `.claude/worktrees/scratch/`, body drafts under `/tmp/claude-1000/`. Nothing near an issue number says close, fix or resolve. Issue #810 stays open.

## Risks and open questions

1. **Day 5 and day 14 are new fits on a published study.** They ran, at the primary setting only, exploratory, in a new folder, and the published run stays untouched.
2. **ENS at day 14 is a different quantity from ENS at day 1**: ENS's steps turn 6-hourly beyond 144 hours, so the day-14 band is wholly 6-hourly and the clear-sky-index resample runs on 6-hour steps. The new arms reuse the ENS-horizons page's conversion rule (`ens_forecast_horizons`). The new arms use this study's rows and folds (rows from 2024-12-01, after the IFS 49r1 change), so the ENS-horizons page's older ENS numbers, which span that change (issue #892), are not comparable, and the page says so.
3. **Two lead figures overlap** (multi-lead rows and by-lead-day lines). Recommendation: keep both; the rows answer absolute skill, the lines answer bracket evidence.
4. **A seventh colour does not exist**, so solar's ARPEGE line is black and dashed. Recommendation: accept.
5. **The shared methods page and the file move** (`studies/forecasts/matched-lead.md`) belong to "Past weather: Solar". This branch does not touch `mkdocs.yml`; if that PR merges first, this branch rebases and fixes the asset paths.

## Departures from this plan, as built

- **Row labels "day 5: not fitted" on the leaderboard** were not drawn; a subtitle sentence says that a lead with no mark was not fitted.
- **The rows that keep the second setting** are not listed on the page; the page states the rule and the two near-the-line results that have no second-setting fit.
- **The V1, V1b, and V3 checks** sit under "How the leads are matched", not under a heading of their own.
- **Results order:** error by lead comes second, and the UKV eras have their own heading.
- **Byte-identical regeneration of the published `report.md`** was not confirmed in this branch; the published losses and report were not touched.
