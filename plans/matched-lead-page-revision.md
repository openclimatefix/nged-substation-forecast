# Revise the matched-lead NWP study page (#912)

## The problem

The page `docs/studies/nwp-forecasts-at-matched-leads.md` (1,145 lines, merged in PR #901 for issue #810) has four faults the maintainer named. Its headings lean on how the leads are matched, which is method, so a reader meets methods inside "Results" and the answer late. Its most useful figures (day-1 absolute error against climatology, Figures 5 and 6) sit fifth and sixth, behind the planned contrasts, so a reader sees gaps between products before seeing how good any of them is. Figures 11 and 12 draw two colours for eight forecast products, so a reader cannot tell the lines apart. The second XGBoost setting appears beside every contrast, including exploratory ones, which doubles the numbers a reader must take in.

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
| ENS mean | 0, 1, 2, 3 | 5, 7, 10, 14 exist in the ENS store; the ENS-horizons page fits ENS at those days under a different design |
| GEFS mean | 1, 2, 3 | 5 and 14 exist in the GEFS store (16 days) |
| IFS 0.25° | 1, 2, 3 | previous_day4 to previous_day7 hold data (93% of hours have shortwave radiation) |
| GFS | 1, 2, 3 | previous_day4 to previous_day7 hold data (about 98%) |
| ICON global | 1, 2, 3 | previous_day4 to previous_day6 hold data (about 98%); day 7 is empty |
| ICON-EU | 1, 2, 3 | previous_day4 holds data (97%); day 5 and later are empty (0%), so ICON-EU ends at day 4 |
| ARPEGE (solar only) | 1, 2, 3 | its forecast is shorter than 5 days; day 5 is not checked and is left absent |
| UKV, ICON-D2, AROME, DMI and KNMI HARMONIE-AROME | 1 | Open-Meteo's Previous Runs fills only `_previous_day1` for UKV and ICON-D2 (UKV 75%); the others' day 2 and later are not built |

So the maintainer's five leads 0, 1, 3, 5 and 14 are all fitted only for ENS at 0, 1 and 3. Day 5 needs a refit for ENS, GEFS, IFS 0.25°, GFS and ICON global. Day 14 needs one for ENS and GEFS only, because the Previous Runs archive stops at day 7. ICON-EU, UKV and ICON-D2 cannot be shown at day 5, and the figure marks that absence. Nothing is interpolated.

**Cost of the refit (estimate; the runner request to the coordinator gives the final figure).** New arms are exploratory, so they are fitted at the primary setting only. Solar needs day 5 for 5 products and day 14 for 2 (7 arms), and wind the same (7 arms), against 45 and 44 arms in the saved run. Each arm is 3 fitting seeds, per generator, per fold, at one setting. A build step (`build_forecast_inputs.py`) must first add ENS at days 5 and 14, GEFS at days 5 and 14, and the three Previous Runs days from data already on disk; it downloads nothing. The ENS day-5 and day-14 bands need the same 3-hour to hourly conversion the day 0 to 3 bands use. The rows, folds, eras and fold rotation stay those of the published run, so the new arms are scored on exactly the shared rows (an arm's few missing values stay missing, as for the other exploratory arms).

## Figure design (recommended)

**Recommendation: one figure per technology, products as rows, lead days as coloured marks on each row, plus a table that gives every number.** Rows sort by day-1 error, best first. Each row carries up to seven marks, one per fitted lead day (0, 1, 2, 3, 5, 14), each with its 95% interval as a short line; the marks are dodged vertically inside the row so intervals do not overprint. Lead day is the colour, so Figure 1's key is the lead days (validated for colour-blind separation), and the product names sit on the axis. A lead day a product does not have is drawn as nothing, and the row's right-hand label says "day 5: not fitted" or "day 5: no forecast that long" where the reason is known. The climatology and smart-persistence baselines sit as reference lines through the plot, not as rows.

Trade-offs against the alternatives:

- **Lead days as colour on product rows (recommended)** answers "how good is each product" first, which is what the maintainer called the headline, and shows the rise with lead in one glance. Its weakness is that colour must carry lead, so products cannot also be coloured; the product name on the axis does that job.
- **Lines with lead day on the x axis, one colour per product (today's Figures 11 and 12, recoloured)** show the slope with lead best and hold the ENS bracket shading. Its weakness is seven products of similar level at day 1, which overprint. This design stays as the second lead figure, because the ENS bracket shading is the page's evidence that bracketed products lose.
- **One panel per lead day** is easy to read and repeats every product name five times, so the page grows by four charts. Rejected.

Colours for the lead-day figure and for the recoloured lines are validated with `validate_palette.py` (all-pairs, light mode). The best six-colour set found from the brand colours and the maintainer-approved extra colours (Data Blue, Data Sky, Data Green, Data Amber, Data Deep Teal, Data Burnt Orange) has a worst all-pairs colour-blind distance of 10.5 and a worst normal-vision distance of 19.5, and fails only the lightness-band and surface-contrast checks that direct labels cover. No seventh colour passes. Wind's lines have six products, so each gets one colour. Solar's have seven; ARPEGE, which is in the solar chart only, is drawn grey and dashed with a direct label. The lead-day figure's colour set is validated separately, in the chart-design step.

## Page structure (agreed with the "Past weather: Solar" session)

Title; summary (two paragraphs at most) with the headline figures directly under it; disclaimer; key findings; introduction; data and methods; results; discussion (what to use); limitations; scope; data and code availability; reproducing. The page already has scope, data and code availability, and reproducing. Moves, all verbatim where possible:

- **Summary:** shortened to two paragraphs (headline verdicts, the setting and interval convention stated once). The long definitions in today's opening paragraph (what a difference means, how the products are read) move to Data and methods and the Introduction table. Figures 1 and 2 become the day-1 absolute-error figures (today's 5 and 6, extended to several leads), Figures 3 and 4 the planned contrasts (today's 1 and 2).
- **Introduction:** the question, the product table, the three ways the products are read (moved from the summary), what ENS is here.
- **Data and methods:** everything under today's "How the leads are matched" (the lead rule, the bracket, the three verdicts, the planned/exploratory rule, the ENS-monotonicity test), the XGBoost model, scored hours and folds. The subsection headings are methods headings, not findings. The UKV radiation rebuild moves here from the lead section. Results sections keep only what the results show.
- **Results:** "Do the XGBoost power forecasts work?" first (Figures 5 to 8 today), then absolute error (the multi-lead figure), the planned contrasts, solar per generator, wind per generator, blending, ensemble averaging and timing, other products, error by lead day, the UKV eras. "Other products (exploratory)" keeps the second-setting rule (primary only).
- **Discussion (what to use)**, **What this study cannot separate** (kept as a Limitations subsection), **Limitations**, **Scope**, **Data and code availability**, **Reproducing**: unchanged in content.

Headings that state a result stay result-phrased; headings that name a method (the bracket, the lead rule) move under Data and methods.

## Second XGBoost setting

The rule: keep the second setting for planned contrasts, for deciding contrasts (P4b and the guards) and for any result near the 5% line; drop it for exploratory arms; one table or a marker, never doubled charts; verdicts still need both settings to agree. "Near the 5% line" is the definition the solar session writes on the shared past-weather methods page: an interval bound within 20% of the interval's width from zero. On this page:

- Kept: the planned-contrast tables (P1 to P4b, guards) and the deciding-contrast paragraphs.
- Dropped unless near the line: the exploratory rows of the blending table (control minus ENS, IFS 0.25° minus ENS), and any other exploratory sensitivity figure. Each exploratory row is checked against the definition; a script prints which rows qualify, so the page states the rule and lists the rows that stay.
- The headline charts keep the primary setting as the dot and the second setting as a marker, since they show only planned contrasts. Figure 1 (leaderboard) is primary only.

## ECMWF AIFS (added by the maintainer after the plan was committed)

The maintainer wants ECMWF's AIFS (its machine-learning weather model), the ensemble and the single run, in the page. No AIFS data is on disk. The download coordinator is fetching it and will report the path, cadence, lead range and start date. The plan adds AIFS as new arms once that message arrives: if the archive window is shorter than the study's rows, AIFS gets its own row set with the references (ENS day 1, and the arms it is compared with) refitted on that row set; the era rule is the study's own (rows from 2024-12-01, three eras, the rotated fold offsets); planned and exploratory are labelled (no AIFS contrast was written into the plan before a result existed, so every AIFS number is exploratory and post hoc); the second setting is shown only if a result is near the 5% line. The AIFS arms go into the same runner request as the day 5 and day 14 arms.

## What changes, file by file

PR 1 (`study-skill-page-structure`):

- `.claude/skills/study/SKILL.md`: "Writing the page" gains the twelve-part outline (Summary carries scoped take-home bullets and is the only place a recommendation appears without its evidence; Discussion does not repeat them; Data and methods holds only what is specific to the study and links the shared methods page where one exists; new study pages go under Studies > Past weather or Studies > Forecasts); the second-setting rule in "Design rules" is replaced; the chart section says the first figure of a ranking page is the absolute-error figure and the planned contrasts follow. The line in the `CLAUDE.md` skills table for `study` is checked, and changed only if it now misdescribes the skill.

PR 2 (this branch):

- `docs/studies/nwp-forecasts-at-matched-leads.md`: restructured as above.
- `studies/nwp_forecast_comparison/nwp_forecast_charts.py`: `by_lead_day` colours each product (one colour per product from the validated set); a new `leaderboard_by_lead` figure (products as rows, lead-day marks, absent leads left blank) with its number and title in `FIGURE_NUMBERS` and `TITLES`, and the existing figure numbers renumbered; a check that every SVG title's figure number equals its caption's number. `SERIES_COLOURS` and `PREVIOUS_RUNS_SERIES` are replaced by a per-product map.
- `studies/nwp_forecast_comparison/build_forecast_inputs.py` and `nwp_forecast_comparison.py`: only if the refit runs (after the runner slot): ENS and GEFS at days 5 and 14, Previous Runs at day 5, in a new arm set fitted at the primary setting, written to its own new folder. The existing `report.md` and losses are not overwritten.
- `studies/nwp_forecast_comparison/README.md`: the new folder and what each file holds.
- Inbound anchors: `docs/studies/index.md`, `docs/background/weather-products-survey.md`, `docs/roadmap/*` and the study README and docstrings are grepped for the page's anchors; every renamed heading updates its inbound links in the same commit.
- `packages/studies/`: unchanged. Chart code for this one page stays in the study's chart script, as today; nothing is shared with another study yet.

## Design-philosophy check

R&D and documentation only; nothing runs in production, so nothing degrades. The study fails fast (the build stops on missing runs). No asset check is added. No hypothesis (`H1`, `T1.2`) is claimed. The Discussion states what the evidence supports and does not commit the project to work.

## Tests

Study scripts have no unit tests; the check is their own output. The gates are:

- **Number conservation:** every decimal, and every integer of 10 or more, on the revised page appears in `git show origin/main:docs/studies/nwp-forecasts-at-matched-leads.md`, in `data/studies/nwp_forecast_comparison/report.md`, or in the new leads report. This fails if a sentence is rewritten with a number the report does not hold. Blind spots (a number attached to the wrong row; a deleted qualifier) go in the PR body.
- **SVG title against caption:** for every figure, the figure number in the SVG title equals the number in the caption, and the grep for `Figure \d+` inside SVG text matches `FIGURE_NUMBERS`.
- **Second-setting rule:** every exploratory row that still shows a second setting is checked by hand against the near-the-line definition (about five places on the page); the check is not committed.
- **Lead absence:** the figure script draws a mark only for an arm present in the saved losses, so an absent lead is blank by construction and nothing is filled.
- **Docs gates:** `uv run mkdocs build --strict` with the rendered HTML read, the docs-link check, `uv run pre-commit run --all-files`, and `pymarkdown scan`.
- **Refit check (if it runs):** the shared-row count in the new losses equals the published run's shared rows, and each new arm's per-row loss frame joins the published ENS day-1 losses on `(site, time, seed)` with no missing key.

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

1. **Day 5 and day 14 are new fits on a published study.** Recommendation: run them, primary setting only, exploratory, in a new folder, because the maintainer asked for them and the published run stays untouched. If the runner slot is late, the page ships with days 0 to 3 and the absent leads marked, and a later PR adds the rest.
2. **ENS at day 14 is a different quantity from ENS at day 1** (3-hour steps become 6-hour steps beyond 144 hours, so the hourly conversion differs). The ENS-horizons page settled the conversion; the new arms reuse its rule, and the caption says so. Open: whether the ENS-horizons session's era-cut fix (issue #892) moves these numbers; the refit waits for that session's answer or says on the page that it does not depend on it.
3. **Two lead figures overlap** (multi-lead rows and by-lead-day lines). Recommendation: keep both; the rows answer absolute skill, the lines answer bracket evidence.
4. **A seventh colour does not exist**, so solar's ARPEGE line is grey and dashed. Recommendation: accept.
5. **The shared methods page and the file move** (`studies/forecasts/matched-lead.md`) belong to "Past weather: Solar". This branch does not touch `mkdocs.yml`; if that PR merges first, this branch rebases and fixes the asset paths.
