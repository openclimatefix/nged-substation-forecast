# Plan: WeatherNext 3 and AIFS on the matched-lead page (#974)

**The problem.** Figures 1 and 2 of `docs/studies/forecasts/matched-lead.md` are the headline lead-day leaderboards for solar and wind. They omit AIFS Single, the AIFS ENS mean, and WeatherNext 3 (WN3). The page has no WN3 arm at all, because the WN3 data (#934) did not exist when the page was written. The AIFS arms exist on disk but live on shorter row sets than the other products, so putting them in the same figure without marking that would mislead. The figures also lack the vertical grid lines a reader needs to compare marks across rows.

**The plan.** Read the trial-area subset of the WN3 Icechunk store once, copy it to a workstation folder, and build and fit WN3 arms with the same machinery `fit_aifs.py` uses for AIFS (own row set, reference arms refitted on that row set, GPU-free, write-once folders). Add AIFS Single, the AIFS ENS mean, and WN3 to the leaderboard chart function in `studies/nwp_forecast_comparison/nwp_forecast_charts.py`, mark each product's row set on the mark and in the name, add minor vertical grid lines every 0.5 points, and redraw Figures 1 and 2 once. Add a WN3 section (methods, results, limitations, reproducing block) to the page, with absolute error beside every contrast.

## Verdict, size and departures

**Verdict: worth doing, roughly as described.** Nothing in the issue is stale: #970 is closed (PR #986 merged) and the page still lacks WN3. No open PR or branch touches #974.

**Preconditions still open at the time of writing.** #934 is not finished: `main` of the WN3 store is unpublished, and the parent has not said validation passed. The plan therefore does no read of the store. Every read step below waits for the parent's go-ahead.

**Size: complex, by the study skill's rule that a published page counts as what gets stored.** The five triggers:

1. **What gets stored:** fires. Two or three new write-once folders under `data/studies/`, a workstation copy of the WN3 trial-area subset, and a published page and two redrawn figures. No Patito model, Delta table, or asset changes.
2. **Production serving path:** does not fire. Only `studies/` and `docs/studies/` change. `packages/studies/` changes only if a helper moves there (see risks), and the mutation pass then runs.
3. **Degradation rule:** does not fire. Study code is R&D and fails fast.
4. **More than one defensible design:** fires. How to mark unequal row sets on the figure, which lead days to fit, and whether to fit new AIFS lead days each have more than one defensible answer.
5. **Callers not nameable without searching:** does not fire for the charts (one entry point, `nwp_forecast_charts.py:main`), but `PRODUCT_NAMES` and `PRODUCT_SLUGS` are read by several scripts, so the implementer greps them before editing.

Any trigger firing makes the issue complex. **Reviews bought:** both plan reviews (simplicity, then correctness). For the diff, the study skill requires two Opus scientific-validity reviews, the `implement-issue` diff review, and a `prose-review` of the page. The issue asks for one Opus diff review and a second only if a headline claim changes. This plan follows the study skill and asks the parent to confirm (see open questions).

**Departures from the issue body.** None on scope. Two additions the issue only implies: an ENS mean reference arm refitted on each new row set (needed so the figure can show absolute error beside a same-rows reference), and AIFS fits at extra lead days, if the local downloads cover them.

## What is known about the inputs

**Which AIFS series exist on disk (checked in the saved losses, no network).** `nwp_forecast_comparison_aifs_blends` holds `aifs_single_day{1,2,7,14}` on the `single` row set (16 months: 28,570 solar rows at days 1 and 2) and `aifs_ens_mean_day{1,2,7,14}` on the `ens` row set (11 months: 16,911 solar rows at days 1 and 2). AIFS ENS's mean therefore exists, and is fitted at days 1, 2, 7, and 14. Neither AIFS arm has a mark at days 0, 3, 5, or 10 today, and neither has day 0 at all. The ENS mean reference on each row set exists at days 1, 2, 7, and 14 (`ens_mean_day<N>`), in the same folders. The `ens` row set is a subset of `single`.

**Which lead days WN3 can have.** The store holds runs at 00, 06, 12, and 18 UTC with 360 hourly leads, so day N for N up to 14 can follow the ENS convention (the 00 UTC run issued N days before the hour's own day). Runs start on 2026-01-01, so the WN3 row set holds about 9 months of 2026 hours, and the era rules (the 2026-01-21 UKV upgrade, IFS Cycle 50r1 on 2026-05-12) apply.

**Which scripts produce Figures 1 and 2.** `nwp_forecast_charts.py` `leaderboard_figure` draws them, fed by `lead_board_rows` and `lead_board_products` (arms named `<slug>_day<N>` whose slug is in `PRODUCT_NAMES`), with `leaderboard` from `nwp_forecast_comparison.py` computing each arm's error and month-resampled interval on that arm's own rows. `load` merges the published folder with the extra-lead folders (`--extra-dir`, GPU refits preferred for the marks). The ordering is by day-1 error, so every product needs a day-1 row. The figure's x grid is drawn by the `grid` layer from `lead_board_x_domain`'s ticks. The reproducing block on the page lists the exact command (`nwp_forecast_charts.py --input-dir $P --extra-dir ... --aifs-dir $A --output-dir docs/studies/assets`).

**Existing WN3 code.** `studies/weather_downloads/fetch_weathernext3.py` (writer) and `validate_weathernext3.py` (validator) describe the store: seven `float32` arrays, dimensions `(init_time, lead_time, latitude, longitude)`, native units (J m-2 for the two 1-hour-mean radiation variables, m s-1 for wind), values rounded to 13 significand bits. No reader for the trial area exists yet.

## Reading the trial-area subset, and the egress estimate

**Chunk shape, from the #934 egress comment (Icechunk metadata, no bulk read).** Shape `(5844, 360, 126, 136)`. One shard holds one run: `(1, 360, 128, 136)`. Inner chunks are `(1, 360, 8, 8)`, so one inner chunk holds all 360 leads for an 8 by 8 block of 0.1 degree cells, and there are 272 per run and variable. The trial-area box touches 4 of the 272 (1.5%). The lead axis cannot be subset by fewer bytes, the spatial axes can.

**Estimate.** Measured store size: about 70 MB per run, 10 MB per variable and run, 36 KB per inner chunk on average. Four inner chunks times 1081 runs is about 160 MB per variable, and about 1.1 GB for all 7 variables, plus 76 MB of manifests and a small shard index per run. At the $0.12 per GB internet egress rate the fetch script's docstring quotes for us-east1, 1.2 GB costs about $0.15, or about £0.11. That rate is unverified for this bucket's destination, so the implementer looks it up before the first read. Even the whole 75 GB store at $0.12 per GB is about £7. **The estimate is far below the £30 stop-and-ask threshold, so no approval is needed on the estimate.** The plan still reads nothing until the parent says validation passed and the read may start.

**How to read.** One script, `studies/weather_downloads/copy_weathernext3_trial_area.py`, opens the store read-only on `main` (after publish; on `staging` only if the parent says so), reads only the trial-area box widened to whole inner chunks, all 1081 runs and all 7 variables, and writes a local Zarr under `data/studies/weather/WeatherNext3_trial_area/`. The box comes from the private generator roster at run time and is never printed or written into a committed file. The script counts bytes read, prints a running total and the estimated cost, and aborts above 5 GB (about £0.45), a tenth of the way to the cap, rather than the £30 cap itself. It runs once. Every later step reads the local copy. If the measured bytes exceed the estimate by 2 times, the script stops and asks the parent.

**Runs on the VM are free but not needed.** A read on the VM saves about £0.11, but then the subset has to be moved to the workstation anyway, which is the same egress. Reading from the workstation is simpler.

## What changes, file by file

1. **`studies/weather_downloads/copy_weathernext3_trial_area.py` (new).** Described above. Includes a check that no variable is all-NaN, that init times cover the stored runs, and that the copied values equal a re-read of one run's chunk.
2. **`studies/nwp_forecast_comparison/build_forecast_inputs.py`.** Add `--wn3`, following `--aifs`: build `wn3_mean_day<N>` columns, and `ens_mean_day<N>` (and `ens_mean6_day<N>` where a 6-hourly reference is needed) restricted to the WN3 row set, into a fresh write-once folder. Spatial read: the same H3 resolution-5 overlap-weighted mean the AIFS arms use (`aifs_site_weights` generalised to a 0.1 degree grid, so WN3 and AIFS and ENS are read the same way). Radiation: the WN3 `1hr_mean` fields are already hour means, converted from J m-2 to W m-2; wind speed and direction come from the u and v components at 100 m. The direct-radiation field is used only if the ENS arm's column set needs it, so that every arm has the same column count. Print every arm's column list into the report.
3. **`studies/nwp_forecast_comparison/fit_wn3.py` (new), reusing `fit_aifs.py`'s fit loop.** Fits `wn3_mean_day<N>` and the same-rows references `ens_mean_day<N>` and climatology on the WN3 row set, at the primary setting on the CPU (`device="cpu"`), plus the sensitivity setting for the day-1 contrast. If lifting the loop out of `fit_aifs.py` is bigger than copying the 30 lines that differ, the implementer imports the shared helpers `fit_aifs.py` already uses from `packages/studies`. No new abstraction.
4. **(Conditional) AIFS extra lead days.** If the local `ECMWF-AIFS` and `ECMWF-AIFS-ENS` downloads hold leads for days 0, 3, 5, and 10 (checked from the saved zarr metadata with no network), `build_forecast_inputs.py --aifs --aifs-days 0 3 5 10` and `fit_aifs.py` fit them into a fresh folder, on the CPU, with the ENS mean and the ENS control reference on the same rows. This is optional in the issue. The plan recommends it only for the days that give the AIFS marks a continuous line from day 1 to day 14 in the figure, and skips any day whose download is missing.
5. **`studies/nwp_forecast_comparison/nwp_forecast_charts.py`.**
   - Add `aifs_single`, `aifs_ens_mean`, and `wn3_mean` to `PRODUCT_NAMES` (names: "AIFS Single", "AIFS ENS mean", "WeatherNext 3 mean"), and `--wn3-dir` and the AIFS blends folder to `load` so `leaderboard_losses` holds their arms. Each new arm keeps its own row set in a `row_set` column.
   - `lead_board_rows` gains a `row_set` column (`shared`, `single`, `ens`, `wn3`). Products on a smaller row set draw hollow-ringed marks. The product's name carries the row set in words, for example "AIFS Single (16 months)", "AIFS ENS mean (11 months)", and "WeatherNext 3 mean (9 months)". Each caption states the rows in each set.
   - **A same-rows reference mark for each smaller row set.** The ENS mean fitted on the same rows as the new product appears as a small grey tick beside the product's mark at the same lead day, so the reader can see that the product's error is comparable with the ENS mean's on those months, and not with the ENS mean row's own full-window value. This answers the issue's open question on helping the reader with different row sets. Recommendation: use ticks, not extra rows, to avoid a second copy of the same product list.
   - **Minor vertical grid lines at every 0.5 points of mean absolute error**, in `leaderboard_figure`: a second `grid` layer of thinner, lighter rules at `x = 0.5 k` for every multiple of 0.5 inside the x domain, under the existing ticks' grid. Only these two figures change; `lead_board_x_domain` gains no new argument, and the minor values are computed from its returned range.
   - `check_single_device` requires all marks on one device. The Figures 1 and 2 marks are all GPU fits today. New WN3 marks are CPU fits (the task's device rule), and the existing marks are GPU fits, so the check must either allow a documented per-arm device or the new arms are also fitted on the GPU. Recommendation: fit WN3 and the new AIFS days on the CPU as directed, and let the figure carry a device marker in the caption ("the WN3 marks are CPU fits; the GPU-CPU noise floor of the other marks, quoted on the page, is between -0.09 and +0.014 points"). Check the noise floor text against the page before quoting it.
   - Anonymisation: the leaderboard only carries products, but `check_anonymised` keeps running on every loaded frame, and the A to F and W1 to W3 labels apply to any per-generator chart.
6. **Output folders (write-once, never overwritten).** `data/studies/nwp_forecast_comparison_wn3/` (inputs, losses, predictions, `report.md`, `verification/`), and `data/studies/nwp_forecast_comparison_aifs_leads/` only if item 4 runs. The scripts refuse to overwrite. **The parent gives the runner slot before any script writes.** No existing published folder is written to.
7. **`docs/studies/forecasts/matched-lead.md`.** Text below.
8. **`docs/studies/assets/nwp_forecast_solar_leaderboard.svg` and `nwp_forecast_wind_leaderboard.svg`.** Redrawn once, with every new arm, after `npx svgo@4 --multipass --precision=1 --final-newline` (the chart script runs it).

## The WN3 addition to the page

- **Methods.** A new subsection "How WeatherNext 3 is read", beside "How AIFS is read": which store and run (the ensemble mean, 0.1 degrees, four runs a day), the lead convention, the spatial read, the row set (2026-01-01 onward, so a shorter and different row set from the arms that start in 2024), the fold and era handling, the device, and the checks that passed. Lead days are stated only after the build shows which exist.
- **Results.** A subsection reporting each WN3 arm's absolute error and its interval, beside the ENS mean fitted on the same rows, then the contrasts (WN3 minus the ENS mean, at each fitted lead day). **Absolute error comes first, and the contrasts second**, per the study skill. Every number is printed by a committed script into `report.md` and checked against it; no number is written by hand or invented. The contrast list is fixed before any result exists: one planned contrast, WN3's mean minus ENS's mean at day 1 on the WN3 rows, for solar and wind, with the negative control that shuffles WN3's weather within generator, year-month, and hour of day. Every other number is exploratory, and labelled so on the page. The sensitivity setting runs on the planned contrast.
- **Limitations.** The shorter row set, the era boundaries in 2026, the CPU-versus-GPU difference, which `effective_capacity` table the numbers rest on, and that no WN3 lead has been checked against a native archive.
- **Figures 1 and 2 captions and the two "Key findings" and headline paragraphs** are rewritten to describe what the redrawn figures show. Any headline claim that changes triggers the second Opus review. Sentences that say "the ENS mean and IFS 0.25° have the lowest day-1 errors" are re-checked against the new marks.
- **Reproducing block.** New commands for the copy script, `build_forecast_inputs.py --wn3`, `fit_wn3.py`, the optional AIFS lead-day commands, and the changed chart command.
- **Discussion.** Any recommendation about WN3 is scoped to the nine farms and the 9-month row set, and follows from the reports. No claim that a set has one member, no commitment to adopt WN3.

## Design-philosophy check

This is R&D code, so it fails fast (no degradation path applies). No asset check or serving path is touched. Hypotheses: none are delivered by this issue. The issue trades no principle away.

## Tests

Study scripts carry no unit tests; their check is their own printed output, so the checks are in the scripts and the reports:

- The copy script checks each variable is not all-NaN, and that a re-read of one run's four chunks equals the copy. It fails on a store that reads back as NaN (the fill value) for a missing chunk.
- The build prints each arm's column list and a column-count check that fails if any arm's count differs from the ENS arm's on the same rows (the lost-column trap in the study skill).
- `fit_wn3.py --check` fits one arm at one generator twice and stops unless both runs agree.
- The chart script prints the rows behind Figures 1 and 2 (product, day, value, interval, row set) into the report, and the page's numbers are checked against them.
- If any helper moves into `packages/studies/`, it gets tests that would fail on the bug they exist for, and the mutation pass runs. Recommendation: do not move any helper unless two scripts need it.

## Verification commands

`uv run ruff check .`, `uv run ruff format .`, `uv run ty check`, `uv run pytest`, `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md`, the pydoclint and docs-link checkers CI runs, `uv run mkdocs build --strict` and a read of the rendered page, and the study scripts' own `--check` runs. Every page number is checked against the report by a committed script (`check_page_numbers.py`-style), and the two SVGs are rendered and looked at.

## Risks and open questions

1. **Readiness.** The plan reads the store only after the parent says #934 validation passed. Recommendation: wait; the plan needs no read to be reviewed.
2. **Row sets differ between marks, so absolute error is not comparable across rows without the same-rows reference tick.** Recommendation: draw the tick, and word each caption with the row set's months.
3. **The device rule.** Existing Figures 1 and 2 marks are GPU fits, and the task asks for CPU fits for the new arms. Recommendation: fit the new arms on the CPU as asked, allow one device per contrast, and say so in the caption and the limitations. Alternative: refit the one day-1 WN3 arm on the GPU as a noise floor.
4. **Review count.** The issue asks for one Opus diff review, and the study skill asks for two Opus science reviews plus a prose review. Recommendation: follow the study skill, and tell the reader in the PR body which reviews ran.
5. **Runner slot.** No script that writes under `data/studies/` runs before the parent grants the slot.
6. **Optional AIFS lead days.** Recommendation: fit them only if the local downloads already cover them, and skip the rest.
7. **Egress price.** The $0.12 per GB rate is unverified for this bucket's destination. Recommendation: confirm it before the first read; the decision does not change while the estimate stays below £30 by two orders of magnitude.

## Reviews and what they changed

Not yet run.
