# Plan: WeatherNext 3 and AIFS on the matched-lead page (#974)

**The problem.** Figures 1 and 2 of `docs/studies/forecasts/matched-lead.md` are the headline lead-day leaderboards for solar and wind. They omit AIFS Single, the AIFS ENS mean, and WeatherNext 3 (WN3). The page has no WN3 arm at all, because the WN3 data (#934) did not exist when the page was written. The AIFS arms exist on disk but live on shorter row sets than the other products, so putting them in the same figure without marking that would mislead. The figures also lack the vertical grid lines a reader needs to compare marks across rows.

**The plan.** Read the trial-area subset of the WN3 Icechunk store once, copy it to a workstation folder, and build and fit WN3 arms by adding a `wn3` row set to `fit_aifs.py` (own row set, reference arms refitted on that row set, write-once folders). Add AIFS Single, the AIFS ENS mean, and WN3 to the leaderboard chart function in `studies/nwp_forecast_comparison/nwp_forecast_charts.py`, mark each product's row set on the mark and in the name, add minor vertical grid lines every 0.5 points, and redraw Figures 1 and 2 once. Add a WN3 section (methods, results, limitations, reproducing block) to the page, with absolute error beside every contrast.

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

**Departures from the issue body.** None on scope. One addition the issue only implies: the ENS mean refitted on the WN3 row set, as the same-rows reference. The plan fits no new AIFS lead days: the issue allows drawing each product at the days it has, and the page will say that AIFS Single and the AIFS ENS mean have marks at days 1, 2, 7, and 14 only.

## What is known about the inputs

**Which AIFS series exist on disk (checked in the saved losses, no network).** `nwp_forecast_comparison_aifs_blends` holds `aifs_single_day{1,2,7,14}` on the `single` row set (16 months: 28,570 solar rows at days 1 and 2) and `aifs_ens_mean_day{1,2,7,14}` on the `ens` row set (11 months: 16,911 solar rows at days 1 and 2). AIFS ENS's mean therefore exists, and is fitted at days 1, 2, 7, and 14. Neither AIFS arm has a mark at days 0, 3, 5, or 10 today, and neither has day 0 at all. The ENS mean reference on each row set exists at days 1, 2, 7, and 14 (`ens_mean_day<N>`), in the same folders. The `ens` row set is a subset of `single`.

**Which lead days WN3 can have.** The store holds runs at 00, 06, 12, and 18 UTC with 360 hourly leads, so day N for N up to 14 can follow the ENS convention (the 00 UTC run issued N days before the hour's own day). Runs start on 2026-01-01, so the WN3 row set holds about 9 months of 2026 hours, and the era rules (the 2026-01-21 UKV upgrade, IFS Cycle 50r1 on 2026-05-12) apply.

**Which scripts produce Figures 1 and 2.** `nwp_forecast_charts.py` `leaderboard_figure` draws them, fed by `lead_board_rows` and `lead_board_products` (arms named `<slug>_day<N>` whose slug is in `PRODUCT_NAMES`), with `leaderboard` from `nwp_forecast_comparison.py` computing each arm's error and month-resampled interval on that arm's own rows. `load` merges the published folder with the extra-lead folders (`--extra-dir`, GPU refits preferred for the marks). The ordering is by day-1 error, so every product needs a day-1 row. The figure's x grid is drawn by the `grid` layer from `lead_board_x_domain`'s ticks. The reproducing block on the page lists the exact command (`nwp_forecast_charts.py --input-dir $P --extra-dir ... --aifs-dir $A --output-dir docs/studies/assets`).

**Existing WN3 code.** `studies/weather_downloads/fetch_weathernext3.py` (writer) and `validate_weathernext3.py` (validator) describe the store: seven `float32` arrays, dimensions `(init_time, lead_time, latitude, longitude)`, native units (J m-2 for the two 1-hour-mean radiation variables, m s-1 for wind), values rounded to 13 significand bits. No reader for the trial area exists yet.

## Reading the trial-area subset, and the egress estimate

**Chunk shape, from the #934 egress comment (Icechunk metadata, no bulk read).** Shape `(5844, 360, 126, 136)`. One shard holds one run: `(1, 360, 128, 136)`. Inner chunks are `(1, 360, 8, 8)`, so one inner chunk holds all 360 leads for an 8 by 8 block of 0.1 degree cells, and there are 272 per run and variable. The trial-area box touches 4 of the 272 (1.5%). The lead axis cannot be subset by fewer bytes, the spatial axes can.

**What is read: the 00 UTC runs only, and only the variables the arms use.** Every existing lead in the page (ENS, GEFS, AIFS) reads the 00 UTC run issued N days before the hour's own day, and `fit_aifs.py` states that only 00 UTC runs are read. WN3 follows that convention, so the read takes the roughly 270 runs at 00 UTC, not all 1081. The AIFS-shaped arm has 2 weather columns for solar and 4 for wind, so the variables are total solar radiation, direct solar radiation, and the 100 m u and v wind components: 4 of the 7. The implementer confirms the list against `arm_columns` before the read.

**Estimate.** Measured store size: about 36 KB per inner chunk on average. Four inner chunks, 4 variables, and about 270 runs is about 160 MB, plus 76 MB of manifests and a small shard index per run. Even the earlier upper bound, all 7 variables and all 1081 runs, was about 1.1 GB. At the $0.12 per GB internet egress rate quoted in `fetch_weathernext3.py`, 1.1 GB costs about $0.13, about £0.10. That rate is unverified for this bucket's destination, so the implementer looks it up before the first read. **Both figures are far below the £30 stop-and-ask threshold, so the estimate needs no approval.** No read happens until the maintainer says #934 validation passed. The £30 rule stays: if a measured total approaches £30, stop and ask. There is no byte counter or abort logic.

**How to read.** The trial-area reader lives in `studies/nwp_forecast_comparison/build_forecast_inputs.py`, as the `--wn3` build's input step, and reads from the store straight into the WN3 output folder's `_grid_cells.parquet` and a local zarr under `data/studies/weather/WeatherNext3_trial_area/`. It opens the store read-only on `main`. The box comes from the private generator roster at run time and is never printed or written into a committed file. Every later step reads the local copy. The only check is that no copied variable is all-NaN, because a missing chunk reads as NaN.

## What changes, file by file

1. **`studies/nwp_forecast_comparison/fit_aifs.py`: add a `wn3` row set, and choose this file over `fit_extra_leads.py`.** `fit_aifs.py` already has what WN3 needs: `ROW_SETS` (`RowSet`: first month, eras, fold offsets, era runs, arms, deciding contrast), reference arms refitted on the row set, the permuted-weather negative control (`PERMUTED_PREFIX`), and per-row-set reports. `fit_extra_leads.py` fits arms on the shared published rows and has none of that. The new row set is `wn3`, with `first_month` 2026-01, its eras from the era rules that apply in 2026 (the 2026-01-21 UKV upgrade and IFS Cycle 50r1 on 2026-05-12, which the ENS reference arm crosses), and `deciding=("wn3_mean_day1", "ens_mean_day1")`. The implementer sets `fold_offsets` with `search_fold_offsets`, as for the other row sets. Arms: `wn3_mean_day{1,2,7,14}`, the same-rows `ens_mean_day{1,2,7,14}`, and the permuted controls. One fit loop stays.
2. **`studies/nwp_forecast_comparison/build_forecast_inputs.py`.** Add `--wn3`, following `--aifs`: the read above, and `wn3_mean_day<N>` columns with the same-rows ENS columns, into a fresh write-once folder. Radiation `1hr_mean` fields are already hour means, converted from J m-2 to W m-2. `aifs_site_weights` and `_h3_crop_weights` take the grid size from the module constant `AIFS_CROP_DEGREES` (0.25 degrees), and WN3's grid is 0.1 degrees, so the grid size becomes one keyword argument with the current value as its default. Nothing else in these functions changes.
3. **`studies/nwp_forecast_comparison/nwp_forecast_charts.py`.**
   - Add `aifs_single`, `aifs_ens_mean`, and `wn3_mean` to `PRODUCT_NAMES` and load the two AIFS folders and the WN3 folder into `leaderboard_losses`. `lead_board_rows` gains a `row_set` column.
   - The product name carries its row set in words: "AIFS Single (16 months)", "AIFS ENS mean (11 months)", "WeatherNext 3 mean (9 months)". Each caption states the months.
   - **Keep the same-rows ENS mean reference, drawn as a small grey tick beside each mark of a smaller row set.** The simple version (a suffix plus caption numbers) would mislead: the marks sit in rows of full-window products, and the reader compares errors across rows, but solar and wind error depend on the season, and the WN3 rows are mostly spring and summer. A caption cannot hold one same-rows ENS mean for each of 3 row sets and 4 lead days per figure. The ticks come from `ens_mean_day<N>` arms already in each folder, so they need no new fit.
   - **Minor vertical grid lines at every 0.5 points**, as a second, lighter `grid` layer inside `leaderboard_figure`, at multiples of 0.5 in the x domain.
   - **Device.** Every existing leaderboard mark is a GPU fit: `fit_extra_leads.py` and `fit_aifs.py` both set `DEVICE = "cuda"`, and `check_single_device` requires one device per mark. The new arms use `fit_aifs.py`, so they are GPU fits, and the figure's marks and every WN3 contrast (the arm and its same-rows references are fitted in one run) share one device. Contrasts stay honest, `check_single_device` passes unchanged, and no per-arm device marker or extra noise-floor caption is needed. This departs from the CPU instruction in the task file, on the coordinator's direction and on the evidence that a GPU wind fit sits 0.04 to 0.09 points below a CPU fit.
4. **Output folder (write-once).** `data/studies/nwp_forecast_comparison_wn3/`, written only when the maintainer gives the runner slot, after one Opus review of the scripts. No existing folder is written to. No new AIFS folder.
5. **`docs/studies/forecasts/matched-lead.md`, and the two SVGs.** Text below. The chart script redraws both leaderboard SVGs once and runs `svgo`.

## The page change (kept minimal)

- **One method paragraph** beside "How AIFS is read": which store, the ensemble mean at 0.1 degrees, the 00 UTC runs and the lead convention, the H3 spatial read, the row set (2026-01-01 onward, about 9 months), and the device.
- **One results table** of absolute error and its interval for `wn3_mean` at each fitted day, beside the ENS mean fitted on the same rows, followed by the planned contrast: WN3's mean minus ENS's mean at day 1 on the WN3 rows, for solar and wind, with its negative control (WN3's weather shuffled within generator, year-month, and hour of day) and the sensitivity setting. Every other number is exploratory and labelled so. Every number is printed into `report.md` by the committed script and checked against it by eye and by the existing report-reading habit. No new number-checker script.
- **Figures 1 and 2 captions, and the one limitation sentence** (shorter and different row set, CPU-and-GPU sentence unchanged because the marks are still all GPU fits). The page says that AIFS Single and the AIFS ENS mean have marks at days 1, 2, 7, and 14 only. Headline sentences that name the lowest day-1 error are rechecked against the redrawn figure.
- **Reproducing block:** the WN3 build and fit commands and the changed chart command.

## Design-philosophy check

R&D code: it fails fast. No asset check or serving path is touched. No hypothesis is delivered. No principle is traded away.

## Tests

Study scripts carry no unit tests, so each check is the script's own output. The build prints every arm's column list and fails if a WN3 arm's column count differs from the ENS arm's on the same rows (`expected_column_count`). The read fails if a copied variable is all-NaN. `fit_aifs.py --check` fits one arm twice and stops unless the two agree. The chart script prints the rows behind the figures into the report. No helper moves into `packages/studies/`, so no mutation pass runs; the only shared-code edit is the grid-size keyword argument in `build_forecast_inputs.py`, and the AIFS build must reproduce its existing inputs bit for bit (the existing `check_equal_to_existing` pattern).

## Verification commands

`uv run ruff check .`, `uv run ruff format .`, `uv run ty check`, `uv run pytest`, the markdown lint command, the pydoclint and docs-link checks CI runs, `uv run mkdocs build --strict` with a read of the rendered page, and a look at the two rendered SVGs.

## Risks and open questions

**Decided by the coordinator, not open:** the runner slot comes from the coordinator after one Opus review of the scripts, before the run. After results, the study skill's two Opus science reviews run, then the diff review and a `prose-review`.

1. **Store readiness.** No read until the maintainer says #934 validation passed.
2. **Egress price.** The $0.12 per GB rate is unverified for this bucket's destination. Confirm before the first read.
3. **Row-set comparability.** Marks on different row sets are comparable only through the same-rows ENS mean tick, and the caption says so.
4. **Note, not planned: one shared fit module.** The simplicity review observed that `fit_aifs.py` (3,068 lines) and `fit_extra_leads.py` (1,374 lines) each carry a copy of the fit loop, and one shared fit module would remove the duplication. That refactor is larger than this issue, would need bit-for-bit reproduction of about five million per-row losses, and belongs in its own issue. This plan adds a row set to `fit_aifs.py` and leaves the duplication.

## Reviews and what they changed

**Simplicity review triage.** Accepted: no new AIFS lead days (finding 1); no `fit_wn3.py`, a `wn3` row set in `fit_aifs.py` instead (2); no byte counter or abort logic (5); read only the 00 UTC runs and 4 of 7 variables (6); no new checker script and a minimal page change (7). Partly accepted: `aifs_site_weights` gains one grid-size argument because the WN3 grid (0.1 degrees) differs from the 0.25 degree AIFS grid (8). Verified and changed: the device (3), since every existing mark is a GPU fit, the new arms are GPU fits too, and the per-arm device marker is dropped. Kept after checking: the same-rows ENS mean ticks (4), for the reason in the charts item. Recorded as a note, not done: the shared fit module.
