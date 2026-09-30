# Plan: CERRA and NORA3 as row sets on the past-wind page (#968)

Issue #968 asks for two new row sets on `docs/studies/past-weather/wind.md`, plus the polish and mutation-testing follow-ups held over from PR #937 and #930. **This plan covers the two row sets.** CERRA wind (50, 75, 100 and 150 m, plus 10 m) and NORA3 wind (50 and 100 m) are already on disk. Each gets its own fitting script, its own refitted reference rows on the fixed folds, a hub-height column, and a domain-coverage row. The leaderboard and the page then gain a fifth and a sixth block.

**The polish and mutation follow-ups ship as a separate, simple PR** (no plan, no science) branched from `main`, because they touch no fit and would otherwise sit behind a long fitting run. They are listed under "Departures".

## Delivery order: three PRs

1. **Shared loader PR (first, from `main`, sized simple).** The Study MAIN COORDINATOR decided that #968 owns the CERRA and NORA3 wind loader and that #957 imports it, so the loader lands on `main` before any fitting or page prose. It is a module in `packages/studies/src/studies/` with tests. It reads the CERRA and NORA3 files; derives each wind farm's nearest cell from `cerra_grid.parquet` (`generator_cells.parquet` holds only solar sites) and asserts each cell is strictly inside the 190-cell crop, not on its edge; pivots the heights (10 m from the single-levels product, 50, 75, 100 and 150 m from the height-levels product, stated in the docstring); and builds the centred power hour. It prints counts and pooled distance ranges, never coordinates or which cell serves which farm. It writes nothing under `data/studies/`.
2. **Polish and mutation-testing PR** (separate, from `main`, simple).
3. **This PR:** the fitting scripts, leaderboard blocks and page prose, which import the loader.

## Verdict, size and departures

- **Verdict:** worth doing as described. CERRA and NORA3 are both regional reanalyses that the page does not yet score, and #938 already did the solar equivalent for CERRA.
- **Size: complex, on the fourth trigger only.** What gets stored: only study outputs under `data/studies/` (losses, a report, a write-once leaderboard folder), no Patito contract or Delta table, so this trigger does not fire. Production serving path: no. Degradation rule: no. More than one defensible design: **yes**, because CERRA has no wind direction and only every third hour, so the choice of fitting device is a science decision. Callers of what changes: `past_wind_leaderboard.py` and its chart script are used only by the wind page, so the callers can be named, and this trigger does not fire.
- **Reviews:** one Opus 5.5 plan review (correctness and the device choice), then one Opus 5.5 diff review. A second review of either is added only if the first finds a science defect. This follows the standing token rule from the maintainer, not the skill's default of four reviews.
- **Departures from the issue body:** (1) the polish and mutation items move to their own PR. (2) The issue says "the wind equivalent of #938". CERRA wind cannot copy #938's method, which rebuilds hourly means from 3-hour windows, because wind is an instantaneous analysis every third hour and there is nothing to rebuild.

## Data facts the design rests on

- **CERRA:** 3-hourly instantaneous analysis at 00, 03, ..., 21 UTC, 2019-09-01 to 2026-06-30, 190 cells on a 5.5 km grid, columns `valid_time`, `y_index`, `x_index`, `wind_speed_m_s`. **No direction.** Heights 10, 50, 75, 100 and 150 m, all complete. The three wind farms map to 3 distinct cells inside the 190, 0.8 to 2.5 km away.
- **NORA3:** hourly instantaneous, 2015-01-01 to 2026-08-31, 1,680 cells on a 3 km grid, with speed and direction at 50 and 100 m. **No 10 m field.** The three farms sit inside the downloaded box, 1.0 to 1.5 km from their nearest cells, with 100 m speed present for every hour. The aggregate-to-monthly seam is a 0.55 m/s step, inside the ordinary step range.
- The scored hub height on the page is 80 m. CERRA's 75 m and 100 m levels bracket it. NORA3's 50 m and 100 m levels bracket it.

## What changes, file by file

- **`studies/beam_diffuse_split/cerra_past_wind.py` (new)**, modelled on `cerra_past_solar.py`. It reuses `assemble_rows`, `jobs`, `choose_fold_offsets`, `with_covering_folds`, `check_column_counts` and `refuse_to_overwrite`, takes nearest cells and hourly rows from the shared loader, and replaces the solar-specific rebuild. Rows are the site-hours that have a CERRA value (1 in 3 of all hours). The fitting device is **hub-level speed, 10 m speed, hour of day, day of year and `era_code`, with no direction and no neighbouring-hour columns**, identical in every arm and reference of the block. Arms: CERRA 75 m, 100 m and 150 m as the hub-level input (each with the 10 m column). References refit on the same rows and the same device: ERA5 at 100 m and ERA5 at the same level as the arm, plus whichever no-weather or station reference the main block uses (mirror `BLOCK_SETTINGS`). All fits use `colsample_bytree=1` and the same device (CPU unless told otherwise).
- **`studies/beam_diffuse_split/nora3_past_wind.py` (new)**, same shape. Rows are every site-hour. The device is **hub-level speed, direction as sine and cosine, hour of day, day of year and `era_code`, with no 10 m column and no context columns**, so the column count matches the arm's references. Arms: NORA3 50 m and 100 m. References: ERA5 100 m and the block's other reference, refit on the same device.
- **Nearest-cell lookup for NORA3** uses the roster coordinates in memory only, as `cov.py` does. The script prints counts and pooled distance ranges, never coordinates or which cell serves which farm.
- **Covering folds:** each script calls `choose_fold_offsets` on its own span and reports `uncovered_share` against the main study's published fold, split into avoidable and unavoidable months (the definitions the page already uses). CERRA ends 2026-06-30 and NORA3 on 2026-08-31, so both end before the main rows.
- **`studies/beam_diffuse_split/past_wind_leaderboard.py`:** new `ARM_LABELS`, `CERRA_ARMS`, `NORA3_ARMS`, `CERRA_PLANNED`, `NORA3_PLANNED`, two `RowSet`s in `ROW_SETS`, two `BLOCK_SETTINGS` entries (hub-height caption, reference name, and a **domain-coverage row**), "four" to "six" in `REPORT_INTRODUCTION`.
- **`studies/beam_diffuse_split/past_wind_leaderboard_charts.py`:** `BLOCK_LABELS` and `UNCOVERED_MONTH_SHARES` entries, caption lines stating the time step (CERRA 3-hourly, one hour in three) and the device (no direction, or no 10 m), "four" to "six" in titles.
- **`studies/beam_diffuse_split/sources.py`:** `WIND_LEADERBOARD_DIR` moves to `wind_leaderboard_3` (the folder is write-once).
- **`studies/beam_diffuse_split/figure_numbers.py`:** figure entries for any new SVG (per-generator lettered figures for the two new row sets if the page shows them).
- **Tests:** `tests/test_cerra_past_wind.py` and `tests/test_nora3_past_wind.py` (device columns, the 1-in-3 row selection, the same-width check across arm and reference, refuse-to-overwrite), plus two leaderboard tests and one added to `test_past_leaderboard_row_set_options.py` pinning that the new row sets use the default report options.
- **Docs:** `docs/studies/past-weather/wind.md` (a CERRA block and a NORA3 block in Data and methods, Results, Discussion and Limitations; the summary and key findings only if a planned contrast is significant), `docs/studies/past-weather/methods.md` (the wind row-set table, folded into the follow-up PR's table so both PRs do not add it twice: this PR adds only the two new rows), and the data-sources survey pages that name CERRA and NORA3 as scored.

## Planned contrasts (written here before any fit)

The planned contrasts are the ones written before the first fit; anything else run afterwards is labelled exploratory or post hoc on the page.

- **CERRA (2):** CERRA 100 m against ERA5 100 m on the same rows and device. CERRA 75 m against CERRA 100 m, because 75 m and 100 m bracket the 80 m hub height.
- **NORA3 (2):** NORA3 100 m against ERA5 100 m. NORA3 50 m against NORA3 100 m.
- Each also runs at the second setting, as the other blocks do. Absolute skill is reported beside every contrast.
- **Exploratory:** CERRA 150 m against CERRA 100 m, CERRA with 10 m removed, NORA3 against CERRA on the site-hours the two share (only where a common-row refit exists).

## Design-philosophy check

This is R&D code (`studies/`), so it fails fast: `refuse_to_overwrite`, `raise_on_uncovered_months`, and a `check_column_counts` error on any mismatched pair. Nothing touches production, no asset check is added, and no engineering hypothesis label is claimed.

## Verification

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest`, `uv run pre-commit run --all-files` (pydoclint, markdown lint), `uv run mkdocs build --strict` and a read of the rendered wind page, the leaderboard `--check-only` run, and the number-conservation and page-number gates. **Fits write under `data/studies/`, so the runner slot is requested from the Study MAIN COORDINATOR first.**

## Risks and open questions

1. **Is a direction-free CERRA device fair?** The other blocks give the arm direction. The plan gives the CERRA block's references the same reduced device so the contrast is like-for-like, but the CERRA absolute skill will not be comparable to the main block's. The page must say so beside every CERRA number. Recommendation: keep it and state the caveat.
2. **1 in 3 hours:** the CERRA block has a third of the rows of the main block, so its intervals are wider and its bootstrap resamples fewer calendar months' worth of hours. Recommendation: report the resampled-month count on the page.
3. **Cross-row-set comparison (NORA3 against CERRA)** is exploratory only, since the two blocks use different row spans and devices.
4. **Device for NORA3 without 10 m:** dropping the 10 m column from the references loses a feature they normally have. Recommendation: refit references without it, and add one exploratory reference row that keeps it.
5. **Fit time** is not recorded anywhere for the wind scripts. Estimate after a one-arm timing run.

## Rejected findings

None yet: no review has run.
