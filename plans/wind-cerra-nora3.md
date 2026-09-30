# Plan: CERRA and NORA3 as row sets on the past-wind page (#968)

Issue #968 asks for CERRA and NORA3 wind to be scored as their own row sets on `docs/studies/past-weather/wind.md`, against the arms the page already covers, plus the polish and mutation-testing follow-ups from PR #937 and #930. **This plan covers the two row sets.** Each row set follows the `wind_icon_dream.py` pattern: take the main block's rows, join the new product, and refit all five main products plus the new product on those rows, on the fixed folds, with the same inputs and device in every arm. The leaderboard and page gain a fifth and a sixth block.

**The polish and mutation follow-ups ship as a separate simple PR**, and the shared loader ships first (below).

## Delivery order: three PRs

1. **Shared loader PR (first, from `main`, sized simple).** The Study MAIN COORDINATOR decided #968 owns the CERRA and NORA3 wind loader and #957 imports it. It is a module in `packages/studies/src/studies/` with tests: it reads the CERRA and NORA3 files, derives each wind farm's nearest cell from `cerra_grid.parquet` and asserts each is strictly inside the 190-cell crop, pivots the heights (10 m from CERRA's single-levels product, 50, 75, 100 and 150 m from its height-levels product), and builds the centred power hour. It prints counts and pooled distance ranges, never coordinates or which cell serves which farm, and writes nothing under `data/studies/`.
2. **Polish and mutation-testing PR** (from `main`, simple).
3. **This PR:** one fitting script, leaderboard blocks and page prose, importing the loader.

## Verdict, size and departures

- **Verdict:** worth doing, with the changes below from the plan review.
- **Size: complex, on the fourth trigger only.** Stored: study outputs under `data/studies/` only, no Patito contract or Delta table. Serving path: no. Degradation rule: no. More than one defensible design: yes, because the inputs each product can supply differ. Callers of what changes: `past_wind_leaderboard.py` and its chart script are used only by the wind page, so they can be named.
- **Reviews:** one Opus 5.5 plan review (done, triaged below), one Opus 5.5 diff review. A second review only if the first finds a science defect.
- **Departures from the issue body:** the polish and mutation items move to their own PR. CERRA wind cannot copy #938's method, which rebuilds hourly means from 3-hour windows, because wind is an instantaneous analysis every third hour.

## Data facts

- **CERRA:** 3-hourly instantaneous analysis at 00, 03, ..., 21 UTC, 2019-09-01 to 2026-06-30, 190 cells on a 5.5 km grid. Wind speed at 10, 50, 75, 100 and 150 m, all complete. **Direction is not downloaded**; whether CDS serves it is an open question to the Data DOWNLOAD COORDINATOR. The three wind farms map to 3 distinct cells inside the 190, 0.8 to 2.5 km away.
- **NORA3:** hourly instantaneous, 2015-01-01 to 2026-08-31, 1,680 cells on a 3 km grid, speed and direction at 50 and 100 m downloaded. **10 m is served but not downloaded**; a refetch has been requested. The three farms sit inside the downloaded box, 1.0 to 1.5 km from their nearest cells, with 100 m speed present in every hour.
- **Scored height:** the page scores ERA5 and UKV at 100 m and the ICON products at 80 m (their served 100 m wind is a rescaled 120 m value). Farm hub heights are unknown. CERRA and NORA3 are scored at **100 m**.

## Design

- **Rows.** The main rows from 2024-08-12, inner-joined to the new product, as `wind_icon_dream.py` does. CERRA rows are the site-hours with a CERRA value (1 in 3 hours) up to 2026-06-30. NORA3 rows run to 2026-08-31. Covering folds come from `choose_fold_offsets`, with the uncovered-month shares reported as the page already defines them.
- **Arms and references.** Refit all five main products and the new product at its 100 m level on the joined rows, same device, `colsample_bytree=1`, CPU unless told otherwise. Each block has its own refitted references, so the "ERA5" row in each block is that block's fit, and the page says so.
- **Inputs.** The main block's inputs column for column: hub-level speed, that height's direction as sine and cosine, 10 m speed, hour of day, day of year, and `era_code`. **NORA3 waits for its 10 m refetch.** **CERRA waits for the answer on direction.** If CDS does not serve CERRA direction, CERRA uses the same inputs without the direction pair, and its references are refit without it, so every arm in the block has equal width (`check_column_counts` enforces this). CERRA direction is never borrowed from ERA5, which would put ERA5's information inside the CERRA-against-ERA5 contrast. The neighbouring-hour context columns are not built for CERRA, which has no neighbouring hours.
- **One fitting script**, `studies/beam_diffuse_split/reanalysis_past_wind.py`, takes the product as an argument and writes `losses.parquet` and `report.md` per product, refusing to overwrite. Its tests check the 1-in-3 row selection, the equal-width check and the write-once refusal.
- **Leaderboard.** `past_wind_leaderboard.py` gets `ARM_LABELS`, the arm and planned tuples, two `RowSet`s and two `BLOCK_SETTINGS` entries. The domain-coverage text goes in `BlockSetting.note`. `past_wind_leaderboard_charts.py` gets `BLOCK_LABELS` and `UNCOVERED_MONTH_SHARES` entries, and the caption notes `CHANCE_NOTE`, `PLANNING_NOTE` and `REFERENCE_ROW_NOTE` are widened, because they describe only four blocks. `sources.py` moves `WIND_LEADERBOARD_DIR` to `wind_leaderboard_3` (write-once). `figure_numbers.py`: `wind_figure_number` has letters a to c and `WindRowSetType` lists three lettered row sets, so per-generator lettered figures for the new row sets need both extended or are left out.
- **Tests.** New tests for the fitting script (above); the existing loops over `ROW_SETS` in `test_past_wind_leaderboard.py` cover the new blocks, so no test is added to `test_past_leaderboard_row_set_options.py`.
- **Docs.** `wind.md` (Data and methods, Results, Discussion, Limitations for each new block), the two new rows in `methods.md`'s wind row-set table, and the survey pages that name CERRA and NORA3 as scored.

## Planned contrasts (written here before any fit)

Everything else run afterwards is labelled exploratory or post hoc.

- **CERRA 100 m against ERA5 100 m**, and **NORA3 100 m against ERA5 100 m**, each on its own block's rows and inputs.
- **Each new product against the main block's leading product**, on the same rows.
- Each also runs at the second setting, as the other blocks do, and absolute skill is reported beside every contrast.
- **Exploratory:** CERRA at 75 m and 150 m, NORA3 at 50 m, and NORA3 against CERRA on shared site-hours. Comparing heights of one product is #957's question, not this issue's.

## Design-philosophy check

R&D code in `studies/`, so it fails fast: `refuse_to_overwrite`, `raise_on_uncovered_months`, and a `check_column_counts` error on a mismatched pair. Nothing touches production, and no asset check is added.

## Verification

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest`, `uv run pre-commit run --all-files`, `uv run mkdocs build --strict` and a read of the rendered wind page, the leaderboard `--check-only` run, and the number-conservation and page-number gates. **Fits write under `data/studies/`, so the runner slot is requested from the Study MAIN COORDINATOR first.**

## Risks and open questions

1. **CERRA direction and NORA3 10 m** depend on two downloads asked of the Data DOWNLOAD COORDINATOR. Recommendation: fit each block once its inputs are downloaded, with the CERRA no-direction fallback.
2. **1 in 3 hours:** the CERRA block has a third of the main rows, so its intervals are wider. The page reports the resampled-month count.
3. **The five main products refit per block** costs more fits than the ERA5-only plan, and the fit time is not recorded for the wind scripts. Estimate after a one-arm timing run.

## Rejected findings

- **"Drop the CERRA 75 m and 150 m and NORA3 50 m arms":** partly taken. They stay as exploratory arms only if the timing run shows the extra fits are cheap.
- **Review claim that `cov.py` does not exist:** it exists in the scratch folder `.claude/worktrees/scratch/nora3-coverage/`, outside the repo, so it is not a repo file. The plan now describes the lookup rather than citing the script.
