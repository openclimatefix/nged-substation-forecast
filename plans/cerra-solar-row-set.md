# Plan: score CERRA in the past-solar study as its own row set (#938)

**Problem.** The Copernicus regional reanalysis CERRA has been downloaded for the studies (#841)
but never scored. The maintainer wants CERRA comparisons published soon. The past-solar page and
its leaderboard hold four row sets (main, extra Open-Meteo, ENS, station), and the restructure
that just merged (#911, #917) left one slot for CERRA in each of the page's four places.

**Solution.** One new script, `studies/beam_diffuse_split/cerra_past_solar.py`, scores CERRA on
its own row set, with its own refitted CAMS and ERA5 reference arms on folds that cover every
calendar month, and writes to its own write-once folder
`data/studies/beam_diffuse_split/past_weather_v2/cerra_past_solar/`. The leaderboard script and
the two charts gain a fifth block for that row set, and the page and survey record the result.
CERRA publishes 3-hour accumulations only, so the script rebuilds hourly means with the study's
tested clear-sky-index code and adds ERA5 and CAMS arms given the same 3-hour treatment, which
separates CERRA's step width from its physics.

## Verdict, size and departures

**Verdict: worth implementing as described.** The data is on disk, the machinery (`ens_past_solar`,
`studies.resample`, `studies.cross_validation`) exists, and a regional 5.5 km reanalysis with its
own direct field is the missing point on the past-solar ranking between ERA5 and the satellites.

**Size: Complex.** Answers to the five triggers:

1. *What gets stored:* a new write-once output folder (`losses.parquet`, `losses.fingerprint`,
   `report.md`) and a new leaderboard folder `solar_leaderboard_3`. No Delta table, Patito model
   or asset. Does not fire on its own, but the outputs feed published numbers.
2. *Production serving path:* not touched. Nothing under `src/`, `packages/ml_core`,
   `packages/xgboost_forecaster` or the Dagster assets changes. Does not fire.
3. *Degradation rule:* none touched. This is R&D code, which fails fast. Does not fire.
4. *More than one defensible design:* fires. Three hourly rebuilds (hold each 3-hour mean, rebuild
   through the clear-sky index, or download leads 1 and 2 for true hourly means), two window
   choices, and where CERRA sits on the leaderboard each have defensible answers.
5. *Callers not nameable without searching:* fires. The change edits `past_solar_leaderboard.py`,
   `past_solar_leaderboard_charts.py` and the shared `studies.charts` block helpers, whose
   callers include the wind leaderboard and the blending charts, and touches every place the page
   says "four row sets" or "14 products".

**Reviews chosen:** both plan reviews (simplicity first, then correctness), then both diff reviews
in `implement-issue`. The study also gets the Opus fact check, scope-of-claims review and the two
science reviews the `study` skill requires before merge.

**Departures from the issue body:** none. Departures from the coordinator's brief: none. Where
the brief says "a domain-coverage row", this plan reads it as the report's coverage table plus the
products-table row (open question 3).

## What CERRA provides, from the files on disk

Read from `data/studies/weather/CERRA/` (the lineage notes, the parquet schemas, and the values).

- **Variables held:** `surface_solar_radiation_downwards` (global irradiance on a horizontal
  surface, GHI) and `time_integrated_surface_direct_short_wave_radiation_flux` (the direct beam
  on a horizontal surface). Wind at 100 m is also on disk and belongs to the wind study. No
  diffuse field: diffuse is global minus direct, as for ICON-DREAM-EU. No temperature or cloud
  field is downloaded, so CERRA arms take ERA5's `temp_c`, the shared feature every arm on the
  page reads.
- **Direct beam is available, as the producer's own field.** The ratio of direct to global at
  12 UTC has mean 0.31, maximum 0.84, and never exceeds 1 in 474,050 cells, which fits a direct
  beam on a horizontal surface. Step 1 confirms the definition against the CDS dataset page
  before the page says "horizontal".
- **Timing: 3-hour accumulations, not hourly and not instantaneous.** Both fields are
  `product_type=forecast, leadtime_hour=3` (CDS has no analysis product for them), in J m⁻², so
  each value is the energy over the 3 hours ending at the label (init at `valid_time − 3 h`,
  window `(valid − 3 h, valid]`). Labels fall at 00, 03, ..., 21 UTC only (474,050 rows at each
  hour, 190 cells × 2,495 days). Dividing by 10,800 s gives the mean flux in W m⁻². The
  survey's "hourly through forecast leads 1 to 3" describes what CDS could serve, not what is on
  disk.
- **Lead:** 0 to 3 hours after each 3-hourly analysis, so the first hours of a short forecast,
  the same family as ICON-DREAM-EU and ERA5's 1-to-12-hour radiation. The page states the lead as
  the window's own span, and does not compare it at matched lead with ERA5 (ERA5 radiation is 1
  to 12 hours).
- **Coverage:** 2019-09-01 03:00 to 2026-07-01 00:00 UTC, 190 grid cells (a cropped box around
  the roster), 5.5 km grid, no nulls, no NaNs, no duplicate keys. Every month has all 240 to 248
  time steps, so the tail is complete. The domain is the whole of Europe, so every generator is
  inside it. Each generator's nearest cell is 1.0 km to 3.5 km away (pooled range only; the
  per-generator table in `generator_cells.parquet` is never printed or charted).
- **Not provided:** hourly means (needs leads 1 and 2, not downloaded), diffuse, direct normal
  irradiance, cloud, temperature, or uncertainty. A forecast is not available either: CERRA is a
  reanalysis, so it is past weather, with the latency the survey records.

## Design decisions

**Hourly rebuild (the choice that needs approving).** The study scores hourly means. Options:

- *Rebuild through the clear-sky index, then rescale each window to its own mean* (recommended).
  Reuses `studies.resample.clear_sky_index_resample` and `rescale_to_step_means` unchanged, the
  method `ens_past_solar.py` uses for ENS, which `ens_forecast_horizons` picked as its best
  technique for 3-hourly radiation. Windows are contiguous, so the rebuild interpolates between
  neighbouring window midpoints with no lead gap.
- *Hold each window's mean for its three hours.* Simplest, but puts the whole window's energy at
  dawn and dusk hours.
- *Download leads 1 and 2 for exact hourly means.* Exact, but a further CDS download of two
  whole-domain requests per variable per 6-month chunk, queued one job at a time, with roughly
  15 GB of scratch per chunk. Out of proportion for the first CERRA result; recorded as a
  follow-on under #841.

**Positive control for the rebuild.** `era5_3h` and `cams_3h` average ERA5 and CAMS over CERRA's
own windows (00 to 03, ..., 21 to 24 UTC) with `studies.resample.step_means` and rebuild them with
the same code, as `ens_past_solar.py` does for ENS's steps. The gap between `era5_global` and
`era5_3h` is then the price of the step width alone, and `cerra_global − era5_3h` is CERRA against
ERA5 with the step width matched. Any conclusion about CERRA's physics rests on the matched
contrast, not the unmatched one.

**Row set.** The main (`solar_long`) frame's site-hours, cut to CERRA's last complete day
(2026-06-30) and to hours where CERRA, ERA5 and CAMS all have a value, so every arm scores the
same rows (the `study` skill's shared-rows rule). CERRA ends about 10 weeks before the main row
set (2026-09-10), so this row set is a near-subset of main's rows and a shorter window; the report
prints its row count, its first and last day, and the months dropped. The page says so in the
block's caption, as it does for the ENS and station blocks.

**Folds cover every calendar month.** `studies.cross_validation.search_fold_offsets` runs on this
row set's own frame, and `raise_on_uncovered_months` runs before any fit, so the fits cannot
inherit the extra block's 36.9% uncovered share. The report prints the uncovered share this row
set would have under the published main-row folds and under the chosen offsets (0 by
construction), so a reader sees what the covering design changed. If `search_fold_offsets` finds
no covering design, the script raises, and the plan is revised.

**Arms.** All use `colsample_bytree=1`, the eight-column shared feature set of `ens_past_solar`
(geometry and calendar features, `era_code`, `temp_c`) plus the arm's own irradiance columns, and
`_check_column_counts` (as in `station_past_solar.py`) raises if a contrast's two arms differ in
width.

- `era5_global`, `cams_global`: refitted on this row set (reference rows).
- `cerra_global`: CERRA's rebuilt hourly GHI, read at each generator's nearest cell.
- `era5_3h`, `cams_3h`: the positive controls above.
- `cerra_split`: CERRA's GHI, rebuilt direct and diffuse (global minus direct), three columns;
  `cerra_erbs`: CERRA's GHI with Erbs separation, three columns. The same pair the page's "own
  direct beam" section scores for other products (`{product}_split`, `{product}_erbs` in
  `weather_products.py`).

**Planned contrasts (four, fixed in this file before the first fit, each also run at the second
hyperparameter setting):**

1. `cerra_global − era5_global`: CERRA against ERA5, the reanalysis that forces it.
2. `cerra_global − cams_global`: CERRA against the best product on the main leaderboard.
3. `cerra_global − era5_3h`: CERRA against ERA5 with the step width matched.
4. `cerra_split − cerra_erbs`: whether CERRA's own direct field adds information over Erbs
   separation.

**Exploratory (labelled so in the report):** `cerra_global − cams_3h`; `era5_global − era5_3h`
and `cams_global − cams_3h` (the cost of the step width); the four planned contrasts per
generator (sign only, labels A to F, no coordinates or distances); CERRA's error by month of
year; the rebuild's check numbers (share of rebuilt hours below −1 W m⁻², the hour at which the
rebuilt series peaks relative to the sun). The planned/exploratory split is stated in the
report's first lines, and the leaderboard script's `RowSet.planned_contrasts` holds exactly the
four above.

## What changes, file by file

- `studies/beam_diffuse_split/cerra_past_solar.py` (new): reads the two parquet files and
  `generator_cells.parquet`, converts J m⁻² to W m⁻² (÷ 10,800), builds the frame with
  `blend_products._solar_frame` and the ENS script's `_era5_and_cams_hourly`, `build_rows()`,
  `jobs()`, `_fingerprint()`, `_report()`, `main()` with `--report-only`, `refuse_to_overwrite`,
  `check_no_missing`, and its own `OUTPUT_DIR = STUDY_DATA_DIR / "past_weather_v2" /
  "cerra_past_solar"`. Module docstring follows `ens_past_solar.py`: data, lead, arms, planned
  contrasts, exploratory list, the statement that the plan was committed before the first fit,
  and "only one agent may run it at a time". The report prints pooled distance and coverage
  ranges only, never per-generator values.
- `studies/beam_diffuse_split/past_solar_leaderboard.py`: a fifth `RowSet` for
  `cerra_past_solar` (with `CERRA_PLANNED`), and a `SOLAR_LEADERBOARD_DIR` bump to
  `solar_leaderboard_3` in `sources.py` (the write-once rule, as for `_2`); a re-run of the saved
  losses only, no refits. Row-count assertions and the number-checks against the report follow
  the existing four.
- `studies/beam_diffuse_split/past_solar_leaderboard_charts.py`: a fifth block in Figures 1 and 2,
  the "four row sets" title and caption words become "five", the per-block share and lead
  captions gain CERRA's (3-hour accumulation, lead 0 to 3 hours, window ending 2026-06-30).
  `figure_numbers.py` is unchanged: no figure is added, so the page keeps 17 figures. The two
  SVGs are redrawn, optimised with svgo, and their titles re-checked (width 681).
- `docs/studies/past-weather/solar.md`: fill the four `SLOT` comments (a Key findings bullet; a
  products-table row and the products count 14 → 15 in every place the page writes it; a Scope
  bullet if CERRA leaves a use case out; a Data and code availability source line), add a
  Summary sentence, a "The CERRA arms" subsection under Data and methods, a Results subsection
  headed with the conclusion, and a Limitations entry for the window, the 3-hour step and the
  uncovered-month share. Every number comes from `cerra_past_solar/report.md` and
  `solar_leaderboard_3`. "Four row sets" changes to "five" wherever the page counts them.
- `docs/background/weather-products-survey.md`: the CERRA rows' "Scored in" cells and the sentence
  "downloaded for the studies, not yet scored" (line 327) link to the new subsection. The survey's
  claim of "hourly through forecast leads 1 to 3" gets the on-disk fact beside it (3-hour
  accumulations, lead 3 only).
- `docs/studies/past-weather/methods.md` and `index.md`: only if they name the row-set count.
- `docs/roadmap/data-sources.md`: CERRA row's status wording, only if it says "not scored".
- `mkdocs.yml`: no change expected (no new page).

## Design-philosophy check

The change is R&D code that runs nowhere near production: the CERRA script fails fast (raises on
an uncovered month, a column-count mismatch, an off-by-one window check, missing values), and no
asset check or degradation path is added or edited. CERRA is past weather with a latency of weeks,
so the page's Scope says it cannot serve the live service (the survey already deprioritises it
for that reason). No engineering hypothesis (`H1`, `T1.2`) is claimed. No design principle is
traded away.

## Tests

- `packages/studies/tests/test_resample.py` gains none: `clear_sky_index_resample` and
  `rescale_to_step_means` are reused unchanged.
- New `tests/test_cerra_past_solar.py`, on synthetic frames, asserting what fails on `main` today
  (the module does not exist):
    - the J m⁻² to W m⁻² conversion (10,800,000 J m⁻² over 3 hours is 1,000 W m⁻²) and that a
      window labelled `t` covers `(t − 3 h, t]`: a synthetic day whose window energy is known
      rebuilds to hours whose mean over the window equals the window's mean, and whose peak falls
      in the hour ending at solar noon ± 1 hour rather than a window boundary;
    - `build_rows` drops every hour after 2026-06-30 and every hour where any of CERRA, ERA5 or
      CAMS is missing, so all arms share the same rows;
    - the four planned contrasts are exactly the four in this file (a set-equality assertion, the
      way `test_past_leaderboard_row_set_options.py` pins the other row sets);
    - `_check_column_counts` raises for a contrast whose arms differ in width;
    - the chosen fold offsets leave no uncovered month on a synthetic frame that reproduces the
    extra block's gap (a frame that fails with all-zero offsets).
- `tests/test_past_solar_leaderboard.py`: the five-row-set list, `CERRA_PLANNED`, and that the
  leaderboard refuses to write into a folder that already exists.
- `tests/test_past_solar_leaderboard_charts.py`: the fifth block's title, its reference rows
  (CAMS and ERA5), and the SVG width of 681 for both charts.
- `tests/test_figure_numbers.py` is unchanged (17 figures) and must still pass.

## Docs to update

The page (filled slots, "four" → "five", 14 → 15 products, the figure captions), the survey, and
the data-sources roadmap row, as above, each written in the present tense. No plan text survives
in `docs/`. Ship-time triage: this issue completes no roadmap item, so no roadmap page is deleted.

## Verification commands

The green-before-push set from `implement-issue`, plus the studies gates:

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`,
  `uv run pytest`, `uv run pydoclint`, `uv run pymarkdown scan -r docs README.md CLAUDE.md
  packages/*/README.md`, `uv run pre-commit run --all-files`.
- `uv run mkdocs build --strict`, then reading the rendered HTML of `past-weather/solar` for the
  new block, the caption words and the figure widths; the docs-link check.
- The page gates from the restructure: number-conservation (the `drop2.py` approach), planned
  versus exploratory status of every CERRA contrast, row-set naming ("the CERRA row set"),
  the SVG titles carry their figure numbers, and a repo-wide `git grep` for stale "four row sets"
  or "14 products".
- The data-validation skill on the CERRA files before any fit (gaps, duplicates, nulls, units,
  timestamp convention, the sun-height check of the rebuilt hours, and a correlation of daily CERRA
  and ERA5 means as a sanity check on the tail to 2026-06-30).
- Runs, in order, each only after the coordinator grants the runner slot: `cerra_past_solar.py`
  (fits, writes `cerra_past_solar/`), then `past_solar_leaderboard.py` (reads saved losses, writes
  `solar_leaderboard_3/`), then `past_solar_leaderboard_charts.py` (writes the two SVGs). Each
  refuses to overwrite.

## Risks and open questions

1. **Hourly rebuild.** Recommendation: clear-sky-index rebuild with the positive-control arms;
   download leads 1 and 2 later only if the rebuild's check numbers look poor. A reviewer may
   prefer the download first. That choice costs a queue of CDS jobs before any result.
2. **Fifth block versus a separate figure.** Recommendation: a fifth block in Figures 1 and 2
   (the brief's "own row set" on the same leaderboard), no new figure, so the figure map and the
   17-figure count stay. Cost: both charts grow taller by about one block (about 480 px on
   Figure 2).
3. **What "a domain-coverage row" means.** This plan reads it as (a) the report's coverage table
   (row count, first and last day, months dropped, the uncovered-month share under published and
   covering folds, and the pooled nearest-cell distance range 1.0 km to 3.5 km) and (b) CERRA's
   row in the products table (domain: the whole of Europe, so every generator is inside it). It
   does not add a CERRA outline to Figure 3, whose outlines are national-scale. If the maintainer
   wants a drawn outline, that is one geojson and a figure edit.
4. **A shorter window than main.** CERRA's row set loses the last 10 weeks of the main row set
   (2026-07-01 to 2026-09-10). The page reports CERRA's numbers only beside its own CAMS and ERA5
   rows, never against the main block's numbers. The block caption says so.
5. **CERRA's end date.** The lineage shows chunks to 2026-06-30 with plausible values, and the
   monthly noon-hour means follow the seasons through the last month. The data-validation step
   checks CERRA against ERA5 for the last months before any fit. If the tail is bad, the window
   shortens and the plan's row count changes, not its design.
6. **Merge order with PR #937** (past-wind prose) and any run of the wind leaderboard: both read
   `sources.py` and `studies.charts`. Merge `origin/main` before each push and rebase nothing.
7. **Sensitive data.** The report and page carry pooled ranges and labels A to F only. Reviewers
   are asked to grep the report and page for coordinates, distances per generator and station
   names before merge.

## Reviews

Filled in as each finishes: simplicity review, correctness review, and the triage of each
(findings kept and findings rejected with a one-line reason).
