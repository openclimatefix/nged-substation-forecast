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
  hour, 190 cells × 2,495 days). Night values are about 6e-15 rather than 0, which the
  clear-sky mask zeroes. Dividing by 10,800 s gives the mean flux in W m⁻². The
  survey's "hourly through forecast leads 1 to 3" describes what CDS could serve, not what is on
  disk.
- **Lead:** 0 to 3 hours after each 3-hourly analysis, so the first hours of a short forecast,
  the same family as ICON-DREAM-EU and ERA5's 1-to-12-hour radiation. The page states the lead as
  the window's own span, and does not compare it at matched lead with ERA5 (ERA5 radiation is 1
  to 12 hours).
- **Coverage:** 2019-09-01 03:00 to 2026-07-01 00:00 UTC, 190 grid cells (a cropped box around
  the roster), 5.5 km grid, no nulls, no NaNs, no duplicate keys. Every month has all 240 to 248
  time steps (February has 224 or 232, September 2019 has 239, and the last label, 2026-07-01
  00:00, is the window 21 to 24 UTC on 30 June), so no month is short. The domain is the whole of
  Europe, so every generator is
  inside it. Each generator's nearest cell is 1.0 km to 3.5 km away (pooled range only; the
  per-generator table in `generator_cells.parquet` is never printed or charted).
- **Not provided:** hourly means (needs leads 1 and 2, not downloaded), diffuse, direct normal
  irradiance, cloud, temperature, or uncertainty. A forecast is not available either: CERRA is a
  reanalysis, so it is past weather, with the latency the survey records.

## Design decisions

**Hourly rebuild (the choice that needs approving).** The study scores hourly means. Options:

- *Rebuild through the clear-sky index* (recommended). One new pure function in the script,
  `rebuild_hourly_from_windows`, takes each site's 3-hour window means, ending at 00, 03, ..., 21
  UTC, and returns the hour-ending-at-label means, using `studies.resample.clear_sky_index_resample`
  and `studies.baselines.hourly_clear_sky` unchanged. That is the method `ens_past_solar.py` uses
  for ENS's steps, which `ens_forecast_horizons` picked as its best technique for 3-hourly
  radiation. The windows are contiguous, so the rebuild interpolates between neighbouring window
  midpoints with no lead gap. No rescaling to the window's mean is applied: `ens_past_solar.py`
  does not rescale either (`rescale_to_step_means` is used only by `ens_forecast_horizons.py`), and
  the same function serves all three arms, so the matched contrast differs only in the input.
  The script does not import `ens_past_solar`'s private `_three_hourly_rebuilt`, which averages
  hourly data into ENS's 00 and 12 UTC run phases and cannot rebuild data that are already steps.
- *Hold each window's mean for its three hours.* Simplest, but puts the whole window's energy at
  dawn and dusk hours.
- *Download leads 1 and 2 for exact hourly means.* Exact, but a further CDS download of two
  whole-domain requests per variable per 6-month chunk, queued one job at a time, with roughly
  15 GB of scratch per chunk. Out of proportion for the first CERRA result; recorded as a
  follow-on under #841.

**Positive control for the rebuild.** `era5_3h` and `cams_3h` average ERA5's and CAMS's hourly
means over CERRA's own windows (the three hours ending at 00, 03, ..., 21 UTC, a plain mean where
all three hours are present, and dropped where not) and rebuild them with the same
`rebuild_hourly_from_windows`, so their windows have CERRA's phase, not the ENS runs' 00 and 12
UTC phase. The 0.287-point step-width figure on the page belongs to ENS's phase and does not carry
over to CERRA's windows: the report measures it again as `era5_global − era5_3h`. The gap between
`era5_global` and
`era5_3h` is then the price of the step width alone, and `cerra_global − era5_3h` is CERRA against
ERA5 with the step width matched. Any conclusion about CERRA's physics rests on the matched
contrast, not the unmatched one.

**Row set.** The main (`solar_long`) frame's site-hours, cut to hours ending at or before the
window end 2026-07-01 00:00 UTC (the last CERRA label, which belongs to 30 June) and to hours where
CERRA, ERA5 and CAMS all have a value, so every arm scores the same rows (the `study` skill's
shared-rows rule). CERRA is on disk from 2019, and the row set starts in December 2022 because the
power data, ERA5 and CAMS limit it, about 43 months to June 2026. CERRA ends about 10 weeks before
the main row
set (2026-09-10), so this row set is a near-subset of main's rows and a shorter window; the report
prints its row count, its first and last day, and the months dropped. The page says so in the
block's caption, as it does for the ENS and station blocks.

**Folds cover every calendar month.** The frame is folded with `studies.cross_validation.cut_eras`
on `first_months=(UKV_UPGRADE_MONTH,)` (the one era boundary in this window; the era after it is
five months, February to June 2026), with the offsets `search_fold_offsets` returns for this
row set's own frame (at most five candidates, `N_FOLDS=5`, one era after the first), and
`raise_on_uncovered_months` runs before any fit. `ens_past_solar.build_rows` ends in `with_eras`,
which cuts main-style folds, so the new script builds its own rows and does not end there. The
report prints the uncovered share this row
set would have under the published main-row folds and under the chosen offsets (0 by
construction), so a reader sees what the covering design changed. If `search_fold_offsets` finds
no covering design, the script raises. The fallback is then to report the uncovered share, and the
maintainer decides whether to proceed.

**Arms.** All use `colsample_bytree=1`, the eight-column shared feature set of `ens_past_solar`
(geometry and calendar features, `era_code`, `temp_c`) plus the arm's own irradiance columns, and
`_check_column_counts` (as in `station_past_solar.py`) raises if a contrast's two arms differ in
width.

- `era5_global`, `cams_global`: refitted on this row set (reference rows).
- `cerra_global`: CERRA's rebuilt hourly GHI, read at each generator's nearest cell.
- `era5_3h`, `cams_3h`: the positive controls above.
- `cerra_split`: CERRA's rebuilt GHI and rebuilt direct, with diffuse as global minus direct
  clipped at zero, three columns (the report counts the hours clipped);
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
generator (sign only, labels A to F, no coordinates or distances); the rebuild's check numbers
(share of rebuilt hours below −1 W m⁻², the hour at which the rebuilt series peaks relative to
the sun). The planned/exploratory split is stated in the
report's first lines, and the leaderboard script's `RowSet.planned_contrasts` holds exactly the
four above.

## What changes, file by file

- `studies/beam_diffuse_split/cerra_past_solar.py` (new, imports what exists and copies nothing):
  reads the two parquet files, converts J m⁻² to W m⁻² (÷ 10,800), and builds the frame with
  `blend_products._solar_frame` and `ens_past_solar`'s public-by-import helper
  `_era5_and_cams_hourly` (made public in the same commit if importing a private name fails the
  linter). Pure functions of frames, so they can be tested without `data/`:
  `windows_from_accumulation`, `rebuild_hourly_from_windows` (all three arms), `windowed_mean` (the
  ERA5 and CAMS 3-hour means), `join_rows` (the shared-rows cut and dtype checks). Then `jobs()`,
  `_fingerprint()`, `_report()`, `main()` with `--report-only`, `refuse_to_overwrite`,
  `check_no_missing`, and its own `OUTPUT_DIR = STUDY_DATA_DIR / "past_weather_v2" /
  "cerra_past_solar"`. It asserts CERRA's time column is a naive `datetime[ns]` in UTC and casts it
  to the study frames' UTC `datetime[us]` explicitly. Module docstring follows `ens_past_solar.py`:
  data, lead, arms, planned contrasts, exploratory list, the statement that the plan was committed
  before the first fit, and "only one agent may run it at a time". The report prints pooled ranges
  and the cell count (190) only: never per-generator distances, the cell indices, the cropped
  box's extent, or coordinates.
- **The nearest-cell table has no recorded provenance.** No script in the repo writes
  `generator_cells.parquet` (created 2026-09-24 by an unrecorded step), and the CERRA files hold
  cell indices with no latitude or longitude, so its A to F labels cannot be checked against
  `_pv_sites()`. Step 1 is a one-field CDS request (one time step of one variable, about 10 MB)
  that fetches the grid's latitude and longitude, a `--write-grid` mode of `fetch_cerra.py` that
  records the request in the lineage note. The script then derives each site's nearest cell from
  `_pv_sites()` coordinates and asserts it equals `generator_cells.parquet`; a mismatch stops the
  run. The request writes under `data/studies/weather/CERRA/`, so it needs the coordinator's slot.
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
- `docs/studies/past-weather/methods.md`: required. It says "Five row sets" with a table (line 13)
  and "The 14 planned contrasts" (line 85), which become six and 18. `index.md` is checked.
- `docs/studies/past-weather/solar.md` also names `solar_leaderboard_2` under "Reproducing the
  figures" (near line 1635), which changes to `solar_leaderboard_3`.
- `docs/roadmap/data-sources.md`: the CERRA row (line 231, "Superseded by ERA5") gains the scored
  result. `git grep "four row sets"` also matches the wind files, whose "four" is the wind page's
  and stays.
- `mkdocs.yml`: no change expected (no new page).

## Design-philosophy check

The change is R&D code that runs nowhere near production: the CERRA script fails fast (raises on
an uncovered month, a column-count mismatch, an off-by-one window check, missing values), and no
asset check or degradation path is added or edited. CERRA is past weather with a latency of weeks,
so the page's Scope says it cannot serve the live service (the survey already deprioritises it
for that reason). No engineering hypothesis (`H1`, `T1.2`) is claimed. No design principle is
traded away.

## Tests

- New `tests/test_cerra_past_solar.py`, on synthetic frames, importing the script's pure
  functions. A test is worth having only if it fails when the code is wrong:
    - the J m⁻² to W m⁻² conversion (10,800,000 J m⁻² over 3 hours is 1,000 W m⁻²) fails on a wrong
      divisor;
    - an asymmetric synthetic day whose window energy is known: the rebuilt hour-ending means peak
      in the exact hour (not within ± 1 hour), a start-labelled reading of the windows moves the
      peak
      by 3 hours and fails, and the rebuilt hours of a window average within a stated tolerance of
      the window mean (no rescaling is applied, so the tolerance is stated, not zero);
    - `join_rows` drops every hour after the window end 2026-07-01 00:00 and every hour where any of
      CERRA, ERA5 or CAMS is missing, and raises on a time column of the wrong dtype or zone;
    - the rebuilt direct and clipped diffuse are non-negative and the count of clipped hours is
      returned;
    - the four planned contrasts are exactly the four in this file (set equality, extending
      `test_past_leaderboard_row_set_options.py`, which pins the other row sets' options);
    - `_check_column_counts` raises for a contrast whose arms differ in width;
    - the chosen fold offsets leave no uncovered month on a synthetic frame that fails with all-zero
      offsets.
- `tests/test_past_solar_leaderboard.py`: the five-row-set list, `CERRA_PLANNED`.
- `tests/test_past_solar_leaderboard_charts.py`: the fifth block's title and its reference rows
  (CAMS and ERA5), and that both charts have the same width (not a fixed 681, which a longer label
  moves).
- `tests/test_figure_numbers.py` is unchanged (17 figures) and must still pass.
- Mechanical checks in the script on the built row set (they run in every fit, not only in
  tests): the hour of the peak of the rebuilt CERRA clear-day composite equals ERA5's, since a
  wrong window convention shifts the series by 3 hours without any value looking wrong.

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
- `git grep -n "solar_leaderboard_2"` returns only the superseded folder's own mentions, and
  `grep -E "km|latitude|y_index|x_index"` on the report and the page shows only pooled values.
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

**The limit of not downloading leads 1 and 2, stated plainly (maintainer's coordinator, decided
before the reviews).** CERRA is scored from 3-hour window means only. The rebuild to hourly values
is a model, so CERRA's hourly values inside a window are not CERRA's own. `era5_3h` and `cams_3h`
match the step width for ERA5 and CAMS, so `cerra_global − era5_3h` is the like-for-like contrast,
and `cerra_global − era5_global` and `cerra_global − cams_global` carry an unmatched step width
that the page states beside each. The page never reads CERRA's error as a limit on CERRA's
physics at hourly resolution.

## Reviews

### Correctness review, and its triage

Sixteen findings, all kept in the plan above except where noted:

- Blocking, kept: the ENS script does not rescale windows, so one shared rebuild without rescaling
  serves all three arms (finding 1); the rescale function cannot run on ENS's target grid and is
  dropped from this plan (2); the ENS helper cannot rebuild step data, so a new pure function does
  (3); folds come from `cut_eras` with `search_fold_offsets`' offsets, the era boundary and the
  fallback are named (4); the nearest-cell table's provenance is rebuilt from the grid's
  coordinates (5).
- Serious, kept: February's step count and the window-end wording (6); the dtype assertion, the
  peak-hour check and the row-set cut as a window end (7); the direct-and-diffuse rebuild and
  clipping (8).
- Tests, kept: the peak test asserts the exact hour, `build_rows` is split into pure functions,
  the always-passing "refuses to overwrite" test is dropped, the width test compares the two
  charts (9).
- Design, kept: the shorter window's reason, the confounds of contrasts 1 and 2 (grid spacing and
  radiation scheme as well as step width, so the page never reads the gap as CERRA's physics), the
  six arms listed, the 0.287-point figure not carried over, the generators not six replicates
  (10 to 12).
- Anonymisation, kept: the report prints the cell count only (13). Missed places, kept: the
  methods page counts, the reproduction path, the roadmap row, the extra greps (14 to 16).

### Simplicity review, and its triage

Seven findings, triaged against the code:

1. *Cut `era5_3h` and `cams_3h`.* Rejected. Without a step-width-matched arm, the headline contrast
   confounds CERRA's physics with the 3-hour step, and the ENS section already measures that the
   step alone moves ERA5's error by 0.287 points. The arms are imported code, not new plumbing.
2. *Cut `cerra_split` and `cerra_erbs`.* Rejected. CERRA's own direct field is what the survey
   ranks it on, and the brief asks what CERRA does and does not provide.
3. *Cut the exploratory list.* Accepted in part: the by-month-of-year table is dropped. The
   per-generator sign check stays, because the page reports every planned contrast at all six
   generators (labels only), and so do the other row sets.
4. *`search_fold_offsets` is unnecessary.* Rejected on the saved evidence: the main row set's own
   folds leave 2,129 of 76,727 scored hours (2.8%, one farm, May to July) with no training row for
   their calendar month, and this row set is a near-subset of main's rows. Step 1 computes the
   uncovered share on the frame with `calendar_month_coverage` first, and uses
   `search_fold_offsets` only for the offsets that remove it.
5. *Do not make CERRA a `weather_products.py` panel; import rather than copy.* Accepted: the new
   script imports `_three_hourly_rebuilt`-style helpers and `_era5_and_cams_hourly` from
   `ens_past_solar.py`'s module (moving them to a shared module only if the import creates a
   cycle), never copies them.
6. *Keep the fifth block.* Accepted as planned.
7. *Smallest equivalent change.* Not adopted whole: it drops the matched arm (finding 1) and the
   beam contrast (finding 2).
