# Weather forecasts for power, compared at matched lead times

Study for [issue #810](https://github.com/openclimatefix/nged-substation-forecast/issues/810). The
question is which forecast product, or which blend of products, gives the most accurate power
forecast at the day-ahead lead the live service delivers, and at the two days after it. The design
and the results are on the published page, [Which weather forecast is best at day-ahead
lead?](../../docs/studies/nwp-forecasts-at-matched-leads.md).

## Scripts

- `verify_previous_runs_leads.py` runs three checks and writes one markdown table each under
  `--output-dir`. V1 (a gate) compares Open-Meteo's GFS `previous_dayN` values with the Dynamical.org
  GFS runs, for candidate run offsets `k` from 0 to 12 hours, and passes only if the mean absolute
  difference is lowest at `k = 0` for `N = 1` and `N = 2`. Each site scores every grid cell of the
  Dynamical.org extract and keeps the best, so no site coordinate is read. V1 fails if nothing is
  scored. V1b gives, per product, field and day, each UTC hour's mean absolute second difference of
  the `previous_dayN` series divided by its mean over all hours, and reads a 3- or 6-hourly run
  cycle where the switch hours stand out by a ratio of 1.3 or more. V3 measures each product's
  radiation timestamp convention from the offset at which its `previous_day1` series tracks the
  sun best, and tests each candidate rebuild of UKV's hourly value from its snapshots; `--v3-only`
  runs V3 alone, and the study's `report.md` quotes `v3_conventions.md` from a `verification/`
  directory beside it. Outputs: `v1_wind.md`, `v1_radiation.md`,
  `v1b_run_switches.md`, `v3_conventions.md`.
- `check_input_steps.py` (V1c) divides each month's mean day-1 radiation and 100 m wind speed of UKV
  and IFS 0.25° by ICON-EU's at the same site, and flags a month-to-month change of 15% or more.
  Output: `v1c_steps.md`.
- `build_forecast_inputs.py` writes `solar_forecast_inputs.parquet` and
  `wind_forecast_inputs.parquet` under `--output-dir` (default `data/studies/nwp_forecast_comparison/`).
  Each file has one row per generator-hour with the target, the capacity, the calendar and
  sun-position columns, every Previous Runs product's weather columns at its planned day offsets
  (wind speeds in m/s), ECMWF ENS's mean at days 0 to 3 and control member at days 0 to 3, the
  no-weather baselines' inputs, and, where GEFS is built, the GEFS mean at days 1 to 3. ENS and GEFS
  are built directly at the chosen upsampling combination (`clear_sky` for solar, `speed_components`
  for wind) through the public functions of `studies/beam_diffuse_split/ens_forecast_horizons.py`.
  GEFS reads each site's nearest 0.25° cell, converts its alternating 3- and 6-hour radiation
  windows to 3-hour step means (`studies.resample.gefs_step_means`) on whole runs before any band is
  sliced, and averages the 31 members. It runs only when
  `data/studies/weather/GEFS_window_2024-11-01_None/_month_cache/` covers 2024-11 to the month the
  rows end on (the last month may be a `.partial.parquet`), or when `--gefs-window-dir` names a
  `GEFS_window_*` extract. The build raises if any 00 UTC run the rows need is missing or
  incomplete.
- `nwp_forecast_comparison.py` reads those files and takes the rows where the target, the
  baselines' inputs, and every planned arm's columns are present. It cuts folds with
  `studies.cross_validation.cut_eras`, fits every arm out of fold, scores the no-weather
  baselines, and writes `report.md`, `<domain>_losses.parquet` and `<domain>_predictions.parquet`
  under `--output-dir`. Every contrast is computed within one hyperparameter setting, and a
  planned verdict stands only if both settings give it. `--dry-run` stops after printing the
  coverage table and the job list. `--report-only` writes the report from the saved losses.
  `--fit-missing` fits only the (arm, setting) pairs the saved losses lack. `--synthetic-losses`
  fabricates losses instead of fitting, to exercise the report, and refuses to write under
  `data/studies/`.
- `verify_extra_leads.py` reads only, and writes two Markdown files under
  `<output-dir>/verification/`: `gefs_radiation_window.md` checks that GEFS's radiation beyond 240 h
  is a 6-hour window mean, and `day0_matches_past_series.md` checks that Open-Meteo's unsuffixed
  Previous Runs series, which the extra-lead build reads as day 0 for ICON-D2 and ICON-EU, equals
  the series the past-weather studies read.
- `build_forecast_inputs.py --extra-leads --published-dir PUBLISHED --output-dir DIR` writes the
  extra lead days' input columns (ENS at days 5 and 14, GEFS at days 5, 10, and 14, IFS 0.25°, GFS,
  and ICON global at day 5, and ICON-D2 and ICON-EU at day 0) onto the published inputs' own
  `(site, time)` keys, to a new folder.
- `fit_extra_leads.py` fits those arms on a GPU at the primary setting only, together with the
  published arms they are compared with, refitted on the same device. `--check` fits one arm at one
  generator twice and stops unless the two runs agree. `--report-only` writes the report from the
  saved losses. It never writes to the published folder.
- `build_forecast_inputs.py --extra-leads --batch third` writes the native GFS arms
  `gfs_native_day<N>_*` at days 0, 1, 2, 3, 5, 7, 10, and 14, read from Dynamical.org's GFS store
  (`data/studies/weather/GFS/`) at each generator's nearest 0.25 degree cell. Day 1 and above read
  the 00 UTC run issued that many days before, at leads from 24 hours per day. Day 0 reads the
  freshest of the four runs a day, at a lead of 1 to 6 hours for solar and 0 to 5 hours for wind.
  The store's radiation is a mean since the last 6-hourly reset, with the lead labelling the end of
  the window, so `studies.gfs_native.step_means` recovers the mean over each hour (or, beyond lead
  120, each 3 hours) before use. Days 0 to 4 read hourly leads directly, and days 5 and above are
  upsampled from 3-hourly leads as the ENS and GEFS arms are.
- `verify_gfs_native.py` reads only, and writes `gfs_radiation_window.md` (whether the store's
  radiation follows the reset rule, at every window length including the windows shorter than 6
  hours and lead 0), `gfs_served_runs.md` (the run and lead each target hour reads), and, with
  `--built-dir`, `gfs_built_columns.md` (a sample of built values recomputed from the store in
  plain Python) under `<output-dir>/verification/`. It exits non-zero if any check fails.
- `fit_extra_leads.py --batch third` fits the eight native GFS arms per technology on a GPU and
  refits nothing: its contrasts against the ENS mean and against Open-Meteo's GFS read the earlier
  batches' GPU fits, through one `--context-dir` for each earlier batch's folder.
- `build_forecast_inputs.py --extra-leads --batch fourth` writes the arms `ifs_single_day<N>_*` at
  days 0, 1, 2, 3, 5, and 7, read from Open-Meteo's Single Runs archive of ECMWF IFS HRES
  (`data/studies/weather/ECMWF-IFS-SINGLE-RUNS/`): one 00 UTC run a day with hourly leads 0 to 240.
  Day `N` reads the 00 UTC run issued `N` days before the hour's own day, at leads from 24 hours per
  day, so day 0 is the run of the hour's own day. Day 0 covers hours before the 00 UTC run is
  published, as ENS's day 0 does, so day 0 is not a forecast that could have been used in advance
  for those hours. There is no day 10, because day 10 needs leads 240 to 264 hours and the runs end
  at 240 (`studies.ifs_single_runs`). Values are used as the archive serves them: radiation clipped
  at zero, wind as speed and the sine and cosine of the direction (the Previous Runs arms' method,
  not `speed_components`, because the archive serves no components). Hourly values from lead 91
  hours are interpolated from IFS HRES's 3-hourly and 6-hourly steps. Sites B and D share a source
  cell and carry identical series. The arm is a row of its own, "IFS HRES (9 km, Open-Meteo)"
  (Open-Meteo's `ecmwf_ifs`, on ECMWF's O1280 grid), never merged with "IFS 0.25°", which is a
  coarser product and not another version of it. The archive holds 00 UTC runs only, and
  Open-Meteo's processing of HRES has not been checked against a native archive. The IFS cycle
  changed inside the span (cycle 50r1, 12 May 2026, from ECMWF's pages); the arm adds no era feature
  and uses the shared rows' `era_code`, which is cut on the target hour's month. Days on which the
  archive lacks the serving run are gaps: their columns are null and they are never filled from
  another run.
- `verify_ifs_single.py` reads only, and writes `ifs_single_source.md` (runs present and absent by
  count, complete leads, radiation below zero, the hour-ending label, and the interpolated native
  steps), `ifs_single_served_runs.md` (the run and lead each target hour reads, and that day 10 is
  unservable), and, with `--built-dir`, `ifs_single_built_columns.md` (the gap rows by count, and a
  sample of built values recomputed from the archive in plain Python) under
  `<output-dir>/verification/`. It exits non-zero if any check fails.
- `fit_extra_leads.py --batch fourth` fits the six IFS HRES (9 km, Open-Meteo) arms per technology
  on a GPU and refits nothing. Each arm is fitted and scored on the shared rows minus the rows with
  a gap in its own columns. Its contrasts against the ENS mean at the same day, IFS 0.25° at days 1,
  2, 3, 5, and 7, and ICON-EU at days 1, 2, and 3 are computed on the rows both arms score, from the
  earlier batches' losses read through `--context-dir` (the first and the second, and no refit).
  Each prints its row and month counts and the absolute error of both arms, and all are exploratory.
  The other arm's fold training sets contained the gap days.
- `verify_aifs_steps.py --published-dir PUBLISHED --output-dir DIR` reads only. It checks, at each
  of days 1, 2, 7, and 14, that AIFS Single's radiation is a 6-hour mean ending at the lead and its
  wind is instantaneous (both against ERA5), and it checks the units and the crop's grid
  orientation (against GEFS), and writes `verification/aifs_steps.md`. With `--wiring`, after the
  build, it checks that the default ENS path is unchanged, the 6-hourly ENS columns are wired, and
  each built AIFS Single 100 m speed equals the raw store's weighted mean at the run and lead its
  day names, and writes `verification/aifs_wiring.md`. It exits non-zero on a failed check.
- `build_forecast_inputs.py --aifs --published-dir PUBLISHED --output-dir DIR` writes the AIFS
  Single and AIFS ENS columns at days 1 and 2 (the 00 UTC run of the day before, as ENS is read),
  ENS's mean and control member on 6-hourly steps, and each AIFS arm's and ENS arm's run time, onto
  the published inputs' `(site, time)` keys, to a new folder. `--aifs-days 1 2 7 14` builds the
  other days as well: day `N` reads the 00 UTC run `N` days before, at leads from `24 N` hours, and
  ENS's columns at days 7 and 14 are its native 6-hourly reads. Each site reads the H3 resolution-5
  overlap-weighted mean of the crop's cells, as ENS's stored table does; AIFS Single also has a
  nearest-cell arm at day 1. `--aifs-weather-dir` names the folder holding the two downloads. The
  build refuses the published folder and the day-1 and day-2 AIFS folder as its output.
- `fit_aifs.py` fits every AIFS arm and reference on a GPU, on two nested row sets (`single`, and
  `ens` where AIFS ENS also exists), with folds cut inside the AIFS version eras. One contrast is
  deciding (AIFS Single against ENS's control member at day 1); AIFS ENS contrasts are
  descriptive; all others are exploratory. `--check` fits one arm twice and prints a time
  estimate. A row set whose losses file exists is not refitted, so a rerun after a crash resumes
  where it stopped; `report.md` is always written once.
- `fit_aifs.py --blends` fits AIFS at days 1, 2, 7, and 14 and the blends of ENS's mean with AIFS
  Single (`single` row set) and with the AIFS ENS mean (`ens` row set), each blend with a control
  that shuffles AIFS within site, year-month, and hour of day (seed 0, and seed 1000 for the
  two-seed day-14 gate) and a mirror control that shuffles ENS's mean instead. Each (row set, day)
  is a stage with its own frame, losses, predictions, and stamp; an hour whose day-`N` run lies
  outside its AIFS version era is dropped row by row and its month kept. Four contrasts per
  technology are deciding, named before any fit but not planned in the published page's sense: H7
  and H14 (AIFS Single against ENS's control member) and B7 and B14 (a blend against ENS's mean
  alone and against its control). The report prints the day-14 reading rule, the smoothing reading
  beside H7 and H14, each forecast's weather-column spread, and every arm's columns. It reads the
  extra-lead folders beside the published one and checks that their ENS columns equal the build's.
  It never writes to the published folder, the day-1 and day-2 AIFS folder, or the extra-lead
  folders.
- `fit_aifs.py --p4-controls` refits the published P4a and P4b blends, their published control, a
  second control (the same shuffle under seed 1000 plus the product's index), and ENS's day-1 mean
  on a GPU, at both settings, on the published run's own rows and folds. The report gives each
  blend's contrast with ENS and with both controls, the seed-to-seed gap between the controls, and
  both arms' absolute errors beside every contrast. If the two controls disagree on the guard's
  verdict, the blend claim is unresolved.
- `nwp_forecast_charts.py --aifs-blends-dir DIR --output-dir DIR` draws only the AIFS lead chart
  (`nwp_forecast_<domain>_aifs_leads.svg`) from the blends fit's losses, so no other chart is
  rewritten, and prints each chart's caption, which is its title.
- `nwp_forecast_charts.py` reads the saved losses and predictions from `--input-dir`, and the extra
  lead days' losses from `--extra-dir`, and, with `--aifs-dir`, the AIFS losses, and writes six SVG
  charts per technology (seven with the AIFS chart) to `--output-dir`, each optimised with `svgo`
  (skip with `--no-svgo`): the leaderboard of every product's absolute error at every fitted lead
  day, the planned contrasts P1a to P4b at both settings, one chosen week of out-of-fold forecasts
  against measured output, the contrasts at each generator alone, the blends against ENS alone and
  their controls, and error by lead day with ENS's day-0 and day-1 intervals shaded. It computes
  every interval itself with the report's own functions and refuses any site label that is not an
  anonymised label. It has no default output directory: charts go to `docs/studies/assets/` only
  once a real report exists.

## Outputs

- `report.md` holds, per technology: every arm's column list; the rows each planned product drops;
  the UKV straddle share; both settings' leaderboards; the planned contrasts P1a to P4b with their
  verdicts and the ENS monotonicity table by 3-hour band; the blend verdict; and the exploratory
  brackets at every day, UKV's bracket by era, the exact-lead wind re-read, the ENS control
  member, and the ENS day-1 reconciliation with the horizons page. Every contrast row is labelled
  planned or exploratory.
- `<domain>_losses.parquet` holds one row per (arm, setting, site, time, seed) with the capped
  error in megawatts and as a fraction of capacity. `<domain>_predictions.parquet` holds the
  capped prediction for the same keys.
- `data/studies/nwp_forecast_comparison_leads/` holds the extra lead days and is write-once.
  `<domain>_extra_lead_inputs.parquet` holds the extra columns, `<domain>_losses.parquet` and
  `<domain>_predictions.parquet` the GPU fits in the same layout as the published files,
  `report.md` the absolute errors, contrasts, and device noise floor, and `verification/` the two
  checks of `verify_extra_leads.py`.
- `data/studies/nwp_forecast_comparison_aifs/` holds the AIFS arms and is write-once.
  `<domain>_aifs_inputs.parquet` holds the AIFS columns, `<domain>_<row_set>_losses.parquet` and
  `<domain>_<row_set>_predictions.parquet` the GPU fits in the published layout (`row_set` is
  `single` or `ens`), `report.md` the absolute errors, the deciding contrast at both settings, the
  listed contrasts, and the per-era contrasts, and `verification/` the checks of
  `verify_aifs_steps.py`.

- `data/studies/nwp_forecast_comparison_aifs_blends/` holds the blends fit and is write-once.
  `<domain>_aifs_inputs.parquet` holds the AIFS columns at days 1, 2, 7, and 14,
  `<domain>_<row_set>_day<N>_losses.parquet`, `.json`, and `_predictions.parquet` hold each stage's
  GPU fits, the stamp that names the device, the input files' SHA-256, and the seeds, and the
  stage's predictions, `report.md` the report, and `verification/` the checks of
  `verify_aifs_steps.py`.
- `data/studies/nwp_forecast_comparison_p4_seeds/` holds the P4 refit and is write-once.
  `<domain>_p4_losses.parquet`, `.json`, and `_predictions.parquet` hold the GPU fits, their stamp,
  and their predictions, and `report.md` the contrasts.

## Running the blends fit and the P4 refit

Run these in order, one job at a time, from the repository root. `D` is the shared data folder.

```bash
D=/home/jack/dev/nged-substation-forecast/data/studies
uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --aifs \
  --aifs-days 1 2 7 14 --published-dir $D/nwp_forecast_comparison \
  --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/verify_aifs_steps.py \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/verify_aifs_steps.py --wiring \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/fit_aifs.py --blends --check \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/fit_aifs.py --blends --workers 1 \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/fit_aifs.py --p4-controls --check \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_p4_seeds
uv run python studies/nwp_forecast_comparison/fit_aifs.py --p4-controls --workers 1 \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_p4_seeds
uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py \
  --aifs-blends-dir $D/nwp_forecast_comparison_aifs_blends --output-dir docs/studies/assets
```

After the blends fit and again after the P4 refit, check both baselines:

```bash
B=/home/jack/dev/nged-substation-forecast/.claude/worktrees/scratch/matched-lead
sha256sum -c $B/sha-before-aifs-blends.txt
sha256sum -c /tmp/claude-1000/pub-sha-before.txt
```

Before the first command, record a SHA-256 baseline of every file in the published folder, the
day-1 and day-2 AIFS folder, and the three extra-lead folders (`nwp_forecast_comparison_leads_day10`,
`_day10b`, and `_day10d`), and check it after the last fit. Check CPU load with `uptime` before each
`--check`, and run only one fit at a time. Each `--check` fits one arm at one wind site twice on
the GPU, stops unless the two fingerprints agree, and prints the fit's runtime estimate.

## Folds

`assign_folds_with_eras()` and `coverage_table()` use `cut_eras`, `calendar_month_coverage` and
`raise_on_uncovered_months` from `studies.cross_validation`. This study rotates the third
era's folds by 3 (`NWP_ERA_FOLD_OFFSETS`), because a rotation of 2 leaves two solar (site, fold,
calendar month) cells uncovered on these rows.
