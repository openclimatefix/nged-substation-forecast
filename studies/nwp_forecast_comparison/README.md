# Weather forecasts for power, compared at matched lead times

Study for [issue #810](https://github.com/openclimatefix/nged-substation-forecast/issues/810). The
question is which forecast product, or which blend of products, gives the most accurate power
forecast at the day-ahead lead the live service delivers, and at the two days after it. The design
and the results are on the published page, [Which weather forecast is best at day-ahead
lead?](../../docs/studies/forecasts/matched-lead.md). How the extra products (AIFS, WeatherNext 3,
the extra lead days, and day 0) are read, and their figureless results, are on its [companion
page](../../docs/studies/forecasts/matched-lead-extra-products.md).

## The scripts and the pages they feed

**Five pages draw on the scripts in this folder, so find a page's scripts in this table.** The
folder holds the matched-lead study, its extra products, the ENS-horizons study, the blends with
ENS, and the UKV-from-CEDA blends. Tests of these scripts are in
`packages/studies/tests/nwp_forecast_comparison/`. A script imports only from this folder, from
`studies.*` (`packages/studies/`), and from the other reviewed packages.

| Script | Published page |
|---|---|
| `nwp_forecast_comparison.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `build_forecast_inputs.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `verify_previous_runs_leads.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `check_input_steps.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `nwp_forecast_charts.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `leaderboard_by_day.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `fit_extra_leads.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `verify_extra_leads.py` | [Matched-lead extra products](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead-extra-products/) |
| `fit_aifs.py` | [Matched-lead extra products](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead-extra-products/) |
| `verify_aifs_steps.py` | [Matched-lead extra products](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead-extra-products/) |
| `verify_gfs_native.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `verify_ifs_single.py` | [Matched lead](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead/) |
| `build_wn3_inputs.py` | [Matched-lead extra products](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead-extra-products/) |
| `verify_wn3_steps.py` | [Matched-lead extra products](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead-extra-products/) |
| `fit_day5_aifs_wn3.py` | [Matched-lead extra products](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/matched-lead-extra-products/) |
| `fit_product_blends.py` | [Blends with ENS](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/blends-with-ens/) |
| `dot_interval_vs_ens.py` | [Blends with ENS](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/blends-with-ens/) |
| `fetch_ens_forecast_horizons.py` | [ENS horizons](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ens-horizons/) |
| `fetch_ens_day4_supplement.py` | [ENS horizons](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ens-horizons/) |
| `ens_forecast_horizons.py` | [ENS horizons](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ens-horizons/) |
| `ens_forecast_charts.py` | [ENS horizons](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ens-horizons/) |
| `build_ukv_ceda_inputs.py` | [UKV from CEDA](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ukv-ceda-blends/) |
| `verify_ukv_ceda_inputs.py` | [UKV from CEDA](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ukv-ceda-blends/) |
| `fit_ukv_ceda_blends.py` | [UKV from CEDA](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ukv-ceda-blends/) |
| `check_arm_columns_unchanged.py` | [UKV from CEDA](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ukv-ceda-blends/) |
| `ukv_ceda_blends_charts.py` | [UKV from CEDA](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ukv-ceda-blends/) |

## Scripts

- `fetch_ens_forecast_horizons.py` extracts every ECMWF ENS member's radiation, temperature, and 10
  m and 100 m wind at each metered generator's H3 cell, at the leads the horizon study scores, from
  the production NWP Delta table, into `data/studies/ens_forecast_horizons/ens_members.parquet`.
- `ens_forecast_horizons.py` scores ENS-driven power forecasts at eight horizons, for solar and
  wind, at hourly resolution on the past-weather studies' own rows: first how to upsample ENS's 3-
  and 6-hourly steps to hourly, then three ways of using the 51 members (the control member, the
  ensemble mean, and each member through the power model), four baselines that read no weather
  forecast, and two past-weather references that are not forecasts. Writes its losses, predictions,
  member-forecast summaries, intervals, leaderboard, and `report.md` to
  `data/studies/ens_forecast_horizons/`.
- `ens_forecast_charts.py` draws the ENS horizon page's anonymised charts from
  `ens_forecast_horizons.py`'s outputs, checking each number against the report.
- `fetch_ens_day4_supplement.py` extracts the ENS leads that the day-4 band needs and
  `ens_members.parquet` lacks.
- `verify_previous_runs_leads.py` runs three checks and writes one markdown table each under
  `--output-dir`. V1 (a gate) compares Open-Meteo's GFS `previous_dayN` values with the
  Dynamical.org GFS runs, for candidate run offsets `k` from 0 to 12 hours, and passes only if the
  mean absolute difference is lowest at `k = 0` for `N = 1` and `N = 2`. Each site scores every grid
  cell of the Dynamical.org extract and keeps the best, so no site coordinate is read. V1 fails if
  nothing is scored. V1b gives, per product, field and day, each UTC hour's mean absolute second
  difference of the `previous_dayN` series divided by its mean over all hours, and reads a 3- or
  6-hourly run cycle where the switch hours stand out by a ratio of 1.3 or more. V3 measures each
  product's radiation timestamp convention from the offset at which its `previous_day1` series
  tracks the sun best, and tests each candidate rebuild of UKV's hourly value from its snapshots;
  `--v3-only` runs V3 alone, and the study's `report.md` quotes `v3_conventions.md` from a
  `verification/` directory beside it. Outputs: `v1_wind.md`, `v1_radiation.md`,
  `v1b_run_switches.md`, `v3_conventions.md`.
- `check_input_steps.py` (V1c) divides each month's mean day-1 radiation and 100 m wind speed of UKV
  and IFS 0.25° by ICON-EU's at the same site, and flags a month-to-month change of 15% or more.
  Output: `v1c_steps.md`.
- `build_forecast_inputs.py` writes `solar_forecast_inputs.parquet` and
  `wind_forecast_inputs.parquet` under `--output-dir` (default
  `data/studies/nwp_forecast_comparison/`). Each file has one row per generator-hour with the
  target, the capacity, the calendar and sun-position columns, every Previous Runs product's weather
  columns at its planned day offsets (wind speeds in m/s), ECMWF ENS's mean at days 0 to 3 and
  control member at days 0 to 3, the no-weather baselines' inputs, and, where GEFS is built, the
  GEFS mean at days 1 to 3. ENS and GEFS are built directly at the chosen upsampling combination
  (`clear_sky` for solar, `speed_components` for wind) through the public functions of
  `studies/nwp_forecast_comparison/ens_forecast_horizons.py`. GEFS reads each site's nearest 0.25°
  cell, converts its alternating 3- and 6-hour radiation windows to 3-hour step means
  (`studies.resample.gefs_step_means`) on whole runs before any band is sliced, and averages the 31
  members. It runs only when `data/studies/weather/GEFS_window_2024-11-01_None/_month_cache/` covers
  2024-11 to the month the rows end on (the last month may be a `.partial.parquet`), or when
  `--gefs-window-dir` names a `GEFS_window_*` extract. The build raises if any 00 UTC run the rows
  need is missing or incomplete.
- `nwp_forecast_comparison.py` reads those files and takes the rows where the target, the baselines'
  inputs, and every planned arm's columns are present. It cuts folds with
  `studies.cross_validation.cut_eras`, fits every arm out of fold, scores the no-weather baselines,
  and writes `report.md`, `<domain>_losses.parquet` and `<domain>_predictions.parquet` under
  `--output-dir`. Every contrast is computed within one hyperparameter setting, and a planned
  verdict stands only if both settings give it. `--dry-run` stops after printing the coverage table
  and the job list. `--report-only` writes the report from the saved losses. `--fit-missing` fits
  only the (arm, setting) pairs the saved losses lack. `--synthetic-losses` fabricates losses
  instead of fitting, to exercise the report, and refuses to write under `data/studies/`.
- `verify_extra_leads.py` reads only, and writes two Markdown files under
  `<output-dir>/verification/`: `gefs_radiation_window.md` checks that GEFS's radiation beyond 240 h
  is a 6-hour window mean, and `day0_matches_past_series.md` checks that Open-Meteo's unsuffixed
  Previous Runs series, which the extra-lead build reads as day 0 for ICON-D2 and ICON-EU, equals
  the series the past-weather studies read.
- `build_forecast_inputs.py --extra-leads --published-dir PUBLISHED --output-dir DIR` writes the
  extra lead days' input columns (ENS at days 5 and 14, GEFS at days 5, 10, and 14, IFS 0.25°, GFS,
  and ICON global at day 5, and ICON-D2 and ICON-EU at day 0) onto the published inputs' own `(site,
  time)` keys, to a new folder.
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
  `--built-dir`, `gfs_built_columns.md` (a sample of built values recomputed from the store in plain
  Python) under `<output-dir>/verification/`. It exits non-zero if any check fails.
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
  wind is instantaneous (both against ERA5), and it checks the units and the crop's grid orientation
  (against GEFS), and writes `verification/aifs_steps.md`. With `--wiring`, after the build, it
  checks that the default ENS path is unchanged, the 6-hourly ENS columns are wired, and each built
  AIFS Single 100 m speed equals the raw store's weighted mean at the run and lead its day names,
  and writes `verification/aifs_wiring.md`. It exits non-zero on a failed check.
- `build_forecast_inputs.py --aifs --published-dir PUBLISHED --output-dir DIR` writes the AIFS
  Single and AIFS ENS columns at days 1 and 2 (the 00 UTC run of the day before, as ENS is read),
  ENS's mean and control member on 6-hourly steps, and each AIFS arm's and ENS arm's run time, onto
  the published inputs' `(site, time)` keys, to a new folder. `--aifs-days 1 2 7 14` builds the
  other days as well: day `N` reads the 00 UTC run `N` days before, at leads from `24 N` hours, and
  ENS's columns at days 7 and 14 are its native 6-hourly reads. Each site reads the H3 resolution-5
  overlap-weighted mean of the crop's cells, as ENS's stored table does; AIFS Single also has a
  nearest-cell arm at day 1. `--aifs-weather-dir` names the folder holding the two downloads. The
  build refuses the published folder and the day-1 and day-2 AIFS folder as its output.
- `build_forecast_inputs.py --aifs --aifs-days 0 3 4 10` builds day 0 and days 3, 4, and 10 as well.
  Day 0 reads the 00 UTC run of the same day as the hour, so its weather is a hindcast that a
  service could not have read. AIFS and ENS on 6-hourly steps have no step before lead 6 hours
  (radiation is null at lead 0), and a solar temperature is read at each hour's midpoint, so solar
  day 0 omits the hours ending 01:00 to 06:00 UTC for every arm (AIFS Single, ENS, and WeatherNext
  3), which keeps the rows matched. The build raises if any other scored solar hour's midpoint lies
  before the first step. Wind day 0 is unaffected. WeatherNext 3 stores no lead for one hour of each
  day at day 0 (00:00 UTC for wind, 01:00 UTC for solar), so the wind fit drops that hour and the
  solar drop above already covers it.

  The two fits write into two separate existing folders that hold only a `README.md`, because both
  write `single_day0_*` files and a `report.md`, and each output is written once. The paths are
  absolute because a worktree has no `data/` folder. Run one command at a time:

  ```bash
  D=/home/jack/dev/nged-substation-forecast/data/studies
  P=$D/nwp_forecast_comparison
  LEAN=$D/nwp_forecast_comparison_aifs_extra_days
  WN3=$D/nwp_forecast_comparison_wn3_extra_days
  uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --aifs \
    --aifs-days 0 3 4 10 --published-dir $P --output-dir $LEAN
  uv run python studies/nwp_forecast_comparison/fit_aifs.py --lean-leads --check \
    --days 0 3 4 10 --published-dir $P --output-dir $LEAN
  uv run python studies/nwp_forecast_comparison/fit_aifs.py --lean-leads \
    --days 0 3 4 10 --workers 8 --published-dir $P --output-dir $LEAN
  uv run python studies/nwp_forecast_comparison/build_wn3_inputs.py --build \
    --days 0 3 4 10 --published-dir $P --output-dir $WN3
  uv run python studies/nwp_forecast_comparison/fit_aifs.py --wn3 --check --lookahead-cleared \
    --days 0 3 4 10 --published-dir $P --output-dir $WN3
  uv run python studies/nwp_forecast_comparison/fit_aifs.py --wn3 --lookahead-cleared \
    --days 0 3 4 10 --workers 8 --published-dir $P --output-dir $WN3
  ```

  `--lean-leads` fits only the AIFS Single arm and the ENS mean arm at those days, at the primary
  setting, which is 144 (arm, site) fits. `--wn3` fits the WeatherNext 3 arms at them, which is 156
  primary fits and the sensitivity refits on top. Each `--check` fits one arm twice, prints the time
  and the estimated total, and fits nothing else.
- `fit_aifs.py` fits every AIFS arm and reference on a GPU, on two nested row sets (`single`, and
  `ens` where AIFS ENS also exists), with folds cut inside the AIFS version eras. One contrast is
  deciding (AIFS Single against ENS's control member at day 1); AIFS ENS contrasts are descriptive;
  all others are exploratory. `--check` fits one arm twice and prints a time estimate. A row set
  whose losses file exists is not refitted, so a rerun after a crash resumes where it stopped;
  `report.md` is always written once.
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
- `build_forecast_inputs.py --extra-leads --batch fifth` writes day 4 of nine products onto the
  published inputs' own `(site, time)` keys: ENS's mean and control member (reading the supplement
  `fetch_ens_day4_supplement.py` writes, so the band has no missing step), GEFS's mean, the native
  GFS arm, IFS HRES (9 km, Open-Meteo), and the Previous Runs arms ICON-EU, ICON global, IFS 0.25°,
  and GFS at `previous_day4`. ICON-EU's archive ends at day 4. `previous_day4` is null on at most
  0.6% of the published rows for those four products.
- `fit_extra_leads.py --batch fifth` fits those nine arms on a GPU at the primary setting, on the
  shared rows and folds of the earlier extra-lead folders, with no negative control (no extra-lead
  batch has one). It writes only to a folder named `nwp_forecast_comparison_day4_shared`. Like the
  fourth batch, it scores each arm without the rows where the arm's own columns are null and
  computes each contrast on the rows both arms score. Its contrasts are each arm against the ENS
  mean at day 4, and against the same product at day 3, read from all four earlier batches through
  four `--context-dir` folders.
- `fit_day5_aifs_wn3.py` fits day 5 of AIFS Single, the AIFS ENS mean, and WeatherNext 3, each
  beside ENS's mean, by calling `fit_aifs.run_lean` and `fit_aifs.run_wn3` at day 5, with the
  settings of the days already fitted. It writes only to a folder named
  `nwp_forecast_comparison_day5_aifs_wn3`: the two fits' reports as `report_aifs.md` and
  `report_wn3.md`, and `report.md` joining them. `build_wn3_inputs.py --build` raises if a stored
  WeatherNext 3 run that a band reads holds a `NaN`, and `build_forecast_inputs.py` raises on any
  missing step inside an AIFS or ENS band.
- `nwp_forecast_charts.py --aifs-blends-dir DIR --output-dir DIR` draws only the AIFS lead chart
  (`nwp_forecast_<domain>_aifs_leads.svg`) from the blends fit's losses, so no other chart is
  rewritten, and prints each chart's caption, which is its title.
- `leaderboard_by_day.py` draws the MAE leaderboard of every product at every fitted lead day as one
  stacked panel per day on one shared x axis, reading the published fit, the extra-lead fits
  (including the day-4 folder), the AIFS and WeatherNext 3 fits (including the day-5 folder) from
  `data/studies/`, and raising if any expected file is missing. It writes
  `nwp_forecast_<domain>_leaderboard.svg`, plus a `report.md` and `marks.parquet` of every plotted
  mark, once.
- `nwp_forecast_charts.py` reads the saved losses and predictions from `--input-dir`, and the extra
  lead days' losses from `--extra-dir`, and, with `--aifs-dir`, the AIFS losses, and writes five SVG
  charts per technology (six with the AIFS chart) to `--output-dir`, each optimised with `svgo`
  (skip with `--no-svgo`): the planned contrasts P1a to P4b at both settings, one chosen week of
  out-of-fold forecasts against measured output, the contrasts at each generator alone, the blends
  against ENS alone and their controls, and error by lead day with ENS's day-0 and day-1 intervals
  shaded. It computes every interval itself with the report's own functions and refuses any site
  label that is not an anonymised label. It has no default output directory: charts go to
  `docs/studies/assets/` only once a real report exists.
- `dot_interval_vs_ens.py` fits nothing. It reads the saved losses the leaderboards draw (the
  `leads_day10*` folders, the AIFS and WeatherNext 3 folders for days 1, 2, 7, and 14, their
  `extra_days` folders for days 0, 3, 4, and 10, `day4_shared` for day 4, and `day5_aifs_wn3` for
  day 5) and draws, for each technology, one panel per leaderboard lead day (0, 1, 2, 3, 4, 5, 7,
  10, and 14) of each product's error minus the ENS mean's, as a dot with a 95% interval from
  resampling whole months and a fitting seed. A row with fewer than 6 months gets a hollow dot and
  no interval. Each product is subtracted from the ENS-mean arm the leaderboard draws at the same
  day: `_leads_day10b` at days 2 and 7, `_day4_shared` at day 4, and `_leads_day10` elsewhere for
  the `leads_day10*` products, and the ENS mean in the product's own file for AIFS and WeatherNext
  3. WeatherNext 3's wind reference is `ens_meanvec`. Because those three are fitted on 16, 11, and
     7 months, each of their rows also has a hollow diamond against the leaderboard's 21-month ENS
     mean on the keys the two share, and an interval from fewer than 12 months is dashed. IFS HRES 9
     km lacks 1,197 to 1,536 of the ENS mean's rows at each day, and ICON global at day 4 lacks 288
     for solar, so their paired differences drop those rows; any other arm whose rows differ from
     its reference's makes the script raise. It writes `report.md`, `intervals.parquet`, and a
     `README.md` naming each row's reference to a new `nwp_forecast_comparison_vs_ens_dots_final`
     folder, and two SVGs to `docs/studies/assets/`. It checks that none of the five outputs exists
     before it writes any, and refuses to overwrite them. With `--blends` (and its own
     `--output-dir`) it draws, in place of the single products, each blend of ENS's mean with one
     product at days 1, 2, 7, and 14 where the blend exists, minus the ENS mean alone, at the
     primary setting, and adds `rankings.parquet` and a report section for comparison C5, the AIFS
     Single blend minus each ICON-EU blend. The AIFS Single blends are read from
     `nwp_forecast_comparison_aifs_blends`, and the ICON-EU, UKV, and WeatherNext 3 blends from
     `nwp_forecast_comparison_product_blends`.
- `fit_product_blends.py` fits ENS plus one product for ICON-EU (optimistic and conservative lead,
  days 1 and 2), UKV (day 1), and WeatherNext 3 (days 1, 2, 7, and 14 on the 7 WeatherNext 3
  months), each with a control that shuffles the product's columns. It reuses the AIFS Single blends
  of `nwp_forecast_comparison_aifs_blends`, refits only the (arm, setting) pairs that folder lacks,
  and raises unless its build stamp equals that folder's and every new arm holds the saved ENS
  mean's `(site, time, seed, fold)` keys. It calls `fit_aifs.product_blend_arms`, which leaves
  `blend_arms`, `stage_arms_fitted`, and `wn3_arms` unchanged, and writes only to a folder named
  `nwp_forecast_comparison_product_blends`. `--dry-run` lists the fits without fitting, and
  `--check` fits one arm twice on the GPU. Its `report.md` holds every arm's columns and the
  contrasts C1 to C5 at both settings.

## Outputs

- `report.md` holds, per technology: every arm's column list; the rows each planned product drops;
  the UKV straddle share; both settings' leaderboards; the planned contrasts P1a to P4b with their
  verdicts and the ENS monotonicity table by 3-hour band; the blend verdict; and the exploratory
  brackets at every day, UKV's bracket by era, the exact-lead wind re-read, the ENS control member,
  and the ENS day-1 reconciliation with the horizons page. Every contrast row is labelled planned or
  exploratory.
- `<domain>_losses.parquet` holds one row per (arm, setting, site, time, seed) with the capped error
  in megawatts and as a fraction of capacity. `<domain>_predictions.parquet` holds the capped
  prediction for the same keys.
- `data/studies/nwp_forecast_comparison_leads/` holds the extra lead days and is write-once.
  `<domain>_extra_lead_inputs.parquet` holds the extra columns, `<domain>_losses.parquet` and
  `<domain>_predictions.parquet` the GPU fits in the same layout as the published files, `report.md`
  the absolute errors, contrasts, and device noise floor, and `verification/` the two checks of
  `verify_extra_leads.py`.
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

Before the first command, record a SHA-256 baseline of every file in the published folder, the day-1
and day-2 AIFS folder, and the three extra-lead folders (`nwp_forecast_comparison_leads_day10`,
`_day10b`, and `_day10d`), and check it after the last fit. Check CPU load with `uptime` before each
`--check`, and run only one fit at a time. Each `--check` fits one arm at one wind site twice on the
GPU, stops unless the two fingerprints agree, and prints the fit's runtime estimate.

## Running the day-4 and day-5 fits

Run these in order, one job at a time, from the repository root, after `uptime` shows the CPU is
idle. `D` is the shared data folder.

```bash
D=/home/jack/dev/nged-substation-forecast/data/studies
P=$D/nwp_forecast_comparison
DAY4=$D/nwp_forecast_comparison_day4_shared
DAY5=$D/nwp_forecast_comparison_day5_aifs_wn3
uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --extra-leads \
  --batch fifth --published-dir $P --output-dir $DAY4
CONTEXT="--context-dir $D/nwp_forecast_comparison_leads_day10 \
  --context-dir $D/nwp_forecast_comparison_leads_day10b \
  --context-dir $D/nwp_forecast_comparison_leads_day10c \
  --context-dir $D/nwp_forecast_comparison_leads_day10d"
uv run python studies/nwp_forecast_comparison/fit_extra_leads.py --batch fifth --check \
  --published-dir $P --output-dir $DAY4 $CONTEXT
uv run python studies/nwp_forecast_comparison/fit_extra_leads.py --batch fifth --workers 2 \
  --published-dir $P --output-dir $DAY4 $CONTEXT
uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --aifs --aifs-days 5 \
  --published-dir $P --output-dir $DAY5
uv run python studies/nwp_forecast_comparison/build_wn3_inputs.py --build --days 5 \
  --published-dir $P --output-dir $DAY5
uv run python studies/nwp_forecast_comparison/fit_day5_aifs_wn3.py --check --lookahead-cleared \
  --published-dir $P --output-dir $DAY5
uv run python studies/nwp_forecast_comparison/fit_day5_aifs_wn3.py --lookahead-cleared \
  --workers 8 --published-dir $P --output-dir $DAY5
```

Before the first command, record a SHA-256 baseline of every file in the published folder and in
each earlier extra-lead, AIFS, and WeatherNext 3 folder, and check it after the last fit.

## Running the product blends fit

Run it after `uptime` shows the CPU is idle and `nvidia-smi` shows the GPU is free. The saved AIFS
blend fits took about 8 to 10 s each, so the 315 (arm, site) fits take about 1 hour with 2 workers.
`--dry-run` builds every frame, checks the build stamp against
`nwp_forecast_comparison_aifs_blends`, and lists the fits without fitting.

```bash
D=/home/jack/dev/nged-substation-forecast/data/studies
P=$D/nwp_forecast_comparison
OUT=$D/nwp_forecast_comparison_product_blends
uv run python studies/nwp_forecast_comparison/fit_product_blends.py --dry-run --lookahead-cleared \
  --published-dir $P --output-dir $OUT
uv run python studies/nwp_forecast_comparison/fit_product_blends.py --check --lookahead-cleared \
  --published-dir $P --output-dir $OUT
uv run python studies/nwp_forecast_comparison/fit_product_blends.py --lookahead-cleared \
  --workers 2 --published-dir $P --output-dir $OUT
uv run python studies/nwp_forecast_comparison/dot_interval_vs_ens.py --blends \
  --output-dir $D/nwp_forecast_comparison_blends_vs_ens_dots
```

## Folds

`assign_folds_with_eras()` and `coverage_table()` use `cut_eras`, `calendar_month_coverage` and
`raise_on_uncovered_months` from `studies.cross_validation`. This study rotates the third era's
folds by 3 (`NWP_ERA_FOLD_OFFSETS`), because a rotation of 2 leaves two solar (site, fold, calendar
month) cells uncovered on these rows.

## Does UKV-CEDA lower the error of the ECMWF ENS mean at lead days 1 to 4?

Study for [issue #1016](https://github.com/openclimatefix/nged-substation-forecast/issues/1016). The
question is whether adding the Met Office's UKV (its 2 km UK variable-resolution weather model),
read from the CEDA archive (the Centre for Environmental Data Analysis), to the mean of the European
Centre for Medium-Range Weather Forecasts (ECMWF) ensemble forecast (ENS) lowers the power-forecast
error at lead days 1 to 4, for solar and for wind separately. The rows, eras, folds, hyperparameter
settings, seeds, metric, and paired month-resampled intervals follow [Which weather forecast is best
at day-ahead lead?](../../docs/studies/forecasts/matched-lead.md). The shuffled control and the
reading rule follow [ENS plus one weather product](../../docs/studies/forecasts/blends-with-ens.md).

**The planned blend reads UKV-CEDA's 03 UTC run (UTC is Coordinated Universal Time).** An hour on
day `D` at lead day `N` reads the 03 UTC run of day `D - N` from the `UKV-CEDA-T120` store, at lead
`24 * N + h - 3` hours, where `h` is the hour of day. A service running at 09:00 UTC could read that
run. The lead is 21 to 117 hours over days 1 to 4, inside the store's 120 hours. Day 5 cannot be
built, because its lead exceeds 120 hours for every hour after 03:00 UTC. The UKV-CEDA lead is 3
hours shorter than ENS's lead at every hour, which favours the blend.

**Four arms are fitted per technology and lead day, each at both hyperparameter settings.** An arm
is one XGBoost model per generator, and the arms hold equal column counts (9 for solar, 11 for
wind).

| Arm | Columns after the calendar columns |
|---|---|
| `blend_ukv_ceda_dayN_pad` | ENS's mean, then exact copies of the same columns |
| `blend_ukv_ceda_dayN` | ENS's mean, then UKV-CEDA |
| `blend_ukv_ceda_dayN_control` | ENS's mean, then UKV-CEDA shuffled within generator, year-month, and hour of day (seed 0) |
| `blend_ukv_ceda_dayN_control_b` | the same shuffle under seed 1000 |

UKV-CEDA's wind columns are its native 10 m speed, the sine and cosine of its 10 m direction, and
its 925 hPa speed. None of these columns is the 100 m wind that ENS carries. Two contrasts are
planned before any fit. P1 is the blend minus the padded ENS arm. P2 is the blend minus each
control. The blend lowers the error only if the upper 95% bound of P1 and of both P2 contrasts is
below zero at both settings.

### Run order

Each step after the first needs the step before it to have exited 0. Run `uptime` and `nvidia-smi`
before the build and before every fit, and start only below a load average of about 24.

1. `uv run python studies/nwp_forecast_comparison/check_arm_columns_unchanged.py` reads every
   `*_losses.json` stamp under `data/studies/nwp_forecast_comparison_*` and exits 0 if each arm's
   recorded columns still equal `fit_aifs.arm_features`. The check proves that the one branch added
   to `nwp_forecast_comparison._wind_weather_fields` leaves every earlier arm alone.
2. `uv run python studies/nwp_forecast_comparison/build_ukv_ceda_inputs.py --dry-run` builds one
   month (`--dry-run-month`, default 2026-03), prints the rows lost to each cause, and writes
   nothing.
3. `uv run python studies/nwp_forecast_comparison/build_ukv_ceda_inputs.py` writes
   `<domain>_ukv_ceda_inputs.parquet`, `build.json`, and `README.md` into the write-once folder
   `data/studies/ukv_ceda_blends/`. The build refuses to run until the download covers the window
   and every run slot the store marks as never archived has been fetched again once. Name the days
   CEDA still does not list in `--unlisted-days`.
4. `uv run python studies/nwp_forecast_comparison/verify_ukv_ceda_inputs.py` recomputes a stratified
   sample of built values in plain Python, gates the radiation timestamp at day 1 (see "The
   radiation timestamp" below), compares each lead day's correlation with the Copernicus Atmosphere
   Monitoring Service (CAMS) and ECMWF's fifth reanalysis (ERA5) against Open-Meteo UKV day 1 on the
   rows all four lead days hold, and screens for month-to-month steps. The script writes
   `verify.json`, holding whether every gating check passed and the SHA-256 of each inputs file,
   into the build's folder.
5. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --dry-run` builds every
   frame, checks every arm's columns, and lists the fits.
6. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --check` fits one arm at
   one wind site twice on the graphics processing unit (GPU), stops unless the two fingerprints
   agree, prints a time estimate, and compares ENS's mean alone with its padded copy on the wind
   day-1 rows.
7. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --verified` fits every
   stage, writes the losses once, refits the wind day-1 blend at one site on the central processing
   unit (CPU), and writes `report.md` and `intervals.parquet`. Add `--only-missing` to fit the pairs
   no saved file holds into a new `_added_<k>` file, and `--report-only --report-name NAME` to write
   the report again from the saved losses into new files.
8. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --post-hoc-stale --dry-run`
   lists the post hoc stale-blend fits (see "Post hoc stale blend" below) and the rows each stage
   loses to the blend's extra columns.
9. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --post-hoc-stale
   --report-name report_2` fits the stale blend into new `_added_<k>` files, never touching the
   planned fits, and writes `report_2.md` and `report_2_intervals.parquet`.
10. `uv run python studies/nwp_forecast_comparison/ukv_ceda_blends_charts.py --figures-dir DIR
    --intervals-name report_2_intervals.parquet` draws every figure from the saved intervals and
    losses without fitting.

11. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --post-hoc-permutation
    --dry-run` lists the post hoc permutation test's fits (see "Post hoc permutation test" below),
    and the same command with `--report-name report_3` instead of `--dry-run` fits the controls into
    new `_added_<k>` files and writes `report_3.md` and `report_3_intervals.parquet`.
12. `uv run python studies/nwp_forecast_comparison/build_ukv_ceda_inputs.py --older-run --dry-run`,
    then without `--dry-run` and with the same `--unlisted-days`, builds the older-run inputs into
    the write-once folder `data/studies/ukv_ceda_blends_run15/`. `verify_ukv_ceda_inputs.py
    --older-run` verifies them. `fit_ukv_ceda_blends.py --post-hoc-older-run --report-name report_4`
    fits the older-run blend and writes `report_4.md` (see "Post hoc older run" below).
13. `uv run python studies/nwp_forecast_comparison/ukv_ceda_blends_charts.py --post-hoc-only
    --figures-dir DIR --intervals-name report_4_intervals.parquet` draws the permutation and
    older-run figures.
14. `uv run python studies/nwp_forecast_comparison/fit_ukv_ceda_blends.py --report-only
    --post-hoc-older-run --report-name report_5` writes `report_5.md` and
    `report_5_intervals.parquet` from the saved losses, fitting nothing. The report adds the day-1
    older-run split at lead 48 hours, the share of the planned gain each post hoc blend keeps, and
    each arm's error at each generator. `ukv_ceda_blends_charts.py --figures-dir DIR
    --intervals-name report_5_intervals.parquet` then draws every figure of the page.

### Outputs the report adds after the first science review

- **Three readings.** A technology and lead day reads `lowers the error`, `no detectable
  difference`, or `unresolved: lower than padded ENS, control test not passed`. The third reading
  applies where P1 is below zero at both settings and some P2 bound is not.
- **Bonferroni at both settings.** The report prints P1's 99.375% interval at the primary and the
  sensitivity setting, and the readings table says whether P1 stays below zero at both. The planned
  rule is unchanged: the correction is an extra column, not part of the rule.
- **Largest gain left open.** For every stage the readings table gives the largest gain P1's lower
  bound does not exclude, at each setting. `no detectable difference` never means no gain.
- **Control gap.** The report prints seed-0 control minus seed-1000 control at both settings. The
  two controls carry the same (no) information, so the gap is the size of difference the pipeline
  produces from nothing. The interval resamples months and a fitting seed, not shuffle seeds.
- **Leave-one-month-out.** Post hoc and exploratory, the report recomputes every planned P1 with
  each calendar month dropped in turn from the saved losses, and prints the lowest and highest point
  estimate, the month that moves it most, and the highest 95% upper bound.
- **Padding check.** `--check` runs ENS's mean alone against its padded copy at wind day 1 and solar
  day 1, and writes `padding_check.json` once. The report prints the result.
- **`intervals.parquet` rows.** Beside the planned and exploratory rows, `scope` can be `Bonferroni`
  (P1 at `level` 99.375), `control gap` (`contrast` `control_gap`), `E6 most influential month
  dropped`, `generator error <site>: <arm>` (contrast `generator_error`, the arm's mean absolute
  error at one generator, with no interval), `post hoc stale` (contrasts `stale_p1`, `stale_p2`,
  `stale_p2b`, `stale_vs_fresh`, `fresh_p1_same_rows`, and `training_rows`), and the `error`
  contrast, whose `difference` is an arm's own mean absolute error and whose `scope` is the arm. The
  charts read every figure number from these rows.

### Post hoc stale blend

**The stale blend moves UKV-CEDA's run 24 hours earlier, which changes its lead, its start against
ENS, and at days 1 and 2 its time resolution together.** The arm `blend_ukv_ceda_stale_dayN` is ENS
day `N` plus UKV-CEDA day `N + 1` for `N` of 1 to 3, with the same column counts as the planned
arms. UKV-CEDA's run is the 03 UTC run one day before ENS's run, 21 hours staler than ENS's run,
where the planned blend's run is 3 hours fresher. The arm is fitted for both technologies at both
settings with its own padded ENS reference (`blend_ukv_ceda_stale_dayN_pad`) and both shuffled
controls, all trained on the stage's rows where UKV-CEDA's day `N + 1` is present, about 0.4% fewer
rows than the planned arms. Every contrast scores those rows. The report prints the planned padded
reference minus the stale-rows padded reference as the measured tilt from the planned reference's
extra training rows. The fit count is 216 (arm, site) fits: 3 lead days, 2 settings, 4 arms, and 9
generators. The stale blend is a post hoc, exploratory analysis added after the first science
review.

### Post hoc permutation test

**The permutation test asks whether the solar blend's gain is larger than shuffled controls give by
chance.** For each solar lead day, `--post-hoc-permutation` fits 15 further shuffled controls
(`blend_ukv_ceda_dayN_control_s<seed>`, seeds 2010 to 2150 in steps of 10) at the primary setting
only. Each further control shuffles UKV-CEDA's columns within generator, year-month, and hour of
day, with the same groups as the planned control. The report prints the planned blend's P1 against
the 17 values of (shuffled control minus padded ENS) from the planned two controls and the 15 extra
controls, the rank of P1 among the 18 values, and the one-sided permutation p-value `(1 + controls
at or below P1) / 18`, whose smallest possible value is 0.056. The test is exploratory and post hoc.
The test fits 6 generators x 4 lead days x 15 seeds, 360 fits.

### Post hoc older run

**The older-run blend tests how the gain falls as the UKV-CEDA run gets older.** The arm
`blend_ukv_ceda_run15_dayN` is ENS day `N` plus UKV-CEDA columns built from the 15 UTC run of the
day before ENS's run, for `N` of 1 to 3. For an hour on day `D` at lead day `N`, the run starts at
15:00 UTC on day `D - N - 1` and the lead is `24 * N + h + 9` hours (`h` the hour of the hour's
instant, plus 1 for a solar label). That run starts 9 hours before ENS's 00 UTC run of day `D - N`,
where the planned blend's run starts 3 hours after ENS's run, and leads 12 hours longer than the
planned run. Day 4 cannot be built: its lead reaches 129 hours, beyond the store's 120, for every
hour after 14:00 UTC. The two changes, a longer lead and an earlier start against ENS, move
together, so the arm cannot separate the effect of the lead from the effect of the timing. The store
is hourly only to lead 48 hours, so the older run also has more hours rebuilt from 3-hourly steps
than the planned run has at days 1 and 2. At day 1 the older run is read at a lead of 33 to 56
hours. The report splits the day-1 rows at lead 48 hours, because both runs are hourly on the rows
at or before that lead. The arm has its own padded ENS reference refitted on its rows, one shuffled
control (seed 0), and equal column counts (9 for solar, 11 for wind), at both settings, for both
technologies. The report prints the older-run blend minus its padded ENS (P1), minus its control
(P2), minus the planned blend, and the planned P1 on the same rows. The older-run inputs are built
by `build_ukv_ceda_inputs.py --older-run` with the same coverage guard, stamp checks, and init-time
assertions as the main build, into `data/studies/ukv_ceda_blends_run15/`.

### Scripts

- `build_ukv_ceda_inputs.py` reads the 03 UTC runs of the `UKV-CEDA-T120` Icechunk store, takes each
  generator's nearest 2 km cell, and rebuilds the leads the store lacks. The store holds hourly
  steps to lead 48 hours, then 51 and 54, then every third hour to 120. Radiation is rebuilt through
  the clear-sky index at each instant, temperature by a straight line, and wind through its eastward
  and northward components. A rebuilt value is missing unless both steps either side of it are
  finite. Solar radiation is the mean of the two snapshots at leads `L - 1` and `L`.
- `verify_ukv_ceda_inputs.py` only reads, and exits non-zero if a gating check fails.
- `fit_ukv_ceda_blends.py` imports `fit_aifs` and `nwp_forecast_comparison` from the sibling
  `studies/nwp_forecast_comparison/` directory and changes neither. A stage whose losses exist is
  not refitted, and every output is written once.
- `ukv_ceda_blends_charts.py` draws the headline figure (both settings' intervals, a title and panel
  titles that state the reading), the per-generator figure, the per-arm absolute-error figure with
  no intervals, the per-generator absolute-error figure, the permutation and older-run figures, and
  the week figures from the saved intervals, losses, and predictions.
- `check_arm_columns_unchanged.py` is described in step 1. The check cannot cover the main
  matched-lead fit, which wrote no stamp. No arm prefix there starts with `ukv_ceda`, so the new
  branch of `_wind_weather_fields` is never reached by that fit.

### Changes to shared code

- `studies.ifs_single_runs.served_init_time` and `served_lead_hours` take `run_hour`, the UTC hour
  at which the run starts (default 0).
- `fit_aifs.BLEND_AIFS_PREFIXES` has `ukv_ceda` and `ukv_ceda_stale` entries, and `BlendRoleType`
  has the roles `_pad` and `_control_b`.
- `nwp_forecast_comparison._wind_weather_fields` returns the 10 m and 925 hPa column names for a
  prefix that starts with `ukv_ceda`.

### The radiation timestamp

The day-1 gate in `verify_ukv_ceda_inputs.py` is the median over the solar generators of two peak
offsets, each the offset at which the radiation correlates best with the cosine of the solar zenith
angle. (a) The raw native day-1 snapshots, which are instants, must peak within 15 minutes of 0. (b)
The rebuilt column, a mean of the snapshots at `L - 1` and `L`, must peak 30 plus or minus 10
minutes before (a). The snapshots are not reweighted to land on -30 minutes.

### Reading the results

- UKV-CEDA's radiation is centred about 20 minutes later than the power hour. On the 262 complete 03
  UTC runs of 2026-01-02 to 2026-09-28, the raw snapshots peaked about 10 minutes after their stamp
  (15 minutes at one generator), so the mean of two snapshots peaks about 20 minutes before its
  label instead of 30. Open-Meteo's UKV peaked 13 minutes before its stamp. The offset is a property
  of the archive that no construction choice can remove. The study does not measure how the offset
  changes the error.

- UKV-CEDA is the archive of the Met Office's UKV, which is statistically different from the live
  UKV feed. An XGBoost model trained on UKV-CEDA must not be run on live UKV.
- The 09:00 UTC delivery time of the 03 UTC run is an assumption that has not been measured for
  CEDA.
- The page that reports the results is `docs/studies/forecasts/ukv-ceda-blends.md`.
