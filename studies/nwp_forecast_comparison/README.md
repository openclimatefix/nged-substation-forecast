# Weather forecasts for power, compared at matched lead times

Study for [issue #810](https://github.com/openclimatefix/nged-substation-forecast/issues/810). The
question is which forecast product, or which blend of products, gives the most accurate power
forecast at the day-ahead lead the live service delivers, and at the two days after it. The design
is in `plans/nwp-forecast-comparison.md` on the study's branch.

## Scripts

- `verify_previous_runs_leads.py` runs three checks and writes one markdown table each under
  `--output-dir`. V1 (a gate) compares Open-Meteo's GFS `previous_dayN` values with the Dynamical.org
  GFS runs, for candidate run offsets `k` from 0 to 12 hours, and passes only if the mean absolute
  difference is lowest at `k = 0` for `N = 1` and `N = 2`. Each site scores every grid cell of the
  Dynamical.org extract and keeps the best, so no site coordinate is read. V1 fails if nothing is
  scored. V1b gives, per product, field and day, each UTC hour's mean absolute second difference of
  the `previous_dayN` series divided by its mean over all hours, and reads a 3- or 6-hourly run
  cycle where the switch hours stand out by a ratio of 1.3 or more. V3 reads each product's
  timestamp convention from its `lineage.json`. Outputs: `v1_wind.md`, `v1_radiation.md`,
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
  sliced, and averages the 31 members. It runs only when `data/studies/weather/GEFS/_month_cache/`
  covers 2024-11 to the month the rows end on, or when `--gefs-window-dir` names a `GEFS_window_*`
  extract.
- `nwp_forecast_comparison.py` reads those files and takes the rows where the target, the baselines'
  inputs, and every planned arm's columns are present. It cuts folds with #885's era helper, fits
  every arm out of fold, scores the no-weather baselines, and writes `report.md`,
  `<domain>_losses.parquet` and `<domain>_predictions.parquet` under `--output-dir`. Every contrast
  is computed within one hyperparameter setting, and a planned verdict stands only if both settings
  give it. `--dry-run` stops after printing the coverage table and the job list. `--report-only`
  writes the report from the saved losses. `--fit-missing` fits only the (arm, setting) pairs the
  saved losses lack. `--synthetic-losses` fabricates losses instead of fitting, to exercise the
  report, and refuses to write under `data/studies/`.

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

## Folds and #885

`assign_folds_with_eras()` and `coverage_table()` import `cut_eras`, `calendar_month_coverage` and
`raise_on_uncovered_months` from `studies.cross_validation` inside the function body, because PR #885
adds them and it is not yet on `main`. Each is marked `TODO(#885)`. This study rotates the third
era's folds by 3 (`NWP_ERA_FOLD_OFFSETS`), because a rotation of 2 leaves two solar (site, fold,
calendar month) cells uncovered on these rows.
