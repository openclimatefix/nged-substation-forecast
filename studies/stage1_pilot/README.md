# Stage 1 pilot: does a weather-to-power XGBoost model trained on ERA5 and CAMS improve the ENS point forecast?

This is study 2a of the staged probabilistic forecast plan ([PR #1183 (Plan: document a staged
probabilistic forecast, and schedule the studies that test it)](https://github.com/openclimatefix/nged-substation-forecast/pull/1183)).
It runs on the nine metered generator series only (6 solar farms, 3 wind farms), because ERA5 and
CAMS are on disk only for them. It is exploratory and autonomous: it changes no production code and
imports only from this folder, `studies.*`, and the reviewed packages in `packages/*`. This README
was written before the first fit. Everything not listed under "Planned contrasts" is exploratory.

## Question

**Does a weather-to-power XGBoost model trained on ERA5 and CAMS (a "perfect prognosis" model,
2019-09 onward), applied to ECMWF ENS weather, improve the ENS-era point forecast compared with the
XGBoost model trained on ENS alone?** If it adds nothing, stage 1 leaves the forecast chain.

## Rows and folds

- **Fold A of the quick-wins study, unchanged.** The ENS-trained models train on the control member
  (member 0) for valid times 2024-04 to 2025-06. They are scored on the 53 ECMWF runs initialised
  every 7 days from 2025-07-01, all 51 members, for valid times to 2026-06-30.
- **Scored rows are the half-hours with an observed power**, the rows the stored control was scored
  on. Each error is divided by the series' effective capacity from the fold A frames.
- **Fold B is not run.** It is the follow-up if fold A finishes with time to spare.

## Arms

All arms are scored on the same cases. The control `c0` is the fixed-pipeline production arm of
[PR #1159 (Study: production feature faults and forecast effects)](https://github.com/openclimatefix/nged-substation-forecast/pull/1159),
recipe `production_cleaned` (24 production columns, `colsample_bytree` 1.0, the quick-wins
`new_meter_settling` cleaning), read from
`quick_wins_fixed_pipeline_1156/fits/combined/fold_a/production_cleaned/main/cuda_seed<s>_none`.
The fixed-pipeline fits were run with padding `equal` and `none`; the recipe is in the set that is
never padded, so both runs have the same 24 columns, and this study reads the `none` run.

| Arm | What it is |
| --- | --- |
| `c0` | The stored ENS-trained control, seeds 0, 1, 2. |
| `c0_refit` | The same model refitted here from the same frames with this study's loop. It checks the loop against the stored fits and is the paired control for `c0_plus_s1`. |
| `s1_control` | Stage 1a applied to ENS member 0's weather. |
| `s1_memberavg` | Stage 1a applied to each of the 51 members' weather, predictions averaged. |
| `s1_cams` | Solar farms only. As `s1_memberavg`, with stage 1a trained on CAMS global irradiance in place of ERA5's. |
| `c0_plus_s1` | `c0_refit` with stage 1a's prediction as a 25th column. Training rows carry a forward-chained out-of-fold value (one model per month, trained on ERA5 up to 14 days before the month starts); validation rows carry the pre-validation model's prediction for the row's own member. |
| `era5_ceiling`, `cams_ceiling` | Stage 1a applied to ERA5 (or CAMS) weather at the valid time: the floor for any ENS-driven stage 1a. Not a forecast; independent of lead. |

`c0_refit` and `c0_plus_s1` both use `colsample_bytree` 1.0, so the extra column gives no gain from
column subsampling. The stored `c0` was fitted at 1.0 too.

## Stage 1a

- **One XGBoost model per series**, squared error, production hyperparameters (learning rate 0.05,
  depth 6, minimum child weight 1, row subsample 0.8, 500 rounds) with every column offered to each
  tree, on the GPU, seeds 0, 1, 2. The target is power divided by effective capacity.
- **No power lag and no lead input.** Calendar columns are the sine and cosine of the UTC hour and
  of the day of year.
- **Training window.** Valid times from 2019-09-01 to before 2025-06-17, 14 days before the first
  validation run (the embargo of the wind interpolation study). Rows before each series' settling
  cutoff (the quick-wins `new_meter_settling` rule, applied to the whole ERA5 era with the effective
  capacity as its level) are dropped.
- **Solar farm columns (13):** shortwave radiation, long-wave radiation, 2 m temperature, 2 m dew
  point, surface pressure, precipitation, 10 m wind speed, solar zenith, solar azimuth, hour and day
  of year sine and cosine. These are variables the ENS table serves, plus solar geometry.
- **Wind farm columns (10):** 10 m and 100 m wind speed, sine and cosine of the 10 m and 100 m
  bearing, hour and day of year sine and cosine. ERA5 temperature and pressure are not on disk at
  the wind farms' cells. No cloud cover is used, because ENS did not serve it over the training
  window.
- **Wind** is interpolated as eastward and northward components; speed and bearing are derived
  afterwards. The ENS side takes the speed and bearing columns of the fixed-pipeline frames, which
  are derived the same way.

## ERA5 and CAMS on the ENS convention

- **Half-hourly, period-ending.** A half-hourly value labelled `T` describes the 30 minutes ending
  at `T`, as the power observations and the ENS radiation do.
- **Radiation and precipitation are hour-ending means.** ERA5 irradiance (from the Open-Meteo
  mirror's hourly fields, which agree with the Copernicus archive to 1 W/m² except 12 days in
  December 2021) and CAMS irradiance are spread over the two half-hours of the hour by the
  clear-sky shape, so the two half-hours average back to the hour's value. Long-wave radiation and
  precipitation take the hour's value for both half-hours. ERA5 accumulations are divided by 3,600 s.
- **Instantaneous variables** (temperature, dew point, pressure, wind components) are interpolated
  linearly to the half-hour between their hourly stamps.
- **Cells.** Each solar farm reads the ERA5 cell nearest it; each wind farm reads the nearest cell of
  the wind archive (2019-09 to 2023-12 by cell, 2024-01 on by site). ERA5 releases 0001 (final) and
  0005 (ERA5T) are not distinguished.
- **CAMS** is read in full: no reliability filter, because the training rows are decided by the
  power observations alone.

## Planned contrasts

Each is computed per lead band and per technology, in points of capacity, on the mean absolute error
of the ensemble-mean point forecast.

- **P1 = `s1_memberavg` minus `c0`.** Does stage 1a on ENS weather beat the ENS-trained model?
- **P2 = `c0_plus_s1` minus `c0`.** Does adding stage 1a's prediction to the ENS-trained model help?

**A planned contrast is material** if the mean paired difference is at most -0.15 points, its 95%
month-resampled interval lies wholly below zero, and at least 9 of 12 months agree (a negative
difference). All contrasts are reported, whether material or not. Every other contrast, band,
slope, and CRPS figure is exploratory.

## Metrics and intervals

- **Mean absolute error** of the ensemble-mean forecast, in points of effective capacity, by
  technology and lead band. Lead is the valid time minus `power_fcst_init_time`. Lead bands: 0 to
  24 h, 1 to 2 days (24 to 72 h), 3 to 6 days (72 to 168 h), 7 to 10 days (168 to 264 h), 11 to 15
  days (264 h and over). The mean is the mean over series of each series' mean.
- **Intervals** resample whole validation months with replacement and one of the three fitting seeds,
  the same draw for every arm so contrasts stay paired (`stage1_stats.py`, 2,000 resamples,
  stream seed `20261013`).
- **Hedging diagnostic.** For `s1_memberavg` and `c0` (and, as exploratory extras, `s1_control`,
  `c0_plus_s1`, `s1_cams`), the slope of an ordinary least squares regression of observed power on
  the ensemble-mean forecast, by lead band and technology. A slope below 1 at long leads is the
  signature of a perfect-prognosis model over-reacting to forecast weather.
- **ERA5-input ceiling.** The mean absolute error of stage 1a applied to ERA5 at the validation
  valid times.
- **Fair CRPS** of the 51 member point predictions (the sorted-member form of
  `ml_core.metrics._fair_crps_expr`) for `c0`, `c0_refit`, `c0_plus_s1`, `s1_memberavg`, and
  `s1_cams`, normalised by capacity. These are squared-error point models, so this is not a
  calibrated distribution. It stays outside P1 and P2.
- **GPU repeatability.** One solar farm's `c0_refit` fitted twice with seed 0 on the GPU
  (`noise_floor.md`).

## Random seeds

XGBoost seeds 0, 1, 2 (stage 1a and the ENS-trained models; seed `s` of the ENS-trained model uses
seed `s` of stage 1a for its extra column). The month bootstrap uses stream seed `20261013`. The
anonymous site labels A to F and W1 to W3 come from `studies.anonymise`.

## Anonymisation

No generator name, `time_series_id`, or coordinate appears in any script output, table, or chart.
Series appear as A to F (solar) and W1 to W3 (wind).

## Files

| Script | What it does |
| --- | --- |
| `stage1_inputs.py` | Paths, series, lead bands, ERA5 and CAMS readers, the half-hourly conversion. |
| `stage1_build_weather.py` | Builds stage 1a's inputs from ERA5, CAMS, and the ENS rows; writes `inputs/checks.md`. |
| `stage1_fit_stage1a.py` | Fits stage 1a (`--stage sanity` first, then `--stage all`). |
| `stage1_fit_enstrained.py` | Refits `c0_refit` and `c0_plus_s1`; `--stage noise` measures GPU repeatability. |
| `stage1_score_arms.py` | Scores every arm on the same cases; saves per-row losses. |
| `stage1_stats.py` | Month-resampled intervals for levels, contrasts, and slopes. |
| `stage1_report.py` | Prints `results/report.md` and saves the tables the charts read. |
| `stage1_charts.py` | Draws the charts. |

Outputs are under `data/studies/per_study/stage1_pilot/`: `inputs/`, `stage1a/` (predictions),
`ens_trained/` (forecasts of the refitted arms), `scores/` (per-row losses), `results/` (tables),
and `charts/`.

## Reproduce

```bash
uv run python studies/stage1_pilot/stage1_build_weather.py
uv run python studies/stage1_pilot/stage1_fit_stage1a.py --stage sanity
uv run python studies/stage1_pilot/stage1_fit_stage1a.py --stage all
uv run python studies/stage1_pilot/stage1_fit_enstrained.py --stage noise
uv run python studies/stage1_pilot/stage1_fit_enstrained.py --stage all
uv run python studies/stage1_pilot/stage1_score_arms.py
uv run python studies/stage1_pilot/stage1_report.py
uv run python studies/stage1_pilot/stage1_charts.py
```

Every script skips outputs that exist, so a crashed run resumes where it stopped.

## Limitations

- Nine series in two weather regions: six solar farms share two ERA5 cells, so the effective sample
  is weather episodes (months), not series-hours.
- The held-out year is one year. A forward fold only; fold B (the reversed fold) is not run.
- The ERA5 irradiance and 2 m temperature come from the Open-Meteo mirror at whole W/m².
- Wind farms have no temperature or pressure input.
- The ENS-trained models run on the fixed-pipeline frames, whose weather columns were built with the
  interval-aware radiation resampler and component wind interpolation. Stage 1a is trained on ERA5
  put onto the same convention by this study's own, simpler code, not the production resampler.

## Results

Every number below is printed by `stage1_report.py` into `results/report.md`. Mean absolute error
is in points of capacity, fold A, three fitting seeds, 12 validation months.

- **P1 is material for wind at every lead band and for solar to day 6.** `s1_memberavg` minus `c0`
  is -3.72 points for wind over all leads (95% interval -4.82 to -2.67) and -0.84 for solar
  (-1.24 to -0.43). For solar, 7 to 10 days is -0.64 (-1.18 to -0.11) with 8 of 12 months
  negative, so it fails the month rule, and 11 to 15 days is -0.56 (-1.14 to +0.03).
- **P2 is not material anywhere.** `c0_plus_s1` minus `c0` is -0.005 for solar (-0.04 to +0.05)
  and -0.025 for wind (-0.11 to +0.06) over all leads. Why the extra column helps nowhere is
  not established by this study; a quick check on one wind farm (not in a committed script, so not
  quoted) suggested the ENS-trained model fits its training months far more closely than the
  validation year, which would leave little for an extra input to add.
- **The ERA5-input ceiling** (stage 1a given ERA5 weather) is 6.18 points for solar (CAMS-trained,
  given CAMS: 4.42) and 7.81 for wind, against 7.90 (`s1_memberavg`) and 15.36 for the ENS-driven
  versions. Forecast weather costs wind about 7.5 points over all leads and solar 1.7.
- **Training weather and training length matter more than the recipe.** `s1_recipe_on_ens`
  (stage 1a's columns trained on 15 months of ENS) is worse than `c0` (solar +0.64, wind +1.41),
  and `s1_memberavg` is 1.48 (solar) and 5.13 (wind) better than it.
- **Averaging 51 members helps.** `s1_memberavg` is better than `s1_control` by 0.68 (solar) and
  2.17 (wind) over all leads, and by more at long leads.
- **No sign of over-reaction.** The slope of observed on predicted power for `s1_memberavg` is
  1.04 to 0.97 for solar and 1.04 to 0.83 for wind from the first to the last lead band, against
  0.86 to 1.01 (solar) and 0.05 to 1.07 (wind) for `c0`; the ceiling's slope is 0.94 to 1.00.
- **GPU repeatability.** The same GPU fit repeated gave identical predictions (difference 0.0000).
  `c0_refit` reproduces the stored `c0` exactly.

Caveats: the control `c0` is weak for wind (19 points over all leads), and the P1 gain is the
gain over that control, not over a best-possible ENS-trained model. ENS wind is 13% to 15% slower
and ENS shortwave 6% brighter than ERA5 at the same valid time (`inputs/checks.md`), a bias stage
1a was not trained on. Twenty-four decisions (2 contrasts, 2 technologies, 6 lead bands) are read
against the materiality rule without correction for multiple comparisons.
