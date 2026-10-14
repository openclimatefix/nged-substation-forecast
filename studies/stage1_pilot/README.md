# Stage 1 pilot: does a weather-to-power XGBoost model trained on ERA5 and CAMS improve the ENS point forecast?

This is study 2a of the staged probabilistic forecast plan ([PR #1183 (Plan: document a staged
probabilistic forecast, and schedule the studies that test it)](https://github.com/openclimatefix/nged-substation-forecast/pull/1183)).
It runs on the nine metered generator series only (6 solar farms, 3 wind farms), because ERA5 and
CAMS are on disk only for them. It is exploratory and autonomous: it changes no production code and
imports only from this folder, `studies.*`, and the reviewed packages in `packages/*`.

## What changed after the first run

**The first run's planned contrasts were revised after two scientific reviews, so the revised
contrasts are not pre-registered.** The first run compared stage 1a with the production control
`c0`, which has 24 production columns, including hour, day-of-year, day-of-week, and four power
lags. Two reviewers found that the control looks like it memorises calendar position: its training
rows repeat each valid time about 15 times (one per forecast lead), and the calendar columns
identify the valid time. The maintainer then asked for one feature list in every arm. The
README describes the second run:

- **One feature list in every arm** (see "Columns"), with no calendar column and no power lag. The
  ENS-trained control `ens_trained` now has the same columns as stage 1a, so the only difference
  between them is the training weather and the training length.
- **The planned contrasts keep their names but change their reference**: P1 and P2 are now measured
  against `ens_trained` and are labelled "(revised)". The first run's version of P1 (stage 1a with
  calendar columns minus `c0`) is kept as the exploratory contrast F6, and the production control
  `c0` stays in the tables as a reference row.
- **The first run's scores are kept** (`superseded/with_calendar_and_production_columns/`) and
  enter the report as the `*_calendar` arms, so the effect of the feature lists is measured on
  the same cases (contrasts F1 to F6).

## Question

**Does a weather-to-power XGBoost model trained on ERA5 and CAMS (a "perfect prognosis" model,
2019-09 onward), applied to ECMWF ENS weather, improve the ENS-era point forecast compared with the
same model trained on ENS alone?** If it adds nothing, the staged plan has no use for stage 1.

## Rows and folds

- **Fold A of the quick-wins study, unchanged.** The ENS-trained models train on the control member
  (member 0) for valid times 2024-04 to 2025-06. They are scored on the 53 ECMWF runs initialised
  every 7 days from 2025-07-01, all 51 members, for valid times to 2026-06-30.
- **Scored rows are the half-hours with an observed power**, the rows the stored control was scored
  on. Each error is divided by the series' effective capacity from the fold A frames.
- **Fold B is not run.**

## Arms

All arms are scored on the same cases. Every arm except `c0` and `climatology` uses the columns
under "Columns". The ENS-trained arms train on the 15 ENS months, in which each valid time
appears about 15 times, once per forecast lead, with the same target and different forecast
weather.

| Arm | What it is |
| --- | --- |
| `ens_trained` | Stage 1a's columns trained on the ENS control member's weather, 2024-04 to 2025-06. Applied to each of the 51 members and averaged. The paired control for every stage 1a arm. |
| `s1_memberavg` | Stage 1a (ERA5-trained, 2019-09 to 2025-06-16) applied to each of the 51 members' weather, predictions averaged. |
| `s1_control` | Stage 1a applied to ENS member 0's weather. |
| `ens_trained_control` | `ens_trained` applied to ENS member 0's weather only: the like-for-like reference for `s1_control`. |
| `s1_short` | `s1_memberavg` with stage 1a trained on ERA5 from 2024-04 only, the months `ens_trained` trains on. Separates the training weather from the training length. |
| `s1_cams` | Solar farms only. As `s1_memberavg`, with stage 1a trained on CAMS irradiance. |
| `ens_trained_plus_s1` | `ens_trained` with stage 1a's prediction as an extra column. Training rows carry a forward-chained out-of-fold value (one model per month, trained on ERA5 up to 14 days before the month starts); validation rows carry the pre-validation model's prediction for the row's own member. |
| `climatology` | Mean power by local month and half-hour of day over 2019-09 to 2025-06-16 (cleaned, same settling cut as stage 1a). No weather input. |
| `era5_ceiling`, `cams_ceiling` | Stage 1a applied to ERA5 (or CAMS) weather at the valid time. Not a forecast, and not a bound on ENS-driven versions: ERA5 is itself an estimate with error. |
| `c0` | The stored fixed-pipeline production control of [PR #1159 (Study: production feature faults and forecast effects)](https://github.com/openclimatefix/nged-substation-forecast/pull/1159), recipe `production_cleaned`, seeds 0, 1, 2. A reference row: it is the only arm with the production columns. |
| `*_calendar` | The first run's `s1_memberavg`, `s1_control`, `s1_cams`, `era5_ceiling`, and the ENS-trained stage 1a recipe (`ens_trained_calendar`), with hour and day-of-year sine and cosine added. |

`ens_trained` and `ens_trained_plus_s1` both use `colsample_bytree` 1.0, so the extra column gives no
gain from column subsampling.

## Columns

Every arm except `c0` uses these columns, listed per technology (the report prints them too).

- **Solar farm (9):** shortwave radiation, long-wave radiation, 2 m temperature, 2 m dew point,
  surface pressure, precipitation, 10 m wind speed, solar zenith, solar azimuth. Solar zenith and
  azimuth together carry the time of day and the time of year.
- **Wind farm (6):** 10 m and 100 m wind speed, sine and cosine of the 10 m bearing, sine and cosine
  of the 100 m bearing. ERA5 temperature and pressure are not on disk at the wind farms' cells.
- **`ens_trained_plus_s1`** adds `stage1a_prediction`.
- **`c0`** has the 24 production columns, including hour, day-of-year, and day-of-week features,
  four power lags, and a precipitation-type column (see `stage1_fit_enstrained.py` and the stored
  fit's `features.json`).
- **`*_calendar` arms** add hour and day-of-year sine and cosine (solar 13 columns, wind 10).

## Stage 1a

- **One XGBoost model per series**, squared error, production hyperparameters (learning rate 0.05,
  depth 6, minimum child weight 1, row subsample 0.8, 500 rounds) with every column offered to each
  tree, on the GPU, seeds 0, 1, 2. The target is power divided by effective capacity.
- **No power lag, no lead input, no calendar column.**
- **Training window.** Valid times from 2019-09-01 to before 2025-06-17, 14 days before the first
  validation run (the embargo of the wind interpolation study). Rows before each series' settling
  cutoff (the quick-wins `new_meter_settling` rule, applied to the whole ERA5 era with the effective
  capacity as its level) are dropped.
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
of the ensemble-mean point forecast. Both are revised (see "What changed after the first run").

- **P1 (revised) = `s1_memberavg` minus `ens_trained`.** Does stage 1a on ENS weather beat the
  same columns trained on ENS?
- **P2 (revised) = `ens_trained_plus_s1` minus `ens_trained`.** Does adding stage 1a's prediction to
  the ENS-trained model help?

**A planned contrast is material** if the mean paired difference is at most -0.15 points, its 95%
month-resampled interval lies wholly below zero, and at least 9 of 12 months agree (a negative
difference). All contrasts are reported, whether material or not. Every other contrast, band,
slope, and CRPS figure is exploratory.

## Metrics and intervals

- **Mean absolute error** of the ensemble-mean forecast, in points of effective capacity, by
  technology and lead band. Lead is the valid time minus `power_fcst_init_time`. Lead bands: 0 to
  24 h, 1 to 2 days (24 to 72 h), 3 to 6 days (72 to 168 h), 7 to 10 days (168 to 264 h), 11 to 15
  days (264 h and over). The mean is the mean over series of each series' mean. A level at one
  lead band mixes lead with which cases fall in the band, so compare arms only through paired
  contrasts.
- **Intervals** resample whole validation months with replacement and one of the three fitting seeds,
  the same draw for every arm so contrasts stay paired (`stage1_stats.py`, 2,000 resamples,
  stream seed `20261013`).
- **Over-reaction diagnostic.** The slope of an ordinary least squares regression of observed power
  on the ensemble-mean forecast, by lead band and technology. The slope equals the correlation times
  the spread of observed power over the spread of the forecast. Below 1: the forecast's swings are
  larger than the swings that come true. At 1: on average, observed power equals the forecast at
  every forecast level. Read each slope against `era5_ceiling`, because any noisy forecast has a
  slope below 1.
- **Fair CRPS** of the 51 member point predictions (the sorted-member form of
  `ml_core.metrics._fair_crps_expr`), normalised by capacity. These are squared-error point models,
  so this is not a calibrated distribution. It stays outside P1 and P2.
- **GPU repeatability.** One solar farm's `ens_trained` fitted twice with seed 0 on the GPU
  (`noise_floor.md`).

## Random seeds

XGBoost seeds 0, 1, 2 (stage 1a and the ENS-trained models; seed `s` of the ENS-trained model uses
seed `s` of stage 1a for its extra column). The month bootstrap uses stream seed `20261013`. The
random example days in the charts use seed `20261013`. The anonymous site labels A to F and W1 to W3
come from `studies.anonymise`.

## Anonymisation

No generator name, `time_series_id`, or coordinate appears in any script output, table, or chart.
Series appear as A to F (solar) and W1 to W3 (wind), and no chart carries a calendar date.

## Files

| Script | What it does |
| --- | --- |
| `stage1_inputs.py` | Paths, series, lead bands, feature lists, ERA5 and CAMS readers, the half-hourly conversion. |
| `stage1_build_weather.py` | Builds stage 1a's inputs from ERA5, CAMS, and the ENS rows; writes `inputs/checks.md`. |
| `stage1_fit_stage1a.py` | Fits stage 1a, its CAMS and short-training variants, and the out-of-fold column (`--stage sanity` first, then `--stage all`). |
| `stage1_fit_enstrained.py` | Fits `ens_trained` and `ens_trained_plus_s1`; `--stage noise` measures GPU repeatability. |
| `stage1_score_arms.py` | Scores every arm, and the climatology, on the same cases; saves per-row losses. |
| `stage1_stats.py` | Month-resampled intervals for levels, contrasts, and slopes. |
| `stage1_report.py` | Prints `results/report.md` and saves the tables the charts read. |
| `stage1_tail.py` | Top-decile comparison (exploratory): MAE, slope, and paired differences on the cases where observed power exceeds the series' training-period 90th percentile; writes `results/tail.md`. |
| `stage1_charts.py` | Draws the charts. |

Outputs are under `data/studies/per_study/stage1_pilot/`: `inputs/`, `stage1a/` (predictions),
`ens_trained/` (forecasts of the ENS-trained arms), `scores/` (per-row losses), `results/` (tables),
`charts/`, and `superseded/` (the first run).

## Reproduce

```bash
uv run python studies/stage1_pilot/stage1_build_weather.py
uv run python studies/stage1_pilot/stage1_fit_stage1a.py --stage sanity
uv run python studies/stage1_pilot/stage1_fit_stage1a.py --stage all
uv run python studies/stage1_pilot/stage1_fit_enstrained.py --stage noise
uv run python studies/stage1_pilot/stage1_fit_enstrained.py --stage all
uv run python studies/stage1_pilot/stage1_score_arms.py
uv run python studies/stage1_pilot/stage1_report.py
uv run python studies/stage1_pilot/stage1_tail.py
uv run python studies/stage1_pilot/stage1_charts.py
```

Every script skips outputs that exist, so a crashed run resumes where it stopped. The report reads
the first run's scores from `superseded/with_calendar_and_production_columns/scores/`, which a fresh
reproduction would have to recreate with the first run's feature lists.

## Limitations

- Nine series in two weather regions: six solar farms share two ERA5 cells, so the effective sample
  is weather episodes (months), not series-hours. Every conclusion holds for one held-out year at
  nine generators in two weather regions.
- A percentile interval over 12 resampled months is somewhat narrow: a reviewer's check put it 15%
  to 20% narrower than a t interval. The 24 planned-contrast decisions (2 contrasts, 2
  technologies, 6 lead bands) are not corrected for multiple comparisons.
- Fold A only. Fold B (the reversed fold) is not run.
- The revised P1 and P2 and the feature list were chosen after seeing the first run's results.
- The ENS-trained rows repeat each valid time about 15 times with different forecast weather. A
  de-duplicated training set (one lead per valid time) is not tested.
- ENS is biased against ERA5 at the same valid time: 10 m wind is 13% to 15% slower, 100 m wind 9%
  slower, shortwave 7% brighter (`inputs/checks.md`), a bias stage 1a was not trained on. Mapping
  each ENS variable's quantiles onto ERA5's before applying stage 1a is not tried.
- The ERA5 irradiance and 2 m temperature come from the Open-Meteo mirror at whole W/m².
- Wind farms have no temperature or pressure input.
- Stage 1a is trained on ERA5 put onto the ENS convention by this study's own, simpler code, not the
  production resampler.
- The `sanity` stage of the stage 1a script prints validation-year errors. The stage 1a
  hyperparameters were not changed after seeing them. The feature list was changed for the second
  run, after the reviews.

## Results

Every number below is printed by `stage1_report.py` into `results/report.md`. Mean absolute error
is in points of capacity, fold A, three fitting seeds, 12 validation months, all leads unless a
lead band is named, with 95% intervals in brackets.

**Figure 1 (mean absolute error by lead band):** drawn to `charts/figure_1.svg` by
`stage1_charts.py`.

- **P1 (revised) is material for both technologies up to day 6.** `s1_memberavg` minus
  `ens_trained` is -1.70 for wind over all leads (-2.33 to -1.08) and -0.57 for solar (-0.83 to
  -0.32). By band, the wind difference is -3.18, -4.04 and -2.37 at 0 to 24 h, 1 to 2 days, and
  3 to 6 days, and the solar difference is -1.23, -1.41 and -0.73. At 7 to 10 and 11 to 15 days
  neither technology passes: wind is -0.45 (-1.30 to +0.30) and -0.61 (-1.19 to -0.13, 8 of 12
  months), solar -0.30 and -0.12.
- **Beyond day 6 the ERA5-trained model is close to climatology.** At 7 to 10 days `s1_memberavg`
  is better than climatology by 0.51 for solar (-0.96 to -0.09) and 0.59 for wind (-1.98 to +0.73);
  at 11 to 15 days by 0.10 for solar (-0.41 to +0.24) and 1.03 for wind (-1.81 to -0.39). The
  ENS-trained control is no better than climatology beyond day 6 (+0.02 solar and -0.42 wind at
  11 to 15 days, both intervals spanning zero).
- **The gain is associated with the training weather, not the training length.** With the 51
  members averaged in both arms, `s1_short` (ERA5 from 2024-04 only) minus `ens_trained` is -0.60
  for solar (-0.90 to -0.35) and -1.70 for wind
  (-2.28 to -1.13). `s1_memberavg` minus `s1_short` is +0.02 and -0.01. Both contrasts are
  exploratory.
- **Absolute levels, all leads.** Solar: `ens_trained` 8.25, `s1_memberavg` 7.67 (`s1_cams` 7.54),
  climatology 8.62, the production control `c0` 8.74, `era5_ceiling` 5.98. Wind: `ens_trained`
  17.09, `s1_memberavg` 15.39, climatology 19.16, `c0` 19.08, `era5_ceiling` 7.90. With ENS in
  place of ERA5 as stage 1a's input, error is 1.70 points higher for solar and 7.49 higher for
  wind. At 0 to 24 h for solar, `s1_memberavg` (5.57) is not worse than `era5_ceiling` (5.63), so
  the ceiling is not a bound.
- **Averaging the 51 members helps the ERA5-trained model much more than the ENS-trained one.**
  `s1_memberavg` minus `s1_control` is -0.70 for solar and -2.23 for wind over all leads (wind
  -4.28 at 11 to 15 days). `ens_trained` minus `ens_trained_control` (the ENS-trained model on the
  control member alone) is -0.15 for solar (-0.30 to -0.05) and -0.22 for wind (-0.41 to -0.05).
- **With the control member alone, ERA5 training is associated with lower error at short leads.**
  `s1_control` minus `ens_trained_control` is -1.07 (solar) and -2.75 (wind) at 0 to 24 h, -1.06 and
  -3.56 at 1 to 2 days, +0.55 and +2.09 at 7 to 10 days, and +0.64 and +2.94 at 11 to 15 days. Over
  all leads it is -0.02 (-0.30 to +0.32) for solar and +0.31 (-0.49 to +1.06) for wind. Both are
  exploratory. Over all leads and all cases, the P1 gain is associated with the combination of ERA5
  training and member averaging. At 0 to 24 h, ERA5 training alone is associated with most of the
  gain, and on the wind top decile the control-member contrast (-6.95) is as large as the
  member-averaged one (-6.14).

**Figure 2 (the planned contrasts):** drawn to `charts/figure_2.svg` by `stage1_charts.py`.

- **P2 (revised) is not material anywhere.** `ens_trained_plus_s1` minus `ens_trained` is -0.06 for
  solar (-0.12 to -0.01, 9 of 12 months) and +0.02 for wind (+0.01 to +0.03). The largest solar
  gain is -0.12 at 1 to 2 days (-0.19 to -0.05, 11 of 12 months), short of the -0.15 threshold. On
  the same columns, adding the prediction as an input adds nearly nothing.

**Figure 3 (slope by lead band):** drawn to `charts/figure_3.svg` by `stage1_charts.py`.

- **Slopes.** The slope of observed on predicted power for `s1_memberavg` is 1.03 to 0.99 for solar
  and 1.06 to 0.91 for wind from the first to the last band (0.83 at 7 to 10 days, interval 0.48 to
  1.08), against 0.99 to 1.00 (solar) and 0.95 to 0.97 (wind) for `era5_ceiling`. `s1_control`
  falls to 0.82 (solar) and 0.11 (wind) at 11 to 15 days. A single member's weather is associated
  with the over-reaction expected of a perfect-prognosis model, and averaging the members removes
  most of it. Mild over-reaction cannot be ruled out for wind at 7 to 10 days. The `ens_trained`
  wind slope is 1.5 to 2.1, so its forecast varies less than the outcomes.

**Figure 4 (the effect of the feature lists):** drawn to `charts/figure_4.svg` by
`stage1_charts.py`.

- **Feature lists (post hoc, F1 to F6 and E12 to E14).** Removing hour and day-of-year columns
  lowers the ENS-trained model's error by 1.13 points for solar (-1.71 to -0.65) and 3.40 for wind
  (-4.81 to -1.97), and the ERA5-trained model's by 0.23 for solar (-0.47 to -0.08) and not at all
  for wind (+0.03, -0.08 to +0.15). Replacing the production control's 24 columns by the common
  columns lowers the ENS-trained error by 0.49 (solar) and 1.99 (wind). The production control is
  worse than climatology at 7 to 10 and 11 to 15 days (solar +0.38 and +0.65, wind +1.73 and
  +2.29), and `s1_memberavg` is 1.07 (solar) and 3.69 (wind) better than it, against 0.84 and 3.72
  in the first run.

**Figure 5 (an extreme day and a random day):** drawn to `charts/figure_5.svg` by
`stage1_charts.py`.

**Top decile of observed power (exploratory, `results/tail.md`).** The cases are those where
observed power exceeds the series' 90th percentile of cleaned power in the stage 1a training window
(20,020 solar and 9,747 wind cases per seed). Selecting on the observed value truncates the outcome,
so every arm's slope is well below 1 and only the paired differences are comparable.

- **For wind, the gain from ERA5 training is much larger in the tail.** `s1_memberavg` minus
  `ens_trained` on the top-decile cases is -6.14 points over all leads (-7.66 to -4.33, 12 of 12
  months), against -1.70 on all cases, and -16.54 at 0 to 24 h and -17.56 at 1 to 2 days. With the
  control member alone, `s1_control` minus `ens_trained_control` is -6.95 (-8.39 to -5.23). At 7 to
  10 days it is -1.22 (-2.46 to -0.08) and at 11 to 15 days +0.12 (-1.17 to +1.56). Top-decile wind
  errors are large for every arm except `era5_ceiling` (13 points), so the weather input is
  associated with most of the top-decile error.
- **For solar, the tail gain is larger in size but not in certainty.** `s1_memberavg` minus
  `ens_trained` is -1.05 over all leads (-3.93 to +1.25, 7 of 11 months with a top-decile case), -3.52
  at 1 to 2 days (-6.24 to -0.89), and near zero at 7 to 15 days.

**Other results.**

- **CRPS** (exploratory, 51 member point predictions): 5.60 for `s1_memberavg` and 7.11 for
  `ens_trained` for solar, 11.04 and 13.66 for wind.
- **GPU repeatability.** The same GPU fit repeated gave identical predictions (difference 0.0000).
