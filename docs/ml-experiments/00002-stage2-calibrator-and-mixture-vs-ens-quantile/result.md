---
id: "00002"
slug: stage2-calibrator-and-mixture-vs-ens-quantile
status: done
size: complex
hypothesis: >-
  A shared per-technology calibrator and a learned blend, fed by ERA5-trained stage 1a quantiles
  applied to ENS weather, give a lower continuous ranked probability score than the pooled stage 1a
  mixture and than a quantile model trained on ENS weather.
prediction: >-
  The CRPS falls by at least 0.15 points of capacity for the stage 2 arm against each of the two.
decision_rule: >-
  Material if the mean paired difference is at most -0.15 points, the 95% month-resampled interval
  is wholly below zero, and at least 9 of 12 validation months agree.
benchmark: >-
  a_ens_quantile (the 00001 ens_trained columns, trained with 9 quantile levels on the ENS control
  member, applied to each member and pooled), with a month-by-half-hour climatology as a reference.
row_set_id: fold_a_tuesdays_v1
metric: CRPS estimated from 9 quantile levels, points of capacity
verdict: partly supported
production_status: candidate
adopted_in: none
best_arm: b_mixture_full
headline: >-
  Q3, b_mixture_full minus a_ens_quantile, all leads: wind -0.62 (-0.99 to -0.26), solar -0.24
  (-0.36 to -0.06). Q1, stage 2 minus the mixture: solar +0.40, wind +0.26, not material.
code: studies/stage1_pilot/
pull_request: https://github.com/openclimatefix/nged-substation-forecast/pull/1189
---

# Does a stage 2 calibrator beat the pooled stage 1a mixture?

**Verdict: the pooled ERA5-trained mixture beats the ENS-trained quantile model, and the stage 2
calibrator and learned blend did not improve on it, and were worse for solar.** Study 2b ran in one
held-out year (fold A, 53 weekly runs) for 6 solar farms and 3 wind farms. The full write-up is the
`Results` section of `studies/stage1_pilot/README_stage2.md`.

## Input columns

The columns are those of experiment 00001, with no calendar column and no power lag.

- **Solar farms (9 columns):** `shortwave`, `longwave`, `temperature_2m`,
  `dew_point_temperature_2m`, `pressure_surface`, `precipitation`, `wind_speed_10m`,
  `solar_zenith`, `solar_azimuth`.
- **Wind farms (6 columns):** `wind_speed_10m`, `wind_speed_100m`, `wind_dir_sin_10m`,
  `wind_dir_cos_10m`, `wind_dir_sin_100m`, `wind_dir_cos_100m`.
- **The calibrator** takes the stage 1a quantiles for one ensemble member plus the same columns.

## Absolute scores, all leads

CRPS in points of capacity.

| arm | solar | wind |
| --- | --- | --- |
| `climatology_q` (reference) | 5.65 | 13.03 |
| `a_ens_quantile` (benchmark) | 5.29 | 11.73 |
| `b_mixture_full` | 5.05 | 11.11 |
| `c_stage2_full_blp` | 5.45 | 11.37 |

![CRPS by lead band](figure_1.svg)

## Planned contrasts

![Planned contrasts by lead band](figure_2.svg)

- **Q1 (stage 2 minus the mixture) is not material.** Solar +0.40 (+0.26 to +0.61), wind +0.26
  (-0.17 to +0.64).
- **Q2 (stage 2 minus the ENS-trained quantile arm) is material for wind only.** Wind -0.36 (-0.59 to
  -0.12), solar +0.17 (-0.03 to +0.50).
- **Q3 (mixture minus the ENS-trained quantile arm) is material for both.** Wind -0.62 (-0.99 to
  -0.26), solar -0.24 (-0.36 to -0.06). The gain sits in the first six days.

## What else the study found

- **For the mixture and stage 2 arms, the long ERA5 history changes the CRPS by 0.01 to 0.09
  points** against stage 1a trained on the ENS months only.
- **In the tail the ERA5-trained gain for wind is not larger than at the centre of the
  distribution.** The exceedance Brier scores of the 90th and 95th percentiles improve by 0.50 and
  0.27 over the ENS-trained arm.
- **The ENS-trained arm under-forecasts wind exceedances**, and stage 2's bins above 0.30 are also under-forecast.
- **The learned blend does not help reliably.** It changes the CRPS by +0.00 to +0.08 for stage 2.

## Failures and limits

- One fold, one weather product, single-member training rows, no power lags, no per-series
  calibrators, and 12 validation months.
- Quantile-mapping ENS onto ERA5 before stage 1a is untested.
- Long feature lists were not run in this experiment; experiment 00001 reports that comparison.

## Lessons for later experiments

- Report the pooled mixture beside any stage 2 arm, because the mixture is the reference stage 2
  has to beat.
- Predictions are under `data/studies/per_study/stage2_pilot/` and are not yet in
  `data/ml-experiments/` with a manifest.
