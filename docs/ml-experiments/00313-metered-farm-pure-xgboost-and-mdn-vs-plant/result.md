---
id: '00313'
slug: metered-farm-pure-xgboost-and-mdn-vs-plant
title: >-
  Do an XGBoost model and a mixture-density neural network that see only weather and the calendar
  match the forecasts built on a physical plant model, for metered solar and wind farms?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
status: planned
kind: other
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00313-metered-farm-pure-xgboost-and-mdn-vs-plant/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: the arms are fixed by the brief, and the fits
  copy experiments 310 and 285. Callers: this folder's scripts, which import experiments 310 and
  285 and no new reviewed module. Asset graph: not touched. One Sonnet code review and one Sonnet
  prose review, no Opus.
hypothesis: >-
  A predictor built only from weather and calendar inputs, with no plant model anywhere in the
  pipeline (not even as an input column), scores close to the plant-based forecast in own history
  and falls behind it where the plant carries what missing history or a long lead cannot.
prediction: >-
  Written before any score is read. P1 (solar, 12 to 36 h): the pure mixture-density network (PURE_MDN)
  scores within 2% of the plant-plus-MDN head in own history and at least 3% worse in zero history.
  P2 (solar): the pure XGBoost model (PURE_XGB) is no better than PURE_MDN at the 0.99 pinball loss.
  P3 (wind): PURE_MDN scores within 2% of PURE_XGB. P4 (solar, days 8 to 10): PURE_XGB is no
  better than climatology, and PURE_MDN scores at least 5% worse than the plant-based blend. Each
  can fail.
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months at 98.75% (Bonferroni across the four predictions). A prediction holds when its
  stated percentage change falls inside the stated bound with its interval on the same side of the
  bound. The experiment compares model families and adopts nothing, so there is no porting
  threshold. Stop rule: none beyond a failed fit, which is recorded as `failed`.
benchmark: >-
  Solar: experiment 310's stored plant-based arms (plant alone, GAM head, mixture-density head,
  XGBoost with the plant response as an input, the blend), `climatology`, and `manual_heuristic`.
  Wind: experiment 276's physics core and XGBoost, and experiment 285's neural adapter.
row_set_id: >-
  Solar: experiment 310's rows (farms A to F, 2025-09 to 2026-08, day bands 1 to 10). Wind:
  `fold_a_tuesdays_v1`, the kept rows of experiments 276 and 285.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), and the
  pinball loss at 0.01 and 0.99
code: ml-experiments/00313-metered-farm-pure-xgboost-and-mdn-vs-plant/
---

## Plan

- **Question.** Does a predictor given weather and calendar inputs only match the plant-based
  forecast? "Plant" means the physical solar power model of experiments 218 and 287 or the wind
  physics core of experiment 230. A pure predictor has no plant output and no plant-derived column.
- **What experiment 300's XGBoost used.** Its inputs were the plant forecast (`mu`), the spread of
  the plant response (`plant_spread`), the plant's unit response (`unit_mean`), and the availability
  ratio (lagged power divided by the plant forecast), as well as weather. It is therefore not pure,
  and the pure arms below replace it.
- **Pure arms.** `PURE_XGB` is XGBoost (depth 6, 300 rounds, 11 quantiles, no base margin). `PURE_MDN`
  is a three-component Gaussian mixture (experiment 300's network) on the target directly, five seeds
  averaged. `PURE_BLEND` averages their quantiles level by level with equal weights.
- **Solar inputs.** The 20 weather, sun-geometry, and calendar columns of experiment 310's rung six
  that do not depend on the plant: irradiance means and spread, temperature, sun elevation and
  azimuth, humidity, longwave, precipitation, pressure, 10 m and 100 m wind, 500 hPa height, the
  clear-sky index, lead hours, hour, and day of year. Lagged power is left out as well, because the
  brief asks for weather and calendar only.
- **Wind inputs.** The six wind columns of experiment 276's XGBoost, the 2 m temperature, the hour,
  and the day of year. Experiment 276's own XGBoost starts from a binned power curve as a base
  margin, so it is quoted as a comparator and is not the pure arm.
- **Schedule.** Solar copies experiment 310's three-monthly refits, day bands, and zero-history
  setting. Wind trains on ERA5 before 2025-06-17 as in experiment 285 (Part A, ERA5 as the weather,
  a ceiling because ERA5 arrives about 5 days late) and applies the models to the 51 ensemble
  members (the forecast-time weather), mixing the members as experiment 285 did.
- **Zero history for wind.** The brief allows pooling the public wind farms of experiment 288. That
  experiment never ran its fits, so zero history for wind trains on the other two metered farms,
  which is a deviation.
- **Past-weather caveat.** Solar uses only ensemble weather available at forecast time. Wind's ERA5
  rows are a ceiling and the ensemble rows are the forecast-time figures.
