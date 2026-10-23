---
id: '00270'
slug: metered-farm-xgboost-vs-plant-diagnostic
title: >-
  Why does XGBoost beat the solar-farm power model on metered farms when a better power model barely
  helps?
theme: metered-solar-farms
also_relevant_to:
- weather-products-and-blending
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00270-metered-farm-xgboost-vs-plant-diagnostic/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible order for the ladder of
  added inputs, so the order is fixed in this plan before any fit. Callers: this folder's scripts,
  copied from experiments 224, 243, and 257, and no new reviewed module. Asset graph: not touched.
hypothesis: >-
  XGBoost beats the plant (the solar-farm power model of experiments 218 and 224) on the six metered
  farms because of inputs the plant never sees, namely the farm's own lagged power and the ensemble
  spread, and not because of a better map from the ensemble-mean weather to power. Calendar columns
  and extra weather variables add little once those two are present.
prediction: >-
  A. XGBoost given only the plant's inputs scores more than 1.5% better than the plant, so part of
  the gap is a better weather-to-power map for ensemble-mean inputs (the prediction that could
  fail; the competing outcome is a score within 1% of the plant). B. The ladder of added inputs
  places at least 70% of XGBoost's total gain over the plant in at most two rungs.
decision_rule: >-
  Quantile score in points of capacity on experiment 243's rows, equal weight per farm, month-block
  resampling of the 12 scored months at 98.75% (a Bonferroni correction across the four headline
  contrasts). A is supported when the interval of (XGBoost on the plant's inputs minus the plant)
  lies wholly below -1.5%. Stop rule: if that contrast is above -0.5% and the ladder places the gain
  in lagged power and spread, the contradiction dissolves and the weather-adapter experiment should
  not use more variables. The threshold is zero for a diagnostic, because the experiment changes
  nothing in production.
benchmark: >-
  The plant alone (experiment 243's per-farm offsets), experiment 243's XGBoost, and the
  manual_heuristic and climatology floors of experiment 224
row_set_id: >-
  Experiment 224's scored-year rows at 12 to 36 hour leads that every arm holds (daylight
  half-hours, sun more than 5 degrees above the horizon, farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score in points of capacity, MAE of the median, pinball loss and coverage at 0.01, 0.05,
  0.95, and 0.99, per farm, arm, and setting
code: ml-experiments/00270-metered-farm-xgboost-vs-plant-diagnostic/
---

## Plan

- **Background.** [Experiment 243](../00243-metered-farm-architecture-comparison/result.md) found
  XGBoost 2.8% to 4% better than the plant, with the plant better at the 0.99 quantile.
  [Experiment 218](../00218-metered-solar-farm-plant-fit/result.md) found per-farm plant parameters
  lift the fitted fraction by only 0.003. [Experiment
  257](../00257-metered-farm-fair-plant-arm/result.md) found a linear fit on the lagged power and the
  ensemble spread recovers about 90% of the neural head's gain. This experiment asks which inputs
  XGBoost uses to beat the plant.
- **Rows, folds, refits.** The rows, the own-history and zero-history (leave-one-farm-out) settings,
  and the fit on the tuning months before 2025-09 of experiment 243; one fit, no refit in the scored
  year (screening schedule, 4 cores, CPU).
- **Weather sources (all exist at forecast time).** ECMWF ensemble, 00 UTC run, nearest 0.25 degree
  cell: ensemble-mean and spread of downward shortwave, and mean temperature. The store also serves
  dew point, downward longwave, precipitation, surface pressure, mean-sea-level pressure, 10 m and
  100 m wind, and 500 hPa geopotential height. It does not serve cloud layers or visibility; the
  longwave flux (a cloud proxy) and relative humidity from dew point stand in, and the brief's cloud
  layers and visibility rungs are not run. No Met Office UKV variable is used.
- **Lagged power.** The farm's own observed power one day earlier (two days when the lead is over 24
  hours), which exists at forecast time as in experiment 243.
- **Arms** (all XGBoost arms: 11 quantile outputs, 150 rounds, depth 5, learning rate 0.15,
  subsample 0.8, `colsample` fixed at 1 for every arm, 3 seeds).
    - Plant, the manual_heuristic and climatology floors (control and references).
    - X1: the plant's inputs only (ensemble-mean irradiance, beam, diffuse, temperature, sun
      elevation, azimuth sine and cosine).
    - X2: the plant forecast only (a recalibration of the plant).
    - Ladder, each rung adds to the previous: L1 plant inputs plus the plant forecast; L2 plus the
      ensemble spread (the members' standard deviation of irradiance and of the plant response); L3
      plus lagged power, a missing flag, and the availability ratio; L4 plus hour of day and day of
      year; L5 plus relative humidity, downward longwave, precipitation, and surface pressure; L6
      plus the remaining production-style inputs (wind speed, mean-sea-level pressure, 500 hPa
      height, clear-sky index, lead hours).
    - Shuffled-input controls S2 to S6: rung k with the columns added at rung k permuted among the
      rows within each farm; each should score like rung k-1.
    - P: the plant with its tilt, azimuth, heat loss, scale, and ceiling refitted on the
      ensemble-mean inputs, against the same plant fitted on observed (CAMS) irradiance, own history
      only.
    - G: an additive model (boosted depth-1 trees, no interactions) of the residual from the plant
      forecast, one smooth term per ladder input. Monotonicity is not imposed, which departs from the
      brief.
- **Departures from the brief.** The ladder has no cloud-layer or visibility rung, and no lead band
  below 12 hours, because a 00 UTC run is only available at forecast time for leads of 12 hours or
  more. Hour of day and day of year in L4 can let a model memorise the date with 17 tuning months,
  so S4 is the guard.
- **Report.** Gain over the plant per arm in quantile score and MAE, by clear-sky-index band, hour of
  day, season, and lead band within 12 to 36 hours; tail calibration at 0.01, 0.05, 0.95, and 0.99;
  a partial-dependence plot of XGBoost's gain against clear-sky index, cell temperature, and spread;
  the share of the gap that each explanation (a to e) accounts for.
- **Reviews.** Sonnet reviews the code and the write-up; no Opus unless a result contradicts a
  prediction in a way that changes the next step.
