---
id: '00240'
slug: no-xgboost-forecast-sprinkle-part-2
title: >-
  Can small XGBoost components inside the XGBoost-free solar and wind forecasts close the gap to
  a full XGBoost forecast?
theme: production-reference-and-best-guess
also_relevant_to:
- generator-forecasting
- metered-solar-farms
status: planned
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible (residual correction before or
  after the spread, which inputs), so this plan pre-registers one. Callers: this folder's scripts,
  which copy experiment 224's farm tables and plant forecast and call reviewed `studies.*` modules
  without changing them. A new combination of existing parts, so one Sonnet plan review, one Sonnet
  code review, one Sonnet write-up review, and one Sonnet prose review.
hypothesis: >-
  A small XGBoost residual model on the plant forecast's median error, an XGBoost spread model for the
  quantile offsets around it, or both, close at least half of the gap in quantile score between the
  XGBoost-free metered solar forecast (the plant fed the mean and spread of 51 ensemble members,
  experiment 224's arm 3) and the full XGBoost forecast (experiment 224's arm 5) at 12 to 36 hour leads.
prediction: >-
  Written before any fit. The gap to close is 0.152 points of capacity (5.598 against 5.446, experiment
  234). The residual model gains 0.03 to 0.08 points, because XGBoost's advantage sits at q05 to q95
  where the plant's fixed offsets are widest. The spread model gains 0.04 to 0.10 points. Both
  together gain 0.08 to 0.14 points, about half to nine tenths of the gap. The shuffled-target
  placebo scores within 0.01 points of arm 3. The calendar columns add under 0.01 points.
decision_rule: >-
  Solar, scored year 2025-09-01 to 2026-08-31, 12 to 36 hour leads, six farms pooled over rows,
  quantile score (twice the mean pinball loss over the 11 levels), points of capacity. Contrasts
  against arm 3 with 98.3% intervals (Bonferroni across three) from resampling whole months and the
  three seeds, 20,000 resamples: C1 residual arm R, C2 spread arm S, C3 both arms RS. A sprinkled
  arm is kept when its contrast is at most -0.076 (half the gap) with the interval's upper end
  below zero, and the placebo arm scores at most 0.01 above arm 3. The threshold is set by cost: each
  XGBoost component adds a fitted model to the serving path beside the plant, and a gain under half
  the gap would not justify a second family. Wind: the same rule with the gap of experiment 234 (5.112
  against 4.870, so a threshold of -0.121), after experiment 230 merges. Everything else (per
  farm, tails, calendar control) is descriptive.
benchmark: >-
  Arm 3 (plant fed the mean and spread of the 51 members) and arm 5 (XGBoost) of experiment 224, read
  as stored; wind: B1 and A0r of experiment 230
row_set_id: >-
  Experiment 224's metered solar half-hours, 12 to 36 hour band, scored year (replicate half-year
  reported); wind: experiment 33's Tuesday runs
metric: >-
  Quantile score; MAE of the median; coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99
code: ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2/
production_status: not-for-production
adopted_in: none
---

**Planned; no result yet.** This page is the plan, committed before any fit.

## Question

[Experiment 234](../00234-no-xgboost-forecast-audit-part-1/result.md) found that the XGBoost-free
solar forecast scores 2.8% worse than XGBoost at 12 to 36 hours, and the XGBoost-free wind forecast
5.0% worse. We ask whether a small XGBoost model inside the XGBoost-free forecast closes at least
half of each gap, which would keep the physical model for the upper tail and the interpretable
structure and add XGBoost only where it helps.

## Arms (solar)

- **A3 and A5:** experiment 224's arm 3 and arm 5, as stored.
- **R:** XGBoost on the plant mean's residual (output minus the plant mean). Inputs: the plant mean,
  the standard deviation of the 51 plant responses, the ensemble means of global, direct, and diffuse
  irradiance and of temperature, the spread of global irradiance, the sun's elevation and azimuth,
  the lead time, the hour, the month, and the plant mean's error
  48 hours earlier (known at forecast time at every lead of 12 to 36 hours). The corrected mean enters
  experiment 224's two-number mixture, with offsets refitted on out-of-fold residuals.
- **S:** XGBoost with 11 quantile outputs on the residual from the plant mean, same inputs, so the
  quantiles are the plant mean plus predicted offsets.
- **RS:** R's corrected mean, with S's offsets fitted on R's out-of-fold residual.
- **R0:** R without the hour and the month (control for calendar memorisation).
- **PR:** R fitted on residuals permuted within month and hour (placebo: it keeps the monthly and
  hourly mean residual and removes any link to the weather).
- **A3r:** arm 3 re-run through the same code with the same seeds and shared offset factors, so
  the contrasts with R, S, and RS are not mixed with sampling noise in the factors (descriptive
  contrasts D6 to D8).
- Seeds 0, 1, and 2; `colsample_bytree` 1.0; 150 rounds at learning rate 0.15, depth 3; trained on
  the tuning months before the fold start with 6-fold out-of-fold residuals grouped in blocks of
  consecutive months, so that most calendar
  months stay in every training set. The offset factors of arm 3 are reused unchanged by R, R0, and PR.
  Arm 3 and arm 5 are stored with one seed and repeat once per seed in the paired resampling.
- **Wind:** the same arms on the physics core B1's residual, after experiment 230 merges, using its
  stored B1 forecasts and ensemble features.
