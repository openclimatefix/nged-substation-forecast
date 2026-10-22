---
id: '00248'
slug: metered-farm-architecture-ukv-inputs
title: >-
  Do UKV irradiance and cloud inputs improve the two best neural heads on the plant for a metered
  solar farm?
theme: metered-solar-farms
also_relevant_to:
- differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private predictions under data/ml-experiments/00248-metered-farm-architecture-ukv-inputs/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  defensible way to feed UKV to a neural head (extra input columns, or a blended irradiance
  through the plant), so extra input columns are fixed here, because experiment 229 already
  tested the blend. Callers: this folder's scripts, copied from experiment 243, and no new
  reviewed module. Asset graph: not touched.
hypothesis: >-
  UKV irradiance and cloud, given as extra inputs to the two best neural heads of experiment 243
  (H5, the plant plus 11 quantile offsets, and H2, the plant plus a correction), lower a metered
  solar farm's quantile score at 12 to 36 hours by more than the 1.8% that experiment 229 found
  for a blend of ensemble and UKV irradiance, because a neural head can use UKV cloud layers that
  a blend cannot.
prediction: >-
  H5 with the six UKV irradiance and cloud columns beats H5 with ensemble inputs only by at least
  3% in quantile score, and beats the same head with UKV columns taken from another day (the
  placebo) by at least 3%. The same holds for H2 with a smaller gain, and XGBoost given the same
  columns gains less than H5.
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity) on daylight
  half-hours at 12 to 36 hour leads with a usable UKV run, equal weight per farm, 95% intervals
  from resampling the 10 scored calendar months (2025-03 to 2025-12) 20,000 times, as in
  experiment 229. Supported when H5 with UKV minus H5 with ensemble inputs is at least 3% below
  zero with an interval wholly below zero, and the placebo contrast is also below zero. A gain
  under 3% with an interval below zero is a small gain. The 3% threshold is experiment 229's, set
  because UKV is a second weather source with a publication delay and a research archive that
  differs from the live product.
benchmark: >-
  H5, H2, XGBoost, and the plant with ensemble inputs only, on the same rows, and experiment 229's
  blend arm (quoted from its page)
row_set_id: >-
  Experiment 229's rows: daylight half-hours (sun more than 5 degrees above the horizon) at 12 to
  36 hour leads with a usable UKV run, farms A to F, scored 2025-03 to 2025-08 (replicate) and
  2025-09 to 2025-12 (scored fold), both inside the second UKV era
metric: >-
  Quantile score in points of capacity, MAE of the median, and pinball loss at 0.01 and 0.99,
  per farm, arm, and fold
code: ml-experiments/00248-metered-farm-architecture-ukv-inputs/
production_status: not-for-production
adopted_in: none
---

Planned. The first commit of this branch is the plan, and the write-up replaces this page.

## Plan

- **Why.** Experiment 243 found that a neural head that outputs all 11 quantile offsets on the plant
  (H5) ties XGBoost, and that the best one-number hybrid is H2. Experiment 229 found that UKV adds
  1.8% when blended into the plant's irradiance. This experiment asks whether a neural head
  extracts more from UKV than a blend does.
- **Weather and forecast time.** ECMWF ensemble inputs as in experiment 243. UKV: the six
  irradiance and cloud variables of experiment 229 (shortwave down, and total, low, medium, high,
  and very low cloud) at the farm's nearest cell, from the freshest complete run published 4 hours
  before the forecast time, plus UKV shortwave over clear-sky irradiance. Both exist at forecast
  time. The UKV-CEDA archive is a research archive and differs from the live UKV, so every gain is
  an upper limit for production.
- **Arms.** Ensemble only: the plant, XGBoost, H2, H5. With UKV columns: XGBoost, H2, H5. Control:
  H5 with UKV columns taken from another day of the same farm, month, half-hour, and ensemble
  clear-sky band. Five seeds per neural arm, one fit on the tuning months per fold, no refit.
  Experiment 229 found 22 UKV variables no better than 6, so only the six are used.
- **Folds.** Scored fold 2025-09 to 2025-12, tuned on 2024-04 to 2025-08; replicate 2025-03 to
  2025-08, tuned on 2024-04 to 2025-02. The last two tuning months of each fold select the training
  epoch. Both folds lie in the second UKV era, so no fit and score straddle a UKV physics change.
  Own-history setting only; the zero-history setting is not repeated.
