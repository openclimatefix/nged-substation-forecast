---
id: '00272'
slug: metered-farm-weather-adapter-gam-vs-neural
title: >-
  Does a learned correction to the weather inputs of the solar-farm power model help, and does a
  neural network beat additive curves?
theme: metered-solar-farms
also_relevant_to:
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private predictions under
  data/ml-experiments/00272-metered-farm-weather-adapter-gam-vs-neural/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible form for the
  correction (a factor on irradiance and an offset on temperature, or a residual on power), so the
  factor-and-offset form, fixed in this plan before any fit, is the design chosen because it keeps
  the output non-decreasing in irradiance by construction of the plant. Callers: this folder's
  scripts, copied from experiments 243 and 257, and no new reviewed module. Asset graph: not
  touched.
hypothesis: >-
  Extra weather variables that the solar-farm power model (the plant) ignores, fed through a
  learned correction to the plant's irradiance and temperature, close part of the gap between the
  plant driven by the ensemble weather forecast (about 0.91 of output fitted on sunny half-hours)
  and the plant driven by observed weather (0.973), and a small neural network does this better
  than additive curves.
prediction: >-
  The additive-curve correction (GAM-style) on all variables closes at most a quarter of the 0.91
  to 0.973 gap and lowers the quantile score by 1% to 2% against the plant alone (the prediction
  that could fail). The neural correction beats the additive one by under 1%. Experiment 248's UKV
  columns lowered the neural head by 1.1% and XGBoost by 1.4%.
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), equal weight
  per farm, on experiment 257's rows (daylight half-hours, 12 to 36 hours ahead, 2025-09 to
  2026-08), own-history fits as the primary setting (zero-history is reported beside it), losses
  averaged over seeds per row, 99.4% intervals from resampling the 12 scored months 20,000 times
  (Bonferroni over eight planned contrasts). A1' is the plant with the adapter held at identity and
  the same offset table fitted the same way as every adapter arm; A1' and not the stored plant is
  the reference for C1 and the stop rule. C1: additive curves on all variables (A3) minus A1'. C2:
  A3 minus additive curves on the minimal inputs (A2). C3: neural network on all variables (A5)
  minus A3. C4: A5 minus the neural network on the minimal inputs (A4). C5: adapter plus H5 head
  (A6) minus the identity adapter plus the same head (A6 reference). C6: A3 minus its own placebo.
  C7: A5 minus its own placebo. C8: share of the gap from A1' to the 0.973 reference of experiment
  218 that A3 and A5 close, on sunny half-hours (point estimate and range over farms, no interval,
  so it is reported and does not decide). The neural network is preferred only when C3 is at least
  1.5% below zero with an interval wholly below zero (the cost of serving and retraining a neural
  network beside additive curves). An extra variable set is kept only when its contrast (C2 or C4)
  is at least 0.5% below zero with an interval wholly below zero and the matching placebo contrast
  (C6 or C7) is also wholly below zero. Stop rule: if A3 minus A1' is under 0.5% below zero (a
  worsening counts as under 0.5%) at 4 or more of the 6 farms in own history, stop before the
  neural arms and report that the extra variables carry no usable information through the plant.
benchmark: >-
  The plant alone (A1), stored H5 and XGBoost of experiment 243, the linear quantile regression P2
  of experiment 257, and experiment 224's manual_heuristic and climatology floors
row_set_id: >-
  Experiment 257's rows: daylight half-hours (sun more than 5 degrees above the horizon) at 12 to
  36 hour leads, 2025-09 to 2026-08, farms A to F, that also have every extra variable
metric: >-
  Quantile score, MAE of the median, fraction of sunny-half-hour output fitted by the median, and
  pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99, per farm and arm
code: ml-experiments/00272-metered-farm-weather-adapter-gam-vs-neural/
---

## Plan

- **Background.** [Experiment 218](../00218-metered-solar-farm-plant-fit/result.md) found a better
  plant barely helps (fraction of output fitted 0.950 to 0.953) while the ensemble irradiance caps
  the fit at 0.91 on sunny half-hours against 0.973 with observed weather. [Experiment
  243](../00243-metered-farm-architecture-comparison/result.md) found a neural head on the plant
  4.4% better than the plant, and [experiment
  257](../00257-metered-farm-fair-plant-arm/result.md) found a linear fit on the lagged power and
  the ensemble spread gets 90% of that gain. [Experiment
  248](../00248-metered-farm-architecture-ukv-inputs/result.md) found Met Office UKV columns worth
  about 1%. This experiment asks whether other ensemble variables, entering through a correction to
  the plant's inputs, do better than those heads, which correct the plant's output.
- **Rows and fits.** Experiment 257's rows and folds. One fit on the tuning months before
  2025-09, none in the scored year (screening schedule, 4 CPU cores), the schedule of experiments
  243 and 257, so the contrasts against their stored arms pair on the same half-hours. Own-history
  (fitted on the scored farm's tuning months) and zero-history (leave-one-farm-out, fitted on the
  other five farms) settings. Weekly refits are not run, which departs from the brief; the
  contrasts here are between arms sharing one schedule.
- **Weather sources (all exist at forecast time).** ECMWF ensemble, 00 UTC run, nearest 0.25
  degree cell. Minimal inputs: clear-sky index of ensemble-mean irradiance, relative ensemble
  spread of irradiance, and ensemble-mean temperature. Full list adds downward longwave flux (a
  cloud proxy), relative humidity from temperature and dew point, precipitation, surface pressure,
  mean-sea-level pressure, 10 m and 100 m wind speed, and 500 hPa geopotential height, taken at
  the store's 3-hourly step holding each half-hour (the `extra_<farm>.parquet` files of
  experiment 270). The store serves no cloud layers or visibility, so the brief's cloud-layer
  variables are not run. No UKV variable is used.
- **The plant and the correction.** The plant is experiment 243's fleet-shape plant in torch with
  the per-farm scale and ceiling (the median of the other five farms' values in zero history),
  frozen. Effective beam and diffuse irradiance equal the ensemble values times exp(g(x)), and
  cell temperature is raised by h(x), with g bounded to +-0.4 and h to +-4 K. Both are zero at
  initialisation, so every adapter starts as the plant. The adapter's quantiles are its power plus
  a table of 11 offsets per bin of the unadapted plant forecast. The table is trained jointly with
  the adapter so the training loss is the quantile loss, and then replaced for every arm, by the
  same code, with the per-bin residual quantiles of the frozen arm on the validation months (2025-07
  and 2025-08), so that a more flexible arm cannot get narrower bands by overfitting its tuning
  rows. Quantiles are sorted, and the crossing rate before sorting is reported per arm.
- **Selection and tuning.** Every choice (epoch, penalty strength, which of A3 and A5 goes into A6,
  which arm gets the alternating and blind controls) is made on the validation months only, with
  the same budget for every adapter: the same 60 epochs with early stopping on validation, and two
  penalty strengths (a second-difference penalty of 1e-3 or 1e-2 for the curves, weight decay of 1e-4
  or 1e-2 for the network). Input standardisation uses tuning rows only. The final fit includes the
  validation months for no arm.
- **Arms.** A1' plant at identity with the shared table (stored plant of experiment 243 kept as a
  benchmark). A2 additive curves on the minimal inputs. A3 additive curves on the full list: one
  piecewise-linear curve with 8 knots per variable for g and another for h, summed, 3 seeds. A4
  neural network on the minimal inputs. A5 neural network on the full list, two hidden layers of 32
  units, zero-initialised last layer, 5 seeds. A6 the better of A3 and A5 on validation, with
  experiment 243's H5 head (11 corrections from the lagged power, ensemble spread and the other H5
  inputs, with the adapted power in place of the plant forecast) on the frozen adapter, against
  the same head on the identity adapter. Controls: shuffled (the extra variables, not the minimal
  ones, permuted across rows within farm and split) and placebo (the extra variables permuted
  within farm, split, calendar month, and clear-sky-index band, so other days of the same season and
  cloud class), both with the arms retrained on the permuted columns; blind (the same network with
  no plant, on the raw inputs plus the full list); an alternating-update check (the plant's scale
  unfrozen for two rounds) on the better arm.
- **Rows.** Every half-hour of experiment 257's rows has all extra variables (checked), so the row
  set is unchanged. The observed-weather plant is not re-run, so the 0.973 reference in C8 is
  experiment 218's, from other rows, and the share closed is indicative.
- **Departures from the brief.** The additive curves are trained by gradient descent through the
  non-linear plant, so the fit is not convex and backfitting is not used. Monotonicity in
  irradiance is measured after the fit, and passes if under 1% of grid steps fall as irradiance
  rises. Quantiles are sorted after the fact, with the crossing rate reported, and not built from
  cumulative non-negative offsets. The minimal inputs add temperature to irradiance and spread.
  Cloud layers and visibility are not served. The interval level is 99.4%, not 98.75%, because
  eight contrasts are planned. The extras are held from the store's 3-hourly step. One fit is made
  on the tuning months (no weekly refits).
- **Report.** Quantile score and MAE per arm with intervals, share of the 0.91 to 0.973 gap closed
  (fraction of sunny-half-hour output fitted by the median), tail calibration through
  `studies.tail_calibration`, the crossing rate of the unsorted quantiles per arm, the fitted
  curves of A3 beside the neural network's factor on the same variable, and the Checks bullets of
  the `ml-experiment` skill.
- **Reviews.** Sonnet reviews the code and the write-up, and a Sonnet prose review runs before the
  merge. No Opus unless a result contradicts a prediction in a way that changes the next step.
