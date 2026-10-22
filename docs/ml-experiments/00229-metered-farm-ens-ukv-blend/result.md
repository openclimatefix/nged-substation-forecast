---
id: '00229'
slug: metered-farm-ens-ukv-blend
title: >-
  Does adding the UKV weather forecast to the ECMWF ensemble lower the day-ahead error of a metered solar farm?
theme: metered-solar-farms
also_relevant_to:
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00229-metered-farm-ens-ukv-blend/ only. Serving path: not touched. Degradation
  rule: not touched. Designs: more than one defensible design (a blend of irradiance before the plant,
  or UKV columns in a second-stage tree), so both are registered as separate arms. Callers: this
  folder's scripts, copied from experiments 224, 218, and 57, and no new reviewed module. Asset graph:
  not touched.
hypothesis: >-
  The Met Office UKV weather model (2 km grid, 54 hours) carries information about cloud over a solar
  farm that the 51-member ECMWF ensemble lacks, so giving a metered solar farm's forecast both products
  lowers its day-ahead error against the ensemble alone, and closes part of the gap between the
  ensemble-driven fit (0.91 of output on sunny half-hours) and the fit with observed weather (0.973) of
  experiment 218. Every gain is an upper limit for production, because the UKV archive used here is the
  research archive on CEDA, which differs statistically from the live UKV.
prediction: >-
  The best blend lowers the median over the six farms of the quantile score at 12 to 36 hour leads by at
  least 3% against its matched ensemble-only arm, with a month-block bootstrap interval at the 99.17%
  level that excludes zero, and closes at least a quarter of the 0.91 to 0.973 gap in the fraction of output
  fitted on sunny half-hours. Experiment 53 found a 7.6% lower squared error from UKV at five Primary
  substations in a validation year that could not confirm it, so a null result is plausible.
decision_rule: >-
  Quantile score is twice the mean pinball loss over the 11 levels of `studies.ens_members`, in points of
  capacity, on daylight half-hours (sun more than 5 degrees above the horizon) whose UKV run is complete,
  so that every arm scores the same rows, at 12 to 36 hour leads, over both folds (2025-03-01 to
  2025-12-31) with intervals from
  resampling the 10 scored calendar months 20,000 times. Three blend contrasts are pre-registered:
  C1a is A3 against A1, C1b is A4 against A1x, and C1c is A5 against A1x, each as a percentage change
  of the quantile score per farm, summarised by the median of the six farms. C2 is A5 against A4
  (whether the extra variables matter), C3 is A3 against its shuffled-UKV placebo, and C4 is A4 against
  its shuffled-UKV placebo. The six contrasts share a Bonferroni-corrected level of 99.17%. Supported
  when the blend with the largest median gain among C1a to C1c is at least 3% lower and its interval
  excludes zero, and that blend closes at least a quarter of the gap between the fraction fitted by A1 on
  sunny half-hours and 0.973 (experiment 218's fit with observed weather, on other rows); partly
  supported when only the interval condition holds; not supported otherwise. Stop rule: if the interval
  of the best blend against its matched ensemble-only arm includes zero at 4 or more of the 6 farms, the
  page reports that UKV adds nothing detectable for a metered farm; if UKV alone (A2) scores worse than
  the cloudless-sky baseline at the 0.5 to 6 hour leads of the data gate, the page reports a data
  problem first. The 3% is set by cost: a blend adds a data source with its own download, publication
  delay, and archive break to the serving path, so a gain under 3% would not justify that.
benchmark: >-
  A1 (plant fed the ensemble's two-number summary, the day-ahead arm of experiment 224), A1x (the same
  second-stage tree as A4 given only ensemble columns), and the manual_heuristic and climatology floors
row_set_id: >-
  Metered solar-farm half-hours with sun more than 5 degrees above the horizon, a null `drop_reason` in
  experiment 179's cleaned series, a complete UKV 00 UTC run, valid 12 to 36 hours (band 0) and 36 to 54
  hours (band 1) after the 00 UTC start of the ensemble run, scored 2025-09-01 to 2025-12-31 with a
  replicate of 2025-03-01 to 2025-08-31 (the ensemble history starts in 2024-04, so each fold is tuned
  on every month before it), both inside the second UKV era
metric: >-
  Quantile score in points of capacity, MAE of the median, fraction of output fitted on sunny
  half-hours (clear-sky irradiance above 650 W/m²), and coverage and pinball loss at 0.01, 0.05, 0.95,
  and 0.99, per farm and arm
code: ml-experiments/00229-metered-farm-ens-ukv-blend/
production_status: not-for-production
adopted_in: none
---

Planned. The first commit of this branch is the plan, and the write-up replaces this page once the
experiment has run.

## Plan

- **Weather at forecast time.** The ensemble run is the 00 UTC run of the valid time's date minus the
  band's day shift, as in [experiment 224](../00224-metered-farm-dayahead-two-number/result.md), and
  its weather is assumed published 8 hours after the run starts. The UKV run is the 00 UTC run of the
  same date, assumed published 4 hours after it starts, so the 06 UTC UKV run (published at 10 UTC)
  is never used for a forecast made at 08 UTC. Leads are counted from the 00 UTC start of the run.
- **Product-specific training.** Arm A1 and the tree A1x train on ensemble history, A2 on UKV history,
  and A3, A4, A5 on the pair, all within the second UKV era (2020-01 to 2025-12). No arm sees a UKV
  value from an era other than the one it was fitted on. The third era (from 2026-02) is not scored,
  because it holds no training months of its own.
- **Arms** (all fleet-shape plant, as in experiment 224): A1 ensemble two-number plant; A1x tree on
  ensemble columns; A2 UKV alone through the plant, with residual offsets from UKV tuning rows; A3
  plant on a convex blend of the ensemble and UKV irradiance, weights by lead band and UKV cloud band,
  fitted on the other farms' tuning rows, with the ensemble spread; A4 a second-stage XGBoost tree on
  ensemble columns plus all UKV variables on disk for the farm's cell, predicting output over clear-sky
  output; A5 the same tree restricted to UKV shortwave and cloud variables; controls: A3 and A4 with
  UKV taken from a donor row of the same month, hour band, and cloud band (placebo).
- **Night placebo.** The farm tables hold only daylight half-hours, so a night placebo cannot be built
  from them; the shuffled-donor controls replace it, and the page says so.
- **Data gate.** Before scoring, the UKV hours used are checked for gaps, steps at era boundaries,
  valid-time convention, and grid orientation; UKV shortwave alone is compared with the cloudless-sky
  baseline at the freshest-run leads 0.5 to 6 hours (`studies.ukv_ceda_stores.read_at_instants`).
- **Prior advice.** The plan follows experiment 224's rows and scoring, experiment 218's finding that
  the plant is not the limit, and the product-specific training rule; it departs from the brief's
  lead bands 0.5 to 6 and 6 to 12 hours for scoring, because a forecast made from a 00 UTC run at
  08 UTC has no such leads. Those leads appear only in the data gate.
- **Schedule.** One refit per fold (tuning rows before the fold, no rolling refit), because the
  contrast is a gain of 3% or more.
