---
id: "00259"
slug: wind-plant-plus-quantile-offset-head
title: >-
  Does a neural head that adds 11 quantile offsets to the wind model, trained on 15 months of daily
  ensemble hindcasts, improve the three metered wind farms' forecasts and their 99% quantile?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
date: 2026-10-22
size: >-
  complex. Stored: private hindcast tables, trained weights, and predictions under
  data/ml-experiments/00259-wind-plant-plus-quantile-offset-head/ only. Serving path: not touched
  (the experiment decides whether a later port is worth proposing). Degradation rule: not touched.
  Designs: several are defensible (what the head sees, what it outputs, how the training and
  early-stopping runs are split), so this plan fixes one before any fit. Callers: this folder's
  scripts, which copy experiments 230, 242, and 243, and the existing production feature engineer
  called through the quick-wins study's input reader, whose callers are the studies. One Sonnet plan
  review, one Sonnet code review, one Sonnet write-up review, and one Sonnet prose review; an audit
  runs if a trigger fires.
hypothesis: >-
  For the three metered wind farms (W1 to W3), a small neural network that reads the 51-member
  mixture of a wind model and the ensemble's wind summaries, and adds an offset to each of 11
  quantiles, lowers the quantile score against the wind model alone (R1, the physics core), matches
  the XGBoost best guess (R0), and lowers the pinball loss at the 0.99 quantile, once it can learn
  from about 440 daily runs of the ensemble forecast instead of the 13 to 40 that experiment 221's
  head saw.
prediction: >-
  Written before any fit. C1, H5c minus R1, kept rows, all leads: -0.15 to -0.40 points of capacity.
  C2, H5c minus R0: inside -0.10 to +0.10, with H5c's 0.99 pinball loss 5% to 15% lower than R0's at
  lead days 1 and 2. C3, H5c minus the better non-neural arm (N5-lin or N5-xgb): at least 0.03
  points lower. C4, H5c minus K5: at least 0.05 points lower. The blind share, (BL5 minus R1) divided
  by (H5c minus R1), is under 0.5. H5c's 5% to 95% band covers 0.86 to 0.92 of kept rows.
decision_rule: >-
  Four decisive contrasts, C1 to C4 above, each on the quantile score over all leads on the rows kept
  by experiment 210's mask, read at 98.75% (Bonferroni across four), resampling whole months of the
  ENS run and the five seeds, 50,000 resamples. Supported: C1 wholly below zero, C2 wholly below
  +0.05 (no worse than XGBoost by 0.05), and the tail score T (the mean of the 0.01 and 0.99
  pinball losses) of H5c below R0's with an interval wholly below zero. The threshold for keeping the
  network over the non-neural arm is 0.03 points (C3). The threshold for replacing XGBoost in
  production is C2 wholly below zero, because the change adds a trained network to the serving path.
  H5x beside H5c is descriptive. Everything else (per farm, per lead band, season, the other
  arms, coverage) is descriptive and labelled so. The intervals describe these three farms.
benchmark: >-
  R0, the XGBoost recipe of experiment 230's A0r fitted on ERA5 before 2024-04-01, and R1, the
  physics core (experiment 230's B1) fitted on the same window, both run on the 51 members with the
  spread factor. Experiment 230's stored A0r and B1 are reference rows.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3). Headline on the rows kept by experiment 210's mask; all rows descriptive.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of
  capacity; pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99; the tail score T; coverage of the 5% to
  95% and 1% to 99% bands
production_status: not-for-production
adopted_in: none
code: ml-experiments/00259-wind-plant-plus-quantile-offset-head/
---

**Question.** A neural head that adds 11 quantile offsets to a plant forecast tied XGBoost on a
metered solar farm and cut the 0.99 pinball loss by 29%
([experiment 243](../00243-metered-farm-architecture-comparison/result.md)). The head of
[experiment 221](../00221-wind-correction-head-on-stored-predictions/result.md) failed on wind,
probably because it could only train on the at most 9 months of Tuesday runs that had a stored best
guess. This experiment re-creates the base forecasts on every daily 00 UTC ensemble (ENS) run from
2024-04-01 to 2025-06-16, so the head trains on about 440 runs.

## Data

- **Base fits.** R0 (XGBoost, experiment 230's A0r recipe, with the 0.01 and 0.99 levels) and R1
  (experiment 230's physics core, B1) are fitted on ERA5 valid times from 2019-09-01 to 2024-03-31,
  so their output on the hindcast runs is out of sample. The same fits score the Tuesday runs of the
  scored year. Experiment 256 did not meet its planned rule for adding a temperature column, so R0
  keeps the six wind columns; the head reads the members' mean temperature.
- **Hindcast runs.** Every daily 00 UTC ENS run with init time from 2024-04-01 to 2025-06-16,
  valid times before 2025-06-17, leads to 14.6 days, all 51 members, at the three farms' cells,
  built by the production feature engineer (the quick-wins study's reader), which reproduces the
  Tuesday-run frames of experiment 230 exactly. Rows with no observed power, and rows in experiment
  210's mask, are dropped from training and scoring.
- **Spread factor.** The monthly spread factor of experiment 33 (fitted on earlier cases only) is
  applied to the mixture of R0 and R1 on the hindcast runs and on the scored year, so the head adds
  offsets to the spread-adjusted base.
- **Head inputs, shared by every neural arm unless stated.** The base 11 quantiles as fractions of
  capacity; the members' mean, standard deviation, 10th percentile, and 90th percentile of the 100 m
  wind speed; the sine and cosine of the direction of the members' mean wind vector at 100 m; the
  members' mean 2 m temperature; the lead in hours; a one-hot farm column. No calendar columns.
- **Head form.** Experiment 243's H5: two hidden layers of 64 units with GELU, a zero-initialised
  last layer, 11 corrections of 0.3 times tanh added to the base quantiles, sorted so they cannot
  cross, clamped to 0 to 1, trained on the mean pinball loss over the 11 levels, with AdamW at a
  learning rate of 0.002 and weight decay of 0.0001, 60 epochs, batches of 512, pooled over W1 to
  W3, 5 seeds. Early stopping uses the last 2 months of the training period as validation, by whole
  ENS runs: training uses runs initialised before 2025-04-16 whose valid time is also before
  2025-04-16, and validation uses runs initialised from 2025-04-16. All settings are frozen before
  any scored month is read.

## Arms

All arms use seeds 0 to 4; the deterministic arms repeat their rows once per seed.

- **R0:** the XGBoost recipe on 51 members (pre-2024-04 fit) with the spread factor.
- **R1:** the physics core with residual quantiles on 51 members (pre-2024-04 fit) with the spread
  factor.
- **H5c:** R1 plus the head. **H5x:** R0 plus the same head, trained on R0's mixture.
- **N5-lin:** 11 linear quantile regressions on the head's inputs, fitted by gradient descent on the
  pinball loss, adding offsets to R1, sorted.
- **N5-xgb:** a per-level XGBoost quantile stage with the head's inputs, adding offsets to R1.
  `colsample_bytree` is 1.
- **K5 (calibration only):** the head with only the base quantiles, the lead, and the farm column.
- **BL5 (blind control):** the head with only the lead, the farm column, and the sine and cosine of
  the hour and the day of the year, adding offsets to R1.
- **T5 (table):** one offset per lead day, quantile level, and farm, the pinball-optimal constant
  fitted on the training runs.
- **References:** climatology (experiment 230's CLIM), and experiment 230's stored A0r and B1.

## Stop rules

- If R0 or R1 differs from experiment 230's A0r or B1 by more than 0.10 points on the kept rows,
  report the cause before any head is trained. A shorter training window (4.6 years against 5.8) is
  an expected cause, and it will be recorded as a departure from the plan.
- If under 95% of the daily runs have all 51 members at the farms' cells, stop and report.
- If H5c's score on its own training runs beats R1 by more than three times its gain on the
  early-stopping runs, and H5c is worse than R1 on those runs, stop tuning and report. No setting is
  tuned on a scored month.

## Compute

CPU for the base runs, and at most 2 GPU hours for the heads, run as
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`.
