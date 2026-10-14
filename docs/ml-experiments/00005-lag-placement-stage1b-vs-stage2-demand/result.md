---
id: "00005"
slug: lag-placement-stage1b-vs-stage2-demand
status: planned
size: complex
hypothesis: >-
  For substation demand series, lagged power improves a two-stage probabilistic forecast more when
  it enters stage 1 through a residual-persistence model (stage 1b) than when it enters only a
  stage 2 calibrator.
prediction: >-
  Stage 1b beats the lag-aware stage 2 calibrator by at least 0.15 points of capacity in the
  continuous ranked probability score (CRPS) in the 0 to 24 h and 1 to 2 d lead bands, and the two
  are within 0.15 points from the 3 to 6 d band onward, because residual persistence fades with
  lead. Using both is not more than 0.15 points better than the better one alone.
decision_rule: >-
  Material if the mean paired difference in CRPS over all leads is at most -0.15 points of
  capacity, the Bonferroni-corrected 99% month-and-seed interval is wholly below zero, and at least
  9 of the 12 validation months agree. Five planned contrasts, so each is read at 99%.
benchmark: >-
  R0, the lag-free pooled ERA5-trained stage 1a, and R0c, the same stage 1a with a lag-free stage 2
  calibrator, with R1 (ENS-trained, lags as inputs) and R2 (climatology by day type) as references.
row_set_id: fold_a_every5d_v1
metric: >-
  CRPS estimated as twice the mass-weighted pinball loss of 9 quantile levels, and the mean
  absolute error of the median, both in points of capacity
verdict: pending
code: ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/
pull_request: pending
---

# Where should lagged power enter a two-stage demand forecast?

**Status: planned.** This page was committed before any fit. The fits wait for the separate
input-feature sweep (experiment 00004) to choose stage 1a's input columns.

## Plan

**Question.** A two-stage forecast has two places to put recent observed power. Stage 1b
adds a learned correction to stage 1a's weather-only forecast. Stage 2 calibrates the ensemble's
distribution and could take the lags as inputs. Which placement gives the better probabilistic
forecast, and does using both add anything?

**Series.** The 20 demand series of experiment 00003 whose `substation_type` in the
`TimeSeriesMetadata` table is Primary, BSP, or GSP: 16 primary substations, 2 bulk supply points,
and 2 grid supply points, labelled D1 to D22 as in 00003. D7 and D20 (the battery and the biofuel
generator, both EHV customers) are dropped because the maintainer asked for substation series only.
The 4 grid-flow series are substations of type BSP or GSP, so they stay. The results report
primary substations and grid-flow series separately.

**Row set and folds.** Validation on the 73 runs of `fold_a_every5d_v1` (every fifth day at 00:00
UTC from 2025-07-01 to 2026-06-30), as in 00003. Nothing is scored on or after 2026-07-01. Fitting
seeds 0, 1, and 2; XGBoost hyperparameters as in 00003 (eta 0.05, depth 6, subsample 0.8, 500
rounds), with `colsample_bytree` and `colsample_bynode` both 1.0.

**Lag rule.** A lag is null when it is no longer than the lead (`lag <= lead`), in training and in
scoring. Every arm that trains on lags builds its training rows with the full lead range: each
training row draws its lead uniformly from 0.5 to 351 hours, as 00003's ERA5-trained V2 arm did,
and never keeps only the shortest lead of each valid time (the flaw that made 00003's planned
contrast D3 uninformative).

**Stage 1a.** One XGBoost model per series trained on ERA5, applied to each of the 51 ENS members.
Its input columns are those the sweep chooses; the working default is 00003's V1 (the ten weather
columns plus the local hour and day of week). Each arm is run with the training window starting at
2019-09 (full history) and at 2023-09, because 00003 found that error falls as the start moves
later. R0 is also run from 2024-04. The 2023-09 start was chosen after seeing 00003's results, and
the page says so wherever it is used.

**Arms.**

- **R0:** stage 1a alone, lag-free. Quantiles are the 9 empirical quantiles of the 51 member
  forecasts. Windows: 2019-09, 2023-09, 2024-04.
- **R0c:** stage 1a, then the stage 2 calibrator without lag inputs.
- **R1 (reference):** one XGBoost model on ENS weather, lags as extra inputs, trained on ENS 2024-04
  to 2025-06 with the full lead range.
- **R2 (reference):** day-type climatology, with the quantiles of the training distribution by local
  month and half-hour.
- **L1:** the same as R1, reported once as L1.
- **L2:** stage 1a (lag-free) plus stage 1b. Stage 1b is one XGBoost model per series trained on
  ERA5 history with simulated origins and leads. The target is the residual of observed power over
  stage 1a. The inputs are the residuals at the origin `T0 = t - L` and earlier (at `T0`, `T0 - 1 h`,
  `T0 - 3 h`, the same time of day yesterday and the same time last week, taken from the latest
  such time at or before `T0`), the lead `L`, and the calendar columns if stage 1a has them. Fading
  with lead is learned, not coded. Stage 1a residuals in the training years are out of fold:
  six time blocks of the ERA5 history, with a 14-day purge around each held-out block, so
  overlapping origin and target pairs do not cross the split. The forecast is stage 1a for each
  member plus one residual correction, computed once from observed power, and pooled over members.
  Windows: 2019-09 and 2023-09.
- **L2c:** L2's output, then the lag-free stage 2 calibrator (the matched control for L2+L3).
- **L3:** stage 1a (lag-free), then a stage 2 calibrator whose inputs are the mean, standard
  deviation, and 10th, 50th, and 90th percentiles of the 51 member forecasts, the lead, the four
  power lags (24, 48, 168, and 336 hours), and four lag anomalies (observed power at the lag time
  minus stage 1a's forecast at that time). Lags are nulled by lead.
- **L2+L3:** L2's output as the stage 1 statistics, then the stage 2 calibrator with the lags and
  lag anomalies as well.

**Stage 2 calibrator.** One multi-quantile XGBoost model per series (`reg:quantileerror`, the 9
levels 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95; depth 4, eta 0.05, 300 rounds) on power over
effective capacity. It trains on every third daily ENS run from 2024-04-01 to 2025-06-30 (valid
times up to 2025-06-30), with all 51 members. The stage 1 statistics on those rows come from
forward-chained out-of-fold stage 1a predictions: for each month, stage 1a is trained on ERA5 valid
times that end 14 days before the month starts. Output quantiles are sorted to remove crossing.

**Planned contrasts** (CRPS over all leads, points of capacity; each read at 99%):

- **C1:** L3 minus R0 (does a lag-aware stage 2 beat the lag-free pooled stage 1).
- **C2:** L2 minus L3 (is the lag information better placed in stage 1b).
- **C3:** L2+L3 minus the better of L2 and L3.
- **C4 (matched control):** L3 minus R0c. C1 mixes the effect of the lags with the effect of
  calibrating at all. C4 isolates the lags.
- **C5 (matched control):** L2c minus R0c. L2 is uncalibrated and L3 is calibrated, so C2 mixes
  placement with calibration. C5 and C4 give the like-for-like comparison.

Every other number is exploratory: the contrasts by lead band and by series group, the contrasts
between windows, and the MAE of the median. The decision for the hypothesis rests on C2, read
beside C4 and C5. The page reports all three.

**Calendar guard.** Stage 1a with calendar columns can memorise the calendar position when its
training rows repeat valid times (00001). Stage 1a here trains on ERA5, which has one row per valid
time, and every inner split groups rows by valid time. A control without the calendar columns is
run for R0 and L2.

**Ceiling label.** Lag-time residuals in L2, L3, and L2+L3 use ERA5 weather at the lag time, which
production does not have in time. The page labels those arms a ceiling and says so in every figure
title.

**Absolute scores.** The write-up shows every arm's CRPS and MAE per lead band, not only the
contrasts, with the R2 climatology row.

**Reviews.** The plan and the code get Sonnet reviews, in line with the maintainer's instruction to
use Opus only where vital; a code review for leakage and for the fit stopping at the fold's
`train_end` is the one the experiment cannot skip.

**Size.** Complex. It changes what is stored (predictions and a published page). It does not touch
the production serving path or any degradation rule. More than one design defensibly tests the
hypothesis (the arms above are the design choices). It spans new code in `ml-experiments/` only,
which no other package imports.

## Not tested here

Quantile forecasts of generator series, a second fold, and several weather models.
