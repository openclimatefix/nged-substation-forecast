---
id: "00142"
slug: differentiable-physics-sequence-model-on-gam-k-residuals
title: "Differentiable physics: a small sequence model on GAM-K's recent residuals"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction, calibration-and-uncertainty]
status: planned
size: >-
  medium, a wild-idea experiment. Stored: private files under data/ml-experiments/00142-<slug>/
  only. Serving path: not touched. Degradation rule: not touched. Designs: many are defensible
  (recurrent or convolutional, output heads, input window), and this plan pre-registers one.
  Callers: this folder's scripts only; the folder copies the modules of experiment 134 it needs and
  reads experiment 111's and 115's stored outputs. A new method (a neural network), so the plan
  gets a Sonnet plan review and the code a Sonnet code review; no Opus unless the ML manager asks.
hypothesis: >-
  A small neural network that reads GAM-K's last 14 days of residuals, the calendar, and the target
  day's forecast weather predicts the day-ahead residual better than the hand-built Kalman filter
  with the 14-day error profile (recipe R of experiment 115), and gives better tails when it is
  trained with the pinball loss at the 11 quantiles.
prediction: >-
  Written before any fit, see "Predictions" below. Headline: SEQ-median minus R is -0.03 to +0.05
  points at D+1 (most likely a null); SEQ-quantile minus R is -0.05 to +0.05; the cold-night q95
  coverage of SEQ-quantile rises by at most 5 points from K0's 73 to 78%; the training-versus-
  validation pinball gap is over 20%.
decision_rule: see "Decision rule"
benchmark: >-
  Recipe R (2.436 points at D+1 in the scored year at the five decision substations), K0 (GAM-K as
  stored, 2.524)
row_set_id: >-
  experiment 115's rows: tuning on the odd months and replicate on the even months of 2024-07 to
  2025-06, scored on the 292 runs of fold_a_daily_gap_v1 (2025-07-02 to 2026-06-29), at D5, D10,
  D11, D14, and D16, with D9 beside them
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1 (leads 15.5 to
  39 hours from the 09:00 UTC power_fcst_init_time); every lead band beside it
code: ml-experiments/00142-differentiable-physics-sequence-model-on-gam-k-residuals/
---

**Take-home: not yet run.** This page is the plan, committed before any fit.

## Question

**Can a small neural network, trained on the stored residuals of GAM-K, do the job of the Kalman
filter and recipe R, and does it give better tails?** GAM-K is the convex demand model with a Kalman
filter on ECMWF ensemble weather ([experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)).
Recipe R subtracts 0.75 times the mean of the last 14 days' errors by half-hour
([experiment 115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md)); the
[same-weekday profile](../00134-differentiable-physics-same-weekday-error-profile-on-gam-k/result.md)
added only 0.011 points. The network pools the substations with a substation embedding, so patterns
shared between substations can be learned. Sequence models of recent errors follow
[Hochreiter and Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735) and
[Cho et al. (2014)](https://doi.org/10.3115/v1/D14-1179) (the gated recurrent unit used here);
the pinball loss is the standard training loss of quantile regression
([Koenker and Bassett, 1978](https://doi.org/10.2307/1913643)).

## Design

- **Substations.** The six substations whose member forecasts experiment 111 stored (D5, D10, D11,
  D14, D16, and D9). The other DP series have no stored GAM-K member forecasts, and no demand model
  is refitted. D9 is trained on but reported beside the five decision substations.
- **One sample per (substation, run).** 821 runs per substation, about 4,900 samples. Target: the
  48 half-hours of day D+1 (leads 15.5 to 39 hours), as actual minus K0's median forecast.
- **Input.** The 14 local days before the origin's day, plus a 15th partial day, as vectors of 48
  half-hour residuals from `SiteData.stream` (observed minus the chained GAM fit, driven by the
  analysis proxy, ending at the origin's latest usable reading, 3.5 hours before the 09:00 UTC
  origin) with a reading mask; each day carries its day type (five types, as in experiment 134) and
  a holiday-like flag. Missing readings are zero with mask zero. The target day enters through its
  day type, holiday-like flag, Christmas-window flag, and the calibrated ECMWF ensemble-mean
  temperature and irradiance at its 48 half-hours (published by the origin, so available at
  forecast time). The target day's actuals are never an input.
- **Network.** A one-layer gated recurrent unit (32 hidden units) over the 15 day vectors, a
  substation embedding (4 dimensions), and a two-layer head (64 units) producing 144 numbers per
  sample: a median correction per half-hour, and a lower and an upper log-scale per half-hour that
  stretch K0's own quantile offsets (the distance of each K0 quantile from K0's median), so the 11
  quantiles cannot cross. About 25,000 parameters. Adam, weight decay, dropout 0.1, early stopping
  on a month-block validation fold.
- **Arms.** K0; R; SEQ-median (median correction trained on absolute error; the spread is rebuilt
  by experiment 111's routine as for every other arm); SEQ-quantile (the correction and the two
  scales trained with the pinball loss at the 11 quantiles).
- **Training sets.** Replicate comparison: train on the odd (tuning) months only and score the even
  months. Scored-year comparison: train on all runs whose valid times fall before 2025-07-01
  (embargoing the last 2 days), early stopping on a month-block fold held out of those. Ten seeds,
  averaged by quantile; seed is fixed per job, one GPU job at a time.
- **Compute.** About 5,000 samples and 25,000 parameters: seconds per fit on the GPU. The whole
  experiment is far inside the 2 GPU-hour cap. Worker threads are set to 1.
- **Reuse.** Rows, periods, scoring, bootstrap, breakdowns, and charts are experiment 134's, copied
  into this folder with the prefix `e00142_`. No change to `src/` or `packages/`.

## Planned contrasts (99% month-resampled paired intervals; D-labels only)

- **C1.** SEQ-median minus R at D+1, scored year (decides the verdict), and replicate months.
- **C2.** SEQ-quantile minus R at D+1, scored year, and replicate months.
- **C3.** SEQ-quantile minus SEQ-median at D+1 (does the quantile loss help the tails).
- **C4.** SEQ-median minus K0, and SEQ-quantile minus K0.
- Beside them: all lead bands, winter, winter evening peaks (16:00 to 20:00), cold nights (daily
  mean under about 3 degrees C) with q95 coverage, holidays and the days after them, and
  Christmas; tail coverage and pinball at q01, q05, q95, q99; the training-versus-validation gap.

## Predictions

- **P1.** SEQ-median minus R is between -0.03 and +0.05 points at D+1 in the scored year; the
  most likely outcome is that R is as good (about 5,000 samples cannot teach a 14-day window more
  than a shrunk mean).
- **P2.** SEQ-quantile minus SEQ-median is between -0.05 and +0.02.
- **P3.** Cold-night q95 coverage of SEQ-quantile is below 83% (K0: 73 to 78%).
- **P4.** The pinball loss on the training rows is more than 20% below the validation rows'.
- **P5.** No arm improves holidays or Christmas by more than 0.1 points (the networks see few).

## Decision rule

**The threshold is 0.02 points at D+1.** A serving-path model is a new component that has to be
trained, versioned, and monitored, so a gain smaller than 0.02 points would not pay for it. SEQ is
`supported` if SEQ-median or SEQ-quantile beats R by at least 0.02 points with a 99% interval
excluding zero in the scored year, the replicate months agree in sign, and the cold-night q95
coverage is not worse than R's. It is `small gain` if the interval excludes zero but the gain is
under 0.02. It is `not supported` otherwise, and a clean negative is a fine result. A training-
versus-validation gap over 20% is reported but does not by itself change the verdict.

## Known-answer gate on real residuals (before any scored fit)

**On the tuning months, a level step of +3 points and a Saturday-evening shift of +3 points for
eight weeks are added to the real residual input and to the real targets.** Gate conditions: the
trained network (trained on clean data, not on the injection) adapts within 3 days to the level step
(recovers at least 50% of the step from the third day) and recovers at least 15% of the Saturday
shape by the third injected Saturday, as experiment 134's gate required of RB. The **absorption
test** follows experiment 135: the network is also retrained with the injection present in the
training windows; it passes only if it does not recover more than the clean-trained network from
days it has not seen, i.e. it must not memorise the injected pattern from the calendar. A failing
gate stops the experiment before any replicate or scored forecast is read.

## What this experiment will not say

Five to six substations and one independent winter; the network is trained on GAM-K's residuals on
ensemble-driven weather, not on the production XGBoost blend; the result bears on a research
engine, not on a model release.
