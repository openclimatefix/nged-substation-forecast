---
id: "00142"
slug: differentiable-physics-sequence-model-on-gam-k-residuals
title: "Differentiable physics: a small sequence model on GAM-K's recent residuals"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction, calibration-and-uncertainty]
status: done
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
verdict: inconclusive
best_arm: >-
  none: the known-answer gate failed, so no replicate or scored forecast was read
headline: >-
  On real residuals, the network recovered -10% of an injected level step (R 0%) and 0% of an
  injected Saturday-evening shift (R -9%), against gates of 30% and 15%; it lowered the day-ahead
  mean absolute error on the tuning months' held-out blocks by only 1% to 3% against K0.
lessons:
  - >-
    With about 1,100 training samples (one winter of days per substation), a gated recurrent unit
    plus a linear skip from the residual profile learned no response to injected changes in the
    residuals.
  - >-
    The gate was run twice: the first architecture (recurrent unit alone) recovered -5% and 0%; the
    second, with a linear skip from the 14-day residual profile, recovered -10% and 0%.
production_status: not-for-production
adopted_in: none
date: 2026-10-21
take_home_message: >-
  A small neural network that reads the last 14 days of GAM-K's errors did not learn to adapt to
  changes in them: when we added a known step or a Saturday-evening shift to real errors, it
  corrected for none of it, so we did not score it on the scored year. With about one year of days
  per substation to learn from, the hand-built Kalman filter with recipe R remains the better tool.
advice:
  - >-
    Do not train a sequence network on one substation-year of GAM-K residuals (about 1,100 to 3,200
    samples); it learns no response to changes in the residuals (known-answer gate).
  - >-
    If a learned residual corrector is tried again, pretrain it on simulated or longer history, such
    as the 2021 to 2024 ensemble backfill, before using the stored GAM-K residuals (known-answer
    gate).
code: ml-experiments/00142-differentiable-physics-sequence-model-on-gam-k-residuals/
---

**Take-home: a small neural network trained on one year of GAM-K's errors did not learn to adapt
when those errors change, so it was not scored.** We added a known shift to real errors and asked
whether the network, trained on clean errors, corrected for it. It corrected for none of it, in two
architectures. The experiment therefore stops at the known-answer gate, and the verdict is
`inconclusive`: the day-ahead score was not measured. The search narrows to a network with more
history to learn from, not a different architecture on the same year.

## Question

**Can a small neural network, trained on the stored residuals of GAM-K, do the job of the Kalman
filter and recipe R, and does it give better tails?** GAM-K is the convex demand model with a Kalman
filter on ECMWF ensemble weather ([experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)).
Recipe R subtracts 0.75 times the mean of the last 14 days' errors by half-hour
([experiment 115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md)); the
[same-weekday
profile](../00134-differentiable-physics-same-weekday-error-profile-on-gam-k/result.md)
added only 0.011 points. The network pools the substations with a substation embedding, so patterns
shared between substations can be learned. Sequence models of recent errors follow
[Hochreiter and Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735) and
[Cho et al. (2014)](https://doi.org/10.3115/v1/D14-1179) (the gated recurrent unit used here);
the pinball loss is the standard training loss of quantile regression
([Koenker and Bassett, 1978](https://doi.org/10.2307/1913643)).

## Design

- **Substations.** The six substations whose member forecasts experiment 111 stored (D5, D10, D11,
  D14, D16, and D9). Experiment 111's ensemble store holds seven series (those six and D18), so the
  other DP series have no stored GAM-K member forecasts, and no demand model is refitted. D9 is
  trained on but reported beside the five decision substations.
- **One sample per (substation, run).** 810 runs per substation, of which about 530 precede the
  scored year: about 3,200 training samples, and about 1,100 for the odd-months-only fit.
  Consecutive samples share 13 of 14 input days, so the independent information is about one year of
  days per substation. Target: the
  48 half-hours of day D+1 (leads 15.5 to 39 hours), as actual minus K0's median forecast.
- **Clock.** Every day, the day type, the holiday flag, and the output index use the UTC day and the
  UTC half-hour (0 to 47), the day D+1 being a UTC day. No local clock enters.
- **Input.** The 14 UTC days before the origin's day, plus a 15th partial day, as vectors of 48
  half-hour residuals from `SiteData.stream` (observed minus the chained GAM fit, driven by the
  analysis proxy, ending at the origin's latest usable reading, 3.5 hours before the 09:00 UTC
  origin) with a reading mask; each day carries its day type (five types, as in experiment 134) and
  a holiday-like flag. Missing readings are zero with mask zero. The target day enters through its
  day type, holiday-like flag, Christmas-window flag, and the calibrated ECMWF ensemble-mean
  temperature and irradiance at its 48 half-hours (published by the origin, so available at
  forecast time; past days use the analysis proxy, as in experiment 111). The target day's actuals
  are never an input.
- **Network.** A one-layer gated recurrent unit (32 hidden units) over the 15 day vectors, a
  substation embedding (4 dimensions), and a two-layer head (64 units) producing 144 numbers per
  sample: a median correction per half-hour, and a lower and an upper log-scale per half-hour that
  stretch K0's own quantile offsets (the distance of each K0 quantile from K0's median), so the 11
  quantiles cannot cross. About 35,000 parameters. AdamW, weight decay, dropout 0.1. Early stopping
  uses
  month-block validation folds, purged by 15 days either side of the held-out block; the
  early-stopped model is the one used (no refit).
- **Arms.** K0; R; SEQ-ridge (a ridge regression of the same median correction on the same inputs,
  the linear reference; if the network does not beat it, a network is not needed); SEQ-blind (the
  SEQ-quantile network with the residual input zeroed, so that any gain from the weather and
  day-type scale heads alone can be told apart from reading the residuals); SEQ-median (median
  correction trained on absolute error; the spread is rebuilt
  by experiment 111's routine as for every other arm); SEQ-quantile (the correction and the two
  scales trained with the pinball loss at the 11 quantiles).
- **Training sets.** Replicate comparison: train on the odd (tuning) months only and score the even
  months. Scored-year comparison: train on all runs whose valid times fall before 2025-07-01
  (embargoing the last 2 days), early stopping on a month-block fold held out of those. The final
  forecast averages the fold models (3 folds for the replicate fit, 5 for the scored-year fit) and
  2 seeds each; the seed is fixed per job.
- **Compute.** About 3,000 samples and 35,000 parameters: seconds per fit, on the GPU if it is
  free and on the CPU otherwise, far inside the 2 GPU-hour cap. Every script runs under
  `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` with one thread per worker.
- **Other leads.** The network outputs day D+1 only. Every arm built on it equals K0 at other leads,
  so the all-lead columns show R's lead-band behaviour against K0 and SEQ's equal to K0.
- **Reuse.** Rows, periods, scoring, bootstrap, breakdowns, and charts are experiment 134's, copied
  into this folder with the prefix `e00142_`. No change to `src/` or `packages/`.

## Planned contrasts (99% month-resampled paired intervals; D-labels only)

- **C0.** SEQ-quantile minus SEQ-blind (does reading the residuals help beyond the weather and
  day-type scale heads).
- **C1.** SEQ-median minus R at D+1, scored year (decides the verdict), and replicate months.
- **C2.** SEQ-quantile minus R at D+1, scored year, and replicate months.
- **C3.** SEQ-quantile minus SEQ-median at D+1 (does the quantile loss help the tails).
- **C4.** SEQ-median minus K0, SEQ-quantile minus K0, and SEQ-median minus SEQ-ridge.
- Beside them: all lead bands, winter, winter evening peaks (16:00 to 20:00), cold nights (daily
  mean under about 3 degrees C) with q95 coverage, holidays and the days after them, and
  Christmas; tail coverage and pinball at q01, q05, q95, q99; the training-versus-validation gap.

## Predictions

- **P1.** SEQ-median minus R is between -0.03 and +0.05 points at D+1 in the scored year, with a
  99% interval about 0.03 wide; the most likely outcome is a null.
- **P2.** SEQ-quantile minus SEQ-median is between -0.05 and +0.02, and SEQ-quantile minus R is
  between -0.05 and +0.03.
- **P3.** Cold-night q95 coverage of SEQ-quantile is between 74% and 82% (K0: 73 to 78%); it does
  not
  reach 90%.
- **P4.** SEQ-quantile minus SEQ-blind is between -0.03 and 0.00: most of any tail gain comes from
  the weather and day-type heads, not from the residual history.
- **P5.** SEQ-ridge scores within 0.02 points of SEQ-median.
- **P6.** The validation pinball loss exceeds the training loss by 10% or more in the replicate fit
  and by less than that in the scored-year fit.
- **P7.** Holidays and Christmas have too few rows (about 400 and 100 per substation-year) for any
  contrast to exclude zero.

## Decision rule

**The threshold is 0.02 points at D+1, applied to the point estimate; the 99% interval must also
exclude zero.** A serving-path model is a new component that has to be trained, versioned, and
monitored, so a smaller gain would not pay for it. The primary arm for the verdict is SEQ-median
against R (C1); SEQ-quantile against SEQ-median (C3) decides the tail claim only. SEQ-median is
`supported` if C1 is at most -0.02 with the interval excluding zero in the scored year, the
replicate months agree in sign, and SEQ-median beats SEQ-ridge. It is `small gain` if the interval
excludes zero but the estimate is above -0.02. A tail gain is supported if C3 excludes zero below
zero and cold-night q95 coverage moves at least 5 points towards 95% without SEQ-blind doing the
same. Otherwise the verdict is `not supported`, and a clean negative is a fine result. A failed gate
caps the verdict at `inconclusive`. The scored-year network trains on all pre-scored months while
R's kappa was tuned on the odd months only; this asymmetry favours the network and is a caveat.

## Known-answer gate on real residuals (before any scored fit)

**Two injections are added to the real tuning-month residual inputs and targets of the validation
folds, and the network was trained on clean data.** (a) A level step of +3 points for eight weeks:
the network must recover at least 30% of the step from the third day on and more than R does.
(b) A +3-point shift on Saturdays 17:00 to 21:00 UTC for eight weeks: it must recover at least 15%
of the shift from the third injected Saturday on, as experiment 134's gate required of RB. The
networks are scored on the validation folds that they did not train on, and the gate tests the
architecture's ability to adapt on real residuals, not its usefulness.

**The absorption test.** A second network is trained with the injection present in the training
weeks A (the first four of the eight), and is evaluated on the later held-out weeks B twice, with
the injection present and absent. It passes if, on the clean held-out weeks, its correction at the
injected half-hours is under 25% of the shift: it must adapt from the residual inputs and not
memorise the calendar position. A failed gate stops the experiment before any replicate or scored
forecast is read.

## What this experiment will not say

Five to six substations and one independent winter; the network is trained on GAM-K's residuals on
ensemble-driven weather, not on the production XGBoost blend; the result bears on a research
engine, not on a model release.

## Results

- **Gate, first architecture (gated recurrent unit alone).** Level step of +3 points for eight weeks
  (April to June 2024, before every tuning month): the network recovered -5% of it from the third
  day (gate: at least 30% and more than R); R recovered 0%, because recipe R removes each day's
  mean. Saturday-evening shift of +3 points: the network recovered 0.04% (gate: 15%). The
  absorption test was ill-defined as first written (it compared against zero and not against the
  clean-trained network), so the run's absorption number is not used.
- **Gate, second architecture (post hoc revision before any replicate or scored read).** A linear
  skip from the 14-day mean residual by half-hour, the previous day, and the partial current day was
  added, which can represent recipe R exactly. The network recovered -10% of the level step and
  -0.1% of the Saturday shift; R recovered 0% and -9%. The absorption test (training with the
  shift in four of the eight weeks, evaluating the other four on clean inputs, correction minus the
  clean-trained network's) gave 2.2% on clean weeks, below the 25% limit and 2.6% on injected weeks,
  so the network did not memorise the calendar, but it did not adapt either.
- **Fit quality on the tuning months.** Day-ahead mean absolute error of the median on three
  held-out month blocks (purged by 15 days): network 3.97, 4.04, 4.43 against K0 4.02, 4.06, 4.57
  points, a 1% to 3% reduction; the training error is lower than the validation error by 3% to 26%,
  the largest gap on the block with the fewest training days.
- **Arms and rows not run.** SEQ-median, SEQ-quantile, SEQ-ridge, and SEQ-blind were not scored in
  the replicate months or the scored year, as the plan requires after a failed gate. The code for
  the fits and frames is in the folder but was not run.
- **Predictions.** P1 to P5 and P7 were not tested. P6 held for the replicate-type fit (the
  validation error exceeded the training error by more than 10% in two of three blocks).
- **Checks.** No audit trigger fired except trigger 1 (a failed control), whose explanation is the
  small training set and whose investigation is the first-architecture and second-architecture
  comparison above.
- **Caveats.** The gate period (April to June 2024) is spring, not winter. The skip path could
  represent recipe R, so a failure to recover the shift means the optimiser found no use for it in
  about 1,100 noisy samples, not that the architecture could not express it.
- **What to try next.** Train on longer history (the 2021 to 2024 ensemble backfill, if the
  maintainer approves its use) or on simulated residuals, and re-run this gate before any score.
