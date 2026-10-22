---
id: "00244"
slug: dp-deep-ensemble-heads-cold-night-bands
title: >-
  Differentiable physics: does the disagreement between independently trained correction networks
  set the width of the cold-night bands?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00244-dp-deep-ensemble-heads-cold-night-bands/
  only. Serving path: not touched. Degradation rule: not touched. Designs: several are defensible
  (what the spread is, how the replicate year is trained without leaking, how the placebo shuffles),
  and this plan pre-registers one of each. Callers: this folder's scripts, which copy experiment
  211's network code and experiment 169's linear-programme fit, and read experiments 155, 164, 169,
  and 201's stored rows. A new method for this line (a deep ensemble of heads), so a Sonnet plan and
  code review; no Opus.
hypothesis: >-
  On cold nights the demand forecast's error is partly something the correction network does not
  know (epistemic uncertainty, as opposed to the weather forecast's own uncertainty, which
  experiment 192 found the 51 ensemble members' temperatures do not signal). The disagreement
  between 20 independently trained copies of the pooled correction network therefore predicts the
  absolute cold-night error, and widening the bands by a non-decreasing function of that
  disagreement lifts cold-night q95 coverage to 0.90.
prediction: >-
  Written before any score is read. (1) On cold nights (winter months, night and evening hour
  bands, ERA5 day mean below 2 C) the spread of the 20 median corrections of the seed ensemble
  correlates with the absolute error of arm D's median at Spearman 0.2 or more, in the scored year
  and in the replicate year. (2) E3 lifts cold-night q95 coverage from experiment 169 arm A's 0.896
  to 0.90 or more, with the all-row score within +0.01 of E0 and q05 and q95 coverage within one
  point of nominal on all rows, and the placebo P does not lift it. (3) The seed-to-seed spread of
  the single network's gain over arm D is under 0.02 points across the 20 seeds.
decision_rule: >-
  Gate first. The seed-ensemble spread of the network without error inputs must be larger on rows
  whose forecast-level input carries an injected cold-night error than on the same rows without it
  (paired ratio of mean spreads at least 1.1, and the 95% interval over resampled days excludes 1);
  if it is not, the verdict is capped at `inconclusive` and no contrast is read as evidence.
  `supported` needs prediction 1 to pass in both years and prediction 2 to pass in the scored year,
  with P's cold-night q95 coverage lift under half of E3's. `partly supported` is prediction 1
  passing and prediction 2 failing (the spread is informative but the bands need another
  transform; coverage by spread decile is then the finding). `refuted` is the scored-year Spearman
  under 0.1. Anything else is `inconclusive`. Threshold, set by cost: the band scaling adds 20
  forward passes of a small network at serving time and 3 coefficients for each of 8 quantile
  levels, so it is worth porting only if it reaches 0.90 on cold nights with an all-row cost of at
  most +0.01 points (under 0.4% of the score). Prediction 3 is reported and does not change the
  verdict.
benchmark: >-
  E0 = experiment 169 arm A (arm D with experiment 160's smooth spread table and experiment 164's
  one-sided widening), rescored on these rows; arm D; the single network without error inputs
  (experiments 200 and 211)
row_set_id: >-
  experiment 211's scored rows: scored year 2025-07-02 to 2026-06-29, leads 15.5 to 39 hours,
  outside switching events, at D5, D10, D11, D14, and D16 (D9 beside, in no contrast)
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  and coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles, all rows and cold
  nights
code: ml-experiments/00244-dp-deep-ensemble-heads-cold-night-bands/
---

# Experiment 244: does the disagreement between independently trained correction networks set the cold-night band width?

Status `planned`: the arms, the gate, and the decision rule above are fixed before any score is read.
This page is rewritten as the write-up when the run finishes.

- **Arms.** E0: experiment 169 arm A. E1: 20 networks of the same design trained with different
  seeds, once without error inputs (the reference of experiment 211) and once with the error inputs
  of experiment 200; the mean of the 20 median corrections replaces the single network's
  correction. E2: the same with 20 bootstrap resamples of the training months (whole months). E3:
  E0's quantile offsets times `c + a1 max(z - k1, 0) + a2 max(z - k2, 0)` with `z` the spread of the
  20 median corrections divided by the head set's median spread, knots at the tuning-month 50th and
  90th percentiles, `c` in [0.3, 3] and `a` non-negative, fitted by linear programme on the tuning
  months (odd months of 2024-07 to 2025-06) only. P: E3 with `z` shuffled among rows of the same
  substation, valid month, and hour band.
- **Primary E3.** The reference network's seed-ensemble spread (E1). The bagged spread (E2) and the
  error-input networks are descriptive.
- **Replicate year without leakage.** The networks trained to 2025-06-28 have seen 2024-07 to
  2025-06, so their spread there is in-sample. For that year, two cross-fit sets hold out 2024-07
  to 2024-12 and 2025-01 to 2025-06 in turn (15-day purge, early-stopping block in the other half),
  and predict the held-out half.
- **Gate input.** The networks have no temperature input; a +4 C error on 2% of the scored year's
  cold-night rows is injected through arm D's forecast-level input, shifted by 4 C times the
  substation's cold-night slope of the forecast level on ensemble-mean temperature.

## Plan-review amendments (made before any score or spread was read)

- **Early stopping and purge.** The `full` and `cfx` sets stop on 2025-04-28 to 2025-06-24; `cfy`
  stops on 2024-10-01 to 2024-11-10, because its held-out half contains the usual block. All
  early-stopping blocks come from the training half. A 15-day purge sits on both sides of each
  held-out half and each early-stopping block. The cross-fit networks see 3 to 5 months of ensemble
  weather rows, fewer than the full set's 12, so the replicate-year numbers are not like-for-like
  with the scored year.
- **Normalisation and knots.** The spread `z` is divided by the median spread of the tuning months
  of the same head set, and the knots are the 50th and 90th percentiles of the tuning months. The
  scored year is divided by the full set's own tuning-month median. The tuning-month spread
  distributions of the cross-fit and the full sets are plotted side by side.
- **Months.** E3 is fitted on all rows of the odd months (tuning) of 2024-07 to 2025-06, never on
  cold nights alone. The replicate-year test of prediction 1 uses the even months; the odd months
  are reported beside it. Cold-night row and day counts of each period are printed; if the tuning
  months hold fewer than 300 cold-night rows, the cold-night result is read from the scored year
  and the replicate even months alone.
- **Intervals.** Spearman correlations and coverages carry 95% intervals resampled by valid day.
  Prediction 1 passes when the point estimate is at least 0.2 and the interval's lower bound is
  above 0.1. E3 passes prediction 2 on a point estimate of 0.90 or more, and the write-up states
  whether the lift over E0 exceeds its interval.
- **Placebo.** Primary P shuffles `z` among rows of the same substation, valid month, and hour band,
  20 times (seeds 0 to 19); the reported lift is the mean over shuffles. A second placebo shuffles
  among all substations (descriptive). P is read as a difference from E3 in cold-night q95 coverage
  over resampled days. E3 must lift coverage by at least 0.005 for the "P's lift under half of E3's"
  clause to apply.
- **Gate.** The forecast level (`mean`, divided by 100) is one of the network's inputs, so the
  injection reaches the networks. The cold-night slope of the forecast level on ensemble-mean
  temperature is fitted on the training months (2020 to 2025-06) of each substation. The rows are
  picked with seed 244, the spread is computed twice on the same rows (clean and injected), and
  the number of injected rows and days is printed. The gate reads spread only, no coverage or score.
- **Prediction 3.** The seed-to-seed spread is the standard deviation, across the 20 seeds, of each
  single network's scored-year all-row quantile-score gain over arm D, using the network's own
  bands (arm D offsets), in points of effective capacity.
- **Extra arms and reports.** E1's own score (ensemble-mean centre, arm D offsets) is reported so the
  change of centre is separate from the band scaling; E3's centre is E0's mean. E3 fits the levels
  0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, and 0.99 as experiment 169 did; the middle three levels
  (0.40, 0.50, 0.60) keep E0's offsets. The spread of the log scales is reported as a second
  covariate (Spearman only). E3 on the bagged spread (E2) is fitted too and reported as descriptive.
  The coverage by spread decile is the first figure, and the 20 seed gains and the placebo follow.
  Experiment 201's three stored corrections are not used. Experiments 160, 164, 169, 192, 200, 201,
  209, and 211 are cited in the write-up.
