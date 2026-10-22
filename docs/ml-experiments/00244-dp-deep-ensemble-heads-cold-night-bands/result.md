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
