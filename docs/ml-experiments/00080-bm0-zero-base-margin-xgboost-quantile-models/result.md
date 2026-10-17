---
id: "00080"
slug: bm0-zero-base-margin-xgboost-quantile-models
status: done
size: simple
hypothesis: >-
  XGBoost starts a quantile model from a base score that it estimates from the training labels,
  one value for each of the nine quantile levels (experiment 78 found 0.403 at the 5% level to
  0.899 at the 95% level for one series, with 500 boosting rounds and no early stopping).
  Replacing that start with zero (a base margin of zero in every row, which is the same as a base
  score of zero) lowers the quantile score of the production-style models for Primary substation
  demand (retrained weekly), bulk supply point and grid supply point demand (retrained weekly),
  solar farms, and wind farms. The gap is not a symptom of too few boosting rounds, so four times
  as many rounds, or early stopping, does not remove it.
prediction: >-
  Primary substations, weekly retraining, no correction: zero start minus estimated start is -0.04
  points of capacity (-0.10 to +0.01), about half the -0.071 that experiment 78 found for the
  reference model `d0`, because a Primary substation's quantile fractions are less extreme than a
  solar farm's. Bulk supply and grid supply points are 0.00 to -0.10, solar farms 0.00 to -0.10,
  and wind farms 0.00 to -0.05, each with an interval too wide to resolve on 4, 6, and 3 series.
  The frozen N1 stack (14-day correction and spread factor) gains less than the uncorrected weekly
  model, by about half, because the correction removes part of the start-dependent error. Rounds:
  XGBoost's quantile leaves are set to the quantile of the residuals in each leaf, scaled by the
  learning rate 0.05, so the distance to the fitted level shrinks by 0.95 per round and is
  negligible after 500 rounds. The plan therefore expects the zero-start gap to shrink by less
  than half when the rounds rise from 500 to 2000, and early stopping to land between 300 and 1500
  rounds. If the gap vanishes at 2000 rounds, the estimated start costs the model through the
  rounds budget, and the learning curves (one validation block) will show the estimated start
  still improving at round 500.
decision_rule: >-
  Four verdict contrasts, quantile score over all leads on the 292 runs of `fold_a_daily_gap_v1`,
  zero start minus the estimated start of experiment 48, read at 98.75% (a 95% interval split
  across four) from 20,000 resamples of whole validation months, with each run counted in its
  origin month (experiment 48's interval). C1 is the 16 Primary substations retrained weekly,
  spread factor 1.15, no correction. C2 is the 4 bulk supply and grid supply points, the same
  configuration. C3 is the frozen N1 stack of experiment 33 (frozen model, 14-day correction,
  spread factor) at the 16 Primary substations, the arm that serves today. C4 is the 6 solar
  farms, retrained monthly on ERA5 with the month-by-month spread rule. The wind farms are not a
  verdict contrast, because experiment 48's wind model supplies a power-curve base margin, which
  makes XGBoost ignore its starting value, so a zero start changes nothing (tested). Wind is read
  in an exploratory contrast on wind models without the margin (see "Arms"). Seed 0 only, as
  experiment 48. The threshold is zero for every class, because the change replaces one argument
  of the XGBoost call and adds no pipeline step. The verdict word follows C1, the class that is
  the production recipe: `supported` when the upper end of C1 is below zero, `refuted` when the
  lower end is above zero, `inconclusive` otherwise; `small gain` takes precedence over
  `supported` when the upper end is below zero and the point estimate is above -0.05. Every class
  is reported separately, and the write-up says which classes gain, because the classes are
  different pipelines. A second, exploratory family at 95% reads the rounds: the zero-start gap at
  500, 1000, and 2000 rounds from one 2000-round fit per start and series (the frozen
  configuration of each class, predicted from prefixes of the same boosters), and the gap with
  early stopping on the final 90 days of the training window. The gap is "removed by more rounds"
  when the gap at 500 rounds has an interval excluding zero, the interval at 2000 rounds includes
  zero, and the point estimate at 2000 rounds is less than half the gap at 500 rounds. The rounds
  family is uncorrected and frozen, so it speaks about the frozen model, not the weekly recipe.
  Early stopping scores the mean pinball loss over the nine levels on the held-out block, with the
  recency weights of the training rows, and predicts from the first `best_iteration + 1` rounds.
  The starting values (per level, per series) and the learning curves are printed before any arm
  is scored. The estimated start's reproduction is checked on the first four weekly refits and the
  frozen refit (refitted here, compared with experiment 48's stored quantiles to 0.001).
benchmark: >-
  Experiment 48's stored predictions (`anchor_preds/n1h` weekly regime, `anchor_preds/frozen`,
  `generator_preds/monthly`), assembled with experiment 48's code into the same arms, and
  `manual_heuristic` and `climatology` as stored.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of
  capacity (1 point is 1% of the series' effective capacity), after the spread factor (and the
  14-day correction for C3).
code: >-
  ml-experiments/00080-bm0-zero-base-margin-xgboost-quantile-models/
verdict: inconclusive
best_arm: >-
  No arm beat its twin by a resolvable margin. At the 16 Primary substations the weekly zero start
  scored 5.104 points of capacity against 5.119 for XGBoost's estimated start (C1, -0.017, 98.75%
  interval -0.072 to +0.054).
headline: >-
  Starting XGBoost from zero did not lower the quantile score of any production-style model:
  Primary substations -0.017 points of capacity (98.75% interval -0.072 to +0.054), bulk and grid
  supply points +0.006 (-0.091 to +0.092), the frozen Primary stack -0.033 (-0.084 to +0.031),
  solar farms -0.010 (-0.036 to +0.028). Wind models without the power-curve margin were worse by
  +0.049 (95% interval +0.032 to +0.065).
lessons: >-
  Experiment 78's 0.071-point gain from a zero start does not carry over to the production recipe,
  so the starting value is not worth another experiment for demand or solar. Four times as many
  rounds raised the Primary score by 0.211 points (0.100 to 0.352) with the estimated start, so
  the fixed 500 rounds are not too few, and early stopping on the last 90 days of the training
  window stopped at 241 to 2,000 rounds and scored worse than the fixed 500.
production_status: not-for-production
adopted_in: none
advice:
  - >-
      Do not test the starting value of the production XGBoost quantile models again for
      demand or solar, because C1 to C4 are all within 0.1 points of zero, with 98.75%
      intervals that include zero (C1 to C4).
  - >-
      Keep XGBoost's estimated starting value for wind models without a power-curve base
      margin, because a zero start scores 0.049 points (0.032 to 0.065) worse; with the margin
      the starting value is ignored (wind read).
  - >-
      Keep the fixed 500 boosting rounds for demand models, because 2000 rounds scored 0.211
      points (0.100 to 0.352) worse at the 8 Primary substations and 0.233 (0.084 to 0.453)
      worse at the 4 bulk and grid supply points (rounds read).
  - >-
      Do not pick the number of boosting rounds by early stopping on the last 90 days of the
      training window, because the held-out loss kept falling to 1,000 rounds while the scored
      year worsened beyond 500, and the stopping round ranged from 241 to 2,000 (rounds read).
  - >-
      Run an identical refit of the reference arm to measure the noise floor of GPU XGBoost,
      because two fits of the same 500-round model scored 0.004 points apart at Primary
      substations and 0.007 at grid supply points (reproduction read).
---

# Experiment 80: does a zero starting value lower the error of the production XGBoost models? (BM0)

**Take-home message: starting XGBoost from zero instead of from its own estimate does not lower the
error of the production forecasts, so the 0.071-point gain that experiment 78 found for its
reference model is not a lever for the production recipe.** At the Primary substations, the bulk and
grid supply points, and the solar farms, the difference is under 0.04 points of capacity, with
intervals that include zero. For wind models without a power-curve margin the zero start is worse
by 0.049 points. More boosting rounds do not help either start, and early stopping on the last 90
days of the training window picks the number of rounds badly. The next idea to test is the number
of rounds itself (about 250 to 500 rather than 2000), not the starting value. The verdict is
`inconclusive`: the pre-registered interval for the headline contrast includes zero, which here
means no gain could be shown.

XGBoost's quantile model begins from a starting prediction (the base score) that it estimates from
the training labels, one value per quantile level. We refitted the production-style models with a
starting prediction of zero and compared them with experiment 48's stored models on the same
292 daily forecast runs from July 2025 to June 2026. The classes were Primary substation demand
(retrained weekly), bulk and grid supply point demand (retrained weekly), solar farms (retrained
monthly), and wind farms. We also asked whether more boosting rounds, or early stopping, make the
estimated start catch up.

The quantile score is an error measure in which lower is better, and 1 point is 1% of a series'
capacity.

![Zero start minus estimated start for the four planned contrasts and the wind models](figure_1_contrasts.svg)

![The gap at 500, 1000, and 2000 rounds and with early stopping, by class](figure_2_rounds.svg)

![Held-out loss round by round for the two starts](figure_3_learning_curves.svg)

- **Arms:** XGBoost's estimated start (experiment 48's stored predictions) against a zero start
  (`base_score` 0, which is exactly the same model as a zero base margin), refitted here with the
  same rows, weights, seed 0, spread factor 1.15, and for C3 the 14-day correction.
- **Row set:** `fold_a_daily_gap_v1`, 292 daily runs; intervals resample whole origin months.
- **C1, Primary substations, weekly refits:** 5.104 against 5.119 points; difference -0.017
  (98.75% interval -0.072 to +0.054). Per series it ranges from -0.107 to +0.151.
- **C2, bulk and grid supply points, weekly refits:** the 2 bulk supply series score 9.417 against
  9.409 and the 2 grid supply series 17.258 against 17.223; difference +0.006 (-0.091 to +0.092).
- **C3, Primary substations, frozen model with 14-day correction:** 5.302 against 5.339;
  difference -0.033 (-0.084 to +0.031).
- **C4, solar farms, monthly refits:** 4.961 against 4.971; difference -0.010 (-0.036 to +0.028).
- **Wind (exploratory, 95%):** experiment 48's wind model supplies a power-curve base margin,
  which makes XGBoost ignore its starting value, so a zero start is a no-op there. On wind models
  without the margin the zero start scores 11.100 against 11.051 (+0.049, +0.032 to +0.065), and
  the margin model scores 10.989.
- **Why the start matters little:** the zero start begins with a much higher held-out loss
  (Primary 0.22 against 0.05 in the first rounds) and has nearly caught up by 250 rounds.
- **Rounds (exploratory, 95%):** the zero-start gap at the 8 Primary substations is +0.015
  (-0.067 to +0.101) at 500 rounds, +0.112 (+0.031 to +0.204) at 1000, and +0.150 (+0.071 to
  +0.243) at 2000, so more rounds do not close the gap. The rule "removed by more rounds" is not met
  for any class. For the estimated start, 2000 rounds scored 0.211 points worse than 500
  (+0.100 to +0.352) at Primary substations.
- **Early stopping (exploratory):** the stopping round ranged from 241 to 2,000; the held-out
  loss kept falling past 1,000 rounds while the scored year worsened beyond 500, because the
  held-out block is the three months before the training end and the scored year spans all seasons.
- **Noise floor:** GPU fits are not bit-reproducible. A refit of the estimated-start frozen model
  scored +0.004 (Primary) and -0.007 (grid supply points) points from experiment 48's stored
  predictions, above the planned tolerance of 0.001 and well below the intervals. Single quantiles
  moved by up to 0.06 of capacity.
- **Deviations, decided before any result was read:** weekly zero-start refits were fitted for
  every second weekly refit (26 of 52), because the full set would have taken about 35 hours on the
  shared GPU, and the contrasts keep the runs both arms share (1.7 million of the 3.2 million
  Primary cases). The rounds family covers 8 of the 16 Primary substations (every second), for the
  same reason. The reproduction check covers 2 weekly refits, not 4, and has no pass flag on the
  weekly quantiles.
- **Caveats:** one seed, so intervals reflect months only; with 4 grid supply series, 6 solar
  farms, and 3 wind farms the intervals are wide; the validation block of the early-stopping arm is
  not purged of runs that began before the cut.
- **Next:** test fewer rounds (250 against 500) with the stock start, since 2000 rounds hurt.
