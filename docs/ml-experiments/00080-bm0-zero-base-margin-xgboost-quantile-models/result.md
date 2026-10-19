---
id: "00080"
slug: bm0-zero-base-margin-xgboost-quantile-models
theme: calibration-and-uncertainty
also_relevant_to: []
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
  Starting XGBoost from zero did not lower the quantile score of any production-style XGBoost
  model: Primary substations -0.017 points of capacity (98.75% interval -0.072 to +0.054), bulk
  and grid supply points +0.006 (-0.091 to +0.092), the frozen Primary stack -0.033 (-0.084 to
  +0.031), solar farms -0.010 (-0.036 to +0.028). Wind XGBoost models without the power-curve
  margin were worse by +0.049 (95% interval +0.032 to +0.065).
lessons: >-
  Experiment 78's 0.071-point gain from a zero start does not carry over to the XGBoost recipe
  that serves, so the starting value is not worth another experiment for demand or solar. Four
  times as many rounds raised the Primary-substation score by 0.211 points (0.100 to 0.352) with
  the estimated start, so the fixed 500 rounds are not too few. Early stopping on the last 90 days
  of the training window gave no detectable gain at Primary substations and stopped anywhere from
  90 to 1,292 rounds at the demand series.
production_status: not-for-production
adopted_in: none
advice:
  - >-
      Do not test the starting value of the XGBoost quantile models again for demand or solar,
      because the four point estimates are within 0.04 points of zero and every 98.75%
      interval includes zero (C1 to C4).
  - >-
      Keep the power-curve base margin in the wind XGBoost model, because it beats the same
      model without the margin by 0.062 points (0.026 to 0.099) and the zero-start model
      without the margin by 0.112 (0.076 to 0.148); without the margin, a zero start is worse
      by 0.049 (0.032 to 0.065) (wind read).
  - >-
      Keep the fixed 500 boosting rounds for demand XGBoost models, because 2,000 rounds
      scored 0.211 points (0.100 to 0.352) worse at 8 Primary substations and 0.233 (0.084 to
      0.453) worse at the 4 bulk and grid supply points (rounds read).
  - >-
      Do not pick the number of boosting rounds by early stopping on the last 90 days of the
      training window for demand, because the stopping round ranged from 90 to 1,292 and the
      matched comparison at Primary substations showed no detectable gain (-0.011, -0.032 to
      +0.008) (early-stopping read).
  - >-
      Run a repeat fit of the reference arm to measure the noise floor of GPU XGBoost, because
      two fits of the same 500-round model scored 0.004 points apart at Primary substations
      and 0.007 at grid supply points (reproduction read).
---

# Experiment 80: does a zero starting value lower the error of the production XGBoost models? (BM0)

**Take-home message: starting an XGBoost model from zero instead of from its own estimate does not
lower the error of the forecasts that serve, so the zero start is not worth porting.** The gain of
0.071 points that experiment 78 found for its reference model did not appear for Primary substation
demand, bulk and grid supply point demand, or solar farms, where every difference was under 0.04
points of capacity with an interval that includes zero. For wind XGBoost models without a
power-curve margin a zero start was worse by 0.049 points. Four times as many boosting rounds did
not help either start. The verdict is `inconclusive`, which here means that no gain could be shown.

XGBoost's quantile model begins from a starting prediction (the base score) that it estimates from
the training labels, one value for each of the nine quantile levels. We refitted XGBoost models for
four classes of series with a starting prediction of zero, and compared them with the stored
models of experiment 48, which use the estimated start, on the same 292 daily forecast runs from
July 2025 to June 2026. The classes are Primary substations (the 33 kV or 66 kV to 11 kV
transformers) and bulk and grid supply points (the larger sites above them), all retrained weekly,
plus solar farms and wind farms, retrained monthly. We also asked whether more boosting rounds
(500 are used now), or early stopping, make the estimated start catch up. The score is the
quantile score, an error measure in which lower is better, and 1 point is 1% of a series' capacity.

![Zero start minus estimated start for the four planned contrasts and the wind models](figure_1_contrasts.svg)

*Each dot is the difference in mean quantile score; negative means the zero start is better.*

![The gap at 500, 1000, and 2000 rounds and with early stopping, by class](figure_2_rounds.svg)

*The zero-start gap does not close with more rounds; the early-stopping dots compare the two starts
at different stopping rounds.*

![Held-out loss round by round for the two starts](figure_3_learning_curves.svg)

*The zero start begins far behind and approaches the estimated start by about 250 rounds.*

- **Arms:** the estimated start (experiment 48's stored predictions) against a zero start
  (`base_score` 0, the same XGBoost model as a zero base margin), refitted here with the same rows,
  weights, seed 0, and spread factor 1.15, and for C3 the 14-day correction. Arm scores below use
  every run of the arm; the differences use only the runs both arms share.
- **Row set:** `fold_a_daily_gap_v1`, 292 daily runs; intervals resample whole origin months.
- **C1, Primary substations, weekly refits:** 5.104 against 5.119 points; difference -0.017
  (98.75% interval -0.072 to +0.054). Per series the difference ranges from -0.107 to +0.151.
- **C2, bulk and grid supply points, weekly refits:** the 2 bulk supply series score 9.417 against
  9.409 and the 2 grid supply series 17.258 against 17.223; difference +0.006 (-0.091 to +0.092).
- **C3, Primary substations, frozen XGBoost model with the 14-day correction:** 5.302 against
  5.339; difference -0.033 (-0.084 to +0.031).
- **C4, solar farms, monthly refits:** 4.961 against 4.971; difference -0.010 (-0.036 to +0.028).
- **Wind (exploratory, 95%):** experiment 48's wind XGBoost model supplies a power-curve base
  margin, which makes XGBoost ignore its starting value, so a zero start changes nothing there. On
  wind XGBoost models without the margin the zero start scores 11.100 against 11.051 (+0.049,
  +0.032 to +0.065), and the model with the margin scores 10.989.
- **Held-out loss by round:** on the 90 days before the training end, the mean pinball loss at
  Primary substations is 0.122 for the zero start against 0.045 for the estimated start at round
  10, 0.0288 against 0.0283 at round 250, and 0.0246 against 0.0214 at round 500.
- **Rounds (exploratory, 95%):** at 8 Primary substations the zero-start gap is +0.015 (-0.067 to
  +0.101) at 500 rounds, +0.112 (+0.031 to +0.204) at 1,000, and +0.150 (+0.071 to +0.243) at 2,000.
  The gap is not detectable at 500 rounds and grows with more rounds, so the rule "removed by more
  rounds" is not met for any class. With the estimated start, 2,000 rounds scored 0.211 points
  worse than 500 (+0.100 to +0.352) at Primary substations.
- **Early stopping (exploratory):** on the last 90 days before the training end, the stopping
  round ranged from 90 to 1,292 at the demand series. Against its own matched 500-round prefix,
  early stopping with the estimated start gained -0.011 (-0.032 to +0.008) at Primary substations
  and -0.035 (-0.070 to -0.003) at solar farms. The comparison of the two starts at each start's own
  stopping round (+0.431 at Primary substations) mostly reflects different stopping rounds.
- **Noise floor:** GPU fits are not bit-reproducible. A repeat fit of the estimated-start frozen
  XGBoost model scored +0.004 (Primary substations) and -0.007 (grid supply points) points from
  experiment 48's stored predictions, above the planned tolerance of 0.001 and well below the
  intervals. In the weekly repeat fits, single quantiles moved by up to 0.06 of capacity.
- **Deviations, made before any result was read:** weekly zero-start refits cover every second
  weekly refit (26 of 52), because all 52 would have taken about 35 hours on the shared GPU. The
  differences therefore use half the runs (1.7 million of 3.2 million Primary-substation cases,
  417,000 of 785,000 supply-point cases). The rounds family covers every second Primary substation
  (8 of 16) and all 4 bulk and grid supply series, 6 solar farms, and 3 wind farms. The
  reproduction check covers 2 weekly refits, not 4, and the planned tolerance was exceeded.
- **Caveats:** one seed, so intervals reflect months only; with 4 bulk and grid supply series, 6
  solar farms, and 3 wind farms the intervals are wide; the early-stopping held-out block is not
  purged of runs that began before the cut.
- **Next:** at the Primary substations, the median best round on the held-out block is 373, so
  fewer than 500 rounds may be enough; the number of rounds, not the starting value, is the open
  question.
