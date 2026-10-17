---
id: "00080"
slug: bm0-zero-base-margin-xgboost-quantile-models
status: planned
size: simple
hypothesis: >-
  XGBoost starts a quantile model from a base score that it estimates from the training labels, one
  value for each of the nine quantile levels (experiment 78 found 0.403 at the 5% level to 0.899 at
  the 95% level for one series, with 500 boosting rounds and no early stopping). Replacing that start
  with zero (a base margin of zero in every row, which is the same as a base score of zero) lowers
  the quantile score of the production-style models for Primary substation demand (retrained
  weekly), bulk supply point and grid supply point demand (retrained weekly), solar farms, and wind
  farms. The gap is not a symptom of too few boosting rounds, so four times as many rounds, or early
  stopping, does not remove it.
prediction: >-
  Primary substations, weekly retraining, no correction: zero start minus estimated start is -0.04
  points of capacity (-0.10 to +0.01), about half the -0.071 that experiment 78 found for the
  reference model `d0`, because a Primary substation's quantile fractions are less extreme than a
  solar farm's. Bulk supply and grid supply points are 0.00 to -0.10, solar farms 0.00 to -0.10, and
  wind farms 0.00 to -0.05, each with an interval too wide to resolve on 4, 6, and 3 series. The
  frozen N1 stack (14-day correction and spread factor) gains less than the uncorrected weekly
  model, by about half, because the correction removes part of the start-dependent error. Rounds:
  XGBoost's quantile leaves are set to the quantile of the residuals in each leaf, scaled by the
  learning rate 0.05, so the distance to the fitted level shrinks by 0.95 per round and is
  negligible after 500 rounds. The plan therefore expects the zero-start gap to shrink by less than
  half when the rounds rise from 500 to 2000, and early stopping to land between 300 and 1500 rounds.
  If the gap vanishes at 2000 rounds, the estimated start costs the model through the rounds budget,
  and the learning curves (one validation block) will show the estimated start still improving at
  round 500.
decision_rule: >-
  Five verdict contrasts, quantile score over all leads on the 292 runs of `fold_a_daily_gap_v1`,
  zero start minus the estimated start of experiment 48, read at 99% (a 95% interval split across
  five) from 20,000 resamples of whole validation months, with each run counted in its origin month
  (experiment 48's interval). C1 is the 16 Primary substations retrained weekly, spread factor 1.15,
  no correction. C2 is the 4 bulk supply and grid supply points, the same configuration. C3 is the
  frozen N1 stack of experiment 33 (frozen model, 14-day correction, spread factor) at the 16
  Primary substations, the arm that serves today. C4 is the 6 solar farms, retrained monthly on ERA5
  with the month-by-month spread rule. C5 is the 3 wind farms, the same configuration; the wind
  model keeps its power-curve base margin in both arms, and the zero start replaces the estimated
  start of the margin-free level only (see "Arms"). Seed 0 only, as experiment 48. The threshold
  is zero for every class, because the change replaces one argument of the XGBoost call and adds no
  pipeline step. The verdict word follows C1, the class that is the production recipe: `supported`
  when the upper end of C1 is below zero, `refuted` when the lower end is above zero, `inconclusive`
  otherwise, and `small gain` when the upper end is below zero and the point estimate is above
  -0.05. A class is read as no change when its interval lies inside plus or minus 0.05. Every class
  is reported separately, and the write-up says which classes gain, because the classes are
  different pipelines. A second, exploratory family at 95% reads the rounds: the zero-start gap at
  500, 1000, and 2000 rounds from one 2000-round fit per start and series (the frozen configuration
  of each class, predicted from prefixes of the same boosters), and the gap with early stopping on
  the final 90 days of the training window. The gap is "removed by more rounds" when its interval at
  2000 rounds includes zero and the point estimate is less than half the gap at 500 rounds. The
  starting values (per level, per series) and the learning curves are printed before any arm is
  scored. The estimated start's reproduction is checked on the first four weekly refits and the
  frozen refit (refitted here, compared with experiment 48's stored quantiles to 0.001).
benchmark: >-
  Experiment 48's stored predictions (`anchor_preds/n1h` weekly regime, `anchor_preds/frozen`,
  `generator_preds/monthly`), assembled with experiment 48's code into the same arms, and
  `manual_heuristic` and `climatology` as stored.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of capacity
  (1 point is 1% of the series' effective capacity), after the spread factor (and the 14-day
  correction for C3).
code: ml-experiments/00080-bm0-zero-base-margin-xgboost-quantile-models/
---

# Experiment 80: does a zero starting value lower the error of the production XGBoost models? (BM0)

**This page is the plan, committed before any code runs.** The take-home message, the verdict, and
the figures are written after the run.

## Question

XGBoost's quantile model begins from a starting prediction (the base score) that it estimates from
the training labels, one value for each quantile level. Experiment 78 found that starting one
reference model from zero lowered its score by 0.071 points of capacity (98.3% interval -0.099 to
-0.047). This experiment tests the same change on the models that serve, or could serve: Primary
substation demand, bulk and grid supply point demand, solar farms, and wind farms.

## Size

- **What gets stored:** predictions under `data/ml-experiments/` only, no production table.
- **Production serving path:** no change; the result is a candidate for a later release.
- **Degradation rule:** none touched.
- **More than one defensible design:** no; the contrast, the rows, and the seeds are those of
  experiment 48, and the only difference is the starting value.
- **Callers:** the code copies experiment 48's fitting and scoring modules into this folder and
  changes one argument of the XGBoost call.

That is **simple**: a hyperparameter permutation with no new data. It gets a Sonnet plan review, a
Sonnet code review, a Sonnet check of the write-up against the printed report, and a Sonnet prose
review.

## Arms

- **Estimated start:** XGBoost's default; experiment 48's stored predictions.
- **Zero start:** a base margin of zero for all nine levels in every training row and at prediction.
  The smoke test checks that this equals `base_score=0` to float precision. For the wind farms the
  power-curve margin stays in both arms, so the wind comparison is between the margin alone and the
  margin plus a re-estimated start. Because a supplied margin makes XGBoost ignore its base score,
  the two wind arms may be identical; the smoke test checks, and if they are, wind is reported as
  not testable and dropped from the family (Bonferroni across four).
- **Rounds family** (frozen configuration of each class): one 2000-round fit per start and series,
  predicted from the first 500, 1000, and 2000 rounds. A second fit per start and series on the
  training window without its last 90 days uses those 90 days for early stopping (patience 100, at
  most 2000 rounds), and its 500-round prefix is the matched fixed-round arm.

## Advice from earlier experiments

Followed: score every XGBoost reference with a start-at-zero twin and test a zero base margin as a
change to the production recipe (experiment 78); train each model on its own history; stay with
experiment 48's rows and seed. Departures: none.

## Lessons for the next experiment

Written after the run.
