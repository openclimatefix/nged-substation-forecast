---
id: "00099"
slug: crps1-crps-type-loss-over-ens-members-primary-demand-pilot
status: planned
size: complex
hypothesis: >-
  Training the Primary-substation quantile model on the continuous ranked probability score (CRPS) of
  the pooled ensemble forecast, with several ensemble members as separate rows, scores lower on the
  quantile score and covers the tails closer to nominal than training the same rows with the
  pinball loss, because the pinball loss makes every member's row learn the whole predictive spread
  and pooling the 51 members then counts the weather uncertainty twice.
prediction: >-
  Against the pinball loss on the same 6 members per run and the same optimiser, the CRPS loss lowers
  the 11-level quantile score of the 16 Primary substations by 0.05 to 0.15 points of capacity, and
  moves the coverage of the 0.95 quantile at 12 to 36 hours from about 0.94 towards 0.95 and the
  coverage of the 0.05 quantile from about 0.04 towards 0.05.
decision_rule: >-
  One verdict contrast, C1: the quantile score of arm c1 (CRPS loss) minus arm p3 (pinball loss
  written as a custom objective, the same rows and optimiser), over the 292 runs of
  fold_a_daily_gap_v1 at the 16 Primary substations, whole origin months resampled, 20,000
  resamples, 95% interval. Supported when the interval lies wholly below -0.05 points of capacity.
  Small gain when it lies wholly below zero but its upper end is not below -0.05. Refuted when its
  lower end is above -0.05, because a gain of 0.05 points is then ruled out. Inconclusive otherwise.
  The threshold of 0.05 points (about 1% of the best guess's score of 5.26) is set by the cost of a
  custom objective that no other training job has, and of six times the training rows. The coverage
  of the 0.01, 0.05, 0.95, and 0.99 quantiles at 12 to 36 hours is reported next to the verdict and
  decides nothing. The pilot recommends a run on all 51 members when C1 is supported or a small gain
  and, in addition, arm c1's coverage at 0.95 and at 0.05 at 12 to 36 hours is within 0.01 of
  nominal.
benchmark: >-
  Arm p3 (the verdict), arm p2 (pinball, built-in optimiser, same rows), arm p1 (pinball, control
  member only, which is BG2's recipe at the cheaper schedule), the stored weekly-retrained best
  guess for Primary substations (`p_s4_weekly` of experiment 87, a reference whose schedule
  differs), `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss) in points of capacity, over 11 levels
  (0.01, 0.05 to 0.95, 0.99). Reported and deciding nothing: mean absolute error of the median,
  coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, and the threshold-weighted CRPS.
code: ml-experiments/00099-crps1-crps-type-loss-over-ens-members-primary-demand-pilot/
---

# Experiment 99: does training on the CRPS of the pooled ensemble forecast beat the pinball loss? (CRPS1 pilot)

**Status: planned. No fit has run on the validation year.**

We ask whether a substation forecast improves when its model learns from the ensemble's members
jointly, instead of treating each member's row as a separate forecasting problem. The current
recipe trains on the control member only, then predicts for all 51 members and pools their
predicted quantiles. The pooled forecast is therefore trained on a different object than the one it
is scored as. This experiment is a pilot on the 16 Primary substations: it trains on 6 members per
run, with a loss that scores the pooled forecast, and compares it with the pinball loss on the same
rows.

## Design

- **Rows and members:** each daily run contributes the control member and 5 members drawn from
  members 1 to 50 with a seed fixed by the run's date. The same members are used by every arm and
  every refit. Only every third daily run contributes training rows, to keep the CRPS gradient
  affordable; the arms share this subsample, so a contrast between them does not depend on it.
  Prediction uses all 51 members of every run.
- **Schedule:** monthly refits from 2025-07-01 on a rolling 12-month window, 150 rounds, learning
  rate 0.15, recency half-life 0.5 years (the P-roll recipe of experiment 85). Every arm starts each
  quantile level at that quantile of the window's labels, so the arms share their starting values.
  A cheap schedule is enough for a contrast between losses; the absolute scores are not the weekly
  best guess's and are not quoted as such.
- **Loss of arm c1:** each row predicts 11 atoms, one per quantile level. The pooled forecast of a
  group of 6 rows is the mixture of their 66 atoms. The loss is the CRPS of that mixture. The atom
  masses are 0.02, 0.06, 0.04, 0.26, 0.04, 0.16, 0.04, 0.26, 0.04, 0.06, and 0.02, which are the
  masses for which the CRPS-optimal atom of each level is the quantile of that level (the quantile
  score of the existing recipe uses different masses). Six members under-represent the spread of the
  51 pooled at prediction time, so pairs of atoms from different members are weighted by
  `(50/51) × 6/5` and pairs from the same member by `6/51` of their plug-in weight. The gradient, with atoms in any
  order, is checked against a finite difference of a brute-force loss in `e00099_objective.py`.
- **Arms:** p1 pinball on the control member only; p2 pinball on 6 members; p3 pinball on 6 members
  as a custom objective with a constant Hessian; c1 CRPS on 6 members with the same constant Hessian.
  Arm p3 exists so that C1 changes the loss and nothing else.
- **Hessian:** arms p3 and c1 share one constant Hessian, chosen on four Primary substations and a
  stretch of the training period (runs from 2025-01-02 to 2025-03-31, never the validation year).
  The choice and its scores are in the tuning section below. The built-in optimiser of arms p1 and
  p2 differs from the custom one (leaf refresh, Hessian scale, regularisation), so C3 is expected
  to be non-zero and only p3 is the like-for-like control for c1.
- **Level weights:** each level's gradient is divided by twice its atom mass, so every level takes
  pinball-sized steps. Arm c1 is therefore a preconditioned CRPS gradient, and its fixed point is
  the same.
- **Pooling masses:** arm c1 pools the 51 members with the CRPS masses and arm p3 with the masses
  of experiment 87. The set `p3m` is arm p3's model pooled with the CRPS masses, so that C5
  separates the loss from the pooling.
- **Budget:** at most 2 GPU hours of wall-clock time, 6 worker processes of one thread each (the
  slot is 12 cores and up to 24 GB of GPU memory).

## Planned contrasts

All contrasts are quantile-score differences in points of capacity, over the 292 runs, whole origin
months resampled, 20,000 resamples. C1 decides the verdict. The others are read at 95% and decide
nothing.

- **C1 (verdict):** c1 minus p3. The effect of the loss, with rows, optimiser, and schedule fixed.
- **C2:** p2 minus p1. The effect of training on 6 members instead of the control member, with the
  pinball loss.
- **C5:** c1 minus p3m. The effect of the loss with the pooling masses also fixed.
- **C3:** p3 minus p2. The effect of the custom optimiser against the built-in one, a check that
  the pinball arms are comparable.
- **C4:** c1 minus p1. The whole change from the current recipe, at the cheap schedule.
- **Reference:** c1 and p1 against the stored weekly best guess, descriptive only, because its
  schedule is weekly with 500 rounds.

## Not scored

The curtailment-cost metric (`studies.node_curtailment`) scores bulk and grid supply points with
generators behind them, and this pilot trains Primary substations only. The write-up says so and
reports tail calibration and the threshold-weighted CRPS instead.

## Past-weather caveat

The model's weather inputs are the ensemble's own forecast, so no past-weather product enters at
lag times. The lag columns are the substation's own metered power, no younger than 3.5 hours at the
forecast origin.

## Tuning the constant Hessian (training period only)

Each custom arm was fitted at a refit on 2025-01-01 at four Primary substations (D2 to D5) and scored
on the 6-member rows of runs from 2025-01-02 to 2025-03-31 that the fit had not seen. The table gives
the mean quantile score (points of capacity, pooled over 6 members) and the coverage of the 0.05 and
0.95 quantiles. One Hessian is shared by arms p3 and c1: 10, the lowest score for both.

| Hessian per row and level | p3 score | c1 score | p3 q05 / q95 | c1 q05 / q95 |
|---|---|---|---|---|
| 5 | 3.818 | 3.797 | 0.099 / 0.918 | 0.150 / 0.897 |
| 10 | 3.767 | 3.748 | 0.069 / 0.930 | 0.089 / 0.917 |
| 20 | 3.815 | 3.782 | 0.033 / 0.944 | 0.038 / 0.939 |
| 40 | 4.085 | 4.050 | 0.014 / 0.952 | 0.014 / 0.950 |

The built-in pinball arm p2 scored 3.640 on the same rows. These numbers are not results of the
experiment: they use 6-member pooling and one refit.
