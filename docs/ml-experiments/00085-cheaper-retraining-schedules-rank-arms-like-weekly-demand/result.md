---
id: "00085"
slug: cheaper-retraining-schedules-rank-arms-like-weekly-demand
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, a cheaper retraining schedule ranks two pairs of
  arms the way full weekly retraining ranks them: the cadence pair (monthly minus weekly
  retraining) and the starting-value pair (a zero starting value minus XGBoost's estimated
  starting value, the contrast of experiment 80). The cheaper schedules are (a) a monthly refit on
  a rolling 12-month window with 150 rounds and `eta` 0.15 (P-roll), (b) the same refit every
  quarter, (c) a monthly refit at the full setting (whole history, 500 rounds, `eta` 0.05), and
  (d) a weekly refit on the whole history with 150 rounds and `eta` 0.15. A schedule that passes
  can replace weekly retraining in research experiments.
prediction: >-
  Full weekly retraining gives a cadence gap (monthly minus weekly) of about +0.13 points of
  capacity (monthly 5.57 against weekly 5.44 in experiment 48) and a starting-value gap (zero
  start minus estimated start) of about -0.07 (experiment 78 and the pilot of experiment 80). The
  starting-value gap comes from the first rounds of a fit, not from the cadence, so P-roll
  monthly, P-roll quarterly, and the full monthly refit reproduce its sign and lie within 0.05 of
  it. P-roll quarterly scores 0.10 to 0.20 above P-roll monthly. The weekly refit with 150 rounds
  and `eta` 0.15 scores within 0.05 of full weekly retraining, and passes both pairs. The cadence
  gap under P-roll is within 0.05 of the full-setting gap.
decision_rule: >-
  Every contrast is computed on one run set: the daily-gap runs that experiment 80's stored
  zero-start weekly arm serves (28 weekly anchors: 2025-07-01, 07-08, 07-15, then every 14 days to
  2026-06-23). A run is in the set when its latest weekly refit, at or before its start, is one
  of those 28 anchors. Every weekly arm of this experiment is fitted at exactly those anchors, and
  the stored
  arms (`w`, `m`, `pm`) are restricted to the same runs. Two reference contrasts come from stored
  predictions: R1 is full monthly minus full weekly (experiment 48's `n1h` monthly and weekly
  regimes), and R2 is the zero start minus the estimated start at full weekly retraining
  (experiment 80's `n1h_m0` minus experiment 48's weekly). The schedule contrasts are K1(S), the
  schedule's monthly arm minus its weekly arm, and K2(S), the schedule's zero-start arm minus its
  estimated-start arm. K1 is scored for P-roll (`pm` minus `pw`) and for the whole-history
  150-round recipe (`fm` minus `fw`). K1 of quarterly (`pq` minus `pw`) is descriptive, because no
  full-quarterly reference exists on these rows. K2 is scored for P-roll monthly, quarterly, and
  weekly, the full monthly refit (`m0` minus `m`), and the whole-history 150-round weekly refit.
  The quantile score over all leads (spread factor 1.15, no correction) is read per contrast with a
  95% interval from 20,000 resamples of whole origin months, paired on series, run, and valid
  time. GPU fits are not bit-reproducible, so the noise floor N is the largest of 0.01 points, the
  absolute difference in mean score between `pm` and its repeat `pm_rep`, the same for `pm0` and
  `pm0_rep`, and the absolute difference between the two K2(P-roll monthly) values; the page also
  prints how far `pm_rep` is from the stored `pm`. The tolerance is T = max(0.05, N). The
  rank-agreement rule, fixed here: a schedule agrees on a contrast when (i) the reference
  interval excludes zero and the reference size exceeds N, and the schedule's estimate has the
  same sign, its own interval excludes zero, and it lies within T of the reference estimate; or
  (ii) otherwise, the schedule's estimate lies within T of zero. A schedule passes when it agrees
  on every contrast it is scored on, and the page states how many that is. The whole-history
  weekly refit's absolute score against full weekly is reported and does not affect the pass.
  Cost is the mean wall-clock fit seconds of one refit under the schedule's own recipe, measured
  in this experiment's own runs (full weekly uses `m0`'s mean, the same full setting), times the
  schedule's refits in a year (52 weekly, 12 monthly, 4 quarterly) times 16 series, in compute
  minutes. The recommendation: a schedule that passes and costs under a third of full weekly
  becomes the R&D default for the contrasts it passed on; a schedule that fails a contrast is not
  used for that kind of contrast. The threshold is zero in score because the experiment ranks
  schedules and ports nothing to production; the pass rule above is the whole decision.
benchmark: >-
  Experiment 48's stored full weekly, full monthly, and P-roll monthly (`roll12`) predictions and
  experiment 80's stored zero-start weekly predictions, plus `manual_heuristic` and `climatology`
  as stored.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of capacity
  (1 point is 1% of the series' effective capacity); MAE of the median, coverage of the 0.01 to
  0.99 quantile band, and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles are reported
code: ml-experiments/00085-cheaper-retraining-schedules-rank-arms-like-weekly-demand/
---

# Experiment 85: do cheaper retraining schedules rank arms the way weekly retraining does? (SPD1)

**This page is the plan, committed before any code runs.** The take-home message, the verdict, and
the figures are written after the run.

## Question

Weekly retraining costs 52 refits for each series. The maintainer asks whether research
experiments can use a cheaper schedule, and keep weekly retraining for production and best-guess
experiments. This experiment checks whether four cheaper schedules give the same answer as full
weekly retraining on two contrasts for which stored weekly predictions exist.

## Size

- **What gets stored:** predictions under `data/ml-experiments/` only.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, in how to decide that a cheaper schedule "agrees"; the
  rule is fixed above before any fit.
- **Callers:** the code copies experiment 80's fitting and scoring modules, which copy experiment
  48's; nothing outside `ml-experiments/` calls it.

That is **medium**, close to simple: no new data and no new method. It gets a Sonnet plan review,
a Sonnet code review, a Sonnet check of the write-up against the printed report, and a Sonnet
prose review. No Opus review, because no verdict can change production.

## Arms

All arms are the N1 model (one XGBoost model per series, trained on the ENS control member) for
the 16 Primary substations, seed 0, forecasting the daily runs of `fold_a_daily_gap_v1`.

- **Reference (stored, not refitted):** `w` (full weekly), `m` (full monthly), `pm`
  (P-roll monthly), `w0` (full weekly, zero start; only its 28 anchors exist, which fixes the
  run set above).
- **New fits:** `pq` and `pq0` (P-roll quarterly, estimated and zero start), `pm0` (P-roll
  monthly, zero start), `pw` and `pw0` (P-roll weekly at the 28 anchors), `fw` and `fw0`
  (schedule (d): weekly at the 28 anchors, whole history, 150 rounds, `eta` 0.15), `fm` (the
  same recipe, monthly), `m0` (schedule (c): full monthly, zero start), and `pm_rep` and
  `pm0_rep` (repeats of `pm` and `pm0` for the noise floor).
- **Contrasts per schedule:** as in the decision rule. Quarterly is judged on K2 only.

## Not covered, and why

The brief also names two contrasts from earlier experiments: `am` against `d0` (experiment 64) and
the five-model blend against its best single member (experiment 49). Experiment 64 fitted each arm
once, at 2025-06-30, so no stored weekly reference exists and the five series use their own
capacity-table machinery; refitting them under four schedules is a larger experiment. Experiment 49's
blend picks weights from five members of mixed schedules. Both are left to the manager to
schedule if wanted.

## Advice from earlier experiments

Followed: screen on P-roll (experiment 34), report absolute scores, run fits in parallel inside
the slot, score zero start with its estimated-start twin (experiment 78). Departures: none.

## Lessons for the next experiment

Written after the run.
