---
id: '00278'
slug: stack-retraining-cadence
title: >-
  How much forecast skill does the best-guess demand stack lose when its convex model, its neural
  network head, and its solar factors are retrained less often?
theme: production-reference-and-best-guess
also_relevant_to:
- retraining-and-training-history
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00278-stack-retraining-cadence/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (which stage of the stack carries the cadence contrast, how a retrained head is trained); the plan
  pre-registers one of each. Callers: this folder's scripts, copied from experiments 136, 155, 201,
  and 202, which call reviewed `studies.*` modules without changing them. Refits the convex model at
  a new cadence and retrains the neural network head; no download.
hypothesis: >-
  The best-guess stack (arm D, the neural network head, and the two-number solar summary) needs its
  neural network head retrained at least quarterly, but its convex model (a generalised additive
  model with a Kalman filter on its coefficients) tolerates quarterly instead of monthly refits,
  because the filter absorbs the drift between refits.
prediction: >-
  Written before any score is read; each can fail. (P1) Arm D with quarterly refits scores under
  0.02 points worse than arm D with monthly refits on the six-substation quantile score. (P2) The
  head trained once before 2025-07 scores at least 0.03 points worse than a quarterly retrained head
  on the rows of the last quarter of the scored year (2026-04 to 2026-06). (P3) A monthly retrained
  head scores within 0.01 points of a quarterly retrained head. (P4) Quarterly solar factors score
  within 0.01 points of monthly solar factors on daylight rows. The stack's convex model already
  refits monthly (experiment 136), not weekly, and experiment 136 found weekly minus monthly at
  +0.004, so weekly refits are expected to add nothing.
decision_rule: >-
  Contrasts are read on the six-substation quantile score with the whole-month paired bootstrap, as
  99% two-sided intervals (a Bonferroni correction for five contrasts at the 5% level). A cadence is
  adequate when the upper end of its interval is below its threshold, and needed when the lower end
  is above its threshold; an interval that is neither is inconclusive. The thresholds are cost-set:
  a cadence change adds no model, data source, or pipeline step to the stack and only trades fit
  time against skill, so the threshold is the smallest loss worth the saving. C1, quarterly minus
  monthly GAM refits: +0.02 points. C2, weekly minus monthly GAM refits: -0.02 points (weekly is
  worth porting only if it is at least 0.02 better). C3, head trained once minus head retrained
  quarterly, on the rows of 2026-04 to 2026-06: +0.03 points. C4, monthly retrained head minus
  quarterly retrained head: -0.01 points. C5, quarterly minus monthly solar factors on daylight
  rows: +0.01 points. P1 is judged on C1 and P2 on C3 (P2 holds when C3's lower end is above 0
  and its point estimate is at least 0.03). Stop rule: if monthly and quarterly GAM refits differ by
  under 0.01 points at 5 of 6 substations, stop before the head arms and report the GAM result; the
  verdict is then `partly supported` at best, and the head half is recorded as not tested. The solar
  factor arm (C5) is cheap and runs regardless. `supported` needs P1 and P2; `partly supported`
  needs exactly one of them (or P1 with the head not tested); `not supported` otherwise.
benchmark: >-
  Arm D with monthly refits (experiment 155's arm D, rebuilt here and gated against its stored
  forecasts), the stack of experiment 251, and BG2 with weekly and monthly refits (experiment 87).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  mean absolute error of the median, and tail-weighted CRPS at the 95th percentile of earlier
  power, by months since the last refit, in winter (November to March) and summer (April to
  September) separately; coverage and pinball loss at q01, q05, q95, q99.
code: ml-experiments/00278-stack-retraining-cadence/
---

**Planned.** We ask how much skill the best-guess demand stack loses when each part is retrained
less often.

## Plan

- **What the stack uses today (checked in the code).** The convex model refits monthly on a
  24-month window ([experiment
  136](../00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md), arm A2), and the Kalman
  filter runs throughout. The experiment 220 page's "weekly-refit" wording is wrong for the convex
  model: weekly refits (experiment 136's arm A3) moved the score by +0.004, so the stack uses
  monthly. The network head is trained once on rows before 2025-07 ([experiment
  201](../00201-dp-gam-residual-mlp/result.md)). The solar factors refit monthly on the trailing 365
  days ([experiment 202](../00202-dp-ens-two-number-rolling-replay/result.md)). BG2 refits weekly.
- **Rows.** The six substations (D5, D9, D10, D11, D14, D16) and the scored year of [experiment
  251](../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md): the 292 runs of
  `fold_a_daily_gap_v1` from 2025-07-02 to 2026-06-30, leads up to 168 hours, outside the
  hand-filtered load shifts.
- **Convex model arms.** Rebuild experiment 136's arm A2 (monthly) and A3 (weekly) and add A4
  (quarterly: the cuts at the start of January, April, July, and October). Each arm passes through
  experiment 155's arm D recipe (winter night offset and recipe R, with the spread table refitted on
  the arm's own tuning months). The rebuilt monthly arm D is gated against the stored forecasts of
  experiment 155 (within 0.005 points). The cadence contrast is read at the arm D level, before the
  head and the solar summary, because those two stages are trained on monthly arm D. A post hoc
  check adds the stored head correction to the quarterly arm D rows to approximate the stack; it is
  an additive approximation (the head was trained on monthly arm D) and is not a verdict input.
  Arm D's spread table is fitted on fixed tuning months (2024-07 to 2025-06), so the spread does not
  follow the refit cadence; spread drift shows only in the coverage by months since the refit.
- **Cut rule for the quarterly arm.** A quarterly refit is the monthly fit at a cut in January,
  April, July, or October (the fits are experiment 136's stored T-all fits, so no GAM is refitted).
  The first fit of each site is its first such cut. Weekly and quarterly arms are gated by
  reproducing experiment 136's stored A3 forecasts and by the monthly gate.
- **Head arms.** The head of experiment 201 (its arm 2) is retrained with three seeds at the start
  of each quarter (2025-07, 2025-10, 2026-01, 2026-04) and each month of the scored year, on all
  rows whose run origin plus 168 hours (the longest lead) precedes the retrain date, with validation months chosen as in experiment
  201. The head trained once is experiment 201's stored forecasts. Each retrain uses the GPU, one fit
  at a time.
- **Solar factors.** Experiment 202's replay is rerun with the two factors refit quarterly instead
  of monthly. The replay takes a map from each month to the month-start whose fit it uses, so the
  quarterly arm uses the quarter-start fit for all three months of the quarter.
- **Reports.** Score against months since the last refit for each part, winter and summer separately,
  and the fit time of each part. Compare absolute scores with BG2 refitted weekly (`p_s6_bg2`) and
  monthly (`p_a3_monthly_in_weekly_slot`) of experiment 87.
- **Compute.** Convex model fits on 6 CPU cores, the head on the GPU, under `systemd-run` with a
  16 GB memory cap.
- **Size, by the five triggers.** What gets stored: private files only. Serving path: not touched.
  Degradation rule: not touched. More than one defensible design: yes (two, recorded above). Code
  whose callers could not be named without searching: no, scripts copy earlier experiments.
