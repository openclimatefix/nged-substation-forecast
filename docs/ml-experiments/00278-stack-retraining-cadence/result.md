---
id: '00278'
slug: stack-retraining-cadence
title: >-
  How much forecast skill does the best-guess demand stack lose when its convex model, its neural
  network head, and its solar factors are retrained less often?
theme: production-reference-and-best-guess
also_relevant_to:
- retraining-and-training-history
status: done
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
verdict: not supported
best_arm: >-
  Head trained once (the current stack), 2.578 points against 2.587 for the head retrained
  quarterly or monthly, on six substations over 2025-07 to 2026-06 (once minus quarterly -0.008,
  99% interval -0.017 to 0.000)
headline: >-
  Retraining the network head quarterly or monthly does not beat training it once (once minus
  quarterly -0.008 points; the plan predicted +0.03 or more). Quarterly instead of monthly refits
  of the convex model scored 0.010 points worse overall (99% interval -0.014 to +0.032) and 0.022
  worse in winter (95% interval +0.014 to +0.027). Quarterly solar factors are level with monthly ones
  (-0.0001).
lessons:
  - >-
    The stack's convex model already refits monthly, not weekly. Weekly refits add nothing (+0.001,
    99% interval -0.018 to +0.019), and a monthly refit costs 0.4 s per substation, so refitting
    monthly instead of quarterly saves almost no time.
  - >-
    The head's skill does not fade over 12 months: the head trained once is level with or ahead of
    heads retrained every quarter or month, and the same pipeline's head trained at 2025-07 and
    never retrained is also level with the quarterly head (-0.004, 95% interval -0.013 to +0.003).
  - >-
    At every head cadence we tried, the stack on the monthly convex model scored 2.578 to 2.588,
    below BG2 refitted weekly (2.715) or monthly (2.752); we report no paired interval for this.
production_status: not-for-production
adopted_in: none
date: 2026-10-23
take_home_message: >-
  Retraining the small neural network that corrects the demand forecast's median, every quarter or
  every month, scored no better than training it once before the scored year, so it needs no
  retraining schedule over a year. The convex demand model should keep its monthly refits, which
  take 0.4 s per substation: refitting quarterly instead scored 0.010 points worse overall (not shown
  to be acceptable) and weekly refits gained nothing. Refitting the two solar factors quarterly
  instead of monthly changed nothing.
advice:
  - >-
    Keep the convex model's monthly refits, because a refit costs 0.4 s per substation and quarterly
    refits were not shown adequate (C1, figure 1; post hoc winter +0.022).
  - >-
    Do not refit the convex model weekly for the stack: weekly minus monthly was +0.001 points (C2).
  - >-
    Train the network head once on all history and leave it for a year in the first release; a
    retrained head scored no better (C3, figure 1), and retest only if the demand pattern changes.
  - >-
    Refit the two solar factors quarterly if a monthly job is inconvenient: the difference was
    -0.0001 points on daylight rows (C5).
  - >-
    Average three seeds when a head is retrained, because single seeds of one head scored 2.589 to
    2.609 against 2.582 for their average (checks bullet below).
---

**Retraining the small neural network that corrects the median (the head) more often did not help,
and the convex demand model's monthly refit is worth keeping.** We asked how much forecast skill
the best-guess demand stack loses when its three retrained parts are retrained less often. The
stack is the combined demand forecast of [experiment
251](../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md), built from three
parts. The first is the convex model, a generalised additive model (GAM) whose coefficients a
Kalman filter updates (called arm D in experiment 155). The second is the head ([experiment
201](../00201-dp-gam-residual-mlp/result.md)). The third is the solar factors ([experiment
202](../00202-dp-ens-two-number-rolling-replay/result.md)): two numbers per half-hour, the mean and
the standard deviation of the 51 weather-ensemble members' solar responses, each rescaled by a
fitted factor. We refitted each part on a different schedule. We scored the 561,363 half-hourly
rows of 2025-07 to 2026-06 in total across six anonymised substations. The quantile score is twice
the mean pinball loss (the standard error of a quantile forecast) over 11 quantiles, in percentage
points of the substation's effective capacity (its rated demand); lower is better. BG2 is the
XGBoost best-guess demand model of [experiment
87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md).

![Figure 1. The planned contrasts against their thresholds](figure_1_planned_contrasts.svg)

![Figure 2. Each contrast by month](figure_2_by_month.svg)

![Figure 3. Absolute scores against BG2](figure_3_absolute.svg)

![Figure 4. Skill by months since the last refit](figure_4_months_since_refit.svg)

**Verdict `not supported`.** The plan predicted two things. P1: the convex model tolerates quarterly
instead of monthly refits, losing under 0.02 points. P2: the head trained once scores at least 0.03
points worse than the quarterly head. P2 failed, because the head trained once scored 0.008 points
better than the quarterly head. P1's point estimate held (+0.010), but its 99% interval reaches
+0.032, so the decision rule calls P1 inconclusive. P3 held (the monthly head minus the quarterly
head +0.001), and P4 held (quarterly minus monthly solar factors -0.0001). The stop rule (stop
before the head arms if monthly and quarterly refits differ by under 0.01 at 5 of 6 substations)
did not fire: they differ by under 0.01 at 2 of 6. We measured the convex model's cadence on its
own forecasts. We then added the head and the solar factors to the monthly-refit forecasts only,
and did not rebuild the whole stack on quarterly refits. The contrasts C1 to C5 below are the five
planned comparisons: C1 and C2 for the convex model, C3 and C4 for the head, C5 for the solar
factors.

## Results

- **Schedule and what was refitted.** Convex model: experiment 136's stored fits on all substations'
  history with a 24-month window, weekly, monthly (the stack's own schedule), and quarterly (the
  monthly fits cut in January, April, July, and October); the Kalman filter runs throughout. Head:
  the experiment 201 variant that sees calendar flags, three random seeds, retrained at each month
  start from 2025-07 (12 retrains, 36 fits) with every training target dated before the retrain
  date. Solar factors: experiment 202's replay, refitted at the month or the quarter start.
  Retraining the head took 2.7 s per fit on the GPU, and refitting the convex model took 0.4 s per
  substation, so no schedule saves meaningful compute. BG2's weekly and monthly refits are
  experiment 87's.
- **Absolute scores, six substations:** the table gives the quantile score, the mean absolute error
  of the median, and the tail-weighted CRPS at the 95th percentile of earlier power.

  | Forecast | Quantile score | Median error | Tail CRPS |
  |---|---|---|---|
  | BG2 refitted weekly | 2.715 | 4.766 | 0.202 |
  | BG2 refitted monthly | 2.752 | 4.805 | 0.204 |
  | Convex model, monthly refits | 2.691 | 4.623 | 0.191 |
  | Convex model, weekly refits | 2.692 | 4.613 | 0.193 |
  | Convex model, quarterly refits | 2.702 | 4.636 | 0.192 |
  | Stack, head trained once | 2.578 | 4.510 | 0.171 |
  | Stack, head retrained quarterly | 2.587 | 4.526 | 0.170 |
  | Stack, head retrained monthly | 2.588 | 4.528 | 0.170 |

  The stack scores about 0.13 below BG2 refitted weekly at every head schedule.
- **C1, convex model quarterly minus monthly:** +0.010 (99% interval -0.014 to +0.032). Post hoc, at
  the 95% level and excluding October: winter (November to March) +0.022 (+0.014 to +0.027),
  summer (April to September) +0.002 (-0.028 to +0.033). By substation: D10 +0.044, D5 +0.016,
  D16 +0.013, D11 +0.006, D14 -0.000, D9 -0.016.
- **C2, weekly minus monthly:** +0.001 (99% -0.018 to +0.019); post hoc, 95% level, October
  excluded: winter +0.020 (+0.007 to +0.034), summer -0.014 (-0.031 to +0.006).
- **C3, head trained once minus quarterly retrained (stack):** -0.008 over all rows (99% -0.017 to
  0.000); April to June -0.011 (week-block 99% -0.027 to +0.005, because three months are too few to
  resample). Once minus monthly retrained is -0.009.
- **C4, monthly minus quarterly head:** +0.001 (99% -0.006 to +0.008).
- **C5, solar factors quarterly minus monthly, daylight rows:** -0.0001 (99% -0.0008 to +0.0003)
  for the solar summary alone, and -0.0001 (99% -0.0008 to +0.0004) with the head added.
- **Tails (coverage and pinball loss; stack with head trained once against BG2 refitted weekly):**
  q01 coverage 0.013 and loss 0.172 against 0.010 and 0.211; q99 coverage 0.984 and loss 0.211
  against 0.983 and 0.253. Tails move by under 0.002 across head
  cadences.
- **Post hoc:** the head fitted at 2025-07 with this experiment's pipeline and never retrained
  ("own once") scores 2.582, 0.004 below the quarterly head (95% -0.013 to +0.003) and 0.004 above
  experiment 201's stored head (-0.004 to +0.012).
- **What to try next:** a retest of the head's cadence once a second year of ensemble weather
  exists, because one year cannot show slow drift; and a stack rebuilt end to end on quarterly
  convex-model refits, which would test the interaction with the head.

## Checks

- **Median minus mean forecast, per substation:** exactly 0.000 at all six substations at every
  convex-model cadence, because arm D's median is its mean forecast by construction.
- **Mean-forecast error beside each contrast:** the median's absolute error moves with the quantile
  score in C1 (+0.013 against +0.010), C2 (-0.009 against +0.001, within noise), and C3 (once
  4.510 against quarterly 4.526), so trigger 3 did not fire.
- **Triggers (the seven audit triggers of the experiment guide):** trigger 1, a failed control, did
  not fire (the gates reproduced experiment 155's arm D and experiment 202's solar arm exactly).
  Trigger 2, a result contradicting its prediction, fired for P2 and was resolved here: the head
  trained once scored 0.004 better than the same pipeline's head trained at 2025-07 and never
  retrained, a gap whose 95% interval (-0.004 to +0.012) includes zero, so retraining adds nothing
  measurable. Trigger 4, a sign flip when one site is left out, did not fire: leaving one substation
  out leaves C1 between +0.003 and +0.016 and C3 between -0.007 and -0.010. Triggers 5 and 6 (a
  nested arm beating its parent, and interval widths) did not fire. Trigger 7, later work depending
  on an unreplicated result, is open: later work may rely on the one-year result.
- **Departures from the plan:** the last-quarter C3 interval uses week blocks as well as month
  blocks, because three months are too few to resample. The cadence contrast is read at the arm D
  level, as planned; the stack approximation was not run separately.
