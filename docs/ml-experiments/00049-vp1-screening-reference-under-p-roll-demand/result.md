---
id: "00049"
slug: vp1-screening-reference-under-p-roll-demand
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, once the XGBoost quantile model N1 is refitted every
  month without the 14-day error correction, (a) a spread factor chosen on earlier months only
  lowers the quantile score below the fixed factor 1.15, (b) an average of the quantiles of
  retrained and stored XGBoost models, with one weight vector chosen per month on earlier months,
  still lowers it below the best single member, and (c) each adopted input (the Christmas column,
  the COVID indicator, a half-life of 0.5 years) changes the score when dropped, and the substation
  fingerprints of experiment 32 change it when added to a pooled XGBoost model, all under the cheap
  rolling-window refit P-roll.
prediction: >-
  (a) The re-tuned factor moves by at most 0.10 from 1.15 and changes the quantile score by under
  0.03 points of capacity. (b) The blend minus its best member is -0.03 to -0.10, smaller than
  experiment 35's -0.097, because retrained members differ less from each other than a frozen and
  a retrained member do. (c) Each drop-one arm moves the score by under 0.05 points, except the
  Christmas column, which stays worth keeping; the fingerprints move it by under 0.05.
decision_rule: >-
  Six contrasts on the quantile score over all leads (twice the mass-weighted mean pinball loss over
  9 levels, in points of capacity), all with the spread factor, none with the correction, on seed 0,
  with whole validation months resampled. The family of six (Bonferroni, 99.17%) sets the level
  of every interval below. D0 decides the verdict on the spread factor. It is the rolling-origin
  factor minus the fixed 1.15 for the monthly-refitted N1, with threshold zero because the factor
  is one constant per retrain. B1 decides the verdict on the blend. It is the blend with one
  weight vector per month, minus the best single member (chosen with hindsight, so stricter,
  never looser). A blend is worth serving only when the estimate is at or below -0.05 points and
  the interval is wholly below zero, because serving it adds a second and third XGBoost model
  (experiment 35's threshold for Primary demand). F1 decides the verdict on the fingerprints. It is
  the pooled-by-type P-roll model with the series identifier and both fingerprint blocks, minus the
  same pooled model without them. Recent-window statistics need a rolling recomputation at every
  forecast origin, so F1 is worth porting only when its interval lies wholly below -0.10 points
  (experiment 32's threshold). C1, C2, and C3 decide no verdict. They are P-roll without the
  Christmas column, P-roll without the COVID indicator, and P-roll with a half-life of 2 years in
  place of 0.5 years, each minus the P-roll refit C0. A component stays in the screening reference
  when its interval lies wholly above zero. A component can be removed when its interval lies
  wholly inside plus or minus 0.03 points, because keeping it adds a column or a weight rule to
  maintain. Otherwise the component is unresolved; the margin may be unresolvable at 99.17%, and
  the page then says so. Verdict: supported when D0, B1, or F1 meets its porting rule; small gain
  when B1 lies wholly below zero but misses its threshold; inconclusive otherwise. Absolute scores
  are reported for every arm with a 95% interval. P-roll (5.15 in experiment 34) is the screening
  reference row that PVDIS-5, P36, and AN1 take. Stage 1 starts only once the manager confirms the
  cadence that RT2 leaves in force.
benchmark: >-
  P-roll on 12 months with the half-life of 0.5 years (experiment 34's `roll12`, quantile score 5.15
  with the spread factor), full monthly retraining (5.12), weekly retraining (5.04), the stack
  (5.32), `manual_heuristic` (5.89), and `climatology` (7.46).
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity;
  MAE of the median and the exceedance rate of the 0.95 quantile are reported and decide nothing
production_status: not-for-production
adopted_in: none
code: ml-experiments/00049-vp1-screening-reference-under-p-roll-demand/
---

# Experiment 49: which Primary demand components still improve the score under monthly retraining?

**We ask which parts of the Primary substation demand forecast still improve the score once the
XGBoost model is refitted every month.** Experiment 34 found that refitting monthly lowered the
forecast error, that the 14-day correction stopped helping, and that a cheap stand-in for monthly
refitting (refit monthly on the latest 12 months of data, "P-roll") gives the same answers at a
third of the cost. Every input the forecast uses was chosen on an XGBoost model trained once. We
therefore (1) re-choose the spread factor, which widens the uncertainty band, on earlier months
only, (2) re-run the experiment 35 blend with retrained XGBoost models, and (3) drop each adopted
input in turn, and add the substation fingerprints of experiment 32 to a pooled XGBoost model, all
under P-roll. Stage 0 uses only predictions that experiments 33 and 34 stored. Stage 1 refits the
XGBoost models.

## Plan

**Sizing.** The experiment stores new prediction tables under `data/ml-experiments/`, which are
private and never committed. It does not touch the serving path or a degradation rule. More than
one design would test the blend (this plan fixes five members) and the fingerprints (this plan uses
both blocks). No code has callers that could not be named without searching, because the scripts
are copies of those of experiments 34 and 35. The size is medium. The experiment gets a Sonnet plan
review, a Sonnet code review, a Sonnet write-up review, and one Opus review of the result, as the
manager's brief sets.

**Earlier advice followed.** Screen on P-roll, not a frozen XGBoost model, and leave the 14-day
correction out of every arm (experiment 34). Report absolute scores. The plan departs from
experiment 32's advice not to add fingerprints to a pooled model that has the series identifier,
because experiment 34 left that contrast (R4) untested under retraining.

**Rows and files.** Every arm is restricted to seed 0 and to the 768,073 Primary cases in
experiment 33's `metrics/Primary/shared_keys.parquet` that every arm holds, with valid time before
2026-07-01. Stage 0 reads experiment 34's `work/arms/` files `monthly`, `weekly`, and `roll12`
(spread factor 1.00, no correction), their `_s` files (factor 1.15), `a7_monthly_s`,
`type_id_monthly_s`, `type_id_roll12_s`, and from `work/reference/` `n1_cs` (the stack),
`manual_heuristic`, and `climatology`. The fixed factor 1.15 was chosen in experiment 19 on
validation months that overlap these, so D0 is not fully out of sample.

**Stage 0 (CPU, stored predictions).**

- **D0.** For the monthly, weekly, and P-roll arms, apply factors 1.000 to 1.300 in steps of 0.025
  about the median. The rolling-origin factor for runs started in month m is the factor that
  minimises the all-lead score on cases whose run started before month m and whose valid time is
  before month m's first day (experiment 35's rule). July 2025 has no history and uses 1.15. D0 is
  the monthly arm; weekly and P-roll are descriptive. The report gives the score curve over the
  factors and the in-sample best factor.
- **B1.** The members are `monthly_s`, `weekly_s`, `a7_monthly_s` (weekly analogues, monthly
  refits), `type_id_monthly_s` (pooled by type), and the stack `n1_cs`. One weight vector per
  month, on a grid of quarters, is chosen on earlier months by the same rule. July 2025 uses equal
  weights. Experiment 35's corner-cell stop rule is not applied. Descriptive arms: weights per lead
  band, equal weights, and experiment 35's three members with the retrained N1 (`monthly_s`,
  `a7_monthly_s`, `n1_cs`).

**Stage 1 (GPU, seed 0, `eta` 0.15, 150 rounds, 12-month window, spread factor 1.15).**

- **C0 and C0b (references).** Refit P-roll (C0) and the pooled-by-type P-roll model with the
  series identifier (C0b) with the same code as the other arms. Every stage 1 contrast uses these
  refits as its reference, so that GPU non-determinism cannot pass for a component effect. Each
  refit's score is also compared with the stored `roll12_s` and `type_id_roll12_s`; a difference
  above 0.03 points is reported as non-reproducible.
- **C1.** P-roll without `days_to_christmas`. **C2.** P-roll without `covid_indicator`. **C3.**
  P-roll with a half-life of 2 years.
- **F1.** C0b plus 7 training-window statistics and 3 recent-window statistics, defined as in
  experiment 32 except for the window. The training-window statistics of a refit use actuals in
  the refit's 12-month window, strictly before the refit's cutoff (3.5 hours before the refit
  time). The recent-window statistics of a case use the 14 days of actuals ending 2 days before the
  case's run start, as experiment 32 does, so they respect the 3.5-hour publication delay. Training
  rows apply the same rule at each training run. No statistic is weighted by the half-life.
- GPU: about 5 arms at 0.2 hours each plus the pooled arms at about 0.3 to 0.5 each; at most 8 GB;
  2 cores, two workers of one thread each.

**Interval method.** Whole validation months are resampled with
`studies.bootstrap.bootstrap_difference_at_level`. Seeds add nothing, because the arms use seed 0.
Experiment 34 found that months, not seeds, set the interval width, which is about plus or minus
0.1 to 0.3 points for a structural change and under 0.05 for a column drop.

**Where the evidence is thin.** P-roll passed a rank test of three contrasts, and this experiment
uses it for four more. One validation year with 12 months sets every interval. The blend rests on
one experiment's members.

## Results

Not yet run.
