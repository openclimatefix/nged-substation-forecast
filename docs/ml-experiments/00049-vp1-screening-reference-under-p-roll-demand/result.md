---
id: "00049"
slug: vp1-screening-reference-under-p-roll-demand
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, once the quantile model N1 is retrained every month
  (without the 14-day error correction), (a) a spread factor chosen on earlier months only lowers
  the quantile score below the fixed factor 1.15, (b) a lead-band blend of the retrained models
  still lowers it below the best single retrained model, and (c) each adopted input (the Christmas
  column, the COVID indicator, a half-life of 0.5 years) and the substation fingerprints of
  experiment 32 in the pool change the score measurably when they are dropped or added under the
  cheap rolling-window refit P-roll.
prediction: >-
  (a) The re-tuned factor moves by at most 0.10 from 1.15 and changes the quantile score by under
  0.03 points of capacity. (b) The blend minus its best member is -0.03 to -0.10, smaller than
  experiment 35's -0.097, because retrained members differ less from each other than a frozen and
  a retrained member do. (c) Each drop-one arm moves the score by under 0.05 points, except the
  Christmas column, which stays worth keeping; the fingerprints in the pool move it by under 0.05.
decision_rule: >-
  Six decisive contrasts on the quantile score over all leads (twice the mass-weighted mean
  pinball loss over nine levels, in points of capacity), read at 99.17% (Bonferroni across six),
  whole validation months resampled, seed 0. D0 is the rolling-origin spread factor minus the fixed
  1.15 on the monthly-retrained N1 without the correction; the threshold is zero, because the
  factor is one constant per retrain and adds no pipeline step. B1 is the rolling-origin one-vector
  blend of five retrained or stored members minus the best single member among them (chosen with
  hindsight, so stricter, never looser); a blend is worth serving only when the estimate is at or
  below -0.05 points and the interval wholly below zero, because serving it adds a second and third
  model (the threshold of experiment 35 for Primary demand). C1 to C3 are the drop-one arms of P-roll
  minus P-roll: no Christmas column (C1), no COVID indicator (C2), and a half-life of 2 years in
  place of 0.5 (C3). A component earns its place when its drop-one contrast lies wholly above zero.
  It can be removed when the 99.17% interval lies wholly inside plus or minus 0.03 points (an
  equivalence margin of about 0.6% of the score), because keeping it adds a column or a weight
  rule to maintain. Otherwise the component is unresolved. F1 is the pooled-by-type P-roll model
  with the series identifier and both fingerprint blocks of experiment 32 (training-window and
  recent-window statistics) minus the same model without them. Recent-window statistics need a
  rolling recomputation at every forecast origin, so F1 is worth porting only when its interval
  lies wholly below -0.10 points (the threshold of experiment 32). Verdict: supported when D0, B1,
  or F1 meets its porting rule; small gain when D0 or B1 lies wholly below zero but misses its
  threshold; refuted when D0, B1, and F1 all lie wholly at or above zero; inconclusive otherwise. C1
  to C3 decide no verdict; they say which components stay in the screening reference. The score of
  each arm is also reported absolutely, with its own interval, and the screening reference row
  (P-roll, 5.15 in experiment 34) is the output that PVDIS-5, P36, and AN1 take. Stage 1 starts only
  once the manager confirms the cadence that RT2 leaves in force.
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

# Experiment 49: which Primary demand components still earn their place under monthly retraining?

**We ask which parts of the Primary substation demand forecast still pay for their upkeep once the
model is refitted every month.** Experiment 34 found that refitting monthly lowered the forecast
error, that the 14-day correction stopped helping, and that a cheap stand-in for monthly refitting
(refit monthly on the latest 12 months of data, "P-roll") gives the same answers at a third of the
cost. Every input the forecast uses was chosen on a model trained once. We therefore (1) re-choose
the spread factor, which widens the uncertainty band, on earlier months only, (2) re-run the
experiment 35 blend with retrained models, and (3) drop each adopted input in turn, and add the
substation fingerprints of experiment 32 to a shared model, all under P-roll. Stage 0 uses only
predictions already stored by experiments 34 and 35. Stage 1 refits the models.

## Plan

**Sizing.** Stored: new prediction tables under `data/ml-experiments/`, which are private and never
committed. Serving path: not touched. Degradation rule: not touched. More than one defensible
design: yes, for the blend's members (this plan fixes five) and for the fingerprints (this plan uses
both blocks). Code whose callers could not be named without searching: no; the scripts are copies of
experiment 34's and 35's. Size: medium. This plan gets a Sonnet plan review, a Sonnet code review, a
Sonnet write-up review, and one Opus review of the result, as the manager's brief sets.

**Earlier advice followed.** Screen on P-roll, not a frozen model (experiment 34); leave the
14-day correction out of every arm (experiment 34); apply no correction before blending (experiment
35); report absolute scores. The plan departs from experiment 32's advice not to add fingerprints
to a model that has the identifier only because experiment 34 left that contrast (R4) untested
under retraining.

**Stage 0 (CPU, stored predictions).**

- **D0.** For each of the monthly-retrained N1 (`monthly`), weekly-retrained N1 (`weekly`), and
  P-roll (`roll12`) arms, apply factors 1.000 to 1.300 in steps of 0.025 about the median. The
  rolling-origin factor for runs started in month m is the factor that minimises the all-lead score
  on cases whose run started before month m and whose valid time is before month m (the rule of
  experiment 35); July 2025 uses 1.15. D0 is the monthly arm; weekly and P-roll are descriptive.
  The report also gives the score curve over the factors and the in-sample best factor.
- **B1.** The members are `monthly`, `weekly`, `a7_monthly` (weekly analogues, monthly refits),
  `type_id_monthly` (pooled by type), and the stack `n1_cs`; all carry the spread factor 1.15, and
  `n1_cs` carries the correction as the stack does. One weight vector per month on a grid of
  quarters, chosen on earlier months as in experiment 35. Descriptive: experiment 35's three
  members with the retrained N1 (`monthly`, `a7_monthly`, `n1_cs`), the equal-weight blend, and one
  global weight vector.

**Stage 1 (GPU, P-roll, seed 0, `eta` 0.15, 150 rounds, 12-month window).**

- **C0 (reference reproduction).** Refit P-roll as stored; the score must match the stored `roll12`
  within 0.01 or the refit is reported as non-reproducible.
- **C1.** P-roll without `days_to_christmas`. **C2.** P-roll without `covid_indicator`. **C3.**
  P-roll with a half-life of 2 years.
- **F1.** Pooled-by-type P-roll (two pools, MW and MVA, 8 series each), series identifier plus
  training-window statistics (7 columns, recomputed over each refit's window) plus recent-window
  statistics (3 columns, per case), minus the same pool without the statistics (stored as
  `type_id_roll12`). The statistics are those of experiment 32.
- GPU: about 4 arms at 0.2 hours each plus the pooled arm at about 0.3 to 0.5; at most 8 GB; 2
  cores, two workers each of one thread.

**Interval method.** Whole validation months are resampled with
`studies.bootstrap.bootstrap_difference_at_level`; only seed 0 exists for most arms, so seeds add
nothing. Experiment 34 found that months, not seeds, set the interval width, which is about plus or
minus 0.1 to 0.3 points for a structural change and under 0.05 for a column drop; the equivalence
margin of 0.03 may therefore not be resolvable, and the page will say which components are
`unresolved`.

**Where the evidence is thin.** P-roll passed a rank test of three contrasts; this experiment uses
it for four more. One validation year with 12 months sets every interval. The blend rests on one
experiment's members.

## Results

Not yet run.
