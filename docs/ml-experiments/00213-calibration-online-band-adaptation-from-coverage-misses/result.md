---
id: '00213'
slug: calibration-online-band-adaptation-from-coverage-misses
part: CAL-4
theme: calibration-and-uncertainty
also_relevant_to:
- curtailment-cost
- production-reference-and-best-guess
status: planned
size: >-
  simple. Stored: private files under
  data/ml-experiments/00213-calibration-online-band-adaptation-from-coverage-misses/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: the update rule, the gamma grid, the
  clamp, and the verdict rules are pre-registered below, so no choice of design is left open.
  Callers: this folder's scripts, which read experiment 169's stored arm A forecasts and call the
  reviewed `studies.*` modules without changing them. No refit of the mean forecast and no fitted
  spread model.
hypothesis: >-
  Moving a scale on each quantile offset of experiment 169's arm A up after a reading falls outside
  that quantile, and down after a reading falls inside it, in the style of adaptive conformal
  inference (Gibbs and Candes, 2021), lifts the cold-night q95 coverage to 0.90 or more without a
  second fitted spread model, because a run of misses on a cold spell widens the band before the
  spell ends.
prediction: >-
  Written before any score is read. Arm A reproduces its 0.896 cold-night q95 coverage and 2.7519
  all-row score. The adaptive arm lifts the cold-night q95 coverage by 0.00 to 0.03 in the scored
  year (to 0.90 or more with probability about 0.4), keeps the all-row q95 coverage between 0.94 and
  0.96, and changes the all-row quantile score by -0.01 to +0.01. It may fail because cold nights are
  rare and the feedback is slow: on a cold spell the scale learns after the misses, not before, so
  the first night of each spell gains nothing and the three days after it gain most.
decision_rule: >-
  The reference is arm A (experiment 169's arm A, which is experiment 164's fix O on experiment
  160's smooth spread table), read from experiment 169's stored forecasts. The planned contrasts are
  C1 (arm ACI minus arm A) in quantile score over leads 0.5 to 168 hours on all rows, and C2 (arm ACI
  minus arm A) in cold-night q95 coverage, at the five decision substations D5, D10, D11, D14, and D16
  (D9 beside them, in no contrast). Intervals are paired month-resampled bootstrap intervals with
  20,000 resamples at the 95% level (two contrasts, one of them descriptive, so no correction). Arm
  ACI passes in a period when all three hold: its cold-night q95 coverage, pooled over the five
  substations, is 0.90 or more; its all-row q95 coverage lies between 0.94 and 0.96, and its all-row
  q05 coverage is no more than 0.01 further from 0.05 than arm A's (arm A's q05 coverage is 0.065, so
  the literal rule of one point from nominal would fail arm A itself); and the upper bound of C1 is
  at or under +0.01 points. The hypothesis is `supported` when ACI passes in the scored year and in
  the replicate months, `partly supported` when it passes in the scored year only, `refuted` when
  ACI's cold-night q95 coverage exceeds arm A's by less than 0.01 in the scored year, and
  `inconclusive` otherwise. The threshold is set by the effort of porting: the rule adds no fitted
  model, only one state of 32 numbers per series (4 hour bands times 8 quantile levels) updated after
  each reading, so a change whose upper bound stays under 0.01 points (under 0.5% of the score) is
  worth porting. The step size gamma is chosen on the tuning months only, from 0.01, 0.03, and 0.1,
  as the value with the highest cold-night q95 coverage among those whose C1 upper bound is at or
  under +0.01; ties go to the smaller gamma. Nothing is fitted on the replicate months or the scored
  year. The verdict is capped at `inconclusive` if arm A rescored from experiment 169's stored
  forecasts differs from the printed 2.7519 by more than 0.001 points, or if the gamma = 0 run
  differs from arm A at all.
benchmark: >-
  Arm A (2.7519 points over leads 0.5 to 168 hours in the scored year; cold-night q95 coverage
  0.896), the constant-widening control CW (the upper offsets times the intercept-only values of
  experiment 169's audit: 1.06 at q90 and q95, 1.11 at q99; cold-night q95 coverage 0.910), BG2,
  climatology, and manual_heuristic as stored by experiment 87, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours; coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles on all rows, on cold
  nights split into the first night of a spell, the 3 days after it, and later nights, on bank
  holidays, and over Christmas
code: ml-experiments/00213-calibration-online-band-adaptation-from-coverage-misses/
---

# Experiment 213: online band adaptation from recent coverage misses (planned)

**This page is the pre-registered plan. The write-up replaces it when the run finishes.**

Experiments [160](../00160-calibration-band-width-fix-on-stage-2-winner/result.md),
[164](../00164-calibration-cold-night-tail-under-coverage/result.md), and
[169](../00169-calibration-combined-fixes-and-ensemble-spread/result.md) widened the demand
forecast's quantile bands with fixed multipliers and found that cold-night q95 coverage stays
near 0.90 (0.896 for arm A, 0.907 for a fitted widening whose gain came from a constant). This
experiment asks whether the band can instead learn from the readings that arrive after each
forecast is issued. It re-implements adaptive conformal inference
([Gibbs and Candes (2021)](https://arxiv.org/abs/2106.00170)) on the quantile offsets of arm A, so
the idea is not ours; the contribution is testing it on cold spells. The mean forecast is
unchanged.

## Update rule (pre-registered)

- **State.** For each series, hour band b (night, morning, midday, evening, as `studies.dp_calendar`
  orders them), and quantile level q in {0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99}, a log
  scale `u[b, q]`, starting at 0. The levels 0.40, 0.50, and 0.60 are not scaled.
- **Update.** A reading at valid time v with hour band b is compared with arm A's own quantile
  from the most recent run issued before v. For an upper level q the miss is `actual > quantile`;
  for a lower level it is `actual < quantile`. After each reading, in valid-time order,
  `u[b, q] += gamma * (miss - alpha_q)`, where `alpha_q = 1 - q` for the upper levels and `alpha_q =
  q` for the lower ones. This is a step of stochastic gradient descent on the pinball loss of the
  scale. `u` is clamped to `[ln 0.7, ln 2.0]`.
- **Use.** A forecast issued at time T for valid time t in hour band b uses the state after every
  reading with a valid time of at most T minus 3.5 hours. Each quantile offset from the mean is
  multiplied by `exp(u[b, q])`, then the quantiles are sorted per row. Readings that are inside
  a hand-filtered switching event, or that are missing, do not update the state.
- **Gamma.** The grid is 0.01, 0.03, and 0.1 per reading (12 readings per hour band per day), chosen
  on the tuning months as the decision rule says. The tuning and replicate months are alternate
  months, so the state restarts at 0 at the first run of each block of consecutive months; the
  scored year is continuous, with the runs that experiment 169 stored.

## Arms and contrasts

- **A.** Experiment 169's arm A, as stored.
- **ACI.** A with the update above, at the tuned gamma. ACI0 is gamma = 0 (a reproduction check:
  it must equal A).
- **CW.** A with the intercept-only widening of experiment 169's audit (a reference, post hoc: it
  is a fixed multiplier, fitted on tuning rows by that audit).
- **C1, C2.** As in the decision rule. C3 (descriptive): ACI minus CW in cold-night q95 coverage.
- **Reported, in no contrast.** Cold-night q95 coverage on the first night of each cold spell, on
  the 3 days after it, and on later nights (a spell is a run of consecutive local dates at one
  series, each holding a cold night; cold nights are as in experiment 169). Effects on bank
  holidays and Christmas. All-row q05 and q95 coverage. Tail table at 0.01, 0.05, 0.95, and 0.99.
  Leave-one-substation-out cold-night coverage. The median forecast minus the mean forecast.

## Sizing

- **Stored:** private files only. **Serving path:** not touched. **Degradation rule:** not
  touched. **More than one defensible design:** the update rule and the clamp are pre-registered;
  the alternatives (per-lead state, a decaying step) are named as next steps. **Callers:** this
  folder's scripts only.
- **Earlier advice followed:** experiment 169 advises judging cold-night coverage over more than
  one winter and testing a bias correction; this experiment tests neither, and only the online
  band. It departs from the advice to test a bias correction first because the maintainer asked
  for this question.
