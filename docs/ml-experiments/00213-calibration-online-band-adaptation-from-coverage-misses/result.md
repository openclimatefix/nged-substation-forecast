---
id: '00213'
slug: calibration-online-band-adaptation-from-coverage-misses
part: CAL-4
theme: calibration-and-uncertainty
also_relevant_to:
- curtailment-cost
- production-reference-and-best-guess
status: done
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
  0.96, and changes the all-row quantile score by -0.01 to +0.01. It may fail because cold nights
  are rare and the feedback is slow: on a cold spell the scale learns after the misses, not before,
  so the first night of each spell gains nothing and the three days after it gain most.
decision_rule: >-
  The reference is arm A (experiment 169's arm A, which is experiment 164's fix O on experiment
  160's smooth spread table), read from experiment 169's stored forecasts. The planned contrasts are
  C1 (arm ACI minus arm A) in quantile score over leads 0.5 to 168 hours on all rows, and C2 (arm
  ACI minus arm A) in cold-night q95 coverage, at the five decision substations D5, D10, D11, D14,
  and D16 (D9 beside them, in no contrast). Intervals are paired month-resampled bootstrap intervals
  with 20,000 resamples at the 95% level (two contrasts, one of them descriptive, so no correction).
  Arm ACI passes in a period when all three hold: its cold-night q95 coverage, pooled over the five
  substations, is 0.90 or more; its all-row q95 coverage lies between 0.94 and 0.96, and its all-row
  q05 coverage is no more than 0.01 further from 0.05 than arm A's (arm A's q05 coverage is 0.065,
  so the literal rule of one point from nominal would fail arm A itself); and the upper bound of C1
  is at or under +0.01 points. The hypothesis is `supported` when ACI passes in the scored year and
  in the replicate months, `partly supported` when it passes in the scored year only, `refuted` when
  ACI's cold-night q95 coverage exceeds arm A's by less than 0.01 in the scored year, and
  `inconclusive` otherwise. The threshold is set by the effort of porting: the rule adds no fitted
  model, only one state of 32 numbers per series (4 hour bands times 8 quantile levels) updated
  after each reading, so a change whose upper bound stays under 0.01 points (under 0.5% of the
  score) is worth porting. The step size gamma is chosen on the tuning months only, from 0.01, 0.03,
  and 0.1, as the value with the highest cold-night q95 coverage among those whose C1 upper bound is
  at or under +0.01; ties go to the smaller gamma. Nothing is fitted on the replicate months or the
  scored year. The verdict is capped at `inconclusive` if arm A rescored from experiment 169's
  stored forecasts differs from the printed 2.7519 by more than 0.001 points, or if the gamma = 0
  run differs from arm A at all.
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
verdict: inconclusive
best_arm: >-
  None. ACI at gamma 0.01 (the smallest value, chosen by a fallback because no gamma met the
  tuning-month bound) reaches cold-night q95 coverage 0.906 in the scored year against arm A's
  0.896, with all-row quantile score +0.0132 (-0.0010 to +0.0295), over the +0.01 limit; in the
  replicate months its coverage falls to 0.892 against 0.913. The constant-widening control CW
  reaches 0.910 and 0.920 at +0.0013 and +0.0043
headline: >-
  Learning each band's width online from recent misses does not lift cold-night q95 coverage
  reliably. In the scored year the best-tuned step lifts it from 0.896 to 0.906, with a
  worse all-row score (+0.013) and worse bank-holiday and Christmas scores. In the replicate months
  it lowers coverage from 0.913 to 0.892. A fixed widening of the upper quantiles does at least as
  well on both periods with almost no change in the all-row score.
lessons:
  - >-
    The scale learned from misses helps only on the later nights of a cold spell (more than 3 days
    after the first night): coverage there rises from 0.884 to 0.908 (scored year), against 0.904
    for a fixed widening. It does not help on the first night (0.870 to 0.857) or the 3 days after
    it (0.917 to 0.916).
  - >-
    A larger step (gamma 0.03) reaches 0.941 cold-night coverage in the scored year but raises the
    all-row score by 0.030 points (worse) and falls below arm A in the replicate months, so no step
    size passes both periods.
  - >-
    The scale mostly narrows the lower band (all-row q05 coverage 0.065 to 0.055, q01 0.022 to
    0.017) and leaves the all-row q95 coverage at 0.947.
take_home_message: >-
  Letting the forecast widen or narrow its bands after each recent miss does not fix the
  under-coverage on cold nights: it lifts the 95% quantile's coverage from 0.896 to 0.906 in the
  scored year and lowers it in the other months, and it worsens the all-row, bank-holiday, and
  Christmas scores. A fixed widening of the upper quantiles does as well, so the cold-night gap is a
  shortfall that a signal arriving after the misses cannot learn in time, and the next step is to
  correct the mean forecast on the coldest days before any further work on the bands.
advice:
  - >-
    Do not add an online coverage-miss update to the band scale for cold nights: no step size passes
    both the scored year and the replicate months, and the best one raises the all-row score by
    0.013 points (worse; C1, ACI minus arm A).
  - >-
    If cold-night q95 coverage matters, use a fixed upper widening (CW: 1.06, 1.06, and 1.11 at q90,
    q95, and q99), which reaches 0.910 in the scored year and 0.920 in the replicate months at
    +0.001 and +0.004 points (C3, post hoc).
  - >-
    Test a bias correction of the mean forecast at ensemble-mean temperatures of 0 degrees C or
    below, as experiment 169 advises, before more work on band width: bands that learn from misses
    cannot see a cold spell coming (the first night of a spell gains nothing).
date: 2026-10-22
production_status: not-for-production
adopted_in: none
---

# Experiment 213: online band adaptation from recent coverage misses

**Learning the band widths from the misses of the last few days did not lift cold-night coverage
reliably, and a fixed widening of the upper quantiles did as well, with almost no change in the
all-row score.** A quantile is a level of the forecast's error distribution: the q95 is the value
that the demand reading should fall at or under on 95% of rows. Coverage is the share of rows on
which it does. A cold night is a winter night (November to March, evening or night hours) whose ERA5
day-mean temperature (the ECMWF reanalysis, an estimate of past weather) is below 2 °C.

**We asked whether the quantile offsets of the best demand forecast so far can learn from recent
misses.** That forecast is arm A of [experiment
169](../00169-calibration-combined-fixes-and-ensemble-spread/result.md). An offset is the distance
from the mean forecast to a quantile. After every reading that is at least 3.5 hours old, the offset
grows if the reading fell outside the quantile and shrinks if it fell inside. The aim was cold-night
q95 coverage of 0.90 or more without a second fitted model of the forecast's spread. The update is
that of adaptive conformal inference (ACI, [Gibbs and Candes
(2021)](https://arxiv.org/abs/2106.00170)).

**The result is a small gain in one period, a loss in the other, and a worse score.** In the scored
year (2025-07-02 to 2026-06-29) the best of three learning rates (gamma, the size of each update)
lifts coverage from 0.896 to 0.906. It also worsens the all-row quantile score by 0.013 points of
effective capacity (percentage points of each substation's rated demand; lower is better), which is
over the pre-registered limit of 0.010. In the replicate months (other months of the earlier year),
ACI at gamma 0.01 lowers coverage from 0.913 to 0.892. The pre-registered rule gives the verdict
`inconclusive`, because the coverage gain (+0.0102) is just over the 0.01 below which the rule says
`refuted`. In substance, online adaptation did not do what it was meant to do.

![Cold-night q95 coverage by arm](figure_1_cold_night_coverage.svg)

![Cold-night q95 coverage by place in the cold spell](figure_2_cold_spell_stage.svg)

- **Arms.** Arm A: experiment 169's arm A. Arm ACI: arm A with each of the quantile offsets at q01,
  q05, q10, q25, q75, q90, q95, and q99 times `exp(u)`, where `u` is learned per series, hour band,
  and level by `u += gamma * (miss - alpha)` after each reading and clamped to [ln 0.7, ln 2.0]. The
  gamma grid is 0.01, 0.03, and 0.1. Arm CW (reference, post hoc): arm A with the q90, q95, and q99
  offsets times 1.06, 1.06, and 1.11, the intercept-only values of experiment 169's audit.
- **Contrasts.** C1: ACI minus A in all-row quantile score. C2: ACI minus A in cold-night q95
  coverage. C3 (descriptive, post hoc): ACI minus CW in cold-night q95 coverage. Each is a paired
  difference with a month-resampled 95% interval.
- **Row set and schedule.** `fold_a_daily_gap_v1`, the five decision substations D5, D10, D11, D14,
  and D16 (D9 written beside, in no contrast), leads 0.5 to 168 hours, rows outside the
  hand-filtered switching events: 464,363 rows in the scored year and 299,306 in the replicate
  months. The mean forecasts are experiment 155's stored monthly refits, so no contrast depends on
  the retraining schedule. The tuning and replicate months are alternate months, so the state
  restarts at 0 whenever readings are more than 36 hours apart; the restart may handicap ACI in
  those months.
- **Gamma selection (deviation from the plan).** No gamma had a tuning-month C1 upper bound at or
  under +0.01 (+0.012 at gamma 0.01, +0.051 at 0.03, +0.108 at 0.1), so the pre-registered selection
  rule selected no gamma. The script fell back to the smallest gamma, 0.01, which the plan did not
  state. Cold-night q95 coverage in the tuning months: A 0.918, ACI 0.925, 0.921, and 0.901 at the
  three gammas.
- **Headline, scored year (gamma 0.01).** Cold-night q95 coverage 0.906 against A's 0.896 (+0.010,
  95% interval -0.007 to +0.025; 5,395 rows). All-row q95 coverage 0.947, all-row q05 coverage 0.055
  (A: 0.948 and 0.065). C1, all rows: +0.0132 (-0.0010 to +0.0295). Of the pre-registered pass
  conditions (cold-night coverage, all-row coverage, score bound), only the score bound fails.
- **Headline, replicate months.** Cold-night q95 coverage 0.892 against A's 0.913 (-0.021, -0.041 to
  0.000). C1 -0.0017 (-0.0208 to +0.0189). ACI fails the coverage condition and the score bound.
- **Gamma sweep, scored year (cold-night q95 coverage, C1).** 0.01: 0.906, +0.013; 0.03: 0.941,
  +0.030; 0.1: 0.909, +0.034. Replicate months: 0.892, 0.874, and 0.889, so no step size passes both
  periods (post hoc reading of the sweep, not a selection).
- **Absolute scores, scored year, decision substations.** A 2.7519, CW 2.7532, ACI 2.7651, BG2 (the
  best-guess forecast of experiment 87) 2.7804, manual heuristic 3.4968, climatology 3.8038.
- **Cold spells, scored year (cold-night q95 coverage; A, ACI, CW).** First night of a spell: 0.870,
  0.857, 0.876 (516 rows). The 3 days after it: 0.917, 0.916, 0.925 (2,242 rows). Later nights:
  0.884, 0.908, 0.904 (2,637 rows). A night is keyed to the evening on which it starts. The stages
  hold 2, 5, and 8 distinct night dates, so these are point values without intervals. Replicate
  months: first night 1.000 for all three arms (804 rows); 3 days after 0.884, 0.856, 0.893; later
  nights 1.000 (112 rows).

- **Special days, scored year (A, ACI).** Bank holidays (9,812 rows): q95 coverage 0.934, 0.928; C1
  +0.086 (+0.019 to +0.131), so the score worsens. Christmas, 24 December to 1 January (8,641 rows):
  q95 coverage 0.934, 0.951; q05 coverage 0.075, 0.094; C1 +0.093, and no interval is quoted because
  the period lies in one month, which leaves the month-resampled bootstrap nothing to resample.
- **Tail calibration, scored year (coverage / pinball loss; A, ACI).** All rows: q01 0.022 / 0.216
  and 0.017 / 0.210; q05 0.065 / 0.714 and 0.055 / 0.718; q95 0.948 / 0.745 and 0.947 / 0.756; q99
  0.982 / 0.232 and 0.984 / 0.233. Cold nights: q01 0.045 / 0.392 and 0.031 / 0.354; q05 0.090 /
  0.950 and 0.099 / 0.943; q95 0.896 / 0.819 and 0.906 / 0.858; q99 0.959 / 0.282 and 0.959 / 0.291.
- **Against the constant widening.** ACI minus CW in cold-night q95 coverage: -0.004 (-0.019 to
  +0.006) in the scored year and -0.028 (-0.055 to 0.000) in the replicate months.
- **Caveats.** The scored year holds 10 cold days, so cold-night figures rest on a handful of
  spells, and the interval on the cold-night coverage gain is wide (-0.007 to +0.025). Leaving one
  substation out moves ACI's scored-year cold-night coverage to 0.901 (D5), 0.921 (D10), 0.925
  (D11), 0.879 (D14), and 0.907 (D16), so the 0.90 pass depends on D14. The chosen gamma is the
  fallback described above. Arm A's q05 coverage (0.065) already misses the plan's rule that
  coverage stay within 0.01 of nominal, so the plan used "no further from 0.05 than A, plus 0.01".
  The replicate-month result may reflect the monthly state restarts. That explanation was not
  tested. The curtailment-cost metric was not run: experiment 169's figures show that it cannot rank
  forecasts this close.
- **What to try next.** (1) A bias correction of the mean forecast at ensemble-mean temperatures of
  0 °C or below, since bands that learn from misses cannot react before a cold spell starts. (2) If
  an online band is still wanted, one that learns separately for ensemble-cold days, scored over at
  least two winters, with a continuous history instead of alternate months.

## Checks

- **Median forecast minus mean forecast, per site.** 0.000 for every arm at every substation: no arm
  scales the q50 offset.
- **Mean-forecast error beside each quantile-score contrast.** The mean absolute error of the mean
  forecast is identical across all arms (4.7096 points in the scored year), because the mean is
  unchanged, so the quantile-score contrasts cannot come from the mean.
- **Reproduction.** Arm A rescored from experiment 169's stored forecasts gives 2.7521 on all five
  decision substations' rows in this experiment's scorer (2.7519 on the rows shared with BG2), a gap
  of 0.0002 from the printed 2.7519, inside the 0.001 cap. A gamma of 0 reproduces arm A's quantiles
  exactly at every substation and period (the script raises otherwise).
- **Prediction check.** Cold-night coverage rose by 0.010 in the scored year (predicted 0.00 to
  0.03), but fell by 0.021 in the replicate months. The all-row score changed by +0.013 (predicted
  -0.01 to +0.01), just outside the range.
- **Audit triggers that fired.** Two of the seven triggers in the `ml-experiment` skill fired.
  (Trigger 2) The prediction failed in the replicate months, where coverage fell, and the monthly
  state restart is an untested explanation. (Trigger 6) The C1 interval in the scored year (-0.001
  to +0.030, width 0.031) is 2.3 times as wide as experiment 169's C2 interval on the same rows
  (width 0.013). An online scale that changes from month to month is expected to vary more than a
  fixed multiplier, but that expectation was not tested. No Opus audit was run, because this
  experiment's brief excluded Opus. The verdict is `inconclusive`, so an audit finding could only
  confirm or weaken it, and the decision to merge without an audit is left to the maintainer.
