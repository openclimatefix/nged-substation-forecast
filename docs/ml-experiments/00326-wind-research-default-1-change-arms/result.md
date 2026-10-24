---
id: "00326"
slug: wind-research-default-1-change-arms
title: >-
  Does adding temperature and air density, the clock, the neural adapter's effective wind speed, or
  one wind-speed scale factor improve Wind farms research default 1?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: finished
kind: change
date: 2026-10-24
size: >-
  complex by the research-default rule, because the experiment adds four arms to a family that
  adopts at most one. Stored: private mixtures, tables, and scored quantiles under
  data/ml-experiments/00326-wind-research-default-1-change-arms/ only, plus the default plus x
  forecasts of the adopted arm if there is one. Serving path: not touched (arm (c) would need a
  neural adapter in the serving path to port, which the page says and does not do). Degradation
  rule: not touched. Designs: several are defensible for the placebos and for the replicate period,
  so this plan fixes one of each before any score is read. Callers: this folder's scripts, which
  import experiments 259, 285, 303, and 307 and call no reviewed module beyond `studies.*`. One
  Sonnet code review and one Sonnet prose review (maintainer's brief), no Opus.
hypothesis: >-
  Each of four changes, added alone to Wind farms research default 1, lowers the quantile score on
  the kept rows of the three metered wind farms W1 to W3 at 1 to 10 day leads: (a) 2 m temperature
  and an air-density column as XGBoost inputs, (b) hour of day and day of year as XGBoost inputs,
  (c) the ERA5-trained neural adapter's effective wind speed as an extra XGBoost column, and (d)
  one wind-speed scale factor, fitted on the hindcast to the implied wind, applied to every
  member's 100 m speed before the XGBoost model. At most one is adopted, in the order c, b, a, d.
prediction: >-
  Written before any score. (c) lowers the kept-row score by 0.05 to 0.15 points, with a larger gain
  at lead days 1 to 2 than at days 3 to 10. (b) lowers it by 0.05 to 0.15, but its month-shuffled
  placebo recovers more than half of that gain (calendar columns act as a date fingerprint). (a)
  lowers it by under 0.1. (d) lowers it by under 0.05, because the offset table already corrects
  the median. No arm raises the 0.01 or 0.99 pinball loss by more than 0.05.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order c, b, a, d, and the
  first arm that passes all five is adopted (at most one). Condition 1, gain: the 99.75% month-block
  interval of (default plus x, minus default) on the kept-row quantile score, scored at 1 to 10
  days, lies wholly below zero (level 1 - 0.01/4 for four arms), and the point estimate is at least
  the tier's minimum gain: 0.02 points for (c), a new fitted step, and 0 for (a), (b), and (d).
  Condition 2, no harm: the one-sided 95% upper bound of the harm to the pooled 0.01 pinball loss
  and to the pooled 0.99 pinball loss is at most 0.05 points each; each farm on its own, rows with
  wind above
  rated, and lead days 1 to 2 are reported without a pass or fail. Condition 3, replication: the
  replicate period of 2025-01 to 2025-06 gives the same sign, at least half of the smaller of the
  main estimate and the minimum gain, and a t test on the six monthly paired differences does not
  contradict the sign. Condition 4: the reproduction gate, the placebos, the known-answer test for
  (d), and the leakage check pass, and no audit trigger is unresolved. Condition 5: every column is
  available at forecast time, and the page says if the port needs a serving-path change. A change
  that wins on condition 1 and fails another stays a candidate. Three farms on 53 runs give wide
  intervals, so a candidate is the likely outcome.
benchmark: >-
  The default: Wind farms research default 1, the ERA5-trained XGBoost of experiment 33 on the 51
  members with the spread factor, plus experiment 259's offset table (experiment 285's xgb_B4,
  experiment 303's P1, 11.352 points on the kept rows). Also reported beside it: climatology, the
  persistence on hours 0 to 6, and the frozen XGBoost reference (the default without the table).
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball
  losses.
code: ml-experiments/00326-wind-research-default-1-change-arms/
verdict: supported
best_arm: >-
  (c), the default plus the ERA5-trained adapter's effective wind speed as an extra XGBoost column,
  11.253 points on the kept rows against 11.352 for the default (-0.099, 99.75% interval -0.185 to
  -0.011)
headline: >-
  Of four changes added alone to Wind farms research default 1, only the adapter's effective speed
  (c) had a 99.75% interval wholly below zero (-0.099 points, scored at 1 to 10 days), and it passes
  all five adoption conditions. Adopting it needs the maintainer's approval because a port puts a
  neural adapter in the serving path. Hour and day of year (b) gained as much (-0.092) but its
  interval includes zero.
production_status: candidate
adopted_in: none
lessons: >-
  The adapter's gain is flat across lead (-0.086 at days 1 to 2, -0.093 at days 3 to 10), so it is
  a correction of the ensemble's wind-speed level and not of short-range detail. The hour-and-day
  columns gained as much as the adapter in the scored year, and a month-shuffled placebo recovered
  none of that gain, so the placebo does not explain the gain; the replicate period is too short to confirm
  or refute it. A single speed factor does nothing once the offset table is applied.
take_home_message: >-
  Feeding XGBoost the neural adapter's corrected wind speed lowers the wind forecast's score by
  0.10 points of capacity, about 0.9% of the default's score, at 1 to 10 day leads (99.75% interval
  -0.19 to -0.01). The change is negative at every farm and at short and long leads. The three
  other changes were not separable from zero. Only a port that puts the adapter in the production
  service can use the gain, so the maintainer decides.
advice:
  - >-
    Give XGBoost the ERA5-trained adapter's effective speed as an extra column if the wind forecast
    is to be improved, and decide separately whether a neural adapter in the serving path is worth
    0.10 points (C1, arm (c)).
  - >-
    Do not add a single wind-speed scale factor on top of the offset table: it scored 0.014 points
    worse than the default and made lead days 1 to 2 worse by 0.091 (arm (d)).
  - >-
    Test hour of day and day of year again on a longer replicate period before adopting them: they
    gained 0.092 points in the scored year, their interval included zero, and the 6-month replicate
    could not separate them from zero (arm (b)).
  - >-
    Do not expect temperature and air density to help a wind XGBoost model that already has the
    wind speed and direction: arm (a) gained 0.032 points with an interval from -0.136 to +0.060,
    and air density at a fixed pressure is a function of temperature alone (arm (a)).
---

**Take-home: giving XGBoost the neural adapter's corrected wind speed lowers the wind forecast's
quantile score by 0.099 points of capacity (about 0.9%), scored at 1 to 10 day leads.** Of four
changes, only this one has a 99.75% interval wholly below zero. We asked whether any of four
changes, each added alone to Wind farms research default 1, lowers the quantile score of the three
metered wind farms (W1 to W3). The default is an XGBoost model (gradient-boosted trees) fitted to
ERA5 (our estimate of past weather). It runs on the 51 members of the ensemble forecast from ECMWF
(the European weather centre), then adds one correction per lead day, quantile level, and farm,
fitted on 442 daily ensemble forecasts. The four changes were: (a) 2 m temperature and air density
as extra inputs, (b) hour of day and day of year, (c) the effective wind speed of a small neural
adapter (a neural network fitted to ERA5 that rescales the wind speed that a physics-based
wind-power model reads), and (d) one wind-speed factor of 1.13 applied to every member. The score is
a quantile score over 11 quantile levels, in points of capacity (1 point is 1% of a farm's capacity;
lower is better), on the kept rows: the rows left after the mask of [experiment
210](../00210-wind-idle-noise-mva-artefact/result.md) removes half-hours when a farm is idle. The
default scores 11.352 points on the kept rows. The adapter (c) passes all five adoption conditions
of the [research-default rule](../research-default.md). Porting (c) puts a neural network in the
production service, and the rule leaves that adoption to the maintainer, so this page names (c) a
candidate and logs no version. The verdict is `supported`.

![Each change minus the default, scored year and replicate period](figure_1.svg)

![Each change's gain at lead days 1 to 2 and days 3 to 10](figure_2.svg)

![Arm (c) and (b) minus the default by month](figure_3.svg)

## Design

- **Family and default.** Wind farms research default 1: [experiment
  33](../00033-best-guess-architecture-vs-production/result.md)'s ERA5-trained XGBoost on the
  members, with the spread factor and [experiment
  259](../00259-wind-plant-plus-quantile-offset-head/result.md)'s offset table, as [experiment
  303](../00303-wind-adapter-in-production-setting/result.md)'s P1 and [experiment
  285](../00285-wind-effective-wind-adapters/result.md)'s `xgb_B4`. Rows `fold_a_tuesdays_v1` (53
  Tuesday runs, July 2025 to June 2026), 3 farms, 3 seeds. Scored at 1 to 10 days. The XGBoost
  models and the adapter are fitted once on ERA5 before 2024-04-01 and the table once on the
  hindcast, not refitted quarterly and monthly as the family settings say.
- **Arms and tier.** Tier 2 (inside the fitted stage), code imported from experiments 259, 285,
  303, and 307. (a) and (b) added columns, (d) a changed setting (minimum gain 0); (c) a new fitted
  step (minimum gain 0.02 points). Adoption order c, b, a, d; interval read at 99.75% (four arms).
  Arm (c) is experiment 303's P8, read from storage.
- **Controls.** Column subsampling off. Month-shuffled placebos for (a) and (b) (the added columns
  taken from a randomly paired other calendar month, in the ERA5 rows and the member rows) and a
  permuted-column placebo for (c). Known-answer test for (d): the refit factor after scaling the
  members by 0.85 times 0.85 equals the unscaled factor (difference 3e-8). Reproduction gate: the
  refit default is bit-identical to P1 (11.352 points, difference 0).
- **Replicate.** The offset table and (d)'s factor refitted on hindcast cases with valid times
  before 2025-01-01; the daily runs of 2025-01-01 to 2025-06-16 scored. Six months.
- **Intervals** resample whole months and the 3 seeds (50,000 resamples); the farms are not
  resampled, so every interval is conditional on these three farms.

## Results

| Arm | Kept rows | All rows | Days 1 to 2 | Days 3 to 10 | 0.01 pinball | 0.99 pinball | 5%-95% coverage |
|---|---|---|---|---|---|---|---|
| Default | 11.352 | 10.825 | 6.612 | 11.191 | 0.284 | 0.640 | 92.1% |
| Default plus (c) | 11.253 | 10.746 | 6.525 | 11.097 | 0.283 | 0.639 | 92.0% |
| Default plus (b) | 11.260 | 10.763 | 6.542 | 11.110 | 0.282 | 0.643 | 91.1% |
| Default plus (a) | 11.320 | 10.804 | 6.600 | 11.154 | 0.283 | 0.637 | 92.0% |
| Default plus (d) | 11.366 | 10.843 | 6.702 | 11.211 | 0.283 | 0.630 | 91.5% |

- **Benchmarks, kept rows.** The default scores 11.352. The frozen XGBoost reference (the default
  without the table) scores 11.697, so the table gains 0.345 (95% interval 0.274 to 0.418).
  Climatology scores 13.547, so the default is 2.195 better (1.577 to 2.808). Persistence is scored
  on hours 0 to 6 only (1,696 kept rows): 15.566 against the default's 6.568 on the same rows.
- **(c) minus default: -0.099 (99.75% interval -0.185 to -0.011; 95% -0.157 to -0.041).**
  Predicted -0.05 to -0.15, held. The prediction of a larger gain at days 1 to 2 than days 3 to 10
  failed; the gains are -0.086 (-0.155 to -0.021) and -0.093 (-0.167 to -0.021). The gain of (c) is
  negative at W1, W2, and W3 (-0.091, -0.129, -0.078) and without any one farm (-0.104, -0.084,
  -0.109), and negative in 10 of 12 months. Placebo for (c): -0.001 (99.75% interval -0.002 to
  +0.001), so the share recovered is 0.6%.
- **(b) minus default: -0.092 (99.75% interval -0.190 to +0.014).** The prediction (-0.05 to
  -0.15) held. The prediction that its placebo recovers over half the gain failed. The placebo is
  -0.001 (-0.069 to +0.071), a share of 0.8%, so the gain is not a date fingerprint. The ERA5 fits
  cover 2019 to 2024 and never see 2025 and 2026 dates, so a fingerprint was unlikely; the placebo
  tested it.
- **(a) minus default: -0.032 (-0.136 to +0.060).** The prediction (a gain under 0.1) held.
  Placebo -0.001.
- **(d) minus default: +0.014 (-0.094 to +0.137).** The prediction (a gain under 0.05) failed in
  sign: the factor lowers the 0.99 pinball loss by 0.010 and the below-median share rises to 42.4%
  from 40.4%, but days 1 to 2 are worse by 0.091 and W3 by 0.262.
- **Tails.** No arm raises the 0.01 pinball loss (upper 95% bound at most +0.0001) or the 0.99
  pinball loss (at most +0.031, arm (b)) by more than 0.05, as predicted. Coverage of the 5%-95%
  band falls from 92.1% to 91.1% for (b) and stays at 92.0% for (c).

## Adoption conditions for (c)

- **1, gain: met.** The 99.75% interval is wholly below zero and the gain 0.099 exceeds 0.02.
- **2, no harm: met.** The one-sided 95% upper bounds of the harm are -0.0006 at 0.01 and +0.013
  at 0.99, against 0.05. Reported without pass or fail: each farm is better; days 1 to 2 are better
  by 0.086; rows with wind above rated are better by 0.45, on a small subset whose interval the
  month resampling cannot give.
- **3, replication: met.** The replicate gain is -0.084 (99.75% interval -0.211 to +0.062), the
  same sign and above half of 0.02. The t test on the six monthly differences gives p = 0.13, so the
  sign is not contradicted but not established either.
- **4, controls: met.** The reproduction gate, the permuted-column placebo, and the known-answer
  test passed. No audit trigger fired for (c) (see Checks).
- **5, serving: met on inputs, with a port that touches the serving path.** The inputs are
  ensemble weather only. Porting needs the adapter, a neural network, in the production service, so
  the rule requires the maintainer's approval before adoption.

**Only (c) passes condition 1: (b), (a), and (d) do not.** (b) is next in the adoption order. Each
of the four arms has a wide 99.75% interval because there are three farms and 12 months, as the
research-default page predicted for a family of three farms.

## Checks

- **Median minus mean forecast.** Kept rows: default -4.805 points, (c) -4.720, (b) -4.548. The
  median sits below the mean in every arm, as in [experiment
  259](../00259-wind-plant-plus-quantile-offset-head/result.md).
- **Mean absolute error of the mean forecast beside the quantile score.** (c): quantile score
  -0.099, mean absolute error of the mean forecast -0.122, median absolute error -0.130, all three
  measures fall. (b): -0.092, -0.100, -0.109. (d): +0.014, +0.116, -0.019, so (d)'s mean forecast is
  worse.
- **Triggers.** Trigger 2 (a result contradicts its own prediction) fired for the lead pattern of
  (c) and the placebo of (b), and is explained above by tested placebos and a flat gain across lead.
  Trigger 2 bears on (b) and on the prediction about lead, not on (c)'s gain. Triggers 1, 3, 4
  (leaving any one farm out keeps the sign of (c)), 5, 6 (the 95% interval of (c) matches experiment
  303's P8 against P1, which is the same rows), and 7 did not fire.
- **Reproduction.** The refit default equals experiment 303's P1 to the last digit. Refit default
  versus stored default gives no difference, so every contrast reads the same against either.

## Caveats

- **Scored at 1 to 10 days, three farms, 12 months.** Intervals do not cover other farms.
- **The fits are single ERA5 fits,** not the quarterly refits the family settings name, so the
  contrasts could differ under refitting.
- **The replicate is 6 months** and its interval is wide; its sign agrees with the scored year for
  (c) and (b), (a) is null in both periods, and (d) is worse in both.
- **The curtailment-cost metric of experiment 277 is not scored.** It needs a substation node with
  its generators, and W1 to W3 have no node forecast in this experiment. Tail calibration is above.

## What the next experiment should answer

- **Does the adapter's gain survive quarterly refits of the XGBoost model and the adapter?** Run
  (c) with the family's refit cadence.
- **Is hour of day and day of year real?** Test (b) alone on a longer replicate, with the adapter
  already in the default.
