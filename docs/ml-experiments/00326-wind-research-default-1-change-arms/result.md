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
status: planned
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
  the tier's minimum gain: 0.02 points for (c), a new fitted step, and 0 for (a), (b), and (d). Condition 2,
  no harm: the one-sided 95% upper bound of the harm to the pooled 0.01 pinball loss and to the
  pooled 0.99 pinball loss is at most 0.05 points each; each farm on its own, rows with wind above
  rated, and lead days 1 to 2 are reported without a pass or fail. Condition 3, replication: the
  replicate period of 2025-01 to 2025-06 gives the same sign, at least half of the smaller of the
  main estimate and the minimum gain, and a t test on the 6 monthly paired differences does not
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
---

**Plan only: no score has been read.** This page is committed before any code runs, as the
`ml-experiment` skill requires. It holds the five plan items of the
[research-default rule](../research-default.md) and is filled in with results afterwards.

## Plan item 1: family and version

- **Family and version.** Wind farms research default 1 (family "Wind farms W1 to W3"), named in
  the maintainer's weekend plan of 2026-10-24: the ERA5-trained XGBoost quantile model of
  [experiment 33](../00033-best-guess-architecture-vs-production/result.md) (a binned power curve
  as the base margin, a monthly spread factor), run on the 51 ensemble members and mixed with equal
  weights, plus the offset table of [experiment
  259](../00259-wind-plant-plus-quantile-offset-head/result.md) (one offset per lead day, quantile
  level, and farm, fitted on the 442 daily hindcast runs of 2024-04-01 to 2025-06-16). The table
  is applied to the default's spread-adjusted mixture, as in experiment 303's P1.
- **Family settings.** Rows `fold_a_tuesdays_v1`; kept rows (experiment 210 mask) headline, all
  rows beside; lead range 1 to 10 days, so every headline says "scored at 1 to 10 days"; refits:
  the ERA5-trained XGBoost quarterly on ERA5 and the table monthly (here a single ERA5 fit before
  2024-04-01 and a single table, as in experiments 259 and 303, which the page states as a
  caveat); metric the quantile score over 11 levels in points of capacity; tolerance 0.02
  headline and 0.05 on the 0.01 and 0.99 tails; reported groups each farm, wind above rated, and
  lead days 1 to 2.
- **Rows.** The 3 farms, 3 seeds, and the 53 Tuesday runs of the scored year; the replicate is the
  daily hindcast runs initialised 2025-01-01 to 2025-06-16.

## Plan item 2: arms

- **Default.** Read the stored default (experiment 285's `xgb_B0` mixtures and experiment 303's
  `P1`, `P8`). The reproduction gate refits the default with this folder's code and requires its
  kept-row score (spread factor and table applied) to match experiment 303's P1 (11.352) within
  0.01. The run stops and reports if the gate fails.
- **Default plus (a).** Seventh and eighth XGBoost columns: 2 m temperature (degrees Celsius) and
  air density (fixed surface pressure of 100,000 Pa divided by the dry-air gas constant times the
  temperature in kelvin, as in experiment 256). With a fixed pressure the density is a monotone
  function of the temperature, so the density column adds no information to a tree model.
- **Default plus (b).** Seventh and eighth columns: hour of day (fraction) and day of year.
- **Default plus (c).** The effective wind speed of experiment 285's adapter A4, fitted to ERA5
  before 2024-04-01 and never fine-tuned, as a seventh column. This is experiment 303's P8, read
  from storage for the scored year and the hindcast.
- **Default plus (d).** Every member's 100 m wind speed multiplied by one factor before the
  XGBoost model (the feature and the base-margin power curve alike; the 10 m speed is not scaled).
  The factor is the mean of the implied 100 m wind (experiment 307) over the mean of the ensemble
  mean 100 m wind, on the unmasked, invertible hindcast cases (experiment 307 found 1.13).
- **Adoption order** c, b, a, d. With k = 4 arms the interval for condition 1 is read at 1 - 0.01/4.

## Plan item 3: tier

All four arms act inside the fitted stage (tier 2), so the default's code is imported from the
experiment folders that hold it. Complexity tiers: (a) and (b) added columns, (d) a changed
setting, minimum gain 0; (c) a new fitted step, minimum gain 0.02 points.

## Plan item 4: predictions that can fail

See `prediction` above. Each can fail: (c) may not beat 0.05 or may gain more at days 3 to 10; (b)
may not lose half its gain under the placebo; (a) may reach 0.1; (d) may exceed 0.05; any arm may
raise a tail loss by more than 0.05.

## Plan item 5: controls

- **Column subsampling off** in every XGBoost fit (`colsample_bytree`, `colsample_bylevel`, and
  `colsample_bynode` all 1.0).
- **Month-shuffled placebo for (a) and (b).** Same columns, but each calendar month's rows take
  the two added columns from a randomly paired other calendar month of the same set, at the same
  position within the month (modulo the paired month's length), with a fixed seed. The placebo
  keeps each column's distribution and its autocorrelation within a month and breaks the link to
  the date and the wind. It is applied to the ERA5 training rows and to each set of member rows.
  The share of the gain recovered is (placebo minus default) over (arm minus default).
- **Shuffled-column placebo for (c).** The effective-speed column permuted at random across rows
  within the training rows and within each set of member rows, with a fixed seed. Needs the
  ERA5-trained adapter, which is refitted (three seeds, three farms) and checked to reproduce
  experiment 303's P8 effective speed through the stored P8 mixtures' score.
- **Known-answer test for (d).** Multiply the hindcast members by a known factor of 0.85, refit
  the factor with the same routine, and require it to recover 1/0.85 to within 0.01.
- **Leakage check.** Every column is available at forecast time: the ensemble's wind and
  temperature, the clock of the valid time, and an adapter that reads only ensemble weather. The
  XGBoost models and the adapter are fitted on ERA5 valid times before 2024-04-01, so no row of
  the hindcast or the scored year is in a fit. The offset table and the factor for (d) are fitted
  on hindcast cases only.
- **Weather at lag times.** None used at forecast time, so the ERA5 delay does not apply.

## Replicate period (condition 3)

The offset tables and the factor for (d) are refitted on hindcast cases with valid times before
2025-01-01 (2024-04 to 2024-12), and the daily hindcast runs initialised 2025-01-01 to 2025-06-16
are scored on the cases the refits did not see. The spread factor is fitted month by month on
earlier cases as in the scored year. The nominal 99% interval undercovers on 6 months, so a t test
on the 6 monthly paired differences is added.

## Reporting

Take-home first, `kind: change`, the lead range as the headline caveat, the default's absolute
score against climatology, the persistence on hours 0 to 6, and the frozen XGBoost reference with
intervals, the five conditions one line each, the tail calibration (coverage and pinball at 0.01,
0.05, 0.95, and 0.99), and the curtailment-cost metric of experiment 277 for the winning arm.
Intervals resample whole months and the three seeds; farms are not resampled, so every interval is
conditional on three farms.
