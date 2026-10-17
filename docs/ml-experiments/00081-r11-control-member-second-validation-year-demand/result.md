---
id: "00081"
slug: r11-control-member-second-validation-year-demand
status: planned
size: complex
hypothesis: >-
  The Primary substation demand findings of experiments 48 and 49 keep their sign on forecast runs
  the project has not scored: the daily 00:00 UTC ECMWF ensemble (ENS) control-member runs of
  2023-07-01 to 2024-03-31 (July to March, with no April to June). For the 13 Primary substations
  that have power data throughout, weekly refitting of the XGBoost demand model N1 without error
  correction scores below the stack (frozen N1 with the 14-day correction and the spread factor
  1.15), and weekly refitting scores below monthly refitting.
prediction: >-
  Weekly minus the stack is -0.05 to -0.25 (experiment 48 found -0.20 on 51-member forecasts; one
  member gives a noisier forecast, so the gap may shrink or grow). Weekly minus monthly is -0.03
  to -0.15. Monthly minus the stack is -0.20 to +0.10, unresolved as in experiment 48. Dropping the
  Christmas column and lengthening the half-life from 0.5 to 2 years each move the monthly
  rolling-window refit's score by -0.03 to +0.03. The blend of four models scores 0.03 to 0.10 below
  its best member. Weekly refitting's training weight at 9 km grows through the fold (about 65%
  by March 2024 at a half-life of 0.5 years), so any gain from adopting the 9 km product should be
  larger late in the fold than early.
decision_rule: >-
  Six planned contrasts on the quantile score (twice the mass-weighted mean pinball loss over 9
  levels) in points of capacity, over the 13 Primaries and all 274 runs of fold B, seed 0, paired.
  Each run forecasts about 14.6 days, so neighbouring weeks share weather; the block is therefore 4
  weeks of forecast start (about 10 blocks), with 50,000 resamples, at 99.17% (a Bonferroni family
  of six). A t-interval over the 9 monthly mean differences is reported beside every contrast. The
  block bootstrap governs the verdict, and a result whose t-interval includes zero is labelled
  fragile in the verdict line, as experiment 48's was. C1 weekly (no correction) minus the stack.
  C2 monthly minus the stack (descriptive, because experiment 48's interval crossed zero). C3
  weekly minus monthly. C4 the monthly rolling 12-month refit without the Christmas column minus
  the same refit with it. C5 the same refit with a half-life of 2 years minus 0.5 years. C6 the
  blend (one weight vector per month, chosen on earlier months, over weekly, monthly, rolling, and
  stack) minus its best single member chosen with hindsight. C1 and C3 replicate (rule R-a) when
  the estimate is negative and its 99.17% interval lies wholly below zero. They hold when pooled
  (rule R-b) when fold B's estimate is negative, its unadjusted 95% interval lies wholly below zero,
  and the precision-weighted mean of the experiment 48 estimate (C1 -0.20, C3 -0.13) and fold B's,
  with standard errors from each interval's half-width, has a 95% interval below zero. R-b treats
  the two years as independent although both score the same series, and the page says so. Thresholds
  follow the cost of porting. C1 and C3 have threshold zero, because retraining replaces the
  correction step and adds a scheduled job. C6 has threshold -0.05 and no replication rule,
  because the blend adds three XGBoost models and its four members are not experiment 49's five.
  C4 and C5 follow experiment 49 and have no replication rule: a component stays when the interval
  lies wholly above zero (removing it hurts), and can go when the interval lies wholly inside plus
  or minus 0.03. For C4 the column also gets a pooled read over the winters of 2022-23 (fold C),
  2023-24 (fold B), and experiment 49's 2025-26, descriptive. Verdict, from fold B: supported when
  C1 and C3 both replicate by R-a; partly supported when one replicates by R-a or both hold by R-b;
  refuted when the C1 or the C3 interval lies wholly above zero; inconclusive otherwise. Fold C
  (below) cannot change the verdict. Also pre-registered and descriptive: C1 without the series
  that carried experiment 48's gain (D8, D12, and D5 where they have data in fold B), the 0.95
  exceedance rate of every arm, the Christmas window 2023-12-18 to 2024-01-04 as a number without
  an interval, and every other read (lead bands, months, series), all at 95% and labelled so.
benchmark: >-
  The stack (frozen N1 fitted on the control member up to 2023-06-30, half-life 2 years, 500
  rounds, with the 14-day correction and the spread factor), `manual_heuristic`, and
  `climatology`, each scored on the same runs. The climatology and every effective capacity that
  feeds a feature are fitted on data before the fold.
row_set_id: control_fold_b_daily_v1
metric: >-
  Quantile score in points of capacity (1 point is 1% of the series' effective capacity); MAE of
  the median and the exceedance rate of the 0.95 quantile are reported and decide nothing. Every
  number is a control-only number and is never compared with a 51-member score.
code: ml-experiments/00081-r11-control-member-second-validation-year-demand/
production_status: not-for-production
adopted_in: none
---

# Experiment 81: do the weekly-retraining findings hold on July-to-March forecast runs the project has not scored?

**Status: planned. No code has run.** This page is the plan, committed before any fit. The
sections below are the question, the design, what the folds cannot say, and the sizing. An Opus plan
review changed the block length, the replication rules, the series count, the run counts, the
leakage guards, and the GPU estimate, and dropped the solar stage.

## The question

**Two findings that decide the next model release rest on one validation year.** Experiment 48
found that refitting the Primary substation demand model every week scored 0.20 points of capacity
below the best setup then known, but the gap crosses zero with a t-interval over the 12 months and
shrinks to -0.08 without three of the 16 substations. Experiment 49 found that dropping the
Christmas column and lengthening the half-life changed the score by under 0.02 points, on one
winter. We ask whether the same contrasts have the same sign on forecast runs from 2023-07 to
2024-03, for which the project holds only the ENS control member. Those nine months hold no April to
June, so they are a second set of runs, not a second year.

## Design

**Rows and folds.** The ENS control-member backfill has 1,104 daily runs from 2021-03-21 to
2024-03-31; the runs of 2022-09-10 and 2024-01-03 are missing. Fold B, the one the maintainer asked
for, is the 274 daily 00:00 UTC runs of 2023-07-01 to 2024-03-31 (row set
`control_fold_b_daily_v1`). The ECMWF ensemble moved from 18 km to 9 km native resolution on
2023-06-27, so every run in fold B is 9 km while 2.3 years of its training history is 18 km. Fold C
is the 273 runs of 2022-07-01 to 2023-03-31 (row set `control_fold_c_daily_v1`), all 18 km, with 15
months of training history. Fold C is a second replication on a different period. It has its own
confounds: a short history that starves the frozen model and so favours retraining, the fall in
demand in the winter of 2022-23, and training rows from the 2021 lockdown months. It therefore
cannot show that a fold B gain is adaptation to the 9 km product, and the page draws no such
inference. A direct descriptive check of the product change fits a frozen model to 2023-03-31 and
reports its score relative to `manual_heuristic`, which does not depend on the weather product, for
April to June 2023 (18 km) and July to September 2023 (9 km).

**Series.** Of the 16 Primaries, three lack power through the backfill period: the first valid
reading of series 7 is 2023-08-23, of series 13 is 2024-12-11, and of series 12 is 2025-02-06. The
experiment scores the other 13, so every frozen model has the same history. D12 of experiment 48
is read against this list when the labels are matched; a series with no data in fold B drops out of
the "without D8, D12, and D5" read.

**Training rule and leakage guards.** Every model trains on its own history: the ENS control
member's rows from 2021-03-21. A refit at time T uses only rows whose valid time precedes T less
the 3-hour telemetry delay, never rows cut by init time, and an assert checks the largest valid
time in every training set. The correction reads no forecast issued before the fold's first run,
because the frozen model's earlier forecasts are in-sample. The built frames assert that every
power lag is null where the lag is no longer than the lead. Climatology and effective capacities are
fitted before the fold: nothing from 2024-04-01 onwards enters a fold B fit, and nothing from
2023-04-01 onwards enters a fold C fit. Frames are built as in experiment 48 (production's
half-hourly upsampling, each series' H3 cells, power lags from the observed power), from one member
in place of 51. The backfill has no precipitation type, which the final feature list does not use.

**Arms (Primary, 13 series, N1 recipe of experiment 34, spread factor 1.15).** Fold B runs all of
these. Fold C drops the blend and the frozen arm without correction.

- **Stack:** frozen N1 (fitted to 2023-06-30 in fold B, to 2022-06-30 in fold C; half-life 2
  years; 500 rounds) with the 14-day correction built from the forecasts as issued.
- **Weekly** and **monthly:** full refits, half-life 0.5 years, no correction.
- **Rolling:** experiment 49's P-roll, a refit every month on the latest 12 months, 150 rounds,
  learning rate 0.15, half-life 0.5 years; **rolling without the Christmas column;** and **rolling
  with a 2-year half-life.**
- **Frozen without correction** (descriptive), `manual_heuristic`, and `climatology`.
- **Blend:** weekly, monthly, rolling, and stack under a monthly weight vector. It has four members,
  not the five of experiment 49, so C6 asks whether blending a corrected frozen model with retrained
  models helps, not whether experiment 49's five-model result repeats.

**Spread factor.** The factor 1.15 was chosen on forecasts pooled over 51 members. One member's
quantiles may be narrower, so every arm reports its 0.95 exceedance rate, and a descriptive check
re-chooses the factor on fold B's first three months and reports its effect on later months.

**Arms left out.** Substation fingerprints are not run, because experiment 49 bounded their effect
below 0.05 points and the porting threshold is 0.10. The solar and analogue contrasts of
experiments 64, 78, and 74 are not run: fold B holds 92 days of daylight April-to-September rows,
about 3 four-week blocks, and the sunshine anomaly those experiments used is an ensemble mean that
one member cannot give. Testing them needs a summer fold with a recalibrated control-only anomaly,
which this plan does not provide.

## What the control-only folds cannot say

- **Absolute skill.** One member cannot give an ensemble mean, spread, CRPS, coverage, or a
  51-member quantile score. Every number is labelled "control-only fold", and only the sign and size
  of paired contrasts are set beside experiment 48's and 49's.
- **Whether a gain is retraining or adaptation to the 9 km product.** Neither fold separates them.
- **Independence of the periods.** All periods score the same 13 substations.
- **April to June.** Neither fold holds those months.
- **Fine intervals.** Each fold has only about 10 four-week blocks, so the intervals resolve less
  than experiment 48's.
- **The product production serves.** Our own ENS archive starts on 2024-04-01 with a different
  product.
- **Generators.** Only Primaries are scored, and no NGED generator series appears.

## What the write-up must also answer

- **More history or the full ensemble?** The finished page ends with one bullet saying whether more
  ENS history, or the full 51-member ensemble (spread and CRPS features), would plausibly change any
  conclusion tested here, and which of the two. This plan proposes no fetch.

## Sizing, by the five triggers

- **Stored:** predictions under `data/ml-experiments/`, private; nothing in production.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, the fold choice and the replication rule, which is why
  this plan had an Opus review.
- **Callers not nameable without searching:** no; the scripts copy experiments 34, 48, and 49.

The experiment is complex because it builds new frames from a new archive and its plan could change
the verdict on weekly retraining. Review set: one Opus plan review (done), one Sonnet code review,
one Sonnet write-up check, one Sonnet prose review.

**Earlier advice followed.** Test weekly retraining and the Christmas column on a further period
(48, 49); read contrasts at a Bonferroni level (34); leave the spread factor at 1.15 (49), with the
check above. Departures: weekly blocks become 4-week blocks, because runs overlap by 14.6 days.

**Plan review (Opus), triaged.** Accepted: 4-week blocks; the block bootstrap governing and the
t-interval labelling a result fragile; R-b needing fold B's own interval; one rule per contrast;
C6 without a replication rule; C2 descriptive and a refuted branch for C3; the Christmas window as
a number only, with a pooled read over winters; fold C reported as a replication with its
confounds and the adaptation inference removed; the reversed adaptation prediction; the direct
9 km check; cutting training by valid time; the correction warm-up rule; lag asserts; fitting
climatology and capacities before the fold; the two missing runs; the 13-series set (checked
against the power data); the exceedance rate and spread check; scoping to July to March; dropping
the solar stage; and a GPU estimate scaled from experiment 48. Rejected: none.

**Resources.** Scaling experiment 48's timings (weekly refits 2.4 GPU hours for 52 refits) gives
about 4 to 5 GPU hours for fold B and about 2.5 for fold C, run as parallel workers inside slot E (6
cores), one GPU fit at a time. No GPU fit starts before the ML manager says go.
