---
id: "00081"
slug: r11-control-member-second-validation-year-demand
status: done
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
verdict: supported
best_arm: >-
  Weekly refitting with no correction: 4.494 points of capacity (control-only fold), against 4.734
  for the stack, 4.533 for monthly refitting, and 5.211 for `manual_heuristic`. The blend of four
  models scored 4.457.
headline: >-
  C1, weekly minus the stack: -0.240 points (99.17% interval -0.360 to -0.145). C3, weekly minus
  monthly: -0.039 (-0.064 to -0.012). C6, blend minus weekly: -0.038 (-0.070 to -0.007). C4 and C5
  are unresolved. Fold C has the same signs with intervals that include zero.
lessons: >-
  Weekly retraining beat the stack on a second set of runs, with a gain as large as in experiment 48
  and a lower bound that excludes zero without the three series that carried experiment 48. The
  Christmas column helps inside the Christmas window in both winters, which the whole-fold contrast
  could not show. The 14-day correction did not help a frozen model here. The gain may include
  adaptation to the 9 km weather product, which the folds cannot separate.
advice:
  - >-
    Port weekly retraining of the Primary demand model and drop the 14-day correction for it, because
    the gain held on a second set of runs at -0.24 (-0.36 to -0.15) and -0.09 (-0.19 to -0.01)
    without D8, D12, and D5 (C1).
  - >-
    Keep the Christmas column, because dropping it raised the error by 1.2 points inside
    2023-12-18 to 2024-01-04 and by 1.8 points inside the same window of the winter before
    (Christmas window; C4 could not resolve it over a whole fold).
  - >-
    Do not rely on the five-model blend result for a four-model blend: the blend of weekly, monthly,
    rolling, and stack scored 0.038 below weekly alone, short of the -0.05 threshold (C6).
  - >-
    Score a weather-product change separately before crediting retraining, because the frozen model
    lost about 3 points of its advantage over `manual_heuristic` when the runs moved from 18 km to
    9 km (9 km check).
production_status: candidate
adopted_in: none
---

# Experiment 81: do the weekly-retraining findings hold on July-to-March forecast runs the project has not scored?

**Take-home: weekly retraining of the Primary demand model beat the stack again, on forecast runs
from a different period, by about as much as before (0.24 points of capacity), so the
case for porting it now rests on two periods instead of one.** The gain is concentrated: without
the three substations that carried experiment 48, it is 0.09 points (95% interval -0.19 to -0.01).
The Christmas column earns its place, and the five-model blend does not carry over as a
four-model blend. The verdict is **supported**.

We scored 13 Primary substations on 274 daily runs from 2023-07 to 2024-03, for which the project
holds only the control member of the ECMWF ensemble (ENS), so every number is a control-only number
and no absolute score is comparable with experiments 48 or 49. The error is the quantile score
in points of capacity (1 point is 1% of a substation's capacity); smaller is better. The stack is a
model fitted once, with a 14-day correction built from its recent errors.

![Planned contrasts, fold B and fold C](figure_1.svg)

![Absolute score of every arm, fold B](figure_2.svg)

![Score by lead band, fold B](figure_3.svg)

![Each series on its own, fold B](figure_4.svg)

![Frozen model against `manual_heuristic` across the 9 km change](figure_5.svg)

- **Arms:** stack; weekly, monthly, and rolling-12-month refits (half-life 0.5 years, no
  correction); rolling without the Christmas column; rolling with a 2-year half-life; a blend of
  weekly, monthly, rolling, and stack. Baselines: `manual_heuristic`, `climatology`.
- **Rows:** fold B is `control_fold_b_daily_v1` (274 runs, 2023-07-01 to 2024-03-31, 9 km). Fold C
  is `control_fold_c_daily_v1` (273 runs, 2022-07-01 to 2023-03-31, 18 km, 15 months of history).
  Seed 0. Intervals use 4-week blocks of forecast start (10 blocks).
- **Absolute scores, fold B:** blend 4.457, weekly 4.494, monthly 4.533, rolling 4.681, stack
  4.734, `manual_heuristic` 5.211, `climatology` 5.700 (95% intervals about plus or minus 0.3).
  Fold C: weekly 4.020, monthly 4.054, rolling 4.073, stack 4.100, `manual_heuristic` 4.845,
  `climatology` 6.099.
- **C1, weekly minus the stack:** fold B -0.240 (99.17% -0.360 to -0.145; 95% month t-interval
  -0.359 to -0.121). Fold C -0.080 (-0.187 to +0.051). Without D8, D12, and D5, fold B -0.093
  (95% -0.189 to -0.014) and fold C -0.084 (-0.170 to +0.018). D8 alone gains 2.05 (95% -2.60 to
  -1.44).
- **C3, weekly minus monthly:** fold B -0.039 (-0.064 to -0.012; the month t-interval, -0.078 to
  -0.000, only just excludes zero). Fold C -0.035 (-0.085 to +0.014).
- **C2, monthly minus the stack (descriptive):** fold B -0.201 (-0.324 to -0.105); fold C -0.045
  (-0.162 to +0.082).
- **C4, rolling without the Christmas column minus with it:** fold B +0.011 (-0.115 to +0.188),
  fold C +0.041 (-0.054 to +0.252), unresolved. Inside 2023-12-18 to 2024-01-04 dropping the column
  raised the score by 1.20 points, against -0.07 outside the window; the winter before, +1.79
  against -0.08.
- **C5, half-life 2 years minus 0.5 years:** fold B -0.019 (-0.060 to +0.040), fold C +0.014
  (-0.027 to +0.058), unresolved.
- **C6, blend minus weekly:** -0.038 (-0.070 to -0.007), a small gain that misses the -0.05 threshold.
  The blend has four members; experiment 49's had five.
- **Correction and spread:** the 14-day correction made the frozen model worse here (+0.091, -0.005
  to +0.187). The 0.95 exceedance rate is 0.052 for weekly with the spread factor and 0.073 without
  it; re-choosing the factor on the first three months (1.10) changed later scores by +0.002.
- **9 km check (descriptive):** the frozen model beat `manual_heuristic` by 9.1% on runs before
  2023-06-27 (18 km) and by 6.3% after (9 km), April to September 2023. The weeks differ in
  season, so this is a hint that part of weekly retraining's fold B gain is adaptation to the 9 km
  product, not a measurement.
- **Caveats:** neither fold can separate retraining from adaptation to a weather product. Effective
  capacities come from the stage 1 pilot's table, built from all data, so they scale scores and
  inputs but were not fitted before each fold. The spread check chose its factor on cases whose
  targets reach 14.6 days into the first scored month. R-b pools two periods that score the same 13
  substations as if independent. The 13 substations exclude three that lack data (series 7, 13, 12).
- **More history or the full ensemble?** More history, not the full ensemble. The folds answer
  retraining and column questions, which depend on how much the weather-to-demand relation drifts,
  so a longer or more varied history would narrow C4 and test the 9 km confound. Ensemble spread and
  CRPS features would not change any conclusion tested here, because none of the six contrasts
  uses them.
- **Next:** a summer fold is the open test for the solar arms of experiments 64, 78, and 74; this
  experiment could not run them.
