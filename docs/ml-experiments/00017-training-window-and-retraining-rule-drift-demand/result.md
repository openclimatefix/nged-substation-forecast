---
id: "00017"
slug: training-window-and-retraining-rule-drift-demand
status: done
size: medium
hypothesis: >-
  For the stage 1a XGBoost demand model trained on ERA5 weather, Primary substation demand drifts
  (experiment 13 saw every arm under-forecast its validation year by 1.3 to 3.1 points of capacity),
  so a model trained on a recent window, weighted towards recent rows, retrained monthly, or
  corrected by the error of the last 14 days, has a lower error than the same model trained on the
  whole history and never retrained.
prediction: >-
  Against the expanding window retrained monthly (A), the best recency arm lowers the mean absolute
  error of the median by 0.1 to 0.3 points of capacity, the 12-month window lowers it by less than
  that, the 6-month window raises it (it never sees a full year of weather), a model frozen at its
  first origin raises it by at least 0.5 points, and the 14-day correction lowers it by 0.1 to 0.3
  points.
decision_rule: >-
  Each planned contrast is the arm minus A, as the mean over all scored rows of the per-row absolute
  error of the median in points of capacity, with a 99.44% interval (Bonferroni across the nine
  planned contrasts) from 20,000 resamples of whole validation months. An arm counts as a gain when
  its interval lies wholly below zero. The threshold for porting is zero for C1 to C6 (a different
  window or weight costs nothing in production) and 0.10 points for C8 (a scheduled retrain) and C9
  (a new pipeline step that reads observed power), because each adds a job to the live service. C7
  measures what skipping retraining loses. The hypothesis is supported if at least one of C1 to C6,
  C8, or C9 passes its threshold. It is refuted if every interval lies wholly above or straddles
  zero with an upper bound under 0.05. Otherwise it is inconclusive.
benchmark: >-
  A, the expanding window with equal weights retrained monthly, and a month-by-half-hour
  climatology refitted each month.
row_set_id: era5_rolling_monthly_2022_01_to_2026_06_v1
metric: >-
  mean absolute error of the median, points of capacity (power over effective capacity, times 100)
verdict: supported
best_arm: >-
  Arm K, the monthly-retrained all-history model plus the mean error of its own forecasts over the
  14 days ending 2 days before (6.10 points of capacity, against 7.62 for A). Among arms with no
  access to recent power, the half-life of 0.5 years (6.70)
headline: >-
  C9, the 14-day correction against A: -1.52 points (99.44% interval -1.95 to -1.13). C4, a
  half-life of 0.5 years: -0.92 (-1.22 to -0.64). C2, a 12-month window: -0.84 (-1.19 to -0.53).
lessons: >-
  Demand drifted by several points of capacity in both directions over 2022 to 2026, so any rule
  that follows recent data beats all history, and retraining monthly beats retraining yearly by
  0.5 points. A frozen model loses 3.8 points.
advice:
  - >-
    Weight training rows with a half-life of 0.5 years, or train on the last 12 months, in the stage
    1a demand model for Primary substations, in place of the 2-year half-life (C4, C2, C6).
  - >-
    Retrain monthly rather than yearly, and never leave a demand model frozen (C7, C8).
  - >-
    Test the 14-day error correction on forecast-time weather and a realistic power delay before
    porting it, because its 1.5-point gain is measured on ERA5 weather (C9).
  - >-
    Score a training-window change over several years of rolling origins, because the sign of the
    all-history model's bias reverses between 2025 and 2026 (figure 2).
production_status: candidate
adopted_in: none
---

# Experiment 17: training window, recency, and retraining rule for demand

**Supported: every rule that follows recent data beat all history, and the 14-day correction beat
all of them.** Scores are the mean absolute error of the median in points of capacity, on 16
Primary substations, with monthly forecasts from January 2022 to June 2026 (1.07 million
half-hours). Weather is ERA5, which arrives about 5 days late, so the scores are a ceiling on what
forecast-time weather reaches. The fitting code is in `ml-experiments/00017-*/`; the sliding
windows are 183, 365, and 548 days.

![Planned contrasts against arm A](figure_1.svg)

| Arm | Error (95% interval) | Against A (99.44% interval) |
|---|---|---|
| A: all history, monthly retrain | 7.62 (7.28 to 7.98) | reference |
| W6: 6-month window | 7.32 (6.97 to 7.70) | -0.30 (-0.91 to +0.32) |
| W12: 12-month window | 6.77 (6.48 to 7.11) | -0.84 (-1.19 to -0.53) |
| W18: 18-month window | 6.82 (6.52 to 7.17) | -0.79 (-1.15 to -0.48) |
| H0.5: half-life 0.5 years | 6.70 (6.42 to 7.00) | -0.92 (-1.22 to -0.64) |
| H1: half-life 1 year | 6.91 (6.63 to 7.23) | -0.71 (-0.92 to -0.51) |
| H2: half-life 2 years | 7.16 (6.86 to 7.50) | -0.45 (-0.57 to -0.34) |
| F: never retrained | 11.42 (10.86 to 12.02) | +3.80 (+3.12 to +4.47) |
| Y: retrained yearly | 8.14 (7.76 to 8.54) | +0.52 (+0.30 to +0.81) |
| K: A plus 14-day correction | 6.10 (5.88 to 6.32) | -1.52 (-1.95 to -1.13) |
| Month-by-half-hour climatology | 8.44 (8.02 to 8.88) | not paired |

**The drift reverses sign, and that is why recent rows win.** A's mean error (observed minus
forecast) runs from about -3.8 points in 2022 and 2023 (forecast too high) to +1.8 in the first half
of 2026 (forecast too low). In 2026 the 12-month window and the half-lives overshoot to +3.0 to +3.4,
yet still score 8.3 to 8.6 against A's 9.4. Only the 6-month window and the correction stay near
zero, the correction within 0.2 points in every year (figure 2).

![Mean signed error by calendar year](figure_2.svg)

**Three predictions missed, and one held with a larger size.** The recency and correction gains were
0.5 to 1.5 points, against the 0.1 to 0.3 predicted. The 6-month window lowered the error by 0.3
points where a rise was predicted, though its interval straddles zero. The frozen XGBoost model
raised the error by 3.8 points, which meets the predicted rise of at least 0.5 and exceeds it.

**Failures and limits.** No fit failed. The correction needs observed power 2 days old, so it is
only as good as the live telemetry delay. All scores use ERA5 weather.

## Post hoc contrasts

**The correction stacks with shorter memory.** The correction applied to H0.5 scores 5.82 (95%
interval 5.63 to 6.02), 0.27 points (0.21 to 0.34) below the correction on A. Applied to W12, it
scores 5.84. These contrasts were not pre-registered.
