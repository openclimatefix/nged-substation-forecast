---
id: "00017"
slug: training-window-and-retraining-rule-drift-demand
status: planned
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
verdict: null
---

# Experiment 17: training window, recency, and retraining rule for demand

**Status: planned. No fit has run.**

## Design

**Setup.** The rows are the 16 Primary substations, scored on ERA5 weather as in experiment 6, with
each validation month 2022-01 to 2026-06 (54 months) forecast by a model trained on rows ending 7
days before the month starts. A series enters at the first month with at least 12 months of
observed history before the purge, so the row set is shared by every arm. Every arm uses experiment
6's Primary inputs (weather block, hour, weekday, COVID indicator, Christmas-window days),
`reg:absoluteerror`, 500 rounds, seed 0, one model per series. ERA5 arrives about 5 days late, so
the scores are a ceiling on what the live service reaches with forecast weather (experiment 6's
caveat). No calendar columns beyond the clock are used, so no model memorises the day of the year.

**Arms.**

- **A:** all history, equal weights, retrained monthly (reference).
- **W6, W12, W18:** the last 6, 12, or 18 months before the purge, retrained monthly (C1 to C3).
- **H0.5, H1, H2:** all history, weights halving every 0.5, 1, or 2 years, retrained monthly (C4 to
  C6; H2 repeats experiment 6's recency result on more years).
- **F:** A fitted once at the series' first origin and never retrained (C7).
- **Y:** A refitted every 12 months from the first origin (C8).
- **K:** A plus the mean signed error of A's own earlier out-of-sample forecasts over the 14 days
  ending 1 day before the valid day, added to the forecast; no correction while fewer than 7 days of
  errors exist (C9).
- **Climatology:** month, half-hour, and weekend median over the same training rows as A.

**Post hoc, labelled as such in the write-up:** the correction K applied to the best window or
recency arm, and the loss by calendar year for each arm.

**Prior advice followed:** weather block, clock, COVID indicator, and windowed Christmas days
(experiment 6); recency with a half-life of 2 years (6, P7), now tested against shorter half-lives.
Departed from: experiment 6 scored one selection and one confirmation year, and this experiment
scores 54 months so the ranking does not rest on one year of drift.

## Size

- **Stores:** only the experiment's own predictions under `data/ml-experiments/`; nothing in the
  pipeline.
- **Production serving path:** not touched; the result decides a later port.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one exists (pooled models, other correction forms), so the
  correction is kept to its simplest form and labelled so.
- **Callers:** the scripts copy experiment 6's fitting code; no other caller exists.

Sized medium: one Sonnet review of the code for leakage (fit ends at the origin's `train_end`, the
correction uses only errors known the day before) and one of the write-up.
