---
id: "00346"
slug: wind-research-default-1-offset-table-refit-cadence
title: >-
  Does refitting the offset table of Wind farms research default 1 every month, on recent runs only,
  improve the wind forecast?
theme: generator-forecasting
also_relevant_to:
  - retraining-and-training-history
  - calibration-and-uncertainty
status: planned
kind: change
date: 2026-10-24
size: >-
  medium. Stored: member mixtures of the daily ensemble runs after 2025-06-16 and the scored
  quantiles of each arm, under data/ml-experiments/00346-wind-research-default-1-offset-table-refit-cadence/
  only; no forecasts go through write_predictions unless an arm is adopted. Serving path: not
  touched (an adopted arm needs a monthly job that refits the table; the boosted model is
  unchanged). Degradation rule: not touched. Designs: the arms are fixed by the maintainer's
  brief. Callers: this folder's scripts, which import experiments 342, 340, 336, 326, 303, 285,
  259, and 230. One Sonnet code review and one Sonnet prose review, no Opus.
hypothesis: >-
  Experiment 342 found that the default's offset table, fitted once on the 442 daily runs of
  2024-04-01 to 2025-06-16, leaves the upper tail too low in the scored year (observed power above
  q99 in 1.8% of half-hours against 1% intended) although it fits the hindcast, and blamed drift
  between the hindcast years and the scored year. The ensemble table holds daily runs up to
  2026-10-24, so the table can be refitted each month on runs that had already finished. A table
  refitted monthly on a trailing 6-month window (x2) scores lower than the table fitted once.
prediction: >-
  Written before any score. x2 lowers the kept-rows quantile score by 0.05 to 0.15 points. x1
  (expanding window) gains less than x2, because it carries the stale years. x3 (trailing 3 months)
  is noisier than x2 and gains less. The 0.99 pinball loss moves by under 0.05 points. The gain is
  larger in the second half of the scored year than in the first. The out-of-time months
  (2026-07 to 2026-10) show a gain at least as large as the scored year.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order x2, x1, x3; at most
  one arm is adopted, with every interval at level 1 - 0.01/3 (99.67%, month-block bootstrap over
  the scored months and the 3 seeds). The arms are a changed setting (cadence and window), so the
  minimum gain is 0 points: condition 1 needs the interval of (arm minus default) on the headline
  quantile score of the kept rows at 1 to 10 days to lie wholly below zero. Tolerances: 0.02 points
  on the headline, 0.05 on the 0.01 and 0.99 pinball losses. The shuffled-date placebo must keep
  under half of x2's gain, the leakage check must show every table using valid times before the
  scored run's month start, the known-answer test must show the trailing windows tracking an
  injected drift, and the default must reproduce experiment 259's score within 0.01. The
  out-of-time months are descriptive and decide nothing.
benchmark: >-
  The default: Wind farms research default 1 (the offset table fitted once on the hindcast;
  experiment 303's P1 refitted in experiment 336, 11.352 points on the kept rows). Reference rows,
  descriptive: a quarterly refit on an expanding window, and (from experiment 342) the default
  with a q99 shift fitted on the scored year itself.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Out-of-time check: the Tuesday runs of 2026-07-07 to 2026-10-20.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerances 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball
  losses.
code: ml-experiments/00346-wind-research-default-1-offset-table-refit-cadence/
---

**Planned, not run.** Table cells are per farm, lead day, and quantile level, as in experiment 259.
Each arm refits the table on the first day of every month, from the spread-adjusted quantiles of
daily runs whose valid times precede that day by the telemetry delay (3.5 hours): x1 on an
expanding window from 2024-04, x2 on the trailing 6 months, x3 on the trailing 3 months. The
reference row refits quarterly on an expanding window. The ERA5-trained XGBoost model and the
spread factor are unchanged. Controls: a shuffled-date placebo, a leakage printout, and a
known-answer drift test.
