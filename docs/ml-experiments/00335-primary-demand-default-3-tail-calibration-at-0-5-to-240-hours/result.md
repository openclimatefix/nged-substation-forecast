---
id: '00335'
slug: primary-demand-default-3-tail-calibration-at-0-5-to-240-hours
title: >-
  Can a per-lead-band calibration of the 0.01, 0.05, 0.95, and 0.99 quantiles bring the tails of
  Primary demand research default 3 to nominal at every lead from 0.5 to 240 hours?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
  - curtailment-cost
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00335-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the three arms are fixed
  by the question. Callers: this folder's scripts, which import the folders of experiments 317, 319,
  320, and 322 and call reviewed `studies.*` modules without changing them. No change under
  `packages/`.
hypothesis: >-
  The 0.01, 0.05, 0.95, and 0.99 quantiles of Primary demand research default 3 miss their nominal
  rates past 96 hours (experiments 317 and 320 found 2.1% to 2.9% of observations below q01 and a
  1.8% upper tail at 7 to 10 days). A calibration fitted per 24-hour lead band and per group
  (season, day or night, cold winter night) on the forecast errors of earlier origins brings the
  tails to nominal at every lead from 0.5 to 240 hours without costing quantile score.
prediction: >-
  Written before any score is read; each can fail. (Stage 0) The tail rates are off by at least 0.8
  percentage points at every lead band past 96 hours, the miss is larger in the upper tail on cold
  winter nights and in the lower tail on sunny daylight, and it varies more by Primary than by lead
  past 168 hours. (PC1, C1 per-band per-group widening) The shares below q01 and above q99 fall to
  within 0.3 points of 1% at every lead band, the 5% to 95% band covers 89% to 91%, cold-winter-night
  q95 coverage reaches 0.89 to 0.91 at 168 to 240 hours, q01 and q99 pinball improve by at least
  0.03 points pooled, and the pooled quantile score stays within 0.01 points of the default. (PC2,
  C2 parametric tail) C2 beats C1 on q01 and q99 pinball by at least 0.01 points at 168 to 240
  hours. (PC3, C3 adaptive conformal) Rolling coverage stays within 3 points of nominal in every
  month with a pooled score no worse than C1's, and C3 is adopted only if its worst-month tail miss
  is at least 1 point smaller than C1's.
decision_rule: >-
  Family rule of research-default.md, three pre-registered arms (C1, C2, C3), at most one adopted.
  The arms change a setting on stored forecasts (tier 1), so the minimum gain on the quantile score
  is 0 points; the case for adoption rests on tail pinball and coverage improving with no harm
  elsewhere. An arm is adopted when its tail pinball improves, the headline score is no worse than
  the tolerance, no protected group breaks its bound at 0.5 to 240 hours (pooled 0.02 points, q01
  and q99 pinball 0.05, cold winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative), and
  leaving one Primary out does not flip the sign. The reproduction gate (the default equals
  experiment 322's default 3 within 0.01 points: 2.727 at 0.5 to 240 hours, 3.077 at 168 to 240
  hours) comes first. If stage 0 shows tail rates within 1.2% and 98.8% to 99.2% at every lead band,
  the result is "already calibrated" and the arms are not run.
benchmark: >-
  Primary demand research default 3 (experiment 322's D1). The manual heuristic and climatology
  are references; BG2 is a reference at 0.5 to 168 hours only.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  at 0.5 to 240 hours, six Primaries, scored year 2025-07-02 to 2026-06-30, origin 09:00 UTC, rows
  outside the hand-filtered events. Share of observations below q01, q05 and above q95, q99, and
  coverage of the 5% to 95% band, per 24-hour lead band. Pinball loss at q01, q05, q95, q99. Threshold-weighted CRPS. 12-month block bootstrap intervals.
---

## Plan

Planned arms: default (Primary demand research default 3), C1 (per-lead-band, per-group one-sided
widening of q01, q05, q95, q99, factor bounded to 0.8 to 1.6, shrunk to 1 under 500 half-hours per
cell), C2 (Student-t tail beyond q10 and q90 per lead band and group), and C3 (adaptive conformal,
step 0.005, rolling 60 days). Every calibration is fitted as-of: on completed origins before each
monthly cut, over the previous 12 months. The replicate year (even months of 2024-25) holds forecasts
at 168 to 240 hours only, so cuts early in the scored year have no history below 168 hours and use
a factor of 1 there.

The five plan items, the sizing, and the stop rule are as in the brief. Results follow once run.
