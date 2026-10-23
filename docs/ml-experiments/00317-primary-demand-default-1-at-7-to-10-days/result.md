---
id: '00317'
slug: primary-demand-default-1-at-7-to-10-days
title: >-
  How far past 7 days does the Primary demand research default still beat the manual heuristic and
  climatology?
theme: production-reference-and-best-guess
also_relevant_to:
- calibration-and-uncertainty
status: planned
kind: other
size: >-
  medium. Stored: private files under
  data/ml-experiments/00317-primary-demand-default-1-at-7-to-10-days/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one is defensible, because the arms are fixed
  by the question. Callers: this folder's scripts, which import the folders of experiments 136, 155,
  160, and 164 and call reviewed `studies.*` modules without changing them. No change under
  `packages/`: the forecast horizon constant of `studies.dp_ens_weather` is set at run time inside
  this folder's scripts.
hypothesis: >-
  The ensemble weather already reaches 10 days, so the convex model of Primary demand research
  default 1 can be driven to 240 hours, and with a spread table fitted per 24-hour lead band its
  quantile score stays better than the manual heuristic and climatology through day 9 and ties
  climatology on day 10.
prediction: >-
  Written before any score is read; each can fail. (P1) The extended default beats the manual
  heuristic and climatology on the quantile score by at least 5% in each 24-hour lead band from 144
  to 216 hours (the 24-hour bands 144 to 168, 168 to 192, and 192 to 216), and is within 2% of
  climatology in the 216 to 240 hour band (day 10). If it beats both by 5% in every band to 240
  hours, the research default serves NGED's whole 1 to 10 day range. If it ties climatology by day 8,
  the useful range is 7 days. (P2) Coverage of the 90% interval and the pinball losses at q01 and
  q99 stay within 3 percentage points and 0.05 points of their 0 to 168 hour values in the 168 to
  216 hour bands, and degrade in the 216 to 240 hour band. (P3) The gain over the heuristic at days 7
  to 10 is largest on cold winter nights and smallest on sunny daylight half-hours. Reproduction
  gate: the extended chain below 168 hours equals the stored forecasts of experiments 136, 155,
  160, and 164 and the stored arm A of experiment 169 to within 0.01 points, or the run stops.
decision_rule: >-
  Audit, so no threshold and no adoption. `supported` if P1 holds with the 12-month block bootstrap
  95% interval of the extended default minus each benchmark wholly below zero in every band to 240
  hours. `partly supported` if the interval is wholly below zero to some lead between 168 and 240
  hours and not beyond; the write-up names that lead. `not supported` if the default ties or loses
  to climatology already in the 168 to 192 hour band. The gate comes first.
benchmark: >-
  Manual heuristic and climatology as stored by experiment 87, which already cover leads to 351
  hours on the same rows; BG2 beside them to 168 hours for reference only.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, rows outside the hand-filtered events, origin 09:00 UTC. Coverage and pinball loss at
  q01, q05, q95, q99 by lead band.
code: ml-experiments/00317-primary-demand-default-1-at-7-to-10-days/
---

# Plan: how far past 7 days does Primary demand research default 1 still beat the benchmarks?

**Status: planned. No score has been read.**

- **Question.** NGED care about forecasts 1 to 10 days ahead, and every demand result so far stops at
  168 hours. This audit extends the convex model of Primary demand research default 1 to 240
  hours and asks at which lead its skill over the manual heuristic and climatology stops.
- **Arm 1 (default extended).** The chain of experiments 136 (convex model with the Kalman filter),
  155 (winter night offset and recipe R), 160 (spread table interpolated between lead-band centres),
  and 164 (fix O, cold-day widening), rebuilt with the forecast window set to 480 half-hours and one
  new spread-table lead band per day (168 to 192, 192 to 216, 216 to 240 hours). Below 168 hours
  the chain must reproduce the stored forecasts (the gate).
- **Planned departure from the brief.** The neural-network median correction (H, experiment 201)
  was trained at day-ahead leads, and the two-number solar summary (S, experiment 202) was fitted
  to leads below 168 hours. Neither is refitted here. Arm 1 below 168 hours is the stored A+H+S
  (2.579 on six Primaries); the same figure and every contrast also report arm A (the chain without H
  and S) over the whole range 0.5 to 240 hours, so the lead curve is one model. The write-up shows
  how much H and S change the score at days 5 to 7, to say what their absence past day 7 costs.
- **Arms 2 and 3.** Manual heuristic and climatology from experiment 87's stored forecasts, which
  hold leads to 351 hours on the same 292 runs.
- **Weather calibration limit.** The temperature and irradiance calibration has four lead bands and
  its last band stands for every lead above 144 hours, so days 8 to 10 use the 144 to 168 hour
  calibration. This is a caveat on the result.
- **Stop rule.** If the stored ensemble lacks a weather variable past 168 hours at any Primary, the
  run reports which and scores the leads available.
- **Compute.** At most 12 cores, one worker per site with `OMP_NUM_THREADS=1`,
  `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` per job. Sonnet only.
