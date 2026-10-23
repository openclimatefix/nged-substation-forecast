---
id: "00261"
slug: wind-low-bias-decomposition
title: >-
  At which stage of the wind forecast does the low bias enter: the weather input, the mixing of the
  51 ensemble members, or the physics core and the farms?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
date: 2026-10-23
size: >-
  medium. Stored: private tables under data/ml-experiments/00261-wind-low-bias-decomposition/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible way to
  attribute the shift (swap the weather input stage by stage, holding the fitted core fixed).
  Callers: this folder's scripts, which copy experiment 259's. One code review, one write-up review,
  and one prose review, each by a fresh Sonnet sub-agent.
hypothesis: >-
  The low bias in the three metered wind farms' forecast (observed power below the physics core's
  median on 34% of rows, not 50%) comes mostly from the ensemble's weather input, not from the core,
  the farms, or the mixing of the 51 members.
prediction: >-
  Written before any run. (1) The core fed ERA5 100 m wind at the valid time puts observed power
  below its median on 45% to 55% of kept rows over the scored year. (2) The ensemble-mean 100 m
  wind speed is lower than ERA5 by 0.3 to 0.8 m/s for ERA5 speeds of 5 to 12 m/s, and the power the
  speed bias implies (core curve slope times speed bias) explains at least 70% of the shift in the
  median between the oracle and the final chain. (3) The member mixing (the mixture's median minus
  the core's output on the ensemble-mean wind) explains under 25% of the shift.
decision_rule: >-
  Descriptive attribution with no planned hypothesis test. The prediction is supported when all of
  (1), (2), and (3) hold. It fails if the oracle share on kept rows is below 45%, which puts the
  cause in the core, the farms, or the idle rows. Steps 3 to 6 of the brief are skipped when the
  oracle share is within 5 points of 50% and the speed bias explains at least 70% of the median
  shift. Shares carry 95% intervals that resample whole months of the scored year; the farms are
  not resampled.
benchmark: >-
  Experiment 259's stored R1 forecasts (physics core fitted to ERA5 before 2024-04-01, 51-member
  mixture, spread factor), on its 53 Tuesday runs of 2025-07 to 2026-06 and its 442 daily hindcast
  runs of 2024-04 to 2025-06.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3); the hindcast rows are descriptive. Headline on the rows kept by experiment 210's mask,
  all rows beside it.
metric: >-
  Share of rows where observed power is below the forecast's median (calibrated: 50%), by stage,
  farm, and lead day; ensemble-mean minus ERA5 100 m wind speed in m/s; median shift in points of
  capacity
production_status: not-for-production
adopted_in: none
code: ml-experiments/00261-wind-low-bias-decomposition/
---

**Planned, not yet run.** Experiment 259 found that observed power was below the physics core's
median on 34% of kept rows instead of 50%, from the first month after the base fits end. This
experiment swaps the weather input one stage at a time, holding experiment 259's fitted core
fixed: ERA5 wind at the valid time, the ensemble-mean wind, then the 51-member mixture. The change in the
share of rows below the median, and in the median itself, at each swap says where the 16-point
shortfall enters.
