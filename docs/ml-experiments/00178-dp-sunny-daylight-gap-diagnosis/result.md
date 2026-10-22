---
id: '00178'
slug: dp-sunny-daylight-gap-diagnosis
title: >-
  Where does the best GAM lose to the XGBoost model of BG2 on sunny daylight half-hours: which hours,
  clear-sky bands, leads, or substations?
theme: differentiable-physics-embedded-generation
also_relevant_to:
- production-reference-and-best-guess
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00178-dp-sunny-daylight-gap-diagnosis/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because the maintainer fixed the four cuts. Callers: this folder's scripts, which read experiment
  164's stored forecasts (the GAM that experiment 169 calls arm A), experiment 87's BG2 forecasts,
  experiment 106's feature cache, and experiment 111's stored ensemble irradiance, and call
  reviewed `studies.*` modules without changing them. No forecast model is fitted.
hypothesis: >-
  On sunny daylight half-hours the best GAM scores worse than the BG2 XGBoost model, and the gap is
  concentrated in one level of one cut (hour of day, clear-sky band of the ensemble irradiance,
  lead band, or substation), which names the cause as a fixed-shape mismatch or an irradiance-input
  problem.
prediction: >-
  Written before any score is read. At least 60% of the summed GAM-minus-BG2 absolute error on sunny
  daylight half-hours sits in one level of one cut. The most likely level is the morning and evening
  shoulder hours (a share of 60% to 70%), because experiments 139 and 157 made capacity accuracy an
  unlikely cause and the fixed PV shape is south-facing. A fail (no level holds 60%) means the gap is
  spread evenly, which points at the irradiance input as a whole.
decision_rule: >-
  Each of the four cuts divides the sunny daylight half-hours into levels (hour of day in 2-hour
  bins, ensemble clear-sky index in four bands, lead in four bands, substation). The gap is the sum
  over rows of the GAM's absolute error minus BG2's absolute error. A level concentrates the gap
  when it holds at least 60% of the total gap while holding fewer than 40% of the rows. The
  hypothesis is `supported` when one level of one cut does, `refuted` when no level of any cut
  does, and `inconclusive` when the total gap's month-block 95% interval includes zero. No threshold
  of effect size is needed because the experiment changes nothing in production.
benchmark: >-
  BG2 (experiment 87, `p_s6_bg2`), same substations, runs, and half-hours as the GAM
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median forecast (q50) in MW, and pinball loss averaged over the 11
  quantiles in MW, on rows outside the hand-filtered switching events
---

# Plan: where does the GAM lose to BG2 on sunny daylight half-hours?

Planned. The write-up replaces this page when the analysis has run.
