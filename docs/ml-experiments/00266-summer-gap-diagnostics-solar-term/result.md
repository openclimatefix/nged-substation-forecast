---
id: '00266'
slug: summer-gap-diagnostics-solar-term
title: >-
  Which part of the best-guess demand stack makes it worse than BG2 in April to June and on sunny
  half-hours?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
  - differentiable-physics-forecast-engine
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00266-summer-gap-diagnostics-solar-term/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible for the weather-input test (a refit with realised irradiance, or a stratification of
  stored forecasts); this plan pre-registers the stratification and names the refit as the cheapest
  fix if it cannot decide. Callers: this folder's scripts, which read the stored forecasts of
  experiments 111, 139, 178, 220, 228, and 251 and call reviewed `studies.*` modules without
  changing them. No refits, no GPU, no downloads.
hypothesis: >-
  The best-guess stack (experiment 251's arm 1) trails BG2 in April to June 2026 because of its
  explicit solar term, and not because of the neural-network head, the two-number solar summary,
  the Kalman filter, or the heating and lighting terms. The same defect explains the D5 loss on
  sunny half-hours.
prediction: >-
  Written before any stored forecast is cut by weather. Each line can fail. (P0) The gate holds:
  arm 1 minus BG2 on April, May, and June reproduces experiment 251's +0.162, +0.220, and +0.054
  within 0.005. (P1) The April to June gap sits on daylight half-hours with a realised clear-sky
  index above 0.6, not at night: daylight rows hold at least 60% of the summed April to June gap.
  (P2) The explicit solar term is worth less in April to June than in July to September: the
  quantile score gain of the GAM-K arm with the pinned capacity over the arm with no solar term
  (experiment 139) is at least 0.1 points smaller in April to June than in July to September. (P3)
  The pinned capacity is too low in spring: on clear midday half-hours the residual-implied
  capacity correction is at least +5% of the pinned capacity in April to June at three or more of
  the five Primaries that have a pin, and larger than in November to February. (P4) The ensemble
  irradiance is not the cause: the gap on half-hours where the ensemble-mean clear-sky index is
  within 0.1 of the realised one is at least 60% as large as the gap on the other half-hours. (P5)
  Removing the network head or the two-number summary from the stack changes the April to June
  score by under 0.05 points. (P6) The mean clear-sky midday residual is a level bias, not a shape
  bias: the morning and afternoon mean residuals differ by under 30% of their average size.
decision_rule: >-
  Each component is implicated only if its signature below holds with a 95% interval (resampling
  forecast days) that excludes zero, and a component whose signature fails is cleared. (Solar
  capacity) P3 holds at three or more Primaries and the clear-sky gap in April to June is at least
  +0.15 points. (Solar shape) the morning and afternoon mean residuals on clear half-hours differ
  by at least 30% of their average size, with the residual changing sign, and capacity is cleared.
  (Ensemble irradiance) the gap on half-hours where the ensemble and realised clear-sky indices
  disagree by more than 0.2 holds at least 60% of the April to June gap, and the gap on agreeing
  half-hours is under +0.05. (Network head) arm 1 minus the stack without the head (experiment
  220's A+S) is at least +0.05 points in April to June. (Two-number summary) arm 1 minus the stack
  without the summary (A+H) is at least +0.05. (Kalman filter) the April to June gap of GAM-K
  minus the filter-free GAM (experiment 111) is at least +0.05 points. (Heating and lighting) night
  and non-daylight rows hold at least 40% of the summed April to June gap. The output is a ranked
  list of causes, each with the single cheapest fix experiment it would justify. The threshold is
  zero because this experiment ports nothing: it only ranks causes. The 0.05 and 0.15 point
  thresholds are the smallest gaps that would change a card's best-guess line.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2), on the 561,207 rows of experiments 228 and 251 (six substations,
  2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside hand-filtered load
  shifts). Solar-term arms come from experiment 139 and experiment 111 on their own stored rows,
  which share the row set.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  signed residual of the median, residual-implied capacity correction in MW and in percent of the
  pinned capacity, with intervals that resample forecast days.
code: ml-experiments/00266-summer-gap-diagnostics-solar-term/
---

# Experiment 266: which part of the best-guess stack makes it worse than BG2 in April to June?

**Planned.** This page holds the plan only; results are added when the diagnostics have run.

[Experiment 251](../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) found
that the best-guess stack beats BG2 over the year, but only through November to March. In April to
June the stack is 0.05 to 0.22 points worse, and at D5 on sunny half-hours it is 1.30 points worse.
The maintainer suspects that unmetered rooftop solar is modelled badly. This experiment runs six
diagnostics on forecasts that earlier experiments already stored, so nothing is refitted.

## Plan

- **D1, error map.** Quantile score of arm 1 minus BG2 by month, by realised clear-sky index band
  (CAMS irradiance over clear sky, from [experiment
  178](../00178-dp-sunny-daylight-gap-diagnosis/result.md)'s rows), by ensemble-mean clear-sky
  index band, and by hour of day, at the six substations.
- **D2, solar term on and off.** The GAM-K arms of [experiment
  139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md) (no solar term, pinned
  capacity, as-of capacity, hindsight capacity) minus BG2 by month. This tests whether the gap moves
  with the solar term.
- **D3, capacity bias.** The pinned capacity of each monthly GAM-K fit against the capacity the
  residual implies: on clear midday half-hours, regress the signed residual on the unit solar shape,
  per substation and month.
- **D4, weather input.** Split the gap by whether the ensemble-mean and realised clear-sky indices
  agree, and read the ERA5-driven filter arm of [experiment
  111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md) beside the
  ensemble-driven one. A refit with realised irradiance as the solar driver is the cheapest fix if
  this cannot decide.
- **D5, residual shape.** Mean signed residual at clear midday by month, morning against afternoon,
  and at D5 against the 60% export clip.
- **D6, ablations in April to June.** Arm 1 against A+S (no head), A+H (no summary), and arm D (no
  head and no summary), from [experiment
  220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md)'s stored arms.

## Checks

To be written with the results.
