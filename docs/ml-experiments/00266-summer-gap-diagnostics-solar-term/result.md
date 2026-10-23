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
  The best-guess demand stack (experiment 251's arm 1) trails BG2, the production-style XGBoost
  forecast, in April to June 2026 mainly because of its explicit solar term (the subtracted
  estimate of unmetered rooftop solar output), and not because of the neural-network head, the
  two-number summary of the ensemble's solar response, the Kalman filter, or the heating and
  lighting terms. The same defect explains the loss at D5, an anonymised Primary substation, on
  sunny half-hours.
prediction: >-
  Written before any stored forecast is cut by weather, hour, or substation. Each line can fail.
  (P0, gate) Arm 1 minus BG2 reproduces experiment 251's +0.162, +0.220, and +0.054 for April, May,
  and June within 0.005, and every arm used has exactly experiment 251's 561,207 row keys. (P1)
  The mean gap per row on daylight half-hours in April to June is at least twice the mean gap per
  row on night half-hours. (P2) The explicit solar term is worth less in April to June than in July
  to September: the quantile score gain of the older GAM-K model with the pinned capacity over the
  same model with no solar term (experiment 139, a proxy for the stack) is at least 0.1 points
  smaller in April to June. (P3) The pinned capacity is too low in spring: on clear midday
  half-hours at 12 to 36 hours ahead, the capacity correction implied by arm D's residual is at
  least +5% of the pinned capacity in April to June at three or more of the five Primaries that
  have a pin, and larger than in November to February. (P4) The ensemble irradiance is not the
  cause: the gap per row on daylight half-hours where the ensemble-mean and realised clear-sky
  indices agree (within 0.1) is at least 60% of the gap per row where they disagree (by more than
  0.2). (P5) Removing the neural-network head or the two-number summary changes the April to June
  score by under 0.05 points. (P6) The clear-sky midday residual is a level bias: the morning and
  afternoon mean residuals (before and after 12:00 UTC) have an asymmetry
  |m - a| / ((|m| + |a|) / 2) under 0.3. (P7) At D5 on sunny half-hours, arm 1 is worse than BG2 by
  more than 1.0 points (experiment 251: +1.298), and arm 2c, which adds the 60% export clip and
  leaves out the head at D5, by more than 0.3 (experiment 251: +0.540).
decision_rule: >-
  A component is implicated only if the lower bound of its 95% interval is at or above its
  threshold, and cleared if the upper bound is below it; otherwise it is undecided. Intervals
  resample whole ISO weeks of forecast initialisation time, the same blocks for every contrast, and
  April to June holds about 13 such weeks, so intervals are wide and the page reports that count.
  Thresholds, in points of quantile score (twice the mean pinball loss, in points of effective
  capacity): (Solar capacity) P3 holds and the clear-sky gap in April to June is at least +0.15.
  (Solar shape) P6 fails, with capacity cleared. (Ensemble irradiance) P4 fails: the gap per row on
  disagreeing half-hours exceeds the gap per row on agreeing ones by at least 0.15, and the
  agreeing-rows gap is under +0.05. (Neural-network head) arm 1 minus A+S (the stack without the
  head) is at least +0.05 in April to June. (Two-number summary) arm 1 minus A+H (the stack without
  the summary) is at least +0.05. (Kalman filter) the older GAM-K model minus the same model
  without the filter (experiment 111's gamk and gam0) is at least +0.05 in April to June.
  (Heating and lighting) night half-hours hold a gap per row of at least +0.15. Arm D (the convex
  model with the Kalman filter, with no head and no summary) minus BG2 in April to June tells
  whether the gap exists before the head and summary are added: a head or summary can hide a defect
  in arm D, so a component that single removal clears while arm D minus BG2 is at least +0.15 is
  reported as masked, not cleared. When several components are implicated, they are ranked by the
  gap per row that their own cut or contrast accounts for, and the page states that the shares
  overlap and do not add. When no component is implicated, the page says the diagnostics could not
  locate the cause and names the undecided components. The output is a ranked list of causes, each
  with the single cheapest fix experiment it would justify. The ERA5-driven filter arm of
  experiment 111 is reported as descriptive only, because it changes the filter's input and not the
  solar driver. The 0.05 and 0.15 point thresholds are the smallest gaps that a paired contrast on
  these rows resolves at the intervals of experiment 251.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2) on the 561,207 rows of experiments 228 and 251 (six substations,
  2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside hand-filtered load
  shifts). The older GAM-K arms of experiments 139 and 111 are scored on the same rows after an
  exact key-equality gate; an arm that lacks rows is reported on the common rows only, with the
  row count.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity,
  meaning percentage points of a substation's rated demand), signed residual of the median in
  points, and residual-implied capacity correction in points and in percent of the pinned
  capacity.
code: ml-experiments/00266-summer-gap-diagnostics-solar-term/
---

# Experiment 266: which part of the best-guess stack makes it worse than BG2 in April to June?

**Planned.** This page holds the plan only; results are added when the diagnostics have run.

[Experiment 251](../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) found
that the best-guess demand stack beats BG2 over the year, but only through November to March. In
April to June the stack is 0.05 to 0.22 points worse, and at D5 on sunny half-hours it is 1.30
points worse. The maintainer suspects that unmetered rooftop solar is modelled badly. This
experiment runs six diagnostics on forecasts that earlier experiments already stored, so nothing is
refitted.

Terms. **Arm 1** (the stack) is the convex demand model with a Kalman filter (arm D), plus a small
neural network that corrects the median (the neural-network head), plus a mean-and-standard-deviation
summary of the 51 ensemble weather members' solar response (the two-number summary). **A+S** is
arm 1 without the head, **A+H** is arm 1 without the summary, and **arm D** has neither.
**GAM-K** is the older generalised additive model with a Kalman filter of experiments 111 and 139,
and **gam0** is the same model without the filter. **Primaries** are primary substations, which feed
homes and businesses. The **pin** is the solar capacity held fixed for each fit, and the **clear-sky
index** is irradiance divided by clear-sky irradiance.

## Plan

- **Diagnostic 1, error map.** Quantile score of arm 1 minus BG2 by month, by realised clear-sky
  index band (CAMS irradiance over clear-sky irradiance, from [experiment
  178](../00178-dp-sunny-daylight-gap-diagnosis/result.md)'s rows), by ensemble-mean clear-sky index
  band, and by hour of day, at the six substations. Bands apply to daylight half-hours only (clear-sky
  irradiance above 20 W/m²); night half-hours form their own cut.
- **Diagnostic 2, solar term on and off.** The GAM-K arms of [experiment
  139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md) (no solar term, pinned
  capacity, as-of capacity, hindsight capacity) minus BG2 by month. These arms are a proxy for the
  stack, because the stack with its solar term removed was never stored. The result feeds the solar
  capacity cause through P2.
- **Diagnostic 3, capacity bias.** The capacity held by each monthly GAM-K fit against the capacity
  that arm D's residual implies: on clear midday half-hours at 12 to 36 hours ahead, regress the
  signed residual on the unit solar shape, per substation and month. The implied correction is
  measured after the Kalman filter and so is not independent of the fit. The independent reference
  is the recovery factor of the capacity estimator in [experiment
  231](../00231-pv-capacity-audit-where-recovery-goes/result.md) and [experiment
  241](../00241-pv-capacity-calibrated-interval/result.md), which found 0.83 of an injected farm
  recovered.
- **Diagnostic 4, weather input.** Split the daylight gap by whether the ensemble-mean and realised
  clear-sky indices agree (within 0.1), disagree (by more than 0.2), or lie between, and report
  [experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)'s
  ERA5-driven filter arm beside the ensemble-driven one as description only. A refit with realised
  irradiance as the solar driver is the cheapest fix if the split cannot decide.
- **Diagnostic 5, residual shape.** Mean signed residual of arm 1 at clear midday by month, before
  and after 12:00 UTC, and at D5 against arm 2c.
- **Diagnostic 6, ablations in April to June.** Arm 1 against A+S, A+H, and arm D, from
  [experiment 220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md)'s stored arms.

## Checks

To be written with the results: median minus mean forecast per substation, the error of the mean
forecast beside each contrast, and which audit triggers fired.
