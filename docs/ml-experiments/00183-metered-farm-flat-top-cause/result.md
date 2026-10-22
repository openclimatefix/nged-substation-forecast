---
id: '00183'
slug: metered-farm-flat-top-cause
title: >-
  Is the flat top of metered solar farms C, D, E, and F a fixed export cap or a panel-plane effect?
theme: solar-disaggregation
also_relevant_to:
- differentiable-physics-embedded-generation
status: planned
size: >-
  simple. Stored: a private table under data/ml-experiments/00183-metered-farm-flat-top-cause/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because no model is fitted. Callers: this folder's script only, which calls reviewed `studies.*`
  modules without changing them.
hypothesis: >-
  The flat top at farms C, D, E, and F is a fixed export cap, so its level does not change with
  season or with the sun's azimuth.
prediction: >-
  At least 3 of the 4 farms have a flat level that varies by under 3% across the months April to
  September.
decision_rule: >-
  For each farm, take the 90th percentile of power over capacity in each month from April to
  September, over half-hours whose clear-sky irradiance on a horizontal plane exceeds 650 W/m². The
  level is seasonally constant when the range of the monthly levels over their mean is under 3%.
  Supported when at least 3 of the 4 farms are seasonally constant. When 3 or more farms vary by more
  than 3%, geometry is the cause, and the per-farm plant fit (follow-up 2) must fit tilt and
  azimuth, not only a clip level. Replication: the same test within each half of the years (even
  and odd calendar years). No score is computed, so no threshold for porting applies.
benchmark: none (data exploration, no fitting)
row_set_id: none
metric: seasonal range of the flat level over its mean, per farm and per half of the years
---

Planned. See the brief `metered_farm_flat_top_brief.md` (Discuss: metered PV proposal 7).
