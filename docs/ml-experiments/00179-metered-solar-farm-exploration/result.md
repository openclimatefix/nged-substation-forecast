---
id: '00179'
slug: metered-solar-farm-exploration
title: >-
  Do the 6 metered solar farms show an AC clipping plateau, step drops, or slow drift, and is their
  cleaned series fit to reuse?
theme: solar-disaggregation
also_relevant_to:
- differentiable-physics-embedded-generation
status: planned
size: >-
  simple. Stored: a private cleaned series and private tables under
  data/ml-experiments/00179-metered-solar-farm-exploration/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: one defensible design, because no model is fitted.
  Callers: this folder's scripts only, which call reviewed `studies.*` modules without changing
  them.
hypothesis: >-
  At least 4 of the 6 metered solar farms show AC clipping, meaning output that flattens below the
  level that clear-sky irradiance alone would give, so that the AC limit can be separated from the DC
  capacity.
prediction: >-
  At least 4 of the 6 farms show a clipping plateau on clear days. Some farms also show step drops
  (outages) and slow drift (soiling or degradation).
decision_rule: >-
  A farm shows clipping when, on clear-sky half-hours with the highest clear-sky irradiance, the 99th
  percentile of output over capacity is within 2% of the 95th and the output is flat as irradiance
  rises further (slope of the binned median under 10% of the slope at lower irradiance). Supported
  when at least 4 farms show it. If fewer than 3 show it, follow-up 4 (DC versus AC loss) is
  dropped for those farms. No score is computed, so no threshold for porting applies.
benchmark: none (data exploration, no fitting)
row_set_id: none
metric: counts of farms with clipping, step drops, and drift; data-validation check results
---

# Metered solar farm exploration (planned)

Exploration only: output against clear-sky irradiance for each of the 6 farms (labels A to F),
then the `data-validation` checklist. Anonymised figures and tables only.
