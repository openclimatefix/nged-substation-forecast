---
id: '00237'
slug: dp-two-number-solar-summary-with-clip
title: >-
  Does the two-number solar summary keep its gain when the solar term is clipped at D5 and D9?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - weather-products-and-blending
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00237-dp-two-number-solar-summary-with-clip/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible, because the summary can be taken before or after the clip, and we take it after, from
  each member passed through the clipped fit. Callers: scripts copy experiment 180's and 214's fit
  and forecast code and experiment 184's member pass and experiment 193's arms. Asset graph: not
  touched.
hypothesis: >-
  The mean and standard deviation of the 51 ensemble members' solar responses lower the daylight
  quantile score of the demand forecast also when each member is passed through the clipped solar
  term, because cloud uncertainty and the clip both act at midday only partly on the same rows.
prediction: >-
  Written in the maintainer's brief before any run. At D5, the daylight gain from the two numbers
  with the clip is at least 70% of the gain without the clip (about 2%). Below 70% the two effects
  overlap, and the combined arm of the final assembly should be scored per Primary, not pooled.
decision_rule: >-
  Retained share = the two-number arm's daylight gain with the clip divided by its gain without the
  clip, per Primary, months resampled as whole blocks in pairs, 95% interval. `supported` when the
  retained share is at least 70% at D5 and its interval does not fall wholly below 70%; `partly
  supported` when the point estimate is at least 70% and the interval includes less; `not supported`
  otherwise. The threshold is 70% because a clip-aware summary needs a second code path beside the
  member pass; the brief set it.
benchmark: >-
  Arm A (experiment 155's arm D forecast) and arm M (51 members) in each condition, on
  `fold_a_daily_gap_v1`
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  on daylight half-hours, as a percentage change from arm A; mean absolute error of the median on
  the sunniest third of daylight half-hours
code: ml-experiments/00237-dp-two-number-solar-summary-with-clip/
---

Planned. The brief fixes D5 and D9 (the as-of clip of experiment 214), the arms of experiment 193
(A, M, Q2), and the scored year with the replicate months as a second check.
