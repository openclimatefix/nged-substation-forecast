---
id: '00232'
slug: dp-pv-clip-with-correction-heads
title: >-
  Do the correction heads close D5's sunny-daylight gap, and do they stack with the 60% clip?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
size: >-
  simple. Stored: one private scores file under data/ml-experiments/00232-dp-pv-clip-with-correction-heads/.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible design, a
  rescore of stored forecasts with no fit. Callers: scripts call `studies.bootstrap` and
  `studies.dp_features` unchanged. Asset graph: not touched.
hypothesis: >-
  A correction head learns demand structure, not an export limit, so it leaves D5's sunny-daylight
  gap to BG2 mostly open and adds to the 60% clip of experiment 180.
prediction: >-
  Written before the rescore, in the maintainer's brief. The head alone closes under 30% of D5's
  gap (+0.202 MW). The head with the clip leaves D5 at +0.05 MW or less. The two effects add to
  within 0.02 MW. If the head alone closes more than half, the clip is a special case of what the
  head learns.
decision_rule: >-
  Contrasts on D5's sunny daylight half-hours, all leads to 168 hours, mean absolute error of the
  median forecast: head minus control, clip-and-head minus clip, and the overlap (clip-and-head
  minus clip minus head plus control), month-block bootstrap at the 98.75% level (four planned
  contrasts, with the clip alone as the fourth). Placebos D10 and D16 move by under 0.02 MW. The
  threshold is zero because a head already exists in experiment 201 and the clip is one stored
  shape per substation.
benchmark: >-
  Experiment 180's control (arm D) and its 60% clip, and BG2, on `fold_a_daily_gap_v1`
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of q50 and pinball loss over 11 quantiles, MW, on experiment 178's sunny
  daylight half-hours
code: ml-experiments/00232-dp-pv-clip-with-correction-heads/
---

Planned. The brief of the maintainer fixes the arms: experiment 180's control, its 60% clip, the
stored heads of experiment 201 (arms 1 to 3, three seeds averaged), and the clip with each head.
