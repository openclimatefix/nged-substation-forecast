---
id: "00222"
slug: dp-network-before-filter-order
title: "Differentiable physics: does a network that corrects the convex model before the Kalman filter beat one that corrects after it?"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction, calibration-and-uncertainty]
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00222-dp-network-before-filter-order/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (which inputs the network reads, whether it trains on the filter's residual stream or
  on forecast rows, how the network's correction reaches the filter), and this plan pre-registers
  one. Callers: this folder's scripts, which copy the filter, the T-all fit and the stream code of
  experiments 136 and 201, the corrections of experiment 155 and the network of experiment 201, and
  call the reviewed `studies.*` modules without changing them. A new method, so a Sonnet plan
  review and a Sonnet code review; no Opus.
hypothesis: >-
  Experiments 200 and 201 correct arm D after the Kalman filter. Their gain is mostly a systematic
  error of the convex model at some temperatures and loads, not a recent-error effect, and that
  systematic error currently sits in the filter's level state. A network that reads only the
  convex model's own output and inputs, and corrects the convex model before the filter, leaves the
  filter a cleaner residual stream and so beats the same kind of network applied after the filter.
prediction: >-
  Written before any score is read. (1) PRE minus POST at day-ahead leads (15.5 to 39 hours) is at
  most -0.02 points, with the 99% month-resampled paired interval wholly below zero. (2) The mean
  absolute value of the filter's level is smaller under PRE than under R0 on the extreme-temperature
  days (the lowest and highest temperature deciles). (3) PRE-T, with the filter noise setting
  retuned on the tuning months from a grid of 3, scores within 0.01 points of PRE. If prediction 1
  fails, the order does not matter and the simpler serving design (network after the filter, filter
  state unchanged) stands.
decision_rule: >-
  Known-answer gate first (G1: PRE's network recovers at least 50% of an injected 3% under-forecast
  on days above the 95th-percentile temperature; G2: PRE's filter recovers an injected 3-point
  8-week level step within 5 percentage points of R0's recovery). If G1 or G2 fails, the verdict is
  `inconclusive` and the contrasts are not evidence. Three planned contrasts, each at the 99%
  level of the month-resampled paired bootstrap (12 months of the scored year, 20,000 resamples):
  C1 PRE minus POST at D+1; C2 PRE minus R0 at D+1 and at all leads 0.5 to 168 hours; C3 PRE-T
  minus PRE at D+1. `supported` when C1 is at most -0.02 with its interval wholly below zero, and
  the Checks list shows no sign flip when one substation is left out. `partly supported` when C1's
  interval is wholly below zero but its point estimate is above -0.02. `not supported` otherwise (the
  order does not matter). The threshold -0.02 is set by cost: a network at the GAM stage replaces
  the network after the filter, so the cost is the same one trained component, and a smaller gain
  does not justify redesigning the serving order.
benchmark: >-
  Arm D of experiment 155 (R0) and experiment 201's arm 2 network (POST), on the scored year
row_set_id: >-
  experiment 155 arm D rows of the scored year (2025-07-02 to 2026-06-29), 6 substations (D5, D9,
  D10, D11, D14, D16; five decision substations, D9 beside them), outside switching events
metric: >-
  Quantile score over the 11 quantiles (twice the mean pinball loss) in points of effective
  capacity, at D+1 (15.5 to 39 hours) and over leads 0.5 to 168 hours, with coverage and pinball
  loss at q01, q05, q95, and q99
date: 2026-10-22
code: ml-experiments/00222-dp-network-before-filter-order/
---

**Planned.** The plan is the front matter above. The write-up replaces this body once the scores exist.
