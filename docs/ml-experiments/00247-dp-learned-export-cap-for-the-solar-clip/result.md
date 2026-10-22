---
id: '00247'
slug: dp-learned-export-cap-for-the-solar-clip
title: >-
  Can the export cap of the solar clip be learned from the metered series instead of set to 60%?
theme: differentiable-physics-embedded-generation
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (an upper envelope of clear-sky export, an in-sample profile, or a softer saturating
  clip), so we test three. Callers: scripts copy the fit and forecast code of experiments 180 and
  214. Asset graph: not touched.
hypothesis: >-
  A cap on the solar term learned at each monthly cut from the metered series before that cut
  lowers the sunny-daylight error at D5 almost as much as experiment 180's hand-set 60% clip,
  because the clip only needs the level at which clear-sky midday export stops growing.
prediction: >-
  Written in the maintainer's brief before any run. At D5, at least one learned arm closes at least
  80% of the sunny-daylight gain of the hand-set clip, and no learned arm worsens the sunny-daylight
  mean absolute error at D10 or D16 by more than 0.02 MW.
decision_rule: >-
  Closure = (error of arm A minus error of the learned arm) divided by (error of arm A minus error
  of the 60% clip), on D5's sunniest third of daylight half-hours, months resampled as whole blocks,
  95% interval. `supported` when the closure is at least 80% and its interval does not fall wholly
  below 80%; `partly supported` when the point estimate is at least 80% and the interval includes
  less; `not supported` otherwise. The placebos must not worsen by more than 0.02 MW. The threshold
  is 80% because the learned cap adds a fitting step at every cut, and the brief set it.
benchmark: >-
  Arm A (no clip), the hand-set 60% clip of experiment 180, and experiment 214's as-of ceiling
  selection, on experiment 178's sunny-daylight cut
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median in MW on the sunniest third of daylight half-hours (experiment
  178's cut), at D5, D9, D10, and D16
code: ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip/
verdict: null
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**What already exists.** Experiment 214 picks the clip ceiling as-of from a grid of shares, scored
out of sample, and experiment 110's `profile_clip` fits the ceiling in sample. Neither learns the
cap from the series alone without a grid of candidate shares scored on forecast error.

**Arms.** All arms refit the monthly cuts of experiment 180 with the same chain.

- `p`: the cap that `profile_clip` fits in sample on the 12 months before each cut.
- `e`: the 0.90 quantile of implied solar on clear-sky midday half-hours before the cut, as a share
  of the held capacity.
- `sat`: a softer saturating clip at 60% (smoothing 0.5 per point).

Placebos are D10 and D16. Experiment 4 reads the winner.
