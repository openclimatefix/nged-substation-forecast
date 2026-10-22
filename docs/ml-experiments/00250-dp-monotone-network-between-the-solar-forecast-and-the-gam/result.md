---
id: '00250'
slug: dp-monotone-network-between-the-solar-forecast-and-the-gam
title: >-
  Does a small monotone neural network between the solar forecast and the demand model beat the hard clip?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00250-dp-monotone-network-between-the-solar-forecast-and-the-gam/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (where the
  network sits, how it is kept monotone, which inputs it reads), so the plan fixes one. Callers:
  scripts copy the fit, forecast, and member-pass code of experiments 180, 214, 237, and 247. Asset
  graph: not touched.
hypothesis: >-
  A small neural network that is monotone in the explicit solar forecast, saturates, and reads the
  clear-sky index, the hour, and the clip cap maps the solar forecast to the solar output that the
  metered series shows, and so closes more of D5's sunny-daylight gap to BG2 than the hard clip,
  because it can learn how the saturation depends on the weather.
prediction: >-
  Written in the maintainer's brief before any run. At D5, the network closes at least 50% of the
  sunny-daylight gap to BG2 that the 60% clip leaves, does not worsen D10 or D16 by more than 0.01
  MW against the clip, and its 95% quantile has coverage within 2 points of 95%.
decision_rule: >-
  Closure beyond the clip = (clip's gap to BG2 minus the network's gap to BG2) divided by the clip's
  gap to BG2, on D5's sunniest third of daylight half-hours, months resampled as whole blocks in the
  same draw for both arms, 95% interval. `supported` when the closure is at least 50% with an
  interval not wholly below 50%, both placebo changes are at most 0.01 MW, and the q95 coverage is
  93% to 97%; `partly supported` when the closure point estimate passes and a placebo or coverage
  check fails or the interval includes less; `not supported` otherwise. The threshold is 50% because
  the network adds a trained component to the serving path; the brief set it.
benchmark: >-
  The unclipped fit, the hand-set 60% clip of experiment 180, and the profile-fit cap of experiment
  247, on experiment 178's sunny-daylight cut, against BG2
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median in MW on the sunniest third of daylight half-hours (experiment
  178's cut), as a gap to BG2, at D5 and D9, with D10 and D16 as placebos; q95 coverage
code: ml-experiments/00250-dp-monotone-network-between-the-solar-forecast-and-the-gam/
verdict: null
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**Design.** For each run, the median is the demand model's fit with its solar term replaced: the
demand without solar (the unclipped forecast plus the unclipped solar term) minus the network's
output. The network reads the solar term from the ensemble-mean weather, the clear-sky index, the
hour, and a cap (60% for the clip arms, the profile fit's share for the learned arm). It has weights
that are non-negative along the solar path, at most 2 hidden layers of 16 units, a saturating output,
weight decay, and one fit per monthly cut on forecasts whose 168-hour window ended before the cut.
The band is the arm's own, scaled by one factor per lead band fitted on the same rows.

**Arms.** Hard clip (experiment 180, 60%), learned cap (experiment 247's profile fit), the network on
the unclipped solar term, and the network on the clipped solar term. At D5 and D9 a fifth arm adds
the mean and standard deviation of the 51 members' solar responses from experiment 237. Placebos are
D10 and D16, where the network has no two-number input.
