---
id: '00192'
slug: dp-ens-members-temperature-cold-nights
part: CAL-6
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00192-dp-ens-members-temperature-cold-nights/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design (the
  linear-in-weights member pass of experiment 187, applied to the temperature inputs and to night
  rows). Callers: this folder's scripts, copied from experiment 187's.
hypothesis: >-
  Experiment 187 found that passing the 51 ECMWF ensemble members through the solar term of the demand
  model lowers the daylight quantile score by 2.2%. The demand response to temperature is also
  nonlinear (the heating hinge and the lighting ramps), and the ensemble mean smooths away the cold
  tail. Passing each member's temperature through the temperature terms therefore lowers the quantile
  score on cold winter nights.
prediction: >-
  Written before any score is read. Arm T (members through the temperature terms only) or arm ST
  (members through the solar and temperature terms) lowers the cold-night quantile score (ensemble-mean
  temperature under 2 degrees C, local 17:00 to 08:00) by at least 1.5% against arm A, with a 97.5%
  month-block interval wholly below zero, and moves the cold-night q95 coverage (0.896 for arm A) to
  within 0.02 of 0.90. The gain is larger than experiment 187's daylight gain of 2.2%. Non-cold rows
  worsen by no more than 0.3%. The prediction that could fail is the size: experiment 184's arm F
  (every weather input from the member) scored the same as arm M in daylight, which says temperature
  spread added nothing in daylight, and lighting depends on solar elevation, not on weather, so the
  member temperature acts only through the heating hinge and the PV cell-temperature derating.
decision_rule: >-
  Reference: arm A, experiment 187's full best-guess stack. Two planned contrasts (T minus A and ST
  minus A, in the quantile score over cold-night rows of D5, D10, D11, D14, and D16, leads 0.5 to 168
  hours, outside the hand-filtered switching events), a paired month-resampled bootstrap with 20,000
  resamples at the 97.5% level. `supported` when the better of T and ST is at or under -1.5% with an
  interval wholly under zero in the scored year, cold-night q95 coverage is within 0.02 of 0.90, and
  all non-cold rows are within +0.3%. `partly supported` when the same holds in the scored year but
  the replicate months show a gain with an interval that includes zero. `small gain` when the interval
  is wholly under zero and the point estimate is above -1.5%. `not supported` otherwise.
  `inconclusive` when the known-answer gate fails. The 1.5% threshold is set by cost: the member pass
  adds a 51-member temperature response and fitted band factors to the weather layer. Cuts reported,
  none deciding: coldest 5% of night rows, all rows, lead band, winter evenings, and each substation.
  Controls: a shuffled-members placebo (members from another day of the same substation, month, hour
  band, lead day, and temperature band), a rescale-only placebo (zero responses, factors refitted), a
  two-number summary (Gaussian with each row's own mean and spread of the responses), a fake-ensemble
  known-answer gate (temperature of the members scaled around the truth, response monotone in the
  member's temperature), and a comparison at equal coverage.
benchmark: >-
  Arm A (experiment 187), arm M (experiment 187's solar-only members) for the daylight gain, and the
  rescale-only placebo for equal coverage
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score in points of effective capacity, as a percentage change from arm A, with absolute
  scores, MAE, q05 and q95 coverage, and interval width
code: ml-experiments/00192-dp-ens-members-temperature-cold-nights/
production_status: not-for-production
adopted_in: none
---

# Experiment 192: ensemble members through the temperature terms, scored on cold nights

Planned. Written before any code runs; see the front matter for the hypothesis, prediction, and
decision rule.
