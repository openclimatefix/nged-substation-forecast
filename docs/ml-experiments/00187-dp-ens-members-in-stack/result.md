---
id: '00187'
slug: dp-ens-members-in-stack
part: CAL-5
title: >-
  Does passing the 51 ensemble members through the solar term still lower the daylight score once a
  corrected known-answer gate is applied, and how much of the gain needs the day's own members?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00187-dp-ens-members-in-stack/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible design for the
  gate (monotone in the member's own unit PV shape) and one for the stack (experiment 184's arm A is
  already the full stack). Callers: this folder's scripts, copied from experiment 184's, which read
  experiment 184's stored arms and member deltas. No mean forecast is refitted.
hypothesis: >-
  Experiment 184 found that running each of the 51 ensemble members through the solar term of the
  convex demand model lowers the daylight quantile score by 2.2%, but its verdict was capped at
  `inconclusive` because a monotone gate failed on rows where the unit PV shape itself falls as
  irradiance rises. With a gate that matches that shape, the same arm passes, and its gain lies
  where the best-guess stack leaves the largest sunny-daylight error, at the substation D5.
prediction: >-
  Written before any new score is read, although experiment 184's scores for arm M are already
  known, so the scored contrasts are a re-derivation and not a blind test. The prediction that can
  fail: the corrected gate G3' passes on every daylight row of the fake ensemble; arm M lowers the
  daylight quantile score by at least 1% with a 97.5% month-block interval below zero, in the scored
  year and in the replicate months; night changes by under 0.5%; and the daylight q05-to-q95
  coverage lies within 0.02 of 0.90. The share of arm M's gain that survives the stricter placebo
  (members shuffled within substation, month, hour band, lead day, and clear-sky band) is between
  25% and 45%, as the audit of experiment 184 measured (-0.78% of -2.23%, 35%).
decision_rule: >-
  Reference: experiment 184's arm A (experiment 164's fix O on experiment 160's table on experiment
  155's arm D mean). Arm D is the Kalman-filtered convex model plus the winter night offset of
  experiment 137 and the 14-day error profile of recipe R (experiment 115), so arm A already is the
  full best-guess stack that the brief lists; no part is added here. C1 is arm M minus arm A in the
  quantile score (twice the mean pinball loss over 11 quantiles) over daylight half-hours (sun 5
  degrees or more above the horizon), leads 0.5 to 168 hours, outside the hand-filtered switching
  events, at D5, D10, D11, D14, and D16, with a paired month-resampled bootstrap (20,000 resamples)
  at the 95% level (one planned contrast). `supported` when gate G3' passes on every fake-ensemble
  daylight row, C1 is at or under -1% with an interval wholly under zero in both periods, night is
  within +0.5%, and the daylight q05-to-q95 coverage is within 0.02 of 0.90. `partly supported` when
  the same holds in the scored year only. `small gain` when the interval is wholly under zero but
  the point estimate is above -1%. `not supported` otherwise. `inconclusive` when G3' fails. The 1%
  threshold is set by cost: the member pass adds a 51-member solar response and four fitted factors
  to the weather layer, which a gain the interval cannot separate from 1% does not justify. G3' is
  justified from the shape alone. The member's response is the fitted solar coefficient times the
  difference between the member's unit PV shape and the ensemble mean's unit PV shape, and
  `SunGeometry.pv_shape_and_ghi` splits irradiance into direct and diffuse parts before tilting it,
  so the shape can fall as irradiance rises at low sun (experiment 184 saw this at elevations of 14
  to 16 degrees). The response must therefore be monotone in the member's unit PV shape (non-
  increasing, because solar output lowers demand) on every fake-ensemble daylight row, and no
  statement is made about irradiance. The old irradiance-monotone check is reported beside it as a
  diagnostic. Placebo shares (the percentage of C1 that each placebo recovers) are reported for
  experiment 184's placebo MP, for the stricter shuffle, for a Gaussian with each row's own mean and
  spread of the responses, and for the same Gaussian reflected about its mean. Cuts reported, none
  deciding: lead band, clear-sky band, substation, the sunniest third, and the gap to BG2.
benchmark: >-
  Arm A (2.7519 points over all rows in the scored year), BG2 for the sunny-daylight gap (experiment
  178), climatology and manual_heuristic for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score in points of effective capacity over daylight and night half-hours, as a
  percentage change from arm A, with absolute scores, q05-to-q95 coverage, and tail pinball loss
code: ml-experiments/00187-dp-ens-members-in-stack/
production_status: not-for-production
adopted_in: none
---

# Experiment 187: ensemble members through the solar term, inside the best-guess stack

Planned. The write-up replaces this page when the scores exist.

- **Why arm A is already the stack.** Experiment 184's arm A is experiment 164's fix O on experiment
  160's table on experiment 155's arm D, and arm D is the Kalman-filtered model plus the winter night
  offset and recipe R. The brief's step 3 (adding those parts) therefore changes nothing, and this
  experiment re-scores experiment 184's stored arms with fresh scripts.
- **Scripts.** `e00187_gates.py` (G3'), `e00187_score.py` (copied), `e00187_placebos.py`,
  `e00187_report.py`.
- **Optional neural arm.** Not run. The Gaussian summary of the member responses already scores
  within 0.1 points of arm M in experiment 184's audit, so a learned scale has little room to gain.
