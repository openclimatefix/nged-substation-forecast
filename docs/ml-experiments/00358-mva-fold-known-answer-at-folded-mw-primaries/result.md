---
id: "00358"
slug: mva-fold-known-answer-at-folded-mw-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
size: >-
  medium. Stored: predictions and small result files in the private folder only. Production serving
  path: not touched. Degradation rule: not touched. Designs: more than one would test the
  hypothesis (full ensemble engine with monthly refits, or one frozen fit on past weather); the
  frozen fit is chosen because the question is whether the sign is identified, not the lead-time
  skill. Callers: the code reused is experiment 110's `fit_gam_through_meter` and injection helpers,
  whose callers are all in experiment 110.
hypothesis: >-
  When the signed power of a Primary substation is folded to apparent power (its magnitude), the
  convex model fitted through the meter (experiment 110's L0) recovers the sign of the flow and the
  solar capacity, and a model that reads the folded reading as signed power does not.
prediction: >-
  On sunny daylight half-hours with the true flow at least 5 points of capacity from zero, L0's
  balanced accuracy of the flow's sign is 0.85 to 0.95 at injection level 2.0 and 0.75 to 0.85 at
  level 1.0, at 4 of the 5 sites; the capacity L0 fits is within 15% of the capacity fitted to the
  signed series; L0's mean absolute error on the folded reading is lower than the naive model's by
  at least 0.10 points at level 2.0.
decision_rule: >-
  Passes (and step 2 runs) when, at the scored cut, under zero reactive power, balanced accuracy on
  sunny half-hours is at least 0.80 at 4 of 5 sites at both level 1.0 and level 2.0, and the fitted
  capacity is within 15% of the signed fit's at 4 of 5 sites at level 1.0. Fails (and the series
  stops) otherwise. Threshold is the design brief's K1 and K4; there is no cost-based threshold
  because the experiment ports nothing.
benchmark: >-
  The same convex model fitted to the signed injected series (arm S, ceiling), and the same model
  fitted to the folded reading as if it were signed power (arm N)
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16; scored year 2025-07-01 to 2026-06-30, replicate
  2024-07-01 to 2025-06-30; frozen fits on past weather, no ensemble
metric: >-
  Balanced accuracy of the sign of the real-power forecast; ratio of fitted solar capacity to the
  signed fit's; mean absolute error of the folded-reading forecast in points of effective capacity
---

# MVA fold: known-answer test at folded signed Primaries (planned)

This is step 1 of the sequence in the MVA Primaries brief. It folds five Primaries with known signed
power to apparent power, and checks whether the convex model fitted through the meter finds the
direction and the solar capacity again.

## Plan

- **Sites.** D5, D10, D11, D14, and D16, the five signed-MW decision Primaries.
- **Folded readings.** The signed power with a real solar farm's output subtracted at three levels:
  none (natural), level 1.0 (the farm scaled to the site's median absolute midday power), and level
  2.0 (twice that). The reading becomes `sqrt(P² + Q²)`. Q0 sets `Q = 0`. Q1 sets `Q = tan(φ) D` at
  power factor 0.97, with D from the pinned base fit.
- **Arms.** L0: experiment 110's `fit_gam_through_meter` with `qmode="zero"`, capacity fitted. S:
  the same model fitted to the signed series. N: the same model fitted to the reading as if signed.
  One frozen fit per arm at each cut (2024-07-01 and 2025-07-01), predicting the following year on
  past weather. Lab conditions: past CAMS irradiance, no ensemble, no Kalman filter, point forecasts
  only.
- **Sunny half-hours.** April to September, starting 10:00 to 13:30 UTC, with CAMS irradiance of at
  least 500 W/m². Sign accuracy on rows with true `|P|` of at least 5 points is reported first.
- **Checks.** Reverse-flow share in the truth; capacity rise against the injected capacity;
  mean-versus-median is not applicable to point forecasts; leave-one-site-out of the verdict.

## Departures from the design brief

- The ensemble engine, monthly refits, latent filter, and folded quantiles belong to the later arms F
  and A of the design brief and are not tested here.
- K2 and K5 use the mean forecast's error and not quantile scores, because there is no spread model
  in a frozen point fit.
