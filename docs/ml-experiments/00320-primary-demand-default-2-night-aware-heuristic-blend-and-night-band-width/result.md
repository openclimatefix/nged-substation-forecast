---
id: '00320'
slug: primary-demand-default-2-night-aware-heuristic-blend-and-night-band-width
title: >-
  Can a blend with the manual heuristic that leaves cold winter nights alone, or a night band-width
  correction, improve Primary demand research default 2 at 0.5 to 240 hours?
theme: production-reference-and-best-guess
also_relevant_to:
- calibration-and-uncertainty
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00320-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the arms are fixed by
  the question. Callers: this folder's scripts, which import the folders of experiments 184, 317,
  and 319 and call reviewed `studies.*` modules without changing them. No change under `packages/`.
hypothesis: >-
  Primary demand research default 2 has two weak spots. (1) At 168 to 240 hours a blend with the
  manual heuristic lowers the score by 4.5%, but a single weight per lead band and season harms
  cold winter nights, where the convex chain is the better forecast. Default plus B1 (the blend
  switched off on half-hours forecast to be cold winter nights) or default plus B2 (a chain weight
  that is a logistic function of the ensemble-mean temperature, per lead band and day or night)
  keeps the gain without the harm. (2) The prediction bands are too wide at night, most clearly at
  one Primary (D10, whose band heads gained 0.144 points, 88% of it at night). Default plus N (a
  width multiplier on the quantile offsets from the median, fitted per Primary and per night or day
  block from the previous 90 days) removes that excess width at every Primary.
prediction: >-
  Written before any score is read; each can fail. (PB1) B1 keeps at least 70% of the blend's gain
  at 168 to 240 hours (a gain of at least 0.045 points against default 2 pooled), brings cold
  winter nights to a harm of +0.03 points or less, passes every other no-harm bound, and stays at
  least 8% better than the manual heuristic. If B1 keeps under 50% of the gain, the blend's gain
  was concentrated on cold nights and the heuristic is not a useful partner outside them. (PB2) B2
  scores at least 0.01 points better than B1 at 168 to 240 hours; otherwise the hard rule is kept
  for its simplicity. (PN) N gives D10 at least 0.08 points, moves the other five Primaries
  together by under 0.02 points, improves the pooled six-Primary score by at least 0.013 points,
  and keeps the q01 and q99 pinball on cold winter nights within 0.05 points. If N gives D10 under
  0.04 points, the band heads' gain at D10 depends on the weather state and not on the hour of the
  day. (PX) B1 or B2 and N are additive within 25% of the sum; the combination is run only if both
  pass.
decision_rule: >-
  Family rule of research-default.md, three pre-registered arms read at the 1 - 0.01/3 level. B1
  and N are changed settings (minimum gain 0 points); B2 is a new fitted step (minimum gain 0.02
  points at 0.5 to 240 hours pooled, or 0.02 at 168 to 240 hours for a lead-limited arm). An arm is
  adopted when its interval at its scored window lies wholly below the default's score, its point
  estimate reaches the minimum gain, no no-harm bound is broken at 0.5 to 240 hours or at 168 to
  240 hours (pooled six-Primary 0.02 points, q01 and q99 pinball 0.05, cold winter nights 0.05,
  sunny daylight 0.05, twCRPS 2% relative), and leaving one Primary out does not flip the sign. At
  most one arm is adopted. The reproduction gate (default 2 equals experiment 319's default plus x1
  to within 0.01 points) comes first and stops the run if it fails.
benchmark: >-
  Primary demand research default 2 (experiment 319's default plus x1: the convex chain with the
  two-number solar summary to 240 hours; experiment 251's A+H+S below 168 hours). The manual
  heuristic and climatology as stored by experiment 87, and BG2 to 168 hours, are references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, rows outside the hand-filtered events, origin 09:00 UTC. Coverage and pinball loss at
  q01, q05, q95, q99 by lead band.
code: ml-experiments/00320-primary-demand-default-2-night-aware-heuristic-blend-and-night-band-width/
---

## Plan

- **Arms.** Default is Primary demand research default 2. B1 applies experiment 319's blend (chain
  weight per 24-hour lead band and season, refitted each month on the trailing 365 completed days)
  at 168 to 240 hours, but sets the chain weight to 1 on half-hours in the forecast cold winter
  night group. That group is experiment 317's: November to March, after dark, local 18:00 to
  07:00, with the ensemble-mean 24-hour temperature at the origin below 2 C (the realised
  temperature of the scored group is not used by the rule). B2 replaces the table of weights by
  `w = sigmoid(a + b * (T - 10) / 5)` per lead band and per daylight or night, fitted on the same
  trailing rows by pinball loss. N multiplies the offsets of all 11 quantiles from the median by a
  factor in 0.6 to 1.2, fitted per Primary and per night (21:00 to 07:30 UTC) or day block on the
  rows of the previous 90 days whose reading had arrived at the month's first origin, and applied
  at 0.5 to 240 hours. The weights and factors are fitted on the convex chain's rows of the history
  (the default's own rows exist only for the scored year) and applied to the default's quantiles.
- **References.** The 319 blend with no cold-night rule (to reproduce its harm), the manual
  heuristic, climatology, BG2 to 168 hours.
- **Replicate.** The tuning-period fit applied to the second year at 168 to 240 hours, as in
  experiment 319.
- **Exploratory.** The optimal chain weight by 24-hour lead band from 24 to 240 hours (figure only).
- **Audit triggers** are read after scoring; see the Checks section of the finished page.
