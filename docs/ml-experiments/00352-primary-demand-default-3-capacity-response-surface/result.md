---
id: "00352"
slug: primary-demand-default-3-capacity-response-surface
title: >-
  How much does the score of Primary demand research default 3 on sunny half-hours depend on the
  solar capacity it is given, with and without the Kalman filter?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
  - production-reference-and-best-guess
status: planned
size: >-
  simple to medium. Stored: private files under data/ml-experiments/00352-*/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible design (scale the pinned capacity,
  filter on or off). Callers: this folder's scripts, which import experiment 349's folder and the
  reviewed `studies.*` modules without changing them. No change under `packages/`.
kind: other
hypothesis: >-
  The score of the first (convex) stage of Primary demand research default 3 on sunny daylight
  half-hours is nearly flat in the solar capacity over 0.8 to 1.2 times the pinned capacity when the
  Kalman filter is on, because the filter absorbs the difference. Experiments 349 and 350 found no
  reliable gain from a different capacity, and a stored-data analysis suggested a flat response.
  Without the filter the response is steeper.
prediction: >-
  Written before any fit. With the filter, the pooled sunny score at 0.5 to 168 hours changes by
  under 0.1 points between any two scales in 0.8 to 1.2 (the range of the five-point curve over
  those three scales), and by 0.10 to 0.25 points at scale 0.6. Without the filter the same range is
  more than 0.3 points. The filter's absorption is therefore the reason capacity precision is not
  worth chasing for demand. D5 improves at scale 0.8 because of its export limit.
decision_rule: >-
  Scale s multiplies the pinned capacity of Primary demand research default 3's first stage at every
  week; the pin at s = 1.0 with the filter is arm A and must reproduce experiment 349's stored arm A
  forecasts exactly (reproduction gate; a failure makes the run `invalid`). The planned contrasts,
  all on pooled sunny daylight half-hours at 0.5 to 168 hours, are C1 and C2 (with the filter, scale
  0.8 and 1.2 each minus 1.0), C3 and C4 (without the filter, the same two), and C5 (no filter minus
  filter at scale 1.0). The hypothesis is `supported` when C1 and C2 have 95% month-resampled
  intervals inside -0.1 to +0.1 and the no-filter range across 0.8 to 1.2 (the largest minus the
  smallest pooled score) exceeds 0.3. It is `partly supported` when exactly one of the two holds.
  It is `refuted` when the with-filter range exceeds 0.3. Otherwise it is `inconclusive`. The
  threshold of 0.1 is about twice experiment 349's 0.047 effect, so a flat curve is one on which a
  capacity change of the size that experiment tested could not be seen. There is no port cost,
  because the experiment changes nothing in production. The no-harm groups are all half-hours,
  cold winter nights, and the tails; they are reported, not decisive.
benchmark: >-
  Primary demand research default 3's first stage at scale 1.0 with the filter (arm A of experiment
  349). The research default's own absolute scores against manual_heuristic and climatology are in
  the research default log.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity) at
  the six Primaries D5, D9, D10, D11, D14, and D16, scored year 2025-07-02 to 2026-06-30, rows
  outside the hand-filtered events, origin 09:00 UTC, at 0.5 to 168 hours and 24 to 168 hours.
  Headline: sunniest third of each Primary's daylight half-hours, as in experiment 349.
code: ml-experiments/00352-primary-demand-default-3-capacity-response-surface/
---

## Plan

- **Question.** How far does the sunny-half-hour score move when the solar capacity moves, and does
  the Kalman filter hide the difference? This tests the explanation in the stored-data analysis of
  why experiments [349](../00349-primary-demand-default-3-exp-269-capacity-in-the-stack/result.md)
  and [350](../00350-primary-demand-capacity-free-irradiance-curve/result.md) found no reliable gain
  from a different capacity.
- **Arms.** Scale s in {0.6, 0.8, 1.0, 1.2, 1.4} times the pin, each with and without the filter
  (10 arms). Without the filter the forecast is the fitted mean plus the spread table, with no
  level or anomaly correction. The spread table is refitted per arm on the tuning months.
- **Code.** Tier 2: the first stage, filter, and forecast code of experiment 349's folder, plus a
  scale on the pinned weekly path. The later stages (neural head, solar summary, blend) are not
  refitted, as in experiment 349.
- **Part 1, stored data.** Experiment 349's sunniest-third contrast (arm B minus arm A) broken down
  by Primary, lead band (0 to 24, 24 to 72, 72 to 168 hours), and month.
- **Scoring.** Month-resampled 95% intervals, `fold_a_daily_gap_v1`, six Primaries. Tail coverage
  and pinball loss at q01, q05, q95, and q99 are reported beside the headline.
