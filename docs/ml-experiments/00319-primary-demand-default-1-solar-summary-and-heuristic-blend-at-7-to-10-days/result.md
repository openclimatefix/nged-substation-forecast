---
id: '00319'
slug: primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days
title: >-
  Does extending the two-number solar summary, or blending with the manual heuristic, improve
  Primary demand research default 1 at 7 to 10 days?
theme: production-reference-and-best-guess
also_relevant_to:
- calibration-and-uncertainty
- weather-products-and-blending
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00319-*/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one is defensible, because the two arms are
  fixed by the question. Callers: this folder's scripts, which import the folders of experiments
  184, 202, and 317 and call reviewed `studies.*` modules without changing them. No change under
  `packages/`.
hypothesis: >-
  Primary demand research default 1 has two weak spots at 168 to 240 hours: sunny daylight
  half-hours, where it scores 5.2% worse than the manual heuristic, and April to October, where
  it beats climatology by only 4%. Default plus x1 (the two-number solar summary S, the mean and
  standard deviation of the 51 ensemble members' solar responses, extended from 0.5 to 168 hours
  out to 240 hours) lowers the sunny-daylight score at 168 to 240 hours. Default plus x2 (a
  blend of the convex chain's median with the manual heuristic's, with the weight fitted per
  24-hour lead band and season group) lowers the pooled score at 168 to 240 hours, mostly in
  April to October.
prediction: >-
  Written before any score is read; each can fail. (P1, x1) At 168 to 240 hours on sunny daylight
  half-hours, S extended lowers the quantile score by at least 3% against the default and brings
  the convex chain to within 2% of the manual heuristic (from -5.2%); the pooled 168-to-240-hour
  score falls by at least 1%; below 168 hours the arm equals the default. If S gains under 1% on
  sunny daylight, the response of the mean is no worse than the mean of the responses at long
  leads, and the weak spot is the sunshine forecast's lack of skill. (P2, x2) The blend beats the
  convex chain pooled at 168 to 240 hours by at least 2%, and beats the manual heuristic by at
  least 7% with the 95% interval above zero in each 24-hour band, with at least 70% of the gain
  in April to October. If the blend gain is under 1%, the convex chain and the heuristic carry the
  same information at 7 to 10 days, and the heuristic is a floor and not a competitor. (P3) The
  two changes are additive: their combination would score within 1% of the sum of their gains
  (the combination is run only if both arms pass).
decision_rule: >-
  Family rule of research-default.md, two pre-registered arms read at the 1 - 0.01/2 level. x1
  is a changed setting (lead range of S), minimum gain 0 points; x2 is a new fitted step,
  minimum gain 0.02 points of the quantile score at 0.5 to 240 hours pooled. An arm is adopted
  when its interval at 168 to 240 hours lies wholly below the default's score, its point estimate
  reaches the minimum gain, no no-harm bound is broken at 0.5 to 240 hours or at 168 to 240 hours
  (pooled six-Primary 0.02 points, q01 and q99 pinball 0.05, cold winter nights 0.05, sunny
  daylight 0.05, twCRPS 2% relative), and leaving one Primary out does not flip the sign. At most
  one arm is adopted. The reproduction gate comes first and stops the run if it fails. The blend
  changes only rows past 168 hours, so below 168 hours both arms equal the default.
benchmark: >-
  The manual heuristic and climatology as stored by experiment 87, on the rows of experiment 317;
  BG2 beside them to 168 hours for reference only. Primary demand research default 1 is the
  default (the convex chain of experiment 317 from 168 hours, and experiment 251's A+H+S below).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, rows outside the hand-filtered events, origin 09:00 UTC. Coverage and pinball loss at
  q01, q05, q95, q99 by lead band.
code: ml-experiments/00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days/
---

## Plan

Lead range of every figure: 0.5 to 240 hours (NGED's 1 to 10 days). Schedule: monthly refits as
in experiment 202; the default's chain is experiment 317's stored forecasts.

- **x1.** Member-level solar responses are built past 168 hours by running experiment 184's
  member code with the 480-half-hour forecast window of experiment 317. Offset factor `f` and
  spread factor `s` are fitted per 24-hour lead band (168 to 192, 192 to 216, 216 to 240 hours)
  on the trailing 12 months whose runs had completed before each scored month (experiment 202's
  rolling rule), and the daylight bands are rebuilt as in experiment 193.
- **x2.** The heuristic is rebuilt for 2024-07 onward from the feature cache with the recipe of
  experiment 161 (weekly lags 1 to 6 and annual lags 49 to 55 weeks, 3.5-hour telemetry delay),
  and gated against experiment 87's stored heuristic on the scored year. The chain's weight is
  fitted per 24-hour lead band and season group by minimising pinball loss on the same
  trailing-12-month completed-history rule, and shifts all 11 quantiles by the change in the median.
- **Stop rules.** The gate fails, or a member response cannot be built past 168 hours at one of
  the six Primaries.
