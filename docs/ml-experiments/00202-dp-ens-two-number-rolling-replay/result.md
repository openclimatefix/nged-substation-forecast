---
id: '00202'
slug: dp-ens-two-number-rolling-replay
part: CAL-7
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00202-dp-ens-two-number-rolling-replay/
  only. Serving path: not touched (the experiment measures what the weather layer would keep, and
  changes no production code). Degradation rule: not touched (a missing run is not replayed, see
  the plan). Designs: two defensible designs for the factor fit (an expanding window or a trailing
  12-month window); the trailing window is registered because it is what a monthly production
  refit would use. Callers: this folder's scripts, copied from experiment 193's, which read
  experiment 184's stored arms and member responses. No mean forecast is refitted.
hypothesis: >-
  Experiment 193 found that two numbers per half-hour (the mean and the standard deviation of the 51
  ensemble members' solar responses) lower the daylight quantile score by 2.20% in the scored year.
  Its two fitted factors per lead band were fitted on interleaved months (the odd months of
  2024-07 to 2025-06), some of them later than the replicate months they were applied to, and its
  numbers were held in float64. A replay of the live weather layer, which stores the two numbers
  as float32 columns computed from the one ensemble run published before each origin and refits
  the factors each month on history that has completed before that month, keeps most of the gain.
prediction: >-
  The prediction that can fail: the replayed arm R2 lowers the daylight quantile score of arm A by
  at least 1.5% over the five substations D5, D10, D11, D14, and D16 in the scored year (193's
  fixed-factor Q2: 2.20%), with a 95% month-block interval wholly below zero, and its night score
  is unchanged within 0.5%. My expectation is -1.9% to -2.2%, because the rolling factors differ
  from the fixed ones by a few hundredths in each lead band. The first rolling months (2025-07 and
  2025-08) are the likeliest to fall short, since they fit on 12 months that include only
  one summer.
decision_rule: >-
  Reference: experiment 187's arm A. Comparator: 193's fixed-factor Q2, for the size of the loss
  that the replay causes. C1 is R2 minus arm A in the quantile score (twice the mean pinball loss
  over 11 quantiles) over daylight half-hours (sun 5 degrees or more above the horizon), leads 0.5
  to 168 hours, outside the hand-filtered switching events, pooled over D5, D10, D11, D14, and D16
  in the scored year (292 runs), as a percentage change of arm A's daylight score, with a paired
  month-resampled bootstrap (20,000 resamples) at the 95% level (one decisive contrast). `supported`
  when C1 is at or under -1.5%, its interval is wholly under zero, the night score is within 0.5% of
  arm A's, and each of the five substations' point estimates is under zero; `partly supported`
  when C1 meets the three pooled conditions but a substation's point estimate is above zero;
  `small gain` when the interval is wholly under zero but C1 is above -1.5%; `not supported`
  otherwise, and the page then says so first. The 1.5% threshold is set by cost: the weather layer
  and the demand model must carry a solar-term pass over the 51 members once per run and two
  stored columns, and a gain under 1.5% would not pay for a second code path beside the existing
  daylight bands. Per-substation, lead-band, and cloud-band cuts are reported and none decides
  the verdict beyond the substation condition above. The ensemble-slot cut (the four 6-hourly
  forecast slots) cannot be scored, because experiments 184 to 193 store forecasts at one origin
  per run only (09:00 UTC, after the 00 UTC run); the page says so and reports the gain by valid
  hour of day, a proxy for the age of the run at the valid time, instead.
benchmark: >-
  Arm A and arm M (experiment 187), 193's fixed-factor Q2, climatology and manual_heuristic for
  absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score in points of effective capacity over daylight and night half-hours, as a
  percentage change from arm A, with absolute scores, q05-to-q95 coverage, and tail pinball loss
code: ml-experiments/00202-dp-ens-two-number-rolling-replay/
production_status: not-for-production
adopted_in: none
---

# Experiment 202: the two-number solar summary in a rolling replay of the live weather layer

**Planned. This page holds the pre-registered design; the write-up replaces it when the run
finishes.**

Experiment [193](../00193-dp-ens-two-number-solar-summary/result.md) found that two numbers per
half-hour keep 98% of the daylight gain that running all 51 ensemble members through the solar term
gives. Its fitted factors, however, came from interleaved months and from float64 numbers, so the
experiment does not show that a weather layer that stores two float32 columns and refits monthly
would keep the gain. This experiment replays that weather layer over the scored year of
experiment 193, using at each origin only the ensemble run published before it and only history
whose actuals had completed.

## Plan

- **Rows.** Experiment 193's: `fold_a_daily_gap_v1`, the 292 scored runs, D5, D10, D11, D14, and
  D16 (D9 beside them, in no contrast), leads 0.5 to 168 hours, outside switching events. Arm A and
  arm M are read from experiment 184's stored arrays.
- **Replay of the two columns.** For each run, the mean and standard deviation of the 51 members'
  stored solar responses are computed from that run alone and rounded to float32, as the weather
  layer would store them. A gate asserts each run's ensemble initialisation plus the 9-hour
  publication delay is not after the origin.
- **Rolling factor fit.** Before each scored month, the offset factor and spread factor per lead
  band (experiment 193's grid search over 20,000 sampled rows per band) are refitted on the
  runs of the trailing 12 months whose last scored valid time (origin plus 168 hours) is before the
  month's first origin. Nothing from the month being scored enters the fit.
- **Arms.** A (experiment 187), M (the 51-member arm, 193's fixed factors), Q2 (193's fixed-factor
  two-number arm), R2 (the rolling replay, the decisive arm), R2o (R2 with factors fitted on the
  month being scored; a lower bound on what a perfect refit would give), and a rescale-only
  placebo RR (every response zero, rolling factors).
- **Gates.** G1: the code path reproduces 193's Q2 daylight change (-2.20%) to 0.01 percentage
  points when given 193's fixed factors in float64. G2: rounding the two columns to float32 moves
  the daylight score by under 0.005 points. G3: the availability assertion above. G4: R2 beats RR
  by the interval of the paired contrast; if the placebo recovers more than 20% of R2's gain the
  page flags it. A failed gate makes the verdict `invalid`.
- **Reported cuts.** Per substation, lead band, clear-sky band (overcast, partly cloudy, clear),
  sunniest third, and the first and second halves of the scored year (to show whether the early
  months with less history fall short), plus the gain by valid hour of day. Absolute scores, tail
  coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, and the median-minus-mean checks.
- **Compute.** CPU only, 4 cores, thread caps 1, under 1 core-hour expected (the stored arrays
  are read and the factor grid is small), well inside the 8 core-hours of the brief.

## Reading of the brief

The brief's "at least 1.5% daylight gain on D5, D10, D11, D14, and D16" is read as the pooled
change over those five substations, because experiment 193's single-substation gains at D10 and D11
(1.18% and 0.99%) lie under 1.5%. The per-substation points are reported, and the verdict rule
above requires each to be negative.
