---
id: '00193'
slug: dp-ens-two-number-solar-summary
part: CAL-6
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00193-dp-ens-two-number-solar-summary/
  only. Serving path: not touched (the experiment measures what a cheaper weather-layer summary
  would keep). Degradation rule: not touched. Designs: two defensible designs for the two-number
  band (a Gaussian-points mixture, or a closed-form shift of arm A's offsets); the Gaussian-points
  mixture is registered because experiment 187's placebo already tested it. Callers: this folder's
  scripts, copied from experiment 187's, which read experiment 184's stored arms and member
  responses. No mean forecast is refitted.
hypothesis: >-
  Experiment 187 found that passing each of the 51 ensemble members through the solar term lowers
  the daylight quantile score of the full best-guess stack by 2.23%, and that a Gaussian with each
  row's own mean and spread of the 51 responses recovers 95% of that gain. Two numbers per
  half-hour (the mean and the standard deviation of the members' solar responses) therefore carry
  nearly all of the gain, so a serving path can store two columns instead of running 51 passes.
prediction: >-
  Q2 keeps at least 80% of arm M's daylight gain (a score change of -1.78% or lower in the scored
  year, against M's -2.23%) with a 95% month-block interval wholly below zero. Prediction that can
  fail: the placebo recovered 95% with the factors of M refitted, so Q2 should land near -2.1%. Q3
  (adding the 10th and 90th percentiles) adds under 0.2 points of that percentage, because the
  skew was worth about 0.3 points in experiment 187 and the 10th and 90th percentiles capture only
  part of it. The rescale-only placebo moves the score by about -0.2%, the stricter shuffle
  recovers under 45% of Q2's gain, and the night placebo does not lower the night score.
decision_rule: >-
  Reference: experiment 187's arm A (the full best-guess stack). Comparator: arm M. C1 is Q2 minus
  arm A in the quantile score (twice the mean pinball loss over 11 quantiles) over daylight
  half-hours (sun 5 degrees or more above the horizon), leads 0.5 to 168 hours, outside the
  hand-filtered switching events, at D5, D10, D11, D14, and D16, with a paired month-resampled
  bootstrap (20,000 resamples) at the 95% level (one decisive contrast). `supported` when, in the
  scored year, C1 is at or under 0.8 times M's change (-1.78%) with an interval wholly under zero,
  and the same two conditions hold in the replicate months with 0.8 times M's replicate change
  (-1.41%); `partly supported` when they hold in the scored year only; `small gain` when the
  interval is wholly under zero but the point estimate is above 0.8 times M's change; `not
  supported` otherwise, and the page then says so first. The retained share of M's gain (C1
  divided by M's change, with a month-block interval) is reported beside the verdict. The 80%
  threshold is set by cost: a two-column summary replaces a 51-member solar response in the
  weather layer and the serving path, and a summary that gave up more than a fifth of the gain
  would not justify a second code path beside the member pass. Cuts reported, none deciding: lead
  band, clear-sky band, substation (D5 apart), the sunniest third and the gap to BG2, absolute
  quantile score, mean absolute error of the mean forecast, q05-to-q95 coverage, tails.
benchmark: >-
  Arm A and arm M (experiment 187; 2.7521 points over all rows in the scored year), BG2 for the
  sunny-daylight gap (experiment 178), climatology and manual_heuristic for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score in points of effective capacity over daylight and night half-hours, as a
  percentage change from arm A, with absolute scores, q05-to-q95 coverage, and tail pinball loss
code: ml-experiments/00193-dp-ens-two-number-solar-summary/
production_status: candidate
adopted_in: none
---

# Experiment 193: two numbers per half-hour in place of 51 ensemble members in the solar term

**Planned. This page is the plan; the write-up follows the run.** Experiment
[187](../00187-dp-ens-members-in-stack/result.md) lowered the daylight forecast score by 2.2% by
passing each of the 51 weather-ensemble members through the solar part of the demand model, and
its placebo suggested that two numbers per half-hour carry nearly all of that gain. This
experiment tests that on the same rows.

- **Arms.** A (reference) and M (51 members, experiment 187), plus Q2 (mean and standard deviation
  of the members' solar responses, with a fitted offset factor and a fitted spread factor per
  lead band, fitted on the tuning months), Q3 (Q2 with the 10th and 90th percentiles added, as a
  two-piece skewed shape), and Q5 (a small monotone neural scale function; optional and not run
  unless Q2 fails).
- **Controls.** Rescale-only (every response zero), the shuffle within substation, month, and hour
  band, the shuffle within lead day and clear-sky band as well, and a night placebo that gives
  night rows a daylight row's two numbers.
- **Reuse.** Experiment 184's stored member responses and arm A forecasts; experiment 187's
  scoring and placebo code, copied.
