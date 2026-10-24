---
id: '00294'
slug: regime-aware-model-at-d8-engine-state-term
title: >-
  Does a regime-aware model beat a single model at a Primary with a persistent generator regime?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: done
size: >-
  medium. Stored: private prediction files under
  data/ml-experiments/00294-regime-aware-model-at-d8-engine-state-term/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible (which engine state
  the forecast sees, and which history the single model trains on), so the plan fixes both below.
  Callers: this folder's scripts, which copy experiment 191's GAM-K engine and experiment 291's
  engine detector and call reviewed `studies.*` modules unchanged. Asset graph: not touched.
hypothesis: >-
  D8's metered power excludes a gas engine's output (experiment 194) while the engine, which runs
  in spells (experiment 291), takes net load below what the weather and calendar predict. A
  single model trained across those spells has no way to know which half-hours the engine ran, so
  its forecast and its bands carry the engine as noise. A model that keeps the calendar, weather,
  and solar shapes shared across the whole history but adds one regime-specific term, an offset
  per half-hour the engine runs (non-negative, taking power off net load), forecasts D8 better
  than the single model. The offset is estimated from the data, and its size is not assumed.
prediction: >-
  Written before any forecast is scored. (P1) The regime-aware model (arm 3) lowers D8's quantile
  score, our stand-in for the continuous ranked probability score (CRPS), by at least 5% against
  the single model on the whole history (arm 1), and lowers D8's lower-tail
  threshold-weighted score (twCRPS) by at least 5%. (P2) Arm 3 scores within 3% of the band-widening
  comparator (arm 4, experiment 264's post hoc baseline) on D8's quantile score. (P3) The same
  engine term does not raise the quantile score at D5, D10, or D11, which have no engine, by more
  than 0.5%. (P4) The fitted offset per running engine lies between 1 and 3 MW in at least 80% of
  the monthly refits that have a running spell in their window (the engine's recorded unit was
  1.97 MW). Each can fail. P1 fails if the band-widening comparator alone does as well and the
  engine term adds nothing beyond it.
decision_rule: >-
  Fixed before any score is read. Rows: D8, D5, D10, and D11 in the scored year (valid times from
  2025-07-02 to 2026-06-30), leads 0.5 to 168 hours, outside the hand-filtered events, on the
  stored runs of experiment 191 (seed 0). The quantile score is twice the mean pinball loss over
  the 11 quantiles, in points of effective capacity. The twCRPS is the same score applied to
  min(forecast quantile, t) and min(reading, t) with t the 10th percentile of D8's readings in the
  scored year (a lower-tail weight, computed from the readings once and used for every arm). The
  deciding arm 3 uses the engine state of the last settled day (see Plan), because the state
  at the target half-hour is built from the very reading being forecast. Intervals resample the
  12 calendar months (20,000 draws) at 98.75%, a Bonferroni correction over the four planned
  contrasts. C1: arm 3 minus arm 1, quantile score, relative to arm 1. C2: arm 3 minus arm 4,
  relative to arm 4. C3: arm 3 minus arm 1 at each control. C4: C1 on the twCRPS. `supported`: C1
  and C4 are each at most -5% with the interval wholly below zero, C2's upper end is below +3%,
  and C3's upper end is at most +0.5% at all three controls. `partly supported`: C1 holds and one
  of C2 and C3 fails, or C1 holds on the quantile score and not the twCRPS. `not supported`:
  otherwise, which includes arm 4 scoring better than arm 3 by more than 3% with the interval above
  zero. `inconclusive` if the reproduction gate fails (below). The 5% threshold is set by cost: the
  term adds an engine detector and a stored state column to the training loop and to the
  forecast's inputs. The 3% and 0.5% limits are not copied from an earlier experiment's gain:
  3% is under the noise between two sensible band-widening rules in experiment 264 (arm 3 and arm
  2b differed by 2.5%), and 0.5% is experiment 264's no-harm rule. The reproduction gate: arm 2
  (experiment 191's reference, rebuilt here) must reproduce experiment 191's stored D8 quantile score
  of 10.95 to within 0.05.
benchmark: >-
  Arm 1 (the single model), arm 2 (experiment 191's masked reference), and arm 4 (experiment 264's
  stored post hoc baseline "arm_emp" and its arm 2b), on the same keys at D8.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (CRPS stand-in), threshold-weighted quantile score, q05-to-q95 coverage, mean
  absolute error of the mean forecast, tail calibration at q01, q05, q95, and q99, and a Brier
  stand-in for the curtailment proxy, each for all rows, ordinary rows, and engine-running rows, with
  winter separately.
code: ml-experiments/00294-regime-aware-model-at-d8-engine-state-term/
verdict: not supported
best_arm: >-
  None met the pre-registered rule. Arm 3 (the engine state of the last settled day) scores 10.3%
  worse than the single model (interval 8.0% to 12.8% worse). The post hoc arm trained on the
  lagged state scores 2.8% better (interval 5.3% to 0.8% better; quantile score 8.83 against 9.09)
  and 15.5% better on the lower-tail twCRPS (24.5% to 5.5% better).
headline: >-
  A regime-aware forecast gains 24% at D8 when the engine state is read at the target half-hour,
  but that state is built from the reading being forecast. With the state of the last settled day,
  the coefficient fitted on the contemporaneous state is more than 20 times too large and the
  forecast gets 10% worse; the coefficient trained on the day-old state gains 3%.
lessons:
  - >-
    The engine's offset is stable across the 12 monthly refits (3.5 to 3.6 MW per half-hour with one
    or more units running, 1.4 to 1.5 MW per running unit), but the last settled day's state is worth
    only 0.15 MW out of sample (interval -0.06 to 0.32), consistent with a coefficient fitted on the
    contemporaneous state overshooting when the state is a day old (the matched arm removes the
    loss).
  - >-
    Training D8 on its whole history, with its own spread, gives a quantile score 17% lower than
    experiment 191's reference rows, and about the same as widening the bands from past residuals
    (experiment 264).
  - >-
    The break indicator alone lowers D8's quantile score by 2.2% (interval 4.2% to 0.5%) and
    raises it by 0.4% at D10 and 0.5% at D11.
take_home_message: >-
  Telling the forecast when D8's gas engine is running would make it much better when the state is
  read at the half-hour being forecast, but the detector builds that state from the very reading
  being forecast. With only the state of the last settled day, the engine term made the
  forecast 10% worse; trained on that stale state, it gained only 3%, no better than widening the
  bands from past residuals. A forecast of whether the engine runs, or NGED's own record of its
  running hours, would be the data to test next.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    At D8, train an engine-state coefficient on the state a forecast will actually see, not on the
    contemporaneous state, because the last settled day's state is worth 0.15 MW per unit of
    probability and the contemporaneous fit gives 3.6 MW (Results, term out of sample).
  - >-
    Do not score a regime term with the detector's state at the target half-hour as a forecast
    gain: that state is built from the reading being forecast, and it lowers the quantile score
    by 24% (Figure 1, state-at-the-target arm).
  - >-
    Train D8 on its whole history with its own spread, not on experiment 191's rows with donor
    spreads, when comparing band methods: the two differ by 17% in quantile score, with the source
    of the spread confounded with the history (Figure 1, arm 2).
  - >-
    Test a break indicator at Primaries without an engine before using it: it raised the score by
    0.4% at D10 and 0.5% at D11 (Figure 5).
date: 2026-10-23
---

**Telling the forecast when D8's gas engine runs would help a lot, but only when the state is read
at the half-hour being forecast, and that state is built from the reading being forecast.** D8 is
one of the 16 Primary substations of National Grid Electricity Distribution (NGED) in the trial. Its
meter stopped including the output of a gas engine of up to three units of about 2 MW each on
2023-08-15, so we take D8's multi-day drops in power (2 to 19 days, about 2.7 MW) to be the engine
running ([experiment
194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md),
[experiment 291](../00291-d8-drops-engine-spells-load-shift/result.md)). We asked whether a forecast
with an extra term for the half-hours when the engine runs beats a forecast without it. Both
forecasts are the same generalised additive model with a Kalman filter on recent errors, trained on
all of D8's history. The engine's running state is estimated by a detector, not observed. We tested
five ways of giving the term its state. The one usable at forecast time, the state of the last day
whose readings had arrived, made the quantile score 10.3% worse than the forecast without the term.
The quantile score is twice the mean pinball loss over 11 quantiles, our stand-in for the continuous
ranked probability score (CRPS), and smaller is better.

**The term's gain on D8 is partly circular, because the engine states and the drops come from the
same residual.** The states are the probabilities of the detector of experiment
[196](../00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate/result.md), as run in
experiment 291 ([PR 1594, "Experiment 291: are D8's multi-day drops generator-engine
spells?"](https://github.com/openclimatefix/nged-substation-forecast/pull/1594)). Experiment 291
found the drops in the residual that this detector reads, so the states cannot confirm that the
drops are an engine. The detector's filtered probability uses readings only up to each half-hour,
but its inputs are not causal: the expected-power series it reads is centred on a 30-day window, so
a state can carry up to 15 days of later readings. That flatters every arm that uses a state, which
makes arm 3's loss conservative and leaves the post hoc gain at risk.

## Arms and rows

- **Rows.** The 292 daily runs of experiment 191, scored from 2025-07-02 to 2026-06-30, leads 0.5 to
  168 hours, outside the hand-filtered events, seed 0: 97,012 rows at D8, 84,137 at D5, 96,370 at
  D10, 97,046 at D11. Every arm is refitted monthly, as in experiment 191, so no contrast depends on
  weekly refits.
- **Arm 1, the single-model arm.** Experiment 191's generalised additive model with a Kalman filter
  (GAM-K), no 24-month window, D8's whole history, its own tuning months for the spread.
- **Arm 2.** Experiment 191's reference rows (D8 from 2025-04-01, donor spread). Arm 2 reproduces
  experiment 191's stored forecasts exactly (largest difference 0.0). Arm 2 differs from arm 1 in
  both the history and the source of the spread.
- **Arm 3, deciding.** Arm 1 plus a break indicator (1 from 2023-08-15) and the filtered probability
  of one or more running units, with a non-negative coefficient, read from the same half-hour of the
  last settled day at forecast time.
- **Other arms.** Break indicator only. Engine state at the target half-hour from the smoothed
  posterior, which uses the reading (leaky ceiling). Post hoc: expected number of units (arm
  3-units), and arm 3 trained on the filtered probability lagged 1 to 7 days (matched). Arm 3-window
  is arm 3's engine term on arm 2's rows.
- **Arm 4.** Experiment 264's stored `arm_emp` (the median plus the past 365 days' residual
  quantiles) and `arm2b`.
- **Lab shortcuts.** The detector's parameters were fitted on 2023-09 to 2026-01 and its expected
  power comes from a model fitted with hindsight. The filtered probability uses only readings up to
  the half-hour, and the last-day state ignores the 3.5-hour delivery delay of readings, so about 15%
  of target half-hours read a state built from readings not yet delivered.

## Figures

![Quantile score and lower-tail score of each arm against the single model](figure_1_relative_score.svg)

*Figure 1. Only the arm that sees the state at the target half-hour (24% better) gains more than the
post hoc arm trained on the lagged state (3% better). The last-day state, arm 3, is 10% worse.*

![The fitted engine offset of each monthly refit](figure_2_engine_offset.svg)

*Figure 2. The offset per half-hour with the engine on is 3.5 to 3.6 MW at every refit; per running
unit it is 1.4 to 1.5 MW, against the recorded unit of 1.97 MW. Trained on a day-old state, the
offset is 0.9 to 1.0 MW.*

![The mean error of each arm on engine-running half-hours and on other half-hours](figure_3_shortfall_by_state.svg)

*Figure 3. The single model reads 1.1 MW below its forecast while the engine runs and 0.8 MW above
it otherwise. Only the target-state arm removes both gaps.*

![Three weeks of D8 readings against forecasts](figure_4_example_weeks.svg)

*Figure 4. In December 2025 the forecast that sees the engine state at the target follows the
readings down during the shaded spells; the others barely move.*

![The engine term against the single model at D5, D10, and D11](figure_5_controls.svg)

*Figure 5. At the three Primaries without an engine, the arms move the quantile score by -0.9% to
+2.0%.*

## Results

- **Quantile score at D8 (points; smaller is better; one point is 0.1 MW).** Arm 1 9.09; arm 2
  10.95; arm 3 10.02; break indicator only 8.89; matched 8.83; units 9.96; target state 6.87; arm 4
  `arm_emp` 8.88, `arm2b` 9.14.
- **C1, arm 3 against arm 1:** +10.3% (98.75% interval +8.0% to +12.8%). Leaving out any one month
  keeps the sign (+9.8% to +10.8%).
- **C2, arm 3 against arm 4:** +12.9% (+1.2% to +25.0%).
- **C4, lower-tail twCRPS, arm 3 against arm 1:** +0.6% (-10.9% to +14.0%).
- **C3, controls, arm 3 against arm 1:** D5 -0.9% (-2.5% to +0.1%); D10 +0.4% (-0.6% to +1.5%); D11
  +2.0% (-0.2% to +6.1%). The upper ends at D10 and D11 exceed the +0.5% limit, so C3 fails.
- **Post hoc matched arm against arm 1:** quantile score -2.8% (-5.3% to -0.8%), twCRPS -15.5%
  (-24.5% to -5.5%), and against arm 4 -0.5% (-11.1% to +9.5%). The arm is post hoc, so its result
  does not enter the decision.
- **Break indicator only, against arm 1:** -2.2% (-4.2% to -0.5%).
- **Prediction P4, offset of 1 to 3 MW:** P4 fails as registered, because the registered column is
  the probability of one or more units running, and its offset is 3.5 to 3.6 MW (0 of 12 refits in
  range). The post hoc column of expected running units gives 1.4 to 1.5 MW per unit, which is in
  range.
- **The term's value out of sample.** The slope of arm 1's error on the state of the last settled
  day, fitted with an intercept, is 0.15 MW per unit of probability (95% interval -0.06 to 0.32),
  against 3.6 MW fitted, so it cannot be told from zero. On the state at the target it is 2.20 MW
  (1.72 to 2.65), close to one engine unit but circular. The last-day flag (probability above 0.5)
  is set in 65% of engine-running half-hours and 37% of other half-hours.
- **Arm 1 against arm 2.** Arm 1's quantile score is 17% lower than arm 2's, but arm 1's spread
  comes from D8's own tuning months and arm 2's from donor Primaries.

| Quantile | Arm 1 below / pinball | Arm 3 | Matched | Arm 4 `arm_emp` |
|---|---|---|---|---|
| q01 | 0.029 / 0.605 | 0.019 / 0.602 | 0.021 / 0.548 | 0.036 / 0.806 |
| q05 | 0.065 / 2.255 | 0.062 / 2.591 | 0.061 / 2.216 | 0.076 / 2.278 |
| q95 | 0.893 / 2.197 | 0.956 / 2.405 | 0.913 / 2.126 | 0.923 / 2.443 |
| q99 | 0.946 / 0.751 | 0.985 / 0.587 | 0.961 / 0.630 | 0.968 / 0.827 |

*Share of readings below each quantile (the target is the level) and the pinball loss in points, at
D8, all rows.*

## Checks

- **Median minus mean forecast,** per arm: 0.00 points for every arm built here (printed in the
  report); the -0.96 points for arm 4 `arm_emp` comes from experiment 264's stored forecasts.
- **Error of the mean forecast beside the quantile score.** Arm 3's mean absolute error is 17.38
  against arm 1's 16.29 (worse, as is its quantile score); the matched arm's is 15.78 (better, as is
  its score). The two metrics agree.
- **Audit triggers.** One trigger fired: arm 3 contains the single-model arm and the indicator-only
  arm, and both score better than arm 3. We tested an explanation, that the coefficient was trained
  on the state at the half-hour and applied to a day-old state, and the matched arm trained on the
  lagged state removes the loss. No independent audit of the method was run. The other six triggers
  (a failed control, a result against its prediction without a tested explanation, a disagreement
  between the quantile score and the error of the mean forecast, a leave-one-site-out flip, an
  interval width against a sister experiment, and an unreplicated result that a later design depends
  on) did not fire.
- **Row set.** All arms share the same 97,012 D8 rows (keys joined across arms).

## What the next experiment should answer

**The next experiment should answer whether the engine's state can be known before the half-hour it
affects.** The state of the last settled day is worth 0.15 MW out of sample, which cannot be told
from zero. Options that this experiment did not test are a forecast of the state from the gas price,
a longer or lead-dependent lag, and NGED's own record of the engine's running hours. Any of them
should be scored with a state that no reading of the target half-hour enters.
