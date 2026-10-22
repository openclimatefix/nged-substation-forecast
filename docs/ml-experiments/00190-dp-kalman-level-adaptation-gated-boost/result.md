---
id: '00190'
slug: dp-kalman-level-adaptation-gated-boost
part: KLA
theme: differentiable-physics-forecast-engine
also_relevant_to:
- lagged-power-and-error-correction
- production-reference-and-best-guess
status: done
size: >-
  medium. Stored: private files under data/ml-experiments/00190-dp-kalman-level-adaptation-gated-boost/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (how the boost is gated, whether it applies to the whole gap between a band's readings),
  and this plan pre-registers one and runs the always-on and shared-level variants as arms. Callers:
  this folder's scripts, which copy the filter and forecast code of experiment 136 and the
  corrections of experiment 155; they call the reviewed `studies.*` modules unchanged. No new
  data.
hypothesis: >-
  The Kalman filter of the convex demand model (GAM-K) has a level state that is a random walk with
  one shared process-noise setting. A single setting must trade speed on a real level shift against
  noise on ordinary days. A gate that multiplies the level's process noise by m only while the last k
  one-step innovations (pooled over the four hour bands) share one sign and their mean exceeds z
  times the day-before innovation spread gives speed when a shift looks real and keeps the quiet
  setting otherwise. The mechanism predicts that the gate lowers the excess loss on days where the
  residual keeps one sign for hours (the 15 smooth-offset days of experiment 175), without harming
  ordinary days.
prediction: >-
  Written before any score is read. P1: on the 15 smooth-offset days, A1 lowers the excess loss by
  at least 20% against A0, with the gain concentrated more than 6 hours after the offset starts.
  P2: on all rows A1 minus A0 lies within -0.03 to +0.01 points, and its 95% upper bound is at most
  +0.01. P3: A1 beats A2 (always-on boost) on all rows by at least 0.01 points. P4: A3 (shared
  level) beats A1 on the smooth-offset days. P5: on the 17 block days A1 helps by under 10%.
decision_rule: >-
  The known-answer gate comes first: A1 must recover at least 50% of the injected offset's excess
  loss for the 16-hour and 36-hour offsets at the 0.10 size, and A1's score on the 40 left-alone days
  must be within +0.005 of A0's at 95%. If the gate fails the verdict is `inconclusive` and no
  contrast is read as evidence about the real days. Then the headline contrast is A1 minus A0 over
  all rows (leads 0.5 to 168 hours, the five decision Primaries D5, D10, D11, D14, and D16, the
  scored year), read at the 95% level of the month-resampled paired bootstrap (20,000 resamples).
  The hypothesis is `supported` when P1 and P2 hold and P3 holds; `partly supported` when P1 and P2
  hold and P3 fails (a plain faster filter is as good); `not supported` when P1 fails with the gate
  passed; `inconclusive` otherwise. The smallest gain worth porting is 0.02 points over all rows, or
  a 20% cut of excess loss on the smooth-offset days with no harm elsewhere, because the gate adds a
  run counter and three settings to the serving path; a smaller all-row gain is reported as a
  `small gain`. m, k, and z are chosen on the odd tuning months of the replicate year only, from
  the grid m in {3, 5, 10}, k in {4, 6, 8}, z in {0.5, 1.0}; the pre-registered point (m = 5, k = 6,
  z = 1.0) is kept unless a grid cell lowers the tuning-month score by at least 0.005 points. The
  settings are frozen before the scored year is read.
benchmark: >-
  Arm D of experiment 155 as stored (2.756 points over leads 0.5 to 168 hours at the five decision
  Primaries in the scored year) and the calibrated spread of experiment 169 arm A (2.7519). A0
  rebuilt here must reproduce 2.756 within 0.005 or the experiment stops.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles) in points of effective
  capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168 hours; also on
  rows within 24 hours after a gate trigger, on the 15 smooth-offset days and 17 block days of
  experiment 175, and on cold nights; tail coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00190-dp-kalman-level-adaptation-gated-boost/
verdict: inconclusive
best_arm: >-
  A0 (arm D of experiment 155, unchanged), 2.7562 points over leads 0.5 to 168 hours in the scored
  year, against 2.7564 for the gated boost (A1), 2.7791 for the always-on boost (A2), and 2.8336 for
  the shared-level boost (A3)
headline: >-
  The known-answer gate failed: the gated boost removed -0.6% to +0.4% of the excess loss of an
  injected offset of 0.10 of the median load, against a pass mark of 50%. On the scored year the
  gated boost scored +0.0002 points against arm D (95% interval -0.0046 to +0.0062), so the
  verdict is `inconclusive` and the contrasts are not evidence about the real days.
lessons:
  - >-
    No setting of the level's process noise recovered an injected offset: the gate's recovery was
    within about 5 percentage points of zero for the tuned gated boost, the always-on boost, the
    shared-level boost, and a post hoc multiplier of 100 on every one of them.
  - >-
    The counter was on for 0.6% to 1.2% of readings, so the gated boost is close to arm D by
    construction, and the boost that is on all the time scored 0.023 points worse.
  - >-
    On the smooth-offset day plotted in figure 4, all four arms forecast about 10 points too high
    through the whole day, so a faster filter did not change a forecast that was wrong from the
    first half-hour.
take_home_message: >-
  Making the Kalman filter's level state faster when the residual keeps one sign did not help:
  the gated version scored the same as the current filter (+0.0002 points), and a version that is
  always fast scored 0.023 points worse. The known-answer gate also failed, because no setting of
  the level's process noise recovered an injected offset, so the filter's speed is not what limits
  the forecast on smooth-offset days. The next lever is the mean model or a new input, not the
  filter.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not raise the level's process noise in the GAM-K filter, gated or not: the gated boost
    scored +0.0002 points against arm D and the always-on boost +0.023 (C1, C2).
  - >-
    Look for the cause of a smooth same-sign offset in the mean model or its inputs, not in the
    filter's speed: no boost recovered an injected offset, even at 100 times the noise (gate).
  - >-
    Run a known-answer gate that injects an offset into the observed power before reading any
    filter contrast, and report recovery by hours since the offset started (figure 3).
---

# Faster Kalman level state when the residual keeps one sign

---

**Take-home: a faster level state does not rescue the forecast on days when the residual keeps one
sign.** The gated version scored the same as the current filter, the always-on version scored worse,
and the known-answer gate failed, so the filter's speed is not the limit. The next lever is the mean
model or a new input.

We asked whether a Kalman filter that speeds up its level state when the recent forecast errors keep
one sign cuts the loss on days like those, without harming ordinary days. The filter is that of the
convex demand model ([experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)),
and the forecast it corrects is the best-guess forecast of
[experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md) (arm
D).
We first added a known one-signed offset to the observed power of ordinary days and measured how
much
of its extra loss each filter removed. We then scored the filters on the real scored year at the
five
decision Primaries. The tested method is a variant of the Kalman filter on additive-model
coefficients, not a method of ours. Arm D is rebuilt here bit for bit (2.7562 against 2.7562).

**Verdict: `inconclusive`, by the pre-registered rule that a failed gate caps the verdict.** The
gate
asked for at least 50% recovery at the 0.10 size for 16-hour and 36-hour offsets. The gated boost
recovered -0.6% (95% interval -3.5% to +0.8%) and +0.4% (-1.3% to +2.2%).

![Contrasts over all rows](figure_1_contrasts.svg)

![Known-answer gate](figure_2_recovery.svg)

![Recovery by hours since the offset started](figure_3_curve.svg)

![Example smooth-offset day](figure_4_example_day.svg)

## Arms and method

- **A0:** arm D of experiment 155: the T-all fits on the ensemble-mean weather (experiment 136's
  A2), the
  winter night offset, and recipe R, with the spread refitted on the tuning months.
- **A1, A2, A3:** only the level's process noise changes, as in the plan. A1 multiplies it by m
  while the
  counter is on. A2 multiplies it at every reading. A3 is A1 with the level increment shared across
  the four hour bands. The counter reads the one-step innovations in time order across bands.
- **Tuning:** the grid of 18 settings was scored on the odd tuning months of 2024-07 to 2025-06 at
  stage 1, with A0's spread held fixed. The pre-registered point (m 5, k 6, z 1.0) scored 2.7506
  against A0's 2.7486. The best cell (m 5, k 8, z 0.5) scored 2.7450, 0.0056 below the
  pre-registered
  point, which passes the 0.005 rule, so the frozen setting is **m 5, k 8, z 0.5**.
- **Row set:** `fold_a_daily_gap_v1`, the five decision Primaries (D5, D10, D11, D14, D16),
  leads 0.5 to 168 hours, 464,363 rows outside switching events. Telemetry delay 3.5 hours.
  Retraining
  schedule: monthly T-all fits stored by experiment 136; the contrast could depend on the
  schedule only through the filter's residuals, which every arm shares.
- **Gate (lab shortcuts):** offsets of 0.05, 0.10, and 0.20 of the day's median load lasting 8, 16,
  and 36 hours, 8 days per substation per cell (40 per cell), on days of the even months. The
  forecast
  of a half-hour is the proxy-weather prediction plus the filter's correction, with origins every 6
  hours. The loss uses a quantile table from the clean tuning-month errors. A matched 40 days
  without injection show the harm.

## Results

**Scores, scored year, quantile score in points (lower is better):**

- A0 2.7562, A1 2.7564, A2 2.7791, A3 2.8336 (spread refitted). Stage-1 versions: 2.7910, 2.7904,
  2.8232, 2.8969.
- With experiment 169's calibrated spread held fixed: A0 2.7521, A1 2.7568, A2 2.7897, A3 2.8499.
- C1 A1 minus A0: +0.0002 (95% interval -0.0046 to +0.0062; 1 of 5 substations better). With the
  fixed spread: +0.0048 (+0.0004 to +0.0103), a small loss.
- C2 A2 minus A0: +0.0229 (-0.0023 to +0.0482). C3 A1 minus A2: -0.0227 (-0.0489 to +0.0045). C4 A3
  minus A1: +0.0773 (+0.0450 to +0.1136; 0 of 5 substations better).
- The gate's counter was on for 0.6% to 1.2% of readings at the five substations (326 to 585
  triggers).

**Pre-registered predictions:**

- **P1 fails.** On the 15 smooth-offset days of experiment 175, A1 changed the excess loss by -1.4%
  (interval -2.5% to +0.3%); a cut of 20% was predicted.
- **P2 holds with the refitted spread.** A1 minus A0 is +0.0002 and the upper bound is +0.0062. With
  the
  fixed spread the upper bound is +0.0103, just above the +0.01 limit.
- **P3 holds on the point estimate.** A1 beats A2 by 0.023 points, with an interval that includes
  zero
  (excluding it with the fixed spread: -0.033, -0.057 to -0.007).
- **P4 fails.** A3 changed the excess loss on the smooth-offset days by -9.5% (-20.4% to +6.4%),
  worse than A1.
- **P5 holds.** On the 17 block days A1 cut the excess loss by 1.5% (+0.1% to +3.7%).

**Known-answer gate:**

- A1 recovered -0.4%, -0.6%, and +0.4% of the offset's excess loss at the 0.10 size for 8, 16, and
  36
  hours. At the 0.20 size the values for A1 were within about 3% of zero.
- The excess loss of A0 at the 0.10 size was 1.5, 1.0, and 0.9 points of quantile score, so there
  was
  loss to recover.
- A1 minus A0 on the 40 left-alone days: -0.001 points (-0.014 to +0.014); A2 +0.079 (+0.001 to
  +0.159); A3 +0.015 (-0.074 to +0.111).
- Post hoc, a multiplier of 100 on A1, A2, and A3 recovered no more (-1.1 to +0.1 at the 0.10 and
  0.20 sizes, 40 cases per cell).
- Wiring check (`e00190_wiring_check.py`): on a stream with no noise, A0 follows a 7.8-point offset
  to
  between 4.9 and 7.1 points, A1 to between 5.2 and 7.1, and A3 reaches the bands whose own
  windows have not been read (8.6 in one band). The code works; the real residual's noise is what
  limits recovery.

## Checks

- Median forecast minus mean forecast per site: 0.00 at every site and arm.
- Mean-forecast error beside the quantile score: C1 on the mean absolute error is +0.0060 (-0.0011
  to
  +0.0139), the same direction as the quantile score; C3 is -0.0567 (-0.0971 to -0.0157).
- Tails (all rows, q95 coverage 0.947 to 0.949 for every arm; q99 0.983 to 0.984). On the 5,395
  cold-night
  rows: q95 coverage 0.837 (A0), 0.841 (A1), 0.858 (A2), 0.763 (A3); quantile score 3.342, 3.317,
  3.309,
  3.765.
- Rows within 24 hours after a trigger (194,765 rows): C1 +0.0009 (-0.0060 to +0.0083).
- Audit triggers: trigger 1 fired (the known-answer gate failed). The failure was explained by the
  noise in the residual and tested by the wiring check above, and a fresh Opus audit has not been
  run.
- Curtailment cost was not scored: this experiment has no generator forecasts to feed the
  node-plus-generators metric.

## Caveats

- The gate runs on the proxy-weather forecast, not on ensemble weather, so it removes weather error
  from the
  comparison and makes recovery easier than in the real forecast.
- The counter's rule is one of several defensible gates, but no setting tried, up to 100 times the
  noise,
  recovered an offset.
- Tuning used the stage-1 score, not arm D's, to keep the 18 settings cheap.

## What to try next

- Look at the mean model: on the plotted day every arm forecasts about 10 points too high all day,
  so the
  error is in the weather-driven part or a missing input, as experiment 175 suggested.
- A small learned correction that reads whether several neighbouring Primaries show the same-sign
  residual
  (a neural head on the filter's innovations). The prediction that could fail: it recovers at least
  20%
  of the excess loss on the 15 smooth-offset days, which none of the four filters here did.
