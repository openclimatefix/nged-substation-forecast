---
id: "00348"
slug: wind-plant-mdn-upper-tail
title: >-
  Does fixing the upper tail of the frozen physics plant plus mixture head let it beat Wind farms
  research default 1 on the three metered wind farms?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: finished
date: 2026-10-25
kind: change
size: >-
  medium. Stored: private head outputs and predictions under
  data/ml-experiments/00348-wind-plant-mdn-upper-tail/ only. Serving path: not touched (the page
  says what serving each arm needs). Degradation rule: not touched. Designs: several are defensible
  for the tail (loss weights, a point mass, a monotone increment network, a residual band), so the
  plan fixes five before any score is read. Callers: this folder's scripts, which import experiments
  230, 303, 310, 326, 336, and 347 and call no reviewed module beyond `studies.*`. Sonnet only, one
  code review and one prose review.
hypothesis: >-
  Experiment 347's best arm (y2: Stage 3 frozen physics plant, 51 ensemble members through it, a
  three-component Gaussian mixture head on the summary of the resulting power values) beats Wind
  farms research default 1 by 0.234 points over true days 1 to 10 but is over-confident in the upper
  tail (2.3% of observations above q99 over every lead, and a 0.99 pinball loss up to 0.084 worse in
  experiment 347's report). One of five tail fixes (z1 tail-weighted loss, z2 capped quantiles with
  a point mass at the plant's scale, z3 monotone increments above q90, z4 z1 plus z2, z5 the solar
  recipe of a plant residual band blended with the default's XGBoost quantiles) brings the share
  above q99 under 1.5% and the 0.99 pinball loss to within the tolerance, while keeping the headline
  gain.
prediction: >-
  Written before any score is read. z4 brings the share above q99 from 2.3% to under 1.5% and the
  0.99 pinball loss to within 0.03 of the default's. z2 does most of that at W1 and W2 and less at
  W3. z1 alone lowers the 0.99 pinball loss by under 0.03 and raises the median-band loss by under
  0.03. z5 lowers the 0.99 pinball loss by at least 10% against the default, puts the share above
  q99 under 1.5%, and is within 0.1 of z4 on the headline. The headline gain of every arm over the
  default stays within 0.05 of y2's (-0.234 over days 1 to 10), and no arm lowers the headline by
  more than 0.05 beyond y2. A head fed only the plant's median reproduces at least 70% of the gain.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order z4, z5, z3, z2, z1;
  the first arm that passes all five is adopted (at most one). Condition 1: the 99.8% month-block
  interval (level 1 - 0.01/5) of (arm minus default) on the quantile score on the kept rows, scored
  at true days 1 to 10 lies wholly below zero and the point estimate is at least the minimum gain
  (0.03 for z1, z3, z4, new heads; 0.02 for z2 and z5, new fitted steps). Condition 2: the one-sided
  95% upper bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most 0.05 points at
  each of the 0.01 and 0.99 levels (the headline's own tolerance is 0.02). Condition 3: the
  replicate (hindcast 2024-04 to 2024-12, scored on the daily runs of 2025-01 to 2025-06) has the
  same sign and a t test on monthly differences does not contradict it. Condition 4: reproduction
  gate (the y2 refit matches experiment 347's within 0.01 on the kept rows), shuffled-summary
  placebo, five seeds, leakage check, and no audit trigger unresolved. Condition 5: the page says
  what serving needs. A winner on condition 1 that fails another stays a candidate.
benchmark: >-
  The default: Wind farms research default 1 (the research-default log names no wind version yet;
  experiments 259, 339, and 347 use the ERA5-trained XGBoost on the 51 members plus experiment 259's
  offset table, 11.352 points on the kept rows over every lead), and no later version is logged.
  Beside it, descriptive: y2 as in experiment 347, XGBoost alone, XGBoost with the adapter column
  (experiment 326 arm c), experiment 324's W4.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all half-hours beside). Hindcast for the head: 2024-04-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, headline scope true days 1 to 10 (lead 24 to
  240 hours, kept rows), with days 1 to 2, days 3 to 10, and every lead beside; tolerance 0.02 for
  the headline and 0.05 for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00348-wind-plant-mdn-upper-tail/
verdict: partly supported
best_arm: >-
  z5 (plant residual band blended per quantile with the default's XGBoost quantiles): 10.605 points
  on the kept rows at true days 1 to 10 against 10.752 for the default (-0.147, 99.8% interval
  -0.274 to -0.059), with the 0.99 pinball loss unchanged (0.602 against 0.603). The six head arms
  (y2, y2p, z1 to z4) score 0.225 to 0.245 better than the default, and every 99.8% interval still
  includes zero.
headline: >-
  Tail weights, caps, and a point mass do not make the plant plus mixture head adoptable: every head
  arm keeps experiment 347's gain over Wind farms research default 1 (0.225 to 0.245 points over
  true days 1 to 10, 99.8% intervals reaching +0.05 to +0.06), the weighted loss and the monotone
  increments leave 2.0% to 2.2% of observations above q99, and the point mass collapses q99 onto the
  farm's scale and raises the 0.99 pinball loss by 0.15. Of the five arms in the plan, the solar
  recipe is the only one whose interval excludes zero (-0.147), and it keeps the default's 0.99
  pinball loss.
production_status: candidate
adopted_in: none
lessons: >-
  A point mass at the top of the mixture cannot be fitted with a pinball loss alone: once the mass
  weight exceeds 1% the 0.99 quantile equals the scale for almost every row and the weight gets no
  gradient from the top levels, so q99 stops depending on the forecast. The head's gain over the
  default is mostly recalibration of the plant's median (a head given only the median reproduces 66%
  to 77% of it), so the tail needs its own information, which the XGBoost quantiles supply.
take_home_message: >-
  The frozen physics plant with a mixture head stays about 0.23 points of capacity better than the
  research default over days 1 to 10, but none of the four tail fixes inside the head brings the top
  quantile into calibration without losing score at that quantile, and none of their intervals
  excludes zero. Blending the plant's residual band with the default's XGBoost quantiles gains less
  (0.15 points) but is the arm of the five whose gain is clear and whose 0.99 quantile is no worse,
  so that recipe, not a better head, is the next step.
advice:
  - >-
    Do not train a point mass at the farm's scale with a pinball loss alone: q99 sat on the scale
    for almost every row and the 0.99 pinball loss rose by 0.15 (z2, z4, tail table). Fit the mass
    weight with a likelihood or a classification loss on saturated half-hours first.
  - >-
    Blend the plant's quantiles with the default's XGBoost quantiles per quantile and lead band
    before adding a tail loss weight, a point mass, or monotone increments to the head: it was the
    arm with a clear gain and no loss at q99 (z5).
  - >-
    Score a wind head at true days 1 to 10 and report the share above q99 beside the 0.99 pinball
    loss: z2 and z4 cut the share to 1.1% while the loss rose, so the share alone misleads.
  - >-
    Treat a base-only head as the reference for any gain claimed over the default: it reproduced 66%
    to 77% of every head arm's gain (controls bullet).
---
 **Take-home: the frozen physics plant with a mixture head stays about 0.23 points of capacity
better than Wind farms research default 1 over days 1 to 10, but no tail fix inside the head
calibrates the top quantile without raising the pinball loss at that quantile, and the plain recipe
of blending the plant's residual band with the default's XGBoost quantiles is the only arm whose
gain is clear.** Verdict: partly supported. The next step is to build on that blend, and to fit a
point mass with a likelihood.

We asked whether five changes to the upper tail would make experiment 347's best arm (a physics
model of each farm run on the 51 weather-ensemble members, then a three-component Gaussian mixture
head) adoptable. The head scored 0.23 points better than the default but put 2.2% of observed power
above its 0.99 quantile (2.3% over every lead). The five changes were a loss that weights the top
quantiles (z1), a point mass at the farm's maximum plus a cap (z2), monotone increments above the
0.9 quantile (z3), z1 plus z2 (z4), and a blend of the plant's residual band with the default's
XGBoost quantiles, the recipe used for the metered solar farms (z5; experiments
[270](../00270-metered-farm-xgboost-vs-plant-diagnostic/result.md),
[287](../00287-metered-farm-spread-binned-plant-heads/result.md), and
[300](../00300-metered-farm-best-guess-assembly/result.md)). All scores are in points of capacity (1
point is 1% of a farm's capacity). Terms: W1 to W3 are the three metered wind farms, whose names are
withheld. The forecast 99th percentile, q99, should lie below 99% of observations. The pinball loss
at a quantile is the standard score of that quantile. The "kept" half-hours are those that survive
experiment 210's mask of idle or unreliable periods. The "scale" is each farm's fitted maximum
output in the plant model. "True days 1 to 10" means lead hours 24 to 240, on the 53 Tuesday runs of
`fold_a_tuesdays_v1`, kept half-hours, **scored at true days 1 to 10 (lead 24 to 240 hours)**.

![Every arm minus the default at true days 1 to 10, with 95% and 99.8% intervals](figure_1_headline.svg)

![Share of observations above q99 and 0.99 pinball loss for every arm](figure_2_upper_tail.svg)

![Forecast and observed q95 and q99 per farm for arm z4, whose q99 sits on the farm's scale](figure_3_power_curve_z4.svg)

![Forecast and observed q95 and q99 per farm for arm z5](figure_3_power_curve_z5.svg)

![Forecast and observed q95 and q99 per farm for the default](figure_3_power_curve_default.svg)

## Results

- **Arms.** Base is experiment 347's y2 (Stage 3 plant, summary of the 51 member powers, head with
  three Gaussian components). z1 to z4 as in the plan. The reference `y2p` is y2 trained with the
  pinball loss, weights 1. z5 works on stored quantiles: the plant's residual band in six
  equal-count bins of the plant's mean forecast and four lead bands, blended per quantile and lead
  band with weights chosen on earlier scored months. Wind farms research default 1 is XGBoost plus
  experiment 259's offset table (the research-default log has no wind version yet, so this page uses
  the name experiments 259, 339, and 347 gave it). Code reuse: experiments 230, 303, 310, 326, 336,
  and 347. - **Headline, true days 1 to 10 (kept rows, default 10.752).** y2 10.518 (-0.234; 99.8%
  interval -0.512 to +0.056); y2p 10.507 (-0.245); z1 10.527 (-0.225; -0.496 to +0.060); z2 10.508
  (-0.244; -0.523 to +0.059); z3 10.514 (-0.238; -0.501 to +0.048); z4 10.507 (-0.245; -0.516 to
  +0.048); z5 10.605 (-0.147; -0.274 to -0.059). Days 1 to 2: head arms -0.05 to -0.07, z5 -0.172.
  Days 3 to 10: head arms -0.28 to -0.31, z5 -0.158. Every head arm is within 0.011 of y2, and every
  head arm is within 0.02 of y2p (z1 is the furthest, 0.020 worse). - **Reference rows (true days 1
  to 10).** XGBoost alone 11.102; XGBoost with the adapter column 10.639; experiment 324's head
  10.523; climatology 13.860. Seed sd of the head arms on kept rows is 0.011 to 0.044. - **Standard
  architecture (physics plant + mixture head).** The best head arm (z4, 10.507) sits 0.016 points
  (0.1%) below the best other family row (experiment 324's W4, 10.523) and 0.245 points (2.3%) below
  the default. - **Replicate** (heads trained on 2024-04 to 2024-11, scored on the daily runs of
  2025-01 to 2025-06, default 9.728): y2 -0.376; y2p -0.354; z1 -0.343; z2 -0.356; z3 -0.329; z4
  -0.331; z5 -0.209 (95% -0.371 to -0.070; t test on six monthly differences p = 0.028). All have
  the same sign.

| Arm (true days 1 to 10) | Below q01 (%) | Above q95 (%) | Above q99 (%) | p01 | p95 | p99 | 5 to 95 coverage (%) |
|---|---|---|---|---|---|---|---|
| default | 0.55 | 6.62 | 1.66 | 0.285 | 2.415 | 0.603 | 92.0 |
| y2 | 0.81 | 6.69 | 2.15 | 0.283 | 2.382 | 0.626 | 88.9 |
| z1 | 0.52 | 6.43 | 1.95 | 0.283 | 2.384 | 0.613 | 90.8 |
| z2 | 0.62 | 6.43 | 1.11 | 0.283 | 2.387 | 0.756 | 89.9 |
| z3 | 0.53 | 6.68 | 2.21 | 0.283 | 2.393 | 0.620 | 90.5 |
| z4 | 0.63 | 6.48 | 1.11 | 0.283 | 2.388 | 0.756 | 90.0 |
| z5 | 0.68 | 6.62 | 1.79 | 0.287 | 2.407 | 0.602 | 90.8 |
| z5a (plant + offsets) | 0.63 | 6.32 | 1.59 | 0.285 | 2.419 | 0.610 | 89.9 |
| z5b (plant + residual band) | 1.02 | 7.18 | 2.32 | 0.295 | 2.543 | 0.650 | 89.0 |
| y2p | 0.58 | 6.38 | 1.93 | 0.283 | 2.377 | 0.612 | 90.3 |

- **Tails.** z2 and z4 put q99 on the farm's scale for almost every row (figure 3), so their 1.1%
  above q99 is the share of observations above the scale. The pinball loss gives the point mass's
  weight no gradient at levels above 1 minus that weight. The head therefore cannot learn to lower
  the weight. By farm, the share above q99 is 0.69%, 2.57%, and 0.04% for W1, W2, and W3 (W2's scale
  is below capacity).
- **Controls.** The reproduction gate holds (y2 refit 11.081 against 11.081 on all leads, kept rows).
  A base-only head (plant median, lead, farm) reproduces 71% (y2), 74% (y2p), 75% (z1), 66% (z2),
  77% (z3), and 75% (z4) of each arm's gain. The shuffled-summary placebo scores 5.6 to 5.8 points
  worse than the default. The plant is frozen and the head sees ensemble weather only.
- **Checks.** Median minus mean forecast per farm (true days 1 to 10) is -3.3 to -4.5 points for the
  default and -3.0 to -4.1 for every arm, so every arm's median stays 3.0 to 4.1 points below its
  mean, close to the default's. The mean-forecast squared error and the median absolute error move
  in the same direction as the quantile score for every arm. No audit trigger fired: the head arms'
  failed predictions (below) are explained in the tails bullet, no control failed, and no sister
  experiment has an interval to compare with.
- **Pre-registered predictions.**
    - z4 brings the share above q99 under 1.5% (holds, 1.11%) and the 0.99 pinball loss within
      0.03 of the default's (fails, 0.153 above), so the joint prediction fails.
    - z2 does most at W1 and W2 and less at W3: fails (by share above q99, W2 is worst at 2.57%
      and W3 best at 0.04%).
    - z1 lowers the 0.99 pinball loss by under 0.03 (holds, 0.013 against y2) and raises the
      quantile score by under 0.03 (holds, 0.009).
    - The headline stays within 0.05 of y2's for every arm (holds), and the base-only head
      reproduces at least 70% of the gain (holds for y2, y2p, z1, z3, and z4; fails for z2).
    - z5 lowers the 0.99 pinball loss by 10% (fails, 0.2%), puts the share above q99 under 1.5%
      (fails, 1.79%), and is within 0.1 of z4 on the headline (holds, 0.098).
- **Adoption (order z4, z5, z3, z2, z1).** No head arm passes condition 1 (every 99.8% interval
  includes zero), and z2 and z4 also fail condition 2 (0.99 pinball loss harm up to 0.212). z5
  passes conditions 1 to 4 (gain 0.147, harm to p01 and p99 at most 0.013, replicate -0.209),
  The replicate of z5 was added after the main scores were read, because the plan said that the
  default's replicate forecasts did not exist, and they do. Therefore z5 stays a candidate, and
  nothing is adopted.
- **Serving.** z1 to z4 need what y2 needs: the plant run on each member, then the head. z5 needs
  the plant, a stored table of offsets by bin, and the default's XGBoost quantiles.
- **Curtailment metric.** Not scored: `studies.node_curtailment` needs a node's export and its
  generators' forecasts together, and this experiment forecasts the three farms alone.
- **Caveats.** Scored at true days 1 to 10 only; three farms and 53 runs give wide intervals. z5's
  blend weights for the first runs of a month use actuals of the earlier month that were not yet
  observed at those runs' issue times, a small look-ahead. z5 fits on earlier scored months. Each
  head was fitted once on the hindcast and never refitted. Whether refitting changes the contrast
  was not tested.
- **What to try next.** (1) Fit the point mass's weight with a likelihood or a classification loss
  on saturated half-hours, then retest z2. (2) Apply the solar recipe to the head's own quantiles
  (plant plus head, then band, then blend), a tier 1 step. (3) Repeat z5 with a pre-registered
  replicate before any adoption.
