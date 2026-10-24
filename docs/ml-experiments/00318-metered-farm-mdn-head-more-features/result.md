---
id: '00318'
slug: metered-farm-mdn-head-more-features
title: >-
  Does giving the mixture-density head of the metered solar-farm forecast more inputs, especially
  the
  forecast lead, lower its quantile score?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: finished
kind: change
date: 2026-10-23
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00318-metered-farm-mdn-head-more-features/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: the arms are fixed by the brief, so no design choice is
  open. Callers: this folder's scripts, which import experiment 310's feature and fit modules and
  experiment 313's pure columns, and no new reviewed module. Asset graph: not touched. One Sonnet
  code review and one Sonnet prose review, no Opus.
hypothesis: >-
  Giving the mixture-density head of Metered solar farms research default 1 the forecast lead (lead
  in whole days), the hour of day and day of year as sine and cosine pairs, and the ensemble spread
  of the shortwave irradiance lowers its quantile score, mostly beyond 36 hours and in the lower
  tail, and one head trained on all lead bands beats one head per band.
prediction: >-
  Written before any score is read. P1: lead features lower the score by at least 0.02 points at
  leads beyond 36 hours (days 3 to 10) and change it by under 0.02 points within 12 to 36 hours (day
  1), in own history. P2: the full feature set lowers the 0.99 pinball loss by at least 3% in own
  history. P3: the shuffled-feature control scores within 0.02 points of the default, so any gain is
  attributable to the features. P4: lead features move the coverage of the 0.01 quantile to within
  0.5 points of 1% at days 3 to 10 (experiment 310: 3.3% to 3.5% for the blend). P5 (the pure
  mixture-density network with lead features, trained on all lead bands) closes experiment 313's
  gap to the plant-based head at days 8 to 10 to under 1.5%. P6 (the pure network given the plant
  forecast as an extra column) beats both the pure network and the default by under 0.5%, so the
  plant is redundant. Each can fail. "Beyond 36 hours" means the bands day 3, days 4 to 5, days 6 to
  7, and days 8 to 10; P1, P2, and P4 hold when
  the point estimate meets the stated size in every one of those four bands (P2 and P4: pooled over
  them).
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months, at 99.17% (Bonferroni across six planned contrasts per band: lead minus default,
  full minus default, full pooled minus default, full pooled minus full, pure with lead (pooled)
  minus pure pooled, and the pure network with the plant column minus pure). The research-default
  rule sets
  the tier: a new input, so a minimum gain of 0.02 points, a no-harm tolerance of 0.02 points on the
  pooled headline and 0.05 on the 0.01 and 0.99 pinball tails, and a replicate on the scored year's
  second half. At most one arm is adopted. The adoption test is on the mixture-density head alone;
  the blend is not rescored, so the best outcome is a candidate. Stop rule: if no arm beats the
  default by 0.02 points at any band, report that the head's inputs are not limiting and add no
  arms.
benchmark: >-
  Metered solar farms research default 1's mixture-density head ("default", five seeds refitted
  here), experiment 313's pure network, experiment 310's `climatology` and `manual_heuristic` rows
row_set_id: >-
  Experiment 310's rows (farms A to F, 2025-09 to 2026-08, day bands 1 to 10)
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99
code: ml-experiments/00318-metered-farm-mdn-head-more-features/
verdict: partly supported
best_arm: >-
  All added inputs with one head trained on all lead bands ("full_pooled"): the 0.99 pinball loss
  falls 5.0% to 6.3% at days 2 to 7 in own history, and the quantile score does not change
  (7.85 against 7.75 points at days 8 to 10).
headline: >-
  Adding the lead, the clock as sine and cosine pairs, and the spread of the ensemble's irradiance
  to
  the mixture-density head lowers its 0.99 pinball loss by 4.8% to 5.6% at days 3 to 7 (the
  intervals exclude zero) and leaves the quantile score unchanged within 0.13 points; the lead alone
  changes
  nothing, and the 0.01 quantile still lets 3.3% to 3.7% of actual values through.
lessons: >-
  The head already holds the lead in hours, so a second lead column adds nothing. The sine and
  cosine clock and the irradiance spread help the upper tail only. One head for all lead bands
  helps the network with no plant (3.0% at days 8 to 10) and does nothing for the plant-based head.
take_home_message: >-
  Giving the solar-farm head more inputs does not improve its overall score at any lead from 1 to 10
  days, so the head's inputs are not what limits the forecast. The extra clock and cloud-spread
  inputs
  do lower the loss at the 0.99 quantile (the tail that matters for curtailment) by about 5% at days
  3 to 7, and none of the changes fixes the lower tail, where 3.3% to 3.7% of actual values fall
  below the 0.01 quantile, where 1% is expected. Training one head for all leads makes the network
  with no plant nearly as
  good as the plant-based head at days 8 to 10 (0.4% worse), so the plant adds little there.
advice:
- >-
  Do not add a lead-day column to the mixture-density head: the head already holds the lead in
  hours,
  and the column changes the quantile score by under 0.01 points at every band (C1).
- >-
  Add the sine and cosine clock and the irradiance spread to the head only if the 0.99 quantile
  matters most: they lower the 0.99 pinball loss by 4.8% to 5.6% at days 3 to 7 and leave the
  quantile score unchanged within 0.13 points (C2, Figure 1).
- >-
  Train the mixture-density network that has no plant as one head for all lead bands, not one head
  per band: the quantile score at days 8 to 10 falls by 3.0% (0.24 points, interval -0.69 to -0.03
  points) (C4).
- >-
  Look for the cause of the lower tail elsewhere than in the head's inputs: no head here brings the
  share of actual values below the 0.01 quantile down from 3.3% to 3.7% towards the expected 1%
  (Figure 4).
production_status: candidate
adopted_in: none
---

**Giving the mixture-density head of the solar-farm forecast more inputs does not lower its quantile
score at any lead from 1 to 10 days. It lowers only the 0.99 pinball loss (the loss on the upper
tail), by about 5% at days 3 to 7.** National Grid Electricity
Distribution (NGED) meters six solar farms, labelled A to F. Metered solar farms research default 1
([experiment 300](../00300-metered-farm-best-guess-assembly/result.md), scored to 10 days by
[experiment 310](../00310-metered-farm-long-lead-scoring/result.md)) starts from a physical model of
the farm's output (the "plant") and corrects the plant's 11 quantiles with a three-component mixture
density network (the "head"). We refitted that head with more inputs: the lead in whole days, the
hour of day and day of year as sine and cosine pairs, and the ensemble's spread of shortwave
irradiance. We also trained one head for all six lead bands instead of one head per band, and added
the two arms that [experiment 313](../00313-metered-farm-pure-xgboost-and-mdn-vs-plant/result.md)
motivates: the network with no plant, and the same network given the plant forecast as a column. The
quantile score is twice the mean pinball loss over the 11 quantiles, in points of capacity (1 point
is 1% of the farm's capacity), and lower is better. "Own history" means the head trains on the
  farm's
earlier months, and "zero history" means it trains on the other five farms only.

**The verdict is partly supported, and nothing is adopted.** The extra inputs fail the research
default's minimum gain of 0.02 points on the quantile score at every band, so the head of research
default 1 stays as it is. The gain that appears is on the 0.99 pinball loss, which replicates in
both halves
  of
the scored year and in zero history. Scores are at leads of 12 to 252 hours (days 1 to 10), with
quarterly refits.

![Figure 1: change in the 0.99 pinball loss against the default head](figure_1_pinball99_change.svg)

![Figure 2: change in the quantile score against the default head](figure_2_score_change.svg)

![Figure 3: absolute quantile score by lead band](figure_3_absolute_score.svg)

![Figure 4: share of actual values below the 0.01 quantile](figure_4_q01_coverage.svg)

## Results

- **Arms.** "default" is experiment 310's head (plant forecast, plant spread, availability ratio,
  lagged power and its missing flag, clear-sky index, lead hours). "lead" adds the lead in whole
  days.
  "full" adds the lead in whole days, the hour-of-day and day-of-year sine and cosine pairs, and the
  irradiance spread (cloud cover itself is
  not stored, and the clear-sky index was already an input). "full_pooled" and "default_pooled"
  train
  one head on all six bands. "full_shuffled" shuffles the added columns (control), and "full_nolag"
  drops lagged power and the availability ratio (the named control for the previous day's plant
  error, which experiment 310 found adds nothing). "pure_pooled" and "pure_lead_pooled" are the
  network with experiment 313's no-plant inputs, pooled, without and with the lead day.
  "pure_plant" is the no-plant inputs plus the plant forecast as a column, per band.
- **Absolute scores, own history (default; climatology; `manual_heuristic`).** 5.36 (8.64; 8.69) at
  day
  1, 6.08 (8.64; 8.68) at day 3, 7.35 (8.64; 8.73) at days 6 to 7, and 7.75 (8.64; 8.83) at days 8
  to
  10. Zero-history default scores 5.53 points at day 1 and 7.91 points at days 8 to 10. Experiment
  313's pure
  network scores 5.34 and 8.03.
- **C1, lead minus default, quantile score (own history).** +0.002 at day 1 to -0.006 at days 8 to
  10; every interval lies within ±0.03 points. The 0.99 pinball loss changes by at most 0.002.
  Prediction P1 (a gain of 0.02 points beyond 36 hours) fails.
- **C2, full minus default, quantile score (own history).** +0.005 (-0.094 to +0.073) at day 1,
  +0.023 at day 3, +0.037 at days 4 to 5, +0.125 (-0.091 to +0.422) at days 6 to 7, and +0.102 at
  days
  8 to 10. No interval excludes zero, and the point estimates are above the no-harm tolerance of
  0.02
  points from day 3. The second half-year drives the rise (+0.20 at days 6 to 7); the first half-
  year
  is flat. In zero history the change is -0.024 to +0.082.
- **C2, 0.99 pinball loss (own history).** -4.8% (-0.016 points, interval -0.030 to -0.006) at day
  3,
  -5.6% at days 4 to 5, -5.3% at days 6 to 7, and -2.6% (interval includes zero) at days 8 to 10;
  -1.4% and -2.4% at days 1 and 2 (intervals include zero). The change replicates: in the first
  half-year it is -6.2% to -8.1% at days 3 to 7, and in the second half-year -2.9% to -4.2% at days
  1 to 7. In zero history it is -2.9% to -4.7% at every band. Prediction P2 (3% or more, pooled over
  days 3 to 10) holds at -4.6%.
- **C3, one head for all leads (own history).** `full_pooled` minus `full`: -0.024 to -0.001 points
  at days 1 to 3 and days 8 to 10, -0.035 (-0.083 to +0.014) at days 6 to 7; no interval excludes
  zero.
  `default_pooled` minus `default` is -0.015 at day 1 and -0.005 at days 8 to 10. `full_pooled`
  minus
  `default` lowers the 0.99 pinball loss by 5.0% to 6.3% at days 2 to 7 and 3.0% at days 8 to 10.
  Pooling alone does nothing for the plant-based head.
- **C4/C5, the network with no plant.** `pure_pooled` minus experiment 313's per-band `pure_mdn`:
  -0.04% at day 1 and -3.0% (-0.24 points, interval -0.69 to -0.03) at days 8 to 10, so one head for
  all leads closes most of experiment 313's gap: `pure_pooled` scores 0.4% above the default at days
  8 to 10 and 0.4% below it at day 1. Adding the lead day to `pure_pooled` (`pure_lead_pooled` minus
  `pure_pooled`) makes the score worse by 0.008 to 0.033 points (days 8 to 10 interval +0.003 to
  +0.063). Prediction P5 (a gap under 1.5% at days 8 to 10) holds, at 0.85% (`pure_lead_pooled`
  against the default), but pooling did the
  work and the lead day did not.
- **C6, the plant forecast as a column of the no-plant network (`pure_plant`).** `pure_plant`
  differs from experiment 313's pure network by -0.4% to +0.2% at every band (intervals include
  zero), so the
  plant column adds nothing measurable to the network. Against the default it scores 0.244 points
  (3.1%) worse at days 8 to 10 (interval includes zero) and 0.012 better at day 1. Prediction P6
  holds only for the first comparison; the comparison with the default mixes in a different
  parametrisation (the default models the residual from the plant, `pure_plant` models the output).
- **Controls.** `full_shuffled` minus default is +0.009 at day 1 and -0.003 at days 8 to 10
  (intervals
  within ±0.04), so P3 holds and the 0.99 gain comes from the added inputs. `full_nolag` minus
  `full`
  is +0.006 and +0.007 points at days 1 and 8 to 10 (intervals include zero), so the previous day's
  plant error adds nothing again.
- **Lower tail (P4).** The share of actual values below the 0.01 quantile is 0.0355 for the default
  at day 1 and 0.0329 at days 8 to 10; 0.0371 and 0.0339 for `full`; 0.0358 and 0.0343 for
  `full_pooled`; 0.0343 and 0.0328 for `pure_pooled`. Against a nominal 0.01 every arm is 3.3 to 3.7
  times too high, and no arm moves it by more than 0.2 points, so P4 fails. Coverage of the 0.99
  quantile is 0.982 to 0.989 for every arm.

## Checks

- **Median minus mean forecast, own history, farms pooled (points of capacity).** Default +0.40 at
  day 1 and +0.30 at days 8 to 10; `full` +0.36 and +0.51; `full_pooled` +0.40 and +0.31. Per farm
  at
  day 1 (default): A -0.10, B +0.22, C +0.12, D +0.61, E +1.15, F +0.41.
- **Error of the mean forecast beside each score contrast.** `full` minus default on the absolute
  error of the mean forecast is -0.015 at day 1 and +0.113 at days 6 to 7 (intervals include zero),
  the same direction as the quantile score; the 0.99 gain has no counterpart because it concerns a
  tail the mean does not see.
- **Audit triggers fired: 2 and 5.** Trigger 2 (a result contradicts its prediction): P1 and P4
  failed. We tested the explanation that the head already holds the lead in hours (C1, and
  `pure_lead_pooled` getting worse). Trigger 5 (a nested arm beats the arm containing it): default
  beats `full` by 0.10 to 0.13 points at days 6 to 10, inside the interval's half-width (about 0.25
  points). A fresh Sonnet agent re-derived the quantile score, the 0.99 pinball loss, and the 0.01
  coverage from the raw predictions and found every contrast within 0.0013 points of the stored
  value (stored absolute scores run 0.002 to 0.003 points above the agent's, an unexplained gap that
  affects both arms of a contrast alike), no duplicate keys, no missing values, and identical row
  counts across arms. Bands d1, d2, and d4 to 5, the zero-history setting, and the interval limits
  were not re-derived. Trigger 4 did not fire: dropping any one farm
  leaves the 0.99 change of `full` negative at days 3 to 10 (-0.0035 to -0.023). Triggers 1 (a
  failed control), 3 (the mean-forecast error disagrees), 6 (an interval much wider or narrower than
  a sister experiment's), and 7 (a later design depends on this result) did not fire.

## Method and schedule

- **Schedule.** Quarterly refits on all earlier months (screening schedule), so absolute scores
  could
  shift slightly under weekly refits, and the contrasts were not tested for dependence on the
  schedule. The head is the mixture network of experiment 310 (copied from its fitter), with five
  seeds averaged in own history and three in zero history. The default reproduces experiment 310's
  head (-0.14%, from two extra seeds).
- **Rows.** Experiment 310's scored-year rows (2025-09 to 2026-08, farms A to F, 12 to 252 hours),
  common to every arm and to the floors; intervals resample the 12 scored months at 99.17%
  (Bonferroni, six planned contrasts per band and setting). The half-year intervals resample six
  months and are coarse. The pooled arms and the arms `lead`, `full_shuffled`, `full_nolag`, and
  `pure_plant` ran in own history only.
- **Reuse.** The scripts import experiment 310's folder; the no-plant columns are copied from
  experiment 313.

## Caveats and next steps

- **Leads.** The scored range is 12 to 252 hours (days 1 to 10), which covers NGED's horizon.
- **The blend was not rescored.** The head is one of three blend members, so a head gain of 0.05
  points or more at the 0.99 quantile may shrink in the blend.
- **Experiment 313's caveats stand.** Experiment 313's intervals (-2.4% to +1.0% at day 1, -1.3% to
  +10.4% at
  days 8 to 10) show no detectable difference, not equivalence. Its zero-history test uses six farms
  in
  one region with one fleet-shape plant, so it cannot show whether a physical plant transfers better
  to
  a farm with a different tilt or climate. The plant also keeps value that is not accuracy:
  explainable
  parameters, a farm card, and a ceiling check.
- **Lead day is redundant.** `lead_day` is a step function of `lead_hours`, which every arm already
  holds, so C1 and `pure_lead_pooled` test only a coarser copy; the effect of an hour-of-lead
  feature
  cannot be separated here.
- **Pooled arms confound pooling with training budget.** A pooled head sees about six times the rows
  at the same number of epochs, so `full_pooled` and `pure_pooled` also take more gradient steps.
- **Next.** The lower tail needs a different remedy, such as a dedicated tail head or recalibration
  of
  the 0.01 quantile on the farm's own history; the 0.99 gain should be retested inside the blend.
