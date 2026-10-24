---
id: "00339"
slug: wind-research-default-1-adapter-plus-mixture-head
title: >-
  Does a mixture-density head on the wind adapter's quantiles improve Wind farms research default 1
  once the idle half-hours are handled?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: finished
kind: change
date: 2026-10-24
size: >-
  complex by the research-default rule. Stored: head outputs and tables under
  data/ml-experiments/00339-wind-research-default-1-adapter-plus-mixture-head/ only, plus the
  forecasts of the adopted arm if there is one. Serving path: not touched, but arm x2 would add a
  rule on the ensemble-mean wind speed and a floor value per farm, and every head arm would add a
  small neural network to serving; the page says so and does not do either. Degradation rule: not
  touched. Designs: several are defensible for handling the idle half-hours (rule, training set),
  so the plan fixes three arms and the order before any score is read. Callers: this folder's
  scripts, which import experiments 324, 326, 336, and 303 and call no reviewed module beyond
  `studies.*`. One Sonnet code review and one Sonnet prose review (maintainer's brief), no Opus.
hypothesis: >-
  A three-component Gaussian mixture head on the hindcast, fed the quantiles of the physics core
  plus adapter and the ensemble members' summaries of speed, direction, and temperature (experiment
  324's W4), lowers the quantile score of Wind farms research default 1 on the kept rows of the
  three metered wind farms W1 to W3 at 1 to 10 day leads. The head loses on the idle half-hours
  because it is trained on kept rows only, and the maintainer's view is that a forecast-time rule
  (predict the idle floor where the ensemble-mean wind speed is below cut-in) needs no idle
  detector. Three arms are tested: x1 the head as in experiment 324, x2 x1 plus the rule, x3 the
  head trained on all hindcast half-hours.
prediction: >-
  Written before any score. x1 keeps at least 0.2 of its 0.27 gain on the kept rows in the
  replicate period. x2 removes at least half of x1's loss on the all-half-hours rows (the +0.208
  from the idle rows) but recovers under a third of it, because the 210 mask covers idle readings
  at higher wind that the cut-in rule cannot reach. x3 loses under 0.1 on the kept rows against x1
  and gains on the all-half-hours rows. None of the arms fixes the upper tail (the share above q99
  stays above 1.8%). The head is no better than the default for a farm with no history
  (leave-one-farm-out).
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order x2, x3, x1, and the
  first arm that passes all five is adopted (at most one). Condition 1, gain: the 99.67% month-block
  interval of (default plus x, minus default) on the kept-row quantile score, scored at 1 to 10
  days, lies wholly below zero (level 1 - 0.01/3 for three arms), and the point estimate is at
  least 0.03 points (a new head, minimum gain 0.03). Condition 2, no harm: the one-sided 95% upper
  bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most 0.05 points each. The
  all-half-hours score is a reported group with tolerance 0.05 and no pass or fail; the page states
  whether each arm meets it. Condition 3, replication: heads trained on 2024-04 to 2024-11 (early
  stopping on December 2024) and
  scored on the daily runs of 2025-01 to 2025-06 give the same sign, and a t test on the monthly
  paired differences does not contradict it. Condition 4: the reproduction gate, the shuffled-input
  placebo, the seed check, and the leakage check pass, and no audit trigger is unresolved.
  Condition 5: the page says what serving needs. A change that wins on condition 1 and fails
  another stays a candidate. The default plus the rule alone is descriptive and never adopted.
benchmark: >-
  The default: Wind farms research default 1, the ERA5-trained XGBoost on the 51 members plus
  experiment 259's offset table (experiment 303's P1, 11.352 points on the kept rows, as
  experiments 326 and 336 reproduced exactly). Also beside it: climatology, persistence on hours
  0 to 6, the default without the table, the physics core plus adapter alone and with a table, and
  the weather-only mixture network of experiment 313.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all half-hours beside). Replicate: the daily hindcast runs of 2025-01-01 to
  2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline, 0.05 for the all-half-hours rows and
  for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00339-wind-research-default-1-adapter-plus-mixture-head/
verdict: partly supported
best_arm: >-
  x1, the head on the physics core plus adapter's quantiles trained on kept rows only, 11.080
  points on the kept rows against 11.352 for the default (-0.272, 99.67% interval -0.433 to
  -0.108). Not adopted: the 0.99 pinball loss rises by 0.050 (one-sided 95% bound 0.092, limit
  0.05), and a post hoc control shows most of the gain is recalibration of the base quantiles.
headline: >-
  The mixture head on the adapter's quantiles beats the default by 0.27 points on the kept rows
  (replicate period 0.32), and beats each single family (XGBoost, physics core plus adapter, with
  and without the offset table, and the weather-only mixture network) by at least 0.18 points. The
  idle-floor rule (x2) recovers none of the head's 2.1 point loss on the idle-band rows, and
  training on those rows (x3) removes the whole gain. A head given only the 11 base quantiles
  scores as well as the full head (11.050), so the gain is recalibration, not weather information.
production_status: candidate
adopted_in: none
lessons: >-
  The head is a recalibrator of the physics core's quantiles. It raises the lower and middle
  quantiles by about 2.5 points of capacity and lowers the top quantile by 3.3, which fixes the
  lower tail (4.8% of half-hours fall below q05, nominal 5%) and leaves the upper tail
  over-confident (2.4% above q99). A constant shift of the default's quantiles fitted on the
  hindcast
  reproduces 0.144 of the 0.272 points (about half), and an oracle shift fitted on the scored rows
  reproduces 0.181 (about two thirds).
take_home_message: >-
  A small neural-network head that re-draws the physics-plus-adapter wind forecast's quantiles
  gains 0.27 points of capacity (2.4%) on the farms' normal half-hours and keeps that gain on the
  replicate months, but the same head given no weather summaries does as well, so the gain is a
  recalibration that a simpler correction partly reproduces. A simple rule that predicts the idle
  floor at low wind recovers none of the head's loss on the idle-band readings, and training the
  head on those readings removes the gain.
advice:
  - >-
    Do not add a forecast-time rule that sets the quantiles to an idle floor where the
    ensemble-mean wind speed is below cut-in: the rule applies to 0.77% of half-hours and 6.7% of
    the idle-band rows, and it scored worse than the head on the rows it touched (arm x2).
  - >-
    Do not train a wind mixture head on the idle-band rows to protect those rows: the kept-row
    gain fell from 0.27 to 0.00 (x3 minus default +0.007, arm x3).
  - >-
    Compare any wind head against a head fed only the base quantiles and against a hindcast-fitted
    constant shift of the default's quantiles, because together they reproduce about half to all of
    the head's gain
    (post hoc controls, audit).
  - >-
    Write wind head experiments with a tail target: the head's 0.99 pinball loss is 0.05 points
    worse and 2.4% of half-hours fall above q99 (arm x1).
  - >-
    Expect no gain from a mixture head on a farm with no history: the leave-one-farm-out head
    scored 0.15 worse than the default (x1 leave-one-farm-out).
---

**Take-home: a small neural network that corrects the wind forecast's quantiles lowers the quantile
score by 0.27 points of capacity on the farms' normal half-hours, but the gain is a recalibration,
and no arm passes the adoption conditions.** The quantile score measures how well a forecast's 11
quantiles match the observed power, in points of capacity (1 point is 1% of a farm's capacity; lower
is better).

We asked whether the mixture-density head of [experiment
324](../00324-wind-farm-solar-cross-breeds-long-run/result.md) can improve Wind farms research
default 1, called "the default" below. The head is a small neural network that fits three bell
curves to each forecast. It takes the 11 quantiles of the physics core plus adapter (a physics-based
wind-power forecast with a learned correction to the wind speed) and summaries of the 51 ensemble
members (the 51 perturbed runs of the ECMWF weather forecast) for speed, direction, and temperature.
The default is an XGBoost model trained on ERA5 (our estimate of past weather) and run on the 51
members, plus a table of per-quantile offsets.

The head scored 11.080 against the default's 11.352 on the kept rows, the half-hours that
[experiment 210](../00210-wind-idle-noise-mva-artefact/result.md)'s mask does not flag as idle-band
readings (small noisy power readings while a farm is idle). On the idle-band rows the head scored
8.146 against 6.051. We tested two repairs: x2 adds a rule that forecasts the farm's idle floor (the
median power of its idle-band readings) wherever the ensemble-mean wind speed is below the speed at
which the turbines start generating (the cut-in speed), and x3 trains the head on every half-hour.
The rule recovered none of the 2.1-point loss. Training on every half-hour (x3) removed the
0.27-point gain.

The pinball loss at the 0.99 quantile, the score of the upper tail that matters for curtailment,
rose by 0.05 points. The adoption rule allows a rise of at most 0.05 with 95% confidence, and the
bound is 0.092. A control run afterwards, in which the head saw only the 11 quantiles and no weather
summaries, scored 11.050, so the weather summaries add nothing. The verdict is `partly supported`,
and the head stays a candidate: no arm is adopted.

![Each arm minus the default: kept rows, all half-hours, idle-band rows, replicate](figure_1.svg)

![Quantile score by lead day](figure_2.svg)

![Tail shares on the kept rows](figure_3.svg)

![Each model family alone, and the head, minus the default](figure_4.svg)

## Results

"Default plus x" is an arm. Scores are quantile scores in points of capacity (lower is better),
scored at 1 to 10 days, on 107,121 half-hours of three farms (W1 to W3), three seeds. Intervals are
99.67% (three arms), resampling whole months and seeds.

| Arm | Kept rows | All half-hours | Idle-band rows | Days 1 to 2 | Days 3 to 10 | 0.01 pinball | 0.99 pinball | 5%-95% coverage |
|---|---|---|---|---|---|---|---|---|
| Default | 11.352 | 10.825 | 6.051 | 6.612 | 11.759 | 0.284 | 0.640 | 92.1% |
| x1 head, kept rows | 11.080 | 10.789 | 8.146 | 6.472 | 11.763 | 0.283 | 0.691 | 88.2% |
| x2 x1 plus rule | 11.084 | 10.795 | 8.176 | 6.486 | 11.767 | 0.283 | 0.697 | 88.1% |
| x3 head, all rows | 11.359 | 10.844 | 6.172 | 6.462 | 11.832 | 0.281 | 0.729 | 89.1% |
| Default plus the rule | 11.354 | 10.830 | 6.083 | 6.626 | 11.762 | 0.284 | 0.647 | 92.0% |
| Head, base quantiles only (post hoc) | 11.050 | 10.744 | 7.969 | | | | | |
| Default plus +3 point shift (post hoc, oracle) | 11.171 | 10.872 | 8.162 | | | | | |

- **x1 minus default: -0.272 on the kept rows (-0.433 to -0.108), -0.037 on all half-hours (-0.163
  to +0.089).** Seeds give -0.274, -0.285, and -0.256; five head seeds on one base seed score 11.076
  on average (sd 0.023). The gain holds at each farm (-0.17, -0.39, -0.25), without any one farm,
  and at every lead day. The replicate period (January to June 2025, heads trained on 2024-04 to
  2024-11) gains 0.321 (-0.562 to -0.031), t test on six monthly differences p = 0.01.
- **The 0.99 pinball loss rises by +0.050 (one-sided 95% bound 0.092) and the 0.01 loss falls by
  0.001.** Adoption condition 2 allows 0.05, so x1 and x2 fail it. Below q05 the share of half-hours
  moves from 1.3% to 4.8% (nominal 5%), above q99 from 1.8% to 2.4% (nominal 1%).
- **x2 minus x1: +0.003 on the kept rows.** The rule flags 0.77% of half-hours and 6.7% of the
  idle-band rows. On the idle-band rows x2 loses 2.13 points against x1's 2.10, so x2 makes the loss
  1% larger instead of recovering any of it. The plan text for x2 contradicts itself (it says "at
  least half" and then "under a third"); x2 fails both halves. On the rows the rule flags, the idle
  floor scores 3.05 against 2.39 for the default, because the farms still produce small real output
  there ([experiment 336](../00336-wind-research-default-1-below-cut-in-zeroing/result.md)). Default
  plus the rule is +0.002 on the kept rows.
- **x3 minus x1: +0.279 on the kept rows.** Training on all half-hours gives the head +0.007 against
  the default on the kept rows (-0.183 to +0.252) and +0.122 on the idle-band rows, so x3 recovers
  94% of x1's idle-band loss. It also gives up the whole kept-row gain, against a predicted loss of
  under 0.1 relative to x1. The 0.99 pinball loss is 0.089 worse than the default.
- **Placebo and controls.** Feeding the trained head member summaries from another month of the same
  farm scores 12.095 on the kept rows, 0.743 worse than the default, so the head reads its
  summaries. The head with no summaries scores 11.050 (-0.302 against the default), 0.030 better
  than x1 (x1 minus base-only +0.030, 95% interval -0.032 to +0.088, so no difference).
- **Leave one farm out.** A head trained on the other two farms scores +0.153 against the default on
  the kept rows (95% -0.129 to +0.452), worse at W2 and W3 and better at W1. Predicted: no better
  than the default; held.
- **Standard architecture against the best single family.** The best family is the physics core plus
  adapter plus the offset table (P3, 11.262), 0.090 better than the default (0.015 to 0.166). The
  standard architecture (x1: physics core + adapter + head) is 0.182 better than P3 on the kept rows
  (-0.285 to -0.082, 1.6%), and 0.032 worse on all half-hours (-0.061 to +0.131). The weather-only
  mixture network scores 11.578, the physics core plus adapter 11.606.
- **Benchmarks, kept rows.** Climatology 13.547. The default without the offset table 11.697.
  Persistence is scored on hours 0 to 6 only (1,696 kept rows): 15.566 against 6.568 for the default
  and 6.248 for x1.
- **Tolerance on all half-hours (0.05, reported without pass or fail).** x1 (-0.037), x2 (-0.031),
  and x3 (+0.019) all meet it.
- **Serving.** x1 needs a small neural network (about 2 minutes to train per farm on a graphics
  processing unit), its input quantiles from the physics core plus adapter, and the 11
  member-summary columns. x2 adds a rule and a floor value per farm (1.4%, 0.4%, and 2.6% of
  capacity at W1 to W3). x3 needs the network only.

## Adoption conditions

| Arm | 1, gain | 2, no harm | 3, replication | 4, controls |
|---|---|---|---|---|
| x2 | met (0.268) | **failed** (0.99 bound 0.097) | met | met, but see Checks |
| x3 | failed (+0.007) | failed (0.99 bound 0.156) | not met | seed spread too large for a zero gain |
| x1 | met (0.272) | **failed** (0.99 bound 0.092) | met | met, but see Checks |

No arm is adopted, so no forecasts are stored.

## Checks

- **Median minus mean forecast, kept rows, per farm (points).** Default -4.95, -5.39, -4.03; x1
  -4.04, -4.30, -3.25; x3 -4.23, -4.42, -3.85. The head moves the median up relative to the mean at
  every farm.
- **Error of the mean forecast beside the quantile score.** x1: quantile score -0.272, absolute
  error of the median -0.410 (-0.566 to -0.245), absolute error of the mean forecast +0.043 (-0.091
  to +0.184). **Trigger 3 fired nominally for x1**, with the mean forecast not significantly
  different.
- **Triggers fired:** audit trigger 2 (x2 and x3 contradicted their predictions; explanations tested
  above) and audit trigger 3 (quantile score and mean-forecast error disagree, x1). A Sonnet audit
  (private report `briefs/e339_audit_report.md`) re-derived the contrasts and found no leakage.
  **Triage:** accepted all findings. Its main finding: about half of x1's gain (a shift fitted on
  the hindcast gives -0.144, a per-quantile shift -0.150) is a level shift of the default's
  quantiles, and x1 still gains -0.122 (-0.21 to -0.03) beyond the hindcast-shifted default. We ran
  the control the audit asked for: the head with no member summaries does as well as x1, so the
  member summaries add nothing and the head's value is recalibration of the physics core's
  quantiles. The audit also notes that the spread factor of the base quantiles is fitted on earlier
  cases including the scored year, a feature the default shares.
- **Leaving one farm out keeps the sign** for x1 and x2 (-0.32, -0.21, -0.28 without each farm).

## Caveats

- **Scored at 1 to 10 days, 3 farms, 12 months.** Intervals do not cover other farms.
- **x1 is bit-identical to experiment 324's W4** (the stored arrays match), so x1 re-scores
  experiment 324's stored predictions against new contrasts, and no new fit was run for it.
- **x1 and x2 never see idle-band rows in training;** only x3 does.
- **The replicate heads use December 2024 only to stop training early,** and the replicate has six
  months.
- **The oracle shift is fitted on the scored rows** and is an optimistic bound on a bias correction.
- **No audit by Opus,** by the maintainer's instruction; the audit and reviews were by Sonnet.

## What the next experiment should answer

- **Does a hindcast-fitted correction of the default's quantiles (a shift or a small head on the
  default's own quantiles) match the head's 0.27 points?** The post hoc controls say a shift reaches
  about half and a base-only head reaches all of it.
- **Can the head's upper tail be repaired?** Above q99 the head covers 2.4% against 1% nominal; a
  tail-specific term in the mixture is the obvious test.
