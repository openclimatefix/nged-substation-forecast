---
id: "00211"
slug: dp-boosted-tree-calibrator-and-input-ablations
title: >-
  Differentiable physics: can a boosted-tree calibrator capture the gain of the network without error
  inputs?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: done
size: >-
  medium. Stored: private files under data/ml-experiments/00211-dp-boosted-tree-calibrator-and-input-ablations/
  only. Serving path: not touched. Degradation rule: not touched. Designs: several are defensible
  (which inputs the tree model gets, how its quantiles are built), and this plan pre-registers one.
  Callers: this folder's scripts, which copy experiment 200's network code and read experiments 200,
  201, and 209's stored rows. A new method for this line (a boosted-tree model), so a Sonnet plan
  and code review; no Opus.
hypothesis: >-
  The 0.065 points by which the network without error inputs (experiment 200) beats the best lookup
  table (experiment 209) comes from a smooth or higher-order dependence on a few inputs, so a
  gradient-boosted tree model fitted with the pinball loss on the same inputs captures it, and a
  boosted-tree calibrator can replace the neural network. One-at-a-time removal of the network's
  inputs shows which input carries the gain.
prediction: >-
  Written before any fit. The boosted-tree model scores 0.07 to 0.10 points below arm D, within 0.02
  of the network without error inputs (boosted-tree minus network between -0.02 and +0.02). Removing
  arm D's forecast level or the ensemble-era flag from the network removes at least half of its gain
  (experiment 200's audit found each alone cut the gain from 0.099 to about 0.05); removing lead or
  half-hour changes the score by under 0.02 points.
decision_rule: >-
  The boosted-tree model matches the network if its score minus the network's lies within 0.02
  points and the 99% month-resampled paired interval includes zero. A boosted tree is then a
  production candidate, because serving stays close to loading a model and calling predict. It does
  not match if the difference is above 0.02 points or the interval lies wholly above zero. The
  threshold 0.02 is set by cost: a neural network head is a new trained component on the serving
  path, so a tree model needs to give up under 0.02 points to be worth preferring the network.
date: 2026-10-22
verdict: supported
best_arm: >-
  boosted-tree model, 0.098 points below arm D (the stored network without error inputs: 0.100)
production_status: candidate
adopted_in: none
headline: >-
  A boosted-tree model scores 0.0015 points above the network without error inputs (99% interval
  -0.028 to +0.026), so it matches the network by the pre-registered rule, and the network's gain
  depends on arm D's forecast level and the ensemble-era flag.
take_home_message: >-
  A gradient-boosted tree model given the same inputs as the neural network, and no recent errors,
  lowers the error score of our best day-ahead forecast by about as much as the network does
  (0.098 against 0.100 points). The interval is wide, so a loss of up to 0.03 points against the
  network is not ruled out.
  The gain comes mostly from two inputs, the forecast level and the weather-data era.
lessons:
  - >-
    Removing arm D's forecast level or the ensemble-era flag gives back 0.041 and 0.042 points, about
    45% of the network's gain over arm D each; removing lead or half-hour gives back under 0.01.
  - >-
    The tree model needs no monotone constraint or tuning beyond early stopping to match the network
    on these inputs.
advice:
  - >-
    Prefer a boosted-tree calibrator to a neural network if arm D's recalibration is taken to
    production, after the check on more substations below; it scores 0.0015 points above the
    network, inside the 0.02 limit (xgboost minus
    stored network).
  - >-
    Keep arm D's forecast level and the weather-data era flag as inputs to any calibrator; removing
    either gives back about 45% of the gain (no_level and no_era minus reference).
  - >-
    Before porting, check the tree model on more than 5 substations: the 99% interval allows a loss
    of 0.026 points against the network (xgboost minus stored network).
benchmark: >-
  Arm D of experiment 155 (2.432 points at day D+1 on the five decision substations), the network
  without error inputs (2.332), and experiment 209's table T0 (2.397)
row_set_id: >-
  experiment 200's scored rows: arm D of the scored year 2025-07-02 to 2026-06-29, leads 15.5 to 39
  hours, outside switching events
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  day D+1, outside switching events
code: ml-experiments/00211-dp-boosted-tree-calibrator-and-input-ablations/
---

**Take-home: a boosted-tree model captured 0.064 of the 0.065 points that lookup tables missed, so
a boosted-tree calibrator is a candidate to replace the neural network, though the interval allows a
loss of up to 0.026 points.** Arm D is
our best day-ahead demand forecast, from [experiment
155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md). In [experiment
200](../00200-dp-pooled-correction-head-on-kalman-state/result.md) a neural network that was given
no recent forecast errors (the "network without error inputs") lowered arm D's score by 0.10 points.
[Experiment 209](../00209-dp-pinball-table-vs-network-head-on-arm-d/result.md) showed that no lookup
table gets beyond 0.041. We asked whether a gradient-boosted tree model gets the full 0.10 points
without a neural network
on the serving path, and which input of the neural network carries the gain.

**Answer: yes by the rule fixed before the run, and forecast level and weather era carry the gain.**
All scores are quantile scores in points of effective capacity, where lower is better. On the five
decision substations in 2025-07-02 to 2026-06-29, the boosted-tree model scores 0.098 points below
arm D, and 0.0015 points above the network without error inputs (99% interval over resampled months
-0.028 to +0.026). The point estimate is within the 0.02 limit and the interval includes zero, so the
verdict is `supported`. The interval is also wide enough to include a loss of 0.026 points, above
the limit, so the data do not show that the tree model is as good as the network, only that they
could not separate the two.

![Each arm's score minus arm D's, with 99% intervals. The lookup table gains 0.035 points; the
network, its retrained copy, and the boosted-tree model gain 0.09 to 0.10; removing forecast level or
the era flag from the network removes about
half.](figure_1_tree_and_ablations.svg)

## Results

- **Setup.** Rows, training window (2020 to 2025-06-27), validation block, and scored rows are
  those of [experiment 200](../00200-dp-pooled-correction-head-on-kalman-state/result.md). The
  boosted-tree model is XGBoost with the pinball loss ([Koenker and Bassett
  (1978)](https://doi.org/10.2307/1913643)) at the 11 quantiles, fitted to arm D's quantile
  residuals on the network's inputs: substation, lead, half-hour, weekday, arm D's forecast level,
  the ensemble-era flag, and four of arm D's quantile offsets. Column subsampling is off, so that
  every arm sees every input. Depth 6, learning rate 0.1, minimum child weight 50, early stopping
  at round 288 on the validation block. The retrained network and its four ablations ran on CPU,
  with one seed each.
- **Scores (five decision substations, points; lower is better).** Arm D 2.432, lookup table (T0)
  2.397, network without error inputs (stored, experiment 200) 2.332, the same network retrained
  here 2.341, boosted-tree model 2.334.
- **Decision contrast.** Boosted-tree model minus stored network +0.0015 (-0.028 to +0.026). Minus
  the retrained network -0.0072 (-0.036 to +0.018). Minus table T0 -0.064 (-0.085 to -0.044).
  Minus arm D -0.098 (-0.131 to -0.065).
- **Ablations (each minus the retrained network, 99% intervals).** No lead -0.0009 (-0.0049 to
  +0.0030). No half-hour +0.0097 (-0.0031 to +0.0253). No forecast level +0.041 (+0.021 to +0.065).
  No era flag +0.042 (+0.021 to +0.065). The retrained network gains 0.091 over arm D, so removing
  the forecast level or the era flag gives back 45% and 46% of the gain.
- **Retrained against stored network.** The retrained network scores 0.0087 above the stored one
  (+0.005 to +0.012), from a different seed and a CPU run. The ablation contrasts use the retrained
  network as the reference for this reason.
- **Tails (boosted-tree model).** Coverage at q95 95.0% and at q99 99.3%. Pinball at q99 0.373 and
  at q01 0.338 against T0's 0.385 and 0.352. Coverage at q05 and q01 was not computed.
- **Leave one substation out.** Boosted-tree minus retrained network stays between -0.015 and
  -0.002 when any one decision substation is dropped.
- **All six substations.** The decision contrast is +0.0038 (-0.023 to +0.029); no ablation
  contrast changes by more than 0.005.
- **Pre-registered predictions.** Boosted tree within 0.02 of the network: held. Removing forecast
  level or era removes at least half the gain: the point estimates (45% and 46%) fall just under
  half, and the intervals include half. Removing lead or half-hour changes the score by under 0.02:
  held.
- **Caveats.** One seed for each network arm. The tree model has no ablations of its own, so the
  experiment does not show that the tree model relies on the same two inputs. The 99% interval
  rests on about 12 resampled months and five substations.

## Checks

- **Median minus mean forecast per site and the error of the median forecast** were not computed:
  the stored rows hold quantile scores only.
- **Triggers fired:** none. The result agrees with the pre-registered prediction, except that the
  two largest ablations removed 45% and 46% of the gain, just under the predicted half. No input
  was changed to make it hold, and the rows are those audited in experiment 200.
- **Row alignment:** arm D's losses agree with experiment 200's stored rows within 0.001, and the
  table T0 rows are those of experiment 209.

## What to try next

- Fit the boosted-tree model on all 16 Primary substations and a second scored year, to narrow the
  interval that currently allows a 0.026 loss.
- Ablate the tree model's inputs, to check that it relies on forecast level and era as the network
  does.
