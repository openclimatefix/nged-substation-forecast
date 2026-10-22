---
id: "00211"
slug: dp-boosted-tree-calibrator-and-input-ablations
title: >-
  Differentiable physics: can a boosted-tree calibrator capture the gain of the network without error
  inputs?
theme: differentiable-physics-forecast-engine
also_relevant_to: [calibration-and-uncertainty]
status: planned
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

**Planned, not run.** Experiment 209 found that no lookup table matches the network without error
inputs. This experiment asks whether a boosted-tree model does, and which input of the network
carries the gain.

## Plan

- **Rows.** Experiment 200's training window, validation block, and scored rows, from experiment
  201's shared feature files.
- **Part 1, ablations.** Retrain the network without error inputs on CPU, once with its full inputs
  as the reference and once with each of lead, half-hour (sine and cosine), arm D's forecast level,
  and the ensemble-era flag set to zero. One seed each, 4 cores.
- **Part 2, boosted tree.** XGBoost with the pinball loss at the 11 quantiles on the same inputs
  (site, lead, half-hour, weekday, forecast level, era, and arm D's quantile offsets), with column
  subsampling off so that arms are fair, early stopping on the validation block.
- **Contrasts (99% month-resampled paired intervals).** Boosted tree minus network without error
  inputs (decides the verdict), boosted tree minus table T0, each ablation minus the reference
  network.
