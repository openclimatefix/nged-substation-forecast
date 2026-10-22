---
id: "00195"
slug: switching-neural-plus-detector-features
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00195-switching-neural-plus-detector-features/
  only (candidate tables, trained weights). Serving path: not touched. Degradation rule: not touched.
  Designs: more than one is defensible (features only, re-ranking, union, stacking), so all four are
  pre-registered arms. Callers: this folder's scripts only, which copy experiment 189's injector and
  experiment 130's detector code. One Opus audit is allowed because injector leakage is the main risk.
hypothesis: >-
  A learned score that reads the features the rule-based switching detectors already compute (the
  24-hour step statistic of RO-NC, the opposite step at the metered neighbour of the pair alarm, and
  related quantities), trained only on injected transfers, finds more of the 35 real transfers lasting
  over 1 day than RO-NC alone at the same number of unexplained alarms, and more than a rule-based union
  of RO-NC and the pair alarm.
prediction: >-
  Written before any real event is scored. At 80 unexplained odd-month alarms, with events held out by
  Primary pair, the best learned arm raises the MW-weighted recall of the 35 real transfers above
  RO-NC's 0.657 by at least 0.05, and finds at least 22 of the 35 (RO-NC: 18). Logistic regression and
  the neural network match within 2 events (the gain is in the features). Recall on injected transfers
  with an unmetered receiver falls by less than 15 percentage points against metered receivers.
decision_rule: >-
  Primary bar: `supported` for an arm when its MW-weighted recall minus RO-NC's has a 95% interval
  (resampling whole Primary pairs) with the lower end above zero AND it finds at least as many events
  as the rule-based RO-NC plus pair-alarm union at the same total number of alarms. Secondary count
  bar: at least 22 of the 35, or at least 3 events RO-NC misses without losing any it finds. Downstream
  bar: with the arm's alarms feeding the gated mask of experiments 163 and 171, the masked score on the
  real transfers and the no-harm interval must not get worse (reported only if the harness can be run
  inside the slot). `partly supported` when the secondary bar holds and the primary does not; otherwise
  `refuted` (or `inconclusive` when the interval is too wide to tell). No porting threshold is set: a
  learned score would add a model and a pipeline step.
benchmark: >-
  RO-NC on the convex-model residual (18 of 35 found, MW-weighted recall 0.657), the pair alarm of
  experiment 156 (5 of 35), the rule-based union of the two, and experiment 189's raw-window network (7).
row_set_id: >-
  none (scores alarms against the 35 real transfers of experiment 41 and freshly injected transfers; no
  forecast)
metric: >-
  Real transfers found out of 35 (start within 2 hours, end within the larger of 6 hours and 10% of the
  duration, experiment 130's rule) at 80 unexplained odd-month alarms, MW-weighted recall, median onset
  error, and recall on injected worlds.
code: ml-experiments/00195-switching-neural-plus-detector-features/
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

# Experiment 195: does a learned score on the detectors' own features find more real switching than the rules?

Planned. Arms: F (features only: logistic regression, gradient-boosted tree, small neural network),
R (re-ranker of a loosened RO-NC candidate list, features plus a raw residual window), U (union and
intersection of the learned alarms with RO-NC and the pair alarm), C (stacked logistic score fitted on
injected validation worlds only). Leak controls: training on injected worlds and real stretches with no
logged or labelled event only; no real event, no D8 drop and none of the 17 block days in training or
model selection; the series identifier and the Primary are not inputs. Reuses experiment 189's injector
and held-out-by-pair folds.
