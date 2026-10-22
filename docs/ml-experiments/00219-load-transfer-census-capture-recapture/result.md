---
id: "00219"
slug: load-transfer-census-capture-recapture
title: >-
  How many large load transfers exist, estimated without labels by capture-recapture between
  detectors?
theme: switching-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private tables under data/ml-experiments/00219-load-transfer-census-capture-recapture/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  (the matching rule and the size cut-off are fixed in advance and varied as sensitivities).
  Callers: this folder's scripts only, which copy experiment 148's interval and end-rule code and read
  stored alarm tables of experiments 148, 156, 195 and 217.
hypothesis: >-
  Three detectors that read the same demand-model residual find overlapping but not identical sets of
  large load transfers. Capture-recapture on those sets, stratified by event size, estimates the
  number of large transfers and each detector's recall without any switching log.
prediction: >-
  The label-free estimate of RO-NC's recall is within 25% (relative) of its recall on the 35 logged
  transfers (0.51 by count, 0.657 weighted by MW moved), and the injected-event arm recovers the known
  injected count within 25% in at least two of the three size strata. It fails if the dependence
  between detectors (all read the same residual) makes the estimate far off.
decision_rule: >-
  Verdict `supported` when both parts hold, `partly supported` when exactly one holds, `refuted` when
  neither holds. The 35 logged transfers are read only after the estimates are frozen, and no
  threshold is tuned on them. Nothing is ported, so no effect-size threshold applies. The page names
  the strata in which the estimate is unstable and reports bounds for them.
benchmark: >-
  Recall of RO-NC, the pair alarm and the learned score on the 35 logged transfers, and the known
  injected count in experiment 195's held-out injected worlds.
row_set_id: none (event counts, not forecasts)
metric: >-
  Estimated number of large events, events per Primary-year, and each detector's estimated recall,
  with 95% intervals that resample whole calendar months.
---

**Planned, not yet run.** This page is the pre-registration. The write-up replaces it.

## Design (pre-registered)

- **Detectors.** The refit-offset neighbour-count detector (RO-NC) on the arm B residual, with
  experiment 148's 6-hour-lookahead end rule (stored intervals); the pair alarm of experiment 156
  (4-hour window); the learned re-ranker score of experiment 195 at its 80-unexplained-odd-month-alarm
  operating point. Ends for the pair alarm and the learned score come from the same end rule, rebuilt
  with copies of experiment 148's code.
- **Large event.** An interval of at least 3 hours whose step is at least 10% of the Primary's 99th
  percentile of cleaned demand (the maintainer-approved operational form in the edge-flow brief).
  Sensitivities: 5% and 20%.
- **Matching.** Two intervals are the same event when they share a Primary and overlap by at least
  50% of the shorter interval. A pair event is keyed on both Primaries.
- **Estimators.** Lincoln-Petersen N = n1 n2 / m for each pair of detectors (Chapman form as a
  sensitivity), and the three-list estimates (no-three-way-interaction, and Chao's lower bound).
  Strata: step size, duration, and metered partner or not.
- **Injected arm.** Experiment 195's held-out injected worlds, which give a found/not-found flag per
  injected event per detector. The same estimators run on those flags, and the estimate is compared
  with the known count. Caveat: those flags come from truth-aware matching, so the arm tests the
  estimator and not the blind matching rule.
- **Check at the end.** Each detector's label-free recall against its recall on the 35 logged
  transfers (RO-NC 0.51 by count, 0.657 MW-weighted; pair alarm 5 of 35; learned score 22 to 24 of
  35). The 35 enter no estimate.
