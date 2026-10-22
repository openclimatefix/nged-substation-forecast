---
id: "00212"
slug: switching-detector-masks-in-gam-training
title: >-
  Can alarms raised by a live switching detector replace hand labels when the forecast model is
  trained?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private fits and score tables under
  data/ml-experiments/00212-switching-detector-masks-in-gam-training/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for when a detector
  alarm is known at a training cut and for how injected transfers are scored, so both are fixed
  below before any score is read. Callers: this folder's scripts only; they copy the modules of
  experiments 111, 188, 195, and 203 and call the reviewed `studies.*` modules without changing them.
hypothesis: >-
  A convex additive model of substation demand (GAM) is trained on readings that include switching
  transfers, which move load between substations for hours or days. Every earlier GAM in this
  project trained with the logged and hand-labelled transfers set missing, so nobody has measured
  what the masking is worth. If masking the detector's own alarms during training does about as well
  as masking the hand labels, a live service with no labels can train as well as the project does
  now.
prediction: >-
  Written before any score is read. Arms are (a) no training mask, (b) the logged and hand-labelled
  transfers masked, (c) the alarm intervals of experiment 195's learned score masked, causally.
  Scored with the Kalman-filtered GAM (GAM-K) forecast and experiment 203's spread, in points of
  effective capacity (lower is better), pooled over D5, D10, D11, D14, and D16. Arm (c) minus arm (b)
  is within +/-0.02 points on clean held-out rows. Arm (a) minus arm (c) is at least +0.05 points on
  the rows after a logged transfer. The prediction fails if arm (c) is no better than arm (a).
decision_rule: >-
  Pre-registered before any forecast score is read. Fits: experiment 111's monthly refit schedule
  and `gamk` recipe, from its first cut to the 2026-06 cut, refitted at each of the five
  substations for each arm. The positive control is arm (b): its refitted coefficients must equal
  experiment 111's stored `gamk` fits and its scores must reproduce experiment 203's K0 scores on
  the same rows to the printed precision; otherwise the page is `invalid`. Arm (a) trains on the
  same rows as arm (b) plus the rows inside the hand-filtered switching mask. Arm (c) trains on the
  rows of arm (a) minus the half-hours inside detector alarm intervals. An interval enters a
  cut's training only if the alarm is known by the cut: a step is known 192 half-hours (4 days)
  after its stamp for the re-ranker (experiment 203's rule), and an interval that is still open at
  the cut ends at the cut. The interval starts at the step stamp, as experiment 188 found best.
  The fold that held out the substation's Primary pair supplies the alarms. Rows: CLEAN is
  experiment 203's ordinary rows (outside every logged transfer and its 72 hours after) of the
  scored year (runs from 2025-07-01); POST is the ordinary rows, over all cached runs from 2024-04,
  whose latest monthly fit has a logged transfer that ended in the 62 days before its cut inside its
  training window. Score: twice the mean pinball loss over 11 quantiles, with experiment 203's K0
  spread and the Kalman hyperparameters of experiment 111, so contrasts are in location only; the
  Kalman residual stream is rebuilt from each arm's own fits. Intervals: 95%, rows resampled by
  calendar month, 20,000 draws. G1: the point estimate of (c) minus (b) on CLEAN lies in
  [-0.02, +0.02]. G2: the point estimate of (a) minus (c) on POST is at least +0.05. The interval of
  each contrast is reported, and a contrast whose interval is wider than +/-0.05 is called "not
  resolved" in the text, whatever its point estimate. Injected worlds (second evidence, the true
  normal running arrangement): the evaluation worlds of experiment 195 (fold k holds out the
  substation's Primary pair) supply drawn transfers on the five substations' series. Each world's
  moved MW is added to the training readings only, the forecasts are scored against the true
  readings on CLEAN rows, and arm (b) in a world masks the true onset to end of every injected
  transfer plus the real labels. Arm (c) in a world masks, from onset to the later of the end and
  the known time, the injected transfers that experiment 195's `injected_f<k>.parquet` marks as
  found by the re-ranker (`found`, with the onset located as `onset_found`), plus the real-record
  alarm intervals. This replays the detector's recall and leaves out its false alarms on injected
  readings; the false alarms of the real record enter through the real alarm intervals. G3: (a)
  minus (b) in the worlds is at least +0.05 for the worlds to count (otherwise the injection did
  not damage the fit and the world result is reported as uninformative), and (c) closes at least
  half of that gap. Verdict: `supported` when G1 and G2 hold; `partly supported` when exactly one
  holds; `not supported` when (a) minus (c) on POST is zero or negative (the brief's failure case);
  `inconclusive` when (a) minus (b) is under 0.02 on both POST and the worlds (nothing for a mask to
  repair), or when POST holds fewer than 5 logged transfers. The 0.02 and 0.05 limits are the
  brief's and are set by the cost of replacing a hand label with a detector in the training loop.
  Reported beside, not bars: winter (November to March) CLEAN rows separately; tail coverage and
  pinball loss at q01, q05, q95, q99 (`studies.tail_calibration`); error of the mean forecast beside
  each contrast; leave-one-substation-out; median minus mean forecast per site; number of training
  half-hours dropped per arm and substation; share of arm (c)'s dropped half-hours that lie outside
  logged transfers.
benchmark: >-
  Arm (b), the logged and hand-labelled transfers masked, which is experiment 111's `gamk` fit; arm
  (a) is the new no-mask comparator.
row_set_id: >-
  experiment 203's ordinary rows of the scored year (CLEAN) and the post-transfer rows of all cached
  runs from 2024-04 (POST) at five decision Primary substations; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity.
code: ml-experiments/00212-switching-detector-masks-in-gam-training/
date: 2026-10-22
---

**Planned. This page holds the pre-registered plan; the write-up replaces it when the fits finish.**

A "mask" sets readings missing so that a fit or a filter ignores them. A switching transfer moves a
load from one Primary substation to its neighbours for hours or days, so the readings during it do
not describe the substation's normal running arrangement. Every GAM fitted so far already masked the
logged and hand-labelled transfers, so no experiment has measured what that masking is worth, and
none has asked whether a detector that only sees power could do the same job. This experiment
refits the GAM-K forecast of [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md) three ways
(no training mask, the hand labels masked, the alarms of the learned score of [experiment
195](../00195-switching-neural-plus-detector-features/result.md) masked) and scores each on clean
held-out rows and on rows after a logged transfer. It repeats the comparison on drawn transfers
added to the training readings, scored against the true readings.

The replay of the alarms reuses the stamp offset, the known-time rule, and the end rule of
[experiment 203](../00203-switching-replay-learned-score-in-masked-forecast/result.md) and the onset
stamp of [experiment
188](../00188-switching-d10-audit-and-onset-location-in-look-back/result.md).

## Five size triggers

- **Stored:** private fits and score tables only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one design:** yes, for when an alarm is known at a cut and how injected transfers
  are scored; both are fixed above.
- **Callers:** this folder's scripts only.

## Earlier advice followed

- Experiment 203: test the detector's alarms before judging them on forecast error (followed:
  training mask, causal).
- Experiment 203: drop scoring windows that overlap the next transfer (followed: CLEAN excludes
  every logged transfer and the 72 hours after).
- Experiment 203: report harm on odd and even months separately (followed: odd and even months of
  the real-record alarms are reported, because the learned score trained on the even months'
  candidates).
