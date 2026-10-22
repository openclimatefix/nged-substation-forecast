---
id: "00212"
slug: switching-detector-masks-in-gam-training
title: >-
  Can alarms raised by a live switching detector replace hand labels when the forecast model is
  trained?
theme: lsc-detectors
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
  Pre-registered before any forecast score is read (revised after the plan review). Fits: experiment
  111's monthly refit schedule and `gamk` recipe, from its first cut to the 2026-06 cut, refitted at
  each of five substations (D5, D10, D11, D14, D16; D9 is dropped for compute) for each arm, with
  one thread per fit. Arms (a) and (c) replace `default_rows` of the frozen `SiteFeatures` with
  `dataclasses.replace`; no reviewed module changes. Positive control: arm (b) coefficients equal
  experiment 111's stored `gamk` fits to 1e-8 relative, and its CLEAN scores reproduce experiment
  203's K0 per-substation scores to the printed precision; otherwise the page is `invalid`. Arm (a)
  trains on the rows of arm (b) plus the rows inside the hand-filtered switching mask. Arm (c)
  trains on the rows of arm (a) minus the half-hours inside detector alarm intervals, padded by the
  same amount as the hand mask. Causality at a cut: an alarm is used only if its step is known by
  the cut (192 half-hours after the stamp for the re-ranker, experiment 203's rule); its end counts
  only if known by the cut (experiment 203's `known_from` logic, copied verbatim), otherwise the
  interval runs to the cut; an alarm raised in the last 4 days before a cut is unknown, so arm (c)
  equals arm (a) there. The interval starts at the step stamp (experiment 188). The fold that held
  out the substation's Primary pair supplies the alarms. Rows: CLEAN is experiment 203's ordinary
  rows (outside every logged transfer and its 72 hours after) of the scored year (runs from
  2025-07-01). POST is the ordinary rows, over all cached runs from 2024-04, whose latest monthly
  fit has, inside its training window, a logged transfer (the table experiment 203 uses, unpadded)
  that ended before the cut; POST is split into POST-OLD (the transfer ended more than 4 days
  before the cut, so the detector could know it) and POST-NEW (it ended within 4 days), and the
  headline is POST-OLD. Intervals for POST resample whole logged transfers (clusters), for CLEAN
  whole calendar months, 20,000 draws, 95%. Count of POST transfers and months is printed first.
  Score: twice the mean pinball loss over 11 quantiles, experiment 203's K0 spread and experiment
  111's Kalman hyperparameters (tuned with masks on; a limit), the residual stream rebuilt from each
  arm's own fits. Contrasts are differences of scores, positive meaning the first arm is worse.
  Injected worlds: experiment 195's `eval_held` worlds (`worlds/f<k>_eval_held_<w>_events.parquet`;
  never `eval_fitted`), fold k for the substations whose Primary pair fold k holds out, and the
  re-ranker's rows (`arm` = R re-ranker) of `injected_f<k>.parquet` for `found` and `onset_found`.
  Moved MW is derived from the event's depth_z with 195's injector and per-series noise scale,
  receivers included. At most 4 worlds per fold are refitted (compute cap). The MW is added to the
  training readings only; the Kalman filter and the scoring read the true readings, on CLEAN rows.
  In a world, arm (b) masks the true onset to end of each injected transfer plus the real labels,
  and arms (a) and (c) differ from the real-record arms only by the injected transfers: in a world,
  arm (a) still holds the real unmasked transfers, so the world's (a) minus (b) is reported net of
  the real record's (a) minus (b). Arm (c) in a world masks, from the step stamp to the known end,
  the injected transfers marked `found` (an event marked only `onset_found` is left unmasked, which
  understates arm (c)), plus the real-record alarms. This replays the detector's recall and leaves
  out its false alarms on injected readings. Verdict, an ordered table applied top to bottom.
  `invalid`: the positive control fails. `inconclusive`: fewer than 5 POST-OLD transfers, or (a)
  minus (b) is under 0.05 on POST-OLD and under 0.05 net in the worlds (nothing for a mask to
  repair). `not supported`: (a) minus (c) on POST-OLD is zero or negative. Otherwise three gates:
  G1, (c) minus (b) on CLEAN has a point estimate in [-0.02, +0.02] and a 95% interval inside
  +/-0.05; G2, (a) minus (c) on POST-OLD has a point estimate of at least +0.05 and an interval
  wholly above zero; G3, in the worlds (c) minus (b) is under 0.05 net and the interval of (a) minus
  (c) is wholly above zero. `supported` when G1, G2 and G3 all hold; `partly supported` otherwise,
  naming the failed gates. A contrast whose interval is wider than +/-0.05 is called "not resolved"
  in the text and cannot pass a gate. The 0.02 and 0.05 limits are the brief's and are set by the
  cost of replacing a hand label with a detector in the training loop. Reported beside, not bars:
  (a) minus (b) and (a) minus (c) on CLEAN; arm (c) split by odd and even months of the learned
  score's training; winter (November to March) CLEAN rows; tail coverage and pinball loss at q01,
  q05, q95, q99 (`studies.tail_calibration`); error of the mean forecast beside each contrast;
  leave-one-substation-out; median minus mean forecast per site; half-hours dropped per arm and
  substation, the share of logged-transfer half-hours arm (c) covers, and the share of its dropped
  half-hours outside logged transfers. Post hoc and labelled: a control that builds experiment
  195's residual stream from arm (a)'s fits if its code allows it.
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

## Lab shortcuts

- **The alarms are not label-free.** Experiment 195's candidate features come from forecast errors
  of a GAM trained with the labelled transfers masked, so arm (c) receives the hand labels
  indirectly. The conclusion is that the detector replaces labels given label-masked residuals.
- **The learned score was trained on the even months of the whole record,** including time after
  an early cut, and the report splits arm (c) by odd and even months.
- **The 80-alarm threshold is label-calibrated:** the unexplained-alarm budget uses the logged
  transfers. At the scale of version 2 there are no labels, so that budget does not exist.
- **The Kalman hyperparameters and step penalty were tuned with the masks on.**

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
