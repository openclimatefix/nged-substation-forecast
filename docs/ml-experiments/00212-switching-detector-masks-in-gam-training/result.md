---
id: "00212"
slug: switching-detector-masks-in-gam-training
title: >-
  Can alarms raised by a live switching detector replace hand labels when the forecast model is
  trained?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
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
verdict: inconclusive
best_arm: >-
  None. Arm B (hand labels masked) scores 2.789 points of effective capacity on clean rows; arm A
  (no training mask) 2.791 and arm C (detector alarms masked) 2.796.
headline: >-
  Training with no mask at all costs 0.002 points on clean rows (-0.003 to +0.007) and 0.003 points
  on rows after a logged transfer (-0.000 to +0.006), so there is almost nothing for a detector mask
  to repair, and the 0.05-point bar cannot be reached on real data. Injected transfers cost 0.009
  points (+0.004 to +0.015).
lessons: >-
  Hand labels cover 0 to 2,406 of about 35,000 training half-hours per substation, and the 24-month
  window plus the ridge penalty dilute a transfer's pull on the coefficients. The detector's alarms
  match the hand mask poorly (0% to 67% of it covered), and replacing the hand labels with them
  costs 0.007 points on clean rows, 0.027 in winter.
take_home_message: >-
  Masking switching transfers when the forecast model is trained barely matters: with no mask the
  score is only 0.002 points worse on clean rows and 0.003 worse after a logged transfer, and drawn
  transfers added to the training readings cost just 0.009 points. Training with the
  detector's alarms masked scores 0.007 points worse than with hand labels masked on clean rows, so
  a live detector can replace hand labels at this cost. The question this experiment asked has
  nothing to answer on this record, because the unmasked fit is already almost as good, so the
  detector has to be judged on what it does to the forecast's recent-power input, not on training.
advice:
  - >-
    Do not spend experiments on training-time masks for switching transfers with the 24-month
    convex GAM: no mask costs 0.002 to 0.009 points (headline contrasts, figure 1).
  - >-
    Judge a switching detector on the mask in the recent-power correction (experiments 163 and 171),
    where masking changes the score by several points, not on GAM training (figure 1).
  - >-
    Add an unmasked arm whenever a training mask is justified by earlier fits that already used it:
    arm A here is new and shows the mask is worth 0.002 points (Results).
  - >-
    Treat the real record's contrasts as small by construction: the hand mask is at most 7% of the
    training half-hours (Results).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**Masking switching transfers when the forecast model is trained barely matters, so a detector has
almost nothing to replace.** A switching transfer moves load between Primary substations for hours
or days. Every earlier forecast in this project trained with the logged and hand-labelled transfers
set missing, and nobody had measured what that was worth. We refitted experiment 111's Kalman-filtered
additive model (GAM-K) three ways at five Primary substations: with no training mask (arm A), with
the hand labels masked (arm B, experiment 111's own fits), and with the alarms of [experiment
195](../00195-switching-neural-plus-detector-features/result.md)'s learned score masked (arm C),
known as a live service would know them. We scored each on clean held-out rows and on rows after a
logged transfer. We repeated the comparison on 476 drawn transfers added to the training readings
and scored against the true readings. Arm A scores 0.002 points of effective capacity worse than arm
B on clean rows and 0.003 worse on the rows after a logged transfer. Drawn transfers raise that to
0.009. All three are far below the 0.05-point bar that the plan set, so the verdict is
`inconclusive`: nothing for a mask to repair, rather than a detector that fails.

**The replay builds on three earlier experiments.** The alarm intervals, the 4-day delay before an
alarm is known, and the end rule come from [experiment
203](../00203-switching-replay-learned-score-in-masked-forecast/result.md). The step stamp as the
mask's start comes from [experiment
188](../00188-switching-d10-audit-and-onset-location-in-look-back/result.md). The refit schedule
and the GAM-K recipe come from [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md).

![Score difference between training masks, with 95% intervals](figure_1_contrasts.svg)

## Results

- **Verdict `inconclusive` under the pre-registered table.** The positive control passes: arm B's
  coefficients equal experiment 111's stored fits exactly (maximum difference 0) at all 27 monthly
  cuts of the five substations, and its scores equal experiment 203's K0 scores row by row
  (maximum difference 0). Arm A minus arm B is under 0.05 on post-transfer rows (+0.0025) and in
  the worlds (+0.0091 net of the real record), which is the table's first rule.
- **Gates, reported although the table stops earlier.** G1 holds: arm C minus arm B on clean rows is
  +0.0065 (-0.0069 to +0.0246). G2 fails: arm A minus arm C on post-transfer rows is +0.0005 (-0.0077
  to +0.0072), against the 0.05 bar. G3 holds on its two conditions (arm C minus B +0.0051, A minus
  C +0.0040 with an interval of +0.00003 to +0.0093), but the worlds' gap that arm C could close
  is only 0.009.
- **Absolute scores (mean quantile score, points of effective capacity).** Clean rows (scored year,
  462,137 rows): A 2.7913, B 2.7890, C 2.7955. Post-transfer rows, transfers known to the detector
  (1,130,781 rows, 17 logged transfers): A 2.8215, B 2.8191, C 2.8210. Post-transfer rows, transfers
  that ended within 4 days of the cut (14,895 rows, 4 months): A 3.5462, B 3.6304, C 3.5452.
- **Row sets.** Clean rows are experiment 203's ordinary rows of the scored year (outside every
  logged transfer and 72 hours after). Post-transfer rows are every cached run from 2024-04 whose
  latest monthly fit has, inside its 24-month window, a logged transfer that ended at least 4 days
  before the cut (a transfer the detector can know about); they are 86% of the ordinary rows, so
  the contrast is diffuse, not local to the weeks after a transfer. The 4-day rows are too few
  (4 months) to resolve anything: arm A minus B is -0.0842 (-0.2209 to -0.0064) there, arm B scoring
  worse than arm A.
- **Why the masks matter so little.** The hand mask holds 0 to 2,406 half-hours in the 24-month
  window at the last cut (D5 2,406, D10 1,044, D11 0, D14 592, D16 2,085) out of about 35,000
  training half-hours, which is why arm A trains on exactly that many more half-hours than arm B.
  The ridge penalty and the long window dilute a transfer's pull on the coefficients. Arm C's alarms
  cover the hand mask poorly (share of it covered, per fold: D5 41%, D10 48% to 52%, D11 no hand mask,
  D14 32% to 43%, D16 67%) and mostly lie outside it (50% to 100% of an alarm's half-hours).
- **Detector against hand labels.** Arm C minus arm B on clean rows is +0.0065 (-0.0069 to
  +0.0246); each of the three folds gives +0.0057, +0.0070, or +0.0068. In winter (November to
  March, 182,832 rows) the contrast is +0.0267 (-0.0015 to +0.0632), outside the plan's +/-0.02
  band on its point estimate. On the odd months that the learned score was trained on it is
  +0.0167 (-0.0047 to +0.0483), and on the even months -0.0036 (-0.0153 to +0.0102).
- **Injected worlds.** 12 of experiment 195's `eval_held` worlds (3 folds, 4 worlds each) put 476
  drawn transfers on 47 substation-and-world cells; the re-ranker found 125 of them. Net of the
  real record (each arm against the real-record arm of the same fold), arm A minus arm B is +0.0091
  (+0.0039 to +0.0150), arm A minus arm C +0.0040 (+0.00003 to +0.0093), arm C minus arm B +0.0051
  (+0.0013 to +0.0101). One cell dominates arm A minus B:
  worlds with 17 to 19 transfers on D5 cost 0.10 and 0.13 points, and the other cells sit below
  0.05, so the average rests on a few D5 worlds.
- **Checks: mean-forecast error.** The error of the mean forecast points the same way as every
  quantile-score contrast on clean rows: C minus B +0.0061 (-0.0198 to +0.0382), A minus B +0.0053
  (-0.0034 to +0.0135), A minus C -0.0008 (-0.0404 to +0.0312).
- **Checks: median minus mean forecast per site (clean rows, points).** B: D5 -0.1345, D10 -0.0639,
  D11 -0.0266, D14 +0.0331, D16 +0.0144; A differs from B by at most 0.0006.
- **Checks: leave one substation out.** Arm C minus B on clean rows stays between +0.0006 and
  +0.0105 and positive without each substation. Arm A minus B changes sign without D16 (-0.0005)
  and stays between +0.0028 and +0.0035 otherwise. Arm A minus C on post-transfer rows runs from -0.0024
  (without D16) to +0.0037 (without D11) and changes sign without D5 or D16; its interval includes
  zero in every case.
- **Checks: audit triggers.** Trigger 2 fired in form: the result contradicts the prediction that
  arm A minus arm C would reach 0.05. The explanation (the hand mask is at most 7% of training
  half-hours) is tested by the worlds, where 476 drawn transfers still cost only 0.009 points.
  Trigger 4 fired for arm A minus B without D16 and for arm A minus C on
  post-transfer rows (G2) without D5 or D16, sign flips of contrasts under 0.003 that sit far from
  the 0.05 bar either way; the verdict is unchanged. No audit was run.
- **Tail calibration on clean rows (share of readings below the quantile; nominal 0.01, 0.05, 0.95,
  0.99).** A: 0.0150, 0.0484, 0.9564, 0.9873. B: 0.0148, 0.0478, 0.9564, 0.9872. C (fold 0): 0.0150,
  0.0480, 0.9555, 0.9865. Pinball loss at those quantiles and the curtailment-cost saving were not
  computed, because the contrasts are under 0.01 points.
- **Schedule and lab shortcuts.** Monthly refits on a 24-month window (experiment 111's schedule),
  which a weekly schedule could change only by refitting more often on the same rows. The alarms are
  not label-free: experiment 195's features come from the residual of a GAM trained with the
  labelled transfers masked, its score was trained on the even months of the whole record, and its
  80-alarm threshold counts alarms the logged transfers do not explain. Deviations from the plan: D9
  is not scored; the detector's intervals are not padded like the hand mask (the cache does not
  record the hand mask's padding); in the worlds a receiver takes an equal share of the moved MW,
  the random drift of experiment 195's profile is omitted, and 4 of the 6 worlds per fold are used.
  The post-transfer interval resamples the 17 logged transfers (clusters), the clean interval
  resamples 12 calendar months. Arm C is the mean of the scores of the three folds' alarm tables
  (as experiment 203 scores ordinary rows), because a substation sits in pairs of several folds;
  the three folds' own contrasts are listed above. An alarm known exactly at the cut counts as
  known (one half-hour, copied from experiment 203). The positive controls are in
  `printed_report.txt` and in the fit script's `--control` output.
- **What this narrows.** A training-time mask is not where a detector earns its keep. The forecast's
  recent-power correction is: experiments 163 and 171 found several points there. The next
  experiment is to judge the live detector only by that mask, and to learn how the unmasked GAM
  copes at the one site (D5) where large transfers cost up to 0.13 points.
