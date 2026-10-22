---
id: "00203"
slug: switching-replay-learned-score-in-masked-forecast
title: >-
  Do the extra real transfers that experiment 195's learned score finds lower the masked forecast's
  error, without harm on ordinary days?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00203-switching-replay-learned-score-in-masked-forecast/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible for when a
  learned alarm counts as known and for how its end is known, so both are fixed below before any
  score is read. Callers: this folder's scripts only; they are copies of the modules of experiments
  163, 171, and 188, and read the stored alarm tables of experiments 147 and 195 and the stored
  scores of experiments 163 and 171.
hypothesis: >-
  Experiment 195's learned score finds 22 to 24 of the 35 real transfers lasting over 1 day,
  against 19 for the refit-offset neighbour-count detector (RO-NC). If its alarms drive the gated
  mask of experiments 163 and 171 (readings set missing from the detector's step stamp until the
  transfer ends), the forecast scores better after real transfers than the pair-alarm mask does,
  and the mask still does no harm on ordinary days. The mask ignores recent power inside an
  alarm, so extra true alarms should help and extra false alarms should harm.
prediction: >-
  Written before any forecast score is read. On the real transfers of experiment 163 (the 72 hours
  after each logged transfer ends), the learned-score mask scores below the pair-alarm mask's 6.32
  points of effective capacity by at least 0.3 points (best guess 2: 10.09; no mask: 13.69). On
  the pooled rows outside logged transfers (27 months, six Primary substations), the learned-score
  gated mask minus no mask has a 95% interval inside +/-0.05 points (experiment 171's pair-alarm
  gated mask: +0.030, -0.011 to +0.074, a 0.085-wide interval). The prediction fails if the extra
  found transfers do not lower the score, or if the harm interval stays wider than +/-0.05.
decision_rule: >-
  Pre-registered before any forecast score is read. Alarm arms (all from experiment 195's
  stored `alarms_f<k>.parquet`; the three folds hold out whole Primary pairs): LEARNED_R (the "R
  re-ranker", 24 of 35 found; primary), LEARNED_F (the "F logistic" regression, 22 of 35;
  secondary), and RONC_195 (195's re-run of RO-NC, 19 of 35; control). Onset: the alarm's step
  stamp (experiment 188 found that masking from the step stamp removes 95% of the extra error of
  masking from the alarm time, and that searching the look-back adds nothing). Causal only: an
  alarm is known 96 half-hours (2 days) after its step for LEARNED_F and RONC_195, which is
  experiment 156's RO-NC lag and the reach of the `persist` feature, and 192 half-hours (4 days)
  after its step for LEARNED_R, whose raw window reads 4 days after the step. The mask begins at
  the first cached forecast run whose newest usable stamp is at or after the known stamp. The
  cached runs are daily (09:00 UTC), not 6-hourly, because the cached GAM-K member forecasts exist
  only for those runs; a 6-hourly replay needs new ensemble forecasts and is not run. End: the
  alarm's `end_est`, known at the later of experiment 147's stored live `known_from` for the same
  (series, step) when one exists and `end_est` + 96 half-hours; this is the lab shortcut of using
  experiment 147's end rule for alarms it did not raise. Each fold's alarm table is replayed
  separately. A logged transfer is scored with the fold that holds out its Primary pair. A row on
  an ordinary day is scored with all three folds' tables and the three differences are averaged
  row by row (the mask of a learned score trained without that row's pairs). Rows and scores:
  experiment 163's AFTER set (the 72 hours after each logged transfer ends; rows with a reading, on
  runs of the scored year and replicate months) is the headline. Its "shared" scope, the rows
  where best guess 2 also has a score, holds 7 transfers; its "stack-only" scope holds 11. Ordinary
  days are experiment 171's rows: every cached run (2024-04-10 to 2026-06-29) at the six decision
  substations (D5, D9, D10, D11, D14, D16), outside every logged transfer and its 72 hours after.
  Comparison arms are K0 (no mask), PAIR_NEW (every pair alarm, experiment 163's
  MASK_NEW) and PAIR_GATED (balance score of 0.7 or more, experiment 171's MASK_GATED), recomputed
  here on the same rows, and best guess 2, read from experiment 87's stored predictions. Positive
  control: PAIR_NEW, K0, and the gated contrast must reproduce experiment 163's 6.321, 13.686 and
  10.091 (best guess 2) on the shared AFTER rows, and experiment 171's +0.030 on the pooled rows,
  to the printed precision; if they do not, the page is `invalid`. Score: twice the mean pinball loss over 11 quantiles in
  points of effective capacity, K0 spread, GAM-K cached members plus the frozen Kalman filter.
  Gate of the harm test: a learned alarm masks whenever it is active (the learned score already
  has its threshold at 80 unexplained odd-month alarms; the balance gate belongs to the pair
  alarm). Intervals: for the gain, transfers resampled whole (95%, 20,000 draws); for the harm,
  rows resampled by whole calendar month as in experiment 171. G1 (gain, the brief's bar): LEARNED_R minus the
  pair-alarm mask on the AFTER rows of 163's 7 transfers (those with best guess 2 scores) has a
  point estimate of -0.3 points or lower. Experiment 163's oracle mask (true onset and end known
  at once) scores 6.354 on those rows, against 6.321 for the pair-alarm mask, so G1 requires the
  learned mask to beat the oracle by 0.33 points; it is kept because the brief states it, and the
  page reports it as the oracle-capped bar. G1' (gain, where the oracle leaves room): the same
  contrast on the AFTER rows of all 11 transfers of 163's stack-only set (6.083 for the pair-alarm
  mask, 4.79 for the oracle), with the point estimate -0.3 points or lower. The intervals of G1
  and G1' are reported, and with 7 and 11 transfers neither is expected to exclude zero. G2
  (no harm): the pooled LEARNED_R gated mask minus K0 on ordinary days has a 95% interval inside
  -0.05 to +0.05. Verdict: `supported` when (G1 or G1') and G2 hold, `partly supported` when exactly one of
  (G1 or G1') and G2 holds, `not supported` when neither holds, and `inconclusive` only for a failed fit or fewer
  than 5 scored transfers. The threshold of 0.3 points is the brief's and
  the harm limit of 0.05 is the brief's; both are set by the cost of porting a learned score,
  which adds a model and a pipeline step to the serving path. Reported beside the contrast (not
  bars): winter (November to March) rows separately for the harm test; the same contrasts for
  LEARNED_F and RONC_195; the larger set of logged transfers; the number of alarms, masked
  half-hours, and share of masked half-hours outside logged transfers per substation and fold;
  tail coverage and pinball loss at q01, q05, q95, q99 of the headline arms on the AFTER rows
  (`studies.tail_calibration`); leave-one-substation-out for both contrasts; median forecast minus
  mean forecast per site. Post hoc and labelled: LEARNED_R with the 96 half-hour lag (if the 2
  extra days of look-ahead were available). Curtailment-cost saving is not reported, because the
  mask changes the forecast of six Primary substations' demand and the node-curtailment metric
  needs a node with generators.
benchmark: >-
  The pair-alarm mask (6.32 points on the AFTER rows of the 7 transfers), best guess 2 (10.09), and
  no mask (13.69), from experiment 163; the pair-alarm gated mask minus no mask on ordinary days
  (+0.030, -0.011 to +0.074), from experiment 171.
row_set_id: >-
  experiment 163's AFTER rows (scored year and replicate months, 7 transfers) and every cached run
  (2024-04-10 to 2026-06-29) of six Primary substations; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity.
code: ml-experiments/00203-switching-replay-learned-score-in-masked-forecast/
---

**Planned.** The write-up is added when the replay has run.
