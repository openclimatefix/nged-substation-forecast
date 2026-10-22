---
id: "00203"
slug: switching-replay-learned-score-in-masked-forecast
title: >-
  Do the extra real transfers that experiment 195's learned score finds lower the masked forecast's
  error, without harm on ordinary days?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00203-switching-replay-learned-score-in-masked-forecast/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible for when a
  learned alarm counts as known and for how its end is known, so both are fixed below before any
  score is read. Callers: this folder's scripts only; they are copies of the modules of experiments
  163, 171, and 188, and read the stored alarm tables of experiments 147 and 195 and the stored
  scores of experiments 163 and 171.
hypothesis: >-
  Experiment 195's learned score finds 22 to 24 of the 35 real transfers lasting over 1 day, against
  19 for the refit-offset neighbour-count detector (RO-NC). If its alarms drive the gated mask of
  experiments 163 and 171 (readings set missing from the detector's step stamp until the transfer
  ends), the forecast scores better after real transfers than the pair-alarm mask does, and the mask
  still does no harm on ordinary days. The mask ignores recent power inside an alarm, so extra true
  alarms should help and extra false alarms should harm.
prediction: >-
  Written before any forecast score is read. On the real transfers of experiment 163 (the 72 hours
  after each logged transfer ends), the learned-score mask scores below the pair-alarm mask's 6.32
  points of effective capacity by at least 0.3 points (best guess 2: 10.09; no mask: 13.69). On the
  pooled rows outside logged transfers (27 months, six Primary substations), the learned-score gated
  mask minus no mask has a 95% interval inside +/-0.05 points (experiment 171's pair-alarm gated
  mask: +0.030, -0.011 to +0.074, a 0.085-wide interval). The prediction fails if the extra found
  transfers do not lower the score, or if the harm interval stays wider than +/-0.05.
decision_rule: >-
  Pre-registered before any forecast score is read. Alarm arms (all from experiment 195's stored
  `alarms_f<k>.parquet`; the three folds hold out whole Primary pairs): LEARNED_R (the "R
  re-ranker", 24 of 35 found; primary), LEARNED_F (the "F logistic" regression, 22 of 35;
  secondary), and RONC_195 (195's re-run of RO-NC, 19 of 35; control). Onset: the alarm's step stamp
  (experiment 188 found that masking from the step stamp removes 95% of the extra error of masking
  from the alarm time, and that searching the look-back adds nothing). Causal only: an alarm is
  known 96 half-hours (2 days) after its step for LEARNED_F and RONC_195, which is experiment 156's
  RO-NC lag and the reach of the `persist` feature, and 192 half-hours (4 days) after its step for
  LEARNED_R, whose raw window reads 4 days after the step. The mask begins at the first cached
  forecast run whose newest usable stamp is at or after the known stamp. The cached runs are daily
  (09:00 UTC), not 6-hourly, because the cached GAM-K member forecasts exist only for those runs; a
  6-hourly replay needs new ensemble forecasts and is not run. End: the alarm's `end_est`, known at
  the later of experiment 147's stored live `known_from` for the same (series, step) when one exists
  and `end_est` + 96 half-hours; this is the lab shortcut of using experiment 147's end rule for
  alarms it did not raise. Each fold's alarm table is replayed separately. A logged transfer is
  scored with the fold that holds out its Primary pair. A row on an ordinary day is scored with all
  three folds' tables and the three differences are averaged row by row (the mask of a learned score
  trained without that row's pairs). Rows and scores: experiment 163's AFTER set (the 72 hours after
  each logged transfer ends; rows with a reading, on runs of the scored year and replicate months)
  is the headline. Its "shared" scope, the rows where best guess 2 also has a score, holds 7
  transfers; its "stack-only" scope holds 11. Ordinary days are experiment 171's rows: every cached
  run (2024-04-10 to 2026-06-29) at the six decision substations (D5, D9, D10, D11, D14, D16),
  outside every logged transfer and its 72 hours after. Comparison arms are K0 (no mask), PAIR_NEW
  (every pair alarm, experiment 163's MASK_NEW) and PAIR_GATED (balance score of 0.7 or more,
  experiment 171's MASK_GATED), recomputed here on the same rows, and best guess 2, read from
  experiment 87's stored predictions. Positive control: PAIR_NEW, K0, and the gated contrast must
  reproduce experiment 163's 6.321, 13.686 and 10.091 (best guess 2) on the shared AFTER rows, and
  experiment 171's +0.030 on the pooled rows, to the printed precision; if they do not, the page is
  `invalid`. Score: twice the mean pinball loss over 11 quantiles in points of effective capacity,
  K0 spread, GAM-K cached members plus the frozen Kalman filter. Gate of the harm test: a learned
  alarm masks whenever it is active (the learned score already has its threshold at 80 unexplained
  odd-month alarms; the balance gate belongs to the pair alarm). Intervals: for the gain, transfers
  resampled whole (95%, 20,000 draws); for the harm, rows resampled by whole calendar month as in
  experiment 171. G1 (gain, the brief's bar): LEARNED_R minus the pair-alarm mask on the AFTER rows
  of 163's 7 transfers (those with best guess 2 scores) has a point estimate of -0.3 points or
  lower. Experiment 163's oracle mask (true onset and end known at once) scores 6.354 on those rows,
  against 6.321 for the pair-alarm mask, so G1 requires the learned mask to beat the oracle by 0.33
  points; it is kept because the brief states it, and the page reports it as the oracle-capped bar.
  G1' (gain, where the oracle leaves room): the same contrast on the AFTER rows of all 11 transfers
  of 163's stack-only set (6.083 for the pair-alarm mask, 4.79 for the oracle), with the point
  estimate -0.3 points or lower. The intervals of G1 and G1' are reported, and with 7 and 11
  transfers neither is expected to exclude zero. G2 (no harm): the pooled LEARNED_R gated mask minus
  K0 on ordinary days has a 95% interval inside -0.05 to +0.05. Verdict: `supported` when (G1 or
  G1') and G2 hold, `partly supported` when exactly one of (G1 or G1') and G2 holds, `not supported`
  when neither holds, and `inconclusive` only for a failed fit or fewer than 5 scored transfers. The
  threshold of 0.3 points is the brief's and the harm limit of 0.05 is the brief's; both are set by
  the cost of porting a learned score, which adds a model and a pipeline step to the serving path.
  Reported beside the contrast (not bars): winter (November to March) rows separately for the harm
  test; the same contrasts for LEARNED_F and RONC_195; the larger set of logged transfers; the
  number of alarms, masked half-hours, and share of masked half-hours outside logged transfers per
  substation and fold; tail coverage and pinball loss at q01, q05, q95, q99 of the headline arms on
  the AFTER rows (`studies.tail_calibration`); leave-one-substation-out for both contrasts; median
  forecast minus mean forecast per site. Post hoc and labelled: LEARNED_R with the 96 half-hour lag
  (if the 2 extra days of look-ahead were available). Curtailment-cost saving is not reported,
  because the mask changes the forecast of six Primary substations' demand and the node-curtailment
  metric needs a node with generators.
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
verdict: partly supported
best_arm: >-
  None reliably. The re-ranker mask (alarms known 4 days after the step) scores 5.623 points of
  effective capacity on the 72 hours after 11 real transfers against 6.083 for the pair-alarm mask
  (-0.460, 95% interval -3.505 to +3.718), but +0.989 (-2.229 to +6.963) on the 7 transfers that
  best guess 2 also scores.
headline: >-
  Masking with experiment 195's learned alarms did not lower the error after real transfers by a
  reliable margin (11 transfers -0.460, interval -3.505 to +3.718; 7 transfers +0.989), and it did
  not meet the no-harm limit on ordinary days (+0.031, -0.020 to +0.086, against +/-0.05). The sign
  of the transfer contrasts rests on three transfers, one of which is scored inside the next
  transfer.
lessons: >-
  The learned alarms are known 2 to 4 days after the step, so on long transfers they fire near the
  end, while the pair alarm fires at the start. Finding more transfers in experiment 195's scoring
  did not become a lower forecast error. The 72 hours after one 580-hour transfer lie inside the
  next transfer, which makes the 7-transfer contrast unreliable. The learned scores were trained
  partly on the ordinary-day rows they are scored on.
take_home_message: >-
  Masking the forecast with the alarms of experiment 195's learned score did not lower the error
  after real switching transfers by a reliable margin, and it did not meet the limit for no harm on
  ordinary days. The learned alarms arrive 2 to 4 days after a step, so they miss the start of the
  long transfers that the pair alarm catches. A candidate next step is a mask that takes the pair
  alarm where it fires and the learned alarm elsewhere.
advice:
  - >-
    Avoid a learned alarm that reads days after a step as the only mask trigger for long transfers:
    the re-ranker's alarm is known before the transfer ends for 8 of 45 logged transfers (Results,
    one-transfer bullet).
  - >-
    Test a union of the pair alarm and the learned alarms in the masked forecast before judging a
    learned score on forecast error (What this narrows).
  - >-
    Drop scoring windows that overlap the next transfer on the same series before comparing masks
    after transfers: one overlap decides the 7-transfer contrast (Results).
  - >-
    Report the harm separately on the months a learned score was trained on and on the other months
    (Results, ordinary-days bullet).
  - >-
    Expect to need far more than 13 transfers to exclude a 0.3 point difference: the 13-transfer
    interval is 5.8 points wide (figure 1).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**Masking the forecast with the alarms of experiment 195's learned score did not lower the error
after real switching transfers by a reliable margin, and it did not meet the no-harm limit on
ordinary days.** A switching transfer moves load between Primary substations for hours or days. A
"mask" sets the forecast's recent power readings aside while an alarm says a transfer is under way.
The pair alarm (a step at one substation matched by an opposite step at a neighbour) fed the mask of
experiment 163, and experiment 195's learned score, a statistical score trained to flag transfers,
was meant to find more transfers. On the 7 logged transfers that best guess 2 (the best-guess
forecast of experiment 87) also scores, the learned-score mask scored 0.99 points of effective
capacity worse than the pair-alarm mask (7.31 against 6.32). On the wider set of 11 transfers it
scored 0.46 points better (5.62 against 6.08), but the 95% interval for that difference runs from
-3.5 to +3.7. The 11-transfer sign flips when the substation labelled D16 is left out. The pattern
in the transfers is consistent with this explanation: the learned alarms often fire late in a
transfer or near its end, while the pair alarm fires at the start of the long ones.

**Switching transfers contaminate the forecast's recent-power correction.** The forecast is a
generalised additive model with a Kalman filter ([Kalman, 1960](https://doi.org/10.1115/1.3662552))
on its recent errors (GAM-K, as in [experiment
163](../00163-switching-real-event-comparison-gam-stack-against-bg2/result.md)). Experiment 163
found that masking while the pair alarm is active brings the error after real transfers down to that
of a forecast with no recent power. [Experiment
195](../00195-switching-neural-plus-detector-features/result.md) found that a learned score on the
detectors' features matches 22 to 24 of 35 real transfers lasting over 1 day against 19 for the
refit-offset neighbour-count detector (RO-NC). This experiment replays those alarms through the mask
of experiments 163 and 171 and scores the forecast.

**The replay masks the unchanged GAM-K forecast with experiment 195's alarms.** Nothing is refitted.
An alarm sets readings missing from its step stamp, once the alarm is known, until its end is known.
Three alarm sources were replayed: the "R re-ranker" (a small neural network, known 4 days after the
step because it reads 4 days after the step), a logistic regression on the same features (known 2
days after the step), and experiment 195's re-run of RO-NC (2 days). The pair alarm and the unmasked
forecast were recomputed on the same rows as controls. A score is twice the mean pinball loss over
11 quantiles, in points of effective capacity (percent of a substation's effective capacity; lower
is better), on the six Primary substations of experiments 163 and 171.

![Learned-score mask minus pair-alarm mask after real transfers, by transfer set and alarm source](figure_1_gain_after_transfers.svg)

![Mask minus no mask on ordinary days, by alarm source, pooled and November to March](figure_2_harm_ordinary_days.svg)

![Mean score per transfer for no mask, the pair-alarm mask, and the re-ranker mask](figure_3_per_transfer.svg)

## Results

- **Verdict: `partly supported` under the pre-registered rule, and a weak result.** G1' (gain on the
  11-transfer set) holds on its point estimate (-0.46, bar -0.3). G1 (the brief's bar on the 7
  transfers that best guess 2 also scores) fails (+0.99). G2 (no harm on ordinary days) fails
  because its upper bound, +0.086, is above the +0.05 limit. The G1' interval, -3.5 to +3.7,
  includes zero, and leaving D16 out gives +0.64.
- **Units.** A transfer is one logged event. The 7, 11, and 13 transfers are distinct events, and a
  bootstrap resamples whole transfers. Two of the 11 touch two of the six substations, so the 11
  transfers hold 13 scoring windows (one substation and one transfer each).
- **Arms.** Alarm sources LEARNED_R (re-ranker, known 192 half-hours after the step), LEARNED_F
  (logistic regression, 96), RONC_195 (96), and, post hoc, LEARNED_R96 (the re-ranker as if known at
  96). Controls: K0 (no mask), PAIR_NEW (every pair alarm), and PAIR_GATED (balance score of at
  least 0.7). Nothing is refitted: the cached daily GAM-K runs of 2024-04 to 2026-06 are reused. Two
  shortcuts apply. The end of an alarm that experiment 147 also raised is known at the later of
  experiment 147's stamp and the estimated end plus 60 half-hours, where the plan said 96 (a re-run
  with 96 leaves every transfer score unchanged and moves the ordinary-day contrast by +0.0005). The
  cached runs are daily (09:00 UTC) and not 6-hourly, because the cached member forecasts exist only
  for those runs.
- **The controls reproduce.** On the 7 transfers, K0, PAIR_NEW, and best guess 2 score 13.686,
  6.321, and 10.091, as experiment 163 printed. The pooled gated pair-alarm contrast is +0.030
  (-0.012 to +0.074), against +0.030 (-0.011 to +0.074) in experiment 171; the bootstrap draws
  differ in the third decimal.
- **Mean score of the 72 hours after a transfer ends (points).** The 7 transfers that best guess 2
  also scores: K0 13.686, PAIR_NEW 6.321, PAIR_GATED 6.336, LEARNED_R 7.310, LEARNED_F 7.370,
  RONC_195 7.361, best guess 2 10.091. The 11 transfers of experiment 163's scored and replicate
  months: K0 10.129, PAIR_NEW 6.083, PAIR_GATED 5.752, LEARNED_R 5.623, LEARNED_F 5.660, RONC_195
  5.707. All 13 transfers in every cached run (post hoc): K0 8.313, PAIR_NEW 5.351, PAIR_GATED
  5.119, LEARNED_R 4.829, LEARNED_F 4.795, RONC_195 4.782.
- **Gain contrasts (arm minus PAIR_NEW, 95% interval from resampling whole transfers).** LEARNED_R:
  7 transfers +0.989 (-2.229 to +6.963); 11 transfers -0.460 (-3.505 to +3.718); 13 transfers -0.522
  (-3.086 to +2.704). LEARNED_F and RONC_195 differ from LEARNED_R by under 0.1 points on each set.
  Each learned mask beats no mask by 3.5 to 6.4 points, and so does the pair alarm (-7.365 on the 7
  transfers).
- **One transfer decides the 7-transfer result, and its scoring window overlaps the next transfer.**
  A 580-hour transfer at D5 (card ce4b82, where a card is the identifier of a logged transfer)
  scores 22.45 with no mask, 12.81 with the pair alarm, and 22.45 with the re-ranker. No learned
  alarm lies near its onset. The only learned alarm there is the onset of the next transfer on the
  same series (card b2f7c0), known after the 72-hour scoring window closes. The 72 hours after ce4b
  ends lie wholly inside b2f7c0, so those 849 rows measure the forecast during b2f7c0. The overlap
  comes from experiment 163's row set, and no other of the 13 scoring windows of the 11 transfers
  overlaps another transfer. Leaving ce4b out would move G1 to -1.608 and G1' to -2.126 (post hoc,
  and not used to claim a pass).
- **The learned masks gain where the pair alarm does harm.** At D16 in the replicate months (card
  500860) the pair-alarm mask scores 11.07 against 7.71 with no mask, because its fragmented weak
  alarms keep the mask open until 166 half-hours after the end. The re-ranker scores 2.32. Three
  transfers carry the sign of the 11-transfer gain: ce4b adds +1.36 to the contrast, card 500860
  adds -0.90, and b2f7c0 at D5 adds -0.77. Leaving card 500860 out gives +0.488.
- **At these substations, the learned alarms do not come earlier.** Of the 45 logged transfers at
  the six substations, an alarm matching the transfer is known before the transfer ends for 16 with
  the pair alarm, 8 with the re-ranker, 11 with the logistic regression, and 10 with RO-NC
  (`printed_found.txt`). An alarm matches at all for 20 (pair alarm), 19 (re-ranker), 27 (logistic
  regression), and 20 (RO-NC) of the 45. The re-ranker's 19 include ce4b, which is really the onset
  of b2f7c0. Experiment 195 counted 35 transfers over 1 day and a stricter rule, so its counts of 24
  and 19 are not comparable.
- **Ordinary days (G2): mask minus no mask on 1,579,711 rows over 27 months, where no mask scores
  2.718.** LEARNED_R +0.031 (-0.020 to +0.086), and +0.031 (-0.097 to +0.175) from November to
  March. LEARNED_F +0.109 (+0.022 to +0.226), and +0.286 in winter. RONC_195 +0.144 (+0.054 to
  +0.258), and +0.291 in winter. The gated pair alarm +0.030 (-0.012 to +0.074). Per fold, LEARNED_R
  gives +0.039, +0.017, and +0.036. None of the learned masks meets the +/-0.05 limit.
- **The learned scores are partly in-sample on ordinary days.** Experiment 195 trained them with the
  real record's even-month candidates as negatives. The re-ranker's contrast is +0.048 (-0.052 to
  +0.156) on odd months, which it was not trained on, and +0.014 (-0.006 to +0.036) on even months.
  The odd-month figure is the fairer estimate of the harm a deployed score would do.
- **The learned masks cover fewer half-hours than the gated pair alarm.** From the half-hour an
  alarm is known to the half-hour its end is known, the re-ranker's alarms span about 18,800
  half-hours per fold (three folds), against 33,557 for the gated pair alarm over the same rows.
- **Leaving one substation out (ordinary days, pooled).** D10 carries the pair alarm's harm: +0.002
  without D10. The re-ranker's estimate without D10 is +0.012, the logistic regression's -0.000, and
  RO-NC's +0.028.
- **Tail calibration on the 11-transfer rows (coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles,
  then pinball loss in points).** No mask: coverage 0.297, 0.353, 0.841, 0.893; loss 3.55, 4.98,
  2.69, 1.47. Pair-alarm mask: 0.172, 0.247, 0.898, 0.937; 1.47, 2.53, 1.42, 0.57. Re-ranker mask:
  0.130, 0.178, 0.935, 0.980; 2.14, 3.01, 0.93, 0.25. The re-ranker mask has better upper-tail
  coverage and loss than the pair-alarm mask and worse lower-tail loss. Every mask leaves the lower
  quantiles too high: the measured power falls below the 0.01 quantile in 13% to 30% of rows,
  because these rows follow a transfer and the spread is that of the unmasked forecast.
- **Checks: median and mean forecast.** The median forecast minus the mean forecast per site
  (ordinary rows, points) is the same for K0 and LEARNED_R to three decimals (D10 -0.058, D11
  -0.030, D14 +0.037, D16 +0.017, D5 -0.146, D9 -0.019), because the mask changes the correction
  only through the mean.
- **Checks: mean-forecast error and audit.** The error of the mean forecast points the same way as
  every quantile-score contrast: LEARNED_R minus PAIR_NEW +0.525 (7 transfers) and -1.221 (11
  transfers), and LEARNED_R minus K0 on ordinary days +0.042 (-0.031 to +0.123). Leaving D16 out
  flips the 11-transfer gain from -0.46 to +0.64, which triggers an audit.
- **Checks: audit findings.** An independent audit re-derived G1, G1', and G2 from the stored
  per-row scores with its own code and reproduced them (+0.989, -0.460, and +0.031 with -0.020 to
  +0.088). It matched the controls to experiment 163's and 171's stored rows within 2e-6, and found
  the onset stamps causal and the fold mapping equal to experiment 195's. All of its findings are in
  this page (the ce4b overlap, the in-sample caveat, the end-rule deviation, and the timing counts);
  none was rejected.
- **Caveats.** The learned alarms come from three folds that hold out whole Primary pairs. Each
  transfer uses the fold that held out its pair, and ordinary rows average the three folds' scores.
  The 7, 11, and 13 transfers are few, so every gain interval is wide. Experiment 195 found the
  extra transfers small (0.4 to 0.9 MW each). The pair alarm raised 242 alarms and the learned
  alarms 104 to 150 per fold.
- **What this narrows.** Finding more transfers is not the bottleneck: the re-ranker matches no more
  transfers than RO-NC at these substations and masks no better. The missing piece is an early alarm
  on the long transfers, which among the alarms tested here only the pair alarm gives. A candidate
  next experiment is the union of the pair alarm and the learned alarms (pair alarm when its balance
  score passes, learned alarm otherwise), judged on the 13-transfer set and the ordinary rows.
  Experiment 195's look-ahead features cannot give an early alarm, so another candidate is a small
  neural network that reads only data up to the alarm, such as the first 24 hours after a step.
