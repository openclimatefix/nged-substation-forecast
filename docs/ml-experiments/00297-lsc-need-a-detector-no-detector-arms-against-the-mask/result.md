---
id: "00297"
slug: lsc-need-a-detector-no-detector-arms-against-the-mask
title: >-
  Do forecasts with no load shift detector match the best LSC detector?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
  - calibration-and-uncertainty
status: done
size: >-
  complex. Stored: private score tables, fitted band tables, and a fitted quantile head under
  data/ml-experiments/00297-lsc-need-a-detector-no-detector-arms-against-the-mask/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible for the
  band model, for the lead-decay time constant, and for the stand-in detector, so each is fixed
  below before any score is read. Callers: this folder's scripts only; they are copies of the
  modules of experiment 283 and read the stored worlds of experiment 141. Asset graph: not touched.
  The maintainer's brief limits the reviewers to Sonnet, so the plan review and the code review are
  Sonnet reviews.
hypothesis: >-
  A forecast that never learns which readings are load shifts can match a forecast that uses a
  detector, because its bands and its recent-power correction can be made to tell a load shift from
  ordinary variation by themselves. This is question 2 of the LSC decision program.
prediction: >-
  Written before any score is read. Each of three parts can fail separately. (P1) GAM_ONLY, the
  forecast with no recent-power correction and bands fitted on odd months, costs at most 0.10
  points of CRPS on ordinary rows at leads of 24 to 168 hours, and the share of true-normal-
  arrangement readings below its q01 or above its q99 in the 48 hours after a shift lies within
  2 percentage points of 1%. (P2) The best band model brings the share of readings below q01 in
  the 48 hours after a shift from K0's share (about 9.1% in the program brief) to 3% or less, with
  a change in ordinary-row CRPS of under 0.01 points. (P3) The lead-decay filter recovers at least
  60% of the PERFECT gain in twCRPS against the true normal-arrangement series in the 48 hours
  after a shift, with ordinary-row harm within +/-0.02 points.
decision_rule: >-
  Fixed before any score is read. Arms, all on the rows, targets, row groups, metrics, and
  intervals of experiment 283 unchanged (injected worlds of experiment 141, six Primary
  substations, cached member forecasts of the GAM-K stack with ensemble weather). Reference arms
  (rerun, and checked against experiment 283's stored numbers): K0, PERFECT (readings in the true
  span set aside), PERFECT_ADD (post hoc, true megawatts added back, observed power only). No-
  detector arms: GAM_ONLY (no correction, spread fitted on the residuals of the other month
  parity); BAND_TABLE_PLAIN (K0 centre, empirical quantile offsets by lead band and hour band);
  BAND_TABLE_DISAGREE (the same offsets also binned by u, the K0 correction in units of its own
  history); BAND_HEAD (an XGBoost quantile model of the 11 residual quantiles reading u, the K0
  correction, lead, hour band, and quarter); ASOF_WIDEN (GAM_ONLY's spread scaled by the ratio of
  the last 7 days of GAM_ONLY residuals known at the origin to that site's usual level, clipped to
  1 to 3; a simplified port of experiment 264's as-of widening, which is the comparator); and
  LEAD_DECAY (the correction at lead 0, c0, times exp(-lead/T(u)), where u is c0 in units of its
  own history and T is fitted per u bin from the centre's squared error). Every table, head, and T
  is fitted on rows whose target month has one parity and scored on rows whose month has the other
  parity, then the roles swap, so every row is scored out of sample and no score is read on fitted
  rows. Ordinary rows weigh 4 in fitting and scoring, as in experiment 283. Detector arms: DET_MASK
  and DET_ADD (the RO-NC alarm intervals stored in experiment 141's worlds, readings set aside, or
  the detected megawatts added back, with their false alarms), and PERFECT. Experiment 141's worlds
  are the only injected worlds that any detector has been run on, so RO-NC stands in for the
  carried LR_b20 mask; the page says so at the top. Metrics, targets, and row groups G0 to G5 as in
  experiment 283; every arm is scored against both PERFECT (the true normal-arrangement series)
  and PERFECT_ADD (observed power). Rule for question 2: a detector is needed only if a detector
  arm beats the best no-detector arm with a 95% interval above zero on twCRPS or on curtailment
  cost in the near group (G5), and the detector arm's ordinary-row (G1) harm interval, against the
  same no-detector arm, lies inside +/-0.05 points. The best no-detector arm is the one with the
  lowest pooled twCRPS in G5 on the same target; because it is chosen on the scored rows, this
  leans towards "no detector needed", and every no-detector arm is reported against the detector
  separately. Amendments after the plan review, fixed before any score is read. (A1) Selection: the
  44 worlds are split by seed parity into half A (even seeds) and half B (odd seeds); the best
  no-detector arm is chosen on half A by pooled G5 twCRPS at the 95th percentile against the true
  normal-arrangement series, and every detector contrast is read on half B only. (A2) The two
  decision contrasts are fixed: DET_MASK against the best arm on twCRPS against the true series (all
  leads, G5), and DET_ADD against the best arm on curtailment cost against observed power (12 to 36
  hour leads, G5, limit 0.5); the same two contrasts are read for PERFECT and PERFECT_ADD, and all
  other combinations are reported without a decision. No multiplicity correction is applied across
  the four; the page says so. (A3) Every fitted table, head, and T is fitted on residuals against
  observed power (the stream a no-detector forecaster sees), never against the true series. Fit rows
  are all rows of the other parity in the stored row set; parity is the target month for ordinary
  rows and the onset month of the shift for rows near a shift. (A4) LEAD_DECAY replaces the filter's
  own projection: the correction is c0 * exp(-lead/T(u)), where c0 is K0's correction at the run's
  first half-hour. u and the as-of usual level use site scales computed on the real record without
  injected shifts. (A5) GAM_ONLY_K0SPREAD (no correction, experiment 115's K0 spread) is added, so
  P1 separates the cost of dropping the correction from the cost of the new spread. (A6) Units: every
  absolute difference is in points of effective capacity and every relative gain is a percentage of
  the comparator's score; the bars in P1 to P3 and the G1 bars apply to the point estimate, the
  interval is reported beside. P1's coverage bar applies to each tail separately, in the 48 hours
  after a shift, against the true series. P3's recovery is (K0 - LEAD_DECAY) / (K0 - PERFECT) on G4
  twCRPS against the true series, bootstrapped per draw. The 9.1% in the brief is not reproduced
  here; K0's own share is recomputed and used as the baseline for P2. The rule is applied to the
  stand-in detector and to PERFECT; if PERFECT does not
  beat the best no-detector arm, no detector can. Every verdict on question 2 is labelled "against
  RO-NC". If RO-NC beats the best no-detector arm, the conclusion is "a detector is needed". If
  RO-NC does not, the page does not conclude "a detector is not needed": it says "not shown with
  RO-NC" and names the next experiment as running LR_b20 on the injected worlds, after checking
  whether the learned score needs retraining there (experiment 208 noted it was trained on the
  real record). Only a PERFECT arm that fails to beat the best no-detector arm supports "no
  detector can add value". Reported with and without D11 (experiment 283
  found one Primary supplies 106 of 267 shifts reaching the 48-hour group). Intervals: whole
  shifts resampled for shift groups, whole calendar months for ordinary rows, 95%, 20,000 draws.
  A Primary has no published limit and no known farm, so curtailment is a relative comparison
  between arms; this goes at the top of the page. Verdict: `supported` when P1 to P3 all hold;
  `partly supported` when one or two hold; `not supported` when none hold.
benchmark: >-
  Experiment 283's K0, PERFECT, and PERFECT_ADD on the same rows (twCRPS gain +23.1% (8.8 to 39.1)
  in the 48 hours after a shift; curtailment cost gain +8.1% (6.4 to 10.0) within 7 days).
row_set_id: >-
  Experiment 283's injected-world rows (44 worlds of experiment 141, six Primary substations);
  not comparable with the leaderboard
metric: >-
  twCRPS at the 95th and 99th percentile of earlier power, CRPS, MAE of the median and of the mean,
  avoidable curtailment cost (node-only, scaled limit 0.5 pre-registered, 0.75 and 0.9 swept),
  q01 and q99 coverage after a shift
code: ml-experiments/00297-lsc-need-a-detector-no-detector-arms-against-the-mask/
verdict: not supported
verdict_note: >-
  None of the three predictions held as written. The RO-NC mask does not beat the best no-detector
  arm, so a detector is "not shown with RO-NC". Perfect knowledge beats that arm on curtailment cost
  but not on the upper-tail score, and the margin on score changes sign when one Primary substation
  is left out.
best_arm: >-
  Band quantile head (no detector): 11.5% (9.7 to 13.4) lower upper-tail score than K0 within 7 days
  of a shift, and the share of readings above q99 in the 48 hours after a shift falls from 14.2% to
  3.5%.
headline: >-
  A quantile model that reads the Kalman filter's recent correction beats the RO-NC mask. Within 7
  days of a shift it beats the perfect mask on the upper-tail score, but leaving out one Primary
  substation (D16) removes that margin. Only perfect knowledge with the true megawatts added back
  beats it on curtailment cost, by 6.5% (3.9 to 9.2).
lessons: >-
  Under half of what the best no-detector model gains survives reweighting ordinary rows like the
  near-shift rows, so part of its gain is better calibrated bands and the rest is a response to the
  filter's correction. RO-NC loses 35% of the upper-tail score within 7 days of a shift, and false
  alarms are the likely cause.
take_home_message: >-
  A forecast with no detector that recalibrates its bands and reads how far the recent-power
  correction has moved scores better than the RO-NC detector near load shifts, so with RO-NC a
  detector is not shown to be needed. Perfect knowledge with the true megawatts added back still
  lowers curtailment cost by 6.5% (3.9 to 9.2) against that forecast, so the next step is to run the
  better LSC detector (LR_b20) on the same simulated shifts and judge it against a forecast that has
  the recalibrated bands.
advice:
  - >-
    Compare any load shift detector with the band quantile head, not with K0: K0's bands are too
    narrow after a shift (14.2% of readings above q99, against 3.5% for the head) (figure 3).
  - >-
    Judge a detector's near-shift effect on the rows where a shift acts (before the alarm, during,
    and 48 hours after), not on the whole 7-day window: 90% of the window's rows are margin rows
    where a detector does nothing (Decision contrasts).
  - >-
    Run LR_b20 on the injected worlds before concluding that no detector is needed: RO-NC loses 35%
    (23 to 49) of the upper-tail score within 7 days of a shift (figure 1).
  - >-
    Give a detector's perfect-knowledge ceiling the same recalibrated bands as the best no-detector
    arm before ranking them, because the perfect mask keeps K0's bands and scores 0.048 points
    worse than the head on ordinary rows (Decision contrasts).
  - >-
    Report each Primary substation: D16 alone moves the perfect mask against the head from -5.7% to
    +0.4%, and D11 alone moves the RO-NC megawatts-added-back arm from -3.9% to +1.4% (Decision
    contrasts).
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**This experiment asks whether a forecast needs a load shift detector at all.** A load shift moves
demand from one Primary substation to its neighbours for
hours or days. A detector finds those shifts in the power readings, and the forecast then sets aside
the readings inside each detected shift (a "mask"). We simulated 411 load shifts (44 copies of the
real power record, six Primary substations labelled D5, D9, D10, D11, D14, and D16, from [experiment
141](../00141-switching-detector-fed-into-gam-k/result.md)). We compared forecasts that use a
detector with seven forecasts that use none, scored as in [experiment
283](../00283-lsc-perfect-knowledge-value-to-forecast-and-curtailment/result.md). The detector is
RO-NC (the refit-offset neighbour-count detector), because LR_b20 (a learned detector built on RO-NC's
alarms) was never run on the simulated shifts. **Against RO-NC, a detector is not shown to be
needed.** The page does not conclude that no detector is needed, because RO-NC is the weaker of the
two detectors.

**K0 is the Kalman filter on recent power with no knowledge of shifts; every gain is a percentage of
K0's score, and positive is better.** The "upper-tail score" is the continuous ranked probability
score weighted towards values above the 95th percentile of earlier power (twCRPS). G1 is ordinary
rows, G4 is the 48 hours after a shift ends, G5 is the 7 days either side of a shift and the shift
itself, and "shift rows" are the before-alarm, during, and after rows of G5. Primary substations have
no published export limit, so curtailment figures compare forecasts and state no money saved.
Intervals resample whole shifts, or whole calendar months for ordinary rows.

![Gain in the upper-tail score over K0 by row group](figure_1_upper_tail_gain_over_k0.svg)

![Gain in curtailment cost over K0](figure_2_curtailment_cost_gain_over_k0.svg)

![Share of readings outside the 1% to 99% band after a shift](figure_3_tail_coverage_after_a_shift.svg)

## Gains over K0 near a shift

| Arm | Upper-tail score, 48 h after (G4) | Upper-tail score, within 7 days (G5) | Curtailment cost, within 7 days (G5) |
|---|---|---|---|
| GAM_ONLY (no correction) | +18.4 (9.8 to 30.0) | +2.3 (-2.5 to 7.5) | -7.5 (-10.8 to -4.6) |
| Band table by disagreement | +1.6 (-5.4 to 7.2) | +6.7 (5.0 to 8.7) | -0.9 (-2.0 to +0.0) |
| Band quantile head | +14.7 (6.7 to 20.9) | +11.5 (9.7 to 13.4) | +2.0 (1.0 to 3.1) |
| As-of widening | +14.2 (6.3 to 24.6) | -0.3 (-4.6 to 4.5) | -4.0 (-5.7 to -2.3) |
| Lead-decay filter | +1.3 (-2.6 to 6.6) | +9.6 (6.7 to 12.9) | -3.5 (-6.0 to -1.6) |
| RO-NC mask | +13.9 (0.9 to 26.5) | -35.0 (-48.8 to -23.0) | -6.1 (-9.9 to -2.7) |
| Perfect mask | +23.1 (8.7 to 39.1) | +6.3 (2.8 to 10.0) | -1.4 (-4.9 to +1.6) |
| Perfect, megawatts added back (post hoc, scored against observed power) | +18.1 (6.0 to 31.5) | +6.6 (3.3 to 9.7) | +8.1 (6.4 to 10.0) |

Every cell is a percentage gain over K0 with a 95% interval.

## Decision contrasts

- **Decision rule against RO-NC (half B of the worlds; the best no-detector arm, the band quantile
  head, was chosen on half A).** The RO-NC mask loses 57.9% (34.1 to 88.8) of the upper-tail score
  against the head within 7 days of a shift, and scores 0.178 points (0.070 to 0.298) worse on
  ordinary rows. RO-NC with megawatts added back loses 3.9% (0.1 to 9.1) of curtailment cost. The
  decision rule is not met.
- **Perfect knowledge against the same head.** On the upper-tail score within 7 days it loses 5.7%
  (1.1 to 10.9). On the 10.4% of those rows where a shift acts it gains 10.2% (-7.0 to +25.0), so
  the interval spans zero. On curtailment cost, with the true megawatts added back, it gains 6.5%
  (3.9 to 9.2) in G5 and 14.9% (6.3 to 23.4) on shift rows. The ordinary-row clause fails: the
  perfect mask scores 0.048 points worse than the head on ordinary rows (interval for perfect minus
  head: -0.149 to +0.019). The head also costs 2.9% (0.7 to 5.7) more curtailment than K0 on
  ordinary rows.
- **Leaving out one Primary substation changes the contrasts.** Without D16, the perfect mask
  against the head moves from -5.7% to +0.4% (-5.1 to +5.6) on the upper-tail score in G5. Without
  D11, RO-NC with megawatts added back moves from -3.9% to +1.4% (-2.4 to +5.2) on cost.
- **Held-out check of the head (post hoc).** The head's fit used the same worlds as its scoring, on
  the other month parity. Refitting it on even-seed worlds only and scoring odd-seed worlds only
  keeps the gain: +10.1% (8.2 to 12.3) within 7 days against +11.8% (9.2 to 14.9) for the
  cross-fitted head on the same worlds, and 4.07% (2.44 to 5.94) of readings above q99 after a
  shift against 3.85% (2.30 to 5.68). The injector and the real record are shared, so a leak
  through either is not removed.

## Predictions

- **P1 failed.** GAM_ONLY loses 0.222 points (0.029 to 0.374) of CRPS on ordinary rows at leads of
  24 to 168 hours, against the bar of 0.10. In the 48 hours after a shift 0.16% (0.07 to 0.28) of
  true readings fall below its q01 and 3.97% (2.46 to 6.25) rise above its q99, outside the
  2-percentage-point bar.
- **P2 failed as written.** K0's share below q01 after a shift is 2.24% (1.14 to 3.55), not the
  brief's 9.1%, so K0 meets the 3% bar. The share above q99 is the real fault: 14.23% (10.92 to
  17.74) for K0 and 3.54% (2.22 to 5.05) for the head. The head lowers ordinary-row CRPS by 0.046
  points (-0.023 to +0.150), more than the 0.01 bar allows.
- **P3 failed.** The lead-decay filter recovers 5.8% (-11.9 to 30.0) of the perfect mask's gain in
  the 48 hours after a shift, against 60%. Arms outside the prediction recover more: GAM_ONLY 79.5%
  (57.2 to 134.1), as-of widening 61.7% (41.5 to 93.4), and the head 63.6% (22.4 to 210.6).

## Checks

- **Median minus mean forecast, per site:** the mean absolute error of the median forecast minus
  that of the mean forecast lies between -0.274 (band head) and +0.088 (band table, plain) points.
  The band arms keep K0's mean forecast, so their mean-forecast error equals K0's by construction,
  and the head moves the median by design.
- **Absolute CRPS against the true series (points of effective capacity, K0 / head / perfect
  mask):** 2.71 / 2.66 / 2.71 on ordinary rows, 4.14 / 3.43 / 2.69 in G4, and 2.96 / 2.74 / 2.64 in
  G5. The printed report holds every arm.
- **Triggers that fired:** the prediction-contradiction trigger (all three predictions failed; the
  tested explanation is K0's measured 2.24% below q01 against the brief's 9.1%), the
  leave-one-site-out trigger (D16 and D11 flip contrasts), and the replication trigger (the next
  program step builds its bar on the best no-detector arm). An audit by a fresh Sonnet reviewer
  reproduced the headline contrasts from the stored scores. Its main finding, that the head's
  advantage over the perfect mask in G5 sits in margin rows (89.6% of G5), is in the contrasts
  above. Its estimate that the head's gain falls from +11.5% to +4.9% when ordinary rows are
  reweighted like near-shift rows is not in the printed report, and BAND_TABLE_PLAIN (+1.8%, 1.1 to
  2.6, the same arm without the filter level) is the printed control.

## How BAND_HEAD was trained, and its real-record results (added by experiment 312)

- **Training:** every head is fitted on the rows of all 44 worlds whose month has one parity and
  scored on the other parity, then the roles swap. Rows near a shift take the parity of the shift's
  onset month, and ordinary rows weigh 4. A world shares the injector, the real record, and the
  months with the other worlds, so the held-out check above removes the scored worlds and seeds
  from fitting but not the injector.
- **Real record ([experiment
  312](../00312-lsc-how-good-must-a-detector-be-against-the-band-head/result.md)):** against K0 the
  head changes the quantile score on ordinary rows by +1.1% (-1.1 to +3.7), so it does no harm.
  Readings fall below q01 in 0.9% of ordinary rows and above q99 in 0.9%, against 1.8% and 1.1% for
  K0. In the 48 hours after a logged transfer the shares are 1.3% and 1.1%, against 2.4% and 1.0%.
- **Not compared:** experiment 264's rows are at D8 and experiment 209's heads belong to arm D, so
  neither shares rows with BAND_HEAD.

## Caveats

- **RO-NC is a weak stand-in.** Its false alarms are the likely cause of the 35% loss, although the
  report does not separate them from late ends. LR_b20 may behave differently, and experiment 208
  noted that its learned score was trained on the real record.
- **The comparison is not like for like on bands.** The detector arms keep K0's bands, and the head
  has refitted bands, so part of the head's lead is a calibration gain that a detector forecast
  could also have.
- **Lab conditions.** The megawatts-added-back arms use the true megawatts and the true end of each
  shift, and the detector arms use hindsight ends.
- **The simulated shifts are all 25% of load or more and last over a day.**
- **Not done.** The head was not scored on the real record (7 logged transfers), which resolves no
  curtailment difference ([experiment
  277](../00277-lsc-curtailment-metric-on-existing-forecasts/result.md)). The as-of widening arm is
  a simplified port of [experiment 264](../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md)'s
  idea, and no comparison with experiment 209's heads was made.
- **Spreads and folds.** GAM_ONLY and as-of widening fit their spreads on the clean series, which
  flatters both arms slightly. Rows near a shift are folded by onset month, so a few target
  timestamps fall in the other parity's months.

## Arms and rows

- **Arms:** K0; GAM_ONLY (no correction, spread fitted on the other month parity); the same with
  K0's bands; a band table (lead band by hour band); a band table by disagreement (also binned by the
  filter's correction in units of its own history); the band quantile head (XGBoost, 11 quantiles,
  depth 3, 100 rounds, trained on residuals against observed power); as-of widening; the lead-decay
  filter; RO-NC with and without megawatts added back; perfect knowledge with and without megawatts
  added back.
- **Training:** every fitted arm is trained on all worlds' rows of one month parity and scored on the
  other parity, then the roles swap. Rows near a shift take the parity of the shift's onset month.
- **Rows and schedule:** experiment 283's rows (22.9 million weighted rows, 411 shifts); no
  retraining, the forecasts are cached member forecasts of experiments 111 and 115.
- **Weather:** the cached ensemble member forecasts; the readings are the real residual stream plus
  injected megawatts.

## What to try next

- **Run LR_b20 on the injected worlds**, after checking whether the learned score needs retraining
  there, and judge it against the head and against perfect knowledge on shift rows.
- **Give the detector arms the recalibrated bands**, so that detector and no-detector forecasts
  differ only in the detector.
