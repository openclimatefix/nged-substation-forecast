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
  A small quantile model that reads the filter's recent correction beats the RO-NC mask, and in the
  7-day window beats the perfect mask on the upper-tail score. Only perfect knowledge with the true
  megawatts added back beats it on curtailment cost, by 6.5% (3.9 to 9.3).
lessons: >-
  Most of what the best no-detector model gains is better calibrated bands, not a response to the
  shift itself, so a detector has to be compared with a forecast whose bands are already recalibrated.
  The RO-NC mask loses 35% of the upper-tail score within 7 days of a shift because its false alarms
  cost more than its true alarms gain.
take_home_message: >-
  A forecast with no detector that simply recalibrates its bands and reads how far the recent-power
  correction has moved scores better than the RO-NC detector near load shifts, so with RO-NC a
  detector is not shown to be needed. Perfect knowledge with the true megawatts added back still
  lowers curtailment cost by 6.5% (3.9% to 9.3%) against that forecast, so the next step is to run
  the better LSC detector (LR_b20) on the same simulated shifts and judge it against a forecast
  that has the recalibrated bands.
advice:
  - >-
    Compare any load shift detector with the band quantile head, not with K0: K0's bands are too
    narrow after a shift (14.2% of readings above q99, against 3.5% for the head) (figure 3).
  - >-
    Judge a detector's near-shift effect on the rows where a shift acts (before the alarm, during,
    and 48 hours after), not on the whole 7-day window: 90% of the window's rows are margin rows
    where the detector does nothing (Decision contrasts).
  - >-
    Run LR_b20 on the injected worlds before concluding that no detector is needed: RO-NC's false
    alarms lose 35% (23 to 49) of the upper-tail score within 7 days of a shift (figure 1).
  - >-
    Give a detector's perfect-knowledge ceiling the same recalibrated bands as the best no-detector
    arm before ranking them, because the perfect mask keeps K0's bands and loses ordinary-row
    score by 1.8% to the head (Decision contrasts).
  - >-
    Report each Primary substation: D16 alone moves the perfect mask against the head from -5.9% to
    +0.4%, and D11 alone moves the RO-NC megawatts-added-back arm from -3.9% to +1.4% (Decision
    contrasts).
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**Experiment 297 asks whether a forecast needs a load shift detector at all.** A load shift moves
demand from one Primary substation to its neighbours for hours or days. We simulated 411 such shifts
(44 copies of the real power record, six Primary substations, from [experiment
141](../00141-switching-detector-fed-into-gam-k/result.md)) and compared a forecast that has a
detector with seven forecasts that have none, scored as in [experiment
283](../00283-lsc-perfect-knowledge-value-to-forecast-and-curtailment/result.md). The detector is
RO-NC, because the carried LR_b20 mask was never run on the simulated shifts. **Against RO-NC, a
detector is not shown to be needed.** The page does not conclude that no detector is needed, because
RO-NC is the weaker detector.

**Primary substations have no published export limit, so curtailment figures compare forecasts and
state no money saved.** Gains are percentages of K0's score (K0 is the Kalman filter on recent power
with no knowledge of shifts); positive is better. Intervals resample whole shifts, or whole calendar
months for ordinary rows.

![Gain in the upper-tail score over K0 by row group](figure_1_upper_tail_gain_over_k0.svg)

![Gain in curtailment cost over K0](figure_2_curtailment_cost_gain_over_k0.svg)

![Share of readings outside the 1% to 99% band after a shift](figure_3_tail_coverage_after_a_shift.svg)

## Results

- **Decision rule against RO-NC (half B of the worlds; the best no-detector arm was the band
  quantile head, chosen on half A).** The RO-NC mask loses 57.9% (33.7 to 88.6) of the upper-tail
  score within 7 days of a shift (G5), and loses on ordinary rows (-0.178 points, -0.299 to -0.071).
  RO-NC with megawatts added back loses 3.9% (0.1 to 9.0) of curtailment cost. The rule is not met.
- **Perfect knowledge against the same head.** On the upper-tail score within 7 days it loses 5.7%
  (1.1 to 10.8); on the 10.4% of those rows where a shift acts it gains 10.2% (-7.0 to +25.4), which
  does not resolve. On curtailment cost, with the true megawatts added back, it gains 6.5% (3.9 to
  9.3) in G5 and 14.9% (6.3 to 23.4) on shift rows. The ordinary-row clause fails: the head scores
  0.048 points better (-0.148 to +0.019) and costs 2.7% (0.5 to 5.3) more than K0 on ordinary rows.
- **The decision contrasts are not robust to one Primary substation.** Without D16 the perfect mask
  against the head moves from -5.7% to +0.4% (-5.1 to +5.5) on score in G5. Without D11 the RO-NC
  megawatts-added-back arm moves from -3.9% to +1.4% (-2.4 to +5.3) on cost.
- **P1 failed.** The forecast with no correction (GAM_ONLY) loses 0.222 points (0.021 to 0.374) of
  CRPS on ordinary rows at leads of 24 to 168 hours, against the bar of 0.10. In the 48 hours after
  a shift 0.16% (0.07 to 0.28) of true readings fall below its q01 and 3.97% (2.45 to 6.25) rise
  above its q99, and the upper tail is outside the 2-point bar.
- **P2 failed as written.** K0's share below q01 after a shift is 2.24% (1.14 to 3.58), not the
  brief's 9.1%, so the 3% bar is met by K0 itself. The share above q99 is the real fault: 14.23%
  (10.85 to 17.84) for K0, 3.54% (2.26 to 5.02) for the head. The head changes ordinary-row CRPS by
  +0.046 points (-0.024 to +0.151), above the 0.01 bar.
- **P3 failed.** The lead-decay filter recovers 5.8% (-11.7 to 29.6) of the perfect mask's gain in
  the 48 hours after a shift, against 60%. The recovery by GAM_ONLY is 79.5% (57.1 to 132.7) and by
  the head 63.6% (22.6 to 218.7), but those arms were not the ones predicted.
- **The head's edge is mostly calibration (audit estimate).** Its 1% to 99% band is about 20%
  wider than K0's on every row group. Weighting ordinary rows like the near-shift rows gives the head
  +4.9% over K0, against +11.5% near shifts. The band table without the filter level (BAND_TABLE_PLAIN)
  gains 1.8% (1.1 to 2.6) near shifts, so the filter level carries the rest.

## Gains over K0 near a shift

| Arm | Upper-tail score, 48 h after (G4) | Upper-tail score, within 7 days (G5) | Cost, within 7 days (G5) |
|---|---|---|---|
| GAM_ONLY (no correction) | +18.4 (9.8 to 30.0) | +2.3 (-2.5 to 7.5) | -7.5 (-10.7 to -4.6) |
| Band table by disagreement | +1.6 (-5.4 to 7.2) | +6.7 (5.0 to 8.7) | -0.9 (-2.0 to +0.0) |
| Band quantile head | +14.7 (6.7 to 21.1) | +11.5 (9.7 to 13.4) | +2.0 (1.0 to 3.1) |
| As-of widening | +14.2 (6.3 to 24.6) | -0.3 (-4.7 to 4.4) | -4.0 (-5.7 to -2.3) |
| Lead-decay filter | +1.3 (-2.6 to 6.7) | +9.6 (6.6 to 12.9) | -3.5 (-6.1 to -1.6) |
| RO-NC mask | +13.9 (0.6 to 26.5) | -35.0 (-48.8 to -22.9) | -6.1 (-9.9 to -2.7) |
| Perfect mask | +23.1 (8.6 to 39.0) | +6.3 (2.8 to 10.0) | -1.4 (-4.9 to +1.6) |
| Perfect, megawatts added back (post hoc) | not scored | not scored | +8.1 (6.4 to 10.0) |

## Checks

- **Median minus mean forecast, per site:** the mean absolute error of the median minus that of the
  mean forecast lies between -0.274 (band head) and +0.088 (band table, plain) points. The band
  arms keep K0's mean forecast, so their mean error equals K0's by construction, and the head moves
  the median by design.
- **Absolute scores (points of effective capacity, K0 / head / perfect mask):** CRPS against the
  true series is 2.71 / 2.66 / 2.71 on ordinary rows, 4.14 / 3.43 / n.a. 48 hours after a shift
  (G4), and 2.96 / 2.74 / n.a. within 7 days (G5). The full table is in the printed report.
- **Triggers that fired:** trigger 2 (all three predictions failed; the tested explanation is K0's
  measured 2.24% below q01 against the brief's 9.1%), trigger 4 (leaving out D16 or D11 flips
  decision contrasts), and trigger 7 (step 6 of the program builds its bar on the best no-detector
  arm). A Sonnet audit re-derived the headline contrasts from the stored scores and reproduced them.
  We accepted its findings 1 to 4 (report shift rows apart from margin rows, report per-site
  contrasts, add the ordinary-row cost clause, state that the head's edge is mostly calibration) and
  documented findings 5 and 6 without refitting, because the clean-series spread in GAM_ONLY and
  ASOF_WIDEN flatters two arms that are not the best arm. Its finding 2 asked for a control with no
  filter input; BAND_TABLE_PLAIN is that control.

## Caveats

- **RO-NC is a weak stand-in.** Its false alarms cause the 35% loss; LR_b20 may behave differently,
  and experiment 208 noted the learned score was trained on the real record.
- **The comparison is not like for like on bands.** The detector arms keep K0's bands, and the head
  has refitted bands, so the head's lead is partly a calibration gain a detector forecast could
  also have.
- **Lab conditions.** The megawatts-added-back arms use the true megawatts and the true end of the
  shift, and the detector arms use hindsight ends.
- **Simulated shifts are all 25% of load or more and last over a day.** The 144 half-hours after the
  end and 7-day margins give 89.6% margin rows in G5.
- **Cross-fitting folds rows near a shift by onset month**, so a few target timestamps fall in the
  other parity's months. The effect is probably small and was not quantified.

## Arms and rows

- **Arms:** K0; GAM_ONLY (no correction, spread fitted on the other month parity); GAM_ONLY_K0SPREAD;
  band table (lead band by hour band); band table by disagreement; band quantile head (XGBoost, 11
  quantiles, depth 3, 100 rounds); as-of widening; lead-decay filter; RO-NC mask and RO-NC with
  megawatts added back; perfect mask and perfect with megawatts added back.
- **Rows and schedule:** experiment 283's rows (22.9 million weighted rows, 411 shifts); no
  retraining, the forecasts are cached member forecasts of experiments 111 and 115; the fitted arms
  are cross-fitted by month parity, so the contrast does not depend on a retraining schedule.
- **Weather:** the cached ensemble member forecasts; the readings are the real residual stream plus
  injected megawatts.

## What to try next

- **Run LR_b20 on the injected worlds**, after checking whether the learned score needs retraining
  there, and judge it against the head and against perfect knowledge on shift rows.
- **Give the detector arms the recalibrated bands**, so that detector and no-detector forecasts
  differ only in the detector.
