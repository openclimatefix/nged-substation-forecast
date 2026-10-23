---
id: "00297"
slug: lsc-need-a-detector-no-detector-arms-against-the-mask
title: >-
  Do forecasts with no load shift detector match the best LSC detector?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
  - calibration-and-uncertainty
status: planned
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
date: 2026-10-23
---

**Question.** The [LSC decision program](../themes/lsc-detectors/plan.md) asks, in step 5, whether a
forecast needs a load shift detector at all, or whether bands and a recent-power correction that
fade at the right speed do just as well. This page is the plan. It fixes the arms, the rule, and
the three predictions before any score is read.

**What is new against experiment 283.** [Experiment
283](../00283-lsc-perfect-knowledge-value-to-forecast-and-curtailment/result.md) found that
perfect knowledge of load shifts lowers twCRPS by 23.1% (8.8 to 39.1) in the 48 hours after a shift
ends. Here the same injected worlds, rows, targets, row groups, and metrics score forecasts that
have no detector. The band model builds on the as-of widening of [experiment
264](../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md), which this plan simplifies to one
comparator arm.

## Plan notes

- **Departures from the brief's step 5.** The forecasts are the GAM-K cached ensemble members of
  experiments 111 and 115, which the injected worlds were built on, not arm D of experiment 155 or
  the experiment 220 stack, because neither has forecasts on the injected worlds. The detector is
  RO-NC (the `R0_M48` intervals in each world) instead of LR_b20, as above. BAND_TABLE_PLAIN,
  DET_ADD, and GAM_ONLY_K0SPREAD are added to isolate effects. Cross-fitting replaces the odd-month
  fit.
- **Stand-in detector.** Experiment 141's worlds carry RO-NC alarm intervals and no other
  detector's alarms. The carried LR_b20 mask of experiments 227 and 258 was scored on the real
  record only, so applying it to the injected worlds would need a new reviewed pipeline. The page
  reports RO-NC and PERFECT, and names the substitution at the top.
- **Cross-fitting.** Fitting on one month parity and scoring on the other uses 12 and 11 months in
  turn, instead of experiment 283's single odd-month fit, so that every row is scored out of
  sample.
- **Lead range.** The cached forecasts reach 168 hours, so P1's "24 hours to 10 days" is read as
  24 to 168 hours.
- **Lab conditions.** The detector arms use the injected worlds' true megawatts only where the
  PERFECT references do.
