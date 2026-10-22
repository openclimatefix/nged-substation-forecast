---
id: "00188"
slug: switching-d10-audit-and-onset-location-in-look-back
title: >-
  Are D10's extra errors from unlogged switching, and does locating the onset inside the look-back
  help the mask?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00188-switching-d10-audit-and-onset-location-in-look-back/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible rule for
  "unlogged event" (part 1) and for locating an onset (part 2), so both are fixed below before any
  score is read. Callers: this folder's scripts only; they are copies of the modules of experiments
  156 and 171 and read those experiments' stored tables, cached forecasts and injected worlds.
hypothesis: >-
  Part 1. Experiment 171 found that ignoring recent power while a gated pair alarm is active scores
  0.030 points of effective capacity worse than no mask on rows outside logged transfers (95%
  interval -0.011 to +0.074), and that leaving out one substation, D10, moves the estimate to
  +0.002. If D10's gated alarms are real switching that NGED did not log, those rows are not
  ordinary days, and treating them as events leaves a no-harm interval inside +/-0.03. Part 2. The
  alarm for a transfer is known hours to days after the step, but the forecast runs every 6 hours,
  so at each run the readings since the step are in the look-back. A step-fitting search over those
  readings finds the onset more precisely than the alarm time, and masking from the found onset
  gives a smaller forecast error in the first 24 hours after the alarm than masking from the alarm
  time.
prediction: >-
  Written before any result. Part 1: with the confirmed unlogged-event rows removed as well, the
  pooled 27-month gated-mask-minus-no-mask interval lies inside -0.03 to +0.03 points of effective
  capacity, and the November to March interval lies inside -0.05 to +0.05. The prediction fails if
  the harm stays spread across ordinary days, that is, if most of D10's gated alarms are not
  confirmed by the rule below or the interval stays wider than the limit. Part 2: masking from the
  located onset cuts the extra error (the score above the score of a mask that starts at the true
  onset) in the first 24 hours after the alarm by at least one third, compared with masking from
  the alarm time. The prediction fails if the located onset is no better than the alarm time or if
  the located onset's median error is larger than the 6-hour spacing of the forecast origins
  (12 half-hours).
decision_rule: >-
  Pre-registered before any forecast score is read. PART 1 (D10 audit). Candidate events are all
  gated pair-alarm intervals (balance score 0.7 or more, experiment 171) at the six decision
  substations. A candidate is a confirmed unlogged event when it is not matched to a logged
  transfer (experiment 171's rule: step from 24 hours before the logged onset to the logged end)
  and a measure that is independent of the alarm's own statistic confirms it. The measure is the
  difference between the median residual in the 12 hours after the alarm's step stamp (starting 1
  hour after) and the median residual in the 12 hours before (ending 1 hour before), for the alarm's
  series (own step) and for each partner series named by the alarm (partner steps). The candidate is
  confirmed when |own step| is at least 0.3 MW, the sum of partner steps has the opposite sign, and
  the ratio |sum of partner steps| / |own step| lies in 0.5 to 2.0. The thresholds are fixed here;
  the gate itself uses peak tests and ratios of 0.7 to 1.43 on a different statistic. A placebo
  applies the same measure at the same stamps shifted by +/-7, 14, and 21 days and reports the
  share confirmed, which is the rate at which the rule confirms by chance. Rows: experiment 171's
  rows (every cached run, six Primary substations, 2024-04 to 2026-06, rows with a reading outside
  logged transfers) with the rows whose valid time lies from a confirmed event's step to 72 hours
  after its end also removed. Scored and bootstrapped exactly as in experiment 171 (twice the mean
  pinball loss over 11 quantiles, whole calendar months resampled, 20,000 draws, 95%). G1': pooled
  MASK_GATED minus K0 has an interval inside -0.03 to +0.03. G2': the same on November to March
  rows inside -0.05 to +0.05. Verdict for part 1 follows experiment 171: `supported` when both
  hold, `partly supported` when only G1' holds, `refuted` when the pooled point estimate is above
  +0.03, `inconclusive` otherwise. Reported beside it: how many gated alarms per substation are
  confirmed, the contrast on the rows of confirmed events alone, on the rows of unconfirmed gated
  alarms alone (the harm that remains), and D10 alone. Because the rows with a gated alarm active
  carry the whole difference, removing confirmed events can lower the contrast by construction; the
  informative results are the confirmed share, the placebo share, and the unconfirmed-alarm
  contrast. PART 2 (onset location). The alarm families are the pair alarm (every pair alarm,
  experiment 156's headline window) and RO-NC (experiment 147's intervals). Truth onset: the
  injected onset in experiment 141's 44 injected worlds (primary; exact), and the logged onset for
  real transfers (secondary; the log is coarser than the stamp). Forecast origins: every 6 hours,
  phase of experiment 171's runs, so the cached daily runs sit on the grid; an alarm is actionable
  at the first origin whose latest data stamp is at or after the alarm's known stamp. LOCATED
  onset: at that first origin, using only data up to its latest stamp, search stamps t from 24
  hours before to 24 hours after the alarm's own step stamp (at least 4 half-hours of data after t)
  for the largest |D(t)| of the right sign, where D(t) is the mean of 12 half-hours from t minus
  the mean of 12 half-hours before t (the after-window shrinks to the data available), on the
  alarm series' residual; for the pair alarm D is the own step minus the sum of the partners' steps
  over the same stamps. Location error is the located stamp minus the true onset, in half-hours;
  reported as median and 90th percentile of the absolute error, and the median signed error, for
  the alarm time (known stamp minus true onset), the detector's own step stamp, and the located
  onset. Mask arms (K0 filter with readings set missing, experiments 156 and 171): MASK_ALARM
  (readings missing from the first actionable origin's latest stamp onwards), MASK_STEP (from the
  detector's step stamp, experiment 171's arm), MASK_LOCATED (from the located onset), MASK_TRUE
  (from the true onset), and K0 (no mask). The end rule is experiment 148's, the same in every
  arm. Rows: all leads of every cached run whose forecast origin falls within 24 hours after the
  alarm's first actionable origin, from the 44 injected worlds (events at the six decision
  substations that the alarm finds) and from the real logged transfers it finds; scored against
  the true normal-running-arrangement series (injected) or the reading (real). Extra error of an
  arm = its mean quantile score minus MASK_TRUE's. Decisive contrast C2: 1 - (extra error of
  MASK_LOCATED) / (extra error of MASK_ALARM), pooled over alarms found in injected worlds, with a
  95% interval from resampling events. Part 2 is `supported` when the point estimate is at least
  1/3 and the interval excludes 0, `partly supported` when the point estimate is positive with an
  interval that excludes 0 but below 1/3, `refuted` when the point estimate is at most 0, and
  `inconclusive` otherwise. C1: the location error contrast (located minus alarm time), reported
  per family. The real transfers are reported in the same form and decide nothing: there are few
  of them. Matching, fixed before any mask score was read: in an injected world an alarm finds a
  transfer when an interval of the family on one of the substation's series has a step stamp within
  48 half-hours (24 hours) of the true onset, and the earliest-known such interval is used; for a
  real transfer experiment 171's rule applies (a step from 24 hours before the logged onset to the
  logged end). The cached forecasts are daily, so each found transfer is scored on the one run
  whose newest usable stamp lies from the first usable origin to 24 hours after it, all leads. A
  threshold of 1/3 is the maintainer's figure for porting a located-onset step, which
  adds a search to the serving path and so needs a gain this size. Post hoc and labelled: the
  located onset recomputed at each later origin; the mask from the located onset with a gated alarm
  only; the quantised origin's share of the error.
benchmark: >-
  Part 1: experiment 171's unmasked stack (K0) on the same rows, and its +0.030 (-0.011 to +0.074).
  Part 2: MASK_ALARM (mask from the alarm time) and MASK_TRUE (the best a located onset could do).
row_set_id: >-
  Part 1: experiment 171's rows (every cached run of six Primary substations, 2024-04-10 to
  2026-06-29) less confirmed unlogged events. Part 2: runs within 24 hours after each alarm, in
  experiment 141's 44 injected worlds and in real logged transfers; not comparable with the
  leaderboard.
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; onset-location error in half-hours.
code: ml-experiments/00188-switching-d10-audit-and-onset-location-in-look-back/
production_status: not-for-production
adopted_in: none
---

*Plan. The results sections are added after the run.*
