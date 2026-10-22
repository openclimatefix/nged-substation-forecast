---
id: "00208"
slug: switching-value-of-a-detector-by-phase-and-quality
title: >-
  When does the forecast benefit from knowing about switching transfers, and what is a better or
  worse detector worth?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00208-switching-value-of-a-detector-by-phase-and-quality/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible for how a
  detector's quality is degraded and for how a phase is defined, so both are fixed below before any
  score is read. Callers: this folder's scripts only; they are copies of the modules of experiments
  151 and 203, and read the stored worlds of experiment 141 and the stored alarm tables of
  experiments 147, 156, and 195.
hypothesis: >-
  Knowing about a switching transfer lowers the forecast error mainly in the 48 hours after the
  transfer ends and in the days before an alarm would normally arrive, and the benefit shrinks
  roughly in proportion to the share of transfers missed and to the delay of the alarm. A detector
  of the quality of the refit-offset neighbour-count detector (RO-NC) recovers well under half of
  the perfect-knowledge benefit, so the quality it must reach to be worth building is higher than
  RO-NC's.
prediction: >-
  Written before any score is read. Injected transfers, 44 worlds, six Primary substations, scored
  against the true normal-arrangement series, masked forecast (readings set missing while a
  transfer is known). (1) Perfect knowledge (known at the true onset, end known at once) lowers the
  mean quantile score by at least 3 points of effective capacity in the 48 hours after the end of
  transfers of 25% of load or more lasting over 1 day, with a 95% interval above zero, and by under
  0.1 points pooled over the whole year. (2) RO-NC, known as a live service would know it, recovers
  15% to 40% of the perfect-knowledge gain in that phase, so it fails the bar below. (3) In the
  quality sweep, recovered share falls roughly linearly with recall (50% recall recovers 35% to
  60%), loses most of its value once the delay exceeds 12 hours, and does not depend on the
  megawatt (MW) error for the mask design (the mask ignores MW) while the subtraction design loses
  at least half its gain at a 50% MW error. (4) Precision matters through harm on ordinary days:
  false alarms at precision 0.5 raise the ordinary-day score by more than 0.05 points, and at
  precision 0.75 by 0.02 to 0.06. The prediction fails if the perfect-knowledge gain in the after
  phase is under 3 points or has an interval including zero, or if RO-NC recovers 50% or more.
decision_rule: >-
  Pre-registered before any score is read. Knowledge sources (all on the same forecast stack and
  rows; GAM-K's cached member forecasts plus the Kalman filter correction with experiment 115's K0
  spread, so a contrast is a location-only difference): (a) NONE (K0, no mask); (b) PERFECT, the
  injected transfers' true start and end, known at the true onset with the end known at once (lab
  conditions), plus PERFECT_LAG, the same truth known after the live service's delays (onset 96
  half-hours after the step, end 60 half-hours after the end, experiment 151's lags); (c) LOGGED,
  NGED's switching table on the real record (experiment 163's oracle mask: logged onset, end known
  at once); (d) DETECTOR, on injected worlds the stored RO-NC intervals of experiment 141's worlds
  (the learned score of experiment 195 was trained on the real record and does not exist for
  injected worlds), and on the real record experiment 195's learned score (LEARNED_R, alarms known 4
  days after the step, from experiment 203's tables) with the step stamp as onset; (e) PAIR_NEW and
  PAIR_GATED (experiments 156 and 166) and RONC_195 on the real record, and RO-NC on injected
  worlds. Designs: MASK (readings inside a known span set missing, experiments 163 and 171) is the
  headline; SUB (the span's MW subtracted from the stream, experiment 88's correction) is run for
  the arms that carry MW. Phases of a row, fixed by the true transfer and the forecast origin
  alone, so every arm is scored on the same rows: BEFORE_ALARM (the forecast's data cutoff is at or
  after the true onset and before onset + 96 half-hours), DURING (cutoff at or after onset + 96
  half-hours and target time before the true end), AFTER (target time within 96 half-hours of the
  true end, cutoff at or after the onset), ORDINARY (rows outside every true transfer and 144
  half-hours after it, every fourth daily run on injected worlds; experiment 171's rows on the real
  record). Gain: K0 score minus arm score in points of effective capacity, so positive is better.
  Share recovered: arm gain divided by PERFECT gain in the same phase and stratum. Strata: size
  (moved share of the receiver's load under 10%, 10% to 25%, 25% or more), duration (2 to 6 h, 6 to
  24 h, over 1 d), season of the onset (November to March against April to October), receiver
  (every receiver metered against some load going to an unmetered receiver), substation, forecast
  lead (0 to 12 h, 12 to 36 h, 36 h to 7 d), and the quantiles q05, q95, and the median reported
  separately (pinball loss; coverage at q01, q05, q95, q99 through `studies.tail_calibration`).
  Intervals: injected transfers resampled whole (units: world and event), 95%, 20,000 draws;
  ordinary-day harm resampled by whole calendar month. Quality sweep, each degradation applied to
  PERFECT with a fixed seed, one draw per (world, substation, level): recall 0.25, 0.5, 0.75 (events
  dropped at random); onset delay 6, 12, 24, 48 half-hours; MW blur by a factor drawn from
  1 +/- 0.25 and 1 +/- 0.5 (SUB only); repair of RO-NC's misses in steps of 25%, 50%, 75%, and 100%
  (missed events added as true spans with the live delays), and RO-NC with its false alarms
  removed. Precision is swept on the real record, where ordinary days are plentiful: false alarms
  are added to PERFECT's real-record counterpart (LOGGED) to reach precision 0.75 and 0.5 (as many
  extra spans as needed, onset drawn uniformly over rows outside every logged transfer and its 72
  hours after, duration and MW drawn from the logged transfers, five draws per level), and the
  ordinary-day harm is the mean of the loss differences over all ordinary rows. The same harm is
  reported for LEARNED_R, PAIR_GATED, and RONC_195. Positive controls: PAIR_NEW, K0, and the gated
  contrast must reproduce experiment 203's 6.321, 13.686 and the gated +0.030 on its rows, and on
  the injected worlds K0 and CLEAN (the stream without the injected MW) must agree before the first
  injected stamp, as in experiment 151; failing either makes the page `invalid`. Bar for "worth
  having" (fixed here before any score is read, the brief's example): detector (d) recovers at
  least 50% of the PERFECT gain in the AFTER phase of transfers of 25% of load or more and lasting
  over 1 day, with the point estimate at or above 50%, and its ordinary-day harm has a 95% interval
  inside +/-0.05 points. The threshold is set by the cost of porting a detector, which adds a
  model, alarm tables, and a pipeline step to the serving path. Quality required: for each axis,
  the worst tested level with the other axes at their perfect setting that still recovers 50%.
  Verdict: `supported` when PERFECT's AFTER gain is at least 3 points with an interval above zero
  and the bar holds for (d); `partly supported` when the first holds and the bar fails (the detector
  is not worth building as it is, and the value curve gives the quality it must reach);
  `not supported` when PERFECT's AFTER gain is under 3 points or its interval includes zero in every
  phase; `inconclusive` for a failed fit or fewer than 20 injected transfers in the headline
  stratum. Reported beside the contrasts (not bars): the median forecast minus the mean forecast
  per site; the error of the mean forecast beside each quantile-score contrast; leave-one-substation
  -out of the headline contrasts; which labels the results need (b and c need NGED labels; d and e a
  2,500-series deployment could reproduce) and the decision each supports: build, keep, or drop.
  Post hoc, labelled: any other degradation or combination of degradations the first results
  suggest.
benchmark: >-
  The perfect-knowledge bound of experiment 44 (a lowering of 0.42% of the best guess's quantile
  score on large events, interval -0.47% to +1.96%), the injected-transfer gains of experiments 88
  and 151, and the real-transfer scores and harm of experiments 163, 171, and 203.
row_set_id: >-
  Experiment 151's injected-world rows (44 worlds of experiment 141, six Primary substations, the
  replicate and scored months) and experiment 203's rows (the real record's after-transfer and
  ordinary rows); not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; gain is the K0 score minus the arm's score.
code: ml-experiments/00208-switching-value-of-a-detector-by-phase-and-quality/
date: 2026-10-22
---

Planned. The write-up replaces this page when the experiment finishes.
