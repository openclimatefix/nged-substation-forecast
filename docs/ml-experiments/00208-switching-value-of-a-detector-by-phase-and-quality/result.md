---
id: "00208"
slug: switching-value-of-a-detector-by-phase-and-quality
title: >-
  When does the forecast benefit from knowing about switching transfers, and what is a better or
  worse detector worth?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
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
verdict: not supported
best_arm: >-
  Perfect knowledge of the transfer, masked: 1.50 points of effective capacity gained over no mask
  (95% interval 1.08 to 1.96) in the 48 hours after a long, large injected transfer ends, against
  the pre-registered bar of 3 points. RO-NC recovers 53% of that gain (35% to 69%).
headline: >-
  Knowing about a transfer lowers the error by 1.50 points (1.08 to 1.96) in the 48 hours after it
  ends and by 1.46 points (0.69 to 2.29) during it, but only 0.61 points (0.43 to 0.81) before an
  alarm would normally arrive, and about 0.005 points pooled over a year. A mask driven by RO-NC
  recovers 53% (35% to 69%) of the 48-hour gain, and every real alarm source costs more on ordinary
  days (+0.03 to +0.14 points) than the whole-year gain it brings (about +0.02 on the real record).
lessons: >-
  The benefit of knowing about a transfer is real but small once pooled over a year, because
  transfers are rare. The detector's recall decides how much of the benefit arrives (recall 0.25 or
  0.5 gives 37% or 63% of it), delay of up to 2 days after the onset costs nothing after the
  transfer ends, and error in the moved megawatts costs little. The binding constraint is harm on
  ordinary days, which grows with the number of alarms: random false alarms at precision 0.5 (46
  alarms) cost +0.013 points, while the real sources raise 104 to 242 alarms for 15 to 20 matched
  logged transfers and cost +0.03 to +0.14.
take_home_message: >-
  Telling the forecast about a switching transfer lowers its error by about 1.5 points of capacity
  during the transfer and in the 48 hours after it ends, but only about 0.005 points over a whole
  year, because transfers are rare. A detector that finds 4 transfers in 10, as RO-NC does, gets
  about half of that, and every real alarm source costs more on ordinary days than it gains, so the
  mask is not worth building on today's detectors. The next step is an alarm source with far fewer
  false alarms per matched transfer.
advice:
  - >-
    Count a detector's alarms per matched logged transfer before judging its recall: ordinary-day
    harm is about +0.0003 points per alarm for random false alarms, the pair alarm, and the
    re-ranker, and about five times that for RO-NC (ordinary-day bullet).
  - >-
    Put the benefit of knowing a transfer in a whole-year pooled number as well as per phase: perfect
    knowledge gains 1.50 points after a transfer and 0.005 points over the year (headline figure).
  - >-
    Aim a detector at recall of about 0.4 or more and accept an alarm known 2 days after the onset:
    recall 0.25 recovers 37% and delay up to 2 days costs nothing after the end (value curves).
  - >-
    Do not spend effort on the moved megawatts for the mask design, and little for subtraction: a
    50% error in the megawatts still recovers 72% of the gain (value curves).
  - >-
    Compare any transfer correction with a forecast that ignores recent power: it recovers 71% (59%
    to 80%) of the perfect-knowledge gain with no detector at all (headline figure).
  - >-
    Score masks against the normal-arrangement series, not metered power, when the aim is a forecast
    under the normal running arrangement: against metered power the mask loses 4.75 points during a
    real transfer (real-record bullet).
  - >-
    Simulate transfers smaller than 25% of load and shorter than 6 hours before judging a detector on
    them: the stored worlds hold none, so the size and duration strata cannot be scored (Results).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**Telling the forecast about a switching transfer lowers its error by about 1.5 points of capacity
during the transfer and in the 48 hours after it ends, but only about 0.005 points over a whole
year, and no real alarm source gains more than it costs on ordinary days.** A switching transfer
moves load from one Primary substation to its neighbours for hours or days. We asked three questions
about the "mask", which sets the forecast's recent power readings aside while a transfer is known:
whether a detector is worth having, where exactly the benefit lies, and what a better or worse
detector would be worth. We simulated transfers on the real power record of six substations (44
simulated copies of the record, 371 transfers), degraded perfect knowledge of them in controlled
steps, and replayed the real alarms and NGED's switching table on the real record. The forecast and
its masks are those of [experiment
163](../00163-switching-real-event-comparison-gam-stack-against-bg2/result.md), [experiment
171](../00171-switching-gated-mask-no-harm-pooled-months/result.md), and [experiment
203](../00203-switching-replay-learned-score-in-masked-forecast/result.md), and the simulated
transfers are those of [experiment 141](../00141-switching-detector-fed-into-gam-k/result.md) scored
as in [experiment 151](../00151-nra-forecast-rescored-against-the-true-nra-series/result.md). The
pre-registered verdict is `not supported`, because perfect knowledge gains 1.50 points where the bar
was 3.

**Answers to the three questions.** (1) Perfect knowledge helps, but modestly: 0.005 points over the
year, in line with the 0.42% of experiment 44. (2) The benefit lies during the transfer (+1.46, 0.69
to 2.29) and in the 48 hours after it ends (+1.50, 1.08 to 1.96), grows with forecast lead (+0.81 at
0 to 12 hours, +1.79 at 36 hours to 7 days), and differs by substation from +0.12 to +4.88. (3)
RO-NC (the refit-offset neighbour-count detector) finds 37% of the injected transfers and recovers
53% (35% to 69%) of the 48-hour gain. The value curve is steep in recall and flat in delay and
megawatt error.

![RO-NC and the other sources' share of the perfect-knowledge gain](figure_1_share_recovered.svg)

![Gain by phase](figure_2_gain_by_phase.svg)

![Value curves](figure_3_value_curves.svg)

![Strata](figure_4_strata.svg)

![Harm on ordinary days](figure_5_harm_on_ordinary_days.svg)

![Real logged transfers](figure_6_real_transfers.svg)

## Results

- **Arms and rows.** Sources (a) none (K0), (b) PERFECT (known at the true onset, end known at once)
  and PERFECT_LAG (the live delays: onset 96 half-hours after the step, end 60 half-hours after the
  end), (c) LOGGED (NGED's switching table, the oracle mask of experiment 163), (d) the detector,
  which is the stored RO-NC intervals on injected worlds and experiment 195's learned score
  LEARNED_R on the real record, (e) PAIR_NEW, PAIR_GATED, RONC, RONC_195. Designs are MASK
  (headline) and SUB (subtract the span's megawatts). References: CLEAN (recent power without the
  injected transfer) and GAM_ONLY (no recent power). Injected rows are 213,533 forecast rows from
  371 transfers (the 25%-of-load-or-more stratum is 301 to 353 transfers per phase). Retraining:
  none, the cached GAM-K forecasts are reused. No weather source is introduced; weather is the
  analysis proxy of experiments 141 and 163 and is not available at forecast time beyond the cached
  ensemble members.
- **Headline gain over K0 in points (25% of load or more, lasting over 1 day; 95% interval from
  resampling whole transfers).** PERFECT: before the alarm +0.61 (0.43 to 0.81), during +1.46 (0.69
  to 2.29), after +1.50 (1.08 to 1.96). PERFECT_LAG: +0.29, +1.46, +1.47. RO-NC (DET): -0.09 (-0.34
  to +0.16), +0.44 (-0.02 to +1.03), +0.79 (0.44 to 1.19). K0's mean score in the 48 hours after is
  4.17 and CLEAN's is 2.46. Share of the PERFECT gain after the transfer: RO-NC 53% (35% to 69%),
  RO-NC without its false alarms 61% (45% to 77%), PERFECT_LAG 98%, no recent power at all 71% (59%
  to 80%), subtracting RO-NC's megawatts 30% (2% to 52%).
- **Pre-registered bar, "worth having".** The recovery clause holds on the point estimate on
  injected worlds (53%) and fails its lower bound (35%); leaving out D11 raises it to 86%, and
  leaving out D9 lowers it to 44%. On the real record LEARNED_R recovers 85% of LOGGED's gain in the
  48 hours after (6 transfer-substation windows; its gain interval, +0.11 to +11.99 points, is too
  wide to inform). The harm clause fails for every real source (next bullet). Whole-year pooled gain
  on injected worlds: PERFECT +0.0053, RO-NC +0.0013, PERFECT_LAG +0.0045.
- **Ordinary days (points of effective capacity, mask minus no mask, 1,579,711 rows, 27 months;
  interval from resampling whole calendar months).** PAIR_GATED +0.030 (-0.012 to +0.074), PAIR_NEW
  +0.055 (+0.009 to +0.105), LEARNED_R +0.031 (-0.020 to +0.086), LEARNED_F +0.109 (+0.022 to
  +0.226), RONC_195 +0.144 (+0.054 to +0.258). None has an interval inside +/-0.05. Random false
  alarms added to LOGGED, at 16 alarms (precision 0.75) cost -0.000 (-0.000 to +0.000) and at 46
  alarms (precision 0.5) cost +0.013 (+0.005 to +0.024). Per alarm, the cost is about +0.0003 points
  for the random false alarms (46 alarms), the pair alarm (+0.055, 242 alarms), and the re-ranker
  (+0.031, 104 to 150 alarms per fold), and about +0.0014 for RO-NC (+0.144, 104 alarms). The real
  sources raise 104 to 242 alarms of which 15 to 20 match a logged transfer, so their precision
  against the log is under 0.2 (an alarm that matches no logged transfer may still be a real,
  unlogged transfer). Perfect recall with precision 0.75 or 0.5 would net about +0.020 or +0.007
  points pooled (LOGGED's pooled gain less the false-alarm harm).
- **Real logged transfers (scored against metered power).** In the 48 hours after a transfer ends,
  LOGGED gains +7.40 (2.32 to 10.22), PAIR_GATED +6.35, LEARNED_R +6.31, RONC +6.18 (6 windows,
  intervals wide). Pooled over ordinary rows plus experiment 163's after rows the gain is about
  +0.017 for every real source (+0.020 for LOGGED), below the ordinary-day harm of +0.03 to +0.14.
  During a transfer the mask loses 4.75 points (LOGGED, -7.2 to -1.5), because metered power really
  does include the transfer; that is not a failure of the mask for a normal-arrangement forecast.
- **Quality sweep, share of the PERFECT gain recovered, after the transfer (during, before the
  alarm).** Recall 0.25: 37% (37%, 36%); 0.5: 63% (72%, 48%); 0.75: 77% (96%, 66%). Onset delay up
  to 2 days: 100%; 3 days (post hoc): 83% (90%, 36%); 4 days (post hoc): 64% (81%, 24%). Error in
  the megawatts (subtraction design only): 25% error 86%, 50% error 72%. Repairing RO-NC's misses in
  steps of 25% (from 53%): 59%, 72%, 73%, and 85% (69% to 99%) at 100%. Quality required for 50%
  after the transfer, other axes perfect: recall about 0.4 (0.25 fails, 0.5 passes), and no tested
  delay or megawatt error fails. RO-NC's recall is 37% (393 of 1,063 transfer-substation pairs), and
  19 of 45 logged transfers on the real record.
- **Strata (perfect knowledge, after the transfer).** Winter (November to March) +1.60 (0.94 to
  2.26) against summer +1.48 (1.00 to 2.00), and RO-NC recovers 74% in winter against 50% in summer.
  Substations: D10 +4.88, D11 +2.08, D9 +1.54, D14 +1.36, D16 +0.59 (-0.27 to +1.28), D5 +0.12
  (-0.25 to +0.53). Size and duration cannot be stratified: every simulated transfer moves 25% of
  the load or more, and all but 5 last over 1 day. The receiver metered or unmetered stratum is not
  scored, because the stored worlds do not record it.
- **Tails (after the transfer, headline stratum; coverage at q95 and q99, pinball at q05 and q95).**
  K0: 0.758 and 0.864; 1.00 and 1.41. PERFECT: 0.948 and 0.983; 0.63 and 0.77. RO-NC: 0.845 and
  0.916; 0.78 and 1.06.
- **Checks.** The median minus the mean forecast per substation is identical for K0 and PERFECT (D5
  -0.31, D10 -0.09, D9 -0.11, D11 -0.04, D14 +0.10, D16 +0.06), because the mask moves the
  correction only through the mean. The mean-forecast error (mean absolute, after the transfer)
  agrees with every score contrast: K0 6.90, RO-NC 5.74, PERFECT 4.63, CLEAN 4.27. Controls
  reproduce: K0 10.129 and PAIR_NEW 6.083 on experiment 163's 11 transfers, and PAIR_GATED +0.030
  (-0.012 to +0.074) with K0 2.718 on ordinary rows; K0 equals CLEAN before the first injected stamp
  (asserted in every job). Audit triggers: trigger 2 fired (perfect-knowledge gain 1.50 against a
  predicted 3 or more, and RO-NC recovering 53% against a predicted 15% to 40%). The explanation was
  tested: simulated transfers move a median of 9.5 points of capacity, the logged transfers a median
  of 15.4, and K0's mean score after is 4.17 on injected worlds against 13.6 on the real record, so
  the real transfers are about 1.6 times larger and the real gain larger. Triggers 1, 3, 4, 5, 6,
  and 7 did not fire (leaving out D11 moves the perfect gain to 0.97 and RO-NC's share to 86%, with
  the same sign and verdict). No Opus audit was run. A Sonnet code review found nine issues, of
  which the delay cap, the repair baseline, the real-record phase labels, the baseline of the
  false-alarm harm, and a missing fold check were fixed and rerun; its finding that the ordinary
  rows need the scored-period filter was rejected, because experiment 171 scored every cached run.
- **Deviations from the plan.** The learned score of experiment 195 does not exist on injected
  worlds, so RO-NC stands in there. The positive controls are experiment 163's published numbers on
  its 11 transfers (10.129 and 6.083), because the plan's 6.321 and 13.686 need best guess 2's
  shared rows. Coverage and pinball use `studies.tail_calibration.pinball` and transfer resampling,
  not its month-resampled table. The delay sweep's pre-registered steps (up to 24 hours) never reach
  the during and after rows, whose cutoffs are at least 96 half-hours after the onset, so delays of
  2, 3, and 4 days were added post hoc. A debug run on 8 of the 44 worlds was read before the final
  run. The real-record before-alarm and during phases are scored against metered power, not the
  normal arrangement.
- **Scale note for 2,500 series.** Sources (b) and (c) need NGED's switching labels. Sources (d) and
  (e) and the quality sweep can be reproduced without labels. The decision this supports: on today's
  detectors, drop the automatic mask in the forecast, and keep the alarms for the operator and for a
  normal-arrangement forecast. The quality to reach is recall of about 0.4 and a harm interval
  inside +/-0.05 on ordinary days.
- **What to try next.** An alarm source with an order of magnitude fewer alarms per matched transfer
  (for example a pair alarm that also demands balanced steps), a soft use of the alarm that does not
  switch recent power off, and a small learned gate that scores each alarm before masking, with the
  convex forecast as the comparison arm. Simulate transfers below 25% of load and under 6 hours
  first, since the stored worlds hold none.
