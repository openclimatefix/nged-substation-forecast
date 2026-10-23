---
id: "00312"
slug: lsc-how-good-must-a-detector-be-against-the-band-head
title: >-
  How good must an LSC detector be to beat the best no-detector forecast?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
  - calibration-and-uncertainty
status: done
size: >-
  complex. Stored: private score tables and fitted quantile heads under
  data/ml-experiments/00312-lsc-how-good-must-a-detector-be-against-the-band-head/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible for how a
  false-alarm span is drawn, for what "onset delay" means, and for the combined head, so each is
  fixed below before any score is read. Callers: this folder's scripts only; they import the
  modules of experiment 297's folder and read experiment 297's stored scores and experiment 141's
  stored worlds. Asset graph: not touched. The maintainer's brief limits the reviewers to Sonnet,
  so the plan review and the code review are Sonnet reviews.
hypothesis: >-
  A load shift detector that is far from perfect can still beat the best forecast that uses no
  detector, so a detector need not be near-perfect to be worth building. The only detector-based
  arm that beat that forecast in experiment 297 was the one that subtracts the true megawatts from
  the readings during a shift and adds them back to the forecast. This experiment degrades that
  arm one axis at a time to find where it stops beating the no-detector forecast. It also asks
  whether the surprise of recent readings against GAM_ONLY's residual bands (band position),
  or perfect knowledge of a shift, added as inputs to the no-detector quantile head, improves the
  head. This is question 3 of the LSC decision program, redefined by the Discuss ruling of
  2026-10-23.
prediction: >-
  Written before any score is read. Each part can fail separately. (P1) Recall: the added-back arm
  at recall 0.5 still beats BAND_HEAD on curtailment cost within 7 days of a shift (G5), with a 95%
  interval above zero, but at recall 0.25 it does not. (P2) Megawatt error and delay: a random
  error of +/-25% in the megawatts, and an onset delay of 12 half-hours (6 hours), each still beat
  BAND_HEAD on cost in G5, and a delay of 48 half-hours does not. (P3) False alarms: at precision
  0.75 the arm beats BAND_HEAD on cost in G5 and its ordinary-row (G1) cost is within 3% of
  BAND_HEAD's; at precision 0.5 it does not beat BAND_HEAD. (P4) Band position as a detector: at an
  alarm rate equal to RO-NC's alarms per matched shift, it matches at least as many injected shifts
  as RO-NC does, with a median onset delay under 48 hours. (P5) Adding band position as a head input
  lowers the upper-tail score (twCRPS at the 95th percentile of earlier power) against observed
  power in the rows near a shift (G5) by at least 2% against BAND_HEAD, with a 95% interval above
  zero, on worlds and months held out from fitting; adding the perfect-knowledge indicator does
  the same by at least 5%.
decision_rule: >-
  Fixed before any score is read. Rows, worlds, targets, row groups, weights, and intervals are
  experiment 297's, unchanged (44 worlds of experiment 141, six Primary substations labelled D5,
  D9, D10, D11, D14, and D16, cached ensemble-weather member forecasts, ordinary rows weigh 4,
  intervals resample whole shifts or whole calendar months, 95%, 20,000 draws). Bar: BAND_HEAD,
  the cross-fitted head of experiment 297, read from its stored scores. Degraded arms are scored
  against observed power only, because adding megawatts back forecasts observed power. Metrics:
  curtailment cost (12 to 36 hour leads, node only, scaled limit 0.5 pre-registered, 0.75 and 0.9
  swept) and twCRPS at the 95th percentile of earlier power, read in G5, in the shift rows (before
  the alarm, during, and 48 hours after a shift) and in each of those three phases separately;
  ordinary-row (G1) harm is read on the quantile score and on cost. Degraded arms: the arm
  PERFECT_ADD of experiment 297, rerun and checked against its stored scores, then one axis at a
  time with the other axes perfect. Recall 0.25, 0.5, 0.75: each true shift is kept with that
  probability (fixed seed per world and substation); a missed shift is not subtracted and not
  added back. Onset delay 6, 12, 24, and 48 half-hours: the shift is known that many half-hours
  after its true onset, its end is known when it ends, as in experiment 208. Megawatt error: every
  shift's megawatts are scaled by 1 plus or minus 0.25 (or 0.5), the sign drawn at random per
  shift (blur), and by a fixed 1.25 or 0.75 (bias). False alarms at precision 0.75 and 0.5: for
  every true shift, one third (precision 0.75) or one (precision 0.5) extra false span is added.
  A false span starts at a random half-hour after the first replicate row, outside every true
  shift and its 72-hour tail, lasts as long as a randomly chosen true shift at that substation,
  carries a constant megawatt value equal to the mean absolute megawatts of a randomly chosen true
  shift at that substation with a random sign, and is known at its onset, with its end known when
  it ends (so that false alarms are the only imperfection). One post hoc mixed arm combines recall
  0.5, delay 12, blur 25%, and precision 0.75. Break-even for an axis: the worst level, on the
  axis's grid, at which the degraded arm's gain over BAND_HEAD has a 95% interval above zero; if
  no level qualifies the page says "none of the levels", and if the perfect level does not
  qualify the page says "perfect knowledge does not beat BAND_HEAD on this metric". The break-even
  is stated with the leave-one-substation-out check (D11, D16) beside it, because experiment 297
  found that one substation moved its contrasts. Band position: the normal score of each
  half-hour's measured power minus GAM_ONLY (the residual stream), inside the empirical quantile
  function of that substation and hour band, fitted on odd-month ordinary half-hours of the real
  record (experiment 309's method); `pit24` is the mean normal score over the 48 half-hours before
  the forecast cutoff and `ext24` the mean absolute normal score. As a detector it enters an alarm
  state when |pit24| (or ext24) exceeds a threshold k at two consecutive 6-hourly runs and leaves it
  when it stays under k/2 at two runs or after 14 days (experiment 309's rule). k is picked on
  odd-month rows to give a target number of alarms per matched shift (RO-NC's alarms per match
  measured on the same worlds, 1.33 and 2.0 for precision 0.75 and 0.5, and 2.7 for the carried
  LR_b20 mask's rate on the real record), and recall, median onset delay, and alarms per match are
  read on even-month events and rows. A shift counts as matched when an alarm opens between its
  onset and its end. New heads: BAND_HEAD's inputs (filter level u, K0 correction, lead, hour
  band, quarter, as-of ratio), XGBoost settings and seed unchanged, plus (a) pit24 and ext24;
  (b) the three perfect-knowledge indicators (a shift is known to be active at the origin, a shift
  ended within 72 hours before the origin, and the target lies inside a true shift); (c) (a) and
  (b) together; (d) (b) and the added-back megawatts at the target. Each head is fitted twice: by
  cross-fitting on month parity over all worlds, as BAND_HEAD was, and on even-seed worlds and the
  other month parity only, scored on odd-seed worlds and the month parity not fitted, so that no
  scored world, injected shift, seed, or month was seen in fitting. Every new head is read against
  the stored BAND_HEAD fit of the same kind (cross-fitted against cross-fitted, held-out against
  held-out). The injector and the real record are shared by every fit, so a leak through them is
  not removed; a further control fits a head on shifts smaller than the median size and scores
  shifts larger than it. Band position also uses odd months for its quantile function while the
  head cross-fits on both parities; the page reports the contrasts on even-month rows separately,
  which the quantile function never saw. Real record (the precondition): BAND_HEAD, K0, and
  GAM_ONLY scored on the real record of the six substations with no injection, in the 7 logged
  transfers' windows and on ordinary rows, with coverage at q01, q05, q95, and q99. Where BAND_HEAD
  sits against experiment 264's as-of widening and experiment 209's heads is stated from those
  pages' printed numbers where they share rows, and otherwise as a statement of what was not
  compared. No multiplicity correction is applied across the contrasts; the page says so.
  Amendments after the plan review, fixed before any score is read. (A1) Headline cell: curtailment
  cost at limit 0.5, G5, against the cross-fitted BAND_HEAD, observed power; the break-even of an
  axis is the weakest level such that it and every better level have intervals above zero, and every
  other cell (twCRPS, shift rows, phases, limits 0.75 and 0.9, the held-out check on odd-seed
  worlds) is reported but cannot flip a verdict. (A2) Delays 96, 144, and 192 half-hours are added
  to the grid, because delays of 6 to 48 half-hours end before the during and after rows begin; the
  page says which phase each delay moves. (A3) Precision is true kept spans over all spans, false
  spans are counted from the kept spans and drawn over the scored period only, and the realised
  precision of each arm is reported. (A4) Ordinary-row harm of a degraded arm is read as its
  quantile score and cost minus the perfect arm's on G1 rows, with K0 beside as a control, because
  K0 plus spread alone is within 3% of BAND_HEAD on G1; harm is a higher score. (A5) Band position's
  detector rule: RO-NC is re-scored with the same match rule (an alarm known between the onset and
  the end), the alarms per match RO-NC shows on these worlds is the rate P4 uses, recall is reported
  at all four rates, and the real transfers inside the stream count as false alarms. (A6) P5's
  primary reading is the even-month rows, which band position's quantile function never saw;
  all-month rows are secondary, and cost is a secondary reading beside twCRPS. (A7) The three
  indicators and the added-back megawatts are oracle inputs: heads (b) to (d) are upper bounds, not
  a detector design, and the small-against-large shift control tests extrapolation only. (A8) The
  real record is scored with the head fitted on the other month parity; its rows overlap the
  training rows (the real record is the background of every world), and the 7 logged transfers are
  descriptive with coverage given per substation. (A9) Experiment 264's rows are at D8, outside the
  six Primaries, and experiment 209's heads are arm D's; unless the rows overlap, the page says no
  placement is possible and gives experiment 297's as-of widening arm as the nearest comparison.
  (A10) Each clause of P1 to P5 is scored separately; `partly supported` needs at least two of the
  five predictions to hold. (A11) The brief's stop condition was checked: PERFECT_ADD beats
  BAND_HEAD on cost in experiment 297, so the experiment runs.
  Verdict: `supported` when P1 to P5 all hold, `partly supported` when at least two hold, `not supported`
  when none holds.
benchmark: >-
  Experiment 297's BAND_HEAD (and BAND_HEAD_HELDOUT for the held-out contrasts) and its
  PERFECT_ADD arm on the same rows (curtailment cost gain over K0 within 7 days: head +2.0% (1.0 to
  3.1), PERFECT_ADD +8.1% (6.4 to 10.0); PERFECT_ADD over the head +6.5% (3.9 to 9.2)).
row_set_id: >-
  Experiment 297's injected-world rows (44 worlds of experiment 141, six Primary substations); not
  comparable with the leaderboard
metric: >-
  Curtailment cost (scaled limit 0.5, with 0.75 and 0.9 swept) and twCRPS at the 95th percentile of
  earlier power against observed power, quantile score, q01 and q99 coverage, and for the band
  position detector, recall, onset delay, and alarms per matched shift
code: ml-experiments/00312-lsc-how-good-must-a-detector-be-against-the-band-head/
verdict: not supported
verdict_note: >-
  Only P1 of the five pre-registered predictions held, and the plan scored exactly one as not
  supported. The added-back-megawatts arm is more tolerant than predicted on delay, megawatt error,
  and false alarms, and band position and the perfect-knowledge inputs add far less to the quantile
  head than predicted.
best_arm: >-
  Perfect megawatts added back, degraded to recall 0.5: 3.5% (1.9 to 5.1) lower curtailment cost than
  BAND_HEAD within 7 days of a shift.
headline: >-
  A detector does not have to be near-perfect to beat BAND_HEAD on curtailment cost. With the true
  megawatts added back, the arm still wins when it finds half the shifts, knows each shift 3 days
  late, has megawatts wrong by 50%, or raises false alarms at 1 in 2. It loses at 25% found, and RO-NC
  found 33% (28 to 38) on these worlds.
lessons: >-
  The break-even recall (0.5) sits just above RO-NC's recall (0.33), so a detector that finds more
  shifts than RO-NC, even with poor timing and size estimates, is the target. Band position is a
  weaker detector than RO-NC at every alarm rate, and as an input it gives the quantile head under 2%.
take_home_message: >-
  A load shift detector does not need to be near-perfect to improve on the best forecast that uses no
  detector: with the true megawatts added back, finding half the shifts, learning of each up to 3 days
  late, or misjudging the megawatts by half still lowers curtailment cost against that forecast. The
  detector tested earlier (RO-NC) finds only a third of the shifts, which is below the break-even of
  one half, so the next step is a detector that finds more shifts, and the surprise of recent readings
  against the forecast's bands (band position) is not that detector.
advice:
  - >-
    Aim a load shift detector at finding at least half of the shifts, and judge it on curtailment
    cost against the band quantile head, because 50% found still wins and 25% found does not (Break-even
    figure).
  - >-
    Do not spend effort on timing finer than about 2 days or on megawatts closer than about 25%, because
    a 3-day delay or a 50% megawatt error still beats the band quantile head on curtailment cost
    (Break-even figure).
  - >-
    Score false alarms on ordinary rows, not within 7 days of a shift: the near-shift cost cell cannot
    see them, and precision 0.5 raised the ordinary-row quantile score by 2.5% (2.1 to 3.0) (Break-even
    figure and G1 bullets).
  - >-
    Do not use band position (the normal score of recent readings against the no-lag forecast's
    residual bands) as a detector or as a head input, because it found at most 19% of shifts at 18
    alarms per match and lowered the head's upper-tail score by 0.2% (-1.5 to +0.9) (Band position
    figures).
  - >-
    Give a head built on perfect-knowledge inputs the added-back centre instead of K0's centre before
    judging it, because the head with the added-back megawatts as an input gained only 1.5% (0.5 to
    3.0) cost against the same head without them, against 6.3% for the added-back arm (Head inputs
    figure).
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**This experiment asks how accurate a load shift detector has to be to improve on a forecast that
uses no detector.** A load shift moves demand from one Primary substation to its neighbours for
hours or days. We worked on the 44 simulated copies of the real power record from [experiment
297](../00297-lsc-need-a-detector-no-detector-arms-against-the-mask/result.md), with 411 injected
shifts at six Primary substations (labelled D5, D9, D10, D11, D14, and D16). Experiment 297 found
that a quantile model of the forecast's error (BAND_HEAD) beats the RO-NC mask (RO-NC is the
refit-offset neighbour-count detector). Only perfect knowledge, with the true megawatts of each
shift subtracted from the readings the Kalman filter sees and added back to the forecast, still beat
BAND_HEAD on curtailment cost. We spoilt that perfect arm one property at a time: the share of
shifts found, the delay before a shift is known, the error in the megawatts, and false alarms. We
report the worst level that still beats BAND_HEAD. We also tested whether band position (how
surprising recent readings are against the no-lag forecast's own error bands) can detect shifts, and
whether it or perfect knowledge improves BAND_HEAD as an input.

**The arm tolerates much more than we predicted.** It still beats BAND_HEAD within 7 days of a shift
when it finds half the shifts, learns of each shift 3 days late, has the megawatts wrong by 50%, or
raises false alarms at 1 in 2. It stops winning at 25% of shifts found. RO-NC found 33% (28 to 38)
of the shifts on these worlds. Band position finds fewer shifts than RO-NC at every alarm rate, and
as an input it lowers the quantile head's
curtailment cost by under 2.5% and leaves its upper-tail score unchanged.

Every figure is the lab-condition arm: the true megawatts and the true end of each shift are used,
and a Primary substation has no published limit, so cost figures compare forecasts and state no
money saved. Gains are percentages of BAND_HEAD's score, and positive is better for the arm.
Intervals resample whole shifts, or whole calendar months for ordinary rows, 95%.

![Gain in curtailment cost over BAND_HEAD for each degraded arm](figure_1_break_even_curtailment_cost.svg)

![Gain in upper-tail score over BAND_HEAD on shift rows for each degraded arm](figure_2_break_even_upper_tail_shift_rows.svg)

![Gain from extra head inputs over the head without them](figure_3_head_inputs.svg)

![Band position as a detector against RO-NC](figure_4_band_position_detector.svg)

![Real-record tail coverage of the head, K0, and the no-correction forecast](figure_5_real_record_coverage.svg)

## Break-even against BAND_HEAD

- **Headline cell (curtailment cost, scaled limit 0.5, within 7 days of a shift):** perfect
  knowledge gains 6.3% (4.5 to 8.1) over BAND_HEAD, the figure of experiment 297. Without D11 it
  gains 9.0% (6.0 to 12.2), and without D16 7.2% (4.9 to 9.6).
- **Recall:** 75% found gains 4.6% (3.2 to 6.1), 50% found 3.5% (1.9 to 5.1), and 25% found 0.5%
  (-0.9 to 2.0). The break-even is 50%. Without D11 the 25% level wins, at 3.1% (1.0 to 5.5).
- **Delay:** 6 hours gains 5.3%, 24 hours 3.7% (2.5 to 4.9), 2 days 2.1% (1.0 to 3.2), 3 days 1.4%
  (0.3 to 2.5), and 4 days 0.7% (-0.3 to 1.7). The break-even is 3 days, and 2 days without D16 (3
  days without D16 gains 1.0%, -0.2 to 2.3). Delays of 6 to 48 half-hours act only on the rows
  before the alarm; the delays of 2 to 4 days also reach the rows during and after a shift.
- **Megawatt error:** random errors of 25% and 50% gain 5.9% (4.1 to 7.7) and 3.7% (2.2 to 5.2), an
  error of 25% too high 4.7% (3.1 to 6.4), and 25% too low 6.5% (4.8 to 8.4). Every level tested
  wins.
- **False alarms:** precision 0.75 gains 6.1% (4.2 to 7.9) and precision 0.5 gains 5.8% (4.0 to 7.7)
  near shifts, because the near-shift rows seldom overlap a false span (a false span avoids every
  shift and its 72-hour tail). On ordinary rows (G1), precision 0.75 raises the quantile score by
  0.9% (0.7 to 1.1) and the cost by 0.6% (0.4 to 0.7) against the perfect arm, and precision 0.5 by
  2.5% (2.1 to 3.0) and 1.6% (1.2 to 2.0). K0 plus spread alone is 2.8% (0.7 to 5.3) cheaper than
  BAND_HEAD on ordinary-row cost.
- **All four mildly together (recall 0.5, 6-hour delay, 25% random error, precision 0.75; post
  hoc):** 1.4% (0.1 to 2.8), and -0.4 to 2.9 without D16.
- **Upper-tail score on shift rows (before the alarm, during, and 48 hours after a shift):** perfect
  knowledge gains 28.8% (14.3 to 41.3). The tolerances are narrower: 50% found gains 10.4% (0.4 to
  20.8) and 25% found does not, a delay of 24 hours gains 13.3% (2.3 to 22.7) and 2 days does not,
  and a 25% random error gains 21.3% (6.6 to 33.4) and 50% does not. Within 7 days of a shift every
  degraded arm scores worse than, or no better than, BAND_HEAD on the upper tail (perfect knowledge
  -2.8%, -7.2 to +1.2), because the arms keep K0's bands.
- **RO-NC on these worlds:** it found 33% (28 to 38) of the shifts, a median 48 hours after onset,
  with about 10 alarms per matched shift, below the recall break-even of 50%.

## Band position and the head inputs

- **As a detector:** at RO-NC's rate of about 10.6 alarms per match (odd months) the mean normal
  score finds 17% (13 to 21) of even-month shifts, with a median delay of 48 hours, and the mean
  absolute score finds 19% (15 to 23). Neither threshold reaches 2.7 alarms per match or fewer than
  4: at higher thresholds recall falls to 0.
- **As a head input (even-month origins):** the upper-tail score within 7 days of a shift changes by
  -0.2% (-1.5 to +0.9) and the cost gains 1.6% (0.8 to 2.7). Held out (fitted on even-seed worlds
  only, scored on odd-seed worlds) the score changes by -1.4% (-2.9 to -0.0) and the cost gains 1.4%
  (0.5 to 2.3).
- **Perfect-knowledge indicators:** the upper-tail score changes by +0.5% (-0.2 to +1.1) and the
  cost by +0.2% (-0.0 to +0.4). With the added-back megawatts as well, the score gains 2.9% (1.5 to
  5.0) and the cost 1.5% (0.5 to 3.0). The 2% and 5% bars of P5 are not met, and these heads read
  inputs no detector has, so they are upper bounds.
- **Injector-leak controls:** fitting on even-seed worlds and scoring odd-seed worlds, and fitting
  on shifts smaller than the median size and scoring larger ones, leave the numbers within their
  intervals. The injector and the real record are shared by every fit.
- **Head trained on:** all 44 worlds' rows of one month parity, scored on the other, then swapped
  (experiment 297); ordinary rows weigh 4. The head's inputs are K0's filter level, K0's correction,
  lead, hour band, quarter, and the as-of ratio. Refitting reproduces experiment 297's scores
  exactly.

## Real record

- **Rows:** the six substations' real record with no injection, scored with the head fitted on the
  other month parity. These rows overlap the injected worlds' background. The 190 logged transfers
  touching the six series in the replicate and scored months are descriptive; the 7 transfers of the
  brief could not be told apart from them.
- **Ordinary rows:** the head against K0 changes the quantile score by +1.1% (-1.1 to +3.7), so
  there is no harm. Readings fall below q01 in 0.9% of rows and above q99 in 0.9%, against 1.8% and
  1.1% for K0.
- **After a transfer (48 hours):** the head gains 4.7% (-2.6 to +13.2) over K0, and readings fall
  below q01 in 1.3% of rows and above q99 in 1.1% (K0: 2.4% and 1.0%). The no-correction forecast
  leaves 3.7% of readings above q99.
- **Experiments 264 and 209:** experiment 264's rows are at D8, outside these six substations, and
  experiment 209's heads belong to arm D, so no placement is possible. Experiment 297's as-of
  widening arm is the nearest comparison, and BAND_HEAD beat it there.

## Predictions

- **P1 held.** 50% found beats BAND_HEAD on cost and 25% found does not.
- **P2 failed.** A 25% random error and a 6-hour delay beat BAND_HEAD, but so does a 24-hour delay
  (3.7%, 2.5 to 4.9), against the prediction that it would not.
- **P3 failed.** Precision 0.75 wins, and ordinary-row cost stays within 3%, but precision 0.5 also
  beats BAND_HEAD near shifts.
- **P4 failed.** Band position finds 17% to 19% of shifts at RO-NC's alarm rate against RO-NC's 33%,
  with a median delay of 46 to 48 hours.
- **P5 failed.** Band position lowers the upper-tail score by 0.2% (not 2%), and the indicators
  raise it by 0.5% (not 5%).

## Checks

- **Reproduction:** the degraded arm at full quality reproduces experiment 297's PERFECT_ADD, and
  the refitted base head reproduces BAND_HEAD, both exactly (largest score difference 0.0 over
  6,879,769 rows).
- **Median minus mean forecast, per site:** for the head the median forecast's absolute error is
  0.29 points lower than the mean forecast's at D16 and 0.11 lower at D11, by design. For K0 and the
  added-back arm it lies within 0.02 points at every site.
- **Mean-forecast error beside the contrasts:** printed for every arm in the report.
- **Triggers that fired:** a result contradicting its prediction (P2 and P3: the false-alarm cell
  cannot see false spans, which is the tested explanation, and the delay arm uses the true end), and
  leave-one-site-out (D11 and D16 move the break-even levels but not the sign of the headline
  contrast). A fresh Sonnet reviewer re-derived the headline contrasts; no Opus audit ran, because
  the brief limits the work to Sonnet.

## Caveats

- **Lab conditions.** The arm uses the true megawatts and the true end of each shift. A real
  detector's timing, size, and recall errors are not independent of each other.
- **False alarms are drawn at random.** A false span avoids shifts, carries a constant size drawn
  from true shifts, and is known at its onset. The cost cell near shifts cannot show the harm of
  false alarms; the ordinary-row figures do.
- **The shifts are 25% of load or more and last over a day**, and the real record's own transfers
  inside the stream count as false alarms in the detector test.
- **Not done.** BAND_HEAD with the added-back centre, a retrained LR_b20 on the injected worlds, and
  a head built on the added-back forecast.

## Arms and rows

- **Arms:** BAND_HEAD, K0, and PERFECT_ADD from experiment 297; 17 degraded versions of PERFECT_ADD
  (recall 0.25, 0.5, 0.75; delays of 3, 6, 12, 24 hours and 2, 3, 4 days; random megawatt error of
  25% and 50%, always 25% too high or too low; precision 0.75 and 0.5; all four mildly); the head
  with band position, with the three indicators, with both, and with the indicators and the
  added-back megawatts, each fitted three ways (all worlds, held out, small shifts only).
- **Code reused:** [experiment
  297](../00297-lsc-need-a-detector-no-detector-arms-against-the-mask/result.md)'s harness, scores,
  and head, and experiment 309's band-position method.
- **Rows and weather:** experiment 297's 6,879,769 rows (411 shifts, ensemble-weather member
  forecasts), no retraining.

## What to try next

- **Build a detector that finds at least half the shifts**, and test it by degrading nothing: run
  LR_b20 on the injected worlds and read its recall against the break-even of 50%.
- **Put the added-back centre under the head**, so the recalibrated bands and the megawatt
  correction come together.
