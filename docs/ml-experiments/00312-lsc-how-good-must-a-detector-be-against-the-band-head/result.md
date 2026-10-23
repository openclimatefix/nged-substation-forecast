---
id: "00312"
slug: lsc-how-good-must-a-detector-be-against-the-band-head
title: >-
  How good must an LSC detector be to beat the best no-detector forecast?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
  - calibration-and-uncertainty
status: planned
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
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**This experiment asks how accurate a load shift detector has to be to improve on a forecast that
uses no detector.** Experiment 297 found that a quantile model of the forecast's residual
(BAND_HEAD) beats a real detector's mask. Only perfect knowledge, with the true megawatts of each
shift subtracted from the recent readings and added back to the forecast, still beat it on
curtailment cost. We degrade that perfect arm one axis at a time (share of shifts found, delay in
finding them, error in the megawatts, false alarms) and report the worst level that still beats
BAND_HEAD. We also test whether band position, which is how surprising recent readings are against
the forecast's own error bands, can detect shifts, and whether it or perfect knowledge improves
BAND_HEAD when given to it as an input.

The results are filled in after the run.
