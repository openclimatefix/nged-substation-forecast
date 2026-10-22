---
id: '00205'
slug: dp-gamk-gap-robustness-coast-or-frozen-gam
part: GAP1
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- lagged-power-and-error-correction
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00205-dp-gamk-gap-robustness-coast-or-frozen-gam/
  only. Serving path: not touched. Degradation rule: not changed (the result is evidence for a later
  rule about missing power readings, which is a separate issue). Designs: more than one is defensible
  (where a gap sits relative to the forecast, how the bands are widened, whether the fallback keeps
  the later corrections), and this plan pre-registers one of each and runs the alternative of the
  last as a labelled secondary rule. Callers: this folder's scripts, which copy the base-forecast
  and filter code of experiments 136 and 197 and the correction code of experiment 155, and call
  the reviewed `studies.*` modules without changing them. No new modelling and no new fit.
hypothesis: >-
  When power readings go missing, a Kalman filter that coasts (keeps running with no update, so the
  transient decays and the level holds) beats the frozen convex model (GAM) alone for short gaps and
  loses to it for long gaps, so a crossover gap length exists between 6 and 72 hours. The mechanism
  is that the level is a random walk, so a short gap leaves it nearly right, while a long gap lets
  it drift away from the demand. Widening the quantile bands by the filter's own growth in
  predictive variance keeps the 5% and 95% quantiles calibrated at the gap lengths where coasting
  is still better than the frozen model.
prediction: >-
  Written before any score is read. Scored over leads 0.5 to 168 hours at the five decision
  substations, a gap ending at the forecast's data cutoff. Rule (a), coast, beats rule (c), the
  frozen GAM, by at least 0.05 points for gaps of 3 and 6 hours, and loses to rule (c) at 72 hours,
  so the crossover lies between 6 and 72 hours. Recovery after a 24-hour gap, which is the number of
  hours between the readings returning and the score being back within 0.02 points of the no-gap
  score, is under 12 hours. Rule (b), coast with widened bands, has q05 and q95 coverage closer to
  the nominal 5% and 95% than rule (a) at 24 and 72 hours, and a quantile score no more than 0.01
  points worse than rule (a) at 3 and 6 hours. If rule (a) never loses to rule (c), the serving rule
  is to always coast.
decision_rule: >-
  The headline contrasts are C1 to C4, rule (a) minus rule (c) at gap lengths of 3, 6, 24, and 72
  hours, and C5 to C8, rule (b) minus rule (a) at the same lengths, on the full pipeline of
  experiment 155's arm D (the filter, the winter night offset, and recipe R) for rules (a) and (b),
  and on the frozen GAM alone for rule (c). Each is read at the 99.375% level of the month-resampled
  paired bootstrap (12 calendar months of the origins, 20,000 resamples), which is 95% across the
  eight contrasts. The crossover exists when C1 and C2 have point estimates of at most -0.05 with
  intervals wholly below zero, and C4 has an interval wholly above zero; the crossover gap length is
  the shortest tested length at which the interval of C1 to C4 is wholly above zero. The hypothesis
  is `supported` when the crossover exists and the recovery after a 24-hour gap is under 12 hours.
  It is `partly supported` when C1 and C2 hold but C4's interval is not wholly above zero, in which
  case the serving rule is to always coast, and this is the finding. It is `refuted` when C1 or C2
  has an interval not wholly below zero. It is `inconclusive` otherwise. The recovery of a rule is
  the smallest tested k (the hours between the end of the gap and the cutoff, among 0, 3, 6, 12, 24,
  and 48) at which the all-lead score minus the no-gap score is at most 0.02 points at that k and at
  every larger tested k. Rule (b) replaces rule (a) in the recommendation when its mean absolute
  distance of q05 and q95 coverage from nominal is smaller at gap lengths of 24 and 72 hours and its
  score minus rule (a)'s is at most +0.01 at every length. A switch from coasting to the frozen GAM
  adds a branch to the serving path, so a crossover is recommended only when rule (c) beats rule (a)
  by at least 0.02 points at some tested length; otherwise the recommendation is to always coast.
  The verdict is capped at `inconclusive` if the control fails: with no gap injected, the pipeline
  must reproduce experiment 155's arm D on the same rows to within 0.001 points, and a rule (a)
  forecast with a gap of 0 hours must equal it exactly.
benchmark: >-
  The no-gap score of the same runs under the same pipeline (arm D of experiment 155, rebuilt here),
  the frozen GAM alone (rule c), and climatology and manual_heuristic as stored by experiment 111
  for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, over leads 0.5 to 168 hours on rows outside the hand-filtered switching events,
  and under 24 hours and 12 to 36 hours; q05 and q95 coverage; mean absolute error of the mean
  forecast beside every quantile-score contrast
code: ml-experiments/00205-dp-gamk-gap-robustness-coast-or-frozen-gam/
---

# Gap robustness of the convex model with a Kalman filter: coast or fall back to the frozen model

**Planned, nothing has been run.** The design below is fixed before any score is read.

**Question.** In production the convex demand model with a Kalman filter (GAM-K, the Kalman filter on recent errors of [experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md); the method re-implements [de Vilmarest and Wintenberger (2022)](https://arxiv.org/abs/2104.10777)) will sometimes receive no power readings for hours or days. We ask which rule to serve then, and whether a gap length exists beyond which the filter should be dropped.

**Design.**

- **Inputs.** The unfiltered forecast of every run (experiment 136's A2: the T-all fits driven by the calibrated ensemble mean), rebuilt from the stored fits and ensemble arrays as in [experiment 197](../00197-dp-kalman-level-adaptation-retest/result.md), and the filter's residual stream. The full pipeline is the arm D of [experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md): the filter, the winter night offset, and recipe R (minus 0.75 times the 14-day mean error profile).
- **Events.** For each substation, one run per 7-day block of origins is chosen at random (seed 205) from the replicate and scored runs, so the events are at least a few days apart and one event's gap hardly touches the next. The same events are used for every gap length.
- **Gap.** The readings of every hour band are removed for the L hours up to k hours before the event run's data cutoff, with L in {3, 6, 24, 72} and k in {0, 3, 6, 12, 24, 48}. All 24 combinations are scored. Headline contrasts use k = 0, the gap that ends at the cutoff. Larger k measure recovery. Actuals inside a gap are also hidden from recipe R's error profile.
- **Rules.** (a) Coast: the filter runs on the gapped stream, so the transient decays and the level holds, then the offset and R apply. (b) Coast with widened bands: rule (a)'s quantile offsets from its mean are multiplied by sqrt(1 + v / s^2), where v is the filter's extra predictive variance since the band's last reading (the level's q per day times the days elapsed, plus the transient's stationary variance times 1 - phi^(2 n) for n half-hours elapsed, minus the same at one half-hour) and s is (q95 - q05) / 3.29 of the row. (c) Frozen GAM alone: the unfiltered forecast with a spread refitted on the tuning months, with no filter, offset, or R. Secondary rules: (a0) and (b0) are (a) and (b) with the filter alone, no offset or R. (c2) is the frozen GAM with the offset and R, which keeps the later corrections but drops the filter for the event run.
- **Scored.** Five decision substations, D5, D10, D11, D14, and D16, with leads 0.5 to 168 hours. Spread tables are fitted on the tuning months, with no gap, and held fixed.
- **Compute.** CPU only, 4 cores, no new fit; the filter re-runs on a masked stream per (substation, L, k).

Prior advice followed: the Kalman state is the one of experiment 111 with the POOL setting (experiment 136, A2); no faster level is used (experiment 197); recipe R at every lead and no offset on top of R is experiment 155's finding, but this plan keeps arm D, the pipeline that experiment 155 tested, so the offset stays in the gapped pipeline and the comparison stays with the stored arm D.
