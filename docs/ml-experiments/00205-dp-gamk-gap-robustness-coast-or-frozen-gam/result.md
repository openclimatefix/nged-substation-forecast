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
  substations, with a gap that ends at the forecast's data cutoff. Rule (a), coast, beats rule (c2),
  the same pipeline with the filter dropped, by at least 0.05 points for gaps of 3 and 6 hours, and
  loses to it at 72 hours, so a crossover lies between 6 and 72 hours. Rule (a) also beats rule (c),
  the frozen GAM alone, at 3 and 6 hours by more than it beats (c2). After a 24-hour gap the score
  is back within 0.02 points of the no-gap score at k = 6 hours or less, where k is the number of
  hours between the gap's end and the cutoff. Rule (b), coast with widened bands, has q05 and q95
  coverage closer to nominal than rule (a) at 24 and 72 hours, and a quantile score no more than
  0.01 points worse than rule (a) at every length. If rule (a) never loses to rule (c2), the serving
  rule is to always coast.
decision_rule: >-
  Headline family: C1 to C4, rule (a) minus rule (c2) at gap lengths of 3, 6, 24, and 72 hours, and
  C5 to C8, rule (b) minus rule (a) at the same lengths, all at k = 0, over leads 0.5 to 168 hours,
  pooled over D5, D10, D11, D14, and D16. Rule (c2) is the comparator because it differs from rule
  (a) only by the filter; rule (c), the frozen GAM alone, is reported as an absolute floor and its
  contrasts are descriptive. Each headline contrast is read at the 99.375% level of the
  month-resampled paired bootstrap (the 12 calendar months of the origins, 20,000 resamples), which
  is 95% across the eight contrasts; everything else (recovery, coverage, per-substation and
  under-24-hour views) is descriptive. A contrast is `below` when its interval is wholly below zero,
  `above` when wholly above zero, and `straddling` otherwise. The hypothesis is `supported` when C1
  and C2 are `below` with point estimates of at most -0.05, the crossover exists, and the recovery
  after a 24-hour gap is at most 6 hours. The crossover is the smallest tested length L* such that C
  at L* and at every longer tested length is `above`. It is `partly supported` when C1 and C2 hold
  as above but no crossover exists; the serving rule is then to always coast, and that is the
  finding. It is `refuted` when C1 or C2 is `above` or has a point estimate of zero or more. It is
  `inconclusive` otherwise, including when C1 or C2 is `straddling` or `below` with a point estimate
  above -0.05. The recovery of a rule at gap length L is the smallest tested k (0, 3, 6, 12, 24, or
  48) at which the all-lead score minus the no-gap score is at most 0.02 points at that k and every
  larger tested k; a rule that has not recovered by k = 48 fails. Recovery is reported for L = 24 on
  the all-lead and the under-24-hour scores, and the verdict reads the all-lead one. The serving
  recommendation is a separate line: switch from coasting to dropping the filter only if a crossover
  exists and (c2) beats (a) by at least 0.02 points at L*, because the switch adds a branch to the
  serving path; otherwise always coast. Rule (b) replaces rule (a) when its mean absolute distance
  of q05 and q95 coverage from nominal is smaller at 24 and 72 hours by more than the binomial
  standard error of the difference, and its score minus rule (a)'s is at most +0.01 at every length.
  The verdict is capped at `inconclusive` if a control fails: with no gap the rebuilt rule (a)
  reproduces experiment 155's arm D to within 0.001 points on the mean forecast; replacing, with random numbers, every actual at or after a run's origin leaves that run's mean forecast unchanged; and rule (b) with the
  extra variance set to zero equals rule (a) exactly.
benchmark: >-
  The no-gap score of the same event runs under the same pipeline (arm D of experiment 155, rebuilt
  here), rule (c2), and the frozen GAM alone (rule c)
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

**Question.** In production the convex demand model with a Kalman filter (GAM-K, the Kalman filter
on recent errors of [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md); the method
re-implements [de Vilmarest and Wintenberger (2022)](https://arxiv.org/abs/2104.10777)) will
sometimes receive no power readings for hours or days. We ask which rule to serve then, and whether
a gap length exists beyond which the filter should be dropped.

**Design.**

- **Inputs.** The unfiltered forecast of every run (experiment 136's A2: the T-all fits driven by
  the calibrated ensemble mean), rebuilt from the stored fits and ensemble arrays as in [experiment
  197](../00197-dp-kalman-level-adaptation-retest/result.md), and the filter's residual stream. The
  full pipeline is the arm D of [experiment
  155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md): the filter, the
  winter night offset, and recipe R (minus 0.75 times the 14-day mean error profile).
- **Events.** For each substation the origins of the replicate and scored runs (excluding the
    every-fifth-day stride runs and any origin before 2024-07-16) are cut into blocks of 8 days, and
    one run is drawn at random (seed 205) from the first 3 days of each block, so two events are at
    least 6 days apart and no gap window reaches the previous event's cutoff. The same event runs
    are used for every gap length and rule.
- **Gap.** The readings of all four hour bands are removed for the L hours up to k hours before the
    event run's data cutoff, with L in {3, 6, 24, 72} and k in {0, 3, 6, 12, 24, 48}; all 24
    combinations are scored, and the headline contrasts use k = 0. One filter pass per (substation,
    L, k) applies the gaps of all that substation's events together, which is equivalent to one pass
    per event because the windows do not overlap.
- **What each stage sees.** The filter reads the gapped stream. The winter night offset is refitted
    monthly on the errors of the earlier rows of the gapped timeline, as in experiment 155, with the
    actuals inside gaps hidden. Rows of the tuning months keep the no-gap filter's forecast, so
    every rule's spread table is the no-gap one even where a gap reaches back into a tuning month.
    Recipe R is recomputed from the gapped timeline: it uses the errors of the earlier runs from the
    gapped filter, and the actuals whose valid time falls inside any gap are hidden from its error
    blocks (a block with fewer than 40 readings is unusable, as in experiment 155). Scoring reads
    the true actuals of the event run's own rows, which all fall after its origin.
- **Rules.** Each rule's offset, R profile, and spread come from that rule's own forecast errors.
    (a) Coast: the gapped filter, then the offset and R. (b) Coast with widened bands: rule (a)'s 11
    quantiles with their offsets from the mean multiplied by sqrt(1 + v / s^2), where s = (q95 -
    q05) / 3.29 of the row, in points, and v is the extra predictive variance of the level and
    transient, in points squared, over the no-gap filter at the same cutoff. With n and n0 the
    half-hours from the target band's last reading to the cutoff in the gapped and the no-gap
    stream, v = max(0, w(n) - w(n0)), where w(n) = q n / 48 + sigma_z^2 (1 - phi^(2n)) / (1 -
    phi^2), using the single pooled setting (q, phi, sigma_z) of experiment 133 that arm A uses.
    (c2) The no-gap pipeline with the filter dropped for all runs: the frozen GAM plus its own
    offset and R. (c) The frozen GAM alone, with a spread refitted on its tuning months, no offset
    and no R. Secondary: (a0) and (b0) are (a) and (b) with the filter alone, no offset and no R,
    spread refitted on the filtered tuning errors.
- **Scored.** Five decision substations, D5, D10, D11, D14, and D16, over leads 0.5 to 168 hours,
    then under 24 hours and 12 to 36 hours; per-substation contrasts and the number of events per
    substation and length are reported beside the pooled ones.
- **Compute.** CPU only, 4 cores, no new fit; the filter re-runs on a masked stream per (substation,
  L, k).

Prior advice followed: the Kalman state is the one of experiment 111 with the POOL setting
(experiment 136, A2); no faster level is used (experiment 197); recipe R at every lead and no offset
on top of R is experiment 155's finding, but this plan keeps arm D, the pipeline that experiment 155
tested, so the offset stays in the gapped pipeline and the comparison stays with the stored arm D.
