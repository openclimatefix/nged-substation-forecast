---
id: '00205'
slug: dp-gamk-gap-robustness-coast-or-frozen-gam
part: GAP1
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- lagged-power-and-error-correction
status: done
size: >-
  medium. Stored: private files under
  data/ml-experiments/00205-dp-gamk-gap-robustness-coast-or-frozen-gam/
  only. Serving path: not touched. Degradation rule: not changed (the result is evidence for a later
  rule about missing power readings, which is a separate issue). Designs: more than one is
  defensible
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
  month-resampled paired bootstrap (the 18 year-and-month groups of the origins, 20,000
  resamples), which
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
  reproduces experiment 155's arm D to within 0.001 points on the mean forecast; replacing, with
  random numbers, every actual at or after a run's origin leaves that run's mean forecast
  unchanged; and rule (b) with the
  extra variance set to zero equals rule (a) exactly.
benchmark: >-
  The no-gap score of the same event runs under the same pipeline (arm D of experiment 155, rebuilt
  here), rule (c2), and the frozen GAM alone (rule c)
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, over leads 0.5 to 168 hours on rows outside the hand-filtered switching
  events,
  and under 24 hours and 12 to 36 hours; q05 and q95 coverage; mean absolute error of the mean
  forecast beside every quantile-score contrast
code: ml-experiments/00205-dp-gamk-gap-robustness-coast-or-frozen-gam/
verdict: inconclusive
best_arm: >-
  Coast with widened bands (rule b): 2.759, 2.788, 2.857, and 2.948 points over leads 0.5 to 168
  hours after gaps of 3, 6, 24, and 72 hours, against 2.747 with no gap, 2.905 to 2.940 for the
  same pipeline with the filter dropped (c2), and 3.198 for the frozen model alone (c)
headline: >-
  Coasting beats the frozen model alone at every gap length tested, by 0.245 to 0.439 points
  (99.4% intervals wholly below zero). Against dropping only the filter, coasting is better by
  0.147, 0.116, and 0.057 points at 3, 6, and 24 hours and worse by 0.013 at 72 hours, but every
  one of those four intervals spans zero, so no crossover length is established. The verdict is
  inconclusive on the pre-registered rule.
lessons:
  - >-
    In these tests, dropping the whole forecast to the frozen model when readings stop was
    worse: coasting scored 0.245 to
    0.439 points better than the frozen model alone at gaps of 3 to 72 hours (C-a-minus-c).
  - >-
    The filter's advantage over dropping it shrinks as the gap grows: 0.147 points after 3 hours and
    0.013 worse after 72 hours, and the 72-hour estimate crosses zero, but 275 events in all
    cannot resolve it (C1 to C4 all span zero), and one switching date drives the 72-hour sign
    (audit).
  - >-
    Widening the bands by the filter's own variance growth costs nothing in score (-0.004 to +0.000)
    and moves the q05 coverage after a 24-hour gap from 7.7% to 5.5% (nominal 5%).
  - >-
    Recovery is slower with recipe R and the offset than without: with the filter alone the
    score is within
    0.02 points of its no-gap value 24 hours after a 24-hour gap, and with recipe R it is not within
    48 hours (+0.025); recipe R's loss of source days is the suspected cause and was not tested.
take_home_message: >-
  When readings stop for 3, 6, 24, or 72 hours, keep running the Kalman filter, which corrects the
  convex demand model using its recent forecast errors, without feeding it new readings, and widen
  the quantile bands. At each of those four gap lengths that forecast scored 0.25 to 0.44 points
  of quantile score better than the convex demand model with no correction (a quantile score is an
  error measure in percent of the substation's effective capacity, and lower is better; the gain
  is 8% to 14% of the uncorrected model's score). At 72 hours the 99.4% bootstrap interval for the
  gain runs from 0.04 to 0.48 points, so the gain may be small. The data cannot show whether
  dropping only the filter beats coasting after a gap of about 72 hours: at every gap length the
  difference between the two forecasts is smaller than the variation between months. The forecast
  with the filter, winter night offset, and recipe R was still 0.025 points worse than its no-gap
  score 48 hours after a 24-hour gap ended. The filter alone was within 0.02 points of its no-gap
  score 24 hours after the gap ended.
advice:
  - >-
    Serve coasting (the filter running with no update) when power readings are missing, for gaps of
    3 to 72 hours; do not fall back to the frozen demand model alone (C-a-minus-c: -0.245 to -0.439
    points).
  - >-
    Widen the quantile offsets by the filter's variance growth since the last reading when readings
    are missing: it fixed the q05 coverage without a score cost (figure 4, C5 to C8).
  - >-
    Do not port a rule that drops the filter after a long gap: the 72-hour contrast (C4, +0.013,
    interval spanning zero) rests on one switching date and is not evidence of a crossover
    (audit).
  - >-
    Run an experiment that needs recovery after a gap with a filter-alone arm beside the full
    pipeline: the full pipeline stayed above the 0.02-point margin at 48 hours after a 24-hour
    gap while the filter alone recovered (figure 3).
date: 2026-10-22
pr: 1508
production_status: candidate
adopted_in: none
---

# Gap robustness of the convex model with a Kalman filter: coast or fall back

**Take-home: when power readings stop, keep the Kalman filter running without new readings
("coasting") and widen the bands.** At each of the four gap lengths tested (3, 6, 24, and 72 hours)
that forecast scored 0.25 to 0.44 points of quantile score better than the convex demand model with
no correction. A quantile score is an error measure in percent of the substation's effective
capacity, and lower is better. Whether dropping only the filter would beat coasting at about 72
hours could not be told. The formal verdict is `inconclusive`: each comparison planned in advance
against the same forecast with the filter dropped has an interval that includes zero.

We asked what a live forecast should do when power readings go missing: let the filter coast, coast
and also widen the bands, or drop the filter. The demand model is the convex generalised additive
model of [experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)
with a Kalman filter on its recent errors (GAM-K, a method re-implemented from [de Vilmarest and
Wintenberger (2022)](https://arxiv.org/abs/2104.10777)), plus the winter night offset and recipe R
of arm D of [experiment
155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md). We hid every
reading in the 3, 6, 24, or 72 hours ending at the forecast's data cutoff (the time of the last
usable reading), for up to 56 forecast runs (events) at each of five substations (275 events in
all), and scored the forecast over the next 168 hours. With coasting, the quantile score (2.747
points with no gap) worsened by 0.011 points after a 3-hour gap and by 0.206 points after a 72-hour
gap. With the filter dropped, it worsened by 0.158 to 0.193 points at every gap length.

![Each rule's score minus the no-gap score by gap length](figure_1_headline.svg)

![Coasting minus the two frozen alternatives](figure_2_contrast.svg)

![Score of coasting against hours since the gap ended](figure_3_recovery.svg)

![Coverage of q05 and q95 with and without wider bands](figure_4_coverage.svg)

## Headline numbers

All differences are in points of quantile score, over leads of 0.5 to 168 hours. An interval is
the 99.4% bootstrap interval, resampled by calendar month, and "spans zero" means it includes zero.
Rule letters: (a) coast, (b) coast with wider bands, (c2) drop the filter, (c) the frozen model alone.

- **Rules.** (a) The filter on the gapped readings, then the winter night offset (a monthly refitted
    correction to winter night forecasts) and recipe R (subtract 0.75 times the mean forecast-error
    profile of the previous 14 days). (b) Rule (a) with the quantile offsets widened by the filter's
    variance growth since the last reading. (c2) Rule (a) with the filter dropped. (c) The convex
    demand model with no filter, offset, or recipe R, called the frozen model. The plan below fixes
    these rules before any score was read.
- **Events.** Five decision substations (D5, D10, D11, D14, and D16) with 53 to 56 events each (275
    in all), one randomly drawn forecast run per 8-day block. The gap ends at the data cutoff (k = 0,
    where k is the number of hours between the gap's end and the cutoff). The spread tables, which
    hold the quantile offsets from the mean, come from the no-gap forecast of experiment 136's arm
    A2.
- **Absolute scores.** No gap 2.747. Rule (a) 2.758, 2.790, 2.857, and 2.953 at 3, 6, 24, and 72
    hours. Rule (b) 2.759, 2.788, 2.857, and 2.948. Rule (c2) 2.905, 2.905, 2.914, and 2.940. Rule
    (c) 3.198, whatever the gap.
- **Coasting minus dropping the filter (rule (a) minus rule (c2)).** 3 hours -0.147 (-0.321 to
    +0.030). 6 hours -0.116 (-0.287 to +0.060). 24 hours -0.057 (-0.232 to +0.123). 72 hours +0.013
    (-0.152 to +0.203). All four span zero, so no crossover length is established and the
    pre-registered test fails. Over leads under 24 hours the contrast is below zero at 3, 6, and 24
    hours (-0.480, -0.401, and -0.295) and spans zero at 72 hours (-0.167, -0.353 to +0.022).
- **Coasting minus the frozen model alone (rule (a) minus rule (c)).** -0.439 (-0.666 to -0.241),
    -0.408, -0.341, and -0.245 (-0.478 to -0.040) at 3, 6, 24, and 72 hours. The plan treats this
    contrast as descriptive.
- **Wider bands (rule (b) minus rule (a)).** The score changes by -0.004 to +0.000 at every length,
    and every interval spans zero (72 hours: -0.004, -0.033 to +0.014). The share of actual
    readings below the predicted 5% quantile (q05) changes by -0.006, -0.011, -0.023, and -0.028
    (24 hours: -0.032 to -0.015), and the share below q95 by +0.003, +0.007, +0.014, and +0.018.
- **Coverage.** Under rule (a), 7.3%, 7.6%, 7.7%, and 8.4% of actual readings fall below q05 at
    the four gap lengths, and 95.1%, 94.9%, 94.5%, and 94.2% fall below q95. Under rule (b) the
    shares are 6.6%, 6.4%, 5.5%, and 5.6% below q05 and 95.5%, 95.6%, 95.9%, and 95.9% below q95.
    With no gap the shares are 7.1% and 95.1%.
- **Rule (b) meets the plan's criterion for replacing rule (a).** The mean absolute distance of the
    q05 and q95 shares from nominal is 0.016 for rule (a) and 0.007 for rule (b) at 24 hours, and
    0.021 and 0.008 at 72 hours. The intervals of the coverage differences exclude zero (24 hours:
    q05 -0.032 to -0.015, q95 +0.010 to +0.018), and the score difference is at most +0.000
    against the +0.01 limit. The coverage differences use the month-resampled interval rather than
    the plan's binomial standard error, because forecast errors in the same month are correlated.
- **Tails under rule (a), 3 to 72 hours.** The pinball loss at the 1% quantile is 0.222 to 0.257
    and at the 99% quantile 0.231 to 0.242. The share of readings below the 1% quantile is 2.5% to
    3.2% and below the 99% quantile 98.1% to 98.4%. Under rule (b) those shares are 1.6% to 2.2%
    and 98.6% to 98.8%.
- **Recovery.** Recovery is the smallest k from which the score stays within 0.02 points of the
    no-gap score. After a 24-hour gap, rule (a) minus the no-gap score is +0.110, +0.074, +0.061,
    +0.060, +0.012, and +0.025 points when k is 0, 3, 6, 12, 24, and 48 hours, so the pre-registered
    recovery (6 hours or less) fails. The filter alone (rule (a0)) is within 0.02 points from 24 hours
    on. After a 3-hour gap, recovery takes 0 hours; after a 6-hour gap, 24 hours; after a 72-hour
    gap, the score has not recovered at 48 hours.
- **Differences between substations (rule (a) minus rule (c2)).** At 3 hours: D10 -0.171, D11
    -0.228, D14 -0.342, D16 +0.056, and D5 -0.040. At 72 hours: D10 +0.033, D11 -0.012, D14 -0.235,
    D16 +0.222, and D5 +0.061.

## Checks

- **Median minus mean forecast.** Exactly 0.000 at every substation and rule, because the centred
    spread table places the median at the mean. The check cannot expose a shifted forecast here.
- **Mean-error contrast beside the quantile score.** Rule (a) minus rule (c2) on the absolute
    error of the mean forecast is -0.270 (below zero), -0.219, -0.124, and -0.006 points at 3, 6,
    24, and 72 hours. The sign matches the quantile-score contrast at 3, 6, and 24 hours and differs
    at 72 hours, where both contrasts span zero (mean error -0.233 to +0.234).
- **Controls.** The rebuilt no-gap forecast reproduces experiment 155's stored arm D exactly (the
    largest mean difference is 0.0 at all five substations). Widening by zero variance leaves the
    quantiles unchanged. Replacing every actual at or after a run's origin with random numbers
    leaves the mean forecasts of three tested runs unchanged.
- **Audit.** The skill's audit check fired because leaving out D14 moves the 24-hour contrast from
    -0.057 to +0.001, and leaving out D16 moves the 72-hour contrast from +0.013 to -0.040. An
    independent reviewer re-derived the contrasts from the stored forecasts, and they match to 4
    decimal places. The reviewer found no path by which future readings reach the forecast. The
    reviewer traced both sign changes and the 72-hour sign to one date, 2026-01-13, at D5 and D16
    (the first finding below). Both contrasts span zero, so the verdict does not change.

## Audit findings

- **One switching date carries the 72-hour sign and both leave-one-out flips.** On 2026-01-13 the
    no-gap forecast scored 15.3 points at D16 and 16.9 at D5, against 1.7 and 7.6 for rule (c2). The
    14 days before that cutoff all fall inside a hand-filtered switching event, but the filter still
    read those readings. It had learned the changed load, moving its level estimate by +22 points at
    D16 and by -9 points at D5. The hidden gap did not cause the large errors. Dropping that date
    after the fact (a post-hoc cut, not the verdict) gives rule (a) minus rule (c2) of -0.204
    (-0.363 to -0.059) at 3 hours and -0.171 (-0.327 to -0.023) at 6 hours, both below zero. The
    same cut gives -0.112 and -0.037 at 24 and 72 hours, both spanning zero. On that cut the
    pre-registered rule would read `partly supported`. The 72-hour sign, coasting worse, is not a
    basis for a rule that drops the filter.
- **At 3 and 6 hours the contrasts mostly measure the filter's no-gap value.** Every forecast
    starts at 09:00 UTC and its data cutoff is 05:30 UTC, so a 3-hour gap covers only the night hour
    band (and the morning band in summer time). A band here is a block of hours of the day that
    gets its own correction. The contrast splits into +0.011 from the gap and -0.158 from the
    filter's no-gap advantage. A gap ending in the evening would test more.
- **The widening is crude.** Most of the extra variance comes from the filter's short-lived
    transient component. The method applies it unchanged at every lead up to 168 hours, although it
    should shrink as the lead grows. Part of the coverage gain also corrects under-coverage that the
    no-gap forecast already has (7.1% of readings fall below q05, against a target of 5%).

## Caveats

- **Gaps are injected, not real.** Hiding readings in clean data does not reproduce a stuck meter,
    a communications fault that backfills later, or a gap at a switching event.
- **The demand model's monthly refits continue during a gap.** A gap that spans a refit cut would
    also thin the training data, and that effect is not modelled.
- **The quantile offsets differ slightly from those of the no-gap forecast.** The offsets differ by
    up to 1.4 points, because the winter night offset and recipe R move on rows near a gap. Rule
    (c) has its own table.
- **Hidden readings stay hidden for later runs.** A telemetry backfill would restore them, so the
    recovery shown is a worst case.
- **The 275 events lie in 18 calendar months,** and the intervals resample those months.

## What to try next

- **Real outages would be a more realistic test.** They could be scored once the cleaned power
    table's outage periods are listed. Resolving the 72-hour question would take more events or
    real gaps.
- **Masking switching readings from the filter could be tested.** One date shows that unmasked
    readings can raise the quantile score by 14 points. A test that masks them would also settle the
    72-hour comparison.
- **A small neural network could shorten the recovery.** It would take the filter state and the
    hours since the last reading and rescale how fast the transient decays. If it works, the score
    after a 24-hour gap would be within 0.02 points of the no-gap score 12 hours after the gap ends.

## Planned design

The design below was fixed before any score was read. Two details changed after the plan review and
before any run: the comparator for the headline contrasts is rule (c2), and the gap hides the
actuals from the offset and recipe R as well as the filter.

**Question.** In production the convex demand model with a Kalman filter (GAM-K, the Kalman filter
on recent errors of [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md); the method
re-implements [de Vilmarest and Wintenberger (2022)](https://arxiv.org/abs/2104.10777)) will
sometimes receive no power readings for hours or days. We ask which rule to serve then, and whether
a gap length exists beyond which the filter should be dropped.

**Design.**

- **Inputs.** The unfiltered forecast of every run (experiment 136's A2: the T-all fits driven by
    the calibrated ensemble mean), rebuilt from the stored fits and ensemble arrays as in
[experiment
    197](../00197-dp-kalman-level-adaptation-retest/result.md), and the filter's residual stream.
The
    full pipeline is the arm D of [experiment
    155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md): the filter,
the
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
