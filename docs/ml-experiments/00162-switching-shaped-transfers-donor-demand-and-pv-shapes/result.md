---
id: "00162"
slug: switching-shaped-transfers-donor-demand-and-pv-shapes
title: >-
  Do load transfers follow the donor's demand and solar shapes, and does a constant-MW injection
  flatter the switching detectors?
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  medium. Stored: private tables and arrays under
  data/ml-experiments/00162-switching-shaped-transfers-donor-demand-and-pv-shapes/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible design (how
  the demand and solar terms are split, how a shaped transfer is matched to a constant one), so the
  matching rule and the arms are pre-registered. Callers: this folder's scripts only; they copy the
  modules of experiments 130, 147, and 156 they need and read their stored panels, worlds, and
  intervals without changing them.
hypothesis: >-
  A load transfer moves houses, factories, and solar sites, not a constant number of megawatts, so
  the power it moves follows the donor's demand and solar shapes. Part 1: on the logged real
  transfers, a donor residual during the transfer is fitted better by a daily shape, -(a x the
  donor's demand term + b x its solar term) from the convex model (GAM), than by a constant offset,
  and the fit separates a from b only for transfers that last several days and contain both sunny
  and dull days. Part 2: an injection whose moved power follows the donor's solar term vanishes at
  night and is small at midday in winter, so the half-size-reversal end rule and the step threshold
  of the detectors misfire on it, and a constant-MW injection (and, less so, today's injection,
  which moves a share of the donor's net power) flatters the detectors.
prediction: >-
  Written before any result. Part 1: for the logged transfers over 1 day, the shape model
  (constant plus a x demand term plus b x solar term, three parameters) cuts the residual sum of
  squares inside the interval by a median of 10% to 30% relative to a constant, measured on held-out
  alternate days, and a is positive (interval excluding zero) at more than half of the donors; b is
  identified (95% interval excluding zero) only at transfers of 3 days or more with both bright and
  dull days. A pure solar transfer and a pure demand transfer injected into quiet stretches can be
  told apart (the fit puts more than 70% of the moved energy in the correct term) from 3 days of
  both sunny and dull weather onward, and not below 1 day. Part 2: relative to a constant-MW
  injection of the same mean megawatts, a solar-shaped injection lowers the pair alarm's
  MW-weighted recall by 0.05 to 0.20 and lengthens the median lag from step to alarm by 2 to 12
  hours; a demand-shaped injection changes recall by less than 0.05; the end rule (the first
  half-size reversal) ends solar-shaped intervals early on 20% to 50% of the transfers; and the
  mask forecast scores 0.05 to 0.30 points of capacity worse on solar-shaped transfers than on
  constant ones.
decision_rule: >-
  Pre-registered before any fit. Models and tests are given in the plan section "Design"; the
  contrasts are 95% intervals with whole calendar months (part 2) or whole transfers (part 1)
  resampled 20,000 times, and the nine contrasts below share a Bonferroni level of 0.05/9 (99.4%).
  P1: median over the logged real transfers over 1 day of the held-out reduction in residual sum of
  squares of the shape model against the constant, 99.4% interval; `supported` when the lower end is
  above 0. P2: share of donors whose a has a 95% interval above zero, tested against one half.
  P3: on 200 positive-control injections into quiet real stretches, the share of moved energy the
  fit places in the right term, by duration (under 1 day, 1 to 3 days, over 3 days) and sunny share;
  reported, not tested. P4 to P6, on the same injected transfers in the 44 worlds of experiment 141
  (even months), difference in MW-weighted recall of the pair alarm (4-hour window, k of
  experiment 156), RO-NC's recall, and the median onset-known lag: solar-shaped minus constant.
  P7: the share of true interval length covered by the detected interval after the end rule:
  solar-shaped minus constant. P8: mask forecast mean quantile score on rows after onset: MASK-UNION
  on solar-shaped minus constant. P9: the same for the demand-shaped arm. The verdict is `supported`
  when P1 and at least two of P4, P5, P7, and P8 show the solar-shaped arm worse than constant with
  the interval excluding zero; `partly supported` when P1 is supported and fewer than two of those
  four are; `not supported` when P1 is not supported and no part-2 contrast excludes zero;
  `inconclusive` when fewer than 20 logged transfers over 1 day have usable terms. No threshold for
  porting is set, because the experiment changes no production step: its use is deciding whether
  the earlier constant-MW and share-of-net-power results can be trusted.
benchmark: >-
  The constant-MW arm (the same events, with the mean megawatts of the share-of-net-power
  injection of experiment 156 spread evenly over the interval) and, for the forecast, the
  weather-model-only arm and the true normal-running-arrangement series of experiment 151.
row_set_id: >-
  fold_a_daily_gap_v1 rows of experiment 141 (09:00 UTC initialisation, leads 0.5 to 168 h, six
  decision substations, the 44 injected worlds of experiment 141, rows outside the hand-filtered
  switching events), as in experiments 151 and 156; not comparable with the leaderboard
metric: >-
  Held-out reduction in residual sum of squares; fitted a and b with intervals; share of moved
  energy placed in the right term; MW-weighted recall; median onset-known lag (hours); share of the
  true interval covered; mean quantile score in points of effective capacity.
code: ml-experiments/00162-switching-shaped-transfers-donor-demand-and-pv-shapes/
verdict: partly supported
best_arm: >-
  Not applicable: no arm is a candidate for production. The constant-MW injection is the reference
  arm, and the mask driven by either alarm scores 3.023 points of capacity on it, against 3.169 on
  the arm where a daytime load leaves the donor and 3.484 on the arm where solar output leaves it.
headline: >-
  P1: constant plus daily shape reduces held-out squared error by a median +0.067 (99.4% interval
  -0.110 to +0.160) on 25 logged transfers, not supported. Solar-shaped injections, minus constant:
  RO-NC recall -0.398 (-0.601 to -0.269) when a daytime load leaves the donor and -0.405 when solar
  output leaves it; live end-rule coverage -0.403 (-0.552 to -0.280) and -0.419; pair-alarm recall
  -0.437 as drawn but only -0.072 (-0.130 to -0.013) when scored from first light. Mask forecast
  +0.149 points (-0.085 to +0.490), not supported.
lessons: >-
  The injections of experiments 141 and 156 (a drifting share of the donor's net power) are about as
  easy to detect as a constant transfer (pair alarm -0.037, RO-NC -0.006), so those results did not
  depend on the injection shape. A transfer that follows the donor's daytime solar shape is much
  harder for RO-NC and the live end rule. The pair alarm's apparent loss is mostly a scoring effect,
  because a solar-shaped step cannot start before sunrise. The data cannot yet say whether real
  transfers move solar output, because the solar share is not identified at any length we have.
take_home_message: >-
  If a transfer followed the donor's solar shape, the neighbour-count detector (RO-NC) would find
  about half of them instead of 91%, and the live end rule would cover 22% of the true interval
  instead of 62%. The pair alarm loses far less once a step is allowed to start at first light. The
  25 logged real transfers cannot say whether they move solar output, so earlier RO-NC and end-rule
  results may be optimistic by 0.3 to 0.46 in recall. Transfers shaped like the donor's demand or its
  net power are as easy to find as constant ones.
advice:
  - >-
    Treat RO-NC and live end-rule results on injected transfers as a ceiling for transfers that
    follow the donor's solar shape, because they lose about 0.40 recall and 0.40 coverage (P5, P7).
  - >-
    Score a solar-shaped transfer from the donor's first light, not from the drawn onset, because
    the pair-alarm loss falls from -0.44 to -0.07 under that scoring (P4).
  - >-
    Test a detector on injections that move only solar output as well as on demand-shaped ones,
    because constant-MW injections hide the gap for RO-NC and the end rule (P5, P7).
  - >-
    Do not take a bootstrap interval over whole days as the test that a transfer is solar-shaped; it
    flags a quiet stretch 14% to 16% of the time, so calibrate against placebo stretches of the same
    length (controls).
  - >-
    Expect to identify a demand share only from transfers of 3 days or more, and a solar share not
    at all below 7 days, and read these as best cases because the injected shares are recovered
    without bias by construction (figure 3).
  - >-
    Keep masking readings as the response to a suspected transfer, because masking assumes no size
    or shape, and do not subtract a fixed MW (design note, not tested here).
production_status: not-for-production
adopted_in: none
---

**Take-home message.** A load transfer is an operator moving demand from one substation to a
neighbour by switching. If a transfer followed the donor substation's solar shape, the refit-offset
neighbour-count detector (RO-NC) would find about half of the transfers instead of 91%, and the live
end rule would cover 22% of the true interval instead of 62%. The pair alarm of [experiment
156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md) loses much less once a step
may start at first light. The 25 logged real transfers cannot say whether they move solar output: a
daily shape fits them no better than it fits quiet stretches. Earlier RO-NC and end-rule results may
therefore be optimistic by 0.3 to 0.46 in recall. Transfers shaped like the donor's demand, or like
its net power (the shape the earlier experiments injected), are as easy to find as constant ones.

**We asked whether the power a transfer moves follows the donor's demand and solar shapes, and
whether injecting a constant number of megawatts flatters the detectors.** The idea comes from the
roadmap's [magnitude-only mixture model](../../roadmap/switching-events.md): a switch moves houses,
factories, and solar sites, not a constant. Part 1 split the convex model of
[experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md) (a generalised additive
model, GAM) into its demand term and its solar term, and fitted each logged transfer's effect on the
donor's residual as a constant, as a daily shape, and as both, scoring on held-out days. Part 2
re-injected the 1,160 transfers of [experiment 141](../00141-switching-detector-fed-into-gam-k/result.md)
with five rules of the same mean megawatts at the donor (constant, demand-shaped, two solar shapes,
half demand and half daytime load), beside the original share-of-net-power rule, and re-ran the
detectors, the end rule, and the mask forecast of experiment 156. [Experiment
60](../00060-sw2-7-profile-weighted-step-detector/result.md) had already found that a detector
weighted by the donor's profile finds the same share of moved power on five shapes; the solar-only
shapes are new here.

**The verdict is `partly supported`, and the pre-registered rule did not cover this outcome.** The
rule gave `supported` when part 1 and part 2 both held and `not supported` when neither did. Part 2
held for RO-NC and the end rule, and part 1 did not (P1 and P2 failed). The verdict was set to
`partly supported` after the results, and the rule has a gap.

![Contrasts against the constant-MW injection](figure_4_contrasts.svg)

![Mean score by forecast and injection shape](figure_5_scores.svg)

![Held-out reduction in squared error](figure_2_held_out_reduction.svg)

![Share of injected transfers identified, by length](figure_3_identification.svg)

![Demand term and solar output over the day](figure_1_profiles.svg)

- **Arms.** Six injections with the same mean megawatts at the donor (0.865 to 0.872 MW across
  arms): constant (CONST, the reference), share of net power (SHARE, exactly the injection of
  experiments 141 and 156), demand-shaped (DEMAND), a daytime load that leaves the donor (SOLAR,
  proportional to the donor's solar output, subtracted from the donor), half demand and half daytime
  load (MIX), and solar output that leaves the donor, so the donor's net power rises at midday
  (PVOUT). Shaped arms divide the donor's term by its mean over the transfer. Detectors, thresholds,
  and scoring code are those of experiment 156 (4-hour pair-alarm window, k = 3.75); SHARE reproduces
  experiment 156's event table exactly and its mask scores to three decimals. Convex-model terms
  come from monthly refits, and solar plus demand equals experiment 130's stored forecast to 5e-15
  MW on six series. Series 12 has no terms; series 13 has terms from 2025-07 to 2026-06 only. 37
  DEMAND events and 57 SOLAR, MIX, and PVOUT events have a donor without terms, or no daylight, and
  use the donor's net-power shape instead.
- **Weather.** ERA5 at lag times, as in experiments 130 and 156, so every number is a lab ceiling:
  ERA5 arrives about 5 days late and is not available at forecast time.
- **Part 1, logged transfers over 1 day (35; 25 have a fitted series; 37 series fits; 13 logged
  donors).** P1, constant plus shape against constant on held-out days: median +0.067 (95% -0.033 to
  +0.126; 99.4% -0.110 to +0.160), not supported. Shape alone +0.069 (99.4% -0.111 to +0.138). The
  placebo median is +0.004 over all stretches; a placebo matched to the transfers' lengths has a
  median of +0.052, and 39% of such placebos reach +0.067. The demand share is beyond the placebo's
  2.5th percentile at 5 of 13 logged donors (0.38; exact 95% 0.14 to 0.68), the solar share beyond
  the 97.5th percentile at 0 of 13; neither is above one half (P2). Median fitted shares: 0.130
  demand and 0.140 solar. The 181 intervals RO-NC detected over a day (experiment 147): median
  +0.005; they are chosen because a step was found, so they are not evidence about transfers.
- **Receivers (12 receiver fits).** The donor's terms fit the receiver's opposite step better than
  its own terms in 8 of 12 (held-out reduction +0.091 against -0.004). That is the direction a
  transfer predicts, from 12 fits.
- **P3, injected known answers (182 per scenario, the same stretches in each).** The injected
  shares are recovered without bias (demand 0.158 for 0.15, solar 0.349 for 0.35), as the fit
  contains the injected shape, so the identification rates below are best cases. With the
  placebo-calibrated rule, the demand share is identified in 0.10 of transfers of 1 to 3 days, 0.75
  of 3 to 7 days, and 0.96 of over 7 days. The solar share is identified in 0.07, 0.08, and 0.29. The
  under-1-day band reads 0.52 for demand from 33 short stretches and is not trustworthy.
- **P4 to P7, even-month transfers (611), minus constant, 99.4% interval, onset as drawn.**
  Pair-alarm recall: SOLAR -0.437 (-0.633 to -0.300), PVOUT -0.463 (-0.663 to -0.368), MIX -0.279
  (-0.417 to -0.180), DEMAND -0.019 (-0.089 to +0.017), SHARE -0.037 (-0.085 to -0.011). RO-NC recall:
  SOLAR -0.398 (-0.601 to -0.269), PVOUT -0.405 (-0.629 to -0.231), MIX -0.042 (-0.107 to -0.011),
  DEMAND -0.004, SHARE -0.006. Absolute recall (pair alarm, RO-NC): CONST 0.786 and 0.907; SOLAR
  0.349 and 0.509; PVOUT 0.324 and 0.503. P6, the lag from step to alarm, is 6 hours for every arm,
  because the window fixes it. P7, share of the true interval the live end rule covers (missed
  transfers count 0): CONST 0.620, SOLAR 0.217, PVOUT 0.201, MIX 0.496; SOLAR minus CONST -0.403
  (-0.552 to -0.280).
- **Scored from first light (post hoc).** A solar-shaped step cannot start before sunrise, so a
  drawn onset at night is not a fair target for an alarm. Matching the donor's alarm from first
  light changes the pair alarm to SOLAR -0.072 (-0.130 to -0.013) and PVOUT -0.100 (-0.166 to
  -0.020), but RO-NC stays at SOLAR -0.406 and PVOUT -0.460. The end rule, scored on the covered
  share of the interval, loses about 0.40 under either scoring. Most of the pair alarm's loss in the
  as-drawn score is therefore a scoring effect, and the RO-NC and end-rule losses are real.
- **P8 and P9, mask forecast on all 277,056 rows after an onset (SOLAR 274,294 paired rows), minus
  constant.** MASK-UNION: SOLAR +0.149 points of capacity (99.4% -0.085 to +0.490), PVOUT +0.464
  (-0.008 to +0.923), DEMAND +0.066 (-0.028 to +0.158), MIX -0.008 (-0.080 to +0.118), SHARE +0.196
  (-0.115 to +0.322); none is supported at the corrected level. Scores of no correction, weather
  model only, mask from RO-NC, mask from either alarm, true series: CONST 3.576, 3.025, 3.160, 3.023,
  2.416; SHARE 3.549, 3.025, 3.338, 3.219, 2.416; DEMAND 3.473, 3.025, 3.187, 3.089, 2.416; MIX 3.487,
  3.025, 3.281, 3.015, 2.416; SOLAR 3.805, 3.019, 3.505, 3.169, 2.407; PVOUT 4.218, 3.019, 3.791,
  3.484, 2.407.
- **Why the forecast contrast is flat although recall falls.** The audit found that on rows the
  recall metric counts as missed, the mask still improves on no correction by 0.72 points on the
  SOLAR arm, because it masks a window that covers part of the transfer. The recall metric therefore
  understates what the mask achieves on solar-shaped transfers.

## Checks

- **Median minus mean forecast** by site, points of capacity: D10 -0.076, D11 -0.044, D14 +0.084,
  D16 +0.040, D5 -0.347, D9 -0.091, the same for every arm.
- **Mean-forecast contrast** has the quantile score's sign everywhere: SOLAR minus CONST +0.210
  (99.4% -0.102 to +0.661) in the error of the mean forecast and +0.149 in the quantile score.
- **Triggers fired.** A control failed: the 95% bootstrap intervals of the fitted shares flag 16%
  (demand) and 14% (solar) of placebos, so part 1 uses a placebo-calibrated rule. A headline
  interval is more than twice as wide as the same contrast's in experiment 156 (P8 for SOLAR is 0.58
  points wide, experiment 156's C4 was 0.22). A Sonnet code review found the bootstrap intervals too
  narrow, P2 chosen by the fit, a biased energy-share measure, contaminated placebos, and P7 dropping
  missed transfers; each was fixed before the numbers above. An Opus audit then found a wrong sign
  on the solar arm (fixed by relabelling it a daytime load and adding PVOUT), onset scoring, inherited
  weights, and the by-construction recovery in P3. The findings and their triage are in the audit
  report, kept in the private data store.

## Caveats

- **P1 and P2 rest on few transfers.** Twenty-five logged transfers, 13 with a logged donor, and
  placebos of 33 to 73 stretches per length band. A shape effect of the size the controls inject (15%
  of demand) would be found in about three quarters of transfers longer than 3 days; the real
  transfers show less, but few are that long.
- **Real transfers may move solar output in a way the donor's fitted solar term does not
  describe**, for example because the donor's solar capacity changed during a transfer.
- **The solar arms are extreme cases.** The realistic mix of houses and small solar sites is nearer
  the SHARE arm (RO-NC -0.006, pair alarm -0.037) than the SOLAR or PVOUT arms, which move no demand.
- **Missing terms.** Series 12 has no terms, which cuts the real transfers from 35 to 25, and the
  donor's net-power shape stands in for 37 to 57 injected events.
- **Mean-matching a solar shape raises its midday power** to several times the mean, so the solar
  arms also differ from the others in peak size.
- **Lab conditions** as in experiment 156: ERA5 residuals, balanced steps, hindsight end rule in the
  RO-NC intervals that the live end rule replaces only in P7. Recall weights are inherited from the
  constant arm's megawatts.

## What the next experiment should answer

Whether real transfers move solar output. The cheapest test uses the 35 logged transfers with
their metered receivers: fit the receiver's opposite step and the donor's step jointly, so that the
two together identify the solar share from a sum constrained to be zero. A second test replays
RO-NC and the end rule on the solar arms with a daylight-only step test, to see how much recall that
recovers.

## Changes to the pre-registered plan

- **Prior art (added after the plan).** Experiment 60 had already compared constant and profile
  injections for a profile-weighted detector.
- **Interval method (after the code review).** Bootstrap intervals over whole days flagged 14% to
  16% of placebo stretches, so the P2 and P3 tests use the placebo-calibrated rule, and the
  bootstrap intervals are reported beside them. The pre-registered statement that the solar share
  "is identified only for transfers of 3 days or more" rested on the bootstrap intervals.
- **P2 donors** are the logged donors, not the series with the largest fitted effect; P3's share
  of moved energy was dropped for a bias towards the demand term, and recovered shares are reported
  instead; P7 counts missed transfers as 0; controls and placebos avoid RO-NC's detected intervals
  and use the same stretches in each scenario.
- **P1's model** is constant plus shape (three parameters), as the prediction states.
- **After the audit:** the SOLAR arm is a daytime load leaving the donor, and the PVOUT arm was
  added for solar output leaving the donor; first-light scoring, the daylight-only sensitivity, and
  the receiver check are post hoc.
