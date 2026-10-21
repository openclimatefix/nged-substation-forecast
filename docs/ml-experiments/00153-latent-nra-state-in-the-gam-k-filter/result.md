---
id: "00153"
slug: latent-nra-state-in-the-gam-k-filter
title: "A latent normal-running-arrangement state inside the GAM-K filter"
theme: switching-labels-in-the-forecast
also_relevant_to:
  - switching-detectors
status: planned
size: >-
  medium. Stored: private score tables under data/ml-experiments/00153-latent-nra-state-in-the-gam-k-filter/
  only. Serving path: not touched. Degradation rule: not touched (an absent alarm leaves the filter
  equal to GAM-K's). Designs: more than one defensible way to give the transfer its own state; this
  plan runs the one the maintainer's brief names, and does not compare alternatives. Callers: this
  folder's scripts only, which copy experiment 141's modules and experiment 151's scoring harness.
hypothesis: >-
  Experiment 141's two-regime filter lets the level follow a load transfer, and 141 scored observed
  power, so neither tested the forecast of the normal running arrangement (NRA). If the filter
  instead holds the NRA as its own slowly moving state and routes the transfer into a separate,
  fast offset state that is switched on by the causal alarm and left out of the forecast, the
  forecast of the NRA moves towards the true NRA series during and after a transfer, and stays
  where it was when no transfer happens.
prediction: >-
  Written before any fit. On the post-alarm rows (forecast origin after the alarm is known, target
  inside the transfer or in the 48 hours after its true end), the new filter fed the detected
  intervals (NEW-DET) closes 40% to 70% of the gap in quantile score between K0 and the new filter
  fed the true intervals (NEW-ORACLE). Experiment 141's two-regime filter (S-KF) closes under 15%.
  On the placebo (no injected transfer), NEW-DET's median forecast differs from K0's by under 0.1
  points of effective capacity on average, and its quantile score is within 0.005 of K0's.
decision_rule: >-
  One decisive contrast, C1: the share of the gap closed, (K0 - NEW-DET) / (K0 - NEW-ORACLE), in
  mean quantile score against the true NRA series over the post-alarm rows, pooled over the two
  sites, with a 99% interval from 20,000 resamples of whole calendar months of target time (the
  same months for numerator and denominator). `supported` when the lower end of C1 is at least 40%
  and the placebo passes. `not supported` when C1's point value is under 20%. `partly supported`
  otherwise. The threshold follows the NRA rescore (experiment 151) so the two are comparable; the
  port needs the detector, its intervals, and a second filter state on the serving path, so a share
  under 40% at its lower bound does not pay for that. The placebo passes when the 95% interval of
  NEW-DET minus K0 in quantile score on the placebo rows lies within 0.01 points either side of
  zero. Every other contrast (S-KF, the shares against the clean-stream ceiling, the pre-alarm
  rows, the 48 hours after the detected end, the error of the median forecast, and the six-site
  widening) is read at 95% and decides nothing.
benchmark: >-
  K0 (GAM-K's filter on the stream with the transfers in it, no correction); NEW-ORACLE (the new
  filter fed the true intervals); CLEAN (the GAM-K filter on the stream without the injected
  transfers, as in experiment 151); S-KF (experiment 141's chosen two-regime filter).
row_set_id: >-
  fold_a_daily_gap_v1 rows of experiment 141 (09:00 UTC initialisation, 2025-07-02 to 2026-06-29
  and the even months of 2024-07 to 2025-06), leads 0.5 to 168 h, the two sites D11 and D16,
  experiment 141's injected worlds; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles) and absolute error of the
  median forecast against the true NRA series, in points of effective capacity; the share of the
  K0-to-NEW-ORACLE gap closed; coverage and pinball loss at q01, q05, q95, and q99.
code: ml-experiments/00153-latent-nra-state-in-the-gam-k-filter/
---

**This page is the plan. The result sections are filled in after the run.**

## Question

Does a Kalman filter that carries the normal running arrangement (NRA) as a latent state, and routes
a detected load transfer into a separate offset state, give a better forecast of the NRA than the
filters we have? The maintainer wants the switching detector for two purposes: reading recent power
observations better, and producing an NRA forecast. Experiment 141 ([PR 1453](https://github.com/openclimatefix/nged-substation-forecast/pull/1453), not yet merged) scored observed power, which
rewards following a transfer. [Experiment 151](../00151-nra-forecast-rescored-against-the-true-nra-series/result.md)
re-scores the existing filters against the true NRA. This experiment changes the filter: no earlier
design ties the Kalman state to a latent NRA. The edge-flow design of
[experiment 116](../00116-differentiable-physics-nra-edge-flows-between-primaries/result.md) treats the NRA as
the frozen GAM, and 141's regime S is a variance switch on the same level state. The filter is a
standard linear-Gaussian state-space model with a switched process noise, in the family of
[Harvey (1989)](https://doi.org/10.1017/CBO9781107049994) intervention models; we re-implement it, we do not
claim it as new.

## The filter

Per hour band, as in GAM-K, the filter reads the residual r of observed power minus the frozen GAM
forecast. State: the NRA level n (a random walk with GAM-K's fitted process noise, so it moves as
slowly as GAM-K's level), GAM-K's decaying anomaly z, and a transfer offset x. Observation:
r = n + z + x + noise. The offset x has zero mean. While an alarm interval is open its process
noise is a large multiple (4 settings pre-registered, see below) of n's; after the interval's end
is called it decays to zero by a factor of 0.5 per reading and has no process noise. The NRA
forecast is n plus the decayed anomaly, with x not carried forward.

GAM-K's level and the NRA level are the same state: two random-walk levels observed only through
their sum cannot be told apart, so a separate fourth state would add nothing. This differs from
the brief's wording ("two extra states") on that point only.

Intervals come from the merged causal RO-NC alarm with experiment 147's R2 rule (the onset is
known 96 half-hours after the step, the end is known 60 half-hours after the true end, the
6-hour lookahead of
[experiment 148](../00148-switching-end-rule-lookahead-ro-nc/result.md)). When an alarm becomes
known at a forecast origin, the filter is re-run from the interval's onset with x switched on over
[onset, now], so the step is not left in n. The filter is re-run once per distinct set of known
intervals, as experiment 151 does for subtraction.

## Arms

- **K0:** GAM-K's filter on the stream, no correction.
- **S-KF:** experiment 141's two-regime filter, with 141's chosen prior and multiplier, fed the
  detected intervals.
- **NEW-DET:** the new filter fed the detected intervals.
- **NEW-ORACLE:** the new filter fed the true transfer intervals with no delay (the ceiling).
- **CLEAN:** GAM-K's filter on the stream without the injected transfers (reported beside, not the
  denominator of C1).
- **Placebo (no injected transfer):** NEW-DET with the detected intervals shifted by a whole number
  of weeks and applied to the stream without any injected transfer, and scored against the observed
  series. K0 and NEW-DET should be equal there.

The ratio of x's process noise to n's is set on the tuning months (the odd months) from {10, 100,
1000, 10000}, scored by quantile score on NEW-DET's pre-registered post-alarm rows against the true
NRA, as 141 tuned its multiplier; the scored months are not used to choose it.

## Rows and windows

Experiment 151's: rows of experiment 141's worlds whose origin is after the alarm, with targets
inside the transfer or in the 48 hours after its true end; the detection lag (onset to alarm) and
the 48 hours after the detected end are reported separately. First test: sites D11 and D16 (the
two sites with the most injected transfers, 333 and 372 across 141's 44 worlds), the first 8
worlds. Widen to the six sites only if C1 passes.

## Checks and risks

- **Ceiling identity:** with no alarm intervals the new filter equals K0 to 1e-9; checked in the
  run script.
- **Leakage:** a forecast at origin c uses only intervals known at c, and stream values up to c
  minus the 3.5-hour telemetry delay.
- **Oracle gap:** a gap whose own interval includes zero reports no share.
- **Past weather:** the residual stream is on the frozen GAM-K, which uses past weather at lag
  times only through experiment 130's stored panels; the lab shortcut (ERA5 in the residual model)
  is the same as experiments 141 and 151, and the new filter adds no weather input.
- **Sharing:** the scoring harness is a copy of experiment 151's scripts at the time of writing; its
  session is asked to commit them so they can be diffed.

## Results

To be written after the run.
