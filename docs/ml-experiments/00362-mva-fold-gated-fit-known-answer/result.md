---
id: "00362"
slug: mva-fold-gated-fit-known-answer
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
kind: other
size: >-
  medium. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one gate would defensibly test the
  hypothesis (a constraint inside the fit, a start-selection rule, or a prediction-time clip); the
  start-selection rule is chosen because it needs no change to `packages/studies`. Callers: reuses
  experiment 358's and experiment 110's scripts, whose callers are all in those folders.
hypothesis: >-
  A gate that accepts reverse flow only where an independent solar forecast exceeds the fitted
  demand stops the fit through the meter from inventing reverse flow at a substation with none,
  without losing the reverse flow that real solar output produces.
prediction: >-
  At the natural level (no injection) at D5, the gated fit's sunny-half-hour direction accuracy is
  at least 0.95 (the signed fit's is 0.98; the ungated fit's was 0.35). At injection levels 1.0 and
  2.0 the gated fit stays within 0.02 of the signed fit's accuracy at 4 or more of 5 sites, and its
  fitted capacity stays within 15% of the signed fit's at 4 or more of 5 sites at level 1.0.
decision_rule: >-
  Passes (and step 2 runs) when, at the scored cut with reactive power zero, all three hold: D5's
  natural-level accuracy is within 0.02 of the signed fit's; at 4 of 5 sites at both levels the
  gated accuracy is within 0.02 of the signed fit's; and the capacity ratio to the signed fit is
  within 15% at 4 of 5 sites at level 1.0. Fails (and the sequence stops) otherwise. No
  cost-based threshold, because the experiment ports nothing.
benchmark: >-
  The ungated fit through the meter (experiment 358's L0, recomputed here), the signed fit, and the
  naive fit
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16; scored year 2025-07-01 to 2026-06-30, replicate
  2024-07-01 to 2025-06-30; frozen fits on past weather
metric: >-
  Share of sunny half-hours (true flow at least 5 percentage points of capacity from zero) with the
  flow direction right; ratio of fitted capacity to the signed fit's
verdict: refuted
production_status: not-for-production
adopted_in: none
date: 2026-10-28
best_arm: >-
  none: the registered gate (ratio 0.75) changed no fit, and the stricter post hoc gate (ratio 1.0)
  fixed D5 at the 2025 cut only
headline: >-
  A gate that accepts reverse flow only where independent solar output exceeds the fitted demand
  either does nothing (ratio 0.75) or rejects the correct fits when reverse flow is heavy (ratio
  1.0), so it does not stop the fit through the meter from inventing reverse flow at D5.
take_home_message: >-
  Checking a fitted reverse flow against an independent solar forecast does not catch the invented
  reverse flow at the substation that has none, because the invented flow appears on half-hours
  after the fit was made and the independent forecast of solar output is too rough to tell a
  slightly positive flow from a slightly negative one. A looser rule never rejects a fit, and a
  stricter rule rejects the correct fits where reverse flow is heavy. The next idea to test is a
  rule that uses the meter reading alone: accept reverse flow only if the training readings come
  close to zero at midday.
lessons:
  - >-
    An independent solar forecast is too coarse to gate the sign of the power flow, because true
    flow is often within a few percent of zero at midday.
  - >-
    A gate chosen after seeing the results at one cut did not hold at the other cut, so tune any
    such rule on one cut and check it on the other.
advice:
  - >-
    Do not gate reverse flow with an independent solar forecast; use a test on the meter readings,
    such as whether midday readings reach near zero in training (figure 1, post hoc ratio 1.0).
  - >-
    Report a gate's rejection counts per site, because a gate that rejects nothing leaves the fit
    unchanged and a gate that rejects the correct fits looks like a fit with the wrong sign.
---

# MVA fold: gated fit through the meter, known-answer test

**A gate that checks fitted reverse flow against an independent solar forecast does not stop the
fit through the meter from inventing reverse flow.** At the registered setting the gate rejected no
fit in any of 60 cases. At a stricter post hoc setting it fixed D5, the one Primary substation with
no reverse flow, at the 2025 cut only, and it rejected the correct fits at four of five substations
when reverse flow was heavy. Verdict: refuted. The sequence stops here.

Experiment 358 found that the model fitted through an apparent-power meter invents reverse flow at
D5 when no solar output is injected. This experiment tested a gate. The fit tries six starting
points. The gate rejects a start when more than 0.5% of its training half-hours have negative fitted
power while the independent solar output is under a set multiple (the ratio) of the fitted demand.
The independent solar output is the pinned capacity times the solar shape plus, in this lab test,
the injected farm's true output. We tested the registered ratio of 0.75 and, post hoc, the ratios
1.0 and 1.25. Everything else follows [experiment 358](../00358-mva-fold-known-answer-at-folded-mw-primaries/result.md).
The ungated fit (L0) reproduces experiment 358's numbers.

![Figure 1: share of sunny half-hours with the flow direction right, by fitting arm](figure_1_gated_direction_accuracy.svg)

## Results (reactive power zero; scored cut unless stated)

- **Registered gate (ratio 0.75) rejects nothing.** It rejects 0 of 6 starts in all 60 cases, so it
  gives the ungated fit's answer everywhere. D5 with no injection stays at 0.35 against the signed
  fit's 0.98. The rule fails on this one criterion: the gate matches the signed fit within 0.02 at 5
  of 5 substations at both injection levels, but misses D5 at the natural level by 0.63.
- **Why it rejects nothing.** On training half-hours the mirror fit's negative power falls where the
  independent solar output is at least 0.75 times the fitted demand, because true flow at midday is
  near zero and the independent forecast cannot tell a small positive flow from a small negative
  one. The invented reverse flow at D5 appears mostly on the following year's half-hours.
- **Post hoc ratio 1.0.** At the scored cut it fixes D5 with no injection (accuracy 0.96, capacity
  1.02 times the signed fit's) and matches the signed fit at level 1.0. It breaks level 2.0 at D11,
  D14, and D16 (accuracy 0.05 to 0.06 against 0.94 to 0.96, capacity 0.18 to 0.26 times the signed
  fit's), because the correct starts are rejected. At the 2024 cut it does not fix D5 (0.49, with
  every start rejected) and also breaks level 2.0 at D10, D14, and D16.
- **Ratio 1.25.** It breaks level 1.0 at D11, D14, and D5 and level 2.0 at all but D11 at the scored
  cut.

## Checks

- **Triggers.** Audit trigger 1 (a control fails: the D5 placebo) and trigger 2 (the result
  contradicts its prediction) fired. The explanation tested is the training-versus-test mismatch
  above. A fresh Sonnet auditor found the gate code a faithful copy of
  the ungated fit apart from the rejection step, found the units right, and reproduced experiment
  358's D5 numbers. It found no bug behind the null at 0.75: a start is rejected only when its fitted
  solar exceeds about 1.33 times the independent solar, and the wrong D5 starts overshoot by only
  about 1.2 times. It also noted that clipping the forecast power at zero cannot change a sign
  accuracy, so the clip is inert here.
- **Median against mean forecast.** Not applicable: point forecasts from frozen fits.
- **Departures.** Ratios 1.0 and 1.25 are post hoc. The audit used Sonnet because the series brief
  requires Sonnet only.

## Caveats and what to try next

- The gate reads the injected farm's true output, an advantage a real substation lacks, and it
  still failed.
- One farm shape (A) supplies the injected output.
- Try a rule on the meter readings alone: accept reverse flow only if the training readings come
  within a few percent of capacity of zero at midday. The brief's measurement says only Stickney
  (D12), Langrick (D22), and Marsh Lane (D8) do.
