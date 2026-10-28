---
id: "00358"
slug: mva-fold-known-answer-at-folded-mw-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
kind: other
size: >-
  medium. Stored: predictions and small result files in the private folder only. Production serving
  path: not touched. Degradation rule: not touched. Designs: more than one would test the
  hypothesis (full ensemble engine with monthly refits, or one frozen fit on past weather); the
  frozen fit is chosen because the question is whether the sign is identified, not the lead-time
  skill. Callers: the code reused is experiment 110's `fit_gam_through_meter` and injection helpers,
  whose callers are all in experiment 110.
hypothesis: >-
  When the signed power of a Primary substation is folded to apparent power (its magnitude), the
  convex model fitted through the meter (experiment 110's L0) recovers the sign of the flow and the
  solar capacity, and a model that reads the folded reading as signed power does not.
prediction: >-
  On sunny daylight half-hours with the true flow at least 5 percentage points of capacity from
  zero, L0's
  balanced accuracy of the flow's sign is 0.85 to 0.95 at injection level 2.0 and 0.75 to 0.85 at
  level 1.0, at 4 of the 5 sites; the capacity L0 fits is within 15% of the capacity fitted to the
  signed series; L0's mean absolute error on the folded reading is lower than the naive model's by
  at least 0.10 points at level 2.0.
decision_rule: >-
  Passes (and step 2 runs) when, at the scored cut, under zero reactive power, balanced accuracy on
  sunny half-hours is at least 0.80 at 4 of 5 sites at both level 1.0 and level 2.0, and the fitted
  capacity is within 15% of the signed fit's at 4 of 5 sites at level 1.0. Fails (and the series
  stops) otherwise. Threshold is the design brief's K1 and K4; there is no cost-based threshold
  because the experiment ports nothing.
benchmark: >-
  The same convex model fitted to the signed injected series (arm S, ceiling), and the same model
  fitted to the folded reading as if it were signed power (arm N)
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16; scored year 2025-07-01 to 2026-06-30, replicate
  2024-07-01 to 2025-06-30; frozen fits on past weather, no ensemble
metric: >-
  Balanced accuracy of the sign of the real-power forecast; ratio of fitted solar capacity to the
  signed fit's; mean absolute error of the folded-reading forecast in points of effective capacity
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-28
best_arm: >-
  L0 (fitted through the meter): flow-direction accuracy on sunny half-hours within 0.01 of the
  signed fit's at all 5 sites at both injection levels, scored cut
headline: >-
  After injecting solar output, the model fitted through the meter finds the flow direction and the
  solar capacity about as well as a model fitted to the signed power, but it invents reverse flow at
  one site with none, and it fails when the data carry reactive power.
take_home_message: >-
  A substation meter that reports only the size of the power flow, not its direction, still lets
  the model fitted through the meter recover the direction and the solar capacity about as well as
  a model that sees the signed power, as long as reactive power is zero. The model also invented
  reverse flow at the 1 of 5 substations that has none, so reverse flow it reports at a real
  substation needs an independent check. The pre-registered test could not be passed as written,
  because with the larger injection almost every sunny half-hour has reverse flow, so only one flow
  direction is present and balanced accuracy cannot be computed.
lessons:
  - >-
    Balanced accuracy of the flow direction is uninformative where one direction covers over 90% of
    the half-hours. Score sunny half-hours against the signed fit's accuracy instead.
  - >-
    The fit through the meter assumes zero reactive power and collapses to the naive answer when
    reactive power is present at power factor 0.97 and the midday flow is near zero.
advice:
  - >-
    Compare a fit through the meter with the signed fit on the same rows. Do not use balanced
    accuracy where reverse flow is heavy (figure 1).
  - >-
    Check a fit through the meter for invented reverse flow at a substation with none, before
    reading any reverse flow it reports (natural level, D5).
  - >-
    Expect reactive power at power factor 0.97 to break the fit through the meter wherever midday
    flow is near zero (figure 2).
---

# MVA fold: known-answer test at folded signed Primaries

**The model fitted through the meter recovers the flow direction and the solar capacity about as
well as a model that sees the signed power, while reactive power is zero.** The model also invented
reverse flow at the one Primary substation with none (D5), and it failed when the data carried
reactive power at power factor 0.97. Verdict: partly supported. The sequence stops here, because
the pre-registered pass rule fails as written.

We asked whether the direction of the power flow at a Primary substation can be recovered from a
meter that reports only apparent power (the size of the flow in MVA, with no direction). We took
five Primaries whose signed power is known and subtracted the output of a real solar farm at two
sizes: 1.0 times and 2.0 times the site's median absolute midday power. Those are injection levels
1.0 and 2.0, and "natural" means no injection. We then replaced each reading with its magnitude,
which
is what folding means here.

We fitted three models. L0 is the convex demand-plus-solar model fitted through the meter, as in
[experiment 110](../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md).
S is the same model fitted to the signed series, which is the ceiling. N is the same model fitted
to the magnitude as if it were signed power. Fits were frozen at 2025-07-01 (the scored cut) and
scored on the following year with past weather; the replicate cut is 2024-07-01. "Sunny
half-hours" are April to September, starting 10:00 to 13:30 UTC, with irradiance of at least
500 W/m².

![Figure 1: share of sunny half-hours with the flow direction right, by site and injection level](figure_1_sign_accuracy_sunny_half_hours.svg)

![Figure 2: fitted capacity through the meter divided by the signed fit's capacity](figure_2_capacity_through_the_meter.svg)

## Results (scored cut, zero reactive power in the data, point forecasts)

- **Direction, sunny half-hours.** L0's accuracy is within 0.01 of S's at every site and level
  except
  D5 at the natural level. At level 1.0 it is 0.82 to 0.94 against N's 0.26 to 0.70. At level 2.0 it
  is 0.94 to 0.99 against N's 0.01 to 0.06. The replicate cut agrees (L0 within 0.01 of S).
- **Pre-registered rule.** At level 1.0, 4 of 5 sites reach balanced accuracy 0.80 on sunny
  half-hours (D5 is 0.69, and S is 0.66 there too). At level 2.0, balanced accuracy is 0.5 at every
  site, because 92% to 99% of the sunny half-hours are reverse flow. The rule fails as written, and
  the metric cannot tell the arms apart there. Plain accuracy, post hoc, passes at 5 of 5 sites.
- **Capacity.** L0's fitted capacity is within 15% of S's at 5 of 5 sites at both levels. The rise
  from the natural fit is about 75% to 80% of the injected capacity for both L0 and S (D10: 0.80),
  so the capacity shortfall appears in the signed fit too and the folding does not cause it.
- **Reading error.** L0's mean absolute error on the magnitude is below N's at every level-1.0 and
  level-2.0 cell, by 1 to 15 points on sunny half-hours (for example D10 at level 2.0: 10.4 against
  25.3).
- **Placebo fails at D5.** With no injection, L0 gets the direction right on 35% of sunny
  half-hours at D5 (S and N: 98%). Its fitted capacity is 1.30 times S's. The six starting points of
  the fit disagree by 45 percentage points of capacity. The lowest-loss start has a solar term that
  is large and negative at midday, a mirror image that a loss on the magnitude cannot tell from the
  right answer.
- **Reactive power breaks it.** With power factor 0.97 in the data and `Q = 0` in the fit, L0
  collapses to N at D11, D14, and D16 at level 1.0 (accuracy 0.31 to 0.63, capacity 0.70 to 0.77 of
  S).
  The design brief's measurement puts the reactive floor at Stickney (D12) below 1% of capacity, so
  the zero-reactive case is the relevant one there.

## Checks

- **Triggers.** Audit trigger 1 (a failed control) fired: the D5 placebo. A fresh Sonnet auditor
  found no leakage, confirmed
  the D5 result is a mirror-basin weakness and not a bug, traced the Q1 collapse to the `Q = 0`
  assumption, and confirmed the sunny mask. It also flagged that the claim "L0 matches S" excludes
  D5
  at the natural level, which this page states.
- **Median against mean forecast.** Not applicable: these are point forecasts from frozen fits.
- **Departures from the plan.** Plain accuracy is added post hoc beside balanced accuracy. The audit
  used Sonnet
  because the series brief requires it, not Opus.

## Caveats and what to try next

- Frozen fits on past weather, no ensemble, no Kalman filter, no quantiles: sign recovery only.
- One farm (A) supplies the injected output, so the result is for one farm shape.
- At the natural level, D11, D14, and D16 have no reverse flow, so balanced accuracy cannot be
  computed.
- Step 2 should read L0's reverse flow only after a check that rejects the D5-type mirror basin
  (for example, require the lowest-loss start to agree with the start nearest zero reverse flow).
