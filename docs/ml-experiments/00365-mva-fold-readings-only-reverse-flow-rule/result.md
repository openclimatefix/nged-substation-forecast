---
id: "00365"
slug: mva-fold-readings-only-reverse-flow-rule
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
kind: other
size: >-
  simple to medium. Stored: small result files in the private folder only. Production serving path:
  not touched. Degradation rule: not touched. Designs: more than one rule on the readings would test
  the hypothesis (a percentile of midday readings, a count of near-zero readings, or a model-based
  test); the percentile is chosen as the simplest. Callers: reads the stored results of
  experiments 358 and 362 and the feature cache, so no new model fit is needed.
hypothesis: >-
  Reverse flow can be told apart from invented reverse flow using the meter readings alone: if the
  training readings on sunny midday half-hours never come close to zero, the substation has no
  reverse flow, and the naive fit (reading the magnitude as signed power) is right. If they do come
  close to zero, the fit through the meter is used.
prediction: >-
  At the scored cut with reactive power zero, the rule picks the naive fit at D5 with no injection
  (direction accuracy within 0.02 of the signed fit's 0.98), picks the fit through the meter at
  every injected case (levels 1.0 and 2.0), and so matches the signed fit within 0.02 at 4 or more
  of 5 substations in every case. At the 8 MVA-only Primaries (a census, not scored), it picks the
  fit through the meter at Stickney (D12) and picks the naive fit at the five NGED marks "No".
decision_rule: >-
  Passes (and steps 2 to 4 resume) when, at the scored cut with reactive power zero, the rule's
  chosen arm is within 0.02 of the signed fit's sunny-half-hour accuracy at 4 of 5 substations in
  each of the three injection cases (none, 1.0, 2.0), and the same holds at the 2024 cut. Fails
  otherwise, and the sequence stops. No cost-based threshold, because the experiment ports nothing.
benchmark: >-
  The signed fit, the ungated fit through the meter, and the naive fit, all from experiments 358
  and 362
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16 (scored); the 8 MVA-only Primaries (census only); cuts
  2024-07-01 and 2025-07-01
metric: >-
  Share of sunny half-hours with the flow direction right, for the arm the rule chooses; the rule's
  statistic (the lowest 1% of training sunny-midday readings, in percentage points of capacity)
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-28
best_arm: >-
  none: the rule meets its registered pass test, but it still gets D5 with no injection wrong and
  it chooses the naive fit at all 8 MVA-only Primaries
headline: >-
  A rule on the meter readings alone passes its registered test, but it cannot tell reverse flow
  from none: it picks the wrong fit at D5 with no injection, and at Stickney it picks the naive fit.
take_home_message: >-
  Looking for midday meter readings near zero does not reliably tell whether a substation has
  reverse flow. It still picks the wrong fit at the substation with none (D5), because that
  substation's readings also dip towards zero. It also misses heavy reverse flow, because heavy
  reverse flow gives large readings, not small ones: the rule picks the naive fit at Stickney (D12),
  the one MVA-only Primary expected to have it. The registered pass test is met only because it
  allows one of five substations to be wrong. A rule that uses how the readings move with sunshine
  is the next idea to test.
lessons:
  - >-
    Heavy reverse flow produces large meter readings, so a rule that looks for readings near zero
    cannot detect it.
  - >-
    A pass rule that allows 1 of 5 substations to be wrong can pass while the placebo substation
    fails, so state the placebo as its own criterion.
advice:
  - >-
    Do not use the lowest sunny-midday readings to decide whether reverse flow exists (figure 1).
  - >-
    Make the substation with no reverse flow its own pass criterion, not one of 5 in a count.
---

# MVA fold: reverse-flow rule on the meter readings alone

**A rule that looks for sunny-midday meter readings near zero does not reliably separate reverse
flow from none.** It still picks the wrong fit at D5, the one of the five test substations with no
reverse flow, because D5's readings also dip towards zero. It also misses heavy reverse flow:
heavy reverse flow gives large readings, not small ones, so the rule picks the naive fit at all 8
MVA-only Primaries, including Stickney (D12). Verdict: partly supported, because the registered
pass test is met, but only because it allows one substation in five to be wrong. Steps 2 to 4 of
the sequence are paused for a decision.

[Experiment 362](../00362-mva-fold-gated-fit-known-answer/result.md) found that an independent
solar forecast cannot tell invented reverse flow from real. This experiment tested a rule that
uses only the meter readings in the training years. The rule takes the training half-hours from
April to September that start 10:00 to 13:30 UTC with irradiance of at least 500 W/m². If the 1st
percentile of the meter readings on those half-hours is below 5 percentage points of capacity, the
rule uses the fit through the meter (L0). Otherwise it uses the naive fit (N), which reads the
magnitude as signed power. No model was refitted: the score of each case is the stored score of the
fit the rule chooses, from [experiment 358](../00358-mva-fold-known-answer-at-folded-mw-primaries/result.md),
which also defines the injection levels and the signed fit (S) used for comparison.

![Figure 1: lowest sunny-midday readings in training, by substation](figure_1_rule_statistic_by_substation.svg)

## Results (reactive power zero)

- **Registered pass test.** At the 2025 cut, the chosen fit's direction accuracy on sunny
  half-hours is within 0.02 of the signed fit's at 4 of 5 substations with no injection, and at 5 of
  5 at levels 1.0 and 2.0. At the 2024 cut the counts are 4, 4, and 5. The test passes.
- **D5 with no injection still fails.** The statistic is 3.79 at the 2025 cut and 2.88 at the 2024
  cut, both under 5, so the rule chooses L0. L0's accuracy is 0.35 against the signed fit's 0.98.
- **Census, not scored.** The statistic is 26.7 at Stickney (D12), 7.6 at Langrick (D22), and 5.9
  at Marsh Lane (D8), and 22 to 40 at the other five. All 8 exceed 5, so the rule chooses the naive
  fit everywhere, including at D12, where earlier experiments found reverse flow on 97% of summer
  midday half-hours. The prediction that the rule would pick L0 at D12 failed.
- **Power factor 0.97.** With reactive power in the data, the rule still chooses L0, which collapses
  at levels 1.0 and 2.0 (0 of 5 substations within 0.02), as in experiment 358.

## Checks

- **Triggers.** Audit trigger 1 (a failed control: D5) and trigger 2 (a result contradicting its
  prediction: the census) fired. The explanation is stated above and follows from the readings, so
  no separate audit was run: the experiment refits nothing and reads two stored result sets.
- **Median against mean forecast.** Not applicable: no forecasts were made.

## Caveats and what to try next

- The 1st percentile and the threshold of 5 were fixed before any statistic was computed.
- The census statistic uses the metered reading at the MVA-only Primaries, so a reading changed by
  generation added to the magnitude (the Marsh Lane change in April 2025) would also move it.
- Next idea: test whether the sunny-midday reading rises or falls with irradiance. The design
  brief measured a positive correlation of +0.75 to +0.83 at Stickney and a negative one at the
  other MVA-only Primaries. Not tested here.
