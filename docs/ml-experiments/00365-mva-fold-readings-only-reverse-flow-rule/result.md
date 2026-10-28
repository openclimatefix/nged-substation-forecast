---
id: "00365"
slug: mva-fold-readings-only-reverse-flow-rule
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
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
---

# MVA fold: reverse-flow rule on the meter readings alone (planned)

[Experiment 362](../00362-mva-fold-gated-fit-known-answer/result.md) found that an independent solar
forecast cannot gate invented reverse flow. This experiment tests a rule that uses only the meter
readings in the training window.

## Plan

- **Rule.** Take the training half-hours from April to September, starting 10:00 to 13:30 UTC, with
  irradiance of at least 500 W/m². If the 1st percentile of the readings on those half-hours is
  below 5 percentage points of capacity, use the fit through the meter. Otherwise use the naive fit.
  Both numbers are fixed here, before any statistic is computed.
- **Scoring.** No new model fit: each cell's chosen arm is read from the stored results of
  experiments 358 and 362 (same sites, injection cases, cuts, and past weather), so the score of the
  rule is the score of the arm it chooses.
- **Census (not scored).** The statistic and the chosen arm at D2, D3, D4, D8, D12, D19, D21, and
  D22 at the 2025 cut, to preview step 3. NGED's monitor file marks reverse flow "Yes" at D8, "Maybe"
  at D12 and D22, and "No" at the other five.
- **Limits.** The statistic is computed on readings the model also trains on, so a single
  glitch reading could trigger it; the percentile guards against one reading and not against a
  stuck meter.
