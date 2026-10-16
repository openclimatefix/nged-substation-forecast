---
id: "00041"
slug: sw2-0a-labeller-gate
status: planned
size: complex
hypothesis: >-
  A labeller that reads features computed from raw power and the switching logs, and that uses no
  forecasting model as its deciding evidence, can find injected load transfers and reject placebo
  windows well enough to label the real logged closures of the 16 Primary substation demand series.
prediction: >-
  On 300 fresh known-answer cards scored once, the rules-only labeller calls 5% or fewer placebo
  cards visible and reaches a recall of 0.60 to 0.85 on injected transfers of 25% or more of the
  receiving load lasting 6 hours or more. Sonnet reaches 0.80 or more on the same cards and its
  event-level F1 is at least the rules' F1.
decision_rule: >-
  `supported` when every row of the gate table below is met by the labeller we would use (Sonnet if
  its event-level F1 is at least the rules-only labeller's, otherwise the rules). `partly
  supported` when the placebo, onset, end, and magnitude rows are met and recall is under 0.80, in
  which case the bands where recall is under 0.80 become don't-care bands in later scoring.
  `refuted` when the placebo rate is above 10% or recall of the large, long injections is under
  0.50. `inconclusive` otherwise. The gate is scored once on the test cards; the 100 development
  cards are the only place prompts and rules are tuned.
benchmark: >-
  A rules-only labeller on the same features, and the placebo cards.
row_set_id: none (this experiment scores labels against injected known answers, not forecasts)
metric: >-
  Gate table of the SW2-0a plan: placebo false-visible rate, injected recall by size and duration
  band, onset and end error, median relative error of the moved MW, receivers named, jittered-log
  check, Cohen's kappa between the two Sonnet labellers, and event-level F1 against the rules.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00041-sw2-0a-labeller-gate/
---

# Experiment 41: can a labeller pass a known-answer gate on switching events?

**Planned.** The design is in the SW2-0a section of the
[research plan](../research-plan.md#sw2-0a-an-agent-labeller-gated-on-known-answers). This page
becomes the write-up when the gate has been scored.

## Size

- **Stored:** a hindsight model's quantile residuals, a features table per logged operation, cards,
  labels, and injected and placebo test cards, all under `data/ml-experiments/00041-*/`. Nothing in
  `src/`.
- **Serving path:** not touched. The logs exist for the trial area only, so no live code uses them.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one (rules only, Sonnet, Sonnet with a second labeller).
- **Callers:** none outside `ml-experiments/`.

## Pre-registered gate

| Check | Bar |
|---|---|
| Placebo labelled visible | 5% or less (conserved 2% or less) |
| Injected recall, share 25% or more and 6 hours or more | 0.80 or more |
| Onset error, detected injections | 1 hour or less in 90% or more |
| End error | the larger of 2 hours and 10% of duration, in 80% or more |
| Moved MW | median absolute relative error 25% or less |
| Receivers named, share 30% or more | 80% or more |
| Jittered log time (1 to 3 hours off) | onset nearer the truth than the log in 80% or more |
| Kappa between the two Sonnet labellers | 0.7 or more |
| Against the rules-only labeller | Sonnet's event-F1 at least the rules' |

## Stages reported

1. The hindsight model and its two absorption checks.
2. The rules-only labeller against the gate.
3. The Sonnet labeller against the gate, with the real token cost of SW2-0a.
