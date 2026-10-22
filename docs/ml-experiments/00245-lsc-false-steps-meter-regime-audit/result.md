---
id: '00245'
slug: lsc-false-steps-meter-regime-audit
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
size: >-
  medium (descriptive; reads stored outputs only). Stored: private tables under
  data/ml-experiments/00245-lsc-false-steps-meter-regime-audit/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design, because the false steps
  are dated only to a month, so the window and the matching rule are free choices; both are fixed
  below before any score. Callers: this folder's scripts only, which copy the load-shift tests of
  experiments 130 and the alarm tables of experiments 130, 156, and 195. Asset graph: not touched.
  Plan reviews skipped (no fit, no new method); one code review, one write-up review, one prose
  review.
hypothesis: >-
  The false steps that failed experiment 207's gate at D10 and D11 are partly real load moved
  between connected Primaries: they coincide with a balanced, opposite step at a metered neighbour
  or with an alarm of the load-shift detectors, more often than steps drawn at random at the same
  Primary.
prediction: >-
  Written before any score. At least half of the localised false steps at D10 and D11 have a
  balanced opposite neighbour step or an alarm (the explanations defined below), at a rate at
  least twice the matched base rate. The prediction fails if the false steps look like ordinary
  noise or genuine meter changes; the page then says the two failure modes are separate.
decision_rule: >-
  Supported when the share of localised false steps at D10 and D11 with an explanation is at least
  0.5 and its matched-sample 95% interval for the ratio to the base rate excludes values below 2.
  Not supported otherwise. A positive outcome is a candidate cause only, because a month-resolution
  capacity step cannot show that a neighbour caused it. No verdict on experiment 207 is given.
benchmark: >-
  Matched base rate: windows at the same Primary, in the same calendar month, whose largest
  residual step is within 30% of the false step's, with no false step within 31 days.
row_set_id: >-
  Not a forecast. Experiment 207's step path at weight 0.3 (base fit, no injection) for D5, D10,
  D11, D14, and D16, summer rows 2022 to 2024, against the arm B residual panel of experiment 130
  (2022-01-01 to 2026-07-01).
metric: >-
  Share of false steps with an explanation: a balanced opposite step (0.4 to 2.5 times the size) of
  at least 2 noise units at a metered neighbour within 6 hours of the Primary's own largest
  residual step in the month window, or an alarm of RO-NC, the pair alarm, or the learned score of
  experiment 195 at the Primary within 24 hours of that step.
code: ml-experiments/00245-lsc-false-steps-meter-regime-audit/
---

# Are the false steps in the meter-regime test load shifts between Primaries?

Planned, not run. The design below is fixed before any score.

## Design

- **False steps:** exactly experiment 207's definition (step path, weight 0.3, no injection, 5 row
  sets, 2022 to 2024): a month-to-month change above 0.5 MW between two summer months, or the net
  change from the August level to the next April level. A step between two summer months is
  *localised* to the month window of 30 days centred on the first day of the later month. The
  August-to-April net change is dated to the winter only, so it is listed but not scored.
  The same step in several row sets counts once.
- **Own step:** the largest absolute one-day level shift (one day after minus one day before,
  experiment 130's arm B residual, noise units from the odd-month median absolute deviation) at the
  Primary inside the window. It needs 3 noise units or more, else the false step has no visible
  residual step.
- **Explanations:** (1) a metered neighbour of the Primary (experiment 130's adjacency) with a
  step of the opposite sign of at least 2 noise units within 6 hours of the own step, whose size is
  0.4 to 2.5 times the own step; (2) an RO-NC alarm (experiment 130), a pair alarm, or an alarm of
  the logistic learned score in any fold (experiment 195) at the Primary within 24 hours of the own
  step. Also tagged, not scored: flagged or zero readings within a day, a bank holiday within a
  day.
- **Base rate:** for each false step, up to 200 windows at the same Primary, in the same calendar
  month of the other years and at other days of that month, with the own step within 30% of the
  false step's size and no false step within 31 days, scored by the same rules.
- **Gate recount:** experiment 207's false-step test with the explained false steps removed. This
  recounts the stored paths and does not refit; a refit with the alarm intervals masked is not run.
- **Wide-window variant (secondary):** the same explanations anywhere in the 30-day window.

## What the next experiment should answer

Written after the results.
