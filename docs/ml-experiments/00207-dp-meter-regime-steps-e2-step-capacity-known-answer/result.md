---
id: '00207'
slug: dp-meter-regime-steps-e2-step-capacity-known-answer
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private tables under
  data/ml-experiments/00207-dp-meter-regime-steps-e2-step-capacity-known-answer/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one defensible design for the step
  path, but the brief fixes the penalty form and the injection test. Callers: this folder's scripts,
  which copy experiment 157's month-design code; no change to `studies.dp_gam` in this experiment.
  Asset graph: not touched.
hypothesis: >-
  A convex solar-capacity path that allows a few sudden jumps (a total-variation penalty on the
  month-to-month change) finds both the size and the date of a known solar-farm step added to a
  Primary's power, where a smooth path blurs the step and a path that steps once a year places it
  within the wrong month.
prediction: >-
  Written before any fit. On summer months (April to August) the step path recovers the size of an
  injected step with a median ratio of 0.85 to 1.0 per Primary (the summer recovery of experiment
  157 was 0.84 to 0.96 and the shape ceiling is 0.92 to 0.94), and places the step at the right
  month boundary or one boundary away in at least 75% of injections. The size gate may pass at only
  3 of the 5 decision Primaries, so the gate is expected to be close.
decision_rule: >-
  Gate (from the brief): the step path recovers the step size, as a ratio of recovered to injected
  step, in 0.75 to 1.25 at 4 of the 5 decision Primaries (D5, D10, D11, D14, D16), and places the
  date within 30 days, and a no-injection placebo year at each of those Primaries gives fewer than
  1 false step above 0.5 MW. A Primary passes the size test when the median ratio over its
  injections is in the band, and the date test when at least 75% of its injections are placed within
  30 days. If the gate fails, stop and write up why a step cannot be seen in this data, and
  experiments E3 and E4 of the brief are not run.
benchmark: >-
  Two other arms on the same rows and the same injections: experiment 157's smooth capacity path
  (squared change penalty, growth share 0.1) and experiment 129's capacity that steps once per
  season-year (ridge 0.1).
row_set_id: >-
  Experiment 157's training rows (the feature cache of experiment 106) before the origin
  2025-07-01, restricted to the months April to August, at D5, D9, D10, D11, D14, and D16. No
  forecast is scored.
metric: >-
  Step recovery: for an injection at the start of a month in May, June, or July of 2022 to 2024,
  the mean fitted level over the months from the step to the end of that August minus the mean over
  the months from that April to the step, injected fit minus the base fit, divided by the injected
  step (the injection is 30% or 60% of the Primary's pinned capacity). Date error: the distance in
  days from the month boundary of the largest fitted change to the true step date. Capacity unit:
  the 99th-percentile AC output of the farm, in MW.
code: ml-experiments/00207-dp-meter-regime-steps-e2-step-capacity-known-answer/
---

# E2: can a step in solar capacity be seen? A known-answer test

*The plan is committed before any fit. A result section replaces this one when the experiment
finishes.*

**Method.** The convex demand model of `studies.dp_gam` is fitted on the summer rows of each
Primary. Its solar capacity is a level for each calendar month, written as a starting level plus
non-negative up-steps and down-steps at each month boundary. A linear penalty on the sum of the
steps (the total variation of the path) lets the path jump where the data need it and stay flat
elsewhere. The penalty weight is a share of the mean diagonal of the month Gram matrix: 0.3 (the
pre-registered value), with 0.03 and 3 as sensitivity checks. Non-negativity of the level is
checked after the fit, not enforced; a Primary where the path goes negative is reported.

**Injection.** Real farm output (farms A, B, C, D, and F, which have data from 2019-09), scaled to
30% or 60% of the Primary's pinned capacity at the origin, is subtracted from the Primary's power
from a step date onward (on-step) or before it (off-step). The step date is the first of a month
drawn at random from May, June, and July of 2022, 2023, and 2024 (seeded). The fit is repeated for
each farm, share, and direction: 20 injections per Primary, each fitted from the same Gram matrix.

**Placebo.** The no-injection fit of each decision Primary is read for steps above 0.5 MW in the
summers of 2022 to 2024; any step found counts as false, which is conservative because the real
series may hold a true step.

**Notes.** Steps are placed in summer months, where the capacity is identified; a step in winter
could be seen only as a difference between two summers. Reported: the recovered ratio per arm and
Primary with a bootstrap interval over injections, the date error, the placebo count, and the
farm-to-farm spread. Absolute recovery comes beside every contrast.

**Sizing (five triggers).** Stores private tables only; not on the serving path; no degradation
rule; more than one defensible penalty form but the brief fixes it; callers are this folder's
scripts plus a copy of experiment 157's month design. Medium. Reviews: one Sonnet code review, one
Sonnet write-up review, one Sonnet prose review; no Opus unless the manager asks.
