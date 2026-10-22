---
id: '00207'
slug: dp-meter-regime-steps-e2-step-capacity-known-answer
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
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
  date within one month boundary (30 days), and a no-injection placebo year at each of those Primaries gives fewer than
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
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-22
best_arm: >-
  Step path with weight 3 (passes every test at 5 of 5 Primaries); the pre-registered weight 0.3
  recovers 0.98 to 1.00 of the step but fails the false-step test at D10 and D11
headline: >-
  At the 5 decision Primaries the step path recovers a median 0.98 to 1.00 of an injected step (weight
  0.3) and the date within one month in 100% of injections, against 0.87 to 0.91 for the smooth path and 0.24 to 0.49
  for season-year steps. The pre-registered gate fails (all three tests at 3 of 5 Primaries), because
  the no-injection fit shows 1.7 false steps above 0.5 MW per year at D10 and 1.0 at D11.
lessons:
  - >-
    Any capacity path with a free level for each month finds the month and most of the size of an
    added solar-farm step in summer rows; only the capacity that changes once a year misses it.
  - >-
    The limit is false steps, not missed steps: weight 0.3 leaves 1.7 and 1.0 false steps per year at
    D10 and D11, a weight of 3 brings every Primary below 0.7 while keeping 0.89 to 0.96 of the step,
    and a weight of 30 cuts the real step to 0.61 to 0.81.
advice:
  - >-
    Fit a step-capable capacity path with a total-variation weight near 3 (a share of the mean
    diagonal of the month Gram matrix) if a later experiment needs steps, because it passes the
    size, date, and false-step tests at all 5 decision Primaries (gate table).
  - >-
    Count false steps as the net change from one summer to the next as well as single month-to-month
    changes, because a fit with no winter rows spreads a real change over the winter months
    (Checks).
  - >-
    Do not use a capacity that changes once per season-year to look for a step, because it recovers
    24% to 49% of an injected step and misplaces the month in up to 60% of injections (Figure 1).
  - >-
    Do not expect to see a winter step: the fit uses April to August rows only, and a winter step
    was not tested (Notes).
take_home_message: >-
  A solar capacity that is free to change each month finds a known solar-farm step added to a
  Primary's power, to within 2% of its size at the pre-registered penalty, and places it within one
  month of the true date. At that penalty the fit also invents steps above 0.5 MW in a year without
  any, at 2 of 5 Primaries, and a stronger penalty that avoids them still finds 89% to 96% of the
  step. A capacity that changes once a year finds only 24% to 49% of the step.

---

# E2: can a step in solar capacity be seen? A known-answer test

**A solar capacity that can change each month finds a known step in solar output, in size and to
within a month, but at the pre-registered penalty it also invents steps at 2 of 5 Primaries (D10 and
D11), so the gate fails.** We asked whether the demand model
of `studies.dp_gam` (a regression that stays convex, so it has one best fit) can find a sudden step in
the solar capacity it infers for a Primary substation, which is a substation that feeds homes and
businesses. The pinned capacity is the real-time estimate of the Primary's solar capacity that
experiment 45 produced.
We added the recorded output of a real solar farm to a Primary's power, from a known month onward or
before it, at 30% or 60% of the Primary's pinned solar capacity, and checked how much of the step the
fit found. The fit used April to August rows before 2025-07-01 only. We repeated the
test for 20 injections at each of the 5 decision Primaries (5 farms, 2 sizes, on and off), and
compared three capacity paths on the same rows. A path is the capacity in each month. The step path
penalises the sum of the month-to-month changes, so it stays flat except where the data need a jump;
the weight 0.3 is the pre-registered strength of that penalty. At that weight the step path found a
median 0.98 to 1.00 of the injected step (the capacity change found, over the capacity change added), the smooth monthly path of experiment 157 found 0.87 to 0.91,
and experiment 129's capacity that changes once a season-year found 0.24 to 0.49. The result narrows the
search: a step as large as one farm's output is visible in summer data, so the open question is
whether the real series contain such steps (experiment 206 found no meter-failure signal).

**Verdict `partly supported`.** At the pre-registered weight 0.3 the step path passes the size test
(median in 0.75 to 1.25) at 5 of 5 Primaries. The step path passes the date test (at least 75% of
dates within one month boundary of the true date) at 5 of 5 Primaries. The false-step test (fewer
than 1 step above 0.5 MW per placebo year) passes at 3 of 5, so all three tests pass at 3 of 5
Primaries and the pre-registered gate, which needs 4 of 5, fails. D10 and D11 fail with 1.73 and 1.00
false steps per year. At weight 3, one of the pre-registered sensitivity weights, all three tests
pass at 5 of 5, but the weight was picked after seeing the results. The smooth path also passes the
size and date bands, so the claim that it blurs the step is not supported: it shrinks the size to 87%
to 91%.

![Figure 1: recovered step over injected step, by Primary and arm](figure_1.svg)

![Figure 2: false steps and recovered step against the penalty weight](figure_2.svg)

![Figure 3: fitted capacity change at D10 for one on-step and one off-step](figure_3.svg)

## Checks

- **Gate table (tests passed, out of 5 Primaries), size / date / false steps / all three:** weight
  0.03 is 5 / 5 / 2 / 2, weight 0.3 is 5 / 5 / 3 / 3, weight 3 is 5 / 5 / 5 / 5, weight 30 is 1 / 5 /
  5 / 1, and weight 300 is 0 / 2 / 5 / 0 (no step is allowed).
- **False steps per placebo year at weight 0.3 (no injection, mean of 5 row sets over 2022 to
  2024):** D5 0.00, D10 1.73, D11 1.00, D14 0.93, D16 0.60. At weight 3 they are 0.00, 0.60, 0.67,
  0.53, and 0.13. A false step is a month-to-month change above 0.5 MW between two summer months, or
  the net change from the August level to the next April level. The largest false step at weight
  0.3 is 1.79 MW. A step in the no-injection fit may be a real change in capacity, so the count is
  an upper bound.
- **Absolute recovery, median (95% interval over the 20 injections), weight 0.3:** D5 0.98 (0.97 to
  0.99), D10 0.99 (0.96 to 1.00), D11 1.00 (0.99 to 1.01), D14 1.00 (0.98 to 1.02), D16 1.00 (0.97 to
  1.01). Smooth path: 0.87, 0.88, 0.90, 0.91, 0.89. Season-year steps: 0.24, 0.34, 0.49, 0.43, 0.37.
- **Date within one month boundary (31 days):** 100% of injections for the step path (weights 0.3
  and 3) and the smooth path at every Primary and 40% to 90% for season-year steps. The exact month is
  found in 100% of injections for the step path at weight 0.3 and the smooth path, 95% at D11 for
  weight 3, and 15% to 45% for season-year steps. The date is read on month boundaries, so the test
  cannot tell a step placed a few days into a month from one at its start.
- **Spread over the injections:** the median ratio is 0.99 to 1.00 for each combination of size and
  direction, and 0.89 (farm B) to 1.02 (farm D) by farm.
- **Non-negativity of the level was not enforced:** the lowest level is -0.68 points (a point is 1%
  of the Primary's effective capacity) at weight 0.3, in 3 path values.
- **Audit triggers:** the first trigger of the `ml-experiment` skill fired, because the placebo
  control fails at D10 and D11 at the pre-registered weight (1.73 and 1.00 against a limit of 1), although the false
  steps are marginal and may be real steps. No forecast is scored, and no later experiment is built
  on this result.

## Notes

- **Arms:** the step path (weights 0.3, 0.03, 3, 30, and 300 per unit of step, as a share of the mean
  diagonal of the month Gram matrix; 30 and 300 were added after the first results), experiment
  157's smooth monthly path (growth share 0.1), and experiment 129's season-year steps (ridge 0.1).
  All use one Gram matrix, so the row set is the same.
- **Row definition:** half-hourly rows of April to August, before 2025-07-01, with a reading for the
  injected farm, about 26,000 rows per fit. No forecast is scored, so there is no lead range.
  Injected steps were 0.83 to 2.36 MW.
- **Step dates:** the first of May, June, or July of 2022 to 2024 (seeded). A step in winter is
  not tested, because winter rows are not in the fit.
- **D9** is left out: its readings start in 2025, so no 2022 to 2024 step can be injected.
- **Disclosed differences from the plan:** the bounded solver keeps the smooth path's levels at or
  above zero but not the step path's (the lowest step level is -0.68 points at weight 0.3, in 3 path
  values). The date tolerance is 31 days, so a step one month boundary away passes. The first draft
  counted only single month-to-month changes as false steps; a reviewer pointed out that the fit has
  no winter rows, so the count now also includes the net change from one summer to the next.
- **Injection:** the farm's output is scaled by its 99th percentile and subtracted from the target
  from the step month on (on-step) or before it (off-step). Recovery is the change in the fitted
  level across the step, injected fit minus the fit without injection, over the injected capacity.

## What the next experiment should answer

Experiment 206 found no meter-failure signal at the Primaries, so experiments E3 and E4 of the brief
are not run. A later experiment that needs steps would fit the step path with weight near 3 and read
candidate steps as "inferred, unconfirmed". The recovery here is a ceiling: a real meter-regime
step has its own shape and may sit in winter, where this fit cannot see it.
