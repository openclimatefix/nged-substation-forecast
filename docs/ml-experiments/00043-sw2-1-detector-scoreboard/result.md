---
id: "00043"
slug: sw2-1-detector-scoreboard
status: planned
size: complex
hypothesis: >-
  Experiment 37's neighbour-balance detector D1 finds injected load transfers between Primary
  substations more often than its single-series detector D0, when a transfer counts as found only
  if both its start and its end are placed correctly at the time a live system could know them.
prediction: >-
  At the 24-hour window and the operating point fixed in experiment 37, D1's event F1 on injected
  transfers of 25% or more of the receiving load lasting over 1 day is 0.30 to 0.60 and D0's is
  0.15 to 0.45, with D1 above D0 by at least 0.10. Most misses are wrong ends, not missed starts.
decision_rule: >-
  The headline is event F1 on the 25%-or-more, over-1-day band of injected transfers, at the 24-hour
  window. `supported` when D1 minus D0 is at least +0.10 with a 95% interval above 0 (resampling
  whole calendar months). `refuted` when the difference interval lies below +0.05. `inconclusive`
  otherwise. Real labelled events are counted and reported without a verdict, because only 4 meet
  the size and length condition.
benchmark: >-
  The single-series detector D0 at the same window, and a detector that raises no alarm.
row_set_id: none (this experiment scores detector alarms against known events, not forecasts)
metric: >-
  Event F1 (an event is found when its onset is within 2 hours and its end within the larger of 6
  hours and 10% of its duration, both at the detector's causal known-from time), with recall,
  precision, and the end error beside it.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00043-sw2-1-detector-scoreboard/
---

# Experiment 43: how well do the two existing switching detectors find transfers?

**Planned.** This page becomes the write-up once the scoreboard has run. The plan is the SW2-1
section of the [research plan](../research-plan.md#sw2-1-a-fixed-scoreboard-and-an-injection-harness)
with the changes below, and the planning session reviews it.

## Size

- **Stored:** alarm and interval tables, injected-event tables, and the scoreboard under
  `data/ml-experiments/00043-*/`. Nothing in `src/`.
- **Serving path:** not touched. The detectors read ERA5 weather, so they run retrospectively only.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one (event matching rule, operating points, scoring bands).
- **Callers:** none outside `ml-experiments/`. The injection harness and the scoring module are
  written to be reused by SW2-4 to SW2-9.

## Design

- **Detectors and residual.** D0 (one series) and D1 (neighbour balance) at the 4-hour and 24-hour
  windows, copied from experiment 37 with the interval builder of experiment 38 (an alarm becomes an
  onset, an end, and a `known_from` time, the first stamp at which a live system could know the end).
  The residual is experiment 17's `H0.5` residual for the 16 Primaries, from 2022-01 to 2026-06.
- **Operating points are fixed, not re-tuned.** The four thresholds that drew SW2-0b's alarm sample
  (D0 8-half-hour 6.0, D1 8-half-hour 3.75, D0 48-half-hour 4.0, D1 48-half-hour 2.5) are the scoring
  thresholds, because the labelled alarm precision applies only at those thresholds. This replaces
  the plan's odd-month tuning, which experiment 37 already did.
- **Matching rule.** An event is found when the detector's onset is within 2 hours of the true onset
  and its end within the larger of 6 hours and 10% of the duration, each at its `known_from` time.
  The scoreboard also reports onset-only recall, so a wrong end shows as its own loss.
- **Injection harness** (new, reusable module `e00043_inject.py`). 600 transfers from the SW2-0a
  generator (a real operation's duration and sheet sides, a moved share that drifts between 0.5 and 2
  times its start, 1 to 3 receivers, 5% to 40% of the receiving load), added to the `H0.5` residual
  in batches so that no two events on a shared series lie within 7 days. Stratified over size bands
  (under 10%, 10% to 25%, 25% or more) and duration bands (2 to 6 hours, 6 to 24 hours, over 1 day).
- **Real events.** The primary set is the labelled transfers of 25% or more of the receiving load
  lasting over 1 day: 4 events, so the scoreboard reports counts and no interval. A second set, added
  after seeing the count and labelled post hoc, is the 35 visible operations lasting over 1 day,
  scored on recall only, with the 13 conserved operations whose side is mostly missing reclassified
  as one-sided and D1 reported on conserved and one-sided events separately.
- **Precision.** The bracket on the 148 unexplained alarms of SW2-0b: lower 9 over 148, upper (logged
  plus labelled visible) over alarms, by detector setting, minus the 0 of 40 quiet-window rate.
- **Headline and intervals.** Event F1 with recall, precision, and median end error beside it, by
  detector and window and by band, with 95% intervals from resampling whole calendar months.
- **Compute.** CPU only, 4 cores (slot C). Detectors on the residual take minutes; the 600
  injections run in batches; the full run is expected to take about an hour.

## Reviews

A Sonnet reviewer reads the plan and the code (leakage: no switching log, no hindsight window, and
no labelled onset reaches a detector), and a Sonnet reviewer checks the write-up against the printed
numbers. Opus is not used.
