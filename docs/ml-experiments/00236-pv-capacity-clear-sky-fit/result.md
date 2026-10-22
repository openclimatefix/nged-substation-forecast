---
id: "00236"
slug: pv-capacity-clear-sky-fit
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00236-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible design, because the brief fixes
  the row rule. Callers: this folder's scripts, which copy experiment 231 and call the reviewed
  `studies.dp_*` modules without editing them. Asset graph: not touched.
hypothesis: >-
  Fitting the capacity of the unmetered solar behind a Primary substation only on half-hours with a
  clear sky and a high sun, where the irradiance error is smallest, finds more of an injected real
  solar farm than the fit of experiment 129 on all rows.
prediction: >-
  Written before any fit. Recovery rises by at least 0.05 at 4 of the 5 decision Primaries, and the
  80% interval from resampling months is no more than 50% wider than the interval of the fit on all
  rows. Experiment 231 lowers this prior: fitting the farm alone on clear half-hours found no more
  than on all daylight half-hours (0.929 against 0.931), so the expected rise is smaller than
  0.05.
decision_rule: >-
  The row rule is worth adopting if the recovery rises by at least 0.05 at 4 of 5 Primaries and the
  interval is no more than 50% wider. A rise at fewer than 4 Primaries, or a wider interval, means
  the rule is dropped. The coverage of the injected value (23 of 30 cells in experiment 129) is not
  re-run, because experiment 129's coverage test uses a semi-synthetic target and code that this
  experiment does not copy; the interval-width condition stands in for it. No cost of porting is
  weighed, because the experiment scores no forecast and changes no production code.
benchmark: >-
  Experiment 231's rung 4 (experiment 129's arm 2 on all rows) on the same injections: pooled
  recovery 0.827 (80% interval 0.814 to 0.842) at season-year 2024
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before 2026-07-01, at D5,
  D10, D11, D14, and D16, with D9 beside them, restricted to clear half-hours. No forecast is
  scored.
metric: >-
  Recovery: the capacity found with a real farm's output added at 50% of the Primary's pinned
  capacity, minus the capacity found without it, divided by the injected capacity, read at
  season-year 2024 (July 2024 to June 2025). Capacity unit: the 99th percentile of alternating-
  current output over daylight half-hours, in MW
code: ml-experiments/00236-pv-capacity-clear-sky-fit/
---

# PV capacity: does fitting only on clear, high-sun half-hours help?

**Planned, not yet run.** The experiment repeats the injection test of experiment 231: the recorded
output of a real solar farm (farms A to F stand for metered generators, and are never named) is added
to a Primary substation's power at 50% of the Primary's pinned capacity, and the convex capacity fit
of experiment 129 reports how much of the farm it finds again. The only change is the row set. Both
the base fit and the injected fit train only on half-hours where the Primary's CAMS global
irradiance is at least 0.9 of its clear-sky value, the clear-sky value is at least 200 W/m2, and the
sun is at least 20 degrees above the horizon. Fits stop at 2026-07-01, so the maintainer's labelled
exception (data to 2026-07-31) is not used. No forecast is scored.

## Design

- **Rows.** The row rule is applied through month weights and a row mask on the Gram matrices, as
  in experiment 231, so that the base and injected fits share one row set.
- **Demand terms.** The demand columns of the fit are unchanged. On clear days the solar shape is
  closer to a smooth seasonal curve than on all days, so the demand harmonics may absorb more of it;
  the experiment reports that as the contrast between the farm-only scale on clear rows and the
  full fit on clear rows.
- **Reported rows.** The recovery on clear rows, the recovery on all rows (rung 4 of experiment
  231), the number of training rows kept, and the share of each season-year kept.
- **Intervals.** A month-block bootstrap, 200 resamples, stratified as in experiment 231, and the
  range over the five farms.

## Plan checks

- **Reproduction:** with the row rule switched off the script must reproduce experiment 231's rung 4
  at every Primary and farm to within 1e-6.
