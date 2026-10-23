---
id: "00267"
slug: pv-capacity-early-years-and-register-floor
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00267-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one defensible way to read the
  hypothesis, and the plan fixes the choice below. Callers: this folder's script, which copies the
  fitting calls of experiments 129 and 238 and edits no reviewed module. Asset graph: not touched.
hypothesis: >-
  The inferred capacity of unmetered solar at the five decision Primaries falls below the register
  floor in season-year 2021 (July 2021 to June 2022) because the capacity fit recovers only about
  0.78 of an injected farm and the 2021 fit window is thin, not because the register is shared to
  the wrong district. Dividing by the median recovery of the metered farms, and using the output of
  a metered farm within 25 km as the sunshine shape, lifts the estimates to the floor.
prediction: >-
  Written before any fit. Dividing by the median recovery of the other metered farms lifts D10 in
  season-year 2021 from 1.1 to 1.3 to 1.5 MW and D11 from 1.9 to 2.3 to 2.7 MW, and both stay below
  their floors. A metered farm within 25 km as the shape changes D5, D14, and D16 by under 10%. The
  2021 season-year rests on 2 to 3 months of data. At least 3 of 5 Primaries have every season-year
  at or above the floor after the division. Reading of the plan before any fit: the card's estimate
  already divides by the Primary's median matched-farm recovery (the `divisor` column of experiment
  129's real-series table, 0.77 to 0.86), and every Primary has 12 months of rows in season-year
  2021, so the division arm and the thin-window rule are expected to change nothing.
decision_rule: >-
  An arm succeeds if, at all 5 Primaries, the upper end of the 80% range of every season-year with
  register coverage is at or above the register floor (the Feed-in Tariff register's 1 MW cut,
  compared in AC MW, with the DC floor shown beside it), and the estimate stays within 10% of the
  card's estimate (A0) at the Primaries with no shortfall. Passing means "consistent with" the
  register, never "confirmed". An arm is drawn only if experiment 241's held-out injections land
  within 10% of the truth at 4 of 5 Primaries under that arm's shape. The nearby-farm shape is kept
  for the cards only if it also lifts the upper end of D10 and D11 to the floor, because the
  reference needs a metered farm near every unmetered Primary, which costs a data dependency.
benchmark: >-
  The card's estimate: experiment 129, arm 4 at increment penalty 0.1, with its stored 80% range
row_set_id: >-
  Experiment 129's training rows before 2026-07-01 at D5, D10, D11, D14, and D16; no forecast is
  scored
metric: >-
  Capacity of unmetered solar in MW of alternating-current output, per season-year, with an 80%
  range, against the register floor in MW
production_status: not-for-production
adopted_in: none
code: ml-experiments/00267-pv-capacity-early-years-and-register-floor/
---

# PV capacity: why the 2021 estimate falls below the register floor

Status: planned. Arms: A0 the card's estimate; A1 the leave-one-farm-out division of experiment 241
(expected to equal A0); A2 the same fit with the output of the nearest metered farm within 25 km as
the sunshine shape, divided by experiment 238's recovery under that shape; A3 A2 with the number of
fitted months printed and a "thin" mark under 6 months. The plan section of the finished page
records the departures from the brief.
