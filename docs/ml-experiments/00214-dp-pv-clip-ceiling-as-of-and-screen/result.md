---
id: '00214'
slug: dp-pv-clip-ceiling-as-of-and-screen
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00214-dp-pv-clip-ceiling-as-of-and-screen/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  design for the as-of rule, but the brief fixes the 12-month window and the 50% to 100% grid.
  Callers: this folder's scripts copy experiment 180's fit, forecast, and scoring code, and call
  `studies.dp_gam` unchanged. Asset graph: not touched.
hypothesis: >-
  The export-limit signature of experiment 180 is a stable property of D5, not a fit to the scored
  year. A ceiling chosen at each monthly refit only from the 12 months of forecasts before that
  refit keeps most of experiment 180's gain at D5, and the same signature is absent at the other
  Primaries, with D9 the most likely exception.
prediction: >-
  Written before any fit. (a) The as-of ceiling at D5 lies in 50% to 70% at 10 or more of the 12
  monthly cuts of the scored year, and the as-of arm keeps at least 0.08 MW of experiment 180's
  0.133 MW gain on D5 sunny daylight (control minus as-of arm). (b) At least one other Primary
  gains 0.03 MW or more with a clip, and the rest move by under 0.01 MW. Part (b) is the
  uncertain part: experiment 180's stage 0 found the implied solar output per unit shape falls
  towards midday only at D5 (D10, D11, D14, and D16 are flat or rising, and D9's is negative), so
  the author expects (b) to fail and the clip to be a D5 property.
decision_rule: >-
  Planned contrasts, each on sunny daylight half-hours, all leads to 168 hours, mean absolute error
  of the median forecast (q50) in MW, as-of arm minus control, month-block bootstrap on paired
  differences at the 98.33% level (three planned contrasts: D5 as-of, the best other Primary's
  as-of arm, and the rest pooled). `supported`: (a) and (b) both hold, with the D5 interval
  wholly below zero, the other Primary's interval wholly below zero, and the placebos D10 and D16
  moving by under 0.02 MW. `partly supported`: (a) holds and no other Primary gains 0.03 MW, which
  is reported as "the clip is a D5 property". `not supported`: the as-of ceiling at D5 leaves
  50% to 70% at 3 or more cuts, or the as-of arm keeps under 0.08 MW. `inconclusive`: the D5
  interval includes zero. The threshold is 0.08 MW because an as-of clip adds one per-substation
  selection step to each refit, so it needs a gain a quarter of the way to experiment 180's
  target of closing the gap to BG2.
benchmark: >-
  The control (the fixed fleet shape, experiment 180's arm c) and experiment 180's fixed-60% arm
  (arm k60), on `fold_a_daily_gap_v1`
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of q50 in MW and pinball loss averaged over 11 quantiles in MW, on sunny
  daylight half-hours (the top third of a substation's daylight half-hours ranked by the fixed
  fleet shape, as in experiment 180), and over all half-hours, with the tail slice of the sunny
  rows (coverage and pinball at q01, q05, q95, q99)
code: ml-experiments/00214-dp-pv-clip-ceiling-as-of-and-screen/
---

# Plan: does D5's clip ceiling hold as-of, and does another Primary show the signature?

**This experiment tests whether experiment 180's export clip at D5 is a real property of the
substation or a fit to the year it was scored on.** [Experiment
180](../00180-dp-sunny-daylight-fix-d5-solar-shape/result.md) chose the 60% ceiling on its tuning
months and scored it on the scored year, so the ceiling was never chosen as-of. The experiment
follows suggestion 1 of the DP PV overnight brief.

## Design

- **Stage 1, screen (no fit).** Repeat experiment 180's stage-0 diagnostic (implied output per unit
  of the fixed fleet shape, by decile of the shape, clear May to August half-hours) at D5, D9, D10,
  D11, D14, and D16, one panel per summer of 2022, 2023, and 2024, using rows before 2025-07-01
  only. A Primary shows the signature when the mean of its top two deciles is at most 85% of the
  mean of deciles 4 to 6 in all three summers. D9's decile values are negative in experiment 180,
  so the ratio is undefined there; D9 is swept regardless, because the brief names it first.
- **Stage 2, ceiling sweep.** Every Primary that shows the signature, and D9, runs arms with the
  clip at 50%, 60%, 70%, 80%, and 90% of the held capacity (100% is the control), through
  experiment 180's chain (monthly T-all fits, the A2 forecast, and the stage-2 arm `d`). Arms that
  experiment 180 already stored are reused: D5 at 60% and 80% over the scored year, and every
  ceiling at D5 over the tuning and replicate months. D11 and D14 run at 60% only, to check the
  "the rest move under 0.01 MW" part. D10 and D16 at 60% are experiment 180's placebos.
- **As-of ceiling.** At the first day of each month of the scored year (12 cuts), the ceiling is
  the one with the lowest mean absolute error of q50 on that Primary's sunny daylight half-hours
  over the 12 months before the cut, using only forecast origins whose 168-hour window ended before
  the cut. The sunny threshold comes from rows before 2024-07-01. The forecasts of the cut's own
  month are then taken from the arm with that ceiling. Where the control beats every clipped
  ceiling over the window, the as-of choice is no clip. The replicate months use an expanding
  window from 2024-07 with at least 3 months before the cut, which is post hoc contaminated,
  because experiment 180 chose 60% on the tuning months, and is labelled so.
- **Row definition and leads.** Experiment 178's rows: the scored year's 292 runs of
  `fold_a_daily_gap_v1`, outside the hand-filtered switching events, with a reading, leads 0.5 to
  168 hours. The as-of selection and the scored contrast use the same sunny definition.
- **Intervals.** Month-block bootstrap on paired differences (`studies.bootstrap`), 98.33% level.
  Placebo substations: D10 and D16.
- **Compute.** CPU only, at most 4 cores, thread caps all 1, inside `systemd-run --user --scope
  -p MemoryMax=16G -p OOMPolicy=continue`. The brief estimates about 3 core-hours.

## Earlier advice followed and departed from

- **Followed:** clip the solar term at D5 at 60% before comparing other solar shapes (experiment
  180); do not search tilt and azimuth per substation (experiment 180).
- **Departed from:** none.

## The five sizing triggers

- **What gets stored:** private parquet and JSON files under the experiment's data folder.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes for the as-of rule (window, criterion); the brief fixes
  the 12-month window.
- **Callers not nameable without a search:** no; scripts copy experiment 180's.
