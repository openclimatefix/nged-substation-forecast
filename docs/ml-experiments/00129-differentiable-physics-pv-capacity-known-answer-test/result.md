---
id: "00129"
slug: differentiable-physics-pv-capacity-known-answer-test
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  complex. Stored: private files under data/ml-experiments/00129-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (four arms, plus the
  interval and gate designs). Callers: spans dp_gam, pv_physics, pv_fit, and capacity_seasonal in
  packages/studies, which are called, not edited. Asset graph: not touched.
hypothesis: >-
  The AC capacity of the unmetered solar behind each of the five decision Primaries can be inferred
  offline, one value per September-to-August season-year, from CAMS irradiance, ERA5 temperature, and
  the Primary's net power, with an 80% range that holds its coverage when the real panels differ from
  the assumed south-facing shape and when a day-long cloud-driven demand is present.
prediction: >-
  Written before any fit. The matched plant (a real metered farm) is recovered at 0.75 to 0.85 before
  the stated divisor; the east-west plant at 0.65 to 1.0 after it; sun-blind demand (arm 3) equals
  arm 2 within noise on the cloud load; arm 2's drift on a semi-synthetic target is a few per cent per
  5 years, so arm 4 may not run; and no arm passes every gate at every Primary (3 of 5 on recovery).
decision_rule: >-
  An arm earns a substation card if it passes every gate: recovery of both mismatched plants (no
  cloud load) in 0.75 to 1.25 after the divisor at 4 of 5 Primaries; at least 21 of 30
  semi-synthetic cells (5 Primaries x 3 plants x 2 loads) with the true capacity inside the 80%
  range; placebo difference of fits under a threshold fixed after stage B and before stage C; and
  semi-synthetic constant-path drift under 0.05 per 5 years. No forecast gain is scored. A run with no
  passing arm is reported as "the range cannot yet be shown on a card".
benchmark: >-
  Arm 1, the experiment 45 frozen vintage (control, not truth), and experiment 45's semi-synthetic
  constant-path fit (path error 0.027) re-run on the same rows
row_set_id: >-
  dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and D16; fits use data to 2026-08-31 (a
  maintainer-approved exception to the 2026-07-01 hold-out, since no forecast is scored)
metric: >-
  Recovery (difference of fits divided by injected capacity), semi-synthetic coverage of the 80%
  range, placebo difference of fits, and constant-path drift per 5 years, all in points of the
  Primary's capacity or shares of the injected capacity
production_status: not-for-production
---

# Differentiable physics: can the embedded solar capacity behind a Primary be inferred, with an honest range?

**We ask whether the capacity of the small solar panels behind a Primary substation can be inferred
from the power the substation draws, with a range that is honest.** We add the recorded output of
real solar farms to a substation's real power draw, and check how much of it a fitting method finds
again, including when the panels face a different way from the fitted shape and when demand rises on
overcast days. The experiment is planned and nothing has run.

**Exception to the hold-out:** the maintainer approved fitting on data up to 2026-08-31, so that
season-year 2025/26 (September 2025 to August 2026) is complete. No forecast is scored, so no
hold-out is spent. The scripts take the cut as a parameter, and the shared `HOLDOUT_START` constant
is not changed.

## Plan

The full plan, with the Opus review's triage, is `data/ml-experiments/briefs/dp_pv_design_note.md`
(private). Its content is repeated here in short.

- **Deliverable:** a capacity-over-time estimate with an 80% range per Primary, for a substation card
  and for NGED. A forecast gain is not scored here. A follow-on experiment tests whether explicit
  solar modelling lowers the net-demand forecast error on the sunniest half-hours, without raising it
  on dark ones.
- **Inputs:** CAMS global irradiance at the Primary (beam and diffuse fetched at the five points for
  one sensitivity row), ERA5 2 m temperature, cleaned Primary net power with metered generators
  already subtracted, the hand-labelled switching events of experiment 103 (66 events, plus and minus
  24 hours), and five metered solar farms as the injection source.
- **Model:** experiment 110's convex model, with a fixed south-facing tilt-35 shape (orientation is
  not identified at a Primary, experiment 73) and one capacity per season-year, each year the previous
  year plus a non-negative increment. Fits run on the GPU where possible.
- **Arms:** (1) the experiment 45 frozen vintage as control; (2) monotone season-year capacity; (3)
  arm 2 with sun-blind demand; (4) arm 2 or 3 with a penalty on the increments, run only if arm 2's
  measured drift exceeds 0.05 per 5 years.
- **Known-answer test:** farm output at 20, 50 and 100% of the Primary's pinned capacity, as a matched
  plant and two mismatched synthetic plants (east-west, tilt 15, DC:AC 1.3, clipped; south-west, tilt
  25), each with and without a day-long cloud-driven load of 10% of capacity, plus a step-up
  injection. Recovery is the difference of two fits divided by the injected capacity; it measures
  bias only, because the fit is nearly linear in its target. Coverage, drift and the placebo run on a
  semi-synthetic target with a known capacity.
- **Stages, each ending in a one-line report:** A, fetch beam and diffuse irradiance at the five
  points; B, pilot on two Primaries; C, full grid; D, real-data tables; E, register comparison and
  write-up.

## Register comparison

The write-up compares the hindsight capacity per Primary with the Embedded Capacity Register,
Renewable Energy Planning Database and Feed-in Tariff figures that experiment 45 processed. The
registers are lower bounds, so the check is that our range lies above the register total of sites
below 1 MW (sensitivity at 5 MW).

## Stage B findings and rules fixed before stage C

Stage B ran on D10 and D16, on the feature cache that stops at 2026-06-30. The numbers are the pilot's
and are not results.

- **Fits cost about 0.5 s, and the GPU and CPU give identical capacities,** so the whole experiment
  takes minutes.
- **The ridge does not matter.** Ridge 0.01, 0.1 and 1.0 move the recovery by 0.01 or less, so the
  ridge pilot is dropped and 0.1 (experiment 104's value) is kept.
- **A real farm's output is recovered at 0.76 to 0.84 without load,** as experiments 39 and 63 found.
  A day-long cloud-driven load of 10% of capacity adds about 0.21 to arm 2's recovery and 0.20 to arm
  3's.
- **Arm 3 recovers 0.02 to 0.03 more than arm 2 and does not cut the cloud-load bias.** Its placebo
  difference of fits (0.17 to 0.18 of the injection) is larger than arm 2's (0.10).
- **On the semi-synthetic target (true capacity zero), the capacity found from noise is 1 to 4% of the
  injected capacity.** The recovery's drift on a constant farm is +0.18 per 5 years at D10 (standard
  deviation 0.16 over 10 draws) and +0.045 at D16, against a limit of 0.05.

**Rules for stage C, written before it runs.**

- **Arm 3 is descriptive.** It cannot win the verdict if it only matches arm 2. Its purpose is the
  stated claim that sun-blind demand does or does not cut the cloud-load bias, at five Primaries.
- **Arm 4 runs,** because arm 2's drift exceeds 0.05 per 5 years at D10. Its increment penalty is
  tuned on D9, which is not a decision Primary, from the semi-synthetic drift alone.
- **Post-pilot choice:** the placebo gate is "placebo difference of fits at most 20% of the real-sky
  recovery of the same arm and Primary". The 20% is a round number chosen after seeing arm 2 at 0.10 of
  the injection, so the gate is not independent of the pilot.
- **Grid:** the five decision Primaries; 20, 50 and 100% of the pinned capacity; the five real farms,
  the east-west plant, and the south-west plant; with and without the cloud-driven load.
- **Divisor:** recovery is divided by the median recovery of the other four farms (matched plants, no
  load, 100%) at the same Primary and arm; the two synthetic plants use the median of all five.
- **Coverage cells:** 30 cells, one per Primary, plant class (matched with a rotating farm,
  east-west, south-west), and load, each at 100%. The interval is the stratified month-block
  bootstrap of the last complete season-year, divided by recovery ratios from the other Primaries and
  other farms.
