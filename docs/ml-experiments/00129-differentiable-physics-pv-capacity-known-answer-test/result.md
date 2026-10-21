---
id: "00129"
slug: differentiable-physics-pv-capacity-known-answer-test
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
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
  dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and D16; fits use data to 2026-07-31 (a
  maintainer-approved exception to the 2026-07-01 hold-out, since no forecast is scored)
metric: >-
  Recovery (difference of fits divided by injected capacity), semi-synthetic coverage of the 80%
  range, placebo difference of fits, and constant-path drift per 5 years, all in points of the
  Primary's capacity or shares of the injected capacity
verdict: partly supported
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The capacity of the small solar panels behind five Primary substations can be read from the
  substations' power, with an 80% range 1.2 to 1.6 MW wide, and the capacity at the five Primaries
  grew by 0.7 to 2.5 MW between each Primary's first estimable season-year and 2025/26 (arm 2).
  Only one of the three methods tested, which penalises sudden jumps, passed every pre-registered
  gate, and the same penalty makes it slow to follow a sudden rise in capacity. The range is not
  yet safe to show on a card when demand rises on overcast days, so a card should state that bias.
date: 2026-10-21
pr: 1429
headline: >-
  Arm 4 (increments penalised, 0.1) passed all four gates (recovery 5 of 5 Primaries, coverage 23 of
  30 cells, placebo 0.12, pooled drift 0.037 per 5 years); arm 2 failed on drift (0.095) and arm 3
  failed on placebo (0.21) and drift (0.105).
best_arm: >-
  arm 4 at increment penalty 0.1, on the pooled-drift reading of the gate (D10 0.054 and D11 0.050
  sit just over it)
lessons:
  - >-
    The drift gate and the ability to follow a sudden rise in capacity pull in opposite directions.
  - >-
    A day-long load that rises on overcast days adds about 0.2 of the injected capacity to the
    recovered capacity and breaks coverage for the south-west plant.
advice:
  - >-
    Put the 80% range, not only a single value, on a substation card, with a note that a
    synthetic overcast-day load of 10% of capacity raised the estimate by about 0.2 of the true
    capacity (the cloud-load cells).
  - >-
    Refit the capacity only after the final ERA5 for the last month is out, and test a method that
    penalises jumps for its slowness on a step-up injection before adopting it (the step test).
  - >-
    Do not treat arm 3's sun-blind demand terms as a fix for the cloud-load bias: they recover
    0.02 to 0.03 more and raise the placebo share to 0.21 (stage C table).
code: ml-experiments/00129-differentiable-physics-pv-capacity-known-answer-test/
---

# Differentiable physics: can the embedded solar capacity behind a Primary be inferred, with an honest range?

**Take-home message: the capacity of the unmetered solar behind five Primary substations can be read
from their power, with an 80% range 1.2 to 1.6 MW wide, and the capacity grew at all five.** The arm
2 estimate of AC capacity in 2025/26 was 3.3 to 4.5 MW across the five Primary substations, 0.7 to
2.5 MW above each Primary's first estimable season-year (figure 1). A card should show the range
with a note on its bias: a synthetic load of 10% of capacity that rises on overcast days raised the
estimate by about 0.2 of the true capacity. The verdict is `partly supported`. Arm 4, which
penalises steps between season-years, passed every pre-registered gate when the drift gate is read
as a pooled mean, but arm 4 follows a sudden rise in capacity slowly (figure 3). The verdict `partly
supported` is a judgement beyond the pre-registered rule, which gives `supported` under the pooled
reading and no passing arm under a per-Primary reading. Arm 2 and arm 3 follow a sudden rise in
capacity and fail the drift gate, so the growth above is quoted from arm 2 with that caveat. No
forecast gain is scored.

**Terms used below.** D5, D10, D11, D14, and D16 are the five decision Primary substations of
experiment 103's data set. A season-year runs from September to August and is labelled by the year
of the September. The four arms are the experiment 45 frozen capacity (arm 1, a control and not a
truth), arm 2 (a capacity that steps up each season-year), arm 3 (arm 2 plus demand terms that
cannot see the sun), and arm 4 (arm 2 with a penalty on the steps). Recovery is the capacity found
with a known solar farm's output added to a Primary's power, minus the capacity found without it,
divided by the added capacity. The divisor is the median recovery of the other real farms at the
same Primary, which rescales a recovery to 1. A coverage cell is one Primary, one plant class, and
one load, with the known capacity checked against the 80% range. CAMS is the Copernicus Atmosphere
Monitoring Service's satellite irradiance, ERA5 is the European Centre's reanalysis of past weather,
and DESNZ is the UK Department for Energy Security and Net Zero.

**The fits stop at 2026-07-31, not 2026-08-31, because the final ERA5 for August 2026 is not out.**
The maintainer approved fitting to 2026-08-31 as an exception to the hold-out, so that season-year
2025/26 is complete. Season-year 2025/26 therefore ends in July. The code constants that mark where
the hold-out begins, and the grid end, are moved inside this experiment's scripts only
(`E00129_THROUGH_JULY=1`), and `packages/` is unchanged.

![Capacity per Primary and season-year](figure_01_capacity.svg)

**Capacity rose at all five Primaries, and experiment 45's pinned value sits above the range at D16
in every year.** The estimates at D10 and D11 are nearly flat before 2022/23 and then step up, and
the cause of the step was not tested. Arm 2 has no point for D14 and D16 in 2019/20 and 2020/21
because the known-answer test found no estimable recovery there. The dashed line is experiment 45's
pinned capacity, a control and not the truth.

## Gates

**Only arm 4 passed all four gates.** Arms 2, 3, and 4 were tested against the gates fixed before
stage C. Arm 3 is descriptive and cannot win. Arm 4 uses an increment penalty of 0.1. Its coverage
is 23 of 30 cells (77%), against a nominal 80%, so the range reaches the 21-cell gate but not its
nominal level.

| Gate | Limit | Arm 2 | Arm 3 | Arm 4 |
|---|---|---|---|---|
| Recovery of both mismatched plants within 0.75 to 1.25 after the divisor | 4 of 5 Primaries | 5 | 5 | 5 |
| Semi-synthetic coverage cells with the truth in the 80% range | 21 of 30 | 23 | 21 | 23 |
| Placebo difference of fits, share of the real-sky recovery (median, max) | at most 0.20 | 0.12, 0.13 | 0.21, 0.24 | 0.12, 0.12 |
| Drift per 5 years, pooled mean (max Primary) | under 0.05 | 0.095 (0.18) | 0.105 (0.20) | 0.037 (0.054) |

![Recovery of injected solar](figure_02_recovery.svg)

**Real farms' output was recovered at about 0.8 before the divisor, and the published estimates are
already divided by that recovery,, at 20, 50, and 100% of the pinned capacity alike.** A synthetic
load that rises on overcast days, of 0.1 of the injected capacity, adds about 0.2 to the recovery at
every Primary. When that load is present, the true capacity lies inside the 80% range for the
south-west plant in only 0 or 1 of 5 cells.

![Step-up injection](figure_03_step.svg)

**Arm 4 follows a sudden rise in capacity more slowly than arm 2 does.** Half of experiment 45's
pinned capacity was injected as a step-up in March 2022. At the median Primary, arm 2 and arm 3
recover nothing before March 2022 and about 0.82 to 0.87 after it. D10 is the exception, where arm 2
recovers 0.43 and arm 3 0.47 before the step. Arm 4 at penalty 0.1 recovers 0.47 before March 2022,
and at penalty 1.0 recovers 0.65. These are leakage into earlier years, because the true added
capacity there is zero.

## Primary power beside the estimated solar power

![Primary power and estimated solar power](figure_04_timeseries.svg)

**The estimated solar power is not visible as a one-for-one dip in the metered power.** Figure 4
shows the metered power of D5, D10, and D14 on a cloudy, a median, and a sunny day in May to July
2026. At each Primary the days are the lowest, the median, and the highest mean CAMS irradiance
      between 06:00 and 18:00 UTC among days with a full day of readings outside the switching
      events, chosen before any plot was drawn: D5 on 4 May, 15 June, and 2 July; D10 on 3 May, 1
      July, and 2 July; D14 on 4 May, 1 July, and 2 July. On the sunny days the metered power falls
      during daylight by about 1.6 to 2.3 MW between the 06:00 UTC level and the noon level (D5
      about 1.6, D10 about 1.7, D14 about 2.3), against estimated solar peaks of 3.9, 3.3, and 4.5
      MW, because the demand that the solar hides is not constant through the day. The flat tops of
      the estimated solar power are the inverter limit in the solar shape. The estimated solar power
      is arm 2's capacity times that shape before temperature derating, so the estimate is a little
      high at midday.

## Every range reaches the register totals

- **The Feed-in Tariff (FiT) register lies below every 80% range.** The register, shared to each
  Primary by postcode district, lists 1.9 to 2.6 MW of declared net capacity at sites of at most 1
  MW (2.0 to 2.8 MW of DC capacity) and 2.1 to 3.2 MW at sites of at most 5 MW (2.2 to 3.5 MW of DC
  capacity). The ranges reach these totals at all five Primaries. The register's "declared net
  capacity" is its name for AC capacity.
- **DESNZ's domestic solar by constituency, shared by area, gives 2.1 to 3.8 MW.** Its capacity
  basis (DC or AC) is not stated.
- **Holding the 1 September 2025 estimate forward undershoots the hindsight 2025/26 estimate by 0.02
  to 0.69 MW for arm 2.** The largest gaps are at D16 and D10, where capacity is still growing.
- **CAMS' own beam and diffuse irradiance changes the latest arm 2 estimate by 0.04 MW or less.**
  The beam and diffuse columns were already in the downloaded files, so no new download was needed.
  Recovery falls by 0.02 and the divided estimate rises by at most 0.04 MW.

## Caveats and departures from the plan

- **Arm 4 passes the drift gate on its pooled mean only.** The plan did not say whether the limit
  applies per Primary or pooled. Read per Primary, arm 4 fails at D10 (0.054) and D11 (0.050).
- **The arm 4 penalty was fixed at 0.1 without the planned tuning.** The plan tuned the penalty on
  D9 from semi-synthetic drift, but D9's drift is undefined (under 3 estimable years) and every
  penalty from 0.001 to 1.0 gave a recovery of about 0.75. The middle value 0.1 was fixed. Penalty
  0.01 gave drift 0.034 to 0.120, and penalty 1.0 gave 0.009 to 0.014, at the five Primaries.
- **The placebo limit of 0.20 is a post-pilot choice,** set after seeing arm 2 at 0.10 of the
  injection in the pilot.
- **Four planned comparisons were not run.** They are the December-factor comparison, experiment
  45's semi-synthetic constant-path re-run, and the Embedded Capacity Register and the Renewable
  Energy Planning Database comparisons, because experiment 45's processed files do not hold the last
  two. The register figures are lower bounds.
- **The monotone-arm and sun-blind-arm fits are copied from experiment 110 and are not yet in
  `packages/studies`.** Moving them there needs a separate module PR.

## What the next experiment should answer

**The next experiment should test whether explicit solar modelling lowers the net-demand forecast
error on the sunniest half-hours without raising it on dark ones.** Refitting each March instead of
each September, or each quarter, is a further question.

## Plan (written before any fit)

The full plan, with the Opus review's triage, is `data/ml-experiments/briefs/dp_pv_design_note.md`
(private). Its content is repeated here in short.

- **Deliverable:** a capacity-over-time estimate with an 80% range per Primary, for a substation
  card and for NGED. A forecast gain is not scored here. A follow-on experiment tests whether
  explicit solar modelling lowers the net-demand forecast error on the sunniest half-hours, without
  raising it on dark ones.
- **Inputs:** CAMS global irradiance at the Primary (beam and diffuse, found already in the
  downloaded files, for one sensitivity row), ERA5 2 m temperature, cleaned Primary net power with
  metered generators already subtracted, the hand-labelled switching events of experiment 103 (66
  events, plus and minus 24 hours), and five metered solar farms as the injection source.
- **Model:** experiment 110's convex model, with a fixed south-facing tilt-35 shape (orientation is
  not identified at a Primary, experiment 73) and one capacity per season-year, each year the
  previous year plus a non-negative increment. Fits run on the GPU where possible.
- **Arms:** (1) the experiment 45 frozen vintage as control; (2) monotone season-year capacity; (3)
  arm 2 with sun-blind demand; (4) arm 2 or 3 with a penalty on the increments, run only if arm 2's
  measured drift exceeds 0.05 per 5 years.
- **Known-answer test:** farm output at 20, 50 and 100% of the Primary's pinned capacity, as a
  matched plant and two mismatched synthetic plants (east-west, tilt 15, DC:AC 1.3, clipped;
  south-west, tilt 25), each with and without a day-long cloud-driven load of 10% of capacity, plus
  a step-up injection. Recovery is the difference of two fits divided by the injected capacity; it
  measures bias only, because the fit is nearly linear in its target. Coverage, drift and the
  placebo run on a semi-synthetic target with a known capacity.
- **Stages, each ending in a one-line report:** A, a beam and diffuse fetch at the five points (not
  needed: the columns were already downloaded); B, pilot on two Primaries; C, full grid; D,
  real-data tables; E, register comparison and write-up.

## Register comparison

The write-up compares the hindsight capacity per Primary with the Embedded Capacity Register,
Renewable Energy Planning Database and Feed-in Tariff figures that experiment 45 processed. The
registers are lower bounds, so the check is that our range lies above the register total of sites
below 1 MW (sensitivity at 5 MW).

## Stage B findings and rules fixed before stage C

Stage B ran on D10 and D16, on the feature cache that stops at 2026-06-30. The numbers are the
pilot's and are not results.

- **Fits cost about 0.5 s, and the GPU and CPU give identical capacities,** so the whole experiment
  takes minutes.
- **The ridge does not matter.** Ridge 0.01, 0.1 and 1.0 move the recovery by 0.01 or less, so the
  ridge pilot is dropped and 0.1 (experiment 104's value) is kept.
- **A real farm's output is recovered at 0.73 to 0.88 without load,** as experiments 39 and 63
  found. A day-long cloud-driven load of 10% of capacity adds about 0.21 to arm 2's recovery and
  0.20 to arm 3's.
- **Arm 3 recovers 0.02 to 0.03 more than arm 2 and does not cut the cloud-load bias.** Its placebo
  difference of fits (0.17 to 0.18 of the injection) is larger than arm 2's (0.10).
- **On the semi-synthetic target (true capacity zero), the capacity found from noise is 1 to 4% of
  the injected capacity.** The recovery's drift on a constant farm is +0.18 per 5 years at D10
  (standard deviation 0.16 over 10 draws and 5 farms) and +0.045 at D16, against a limit of 0.05.

**Rules for stage C, written before it runs.**

- **Arm 3 is descriptive.** It cannot win the verdict if it only matches arm 2. Its purpose is the
  stated claim that sun-blind demand does or does not cut the cloud-load bias, at five Primaries.
- **Arm 4 runs,** because arm 2's drift exceeds 0.05 per 5 years at D10. Its increment penalty is
  tuned on D9, which is not a decision Primary, from the semi-synthetic drift alone.
- **Post-pilot choice:** the placebo gate is "placebo difference of fits at most 20% of the real-sky
  recovery of the same arm and Primary". The 20% is a round number chosen after seeing arm 2 at 0.10
  of the injection, so the gate is not independent of the pilot.
- **Grid:** the five decision Primaries; 20, 50 and 100% of the pinned capacity; the five real
  farms, the east-west plant, and the south-west plant; with and without the cloud-driven load.
- **Divisor:** recovery is divided by the median recovery of the other four farms (matched plants,
  no load, 100%) at the same Primary and arm; the two synthetic plants use the median of all five.
- **Coverage cells:** 30 cells, one per Primary, plant class (matched with a rotating farm,
  east-west, south-west), and load, each at 100%. The interval is the stratified month-block
  bootstrap of the last complete season-year, divided by recovery ratios from the other Primaries
  and other farms.
