---
id: "00267"
slug: pv-capacity-early-years-and-register-floor
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
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
verdict: not supported
production_status: not-for-production
adopted_in: none
date: 2026-10-23
headline: >-
  The card's estimate already divides by the median recovery of the metered farms, so the
  division arm changes nothing (within 3%), and no Primary's 2021 season-year is thin (12 months at
  all five). Using the nearest metered farm as the sunshine shape lowered the 2021 estimate by 7%
  at D10 and by 22% at D11, so the upper end of the 80% range stays below the register floor at both
  (1.15 MW against 1.90 MW, and 1.55 MW against 2.57 MW).
best_arm: A0, the card's estimate (no arm lifted D10 or D11 to the floor)
take_home_message: >-
  The 2021 shortfall at D10 and D11 is not a missing recovery correction and not a thin fit window:
  the card already divides by the metered farms' recovery, and the 2021 season-year has 12 months of
  data at every Primary. Using a nearby metered farm's output as the sunshine shape makes the
  shortfall worse (the estimates fall by 7% to 22%), so the estimate at those two Primaries stays
  "consistent with the register" only if the register's postcode-district share over-states the
  Primary. Show 2021 at D10 and D11 as a known shortfall and look at the register match next.
lessons:
  - >-
    The brief's predicted lift (1.3 to 1.5 MW at D10, 2.3 to 2.7 MW at D11) is what a second
    division by the recovery gives (1.39 and 2.39 MW), which would overcorrect by about 25%.
  - >-
    The nearest-farm shape lowers the capacity estimate at every Primary and year, because the
    recovery under that shape is higher (0.82 to 0.90 against 0.77 to 0.86), so the raw capacity
    is divided by a larger number.
advice:
  - >-
    Do not divide the card's capacity estimate by a recovery a second time: it already carries the
    division by the Primary's median matched-farm recovery (A1 table).
  - >-
    Look for the cause of the 2021 shortfall at D10 and D11 in the register match (postcode
    district shared to a Primary) before changing the fit (figure 1).
  - >-
    Do not use a nearby metered farm's output as the sunshine shape on the cards: it lowers every
    estimate by 5% to 22% and clears the register floor at fewer season-years (figure 1).
code: ml-experiments/00267-pv-capacity-early-years-and-register-floor/
---

# PV capacity: why the 2021 estimate falls below the register floor

**Take-home message: the 2021 shortfall at D10 and D11 is not a missing correction and not a thin
fit window, and a nearby metered farm's output makes the shortfall worse.** The card's estimate
already divides by the recovery of the metered farms, so the division changes it by under 3%. Every
Primary has 12 months of data in the 2021 season-year. The nearest-farm shape lowers the 2021
estimate by 7% at D10 and 22% at D11, so the upper end of the 80% range stays below the register
floor at both Primaries.

**What was asked.** The card for each of five Primary substations (D5, D10, D11, D14, and D16)
shows the capacity of unmetered solar panels behind the substation, inferred from its power
readings. For the July 2021 to June 2022 season-year, the estimate at D10 (1.1 MW) and D11 (1.9 MW)
is below the capacity the Feed-in Tariff register lists in the Primary's area (1.9 MW and 2.6 MW of
alternating-current capacity at sites of at most 1 MW, which is a lower bound because the register
is partial). We tested two repairs: dividing by the median recovery of the metered solar farms (the
share of an injected farm that the fit finds again, experiment 241), and fitting with the output of
the nearest metered farm as the sunshine shape (experiment 238) in place of the satellite
irradiance. Both are offline checks that score no forecast.

![Capacity of unmetered solar per season-year](figure_01_capacity.svg)

**Figure 1. At D10 and D11 the card's 2021 range (orange) lies below the register floor (dashed
line), and the nearest-farm shape (blue) lowers the estimates.**

## Findings

- **A1, the division by the recovery, is the card's own estimate.** The card divides the fitted
  capacity by the Primary's median recovery of the matched farms (0.77 to 0.86 over the Primaries
  and season-years). Experiment 241's leave-one-farm-out divisor differs from that by at most 0.02
  in recovery, and its estimate by at most 3% (D16 2022: 3.25 MW against 3.33 MW). Its median
  absolute error on a held-out farm is 0.063 of the injected capacity with the card's fit. Dividing
  the card's estimate by 241's recovery as well gives 1.39 MW at D10 and 2.39 MW at D11 in 2021,
  which matches the brief's predicted lift, but it is a second correction of the same bias and
  would overstate capacity by about 25%. Its upper range ends (1.62 MW and 2.62 MW) still sit below
  the D10 floor (1.90 MW) and the D11 DC floor (2.78 MW).
- **A3, the thin-window rule, never fires.** Season-year 2021 has 12 months and 16,390 to 16,608
  training half-hours (of 17,520) at every Primary. Season-years 2022 to 2025 have 13
  calendar months at every Primary by the month in which each half-hour ends.
  Training rows begin in July 2021, so season-year 2021 is a full year, and the register's last
  entry (24 March 2021) falls before the season-year starts. The estimate therefore sits below
  capacity that was already installed.
- **A2, the nearest-farm shape, lowers every estimate.** The nearest farm to each Primary is 3.1 km
  (D5), 7.7 km (D10), 23.1 km (D11), 5.3 km (D14), and 5.8 km (D16) away, so a farm lies within
  25 km of all five. Against the card's estimate in 2021, A2 gives 1.04 MW at D10 (-7%), 1.46 MW at
  D11 (-22%), 2.74 MW at D14 (-17%), 2.08 MW at D16 (-20%), and 3.20 MW at D5 (-14%). The prediction
  of under 10% change at D5, D14, and D16 failed.
- **Register floor, upper end of the 80% range, AC floor.** The card's estimate clears it in 23 of
  25 Primary season-years, and misses D10 2021 (1.31 MW against 1.90 MW) and D11 2021 (2.07 MW
  against 2.57 MW). A2 clears it in 22 of 25 (it also misses D16 2021, 2.18 MW against 2.20 MW) and
  ties at D10 2022 (1.90 MW). Against the DC floor, the card misses the same two season-years and
  A2 misses five.
- **No arm met the decision rule.** The rule asks for the upper end to reach the floor at all five
  Primaries in every season-year with register coverage. A0 and A2 both fail at D10 and D11.
- **The early-year level is not an artefact of the increment penalty.** Experiment 129's arm 2,
  without that penalty, gives 0.82 MW at D10 and 1.38 MW at D11 in 2021, no higher than the card.
- **The capacity rises by about 75% between 2021 and 2022** at both Primaries (D10 1.13 to 1.98 MW,
  D11 1.88 to 3.32 MW). A fitted capacity that rises that fast between two complete years is
  consistent with a level that is too low in 2021, but also with real installations in 2021 and
  2022.
- **The A2 known-answer check passes at 4 of 5 Primaries.** The leave-one-injected-farm-out median
  absolute error under the nearest-farm shape is 0.02 (D10), 0.02 (D11), 0.13 (D14), 0.04 (D16), and
  0.06 (D5), with 2 to 4 injected farms per Primary. D14 misses the 10% bound.

## Checks

- **Reproduction:** refitting the satellite shape with the card's settings reproduces the stored raw
  capacity of every Primary and season-year to within 1e-6, as the script asserts.
- **Median forecast minus mean forecast, and the mean-forecast error:** not applicable, because the
  experiment scores no forecast.
- **Audit triggers** (the seven triggers in the `ml-experiment` skill's "Audit before merge"):
  trigger 2 fires, because the result contradicts the prediction. The prediction's two causes were
  tested (the division and the window), and the A2 fall in the estimate was explained by the higher
  recovery under that shape, but not tested further. No audit was run, because the ML manager's rule
  is Sonnet reviews for capacity experiments; one Sonnet reviewer read the code and the page.

## Limits

- **The register is partial and shared by postcode district,** so passing it means "consistent
  with", and failing it at D10 and D11 may mean the register over-states the Primary's share.
- **The A2 range uses 2 to 4 injected farms,** so its range is narrow and not a calibrated 80%
  interval (experiment 241).
- **Divisors for A2 come from experiment 238's injection fits,** which use no increment penalty and
  data to 2026-07-01, and read the recovery at the 2024 season-year. The card's fit uses the
  penalty and data to 2026-07-31, as experiment 129's real-series fit did.
- **A neighbouring farm's output carries plant shape as well as sunshine** (experiment 238).

## What the next experiment should answer

- **Does the register's district share over-state D10 and D11 in 2021?** Compare the register's
  postcode districts with the Primaries' supply areas. The prediction that could fail is a share
  below 70% of the district's listed capacity at both Primaries.
- **Is the 2021 level fitted too low because 2021 is the first full year?** Fit the 2021 and 2022
  season-years with the 2022 level as a lower bound on 2021, and compare the fit's error on 2021
  half-hours. The prediction that could fail is an error no worse than the free fit.

## Plan: departures from the brief

- **A1** is the card's own estimate, so it is reported as the comparison above and not as a
  separate fit.
- **A2** fits the real series with the nearest of the five metered farms with data from 2019-09,
  within 25 km of the Primary, and divides by experiment 238's median recovery of injected farms
  under that shape. The range is the fit's month-block bootstrap (40 resamples) divided by those
  recoveries.
- **A3** prints the months and training half-hours per season-year, with a "thin" mark under 6
  months, which no season-year reaches.
- **The register floor** is the 1 MW-cut AC capacity (the estimate is AC), with the DC capacity
  shown beside it.
