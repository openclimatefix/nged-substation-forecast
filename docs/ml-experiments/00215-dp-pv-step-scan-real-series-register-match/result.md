---
id: '00215'
slug: dp-pv-step-scan-real-series-register-match
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
size: >-
  medium. Stored: private tables under data/ml-experiments/00215-dp-pv-step-scan-real-series-register-match/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  design for the register match, so the plan fixes the cluster rule and a random-date base rate.
  Callers: this folder's scripts copy experiment 207's step-path code; no change to
  `studies.dp_gam`. Asset graph: not touched.
hypothesis: >-
  The step path of experiment 207 (total-variation weight 3, April to August rows), run on the real
  2019 to 2026 series of the Primaries, finds real changes in solar capacity, and the steps that
  increase capacity fall near the dates when feed-in tariff solar installations were commissioned
  in the Primary's area.
prediction: >-
  Written before any fit. At least one step of 0.5 MW or more at D10 or D16 lies within 60 days
  of a feed-in tariff commissioning cluster, and the two placebo Primaries (D5 and D14) show fewer
  than 1 step of 0.5 MW or more per year. Reasons: experiment 207's no-injection false-step rate at
  weight 3 was 0.00 at D5 and 0.53 at D14 per year, so a rate above 1 per year at a placebo would
  mean real steps or a worse fit on real data. The author expects the match part to be fragile,
  because the register of feed-in tariff installations ends on 2021-03-24, so only steps dated
  before 2021-04-01 (the summer of 2020 and the change from August 2020 to April 2021) can be
  matched.
decision_rule: >-
  Steps are month-to-month changes of the fitted capacity between consecutive summer months above
  0.5 MW, and the net change from each August to the next April above 0.5 MW, at the weight 3
  of experiment 207. A feed-in tariff commissioning cluster at a Primary is at least 0.5 MW of
  installed capacity commissioned within 60 days either side of the step's month boundary,
  summed over postcode districts and weighted by each district's share of units inside the Primary
  (`fit_district_polygon_shares`). Only increases can match; decreases are reported as "inferred,
  unconfirmed". The base rate is the share of month boundaries from 2019-09 to 2021-03 at that
  Primary whose 60-day windows hold a cluster, and a match is called informative only where the
  base rate is under 50%. `supported`: the prediction holds, with the scan code reproducing
  experiment 207's stored no-injection path at weight 3 (gate). `partly supported`: steps of 0.5
  MW or more exist but none matches a cluster, or a placebo shows 1 or more per year. `not
  supported`: no step of 0.5 MW or more at any Primary. `inconclusive`: the gate fails. The
  threshold is zero cost, because the steps only feed the substation cards and change no
  forecast.
benchmark: >-
  Experiment 207's false-step rates at weight 3 (D5 0.00, D10 0.60, D11 0.67, D14 0.53, D16 0.13
  per year) and a random-date base rate for the register match
row_set_id: >-
  Experiment 106's feature cache, April to August rows, from each Primary's first usable instant
  to 2026-06-30, at D5, D9, D10, D11, D14, and D16. No forecast is scored. The fit is a
  retrospective scan over all rows and is not as-of.
metric: >-
  Steps above 0.5 MW per Primary and year, their size in MW and date, and for each increase the
  share-weighted feed-in tariff capacity commissioned within 60 days
code: ml-experiments/00215-dp-pv-step-scan-real-series-register-match/
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-22
best_arm: >-
  Step path at weight 3 on the real series: 7 steps of 0.5 MW or more at 4 of 6 Primaries in 5 years
  (D10 3, D11 2, D14 1, D16 1; D5 and D9 none), none could be matched to an installation date
headline: >-
  The fit finds 0 to 0.6 changes of more than 0.5 MW per year at each Primary (D5 0.00, D9 0.00, D10
  0.60, D11 0.40, D14 0.20, D16 0.20). That is no more than the 0 to 0.67 per year it invents on
  experiment 207's test series with no change (D5 0.00, D10 0.60, D11 0.67, D14 0.53), except at D16
  (0.20 against 0.13). No change can be matched to an installation, because the register ends on
  2021-03-24 and the fit uses only rows from 2021-07.
lessons:
  - >-
    The real series show no more changes above 0.5 MW than experiment 207's fit invents on series
    with no change, so the scan gives no evidence that any real change exists (figure 2).
  - >-
    The feed-in tariff register cannot confirm a change: the 2020-03-23 to 2021-07-19 COVID-19
    lockdown window, which the project excludes from training, removes every row before the
    register's last date (figure 3).
  - >-
    At D10, 1.7 MW of the 2.3 MW rise in fitted capacity is in changes above 0.5 MW; at D5 the
    fitted capacity stays between 2.2 and 2.8 MW (figure 1).
advice:
  - >-
    Do not describe any change from the weight-3 scan as a meter fault or a commissioning; each is
    inferred, unconfirmed, and the count is at the rate of experiment 207's false steps (figure 2).
  - >-
    To match changes to commissioning dates, use a register that covers 2021-07 onward, because the
    feed-in tariff register in the project ends on 2021-03-24 (figure 3).
  - >-
    Treat the first August-to-April change of the scan as a possible fit artefact, because D11 and
    D14 both show about +1.45 MW there and the fit has no earlier summer to anchor it (Results).
take_home_message: >-
  The method that finds jumps in solar capacity finds about as many in the real Primary series as it
  invents when none exists, so we cannot call any of them real. The only list of dated installations
  we hold ends in March 2021, before the months the fit uses, so no change could be checked against
  it. The changes do not alter the forecast, but they should not appear on the substation cards as
  confirmed. A list covering 2021 to 2026, or the metered record of a known installation, would
  settle the question.
---

# Do the real Primary series show jumps in solar capacity that match feed-in tariff installation dates?

**The fit finds no more jumps in solar capacity in the real Primary series than it invents on test
series where none exists, and the list of feed-in tariff solar installations stops before the
months the fit uses, so no jump can be confirmed.** We asked whether the power drawn at six Primary
substations (labelled D5, D9, D10, D11, D14, and D16), which feed homes and businesses, shows sudden
changes in the capacity of the solar panels connected to them. [Experiment
207](../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md) built an estimate of
solar capacity that is allowed to jump from one month to the next, and showed that the estimate
finds a solar farm that we had added to a substation's power. Here we ran that estimate on the real
April-to-August power of each Primary from July 2021 to June 2026, with nothing added, and listed
every change of more than 0.5 MW. We found 7 such changes across 4 of the 6 Primaries. The only
dated record of installations we hold, the feed-in tariff register (a list of small solar
installations paid under a UK subsidy scheme), ends on 2021-03-24. The project excludes every row
before 2021-07-20 from training, because of the COVID-19 lockdown. No change can therefore be matched
to an installation date.

![Figure 1. Fitted solar capacity of each Primary](figure_01_paths.svg)

*Figure 1: the fitted capacity rises at D10 to D16 and stays within 2.2 to 2.8 MW at D5, and the
marked month-to-month changes above 0.5 MW are few.*

![Figure 2. Changes per year against experiment 207's rate](figure_02_step_rate.svg)

*Figure 2: the real series show no more changes per year than experiment 207's series with no
change, except at D16.*

![Figure 3. Months trained on against the register's end](figure_03_register_window.svg)

*Figure 3: no month the fit trains on comes before the register's last commissioning date.*

**The prediction is partly supported: changes of 0.5 MW or more exist at 4 Primaries, none can be
matched to an installation, and the placebo Primaries stay under 1 change per year.** Whether a
change at D10 or D16 lies within 60 days of a register cluster could not be tested, so that part of
the prediction is neither confirmed nor refuted. D5 shows 0 changes and D14 shows 0.20 per year.
The check that the scan code reproduces experiment 207's stored result at D10 passed (largest
difference 1.7e-13, in the units of the fitted capacity).

## Results

We used April to August half-hours from 2021-07 to 2026-06 at each Primary (D9 from 2024-04), after
dropping the lockdown months and the periods of switching, which leaves 34,690 to 35,804 half-hours
per Primary and 19,041 at D9. The fit uses every row, including rows after the date it describes,
so it is a retrospective scan and not a forecast. The penalty on the total change in capacity has
weight 3, as in experiment 207.

| Primary | Month-to-month changes | August-to-April changes | Per year | Experiment 207, per year, no change added |
|---|---|---|---|---|
| D5 | 0 | 0 | 0.00 | 0.00 |
| D9 (2.25 years) | 0 | 0 | 0.00 | none |
| D10 | 2 | 1 | 0.60 | 0.60 |
| D11 | 1 | 1 | 0.40 | 0.67 |
| D14 | 0 | 1 | 0.20 | 0.53 |
| D16 | 0 | 1 | 0.20 | 0.13 |

- **All 7 changes are increases.** D10: +0.60 MW (August 2022), +0.51 MW (May 2023), and +0.54 MW
  from August 2022 to April 2023. D11: +1.08 MW (August 2022) and +1.46 MW from August 2021 to April
  2022. D14: +1.45 MW from August 2021 to April 2022. D16: +0.58 MW from August 2022 to April 2023.
  An August-to-April change is a net change across the winter. Its date is unknown within the
  winter, because the fit uses only April to August rows.
- **The two changes of about +1.45 MW in the first winter, at D11 and D14, are probably fit
  artefacts.** The two values are equal to within 0.02 MW, and the fit has no earlier summer to
  anchor either.
- **No change falls in the register's period, so none can be matched.** Between 2019-09 and
  2021-03, no 60-day window at any Primary holds 0.5 MW or more of share-weighted registered
  capacity (the base rate is 0%). The register holds almost nothing from those months: 393
  installations were commissioned across the whole register from April 2019 to March 2021, against
  860,064 before April 2019, so even a fit that covered 2019 to 2021 could not find a cluster. Every
  change is therefore described as inferred, unconfirmed.
- **Sensitivity (post hoc, not part of the verdict):** at penalty weight 0.3 the fit finds 9, 3, 4,
  and 6 changes at D10, D11, D14, and D16 (none at D5, 2 at D9). At penalty weight 30 it finds one
  change each at D10, D11, and D14. The count falls as the penalty grows, as experiment 207 found.
- **The fitted capacity at D10 rises from 0.4 MW in 2021 to 2.7 MW in 2026, and 1.7 MW of that 2.3
  MW rise is in changes above 0.5 MW.** D5 stays between 2.2 and 2.8 MW.

## Checks

- **Reproduction:** refitted using only the data up to 2025-07-01, experiment 207's cut-off, the
  scan's base path at D10 for weight 3 matches experiment 207's stored path in all 48 months
  (largest difference 1.7e-13, in the units of the fitted capacity). `e00215_scan.py` prints it.
- **Forecast checks:** this experiment scores no forecast, so the forecast checks do not apply.
- **Audit triggers:** none fired. The untested match is explained in the verdict paragraph and is
  not an unexplained contradiction.
- **Plan correction:** the plan expected that changes in the summer of 2020 could be matched. The
  lockdown window removes that summer from training.

## What to try next

- Match the changes to a register that covers 2021 to 2026, or to the metered record of a known
  installation, before reading any change as a commissioning.
- Show the changes on the substation cards as inferred, and only once one is confirmed as real.
- Find a way to use the lockdown months without biasing the fit, so that the fit has a summer
  before 2021 to anchor the first winter.
