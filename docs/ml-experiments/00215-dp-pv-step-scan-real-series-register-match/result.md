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
  (D10 3, D11 2, D14 1, D16 1; D5 and D9 none), none matched to the register
headline: >-
  At weight 3 the real series show 0.0 to 0.6 steps above 0.5 MW per year per Primary (D5 0.00, D9
  0.00, D10 0.60, D11 0.40, D14 0.20, D16 0.20), at or below experiment 207's false-step rate on
  series with no step (0.00, 0.60, 0.67, 0.53) at D5, D10, D11, and D14 and 0.07 above it (0.13) at
  D16. No step can
  be matched, because the register ends on 2021-03-24 and the fit trains from 2021-07.
lessons:
  - >-
    The real series show no more steps above 0.5 MW than experiment 207's fit invents on series
    with no step, so the scan gives no evidence that any real step exists (figure 2).
  - >-
    The feed-in tariff register cannot confirm a step: the lockdown window removes every row before
    2021-07-20, after the register's last date (figure 3).
  - >-
    The fitted capacity grows smoothly at D10, D11, D14, and D16 (from about 0.4 to 2.7 MW at D10 over
    5 years) and stays flat at D5, so most of the growth is not in steps (figure 1).
advice:
  - >-
    Do not describe any step from the weight-3 scan as a meter fault or a commissioning; it is
    inferred, unconfirmed, and at the rate of the false steps in experiment 207 (figure 2).
  - >-
    To match steps to commissioning dates, use a register that covers 2021-07 onward, such as a newer
    installations list, because the feed-in tariff register in the project ends on 2021-03-24
    (figure 3).
  - >-
    Count the first August-to-April change of the scan as a possible fit artefact, because D11 and
    D14 both show +1.45 MW there and the fit has no earlier summer to anchor it (Results).
take_home_message: >-
  Fitting a step-capable solar capacity to the real Primary series finds 7 changes above 0.5 MW in
  5 years, at about the rate that the same fit invents on data with no step, so none can be called
  real. The feed-in tariff register that could confirm them ends in March 2021, before the data the
  fit uses begins, so the match could not be tested. A register or a metered record that covers 2021
  to 2026 is what would settle it.
---

# Do the real Primary series contain solar-capacity steps that match the register?

**The scan finds only as many steps in the real series as the same fit invents on series with no
step, and the register that could confirm them ends before the data begins, so no step can be
called real.** [Experiment 207](../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md)
built a solar capacity that may change by a step each month, and showed that it finds a known
solar-farm step added to a Primary substation's power. A Primary substation feeds homes and
businesses; D5, D9, D10, D11, D14, and D16 are anonymised labels for six of them. We ran the same
fit on each Primary's real power from 2021 to 2026, with no step added. We listed the months in
which the fitted capacity moved by more than 0.5 MW, and compared the increases with the dates when
feed-in tariff solar installations were commissioned. The fit found 7 such changes in 5 years, at
four of the six Primaries. We could not compare them with the register: its last commissioning date
is 2021-03-24, and the fit trains only on rows from 2021-07.

![Figure 1. Fitted solar capacity of each Primary](figure_01_paths.svg)

*Figure 1 shows the fitted capacity of each Primary with the month-to-month steps above 0.5 MW.*

![Figure 2. Steps per year against experiment 207's rate](figure_02_step_rate.svg)

*Figure 2 shows how many steps per year the fit finds in the real series, against how many it finds
in experiment 207's series with no step.*

![Figure 3. Months trained on against the register's end](figure_03_register_window.svg)

*Figure 3 shows that no month the fit trains on comes before the register's last commissioning
date.*

**Verdict `partly supported`.** The decision rule's `partly supported` branch applies: steps of 0.5
MW or more exist at four Primaries and none matches a register cluster. The match could not be
tested, so the prediction that a step at D10 or D16 lies within 60 days of a cluster is neither
confirmed nor refuted. The placebo part holds: D5 shows 0 steps and D14 shows 0.20 per year, both
under 1 per year. The gate passed: the scan code reproduces experiment 207's stored fitted
capacity at D10 to within 2e-13 points.

## Results

Rows: April to August half-hours from 2021-07 to 2026-06 at each Primary (D9 from 2024-04), after
the project's default row filter (outside the lockdown window and the switching masks), 34,690 to
35,804 rows per Primary and 19,041 at D9. The scan is retrospective and uses every row, so it is
not as-of. The fit is experiment 207's step path with total-variation weight 3.

| Weight 3, steps above 0.5 MW | Month-to-month | August-to-April | Per year | Experiment 207 (no step) |
|---|---|---|---|---|
| D5 | 0 | 0 | 0.00 | 0.00 |
| D9 (2.25 years) | 0 | 0 | 0.00 | none |
| D10 | 2 | 1 | 0.60 | 0.60 |
| D11 | 1 | 1 | 0.40 | 0.67 |
| D14 | 0 | 1 | 0.20 | 0.53 |
| D16 | 0 | 1 | 0.20 | 0.13 |

- **The steps (weight 3, all increases):** D10 +0.60 MW (2022-08), +0.51 MW (2023-05) and +0.54 MW
  from August 2022 to April 2023; D11 +1.08 MW (2022-08) and +1.46 MW from August 2021 to April
  2022; D14 +1.45 MW from August 2021 to April 2022; D16 +0.58 MW from August 2022 to April 2023.
  The August-to-April steps are net changes across the winter, which has no data, so their date is
  unknown within the winter.
- **Likely artefact:** the +1.45 MW changes at D11 and D14 in the first winter of the fit are
  equal to within 0.02 MW. The fit has no earlier summer to anchor them, so we treat them as
  possible fit artefacts and not as installations.
- **Match to the register:** none of the steps lies in the register's period. The share of month
  boundaries between 2019-09 and 2021-03 whose 60-day window holds at least 0.5 MW of
  share-weighted registered capacity is 0 at every Primary, because each Primary's weighted
  register total is 1.6 to 3.5 MW over its whole history, 70% to 90% of it commissioned in 2011 to 2015.
  Every step is therefore "inferred, unconfirmed".
- **Sensitivity (post hoc, not part of the verdict):** at weight 0.3 the fit finds 9, 3, 4, and 6
  steps at D10, D11, D14, and D16 (none at D5, 2 at D9), and at weight 30 one step at each of D10,
  D11, and D14. Steps shrink in number as the penalty grows, as experiment 207 found.
- **Capacity:** the fitted capacity at D10 rises from 0.4 MW in 2021 to 2.7 MW in 2026, a rise
  of 2.3 MW, and 1.7 MW of it is in steps above 0.5 MW. D5 stays between 2.2 and 2.8 MW.

## Checks

- **Gate:** the scan's base path at D10 for weight 3, refitted at experiment 207's cut of
  2025-07-01 on the same rows, matches experiment 207's stored path in all 48 months (largest
  difference 1.7e-13 points). `e00215_scan.py` prints it.
- **Forecast checks:** this experiment scores no forecast, so the median-against-mean check and the
  quantile-score contrast do not apply.
- **Audit conditions:** none was met. The prediction's match part could not be tested, and the
  reason is recorded above, so it is not an unexplained contradiction.
- **Plan correction:** the plan expected that steps in the summer of 2020 could be matched. The
  lockdown window removes that summer from training, so the match was untestable from the start.

## What to try next

- Match the steps to a register that covers 2021 to 2026, or to the metered record of a known
  installation, before reading any step as a commissioning.
- Keep the steps out of the substation cards until one is confirmed, and show them as inferred.
- A longer history, with the lockdown months handled, would give the fit a summer before 2021 to
  anchor the first winter.
