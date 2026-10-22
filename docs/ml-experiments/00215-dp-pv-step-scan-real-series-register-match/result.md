---
id: '00215'
slug: dp-pv-step-scan-real-series-register-match
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
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
---

# Plan: do the real series contain solar-capacity steps that match the register?

**This experiment scans the real series of the Primaries for steps in solar capacity, and checks
whether the steps match known commissioning dates.** [Experiment
207](../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md) showed that a
capacity path with a total-variation penalty finds a known step in summer data. This experiment
runs the same path on the real 2019 to 2026 data and compares the steps to the feed-in tariff
register of experiment 45. It follows suggestion 2 of the DP PV overnight brief.

## Design

- **Fit.** Experiment 207's step path at weight 3, copied, on April to August rows from each
  Primary's first usable instant to 2026-06-30, with no injection. Weights 0.3 and 30 are
  reported as sensitivity only.
- **Gate.** With the cut set to 2025-07-01 and no injection, the scan code reproduces the base
  level path that experiment 207 stored at D10 for weight 3.
- **Steps and register match.** As in the decision rule. Steps after 2021-03-31 are reported as
  "inferred, unconfirmed" because the register ends on 2021-03-24. A step is never called a meter
  fault.
- **Compute.** CPU only, at most 4 cores, thread caps 1, inside a 16 GB scope. The brief estimates
  about 3 core-hours; experiment 207's fits take minutes.

## Earlier advice followed and departed from

- **Followed:** fit the step path with a weight near 3 and count false steps as the net change from
  one summer to the next as well as single month-to-month changes (experiment 207).
- **Departed from:** none.

## The five sizing triggers

- **What gets stored:** private tables under the experiment's data folder.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes for the register match; the plan fixes the cluster rule.
- **Callers not nameable without a search:** no; the scripts copy experiment 207's.
