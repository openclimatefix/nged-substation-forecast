---
id: '00268'
slug: lsc-curtailment-metric-v2-member-wise-ens
title: >-
  A node-plus-generators curtailment metric with member-wise ENS inputs
theme: curtailment-cost
also_relevant_to:
- lsc-detectors
status: finished
size: >-
  medium. Stored: private joined frames under data/ml-experiments/00268-lsc-curtailment-metric-v2-member-wise-ens/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  way to turn ensemble members into the node's planned cut, so the choice (net export per member,
  then the 0.95 quantile over members) is fixed in this plan. Callers: the unchanged scoring
  entry point studies.node_curtailment.node_curtailment, experiment 255's scripts, and the
  experiments of the load-shift decision program; the new functions are additive. Asset graph:
  not touched.
hypothesis: >-
  A member-wise input path added to studies.node_curtailment (ensemble members of the node's
  demand and of each farm's output, paired by member, reduced to the quantiles the rule reads)
  gives the same avoidable cost as the quantile path when the members are drawn from the same
  quantile function, and scores nodes that experiment 255 could not (Primaries that connect a
  farm, and the bulk supply points D13 and D17 with member-wise inputs).
prediction: >-
  Re-scoring experiment 255's rows at 0.5 times each node's earlier maximum export with the
  quantile path of the extended module reproduces every arm's avoidable cost exactly (same code
  path), and the member path, fed 100 members interpolated from the same 11 stored quantiles,
  reproduces each arm's avoidable cost to within 1% and the sign and 95% interval inclusion of
  contrasts C1 to C5. A perfect forecast costs zero, and synthetic flows with a known right cut are
  scored exactly.
decision_rule: >-
  Tolerance, fixed before any code runs. On experiment 255's causal rows at f = 0.5, headline price
  GBP 2,000 per MWh, Curtailment Limit 0: (1) the quantile path reproduces each arm's avoidable
  cost to within 0.01% (floating-point noise); (2) the member path reproduces each arm's avoidable
  cost to within 1% and each contrast's point estimate to within 1 percentage point of the
  baseline's avoidable cost, with the same sign and the same answer to "does the 95% interval
  include zero" for C1 to C5. Pass needs both. A failure of (2) alone means the member reduction
  is not a faithful stand-in for the quantile path and is reported as such, with the cause.
  Unit tests (synthetic flows with a known right cut, a perfect forecast costs zero, members
  paired by index not by order, a missing member drops that member only) must pass in CI.
benchmark: >-
  Experiment 255's published avoidable costs and contrasts at f = 0.5 (causal rows)
row_set_id: >-
  Experiment 255's causal rows (Experiment 48's node forecast origins, 09 UTC, 12 to 36 hour leads,
  valid times 2025-09-01 to 2026-06-30, less the node rows whose daylight farm forecast comes from
  a later run than the node origin)
metric: >-
  Avoidable cost in GBP at 0.5 times each node's earlier maximum export, as in experiment 255
code: ml-experiments/00268-lsc-curtailment-metric-v2-member-wise-ens/
verdict: partly supported
verdict_note: >-
  The quantile path reproduces experiment 255 exactly and the member path keeps the sign and 95%
  interval verdict of all five contrasts, but the member path misses the pre-registered 1% limit on
  one arm's avoidable cost (2.0%, climatology) because a third of the stored farm forecasts have
  crossing quantiles.
best_arm: >-
  Not applicable: this experiment scores no forecast. The reproduced H5 minus XGBoost contrast is
  GBP -62,407 on experiment 255's stored quantiles and GBP -69,766 once crossing quantiles are
  sorted.
headline: >-
  Member-wise inputs now enter the curtailment metric and reproduce experiment 255 to within 2.0%
  per arm, and 34% of the stored farm-forecast rows turned out to have crossing quantiles.
take_home_message: >-
  The curtailment metric can now take ensemble members for a substation's demand and for each solar
  farm, and on experiment 255's rows it reproduces the earlier answer. In doing so it found that a
  third of the stored solar-farm forecasts list their quantiles out of order, which moves the cost
  of some forecasts by up to 3.7% without changing which forecast wins. The metric now sorts them,
  so the next experiments can score forecasts on it, although no member-wise farm or bulk supply
  point forecasts exist on disk yet and must be built first.
production_status: not-for-production
adopted_in: none
lessons:
  - >-
    Stored quantile sets can cross, so the median column is not the median of the stored forecast.
  - >-
    No member-wise farm or bulk supply point forecast exists for experiment 255's window.
advice:
  - >-
    Sort every stored quantile set before reading a single column such as the median; the metric
    does this by default and counts the rows that crossed (Numbers).
  - >-
    Build member-wise demand for the two bulk supply points D13 and D17 and member-wise farm output
    for farms A to F before scoring forecasts on member-wise curtailment cost, because neither
    exists on disk (Design).
  - >-
    Report Primary-level curtailment as a demand-only relative proxy, because no file links a farm
    to a Primary and the farms probably connect at the bulk supply point (Design).
---
**The curtailment metric can now take ensemble members, and it reproduces experiment 255, but the
stored farm forecasts it was checked on have quantiles in the wrong order in 34% of rows.** This is
step 2 of the load-shift decision program. We asked whether the node-plus-generators curtailment
metric of `studies.node_curtailment` can read ensemble members (51 weather-model runs of one
forecast) for a substation's demand and for each solar farm, and whether it then gives the answer
that experiment [255](../00255-metered-farm-node-curtailment-value/result.md) found. We added two
functions, scored experiment 255's rows at 0.5 times each substation's earlier maximum export both
from the stored quantiles and from 101 members drawn from them, and compared both with experiment
255's stored costs. **Verdict: partly supported.** The quantile path matches experiment 255 exactly.
The member path keeps the sign and the 95% interval verdict of all five contrasts (largest
difference 0.58 points of the baseline's avoidable cost, under the 1 point allowed), but one arm's
avoidable cost differs by 2.0% against the 1% allowed. The cause is in the stored forecasts:
26,344 of 77,897 solar-farm forecast rows (34%) store quantiles that cross, so a set of members
cannot reproduce them. The metric now sorts each row's quantiles by default.

## Design

- **Scoring nodes.** The step-1 mapping search found no file linking a farm to a Primary, and the
  six farms are probably connected at the bulk supply point. The generator-aware nodes are therefore
  the bulk supply points D13 (farms A, C, E) and D17 (farms B, D, F), with their child Primaries as
  demand. Primary-level curtailment is a demand-only relative proxy: it ranks forecasts and states
  no money saved.
- **Member path.** `net_flow_members` pairs the demand's and each generator's members by member
  number and returns the node's metered flow per member. `quantiles_from_members` reduces members to
  the `q<percent>` columns that `node_curtailment` reads, interpolating linearly between sorted
  members. `rearrange_quantiles` sorts a stored row's quantiles; `NodeCurtailmentAssumptions`
  switches it on by default, and the result counts the rows that crossed.
- **Reproduction check.** Pre-registered in the plan: tolerance 0.01% for the quantile path and 1%
  per arm for the member path, contrasts within 1 point, same sign, same 95% interval verdict. The
  plan said 100 members; the script uses 101 (probabilities 0, 0.01, ..., 1), because only then does
  linear interpolation between sorted members return the stored quantile at every whole-percent
  level. Post hoc: the quantile path with sorting off, and the member path on sorted quantiles.

## Numbers

- **Quantile path, sorting off.** Every arm's avoidable cost equals experiment 255's, with largest
  difference 0.0000%, and every contrast equals it (e.g. H5 minus XGBoost GBP -62,407).
- **Member path on the stored quantiles.** Avoidable cost differs from experiment 255 by -2.00%
  (climatology floor), -1.09% (manual-heuristic floor), -0.79% (plant), -0.78% (H5 with zero
  history), -0.76% (H5), and -0.69% (XGBoost). Contrasts differ by at most 0.58 points of the
  baseline's avoidable cost, keep their signs, and keep their 95% interval verdicts. The 1% arm
  limit is missed by the two floor arms.
- **Why it misses.** 26,344 of 77,897 farm-forecast rows cross in every forecast arm (158,064 rows
  summed over the six arms). With the quantiles sorted first, the member path and the quantile path
  agree to 0.0000% on every arm.
- **Sorting moves the costs.** With sorting on (the default), avoidable cost differs from
  experiment 255's by -3.66% (climatology floor), -1.90% (manual-heuristic floor), +1.08% (XGBoost),
  +0.72% (H5), +0.62% (plant), and -0.59% (H5 with zero history). Contrasts differ by at most 0.81
  points; H5 minus XGBoost is GBP -69,766 against -62,407.
- **Rows.** Experiment 255's causal rows, 2025-09-01 to 2026-06-30, 12 to 36 hour leads, headline
  price GBP 2,000 per MWh, Curtailment Limit 0, f = 0.5.
- **Tests.** 88 tests in `test_node_curtailment.py` pass, including hand-worked member cases: members
  pair by number not by row order, a missing generator member drops only that member, and a 20-member
  case gives exactly 3.25 MWh planned and GBP 6,500.

## Checks

- **Median minus mean forecast per site, and the mean-forecast error beside each contrast:** not
  applicable, because the experiment trains no forecast.
- **Triggers.** Trigger 1 fired (a pre-registered tolerance was missed). The explanation, crossing
  stored quantiles, was tested directly by sorting first, which removes the whole gap. No audit was
  run for the member path's miss; the maintainer or manager can ask for one. The other six triggers
  did not fire.

## Caveats and next steps

- **No member-wise inputs exist for a real run.** Experiment 255's window has no member-wise farm
  output (the stored farm members of experiments 229 and 246 hold no member column) and no
  member-wise demand at D13 or D17 (the stored 51-member demand forecasts of experiment 125 cover
  3 of D13's 7 child Primaries and 3 of D17's 9). Step 3 therefore needs those frames built first.
- **Sorting changes costs by up to 3.7%**, so a number from before 2026-10-23 on stored farm
  forecasts differs slightly from the same number now. Experiment 255's page is unchanged.
- **The metric reads each generator's median only**, so a better farm tail still does not show in
  its cost.
