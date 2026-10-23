---
id: '00268'
slug: lsc-curtailment-metric-v2-member-wise-ens
title: >-
  A node-plus-generators curtailment metric with member-wise ENS inputs
theme: curtailment-cost
also_relevant_to:
- lsc-detectors
status: planned
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
---
**Planned. This experiment extends the curtailment metric so that it reads ensemble members, and
checks that the extension reproduces experiment 255 before any later experiment relies on it.**
It is step 2 of the load-shift decision program (the scoring tool that steps 3 to 7 use).

## Plan

- **Design.** For each node row and each ensemble member, net export is the node's demand-side
  flow minus the generators' output in that member, members paired by member number. The planned
  cut reads the 0.95 quantile of net export over members, as the quantile path does. The cut is
  shared among generators by the median over members of each generator's forecast.
- **Extension.** New functions in `studies.node_curtailment` reduce members to the `q<percent>`
  columns that `node_curtailment` already reads. The scoring entry point does not change.
- **Reproduction check.** Pre-registered in the decision rule above.
- **Nodes.** Primaries that connect a farm (the farm-to-Primary map is a guess unless the
  mapping search settles it; the page states which) and the bulk supply points D13 and D17, with
  the labels D1 to D18 and farms A to F.
- **Compute.** CPU only, at most 4 cores, thread caps 1, `systemd-run --user --scope -p
  MemoryMax=16G -p OOMPolicy=continue`.
- **Reviews.** One Sonnet code review of the `packages/studies` change, one Sonnet review of the
  write-up against the printed report, one Sonnet prose review. No Opus.
