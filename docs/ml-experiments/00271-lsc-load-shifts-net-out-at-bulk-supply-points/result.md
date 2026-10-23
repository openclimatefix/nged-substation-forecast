---
id: '00271'
slug: lsc-load-shifts-net-out-at-bulk-supply-points
title: >-
  Do load shifts net out at the bulk supply points in the curtailment metric?
theme: lsc-detectors
also_relevant_to:
- curtailment-cost
status: planned
size: >-
  medium. Stored: private world tables under
  data/ml-experiments/00271-lsc-load-shifts-net-out-at-bulk-supply-points/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one defensible way to push an
  injected load shift up to a bulk supply point, so the bottom-up delta design is fixed below before
  any score is read. Callers: this folder's scripts, copied from experiments 255 and 258, and the
  unchanged studies.node_curtailment. Asset graph: not touched.
hypothesis: >-
  A load shift between two Primary substations under the same bulk supply point (BSP) leaves the
  power flow at the BSP unchanged, so it barely changes the curtailment cost at the BSP, and only
  a load shift that crosses from one BSP to the other changes the cost noticeably. The step 7
  program brief for the load-shift (LSC) decision program asks this in order to know whether the
  money question can be asked at BSP level.
prediction: >-
  Written before any score is read. On injected worlds, at 0.5 times each BSP's earlier maximum
  export, observed-power target, in the filter forecast without knowledge of load shifts (K0),
  load shifts within a BSP change the avoidable cost by under 1% and load shifts across the BSPs
  change it by under 5%. The prediction fails if the upper end of the 97.5% interval of either
  change exceeds its threshold.
decision_rule: >-
  The change is the avoidable cost in a world with injected load shifts minus the avoidable cost in
  the real record, divided by the real record's, for the same arm, on the causal rows of
  experiment 255 (valid times 2025-09-01 to 2026-06-30, scored at 0.5 times the maximum export
  before 2025-09-01, the limit fixed from real flows in every world), summed over the two BSPs.
  The headline is the event-targeted placement of load shifts (each ends 6 to 72 hours before a day
  with scored events at the BSP), because randomly placed load shifts cover a small share of the
  event half-hours and would confirm the thresholds by dilution; the randomly placed worlds are
  reported beside it as the typical year. C1: K0, within-BSP, targeted; C2: K0, cross-BSP, targeted.
  Each interval is a 97.5% percentile interval from resampling whole worlds (Bonferroni over two
  contrasts, 20,000 draws). Supported when both upper ends are under their thresholds in absolute
  value; partly supported when both hold for random placement only or only one holds for targeted
  placement; not supported otherwise. The thresholds are the program brief's, written before any
  score. The other arms (GAM_ONLY, PERFECT, PERFECT_LAG, a stand-in for the carried mask) are
  descriptive. Stop rule: if a world holds no scored event half-hour inside the placement
  window, the page says so and reports the event census only.
benchmark: >-
  The real record scored by the same rule (no injection), and experiment 255's avoidable cost at
  0.5 times the maximum export.
row_set_id: >-
  The causal rows of experiment 255 (experiment 48's node forecast origins at 09 UTC, 12 to 36 hour
  leads, valid times 2025-09-01 to 2026-06-30, less rows whose daylight farm forecast comes from a
  later run than the node origin)
metric: >-
  Avoidable cost in GBP at GBP 2,000 per MWh (energy proxy, Curtailment Limit 0) at 0.5 times each
  BSP's earlier maximum export, with the mean absolute error and the quantile score of the node
  forecast by row group G0 to G5
code: ml-experiments/00271-lsc-load-shifts-net-out-at-bulk-supply-points/
production_status: not-for-production
adopted_in: none
---

**Plan (written before any score is read).** Load shifts are demand that the network operator moves
between neighbouring Primary substations. Two things matter for the curtailment metric at a BSP:
whether the two Primaries sit under the same BSP, and whether the forecast carries the shift
forward after it ends.

- **Stage 1 (count, done before any injection).** Of the 31 logged load-shift pairs, 22 join two
  Primaries under the same BSP, 4 cross the BSP boundary (D15-D2, D12-D2, D10-D12, D11-D12), and 5
  have an unknown partner. Of the 66 hand-filtered real events, 54 involve two Primaries and 4 of
  those cross (all D10-D12), of which 1 lasts over 1 day (of 28 two-Primary events over 1 day). Of
  the 54 Primary-Primary interconnections, 33 join Primaries under the same BSP, 5 cross, and 16
  touch a substation outside the 30 listed in `Substations.csv`. Real load shifts therefore almost
  never cross a BSP boundary, so a BSP-level metric is nearly blind to them.
- **Design.** The 15 Primary series of experiment 258's harness cover 7 of 7 children of D13 and
  8 of 9 children of D17. A world injects rectangular shifts (moved power 15%, 30%, or 45% of the
  donor's mean load, lasting 1, 2, 4, or 8 days) conserved in MW between a donor and a receiver
  that are logged or interconnected pairs. Each arm's forecast of each affected Primary is rerun
  on the injected readings with the harness. The BSP forecast of an arm is experiment 48's node
  forecast plus the summed change, over the affected Primaries, of the arm's correction between the
  world and the real record, so only the injected shifts move a forecast. The BSP's observed power
  changes only if the shift crosses the boundary; the true normal-running-arrangement target is
  the real record.
- **Arms.** K0 (Kalman filter on recent power, no knowledge), GAM_ONLY (no recent power),
  PERFECT (the shift's true span masked, known at once), PERFECT_LAG (known after the live
  service's delays), and LB20_LIKE (PERFECT_LAG on half of the shifts drawn at random, a stand-in
  for the carried LR_b20 mask's recall because that mask has no alarms on injected readings). The
  carried LR_b20 mask itself is scored on the real record only, as a descriptive row.
- **Reviews.** Sonnet code review, numbers review, and prose review. No Opus.
