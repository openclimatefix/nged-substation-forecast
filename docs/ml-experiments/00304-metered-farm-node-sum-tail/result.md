---
id: '00304'
slug: metered-farm-node-sum-tail
title: >-
  How wide is the tail of the summed output of the solar farms under one bulk supply point?
theme: metered-solar-farms
also_relevant_to:
- curtailment-cost
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private joined node frames and tables under
  data/ml-experiments/00304-metered-farm-node-sum-tail/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to form the quantiles of a
  sum, so all four are arms. Callers: this folder's scripts and one new module,
  studies.quantile_sums, which only these scripts import. Asset graph: not touched.
hypothesis: >-
  For the solar farms that share a bulk supply point, adding each farm's quantiles overstates the
  upper tail of the farms' summed output, adding independent bands understates it, and summing the
  51 ensemble members' farm forecasts member by member gives calibrated 0.95 and 0.99 quantiles,
  because the ensemble carries the farms' shared dependence on cloud.
prediction: >-
  Coverage of the node's summed output at the 0.99 quantile is above 0.995 for the sum of quantiles
  (a), below 0.985 for the independent sum (b), and within 0.5 points of 0.99 for the member-wise
  sum (c), and the same holds at 0.95 within 0.5 points. The member-wise sum (c) fails if its
  coverage at either level is more than 0.5 points from nominal.
decision_rule: >-
  Coverage at 0.95 and 0.99 on the node rows (daylight half-hours at 12 to 36 hour leads where all
  three farms of the node are present, 2025-09 to 2026-08), pooled over the two nodes, with
  intervals from resampling the 12 scored months (2,000 draws) at the 98.75% level (Bonferroni
  across four planned contrasts: the member-wise sum minus nominal at 0.95 and at 0.99, and the
  pinball loss of the member-wise sum minus that of the sum of quantiles and minus that of the
  independent sum, both at 0.99). The prediction for (c) is supported when both coverage intervals
  lie within 0.5 points of nominal. The threshold for porting a sum form is zero, because every form
  is a calculation on forecasts the service already makes. Stop rule: if under 2,000 daylight
  half-hours hold all three farms of a node, report that the node-level test cannot be run and stop.
  The curtailment consequence is scored only if the event census of experiment 255 reaches 20
  half-hours at the limit; it reaches 5 at 0.75 and 2 at 0.9 times the maximum export, so the
  coverage result stands alone.
benchmark: >-
  Nominal coverage, the sum of quantiles (a) as the upper bound of dependence, and the independent
  sum (b) as the lower bound; the farm marginals are experiment 270's spread-binned plant (P2),
  experiment 287's per-quantile blend, and experiment 287's XGBoost arm
row_set_id: >-
  Experiment 270's scored-year rows at 12 to 36 hour leads (daylight half-hours, sun more than 5
  degrees above the horizon, farms A to F, 2025-09 to 2026-08), limited to the half-hours where the
  three farms of a node are all present
metric: >-
  Coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99 of the node's summed output in MW, and the
  Pearson correlation of the farms' residuals by clear-sky band and lead
code: ml-experiments/00304-metered-farm-node-sum-tail/
---

**Draft plan; the write-up replaces this page when the experiment finishes.**

## Plan

- **Nodes.** The six farms sit beneath two bulk supply points, three farms each: {A, C, E} and
  {B, D, F}. The node's output is the sum of its farms' outputs in MW. A node's farms are the only
  members of the sum, so the other generators and the demand at the node are outside this test.
- **Marginals.** Each farm's 11 quantiles from experiment 270's P2, experiment 287's blend, and
  experiment 287's XGBoost arm (seed 0, own history), converted to MW with the farm's effective
  capacity.
- **Sum forms.** (a) the sum of the farms' quantiles at each level; (b) independent bands, drawn by
  sampling each farm's piecewise-linear quantile function with independent uniforms; (c) the
  ensemble's own dependence: the rank of each of the 51 members' plant response within a farm gives
  a uniform, the farm's quantile function maps it to MW, and the members are summed across the
  node's farms, with the 51 ranks widened to 2,040 draws by a uniform inside each rank's interval;
  (d) a Gaussian copula whose correlations between the node's farms are fitted on the residuals
  of the other 11 scored months, per clear-sky band (leave one month out).
- **Controls.** The members shuffled across days of the same hour (breaks the dependence, should
  match (b)), and a one-farm node (all four forms agree and the coverage is the farm's own).
- **Report.** Node-level coverage and pinball at 0.01, 0.05, 0.95, 0.99, and the Pearson correlation
  of farm residuals by clear-sky band and lead, with farm labels A to F only.
- **Weather at lag times.** The 00 UTC ECMWF ensemble run, available at forecast time; no past
  weather from ERA5 enters.
- **Schedule.** No refits: the stored seed-0 predictions of experiments 270 and 287 are read, so the
  schedule is whatever those experiments used.
