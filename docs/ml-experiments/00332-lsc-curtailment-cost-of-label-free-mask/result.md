---
id: '00332'
slug: lsc-curtailment-cost-of-label-free-mask
title: >-
  Does masking load shifts found by a label-free detector lower curtailment cost?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
status: planned
date: 2026-10-24
kind: other
size: >-
  medium. (1) Stored: private score tables under
  data/ml-experiments/00332-lsc-curtailment-cost-of-label-free-mask/ only. (2) Serving path: not
  touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for turning
  the stored Primary forecasts into a bulk-supply-point forecast, so the choice is fixed below.
  (5) Callers: this folder's scripts only; they import the folders of experiments 255, 268, 277, 297,
  309, and 327 and call the reviewed `studies.node_curtailment` unchanged. The maintainer's brief
  limits the work to CPU, Sonnet only, one Sonnet plan review and one Sonnet code review.
hypothesis: >-
  Treating as missing the readings that a label-free load shift detector flags (arm A of experiment
  327) lowers curtailment cost during and after load shifts, against no mask (arm C), without
  raising it on ordinary half-hours.
prediction: >-
  Written before any cost is read. On the injected worlds of experiment 297 (44 worlds at six
  Primaries), at the 0.5 scaled limit and against observed power, arm A minus arm C avoidable cost
  is negative in the DURING and AFTER groups, with a 95% interval below zero, and arm A changes the
  ordinary half-hours (G1) by less than 1% of arm C's cost there. The prediction fails if the
  interval includes zero; the answer is then that the quantile-score gain of experiment 327 does not
  show up as curtailment cost on this record.
decision_rule: >-
  Fixed before any cost is read. Step 0 (stop rule) is answered below: the stored forecasts cover
  all 7 child Primaries of the bulk supply point D13 on the real record, so the stop rule does not
  fire. Rows: experiment 327's. Injected worlds: all rows of experiment 297's worlds, 12 to 36 hour
  leads, six Primaries, arms K0 (no mask, arm C), PERFECT (arm D), A, and B. Real record: even
  calendar months of experiment 258's harness (2025-08, 2025-10, 2025-12, 2026-02, 2026-04,
  2026-06, the months for which farm forecasts exist), on target rows outside every logged or
  labelled span and every arm's alarm spans, as in experiment 327. Metric: `studies.node_curtailment`
  at the headline price, breach ratio 20 (the tau = 0.95 quantile), Curtailment Limit 0, limits
  f = 0.5 (pre-registered), 0.75, and 0.9 (sensitivity) times the node's largest observed flow
  before the first scored half-hour. Part 1 (headline for the prediction): injected worlds at
  Primary level, node-only, one node per Primary with the high tail of demand as the flow that can
  breach (experiment 277's convention). A Primary has no published limit, so Part 1 is a relative
  comparison between arms and states no money saved. Part 2 (the money result): the real record at
  D13, a bulk supply point with 7 child Primaries and the metered farms A, C, and E, scored with
  generators. The D13 node flow is the bulk supply point's own metered flow. The node forecast is
  the sum of the 7 Primaries' arm forecasts (the location of experiment 327's filtered forecast),
  plus quantile offsets fitted on the aggregate residual (metered node flow minus that sum) of odd
  months by half-hour of day and lead band. All arms share the farms' forecasts (experiment 243's
  H5 quantile forecast, seed 0, of which the metric reads only the median) paired by valid time,
  which is a lab shortcut, and the other generators' forecasts of experiment 48. The Primary-level
  groups are G0 all
  rows, G1 ordinary, G2 BEFORE_ALARM, G3 DURING, G4 AFTER (48 hours), G5 within 7 days (the
  experiment 297 phase labels); the real record has G0, G1, G4, and G5 only. Contrasts: A minus C,
  B minus C, D minus C, and A minus B on avoidable cost (cost minus the perfect forecast's cost),
  injected worlds against observed power (headline) and against the true normal-running-arrangement
  series (reported), winter (November to March) separately. Intervals resample whole injected worlds
  (44) for Part 1 and whole calendar months for Part 2, 20,000 draws, 95%. The prediction is
  decided on the DURING and AFTER groups pooled (G3 plus G4 rows counted once) and on G1 at
  f = 0.5. Detectability: the smallest difference the metric resolves is 2.8 times the bootstrap
  standard error of the paired difference, printed beside each observed difference. If arm A loses
  on observed power in DURING, the variant that adds the moved megawatts back (experiment 312) is
  reported as a separate column, never as the headline. The threshold for porting is any
  difference whose 95% interval excludes zero in the predicted direction, because the pipeline stage
  already exists in experiment 327 and this experiment only measures what it costs or saves in
  money. Verdict: supported if both parts of the prediction hold; partly supported if the DURING and
  AFTER interval is below zero but G1 changes by 1% or more; not supported if the DURING and AFTER
  interval includes zero.
benchmark: >-
  Arm C (no mask) of experiment 327, scored by the same metric; arm D (the hand-labelled spans
  masked) as the oracle reference.
row_set_id: >-
  experiment 327's real-record rows (even months) and experiment 297's injected-world rows; not
  comparable with the leaderboard
metric: >-
  Avoidable curtailment cost (GBP at the headline price, relative at Primary level) at the 0.5
  scaled limit, 12 to 36 hour leads.
code: ml-experiments/00332-lsc-curtailment-cost-of-label-free-mask/
---

## Plan

**This page is the plan; the write-up replaces it when the experiment finishes.** The brief is
"Does masking load shifts found by a label-free detector lower curtailment cost?" Experiment
[327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md) scored the label-free
detect-and-mask pipeline on quantile scores only. This experiment scores the same forecasts by the
curtailment-cost metric of experiment
[268](../00268-lsc-curtailment-metric-v2-member-wise-ens/result.md).

**Step 0 (what the stored forecasts support).**

- **Bulk supply points.** D13 has 7 child Primaries and D17 has 9. Experiment 327's real record holds
  all 7 of D13's and 8 of D17's (one Primary of D17 has no stored forecast), so D13 is the only full
  node on the real record.
- **Injected worlds.** They cover six Primaries, which are not a full bulk supply point, so
  Part 1 is a relative comparison between arms.
- **Farms.** Forecasts for farms A, C, and E (D13's farms) from experiments 224 and 243 exist for
  2025-07 to 2026-06 only, so Part 2 covers the six even months in that window.

**Experiment 327 stores a demand forecast, not quantiles.** The quantiles come from the Kalman filter
with each arm's spans set aside, so the scripts re-run that filter (experiment 327's own functions,
unchanged) and write the quantile frames.
