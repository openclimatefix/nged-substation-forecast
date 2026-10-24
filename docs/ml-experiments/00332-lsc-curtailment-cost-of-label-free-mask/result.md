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
  Fixed before any cost is read. Rows are experiment 327's, so the quantile frames come from re-running
  experiment 327's Kalman filter with each arm's spans set aside, using 327's stored demand
  forecasts, hyperparameters and offsets unchanged (nothing is refitted). The metric is
  `studies.node_curtailment` at the headline price of GBP 2,000 per MWh, breach ratio 20 (the 0.95
  quantile), Curtailment Limit 0 hours, and limits of f times the node's largest observed flow, with
  f = 0.5 pre-registered and 0.75 and 0.9 as sensitivity (0.9 holds few event half-hours and is
  not read as evidence). Part 1 is the injected worlds of experiment 297 (44 worlds, six
  Primaries), even calendar months, 12 to 36 hour leads, scored node-only with one node per
  Primary and the high tail of demand as the flow that can breach (export sign +1, as in experiment
  277). A Primary has no published limit, so Part 1 ranks arms and states no money saved. The limit
  of each Primary is f times the largest real observed power of that Primary before the earliest
  scored row, taken from the real record without injected shifts, and is the same for every arm and
  for both targets. Arms: K0 (no mask, called arm C below), A, B, PERFECT (the true spans masked,
  called arm D below), and a perfect forecast (every quantile equal to the target) that gives
  avoidable cost. Groups follow experiment 297: G0 all rows, G1 ordinary, G2 before alarm, G3
  during, G4 after, G5 near an event. Targets: observed power (headline) and the true
  normal-running-arrangement series (reported). The one deciding contrast is A minus C on
  avoidable cost per half-hour, G3 and G4 pooled, observed power, f = 0.5, with the 95% interval
  from resampling whole worlds (20,000 draws). The prediction holds when that interval is below
  zero and the point estimate of A minus C on G1 is within 1% of arm C's G1 cost per half-hour. The
  verdict is supported when both hold, partly supported when only the first holds, and not
  supported otherwise. The leave-one-Primary-out line, the other contrasts (B minus C, D minus C, A
  minus B), the other limits and groups, the winter months (November to March), and the true-series
  target are reported and cannot change the verdict. The leave-one-Primary-out line is printed
  beside the deciding contrast, because experiment 327 found that one Primary held most of its
  gain. The variant that adds the moved megawatts back (experiment 312) is computed for every run
  and is never the headline. Detectability is 2.8 times the bootstrap standard error of each
  paired difference, printed beside it. Part 2 is descriptive and cannot change the verdict. It
  scores the real record at the bulk supply point D13 (Skegness, 7 child Primaries, all with stored
  forecasts), with the metered solar farms A, C, and E as generators and the bulk supply point's own
  metered flow as the node flow. The node forecast is the sum of the 7 Primaries' arm forecasts plus
  quantile offsets fitted on the odd-month residual of that sum against the metered node flow, by
  half-hour of day and lead band, so the offsets absorb the farms' output and the mismatch between
  the node and its Primaries. The farms' forecasts (experiment 243's H5 quantile forecast, seed 0,
  paired by valid time, of which the metric reads the median) only set the allocation of the planned
  cut, are shared by every arm, and are a lab shortcut because their lead can be shorter than the
  demand lead. Part 2 covers the six even months with farm forecasts (2025-08 to 2026-06) on
  target rows outside every logged span and every arm's alarm spans, and the number of excluded
  rows is printed. It cannot show a saving during a shift, because those rows are excluded, and
  experiment 277 found that the real record resolves no difference. Boston (D17) lacks one child
  Primary's forecast, so it is not scored. As in experiment 327, offsets are fitted on odd months
  and scored on even months, so some offsets use later data than the rows they score. The
  long-and-deep stratum and arm E are reported for the real record where they have rows.
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

**Experiment 327 stores a demand forecast per Primary, not quantiles.** The quantile frames are
re-computed by running the Kalman filter with each arm's spans set aside (experiment 327's own
functions, unchanged), and are written under `data/ml-experiments/00332-*/`.
