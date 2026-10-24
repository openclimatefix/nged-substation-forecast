---
id: '00332'
slug: lsc-curtailment-cost-of-label-free-mask
title: >-
  Does masking load shifts found by a label-free detector lower curtailment cost?
theme: lsc-detectors
also_relevant_to:
  - curtailment-cost
status: finished
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
verdict: not supported
best_arm: >-
  None. Against observed power the label-free mask (arm A) lowers cost during and after shifts by
  GBP 39 per half-hour (3.0%), with a 95% interval from -310 to +392 that includes zero.
headline: >-
  Against observed power, masking the load shifts a label-free detector finds does not lower
  curtailment cost during and after shifts (-3.0%, interval -24% to +30%) and raises it by 1.3% on
  ordinary half-hours. Against the true normal-running-arrangement series it lowers cost by 32.8%
  (interval -53% to -17%).
lessons: >-
  The quantile-score gain of experiment 327 shows up as curtailment cost only against the true
  normal-running-arrangement series. Observed power during a shift includes the moved megawatts,
  which a forecast of the normal arrangement cannot predict, so the mask raises cost during a shift
  (+36%) and nothing offsets that in the after window.
take_home_message: >-
  Masking the load shifts that a label-free detector finds lowers curtailment cost by about a third
  when the forecast is judged against the normal running arrangement. Judged against the power the
  network actually carried, which is what a curtailment instruction responds to, the mask changes
  cost by -3% with an interval from -24% to +30%, and it costs 1.3% more on ordinary half-hours.
  The detector therefore stays an offline tool for labelling load shifts, and the live service
  needs the moved megawatts as well as the mask before masking can lower cost.
advice:
  - >-
    Score a masked forecast of the normal running arrangement against both the true series and
    observed power, and headline the one the decision depends on, because the mask changed cost by
    -32.8% against the true series and by -3.0% (interval -24% to +30%) against observed power
    (figure 1).
  - >-
    Do not propose the label-free mask as a research-default change for curtailment on its own,
    because its cost against observed power during and after shifts has an interval that includes
    zero (deciding contrast).
  - >-
    Test a detector together with an estimate of the moved megawatts, because the true spans with
    the true megawatts added back lowered during-shift cost by 17.1% (interval -29.7% to -3.7%)
    while the label-free mask with them added back did not (figure 2).
  - >-
    Read curtailment cost at the 0.5 limit for Primaries, because at 0.75 the metric holds a cost
    of GBP 16 per half-hour and at 0.9 none (limit sensitivity table).
  - >-
    Expect the real record at a bulk supply point to resolve nothing, because the smallest
    resolvable difference at D13 is GBP 139,000 over six months against an observed 56,000 (real
    record bullets).
production_status: not-for-production
adopted_in: none
---

**Masking the load shifts that a label-free detector finds lowers curtailment cost by about a third
when the forecast is judged against the normal running arrangement, but not when it is judged against
the power the network carried.** Against observed power, the mask changes cost during and after
shifts by -3.0% (95% interval -24% to +30%) and raises it by 1.3% on ordinary half-hours. The
detector therefore stays an offline tool for labelling load shifts. A live service would also need
the moved megawatts before masking could lower cost. Verdict: not supported.

We asked whether the quantile-score gain that experiment
[327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md) found for its label-free
detect-and-mask pipeline also shows up as lower curtailment cost. Curtailment cost is the cost of
cutting generation because a forecast let a substation's flow pass a limit, as defined by experiment
[268](../00268-lsc-curtailment-metric-v2-member-wise-ens/result.md). We scored experiment 327's
forecasts on 44 worlds in which transfers of load between six Primary substations had been injected,
and on the real record at one bulk supply point (D13) with its metered solar farms. A Primary has no
published limit, so the injected-world costs rank forecasts and state no money saved.

![Figure 1. Arm A minus arm C by row group](figure_1_a_minus_c_by_group.svg)

![Figure 2. Contrasts against no mask, observed power](figure_2_contrasts_observed.svg)

![Figure 3. Arm A minus arm C at each Primary](figure_3_per_primary.svg)

![Figure 4. The real record at D13](figure_4_real_record.svg)

## Arms, rows, and data

- **Arms.** C is no mask. A masks the readings a label-free detector flags. B masks with a plain
  24 hour residual detector. D masks the true (logged and labelled) spans, as a reference. E is A
  with a second detection pass. All five are experiment 327's, with its filter hyperparameters and
  quantile offsets unchanged. The filter was re-run to rebuild the quantiles, which 327 does not
  store.
- **Injected worlds (Part 1).** Experiment 327's `inj` folder: 44 worlds at six Primaries, even
  calendar months, 12 to 36 hour leads, arms C, A, B, and D only. The limit is f times the Primary's
  largest real observed power before 2024-08-04 (before any scored row), the same for every arm
  and for both targets.
- **Real record (Part 2).** The bulk supply point D13 has 7 child Primaries, all with stored
  forecasts. Its node forecast is the sum of the 7 child forecasts plus quantile offsets fitted on
  odd months. Its generators are the metered farms A, C, and E and 2 other generators. Experiment
  243's quantile forecast for the farms and experiment 48's for the others are paired by valid
  time, which is a lab shortcut. D17 lacks one child forecast, so it is not scored.
- **Headline price** is GBP 2,000 per MWh-equivalent, breach ratio 20, Curtailment Limit 0 hours.
- **Not run.** The long-and-deep stratum of experiment 327 has too few rows at cost-bearing
  limits to score. Arm E has no injected-world rows, so it is scored on the real record only.

## Results

- **Deciding contrast (A minus C, during and after pooled, observed power, limit 0.5).** -GBP 39
  per half-hour (3.0% of C's GBP 1,291), 95% interval -310 to +392, from 10,149 half-hours. The
  smallest difference this design resolves is GBP 514. The interval includes zero, so the
  prediction fails.
- **Ordinary half-hours.** A minus C is +GBP 18.7 per half-hour (+1.3%, interval +16.6 to +21.2),
  above the 1% bound.
- **By group, observed power.** During: +295 (+36%, interval +75 to +641). After (48 hours): -213
  (-13.8%, interval -604 to +311). Before the alarm: +218 (+17.4%, interval +16 to +547).
- **Against the true series.** During and after pooled: -697 (-32.8%, interval -1,126 to -359).
  During: -1,118 (-35.5%). After: -477 (-29.9%). B minus C is -334 (-15.7%), D minus C is -986
  (-46.4%), and A minus B is -363 (interval -654 to -118).
- **Moved megawatts added back (observed power, as experiment 312's perfect-megawatts arm).** D with
  them: during -139 (-17.1%, interval -242 to -30). A with them: during +134 (interval -196 to
  +458), pooled -94 (interval -399 to +289). The true megawatts are used, so these are upper
  bounds.
- **Leave one Primary out (deciding contrast).** Without D11: -117 (interval -223 to +2). Without
  D16: +46 (interval -301 to +599). Per Primary, D14 (-362, interval -654 to -66) and D16 (-282,
  interval -476 to -101) gain, and none of the other four has an interval below zero.
- **Winter (November to March), observed power.** During: +339 (interval +75 to +613, 176
  half-hours). After: -134 (interval -389 to +113, 596 half-hours).
- **Limit sensitivity (observed power, A minus C, during and after).** At 0.75 the base cost is
  GBP 16 per half-hour and the difference is +17.5 (interval -4.9 to +64). At 0.9 no cost arises.
  Neither is evidence.
- **Real record at D13, ordinary half-hours, limit 0.5** (12,544 half-hours, 9,344 rows excluded
  because a child's span or label masks the target). Cost: C GBP 6.41 million, A 6.47 million, B
  6.41 million, D 6.38 million, perfect forecast 0.45 million. A minus C is +GBP 56,300 (+0.9%,
  interval 0 to +160,494) against a smallest resolvable difference of GBP 138,727. B minus C is
  +3,349 (interval -8,258 to +18,306). Cost arises in April and June only. The after-shift groups
  hold no rows in the farm window.
- **Tails.** Quantile q01 and q99 coverage were not recomputed here. Experiment 327 reports them.

## Checks

- **Median minus mean forecast per site, and the mean-forecast error beside each contrast.** Not
  applicable. No forecast was trained, and the stored mean-forecast errors are experiment 327's.
- **Triggers fired.** Trigger 1 (the true-span arm D loses to C during a shift on observed power)
  and trigger 2 (the result contradicts its own prediction): both are explained by the moved
  megawatts, and the add-back arm tests the explanation. Trigger 4 (leaving one Primary out flips
  the sign of the point estimate: without D16 it is +46): the interval includes zero both with and
  without D16, so the verdict does not change. The fresh Sonnet audit that these triggers call for
  ran in place of an Opus audit, on the maintainer's instruction that this run use no Opus. The
  audit re-derived the deciding contrast from the raw files with its own code: -GBP 38.9 per
  half-hour (interval -309 to +388), ordinary +18.68 (+1.27%), and both leave-one-out values. It found no error. It noted
  that a single world holds 22% to 41% of the pooled difference at D11 and D16, which is why the
  intervals resample whole worlds, and that the files hold both the earlier and the scored
  periods, both of which are used, as in experiment 327.
- **Weak node forecast.** At D13 the NGED baseline (generators assumed at capacity, the
  maintainer's unconfirmed understanding) costs GBP 5.89 million, below every demand arm. The node
  forecast, which ignores the farms, is the weak link, so the real-record levels are not money
  saved. Arm contrasts share that weakness.

## What to try next

- **Pair the mask with the moved megawatts.** The true spans with the true megawatts added back
  lowered during-shift cost, so a live service needs an estimate of the moved megawatts, which
  experiment 312 scored with oracle values.
- **Score the real record on a node that carries more shifts.** D13 has no after-shift rows in the
  farm window, so a longer farm forecast history is needed before the money question can be asked
  of real data.
