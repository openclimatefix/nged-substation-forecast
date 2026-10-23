---
id: '00304'
slug: metered-farm-node-sum-tail
title: >-
  How wide is the tail of the summed output of the solar farms under one bulk supply point?
theme: metered-solar-farms
also_relevant_to:
- curtailment-cost
- calibration-and-uncertainty
status: finished
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
verdict: not supported
verdict_note: >-
  The member-wise sum (c) over-covers at 0.95 by 1.8 points (plant) and 2.5 points (blend), so it
  misses the 0.5-point bar, and at 0.99 it is within the bar for the plant only. It differs from the
  sum of quantiles (a) by under 1 point. The fitted Gaussian copula (d) is the closest form.
best_arm: >-
  The Gaussian copula with a correlation fitted on the other 11 months (d), blend marginals, which has
  the lowest pinball loss at 0.99 of the arms (the plant marginals sit closer to nominal at both
  levels): 0.9648 coverage at the 0.95 quantile (98.75% interval 0.956 to 0.9724) and 0.9938 at the
  0.99 quantile (0.9904 to 0.9967), pooled over both nodes, 2025-09 to 2026-08.
headline: >-
  Summing the 51 ensemble members farm by farm gives almost the same band as adding the farms'
  quantiles, because the ensemble treats the three farms as nearly one farm (correlation 0.95 to
  0.97) while their forecast errors correlate 0.54 to 0.78. Its 0.95 quantile covers 96.8% (plant) to
  97.9% (XGBoost) of half-hours.
lessons: >-
  The ensemble's ranks overstate how closely the farms' errors move together, so the member-wise sum
  is as wide as assuming perfect dependence. Independent bands are too narrow by 2 to 5 points at
  0.95. A correlation fitted on past errors is the closest of the four forms tried, and a farm's own
  band already over-covers a little, which every sum inherits.
take_home_message: >-
  The band for the combined output of the three farms beneath one substation comes out too wide if
  the weather forecast's 51 members are summed member by member, because the weather forecast treats
  the farms as nearly one farm and their forecast errors are not that closely tied. The result is
  almost the same as adding each farm's quantiles. Assuming the farms' errors are unrelated is too
  narrow, and a correlation fitted on past errors comes closest to the nominal level. The window
  holds too few half-hours beyond a substation's limit to say what any of this means for curtailment
  cost.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Form the band of the summed output of farms beneath one substation from a Gaussian copula
    whose correlation is fitted on past forecast errors, by clear-sky band, not from the ensemble
    members or from independent bands (C1, Figure 1).
  - >-
    Do not sum the ensemble members' farm forecasts member by member for a band, because the
    members imply a correlation of 0.95 between farms against 0.54 to 0.78 in the forecast errors
    (Figure 2).
  - >-
    Do not treat the farms' errors as independent for a band on the summed output, because the 0.95
    quantile then covers only 90% to 93% of half-hours (Figure 1).
  - >-
    Fit the copula's correlation on at least a year of errors from the farms' own forecast bands
    and score it on months it was not fitted on, because it was fitted here on the other 11 months
    of the scored year (Caveats).
  - >-
    Check each farm's own band before summing, because a farm's 0.95 quantile covers 94.6% to 97.3%
    of half-hours and the sum inherits the excess (One-farm control).
---

**Summing the weather forecast's 51 members farm by farm gives a band almost as wide as adding the farms' quantiles, which is too wide, and a correlation fitted on past forecast errors gives the band closest to the nominal level.** National Grid Electricity Distribution (NGED)
meters six solar farms, labelled A to F on this page, beneath two bulk supply points (132 kV to 33 kV
substations): farms A, C, and E beneath "node 1" and B, D, and F beneath "node 2". Curtailment acts at
the substation, so the tail of the farms' summed output matters more than each farm's tail. Each farm
has a forecast band of 11 quantiles, from [experiment
270](../00270-metered-farm-xgboost-vs-plant-diagnostic/result.md) (the spread-binned power model,
"P2") and [experiment 287](../00287-metered-farm-spread-binned-plant-heads/result.md) (the
per-quantile blend, and [XGBoost (Chen and Guestrin, 2016)](https://doi.org/10.1145/2939672.2939785)).
We formed the substation's band four ways and counted how often the observed sum fell at or below
each quantile. The four ways add the farms' quantiles (a), draw each farm's level independently (b),
sum the 51 ensemble members' farm forecasts member by member (c), and draw the levels from a Gaussian
copula whose correlation is fitted on past errors (d).

**Verdict: not supported.** The prediction was that (c) lands within 0.5 points of the nominal level
at 0.95 and 0.99. It does not at 0.95. In the pooled rows, the 0.95 quantile of (c) covers 96.79% of
half-hours for P2 (98.75% interval from resampling whole months, 95.85% to 97.60%), 97.50% for the
blend (96.76% to 98.14%), and 97.89% for XGBoost (97.14% to 98.53%). The 0.99 quantile of (c) covers
99.27% for P2 (98.90% to 99.59%), within 0.5 points, and 99.64% for the blend (99.42% to 99.84%) and
99.75% for XGBoost, outside it. Adding independent bands (b) is too narrow, as predicted: 90.10%
(P2), 90.78% (blend), and 92.66% (XGBoost) at 0.95, and 95.97%, 96.62%, and 97.90% at 0.99. Adding
each farm's quantiles (a) is above 99.5% at 0.99 for the blend (99.70%) and XGBoost (99.82%), not for
P2 (99.34%), so that part of the prediction held for two of three arms.

![Figure 1: coverage of the node sum, every form and arm](figure_1_coverage.svg)

![Figure 2: correlation between farms implied by the ensemble and seen in the errors](figure_2_dependence.svg)

## Results

- **Rows.** 5,569 half-hours at node 1 and 6,536 at node 2 hold all three farms (daylight, sun more
  than 5 degrees above the horizon, 12 to 36 hours ahead, 2025-09 to 2026-08). The stop rule
  (under 2,000) did not fire. Node 2 holds a farm of 1.19 MW beside farms of 6.03 and 7.80 MW.
- **Why (c) is as wide as (a) (tested explanation).** The ensemble's ranks imply a mean correlation
  of 0.95 to 0.97 between the node's farms in every clear-sky band, while the farms' forecast errors
  correlate 0.54 to 0.78 (Figure 2; blend arm). Summing members therefore treats the three farms
  as nearly one farm, which is the perfect dependence that (a) assumes. The control agrees:
  shuffling the members across days (breaking the dependence) gives 90.02% at 0.95 and 95.35% at
  0.99 for P2, close to the independent form (b).
- **The fitted copula (d) is the closest form.** Coverage of (d) at 0.95 is 95.84% for P2 (94.94% to
  96.78%), 96.48% for the blend (95.60% to 97.24%), and 97.36% for XGBoost (96.48% to 98.12%). At 0.99
  it is 98.90% (P2), 99.38% (blend), and 99.56% (XGBoost). (d) scores lower pinball loss at 0.99 than
  (c) for the blend by 6.5% (interval -0.0070 to -0.0020 MW per half-hour, excluding zero) and for
  XGBoost by 15.6% (-0.0214 to -0.0122), and by 1.6% for P2 (-0.0034 to +0.0011, includes zero).
- **Lower tail.** At the 0.01 quantile the forms cover 0.3% (a), 5.5% to 5.9% (b), 0.4% to 0.5% (c),
  and 1.0% to 1.2% (d) of half-hours, nominal 1.0%. The sum of quantiles (a) and the member-wise
  sum (c) are far too wide below, as above.
- **Planned contrasts on pinball loss at 0.99 (MW per half-hour, 98.75% intervals).** (c) minus (a):
  P2 -0.0002 (-0.3%, -0.0009 to +0.0010), blend -0.0004 (-0.6%, -0.0011 to +0.0007), so the two
  forms score alike; for XGBoost -0.0037 (-3.5%, -0.0045 to -0.0027). (c) minus (b): P2 -0.0268
  (-27.2%, -0.0421 to -0.0107) and blend -0.0162 (-19.2%, -0.0306 to -0.0032), lower because the
  independent band is too narrow; for XGBoost +0.0116 (+12.6%, -0.0070 to +0.0304, includes zero).
- **Both nodes agree.** Coverage of (c) at 0.95 is 97.0% and 96.6% (P2), 97.7% and 97.4% (blend),
  and 98.0% and 97.8% (XGBoost), at node 1 and node 2. No form changes its ranking between the
  nodes.
- **Residual correlations** (blend arm, pairs of farms beneath one node, all bands): 0.45 to 0.83 at
  node 1 and 0.64 to 0.68 at node 2. Correlation is higher under overcast skies (0.68 to 0.95) than
  under clear skies (0.33 to 0.78), and differs by at most 0.08 between 12 to 24 and 24 to 36 hours
  ahead.
- **One-farm control** (blend arm, a farm's own band): the 0.95 quantile covers 94.6% to 97.3% and the
  0.99 quantile 98.8% to 99.6% of half-hours, so the marginals already over-cover a little, and
  every sum inherits that. The 0.01 quantile of the 1.19 MW farm D covers 12.1% and that of the
  17.6 MW farm E covers 5.6%, both too narrow below. Farm D's small share keeps its excess from
  reaching node 2's sum, whereas farm E's share is large and is consistent with the node 1 lower tail
  of the independent form (5.5% to 5.6%).
- **Curtailment cost not scored.** The event census of [experiment
  255](../00255-metered-farm-node-curtailment-value/result.md) holds 5 half-hours beyond 0.75 times
  each node's maximum export and 2 beyond 0.9, under the 20 the plan required, so the coverage
  result stands alone.

## Checks

- **Median minus mean forecast per farm** (blend arm, mean taken as the average of the stored
  quantiles weighted by level spacing, MW): A -0.018, C +0.012, E +0.103, B +0.030, D +0.004,
  F +0.030, against capacities of 5.96, 7.88, 17.6, 7.80, 1.19, and 6.03 MW. The largest is 0.6% of
  capacity, so no spread shifts a forecast.
- **Error of the median forecast beside the pinball contrasts.** The mean absolute error of the sum's
  median is 2.088 (a), 2.079 (b), 2.090 (c), and 2.078 (d) MW for P2, with overlapping intervals,
  so the contrasts on the tail do not come from a shifted median.
- **Audit triggers.** Trigger 2 fired once: (c) over-covers against the prediction. The explanation
  above (Figure 2) was tested in this session, so no audit ran. Triggers 1, 3, 4, 5, 6, and 7 did not
  fire: the shuffled-members control behaves as a control should, the quantile and median errors do
  not disagree, neither node flips a verdict, no nested arm beats its container, there is no sister
  experiment on the same rows, and no later experiment depends on this result yet.

## Caveats

- **The copula (d) used future months.** Its correlation for each month comes from the other 11
  scored months, because the stored predictions cover only the scored year. A production copula would
  use earlier months only.
- **(d) does not use the ensemble.** The plan described (d) as a correction of the member-wise sum;
  it is a Gaussian copula whose correlation is fitted by clear-sky band, and the ensemble's ranks play
  no part in it.
- **Two nodes and six farms.** The intervals resample 12 months, and the two nodes share the same
  weather forecasts, so the nodes are not independent evidence.
- **Marginals.** Farm bands come from experiments 270 and 287 at seed 0, own history. No refits ran,
  so the schedule is the one those experiments used.
- **Weather at lag times.** The ensemble members are from the 00 UTC run, available before the forecast
  is used. No ERA5 data enters.

## What the next experiment should answer

- Does a copula with correlation fitted on earlier months only keep its coverage within 1 point of
  nominal at 0.95 and 0.99 on months it has not seen?
- Does recalibrating each farm's own band, which over-covers slightly, bring the sum closer to
  nominal than the choice of copula does?
