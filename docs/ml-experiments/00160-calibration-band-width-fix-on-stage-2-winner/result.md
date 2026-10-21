---
id: '00160'
slug: calibration-band-width-fix-on-stage-2-winner
part: CAL-1
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
- curtailment-cost
status: planned
size: >-
  simple. Stored: private files under
  data/ml-experiments/00160-calibration-band-width-fix-on-stage-2-winner/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: three variants are pre-registered side by side
  (interpolate, force non-decreasing, both), so no choice of design is left open. Callers: this
  folder's scripts, which read experiment 155's stored forecasts and call the reviewed `studies.*`
  modules without changing them. No new modelling and no refit of the mean.
hypothesis: >-
  The q05 to q95 band of the stage 2 winning arm (experiment 155's arm D with arm A's spread table)
  widens in steps along the lead, because the spread table has only four lead bands (up to 24 hours,
  48, 144, and 168), fitted separately per hour band and calendar quarter. Interpolating each hour
  band by quarter cell between the lead-band centres, forcing the width to be non-decreasing in
  lead, or both, removes the steps without worsening the quantile score.
prediction: >-
  Written before any score is read. Experiment 136 measured identical median widths on days 3 to 6
  of the lead (D5 25 points of effective capacity, D16 14), steps at 24, 48, and 144 hours, and q95
  coverage of 0.91 to 0.98 by lead day. Arm D is expected to show the same staircase: at the five
  decision substations at least 3 pairs of adjacent lead days (out of 6) have median widths equal to
  within 0.5% at D5 and D16. The interpolated and combined variants leave at most 1 such pair at
  every decision substation. The quantile score changes by -0.005 to +0.005 points over leads 0.5 to
  168 hours for every variant, because the tables are the same and only their placement along the
  lead moves. The cold-night q95 coverage stays near 83% in every variant (experiment 155 found
  82.7% to 84.3%), because the fix does not touch the mean.
decision_rule: >-
  The reference is arm D with arm A's spread table, as stored (`d_spread_a` in experiment 155). The
  contrasts are C1 (interpolated minus reference), C2 (non-decreasing minus reference), and C3 (both
  minus reference), in quantile score over leads 0.5 to 168 hours, at the five decision substations
  D5, D10, D11, D14, and D16 in the scored year (D9 beside them, in no contrast), read as a paired
  month-resampled bootstrap at the 95% level with 20,000 resamples. A variant passes when the upper
  bound of its interval is under +0.01 points and the staircase is gone. The staircase is gone when
  the year-median q05 to q95 width has at most 1 pair of adjacent lead days (day 0 is 0 to 24
  hours, day 6 is 144 to 168 hours) equal to within 0.5% at every decision substation, and the
  largest step between adjacent lead days, as a share of the substation's all-lead median width,
  does not exceed the reference's at the median across the five substations. The hypothesis is
  `supported` when at least one variant passes on both the scored year and the replicate months,
  `partly supported` when one passes on the scored year only, `refuted` when every variant's upper
  bound is at or above +0.01 points on the scored year, and `inconclusive` otherwise. The winning
  variant is the passing variant with the lowest all-lead score, with the earlier of C1, C2, and C3
  winning ties within 0.002 points. The threshold of +0.01 is set by cost: the change adds no
  fitted term and no stored state, only a different way of reading the existing spread table at
  prediction time, so it is worth porting at any score change whose upper bound stays under 0.01
  points (under 0.5% of the score). The table is fitted on the tuning months and nothing is fitted on
  the scored year. The verdict is capped at `inconclusive` if the reproduction control (the
  reference rescored from the stored forecasts of experiment 155) differs from experiment 155's
  printed arm D score by more than 0.001 points.
benchmark: >-
  The reference (arm D with arm A's spread; experiment 155 scored arm A at 2.791 and arm D, with its
  own refitted spread, at 2.756 points, and arm A's spread scored 0.002 to 0.005 better than D's
  own); BG2, climatology, and manual_heuristic as stored
  by experiment 87, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours; median q05 to q95 width, q05 to q95 coverage, and q95 coverage by lead day; coverage and
  pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles by hour band and on cold nights (day mean
  temperature below 2 C); the absolute lead-to-lead change in median width; and a scaled-limit
  curtailment-cost proxy
code: ml-experiments/00160-calibration-band-width-fix-on-stage-2-winner/
production_status: not-for-production
adopted_in: none
---

# Experiment 160: does a smooth spread table remove the width staircase?

Plan only; the write-up follows the runs.

## Plan

**Question.** Experiment 136 found that the q05 to q95 band of its stage 1 assembly widens in steps
along the lead. Experiment 155 assembled the stage 2 winner (arm D: the winter night offset and
recipe R on top of experiment 136's A2, with arm A's spread table). This experiment first measures
whether arm D's band is smooth and calibrated by lead day, then changes only how the stored spread
table is read along the lead. It does not refit the mean.

**Start arm.** The maintainer's brief names arm D. Experiment 155's own rule picked arm C (recipe R
alone, 2.754) because C and D tied within 0.005 points and C comes earlier in the order. D is kept
because the winter night offset also shifts the cold-night quantiles, which the tail checks here
read. The band-width fix acts on the spread and not on the mean, so the result is expected to carry
over to arm C.

**Spread table.** Each row's quantiles are its mean plus an offset read from a table indexed by hour
band (4), calendar quarter (4), and lead band (4: up to 24, 48, 144, and 168 hours). Every offset
set has median zero. The experiment recovers the table from the stored forecasts (quantile minus
mean in the tuning rows) and checks that each cell holds one value.

**Variants.** The centres of the four lead bands are 12, 36, 96, and 156 hours.

- Reference: the stored forecasts, each row reading its own lead band's offsets.
- I (C1): each offset is linearly interpolated in lead between the lead-band centres of its hour band
  by quarter cell, and held flat below 12 and above 156 hours.
- M (C2): along the four lead bands of each cell, the upper offsets (q60 to q99) are replaced by
  their running maximum and the lower offsets (q01 to q40) by their running minimum, so the width
  never falls with lead; the lookup stays the stored one.
- IM (C3): I applied to the M table.

Nothing in the three variants is fitted on the scored year, and the mean is the reference's.

**Measurement (task 1, before any variant is scored).** For the reference at each decision
substation: median q05 to q95 width, q05 to q95 coverage, and q95 coverage by lead day; q95 and q99
coverage by hour band; q95 and q99 coverage on cold nights (day mean temperature below 2 C, winter
nights as in experiment 155); the pairs of adjacent lead days whose medians are equal.

**Scoring.** Quantile score, coverage by lead day, tail coverage and pinball loss at the 0.01, 0.05,
0.95, and 0.99 quantiles, the cold-night q95 coverage, width smoothness (mean absolute lead-to-lead
change in median width), and the curtailment-cost proxy: the node-only rule of
`studies.node_curtailment` on each decision substation's demand series, with the limit `L` set to 0.9
times the series' maximum before the scored year, at a price of GBP 2,000 per MWh, leads 12 to 36
hours, reported beside the manual heuristic and the perfect forecast. A Primary substation has no
published export limit in the project, so the proxy shows the tail arithmetic and cannot rank the
variants.

**Advice followed.** Experiment 136's width check (its candidate fix is the one tested here) and
experiment 155's advice to start calibration work from arm D with arm A's spread table. No departures.

**Checks.** The write-up carries the median forecast minus the mean forecast per site, the
mean-forecast error beside each quantile-score contrast, and the audit triggers.
