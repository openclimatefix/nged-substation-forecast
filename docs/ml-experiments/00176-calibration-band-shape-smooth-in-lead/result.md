---
id: '00176'
slug: calibration-band-shape-smooth-in-lead
part: CAL-4
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
status: planned
size: >-
  simple. Stored: private files under
  data/ml-experiments/00176-calibration-band-shape-smooth-in-lead/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: the four arms and the shrinkage constant's selection rule
  are pre-registered below, so no choice of design is left open. Callers: this folder's scripts,
  which read experiment 169's stored forecasts and experiment 160's stored spread tables and call
  the reviewed `studies.*` modules without changing them. No refit of the mean forecast.
hypothesis: >-
  The q05 to q95 band of the combined forecast (experiment 169's arm A) is not smooth along the lead
  because each cell of the spread table (hour band, calendar quarter, one of four lead bands) is a
  quantile of a few hundred correlated residuals, and neighbouring cells are fitted independently.
  Replacing the table with offsets that are a non-decreasing piecewise-linear function of lead
  (fix P), shrinking each cell towards the table pooled across substations (fix S), or both (fix
  PS), gives a band whose median width never falls from one lead day to the next, without worsening
  the quantile score or the cold-night coverage.
prediction: >-
  Written before any score is read. On the scored rows (outside the switching events) arm A already
  has D5 median widths that rise with lead day (19.7, 20.8, 23.9, 24.6, 25.4, 26.6, 27.8 points),
  so the staircase the brief describes is expected to appear on unmasked rows only, and to come
  from the median of a mixture of narrow and wide cells moving with the row mix. Fix P and PS leave
  every substation's all-row quantile score within -0.005 to +0.005 points. Fix S moves it by
  less than 0.003. Cold-night q95 coverage stays within 0.01 of arm A's 0.896.
decision_rule: >-
  The reference is arm A (experiment 169's arm A: experiment 160's interpolated table plus
  experiment 164's fix O). Every arm keeps the mean forecast and the fix-O multipliers, and changes
  only the base offsets. All coefficients are fitted on the tuning months and nothing on the
  replicate months or the scored year. An arm passes when, in the scored year, (1) D5's median q05
  to q95 width by lead day is non-decreasing to within 3% (each lead day is at least 0.97 times the
  day before) both on rows outside the switching events and on all rows, (2) for each of the six
  substations the upper bound of the paired month-resampled 95% interval of the all-row quantile
  score difference from arm A is at or under +0.005 points (20,000 resamples), and (3) the
  cold-night q95 coverage pooled over the five decision substations is at least 0.89. The
  hypothesis is `supported` when an arm passes on the scored year and on the replicate months,
  `partly supported` when it passes on the scored year only, `refuted` when no arm passes (1)
  because the width stays unsmooth, and `inconclusive` otherwise. The threshold is set by the
  effort of porting: each fix changes how the stored table is built and read at prediction time and
  adds no data source, so a change whose upper bound stays at or under +0.005 points is worth
  porting. The verdict is capped at `inconclusive` if arm A rescored from the stored forecasts
  differs from experiment 169's printed 2.7519 by more than 0.001 points.
benchmark: >-
  Arm A (2.7519 points over leads 0.5 to 168 hours in the scored year), BG2, climatology, and
  manual_heuristic as stored by experiment 87, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours; median and mean q05 to q95 width by lead day, with the largest fall between adjacent days;
  coverage at q05, q95, and q99 by lead day; cold-night q95 coverage; pinball loss at the 0.01,
  0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00176-calibration-band-shape-smooth-in-lead/
---

# Experiment 176: a band width that rises smoothly with lead

Planned; the write-up replaces this page when the run finishes.

## Plan

- **Diagnosis.** Print D5's spread table, and the median q05 to q95 width by lead day (day = `ceil(lead / 24) - 1`) for
  the five decision substations and D9, on rows outside the switching events and on all rows.
  Day 7 of the brief's list holds only the rows with a lead of exactly 168.0 hours, because
  `floor(lead / 24)` puts them alone in an eighth day, so the plan bins with `ceil`.
- **Fix P.** For each substation, hour band, and quantile level, offset = A(cell) times h(lead),
  where h is non-decreasing and piecewise linear with knots every 24 hours, fitted by linear
  programme on the pinball loss, pooled across the six substations within an hour band, and A(cell)
  is the substation's own scale for the cell, refitted by weighted quantile given h. Median offsets stay zero.
- **Fix S.** Each cell of experiment 160's table is shrunk towards the table pooled across
  substations (normalised by each substation's mean width) with weight n / (n + k), where n counts
  the cell's independent forecast runs and k is chosen from {0, 5, 20, 80} by three-fold month-block
  cross-validation inside the tuning months.
- **Fix PS.** Fix P, then fix S applied to its cell scales A.
- **Fix O** (experiment 164) is reapplied on top of every arm as its stored multipliers.
