---
id: '00176'
slug: calibration-band-shape-smooth-in-lead
part: CAL-4
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
status: done
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
verdict: inconclusive
best_arm: >-
  None passes all three rules. Arm S (the stored cells shrunk towards the table pooled across
  substations) scores 2.7325 against arm A's 2.7521 in the scored year (-0.0196, 95% interval
  -0.0343 to -0.0069) and has D5 median widths that never fall on either row set, but the gain is
  all D10's (-0.0884), D5 scores +0.0095 worse (replicate months +0.0205, +0.0045 to +0.0403), and
  cold-night q95 coverage falls to 0.8884
headline: >-
  D5's median q05 to q95 width already rises with lead on the rows the experiments score (19.7 to
  27.8 points); the dips and the day-7 jump come from including the switching-event rows and from
  binning the 282 rows at exactly 168.0 hours into an eighth day. Refitting a smooth growth in lead
  to the tuning residuals worsens the score out of sample (+0.0644, +0.0317 to +0.1023 in the
  replicate months), and shrinking the stored cells helps only D10.
lessons:
  - >-
    The staircase and dips the brief describes are in the median of a mixture of narrow and wide
    cells, not in the cells: on the scored rows D5's median width rises on every lead day, and only
    the all-rows median falls (17.6 on lead day 1 after 19.7 on lead day 0, and 19.6 on lead day 3
    after 23.9 on lead day 2).
  - >-
    A smooth growth refitted to six tuning months (arms PB, P, U) beats arm A in the tuning months
    (D10: 3.815 against 4.012) and loses out of sample (D10 replicate months: 3.701 against 3.535),
    so the refit overfits a six-month residual sample whose cells hold about 45 forecast runs each.
  - >-
    Shrinking each stored cell towards the pooled table with a weight of runs / (runs + 80) lowers
    the pooled score by 0.0196 in the scored year, but without D10 the change is -0.0015 (-0.0108 to
    +0.0060).
take_home_message: >-
  D5's forecast band already widens steadily with lead time on the rows we score. The dips we saw
  come from how the median was taken over rows that include switching events, so no smoothing of
  the band is needed to fix them. Refitting a smooth curve made the forecast worse on months it was
  not fitted on, and pulling noisy table cells towards the pooled table helped only one substation.
advice:
  - >-
    Report band width by lead day on the scored rows and per hour band, not as a median over all
    rows: the all-rows median of a mixture of narrow and wide cells moves by 3 to 4 points when
    switching events are included (diagnosis bullet).
  - >-
    Bin lead days with ceil(lead / 24) - 1, so the rows at exactly 168.0 hours stay in day 6
    (diagnosis bullet).
  - >-
    Do not refit the spread table's growth in lead from the six tuning months: arms P, PB, and U
    score +0.005 to +0.009 worse in the scored year and +0.063 to +0.069 worse in the replicate
    months (C1).
  - >-
    If cell noise matters for D10, test shrinking its stored spread cells towards a pooled table on
    more substations before porting; the scored-year gain of -0.0196 is D10's alone (C3).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

# Experiment 176: does the band width need to be smoother in lead?

**D5's forecast band already widens with lead time on the rows we score, so the dips seen on D5
come from the row set and the binning, not from the spread table; smoothing the table did not
help.** The forecast is arm A of [experiment
169](../00169-calibration-combined-fixes-and-ensemble-spread/result.md): the interpolated spread
table of [experiment 160](../00160-calibration-band-width-fix-on-stage-2-winner/result.md) plus the
cold-day widening of [experiment 164](../00164-calibration-cold-night-tail-under-coverage/result.md).
Its band is the q05 to q95 interval, and its width is the q95 minus the q05, in points of effective
capacity (percentage points of a substation's rated demand). A lead day counts from the forecast
time: day 0 is up to 24 hours ahead. A request to the experiment asked why D5's median width by
lead day peaks on day 3, dips, and jumps on day 7, and whether three fixes remove the shape. The
fixes are a growth in lead that is non-decreasing and piecewise linear (fix P), shrinking sparse
cells of the stored table towards the table pooled across the six substations (fix S), and both
(fix PS). We fitted every fix on the tuning months (the odd months of 2024-07 to 2025-06), and
scored it on the scored year (2025-07-02 to 2026-06-29) and on the replicate months (the even
months). The mean forecast is unchanged. No fix passes the pre-registered rule: the score gain of
shrinking the cells is all at D10, and the refits lose score on months they were not fitted on.

![D5 median width by lead day](figure_1_d5_width_by_lead_day.svg)

![Width by lead day at every substation](figure_2_width_all_substations.svg)

![Score contrasts](figure_3_score_contrasts.svg)

![q95 coverage by lead day](figure_4_q95_coverage_by_lead_day.svg)

- **Diagnosis.** On the rows the experiments score (outside the hand-filtered switching events),
  arm A's D5 median width by lead day is 19.7, 20.8, 23.9, 24.6, 25.4, 26.6, and 27.8 points, which
  never falls. On all rows it is 19.7, 17.6, 23.9, 19.6, 20.6, 21.5, and 22.3. The table is bimodal
  (in D5's stored cells hour bands 0 and 3 have q05 to q95 widths of 6 to 28 points, and bands 1
  and 2 of 20 to 51), so the median sits in the gap between the two groups and moves with the row
  mix. We did not identify which cells the switching rows sit in. The brief's "day 7" holds the 282
  rows at a lead of exactly 168.0 hours, all in hour band 1 with a median width of 37.6, because
  `floor(lead / 24)` puts them in an eighth day. Binning by `ceil(lead / 24) - 1` removes day 7. The
  other five substations have no fall larger than 3% from one lead day to the next on either row
  set (figure 2).
- **Arms.** A: the reference. PB: offset = scale(substation, hour band, quarter) times a growth
  curve per hour band, non-decreasing in lead with a knot every 24 hours, fitted on the pinball loss
  by linear programme (HiGHS) pooled over substations and quarters (the first attempt). P: one
  growth curve per hour band and quarter, pooled over substations, with a scale for each
  substation, hour band, and quarter (the brief's design). U: a growth curve per substation, hour
  band, and quarter (no pooling). S: each cell of the stored table shrunk towards the pooled table,
  weight runs / (runs + k) where runs counts the cell's forecast runs and k = 80 was chosen by
  three-fold cross-validation across alternate tuning months (held-out pinball loss 23.50, 21.89,
  21.53, 21.44, 21.45, and 21.46 for k = 0, 5, 20, 80, 320, and 1280). PS: P with its cell scales
  shrunk the same way. Every arm keeps the mean, the offsets at q40, q50, and q60, and experiment
  164's fix-O multipliers (1.65, 1.36, 1.26, and 1.16 at q75, q90, q95, and q99 on forecast-cold
  days). The grid of k began at 0 to 80 and was extended to 1280 after the first run chose 80, the
  edge of the grid; the choice stayed 80. PB, and the first fit's per-hour-band growth, were run
  before the brief's per-quarter design (P), and are reported because their scores match P's.
- **Row set and schedule.** `fold_a_daily_gap_v1`, the five decision substations D5, D10, D11, D14,
  and D16 (D9 beside, in no pooled contrast), leads 0.5 to 168 hours, rows outside the switching
  events, 464,363 rows over the five substations in the scored year. The mean forecasts are
  experiment 155's monthly
  refits, so no contrast depends on the retraining schedule. Arm A rebuilt from experiment 169's
  reference arm and the fix-O multipliers scores 2.7521 over the five substations against the
  printed 2.7519, inside the 0.001 cap.
- **Pre-registered rule, scored year.** (1) D5's median width non-decreasing to within 3% on both
  row sets: A passes on the scored rows and fails on all rows (day 3 is 0.82 of day 2); PB, P, and U
  also fail on all rows; S passes both (lowest ratio 0.995); PS fails on all rows (0.96). (2) Upper
  bound of every substation's contrast at or under +0.005: no arm passes (largest upper bound P
  +0.1008, S +0.0358, PS +0.0336). (3) Cold-night q95 coverage at or above 0.89: A 0.896, P 0.939,
  PS 0.907, and S 0.888, so S misses by 0.002. Replicate months: S's D5 contrast is +0.0205
  (+0.0045 to +0.0403) and A passes everything, as the reference.
- **Score contrasts against A, five substations pooled.** Scored year / replicate months. P:
  +0.0052 (-0.0141 to +0.0290) / +0.0644 (+0.0317 to +0.1023). S: -0.0196 (-0.0343 to -0.0069) /
  -0.0169 (-0.0366 to -0.0004). PS: -0.0215 (-0.0463 to +0.0016) / +0.0214 (-0.0135 to +0.0549). PB:
  +0.0053 / +0.0629. U: +0.0090 / +0.0692. Absolute scores, scored year: A 2.7521, PB 2.7574, P
  2.7573, U 2.7611, S 2.7325, PS 2.7306; replicate months: A 2.6821, PB 2.7449, P 2.7464, U 2.7513,
  S 2.6651, PS 2.7035.
- **Per substation, S against A, scored year.** D10 -0.0884 (-0.1719 to -0.0178), D14 -0.0100
  (-0.0166 to -0.0036), D5 +0.0095 (-0.0214 to +0.0358), D11 -0.0024, D16 -0.0019, D9 +0.0001.
  Leaving D10 out, S's pooled contrast is -0.0015 (-0.0108 to +0.0060) in the scored year and
  -0.0006 (-0.0100 to +0.0087) in the replicate months; leaving any other substation out keeps it at
  -0.022 to -0.026 (scored year).
- **Why the refits lose out of sample (tested).** In the tuning months P beats A (D10 3.815 against
  4.012, D5 3.091 against 3.107) because it is fitted on those residuals. In the replicate months
  it loses (D10 3.701 against 3.535, mostly at q75, 6.94 against 5.71 in pinball units), and its q25
  coverage there is 0.34 against 0.26. Experiment 155 found the same for refitting a spread table
  on the tuning months, so the stored table is a better base than a refit on six months.
- **Tail pinball loss, scored year, five substations (q01, q05, q95, q99).** A: 0.216, 0.714,
  0.745, 0.232. S: 0.217, 0.705, 0.732, 0.224. P: 0.209, 0.707, 0.745, 0.222. PS: 0.213, 0.703,
  0.733, 0.217. Cold-night q05 coverage: A 0.090, S 0.083, P 0.095, PS 0.091.
- **q95 coverage by lead day, scored year.** A: 0.946, 0.948, 0.951, 0.951, 0.952, 0.946, 0.943.
  S: 0.951, 0.953, 0.955, 0.954, 0.956, 0.949, 0.947.
- **Caveats.** The tuning months are six alternate months, so a cell holds about 45 forecast runs
  and the shrinkage constant rests on three folds. The brief's rule (2) asks for an upper bound of
  +0.005 at every substation, which month-resampled intervals of 12 months cannot meet unless the
  effect is near zero (half-widths are 0.01 to 0.03). Fix S shifts some quantiles on forecast-cold
  nights, where D5 holds few rows (cold-night coverage at D5 is 1.0 on few rows). The fit uses
  residuals of the corrected mean, not of the stored table's own tuning forecasts. D9 is in no
  contrast.
- **What to try next.** (1) Test shrinking stored cells on more substations, since D10 alone gives
  the gain. (2) Tell the substation cards to show the width by lead day on the scored rows, per
  hour band.

## Checks

- **Median minus mean forecast, per site.** Exactly 0.000 for every arm and decision substation.
- **Mean-forecast error beside each quantile-score contrast.** The mean absolute error of the mean
  forecast is 4.7096 in the scored year and 4.6207 in the replicate months for every arm, because
  the mean is unchanged.
- **Reproduction control.** Arm A rebuilt from the stored reference and fix-O multipliers differs
  from experiment 169's stored arm A by at most 1.5e-5 points in any quantile and scores 2.7521
  against the printed 2.7519.
- **Triggers.** Trigger 2 (a result contradicts its prediction) fired for P (predicted within
  +-0.005, replicate months +0.0644) and S (predicted under 0.003, scored year -0.0196); both are
  explained by the tests above (the overfit, and D10 alone). Trigger 4 (leaving one site out
  changes the verdict of a contrast) fired for S's pooled gain, which vanishes without D10; the
  experiment's verdict does not depend on it. No other trigger fired.
