---
id: '00164'
slug: calibration-cold-night-tail-under-coverage
part: CAL-2
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
- curtailment-cost
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00164-calibration-cold-night-tail-under-coverage/
  only. Serving path: not touched. Degradation rule: not touched. Designs: three fixes are
  pre-registered side by side, so no choice of design is left open. Callers: this folder's scripts,
  which read experiment 160's stored forecasts and experiment 111's stored ensemble temperatures and
  call the reviewed `studies.*` modules without changing them. No new modelling and no refit of the
  mean.
hypothesis: >-
  The q95 coverage of experiment 160's smooth-table forecast on cold nights (day mean temperature
  below 2 °C) stays near 85% because the spread table is too narrow when it is cold, not because the
  mean forecast is biased. Widening the quantiles on days the ensemble forecasts as cold, by a
  factor fitted on the tuning months, lifts cold-night q95 coverage to 90% or more without
  worsening the all-row score.
prediction: >-
  Written before any score is read. Diagnosis: the mean residual (actual minus mean forecast) on
  cold nights is positive but under 0.3 times the cold-night residual standard deviation, and the
  cold-night residual spread is 1.2 to 1.6 times the all-night spread in the same hour bands and
  lead bands, so the shortfall is mostly a width problem. Fixes, scored year at the five decision
  substations: cold-night q95 coverage reaches 0.89 to 0.93 for the temperature-class factor (T)
  and the one-sided factor (O), and 0.87 to 0.91 for the ensemble-spread factor (S), because S
  adds a single fitted term. The all-row quantile score changes by -0.01 to +0.01 points for every
  fix, because cold nights are about 4% of rows.
decision_rule: >-
  The reference is arm I of experiment 160 (the stored spread table interpolated between the
  lead-band centres), read from experiment 160's stored forecasts. The contrasts are C1 (T minus
  reference), C2 (S minus reference), and C3 (O minus reference), in quantile score over leads 0.5
  to 168 hours on all rows at the five decision substations D5, D10, D11, D14, and D16 in the scored
  year (D9 beside them, in no contrast), read as a paired month-resampled bootstrap with 20,000
  resamples at the 98.33% level (a Bonferroni correction across the three contrasts). A fix passes
  when its cold-night q95 coverage, pooled over the five substations in the scored year, is 0.90
  or more, and the upper bound of its all-row interval is under +0.01 points. Cold nights are
  winter months (November to March), the night and evening hour bands, and an ERA5 day mean below
  2 °C, the definition of experiment 160. The hypothesis is `supported` when at least one fix passes
  on both the scored year and the replicate months, `partly supported` when one passes on the
  scored year only, `refuted` when no fix reaches a cold-night q95 coverage of 0.90 in the scored
  year, and `inconclusive` otherwise (including a fix that reaches 0.90 but whose all-row upper
  bound is at or above +0.01). The winning fix is the passing fix with the lowest all-row
  quantile score, with the earlier of C1, C2, and C3 winning ties within 0.002 points. The
  threshold is set by cost: T adds 6 fitted multipliers per temperature class, O adds 4, and S adds one,
  stored beside the spread table, so each is worth porting at any all-row change whose upper bound
  stays under 0.01 points (under 0.5% of the score). Every multiplier and the ensemble-spread
  coefficient are fitted on the tuning months and nothing is fitted on the replicate months or the
  scored year. The gate (cold, or the temperature class) uses only the ensemble forecast, so it
  exists at forecast time. The verdict is capped at `inconclusive` if the reference rescored from
  experiment 160's stored forecasts differs from experiment 160's printed reference score of 2.7529
  by more than 0.001 points.
benchmark: >-
  The reference (experiment 160's arm I, 2.7529 points over leads 0.5 to 168 hours in the scored
  year); BG2, climatology, and manual_heuristic as stored by experiment 87, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours; coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles on cold nights and
  on all rows; the residual mean, residual spread, and coverage by temperature bin and hour band;
  the median q05 to q95 width by lead day; and a scaled-limit curtailment-cost proxy
code: ml-experiments/00164-calibration-cold-night-tail-under-coverage/
production_status: not-for-production
adopted_in: none
---

# Experiment 164: a wider spread on cold nights (planned)

**Plan, committed before any fix is run.** Experiment 160 found that the q95 of the smooth-table
forecast is exceeded on about 15% of cold-night rows (nominal 5%), and that smoothing the spread
does not move this. NGED's main use of the forecast is reducing curtailment, which depends on the
upper tail, so this experiment first asks whether the shortfall is a width problem or a bias
problem, then tests three fixes fitted on the tuning months.

**Method.** The forecast is arm I of experiment 160: arm D of experiment 155 with the interpolated
spread table. The mean forecast is never refitted. The cold gate for every fix is the experiment-111
ensemble's mean temperature, averaged over the 24 hours centred on the valid time (clipped to the
run's window). That value exists at forecast time. The tuning months are the odd months and the
replicate months the even months of the year before the scored year, as in experiment 155.

- **Diagnosis.** Residual mean, residual standard deviation, and q05 and q95 coverage by ensemble
  temperature bin (below 0, 0 to 2, 2 to 5, 5 to 8, 8 to 12, and above 12 °C) and by hour band,
  with the same breakdown for the ERA5-defined cold nights that the score uses.
- **T: temperature-class factor.** One multiplier on the quantile offsets at the six outer levels (0.01, 0.05, 0.1, 0.9, 0.95, and 0.99) per
  ensemble-temperature class (below 2, 2 to 6, 6 to 10, and 10 °C or more), pooled over the six
  substations. Each multiplier is the empirical quantile of residual over offset that makes the
  tuning coverage nominal, and is kept at 1 where a class has fewer than 500 tuning rows.
- **S: ensemble-spread factor.** On rows with an ensemble temperature below 2 °C, offsets are
  multiplied by one plus a coefficient times the ensemble standard deviation of the 24-hour mean
  temperature. The coefficient is chosen on a grid to minimise the tuning mean pinball loss at the
  0.05, 0.95, 0.01, and 0.99 quantiles on those rows.
- **O: one-sided factor.** Multipliers on the upper offsets only (levels 0.75, 0.9, 0.95, and 0.99)
  on rows with an ensemble temperature below 2 °C, fitted as for T. The median and the lower
  offsets are unchanged.

If the diagnosis shows the mean is biased on cold nights (a residual mean over 0.3 times the
residual spread), the write-up says so, and a shift of the quantiles is added as a post hoc arm,
labelled as such, outside the contrasts.
