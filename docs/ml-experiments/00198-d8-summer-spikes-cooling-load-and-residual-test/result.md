---
id: "00198"
slug: d8-summer-spikes-cooling-load-and-residual-test
title: >-
  Are the summer spikes in D8's power air-conditioning load, and does adding a cooling term lower the
  best-guess forecast's summer error at D8 and at the other Primaries?
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to:
  - differentiable-physics-embedded-generation
  - switching-detectors
status: planned
size: >-
  medium. Stored: private tables and figures under
  data/ml-experiments/00198-d8-summer-spikes-cooling-load-and-residual-test/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible (a cooling
  hinge refitted inside the convex model, or fitted on the stored forecast's residuals), and this
  plan fixes the residual form because D8 has no stored GAM-K forecast and the convex model's
  fitting code would need a reviewed change. Callers: this folder's scripts only; they read
  experiment 155's stored forecasts (arm D), experiment 111's stored ensemble temperatures, and
  experiment 159's stored D8 panel, and call reviewed `studies.*` modules without changing them.
hypothesis: >-
  The summer spikes a colleague saw in D8's power (time series 26, Marsh Lane) are air-conditioning
  load: they rise smoothly with temperature, peak in the afternoon of the hottest days, and repeat
  every year. The best-guess forecast has a heating hinge but no cooling term, so its summer
  residual (metered minus forecast) rises with temperature, and a cooling hinge on the ensemble
  mean temperature fitted on past residuals lowers its summer score.
prediction: >-
  Written before any result is read. I expect cooling to fail. Experiments 159, 186, and 194 found
  D8's large departures are downward shortfalls of 1.5 MW or more that follow an added-back 6 MW
  gas engine and the electricity price, not temperature. An exploratory peek at the stored panel
  while writing this plan (not a test) gave a correlation of 0.13 between D8's daily maximum power
  and the day's maximum ERA5 temperature in June to September. A UK Primary's cooling load is
  small. Predictions: (1) at D8 before the engine's readings stop carrying output (2022-06-01 to
  2023-08-14, a date range taken from experiment 194, so no residual is used to choose rows), the
  partial Spearman correlation between the summer afternoon residual and the hinge variable max(0,
  T - 20) on days with a day maximum of 22 degrees or more, controlling for day of year, is below
  +0.15; (2) the same correlation is below +0.15 at 5 or 6 of the 6 stored Primaries; (3) C1
  (below) lies between -0.02 and +0.01 points with an interval including zero.
decision_rule: >-
  Fixed before any result is read. Cooling hinge: the mean over the 51 ensemble members of max(0,
  T_member - tau), T being the 2 m temperature at the valid time from the run available at
  power_fcst_init_time, with one coefficient per 4-hour band of the day (6 bands, ridge 1.0),
  added to arm D's mean and to every quantile (the spread is not refitted). Fit rows are the four
  non-scored summer months (July, August, and September 2024, and June 2025), before any scored
  row. A tau of 16, 18, 20, 22, or 24 degrees is eligible only if those months hold at least 15
  days and 100 daytime half-hours above it; the counts are printed before any fit. Tau is chosen
  by leave-one-month-out over the four months, and the sign check passes when the held-out gain is
  negative in at least 3 of the 4; a failure caps the verdict at `inconclusive`. The final hinge
  is refitted on all four months. Scored rows are June to September in the scored year (2025-07-02
  to 2025-09-30 and 2026-06-01 to 2026-06-29, off the stride), rows outside the hand-filtered
  switching events, leads 0.5 to 168 hours. Headline contrast C1: cooling arm minus arm D,
  quantile score in points, pooled over the five decision Primaries (D5, D10, D11, D14, D16), with
  a 99% interval from a paired bootstrap over 2-week blocks (adjacent days share forecast runs,
  and 4 month blocks cannot resolve anything), plus C1 with the hottest spell dropped. C1 is also
  reported by lead band (0.5 to 48 hours, 48 to 168 hours) and on hot days (ensemble-mean day
  maximum at or above 25 degrees; if there are fewer than 10 such days in 3 spells, hot days are
  descriptive only). Forecast verdict (does the cooling term help the forecast): `supported` when
  C1's interval lies wholly below -0.01 points on summer rows (about -0.003 over the year; the
  term adds 6 coefficients per Primary to a model that already has a heating hinge, and costs
  nothing at serving time beyond the temperature it already reads), C1 stays below zero with the
  hottest spell dropped, C1 lies below the 5th percentile of the placebo gains, and the known-
  answer check passes. `small gain` when C1's interval lies wholly below zero and the other
  conditions hold but not the -0.01 bar. `refuted` when C1's interval lies wholly above zero. `not
  supported` when C1's interval includes zero. Controls: (a) a placebo in which whole 2-week
  spells of temperature from other years' same calendar weeks (2024 temperatures laid on 2025
  dates, diurnal shape kept) replace the real temperature, 200 draws of refitting and rescoring;
  (b) a known-answer check that adds a synthetic cooling load of 0.1 MW per degree above 20
  degrees, driven by ERA5 temperature and shaped by a bell curve centred on 15:00 UTC, to arm D's
  actuals at one Primary, and passes when the fitted summed effect on hot rows is within 25% of
  the injected summed effect and C1 on the injected data meets the `supported` bar. Their failure
  caps the verdict at `inconclusive`. D8 question (are the spikes cooling load): judged on the
  look-first classification and on D8's correlation of prediction (1); the spikes are called
  cooling load only if the correlation is at least +0.15 and the large departures are upward and
  afternoon-peaked; a negative correlation is reported as a sunny-day or photovoltaic confound,
  not as the opposite of cooling. D8 has no stored GAM-K forecast (experiment 111's ensemble store
  covers D5, D9, D10, D11, D14, D16, and D18), so D8 uses experiment 159's stored panel (metered
  power, the frozen convex model's expected power, whose freeze date is read from experiment 130
  and printed, and ERA5 temperature) as a lab shortcut: the cooling hinge is fitted by median
  regression on the residual of summers 2022 and 2023 before 2023-08-15 and scored on summer 2023
  after the break and summers 2024 to 2025, as the mean absolute error in MW with a 2-week-block
  bootstrap interval, with a sensitivity that adds the ensemble-versus-ERA5 temperature error as
  noise. D8 has no error-correction layer, so its gain is not comparable with the Primaries' and
  stays out of the quantile-score verdict. The correlations of prediction (2) are descriptive and
  enter no verdict.
benchmark: >-
  Experiment 155's arm D as stored (2.791 points over all rows at the five decision Primaries),
  rescored on the summer rows; climatology and manual_heuristic as stored by experiment 111 for
  absolute skill.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity on June to September scored rows outside the hand-filtered switching events,
  then by lead band and on hot days (count and spells reported); coverage and pinball loss at q01,
  q05, q95, q99 on hot and normal days; the partial Spearman correlations per Primary; D8's mean
  absolute residual in MW
code: ml-experiments/00198-d8-summer-spikes-cooling-load-and-residual-test/
date: 2026-10-22
---

*The plan is committed before any code runs; the opening paragraphs and results are written after
the run.*

## Plan

- **Look first.** Plot D8's June to September power against the day's maximum temperature and the
  hour of day, for each year from 2022 to 2026. Say whether the spikes are upward or downward, their
  size in MW, and when they occur.
- **Separate the candidates.** Cooling load rises smoothly with temperature and peaks in the
  mid-afternoon of the hottest days. The engine shortfalls of experiments 186, 194, and 196 are
  downward and follow the price. Switching transfers are steps that repeat at both ends. Metering
  artefacts are readings of zero or a stuck value. The look sorts each large departure into one of
  the four by its shape, hour, and temperature.
- **Cheap test, then fit.** Correlate the summer afternoon residual with temperature at every
  Primary, then fit the cooling hinge of the decision rule above and score it against arm D on the
  same rows.
- **Weather source.** The hinge uses the ensemble-mean temperature at the valid time, which exists at
  forecast time. D8's panel uses ERA5, which arrives about 5 days late; that is a lab shortcut and is
  labelled.
- **Prior advice followed.** Experiment 155's scoring (quantile score, rows outside the hand-filtered
  switching events) and its Primaries; the placebo is built from permuted days, not shifted
  indicators, as experiment 155's audit advised.
