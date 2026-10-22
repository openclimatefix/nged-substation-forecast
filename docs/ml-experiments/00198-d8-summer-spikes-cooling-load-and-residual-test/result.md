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
status: done
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
verdict: inconclusive
best_arm: >-
  Not applicable. The cooling arm scored the same as arm D: C1 -0.0002 points (99% interval -0.0103
  to +0.0147) over 157,860 June-to-September rows at the five decision Primaries.
headline: >-
  D8's summer power does not rise with temperature, and its large summer departures are mostly
  downward shortfalls, so the spikes are not air-conditioning load. A cooling term gave no gain
  at the five Primaries, but the pre-registered sign check and known-answer check failed, so the
  verdict is inconclusive. All six Primaries run 0.6 to 5.1 points above arm D on afternoons of 25
  degrees or more, on few days.
lessons:
  - >-
    The fit months held too few hot days: at the chosen threshold of 24 degrees the median Primary
    had 18 days and 193 daytime half-hours, and the leave-one-month-out sign check held in 2 of 4
    months.
  - >-
    The known-answer check recovered 49% of an injected cooling load of 0.1 MW per degree with the
    ensemble temperature, and 83% with ERA5 temperature (post hoc), so ensemble temperature error
    and arm D's own error correction both blunt a cooling term.
take_home_message: >-
  D8's summer power does not rise with temperature, and its large summer departures from the
  forecast are mostly downward shortfalls that grow from 18% of summer half-hours in 2022 to 52% in
  2025, so the spikes are not air-conditioning load. A cooling term added to the best-guess
  forecast changed the summer score at the five Primaries by 0.0002 points (interval -0.010 to
  +0.015), but the fit months held too few hot days and the known-answer control failed, so the
  experiment cannot rule a cooling term out. On afternoons of 25 degrees or more every Primary
  runs 0.6 to 5.1 points above the forecast, so a term fitted on the long weather history, pooled
  across Primaries, is the next test.
advice:
  - >-
    Do not read D8's summer shortfalls as a temperature effect: its summer power has a partial
    correlation of +0.004 with temperature before 2023-08-15 (D8 section).
  - >-
    Fit any cooling term on the full weather history pooled across Primaries, not on the four
    summer months of arm D's stored forecasts, which hold 18 days above 24 degrees (C1, eligibility
    counts).
  - >-
    Include a known-answer injection in any weather-term test on arm D: the error correction and
    ensemble temperature error recovered only about half of the injected load (known answer).
production_status: not-for-production
adopted_in: none
---

**We asked whether the summer spikes in the power of one substation, D8 (Marsh Lane), are
air-conditioning load, and whether the best-guess forecast needs a cooling term.** We looked at
D8's June-to-September power against temperature and at the size and timing of its departures from
the forecast. We then added a cooling term to the best-guess forecast at the five Primary
substations whose stored forecasts we have (D8 has none) and rescored the summer rows.

**D8's spikes are not cooling load, and the cooling term did not help, but the experiment could not
rule one out.** D8's summer power has a partial correlation of +0.004 with temperature. Its large
departures from expected power are mostly downward (figures 3 and 5), which matches the engine
account of experiments 186 and 194. The cooling term changed the five Primaries' summer score by
-0.0002 points, but the check that the term can be learned from the four fit months failed, so the
verdict is `inconclusive` under the pre-registered rule.

## Figures

![Contrasts of the cooling arm against arm D, real and injected](figure_1_contrasts.svg)

![Arm D's afternoon error against ensemble-mean temperature at six Primaries](figure_2_residual_vs_temperature.svg)

![D8's daily maximum power against the day's maximum temperature](figure_d8_power_vs_temperature.svg)

![D8's afternoon residual against temperature, before and after 2023-08-15](figure_d8_residual_vs_temperature.svg)

![The hour of day of D8's large summer departures](figure_d8_departure_hours.svg)

## Results

**D8 (experiment 159's panel, past weather; a lab shortcut).**

- Departures of 1.5 MW or more from expected power in June to September: downward in 18.3%, 18.8%,
  35.9%, and 52.1% of half-hours in 2022 to 2025, upward in 5.0%, 7.7%, 10.5%, and 6.5%. June 2026
  reverses: 11.3% downward and 21.3% upward.
- Largest upward departure 2.7 to 6.0 MW a year; largest downward 4.6 to 9.3 MW. Median hour of
  both kinds is about 14:00 UTC (13.5 upward, 14.0 downward). Readings flagged as zero are 0.2% or
  fewer of summer half-hours in every year, so metering artefacts do not explain the spikes.
- Mean temperature is 19.0 degrees at upward departures, 17.6 at downward ones, and 17.0 across all
  summer half-hours. The daily maximum power has a correlation of 0.13 with the day's maximum
  temperature (exploratory peek, before the plan).
- Prediction (1): the partial Spearman correlation of the afternoon residual with the hinge
  max(0, T - 20), on days of 22 degrees or more and before 2023-08-15, is +0.004 on 912 half-hours,
  below the +0.15 bar. After 2023-08-15 it is +0.051 on 1,308 half-hours.
- The cooling hinge (tau 20, median regression, fitted on summers 2022 and 2023 before the break)
  changes the mean absolute residual on 15,407 later summer half-hours by -0.001 MW (99% interval
  -0.004 to +0.001), against 1.97 MW before the term. With noise added to the temperature the
  change is -0.0001 MW (-0.002 to +0.001). The noise standard deviation of 1.2 degrees is an
  assumption, not a measured ensemble error.
- The spikes are therefore not cooling load. The upward departures appear mostly when the downward
  shortfall fades in 2026, which fits the expected power having been learned from a window that
  contains the engine's shortfalls. We did not test that account here.

**The five decision Primaries (arm D of experiment 155, ensemble temperature at forecast time).**

- Eligibility: at the median Primary the four fit months hold 111, 91, 70, 41, and 18 days above 16,
  18, 20, 22, and 24 degrees, so all five thresholds were eligible. Leave-one-month-out chose 24
  degrees, with a held-out gain in 2 of 4 months, so the sign check failed.
- C1 (cooling minus arm D, quantile score over leads 0.5 to 168 hours, June to September): -0.0002
  points (99% interval -0.0103 to +0.0147; 14-day blocks, 11 blocks). Leads 0.5 to 48 hours: +0.0067
  (-0.0061 to +0.0344). Leads 48 to 168 hours: -0.0030 (-0.0127 to +0.0067). Without the hottest
  spell (7 days): -0.0009 (-0.0035 to +0.0019). Hot days (12 days at some Primary, 5 spells): -0.0014
  (-0.1241 to +0.1327) on 12,174 rows.
- Absolute summer scores, arm D then cooling arm, in points: D5 3.182 and 3.178; D10 2.969 and 2.972;
  D11 2.300 and 2.307; D14 1.978 and 1.979; D16 2.314 and 2.306.
- Placebo (200 draws of other weeks and years' temperatures, tau held at 24): gains from -0.0014 (5th
  percentile) to +0.0079 (95th), median +0.0019. C1 lies above the 5th percentile, so the term did
  not beat the placebo.
- Known answer (0.1 MW per degree above 20 degrees injected at D10, shaped by an afternoon bell
  curve, driven by ERA5): the fitted effect is 49% of the injected sum, and C1 on the injected data is
  -0.068 (-0.155 to +0.004), so the check failed. Post hoc: forcing tau 20 gives 59%, and replacing
  the ensemble temperature by ERA5 gives 78% (tau chosen) or 83% (tau 20), with C1 -0.150 (-0.352 to
  -0.023) in the last case.
- Prediction (2): the partial correlation of the afternoon residual with the hinge is +0.26 at D10 and
  +0.21 at D9, and +0.04, +0.07, +0.01, and +0.07 at D5, D11, D14, and D16. It was predicted to be
  below +0.15 at 5 or 6 of 6 Primaries, and is below it at 4.
- Post hoc, on the afternoons with an ensemble-mean temperature of 25 degrees or more (leads up to 24
  hours, June to September, all periods), actual power minus arm D averages +0.90 (D5, 232 rows on 26
  days), +4.68 (D10, 85 rows on 8 days), +2.48 (D11, 109 rows on 12 days), +0.60 (D14, 174 rows on 22
  days), +1.17 (D16, 251 rows on 29 days), and +5.09 points (D9, 45 rows on 5 days). At 19 to 25
  degrees it averages -0.97, -1.08, -0.27, -1.36, -0.92, and +0.42 points.

## Checks

- Median minus mean forecast is 0.00 points at every Primary: arm D's median equals its mean.
- The mean forecast's mean absolute error beside each score change, arm D then cooling arm, in points:
  D5 5.548 and 5.538; D10 4.997 and 4.999; D11 4.121 and 4.131; D14 3.458 and 3.454; D16 4.081 and
  4.062. Neither the score nor the error moves in a consistent direction.
- Leaving one Primary out moves C1 between -0.0020 and +0.0019, every 99% interval includes zero.
- Triggers fired: trigger 1 (the known-answer check and the sign check failed) and trigger 2 (the
  correlation prediction failed at 2 of 6 Primaries; the post hoc hot-end table is the tested
  explanation). No audit has been run: a fresh Opus audit is the maintainer's call, because the
  headline is a null, and this page does not claim more than `inconclusive`.

## Caveats

- Arm D's spread was fitted on the tuning months and its mean includes recipe R, which corrects the
  last 14 days' error and may absorb part of a cooling load (known answer).
- Only 4 summer months fit the hinge and 4 score it, from 2024 to 2026, so the scored summer is one
  year and hot days are few. The blocked intervals rest on 11 blocks.
- D8 has no stored forecast from the best-guess model, so its test uses the residual of the convex
  demand model refitted on the 1st and 16th of each month (experiment 130's panel) with ERA5
  temperature. D8's cooling term is not comparable with the Primaries'.
- The pre-registered fit rule deviates in two small ways from the plan text: the eligibility count
  uses the median Primary, and the placebo draws temperatures a year (plus or minus 0 to 28 days)
  away for each 14-day block.

## What to try next

- Fit a cooling hinge on the long weather history, pooled across Primaries, so that the fit holds
  dozens of days above 25 degrees instead of 4 to 18. Keep the known-answer injection as the gate.
- Ask NGED whether the engine's output is still added back to D8's reading, which decides whether
  D8's upward departures are real demand.

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
