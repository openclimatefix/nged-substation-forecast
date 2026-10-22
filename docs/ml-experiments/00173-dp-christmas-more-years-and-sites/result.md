---
id: '00173'
slug: dp-christmas-more-years-and-sites
part: BG-S3
title: >-
  Does the convex demand forecast with the Kalman filter and recipe R stay within 1.3 times its
  ordinary-weekday error over every Christmas back to 2019-20, at all 16 Primary substations?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- calibration-and-uncertainty
- differentiable-physics-demand-calendar-and-weather
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00173-dp-christmas-more-years-and-sites/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (which cut date and training window, whether the weather is observed or forecast, how
  the spread is fitted), and this plan pre-registers one of each. Callers: this folder's scripts,
  which copy experiment 170's filter, recipe R, and event-scoring code and experiment 107's fitting
  pattern; they call the reviewed `studies.*` modules without changing them. No new modelling.
hypothesis: >-
  The convex demand model with a Kalman filter on recent errors and recipe R (the mean error of the
  last 14 days by half-hour, held to 0.75 of its size), which scored 1.13 times an ordinary weekday
  over 24 December to 2 January at five substations in two Christmases (experiment 170), keeps a
  Christmas-to-weekday error ratio under 1.3 in every Christmas back to 2019-20 and at 14 or more of
  16 Primary substations, and does not over- or under-forecast 25 and 26 December by more than 5% of
  that day's demand.
prediction: >-
  Written before any score is read. (P1) The ratio of the quantile score on 24 December to 2
  January to the quantile score on ordinary weekdays at the same half-hours, over leads 0.5 to 168
  hours and pooled over the substations that have data, is under 1.3 in every Christmas. (P2) The
  same ratio, pooled over the Christmases, is under 1.3 at 14 or more of the 16 Primary substations.
  (P3) The signed error on 25 and 26 December (measured power minus the mean forecast, summed over
  rows and substations, divided by the summed measured power), is within plus or minus 5% in every
  Christmas. The risk is a Christmas with unusual weather or demand that the weather-only model does
  not see (the cold snap of December 2022 is the one named in the brief), or an industrial or rural
  substation whose Christmas shutdown is longer than the Christmas window. A fail names the
  Christmas or the type of substation that needs a special-day term.
decision_rule: >-
  Each Christmas is one forecast season: a frozen convex-model fit on a cut date, daily 09:00 UTC
  runs of 168 hours, scored on the half-hours valid from 24 December to 2 January (local dates).
  `supported` needs P1, P2, and P3 all to hold. `partly supported` is P1 and P2 holding with P3
  failing, or P3 holding with one of P1 or P2 failing by no more than 0.1 in the ratio. `refuted`
  is a pooled Christmas ratio of 1.4 or more in at least one Christmas, or a ratio of 1.3 or more at
  more than four substations. `inconclusive` is everything else, and the verdict is capped at
  `inconclusive` if the positive control fails. Pass bands are on point estimates; every ratio is
  also given with a 95% interval from resampling local dates. The 1.3 threshold is the level at
  which a Christmas-specific term (a day-type for the Christmas window) would begin to be worth
  its added complexity in the serving path: a ratio under 1.3 is an error rise of under 30% on 10
  days a year. Positive control: this pipeline run on the Christmases of 2024-25 and 2025-26 at the
  five decision substations (D5, D10, D11, D14, D16) must give a pooled ratio within 0.15 of
  experiment 170's 1.13 and a score on ordinary weekdays no higher than experiment 170's arm D
  (2.756 points), because this experiment's weather is observed and experiment 170's is a forecast.
benchmark: >-
  Experiment 170's arm D Christmas ratio (1.13, two Christmases, five substations, ensemble-mean
  weather); experiment 155's arm D (2.756 points over leads 0.5 to 168 hours); ordinary weekdays
  of the same forecasts as the reference.
row_set_id: dp_christmas_daily_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, and the absolute error of the mean forecast, on the half-hours valid on 24
  and 25 December, 26 December, 27 to 31 December, and 1 to 2 January, against ordinary weekdays
  (Monday to Friday, 8 December to 31 January, outside the Christmas window and any bank holiday)
  at the same local half-hours; signed error of the mean forecast; coverage and pinball loss at the
  0.01, 0.05, 0.95, and 0.99 quantiles
---

**Plan only: no score has been read.** The experiment re-scores the Christmas window of experiment
170 at all 16 Primary substations and at every Christmas with data back to 2019-20.

## Design

- **Forecast.** The convex model (experiment 107's `g1_fc` arm: 12 months of history, fitted PV
  capacity, no drift term, because experiment 170 refuted the drift term) is fitted once per
  Christmas on a cut of 1 November, with the penalties of experiment 104. It forecasts daily
  09:00 UTC runs of 168 hours. The Kalman filter of experiment 133's shared noise setting
  corrects each run from the fit's own residuals, and recipe R subtracts 0.75 of the 14-day mean
  error profile. The winter night offset is left out (it added 0.002 points in experiment 155).
- **Weather.** The weather is observed (ERA5 temperature and CAMS irradiance at the valid time),
  because the ensemble forecasts exist only from 2024. The observed weather is not available at
  forecast time, so the numbers are a ceiling for the ensemble-driven forecast, and the ratio
  isolates the calendar effect from the weather error.
- **Spread.** The 11 quantiles are the corrected mean plus a residual table fitted on the same
  season's runs with valid times from 16 November to 22 December. Christmas 2019-20 has no cut
  with 90 days of training before November: it uses a cut of 16 December (the first date with 90
  days of data), reports only the mean forecast's errors, and is excluded from P1 to P3's
  pass count but shown.
- **Scoring.** The ratio and its interval, the signed error, and the coverage are computed per
  Christmas, per substation, pooled, and by day group (24 and 25 December; 26 December; 27 to 31
  December; 1 and 2 January).

## Checks and reporting

- Report absolute scores beside every ratio, the median minus the mean forecast per substation, and
  the tail table (coverage and pinball loss at 0.01, 0.05, 0.95, 0.99).
- The curtailment-cost metric needs a node with its generators and does not apply to single
  Primary demand series; it is not reported.
