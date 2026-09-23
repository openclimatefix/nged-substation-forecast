# How accurate is a power forecast driven by ECMWF ENS at each horizon?

**At six solar farms and three wind farms in Lincolnshire, an XGBoost model given the ECMWF
ensemble's mean forecast beats every forecast that reads no weather forecast out to day 5 for solar
and day 7 for wind, and loses to the best of those by day 14.** ENS is the ensemble forecast of the
European Centre for Medium-Range Weather Forecasts (ECMWF), 51 runs of one weather model started a
little differently, and the live service reads its 00 UTC run each day. Every error on this page is
a mean absolute error as a percentage of each generator's capacity, and a bracketed pair after a
figure is its 95% interval. At day 1, the day after the run, the ensemble mean gives an error of
8.77% [8.29, 9.25] for solar and 8.16% [7.47, 8.95] for wind. The error grows with every day of
horizon: by day 7 it is 14.30% [13.38, 15.27] for solar and 16.70% [15.24, 18.18] for wind. The best
forecast that reads no weather forecast is climatology, each generator's typical output for the
month and hour, at 14.68% [13.76, 15.62] for solar and 18.29% [16.68, 19.90] for wind. The evidence
is 50,643 generator-hours of solar from April 2024 to September 2026 and 50,268 generator-hours of
wind from August 2024 to September 2026.

**Of three ways of using the ensemble, feeding the mean of its members to the XGBoost model does
best at almost every horizon.** It beats the control member, the one run started from ECMWF's best
estimate of the current weather, and it beats training one XGBoost model on every member and
averaging its 51 forecasts. Every ENS input is first turned from ENS's 3- and 6-hourly steps into
hourly values. For solar, rebuilding the radiation through the clear-sky index lowers the error by
0.30 points [0.22, 0.40] at day 1 against the straight-line interpolation the live service uses
today. For wind, no way of interpolating tested moves the error by a tenth of a point.

![Figure 1: At these farms, an XGBoost model given the ENS ensemble mean beats every no-weather
baseline to day 5 for solar and to day 7 for wind](assets/ens_horizons_leaderboard.svg)

**Figure 1 draws each forecast's own error at each horizon; Figure 2 tests each horizon against day
0 directly.** Figure 1's intervals are wide mainly because every forecast's error rises and falls
together from month to month. Figure 2 compares two XGBoost models on the same months and the same
fitting seed, which cancels that shared swing, so its intervals are much narrower. Two points can
therefore overlap in Figure 1 and still differ by a margin statistically significant at the 5% level
in Figure 2.

![Figure 2: Forecast error rises with every day of horizon, fastest over the first
week](assets/ens_horizons_against_day0.svg)

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Opus 5.5, reusing data-preparation and model-fitting code that
> Claude Opus 5 and Claude Sonnet 5 wrote for the earlier weather-product studies). Several
> independent Claude reviewers have checked the method, the evidence, and the prose adversarially.

## Key findings

**A difference between two errors is in percentage points of capacity, written "points". Where one
forecast "beats" another, the difference is statistically significant at the 5% level.**

- **[The XGBoost models follow the measured output at every generator at day 1; at day 7 they stay
  close to an average day.](#the-xgboost-models-work)**
- **[Rebuilding solar radiation through the clear-sky index beats straight-line interpolation at
  every horizon to day 7; for wind, the way of interpolating makes no material
  difference.](#turning-enss-steps-into-hourly-values)**
- **[Error rises with horizon: from day 0 to day 1 by 0.61 points [0.45, 0.79] for solar and 0.75
  points [0.56, 0.94] for wind, and to day 7 by 6.14 points [5.48, 6.85] and 9.30 points [8.20,
  10.43].](#error-grows-with-horizon)**
- **[The ensemble mean beats the control member, and beats training on every member and averaging
  the forecasts, at almost every
  horizon.](#the-ensemble-mean-is-the-best-of-the-three-ways-tested)**
- **[The ensemble mean beats the best no-weather baseline to day 5 for solar and to day 7 for wind;
  by day 14 it loses to climatology.](#where-ens-stops-beating-the-no-weather-baselines)**
- **[At day 1 the ensemble mean beats ERA5 for solar and not for wind, and loses to the best
  past-weather input a live service can read.](#ens-against-past-weather)**

## Introduction

**Every weather product this project has compared so far was scored as a description of weather that
had already happened, which is not the question the live service faces.** The live service forecasts
power days ahead, from a forecast of the weather. [Issue
#810](https://github.com/openclimatefix/nged-substation-forecast/issues/810) asks which weather
model gives the best power forecast at the horizons the live service runs at. This page is the
first, deliberately small step of that question: it measures how accurate a power forecast driven by
ENS, the forecast the live service already reads, is at each horizon, before any other forecast
product is added.

**Three groups of forecasts are compared at each horizon, and two reference rows show how good the
weather input could be.**

- **ENS, three ways:** the control member alone; the mean of the 51 members' weather; and each
  member through one XGBoost model, the 51 power forecasts then averaged.
- **No-weather baselines:** persistence, diurnal persistence, smart persistence, and climatology,
  defined under "Data and methods".
- **References, which are not forecasts:** ERA5, ECMWF's reanalysis of past weather, available about
  5 days late; and UKV with ICON-EU, the best past-weather input a live service anywhere in Great
  Britain can read on [the blending page](blending-weather-products.md), each available within hours
  of the hour.

| Input | What it is | Grid | Steps | When it can be read |
|---|---|---|---|---|
| ENS | ECMWF's 51-member ensemble forecast, 00 UTC run, from Dynamical.org | 0.25°, averaged over each generator's H3 cell (about 250 km²) | 3-hourly to 144 hours ahead, 6-hourly beyond | about 09:00 UTC on the run's day |
| ERA5 | ECMWF reanalysis | about 31 km | hourly | about 5 days after the hour |
| UKV and ICON-EU | Met Office and German weather service models, as Open-Meteo's archive serves them | 2 km and 7 km | hourly | within hours of the hour |

## Data and methods

### The hours scored

**Every comparison is scored at hourly resolution, on the hours the past-weather studies scored:
every daylight hour at six solar farms, labelled A to F, and every hour at three wind farms,
labelled W1 to W3.** Hours with an exactly-zero half-hour, the commissioning ramp of one solar farm,
and outages are dropped as on [the solar page](weather-products-for-past-solar.md) and [the wind
page](weather-products-for-past-wind.md). An hour is also dropped from every forecast where any
horizon's ENS input or any baseline's input is missing, so every row of a panel is scored on the
same hours. That leaves 50,643 of 54,791 solar hours, from 15 April 2024 to 10 September 2026, and
50,268 of 50,734 wind hours, from 12 August 2024 to 10 September 2026. The ENS archive starts on 1
April 2024, and day 14 needs a run 14 days before the day it forecasts.

### Horizons and issue time

**A horizon is a whole UTC day after the run's own day: day d forecasts a calendar day from the 00
UTC run d days earlier.** The horizons scored are days 0, 1, 2, 3, 5, 7, 10, and 14: every day to
day 3, then days 5, 7, and 10, which span the 3-to-10-day band [the
roadmap](../roadmap/xgboost-improvements.md) names as the live service's primary band, and day 14,
near the end of the run. The live service reads a 00 UTC run from about 09:00 UTC, so day 1, the
day-ahead product, is 15 to 39 hours after the forecast is issued. Most of day 0 is over before the
run can be read, so day 0 is a best case, not a product the live service can deliver.

### Turning ENS's steps into hourly values

**ENS publishes its radiation as a mean over each 3- or 6-hour step and its wind as a value at each
step, so every ENS input is first upsampled to hourly values.** [NWP variable
conventions](../architecture/nwp-variable-conventions.md) sets out why a straight line between steps
misreads a period mean. Each member is upsampled before any is averaged. The techniques tested are
straight-line interpolation of every field, which is what the live service does today; for solar
radiation, the clear-sky-index resample that page describes, with and without scaling each step's
hours to keep the step's mean; a shape-preserving curve for temperature; and for wind, the speed or
the direction, or both, taken from interpolated eastward and northward components. A rule written
before any result chose the combination every later comparison uses: starting from straight lines,
try one change at a time, and keep it only if it lowers the ensemble-mean model's error at both day
1 and day 7. The rule chose the clear-sky index for solar radiation, straight lines for solar
temperature, and the components for wind speed.

### The forecasts

**Each forecast is an XGBoost model fitted per generator, trained on the input it is scored with.**
A solar model is given the hour's sun position, the time of day, the day of the year, which side of
the Met Office's January 2026 upgrade of UKV the hour falls on, and ENS's radiation and air
temperature. A wind model is given the time of day, the day of the year, the same upgrade flag, and
ENS's 100 m wind speed, its direction as a sine and a cosine, and its 10 m wind speed.

- **Control member:** trained and scored on the control member.
- **Ensemble mean:** trained and scored on the mean of the 51 members' weather. The mean wind
  direction is the direction of the mean wind vector.
- **Member by member:** one XGBoost model trained on every member's hours, each member's hour
  weighted 1/51 so that an hour counts once, then applied to each member; the 51 power forecasts are
  averaged. Their median is reported as well.
- **Exploratory: trained on the mean, applied to each member.** An XGBoost model trained on the
  ensemble mean, applied to each member, its 51 forecasts averaged. It breaks the rule of training
  on the input scored.

**The no-weather baselines read only the telemetry up to the forecast's issue time: 09:00 UTC on the
run's day, or the run's 00 UTC start for day 0,** whose day would otherwise hand the baseline the
morning it forecasts. The telemetry is taken as available at once, the best case for persistence.

- **Persistence:** the last observed hour's output, held for every hour forecast.
- **Diurnal persistence:** the output at the same time of day on the last whole day before the
  issue.
- **Smart persistence:** for solar, the clear-sky index of the last 24 observed hours times the
  clear-sky irradiance at the hour forecast; for wind, persistence blended with climatology, the
  weight fitted per generator and horizon on the training months.
- **Climatology:** each generator's median output for the calendar month and hour of day, over the
  training months only. The median, because the error is a mean absolute error, which the median
  minimises.

### Folds, intervals, and planned contrasts

**Each model is trained on some blocks of whole months and scored on the others, fitted three times
from three fitting seeds.** The five blocks, called folds, are cut separately before and after the
UKV upgrade, as on the past-weather pages, so that the references are fitted as there. Hours the
network operator curtailed are left out of training and kept for scoring, with the forecast held to
the export cap then in force. Each error is divided by its own generator's capacity, its 99th
percentile of output.

**Each 95% interval comes from resampling whole calendar months, each time also drawing one of the
three fitting seeds.** A difference is statistically significant at the 5% level when its interval
lies wholly on one side of zero. The test covers month-to-month weather and the fitting seed, not
differences between generators.

**Five contrasts were planned, written into the study code before any result existed:** the ensemble
mean at day 1 and at day 7, each against day 0; member by member against the ensemble mean at day 1;
the ensemble mean against the control member at day 1; and the ensemble mean at day 7 against
climatology. Every other figure is exploratory, and among many exploratory figures about 1 in 20
reaches significance at the 5% level by chance. The planned contrasts are also run with a second,
shallower XGBoost setting.

## Results

### The XGBoost models work

**Given the day-1 ensemble mean, the XGBoost model follows the day-to-day swings in measured output
at every generator.** Figures 3 and 4 plot out-of-fold forecasts against measured output through one
week each, chosen from measured output alone: for solar the week, in February 2025, whose daily
output varies most, for wind the week, in September 2025, with the largest hour-to-hour changes. At
day 7 the solar forecast stays close to an average day, too high on dull days and too low on bright
ones, and the wind forecast mostly misses the swings.

![Figure 3: Given the day-1 ensemble mean, the XGBoost model follows the day-to-day swings at every
solar farm; the day-7 forecast stays close to an average day](assets/ens_horizons_solar_week.svg)

![Figure 4: Given the day-1 ensemble mean, the XGBoost model follows the wind farms' swings; the
day-7 forecast mostly does not](assets/ens_horizons_wind_week.svg)

**Every generator's error rises between day 1 and day 7** (Figure 5).

![Figure 5: Every generator's error rises between day 1 and day
7](assets/ens_horizons_per_generator.svg)

### Turning ENS's steps into hourly values

**For solar, the clear-sky index keeps the shape of the day where a straight line between steps
shifts it late and flattens it.** Figure 6 shows one day, in April 2025, at day 1 and day 7, and a
wind day in September 2025.

![Figure 6: The clear-sky index keeps the solar day's shape, where linear interpolation shifts it
late](assets/ens_upsampling_days.svg)

**Rebuilding solar radiation through the clear-sky index lowers the error at every horizon to day 7:
by 0.30 points [0.22, 0.40] at day 1 and 0.16 points [0.02, 0.31] at day 7.** Beyond day 7 the gain
is not statistically significant at the 5% level. Keeping each step's mean exactly adds nothing, and
costs 0.04 points [0.02, 0.07] at day 1. The shape-preserving curve for temperature moves the error
by less than 0.02 points at every horizon, and the rule did not adopt it.

![Figure 7: Rebuilding solar radiation through the clear-sky index lowers the error at every horizon
to day 7](assets/ens_upsampling_solar.svg)

**For wind, no technique moves the error by as much as a tenth of a point at any horizon.** Taking
the direction from interpolated components lowers the error by 0.02 points at day 1 and raises it,
not significantly, at day 7, so the rule did not adopt it. Taking the speed from components moved
the error by −0.010 at day 1 and −0.011 at day 7, neither significant, and the rule adopted it.

![Figure 8: No way of interpolating ENS's wind moves the wind error by a tenth of a
point](assets/ens_upsampling_wind.svg)

**ENS scored on its own steps is a reference, not a rival, because it is scored on different rows.**
Scored on its own 3-hour steps, the day-1 ensemble mean gives 6.87% [6.36, 7.39] for solar and 8.11%
[7.43, 8.88] for wind. A 3-hour mean of solar power is easier to forecast than one hour of it, and
the wind rows are only the hours at ENS's stamps, so these figures cannot be set against the hourly
ones.

**Giving day 1 only every other step costs 0.23 points [0.17, 0.28] for solar and 0.24 points [0.16,
0.31] for wind.** This emulates the 6-hour steps ENS publishes beyond day 6, so part of the rise
from day 5 to day 7 is the coarser step, not the longer horizon.

### Error grows with horizon

**The ensemble mean's error rises with every day of horizon, most steeply over the first week.**
From day 0 to day 1 it rises by 0.61 points [0.45, 0.79] for solar and 0.75 points [0.56, 0.94] for
wind, and from day 0 to day 7 by 6.14 points [5.48, 6.85] and 9.30 points [8.20, 10.43], all four
planned. By day 14 the solar error is 15.23% [14.22, 16.19] and the wind error 19.46% [17.57,
21.35]. The second XGBoost setting gives the same picture.

### The ensemble mean is the best of the three ways tested

**The ensemble mean beats the control member at every horizon to day 10 for solar and to day 7 for
wind, and training on every member loses to both at most horizons.** At day 1, the ensemble mean
beats the control member by 0.36 points [0.26, 0.47] for solar and 0.40 points [0.26, 0.53] for
wind, and member by member loses to the ensemble mean by 0.48 points [0.34, 0.62] and 0.75 points
[0.52, 0.99], all planned.

![Figure 9: The ensemble mean beats the control member, and beats feeding each member through the
model, at almost every horizon](assets/ens_horizons_ways.svg)

**Training on every member makes the XGBoost model's response to the weather flatter, which is why
member by member loses.** Each member's weather is a noisier guess at the hour than the mean, and a
model trained on noisy inputs learns to respond less to them. Its 51 forecasts then cluster too
tightly: at day 1 only 30% of solar hours and 42% of wind hours fall between the 10th and 90th
percentiles of the 51 forecasts, where a calibrated ensemble would hold 80%. The loss survives
turning off row subsampling, which drops member rows rather than hours: 0.47 points [0.33, 0.62] for
solar and 0.72 points [0.50, 0.95] for wind at day 1.

**In an exploratory check, a model trained on the ensemble mean and applied to each member does no
worse than the ensemble mean.** For solar it is ahead at every horizon, by 0.06 points [0.02, 0.10]
at day 1; for wind the difference is not statistically significant at the 5% level to day 7. This
breaks the rule of training on the input scored, so it is a lead rather than a result.

### Where ENS stops beating the no-weather baselines

**The ensemble mean beats the best no-weather baseline by several points to day 5, and loses to it
by day 14.** For solar the best baseline at every horizon is climatology; for wind it is climatology
or smart persistence, which are within 0.1 points of each other from day 3. At day 7, the planned
contrast against climatology favours the ensemble mean by 0.38 points [-0.04, 0.78] for solar, not
statistically significant at the 5% level at the main setting and significant at the second, and by
1.58 points [0.79, 2.42] for wind. At day 10 neither technology's difference is statistically
significant, and at day 14 climatology is ahead by 0.55 points [0.04, 1.04] for solar and 1.18
points [0.41, 1.96] for wind. Persistence and diurnal persistence are worse than climatology at
every horizon, except wind persistence at day 0.

![Figure 10: The ENS ensemble mean beats the best no-weather baseline by several points to day 5,
and loses to it by day 14](assets/ens_horizons_against_baselines.svg)

### ENS against past weather

**At day 1, the ensemble mean beats ERA5 for solar by 0.29 points [0.05, 0.53], and loses to it for
wind by 0.92 points [0.73, 1.14].** ERA5's solar radiation is itself a short-range forecast,
averaged over a 31 km cell, which may be why a day-ahead forecast can beat it. **UKV with ICON-EU,
the best past-weather input a live service can read, beats the day-1 ensemble mean by 1.29 points
[1.06, 1.55] for solar and 2.20 points [1.93, 2.49] for wind**: the room a better weather input
leaves.

## What to use

**These recommendations rest on nine farms in one part of Lincolnshire over about two years, and on
ENS alone.**

- **Feed the power model the ensemble mean, not the control member.** The mean beats the control at
  day 1 for both technologies, in planned contrasts.
- **Upsample ENS's solar radiation through the clear-sky index before it reaches the model.** It
  beats the straight-line resample at every horizon to day 7. For wind the choice makes no material
  difference.
- **Do not train on every member and average, as tested here.** Whether a model trained on the mean
  and applied to each member is better is a lead for the next step, not a finding.
- **Beyond about a week, an ENS-driven forecast from this set-up is no better than climatology.**

## Limitations

- **One region and about two years.** Six solar farms in a 25 km by 23 km box and three wind farms,
  from April or August 2024 to September 2026. The intervals resample months, not farms.
- **One forecast product.** Other forecast products join this comparison once their archives at
  fixed leads are downloaded.
- **The H3 cell, not the farm.** ENS is averaged over each generator's H3 cell of about 250 km².
- **Day 0 is a best case.** Most of it passes before the live service can read the run.
- **The baselines see telemetry the moment it is measured.** A live service may see it later, which
  would make persistence worse.
- **ECMWF's upgrades of its model inside the period are not treated as breaks,** unlike the Met
  Office's upgrade of UKV.
- **Dropped hours.** 4,148 of 54,791 solar hours and 466 of 50,734 wind hours of the past-weather
  studies' rows since late March 2024 are dropped from every forecast. For solar, 3,582 of them lack
  a baseline's input, because the telemetry before the issue time has a gap, and 1,196 fall in the
  first three weeks of the ENS archive, before day 14 has a run; the two groups overlap. Every wind
  hour dropped lacks a baseline's input. Dropping hours whose preceding telemetry has a gap may
  favour the baselines slightly.
- **The figures rest on the capacity table as rebuilt in September 2026.** Each generator's capacity
  is its 99th percentile of output from the `effective_capacity` table, so a rebuilt table would
  move every figure.

## Reproducing the figures

```bash
uv run python studies/beam_diffuse_split/fetch_ens_forecast_horizons.py
uv run python studies/beam_diffuse_split/ens_forecast_horizons.py
uv run python studies/beam_diffuse_split/ens_forecast_charts.py
```

The report lands in `data/studies/ens_forecast_horizons/report.md`. The run needs the solar, wind,
and blending studies' inputs on disk, and takes about two hours on a 32-core machine.
`ens_forecast_horizons.py --report-only` rebuilds the report from the saved losses.
