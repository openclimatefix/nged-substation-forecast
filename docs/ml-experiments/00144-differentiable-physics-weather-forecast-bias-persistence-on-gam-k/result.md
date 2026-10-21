---
id: "00144"
slug: differentiable-physics-weather-forecast-bias-persistence-on-gam-k
title: "Differentiable physics: does the weather-forecast error persist from day to day?"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction]
status: done
size: >-
  simple. Stored: one private JSON file of autocorrelations. Serving path: not touched. Degradation
  rule: not touched. Designs: one defensible (the stop rule in the brief). Callers: this folder's
  one script only; it reads experiment 111's stored ensemble runs.
hypothesis: >-
  The day-ahead error of the ensemble weather forecast (temperature and irradiance) repeats from
  one day to the next, so the trailing mean of recent weather errors can correct GAM-K's demand
  forecast beyond what the Kalman filter already does.
prediction: >-
  Lag-1 autocorrelation of the daily weather-forecast error of 0.2 to 0.4 in at least one season.
  Below 0.2 in every season ends the experiment, because a demand correction built on that error has no
  signal to use.
decision_rule: >-
  Stop before any demand arm if the lag-1 autocorrelation, averaged over the seven stored
  substations, is under about 0.2 in every season for both variables.
benchmark: GAM-K as stored by experiment 111 (not scored, because the experiment stopped)
row_set_id: >-
  not applicable; step 1 uses every daily run from 2024-04-02 to 2026-06-30 (820 days) at the seven
  substations stored by experiment 111
metric: Pearson autocorrelation of the daily mean weather-forecast error at lags 1 to 7 days
verdict: refuted
best_arm: none; no demand arm was run
headline: >-
  Lag-1 autocorrelation of the day-ahead weather-forecast error is at most +0.11 for temperature and
  +0.05 for irradiance in every season, against a stop threshold of 0.2.
lessons:
  - >-
    The ensemble's day-ahead temperature error against the analysis proxy has a daily standard
    deviation of about 0.3 degrees Celsius at D5 and a lag-1 autocorrelation of at most +0.11.
  - >-
    The only reference we found that is available at forecast time is a later forecast run, so this
    test measures how much ensemble runs revise each other, not how wrong they are against
    observations.
production_status: not-for-production
adopted_in: none
date: 2026-10-21
take_home_message: >-
  In this test, the day-ahead temperature and irradiance forecasts did not repeat their errors from
  one day to the next: the correlation between one day's error and the next day's was at most
  +0.11, against a stop threshold of 0.2. A correction that feeds recent weather errors into the
  demand forecast therefore has little signal to use, so no demand forecast was fitted. Two
  caveats apply: wind was not tested, because experiment 111's ensemble store does not hold it, and
  the reference was a later forecast (the analysis proxy), not observed weather, so a bias against
  observations is not ruled out.
advice:
  - >-
    Do not build a demand correction from recent ensemble-minus-analysis-proxy weather errors: their
    lag-1 autocorrelation is at most 0.11 (step 1).
  - >-
    To test weather-bias persistence against observations, first find an observed source that is
    published before the forecast origin; ERA5 arrives about 5 days late (step 1 caveat).
code: ml-experiments/00144-differentiable-physics-weather-forecast-bias-persistence-on-gam-k/
---

**Take-home: in this test, the day-ahead weather forecast did not repeat its errors from one day
to the next, so there is nothing to feed a demand correction.** The brief set a stop rule at a lag-1
autocorrelation of 0.2 (the correlation between one day's error and the next day's). Every
season's value is under 0.2, so no demand forecast was fitted. Two caveats apply: wind was not
tested, because experiment 111's ensemble store does not hold it, and the reference was a later
forecast, not observed weather.

We asked whether an ECMWF ensemble forecast that was too cold or too cloudy yesterday tends to be
too cold or too cloudy today. For each day we took the ensemble mean forecast published the morning
before. We subtracted a reference for that day that a live service could build afterwards: the
control forecast from each of the following runs, joined end to end. We call this reference the
"analysis proxy". It is a later forecast, not an observation. We then correlated each day's error
with the error 1 to 7 days later, at seven substations over 820 days.

The lag-1 correlation is at most +0.11, so a correction based on it has no signal to use. A search
of `docs/ml-experiments` found no earlier experiment that tested this directly. The closest earlier
result we found is experiment 111's, where ERA5 (the European Centre for Medium-Range Weather
Forecasts reanalysis) residuals in the GAM-K filter (the convex demand model with a Kalman filter)
did not change the score (GAM-K-era5 minus GAM-K: -0.008, interval -0.024 to +0.008). That agrees
with this experiment's result.

## Weather-forecast errors are barely correlated from one day to the next

Each cell is the Pearson correlation averaged over seven substations (D5, D9, D10, D11, D14, D16,
D18), with about 200 days per season. The raw and anomaly versions are both in the JSON file;
"anomaly" removes each calendar month's mean error first.

| Variable | Season | Lag 1 | Lag 2 | Lag 3 | Lag 7 |
|---|---|---|---|---|---|
| Temperature | December to February | +0.11 | +0.07 | -0.06 | -0.05 |
| Temperature | March to May | +0.03 | +0.02 | -0.04 | +0.06 |
| Temperature | June to August | +0.02 | +0.02 | -0.02 | +0.08 |
| Temperature | September to November | +0.10 | 0.00 | -0.04 | +0.03 |
| Irradiance | December to February | +0.03 | -0.13 | -0.02 | -0.02 |
| Irradiance | March to May | -0.01 | -0.09 | -0.01 | +0.02 |
| Irradiance | June to August | +0.04 | +0.04 | +0.08 | +0.06 |
| Irradiance | September to November | +0.05 | +0.02 | -0.10 | -0.11 |

- The anomaly version gives similar values (lag 1 from -0.01 to +0.09).
- `e00144_autocorr.py` prints lags 1 to 7 for both versions and writes `work/autocorrelation.json`.

## Checks and caveats

- **Wind was not tested.** Experiment 111's ensemble store holds temperature and shortwave
  irradiance only. GAM-K takes no wind input.
- **The reference is a later forecast.** The error is the revision between the day-ahead ensemble
  mean and the next run's control forecast, which is 9.5 to 33 hours old when subtracted.
- **The errors are small.** At D5 the daily temperature error has a standard deviation of about
  0.3 degrees Celsius, and the daily irradiance error about 14 W/m².
- **A bias against observed weather could still persist.** ERA5 is published about 5 days after
  the date, too late to use at forecast time. Experiment 111's ERA5 result found no gain, which
  points the same way.
- **No demand forecast was fitted**, so there are no scores, tail calibration, or breakdowns.
