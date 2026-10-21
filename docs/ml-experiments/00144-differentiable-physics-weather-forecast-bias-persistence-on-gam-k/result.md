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
  Below 0.2 in every season ends the experiment, because a ridge correction on that error has no
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
    The ensemble's day-ahead temperature error against its own later analysis proxy has a daily
    standard deviation of only about 0.3 degrees Celsius and no memory beyond a day.
  - >-
    The only weather "truth" available at forecast time is a later forecast run, so this test
    measures how much ensemble runs revise each other, not how wrong they are against observations.
production_status: not-for-production
adopted_in: none
date: 2026-10-21
take_home_message: >-
  Errors in the day-ahead weather forecast do not carry over to the next day, so a correction that
  feeds recent weather errors into the demand forecast has nothing to work with. The Kalman filter
  and the error profile already capture the slow part of the demand error. Only an observed-weather
  source that arrives by forecast time could reopen the idea.
advice:
  - >-
    Do not build a demand correction from recent ensemble-minus-analysis-proxy weather errors: their
    lag-1 autocorrelation is at most 0.11 (step 1).
  - >-
    To test weather-bias persistence against observations, first find an observed source that is
    published before the forecast origin; ERA5 arrives about 5 days late (step 1 caveat).
code: ml-experiments/00144-differentiable-physics-weather-forecast-bias-persistence-on-gam-k/
---

**Take-home: errors in the day-ahead weather forecast do not carry over to the next day, so there is
nothing to feed a demand correction.** The brief set a stop rule at a lag-1 autocorrelation of 0.2,
and every season is well under it. No demand arm was fitted.

We asked whether a weather forecast that was too cold or too cloudy yesterday tends to be too cold
or too cloudy today. For each day, we took the ensemble mean forecast published the morning before,
and subtracted the weather that a live service would have for that day afterwards: the stitched
control members of the following runs ("analysis proxy", the one weather record available at
forecast time). We then correlated each day's error with the error `k` days later. The answer is
about zero, so the idea ends here.

A search of `docs/ml-experiments` found no earlier experiment that tested this directly.
The nearest evidence is experiment 111, where ERA5
residuals in the filter gained nothing (GAM-K-era5 minus GAM-K: -0.008, interval -0.024 to +0.008),
which agrees with the result here: recent weather errors carry no usable information beyond what
the filter already tracks.

## Lag-1 to lag-7 autocorrelation of the daily error

Mean over seven substations (D5, D9, D10, D11, D14, D16, D18) of the Pearson correlation. "Anomaly"
removes each calendar month's mean error first.

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

- The anomaly version gives the same picture (lag 1 from -0.01 to +0.09).
- The full lags 1 to 7 and both versions are in `work/autocorrelation.json`, printed by
  `e00144_autocorr.py`.

## Checks and caveats

- **Wind was not tested.** Experiment 111's ensemble store holds temperature and shortwave only, and
  GAM-K takes no wind input.
- **The proxy is a forecast.** The error is the revision between the day-ahead ensemble mean and the
  next run's control member (lead 9.5 to 33 hours). The daily temperature error has a standard
  deviation of about 0.3 degrees Celsius at D5 and irradiance about 14 W/m², small beside real
  weather errors. A bias against observed weather could still persist; ERA5 is published about 5
  days after the date, too late to use at forecast time.
- **No demand arm, gate, or scored fit was run**, so the 2 degrees Celsius injection gate was not
  needed. The curtailment, tail, and holiday breakdowns are not applicable.
- **Audit triggers:** none fired (no control, contrast, or scored arm).
- **Next:** a test against observed weather would need an observed source available at forecast
  time; no such source is in the project today.
