# Weather inputs by product

**Each weather product reaches the forecast at a different time, and an experiment must name which
product feeds each column at forecast time and which feeds it at training time.** The two are
usually different products. This page is a digest: it holds the table of what exists when, and the
traps that cost experiments a re-run. The long accounts are the [weather products
survey](../../background/weather-products-survey.md), the [data-sources
page](../../roadmap/data-sources.md#weather-data), the [NWP variable
conventions](../../architecture/nwp-variable-conventions.md), and the [ECMWF ENS known
issues](../../architecture/ecmwf-ens-known-issues.md).

## Availability at forecast time

| Product | What it is | Delay before use | Exists at forecast time? | Training rule |
|---|---|---|---|---|
| ECMWF ENS (51 members) | Ensemble forecast, stored by `delta_store.nwp` | A run is usable 9 hours after its `init_time` (`NWP_PUBLICATION_DELAY_HOURS`) | Yes: the production input | Train on ENS history, which starts 2024-04-01 on disk |
| ERA5 | ECMWF reanalysis, an estimate of past weather | About 5 days through the Climate Data Store | No: a ceiling at lag times | Version-1 plan: the training product for every variable except irradiance |
| CAMS solar radiation | Satellite-derived irradiance, point time-series API | 1 day: a request that names today is rejected | No: nothing about the day being forecast | The planned training product for irradiance |
| UKV (Met Office) | 2 km deterministic UK weather model | Three variants exist; see below | Only the live feed | Train on the product that will serve; never train on the Centre for Environmental Data Analysis (CEDA) archive and serve the live feed |
| GEFS, ICON-EU, IFS, AIFS-ENS, WeatherNext 3 | Further forecast products | Differs by product | Studies only | Not in production; see the weather theme |

- **A weather product that is "as accurate as ERA5" is a ceiling for demand, not a forecast.**
  ERA5 weather in place of ENS weather lowered the Primary demand quantile score by 0.10 points
  (exp 22; Primary demand, one validation year; provisional).
- **Name the weather source used at lag times in every write-up, and say whether it exists at
  forecast time** ([data-processing page](../data-processing.md#timestamps-and-availability); exp 37,
  38; lag-time sources; established).

## Training product: the rule

- **Train each XGBoost model on the history of the weather product that will serve it.** The arms
  are ENS-trained, ENS plus the other product as columns, and the other product alone. An
  ERA5-trained XGBoost model is only a reference ([weather theme](../themes/weather-products-and-blending/index.md#gotchas-and-implementation-details);
  exp 22; XGBoost models on Primary demand; provisional).
- **Pair a CAMS-trained XGBoost model with a stage 0 quantile mapping of ENS shortwave.** The
  pair scored 0.102 points better than the ERA5-trained arm for Primary demand, against 0.035
  without the mapping (exp 14; Primary demand; provisional, and it supersedes the advice of exp 9
  to keep ERA5).
- **Calibrate ENS temperature and irradiance before they drive a model trained on past weather.**
  Shifts of up to about 2 K and 20% were needed at five Primaries (exp 111; 5 Primaries; provisional,
  medium confidence).
- **"Better weather adds little" was measured only for an XGBoost model trained on ENS.** Do not
  read it as a result for an ERA5-trained XGBoost model ([weather
  theme](../themes/weather-products-and-blending/index.md#gotchas-and-implementation-details); exp
  22; ENS-trained XGBoost; provisional).
- **Training on ERA5 lowered the solar and wind farm error up to day 6, but for demand a long
  history did not carry over** ([weather theme](../themes/weather-products-and-blending/index.md);
  unchecked).

## ECMWF ENS

- **The step grid and the publication delay are owned by the architecture pages.** See the [step
  grid](../../architecture/nwp-variable-conventions.md#the-forecast-step-grid) and the [ECMWF ENS
  known issues](../../architecture/ecmwf-ens-known-issues.md) (code).
- **The ENS archive on disk starts at 2024-04-01.** Source Cooperative stages about 3 further years
  as raw GRIB1 (complete dates 2021-03-21 to 2024-03-31), and a pilot decoded them bit-exact. See
  the [training-history page](../../roadmap/training-history.md) for whether the full fetch has
  run (disk; pilot only, unchecked).
- **Instantaneous and period-ending variables differ.** See the [NWP variable
  conventions](../../architecture/nwp-variable-conventions.md) (code).
- **Drop an ensemble member that has a missing weather value, and refuse a run whose control member
  is incomplete** (exp 111; 5 Primaries; provisional).
- **Repeated ENS runs repeat each valid time about 15 times.** An ENS-trained model with calendar
  columns can memorise the calendar position, so include a control without them and group inner
  splits by valid time (exp 1, 3; ENS-trained XGBoost; provisional).

## ERA5

- **ERA5 arrives about 5 days late through the Climate Data Store.** A faster public Zarr copy
  (ARCO-ERA5) carries ERA5T at about a week's lag
  ([data-sources page](../../roadmap/data-sources.md#era5-which-access-route)) (unchecked).
- **The half-hourly file's :30 values are interpolated.** Each is the mean of the two neighbouring
  hourly values and contains the next hour's reading (exp 104; ERA5 half-hourly file; provisional).
- **ERA5T hours from the last few months are preliminary.** See the [five
  traps](../../roadmap/data-sources.md#five-traps-when-era5-hourly-fields-are-forecasting-model-inputs)
  (unchecked).
- **ERA5 describes past sunshine and wind worse than several current products.** CAMS beat ERA5 by
  3.7 to 4.1 points of capacity on past sunshine at the solar study's five headline row sets, an
  exploratory comparison ([training-history page](../../roadmap/training-history.md#which-estimate-of-past-weather-to-train-on);
  solar study, 5 row sets; provisional).

## CAMS

- **Use the point time-series API, not the gridded product.** The gridded product lags years
  behind ([data-sources page](../../roadmap/data-sources.md#cams-use-the-point-api-not-the-gridded-product))
  (unchecked).
- **The latency is 1 day.** A forecast job at 09:00 UTC gets nothing about the day it forecasts, so
  no intraday correction rests on CAMS (unchecked).
- **Values are accumulated irradiation in Wh/m²** and each is stamped at the end of its step. Set
  `time_reference` to `universal_time` (unchecked).
- **Snapshot each fetch.** CAMS recomputes a request against the latest software, so the same
  timestamp can return a different value later (unchecked).

## UKV

- **Three UKV variants are different products.** The CEDA archive, Open-Meteo's archive, and the
  live feed differ in lead, irradiance, temperature, and units
  ([data-sources page](../../roadmap/data-sources.md)) (unchecked).
- **Open-Meteo's UKV archive is the T+0 analysis.** It holds no forecast at any lead, so it
  compares with a reanalysis and does not measure forecast skill (unchecked).
- **The CEDA archive has no 100 m wind, lacks four hourly steps, and is licensed for
  non-commercial use.** UKV was upgraded on 2026-01-21 (exp 54, 57; CEDA archive; provisional).
- **UKV shortwave is instantaneous, whereas ENS and CAMS are means labelled at the period end.**
  Check the hour-of-day peak of every source (exp 53; UKV, ENS, CAMS; provisional).

## Other products

- **GEFS did not improve the sunshine anomaly**, and its raw anomaly was about 15% too large
  (exp 50; sunshine anomaly; provisional).
- **ICON-EU and IFS winds scored worse than the best guess** when fed alone to an ERA5-trained
  wind model: 2.5 and 1.3 points worse at days 1 and 2 (exp 52; wind model, days 1 and 2;
  provisional).
- **CERRA runs about 3 months behind real time.** It is deprioritised
  ([data-sources page](../../roadmap/data-sources.md#weather-data)) (unchecked).

## Blending experiments

- **Pass a shuffled-product placebo through every step**, and give a guard arm the variant it
  guards ([weather theme](../themes/weather-products-and-blending/index.md#gotchas-and-implementation-details);
  exp 52, 54; blending arms; provisional).
- **Feed the run the service would have.** Experiment 53 assumed a 3-hour UKV delivery time that no
  experiment has measured (exp 53; UKV; provisional).

Last checked against experiment 183.
