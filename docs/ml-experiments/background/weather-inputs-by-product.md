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

- **Each cell is a digest.** The owner page of each row is linked in the section below.
- **A weather product that is "as accurate as ERA5" is a ceiling for demand, not a forecast.**
  ERA5 weather in place of ENS weather lowered the Primary demand quantile score by 0.10 points
  (experiment 22).
- **Name the weather source used at lag times in every write-up, and say whether it exists at
  forecast time** ([data-processing page](../data-processing.md#timestamps-and-availability),
  experiments 37 and 38).

## Training product: the rule

- **Train each model on the history of the weather product that will serve it.** The arms are
  ENS-trained, ENS plus the other product as columns, and the other product alone. An
  ERA5-trained model is only a reference ([weather theme](../themes/weather-products-and-blending/index.md#gotchas-and-implementation-details),
  experiment 22).
- **Pair a CAMS-trained irradiance model with a stage 0 quantile mapping of ENS shortwave.** The
  pair scored 0.102 points better than the ERA5-trained arm for Primary demand, against 0.035
  without the mapping (experiment 14, which supersedes experiment 9's advice to keep ERA5).
- **Calibrate ENS temperature and irradiance before they drive a model trained on past weather.**
  Shifts of up to about 2 K and 20% were needed at five Primaries (experiment 111; medium
  confidence).
- **"Better weather adds little" was measured only for a model trained on ENS.** Do not read it as
  a result for an ERA5-trained model (experiment 22, [weather
  theme](../themes/weather-products-and-blending/index.md#gotchas-and-implementation-details)).
- **Training on ERA5 lowered the solar and wind farm error up to day 6, but for demand a long
  history did not carry over** ([weather theme](../themes/weather-products-and-blending/index.md)).

## ECMWF ENS

- **The step grid is uneven.** The grid steps 3 hours from 3 h to 144 h and 6 hours from 150 h to
  360 h, so days 6 to 10 sit on the coarse half
  ([NWP variable conventions](../../architecture/nwp-variable-conventions.md#the-forecast-step-grid)).
- **A 00Z run reaches our disk about 10:30 UTC.** The 9-hour delay is the earliest usable time of a
  slot and not the arrival time. A replay of a past `init_time` must apply the delay, because
  joining a run that landed afterwards is lookahead bias
  ([data-processing page](../data-processing.md#timestamps-and-availability)).
- **The ENS archive on disk starts at 2024-04-01.** Source Cooperative stages about 3 further years
  as raw GRIB1 (complete dates 2021-03-21 to 2024-03-31), and a pilot decoded them bit-exact. See
  the [training-history page](../../roadmap/training-history.md) for whether the full fetch has
  run.
- **Instantaneous and period-ending variables differ.** Radiation fields describe the interval
  before their timestamp, and wind components are interpolated before speed and direction are
  derived ([NWP variable conventions](../../architecture/nwp-variable-conventions.md)).
- **Drop an ensemble member that has a missing weather value, and refuse a run whose control member
  is incomplete** (experiment 111).
- **Repeated ENS runs repeat each valid time about 15 times.** An ENS-trained model with calendar
  columns can memorise the calendar position, so include a control without them and group inner
  splits by valid time (experiments 1 and 3).

## ERA5

- **ERA5 arrives about 5 days late through the Climate Data Store.** A faster public Zarr copy
  (ARCO-ERA5) carries ERA5T at about a week's lag
  ([data-sources page](../../roadmap/data-sources.md#era5-which-access-route)).
- **The half-hourly file's :30 values are interpolated.** Each is the mean of the two neighbouring
  hourly values and contains the next hour's reading (experiment 104).
- **ERA5T hours from the last few months are preliminary.** Record `expver` for every hour
  ([five traps](../../roadmap/data-sources.md#five-traps-when-era5-hourly-fields-are-forecasting-model-inputs)).
- **ERA5 describes past sunshine and wind worse than several current products.** CAMS beat ERA5 by
  3.7 to 4.1 points of capacity on past sunshine at the solar study's five headline row sets, an
  exploratory comparison ([training-history page](../../roadmap/training-history.md#which-estimate-of-past-weather-to-train-on)).
- **ERA5 is one of several estimates of past weather.** Older write-ups that say "ERA5-trained"
  predate the others (CAMS, UKV analysis, ICON-D2 for wind).

## CAMS

- **Use the point time-series API, not the gridded product.** The gridded product lags years
  behind ([data-sources page](../../roadmap/data-sources.md#cams-use-the-point-api-not-the-gridded-product)).
- **The latency is 1 day.** A forecast job at 09:00 UTC gets nothing about the day it forecasts, so
  no intraday correction rests on CAMS.
- **Values are accumulated irradiation in Wh/m²** and each is stamped at the end of its step. Set
  `time_reference` to `universal_time`.
- **Snapshot each fetch.** CAMS recomputes a request against the latest software, so the same
  timestamp can return a different value later.
- **Join power and CAMS on `(time - 1 µs).truncate("1h") + 1h`** ([data-processing
  page](../data-processing.md#timestamps-and-availability)).
- **A dark row is a half-hour in which CAMS irradiance is below 1 W/m².** The rule for MVA series
  is in the [data-processing page](../data-processing.md#decide-which-rows-are-demand).

## UKV

- **Three UKV variants are different products.** The CEDA archive, Open-Meteo's archive, and the
  live feed differ in lead, irradiance, temperature, and units
  ([data-sources page](../../roadmap/data-sources.md)).
- **Open-Meteo's UKV archive is the T+0 analysis.** It holds no forecast at any lead, so it
  compares with a reanalysis and does not measure forecast skill.
- **The CEDA archive has no 100 m wind, lacks four hourly steps, and is licensed for
  non-commercial use.** UKV was upgraded on 2026-01-21 (experiments 54 and 57).
- **UKV shortwave is instantaneous, whereas ENS and CAMS are means labelled at the period end.**
  Check the hour-of-day peak of every source (experiment 53).

## Other products

- **GEFS did not improve the sunshine anomaly**, and its raw anomaly was about 15% too large
  (experiment 50).
- **ICON-EU and IFS winds scored worse than the best guess** when fed alone to an ERA5-trained
  wind model: 2.5 and 1.3 points worse at days 1 and 2 (experiment 52).
- **CERRA runs about 3 months behind real time.** It is deprioritised
  ([data-sources page](../../roadmap/data-sources.md#weather-data)).

## Blending experiments

- **Pass a shuffled-product placebo through every step**, and give a guard arm the variant it
  guards ([weather theme](../themes/weather-products-and-blending/index.md#gotchas-and-implementation-details),
  experiments 52 and 54).
- **Feed the run the service would have.** Experiment 53 assumed a 3-hour UKV delivery time that no
  experiment has measured.

Last checked against experiment 183.
