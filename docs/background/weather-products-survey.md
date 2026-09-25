# Weather products surveyed for the weather-product studies

**This page is a survey, not a plan: it records which weather products could join two studies of
past weather and a planned study of forecasts (issue #810), and commits the project to ingesting
none of them.** The list of products the project ingests, plans to ingest, or is researching as a
possible source is the catalogue under [Weather data](../roadmap/data-sources.md#weather-data).

## What the project uses

**The live forecast uses one weather product, ECMWF ENS, and two more are planned.** The studies
score 13 products in total: the seven products in this table, four more Open-Meteo models from the
survey below, ICON-DREAM-EU from the reanalysis table below, and SARAH-3, which the
[catalogue](../roadmap/data-sources.md#weather-data) lists and this page does not. Every other
product on this page is surveyed only, except where the "Scored in" column says otherwise. "Roadmap
status" copies the product's row in the [Weather data](../roadmap/data-sources.md#weather-data)
catalogue; a version label such as v0.5 names a [roadmap milestone](../roadmap/index.md#milestones).

| Product | What it is | Roadmap status | What the project uses it for | Archive read from | Scored in |
|---|---|---|---|---|---|
| **ECMWF ENS** | The European Centre for Medium-Range Weather Forecasts' (ECMWF's) 51-member ensemble forecast | [✅ Ingested today](../roadmap/data-sources.md#weather-data) | The live forecast | [Dynamical.org](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/), a non-profit that republishes weather-model archives | [Sunshine](../studies/weather-products-for-past-solar.md#ecmwf-ens-a-longer-lead-forecast-than-any-other-product-on-this-page) |
| **ERA5** | ECMWF's reanalysis: a weather model re-run over the past and pulled towards observations | [🚧 Planned ingestion, milestone v0.5](../roadmap/data-sources.md#weather-data) | Planned: training history and capacity estimation ([why ERA5 scores worse than current products](../roadmap/data-sources.md#why-era5-describes-past-sunshine-and-wind-worse-than-most-current-weather-products)) | Open-Meteo's copy, for the studies; the planned ingest route is [still open](../roadmap/data-sources.md#era5-which-access-route) | [Sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md) |
| **CAMS** | The Copernicus Atmosphere Monitoring Service's satellite irradiance | [🚧 Planned ingestion, milestone v0.7](../roadmap/data-sources.md#weather-data) | Planned: offline capacity estimation only ([which CAMS route, and its traps](../roadmap/data-sources.md#cams-use-the-point-api-not-the-gridded-product)) | The CAMS Radiation Service point API | [Sunshine](../studies/weather-products-for-past-solar.md) |
| **ICON-EU** | The German weather service's (DWD's) ICON model, 6.5 km Europe configuration | [🔬 Research (v0.9, uncertain)](../roadmap/data-sources.md#weather-data), as Dynamical.org's copy | Studies only | Open-Meteo's Historical Forecast archive | [Sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md) |
| **UKV** | The Met Office's 2 km UK model | [🔬 Research (uncertain)](../roadmap/data-sources.md#weather-data), as the Met Office's AWS feed | Studies only | Open-Meteo's Historical Forecast archive | [Sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md) |
| **ICON-D2** | DWD's ICON model, 2 km central-Europe configuration | Not on the roadmap | Studies only | Open-Meteo's Historical Forecast archive | [Sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md) |
| **ICON global** | DWD's ICON model, global configuration | Not on the roadmap | Studies only | Open-Meteo's Historical Forecast archive | [Sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md) |

**CAMS describes past sunshine best, and UKV and ICON-D2 describe past wind best.** [Which weather
product best describes past sunshine?](../studies/weather-products-for-past-solar.md) scores eight
products from December 2022 to August 2026 and finds CAMS best by a wide margin, and ICON-D2 the
best of the weather models as served. The page also scores four more Open-Meteo models on a shorter
window, [from November 2024 to August 2026](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models).
[Which weather product best describes past wind?](../studies/weather-products-for-past-wind.md)
scores August 2024 to September 2026 and finds UKV and ICON-D2 best. CAMS publishes no wind, so the
wind study scores five of the products in the table above, and a sixth, ICON-DREAM-EU, on a slightly
shorter row set.

## Terms used on this page

**The survey's comparisons turn on how a forecast is run and archived, so these terms are defined
before any section uses them:**

- **Run** — one execution of a weather model from a start time, its initialisation, producing
  values for the hours ahead: "the 00 UTC run".
- **Lead time** — how far ahead of its run's start a value lies. "T+54 h" is the value 54 hours
  after the run started.
- **Fixed lead** — every product compared at the same lead time, so a difference between products
  is not a difference in how old each forecast was.
- **Whole run** — every lead time of a run, kept in an archive, as opposed to only the freshest
  run's value for each hour.
- **Hindcast** — a forecast re-run after the fact with a later model version.
- **Serving path** — the code that produces the live forecast each day, as opposed to training and
  offline analysis.
- **Global, direct, and diffuse irradiance** — global irradiance is all the sunlight reaching a
  horizontal surface. The direct beam is the part arriving straight from the sun, and diffuse is
  the rest, scattered by cloud and air. A solar-farm model needs the split, because tilted panels
  catch the two parts differently. Direct normal irradiance (DNI) is the direct beam measured facing
  the sun.
- **Period-ending** — how the meters label each half-hourly reading: with the time the half-hour
  ends. A product stamped at the middle of its averaging period needs shifting to match.
- **HRES** — ECMWF's single 9 km forecast from its Integrated Forecasting System (IFS). Since IFS
  Cycle 50r1 went live on 2026-05-12, ECMWF no longer runs a separate HRES: the single forecast is
  the ENS control member
  ([ECMWF](https://www.ecmwf.int/en/about/media-centre/news/2026/ifs-cycle-50r1-aifsv2-live),
  [plans](https://www.ecmwf.int/en/about/media-centre/focus/2024/plans-high-resolution-forecast-hres-and-ensemble-forecast-ens)).
  Open-Meteo still calls the product "IFS HRES 9 km", and so does this page. AIFS is ECMWF's
  machine-learned Artificial Intelligence Forecasting System.

## How to read the Open-Meteo rows

**Open-Meteo, the service the studies read, re-serves other centres' forecasts, so a start date, a
latency, or a direct/diffuse split read through Open-Meteo may belong to the archive rather than to
the producer.** The tables below say which. The production service does not read Open-Meteo. Four
conventions apply to every Open-Meteo row:

- **Start dates are the first hour with a non-null value** at Lincoln (53.23°N, 0.54°W), queried on
  2026-09-23. Lincoln stands in for the trial area in the East Midlands, so a start date, a null
  series, or a ratio measured there is a result for that one point. A user elsewhere in Great
  Britain should re-query at their own sites, and the "Covers GB" column says where coverage was
  checked on the grid rather than at Lincoln. ICON-D2, for example, has data at Lincoln but none in
  the west of Great Britain: the [map of ICON-D2's and AROME France's
  domains](../roadmap/data-sources.md#what-comparing-weather-products-on-the-trial-areas-solar-farms-found)
  on the data sources page shows where.
- **Open-Meteo's UKV archive is a backfill before 12 August 2024**, so a UKV start date on this page
  is not the start of UKV's own run history ([what the backfill is and how it changes
  scores](../roadmap/data-sources.md#open-meteos-ukv-archive-is-the-t0-analysis-and-half-of-it-is-backfill)).
- **Latency is the time from a run's initialisation to the whole run being available on
  Open-Meteo**, read from that model's `meta.json` for its latest run on the morning of 2026-09-23:
  00 UTC for the 6-hourly models, a later run for AROME, the Danish Meteorological Institute (DMI),
  and the Royal Netherlands Meteorological Institute (KNMI). Each figure is one run, not an
  average.
- **"Native" means the producer publishes the direct or the diffuse field, and "separation" means
  Open-Meteo derives both from global irradiance** with the Razo–Müller–Witwer separation model
  ([Open-Meteo ECMWF documentation](https://open-meteo.com/en/docs/ecmwf-api)). A split made by
  separation carries no information the global field lacks.
- **The Licence column gives the licence of the data.** Open-Meteo's free API is for non-commercial
  use only ([terms](https://open-meteo.com/en/terms)), so commercial use needs a paid Open-Meteo
  plan.

## Which archives keep whole past forecast runs

**Issue #810 compares models at a fixed lead, so it needs an archive that keeps whole past runs or
serves a fixed lead.** A model run every 3 hours and a model run every 6 hours reach a given hour at
different ages, so a series that always takes the freshest run compares delivery cadence as well as
forecast quality. Open-Meteo publishes three archives of the same forecasts, and Dynamical.org
publishes a fourth kind:

- **The Historical Forecast archive stitches the first hours of each successive run** into one
  continuous series, so a product's lead follows its run frequency rather than being chosen
  ([Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)). The two study
  pages score this archive.
- **The Previous Runs archive serves each variable at a fixed offset of 1 to 7 days**
  ([Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)). The archive starts on
  2024-01-19 at the earliest, and serves day 1 only for UKV, ICON-D2, the Met Office global 10 km
  model, and the HARMONIE-AROME and AROME models (table below).
- **The Single Runs archive keeps whole runs** ([Single Runs
  API](https://open-meteo.com/en/docs/single-runs-api)), but only from the 2026-04-02 00 UTC run:
  `ecmwf_ifs025` and `ncep_gfs013` both reject `run=2026-04-01T00:00`. ECMWF IFS HRES 9 km is the
  exception, below.
- **Dynamical.org keeps whole runs as Icechunk (Zarr) repositories**, listed in its SpatioTemporal
  Asset Catalog (STAC). The start dates below come from Dynamical.org's catalogue pages, checked
  against each store's `init_time` coordinate where the date matters here. The longest-running
  Dynamical.org stores reach back to 2020-10-01 for GEFS, the US National Oceanic and Atmospheric
  Administration's (NOAA's) Global Ensemble Forecast System, and to 2021-05-01 for GFS, NOAA's
  Global Forecast System.

**Open-Meteo's Single Runs archive keeps ECMWF IFS HRES 9 km from 2024-03-14, two years earlier than
the other models.** The model `ecmwf_ifs` rejects `run=2024-03-13T00:00` and serves the 2024-03-14
00 UTC run with 240 hourly steps. Open-Meteo calls the runs from 2024-03-14 "IFS Cycle 49R1
hindcasts". The March to June 2024 runs sampled are 00 and 12 UTC only, and 06 UTC runs are present
from 2024-08-10. Runs from 2026-05-12 06 UTC use IFS Cycle 50r1 ([what that upgrade changed](../roadmap/data-sources.md#nwp-model-upgrades-since-2019)).

**Dynamical.org stops serving data.dynamical.org on 2026-09-30, so any URL on that host has to move
to a supported access pattern.** Every Dynamical.org catalogue page carries the notice
"data.dynamical.org access ends September 30, 2026".

**Which offsets the Previous Runs archive serves is set by the archive, not by the model's
horizon.** UKV's runs in the Single Runs archive hold 55 hourly steps, to T+54 h, and ICON global's
runs reach 180 h, yet the Previous Runs archive serves UKV at day 1 only and ICON global to day 6
only. Any lead-matched comparison is therefore capped at day 1 where one arm is UKV, ICON-D2, the
Met Office global 10 km model, or a HARMONIE-AROME or AROME model.

| Model on Open-Meteo | First day-1 global irradiance at Lincoln | Offsets served |
|---|---|---|
| ICON-D2 | 2024-01-19 | Day 1 |
| Météo-France AROME France | 2024-01-19 | Day 1 |
| KNMI HARMONIE-AROME Europe | 2024-06-27 | Day 1 |
| DMI HARMONIE-AROME DINI | 2024-06-29 | Day 1 |
| UKV | 2024-08-06 | Day 1 |
| Met Office global 10 km | 2025-01-03 | Day 1 |
| Météo-France ARPEGE (Europe and World) | 2024-01-19 | Days 1–3 |
| ICON-EU | 2024-01-19 | Days 1–4 |
| ICON global | 2024-01-19 | Days 1–6 |
| NOAA GFS | 2024-01-19 | Days 1–7 |
| ECMWF IFS 0.25° | 2024-03-06 | Days 1–7 |
| ECMWF AIFS Single | 2025-02-17 | Days 1–7 |
| ECMWF IFS HRES 9 km | 2025-10-01 | Days 1–7 |

| Whole-run archive | Runs kept | First run | Radiation and wind in the archive |
|---|---|---|---|
| [NOAA GEFS 35-day](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/) (Dynamical.org) | 31 members, 00 UTC only, to 840 h | 2020-10-01 | Global irradiance; 10, 80, and 100 m wind |
| [NOAA GFS](https://dynamical.org/catalog/noaa-gfs-forecast/) (Dynamical.org) | 4 a day, to 384 h | 2021-05-01 | Global irradiance; 10, 80, and 100 m wind |
| [Google WeatherNext 2](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0) (Earth Engine) | 64 members, 4 a day, to 15 days | 2022-01-01 | No radiation; 10 and 100 m wind |
| ECMWF IFS HRES 9 km ([Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api)) | 00 and 12 UTC at first; 06 UTC from 2024-08-10 | 2024-03-14 | Global and native direct irradiance; 10, 80, and 100 m wind |
| [ECMWF IFS ENS](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) (Dynamical.org) | 51 members, 00 UTC only, to 360 h (906 runs to 2026-09-23) | 2024-04-01 | Global irradiance; 10 and 100 m wind |
| [ECMWF AIFS Single](https://dynamical.org/catalog/ecmwf-aifs-single-forecast/) (Dynamical.org) | 4 a day, to 360 h | 2024-04-01; radiation and 100 m wind from the 2025-02-24 06 UTC run | Global irradiance; 10 and 100 m wind |
| [ECMWF AIFS-ENS](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/) (Dynamical.org) | 4 a day, to 360 h | 2025-07-02 | Global irradiance; 10 and 100 m wind |
| [DWD ICON-EU](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/) (Dynamical.org) | 00, 06, 12, and 18 UTC, to 120 h | 2026-02-10 | Native direct and diffuse irradiance; 10 m wind only |
| [CEDA NWP-UKV, the Centre for Environmental Data Analysis](https://catalogue.ceda.ac.uk/uuid/78f23c539d304591b137cf986b69a525/) | Up to 8 a day, T+0 to T+120 | 2016-03-16 | Radiation not listed on the catalogue record; access by application |

## The surveyed products

**None of the products from here on is used by the project, except where its Roadmap status says
so.** The tables give each product's roadmap status in words beside the catalogue's symbol, and
the study pages that score it. "Grid" is the producer's native grid spacing, with the serving grid
in brackets where the two differ. "Past runs" names the archive that keeps whole runs or a fixed
lead, from the section above.

**Of the forecast models in the tables below whose solar variables were established, only ECMWF's
IFS 9 km forecasts, GFS, the Met Office global model, ICON-EU, and Google WeatherNext 3 serve a
producer's own direct or diffuse field.** For the other models Open-Meteo serves, Open-Meteo makes
the split by separation, except for DMI HARMONIE-AROME, whose served direct field is probably not a
direct beam at all (note below the first table). The Dynamical.org and Google products serve global
irradiance only, or no radiation. UKV, ICON-D2, and ICON global, which the studies score, also serve
the producer's own direct field, as do UKV and MOGREPS-UK on the Met Office's own feed — see [which
sources carry which irradiance
components](../roadmap/data-sources.md#which-sources-carry-which-irradiance-components).

**Among the reanalyses surveyed, CERRA and COSMO-R6G2 carry the producer's own direct field, as
ERA5 and ICON-DREAM-EU, both already scored, do.**

### Forecast models

| Product | Roadmap status | Scored in | Producer and method | Grid | Covers GB | Archive start | Solar variables | Wind heights | Runs, horizon, step | Latency | Past runs | Access | Licence | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ECMWF IFS ENS** (Dynamical.org) | [✅ Ingested today](../roadmap/data-sources.md#weather-data) | [Sunshine](../studies/weather-products-for-past-solar.md#ecmwf-ens-a-longer-lead-forecast-than-any-other-product-on-this-page) only | ECMWF, 51-member physics ensemble | 9 km (0.25° served) | Yes (global) | 2024-04-01 to present; back-fill in progress ([catalogue row](../roadmap/data-sources.md#weather-data)) | Global only (`ssrd`) | 10, 100 m | 4 a day from ECMWF, Dynamical.org keeps 00 UTC; 360 h; 3-hourly to 144 h, then 6-hourly | Not established | Whole runs | Icechunk (Zarr) | CC BY 4.0 and ECMWF terms | [Dynamical.org](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) |
| **ECMWF IFS ENS 9 km Europe** (Open-Meteo, `ecmwf_ifs_europe_ensemble`) | Not on the roadmap (the catalogue's ENS row is the Dynamical.org feed) | Not scored | ECMWF, 51-member physics ensemble, European domain | 9 km, O1280 (served natively) | Data at Lincoln | 2026-09-23 01 UTC at Lincoln, asking for 7 past days on 2026-09-23 | Global; direct native (ECMWF's `fdir`, per Open-Meteo's downloader) | Not checked | 00 and 06 UTC; 15 days; hourly to 90 h, 3-hourly to 144 h, then 6-hourly | Not established | None beyond the latest runs | Open-Meteo Ensemble API | CC BY 4.0 ([ECMWF's licence and delivery terms](../roadmap/data-sources.md#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost)) | [Open-Meteo](https://open-meteo.com/en/docs/ensemble-api), [downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/EcmwfEcpds/EcmwfEcpdsVariable.swift) |
| **ECMWF IFS HRES 9 km** (`ecmwf_ifs`) | Not on the roadmap | [sunshine](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models) only, on a shorter window | ECMWF, global physics model | 9 km (served natively) | Yes (global) | 2017-01-01, complete to 2026-09-21 | Global; direct native (ECMWF's `fdir`, per Open-Meteo's downloader); diffuse = global − direct | 10, 80, 100 m | 00 and 12 UTC to 15 days, 06 and 18 UTC to 144 h; hourly to 90 h, 3-hourly to 144 h, then 6-hourly | 7.0 h (Open-Meteo) | Single Runs from 2024-03-14; Previous Runs from 2025-10-01 | Open-Meteo | CC BY 4.0 ([ECMWF's licence and delivery terms](../roadmap/data-sources.md#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost)) | [Open-Meteo](https://open-meteo.com/en/docs/ecmwf-api), [downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/EcmwfEcpds/EcmwfEcpdsVariable.swift) |
| **ECMWF IFS 0.25°** (`ecmwf_ifs025`) | Not on the roadmap | Not scored | ECMWF open-data subset of the same run | 9 km (0.25° served) | Yes (global) | Radiation and 100 m wind 2024-03-05; 10 m wind 2024-02-03 | Global; direct and diffuse by separation | 10, 100 m | 00 and 12 UTC to 15 days, 06 and 18 UTC to 144 h; 3-hourly to 144 h, then 6-hourly | 7.8 h (Open-Meteo) | Previous Runs from 2024-03-06; Single Runs from 2026-04-02 | Open-Meteo | CC BY 4.0 | [Open-Meteo](https://open-meteo.com/en/docs/ecmwf-api) |
| **ECMWF AIFS Single** | Not on the roadmap (the catalogue's AIFS-ENS row is the ensemble) | Not scored | ECMWF, machine-learned deterministic model; v1 operational from the 2025-02-25 06 UTC run, adding `ssrd` and 100 m wind | N320, about 0.25° (0.25° served) | Yes (global) | Open-Meteo 2025-02-17; Dynamical.org radiation from 2025-02-24 06 UTC | Global only (`ssrd`); Open-Meteo's direct is by separation | 10, 100 m | 4 a day; 360 h; 6-hourly | 5.7 h (Open-Meteo) | Whole runs (Dynamical.org); Previous Runs from 2025-02-17 | Open-Meteo; Icechunk (Zarr) | CC BY 4.0 and ECMWF terms | [Dynamical.org](https://dynamical.org/catalog/ecmwf-aifs-single-forecast/), [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+Single+v1) |
| **ECMWF AIFS-ENS** | [🔬 Research (v2.1, uncertain)](../roadmap/data-sources.md#weather-data) | Not scored | ECMWF, machine-learned ensemble | 0.25° | Yes (global) | 2025-07-02 | Global only (`ssrd`) | 10, 100 m | 4 a day; 360 h; 6-hourly | Not established | Whole runs | Icechunk (Zarr) | CC BY 4.0 and ECMWF terms | [Dynamical.org](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/) |
| **DWD ICON-EU** (Dynamical.org) | [🔬 Research (v0.9, uncertain)](../roadmap/data-sources.md#weather-data) | Both studies, through Open-Meteo's copy ([sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md)) | DWD, regional physics model | 6.5 km (0.0625° served) | Yes | 2026-02-10 | Direct and diffuse native | 10 m only in Dynamical.org's store; Open-Meteo serves 80 and 120 m | Every 3 hours on Open-Meteo, Dynamical.org keeps 00, 06, 12, and 18 UTC; 120 h; hourly to 78 h | Not established | Whole runs (Dynamical.org); Previous Runs from 2024-01-19 (Open-Meteo) | Icechunk (Zarr); Open-Meteo | CC BY 4.0 | [Dynamical.org](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/), [Open-Meteo](https://open-meteo.com/en/docs/dwd-api) |
| **NOAA GFS** (`ncep_gfs013` for surface fields, `ncep_gfs025` for 80 and 100 m wind) | Not on the roadmap | Not scored | The National Centers for Environmental Prediction of the US National Oceanic and Atmospheric Administration (NOAA NCEP), global physics model | ~13 km (0.11° served for surface fields, 0.25° for 80 and 100 m wind) | Yes (global) | 2021-03-23 | Global; diffuse from NCEP's visible-band diffuse flux (`VDDSF`), not a broadband field; direct = global − that diffuse | 10, 80, 100 m | 4 a day; 16 days | 5.5 h at 0.11°, 6.5 h at 0.25° (Open-Meteo) | Previous Runs from 2024-01-19; whole runs on Dynamical.org from 2021-05-01 | Open-Meteo; Icechunk (Zarr) | Public domain from NOAA; CC BY 4.0 as served by Dynamical.org and Open-Meteo | [NOAA EMC](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php), [Open-Meteo](https://open-meteo.com/en/docs/gfs-api), [downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Gfs/GfsVariableDownloadable.swift), [Dynamical.org](https://dynamical.org/catalog/noaa-gfs-forecast/) |
| **NOAA GEFS 35-day** (Dynamical.org) | Not on the roadmap | Not scored | NOAA NCEP, 31-member ensemble | 0.25° to 240 h, 0.5° after | Yes (global) | 2020-10-01 | Global only | 10, 80, 100 m | 00 UTC; 840 h; 3-hourly to 240 h, then 6-hourly | Not established | Whole runs | Icechunk (Zarr) | Public domain from NOAA; CC BY 4.0 on Dynamical.org | [Dynamical.org](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/) |
| **Météo-France ARPEGE Europe** | Not on the roadmap | [sunshine](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models) only, on a shorter window | Météo-France, stretched-grid global model | ~5 km over Europe (0.1° served) | Bounds of the Europe extract not checked; data present at Lincoln | Global irradiance and 10 m wind 2022-11-09; 100 m wind 2023-06-05 | Global; direct and diffuse by separation | 10, 80, 100 m | 4 a day; 4 days | 3.5 h (Open-Meteo) | Previous Runs from 2024-01-19 | Open-Meteo | Météo-France's own licence, per Open-Meteo; CC BY 4.0 on Open-Meteo | [Météo-France](https://meteofrance.com/actualites-et-dossiers/comprendre-la-meteo/les-modeles-de-prevision-meteo), [Open-Meteo](https://open-meteo.com/en/docs/meteofrance-api), [licences](https://open-meteo.com/en/licence) |
| **Météo-France ARPEGE World** | Not on the roadmap | Not scored | The same ARPEGE run, global extract | ~5 km over Europe (0.25° served) | Yes (global) | 2023-12-27 | Global; direct and diffuse by separation | 10, 100 m | 4 a day; 4 days | 4.0 h (Open-Meteo) | Previous Runs from 2024-01-19 | Open-Meteo | As ARPEGE Europe | [Open-Meteo](https://open-meteo.com/en/docs/meteofrance-api) |
| **Météo-France AROME France** | Not on the roadmap | Not scored | Météo-France, convection-permitting regional model | 1.3 km (0.025° served) | South of 55.4°N only | 2023-12-27 | Global; direct and diffuse by separation | 10, 80, 100 m | 8 a day; 2 days | 2.7 h (Open-Meteo, 03 UTC run) | Previous Runs from 2024-01-19 | Open-Meteo | As ARPEGE Europe | [Open-Meteo](https://open-meteo.com/en/docs/meteofrance-api), [domain](https://api.open-meteo.com/data/meteofrance_arome_france0025/static/meta.json) |
| **DMI HARMONIE-AROME DINI** | Not on the roadmap | [sunshine](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models) only, on a shorter window | The Danish Meteorological Institute (DMI) for the United Weather Centres West (UWC-West) consortium; HARMONIE-AROME is the convection-permitting regional model several European weather services share | 2 km (Open-Meteo's figure) | Yes (39.7–62.7°N, 25.4°W–40.1°E) | 2024-06-28 | Global; a field Open-Meteo serves as direct, probably not DMI's direct beam (note below) | 10, 50, 100, 150, 250, 350, 450 m (Open-Meteo interpolates 80 m) | 8 a day; 2.5 days | 2.8 h (Open-Meteo, 06 UTC run) | Previous Runs from 2024-06-29 | Open-Meteo | CC BY 4.0 | [Open-Meteo](https://open-meteo.com/en/docs/dmi-api), [domain](https://api.open-meteo.com/data/dmi_harmonie_arome_europe/static/meta.json), [downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Dmi/DmiDownloader.swift) |
| **KNMI HARMONIE-AROME Europe** | Not on the roadmap | [sunshine](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models) only, on a shorter window | The Royal Netherlands Meteorological Institute (KNMI) for UWC-West | 5.5 km, as KNMI publishes the grid | Yes (39.7–62.6°N, 25.2°W–38.8°E) | Global irradiance 2024-06-26; 100 m wind 2024-10-09 | Global; direct and diffuse by separation | 10 to 300 m, including 80 and 100 m | Hourly; 2.5 days | 2.6 h (Open-Meteo, 06 UTC run) | Previous Runs from 2024-06-27 | Open-Meteo | CC BY 4.0 | [Open-Meteo](https://open-meteo.com/en/docs/knmi-api), [domain](https://api.open-meteo.com/data/knmi_harmonie_arome_europe/static/meta.json) |
| **Met Office global 10 km** (`ukmo_global_deterministic_10km`) | Not on the roadmap ([why no request has been made](../roadmap/data-sources.md#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost)) | Not scored | Met Office Unified Model, global | ~10 km (0.09° served) | Yes (global) | Direct and 10 m wind 2022-03-01; global irradiance from 2025-01-02 | Global; direct native; diffuse = global − direct | 10 m only | 4 a day; 7 days (Open-Meteo) | 8.4 h (Open-Meteo) | Previous Runs from 2025-01-03 | Open-Meteo; AWS | CC BY-SA 4.0 (AWS) | [Open-Meteo](https://open-meteo.com/en/docs/ukmo-api) |
| **Google WeatherNext 3** | [🔬 Research (after v2, unlikely)](../roadmap/data-sources.md#weather-data) | Not scored | Google DeepMind, machine-learned 64-member ensemble | 0.1° | Yes (global) | 2026-01-01, with 2024 and 2025 being back-filled | Global; direct native (`fdir`); diffuse = global − direct | Not checked | 4 a day to 15 days, plus hourly interim runs to 48 h; hourly | Not established | Not established | Access request; members only in the Cloud Storage Zarr store | CC BY 4.0 once at least 1 h old; real-time data under Google DeepMind's experimental terms | [model specs](https://developers.google.com/weathernext/guides/models), [access](https://developers.google.com/weathernext/guides/access-forecast) |
| **Google WeatherNext 2** | Not on the roadmap (the catalogue's Google row is WeatherNext 3) | Not scored | Google DeepMind, machine-learned 64-member ensemble; Google marks the data experimental, "not intended, validated, or approved for real world use" | 0.25° | Yes (global) | 2022-01-01 to 2026-09-22 in the Earth Engine catalogue | None | 10, 100 m | 4 a day; 15 days; 6-hourly | About 7.5 h (Google's dissemination schedule) | Whole runs | Earth Engine, BigQuery, Cloud Storage Zarr; all need an access request | CC BY 4.0 for data over 48 h old; real-time data under Google DeepMind's experimental terms | [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0) |
| **CEDA NWP-UKV** (the UK's Centre for Environmental Data Analysis) | Not on the roadmap (the catalogue's UKV row is the Met Office's AWS feed: 🔬 Research (uncertain)) | Not scored (the studies score Open-Meteo's UKV) | Met Office UKV as archived at CEDA; statistically different from the UKV served live on AWS and Open-Meteo, so not interchangeable with it for training and inference | ~0.018° (CEDA's figure) | Yes | 2016-03-16 to present | Not listed on the catalogue record | Surface and pressure levels | Up to 8 a day; 120 h | Archive, ongoing | Whole runs | CEDA, by application | Met Office licence via CEDA | [CEDA](https://catalogue.ceda.ac.uk/uuid/78f23c539d304591b137cf986b69a525/) |
| **CEDA Met Office global** (`global-grib`) | Not on the roadmap | Not scored | Met Office global model, raw output | Not established | Yes (global) | Directories from 2016-03 to 2026-09 | Not established; the README needs a login | Not established | Not established | Archive | Not established | CEDA, by application | Not established | [CEDA](https://data.ceda.ac.uk/badc/ukmo-nwp/data/) |

"Latency" in this table is the time from a run's start to the whole run being available on the
named service. MARS is ECMWF's Meteorological Archival and Retrieval System.

**The low direct share Open-Meteo serves for DMI HARMONIE-AROME probably comes from Open-Meteo's
field mapping rather than from DMI's model.** Direct over global irradiance at Lincoln was 0.24,
0.30, and 0.18 in August 2024, June 2025, and June 2026, against 0.46 to 0.62 for KNMI
HARMONIE-AROME, UKV, and ECMWF IFS HRES in the same months. Open-Meteo's
[downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Dmi/DmiDownloader.swift)
serves DMI's "downward short-wave radiation flux" as direct and leaves DMI's "direct solar exposure"
field unused, with a comment that the "direct solar exposure" field "seems to be DNI". The mapping
has not been checked against DMI's own files.

### Reanalyses, hindcasts, and satellite retrievals

| Product | Roadmap status | Scored in | Producer and method | Grid | Covers GB | Period | Solar variables | Wind heights | Time step | Latency | Access | Licence | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CERRA** | [🔬 Research (deprioritised)](../roadmap/data-sources.md#weather-data) | Downloaded for the studies, not yet scored | Copernicus regional reanalysis, HARMONIE-ALADIN with 3D-Var, forced by ERA5 | 5.5 km, 1069 × 1069 points | Yes (European domain) | 1984-09-01 to 2026-06-30 on the CDS on 2026-09-23 | Global and direct native, forecast fields only, accumulated | 10 m; 15, 30, 50, 75, 100, 150 to 500 m | 3-hourly analyses; hourly values from forecast leads 1–3 h | About 12 weeks on 2026-09-23 | Copernicus Climate Data Store (CDS); no area cropping, so every field is the whole domain | CC BY 4.0 | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels), [user guide](https://confluence.ecmwf.int/x/WFQ7E), [Ridal et al. (2024)](https://doi.org/10.1002/qj.4764) |
| **COSMO-R6G2** | Not on the roadmap | Not scored | DWD, COSMO regional reanalysis forced by ERA5 | 0.055° (~6 km), EURO-CORDEX EUR-11 domain | Yes | 2002-01 to 2025-12 | Global and direct native, hourly means | 10 m; 100, 150, 200 m | Hourly means stamped at :30 | Yearly batches: the 2025 files were posted on 2026-05-11 | HTTPS, monthly NetCDF | CC BY 4.0 | [DWD](https://opendata.dwd.de/climate_environment/REA/COSMO_R6G2/), [paper](https://asr.copernicus.org/articles/22/149/2026/) |
| **ICON-DREAM-EU** | Not on the roadmap | Both studies: [sunshine](../studies/weather-products-for-past-solar.md), [wind](../studies/weather-products-for-past-wind.md) | DWD, ICON reanalysis, EU nest | 6.5 km | Expected (ICON's EU nest); grid file not checked | 2010-01 to 2026-08 | Direct and diffuse native, hourly means | 10 m; otherwise model levels only, so a fixed height needs vertical interpolation | Hourly | The August 2026 file was posted on 2026-09-07 (DWD states 2–3 months) | HTTPS, monthly GRIB2 | CC BY 4.0 | [DWD](https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-EU/) |
| **NORA3** | Not on the roadmap | Not scored | MET Norway, HARMONIE-AROME hindcast downscaling ERA5 | 3 km | Yes, checked on the grid | 1958 to 2026-08-31 | Global only | 10 m; 20, 50, 100, 250, 500, 750 m, as grid-relative components needing rotation | Hourly, from leads 3–9 h of each 6-hourly run | About 3 weeks | THREDDS / OPeNDAP | CC BY 4.0 | [THREDDS](https://thredds.met.no/thredds/catalog/nora3/catalog.html), [Solbrekke et al. (2021)](https://doi.org/10.5194/wes-6-1501-2021) |
| **MERRA-2** | Not on the roadmap | Not scored | NASA's Global Modeling and Assimilation Office, Goddard Earth Observing System (GEOS) global reanalysis | 0.5° × 0.625° | Yes (global) | 1980 to present | Global only (`SWGDN`) | 2, 10, 50 m | Hourly means stamped at :30 | About 3 weeks after month end | GES DISC, Earthdata login | NASA open data | [radiation](https://disc.gsfc.nasa.gov/datasets/M2T1NXRAD_5.12.4/summary), [wind](https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary) |
| **ERA5-Land** | Not on the roadmap | Not scored | ECMWF, land component of ERA5 re-run with ERA5 forcing | 0.1° | Yes | 1950 to 2026-09-17 | Global only | 10 m only | Hourly | About 6 days | CDS | CC BY 4.0 | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) |
| **CERRA-Land** | Not on the roadmap | Not scored | Copernicus land-surface reanalysis on CERRA's grid | 5.5 km | Yes (European domain) | 1984-09 to 2025-12-31 | Global only | None | Leads 1–3 h | Not checked | CDS | CC BY 4.0 | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-land) |
| **NOAA GFS and GEFS analyses** (Dynamical.org) | Not on the roadmap | Not scored | The first hours of each GFS or GEFS run, concatenated | 0.25° | Yes (global) | GFS 2021-05-01; GEFS 2000-01-01 (reforecast to 2019, operational after) | Global only | 10, 80, 100 m | GFS hourly; GEFS 3-hourly | Not established | Icechunk (Zarr) | Public domain from NOAA; CC BY 4.0 on Dynamical.org | [GFS](https://dynamical.org/catalog/noaa-gfs-analysis/), [GEFS](https://dynamical.org/catalog/noaa-gefs-analysis/) |
| **HelioClim-3** | Not on the roadmap | Not scored | Transvalor and Mines Paris – PSL, Heliosat-2 retrieval from Meteosat | About 5 km in Europe | Yes | 2004-02 to real time | Global; direct and diffuse by decomposition | None | 15 min | "A few minutes after image reception" | SoDa web service | Paid after 2006 | [SoDa](https://www.soda-pro.com/help/helioclim/helioclim-3-overview) |
| **Solcast** | Not on the roadmap | Not scored | Solcast, multi-satellite retrieval | 90 m (downscaled) | Yes | 2007-01 to 7 days ago | Global, direct normal, diffuse, tilted | Included; heights not listed | 5–60 min | 7 days in the historical API; Solcast's live API is separate | Commercial API | Paid | [Solcast](https://solcast.com/historical-weather-api) |

"Latency" in this table is how far the archive lags real time.

### Weather-station observations

**The studies also read one source of station observations, the Met Office's MIDAS Open, which has
no row in the tables above because it is a set of points rather than a gridded product.** MIDAS is
the Met Office Integrated Data Archive System. MIDAS Open is its open release, published by the
Centre for Environmental Data Analysis (CEDA) under the [Open Government Licence
v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). The release read
here is `dataset-version-202607`, quality-control version 1. The release covers 2017-01-01 to
2025-12-31. CEDA releases MIDAS Open once a year, so the files hold only past weather. The
[catalogue](../roadmap/data-sources.md#weather-data) lists no station source, and the project
ingests none. See [the catalogue's summary of what the station arms
found](../roadmap/data-sources.md#met-office-weather-station-observations-midas-open).

**The studies downloaded 10 of the 86 radiation stations whose records overlap their years, and 38
hourly-weather stations.** The lists were typed by hand, and one distance rule reproduces them: a
station is included if its record runs into 2025 or later and it lies within 100 km of at least one
of the nine anonymised study sites. That rule gives the 10 radiation stations and 37 of the 38
hourly-weather stations. The 38th station's record ends in 2024, and the download holds it although
it fails the record test. The other 76 radiation stations are all 107 km or more from every solar
farm. See [how the stations were
chosen](../studies/weather-products-for-past-solar.md#how-the-stations-were-chosen).

**The variables are hourly global irradiation, 10 m wind, and air temperature.**

- **Global horizontal irradiation** is an hourly total in kJ m⁻² over the hour ending at the
  timestamp, in UTC. Dividing by 3.6 gives the hour's mean irradiance in W m⁻². The diffuse and
  direct columns are empty at all 10 stations.
- **10 m wind speed and direction** are a 10-minute mean over the window from 20 to 10 minutes
  before the timestamp ("HH-20 to HH-10" in the Met Office's notation), with speeds in whole knots
  and directions in steps of 10 degrees. Only 18 of the 38 stations report an hourly wind speed. The
  wind study did not find the averaging window documented for the 4 stations that report the
  automatic-weather-station hourly message type.
- **Air temperature** is a spot reading at the timestamp, reported at all 38 stations. Some stations
  report only once a day.

**Measured against gridded products, a station's irradiance scores between CAMS and ERA5 on
sunshine, and a station's 10 m wind scores worse than ERA5's on wind.** For six solar farms in
Lincolnshire, all sharing one radiation station 17 to 31 km away, the station's irradiance and
temperature score worse than CAMS and better than ERA5, and adding the station's irradiance to CAMS
lowers CAMS's error. The figures and intervals are in [Met Office weather-station observations as a
stand-in for a gridded
product](../studies/weather-products-for-past-solar.md#met-office-weather-station-observations-as-a-stand-in-for-a-gridded-product).
For three wind farms, each farm's nearest anemometer, 6 to 18 km away, gives a larger error than
ERA5's 10 m wind, and lowers UKV's error when added to it. The figures are in [One nearby 10 m
weather station trails ERA5's 10 m wind on its own, and lowers UKV's error when added to
it](../studies/weather-products-for-past-wind.md#one-nearby-10-m-weather-station-trails-era5s-10-m-wind-on-its-own-and-lowers-ukvs-error-when-added-to-it).

**Every station result shares four limits.**

- **The data end on 2025-12-31.** The station arms therefore score fewer hours than the main arms
  and end before the Met Office's UKV upgrade of January 2026. See [what the station files hold, and
  where they
  end](../studies/weather-products-for-past-solar.md#what-the-station-files-hold-and-where-they-end).
- **The quality-control flags are kept as delivered, and no row is dropped on a flag.** Flag 106
  marks whole stations or runs of several months rather than isolated bad hours. The wind study did
  not check whether excluding a flagged station moves its contrasts, and the solar study checked
  only the 107 site-hours whose flag differs from the usual value.
- **Each planned contrast reads one nearest station per farm.** The six solar farms share one
  radiation station, so the results do not say how a different station would compare.
- **MIDAS Open's other wind datasets were not searched for the 12 hourly-weather stations with no
  wind in the downloaded file,** so a nearer anemometer may exist. See [how nearby weather stations
  were added](../studies/weather-products-for-past-wind.md#how-nearby-weather-stations-were-added).

**The studies did not research whether a crowd-sourced observation dataset, such as a network of
amateur weather stations, would add anything to a station arm.**

## Candidates for the past-sunshine study

**Of the products surveyed, ECMWF IFS HRES 9 km, CERRA, and COSMO-R6G2 rank highest for the
past-sunshine study, because each adds a model family, a grid spacing, or a kind of product the
study has not scored over the whole window from December 2022.** A native direct beam counts for
less: the study found that a product's own direct beam improves its mean absolute error by 0.03 to
0.10 percentage points of capacity. A place in this ranking is not a decision to score the product,
and no extension of the study has been agreed. The ranking, with each product's roadmap status and
one reason:

1. **ECMWF IFS HRES 9 km on Open-Meteo** (not on the roadmap) — a 9 km forecast from the IFS model
   behind the 31 km ERA5, from 2017, through the same Open-Meteo interface as the four weather
   models on the main row set, with a native direct beam. The study scores it only from November
   2024, on the [shorter row set of the four extra Open-Meteo
   models](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models), so the
   candidate is to score it over the whole window from December 2022. Before October 2025 the series
   is Open-Meteo's own assembly of IFS runs "employing the most up-to-date version of IFS"
   ([Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)). Open-Meteo does
   not document the series' served lead. A study using the series has to report both the assembly
   and the undocumented lead.
2. **CERRA** (🔬 research, deprioritised) — a regional reanalysis on a 5.5 km grid with a native
   direct field, hourly through forecast leads 1 to 3. Its latency is why the catalogue
   deprioritises CERRA for the live service, and is no obstacle for a study of past weather. The
   download volume is the obstacle: every field is the whole European domain.
3. **COSMO-R6G2** (not on the roadmap) — a second regional-reanalysis family, after CERRA, on a
   ~6 km grid, with hourly global and direct irradiance as plain monthly NetCDF. The archive stops
   at December 2025, 9 months short of the study's September 2026 end, so COSMO-R6G2 can only be
   scored on a shorter window. DWD publishes the archive in yearly batches. The hourly means are
   stamped at :30 and need shifting 30 minutes to be period-ending.
4. **GFS (0.11° surface fields)** (not on the roadmap) — a further global centre, from 2021-03-23,
   whose diffuse field is visible-band rather than broadband.
5. **MERRA-2** (not on the roadmap) — an optional arm from the GEOS model family. The ~50 km grid
   is coarser than every other arm, so a score cannot separate model family from grid spacing, and
   MERRA-2 has no direct field. The hourly means are stamped at :30 and need shifting 30 minutes to
   be period-ending.
6. **HelioClim-3** (not on the roadmap) — a second satellite retrieval beside CAMS, but paid after
   2006, and its direct and diffuse are decompositions of global irradiance.

ICON-DREAM-EU, an ICON reanalysis on a 6.5 km grid with native direct and diffuse, is scored in the
past-sunshine study: [ICON-DREAM-EU beats ERA5 but not the ICON weather
models](../studies/weather-products-for-past-solar.md#icon-dream-eu-beats-era5-but-not-the-icon-weather-models).

The sunshine study scores the DMI and KNMI HARMONIE-AROME forecasts on global irradiance only, on
[its shorter row set from November 2024](../studies/weather-products-for-past-solar.md#the-four-extra-open-meteo-models).
It scores neither direct beam. KNMI's direct beam is Open-Meteo's separation of the global
irradiance. The study finds that the field Open-Meteo serves as DMI's direct beam is exactly zero in
half of the daytime rows and in some rows exceeds the global irradiance, which supports the reading
that the field is not DMI's direct beam.

## Candidates for the past-wind study

**Of the products surveyed, CERRA and NORA3 rank highest for the past-wind study, because both serve
wind close to hub height on grids of 3 to 5.5 km.** Neither covers the whole study window: CERRA
ends on 2026-06-30 and NORA3 on 2026-08-31, against the study's September 2026 end. The wind farms'
hub heights are not known; the study scores 80 m and 100 m wind. A place in this ranking is not a
decision to score the product, and no extension of the study has been agreed. The ranking, with each
product's roadmap status and one reason:

1. **CERRA** (🔬 research, deprioritised) — wind at 75 m and 100 m, bracketing 80 m, on a 5.5 km
   grid, hourly, to June 2026. The download volume is the same obstacle as for sunshine.
2. **NORA3** (not on the roadmap) — HARMONIE-AROME on a 3 km grid with wind at 50 m and 100 m,
   hourly, to August 2026, over OPeNDAP, so a point extraction is cheap. The components are
   grid-relative and need rotating.
3. **COSMO-R6G2** (not on the roadmap) — 100 m wind, hourly, on a ~6 km grid, to December 2025.
4. **ECMWF IFS HRES 9 km** (not on the roadmap) — 10, 80, and 100 m wind from 2017, through the
   existing Open-Meteo interface.
5. **DMI HARMONIE-AROME DINI** (not on the roadmap) — 100 m wind on a 2 km grid from 2024-06-28,
   from a different consortium than UKV and ICON-D2. The 80 m wind Open-Meteo serves is
   interpolated.
6. **GFS** (not on the roadmap) — 80 m and 100 m wind, but only from the 0.25° product.

MERRA-2 earns no wind arm: its highest wind is at 50 m, below both heights the study scores, on a
~50 km grid over flat Lincolnshire, so a score would mainly measure grid spacing.

ICON-DREAM-EU is scored in the past-wind study: [ICON-DREAM-EU does not beat ERA5, and trails
ICON-EU](../studies/weather-products-for-past-wind.md#icon-dream-eu-does-not-beat-era5-and-trails-icon-eu).
The study reads it at DWD's model level 72, about 96 m, directly, rather than interpolating to a
fixed height.

## Candidates for comparing forecast models at a fixed lead (issue #810)

**Several whole-run archives reach back further than Open-Meteo's Previous Runs archive, but
Previous Runs is the only free route surveyed that puts ICON-D2, AROME, the HARMONIE-AROME models,
and UKV as Open-Meteo serves it at a fixed lead, and for those models at day 1 only.** CEDA's UKV
archive keeps whole runs from 2016, but access is by application, and CEDA's UKV is statistically
different from the live UKV. Issue #810 proposes ECMWF ENS via Dynamical.org, the product in
production, as the incumbent each other model is compared against. A place in this list is not a
decision to score the product. Ordered by how far back whole runs reach:

1. **Dynamical.org GEFS 35-day** (not on the roadmap), from 2020-10-01 — an ensemble comparator to
   ECMWF ENS with 3.5 more years of history, 00 UTC runs only.
2. **Dynamical.org GFS** (not on the roadmap), from 2021-05-01 — 4 runs a day to 384 h with global
   irradiance and wind at 10, 80, and 100 m.
3. **Google WeatherNext 2** (not on the roadmap), from 2022-01-01 — an ensemble with wind and no
   radiation, marked experimental by Google.
4. **Open-Meteo Single Runs of ECMWF IFS HRES 9 km** (not on the roadmap), from 2024-03-14 — the one
   whole-run archive in this list with a native direct beam. Three caveats apply: Open-Meteo calls
   the early runs hindcasts, the runs sampled from early 2024 are 00 and 12 UTC only, and the model
   changes to IFS Cycle 50r1 on 2026-05-12, [an upgrade the data sources page
   describes](../roadmap/data-sources.md#nwp-model-upgrades-since-2019).
5. **Dynamical.org AIFS Single** (not on the roadmap), with radiation and 100 m wind from the
   2025-02-24 06 UTC run.
6. **Open-Meteo Previous Runs**, from 2024-01-19 at the earliest — capped at the offsets in the
   table above.

**The matched-lead study scored most of these candidates and found ENS ahead of UKV, ICON-EU, and
GEFS for solar.** The
[study](https://openclimatefix.github.io/nged-substation-forecast/studies/nwp-forecasts-at-matched-leads/)
reads Open-Meteo Previous Runs (UKV, ICON-D2, ICON-EU, ICON global, IFS 0.25°, GFS, ARPEGE, AROME,
and two HARMONIE-AROME products) at whole-day offsets, and Dynamical.org's ECMWF ENS and GEFS as
whole 00 UTC runs. Previous Runs is not a fixed-lead archive, because a value's lead depends on each
product's run cycle, so the study brackets each product between ENS's day-0 and day-1 errors. For
solar, UKV (+1.242 points [+0.824, +1.689]) and ICON-EU (+0.817 [+0.584, +1.069]) lose to ENS, and
GEFS at an identical lead is +1.272 points [+0.937, +1.601] worse. For wind, UKV loses (+0.770
[+0.460, +1.057]), ICON-EU is unresolved (+0.169 [-0.064, +0.384]), and GEFS is +0.858 points
[+0.598, +1.099] worse. Dynamical.org's GEFS is read from 2024-11-30 in the study, although the
archive reaches back to 2020-10-01. Products the study does not fit are the ERA5 and CAMS reference
rows, IFS HRES 9 km, Dynamical.org's own GFS, and Open-Meteo Single Runs.

Dynamical.org's ICON-EU whole runs carry native direct and diffuse irradiance, but start only on
2026-02-10, too short a history to list.

## Products ruled out

**The products ruled out each fail on period, on missing variables, on having no data at Lincoln,
or on adding nothing a product already on this page carries:**

- **UERRA** — ends in 2019, and CERRA supersedes it
  ([CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-uerra-europe-single-levels)).
- **COSMO-REA6** — ends 2019-08-31, and COSMO-R6G2 supersedes it
  ([WDC Climate](https://www.wdc-climate.de/ui/entry?acronym=CR6_EU6)).
- **ERA5-Land** — its global irradiance and 10 m wind come from ERA5's forcing, per the CDS
  description, and it has no direct field and no hub-height wind.
- **CERRA-Land** — global irradiance only, no wind, and ends 2025-12-31.
- **The National Solar Radiation Database (NSRDB), Meteosat Prime Meridian** — covers Europe, but
  ends in 2022 ([NSRDB](https://developer.nlr.gov/docs/solar/nsrdb/nsrdb-msg-v1-0-0-download/)).
- **Open-Meteo Ensemble API as a history source** — values at Lincoln reach only 3 to 4 days back,
  and less than a day for the ECMWF 9 km Europe ensemble, although the API accepts dates 92 days
  back ([Ensemble API](https://open-meteo.com/en/docs/ensemble-api)).
- **ECMWF IFS ENS 46-day** — 1.5° grid and no radiation
  ([Dynamical.org](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree/)).
- **Google WeatherNext Graph** — deprecated, with 10 m wind only and no radiation
  ([Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_59572747_4_0)).
- **ECMWF IFS 0.4° (`ecmwf_ifs04`) on Open-Meteo** — no radiation, 10 m wind only, and the series
  stops 2025-03-07.
- **Météo-France AROME France HD on Open-Meteo** — no radiation at all.
- **Open-Meteo "UK Met Office seamless"** — at Lincoln on 2023-06-10 and 2025-06-10 every global
  irradiance and 10 m wind value equals UKV's.
- **MET Nordic, JMA MSM, ItaliaMeteo ICON-2I, MeteoSwiss ICON-CH1, and KNMI HARMONIE-AROME
  Netherlands on Open-Meteo** — no data at Lincoln.
- **ACCESS-G on Open-Meteo** — no radiation or 100 m wind at Lincoln on 2026-09-01.
- **GEM Global and CMA GRAPES on Open-Meteo** — data at Lincoln, but neither grid spacing nor method
  sets them apart from the global models ranked above.
- **MOGREPS-UK and MOGREPS-G history** — AWS holds a 30-day rolling window
  ([MOGREPS-G](https://registry.opendata.aws/met-office-global-ensemble/)), Open-Meteo's ensemble
  archive 3 to 4 days, and CEDA's `ukmo-nwp` archive has no MOGREPS directory.
