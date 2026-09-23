# Weather products surveyed for the studies

**This page is a survey, not a plan: the project has agreed to ingest none of the products that
appear only here.** The survey asked which further weather products could join the two studies of
past weather — [Which weather product best describes past
sunshine?](../studies/weather-products-for-the-past.md) and [Which weather product best describes
past wind?](../studies/weather-products-for-past-wind.md) — and which archives could support the
forecast comparison in
[issue 810](https://github.com/openclimatefix/nged-substation-forecast/issues/810). It ranks
candidates for those three studies. The list of products the project ingests, or plans to ingest, is
the catalogue under [Weather data](../roadmap/data-sources.md#weather-data).

**Every product in the tables below carries one of four statuses:**

- **In production** — ingested by the running service. ECMWF ENS via Dynamical.org is the only
  product with this status.
- **On the roadmap** — has a row in the [Weather data](../roadmap/data-sources.md#weather-data)
  catalogue, with the milestone that row gives.
- **Used in a study** — scored on one of the two study pages. The six products scored there are
  CAMS, ERA5, ICON-D2, ICON-EU, ICON global, and UKV. CAMS has no wind arm.
- **Researched only** — surveyed here and used nowhere else in the project.

## Most facts about an Open-Meteo product describe Open-Meteo's archive

**Open-Meteo re-serves other centres' forecasts, so a start date, a latency, or a direct/diffuse
split read through Open-Meteo may belong to the archive rather than to the producer.** The tables
below say which. Three conventions apply to every Open-Meteo row:

- **Start dates are the first hour with a non-null value** at Lincoln (53.23°N, 0.54°W), queried on
  2026-09-23. A producer's documentation can give a different date.
- **Latency is the time from a run's initialisation to the whole run being available on
  Open-Meteo**, read from that model's `meta.json` for the 2026-09-23 00 UTC run. The figure is one
  run, not an average.
- **"Native" means the producer publishes the direct or the diffuse field, and "separation" means
  Open-Meteo derives both from global irradiance** with the Razo–Müller–Witwer separation model
  ([Open-Meteo ECMWF documentation](https://open-meteo.com/en/docs/ecmwf-api)). A split made by
  separation carries no information the global field lacks.

## Which archives keep whole past forecast runs

**Issue 810 has to hold the lead time fixed while varying the model, so it needs an archive that
keeps whole past runs or serves a fixed lead.** Open-Meteo publishes three archives of the same
forecasts, and Dynamical.org publishes a fourth kind:

- **The Historical Forecast archive stitches the first hours of each successive run** into one
  continuous series, so a product's lead follows its run frequency rather than being chosen
  ([Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)). The two study
  pages score this archive.
- **The Previous Runs archive serves each variable at a fixed offset of 1 to 7 days**
  ([Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)), but from 2024-01-19 at
  the earliest, and at day 1 only for UKV, ICON-D2, and the HARMONIE and AROME models (table below).
- **The Single Runs archive keeps whole runs** ([Single Runs
  API](https://open-meteo.com/en/docs/single-runs-api)), but only from the 2026-04-02 00 UTC run:
  `ecmwf_ifs025` and `ncep_gfs013` both reject `run=2026-04-01T00:00`. ECMWF IFS HRES 9 km is the
  exception. The model `ecmwf_ifs` rejects `run=2024-03-13T00:00` and serves the 2024-03-14 00 UTC
  run with 240 hourly steps. Open-Meteo calls the runs from 2024-03-14 "IFS Cycle 49R1 hindcasts";
  the March to June 2024 runs sampled are 00 and 12 UTC only, and four runs a day are present from
  2024-09-01 at the latest.
- **Dynamical.org keeps whole runs as Zarr stores.** The start dates below come from Dynamical.org's
  catalogue pages, checked against each store's `init_time` coordinate where the date matters here.
  The longest reach back to 2020-10-01 (GEFS) and 2021-05-01 (GFS).

**Which offsets the Previous Runs archive serves is set by the archive, not by the model's
horizon.** UKV's runs in the Single Runs archive hold 55 hourly steps, to T+54 h, and ICON global's
runs reach 180 h, yet the Previous Runs archive serves UKV at day 1 only and ICON global to day 6
only. A lead-matched comparison with UKV, ICON-D2, or a HARMONIE or AROME model as one arm is
therefore capped at day 1.

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
| ECMWF IFS HRES 9 km ([Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api)) | 00 and 12 UTC at first, 4 a day by 2024-09-01 | 2024-03-14 | Global and native direct irradiance; 10, 80, and 100 m wind |
| [ECMWF IFS ENS](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) (Dynamical.org) | 51 members, 00 UTC only, to 360 h (906 runs to 2026-09-23) | 2024-04-01 | Global irradiance; 10 and 100 m wind |
| [ECMWF AIFS Single](https://dynamical.org/catalog/ecmwf-aifs-single-forecast/) (Dynamical.org) | 4 a day, to 360 h | 2024-04-01, but radiation and 100 m wind are null until the 2025-02-24 06 UTC run | Global irradiance; 10 and 100 m wind |
| [ECMWF AIFS-ENS](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/) (Dynamical.org) | 4 a day, to 360 h | 2025-07-02 | Global irradiance; 10 and 100 m wind |
| [DWD ICON-EU](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/) (Dynamical.org) | 4 a day, to 120 h | 2026-02-10 | Native direct and diffuse irradiance; 10 m wind only |
| [CEDA NWP-UKV](https://catalogue.ceda.ac.uk/uuid/78f23c539d304591b137cf986b69a525/) | Up to 8 a day, T+0 to T+120 | 2016-03-16 | Radiation not listed on the catalogue record; access by application |

## What each product carries

**Of the forecast models surveyed, only ECMWF IFS HRES 9 km, GFS, DMI HARMONIE-AROME, the Met
Office global model, and ICON-EU serve a producer's own direct or diffuse field.** Every other
direct/diffuse split Open-Meteo serves for a forecast model is made by separation. Among the
reanalyses, CERRA, COSMO-R6G2, and ICON-DREAM-EU carry the producer's own direct field. "Grid" is
the producer's native grid spacing, with the serving grid in brackets where the two differ. "Past
runs" names the archive that keeps whole runs or a fixed lead, from the section above.

### Forecast models

| Product | Status | Producer and method | Grid | Covers GB | Archive start | Solar variables | Wind heights | Runs, horizon, step | Latency | Past runs | Access | Licence | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ECMWF IFS ENS** (Dynamical.org) | In production ([catalogue](../roadmap/data-sources.md#weather-data)) | ECMWF, 51-member physics ensemble | 0.25° served | Yes (global) | 2024-04-01 to present | Global only (`ssrd`) | 10, 100 m | 00 UTC; 360 h; 3-hourly to 144 h, then 6-hourly | Not established | Whole runs | Zarr | CC BY 4.0 | [Dynamical.org](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) |
| **ECMWF IFS HRES 9 km** (`ecmwf_ifs`) | Researched only | ECMWF, global physics model | 9 km (served natively) | Yes (global) | 2017-01-01, complete to 2026-09-21 | Global; direct native (ECMWF's `fdir`, per Open-Meteo's downloader); diffuse = global − direct | 10, 80, 100 m | 4 a day; 15 days | 7.0 h (Open-Meteo) | Single Runs from 2024-03-14; Previous Runs from 2025-10-01 | Open-Meteo | CC BY 4.0, as ECMWF's whole catalogue has been since 2025-10-01 | [Open-Meteo](https://open-meteo.com/en/docs/ecmwf-api), [downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/EcmwfEcpds/EcmwfEcpdsVariable.swift), [ECMWF](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-makes-its-entire-real-time-catalogue-open-all) |
| **ECMWF IFS 0.25°** (`ecmwf_ifs025`) | Researched only | ECMWF open-data subset of the same run | 9 km (0.25° served) | Yes (global) | Radiation and 100 m wind 2024-03-05; 10 m wind 2024-02-03 | Global; direct and diffuse by separation | 10, 100 m | 4 a day; 15 days; 3-hourly | 7.8 h (Open-Meteo) | Previous Runs from 2024-03-06; Single Runs from 2026-04-02 | Open-Meteo | CC BY 4.0 | [Open-Meteo](https://open-meteo.com/en/docs/ecmwf-api) |
| **ECMWF AIFS Single** | On the roadmap, v2.1 ([catalogue](../roadmap/data-sources.md#weather-data)) | ECMWF, machine-learned deterministic model | 0.25° | Yes (global) | Open-Meteo 2025-02-17; Dynamical.org radiation from 2025-02-24 06 UTC | Global only (`ssrd`); Open-Meteo's direct is by separation | 10, 100 m | 4 a day; 360 h; 6-hourly | 5.7 h (Open-Meteo) | Whole runs (Dynamical.org); Previous Runs from 2025-02-17 | Open-Meteo; Zarr | CC BY 4.0 | [Dynamical.org](https://dynamical.org/catalog/ecmwf-aifs-single-forecast/) |
| **ECMWF AIFS-ENS** | On the roadmap, v2.1 ([catalogue](../roadmap/data-sources.md#weather-data)) | ECMWF, machine-learned ensemble | 0.25° | Yes (global) | 2025-07-02 | Global only (`ssrd`) | 10, 100 m | 4 a day; 360 h; 6-hourly | Not established | Whole runs | Zarr | CC BY 4.0 | [Dynamical.org](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/) |
| **DWD ICON-EU** (Dynamical.org) | On the roadmap, v0.9 ([catalogue](../roadmap/data-sources.md#weather-data)); the Open-Meteo copy is used in both studies | DWD, regional physics model | 6.5 km (0.0625° served) | Yes | 2026-02-10 | Direct and diffuse native | 10 m only | 4 a day; 120 h; hourly to 78 h | Not established | Whole runs | Zarr | Not checked | [Dynamical.org](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/) |
| **NOAA GFS** (`ncep_gfs013` for surface fields, `ncep_gfs025` for 80 and 100 m wind) | Researched only | The National Centers for Environmental Prediction of the US National Oceanic and Atmospheric Administration (NOAA NCEP), global physics model | ~13 km (0.11° served for surface fields, 0.25° for 80 and 100 m wind) | Yes (global) | 2021-03-23 | Global; diffuse native; direct = global − diffuse | 10, 80, 100 m | 4 a day; 16 days | 5.5 h at 0.11°, 6.5 h at 0.25° (Open-Meteo) | Previous Runs from 2024-01-19; whole runs on Dynamical.org from 2021-05-01 | Open-Meteo; Zarr | Public domain | [NOAA EMC](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php), [Open-Meteo](https://open-meteo.com/en/docs/gfs-api), [Dynamical.org](https://dynamical.org/catalog/noaa-gfs-forecast/) |
| **NOAA GEFS 35-day** (Dynamical.org) | Researched only | NOAA NCEP, 31-member ensemble | 0.25° to 240 h, 0.5° after | Yes (global) | 2020-10-01 | Global only | 10, 80, 100 m | 00 UTC; 840 h; 3-hourly to 240 h, then 6-hourly | Not established | Whole runs | Zarr | Public domain | [Dynamical.org](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/) |
| **Météo-France ARPEGE Europe** | Researched only | Météo-France, stretched-grid global model | ~5 km over Europe (0.1° served) | Bounds of the Europe extract not checked; data present at Lincoln | Global irradiance and 10 m wind 2022-11-09; 100 m wind 2023-06-05 | Global; direct and diffuse by separation | 10, 80, 100 m | 4 a day; 4 days | 3.5 h (Open-Meteo) | Previous Runs from 2024-01-19 | Open-Meteo | Not checked | [Météo-France](https://meteofrance.com/actualites-et-dossiers/comprendre-la-meteo/les-modeles-de-prevision-meteo), [Open-Meteo](https://open-meteo.com/en/docs/meteofrance-api) |
| **Météo-France ARPEGE World** | Researched only | The same ARPEGE run, global extract | ~5 km over Europe (0.25° served) | Yes (global) | 2023-12-27 | Global; direct and diffuse by separation | 10, 100 m | 4 a day; 4 days | 4.0 h (Open-Meteo) | Previous Runs from 2024-01-19 | Open-Meteo | Not checked | [Open-Meteo](https://open-meteo.com/en/docs/meteofrance-api) |
| **Météo-France AROME France** | Researched only | Météo-France, convection-permitting regional model | 1.3 km (0.025° served) | South of 55.4°N only | 2023-12-27 | Global; direct and diffuse by separation | 10, 80, 100 m | 8 a day; 2 days | 2.7 h (Open-Meteo) | Previous Runs from 2024-01-19 | Open-Meteo | Not checked | [Open-Meteo](https://open-meteo.com/en/docs/meteofrance-api), [domain](https://api.open-meteo.com/data/meteofrance_arome_france0025/static/meta.json) |
| **DMI HARMONIE-AROME DINI** | Researched only | The Danish Meteorological Institute (DMI) for the UWC-West consortium, convection-permitting, started from ECMWF IFS | 2 km (Open-Meteo's figure) | Yes (39.7–62.7°N, 25.4°W–40.1°E) | 2024-06-28 | Global; direct native, but see the note below this table on its served direct share | 10 m; 50–450 m, including 80 and 100 m | 8 a day; 2.5 days | 2.8 h (Open-Meteo) | Previous Runs from 2024-06-29 | Open-Meteo | Not checked | [Open-Meteo](https://open-meteo.com/en/docs/dmi-api), [domain](https://api.open-meteo.com/data/dmi_harmonie_arome_europe/static/meta.json) |
| **KNMI HARMONIE-AROME Europe** | Researched only | The Royal Netherlands Meteorological Institute (KNMI) for UWC-West, started from ECMWF IFS | 5.5 km, as KNMI publishes the grid | Yes (39.7–62.6°N, 25.2°W–38.8°E) | Global irradiance 2024-06-26; 100 m wind 2024-10-09 | Global; direct and diffuse by separation | 10 to 300 m, including 80 and 100 m | Hourly; 2.5 days | 3.3 h (Open-Meteo) | Previous Runs from 2024-06-27 | Open-Meteo | Not checked | [Open-Meteo](https://open-meteo.com/en/docs/knmi-api), [domain](https://api.open-meteo.com/data/knmi_harmonie_arome_europe/static/meta.json) |
| **Met Office global 10 km** (`ukmo_global_deterministic_10km`) | Researched only ([why no request has been made](../roadmap/data-sources.md#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost)) | Met Office Unified Model, global | ~10 km (0.09° served) | Yes (global) | Direct and 10 m wind 2022-03-01; global irradiance from 2025-01-02 | Global; direct native; diffuse = global − direct | 10 m only | 4 a day; 7 days (Open-Meteo) | 8.4 h (Open-Meteo) | Previous Runs from 2025-01-03 | Open-Meteo; AWS | CC BY-SA 4.0 (AWS) | [Open-Meteo](https://open-meteo.com/en/docs/ukmo-api) |
| **Google WeatherNext 2** | Researched only | Google DeepMind, machine-learned 64-member ensemble | 0.25° | Yes (global) | 2022-01-01 to 2026-09-22 in the Earth Engine catalogue | None | 10, 100 m | 4 a day; 15 days; 6-hourly | Historical data is anything over 48 h old | Whole runs | Earth Engine, BigQuery, Cloud Storage Zarr | CC BY 4.0 for historical data | [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0) |
| **CEDA NWP-UKV** (the UK's Centre for Environmental Data Analysis) | Researched only | Met Office UKV, raw operational output | ~0.018° (CEDA's figure) | Yes | 2016-03-16 to present | Not listed on the catalogue record | Surface and pressure levels | Up to 8 a day; 120 h | Archive, ongoing | Whole runs | CEDA, by application | Met Office licence via CEDA | [CEDA](https://catalogue.ceda.ac.uk/uuid/78f23c539d304591b137cf986b69a525/) |
| **CEDA Met Office global** (`global-grib`) | Researched only | Met Office global model, raw output | Not established | Yes (global) | Directories from 2016-03 to 2026-09 | Not established; the README needs a login | Not established | Not established | Archive | Not established | CEDA, by application | Not established | [CEDA](https://data.ceda.ac.uk/badc/ukmo-nwp/data/) |

**DMI HARMONIE-AROME's served direct share is about half what the
other models give, for a reason not established.** Direct over global irradiance at Lincoln was
0.24, 0.30, and 0.18 in August 2024, June 2025, and June 2026, against 0.46 to 0.62 for KNMI
HARMONIE-AROME, UKV, and ECMWF IFS HRES in the same months. Whether Open-Meteo's conversion or DMI's
field definition causes the gap has not been checked.

### Reanalyses, hindcasts, and satellite retrievals

| Product | Status | Producer and method | Grid | Covers GB | Period | Solar variables | Wind heights | Time step | Latency | Access | Licence | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CERRA** | On the roadmap, deprioritised ([catalogue](../roadmap/data-sources.md#weather-data)) | Copernicus regional reanalysis, HARMONIE-ALADIN with 3D-Var, forced by ERA5 | 5.5 km, 1069 × 1069 points | Yes (European domain) | 1984-09-01 to 2026-06-30 on the CDS on 2026-09-23 | Global and direct native, forecast fields only, accumulated | 10 m; 15, 30, 50, 75, 100, 150 to 500 m | 3-hourly analyses; hourly values from forecast leads 1–3 h | About 12 weeks | Copernicus CDS; no area cropping, so every field is the whole domain | CC BY 4.0 | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels), [user guide](https://confluence.ecmwf.int/x/WFQ7E), [Ridal et al. (2024)](https://doi.org/10.1002/qj.4764) |
| **COSMO-R6G2** | Researched only | DWD, COSMO regional reanalysis forced by ERA5 | 0.055° (~6 km), EURO-CORDEX EUR-11 domain | Yes | 2002-01 to 2025-12 | Global and direct native, hourly means | 10 m; 100, 150, 200 m | Hourly | About 5 months | HTTPS, monthly NetCDF | CC BY 4.0 | [DWD](https://opendata.dwd.de/climate_environment/REA/COSMO_R6G2/), [paper](https://asr.copernicus.org/articles/22/149/2026/) |
| **ICON-DREAM-EU** | Researched only | DWD, ICON reanalysis, EU nest | 6.5 km | Not checked against the grid file | 2010-01 to 2026-08 | Direct and diffuse native, hourly means | 10 m; otherwise model levels only, so a fixed height needs vertical interpolation | Hourly | 2–3 months, per DWD | HTTPS, monthly GRIB2 | CC BY 4.0 | [DWD](https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-EU/) |
| **NORA3** | Researched only | MET Norway, HARMONIE-AROME hindcast downscaling ERA5 | 3 km | Yes, checked on the grid | 1958 to 2026-08-31 | Global only | 10 m; 20, 50, 100, 250, 500, 750 m, as grid-relative components needing rotation | Hourly, from leads 3–9 h of each 6-hourly run | About 3 weeks | THREDDS / OPeNDAP | CC BY 4.0 | [THREDDS](https://thredds.met.no/thredds/catalog/nora3/catalog.html), [Solbrekke et al. (2021)](https://doi.org/10.5194/wes-6-1501-2021) |
| **MERRA-2** | Researched only | NASA's Global Modeling and Assimilation Office, GEOS global reanalysis | 0.5° × 0.625° | Yes (global) | 1980 to present | Global only (`SWGDN`) | 2, 10, 50 m | Hourly means stamped at the centre of the hour | About 3 weeks after month end | GES DISC, Earthdata login | NASA open data | [radiation](https://disc.gsfc.nasa.gov/datasets/M2T1NXRAD_5.12.4/summary), [wind](https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary) |
| **ERA5-Land** | Researched only | ECMWF, land component of ERA5 re-run with ERA5 forcing | 0.1° | Yes | 1950 to 2026-09-17 | Global only | 10 m only | Hourly | About 6 days | CDS | CC BY 4.0 | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) |
| **CERRA-Land** | Researched only | Copernicus land-surface reanalysis on CERRA's grid | 5.5 km | Yes (European domain) | 1984-09 to 2025-12-31 | Global only | None | Leads 1–3 h | Not checked | CDS | CC BY 4.0 | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-land) |
| **NOAA GFS and GEFS analyses** (Dynamical.org) | Researched only | The first hours of each GFS or GEFS run, concatenated | 0.25° | Yes (global) | GFS 2021-05-01; GEFS 2000-01-01 (reforecast to 2019, operational after) | Global only | 10, 80, 100 m | GFS hourly; GEFS 3-hourly | Not established | Zarr | Public domain | [GFS](https://dynamical.org/catalog/noaa-gfs-analysis/), [GEFS](https://dynamical.org/catalog/noaa-gefs-analysis/) |
| **HelioClim-3** | Researched only | Transvalor and MINES ParisTech, Heliosat-2 retrieval from Meteosat | About 5 km in Europe | Yes | 2004-02 to real time | Global; direct and diffuse by decomposition | None | 15 min | "A few minutes after image reception" | SoDa web service | Paid after 2006 | [SoDa](https://www.soda-pro.com/help/helioclim/helioclim-3-overview) |
| **Solcast** | Researched only | Solcast, multi-satellite retrieval | 90 m (downscaled) | Yes | 2007-01 to 7 days ago | Global, direct normal, diffuse, tilted | Included; heights not listed | 5–60 min | 7 days | Commercial API | Paid | [Solcast](https://solcast.com/historical-weather-api) |

## Candidates for the past-sunshine study

**Of the products surveyed, ECMWF IFS HRES 9 km, CERRA, and COSMO-R6G2 rank highest for the
past-sunshine study, because each adds a producer's own direct beam from a model or a grid spacing
the study has not scored.** The ranking, with one reason each:

1. **ECMWF IFS HRES 9 km on Open-Meteo** — a native direct beam from 2017, through the same
   Open-Meteo interface as the four weather models already scored. Before October 2025 the series
   is Open-Meteo's own assembly of IFS runs "employing the most up-to-date version of IFS"
   ([Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)), and its
   served lead is undocumented, so a study using it has to say both.
2. **CERRA** — a regional reanalysis at 5.5 km with a native direct field, hourly through forecast
   leads 1 to 3. Its 12-week latency is why the catalogue deprioritises CERRA for the live service,
   and is no obstacle for a study of past weather. The download volume is the obstacle: every field
   is the whole European domain.
3. **COSMO-R6G2** — a third regional-reanalysis family at ~6 km with hourly global and direct
   irradiance, as plain monthly NetCDF. The archive stops at December 2025, nine months short of the
   studies' September 2026 end, so COSMO-R6G2 can only be scored on a shorter window.
4. **ICON-DREAM-EU** — an ICON reanalysis at 6.5 km with native direct and diffuse, which sets an
   analysis against the short-range ICON-EU forecast already scored inside one model family.
   Coverage of GB has to be checked against the grid file first.
5. **GFS (0.11° surface fields)** — a further global centre with native diffuse irradiance, from
   2021-03-23.
6. **MERRA-2** — an optional arm from the GEOS model family. The ~50 km grid is coarser than every
   other arm, so a score cannot separate model family from grid spacing, and MERRA-2 has no direct
   field. The hourly means are stamped at :30 and need shifting 30 minutes to match the period-ending
   meters.
7. **HelioClim-3** — a second satellite retrieval beside CAMS, but paid after 2006, and its direct
   and diffuse are decompositions of global irradiance.

The DMI and KNMI HARMONIE-AROME forecasts would add a further model family, but neither adds a
trustworthy direct component: DMI's served direct share is about half other models', and KNMI's
split is by separation. Global irradiance alone would make either a lower-priority arm.

## Candidates for the past-wind study

**Of the products surveyed, CERRA and NORA3 rank highest for the past-wind study, because both serve
wind close to hub height at 3 to 5.5 km for the whole study window.** The ranking, with one reason
each:

1. **CERRA** — wind at 75 m and 100 m, bracketing an 80 m hub, at 5.5 km, hourly, to June 2026. The
   download volume is the same obstacle as for sunshine.
2. **NORA3** — HARMONIE-AROME at 3 km with wind at 50 m and 100 m, hourly, to August 2026, over
   OPeNDAP, so a point extraction is cheap. The components are grid-relative and need rotating.
3. **COSMO-R6G2** — 100 m wind, hourly, at ~6 km, to December 2025.
4. **ECMWF IFS HRES 9 km** — 10, 80, and 100 m wind from 2017, through the existing Open-Meteo
   interface.
5. **DMI HARMONIE-AROME DINI** — 80 m and 100 m wind at 2 km from 2024-06-28, from a different
   consortium than UKV and ICON-D2.
6. **GFS** — 80 m and 100 m wind, but only from the 0.25° product.

MERRA-2 earns no wind arm: its highest wind is at 50 m, below the hub heights in question, on a
~50 km grid over flat Lincolnshire, so a score would mainly measure grid spacing. ICON-DREAM-EU
serves wind only on model levels, so it would need a vertical interpolation no other arm does.

## Candidates for the forecast comparison in issue 810

**Several whole-run archives reach back further than Open-Meteo's Previous Runs archive, but only
Previous Runs puts ICON, UKV, and the HARMONIE models at a fixed lead, and for those models at day 1
only.** ECMWF ENS via Dynamical.org is the incumbent every arm is compared against. Ordered by how
far back whole runs reach:

1. **Dynamical.org GEFS 35-day**, from 2020-10-01 — an ensemble comparator to ECMWF ENS with 3.5
   more years of history, 00 UTC runs only.
2. **Dynamical.org GFS**, from 2021-05-01 — four runs a day to 384 h with global irradiance and
   wind at 10, 80, and 100 m.
3. **Google WeatherNext 2**, from 2022-01-01 — an ensemble with wind and no radiation.
4. **Open-Meteo Single Runs of ECMWF IFS HRES 9 km**, from 2024-03-14 — the one whole-run archive
   here with a native direct beam, but Open-Meteo calls the early runs hindcasts, and the runs
   sampled from 2024 start at 00 and 12 UTC only.
5. **Dynamical.org AIFS Single**, with radiation and 100 m wind from the 2025-02-24 06 UTC run.
6. **Open-Meteo Previous Runs**, from 2024-01-19 at the earliest — the only route surveyed to ICON,
   UKV, AROME, and the HARMONIE models at a fixed lead, capped at the offsets in the table above.

## Products ruled out

**The products ruled out each fail on period, on missing variables, or on having no data at
Lincoln:**

- **UERRA** — ends in 2019, and CERRA supersedes it
  ([CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-uerra-europe-single-levels)).
- **COSMO-REA6** — ends 2019-08-31, and COSMO-R6G2 supersedes it
  ([WDC Climate](https://www.wdc-climate.de/ui/entry?acronym=CR6_EU6)).
- **ERA5-Land** — its global irradiance and 10 m wind come from ERA5's forcing, per the CDS
  description, and it has no direct field and no hub-height wind.
- **CERRA-Land** — global irradiance only, no wind, and ends 2025-12-31.
- **NSRDB Meteosat Prime Meridian** — covers Europe, but ends in 2022
  ([NSRDB](https://developer.nlr.gov/docs/solar/nsrdb/nsrdb-msg-v1-0-0-download/)).
- **Open-Meteo Ensemble API** — values reach only 3 to 4 days back, although the API accepts dates
  92 days back ([Ensemble API](https://open-meteo.com/en/docs/ensemble-api)).
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
