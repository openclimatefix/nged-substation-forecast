# Which weather product best describes past sunshine?

> **This study uses only Flexpectation's own data, and its purpose is to inform Flexpectation's
> choices.** The study scores each weather product only at the solar farms in Flexpectation's trial
> area in Lincolnshire. The study does not compare results across many regions or climates, so a
> result on this page may not hold elsewhere.

**At six metered solar farms in Lincolnshire, two satellite retrievals describe past sunshine far
better than any of the weather models or reanalyses tested.** For each of eight weather products, an
XGBoost model, a gradient-boosted tree, was fitted per generator to predict hourly output from that
product's sunshine, and scored by its mean absolute error as a percentage of the generator's
capacity (its 99th percentile of metered output). The error is 5.09% of capacity given the
Copernicus Atmosphere Monitoring Service's satellite retrieval (CAMS), and 5.49% given SARAH-3,
the satellite climate data record from EUMETSAT's Satellite Application Facility on Climate
Monitoring (CM SAF). The error is 7.76% given ICON-D2, the best of the four weather models on the
main row set, and ICON-D2 stays the best of all eight weather models on the shorter row set that
adds four more. It is 8.77% given ICON-DREAM-EU, the German weather service's reanalysis, and 9.08% given
ERA5, the reanalysis from the European Centre for Medium-Range Weather Forecasts (ECMWF). ICON-D2
is the German weather service's model for Germany and neighbouring countries. A satellite observes
the clouds of the hour itself, where every weather model and reanalysis simulates them from a run
started before the hour, so a large gap is expected.

**For training history and disaggregation the page recommends CAMS where its one-day delay allows,
and for capacity estimation CAMS provisionally. For historical features in the live service it
recommends ICON-EU or rebuilt UKV, because ICON-D2, the more accurate weather model here, does not
cover South West England or South Wales.**
These four consumers of past weather are parts of this project, described in the
[introduction](#introduction). ICON-EU is the German weather service's European model. Rebuilt UKV
is the Met Office's UK variable-resolution model (UKV) with its hourly value rebuilt from its own
snapshots. The evidence is six metered solar farms inside one 25 km by 23 km box in Lincolnshire,
and 76,727 generator-hours from December 2022 to August 2026. Hours near sunrise where Open-Meteo's
UKV archive holds a physically impossible value are left out for every product; see
[Limitations](#limitations). Four more weather models, available from Open-Meteo's archive, are
scored separately, on a shorter, more recent row set: see [The four extra Open-Meteo
models](#the-four-extra-open-meteo-models).

![Figure 1: CAMS has the lowest error of the eight products tested, SARAH-3 the second lowest, and
ERA5 the highest](assets/sunshine_leaderboard.svg)

![Figure 2: CAMS beats SARAH-3 by 0.4 points, and ICON-D2, the best of the four weather models
tested, by more than 2 points](assets/sunshine_headline.svg)

**Figure 1's intervals are wide mainly because every product's error rises and falls together from
month to month.** Some months are harder to describe than others for every product, and resampling
whole months carries that shared swing into each product's own interval. The six generators also
share their weather, so each interval rests on 45 months rather than on thousands of independent
hours. Figure 2 pairs two products on the same hours, which cancels the shared swing. Two products
whose intervals overlap in Figure 1, or in the top panel of Figure 2, can therefore still differ by
a margin that is statistically significant at the 5% level. Only a contrast pairing those two
products tests them directly, and the bottom panel of Figure 2 holds the six planned ones.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Opus 5.5 and Claude Sonnet 5, reusing the data-preparation and
> model-fitting code that Claude Opus 5 wrote for the [beam/diffuse study](beam-diffuse-split.md)).
> Several independent Claude reviewers have checked the method, the evidence, and the prose
> adversarially.

## Key findings

**A difference between two errors is in percentage points of capacity, written "points", and a
bracketed pair after a figure, such as [2.42, 2.90], is its 95% interval.** A weather model's value
"as served" is the value Open-Meteo's archive holds for an hour, which comes from a run of the
weather model that started a few hours earlier.

- **CAMS beats ICON-D2, the best of the weather models tested, by 2.68 points [2.42, 2.90], and the
  gap holds at every generator, in every season, and in each calendar year from 2023 to 2026.** See
  [CAMS describes past sunshine
  best](#cams-describes-past-sunshine-best-of-the-eight-products-tested-by-a-wide-margin).
- **CAMS beats SARAH-3, the second satellite retrieval, by 0.40 points [0.29, 0.50], and, in an
  exploratory comparison, by about as much under each satellite SARAH-3 has used since 2021.**
  SARAH-3's error is the second lowest at each of the six generators, but CAMS's margin over
  SARAH-3 varies widely between them, from 0.03 to 0.64 points. See [SARAH-3 is second to CAMS
  under every satellite](#sarah-3-is-second-to-cams-under-every-satellite).
- **ICON-DREAM-EU, the German weather service's reanalysis, beats ERA5 by 0.32 points [0.12,
  0.52], but trails all three ICON weather models.** In an exploratory comparison, each product's
  raw irradiance, compared directly against CAMS's with no power model involved, ranks the
  products almost exactly as the power-model contrasts do, so this is not an artefact of the power
  model. See [ICON-DREAM-EU beats ERA5 but not the ICON weather
  models](#icon-dream-eu-beats-era5-but-not-the-icon-weather-models).
- **In an exploratory comparison on matched months, January to August, ERA5 trails CAMS by between
  3.6 and 4.7 points in each calendar year from 2021 to 2026, and SARAH-3 by between 3.2 and 4.3
  points.** See [ERA5's
  deficit, year by year](#era5s-deficit-year-by-year).
- **Among the four weather models tested, ICON-D2 (from the German weather service's Icosahedral
  Nonhydrostatic model family) is the best as served, beating ICON-EU by 0.62 points [0.48, 0.76]
  across the record.** In a breakdown added after the first run, ICON-D2's advantage over ICON-EU
  shrinks from 1.03 points 1 hour into each run to 0.32 points 3 hours in. See [ICON-D2 is the best
  weather model
  tested](#icon-d2-is-the-best-weather-model-tested-and-its-advantage-shrinks-within-hours-of-each-run).
- **ICON-EU is 0.10 points ahead of ICON global, the same service's global model, as served [0.05,
  0.15].** In a split added after the first run, most of that small gap sits in the hours ICON
  global is served further ahead than ICON-EU. See [ICON-EU against ICON global and
  UKV](#icon-eu-against-icon-global-and-ukv).
- **Once UKV's hourly value is rebuilt from its own snapshots, UKV is ahead of ICON-EU, by a margin
  close to the 5% threshold.** Against Open-Meteo's hourly value for UKV, ICON-EU is 0.48 points
  ahead [0.29, 0.65]. Rebuilt as the mean of UKV's snapshots at the start and the end of the hour,
  UKV is 0.21 points ahead of ICON-EU [0.03, 0.38]. An XGBoost model given both snapshots as
  separate inputs puts UKV 0.26 points ahead
  [0.09, 0.43]. Every rebuild of UKV was added after the first run. See [ICON-EU against ICON global
  and UKV](#icon-eu-against-icon-global-and-ukv).
- **Rebuilt UKV also beats ERA5, by 0.90 points [0.72, 1.09], but Open-Meteo's hourly value for UKV
  beats ERA5 only across the whole record, not since August 2024.** See [UKV rebuilt from its
  snapshots beats ERA5](#ukv-rebuilt-from-its-snapshots-beats-era5).
- **A product's own published direct beam improves the XGBoost model by 0.03 to 0.10 points for
  every product except ERA5 and SARAH-3.** SARAH-3's direct beam is not tested, because CM SAF
  models it from SARAH-3's own global irradiance. See [A product's own direct beam adds
  little](#a-products-own-direct-beam-adds-little).
- **An XGBoost model trained on five generators and applied to the sixth ranks the products the
  same way, with errors 0.10 to 0.20 points larger.** See [The ranking holds for a generator
  predicted from its neighbours](#the-ranking-holds-for-a-generator-predicted-from-its-neighbours).
- **In an exploratory measure, once the seasonal cycle is removed, the implied capacities of CAMS
  and SARAH-3 are the steadiest of the eight products from month to month, but both satellite
  retrievals swing the most with the seasons.** A product's implied capacity is the ratio of metered
  output to what a fixed
  south-facing panel predicts from the product. See [Implied capacity from month to
  month](#implied-capacity-from-month-to-month).
- **At six solar farms in Lincolnshire, from November 2024, HARMONIE-AROME as Open-Meteo serves it
  from both the Danish Meteorological Institute (DMI)'s and the Royal Netherlands Meteorological
  Institute (KNMI)'s feeds trails the ICON model of similar grid spacing, and ECMWF-IFS-HRES
  against ICON-EU is not resolved.** DMI's and KNMI's feeds carry the same United Weather
  Centres-West (UWC-West) HARMONIE-AROME run, not two independent weather models. KNMI
  HARMONIE-AROME, as Open-Meteo's
  archive serves it, trails ICON-EU by 0.56 points [0.32, 0.75]; KNMI's served run interval is not
  measured here, but KNMI and Open-Meteo document an hourly update, which would put it at a 1-hour
  lead on every row against ICON-EU's 1 to 3 hours. DMI HARMONIE-AROME, as Open-Meteo's archive
  serves it, trails ICON-D2 by 1.02 points [0.73, 1.26]; DMI's own run interval is inferred at 3
  hours from weak peaks in its temperature, the same as ICON-D2's, so the two sit at the same
  served lead on every row if that inference holds, and the loss is positive in 21 of 22 months and
  at all six farms (0.77 to 1.19 points). DMI's and KNMI's feeds serve almost the same global
  irradiance at the first hour after each run, where DMI's loss is larger than at other hours. This
  page does not establish why, and the loss outside that hour, 0.84 points [0.52, 1.11], stands on
  its own. None of the four extra Open-Meteo models has a lower error than CAMS or is shown to beat
  ICON-EU. See [The four extra Open-Meteo
  models](#the-four-extra-open-meteo-models).
- **An XGBoost model given the mean of the 51 members of ECMWF's ensemble forecast (ENS), from the
  00 UTC run's `T+3` band, beats an XGBoost model given ERA5 by 0.877 points [0.677, 1.080] and
  trails an XGBoost model given CAMS by 3.270 points [2.942, 3.585], but the comparison differs in
  five ways besides forecast skill, and 23% of the scored hours ended before ENS was readable from
  Dynamical.org's archive.** The five differences are lead, the 3-hourly steps of ENS's open-data
  subset, native resolution, spatial support (the area each value averages over), and model version
  (Integrated Forecasting System, IFS, Cycles 48r1 to 50r1 for ENS and 41r2 for ERA5). ERA5's
  radiation is 1 to 12 hours ahead and CAMS is a satellite retrieval whose cloud information has no
  forecast step. ENS's scored leads are 5 to 20 hours from a 00 UTC run. A live service reading
  Dynamical.org's archive gets that run from about 09:00 UTC, so 12,648 of the 54,447 scored
  site-hours (23%) ended before the run became readable there. For those hours the ENS value
  reaches a live service up to 4 hours after the hour it describes. Two exploratory comparisons were
  added after the first results. With CAMS averaged over the 3-hour
  steps of ENS's open-data subset, ENS still trails CAMS by 2.434 points [2.145, 2.729]. That
  averaging makes CAMS worse by 0.837 points [0.728, 0.942], but the same treatment makes ERA5
  better by 0.287 points [0.219, 0.359], so how much of the gap the 3-hourly steps explain is not
  measured here. See [ECMWF ENS: a longer-lead forecast than any other product on this
  page](#ecmwf-ens-a-longer-lead-forecast-than-any-other-product-on-this-page).
- **At six solar farms in Lincolnshire, an XGBoost model given the nearest Met Office weather
  station's irradiance and air temperature trails an XGBoost model given CAMS by 2.007 points
  [1.699, 2.248] and beats an XGBoost model given ERA5 by 2.082 points [1.817, 2.363], but all six
  farms share one nearest station, 17 to 31 km away.** Adding the station's irradiance to CAMS
  lowers CAMS's error by a further 0.340 points [0.277, 0.420], measured against CAMS given a
  shuffled copy of the station's irradiance. The station files end on 2025-12-31, so the
  weather-station section's row set is shorter than the page's main row set and ends eight months
  earlier. See [Met Office weather-station observations as a stand-in for a gridded
  product](#met-office-weather-station-observations-as-a-stand-in-for-a-gridded-product).

## Introduction

Several parts of this project read an estimate of weather that has already happened, not a forecast.
This page calls those parts the consumers of past weather, which are parts of the project, not
electricity customers. Capacity estimation infers a generator's size from how its output tracks
sunshine. Training history is the years of past weather that pre-training a forecasting model needs.
Historical features give a forecasting model the weather of hours already past. Disaggregation
separates hidden solar generation from demand at a substation. Each consumer reads one weather
product, and the project has to choose which. This page measures how well 12 products describe
past sunshine at six metered solar farms, and says which product each consumer should read.

**The products differ in how far ahead each value was forecast and in what area they cover, as well
as in accuracy, and both properties matter to a consumer.** A weather model run is one of the
forecasts each weather model starts every few hours. How many hours before the hour it describes a
product's value as served was forecast is the served lead. A lead of zero, written T+0, is the run's
analysis: the weather model's best estimate of the weather at the moment the run starts. ICON-D2
does not cover South West England or South Wales, which are inside the licence area of National Grid
Electricity Distribution (NGED), the distribution network operator this project forecasts for.
ICON-D2's western edge runs from about 2°W on the south coast to about 2.5°W in the Midlands.

![Figure 3: ICON-D2 has no data west of a line running from 1.8°W at 49.9°N to 3.9°W at 57.3°N.
The map also draws AROME France, which this page does not test](../roadmap/assets/weather_product_domains.svg)

| Product | What it is | Served lead | Covers all of Great Britain? | Grid spacing, native and as served | Start of the archive read here | Available after |
|---|---|---|---|---|---|---|
| CAMS | Satellite retrieval from Meteosat images, with clear-sky irradiance from CAMS's modelled aerosol, water vapour and ozone | no forecast step | yes | point values, interpolated to each requested location from the satellite pixels, about 5 km across here | 2004 | about 1 day |
| ERA5 | ECMWF reanalysis, a consistent record back to 1940 | 1 to 12 hours | yes | about 31 km native (TL639); served on a 0.25° grid | 1940 | about 5 days |
| UKV | Met Office model for the UK | T+0, the analysis | yes | 1.5 km over the UK, coarsening to 4 km at the domain's edges; served at 2 km | March 2022, which Open-Meteo backfilled from a source it does not name until August 2024 | about 4 hours |
| ICON-D2 | German weather service (DWD) model for Germany and neighbouring countries | 1 to 3 hours | no | 2.2 km; served at about 2 km | December 2022 | about 1.5 hours |
| ICON-EU | DWD model for Europe, nested inside ICON global | 1 to 3 hours | yes | 6.5 km; served at about 7 km | November 2022 | about 3.5 hours |
| ICON global | DWD global model | 1 to 6 hours | yes | 13 km; served at about 11 km | November 2022 | about 3.5 hours |
| SARAH-3 | Satellite climate data record from Meteosat images, from EUMETSAT's Satellite Application Facility on Climate Monitoring (CM SAF); the years read here, from 2021, are from its Interim Climate Data Record | no forecast step | yes | 0.05° grid, about 3.3 km east to west by 5.6 km north to south here; read at the nearest cell | 1983; read here from January 2021 | 2 to 5 days |
| ICON-DREAM-EU | DWD reanalysis over Europe, built from ICON | 1 to 3 hours | yes | about 6.5 km; read at the nearest cell | 2010; read here from September 2019 | monthly; DWD's readme states 2 to 3 months, but August 2026 was on DWD's server by 23 September 2026 |
| ECMWF-IFS-HRES | ECMWF's global model | 1 to 12 hours before 1 October 2025, when Open-Meteo's archive held only the 00 and 12 UTC runs; 1 to 6 hours from it, when the archive switched to ECMWF's newly opened real-time catalogue and gained the 06 and 18 UTC runs | yes | 9 km; served on the native O1280 grid from 1 October 2025, served grid before then not established | 2017; read here from November 2024 | not established |
| ARPEGE Europe | Météo-France's global model, a stretched grid finest over France, as distributed on its European 0.1° grid | 1 to 6 hours (4-times-daily cycle) | yes | 0.1°, about 11 km, as Open-Meteo documents | November 2022 in Open-Meteo's archive; read here from November 2024 | not established |
| DMI HARMONIE-AROME | HARMONIE-AROME run by UWC-West, the collaboration of the Danish, Dutch, Icelandic and Irish weather services, over north-west Europe up to Iceland (the DINI domain), as distributed by the Danish Meteorological Institute (DMI) | 1 to 3 hours; run interval measured at 3 hours, matching the 3-hourly update Open-Meteo documents | yes | 2 km; Open-Meteo documents it at 2 km | July 2024; read here from November 2024 | not established |
| KNMI HARMONIE-AROME | The same UWC-West HARMONIE-AROME run, as distributed hourly by the Royal Netherlands Meteorological Institute (KNMI) | not measured here; KNMI and Open-Meteo document an hourly update, which would give 1 hour | yes | 2 km model, distributed on a reduced 0.05° grid, about 5.5 km | July 2024; read here from November 2024 | not established |
| ECMWF ENS (`T+3` band) | ECMWF's 51-member global ensemble forecast, from Dynamical.org's IFS ENS catalogue, which archives only the 00 UTC run of ENS's four daily runs; a forecast from a 00 UTC run, where ERA5 and ECMWF-IFS-HRES also serve forecast leads of 1 to 12 hours | 3 to 21 hours, its shortest available band in this download; 5 to 20 hours on the hours scored here | yes | about 9 km native (O1280); served on the open-data 0.25° grid, about 28 km north to south here, and read as the overlap-weighted mean of the 0.25° cells that each generator's H3 resolution-5 cell overlaps | April 2024 | about 09:00 UTC on the run's day, from Dynamical.org's archive, as the [ENS horizons study](ens-forecast-horizons.md) finds; ECMWF disseminates the run's steps 0 to 90 by about 06:55 UTC |
| Met Office weather stations (MIDAS Open, the open release of the Met Office Integrated Data Archive System) | Observations from Met Office stations, read one station at a time, not as a gridded product: the 10 radiation stations and 38 air-temperature stations downloaded for these studies, a subset of the Met Office's network of weather stations; some of the air-temperature stations report only once a day | no forecast step (an observation of the hour itself) | no: sparse points across the UK, and the nearest radiation station is 17 to 31 km from a farm | point observations at each station | 2017; the files read here end on 2025-12-31 | not established; MIDAS Open `dataset-version-202607` ends on 2025-12-31 |

**Most of the latencies come from each service's own documentation:** CAMS's [radiation-service
notes](https://confluence.ecmwf.int/x/jOLjDw), the [ERA5 dataset
page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview),
Open-Meteo's [UKV documentation](https://open-meteo.com/en/docs/ukmo-api), and the publication times
of the German weather service's own open-data files for ICON-D2, ICON-EU and ICON global.
SARAH-3 is CM SAF's [Surface Solar Radiation Data Set – Heliosat, Edition
3](https://doi.org/10.5676/EUM_SAF_CM/SARAH/V003), published by the European Organisation for
the Exploitation of Meteorological Satellites (EUMETSAT) and available by manual order from CM
SAF; its 2-to-5-day figure is this project's own tracking, cross-referenced in the
[disaggregation
roadmap](../roadmap/disaggregation.md#an-irradiance-nowcast-would-be-a-more-useful-product), not
a figure CM SAF itself publishes. DWD publishes ICON-DREAM-EU a month at a time, after the month
ends: its readme states 2 to 3 months' delay, but August 2026 was on DWD's server by 23 September
2026. DMI's and KNMI's joint UWC-West run is documented on [KNMI's data
platform](https://english.knmidata.nl/open-data/harmonie), [Open-Meteo's DMI
documentation](https://open-meteo.com/en/docs/dmi-api), and [Open-Meteo's KNMI
documentation](https://open-meteo.com/en/docs/knmi-api). ARPEGE Europe's grid and Open-Meteo's own
archive start are from [Open-Meteo's Météo-France
documentation](https://open-meteo.com/en/docs/meteofrance-api), and ECMWF-IFS-HRES's O1280 grid
from [Open-Meteo's historical-forecast
documentation](https://open-meteo.com/en/docs/historical-forecast-api).

**Four more weather models are scored in their own section, not on this row set.** ECMWF's 9 km
global model, Météo-France's global model on its European grid, and the UWC-West HARMONIE-AROME run
as the Danish and the Dutch weather services each distribute it are all available from Open-Meteo's
archive at each generator's coordinates. Two of the four start only in July 2024, and ARPEGE changed
its radiation scheme in October 2024 (Météo-France's
[cycle 48t1](https://www.umr-cnrm.fr/old/IMG/pdf/r_r_2024-gb_web_2.pdf)), so a comparison that
includes all four needs its own row set, from November 2024, separate from the row set above: see
[The four extra Open-Meteo models](#the-four-extra-open-meteo-models).

**Nearby weather-station observations are scored in their own section, on their own shorter row
set.** The Met Office's MIDAS Open files end on 2025-12-31, so the XGBoost models given station
observations cannot use the page's hours after that date. See [Met Office weather-station
observations as a stand-in for a gridded
product](#met-office-weather-station-observations-as-a-stand-in-for-a-gridded-product).

## Data and methods

### What each error measures

**Every error on this page is a mean absolute error as a percentage of each generator's own 99th
percentile of output, which the page calls its capacity.** That capacity is a statistic of the
metered output, not the generator's registered or export capacity. Each error is that of an XGBoost
model fitted per generator to predict hourly output from one product, so the figures rank how much
each product's sunshine says about the output once that XGBoost model has been fitted.

### Where each product comes from, and its served lead

**The eight weather models come from Open-Meteo's historical-forecast archive, and ERA5 from
Open-Meteo's copy of the Copernicus archive.** `verify_era5_sources.py` checked that copy against
the Copernicus original. CAMS comes from the CAMS radiation service. The start dates for the weather
models are those of Open-Meteo's archive: DWD has run ICON global and ICON-EU since 2015, and
ICON-D2 since February 2021. SARAH-3 comes from CM SAF's own gridded files and ICON-DREAM-EU from
DWD's own gridded files, each read at the grid cell nearest each generator: 0.8 km to 2.8 km away
for SARAH-3, and 1.6 km to 5.0 km for ICON-DREAM-EU. CAMS is computed for each generator's own
coordinates. For the Open-Meteo products, Open-Meteo serves the grid cell nearest the coordinates
requested.

**Neither SARAH-3 nor ICON-DREAM-EU publishes an hourly mean, so each is converted to the mean over
the hour ending at the label, as every other product serves it.**

- **SARAH-3 publishes a snapshot every 30 minutes.** The hour labelled 12:00 is the mean of the
  snapshots stamped 11:00 and 11:30. The satellite scans Great Britain some minutes after each
  snapshot's stamp, so that pair straddles the hour's midpoint more closely than the snapshots
  stamped 11:30 and 12:00. Over the whole record, SARAH-3's hourly values track the sun's height best
  30 to 35 minutes before the label at every generator, as a mean over the hour ending at the label
  should and as CAMS's do; the later pair of snapshots would peak at the label itself.
- **ICON-DREAM-EU publishes the mean since its latest 3-hourly forecast start**, so the value
  labelled 11:00 averages 09:00 to 11:00. Each hour's own mean is recovered by subtracting the
  running mean an hour earlier, weighted by the number of hours each covers. Assuming the wrong
  start hours leaves 8% to 10% of the recovered hourly means below −1 W m⁻², against 0.06% with
  starts at 00, 03, 06 UTC and so on, the hours ICON-DREAM-EU uses. The recovered values track the
  sun best 30 minutes before the label at every generator.

Each served lead comes from how the product's archive is built:

- **ERA5's hourly radiation is not an analysis.** It comes from the reanalysis's own short
  forecasts, started at 06 and 18 UTC, at steps of 1 to 12 hours ([ERA5
  documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)).
- **Open-Meteo's weather-model archive keeps, for each hour, the freshest run that covers it.** For
  UKV, which runs every hour, the freshest run is the T+0 analysis. `verify_ukv_lineage.py` matched
  Open-Meteo's served UKV value against the Met Office's own files to within 0.55 W m⁻² at five
  instants either side of the January 2026 upgrade, at one generator's location. The check does not
  reach the backfill before August 2024.
- **For ICON-EU, the freshest run is measured.** `verify_icon_lineage.py` reconstructed each hour
  from 08 to 16 UTC on one day, at one place, from every run the German weather service still
  published. At all nine hours the freshest run reproduced Open-Meteo's served value to within
  1 W m⁻².
- **For ICON-D2 the same check does not confirm the mapping on its own.** The freshest run was the
  closest match at 7 of 9 hours, but still differed by up to 44 W m⁻². That mismatch is between
  Open-Meteo's served value and DWD's own files, so it bears on the archive, not on ICON-D2's
  accuracy. The 3-hour pattern in ICON-D2's own errors, described below, is the stronger evidence.
  The lead analysis below assumes that ICON-D2's archive is built the same way as ICON-EU's, which
  runs on the same 3-hourly cycle.
- **ICON global's lead is inferred from its 6-hourly cycle.** DWD publishes ICON global's open data
  only on the weather model's native icosahedral grid, which `verify_icon_lineage.py` does not read.
- **ICON-DREAM-EU's lead is 1 to 3 hours by construction.** Each value comes from one of the
  reanalysis's own short forecasts, started every 3 hours from an analysis.

### How the comparison was made

**Every product is scored on the same hours, with the same XGBoost model and the same features, so a
difference between two products is a difference between their irradiance alone.**

- **One row set.** An hour is kept only if all eight products cover it, which ends the record at
  the end of August 2026, where ICON-DREAM-EU's download stops. An hour is dropped if either of its
  two half-hours of metered power reads zero, whatever any product says about that hour. Two
  products' own values also decide which hours are scored, and both reject only a missing or
  physically impossible reading rather than an inaccurate one. An hour is dropped where the UKV
  snapshot recovered from Open-Meteo's archive at either end of the hour, or at a neighbouring
  hour, exceeds what the sun's geometry allows (see [Limitations](#limitations)). An hour is also
  dropped where SARAH-3 marks either of
  its two snapshots as unusable, which removes 426 of the 300,960 generator-hours SARAH-3's files
  cover (0.14%), all of them in daylight. CAMS is read in full, not only on the hours it
  rates as reliable.
- **One treatment of UKV's change of source.** Open-Meteo's UKV archive before 12 August 2024 is a
  backfill from a source Open-Meteo does not name. Every product's comparison with ERA5 is also
  reported on the hours since that date, and no XGBoost model is told which side of the date an
  hour falls.
- **One XGBoost model per generator.** A gradient-boosted tree (XGBoost) is fitted per generator on
  one product's global horizontal irradiance plus sun position, season, time of day, and ERA5's air
  temperature. Every XGBoost model is given the same temperature.
- **Folds that respect an upgrade.** Each XGBoost model is trained on some blocks of whole months
  and scored on the others, which it has never seen; each held-out block is called a fold. The Met
  Office's PS47 upgrade of UKV went live on 21 January 2026. The record is cut into five folds
  separately before and after that date, so every fold is scored by an XGBoost model trained on both
  versions. Every XGBoost model is also told which side of the date an hour falls. The 11 days from
  21 to 31 January are dropped, because the folds are cut by whole month and January counts as a
  pre-upgrade month.
- **One normalisation.** Each hour's error is divided by its own generator's capacity before any
  mean or difference.
- **Intervals from whole months.** The six generators share their weather, so each 95% interval
  comes from resampling whole calendar months 2,000 times, each time also drawing one of three
  XGBoost fits that differ only in their random seed. This page calls a difference statistically
  significant at the 5% level when its 95% interval from resampling whole months lies wholly on one
  side of zero, and not statistically significant at the 5% level when the interval includes zero.
  The test covers month-to-month variation in the weather and the fitting seed only, not variation
  between generators. The intervals are not corrected for the number of comparisons, so among the
  many exploratory rows some will reach significance by chance.
- **Six planned contrasts on the main, eight-product row set, three more on the four extra
  Open-Meteo models' own, shorter row set, two more on ENS's own row set, and three more on the
  weather-station section's own row set.** A contrast is the
  difference between two products' errors on the same hours. A comparison is planned when it was
  written down before any result existed; every other figure is exploratory, chosen
  or added after results were seen. The distinction matters because with many comparisons, about 1
  in 20 exploratory rows reaches significance at the 5% level by chance, so an exploratory result is
  a lead to follow up rather than a finding. A chart holding both kinds marks each planned row
  "(planned)". A chart whose rows are all one kind says so once, in its subtitle. The ranking rests
  on four planned contrasts: CAMS against ICON-D2, ICON-EU against ICON-D2, ICON-EU against UKV,
  and ICON global against ICON-EU. Two more were written before SARAH-3 and ICON-DREAM-EU were
  scored: SARAH-3 against CAMS, and ICON-DREAM-EU against ERA5. Each planned contrast is also
  refitted with a second set of XGBoost settings, and every one keeps its sign and its statistical
  significance at the 5% level. Every other figure on this page is exploratory. The UKV snapshot
  rebuilds, the hour-by-hour and by-lead breakdowns, and the split of ICON global by lead were
  added after the first run. Three more planned contrasts, including ECMWF's 9 km global model
  against ICON-EU, are scored on the four extra Open-Meteo models' own, shorter row set: see [The
  four extra Open-Meteo models](#the-four-extra-open-meteo-models). Two further planned contrasts,
  ECMWF ENS against ERA5 and against CAMS, are scored on ENS's own row set: see [ECMWF ENS: a
  longer-lead forecast than any other product on this
  page](#ecmwf-ens-a-longer-lead-forecast-than-any-other-product-on-this-page). Three further planned
  contrasts are scored on the weather-station section's own row set: the nearest weather station
  against CAMS and against ERA5, and CAMS with the station against CAMS with a shuffled copy of the
  station's irradiance. See [Met Office weather-station observations as a stand-in for a gridded
  product](#met-office-weather-station-observations-as-a-stand-in-for-a-gridded-product).
- **A longer record for two questions.** The year-by-year comparison with ERA5 and SARAH-3's
  comparison by satellite use a second row set, built the same way from the four products whose
  records reach back to January 2021: ERA5, CAMS, SARAH-3, and ICON-DREAM-EU. That row set holds
  115,594 generator-hours from January 2021 to August 2026. Every other figure on this page uses
  the eight-product row set, except the section [The four extra Open-Meteo
  models](#the-four-extra-open-meteo-models), which uses its own 12-product row set from November
  2024.

**The folds are cut by `studies.cross_validation` and the intervals computed by `studies.bootstrap`,
both covered by tests.**

## Results

### The XGBoost models work

**Given CAMS, the XGBoost model tracks measured power closely at every generator, in a clear, a
variable, and a dull week.** Given ERA5, the XGBoost model still follows the shape of each day, but
runs further from the measured line, most visibly in the dullest week. Figure 4 plots both XGBoost
models' out-of-fold predictions against measured power, each prediction held to the export cap as
every score on this page is.

**The three weeks are chosen by a rule that reads measured power alone, so the choice cannot favour
CAMS or ERA5.** The rule pools the six generators and considers only April to September, so that a
short midwinter day cannot dominate the choice. The clearest week is the one in which the
generators produced the most output relative to their own capacity, and the dullest week the one in
which they produced the least. The most variable week is the one whose daily output swings the most
from day to day. The clearest week and the most variable week both fell in May 2026, and the
dullest week in July 2025.

**Figure 4 shows each week as days 1 to 7, with no calendar dates, because a dated hourly series
would identify the solar farm.** With only 6 solar farms in the study, a generator's hourly
output on known dates could be matched against publicly available generation data. That match
would undo the anonymisation that names the generators only as Generator A to F. The page
therefore gives each week's month and year in the text and no finer date.

![Figure 4: An XGBoost model given CAMS tracks measured power at every generator, across a clear,
a variable, and a dull week](assets/sunshine_models_work_timeseries.svg)

**CAMS has the lowest error at each of the six generators, SARAH-3 the second lowest, and ICON-D2
the third.** Figure 5 plots every product's mean absolute error at each generator separately, one
dot per product per generator. The other five products reorder between generators: at two
generators, for instance, Open-Meteo's hourly UKV scores worse than ERA5. The lead of CAMS, then
SARAH-3, then ICON-D2, described in [CAMS describes past sunshine
best](#cams-describes-past-sunshine-best-of-the-eight-products-tested-by-a-wide-margin), therefore
holds generator by generator, rather than resting on a pooled average that one generator could
dominate.

![Figure 5: CAMS has the lowest error at each of the six generators, SARAH-3 the second lowest, and
ICON-D2 the third](assets/sunshine_models_work_error.svg)

### CAMS describes past sunshine best of the eight products tested, by a wide margin

**CAMS beats the best of the weather models tested by 2.68 points [2.42, 2.90], and the gap holds
at every generator, in every season, and in each calendar year from 2023 to 2026.**

| Product | Mean absolute error, % of capacity |
|---|---|
| CAMS | 5.09 |
| SARAH-3 | 5.49 |
| ICON-D2 | 7.76 |
| UKV given its two snapshots as separate inputs (added after the first run) | 8.13 |
| UKV rebuilt as the mean of its two snapshots (added after the first run) | 8.18 |
| ICON-EU | 8.39 |
| ICON global | 8.48 |
| ICON-DREAM-EU | 8.77 |
| UKV, Open-Meteo's hourly value | 8.86 |
| ERA5 | 9.08 |

Across the six generators, CAMS's margin over ICON-D2 ranges from 2.29 to 3.41 points. By season it
ranges from 1.81 points in winter to 3.07 in autumn, and in each calendar year from 2023 to 2026
(2026 to August) it lies between 2.62 and 2.87.

![Figure 6: CAMS's margin over ICON-D2 holds at every generator, in every season, and every year](assets/sunshine_cams_breakdown.svg)

**ERA5 has the largest error of the eight products, and [Why ERA5 describes past sunshine and wind
worse than most current weather
products](../roadmap/data-sources.md#why-era5-describes-past-sunshine-and-wind-worse-than-most-current-weather-products)
sets out ERA5's documented weaknesses.**

**The gap is not an artefact of lead or of reading CAMS in full.** On the hours ICON-D2 is served 1
hour after its run started, CAMS still beats it by 2.24 points [1.99, 2.45]. Reading CAMS in full is
the conservative choice: on the 69,573 hours CAMS rates as reliable, CAMS's margin over ERA5 widens
from 4.00 to 4.40 points.

### SARAH-3 is second to CAMS under every satellite

**CAMS beats SARAH-3 by 0.40 points [0.29, 0.50], one of the six planned contrasts, and by about as
much under each satellite SARAH-3 has used since 2021.** With the second set of XGBoost settings,
CAMS is 0.42 points ahead [0.32, 0.52]. For a generator predicted from its neighbours, CAMS is 0.45
points ahead [0.34, 0.54]. For XGBoost models trained on the 7 months since UKV's upgrade alone,
CAMS is 0.29 points ahead [0.02, 0.53], an interval that rests on 7 months and is likely too narrow.

**SARAH-3 trails CAMS by about 0.4 points in every period since 2021.** SARAH-3's retrieval over
Europe moved from Meteosat-11 to Meteosat-10 on 21 March 2023, and Meteosat-9 stood in for a
fortnight, 17 to 31 January 2022. On the longer row set from January 2021, in comparisons chosen
after SARAH-3's first results, CAMS is ahead by:

| Satellite behind SARAH-3 | CAMS's margin over SARAH-3 (points) |
|---|---|
| Meteosat-11, 2021 | 0.40 [0.20, 0.60] |
| Meteosat-11, February 2022 to 20 March 2023 | 0.50 [0.32, 0.68] |
| Meteosat-10, from 21 March 2023 | 0.41 [0.32, 0.50] |

January 2022, the month holding the fortnight from Meteosat-9, is one month and too short for an
interval: CAMS is 0.42 points ahead in that month. No XGBoost model is told which satellite an hour
comes from. CAMS also works from Meteosat's images, so a change of satellite reaches both products,
and each span above is also a different stretch of years. The table therefore shows that SARAH-3's
deficit did not change across the switches, not how SARAH-3 alone responds to a satellite. A few
days in each span came from another satellite: SARAH-3's own `platform` attribute shows Meteosat-9
standing in for 19 to 20 December 2021, inside the "Meteosat-11, 2021" span, and Meteosat-11
standing in for a few days in 2024, 2025 and 2026, inside the "Meteosat-10" span.

![Figure 7: CAMS beats SARAH-3 by about 0.4 points under every satellite, and ICON-DREAM-EU beats
ERA5 by 0.3 points](assets/sunshine_new_products.svg)

**Why CAMS beats SARAH-3 is not identified, but the gap is largest in broken cloud and sits at
four of the six generators.** CAMS is computed for each generator's own coordinates. SARAH-3 is
read from a 0.05° grid cell 0.8 km to 2.8 km from each generator. Across the generators CAMS's
margin ranges from 0.03 to 0.64 points, and at two of them it is not statistically significant at
the 5% level. Relative to CAMS's own error, SARAH-3's is about 7.1% larger in overcast hours,
10.7% larger in broken cloud, and 4.3% larger in clear hours, binned on the mean of CAMS's and
SARAH-3's own clearness index, so that neither product alone decides the bins, in comparisons
chosen after the results were seen. That pattern points at how each hourly value is built: CAMS's
hourly value is the service's own integration over the hour, while SARAH-3's is this study's mean
of two of SARAH-3's 30-minute snapshots. Another construction, such as a weighted mean of the
snapshots stamped 11:00, 11:30, and 12:00, was not tested, so part of the gap may belong to this
study's conversion rather than to SARAH-3. A steady bias could not explain the gap either, because
the XGBoost model fitted to each generator corrects a steady bias.

**SARAH-3's direct beam is modelled from its own global irradiance, so this page scores SARAH-3's
global irradiance alone.** CM SAF derives SARAH-3's direct irradiance from its global irradiance
with a model of the diffuse fraction. SARAH-3's published direct fraction varies about as little,
for a given sky clearness and sun angle, as the output of a separation model applied to the global
irradiance would: its median spread is 0.041, below the threshold of 0.05 that
`studies.served_column_checks` applies. An XGBoost model given that direct beam would measure the
diffuse-fraction model rather than the satellite.

### ICON-DREAM-EU beats ERA5 but not the ICON weather models

**ICON-DREAM-EU beats ERA5 by 0.32 points [0.12, 0.52], one of the six planned contrasts, but its
error of 8.77% is higher than that of each of the three ICON weather models.** The two reanalyses,
like SARAH-3, a satellite climate data record, are built to be consistent over time, each with one
fixed version of its weather model or retrieval, though the observations and satellites each uses
change over time. With the second set of XGBoost settings, ICON-DREAM-EU is 0.33
points ahead of ERA5 [0.14, 0.52], and for a generator predicted from its neighbours 0.37 points
ahead [0.17, 0.58]. Only 4 of the 5 folds agree in sign. On the longer row set from January 2021,
ICON-DREAM-EU is 0.39 points ahead [0.23, 0.54].

**Since August 2024, ICON-DREAM-EU's advantage over ERA5 is not statistically significant at the 5%
level. But every product's lead over ERA5 is smaller in 2025 and in 2026 than in 2024, in point
estimate.**
ICON-DREAM-EU is 0.16 points ahead on those hours [−0.09, +0.38], and 0.23 points ahead after UKV's
upgrade [−0.22, +0.57], on 7 months. In the same years CAMS's lead over ERA5 falls from 4.50 points
in 2024 to 3.69 in 2026, and ICON-EU's from 1.13 to 0.47, while ICON-DREAM-EU's gap to ICON-EU shows
no trend across those years. The smaller advantage over ERA5 is therefore not evidence that
ICON-DREAM-EU got worse; every product's lead over ERA5 narrows together. All of these comparisons
are exploratory.

**ICON-DREAM-EU and ERA5 are served at different leads, but ICON-DREAM-EU and ICON-EU can be
compared at equal leads, because both run on the same 3-hourly cycle.** ICON-DREAM-EU's radiation
comes from its own forecasts 1 to 3 hours after each 3-hourly analysis, and ERA5's from forecasts 1
to 12 hours after 06 and 18 UTC. Equal leads would favour ERA5 relative to what is measured here. In
an exploratory comparison, ICON-DREAM-EU trails ICON-D2 by 1.00 points [0.82, 1.17], trails ICON
global by 0.28 points [0.17, 0.39], and is not statistically significantly different from
Open-Meteo's hourly UKV, at −0.10 points [−0.27, +0.09].
ICON-DREAM-EU trails ICON-EU by 0.38 points at equal leads, in an exploratory comparison, and the
gap is statistically significant at the 5% level at each lead: 0.34 points [0.20, 0.47] on hours 1
hour into a run, where no de-averaging is needed, so the conversion to hourly means is not the
main cause. The gap is 0.49 points [0.34, 0.63] at 2 hours and 0.39 points [0.25, 0.54] at 3
hours.

**Before any power model sees it, the raw irradiance already ranks the products almost exactly as
the power-model contrasts above do, which is why ICON-DREAM-EU's modest lead over ERA5 is not an
artefact of the power model.** Each product's own served global irradiance, compared row for row
against CAMS's on the 76,727 daylight generator-hours every product on this page shares — no
XGBoost model, no per-generator recalibration — orders the products almost exactly as the
mean-absolute-error table above does: SARAH-3 closest to CAMS, then ICON-D2, ICON-EU, ICON global,
ICON-DREAM-EU, ERA5, with Open-Meteo's hourly value for UKV the furthest from CAMS. Correlation
with measured output, which does not use CAMS as the reference, gives the same order. Two swaps
stand out. Open-Meteo's hourly value for UKV is the furthest from CAMS by every raw measure here,
but its power-model error is lower than ERA5's. Rebuilt UKV is further from CAMS than ICON-EU,
ICON global, ICON-DREAM-EU, and ERA5 by mean absolute difference (68.53 W/m² against 58.38 to
63.27 W/m²), and correlates less with output than ICON-EU, ICON global, and ICON-DREAM-EU, yet its
power-model error, 8.18%, is lower than all four. Open-Meteo builds UKV's hourly value from the
snapshot at the hour's end, a timing error that depends on the sun's position, and a per-generator
XGBoost model given the sun's position can partly undo that error, which a raw comparison cannot.
Rebuilt as the mean of its two snapshots, UKV's correlation with CAMS rises from 0.889 to 0.912.
UKV also reads about 40 W/m² below CAMS in either form, a steady bias that each per-generator
XGBoost model removes. This comparison is exploratory.

| Product | Bias vs CAMS (W/m²) | MAD vs CAMS (W/m²) | Correlation with CAMS | Correlation with output |
|---|---|---|---|---|
| CAMS | — | — | — | 0.933 |
| SARAH-3 | +9.67 | 32.53 | 0.977 | 0.928 |
| ICON-D2 | −5.32 | 52.46 | 0.935 | 0.889 |
| ICON-EU | −9.27 | 58.38 | 0.921 | 0.875 |
| ICON global | −6.68 | 59.01 | 0.919 | 0.873 |
| ICON-DREAM-EU | −8.47 | 60.57 | 0.915 | 0.868 |
| UKV, Open-Meteo's hourly value | −44.46 | 77.21 | 0.889 | 0.846 |
| UKV rebuilt from its snapshots | −40.11 | 68.53 | 0.912 | 0.867 |
| ERA5 | −1.28 | 63.27 | 0.904 | 0.857 |

Correlation with output is the mean of each generator's own Pearson correlation between the
product's raw global irradiance and its measured output as a fraction of capacity. ICON-DREAM-EU's
mean absolute difference from CAMS is 2.70 W/m² smaller than ERA5's [−4.36, −0.87], ICON-EU's is
0.63 W/m² smaller than ICON global's [−0.93, −0.33], and ICON global's is 1.56 W/m² smaller than
ICON-DREAM-EU's [−2.56, −0.59], each resampling whole months.

### ERA5's deficit, year by year

**On matched months, January to August, ERA5 trails both satellite retrievals by more than 3 points
in every calendar year from 2021 to 2026, and ICON-DREAM-EU by up to about 0.7 points.** Every
figure in this section is exploratory, on the longer row set from January 2021, and each year's
interval comes from resampling that year's own January-to-August months alone; restricting every
year to the same months keeps a partial 2026 from being compared against the other years' full
12 months. Each year's interval rests on 8 months, so the intervals are likely too narrow. ERA5
trails CAMS by between 3.62 points (2026) and 4.74 points (2021). ERA5 trails SARAH-3
by between 3.21 points (2026) and 4.31 points (2021). ERA5 trails ICON-DREAM-EU by about half a
point to 0.68 points in each year from 2021 to 2024, a gap that is statistically significant at the
5% level in each of those years, and by 0.21 points in 2025 and 0.18 points in 2026, where the gap
is not statistically significant at the 5% level.

**No year from 2021 to 2024 has a gap between ERA5 and ICON-DREAM-EU that differs from 2025's by a
margin statistically significant at the 5% level.** Resampling each year's months independently of
2025's, and restricting every year to January to August, the change from each of 2021 to 2024 to
2025 is −0.27 to −0.47 points, and every interval includes zero, so a reader should not conclude
that ICON-DREAM-EU's gap to ERA5 has genuinely narrowed rather than moved within the noise.

![Figure 8: On matched months, ERA5 trails both satellite retrievals by more than 3 points in every
year from 2021 to 2026](assets/sunshine_era5_by_year.svg)

### ICON-D2 is the best weather model tested, and its advantage shrinks within hours of each run

**ICON-D2 beats ICON-EU across the record by 0.62 points [0.48, 0.76], by 1.03 points [0.87, 1.18]
in the first hour after each run, and by only 0.32 points [0.16, 0.48] in the third hour.** ICON-D2
and ICON-EU run on the same 3-hourly cycle, so at any given hour both are served at the same lead.

| Hour (UTC) | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|
| Served lead of both | 3 h | 1 h | 2 h | 3 h | 1 h | 2 h | 3 h | 1 h |
| ICON-D2 − ICON-EU (points) | −0.40 | −1.38 | −0.60 | −0.20 | −1.48 | −0.76 | −0.47 | −0.97 |

The table shows hours 09 to 16 UTC. Each hour is labelled by its end, so 12 UTC is the mean from
11:00 to 12:00. The by-lead figures above average over 07 to 19 UTC. At those two edge hours, with
the sun low, the pattern weakens: 1 hour into a run, ICON-D2's advantage is only 0.35 points at
07 UTC and 0.06 points at 19 UTC.

![Figure 9: ICON-D2's advantage over ICON-EU shrinks within hours of each run](assets/sunshine_icon_d2_leads.svg)

**The time of day does not explain the pattern around noon.** The hours 12 and 13 UTC sit on either
side of solar noon, yet ICON-D2's advantage is 0.20 points at 12 UTC, 3 hours into a run, against
1.48 points at 13 UTC, 1 hour in. ICON-EU shows no such pattern with lead. Against UKV rebuilt from
its snapshots, which is always served at T+0, ICON-EU is 0.20 to 0.24 points behind at every lead
from 1 to 3 hours into its runs, in a comparison added after the first run; only the 1-hour gap is
statistically significant at the 5% level (2 h: 0.21 [−0.01, +0.44]; 3 h: 0.20 [−0.02, +0.42]).

**The mechanism is not identified, and the rate of decay may not hold elsewhere.** ICON-D2 takes the
weather at its boundaries from ICON-EU. The six generators sit about 160 to 200 km east of ICON-D2's
western boundary, so air arriving on a westerly wind may carry ICON-EU's weather to them within
hours. Another candidate is ICON-D2's own data assimilation: an advantage built into each run's
starting state would be expected to decay over the run's first hours. By a 3-hour lead only 0.32
points of the advantage remain, so the as-served advantage should not be assumed to hold a day
ahead. The comparison at longer leads belongs to
[#810](https://github.com/openclimatefix/nged-substation-forecast/issues/810), the study comparing
weather models for UK power forecasting.

### ICON-EU against ICON global and UKV

**Most of ICON global's 0.10-point deficit to ICON-EU sits in the hours ICON global is served
further ahead.** ICON global runs every 6 hours, so half its hours are served 4 to 6 hours after the
run started, where ICON-EU is at 1 to 3 hours. On those hours ICON global is 0.19 points behind
[0.09, 0.30]. On the hours where the two leads are equal ICON global is 0.02 points behind [−0.03,
+0.08]. So at equal leads the difference between ICON global and ICON-EU is not statistically
significant at the 5% level, though a difference as large as 0.08 points is not excluded. The split
also divides the hours of the day: the longer leads fall at 10 to 12 and 16 to 18 UTC, so a
difference in either weather model's skill by time of day would show up here as an effect of lead.

**ICON-EU beats Open-Meteo's hourly value for UKV by 0.48 points [0.29, 0.65], one of the six
planned contrasts, but UKV is ahead of ICON-EU, by a margin close to the 5% threshold, once UKV's
hour is rebuilt from its own snapshots.** Each ICON product's archived value for an hour is a mean
over that hour. UKV publishes a snapshot each hour, and Open-Meteo builds UKV's hourly value from
the snapshot at the hour's end, rescaled by the change in the sun's angle. In comparisons added
after the first run, averaging UKV's snapshots at both ends of the hour cuts UKV's error by 0.68
points, which puts UKV 0.21 points ahead of ICON-EU [0.03, 0.38]. The interval's lower end, 0.03
points, is close to zero, so this result should not be read as settled. An XGBoost model given the
two snapshots as separate inputs puts UKV 0.26 points ahead of ICON-EU [0.09, 0.43].

**Giving each XGBoost model the neighbouring hours improves both products by about 0.2 points, and
leaves rebuilt UKV about as far ahead of ICON-EU.** An XGBoost model given ICON-EU's hourly mean for
the hour before and after, as well as the hour itself, improves by 0.18 points. The same context
improves UKV's rebuilt hour by 0.19 points. With context on both, ICON-EU is 0.23 points behind
rebuilt UKV [0.05, 0.39], and since August 2024 0.26 points behind [0.02, 0.50]. With context,
ICON-EU is 0.09 points behind UKV's two snapshots given as separate inputs [−0.08, +0.25], a
difference that is not statistically significant at the 5% level.

**Every rebuilt UKV construction scores ahead of ICON-EU across the record, and UKV is also served
at the shorter lead.** UKV is served at T+0 and ICON-EU 1 to 3 hours ahead, so equal leads would if
anything favour ICON-EU. On the 7 months after UKV's upgrade alone, ICON-EU is 0.15 points behind
rebuilt UKV in point estimate [−0.23, +0.57], with only 2 of 5 folds agreeing in sign. That interval
rests on 7 months and is likely too narrow.

![Figure 10: UKV rebuilt from its snapshots beats ICON-EU, which beats ICON global and Open-Meteo's
hourly UKV](assets/sunshine_icon_eu_rivals.svg)

### UKV rebuilt from its snapshots beats ERA5

**UKV rebuilt from its snapshots beats ERA5 in every period tested, but Open-Meteo's hourly value
for UKV beats ERA5 only across the whole record.** In comparisons added after the first run, UKV
rebuilt as the mean of its two snapshots beats ERA5 by 0.90 points [0.72, 1.09] across the record,
by 0.79 points [0.57, 1.00] since Open-Meteo's own UKV downloader started in August 2024, and by
0.62 points [0.45, 0.75] after the January 2026 upgrade. The post-upgrade interval rests on 7
months and is likely too narrow, though all 5 folds agree in sign.

**Open-Meteo's hourly value for UKV is 0.22 points ahead of ERA5 across the record [0.06, 0.39], and
0.14 points ahead since August 2024 [−0.04, +0.35].** After the upgrade neither fit finds a
difference that is statistically significant at the 5% level. XGBoost models trained on the whole
record, both sides of the upgrade, score Open-Meteo's hourly UKV 0.06 points worse than ERA5 on the
post-upgrade months [−0.08, +0.24]. XGBoost models trained on the post-upgrade months alone score
the two within 0.01 points of each other [−0.16, +0.14]. Both post-upgrade intervals rest on 7
months and are likely too narrow. No XGBoost model for rebuilt UKV was trained on the post-upgrade
months alone.

**Every UKV contrast with ERA5 mixes the products with their leads.** UKV is served at T+0 and ERA5
at 1 to 12 hours, and equalising the leads could narrow the gap.

![Figure 11: UKV rebuilt from its snapshots beats ERA5 in every period; Open-Meteo's hourly UKV does
not since August 2024](assets/sunshine_ukv_against_era5.svg)

### A product's own direct beam adds little

**For every product tested except ERA5, an XGBoost model given the product's own published direct
beam, instead of a separation model's estimate from the product's own total, improves by 0.03 to
0.10 points.** The separation model is the [Erbs et al.
(1982)](https://doi.org/10.1016/0038-092X(82)90302-4) correlation, applied to each product's own
global irradiance. CAMS improves by 0.08 points and UKV by 0.10. The three ICON weather models and
ICON-DREAM-EU improve by 0.03 to 0.05 points each. ERA5's effect is not statistically significant at
the 5% level. Every effect is smaller than each planned contrast except ICON global against ICON-EU,
which is about as large as UKV's. For CAMS and ERA5 the own-beam effects agree with [the
beam/diffuse study](beam-diffuse-split.md), which tested whether the published beam carries
information or merely encodes the total differently. SARAH-3 is not tested, because its direct beam
is modelled from its own global irradiance, as described under [SARAH-3 is second to CAMS under
every satellite](#sarah-3-is-second-to-cams-under-every-satellite). ICON-DREAM-EU's own-beam gain of
0.034 points [0.010, 0.059] disappears on the record panel's longer row set, where it is −0.005
points [−0.027, +0.018], so this result should not be read as settled.

![Figure 12: Every product with its own direct beam, except ERA5, gains 0.03 to 0.10 points from
it](assets/sunshine_own_beam.svg)

### The ranking holds for a generator predicted from its neighbours

**An XGBoost model trained on five generators and applied to the sixth ranks the products the same
way, with errors 0.10 to 0.20 points larger.** Each held-out generator is predicted by an XGBoost
model that never saw that generator and never saw the held-out months at any generator. The
neighbours share their weather. An XGBoost model trained on the held-out months at the other
generators would already have seen how each of those days turned out, and would overstate how well
an XGBoost model carries over to a generator it has never seen. The prediction is converted back to
megawatts with the generator's own capacity, so this neighbour test assumes the capacity is known
and says nothing about capacity estimation itself. The training generators all sit within 34 km of
the held-out generator. The increase in error is therefore likely smaller here than for a generator
further from its neighbours, which this page does not measure. For disaggregation, the result
supports the ranking but not the size of the error.

![Figure 13: The ranking holds for a generator predicted from its neighbours](assets/sunshine_neighbours.svg)

### Implied capacity from month to month

**Once the seasonal cycle is removed, the implied capacities of CAMS and SARAH-3 are the steadiest
of the eight products from month to month, but both satellite retrievals swing the most with the
seasons.** A month's implied capacity is the ratio of metered output to what a fixed south-facing
panel at 30° tilt predicts per megawatt from each product, over the hours with the sun above 10°
and no curtailment cap. With each calendar month's average removed, CAMS's month-to-month spread is
6.5% and SARAH-3's 6.6%, a difference that is not statistically significant at the 5% level. The
other six products spread by 8.0% to 9.8%, and for each of them the difference from CAMS's spread
is statistically significant at the 5% level. CAMS, however, implies a capacity 7%, 23%, and 10%
below its annual mean in November, December, and January, and SARAH-3 one 4%, 15%, and 7% below.
No other product strays more than 11.1% from its annual mean in any of those three months.
December's figure rests on four Decembers, 22 generator-months at generators that share their
weather, and Figure 14 draws no interval.

**This study cannot say which product is right about December.** December's sun stays low all day,
which is where a satellite retrieval and this page's assumed panel geometry are both at their least
reliable. The two satellite retrievals dip in the same months, by different amounts. The swing may
therefore come from the retrieval or from the assumption that every panel faces south at 30° tilt,
which this page does not verify. Removing the seasonal cycle also needs several years of each
calendar month, which a capacity estimator run on a short window does not have.

![Figure 14: Of the eight products tested, CAMS and SARAH-3 imply the steadiest capacity from
month to month but swing the most with the seasons](assets/sunshine_implied_capacity.svg)

### The four extra Open-Meteo models

**At six solar farms in Lincolnshire, from November 2024, none of the four extra Open-Meteo models
has a lower error than CAMS or is shown to beat ICON-EU, and one of the three planned contrasts is
not resolved.** ECMWF-IFS-HRES, ARPEGE Europe, and the UWC-West HARMONIE-AROME run as DMI and KNMI
each distribute it are fetched at each generator's own coordinates, the same way as UKV and the
three ICON weather models above, but score here on a shorter, later row set: 40,243 common
generator-hours, November 2024 to August 2026, all 12 products sharing every hour. Two of the four
weather models, DMI's and KNMI's HARMONIE-AROME, cover only from July 2024, which the row set's
later start already accommodates.

**The row set starts in November 2024 for two reasons, and ARPEGE's own longer record confirms
that start date is not hiding a step in irradiance.** Open-Meteo's UKV archive is a backfill before
12 August 2024, and Météo-France's
[cycle 48t1](https://www.umr-cnrm.fr/old/IMG/pdf/r_r_2024-gb_web_2.pdf), on 15 October 2024,
replaced ARPEGE's radiation scheme; the first whole month after the later of the two changes is the
start. ARPEGE's own per-site build reads back to January 2024, further than this row set's own
November 2024 start, so it can check whether cycle 48t1 left a step: ARPEGE's ratio to
ECMWF-IFS-HRES shows no step at
cycle 48t1's date, but that ratio cannot isolate ARPEGE's own change, because ECMWF's own
[Cycle 49r1](https://www.ecmwf.int/en/about/media-centre/news/2024/forecast-upgrade-improves-wind-and-temperature-predictions)
went operational on 12 November 2024, inside the same window. ARPEGE's ratio to CAMS does
rise: from 0.94 to 1.07 across the 10 calendar months from January to October 2024, to 1.01 to 1.32
from November 2024, and it is higher in 8 of the 10 calendar months the report holds on both sides
of the change (`report.md`), roughly equal in June and 0.01 lower in April. The row set's November
2024 start rests on the documented change of radiation scheme.

**The three contrasts planned before this row set's first result, each also refitted at a second
XGBoost setting:**

- **KNMI HARMONIE-AROME, as Open-Meteo's archive serves it, trails ICON-EU by 0.56 points of
  capacity [0.32, 0.75], and the gap holds at the second XGBoost setting (0.50 points
  [0.22, 0.73]).** KNMI's served run interval is not measured here (below), so this contrast mixes
  weather-model skill with lead. KNMI and Open-Meteo both document an hourly update for this feed,
  which would put KNMI at a 1-hour lead on every row against ICON-EU's 1 to 3 hours. The plan's own
  reasoning, written before either result existed, made the same prediction: a shorter lead biases
  the contrast in KNMI's favour rather than against it.
- **DMI HARMONIE-AROME, as Open-Meteo's archive serves it, trails ICON-D2 by 1.02 points
  [0.73, 1.26], and the gap holds at the second setting (0.94 points [0.66, 1.18]).** DMI's own
  run interval is inferred at 3 hours from weak peaks in its temperature, the same as ICON-D2's
  (below), so the two sit at the same served lead on every row if that inference holds: DMI trails
  ICON-D2 by 1.09 points [0.78, 1.35] on 07–19 UTC, close to the unsplit figure above, and the loss
  is positive in 21 of 22 months (point estimates, no interval; three months are below 0.2 points)
  and at all six farms (0.77 to 1.19 points, exploratory). DMI's 2 km native grid is matched in
  resolution to ICON-D2's 2.2 km, as KNMI's reduced 0.05° output grid, about 5.5 km, is to
  ICON-EU's 6.5 km, so a coarser grid cell is unlikely to explain either loss.

  **The loss is larger at the first hour after each run than at the rest of the day, where DMI's
  and KNMI's feeds — the same UWC-West HARMONIE-AROME run — serve almost the same global
  irradiance.** Split by the first served hour after each run, exploratory and found after these
  results: 1.55 points [1.22, 1.83] at that hour against 0.84 points [0.52, 1.11] at the rest of
  the day. DMI's and KNMI's own archives read almost the same global irradiance at that hour, 61%
  of rows within 2 W/m² of each other and 36% exactly equal, against 14% within 2 W/m² and 3%
  exactly equal at other hours. This page does not establish why the loss concentrates there, and
  cannot tell apart three candidate explanations: HARMONIE-AROME's own first forecast hour, for
  example cloud spin-up; how Open-Meteo turns that hour's accumulation into an hourly mean; and
  that the same hour is also ICON-D2's own freshest, one hour into its run. KNMI's loss against
  ICON-EU does not concentrate at that first hour the same way, exploratory: 0.70 points
  [0.44, 0.94] at the first hour against 0.54 points [0.26, 0.79] at other hours, both excluding
  zero.

- **ECMWF-IFS-HRES against ICON-EU is not resolved: −0.09 points [−0.37, +0.17], and about 0
  points [−0.28, +0.27] at the second setting.** IFS-HRES's own run interval is measured (below).
  ECMWF runs IFS-HRES every 6 hours, or every 12 for its longest forecasts, where ICON-EU runs
  every 3, so the plan's own reasoning, written before either result existed, expected IFS-HRES's
  longer average lead to bias the pooled contrast against it. Splitting the row set by whether the
  two products' leads match, exploratory and added after the first result, is consistent with that
  direction but does not resolve it: the gap widens to −0.31 points [−0.69, +0.06] at matched
  leads, in IFS-HRES's favour relative to the pooled figure, and narrows to +0.05 points
  [−0.27, +0.35] where IFS-HRES's lead is the longer of the two, with neither interval excluding
  zero. No row fell where IFS-HRES's lead was the shorter. The matched-lead and longer-lead rows
  also differ in hour of day, so the split does not isolate lead on its own. Hour by hour the gap
  ranges from −0.65 to +0.65 points with no steady pattern by time of day, exploratory and found
  after these results. Open-Meteo's historical-forecast archive still serves IFS-HRES throughout
  this row set; only its upstream source changed, on 1 October 2025, from a source Open-Meteo does
  not document, which held only the 00 and 12 UTC runs, to ECMWF's own real-time open-data
  catalogue. That date is a change of source inside this row set as well as a change of cadence.
  The folds are not cut at that date and no XGBoost model is told which side of it an hour falls,
  so the contrast is instead split before and after it: −0.06 points [−0.49, +0.30] before, and
  −0.12 points [−0.49, +0.22] on or after, neither resolved. This page does not establish whether
  the served grid changed across the switch. ECMWF's Cycle 49r1 went operational on 12 November
  2024, inside this row set, which the panel's own splits do not separate out.

**How each served lead was measured, or why it was not (exploratory):** `check_new_products.py`
reads a single-site hourly 2 m temperature fetch for each of the four models and looks for an
hour where the mean absolute second difference of temperature stands out from the rest of the
day — a run switch. Temperature carries its own diurnal cycle, a smooth afternoon warming that
raises this curvature from 12 to 18 UTC for every one of the four models, unlike radiation's sharp
on/off pattern at sunrise and sunset. A ratio read against the whole day's median absorbs that
afternoon hump into the baseline, and a run switch that sits inside the hump can then read as
unremarkable. ECMWF-IFS-HRES's temperature peaks 12 hours apart before 1 October 2025 and 6 hours
apart from it, when Open-Meteo's historical-forecast archive switched its upstream source for
ECMWF-IFS-HRES to ECMWF's own real-time open-data catalogue; ARPEGE's peaks 6 hours apart across
its whole record, matching its public four-times-daily cycle. Against the whole day's median,
neither HARMONIE-AROME model shows a single peak standing out clearly enough to read a cycle off
directly, which is where that afternoon hump hid DMI's own signal. Against each hour's two
neighbours instead — `check_new_products.py`'s local-prominence row, below the whole-day ratio for
each product — DMI HARMONIE-AROME peaks at every hour divisible by 3, 1.07 to 1.29 times its
neighbours, the same phase as ICON-D2's own 3-hourly cycle. This method has no positive control:
no single-site temperature fetch for ICON global, whose run interval is already known from its
documented 6-hourly cycle, is on disk to check the method finds a cadence it already knows. The
strongest independent evidence for DMI's 3-hour cycle is that DMI's and KNMI's radiation are
exactly equal at 36% of rows at 07, 10, 13, 16, and 19 UTC, against 3% at other hours (above),
together with Open-Meteo's own documented 3-hourly update for DMI's feed. KNMI's temperature shows
no peak at a fixed phase, so KNMI gets no measured interval; that absence is what an hourly update,
with a run switch at every hour, would look like, matching the hourly update KNMI and Open-Meteo
both document.

**Each product's own mean absolute error, exploratory:** CAMS 5.20%, SARAH-3 5.61%, ICON-D2
7.81%, ECMWF-IFS-HRES 8.29%, ICON-EU 8.38%, ICON global 8.52%, UKV (Open-Meteo's hourly value)
8.79%, ICON-DREAM-EU 8.81%, DMI HARMONIE-AROME 8.83%, ERA5 8.87%, KNMI HARMONIE-AROME 8.94%, and
ARPEGE Europe 9.30%, the highest.

![Figure 15: CAMS and SARAH-3 still lead when four more weather models are added, November 2024 to
August 2026](assets/sunshine_all_leaderboard.svg)

![Figure 16: HARMONIE-AROME, as Open-Meteo serves it from DMI's and KNMI's feeds, trails the ICON
model of similar grid spacing; IFS-HRES against ICON-EU is not
resolved](assets/sunshine_all_contrasts.svg)

**The direct beam Open-Meteo serves for ARPEGE Europe and KNMI HARMONIE-AROME is not scored,
because Open-Meteo derives it from each weather model's global irradiance with a separation model
(exploratory).** Open-Meteo's documentation says so for both products: only global irradiance is
native, the diffuse share comes from the Razo, Müller and Witwer separation model, and the direct
beam is the remainder
(<https://open-meteo.com/en/docs/meteofrance-api>, <https://open-meteo.com/en/docs/knmi-api>).
`sources.py`'s check agrees: the served direct fraction varies by only 0.018 inside a bin of
similar cloud and sun height, against a threshold of 0.05 — the signature of a model applied to
the product's own global irradiance, carrying no information beyond it. **DMI HARMONIE-AROME's
served direct beam is not scored either, for a different reason: on daytime rows, it is exactly
zero in 50% of them and exceeds the served global flux, which is physically impossible, in 6
(`product_checks.md`).** This page does not establish whether that defect belongs to DMI's model
or to how Open-Meteo's archive serves it; `sources.py` documents the check, and the pattern is not
a separation model's signature. All three get a global arm only. ECMWF-IFS-HRES's own direct beam
passes both checks and is scored: its own split is 0.08 points worse than a synthetic Erbs split
derived from its own global irradiance alone [0.02, 0.14], an exploratory result in the opposite
direction to the finding on the main row set above; on this row set ICON-EU's own beam
(+0.01 [−0.05, +0.07]) and ICON global's (+0.04 [−0.00, +0.08]) show no gain either, and only 2 of
the 5 folds agree in sign on ECMWF-IFS-HRES's own result, so not settled ([A product's own direct
beam adds little](#a-products-own-direct-beam-adds-little)).

**Reproducing this section's figures:**

```bash
uv run python studies/beam_diffuse_split/build_dataset.py --source ecmwf-ifs-hres
uv run python studies/beam_diffuse_split/build_dataset.py --source arpege-europe
uv run python studies/beam_diffuse_split/build_dataset.py --source dmi-harmonie-arome
uv run python studies/beam_diffuse_split/build_dataset.py --source knmi-harmonie-arome
uv run python studies/beam_diffuse_split/fetch_open_meteo_point.py --model ecmwf-ifs-hres  # and arpege-europe, dmi-harmonie-arome, knmi-harmonie-arome
uv run python studies/beam_diffuse_split/check_new_products.py
uv run python studies/beam_diffuse_split/weather_products.py --panel all
uv run python studies/beam_diffuse_split/weather_product_charts.py
```

`check_new_products.py`'s night-jump table also reads a single-site hourly 2 m temperature fetch
for each of the four models, a throwaway download not scripted with its own command
(`sources.temperature_site_b_path_for`); without it, that table prints only "no positive control"
and skips each model's row. The report lands in `past_weather_v2/solar_all/report.md`, and the
direct-beam and run-interval checks in `past_weather_v2/product_checks.md`, both alongside the
eight-product report's own directory (see [Reproducing the
figures](#reproducing-the-figures)).

### ECMWF ENS: a longer-lead forecast than any other product on this page

**An XGBoost model given the mean of ECMWF ENS's 51 members, from the 00 UTC run's shortest-lead
band (`T+3`), beats an XGBoost model given ERA5 by 0.877 points of capacity [0.677, 1.080], and
trails one given CAMS, the best product on this page, by 3.270 points [2.942, 3.585].** Both
contrasts are planned and statistically significant at the 5% level. The control member alone, the
one member run from ECMWF's unperturbed best estimate of the atmosphere, beats ERA5 by 0.637 points
[0.437, 0.836]. The other 50 members each start from a slightly perturbed state. The comparison
differs in five ways besides forecast skill:

- **Lead:** the hours ahead of the target hour that the value was forecast.
- **Step width:** the 3-hourly steps belong to the open-data subset of ENS that Dynamical.org's
  archive holds. ECMWF's full ENS output is hourly to 90 hours ahead, then 3-hourly to 144 hours.
- **Native resolution:** ENS's native grid is about 9 km (O1280) since IFS Cycle 48r1, which ECMWF
  made operational on 27 June 2023. ERA5's native grid is about 31 km (TL639).
- **Spatial support:** the area each value averages over. Both products are served on a 0.25° grid,
  and the section below on ERA5's 3 by 3 block sets out what each value averages over.
- **Model version:** ENS ran IFS Cycle 48r1 until 12 November 2024, then 49r1 until 12 May 2026,
  then 50r1. ERA5 uses Cycle 41r2, which ECMWF made operational on 8 March 2016.

**The comparison is not lead-equal, and ERA5's lead is not like for like with an operational ENS
lead.** ERA5's radiation is a forecast 1 to 12 hours ahead, UKV's archive holds the analysis,
ICON-D2 and ICON-EU read 1 to 3 hours ahead, and CAMS is a satellite retrieval whose cloud
information has no forecast step. Dynamical.org archives only the 00 UTC run of ENS's four daily
runs. `T+3` is the download's label for the first band of that run, covering leads of 3 to 21 hours.
After this section's row set is joined to the rest of the page's hours, the scored leads are 5 to 20
hours, with a mean of 12.63 hours against ERA5's 6.52. ERA5's four-dimensional variational (4D-Var)
assimilation windows run 09 to 21 UTC and 21 to 09 UTC, and its forecasts start from the 06 and 18
UTC analyses. Each of those analyses therefore already uses observations up to 3 hours after the
forecast's start. An operational ENS run uses no observation after its start time, so the 6.11-hour
gap in mean lead is not like for like with an operational ENS lead.

**A live service reading Dynamical.org's archive gets the 00 UTC run from about 09:00 UTC, so 12,648
of the 54,447 scored site-hours (23.2%) ended before the run became readable there.** ECMWF itself
disseminates the run's steps 0 to 90 by about 06:55 UTC, and 2,582 of the scored site-hours (4.7%)
end at or before that time. The [ENS horizons study](ens-forecast-horizons.md) takes the 09:00 UTC
time from Dynamical.org's archive. Every consumer on this page reads a value after its hour has
passed, so for the 12,648 site-hours the ENS value reaches a live service up to 4 hours after the
hour it describes. That delay is shorter than CAMS's one-day delay.

**Splitting the scored hours at 09:00 UTC does not show that ENS's gain over ERA5 depends on whether
the run was readable (exploratory, post hoc).** In the split, ENS beats ERA5 by 0.992 points [0.766,
1.230] on the 41,799 site-hours that end after 09:00 UTC and by 0.499 points [0.297, 0.703] on the
12,648 that end at or before it. ENS trails CAMS by 3.790 points [3.363, 4.184] on the hours that
end after 09:00 UTC and by 1.553 points [1.333, 1.746] on the hours that end at or before it. ENS's
gain is 9.6% of ERA5's error on the hours that end after 09:00 UTC and 9.3% on the hours that end at
or before it, so the split mostly separates midday hours from morning hours.

**The row set is ENS's own, shorter and later than the rest of the page, so each pooled interval
resamples 30 calendar months, against 45 for the main row set.** Generator E covers 23 of the 30
months. ENS's own archive here starts on 1 April 2024, well after the main row set's December 2022
start, so this section scores 54,447 common site-hours from 1 April 2024 to 10 September 2026. That
end date is the last date of the ERA5 grid, which `build_dataset.py` trims every product to. ENS's
own runs go on to 22 September 2026, and the join drops none of the ENS rows dated on or before 10
September 2026. The same shape of caveat applies to the [four extra Open-Meteo
models](#the-four-extra-open-meteo-models) and to
[ICON-DREAM-EU](#icon-dream-eu-beats-era5-but-not-the-icon-weather-models).

**ERA5 and CAMS are refit on this section's own row set, following this page's shared-rows rule, and
every XGBoost model in this section carries eight feature columns.** The eight columns are the
shared solar geometry and calendar features, an era flag for UKV's January 2026 upgrade, and either
the product's own global irradiance or, for ENS, ENS's mean-of-members irradiance and temperature.
The era flag does not mark ENS's own cycle changes (49r1 and 50r1). ERA5's and CAMS's temperature
column is ERA5's own air temperature, one of the shared features every product on this page reads,
because the CAMS radiation service publishes no temperature.

**ENS is scored on global irradiance only, like every other global-only product on this page,
because the free open-data subset of ENS that Dynamical.org's archive is built from carries no
direct-beam field, although ECMWF's full ENS catalogue does.**

**ENS's `T+3` band holds seven 3-hour steps per run, and each run is rebuilt into 19 hourly values
by the clear-sky-index reconstruction that the ENS horizons study picked as best for ENS's
radiation.** The reconstruction interpolates the ratio of ENS's radiation to clear-sky radiation
between the steps, then multiplies the ratio by each hour's clear-sky radiation.

**Averaging all 51 members, the control member and the 50 perturbed members, lowers the error by
0.240 points [0.147, 0.344] against the control member alone (exploratory):** the XGBoost model
given the control member alone scores 8.507% of capacity, against 8.267% for the model given the
mean of the members. Only the member mean and the control member are scored, so the spread of the 51
members, which is what makes ENS a probabilistic forecast, is not assessed here.

![Figure 17: ENS's own forecast trails CAMS by 3.3 points, and beats
ERA5](assets/ens_past_solar_leaderboard.svg)

![Figure 18: ENS beats ERA5 but trails CAMS by more than three
points](assets/ens_past_solar_planned_contrasts.svg)

**Two contrasts are planned: ENS's mean-of-members forecast against ERA5, and against CAMS.** Both
contrasts, the `T+3` band, the member mean, and the clear-sky-index reconstruction were fixed in the
written instructions for the first run, before the first model was fitted. Those instructions are a
working file outside the repository, so the repository does not record the ordering. ERA5's and
CAMS's results on the main row set were published before ENS was planned, so a large gap between ENS
and CAMS was predictable. Both contrasts hold at the second hyperparameter setting. At that setting
ENS beats ERA5 by 0.816 points [0.612, 1.024] and trails CAMS by 3.244 points [2.912, 3.568], each
within 0.07 points of the primary setting's figure, so neither ordering depends on the
hyperparameter setting. Every other comparison in this section is exploratory.

**With CAMS averaged over the 3-hour steps of ENS's open-data subset, ENS still trails CAMS by 2.434
points [2.145, 2.729] (exploratory).** CAMS is averaged over the seven 3-hour steps and rebuilt to
hourly values with the code that rebuilds ENS's own hours. The rebuilt CAMS scores 0.837 points
worse than CAMS as first measured [0.728, 0.942]. The same treatment makes ERA5 better, so how much
of the gap the 3-hourly steps explain is not measured here. The 2.434 points that remain combine
ENS's longer lead, its native resolution and spatial support, and its model version. This section
does not separate those differences.

**ERA5 given the same 3-hour treatment is better by 0.287 points [0.219, 0.359], and ERA5 averaged
over the 3 by 3 block of served 0.25° cells around each generator's nearest cell is better by 0.157
points [0.117, 0.196].** This section does not establish why averaging ERA5 to 3 hours improves it.

**The 3 by 3 block averages over more area than either ENS's value or ERA5's nearest cell, so the
0.157 points measures the effect of averaging over 9 cells, not the effect of ENS's own weighting of
2 to 4 cells.** Both products are served on the same 0.25° grid, where one cell covers 464 km² at
these generators' latitude. ERA5's value is the one cell nearest the generator. ENS's value is the
mean of the grid cells that the generator's H3 resolution-5 cell (253 km²) overlaps, weighted by
overlap. The four H3 cells the generators sit in overlap 2 to 4 grid cells each, 9 distinct grid
cells in all, and the 2 cells nearest the generators for ERA5 are 2 of those 9. The 3 by 3 block is
9 grid cells covering 4,179 km².

**ENS still beats ERA5 in each case, by 0.590 points [0.417, 0.774] against the rebuilt ERA5, and by
0.721 points [0.544, 0.910] against the 3 by 3 ERA5.** The three extra XGBoost models were added
after the first science review, so they are exploratory. Each carries the same eight feature columns
as every other XGBoost model in this section and is scored on the same 54,447 site-hours. At the
second hyperparameter setting every sign and ordering holds, and each of the six contrasts moves by
less than 0.06 points.

![Figure 19: Averaging ERA5 and CAMS over 3-hour steps narrows both of ENS's
gaps](assets/ens_past_solar_exploratory_contrasts.svg)

**How much ENS's longer lead raises ENS's error is not measured here, because lead and time of day
are the same variable for a 00 UTC run.** Every ENS value in this section comes from the 00 UTC run,
so an hour's lead equals its hour of day, and no model can separate lead from the daily cycle of
sunshine. The ENS horizons study reports error rising by 0.61 points [0.45, 0.79] from day 0 to day
1, which is 24 hours of extra lead. Scaled to this section's 6.11 hours of extra lead, that rise is
about 0.16 points, with no interval and from the other page's own rows. That scale is the only guide
here. The 0.16 points is smaller than the 0.287 points by which the 3-hour treatment lowers ERA5's
error, and close to the 0.157 points by which the 3 by 3 average lowers it. Lead, step width, native
resolution, spatial support, and model version move the 0.877-point gain in ways this section cannot
separate, so the gain may be too high or too low.

**The control member alone beats ERA5 by 0.637 points [0.437, 0.836] here (8.507% against 9.144%),
matching the ENS horizons study's day 0 on largely overlapping rows.** That study finds the control
member beating ERA5 by 0.64 points [0.42, 0.87] (8.41% against 9.06%). The two comparisons use
different rows: the ENS horizons study scores 50,643 site-hours at leads 0 to 23 hours from 15 April
2024, and this section scores 54,447 site-hours at leads 5 to 20 hours from 1 April 2024. The ENS
horizons study reports the ensemble mean beating ERA5 by 0.29 points [0.05, 0.53] at day 1, leads 24
to 47 hours, against 0.877 points here at leads 5 to 20 hours.

**ERA5 and CAMS refit here score 9.144% and 4.996%, against 9.080% and 5.085% on the main row set.**
The refit figures differ by +0.064 and -0.089 points because this section's row set holds 54,447
site-hours from April 2024, against the main row set's 76,727 from December 2022.

**The two planned contrasts keep their sign at all six generators, but the six generators are not
independent replication.** The six generators sit in a box 24.9 km north to south by 22.7 km east to
west, on 2 ERA5 cells and 4 distinct ENS inputs, and they share the resampled months. In an
exploratory comparison, ENS beats ERA5 by 0.410 to 1.355 points across the six generators, and
trails CAMS by 2.841 to 3.730 points at every one of them. Generators B and D share one ENS input,
because they fall in the same H3 resolution-5 cell, as do generators C and E. ENS beats ERA5 by
1.041 points at B and 1.355 at D, and by 0.757 points at C and 0.410 at E, so generators that read
identical ENS values differ. Generator E covers 23 months against 30 at each of the other five, and
generator D scores 6,951 site-hours against 9,727 to 10,758 at A, B, C, and F. At generator A, ENS
beats ERA5 in 3 of the 5 folds, and every other per-generator contrast has the same sign in all 5
folds.

**ENS was compared only with ERA5 and CAMS, on rows that begin in April 2024, so this section
supports a narrow reading for each of the four uses in ["What to use"](#what-to-use).** ENS's error
of 8.267% of capacity must not be set against the errors of ICON-EU or UKV, which come from
different, longer row sets. The [ENS horizons study](ens-forecast-horizons.md) is where ENS's role
as a forecast is assessed.

- **Capacity estimation:** this section computes no implied capacity for ENS, so it gives no
  evidence for or against ENS.
- **Training history:** ENS describes past sunshine better than ERA5 by 0.877 points [0.677, 1.080]
  and worse than CAMS by 3.270 points [2.942, 3.585], with the five differences above not separated.
  This section does not train a forecasting model on ENS and run the model on ENS forecasts, so it
  does not show whether ENS's match to its own forecasts offsets its lower accuracy than CAMS in
  describing past sunshine.
- **Historical features in the live service:** the ENS value for an hour can reach the service after
  the hour has ended (23.2% of the scored site-hours at Dynamical.org's archive latency). The
  section gives no comparison with ICON-EU or rebuilt UKV, so it gives no reason to replace either.
- **Disaggregation:** CAMS is 3.270 points more accurate than ENS on these rows, and this section
  does not show that ENS's shorter delay compensates.

Reproducing this section's figures:

```bash
uv run python studies/beam_diffuse_split/ens_past_solar.py
uv run python studies/beam_diffuse_split/ens_past_solar_charts.py
```

The report lands in `past_weather_v2/ens_past_solar/report.md`, alongside the ENS horizons study's
own directory. `ens_past_solar.py` reads `data/studies/weather/ENS/beam_diffuse_ens.parquet` (no
fetch needed) and `weather_products.py`'s own saved solar dataset; `--report-only` rebuilds the
report from the saved losses without refitting, checking a fingerprint against what a fresh run
would now fit.

### Met Office weather-station observations as a stand-in for a gridded product

**At six solar farms in Lincolnshire, an XGBoost model given the nearest Met Office weather
station's irradiance and air temperature has a mean absolute error 2.007 points of capacity above an
XGBoost model given CAMS [1.699, 2.248], and 2.082 points below an XGBoost model given ERA5 [1.817,
2.363].** Adding the station's irradiance to CAMS lowers the error by a further 0.340 points [0.277,
0.420], measured against CAMS given a shuffled copy of the station's irradiance. The shuffled copy
keeps the station's average irradiance for each month and hour but not its hour-to-hour weather, so
that any gain cannot come from the XGBoost model merely having one more input column. These are the
section's three planned contrasts. Each planned contrast is statistically significant at the 5%
level, has the same sign in all five folds of contiguous whole months when the six generators are
pooled, and keeps its sign and significance at the second hyperparameter setting (+1.957 [+1.657,
+2.193], −2.084 [−2.357, −1.828], and −0.321 [−0.401, −0.257]).

**All six farms take the same nearest radiation station, 17 to 31 km away, so these results describe
one station's pyranometer record, from an instrument that measures global horizontal irradiance, set
against two gridded products.** The 95% intervals cover month-to-month weather and the fitting seed.
They do not cover the choice of station, or how a different station or region would compare. The six
farms also share their ERA5 grid cells, so the six per-generator results are not six independent
replications.

**The section asks how well a pyranometer some tens of kilometres from a farm stands in for the
irradiance at the farm, not how accurate the station is.** A station arm is an XGBoost model given
the station's hourly global horizontal irradiance in place of a gridded product's irradiance. In the
files read here, the diffuse and direct irradiance columns are empty at all 10 stations, so no
station arm has a split into direct beam and diffuse light.

**Three contrasts were written down before the first model was fitted, and every other comparison in
this section is exploratory.** The three planned contrasts were written into the study's plan file,
in a commit made before the first XGBoost model was fitted (the commit stays in the git history of
the study's pull request after the plan file is deleted at merge): the nearest station's irradiance
and temperature against CAMS, the same against ERA5, and CAMS with the nearest station against CAMS
with a shuffled copy of the station's irradiance. The shuffled copy is permuted within each farm,
each calendar month of each year, and each hour of day, so the shuffled copy keeps the station's
monthly average at each hour and removes the station's hour-to-hour weather. CAMS with the shuffled
copy is called the padded control below, and CAMS with the real station column is called the blend
below. The padded control carries the same nine feature columns as the blend. ERA5's and CAMS's
results on the main row set were published before this section was planned, so it was known in
advance that CAMS beats ERA5, though not where the station would fall.

#### What the station files hold, and where they end

**The files are the quality-controlled version (`qc-version-1`) of the Met Office's MIDAS Open
dataset, version 202607, from 2017-01-01 to 2025-12-31: hourly global irradiation from the 10
radiation stations downloaded for these studies, and air temperature from the 38 weather stations
downloaded.** Some of the 38 weather stations report once a day, and so cannot meet the coverage
rule below. MIDAS reports each irradiance row as an amount in kJ m⁻² over the hour ending at its
label. This section converts each amount to a mean irradiance in W m⁻², keeping the hour-ending
convention of the page's power aggregation. An air-temperature row is a spot reading at its label.

**This section's row set ends on 2025-12-31 and holds 60,033 common site-hours over 37 calendar
months, against the main row set's 76,727 site-hours from 2022-12-01 to 2026-08-31.** The last hour
in the files is 2025-12-31 23:00 UTC. This page does not establish how often the Met Office releases
a new version of the files. Each pooled interval here therefore resamples 37 months, and treats
neighbouring months as independent.

**Measured power tracks the irradiance of the same hour more strongly than the irradiance of the
hour before or after, for both the station and CAMS, so the station's hours are labelled the same
way as CAMS's.** Across the 47,369 site-hours whose neighbouring hours are also scored (exploratory,
post hoc), the correlation of measured power with the station's irradiance is 0.880 for the same
hour, against 0.768 for the hour before and 0.819 for the hour after. The same three correlations
for CAMS's irradiance are 0.926, 0.804, and 0.824.

**CAMS's irradiance at the farms correlates more strongly with the nearest station's irradiance than
ERA5's irradiance does (exploratory).** Across the six farms, CAMS's irradiance follows the nearest
station's irradiance with a correlation of 0.939 and a mean difference, product minus station, of
−9.7 W m⁻². ERA5's irradiance follows the station's irradiance with a correlation of 0.890 and a
mean difference of −10.3 W m⁻². The mean station irradiance is 273.6 W m⁻². Each product is read at
the farms, 17 to 31 km from the station, so these figures are not a validation of either product
against the station.

**Six irradiance values in the whole files are corrected, and no quality-control flag is used to
drop an hour.** The station-reading code makes two corrections, counted over the whole files before
the row set is cut: the code clips three slightly negative irradiance values to zero, and sets 3
hours to missing where the station reports more than 5 W m⁻² in an hour that starts and ends with
the sun below the horizon. No upper-limit check against clear-sky irradiance was run beyond that
night-time test. The files' quality-control flags are kept as delivered, and the page does not
decode the flag values. At 107 of the 60,033 site-hours, the nearest radiation station's flag
differs from 6, the value the flag carries at most hours. Without those 107 site-hours, the three
planned contrasts are +2.009 [+1.702, +2.251], −2.068 [−2.337, −1.817], and −0.341 [−0.421, −0.277]
(exploratory, post hoc). The station-reading code then drops the hours where a station an arm needs
has no value. Of the 60,611 candidate site-hours, 578 (0.95%) were dropped because a station value
was missing.

#### How the stations were chosen

**The station rule was written before any score existed: the nearest station with a usable value at
no less than 99% of a farm's hours.** Stations are ranked by great-circle distance, with ties going
to the lower station number. A threshold of 100% leaves at least one farm with no eligible station,
because none of the downloaded stations has a usable value at every one of that farm's hours. At
every farm, the rule chose the nearest downloaded radiation station, and no nearer radiation station
was skipped for low coverage. The nearest air-temperature stations are chosen by the same rule from
their own 38 stations, so a farm's nearest air-temperature station can differ from its nearest
radiation station.

**No radiation station left out of the download is nearer to any farm than the nearest radiation
station chosen here.** For radiation, the rule is applied only to the 10 stations downloaded. The
station-metadata file lists 76 more radiation stations whose record overlaps this section's years.
The nearest of those 76 stations is 107 km or more from every farm, so no radiation station in that
file is nearer to a farm than the nearest radiation station chosen here.

| Kind | Rank | Distance range (km) | Distinct stations | Lowest coverage | Most nearer stations skipped |
|---|---|---|---|---|---|
| radiation | 1 | 17 to 31 | 1 | 0.9933 | 0 |
| radiation | 2 | 36 to 51 | 1 | 0.9991 | 0 |
| radiation | 3 | 52 to 89 | 2 | 0.9903 | 1 |
| air temperature | 1 | 7 to 17 | 3 | 0.9956 | 0 |
| air temperature | 2 | 18 to 28 | 2 | 0.9989 | 2 |
| air temperature | 3 | 18 to 31 | 4 | 0.9969 | 2 |

**The table gives ranges pooled over the six farms, because a station-to-farm mapping would narrow
where a metered generator is.**

**Only the third-nearest radiation rank varies across the six farms, so the nearest and
second-nearest arms each rest on a single station.** The nearest radiation station is the same at
every farm, and the second-nearest radiation station is also the same at every farm. The
third-nearest radiation rank holds two stations. The air-temperature stations vary across the six
farms at every rank. The gap between the nearest-station and second-nearest-station arms therefore
compares two single stations.

#### The XGBoost model given the nearest station tracks measured power

**An XGBoost model given the nearest station follows measured power at every generator across a
clear, a variable, and a dull week.** Figure 20 draws out-of-fold predictions, each held to the
export cap as the scores are. The weeks are picked from measured power alone, by the rule the page's
earlier "models work" figure uses, so no input's values enter the choice.

![Figure 20: An XGBoost model given the nearest station follows measured power at every generator,
across a clear, a variable, and a dull week](assets/station_past_solar_models_work.svg)

#### Mean absolute error of each arm, and the three planned contrasts

**CAMS and ERA5 both score worse on this section's shorter row set than on the main row set, so this
section's absolute errors are not comparable with the page's main leaderboard.** CAMS scores 5.299%
of capacity here against 5.085% on the main row set, and ERA5 9.389% against 9.080%. The XGBoost
models fitted on the main row set score 5.170% for CAMS and 9.250% for ERA5 on the 59,873 site-hours
the two row sets share. The remaining 0.129 and 0.139 points are attributed to the shorter training
span, the folds re-cut on the shorter row set, and the 160 site-hours that this section's row set
holds and the main row set lacks, which the common rows of the [blending
study](blending-weather-products.md) do include. No interval is computed for these two differences,
so fitting-seed variation is not separated from them. The station arms carry the same
shorter-history handicap as the CAMS and ERA5 arms refitted on this row set.

**The 160 site-hours that only this section's row set holds do not change what the three planned
contrasts show (exploratory, post hoc).** On the 59,873 site-hours the two row sets share, the three
planned contrasts are +2.008 [+1.701, +2.250], −2.082 [−2.364, −1.817], and −0.341 [−0.420, −0.277].

![Figure 21: The nearest station beats ERA5 but not CAMS, and adds to
CAMS](assets/station_past_solar_leaderboard.svg)

**The station arm sits between CAMS and ERA5.** The XGBoost models score, in percent of capacity:
given CAMS and the nearest station, 4.968% [4.620, 5.287]; given CAMS alone, 5.299% [4.990, 5.599];
given the mean of the three nearest stations, 6.948% [6.402, 7.391]; given the nearest station,
7.307% [6.740, 7.768]; given the second-nearest station, 8.396% [7.775, 8.887]; given ERA5, 9.389%
[8.761, 9.919]; and given the third-nearest station, 9.397% [8.737, 9.931]. Each interval in this
paragraph and in Figure 21 carries the month-to-month swing that all arms share, so two arms whose
intervals overlap can still differ significantly. The intervals for the nearest station and for the
mean of the three nearest stations overlap in Figure 21, but the contrast, which compares the two
arms on the same site-hours and so cancels the shared swing, separates those two arms by 0.359
points [0.245, 0.471]. Only the contrasts test one arm against another.

![Figure 22: The nearest station trails CAMS by 2.007 points, beats ERA5 by 2.082, and lowers CAMS's
error by 0.340](assets/station_past_solar_planned_contrasts.svg)

**Every planned contrast compares two arms with the same number of feature columns, and swapping the
station's air temperature for ERA5's moves the nearest-station arm's error by no more than 0.036
points (exploratory).** In every planned contrast, and in every exploratory contrast except the
three that set an arm carrying an extra column against a plain product, the two arms carry the same
number of feature columns, 8 or 9, with `colsample_bytree` at 1, so every tree sees every column.
The three exceptions are CAMS padded with a shuffled station column against plain CAMS, ERA5 padded
the same way against plain ERA5, and the blend against plain CAMS. The report prints every arm's
feature columns. The nearest-station arm reads the station's air temperature where the CAMS and ERA5
arms read ERA5's air temperature. The nearest-station arm scores 0.012 points below the same arm
given ERA5's air temperature [−0.036, +0.011] (exploratory). The 95% interval puts the effect of the
temperature swap within 0.036 points of zero, small beside the 2-point gaps to CAMS and ERA5, so the
temperature swap is not what separates the arms.

**The nearest station's own irradiance is what carries the nearest-station arm's skill.** The
nearest-station arm scores 6.734 points below the same arm given a shuffled copy of the station's
irradiance [6.082, 7.402] (exploratory). The station's hour-to-hour irradiance is therefore what the
nearest-station arm adds over the shuffled arm, and the station's air temperature, the season and
time-of-day features, and the sun position are not.

**Each planned contrast has the same sign at all six generators, and an interval that excludes zero
at each (exploratory).** The nearest station trails CAMS by between 1.174 and 2.656 points at the
six farms, and beats ERA5 by between 1.555 and 2.528. The blend gains between 0.221 and 0.469 points
over its padded control. At generator E, which has the fewest site-hours, 4 of 5 folds agree in sign
for the CAMS contrast and for the blend contrast. The six farms share one nearest station and two
ERA5 grid cells, so the per-generator results are not independent replications.

![Figure 23: Each planned contrast has the same sign at all six
generators](assets/station_past_solar_per_generator.svg)

**Splitting the planned contrasts by half of the year gives the same signs (exploratory, post
hoc).** The nearest station trails CAMS by 2.457 points [2.280, 2.628] in April to September and by
1.286 points [0.761, 1.794] in October to March. The nearest station beats ERA5 by 2.080 points
[1.784, 2.431] in April to September and by 2.087 points [1.643, 2.523] in October to March. The
blend's gain over the padded control is 0.193 points [0.157, 0.235] in April to September and 0.576
points [0.494, 0.670] in October to March. The split was added after the first results. Each
half-year interval resamples only 18 or 19 calendar months, so these intervals are likely too
narrow.

#### What a station adds to a gridded product

**Adding the nearest station's irradiance to ERA5 lowers ERA5's error by 2.432 points [2.196, 2.673]
against ERA5 with a shuffled copy of the station's irradiance, and adding the station's irradiance
to CAMS lowers CAMS's error by 0.340 points [0.277, 0.420] against the same padding (exploratory for
ERA5, planned for CAMS).** The padded controls themselves (exploratory, post hoc) score within 0.012
points of the plain products (+0.009 [−0.004, +0.021] for CAMS and +0.012 [−0.018, +0.041] for
ERA5). The 95% intervals put any effect of the shuffled column within 0.041 points of zero, small
beside the 0.340-point gain the real station column brings, so the padding does not explain that
gain. Against the XGBoost model given CAMS alone, which carries one column fewer, the blend gains
0.331 points [0.267, 0.411] (exploratory, post hoc).

![Figure 24: The real station column lowers ERA5's error by 2.432 points against a shuffled column,
and CAMS's error by 0.331 points against plain CAMS; a shuffled station column, or the station's own
temperature, moves no error by more than 0.041](assets/station_past_solar_controls.svg)

#### Averaging three stations, and stations further away

**Averaging the three nearest stations beats the nearest station alone by 0.359 points [0.245,
0.471], and the second-nearest station scores 1.089 points worse than the nearest [0.921, 1.254]
(exploratory).** The mean of the three nearest stations is still 1.648 points behind CAMS [1.330,
1.912] and 2.441 points ahead of ERA5 [2.217, 2.671]. The third-nearest station scores 2.091 points
worse than the nearest station [1.848, 2.345]. The third-nearest station does not differ
statistically significantly at the 5% level from ERA5 (+0.009 points [−0.222, +0.244], with 3 of 5
folds agreeing in sign, so the 95% interval runs from 0.222 points better than ERA5 to 0.244 points
worse). The second-nearest station is 0.993 points ahead of ERA5 [0.774, 1.230].

![Figure 25: Averaging three stations beats the nearest station alone, and the third-nearest station
scores no better than ERA5](assets/station_past_solar_stations.svg)

**The rise in error from the nearest to the second- and third-nearest stations cannot be attributed
to distance alone.** The second-nearest station is 36 to 51 km from a farm and the third-nearest 52
to 89 km, against 17 to 31 km for the nearest. But the stations at the three ranks are also
different instruments at different places, with different records, and at one or more farms the
third rank skipped a nearer station that failed the coverage rule.

#### What this section does and does not show

**The section shows that an XGBoost model given one station's pyranometer record, 17 to 31 km from
the farms, predicts the farms' output with a lower mean absolute error than an XGBoost model given
ERA5's value at the farm, and a higher mean absolute error than an XGBoost model given CAMS's
satellite retrieval, at these six farms from December 2022 to December 2025.** The section does not
show how the result changes with the station chosen, because every farm shares one nearest radiation
station. The section does not show how the result changes with distance, because rank, place, and
instrument change together. The section does not test the station's accuracy either. Nor does the
section test any product with a beam and diffuse split, because the diffuse and direct columns are
empty at all 10 stations in the files read here. The files read here end on 2025-12-31, and the
section does not test any near-real-time station feed, so the section says nothing about a live
service.

## What to use

**These recommendations rest on six generators in one part of Lincolnshire, and weigh accuracy
against availability and coverage.**

- **Capacity estimation: provisionally CAMS, fitted over at least a year of history or with November
  to January left out.** Every accuracy figure on this page comes from an XGBoost model fitted to
  each generator's own metered output, and that fit absorbs any steady bias in a product. Capacity
  estimation cannot make that correction, so the evidence for capacity estimation is the
  implied-capacity measure alone. That measure assumes every panel faces south at 30° tilt, and the
  November-to-January rule was chosen after seeing Figure 14. SARAH-3's implied capacity swings less
  with the seasons than CAMS's, but this page cannot say which of the two is right about December.
  The implied-capacity measure is not the project's capacity estimator. Running that estimator with
  each product over rolling windows, and scoring it against known capacity, would test this
  recommendation directly.
- **Training history: CAMS, with a caveat.** CAMS gives the most accurate description of past
  sunshine of the eight products, and SARAH-3, 0.40 points behind, the second most accurate. A
  forecasting model pre-trained on CAMS and then run on a weather forecast meets a different product
  at inference. This page does not measure how much that mismatch changes forecast accuracy, nor
  whether pre-training on ICON-DREAM-EU, which is built from the same ICON weather model as the ICON
  forecasts and reaches back to 2010, narrows the mismatch.
  ICON-DREAM-EU beats ERA5 as a reanalysis, but by less than any of the three ICON weather models
  does. ECMWF-IFS-HRES is archived from 2017, further back than any weather model this page tests
  except ICON-DREAM-EU, and on its own shorter row set it beats ERA5 by 0.58 points [0.36, 0.81]
  (5 of 5 folds, exploratory).
- **Historical features in the live service: ICON-EU, or UKV with its two snapshots averaged;
  with either product, include the neighbouring hours.** Rebuilding UKV's hour from Open-Meteo's
  snapshots produces physically impossible values in the first hours after sunrise (see
  [Limitations](#limitations)), mostly before the January 2026 upgrade, so a service reading
  rebuilt UKV has to screen those values as this study did. Since the upgrade, the difference
  between rebuilt UKV and ICON-EU is not statistically significant at the 5% level [−0.23,
  +0.57]. CAMS arrives a day late and ERA5 about 5 days late, so neither can supply the last few
  hours at run time. ICON-D2 was more accurate at the six
  generators, all of which sit inside its domain. Reading ICON-D2 at generators east of ICON-D2's
  western edge, and ICON-EU elsewhere, is therefore an option, though a forecasting model trained on
  the result would mix two products across generators. Open-Meteo's UKV archive has three stretches:
  a backfill from a source Open-Meteo does not name until August 2024, Open-Meteo's own download of
  the Met Office's files until the January 2026 upgrade, and the upgraded weather model since. A
  forecasting model trained on the whole UKV archive therefore mixes all three. Every figure here
  scores the archive's freshest run for each hour. At run time the most recent hours are not yet in
  the archive in that form: UKV arrives about 4 hours and ICON-EU about 3.5 hours after each run
  starts, so the last few hours come from an older run, at a longer lead than any scored here.
  ECMWF-IFS-HRES, on its own shorter row set, is not distinguishable from ICON-EU (−0.09 points
  [−0.37, +0.17], not resolved) and covers all of Great Britain, so it is a candidate alongside
  ICON-EU where a service already reads Open-Meteo's ECMWF feed, though this page's evidence for it
  is thinner: a shorter row set, and a change of Open-Meteo's ECMWF source on 1 October 2025 inside
  that row set. IFS-HRES is disseminated several hours after each run starts, later than ICON-EU's
  about 3.5 hours (see [ECMWF's dissemination
  schedule](https://confluence.ecmwf.int/display/DAC/Dissemination+schedule)), so at run time its
  last hours come from an older run at a longer lead than scored here.
- **Disaggregation: CAMS where its one-day delay allows, otherwise ICON-EU or rebuilt UKV.** The
  ranking holds for a generator predicted from its neighbours.
- **ECMWF ENS: none of the four uses above follows from this page's evidence alone.** ENS was
  compared only with ERA5 and CAMS, on rows from April 2024, so its 8.267% error must not be set
  against the errors of ICON-EU or UKV, which come from different, longer row sets. The [ENS
  horizons study](ens-forecast-horizons.md) is where ENS's role as a forecast is assessed, and the
  [ENS section above](#ecmwf-ens-a-longer-lead-forecast-than-any-other-product-on-this-page) says
  what the evidence supports for each use.
- **Weather-station observations: at these six farms, the nearest station is a worse input than CAMS
  and a better input than ERA5, the only two gridded products scored on the weather-station
  section's rows, and the evidence rests on one station.** An XGBoost model given the nearest Met
  Office radiation station's irradiance and air temperature, 17 to 31 km away, scores 7.307% of
  capacity on the weather-station section's rows, between CAMS's 5.299% and ERA5's 9.389%. These
  percentages are not comparable with the main leaderboard's percentages, because the
  weather-station section's row set is shorter and scores CAMS and ERA5 worse than the main row set
  does. For capacity estimation, training history, and disaggregation, CAMS covers every hour the
  station files do, so none of those three uses needs a station in place of CAMS. Adding the
  station's irradiance to CAMS lowers CAMS's error by a further 0.340 points [0.277, 0.420]. For
  historical features in the live service, the page recommends ICON-EU or rebuilt UKV because CAMS
  arrives about a day late. The station was not scored against ICON-EU or UKV, and no near-real-time
  station feed was tested, so this page cannot say whether a station would serve that use. Where
  CAMS cannot be used, the nearest station is a better input than ERA5, by 2.082 points [1.817,
  2.363]. The third-nearest station, 52 to 89 km away, does not differ statistically significantly
  at the 5% level from ERA5 (+0.009 points [−0.222, +0.244]).

**Giving an XGBoost model CAMS, ERA5, UKV, ICON-D2, ICON-EU, and ICON global at once beats giving it
CAMS alone, with CAMS's
neighbouring hours and its split of sunlight into direct beam and diffuse light, by 0.13 points
[0.10, 0.17], a post hoc comparison written up in [Does blending weather products beat the best
single weather product?](blending-weather-products.md#solar-a-blend-beats-cams-given-its-neighbouring-hours)**

## Limitations

- **Hours where Open-Meteo's UKV archive holds a physically impossible sunrise value are dropped
  for every product.** Open-Meteo stores UKV's hourly value rounded to 1 W m⁻² and serves each
  snapshot by undoing the rescaling for the sun's angle described under [ICON-EU against ICON
  global and UKV](#icon-eu-against-icon-global-and-ukv). In the first hours after sunrise that
  rescaling ratio can reach the hundreds, so undoing it turns the stored mean, and its rounding,
  into a value with no physical meaning — up to 16,537 W m⁻² against a real GHI ceiling of about
  1,400 W m⁻². The
  study drops each snapshot that exceeds 1.1 times the top-of-atmosphere flux at its own instant,
  1,554 of 238,320 hourly snapshots from March 2022 to September 2026, together with every hour
  built from one. Every dropped hour falls between 05:00 and 10:00 UTC, and almost all of them
  predate UKV's January 2026 upgrade — 2.8% of pre-upgrade hours against 0.1% after it, and none
  from April 2026 onwards. Because every product is scored on one shared row set, dropping these
  snapshots removes 1,768 generator-hours (2.3%) from every product's figures on this page, not
  only UKV's, leaving 76,727 common rows. Two smaller defects in Open-Meteo's UKV archive are not
  corrected here: at 318 raw hours (186 of the 76,727 scored rows) the served UKV beam exceeds the
  served global irradiance, which is physically impossible and bears on the own-beam figure above,
  and Open-Meteo's UKV wind file has a 94-hour gap, which does not bear on this page.
- **The evidence is regional.** All six generators sit in one 25 km by 23 km box, inside ICON-D2's
  domain, about 160 to 200 km east of its western edge, and in only two ERA5 grid cells. The ranking
  is untested elsewhere in Great Britain, and nothing here measures a site near ICON-D2's edge or
  outside its domain. Wind is measured separately on [Which weather product best describes past
  wind?](weather-products-for-past-wind.md).
- **The intervals describe these six generators only.** The intervals resample months, not
  generators, so they say nothing about how a generator elsewhere would rank the products.
- **Every product except CAMS is read at a grid cell, not at the generator.** ERA5 at its nearest
  0.25° cell, the Open-Meteo weather models at the cell Open-Meteo serves, SARAH-3 at a cell 0.8 km
  to 2.8 km away, and ICON-DREAM-EU at a cell 1.6 km to 5.0 km away. A cell that misses a
  generator's own clouds handicaps the product read from it.
- **The four extra Open-Meteo models are scored on a shorter, later, and separate row set.** ECMWF's
  9 km global model, Météo-France's ARPEGE on its European grid, and the UWC-West HARMONIE-AROME
  run as DMI and KNMI each distribute it are compared against each other and against the three ICON
  weather models and UKV, from November 2024, but the ranking above, from December 2022, says
  nothing about them. Three of the four (all but ECMWF-IFS-HRES) score a global arm only: the
  direct beam Open-Meteo serves for ARPEGE Europe and KNMI HARMONIE-AROME is Open-Meteo's own
  separation of their global irradiance, and DMI HARMONIE-AROME's holds a defect this page does not
  attribute to the weather model or to Open-Meteo's archive of it, so no split or Erbs arm is fitted
  for them. See [The four extra Open-Meteo
  models](#the-four-extra-open-meteo-models).
- **The comparison is not lead-equal.** The served lead is part of what a consumer receives, so the
  as-served ranking answers the consumer's question. A comparison at a held-equal lead is a forecast
  comparison, and belongs to
  [#810](https://github.com/openclimatefix/nged-substation-forecast/issues/810), the study comparing
  weather models for UK power forecasting.
- **The ECMWF ENS section is scored on its own shorter, later, and separate row set, and its
  product is a forecast, not past weather.** ENS's own archive here starts on 1 April 2024, so its
  54,447 common site-hours say nothing about the ranking above, from December 2022. ENS is compared
  here with ERA5 and CAMS only, and is not compared directly with ICON-EU or rebuilt UKV. ICON-EU and
  rebuilt UKV also beat ERA5, on the main row set, so ENS's gain over ERA5 does not show that ENS
  competes with the products the ["What to use"](#what-to-use) section recommends. The section
  scores ENS because it is the live service's own forecast product, and this page is where this
  project compares weather products. Of the scored site-hours, 23% end at or before the time a live
  service reading Dynamical.org's archive can read the 00 UTC run. The section does not separate
  ENS's lead, step width, native resolution, spatial support, and model version, so ENS's gaps to
  ERA5 and CAMS are not a measure of forecasting skill alone. Only the member mean and the control
  member are scored, so the spread of the 51 members, which is what makes ENS a probabilistic
  forecast, is not assessed here.
  This section is solar only. See [ECMWF ENS: a longer-lead forecast than any other product on this
  page](#ecmwf-ens-a-longer-lead-forecast-than-any-other-product-on-this-page).
- **ECMWF's IFS Cycles 49r1 and 50r1 fall inside the ENS section's window and are not treated as
  breaks.** ECMWF introduced Cycle 49r1 on 12 November 2024 and Cycle 50r1 on 12 May 2026. The ENS
  section's folds and its `era_code` feature are cut around the Met Office's UKV upgrade only, so an
  XGBoost model given ENS trains and scores across three ECMWF model versions without being told
  which version produced each run.
- **Every accuracy figure is recalibrated per generator.** A product with a large but stable bias
  scores well here. The implied-capacity measure is the only evidence on this page about each
  product's uncorrected bias.
- **The figures rest on the capacity table as rebuilt in September 2026.** Each generator's capacity
  is its 99th percentile of output from the `effective_capacity` table.
  [#825](https://github.com/openclimatefix/nged-substation-forecast/issues/825), which refreshes the
  beam/diffuse study's figures on the current power and capacity tables, may move the absolute
  figures and the size of every contrast, because each generator's errors are divided by its own
  capacity. A change to the capacities alone cannot flip the sign of a contrast that agrees at all
  six generators, such as CAMS against ICON-D2, but a change to the power table can move any figure.

- **The weather-station section is scored on its own shorter row set, and its result rests on one
  station.** The Met Office's MIDAS Open files end on 2025-12-31, so the section's 60,033 site-hours
  run from 2022-12-01 to 2025-12-31, against the main row set's 76,727 site-hours, which run to
  2026-08-31. Only 10 radiation stations were downloaded, by a rule this page does not document, and
  the station-metadata file lists 76 more whose record overlaps the section's years. The nearest of
  those 76 is 107 km or more from every farm, so none is nearer than the station chosen. All six
  farms take the same nearest radiation station, 17 to 31 km away, so the section compares one
  pyranometer with two gridded products. The section's intervals cover month-to-month weather and
  the fitting seed, not the choice of station. The second- and third-nearest stations differ from
  the nearest in place, instrument, and record as well as in distance, so the gaps between the three
  ranks cannot be read as an effect of distance alone. In the files read here, the diffuse and
  direct irradiance columns are empty at all 10 stations, so no station arm has a beam split. The
  only gridded products the weather-station section scores are CAMS and ERA5, so the section does
  not rank the station against the other products on this page. The section is solar only.

## Reproducing the figures

Cut SARAH-3 and ICON-DREAM-EU at each generator, check every product's timestamps against the sun,
build every product's dataset, then run the study. `check_new_products.py` here relies on its
default `--sources`, all six products the second round adds, so its own fetches under [The four
extra Open-Meteo models](#the-four-extra-open-meteo-models) have to run first, or this command
fails on a missing file rather than silently dropping those four models' checks.

```bash
uv run --with netcdf4 python studies/beam_diffuse_split/extract_site_series.py --product sarah-3
uv run python studies/beam_diffuse_split/extract_site_series.py --product icon-dream-eu
uv run python studies/beam_diffuse_split/check_new_products.py
for source in open-meteo ukv icon-d2 icon-eu icon-global cams sarah-3 icon-dream-eu; do
  uv run python studies/beam_diffuse_split/build_dataset.py --source $source
done
uv run python studies/beam_diffuse_split/build_dataset.py --source cams \
  --min-cams-reliability 0 --suffix _allhours
uv run python studies/beam_diffuse_split/weather_products.py --panel long record
```

The eight-product report lands in
`data/studies/beam_diffuse_split/past_weather_v2/solar_long/report.md`, and the longer record's in
`solar_record/report.md` beside it, with each panel's year-by-year table in `era5_by_year.parquet`.
`weather_products.py --report-only` rebuilds a report from the saved losses without refitting, and
neither mode overwrites an existing file. `--concurrent-fits` lowers how many XGBoost models are
fitted at once. The report
also prints the distances between the generators and to ICON-D2's edge, the ERA5 cells they fall in,
and every number the charts share with it. The charts come from `uv run python
studies/beam_diffuse_split/weather_product_charts.py`, which computes the numbers the report does
not print, the leaderboard's intervals and Figures 4 and 5, from the saved losses without refitting
any XGBoost model. `check_new_products.py` writes the timing and direct-beam checks on all six
products the second round adds to `past_weather_v2/product_checks.md`. The served-lead check runs
with `uv run
--with cfgrib python studies/beam_diffuse_split/verify_icon_lineage.py --model icon-eu`, against the
runs the German weather service still publishes, which cover about one day.

The weather-station section has three scripts of its own, run in this order: two that build and
chart the section, and a check on the page's numbers. `station_past_solar.py` builds the section's
row set, fits every arm, and writes `report.md` and `losses.parquet` to
`data/studies/beam_diffuse_split/past_weather_v2/station_past_solar/`.
`station_past_solar_charts.py` draws Figures 20 to 25 from those two files, and stops unless the
saved report matches the report the current code produces. `check_station_page_numbers.py` stops
unless every number the section adds to the page appears in `report.md`. The MIDAS Open files come
from `studies/weather_downloads/fetch_midas_open.py`.

```bash
uv run python studies/beam_diffuse_split/station_past_solar.py
uv run python studies/beam_diffuse_split/station_past_solar_charts.py
uv run python studies/beam_diffuse_split/check_station_page_numbers.py
```
