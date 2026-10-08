# Data sources

The inputs to the forecasting system: NGED's power-flow data, supporting NGED files, weather data,
and electricity price data.

> **Status legend** — ✅ Ingested today · 🚧 Planned ingestion · 🔬 Research. The ECMWF ENS NWP and the
> NGED time-series JSON / metadata are ✅ ingested; the supporting NGED files and the extra weather
> datasets are 🚧 planned (needed for switching-event detection, capacity estimation, and the
> training history), and
> NGED's published electricity-network model and the electricity price data are 🔬 research. See the
> [roadmap index](index.md) for status conventions.

---

## Data from NGED

### Provided on NGED's AWS S3 bucket (live, updated every 6 hours)

| Source | Status | Description |
|---|---|---|
| **Time-series JSON files** | ✅ | Half-hourly power flow + metadata per substation / customer meter in the trial area. Ingested by OCF to produce operational forecasts. Each reading is a **period-ending mean** — power averaged over the preceding 30 minutes, with `time` marking the end of that window, as `PowerTimeSeries` in `packages/contracts/src/contracts/power_schemas.py` records. The irradiance ingest is chosen to match, so [Weather data](#weather-data) prefers a source that accumulates over the interval to a source that samples an instant. |
| **Curtailment (active network management)** | 🚧 | Half-hourly megawatts lost to an active-network-management (ANM) instruction, per curtailed generator. Crucial for distinguishing deliberate ANM ramp-downs from genuine faults or capacity loss. What the feed holds, and how far back it reaches, is [below](#what-the-curtailment-feed-holds). |

### What the curtailment feed holds

**Curtailment reaches us two ways, and the export cap is the one to build on.** The `curtailment/`
prefix sits beside `timeseries/` on the bucket, on the same six-hour window convention, and each
object carries a `data` list of `{startTime, endTime, value}` at half-hourly resolution under a
header naming the generator, its substation number, its licence area, and a `CurtailmentType` such
as `ANM (DANM and TANM)`. Its `value` is a derived volume of megawatts lost. Separately, NGED can
export the raw active-network-management setpoint history on request, one row per change of the
generator's export cap — the signal the derived volume is computed from. Nothing in this repository
ingests either yet.

**Prefer the setpoint history: it reaches further back and it is the physical quantity.** For the
one generator both cover, the setpoint history runs from February 2024 against the bucket feed's
late April 2026, and the two disagree on the hours they share. The cap is also what a model can use
directly, because export cannot exceed it: a generator's output is `min(what the weather allowed,
the cap in force)`. The cap is published as a negative number, generation being negative in NGED's
convention, and its largest magnitude is the connection limit rather than a curtailment. Reading
the cap as a curtailment volume gets this backwards — the measurements are in [the beam/diffuse
results](../studies/beam-diffuse-split.md#one-site-is-curtailed-and-the-export-cap-is-what-makes-its-hours-scorable).

**Ask for the whole history, and expect the first request to be truncated.** The first export we
were sent for that generator began in July 2024; a second request returned an exact superset
reaching back to February 2024, the day the generator's telemetry begins. Nothing in the first file
said it was partial.

**The setpoint feed reads zero for months before the scheme enforces anything, so find the go-live
date before trusting a single reading.** Across all 431 bright hours the export covers before
6 August 2024, the cap forbids export outright — while the generator exported above 5% of capacity
in 423 of them, at a median of 44%. An ingest that honoured those readings would label a plant
running normally as a plant held at zero, and would do it silently. Take the scheme as live from the
first half-hour at which the cap reaches the connection limit: a live scheme on an unconstrained
generator rests at that limit most of the time, and the marker needs no reference to metered output.
Eight of the 2,268 rows also carry a positive value where the rest are negative, so read the
magnitude rather than flipping the sign.

**One generator in the trial area is connected under active network management, and NGED has
confirmed there are no others.** Within the trial area a generator absent from the setpoint record
therefore ran free, which is what lets an absent cap be read as evidence rather than as a gap. That
confirmation does not extend past the trial area: at rollout the population of flexible connections
has to be established again, and until it is, an absent record means only that NGED has not sent
curtailment for that generator. The setpoint export is per generator, so widening it to a new set of
sites is a fresh request each time.

**15-minute power data may become available. We deliberately stay on half-hourly until v2.** There
is no room on the road to v1.0 to re-ingest the power feed at a finer step, so treat 15-minute data
as a v2 item.

**Power forecasts stay half-hourly regardless, so the change is confined to the ingest.** Averaging
each pair of 15-minute readings into the half-hour ending at `:00` or `:30` leaves the training
grid, the metrics, and the leaderboard exactly as they are, because two period-ending 15-minute
means average to the period-ending 30-minute mean the contract already specifies. Everything
downstream stays correct on that basis — `_upsample_nwp_to_half_hourly`, the inference spine in
`production_helpers`, and the row-wise forecast metrics all assume a fixed half-hourly step and
would keep getting one. The milestone that actually wants a finer step is v2 disaggregation, which
can read it from a separate table rather than by widening `PowerTimeSeries`.

**Do that averaging explicitly, because today's ingest would silently keep the wrong half of each
half-hour.** `PowerTimeSeries.drop_implausible_rows` filters to readings aligned to `:00` and `:30`,
logging a count and raising nothing. Fed a 15-minute series it would discard every `:15` and `:45`
reading and keep the `:00`/`:30` ones — each of which covers only the *preceding* 15 minutes. The
result is the second quarter-hour of every half-hour wearing a half-hourly label. `validate` then
passes, because the survivors are a well-formed half-hourly series, and the Dagster asset
materialises green, because the drop count only warns. Around sunrise and sunset, where irradiance
ramps inside the half-hour, that bias is systematic rather than noise. Changing `PowerTimeSeries` is
a data-contract change, so it needs sign-off under the rule in `packages/contracts/README.md`, and
the averaging rule needs to say what happens when one reading of a pair is missing.

### Provided as reference files (mostly static reference / historical)

| File | Status | Description | Known issues |
|---|---|---|---|
| Historical time-series JSON | ✅ | Historical outputs of substations and customer meters. | See [data quality](#data-quality-availability). |
| **Monitor Direction.csv** | 🚧 | Metadata for all substations: meter (analogue) type and power-flow direction. | One solar park's (ID 30) substation number differs from its metadata, to be confirmed with NGED; other sources agree, so low risk. |
| **Primary Substation Interconnections.csv** | 🚧 | List of possible connections between primary substations (not all are in the trial area). | All trial-area substations have ≥ 1 connection; topology appears complete. |
| **Substations.csv** | 🚧 | For each substation, bulk supply point (BSP) and grid supply point (GSP): which BSP & GSP it connects to (names + IDs). | All trial-area substations valid. |
| **Switching Logs.xlsx** | 🚧 | History of every normally-open switching point between primaries, labelled by time-series ID. Primaries outside the trial area are labelled "Unknown". | **Extremely valuable** as the gold-standard *test set* for [switching-event detection](switching-events.md) — lets us validate the unsupervised method on the trial area (labels do **not** exist at scale). Some edges "collapse" into `[substation ID] – unknown`. Two edges present in Interconnections.csv are missing: 900016 (ID 10) ↔ 900019 (ID 13), and 900022 (ID 16) ↔ unknown (910026). Logs go back to ≥ 2019. |
| **MPAN to Substation Number.csv** | 🚧 | Associates each Embedded Capacity Register (ECR) generator to the substation it connects to. | All trial-area generators present, each with two Meter Point Administration Numbers (MPANs, import + export). Three primaries appear with one MPAN each, to be confirmed with NGED. |
| **Peak Loads.xlsx** | 🚧 | Manually selected peak demand per trial-area substation, from 2024/25 (most recent survey). | Covers all 16 trial primaries. Of these, 12 have 2024/25 readings above the recorded peak. Even at the 99th percentile of observed power, 3 primaries (IDs 8, 13, and 25) show 2–4× the recorded peak. One other primary has a recorded peak above 14 MVA, but its telemetry has never exceeded 6.8 MVA. Given these discrepancies, we use the **99th percentile of observed power** as the substation "capacity" proxy, at least initially. |

### NGED's published electricity-network model (the Long Term Development Statement)

**NGED publishes a description of the electricity network itself, and that description does not
arrive in Flexpectation's pipeline.** The two tables above list what NGED sends. The Long Term
Development Statement (LTDS) is separate. The LTDS sits on NGED's open-data portal in two forms. The
LTDS describes the wires rather than the power flowing through them.

| Dataset | Status | Description |
|---|---|---|
| [LTDS Tabular Model](https://connecteddata.nationalgrid.co.uk/dataset/ltds-tabular-model) | 🔬 | Comma-separated tables covering circuits, nodes, two- and three-winding transformers, demand, fault levels, and generation, across NGED's four licence areas. |
| [LTDS Common Information Model](https://connecteddata.nationalgrid.co.uk/dataset/ltds-common-information-model) | 🔬 | The same electricity network in the Common Information Model format, as one archive per licence area, covering 132 kV grid supply points down to 11 kV or 6.6 kV primaries. A separate lookup table maps the model's substations to NGED substation numbers. |

**Between them, the two publications carry the per-branch impedances and thermal ratings a
power-flow model is built from.** Flexpectation holds neither anywhere else. The circuit table holds
22,516 rows. Of those, 6,145 carry positive- and zero-sequence impedances and four seasonal thermal
ratings, at 132, 66, 33, 11, and 6.6 kV. The rest are switching devices such as disconnectors, which
have no impedance.

Three planned pieces of work would draw on the two publications:

- [Tier 3 curtailment costing](cost-savings-metrics.md#tier-3-full-power-flow-modelling), which is
  out of scope today for want of exactly this model.
- [Switching-event detection](switching-events.md), which reasons about which substations can
  exchange load.
- [Capacity estimation](capacity-estimation.md), which needs a firmer limit per node than the 99th
  percentile of observed power.

**No Flexpectation code reads either dataset.** Two questions have to be answered first. Do the LTDS
node names join cleanly to the `time_series_id`s Flexpectation forecasts? The Common Information
Model's substation-number lookup is where that join would start. How far below the primary
substations does the published model reach? Neither question has been investigated.

---

## Data quality & availability

Machine-learning models are only as good as their training data. Known characteristics of NGED's
historical data (full detail + plots in the Milestone 1 report, Appendices A & B):

- Most trial-area time series go back to **late 2019** (≥ 6 years). Exceptions: Wrangle primary
  (ID 12) and ID 13 have only ~1 year.
- The first ~2 months of each series tend to be poor quality (meter ramp-up / calibration) and are
  dropped during cleaning.
- **Gaps**: a couple of missing points every few weeks, especially recently for generators. Solar
  generators legitimately don't report overnight, but not all gaps are nighttime; gaps can last
  hours to months.
- **Meter quality flags**: some meters carry NGED's own quality flags ("analogue not working" or
  "analogue suspect"), which ingestion does not yet act on.
- **False zeros**: substation telemetry has occasional drop-outs to zero, visible as an excess of
  exact zeros in the distribution vs. near-zero values.
- **Not-on assets**: one trial-area generator (ID 19) has not been operating since mid-2024 —
  motivating the [building-blocks](forecast-building-blocks.md) delivery approach.
- **MVA / reverse flow**: primary data is disaggregated from metered generation where possible, but
  not always. The primary with ID 26, for example, has two solar meters that are not reporting, so
  their generation cannot be subtracted. Combined with MVA metering (which reports absolute value),
  midday solar export "bounces" off zero and looks like extra load. See also the [MVA discussion in
  Net-demand disaggregation](disaggregation.md#apparent-power-mva-metering).

These data oddities are detected and reported back to NGED as warnings (see [delivery tables, Table
2](delivery-tables.md#table-2-power_forecast_warnings)).

---

## NGED's network (context)

As of May 2026, NGED's full network (the v2 target scope) consists of:

- **1,161 primary substations** (33/11 kV & 66/11 kV)
- **271 bulk supply points (BSPs)** (132/33 kV & 132/66 kV)
- **52 grid supply points (GSPs)** (400/132 kV & 275/132 kV)
- **~1,500 generators** (industrial customer generators, not domestic):
    - 558 connect directly to GSP/BSP busbars at 33 kV or 132 kV (modelled by NGED as
      generation-only "substations"; mostly have telemetry; curtailable via ANM; comprising 329
      solar, 63 wind, 166 other).
    - ~1,000 are on the 11 kV network downstream of primaries; some metered, some not.
    - Power flow from metered generators connected to primaries is **already subtracted** from the
      primaries' power flow ("Disaggregated Demand").

The **Embedded Capacity Register** behind these figures has limits worth naming, because both
[capacity estimation](capacity-estimation.md) and [disaggregation](disaggregation.md) build on the
register. The register records generation of 50 kW and above. The capacity recorded is the export
limit a site's connection agreement permits rather than what the site can actually generate, which
is why Flexpectation needs to estimate *effective* capacity rather than reading nameplate capacity
off the register. Below 50 kW the register is silent. Most of the panels sit below 50 kW, too: of
the 22,560 MW of solar photovoltaic capacity installed in GB by the end of July 2026, 8,503 MW sits
in arrays smaller than 50 kW. Recovering solar on that scale, with no register to check the answer
against, is what [disaggregation](disaggregation.md) has to do. The figures come from the
[energy-forecasting
review](../background/energy-forecasting-review.md#8-disaggregating-unmetered-solar-and-wind-from-a-substations-net-flow),
which cites the Department for Energy Security and Net Zero's [solar deployment
statistics](https://www.gov.uk/government/statistics/solar-photovoltaics-deployment).

The **v1 trial area** is 32 of these time series — see the [roadmap index](index.md) for the
breakdown.

---

## Weather data

Issues: [#142](https://github.com/openclimatefix/nged-substation-forecast/issues/142) (CAMS solar
radiation), [#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143)
(reanalysis ingestion — ERA5)

The table below lists the weather products the project ingests, plans to ingest, or is researching
as a possible source. The Status column says which, and several rows are marked uncertain,
deprioritised, or unlikely. Products researched only for the studies are on the [weather products
survey](../background/weather-products-survey.md#the-surveyed-products) page.

**ECMWF's own models, Google DeepMind's WeatherNext 3, and NOAA's GFS and GEFS reach NGED's 14-day
forecast horizon; every regional model on this page stops at 7.5 days or less.**

![Horizontal bar chart of sixteen NWPs' longest routine forecast horizon in days, sorted longest to
shortest. GEFS reaches 35 days on its 00 UTC cycle; GFS reaches 16 days; ECMWF ENS, AIFS-ENS, AIFS
Single, ECMWF IFS HRES, and WeatherNext 3 all reach 15 days; ICON global reaches 7.5 days; the Met
Office's global model reaches 7 days; MOGREPS-UK reaches 5.25 days; UKV and ICON-EU reach 5 days;
ARPEGE reaches 4 days; the shared DMI/KNMI HARMONIE-AROME feed reaches 2.5 days; and ICON-D2 and
AROME France both reach 2 days. A dashed line marks NGED's 14-day forecast
horizon.](assets/nwp_horizons.svg)

Each bar is the model's longest routine run, taken from this page, from the [weather products
survey's forecast-model table](../background/weather-products-survey.md#forecast-models), or from
the producer's own documentation — the
source for every bar is recorded next to its horizon in `studies/beam_diffuse_split/nwp_horizons.py`.
Reanalyses and satellite products carry no forecast horizon and are left off.

| Source | Status | Description |
|---|---|---|
| **ECMWF ENS** (Dynamical.org) | ✅ | Main NWP source: 51-member ensemble, distributed as live-updating Zarrs. OCF converts gridded NWP to tabular via the H3 spatial index and stores as Delta Lake, stored as `Float32` rounded to a 13-bit significand, with zstd compression (~40 GB/year for all of GB; ~1 minute to download+convert one day). **The archive currently only extends back to 2024-04-01**; Dynamical.org are back-filling the operational archive from MARS to 2016-03-08 (51 members, 0.25°, 00Z inits only), but at ~0.8 TB/day against ~446 TB remaining the estimate is **~November 2027** — after v1.0, which is why we [extend the training history](training-history.md) with an estimate of past weather instead. Radiation: no direct component, which is what forces [DP forecasting of PV](disaggregation.md) (v2) to find the beam/diffuse split elsewhere — see [which sources carry which irradiance components](#which-sources-carry-which-irradiance-components). |
| **ERA5** (ECMWF global reanalysis) | 🚧 (v0.5) | The reanalysis planned for ingest, serving the [weather-abnormality climatology](xgboost-improvements.md#weather-abnormality-climatology-z-score-features). For **pre-training**, the version-1 plan is ERA5 as the estimate of past weather for every variable except irradiance, and CAMS for irradiance — see [Which estimate of past weather to train on](training-history.md#which-estimate-of-past-weather-to-train-on). A [study of CEDA's UKV archive against ERA5](../studies/past-weather/ukv-ceda-vs-era5.md) found no clear advantage for CEDA's UKV in wind-farm power forecasts. ERA5 is not ingested yet. Covers 1940 to the present, so it spans every power history that predates the ENS archive (2024-04-01). Its 31 km resolution is coarser than CERRA, which is acceptable because weather anomalies are synoptic-scale and the high-resolution *solar* irradiance comes from CAMS regardless. Carries the [beam/diffuse split](#which-sources-carry-which-irradiance-components), which the live ENS feed does not. Its **ERA5T** near-real-time stream lands ~5 days behind real time, and final ERA5 overwrites it ~2–3 months later after quality control. Shares the ECMWF **IFS lineage** with the ENS forecasts, so systematic biases largely cancel when the two are combined. Ingest **2020 to present**, including the 2024+ ENS overlap, which is not optional — see [Extending the training history](training-history.md). [Which access route](#era5-which-access-route) is still open. |
| **CERRA** (Copernicus regional reanalysis for Europe) | 🔬 (deprioritised) | Higher-resolution (5.5 km) European reanalysis. Per the [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels), it now runs from **September 1984 to the present** — monthly updates, but **about 3 months behind real time**: on 2026-09-23 the latest data ended 2026-06-30, 12 weeks earlier. **Not in the active plan**, which takes its estimates of past weather from CAMS and ERA5; that latency of about 3 months would also rule CERRA out of near-real-time capacity estimation. Kept here because its 5.5 km resolution could still earn a place for fine-scale work (e.g. wind over complex terrain) if that ever proves decisive. Its [direct short-wave](#which-sources-carry-which-irradiance-components) is time-integrated from 3-hourly forecast cycles, so temporally coarser than SARAH-3. On past sunshine at six solar farms, an XGBoost model given CERRA's global irradiance is no better than one given ERA5 and 4.045 points of capacity behind one given CAMS (9.164% against 9.135% and 5.118%, on the CERRA row set; [results](../studies/past-weather/solar.md#cerra-is-no-better-than-era5-and-trails-cams-by-about-4-points)). Grid, coverage, wind heights, and access are in the [survey's reanalysis table](../background/weather-products-survey.md#reanalyses-hindcasts-and-satellite-retrievals). |
| **CEDA UKV** (the Met Office's UKV as archived by the Centre for Environmental Data Analysis) | 🔬 (not ingested in version 1) | A candidate estimate of past weather for every variable except irradiance, read from the early time steps of each archived run. The version-1 plan is ERA5 for those variables, and CEDA's UKV is not ingested. In the [CEDA UKV against ERA5 study](../studies/past-weather/ukv-ceda-vs-era5.md), pooled over three wind farms in Lincolnshire, an XGBoost model given ERA5's wind had a lower power error than an XGBoost model given CEDA's 10 m and 925 hPa wind, by 0.125 points of capacity [+0.033, +0.216], which is below the study's 0.16-point margin for a clear difference. CEDA's 1.5 m temperature was closer than ERA5's by 0.124 K [0.113, 0.135] at four weather stations that UKV probably assimilates. CEDA's archive is the first UKV source the studies have found that reaches back to 2019. The fields fetched so far hold 10 m wind and winds at 925 hPa and 1000 hPa, with no 100 m wind and no orography. CEDA's archive lacks a small number of runs and holds a few more only in part. The Met Office's PS47 upgrade on 2026-01-21 changed the regional models, so a UKV series spanning that date is not homogeneous. CEDA's UKV differs from Open-Meteo's UKV and from the Met Office's AWS bucket in runs and lead, grid, irradiance, wind units and heights, eras, and licence — see [how the three archives differ](#ukv-from-cedas-archive-differs-from-ukv-as-open-meteo-serves-it-in-lead-irradiance-temperature-and-units). The catalogue record states the CC BY-NC-SA 4.0 licence, and access is by application; archive start and grid are in the [survey's forecast-model table](../background/weather-products-survey.md#forecast-models). |
| **CM SAF** (Satellite Application Facility on Climate Monitoring) | 🔬 (v2 comparison) | SARAH-3 carries [every irradiance component](#which-sources-carry-which-irradiance-components) on a 0.05° grid at 30 minutes from 1983. Two reasons SARAH-3 is not the first ingest. Its climate data record ends 2020-12-31 and the Interim Climate Data Record extends that record, putting a version seam inside the 2019-onward history we train on. And its 30-minute values are **instantaneous snapshots**, whereas CAMS accumulates over the step, which is what a period-ending meter reading measures — under broken cloud an instantaneous sample and a 30-minute mean can differ a lot. Its gridded delivery would suit the H3 pipeline better than CAMS point requests, and comparing the two resolutions is not straightforward, because the CAMS point service interpolates to the requested location rather than publishing a grid. Latency is 2–5 days ([Pfeifroth et al. (2024)](https://doi.org/10.5194/essd-16-5243-2024)), immaterial offline. Worth a genuine head-to-head against CAMS in v2 — see [Correcting satellite irradiance over Great Britain](disaggregation.md#correcting-satellite-irradiance-over-great-britain). |
| **CAMS solar radiation** (Copernicus Atmosphere Monitoring Service) | 🚧 (v0.7) | The satellite-derived irradiance we ingest, used to estimate **solar PV** capacity, and the planned estimate of past irradiance for [pre-training](training-history.md#which-estimate-of-past-weather-to-train-on). Used **offline only** — capacity estimation and pre-training run over history, and the production serving path takes no dependency on it. The CAMS Radiation Service carries [every irradiance component](#which-sources-carry-which-irradiance-components), under both clear sky and observed cloud, from 2004-02, under CC-BY-4.0, at steps of 1 minute, 15 minutes, 1 hour, 1 day, or 1 month — the beam/diffuse split the [DP solar model](../techniques/differentiable-physics.md#the-core-building-block-differentiablesolarplant) needs. Cloud information comes from Meteosat Second Generation; aerosol, ozone, and water vapour come from the CAMS global forecasting system, so aerosol optical depth is a 3-hourly analysis rather than SARAH-3's monthly climatology. Values are interpolated to the requested location rather than served on a grid. Chosen over SARAH-3 on **delivery and record continuity, not on measured accuracy over Great Britain** — see [CAMS: use the point API, not the gridded product](#cams-use-the-point-api-not-the-gridded-product) for the route and its traps, and [Correcting satellite irradiance over Great Britain](disaggregation.md#correcting-satellite-irradiance-over-great-britain) for what is known about this source's error and what v2 might do about it. |
| **ICON-EU** (Dynamical.org) | 🔬 (v0.9, uncertain) | Possible additional NWP source to test whether it improves skill over ECMWF ENS: a deterministic run from DWD, Germany's national weather service, on a ~6.5 km grid, 4 runs a day out to 5 days. Already carries the [beam/diffuse split](#which-sources-carry-which-irradiance-components) [the PV forward model](disaggregation.md#the-forward-model) needs, and holds the earliest roadmap slot of any source that does. Starts early 2026, so it can't enter the canonical CV folds directly — assessed via ad-hoc ablation first. Archive start, latency, and access routes are in the [survey's forecast-model table](../background/weather-products-survey.md#forecast-models). |
| **AIFS-ENS** (ECMWF) | 🔬 (v2.1, uncertain) | ECMWF's machine-learned ensemble, now operational with the same 51 members and 15-day horizon as the physics ensemble, and more accurate than it on the majority of variables and lead times ([Lang et al. (2026)](https://doi.org/10.1038/s44387-026-00073-7)). Whether that translates into a better substation-load forecast is an open question. AIFS-ENS member *n* starts from the [same initial conditions](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v1) as ECMWF ENS member *n*, so the two can be fed [side by side](xgboost-improvements.md#several-nwp-sources-as-features-v21) rather than swapped. Unlike the physics ensemble, AIFS has [no direct-beam field](#which-sources-carry-which-irradiance-components) to ask for at all, in either the ensemble or the deterministic AIFS Single: on the 2026-09-19 00Z run, AIFS-ENS open data carries 29 parameters and AIFS Single 30, and `ssrd` and `strd` are the only radiation fields in either. **Both AIFS streams arrive on 6-hourly steps across the whole 15-day horizon — 61 steps, against the physics ensemble's 85** — where the physics ensemble is [3-hourly out to 144 hours](../architecture/nwp-variable-conventions.md#the-forecast-step-grid) and only then drops to 6-hourly. Twice the step width over days 0 to 6 costs most on irradiance, whose diurnal cycle varies faster than any other field we use: the [reconstructed solar day](../architecture/nwp-variable-conventions.md#period-ending-variables-are-interpolated-as-though-they-were-instantaneous) lags the true one by half a step width, so 3 hours at AIFS's steps against 1.5 hours at the physics ensemble's, and the modelled clear-sky peak falls from 816 W m⁻² to 756 at 3-hourly steps and 590 at 6-hourly. Same folds problem as ICON-EU: the archive starts mid-2025, so it is an ad-hoc ablation before it is a canonical source. Archive start and access are in the [survey's forecast-model table](../background/weather-products-survey.md#forecast-models). The evidence on whether AIFS improves faster than the physics-based IFS, and the reason each AIFS version is scored separately, are in the survey ([evidence](../background/weather-products-survey.md#aifs-has-not-been-shown-to-improve-faster-than-the-physics-based-ifs), [version eras](../background/weather-products-survey.md#score-each-aifs-version-separately)). |
| **WeatherNext 3** (Google DeepMind) | 🔬 (after v2, unlikely for production; ensemble-mean fetched for a study) | Google DeepMind's machine-learned 64-member ensemble (grid, cycles, and horizons are in the [survey's forecast-model table](../background/weather-products-survey.md#forecast-models)), following [Google's guide to WeatherNext 3 on Cloud Storage](https://developers.google.com/weathernext/guides/gcs). 2 m temperature and dewpoint also arrive at 0.05°, from a neural-network output head trained directly on in-situ surface observations from airport stations, regional station networks, ships, and buoys ([Rasp et al. (2026)](https://arxiv.org/abs/2609.03582); [model specs](https://developers.google.com/weathernext/guides/models)). Carries the [beam/diffuse split](#which-sources-carry-which-irradiance-components) [the PV forward model](disaggregation.md#the-forward-model) needs, as named fields rather than by subtraction. The paper reports a lower [continuous ranked probability score](../techniques/evaluation-metrics.md#crps-continuous-ranked-probability-score) than ECMWF ENS on surface solar radiation, scored on the 6-hour accumulations rather than the hourly fields. Every radiation evaluation in the paper scores against an ECMWF analysis, and none scores against a surface measurement, so any irradiance gain would have to be measured downstream on power. Two structural differences from AIFS-ENS: all 64 members start from one analysis, so the ensemble's spread comes from the model alone rather than from perturbed initial conditions, and no member pairs with an ECMWF ENS member. Access needs [a request form](https://developers.google.com/weathernext/guides/access-forecast), and the survey gives the licence terms. BigQuery and Earth Engine serve six summary statistics per variable (the mean, and the 10th, 25th, 50th, 75th, and 90th percentiles), matching a third route: a Cloud Storage Zarr store of precomputed statistics, distinct from the 64-member store the ensemble itself sits in. The [matched-lead study](../studies/forecasts/matched-lead.md) fetches the ensemble mean from that statistics store, cropped to a wide United Kingdom box and held in OCF's own Icechunk store (`ocf-weathernext3-uk`) — see [the survey's findings](../background/weather-products-survey.md#what-we-learnt-about-weathernexts-precomputed-statistics-store) for the route, the cost, and how to open the result. That fetch is a study input, not production ingestion: the folds problem remains worse than ICON-EU's or AIFS-ENS's, since the archive starts 2026-01-01 with no back-fill before it, so on current priorities we do not expect to reach this source for production inside the Network Innovation Allowance project. The survey lists the other AI weather models and commercial archives of AI forecasts under [AI weather models beyond AIFS](../background/weather-products-survey.md#ai-weather-models-beyond-aifs). |
| **UKV and MOGREPS-UK** (Met Office, via AWS) | 🔬 (uncertain) | The Met Office's 2 km deterministic UK model and its 2.2 km UK ensemble. Both publish [every irradiance component](#which-sources-carry-which-irradiance-components) as its own field, so neither needs a subtraction to give [the PV forward model](disaggregation.md#the-forward-model) its beam/diffuse split, and both are free on AWS under British Crown copyright and CC BY-SA 4.0 ([UKV](https://registry.opendata.aws/met-office-uk-deterministic/), [MOGREPS-UK](https://registry.opendata.aws/met-office-uk-ensemble/)). Verified by listing the `met-office-atmospheric-model-data` and `met-office-uk-ensemble-model-data` buckets rather than from the documentation. Neither the horizons nor the archives suit the canonical folds — MOGREPS-UK is held as a 30-day rolling window — see [which feed carries a direct beam](#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost). How Open-Meteo archives UKV is in the [survey](../background/weather-products-survey.md#which-archives-keep-whole-past-forecast-runs), and how the bucket's UKV differs from CEDA's and Open-Meteo's is in [the comparison of the three archives](#the-met-offices-aws-bucket-keeps-every-hourly-run-on-a-different-grid). |

**ERA6 is a future upgrade, not a current option.** ECMWF began ERA6 production in March 2026, but
the phased release runs from late 2027 (first 20 years) into 2028, so it is out of scope for the
near-term milestones. When it lands it should drop in cleanly — same IFS family, a similar
ERA5T-style near-real-time fast track — and at ~14 km (2× finer than ERA5) it would close most of
the resolution gap with CERRA that motivates keeping CERRA on the list at all.

### Which sources carry which irradiance components

**The Met Office's UKV and MOGREPS-UK are the only forecast sources listed below that publish
global, direct, and diffuse short-wave as three separate fields in a free feed.** Every other free
forecast source makes the reader recover at least one component by arithmetic. ECMWF's own direct
beam, `fdir`, is outside ECMWF's free open-data subset for both the single 9 km forecast (formerly
HRES; see the [weather products
survey](../background/weather-products-survey.md#terms-used-on-this-page)) and the ENS ensemble;
[what taking `fdir` from ECMWF directly
involves](#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost) is below. Both
reanalyses and both satellite products carry a direct component, where the free feed behind the ENS
forecast in production does not. That is why [capacity
estimation](capacity-estimation.md#irradiance-inputs) plans on having the beam/diffuse split, while
[forecasting](disaggregation.md#the-forward-model) has no route to the split through the feed it
uses today.

**Four rows below, and AIFS Single in the AIFS row, are not in the catalogue above**, because the
project plans to ingest none of them: ECMWF's own ENS is the full-catalogue delivery route behind
the free feed, the two ECMWF rows via Open-Meteo were researched only for the studies and are on the
[weather products survey](../background/weather-products-survey.md) page, AIFS Single has no
catalogue row, and the Met Office's global 10 km model is the Met Office model Dynamical.org have on
their tracker rather than one we have asked for.

| Source | Kind | Global | Direct | Diffuse | Where it comes from |
|---|---|---|---|---|---|
| **ECMWF ENS** via Dynamical.org | Forecast | ✅ `ssrd` | ❌ | ❌ | Free ECMWF open data on AWS |
| **ECMWF ENS** from ECMWF | Forecast | ✅ `ssrd` | ✅ `fdir` | By subtraction | ECMWF dissemination or MARS, not the free subset ([licence and delivery below](#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost)) |
| **ECMWF ENS 9 km Europe** via Open-Meteo | Forecast | ✅ `ssrd` | ✅ `fdir` | By subtraction | Open-Meteo's Ensemble API, live runs only ([survey](../background/weather-products-survey.md#forecast-models)) |
| **ECMWF IFS HRES 9 km** via Open-Meteo | Forecast | ✅ `ssrd` | ✅ `fdir` | By subtraction | Open-Meteo, from 2017 ([survey](../background/weather-products-survey.md#forecast-models)) |
| **ECMWF AIFS**, both Single and ENS | Forecast | ✅ `ssrd` | ❌ | ❌ | Free ECMWF open data; no direct field exists in any AIFS feed |
| **ICON-EU** via Dynamical.org | Forecast | By addition | ✅ | ✅ | Free, already ingested by Dynamical.org |
| **UKV** (Met Office) | Forecast | ✅ | ✅ | ✅ | Free on AWS, CC BY-SA 4.0; all three components in every run the bucket's rolling two-year window still holds |
| **MOGREPS-UK** (Met Office) | Forecast | ✅ | ✅ | ✅ | Free on AWS, CC BY-SA 4.0; all three components, plus net short-wave; **30-day rolling archive** |
| **Global 10 km** (Met Office) | Forecast | ✅ | ✅ | By subtraction | Free on AWS, CC BY-SA 4.0; 168-hour horizon, but no global field before the 2024-11-07 12 UTC run and none at T+168 before 2026-01-21 |
| **WeatherNext 3** (Google DeepMind) | Forecast | ✅ | ✅ `fdir` | By subtraction | Access request; CC-BY-4.0 once at least 1 hour old |
| **ERA5** | Reanalysis | ✅ `ssrd` | ✅ `fdir` | By subtraction | Free from the Copernicus Climate Data Store |
| **CERRA** | Reanalysis | ✅ | ✅ | By subtraction | Free from the Copernicus Climate Data Store |
| **CAMS Radiation Service** | Satellite | ✅ | ✅ | ✅ | Free, CC-BY-4.0; also direct normal, and every component under clear sky |
| **CM SAF SARAH-3** | Satellite | ✅ SIS | ✅ SID | By subtraction | Free; also direct normal |

**"By subtraction" means the diffuse component is global minus direct, and "by addition" means
global is direct plus diffuse.** Both are exact inside the model, and both inherit the errors of the
two components they combine. For ECMWF the subtraction lands on a quantity close to, but not
identical with, what a shadow-band diffuse pyranometer measures, for the reasons under [two traps
for whoever builds on ECMWF's direct beam](#two-traps-for-whoever-builds-on-ecmwfs-direct-beam). A
source publishing all three components as their own fields, as UKV, MOGREPS-UK, and the Radiation
Service do, needs no arithmetic at all.

### Which feed carries a direct beam, and what asking for one would cost

**ECMWF's own ENS carries the direct beam. The free open-data subset does not, and the free subset
is what Dynamical.org ingests.** ECMWF's [ENS
catalogue](https://www.ecmwf.int/en/forecasts/datasets/set-iii) lists total-sky direct solar
radiation at the surface (`fdir`, paramId 228021) alongside global short-wave (`ssrd`, paramId 169),
on the same steps: hourly to T+90, 3-hourly to T+144, and 6-hourly to T+360. Reading the GRIB
(gridded binary) index of any single step of an open-data ENS run gives 47 parameters, with `ssrd`
among them and `fdir` not. Across all steps the union is 50 parameters, and `fdir` appears in no
step. Dynamical.org's [ECMWF IFS ENS
dataset](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) names ECMWF Open
Data on the AWS Open Data Registry as its source. So the variable Dynamical.org would have to add is
not in the feed they read. The single 9 km forecast is in the same position: the free subset's
`oper` index for the 2026-09-22 00 UTC run carries `ssrd` and not `fdir`. Open-Meteo serves that
forecast's direct beam, and the 51-member ensemble's direct beam over Europe, only because
Open-Meteo's
[downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/EcmwfEcpds/EcmwfEcpdsVariable.swift)
fetches `fdir` from ECMWF's dissemination system.

**ENS `fdir` reaches us through Dynamical.org by no route open today, and through Open-Meteo only
live.** Open-Meteo's [Ensemble API](https://open-meteo.com/en/docs/ensemble-api) serves the
51-member ensemble at 9 km over Europe with `fdir`, from the 00 and 06 UTC runs, but keeps no
history to train or backtest on
([survey](../background/weather-products-survey.md#forecast-models)). Commercial use needs a paid
Open-Meteo plan.

**Taking `fdir` from ECMWF directly is a delivery contract with ECMWF, not a storage decision.** The
route is ECMWF's dissemination or a MARS subscription in place of, or alongside, the free bucket.
Since 1 October 2025 ECMWF's whole Real-time Catalogue, `fdir` included, has been licensed CC BY 4.0
([ECMWF](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-makes-its-entire-real-time-catalogue-open-all)).
Delivery of the full catalogue "may involve service charges". Dynamical.org's live ENS store reads
ECMWF's free subset, although their back-fill of past ENS runs comes from MARS. The project has not
asked Dynamical.org whether that back-fill could carry `fdir`, or whether Dynamical.org would take
on a delivery feed for live runs. Dynamical.org already publish direct and diffuse short-wave for
ICON-EU, which shows that a missing variable was never the obstacle — the obstacle is which ECMWF
feed the open bucket holds, and which feed the bucket holds is ECMWF's decision rather than
Dynamical.org's.

**The Met Office models are the request worth making, because their free feed already carries a
direct beam.** The one open Met Office request on Dynamical.org's issue tracker is for [the global
10 km deterministic model](https://github.com/dynamical-org/reformatters/issues/646), which
publishes
global and direct short-wave and leaves diffuse to the same subtraction ECMWF would need. UKV and
MOGREPS-UK go further and publish diffuse as its own field, but neither appears on Dynamical.org's
tracker, so either would have to be asked for. None of the three Met Office models needs a
delivery feed that may carry a service charge, which is what sets them apart from `fdir` on ENS.

**Every Met Office model brings a horizon problem, and two bring an archive problem.** The global
model reaches 168 hours, UKV reaches 120 hours on its 03 and 15 UTC runs and 54 hours on the rest,
and MOGREPS-UK reaches 126 hours. No Met Office model covers NGED's 14-day horizon. A Met Office
model would therefore sit alongside ECMWF ENS rather than replace the ECMWF feed, exactly as
ICON-EU would. MOGREPS-UK is held on AWS as a 30-day rolling window, which rules out backtesting
unless we archive the feed ourselves from the day we start. [What we learnt about
MOGREPS-UK](#what-we-learnt-about-mogreps-uk-on-2026-09-26) gives the size of that archive and the
one other source of history we found.

**A backtest on the global 10 km model starts at its 2024-11-07 12 UTC run, not at the start of its
archive, because no earlier run carries global short-wave.** Listing every 6-hourly run the bucket
held on 2026-09-19, back to its earliest at 2024-09-19 18 UTC, shows
`radiation_flux_in_shortwave_direct_downward_at_surface` throughout and
`radiation_flux_in_shortwave_total_downward_at_surface` only from 2024-11-07 12 UTC onwards. That
run carries 4,470 files against the 4,310 of the 00 UTC run the same day, the extra 160 being global
short-wave at 78 lead times and `cloud_amount_on_height_levels` at 82. Compare runs at the same
hour: the 06 and 18 UTC runs stop at 54 hours and carry about 2,842 files whatever their date.

**Global short-wave then stayed one lead time short of the direct field for another 14 months, and
the missing lead time is T+168 — the one a 7-day horizon exists for.** Runs settle on 2024-11-13
at 88 lead times against the direct field's 89, and every 00 and 12 UTC run from then until
2026-01-21 00 UTC drops T+168. The first run carrying global short-wave at all 89 lead times is
2026-01-21 12 UTC. A backtest that needs the full 168-hour horizon therefore starts there, 14 months
after the field first appears.

**UKV and MOGREPS-UK need no cut-off of either kind.** Both carry global, direct, and diffuse
short-wave at the same lead times as each model's other hourly fields — 55 for a UKV run reaching 54
hours, 126 for a MOGREPS-UK run — with no component lagging another. That holds in the earliest UKV
run on AWS and in the earliest complete MOGREPS-UK run, the oldest surviving run of a 30-day rolling
window being part-deleted rather than whole.

### What we learnt about MOGREPS-UK on 2026-09-26

**Each MOGREPS-UK object on AWS expires 30 days after it is written, so the oldest runs are already
partial.** The anonymous bucket `met-office-uk-ensemble-model-data` holds 24 runs a day, one every
hour, under prefixes of the form `uk-ensemble/YYYY/MM/DD/THHMMZ/`. Each run reaches 126 hours and
has 3 members, and the realisation IDs of those members differ from run to run. One file holds a
single variable at a single lead time for all 3 members. An uncropped run has 14,330 such files and
about 258 GB in total, of which the archive's chosen fields are 3,744 files. Objects expire one by
one (the `x-amz-expiration` header gives each object's date), which is why the oldest runs are
already partial.

**Shortwave has 126 hourly steps and looks instantaneous.** Shortwave has no step 0, no
`cell_methods`, and no time bounds, and its `time` equals the valid time. Shortwave therefore looks
like an instantaneous value, unlike DWD's shortwave, which averages since the initialisation time.
Cloud and height-level fields have 127 hourly steps, and screen-level temperature and 10 m wind have
15-minute steps to 11.75 hours and hourly steps after that (163 steps). The 100 m wind field is on
33 height levels. The grid is 970 by 1042 points in a Lambert azimuthal equal-area projection.

**Recording one full run cropped to Great Britain takes 29 minutes and stores 1.06 GB.** The run
took 5.5 GB of downloads and 54,282 requests with 8 threads. At one run an hour, the archive grows
by about 25 GB a day and 9 TB a year. About 40,000 of the requests fetch the 100 m wind files. Those
files gain nothing from extra threads, because the `h5py` library holds a global lock, whereas 4
processes gave about 4 times the throughput.

**Open-Meteo holds the individual members for about 3.5 days and the ensemble mean and spread for
about 93 days.** We probed one Great Britain point on 2026-09-26. The [ensemble
API](https://open-meteo.com/en/docs/ensemble-api) serves the 3 members as `ukmo_uk_ensemble_2km`,
hourly to 126 hours. Runs 1 to 3 days back were complete, 4 days back were half complete, and 5 or
more days back were empty. The historical-forecast and previous-runs APIs accept the model but
return only nulls, so Open-Meteo keeps no per-run history. The [ensemble mean
API](https://open-meteo.com/en/docs/ensemble-mean-api) serves the mean as
`ukmo_uk_ensemble_mean_2km`, with spread as variables carrying a `_spread` suffix (a `_mean` suffix
is an error). The mean series starts on 2026-06-25, and we could not tell whether that start is a
rolling limit, because the API rejects earlier dates. The series is stitched from successive runs
rather than held per run, and its `previous_dayN` variables are null. The mean carries
`temperature_2m`, `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`, `wind_speed_10m`,
and `cloud_cover`, and `wind_speed_100m` is all null. The spread is null for `diffuse_radiation` and
`wind_speed_100m`. The DWD ensemble mean `dwd_icon_d2_eps_ensemble_mean` carries 100 m wind and
`dwd_icon_eu_eps_ensemble_mean` does not, and we did not probe how far back either goes. The
deterministic UKV has stitched history on Open-Meteo from about 2022 to 2023, and its
`previous_day1` variables start only in about 2025.

**The only per-run, per-member MOGREPS-UK history we found is the archive that a recorder builds,
tracked in issue #926 and described in [the ensemble weather
archive](../architecture/ensemble-archive.md).** The 30-day window on AWS adds whatever runs still
survive. A study that wants a MOGREPS-UK ensemble mean for the last 3 months can use Open-Meteo's
stitched mean, but cannot recompute that mean from members.

**A MOGREPS-UK run holds 3 members, and the Met Office's 18-member MOGREPS-UK ensemble is six hourly
runs lagged together.** [Porson et al. (2020)](https://doi.org/10.1002/qj.3844) describe it as "an
18-member ensemble ... created by running three members every hour and time-lagging these over a 6
hr window". The global MOGREPS-G has 18 members in each run, so its members all share one run age,
at 20 km resolution and 6-hourly updates against MOGREPS-UK's 2.2 km and hourly updates. Members
from different lagged runs differ in age by up to 5 hours, so they are not exchangeable the way the
members of one run are. A mean or spread built from lagged runs therefore needs three things. The
build has to decide how to weight members of different ages. It has to keep each member's run age
(the run's initialisation time) beside the member. It must never treat the 3 members of a single run
as a full ensemble, because a mean of 3 members is noisy and a spread from 3 members rests on 2
degrees of freedom. The [ensemble-means
study](../studies/forecasts/ensemble-means.md#what-open-meteos-source-code-does-for-the-mogreps-uk-mean)
records what Open-Meteo's source code does.

### What we learnt about MOGREPS-Global on 2026-09-27

**The anonymous bucket `met-office-global-ensemble-model-data` carries only the full global grid,
not a pre-cropped UK subset.** Every file under its `global-ensemble/YYYY/MM/DD/THHMMZ/` prefixes
covers the whole world on a 960-by-1280-point regular latitude-longitude grid at 20 km spacing
(confirmed from the files' own `mosg__grid_domain=global` attribute), so an archive built from this
bucket crops to Great Britain on fetch, the same way it already crops the other archived products
from their native grids.

**A run has 18 members, one file per variable per lead time holding every member, and reaches 246
hours.** The bucket runs 4 times a day at 00:00, 06:00, 12:00 and 18:00 UTC. Most surface fields —
including all four shortwave fields and 10 m wind — publish hourly to 132 hours and then 3-hourly to
246 hours (171 steps). The two height-level wind fields publish hourly only to 54 hours before
switching to 3-hourly (119 steps), a shorter hourly window than the other fields. Each object
expires 30 days after it is written, the same retention as MOGREPS-UK.

**The bucket carries four shortwave fields and both 10 m and 100 m wind, one field further than
MOGREPS-UK's three shortwave components.** `radiation_flux_in_shortwave_total_downward_at_surface`,
`..._direct_downward_at_surface`, and `..._diffuse_downward_at_surface` match MOGREPS-UK's three;
`radiation_flux_in_shortwave_net_at_surface` (its file name carries no `_downward` segment) has no
MOGREPS-UK equivalent. `wind_speed_on_height_levels` and `wind_direction_on_height_levels` list 32
height levels including exactly 100 m, alongside `wind_speed_at_10m` and `wind_direction_at_10m`.

**The deprecated `registry.opendata.aws/mogreps/` listing is still readable, but its coverage ends
nearly three years before NGED's power data begins, so it does not extend either archive
backwards.** The listing points to two further anonymous buckets, `s3://mogreps-uk` and
`s3://mogreps-g`, both still readable by anonymous S3 request despite the registry page marking the
listing itself as deprecated. `s3://mogreps-uk` holds runs from 2013-01-01 to 2016-12-31;
`s3://mogreps-g` holds runs from 2016-01-01 to 2016-12-31 (files at the bucket root, one per member
per lead time, named `prods_op_mogreps-{uk,g}_YYYYMMDD_HH_00_LLL.nc`). NGED's power telemetry starts
in late 2019, and neither current rolling bucket keeps more than about 30 days, so the gap between
2017-01-01 and roughly 30 days ago is covered by neither the deprecated buckets nor the live ones —
it would need a different source, such as the Met Office directly or CEDA, unverified. The
deprecated buckets also carry a different licence, "Non-Commercial Government Licence" for research
use, rather than the live MOGREPS-UK and MOGREPS-Global buckets' CC BY-SA 4.0, worth checking before
any use beyond research.

### What we learnt about Met Office IMPROVER on 2026-09-26

**The Met Office publishes its IMPROVER post-processing system's output on AWS as the Blended
Probabilistic Forecast (BPF), and the UK gridded buckets carry no irradiance.** The BPF is in the
AWS Open Data programme in `eu-west-2`, as NetCDF under CC BY-SA 4.0, with a
[percentiles bucket](https://registry.opendata.aws/met-office-bpf-uk-gridded-percentiles/) and a
[probabilities bucket](https://registry.opendata.aws/met-office-bpf-uk-gridded-probabilities/) on a
2 km UK grid. Lead times are hourly to 120 hours and then 3-hourly to 186 hours. The registry says
the percentiles are updated "4 times each day", while the listing for 2026-09-15 held 94 blend-time
prefixes, about one every 15 minutes. Each object expires after 30 days. In the percentiles bucket,
the 44 variables listed by file name on 2026-09-15 include total cloud amount, low cloud amount, and
the UV index, and no downward shortwave, direct, diffuse, or sunshine-duration field. We inferred
the probabilities bucket's variables from the percentiles bucket's, and did not inspect the spot or
global buckets.

**The UK gridded buckets carry wind at 10 m only, so it cannot serve either purpose that the
weather-product studies score.** The wind fields are `wind_speed_at_10m`, `wind_direction_at_10m`,
and the 1-hour and 3-hour maximum gust at 10 m. No 100 m wind is listed in the percentiles bucket,
and the same caveat about the other buckets applies. Cloud cover is a weak substitute for the direct
and diffuse shortwave that [the PV forward model](disaggregation.md#the-forward-model) needs.

**The BPF is a calibrated blend, so it cannot stand in for MOGREPS-UK members in the ensemble-means
comparison.** [Roberts et al. (2023)](https://doi.org/10.1175/BAMS-D-21-0273.1) describe IMPROVER
as blending probabilities rather than physical values, with model weights that represent each
model's relative skill for precipitation, temperature, wind speed and direction, cloud cover, and
visibility, and with a radar nowcast added for precipitation. The paper names no irradiance
variable. The variable list and the file names hold percentiles and probabilities and no member
index, so the UK gridded buckets have no raw members from which to compute a mean.

**Recording the BPF whole would take 719 to 807 GB a day per bucket.** One day (2026-09-15)
held about 807 GB in the percentiles bucket and 719 GB in the probabilities bucket, for the whole
grid. A useful field set (cloud, screen temperature, 10 m wind and gust, precipitation rate,
visibility, humidity, and pressure) was 279 GB in the percentiles bucket. We estimate, from the
bounding box, that Great Britain is about 22% of the grid, which gives about 60 GB a day cropped to
Great Britain for that field set. Keeping one blend every 6 hours would be about 1 TB a year, also an
estimate. The bucket holds 30 days, so a longer archive cannot be back-filled. In the searches
behind [issue #801](https://github.com/openclimatefix/nged-substation-forecast/issues/801), we
found no archive of MOGREPS-UK, and we did not verify whether one exists for the BPF.

**The research recommends against archiving the BPF for solar and wind.** A small forward recorder is
worth considering if a study wants the operational blend as a benchmark. See #801 for where such a
recorder would be tracked.

**The IMPROVER paper states that equal weights were used across time-lagged runs at the time of
writing (2023), although the open-source code has options to weight runs by age.** Roberts et al.
say time-lagging is essential for MOGREPS-UK, which was designed as a time-lagged ensemble of
several runs of 3 members each hour, and that IMPROVER also time-lags UKV and MOGREPS-G. They say
options exist for applying different weights to each forecast length, "although at present, equal
weighting is used". The paper was published in March 2023, and we have not checked whether the
operational weights have changed since. In the code at commit
[`acf4ab6`](https://github.com/metoppv/improver/tree/acf4ab6d08cff2e52359bcc62e8bfe0788958218/improver/blending),
the weights can come from four sources. A dictionary gives piecewise-linear weights along a
coordinate such as lead time. A default linear rule runs between a start value and an end value. A
non-linear rule sets the weight of the i-th cycle to `cval**i` and can order the cycles so that the
newest is heaviest. A triangular rule is the fourth. The utility that merges lagged runs into one
ensemble only pools the members, without weighting them by age, and the repository's own
cycle-blending command line uses equal weights. The ensemble-means question of how to weight
members of different ages therefore has a published operational answer of equal weights, as of 2023.

### ECMWF has published no plan to open a direct beam or hourly ensemble steps

**Asked in July 2025 whether the open-data parameter list would grow, ECMWF said it would not grow
during the transition to open data.** ECMWF staff
[replied](https://forum.ecmwf.int/t/what-does-fully-open-data-status-in-2025-mean/12777) that they
were "not planning to expand the existing parameter dataset while we make the transition", and would
"review the parameters themselves and potentially expand in the coming years" — naming no parameter
and no date. Searching ECMWF's [open-data catalogue
page](https://www.ecmwf.int/en/forecasts/datasets/open-data), its news archive, its user forum, and
the `ecmwf/ecmwf-opendata` client repository turned up `fdir`, `cdir`, and `dsrp` nowhere as a
planned addition.

**The open-data parameter list is not frozen, though.** ECMWF
[announced](https://forum.ecmwf.int/t/open-data-transition-early-success/14478) in December 2025
that total cloud cover, snow depth, and snowfall had been added to the IFS open dataset, and all
three are in the feed today. A request for a radiation parameter is therefore a request of a kind
ECMWF granted recently, rather than one the feed's design rules out. Asking is invited, too: the
same July 2025 reply says ECMWF are "happy to take suggestions and we keep a list of requests for
the open data", and when a user asked for `fdir` among the AIFS outputs in June 2025, ECMWF
[replied](https://forum.ecmwf.int/t/aifs-expanded-products/13606) that they would pass the request
to the AIFS developers — a request logged, not a plan.

**The free feed's ensemble steps are 3-hourly to 144 hours, and the one dated commitment to widen
the free feed is about grid spacing rather than steps.** ECMWF's announcement of 1 October 2025,
[opening the whole Real-time
Catalogue](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-makes-its-entire-real-time-catalogue-open-all),
says the free subset is published "at 25 km resolution" and that "later in 2026, the free and open
subset will be extended to include our 9 km resolution forecasts with a 2-hour latency". The hourly
steps the full catalogue carries out to 90 hours appear neither there nor anywhere else we
searched. Hourly steps would be worth having for solar, because the coarser the step, the flatter
and later the reconstructed solar day.

**Removing the licence fee brought neither the direct beam nor a finer step to the free feed.** The
October 2025 change made ECMWF's whole Real-time Catalogue open, and the free feed has gained
parameters since, but the open-data catalogue page, updated for IFS Cycle 50r1 on 13 May 2026, still
lists five radiation parameters — `ssrd`, `strd`, `ssr`, `str`, and `ttr` — and the same 3-hourly
step floor. Listing the index files of the 2026-09-19 00Z run agrees: 47
parameters at any one ensemble step, those same five radiation fields, and no step finer than
3 hours.

### Two traps for whoever builds on ECMWF's direct beam

**`dsrp` is the more convenient field to ask ECMWF for, because `fdir` is a horizontal-plane flux
and
`dsrp` is already direct normal irradiance.** The ENS catalogue linked above carries three direct
fields: `fdir` (paramId 228021, total-sky direct at the surface), `cdir` (paramId 228022, the
clear-sky equivalent), and `dsrp` (paramId 47, direct solar radiation into a plane facing the sun).
Deriving direct normal irradiance from `fdir` means dividing by the cosine of the solar zenith
angle, which is numerically unstable at low sun, and the sun is low over GB for much of the year.
`dsrp` needs no division, so a request naming only `fdir` leaves behind the field that needs none.

**`ssrd` − `fdir` is close to a diffuse pyranometer reading, but not equal to that reading.**
[ECMWF's
radiation
note](https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf)
records that the model treats strongly forward-scattered radiation as unscattered, and that its
diffuse part includes the circumsolar radiation a shadow-band pyranometer excludes. The note puts
the correction on the measurement rather than on the model output: a diffuse measurement has to be
corrected for the shadow band before it is compared with `ssrd` − `fdir`. The note also warns
that these fluxes are accumulated over a period across which the solar zenith angle moves.
Converting an accumulated flux to an instantaneous pyranometer convention is therefore not
straightforward either.

### ERA5: which access route

**The leading candidate is the CDS's own analysis-ready Zarr store, but two open questions need
answering before we commit.** Since
[2026-06-30](https://forum.ecmwf.int/t/access-to-our-arco-era5-data-lake-zarr-store/15123)
Copernicus publish ERA5 single levels (and ERA5-Land) as analysis-ready Zarr, opened straight from
`xarray` with a CDS API key and offered **geo-chunked** — long periods over a small area, which is
our access pattern — as well as time-chunked. Neither of the two open questions is documented:
whether its **subset** of surface and wave variables carries every variable the climatology and gap
filling need (further parameters are added on request), and whether it serves **ERA5T** or only
final ERA5. The Zarr store is also explicitly a **beta** service, and the CDS
reserve the right to withdraw access to protect Data Store performance.

**Three fallbacks, in order.** [ARCO-ERA5](https://github.com/google-research/arco-era5) is a public
Google Cloud Zarr carrying ERA5T at about a week's lag — verify freshness via its
`valid_time_stop_era5t` / `last_updated` metadata. [Earthmover
Icechunk-ERA5](https://registry.opendata.aws/earthmover-era5/) is on AWS and updates daily, but is
paid; its free tier lags 3 months. The plain **CDS API** reaches about 5 days behind real time but
is not analysis-ready. Separately, a precomputed *mean* climatology for the [weather-abnormality
feature](xgboost-improvements.md#weather-abnormality-climatology-z-score-features) is available from
WeatherBench2 at `gs://weatherbench2/datasets/era5-hourly-climatology/`.

### Why ERA5 describes past sunshine and wind worse than most current weather products

**ERA5 has four documented weaknesses: a weather model and assimilation system frozen in 2016, a 31
km grid, hourly sunshine that is a forecast rather than an analysis, and aerosol that is prescribed
rather than observed.** Neither the solar study nor the wind study measured how much each weakness
contributes to ERA5's error at the study farms. The weaknesses come from ECMWF's own descriptions of
ERA5 and from published validations.

**ERA5 runs a weather model frozen in 2016, 10 releases older than the one ERA6 is built on.** A
reanalysis re-runs one weather model over past decades and corrects each hour towards the
observations recorded then; that correcting step is data assimilation. ECMWF numbers each release of
its Integrated Forecasting System (IFS) as a cycle, and ERA5 is produced with cycle 41r2, the
version ECMWF used for its operational forecasts in 2016 ([ERA5
documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)). The freeze is
deliberate: a reanalysis is built to be consistent over decades, so ERA5 applies the same model
physics, grid, and assimilation to 1940 as to last week. The operational IFS then moved through 10
cycles to cycle 49r2, the cycle ERA6 runs on ([Copernicus
announcement](https://climate.copernicus.eu/copernicus-climates-era6-reanalysis-production-starts)).

**ERA5's 2016 assimilation system is unlikely to place cloud from satellite images, which CAMS and
UKV both use.** ERA5 keeps absorbing current observations, about 24 million a day by the end of 2018
([ECMWF Newsletter
159](https://www.ecmwf.int/en/newsletter/159/meteorology/global-reanalysis-goodbye-era-interim-hello-era5)),
but only as far as the 2016 assimilation system can use them. In 2019 ECMWF reported that infrared
radiances were not yet assimilated operationally in all-sky conditions, meaning with cloudy and
clear scenes treated alike ([ECMWF Newsletter
161](https://www.ecmwf.int/en/newsletter/161/meteorology/recent-progress-all-sky-radiance-assimilation)).
That newsletter describes instruments on polar-orbiting satellites, so the step from its statement
to geostationary imagery is this page's inference, not ECMWF's. CAMS reads cloud from Meteosat
images of the hour itself, and UKV assimilates satellite-derived cloud (see [UKV assimilates
satellite cloud](#ukv-assimilates-satellite-cloud-and-carries-a-fixed-aerosol-climatology)).

**ERA5's 31 km grid holds cloud only as a fraction of each grid cell.** A cloud smaller than a 31 km
cell has no position inside the cell, so ERA5 cannot say which farm in the cell the cloud shades.
Validating daily irradiance at 41 Baseline Surface Radiation Network stations worldwide and 294
stations in Europe, [Urraca et al. (2018)](https://doi.org/10.1016/j.solener.2018.02.059) found that
ERA5 overestimates irradiance under cloud and slightly underestimates it under clear skies. The
authors attribute that pattern to ERA5's poor handling of cloud, and judge ERA5's 31 km grid too
coarse for places where surface irradiance varies strongly, such as coasts and mountains.

**ERA5's hourly sunshine is a forecast up to 12 hours old.** ERA5's hourly surface radiation is not
an analysis: ERA5 takes it from short forecasts started at 06 and 18 UTC, at steps of 1 to 12 hours
([ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)). ERA5's
hourly wind is an analysis, so the forecast age applies to sunshine and not to wind.

**ERA5's aerosol is prescribed, never taken from observations of that day's haze.** Aerosol matters
for sunshine because aerosol sets how much sunlight reaches the ground and how that sunlight divides
between the direct beam and the diffuse sky. Cycle 41r2 reads tropospheric aerosol from the monthly
climatology of [Tegen et al. (1997)](https://doi.org/10.1029/97JD01864) ([IFS documentation, cycle
41r2, part IV](https://doi.org/10.21957/tr5rv27xu)). ERA5 takes its tropospheric sulphate, and the
stratospheric sulphate that volcanic eruptions leave, from forcing data prepared for the Coupled
Model Intercomparison Project Phase 5 ([ECMWF Newsletter
159](https://www.ecmwf.int/en/newsletter/159/meteorology/global-reanalysis-goodbye-era-interim-hello-era5),
[ECMWF Newsletter
174](https://www.ecmwf.int/en/newsletter/174/news/updating-land-and-aerosol-properties-improve-reanalyses-and-seasonal)).
ERA5's aerosol therefore varies by month and from year to year, but not with the day's haze. CAMS's
irradiance takes aerosol from CAMS's own reanalysis and global forecasting system ([CAMS radiation
documentation](https://confluence.ecmwf.int/x/jOLjDw)), and UKV carries a fixed climatology (see
[UKV assimilates satellite
cloud](#ukv-assimilates-satellite-cloud-and-carries-a-fixed-aerosol-climatology)).

**The published validations of ERA5's wind cited here come from sites unlike the wind study's farms,
which are onshore in flat Lincolnshire.** Offshore, ERA5 underestimates strong wind speeds. [Gandoin
and Garza (2024)](https://doi.org/10.5194/wes-9-1727-2024) trace the bias to ERA5's surface drag
formulation and its dependence on sea state, worst for short fetches (wind that has crossed only a
short stretch of open water) over shallow seas such as the North Sea. At the IJmuiden mast they
measure an underestimate of almost 10% for the largest wind speeds. Against Doppler lidar at heights
of 100 to 500 m, [Cheynet et al. (2025)](https://doi.org/10.5194/wes-10-733-2025) found ERA5 and the
3 km Norwegian hindcast NORA3 similar offshore, and NORA3 better than ERA5 at two coastal sites and
one complex-terrain site in Norway.

### CAMS: use the point API, not the gridded product

**Two CAMS products exist, and only the point time-series product is current.** The [gridded
product](https://ads.atmosphere.copernicus.eu/datasets/cams-gridded-solar-radiation) would suit the
H3 pipeline — 0.1°, 15-minute, monthly netCDF — but it lags years behind, and version 4.6 (rev2) is
the only version that reaches 2024:

| Version | Years available |
|---|---|
| 4.5 | 2005–2022 |
| 4.6 | 2005–2023 |
| 4.6 (rev2) | 2005–2024 |

**A grid that stops in 2024 is disqualifying for capacity estimation**, because the domestic solar
fleet the estimate has to track has kept growing since. That table was read from the Atmosphere Data
Store catalogue API on 2026-09-09, and a [user who asked for April 2025 onward in May
2026](https://forum.ecmwf.int/t/cams-gridded-solar-radiation-data-for-the-period-01-04-2025-to-31-03-2026/14983)
was given no release date and pointed at the point product instead.

**The [point time-series
product](https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries) runs to
yesterday**, and takes one latitude and longitude per request. That is annoying rather than
disqualifying: one request covers a location over a whole date range, so the count scales with sites
rather than with days. The [v1 trial area](../index.md#scope) needs at most 32 requests, 6 of them
for the solar farms, against a limit of 500 requests per day. The daily cap binds long before the
wait does, at least at the sizes tested: a 14-day request at a 15-minute step returned in 27 seconds
on 2026-09-21. A multi-year request has not been timed. Ask for `observed_cloud` rather than
`clear`: the all-sky response carries the clear-sky columns (`GHIc`, `BHIc`, `DHIc`, `BNIc`)
alongside the all-sky ones, so one request returns both, and their ratio is the clear-sky index that
conditions the capacity fit and the published estimates of CAMS's own error.

**No intraday correction can rest on CAMS, whatever the accuracy of its irradiance.** A request
naming today is rejected outright: probing the retrieval API on 2026-09-21 returned HTTP 400 for
that same day, and a complete set of values for 2026-09-20. The freshest value on offer is
yesterday's last interval, which the period-ending convention below timestamps at 00:00 UTC today. A
forecast job running at 09:00 UTC therefore gets nothing at all about the day it is forecasting. The
most recent daylight irradiance that job can reach dates from yesterday's sunset — between roughly
15:45 UTC midwinter and 20:35 UTC midsummer across NGED's licence areas.

**The latency is 1 day, not the 2 days some third-party documentation still gives.** The CAMS
Radiation Service halved the latency in [February
2026](https://forum.ecmwf.int/t/cams-solar-radiation-timeseries-new-data-availability-and-request-limits-on-ads/14735).
The same change raised the daily cap to the 500 requests above. The [`camsRad`
vignette](https://cran.r-project.org/web/packages/camsRad/vignettes/CAMS_solar_data.html) still
gives the coverage as "up to 2 days ago", so check any latency figure taken from outside the
Atmosphere Data Store against the request form itself.

**Great Britain sits inside the Meteosat Second Generation field of view** that bounds the all-sky
product, with the same low-winter-sun degradation SARAH-3 documents. Meteosat Third Generation is
not yet in the CAMS processing chain. Depending on a service still built on the older satellite is a
supply risk.

**Three traps in the point API each fail silently.**

- **Set `time_reference` to `universal_time`.** The request form also offers true solar time, which
  differs from UTC by the site's longitude plus the equation of time. Across Great Britain that
  offset runs from roughly −38 to +24 minutes, varying by site *and* by time of year, so it exceeds
  a half-hour step at western sites in winter and looks exactly like a retrieval bias.
- **Values are irradiation in Wh/m² accumulated over the step, not irradiance in W/m².** ECMWF
  timestamp each value at the **end** of its integration period, matching NGED's period-ending
  convention above.
- **CAMS recomputes each request on the fly** against the latest software release and the latest
  aerosol and cloud inputs, so the same timestamp can return a different value next year. Snapshot
  each fetch into Delta and treat the snapshot as the source of truth, or experiments stop being
  reproducible.

### What comparing weather products on the trial area's solar farms found

**The satellite retrieval beat the reanalysis by 4.29 points of mean absolute error on the six
metered solar farms, which is the largest effect anything in that experiment varied.** On the
124,849 hours both products cover, CAMS cut an XGBoost forecast's error from 9.66% to 5.37% of P99
output — the 99th percentile of each site's own metered output, which every figure there is
normalised by — and a fitted five-parameter physical model's from 10.03% to 6.20%. The largest
contrast from changing which beam/diffuse split the model saw was 0.126 points, and changing the
model family moved 1.05. Which product feeds the model dominates both. The write-up is [Does a
weather product's beam/diffuse split help a PV forecast?](../studies/beam-diffuse-split.md); the
code sits in `studies/beam_diffuse_split/`.

**Scoring eight products on one common set of hours puts the two satellite retrievals far ahead of
every weather model.** On 76,727 generator-hours from December 2022 to August 2026, CAMS scores
5.09% of capacity and SARAH-3 5.49%. Among the weather models and reanalyses, the German weather
service's ICON-D2 scores 7.76%, ICON-EU 8.39%, ICON global 8.48%, its reanalysis ICON-DREAM-EU
8.77%, the Met Office's UKV 8.86% as Open-Meteo serves its hourly value, and ERA5 9.08%. ICON-D2's
advantage shrinks within hours of each run, and ICON-EU no longer beats UKV once UKV's hour is
rebuilt from its own snapshots. Four more Open-Meteo models are scored on a shorter row set from
November 2024. The write-up, with what each offline consumer should read, is [Which weather product
best describes past sunshine?](../studies/past-weather/solar.md).

**Figures measured on different row sets do not compose into a ranking.** ERA5 scores 9.08% on the
eight-product row set and 8.87% on the shorter row set that adds four more Open-Meteo models: the
same product and the same generators, because restricting to hours every product covers changes
which hours are kept. Open-Meteo's hourly UKV beats ERA5 across the whole record by 0.22 points
[0.06, 0.39], and by 0.14 points since August 2024 [−0.04, +0.35], the second of which is not
statistically significant at the 5% level.

**CAMS infers cloud from Meteosat at the hour in question, whereas the weather models simulate it.**
So a retrieval of an hour that has already happened starts from the cloud field the others have to
predict.

**Three limits bound how far that carries.** UKV differs from ERA5 in aerosol treatment as well as
in resolution, so neither ordering is a clean resolution contrast. Open-Meteo's archive stitches
the first hours of each successive run ([the survey compares Open-Meteo's
archives](../background/weather-products-survey.md#which-archives-keep-whole-past-forecast-runs)), so
a product's effective lead follows its run frequency: hourly UKV is the T+0 analysis, while 3-hourly
ICON-D2 carries a lead of 1 to 3 hours, measured against the German weather service's own files.
ICON-D2 therefore beats UKV while forecasting further ahead than it, and the comparison is not
analysis against analysis. And CAMS is a retrieval rather than a model, so its lead is nil and its
win is partly a win for observing cloud rather than simulating it. Every measurement is of these
eight products on this fleet, and none has been shown to hold for every product at those
resolutions.

**ICON-D2 reaches only the eastern half of NGED's licence area, so its score is not a licence-wide
result.** Probing the Open-Meteo archive along a west-to-east line puts the model's western
boundary near 2.5°W: Bristol, Exeter, Cardiff, Swansea, and Truro return no data, while Birmingham
and Nottingham are covered. The whole of South West England and South Wales therefore sit outside
it, and every metered site in this comparison sits inside it, so nothing here says how ICON-D2
would perform in the west. **A limited-area model from a neighbouring country can be a component of
a blend over part of the licence area, and cannot be the single source feeding all of it.**

![ICON-D2 has no data west of a line running from 1.8°W at 49.9°N to 3.9°W at 57.3°N, and AROME
France has none north of 55.4°N. The UK is drawn in dark grey, and Ireland in light grey for
context](assets/weather_product_domains.svg)

**Two limited-area weather models cover only part of the UK, and every other weather product this
page names covers all of it.** ICON-D2 misses South West England, Wales, the western edge of
the Midlands and of North West England, the Isle of Man, Northern Ireland, and Scotland west of
Edinburgh and north of about 57.5°N, including Orkney and Shetland. Météo-France's AROME France,
listed on the [weather products survey](../background/weather-products-survey.md), misses
everything north of 55.4°N: all of Scotland except its southern fringe, and the northern tip of
Northumberland. AROME France does cover Northern Ireland, whose northernmost point is at 55.3°N.
The ICON-D2 outline is traced from the valid-data mask of one file the German weather service
publishes, the 2 m temperature from the 00 UTC run of 23 September 2026. That outline's western
edge at 53.2°N sits at 2.6°W, which agrees with Open-Meteo returning no data west of about 2.5°W.
The AROME France outline is the plain rectangle in the grid description Open-Meteo publishes, and
Open-Meteo's API confirms the northern edge by returning data at 55.3°N and none at 55.5°N. The
map is drawn by `studies/beam_diffuse_split/weather_product_domains.py`.

**CAMS's own reliability flag is worth acting on, and dropping the flagged hours raises a
P99-normalised error even though it improves the data.** The service flagged 17.9% of the daylight
hours it delivered as less than 90% reliable. Those hours are much darker than the ones kept — 33
W m⁻² of global irradiance against 285, and 3.5% of P99 output against 36.1% — so removing them
removes hours where every model is nearly right, and the remaining mean error rises mechanically.
Any comparison that drops them on one source and not on another is comparing two row sets rather
than two products; restrict to the shared hours first.

**The beam field CAMS publishes is worth having and the one ERA5 publishes is not.** On the 5 km
retrieval, giving a model the product's own split rather than running a separation model locally cut
error by about 1.5% relative. On the 31 km reanalysis it added nothing detectable. Whether the
retrieval's beam field is worth paying for depends on what a supplier charges for it, which that
experiment does not know.

### Met Office weather-station observations (MIDAS Open)

**The project ingests no weather-station observations, and the studies read the Met Office's MIDAS
Open only to test whether a nearby station stands in for a gridded product.** MIDAS Open is released
once a year, and its current release ends on 2025-12-31. MIDAS Open can therefore feed offline work
but not the live forecast. On sunshine, a nearby station's irradiance scored worse than CAMS and
better than ERA5. On wind, a nearby anemometer scored worse than ERA5's 10 m wind. Adding a
station's observations to a gridded product (CAMS on sunshine, UKV on wind) lowered that product's
error, so neither result rules a station out as an extra input. The licence, the stations fetched,
the variables, the results, and the limits of both studies are in the [weather products
survey](../background/weather-products-survey.md#weather-station-observations).

### UKV assimilates satellite cloud, and carries a fixed aerosol climatology

**UKV's radiation scheme sees no time-varying aerosol**, which matters because aerosol sets how
sunlight divides between the direct beam and the diffuse sky. The regional configuration's radiation
uses a fixed five-species climatology, unchanged from RAL1 through the RAL3 package that went
operational at PS47 on 2026-01-21: [Bush et al. (2020)](https://doi.org/10.5194/gmd-13-1999-2020)
describes the climatology, and [Bush et al. (2025)](https://doi.org/10.5194/gmd-18-3819-2025)
records that no radiation parameters changed between RAL2 and RAL3. UKV's only advected aerosol
quantity is the Murk tracer of [Clark et al. (2008)](https://doi.org/10.1002/qj.318), which
diagnoses visibility and does not reach the radiation calculation. CAMS is aerosol-informed by
construction. ERA5's aerosol is prescribed, a monthly climatology plus sulphate forcing, with
nothing assimilated (see [Why ERA5 describes past sunshine and wind worse than most current weather
products](#why-era5-describes-past-sunshine-and-wind-worse-than-most-current-weather-products)). So
any skill ERA5's split shows could come from cloud or, weakly, from prescribed aerosol that varies
by month and year but not with the day's haze, and **any skill UKV's split shows can only be
cloud-sourced**, where CAMS's could be either.

**UKV's 4D-Var assimilates a large volume of satellite-derived cloud, so at short lead times the
model is partly a retrieval.** Satellite-derived cloud fraction was the single largest observation
type by count in UKV's 2013 observation table, at 650,000 a day against 39,000 for SEVIRI radiances
([Tubbs and Kelly
(2013)](https://www-cdn.eumetsat.int/files/2020-04/pdf_conf_p_s7_09_tubbs_v.pdf)), entering the
humidity field as a pseudo-observation by the mechanism of [Renshaw and Francis
(2011)](https://doi.org/10.1002/qj.980). The [PS43 release
notes](https://www.metoffice.gov.uk/services/data/met-office-data-for-reuse/ps43_ftp) confirm the
stream was operational and being refined in December 2019, and the PS47 notes name only latent heat
nudging and the adaptive vertical grid as assimilation methods removed. **No document from 2020
onwards was found positively re-confirming the satellite cloud stream, and whether surface solar
irradiance is assimilated anywhere in the Met Office's systems could not be established.** A UKV
analysis and a satellite product are therefore partly downstream of the same geostationary
satellite, so a contrast between the two weakens at short lead and recovers as the model's own
physics overwrites the initial cloud field.

### Open-Meteo's UKV archive is the T+0 analysis, and half of it is backfill

**Open-Meteo mirrors UKV from the Met Office's own AWS bucket, ingesting every hourly run about four
hours late.** A later run overwrites an earlier one for the same valid time, so the archive holds
the T+0 analysis rather than a forecast at any lead. Sampled at five instants either side of PS47,
the nearest grid cell in the Met Office's own file agrees with the served snapshot to between 0.11
and 0.55 W m⁻², where every other lead sits tens to hundreds of W m⁻² away. **An archive of
analyses is the right object to compare against a reanalysis or a retrieval and the wrong one to
read as forecast skill.**

**Over half the archive predates the ingest that produced the rest.** Open-Meteo's UKV downloader
was created on 2024-08-12, and the archive claims to start on 2022-03-01, so the 29 months between
were backfilled from a source Open-Meteo does not name. The AWS bucket's rolling two-year window
reached back only to about August 2022 on the day the downloader landed, so an AWS backfill does not
account for the stated start either, and the window has since rolled past all of it. **The era that
can be checked against the Met Office's files is essentially the era Open-Meteo ingested live**, and
the earlier half cannot be checked against anything.

**A Met Office upgrade moved how much UKV's published split improves the forecast, by more than
the backfill boundary does.** In the beam/diffuse split experiment, the gain from UKV's direct-beam
share is five and a half times larger over the 8 months after Parallel Suite 47 became operational
on 2026-01-21 than over the 47 months before. Neither other source steps over those same months:
the satellite source's contrast moved by about a third, from −0.100 to −0.068 percentage points,
and the reanalysis stayed null in both eras. UKV's global irradiance went the other way, sitting
about 1 percentage point worse against the reanalysis from the same date.

**One upgrade moved this product's value by a factor of five and a half, so a production ingest of
UKV should score continuously rather than trust a figure measured once**, and should not assume
that one era of UKV stands in for another.

**Three mechanical facts about the served columns, each of which fails silently if assumed wrong.**

- **Only global and direct short-wave are ingested; diffuse is served as their difference.** The Met
  Office publishes its own diffuse field and Open-Meteo does not read it, so the served diffuse
  carries no information the other two columns lack.
- **The default hourly column is a backward-looking mean over the hour ending at its label**, which
  matches the period-ending convention used everywhere else in this project. UKV publishes radiation
  as an instantaneous snapshot, and Open-Meteo divides that snapshot by the ratio of the
  instantaneous cosine of the solar zenith angle to its mean over the preceding hour. Asking for the
  `_instant` suffix multiplies the ratio back to recover the snapshot, which lands half an hour
  later than the hour's centre.
- **That conversion is skipped where the ratio falls below 0.05**, near sunrise and sunset, so the
  round trip between the two columns does not hold at very low sun.

### UKV from CEDA's archive differs from UKV as Open-Meteo serves it in lead, irradiance, temperature, and units

**The project can read UKV from three archives, and the three hold the same Met Office weather
model but not the same series.** An XGBoost model trained on one archive and scored on another
therefore meets inputs that its training never showed. The comparison of CEDA's archive with
Open-Meteo's rests on the [study of UKV from CEDA against UKV from
Open-Meteo](../studies/past-weather/ukv-ceda-vs-openmeteo.md), which compares the two archives at
nine metered generators over the 23 whole months they share. The facts about the Met Office's own
AWS bucket are pilot findings, from a listing of the bucket and a download of two days of runs. The
last column of the table says where each row's evidence comes from.

| Difference | UKV from CEDA's archive | UKV as Open-Meteo serves it | UKV in the Met Office's AWS bucket | Evidence |
|---|---|---|---|---|
| [Runs and lead](#each-archive-serves-a-different-lead) | 8 runs a day; the studies read the 00, 06, 12, and 18 UTC runs at leads of 0 to 5 hours | Every hourly run, a later run overwriting an earlier one, so in effect the T+0 analysis | Every hourly run, each kept whole to 54 hours, or to 120 hours for the 03 and 15 UTC runs | CEDA: catalogue record and download script; Open-Meteo: measured for irradiance; AWS: pilot |
| [History](#the-archives-differ-in-how-much-of-each-physics-era-they-hold) | From 2016-03; the studies' download from 2019-09 | Live ingest from 2024-08-12, and a backfill of unnamed origin before that date | A rolling window of about two years, reaching back to 2024-10-04 on 2026-10-06 | Documented; AWS: pilot |
| [Grid and files](#the-met-offices-aws-bucket-keeps-every-hourly-run-on-a-different-grid) | GRIB files on a 548 by 704 grid on the Ordnance Survey national grid, first row in the north | Values at the requested coordinates | One NetCDF file per variable and valid time, on a 970 by 1042 Lambert azimuthal equal-area grid, first row in the south | Download script; AWS: pilot |
| [Irradiance](#open-meteos-hourly-irradiance-is-a-scaled-snapshot-whose-construction-changes-after-ps47) | One global short-wave snapshot | The snapshot at the hour's end, scaled by a ratio of cosines of the solar zenith angle, and built differently after PS47 | Global, direct, and diffuse short-wave, each a snapshot; the three do not add up in 2024 | Measured; AWS: pilot |
| [Temperature](#temperature-agrees-closely-at-lead-0-and-wind-differs-in-units-level-and-heights) | Hourly instants | Hourly instants, a mean absolute difference of 0.098 K from CEDA's at lead 0 | Hourly instants, and 15-minute files in 2026 runs | Measured; AWS: pilot |
| [Wind](#temperature-agrees-closely-at-lead-0-and-wind-differs-in-units-level-and-heights) | m/s; 10 m, 925 hPa, and 1000 hPa | km/h; 10 m, 50 m, 80 m, and 100 m; at 10 m about 3% below CEDA's at lead 0, and about 6% above in two spans | m/s; 10 m, pressure levels, and 33 or 56 height levels | Measured at 10 m; AWS: pilot |
| [Licence](#the-archives-carry-different-licences) | CC BY-NC-SA 4.0 | Open-Meteo's terms | CC BY-SA 4.0 | Documented |

#### Each archive serves a different lead

**CEDA's value for an hour comes from a run that started up to 5 hours earlier, and Open-Meteo's
value comes from the analysis of that hour.** CEDA's catalogue record describes UKV runs "up to
eight times a day". The project's CEDA download script records 8 runs a day, with files reaching
120 hours for the 03 and 15 UTC runs only. The studies read the 00, 06, 12, and 18 UTC
runs, which reach 54 hours, and take each hour from the latest of those runs that started at or
before the hour. The lead is therefore 0 to 5 hours, and the hour of day fixes the lead. Open-Meteo
ingests every hourly run, and a later run overwrites an earlier run for the same valid time, as [the
section above](#open-meteos-ukv-archive-is-the-t0-analysis-and-half-of-it-is-backfill) describes.
The two archives can hold the same analysis only at 00, 06, 12, and 18 UTC, where CEDA's lead is 0.
Across the nine generators, the mean absolute difference between the two archives grows from
0.098 K at lead 0 to 0.557 K at lead 5 for temperature, and from 0.190 m/s to 0.519 m/s for 10 m
wind speed ([the study's model-free
comparison](../studies/past-weather/ukv-ceda-vs-openmeteo.md#the-archives-agree-closely-at-lead-0-and-drift-apart-with-cedas-lead)).

#### Open-Meteo's hourly irradiance is a scaled snapshot whose construction changes after PS47

**Open-Meteo's `shortwave_radiation` for an hour is not the end-of-hour snapshot that the Met Office
publishes.** As [the section
above](#open-meteos-ukv-archive-is-the-t0-analysis-and-half-of-it-is-backfill) says, Open-Meteo
scales the snapshot by the ratio of the hour's mean cosine of the solar zenith angle to the cosine
at the hour's end. At CEDA's lead-0 hours from 2024-09 to 2025-12, with Open-Meteo's value above
50 W m⁻², the median of CEDA's raw snapshot over Open-Meteo's value is 1.355 at 06 UTC, 1.011 at
12 UTC, and 0.774 at 18 UTC. CEDA's raw snapshot therefore reads 36% high at 06 UTC and 23% low at
18 UTC against Open-Meteo's value. An XGBoost model trained on the raw snapshot and given
Open-Meteo's value meets a scaling by hour of day that its training never showed.

**Scaling CEDA's snapshot by the same cosine ratio reproduces Open-Meteo's value at the median
before the Met Office's PS47 upgrade, and does not after the upgrade.** On the same hours, the
median ratio of the rebuilt CEDA value to Open-Meteo's is 1.005, 1.000, and 0.996 at 06, 12, and
18 UTC. From 2026-02 to 2026-08 the median ratios are 1.111, 1.000, and 0.864, so Open-Meteo builds
the hourly value some other way after PS47. The mismatch after PS47 is not confined to low sun.
With the sun more than 10 degrees above the horizon the median ratio is 1.000 to 1.004 before and
after PS47. The 10th to 90th percentile of the ratio at 10 to 20 degrees of elevation is 0.941 to
1.058 before PS47 and 0.823 to 1.210 after. The mean absolute difference at lead 0 is 8.0 W m⁻²
before PS47 and 22.9 W m⁻² after. Within a few degrees of the horizon the rebuild fails before and
after PS47: before PS47 the median ratio is 0.008 up to 2 degrees of elevation and 0.681 from 2 to
5 degrees ([the study's irradiance
section](../studies/past-weather/ukv-ceda-vs-openmeteo.md#open-meteo-builds-its-hourly-irradiance-differently-after-the-2026-upgrade-and-in-low-sun)).

#### Temperature agrees closely at lead 0, and wind differs in units, level, and heights

**At lead 0 the two archives' temperatures differ by about a tenth of a degree.** Both archives
hold hourly instants of temperature, and the study averages the instants at the two ends of each
hour for both archives. At CEDA's lead 0, the mean difference is −0.021 K (CEDA minus Open-Meteo),
the mean absolute difference is 0.098 K, and the correlation is 0.9997. The mean lead-0 difference
at each of the nine generators lies between −0.045 K and −0.006 K, so any adjustment Open-Meteo
makes for a point's elevation is small at the nine generators. Whether Open-Meteo adjusts
temperature for elevation at all is not established.

**Open-Meteo serves wind in km/h and CEDA's files hold wind in m/s, and after conversion
Open-Meteo's 10 m speed is about 3% below CEDA's.** The study divides Open-Meteo's speed by 3.6 and
stops unless the median ratio of the two archives' 10 m speeds lies between 0.9 and 1.1. At lead 0
the median ratio of Open-Meteo's speed to CEDA's is 0.968 outside the two spans described below,
and the monthly median stays between 0.960 and 0.974 in each of the 20 months outside the spans.
CEDA's 10 m direction is about 1.9 degrees lower than Open-Meteo's at lead 0.

**In two spans, Open-Meteo's 10 m speed is about 6% above CEDA's instead of 3% below.** The spans
are the UTC days 2024-11-07 to 2024-11-30 and 2025-01-16 to 2025-02-18, and the median ratio of
Open-Meteo's speed to CEDA's inside them is 1.063. Against ERA5, CEDA's speed stays steady across
those months while Open-Meteo's speed rises, so the step is in Open-Meteo's served series. The cause
of the step is unknown, and the study drops both spans from its wind fits ([the
step](../studies/past-weather/ukv-ceda-vs-openmeteo.md#open-meteos-10-m-wind-speed-is-about-3-below-cedas-and-steps-up-in-two-spans)).
A training history read from Open-Meteo's UKV contains both spans.

**Even at lead 0, Open-Meteo's value is not the nearest CEDA cell's value.** Of the nine CEDA cells
around each generator, the nearest cell holds the value closest to Open-Meteo's at 27% of lead-0
hours for 10 m wind speed and 47% for temperature, averaged over the generators, where chance is
11%. Interpolation from the native grid and a different grid are both possible causes, and neither
cause is verified.

**The archives offer different wind heights.** CEDA's files hold 10 m wind and wind at the 925 hPa
and 1000 hPa pressure levels, and no wind on height levels, so CEDA carries no 100 m wind.
Open-Meteo serves UKV wind at 10 m, 50 m, 80 m, and 100 m. The study compares 10 m wind only.

#### The Met Office's AWS bucket keeps every hourly run on a different grid

**The Met Office's AWS bucket keeps every hourly UKV run whole for about two years, which neither
CEDA's archive nor Open-Meteo's archive does.** The findings below come from listing the bucket on
2026-10-06 and from a pilot download of all 24 runs of 2024-10-08 and of 2026-10-05 at leads 0 to 5
hours, and are pilot findings rather than a study result.

- **Runs and window.** The bucket `met-office-atmospheric-model-data` holds UKV under the prefix
  `uk-deterministic-2km/`, with 24 runs a day. Most runs reach 54 hours at hourly steps, and the 03
  and 15 UTC runs reach 120 hours, hourly to 54 hours and 3-hourly after. The oldest runs are
  deleted day by day: on 2026-10-06 the first run held was 2024-10-04 21 UTC, and the first whole
  day was 2024-10-06.
- **Files and grid.** Each file holds one variable at one valid time on the whole 970 by 1042 grid,
  a Lambert azimuthal equal-area projection. CEDA's GRIB files sit on a 548 by 704 grid on the
  Ordnance Survey national grid. The first row of the AWS grid is the southernmost row, and the
  first row of CEDA's grid is the northernmost row, so code that assumes one orientation reads the
  other archive upside down. The files are chunked in blocks of 128 by 128 cells, so a range read
  fetches only the blocks over a small area: the pilot requested 0.43 MB per file on average.
- **Every field is a snapshot.** The files for the eight variables the pilot read carry no
  `cell_methods` attribute and no time bounds, so under the CF conventions every field, short-wave
  included, is a value at the valid time. No file labels a field as instantaneous in words.
- **Irradiance components.** The bucket publishes global, direct, and diffuse short-wave as three
  fields, where CEDA's archive holds one short-wave field. On 2024-10-08 the three fields do not add
  up: in daylight, global minus direct minus diffuse has a mean absolute value of 8 to 42 W m⁻² per
  run hour. On 2026-10-05 the residual is 0.06 W m⁻² or less. The date from which the three fields
  add up has not been found.
- **Height levels.** Wind on height levels comes on 33 levels in the files from 2024-10-15 to
  2026-01-20 and on 56 levels from 2026-01-22. The 50 m, 75 m, 100 m, and 150 m levels are in every
  file probed on nine dates, and the 125 m level is only in the 56-level files. Wind on pressure
  levels is published too.
- **Files every quarter hour.** Some variables, among them screen temperature and 10 m wind, also have
  15-minute files out to 54 hours in the 2026 runs. The 2024 runs have none.

#### The archives differ in how much of each physics era they hold

**A series that spans one of the Met Office's upgrades is not homogeneous in any of the three
archives, and the archives span different eras.** The Met Office's PS43 upgrade of 2019-12-04 moved
UKV to the RAL2-M physics. No date or content for PS44 could be found. UKV stayed on the RAL2-M
physics after PS45 in May 2022, and PS46 in May 2025 was a move to a new supercomputer with no
science change intended. PS47 on 2026-01-21 moved UKV to the RAL3 physics, with new microphysics and
a new cloud scheme ([the list of upgrades](#nwp-model-upgrades-since-2019)). CEDA's archive starts
in 2016-03, and the project's CEDA download starts on 2019-09-01, so the download holds three months
before PS43 and the whole era from PS43 to PS47. Open-Meteo's live ingest starts on 2024-08-12, and
the AWS bucket's window reached back to 2024-10-04 on 2026-10-06, so neither Open-Meteo's archive
nor the AWS bucket can test PS43, PS44, or PS45. The 23 whole months that CEDA's and Open-Meteo's
archives share run from 2024-09 to 2025-12, 16 months before PS47 that include PS46, and from
2026-02 to 2026-08, 7 months after PS47. The study drops 2026-01, the month of the upgrade.

#### The archives carry different licences

**CEDA's UKV archive is licensed CC BY-NC-SA 4.0, which allows non-commercial use only and requires
adaptations to be shared alike.** CEDA's catalogue record asks users to cite the data as [Met Office
(2016): NWP-UKV: Met Office UK Atmospheric High Resolution Model
data](https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f), with CEDA as the
publisher. The Met Office's AWS bucket is published under British Crown copyright and CC BY-SA 4.0,
with no non-commercial restriction ([listed above](#weather-data)). Open-Meteo's free API is for
non-commercial use only, and its paid plans cover commercial use
([survey](../background/weather-products-survey.md#how-to-read-the-open-meteo-rows)). Whether CEDA's
licence permits the project's use of CEDA's UKV is a decision for the maintainer.

#### Open-Meteo's UKV is checked against the Met Office's files for irradiance only, and CEDA's for two runs

**Open-Meteo's UKV matches the Met Office's own files for irradiance, and the project has not
checked Open-Meteo's wind or temperature against those files.** At five instants sampled either side
of PS47, Open-Meteo's irradiance snapshot agrees with the nearest cell of the Met Office's own file
to between 0.11 and 0.55 W m⁻². Hours older than the AWS bucket's rolling window, which reached
back to 2024-10-04 on 2026-10-06, can no longer be checked against the bucket.

**CEDA's UKV agreed closely with the AWS bucket's UKV in a spot check of two runs, too few to settle
whether the two differ.** For the 00 and 12 UTC runs of 2024-10-08 at leads 0 to 5 hours, with each
AWS cell matched to the nearest CEDA cell (1.05 km apart on average), the mean difference in screen
temperature was 0.02 K or less, and the AWS bucket's 10 m speed was higher than CEDA's by 0.04 to
0.24 m/s. The project's download notes record that CEDA's archive is statistically different from
the live feed, and that an XGBoost model trained on one should not be used on the other. The
mechanism behind that note has not been established, and the [study of CEDA against
Open-Meteo](../studies/past-weather/ukv-ceda-vs-openmeteo.md) compares CEDA's archive with
Open-Meteo's and not with the Met Office's own files.

## NWP model upgrades since 2019

**The weather models the project reads have changed many times since 2019, and many of those
changes alter irradiance, cloud, or wind.** ECMWF changed the IFS cycle behind ECMWF ENS seven
times between June 2019 and May 2026. The German weather service (DWD) changed ICON in ways that
touch cloud, radiation, or wind on 16 dates between July 2019 and September 2026, counting the
introduction of ICON-D2. The Met Office changed UKV's physics package twice. A model trained on one
version of a weather model and run on the next meets inputs whose relationship to power has moved.
After the Met Office's PS47 upgrade, [the error reduction from giving a model UKV's published
direct-beam share grew five and a half times
larger](#open-meteos-ukv-archive-is-the-t0-analysis-and-half-of-it-is-backfill), comparing the 8
months after the upgrade with the 47 months before. How the live service
plans to handle an upgrade is in [Live service → NWP model
upgrades](live-service.md#nwp-model-upgrades).

**The full list is [`nwp-model-upgrades.csv`](assets/nwp-model-upgrades.csv), 98 changes under 19
product labels, each checked against the producer's own text on 2026-09-23.** The `verified` column
reads `yes` where the check confirmed the row as written, `corrected` where the check changed it, and
`unverified` where no producer text could be found. The `note` column says what the check found. The
list also covers products the project neither reads nor evaluates: NOAA's GFS and GEFS,
Météo-France's ARPEGE and AROME, KNMI's HARMONIE-AROME, MET Norway's NORA3, and CERRA's publication
history.

**The table below keeps only the verified changes that alter irradiance, cloud, or wind over Great
Britain, in the weather models the project reads or is evaluating, and in the two archives the
studies read them through.** It leaves out changes confined to DWD's ICON-EPS ensemble, which the
project does not read. ERA5 has no rows, because ECMWF froze ERA5 at IFS cycle 41r2 for the whole
record. ERA5's near-real-time stream, ERA5T, has correction periods in which it differs from the
final ERA5, but the check found no correction period that touches irradiance or wind. Between PS43
and PS47 the check found no UKV physics change. UKV was still on the RAL2-M physics after PS45 in
May 2022, PS46 in May 2025 was a move to a new supercomputer with no science change intended, and
no date or content for PS44 could be found.

| Takes effect | Model | What changed | Source |
|---|---|---|---|
| 2019-06-11 06 UTC | ECMWF IFS 46r1 | Longwave scattering switched on in the radiation scheme; convection changes; new 200 m wind output | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+46r1) |
| 2020-06-30 06 UTC | ECMWF IFS 47r1 | Revised shortwave albedo of land, snow, and sea ice; updated total solar irradiance; less ocean drag at high wind speeds | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+47r1) |
| 2021-05-11 06 UTC | ECMWF IFS 47r2 | ENS vertical levels raised from 91 to 137 | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+47r2) |
| 2021-10-12 06 UTC | ECMWF IFS 47r3 | New moist physics: cloud vertical overlap, sub-grid cloud, and gustiness revised; resolution unchanged | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+47r3) |
| 2023-06-27 06 UTC | ECMWF IFS 48r1 | ENS grid refined from about 18 km to about 9 km | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+48r1) |
| 2024-11-12 06 UTC | ECMWF IFS 49r1 | 2 m temperature observations assimilated; the SPP stochastic scheme replaces SPPT in the ensemble; ECMWF reports better 10 m wind | [ECMWF](https://www.ecmwf.int/en/about/media-centre/news/2024/forecast-upgrade-improves-wind-and-temperature-predictions) |
| 2026-05-12 06 UTC | ECMWF IFS 50r1 | SPP revised to reduce excessive 10 m wind spread; aerosol climatology updated | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+50r1) |
| 2025-02-25 06 UTC | ECMWF AIFS Single v1 | First operational AIFS Single, on a grid of about 32 km, with `ssrd`, `strd`, and 100 m wind among its outputs | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+Single+v1) |
| 2025-07-01 06 UTC | ECMWF AIFS ENS v1 | First operational AIFS ENS: 51 members on a grid of about 31 km, with `ssrd` and 100 m wind among its outputs | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v1) |
| 2025-08-27 06 UTC | ECMWF AIFS Single v1.1 | Soil-moisture loss weight reduced by a factor of 100 to remove spurious point-rainfall artefacts; an earlier attempt on 2025-07-31 was reverted on 2025-08-01 | [ECMWF](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+Single+v1) |
| 2026-05-12 06 UTC | ECMWF AIFS Single v2 | New wave, snow-cover, and 10 hPa outputs; fine-tuned on operational and experimental 50r1 analyses | [ECMWF](https://www.ecmwf.int/en/newsletter/187/news/implementation-aifs-v2) |
| 2026-05-12 06 UTC | ECMWF AIFS ENS v2 | Multi-scale training loss and stricter variable bounds; fine-tuned on 2018 to 2024 data plus 50r1 data | [ECMWF](https://confluence.ecmwf.int/spaces/FCST/pages/620418893) |
| 2019-07-30 06 UTC | DWD ICON 2.5.0 | Revised cloud-cover scheme, aimed at better cloud and radiation forecasts | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2019/pdf_icon_30_07_2019.pdf?__blob=publicationFile) |
| 2019-10-22 06 UTC | DWD ICON 2.5.1 | Cloud-cover scheme retuned; DWD reports global radiation error down 1–2% and daytime radiation bias down 3–4 W m⁻² | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2019/pdf_icon_22_10_2019.pdf?__blob=publicationFile) |
| 2020-05-19 09 UTC | DWD ICON | Aeolus satellite wind observations assimilated | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2020/pdf_icon_19_05_2020.pdf?__blob=publicationFile) |
| 2021-02-10 09 UTC | DWD ICON-D2 | ICON-D2 replaces COSMO-D2 as DWD's convection-permitting regional model | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon_d2/pdf_2021/pdf_icon_system_10_02_2021.pdf?__blob=publicationFile) |
| 2021-04-14 06 UTC | DWD ICON 2.6.2 | ecRad replaces RRTM as the radiation scheme, with stochastic cloud overlap, in ICON global, ICON-EU, and ICON-D2 (09 UTC) | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2021/pdf_icon_14_04_2021.pdf?__blob=publicationFile) |
| 2021-09-08 | DWD ICON 2.6.3 | Sub-grid orography scheme retuned; small improvement in northern-hemisphere winter wind | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2021/pdf_icon_08_09_2021.pdf?__blob=publicationFile) |
| 2022-11-23 06 UTC | DWD ICON | Vertical levels raised from 90 to 120 in ICON global and from 60 to 74 in ICON-EU; new orography; adaptive surface friction; DWD reports a smaller positive wind bias at turbine height | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2022/pdf_icon_23_11_2022.pdf?__blob=publicationFile) |
| 2023-03-15 06 UTC | DWD ICON | 10 m wind assimilation extended; shallow convection switched off in stable boundary layers, improving stratocumulus and the solar radiation bias; ICON global and ICON-D2 (09 UTC) | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2023/pdf_icon_15_03_2023.pdf?__blob=publicationFile) |
| 2023-11-28 06 UTC | DWD ICON 2.6.6 | Sub-grid orography and turbulence retuned for winter; the cloud-cover scheme gains an inversion diagnostic to improve stratocumulus | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2023/pdf_icon_28_11_2023.pdf?__blob=publicationFile) |
| 2024-01-24 | DWD ICON-D2 | 10 m wind assimilation and adaptive surface friction extended | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/nwv_aenderungen_icon_d2_gesamt.html) |
| 2024-07-09 09 UTC | DWD ICON-D2 | Gust parameterisation revised | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon_d2/pdf_2024/pdf_icon_d2_09_07_2024.pdf?__blob=publicationFile) |
| 2024-12-04 06 UTC | DWD ICON | Adaptive parameter tuning extended; snow cover revised in the surface transfer, which can shift near-surface wind; ICON global and ICON-D2 (09 UTC) | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2024/pdf_icon_04_12_2024.pdf?__blob=publicationFile) |
| 2025-07-23 06 UTC | DWD ICON | Dissipative heating and warm-layer ocean parameterisations added in every ICON configuration | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2025/pdf_icon_23_07_2025.pdf?__blob=publicationFile) |
| 2025-09-24 06 UTC | DWD ICON | Inversion-cloud scheme extended from stratocumulus to stratus, for winter fog and low cloud, and switched on in ICON-D2 | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2025/pdf_icon_24_09_2025.pdf?__blob=publicationFile) |
| 2026-02-18 09 UTC | DWD ICON-D2 | New output field for 10 m wind corrected for sub-grid orography; adaptive surface friction restricted | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2026/pdf_icon_18_02_2026.pdf?__blob=publicationFile) |
| 2026-09-02 12 UTC | DWD ICON | A prognostic aerosol scheme replaces the Tegen climatology for non-dust aerosol in ICON global and ICON-EU; in DWD's tests over central Europe, global radiation increases significantly | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2026/pdf_icon_02_09_2026.pdf?__blob=publicationFile) |
| 2026-10-06 06 UTC, announced | DWD ICON-EU | ICON-EU domain slightly enlarged, mainly to the south | [DWD](https://www.dwd.de/DE/fachnutzer/forschung_lehre/numerische_wettervorhersage/nwv_aenderungen/_functions/DownloadBox_modellaenderungen/icon/pdf_2026/pdf_icon_06_10_2026.pdf?__blob=publicationFile) |
| 2019-12-04 | Met Office PS43 | UKV and MOGREPS-UK move to the RAL2-M physics: mixed-phase ice cloud fraction, sub-grid turbulence, and lying snow | [Met Office authors, GMD 2023](https://gmd.copernicus.org/articles/16/1713/2023/) |
| 2026-01-21 | Met Office PS47 | UKV and MOGREPS-UK move to the RAL3 physics: double-moment microphysics (CASIM), a bimodal cloud scheme, and boundary-layer and land-surface updates | [Met Office](https://datahub.metoffice.gov.uk/support/changes-and-updates) |
| 2021-06 | CAMS Radiation Service v4.0 | APOLLO_NG cloud detection replaces APOLLO | [ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/266592908/CAMS+solar+radiation+time-series+data+documentation) |
| 2022-09 | CAMS Radiation Service v4.5 | New all-sky model, with the bias correction removed | [ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/266592908/CAMS+solar+radiation+time-series+data+documentation) |
| 2023-07 | CAMS Radiation Service v4.6 | The McClear clear-sky model reads inputs from IFS cycle 48r1 from 2023-06-27; earlier data unchanged | [ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/266592908/CAMS+solar+radiation+time-series+data+documentation) |
| 2023-07-04 | Open-Meteo v0.2.69 | Solar-radiation calculation improved, including direct normal irradiance | [Open-Meteo](https://github.com/open-meteo/open-meteo/releases) |
| 2025-10-01 | Open-Meteo | Native 9 km IFS HRES served, with ECMWF's own direct radiation, after ECMWF opened its real-time catalogue. ECMWF's own news says the 9 km catalogue is planned for later in 2026, which conflicts with this date; the conflict is unresolved | [Open-Meteo](https://openmeteo.substack.com/p/ecmwf-transitions-to-open-data) |

**Date an upgrade by the first run it applies to, not by the day it was announced.** ECMWF brings in
each cycle at the 06 UTC run, so the 00 UTC run of the implementation day, which is the run the
project ingests, is still on the old cycle. DWD publishes a notice 5 to 15 days before a change
takes effect. The effective date is in each notice's title ("Gültigkeit ab") and in the PDF's file
name, and the dates above are effective dates. A month-only date, such as the CAMS rows', means the
producer published no day. Open-Meteo's dates are software releases, and the hosted service may
deploy a change on a different day.

**A CAMS version change may rewrite history, because the Radiation Service computes each request
afresh.** The changelog says that v4.6 left data before 2023-06-27 unchanged, and says nothing either
way about v4.0 and v4.5. A series fetched before a version change and one fetched after it may
therefore disagree over the same hours, which is one more reason to [snapshot each
fetch](#cams-use-the-point-api-not-the-gridded-product).

**Two fields in Dynamical.org's ENS archive start part-way through it.** The archive starts on
2024-04-01, but 10 m gust starts on 2024-11-13 and total cloud cover on 2025-11-21. Neither field is
ingested today; a feature built on either would meet nulls across the early part of the record.

## Electricity price data

**The price series below are free and need no API key.** Batteries, pumped storage, and flexible
demand and generation charge and discharge according to the prices, so the prices shape part of what
NGED's substations see. The sources are 🔬 research: the project has downloaded the four price series
(imbalance system price, market index price, N2EX day-ahead auction price, and carbon intensity) for
1 September 2025 to 30 September 2026 for one study, is downloading the balancing-unit-level
acceptance series (`BOAV`, `EBOCF`, and `BOALF`), and ingests none of them.

**Each licence's attribution wording applies wherever the data is shown.** Elexon's Balancing
Mechanism Reporting Service (BMRS) licence requires the wording "Contains BMRS data © Elexon Limited
copyright and database right [year]", which the project has not independently verified. The Carbon
Intensity API's CC BY 4.0 licence comes from its documentation, and the API's responses carry no
licence field.

### Which price series are free, and how far back they go

| Series | Source and licence | Resolution | History found |
|---|---|---|---|
| **Imbalance system price** (Elexon dataset `DISEBSP`) | [Elexon Insights API](https://bmrs.elexon.co.uk/system-prices), `/balancing/settlement/system-prices/{date}`; Elexon's open licence for BMRS data | Half-hourly | A request for 2015-12-01 returned 48 periods |
| **Market index price** (Elexon dataset `MID`, provider `APXMIDP`) | [Elexon Insights API](https://bmrs.elexon.co.uk/market-index-data), `/datasets/MID/stream`; Elexon's open licence for BMRS data | Half-hourly | A request for 2016-10-31 returned data, and a request for 2016-01-01 returned none |
| **Day-ahead auction price** (N2EX) | [National Energy System Operator (NESO) data portal](https://www.neso.energy/data-portal/gb-n2ex-day-ahead-price), Comprehensive Knowledge Archive Network (CKAN) resource `4f27eea5-7038-4f73-9740-e3e4ad47c26a`; [NESO Open Data Licence](https://www.neso.energy/data-portal/neso-open-licence) | Hourly | From 2021-09-20 |
| **Carbon intensity** (national) | [Carbon Intensity API](https://carbonintensity.org.uk/); CC BY 4.0 (unverified) | Half-hourly | A request for 2017-09-30 returned data |
| **Bid and offer prices** (Elexon datasets `BOD`, `BOALF`, `BOAV`, `EBOCF`) | [Elexon Insights API](https://bmrs.elexon.co.uk/); Elexon's open licence for BMRS data | Per balancing unit, per half-hour | Not checked |

**The imbalance system price, the market index price, and the N2EX day-ahead auction price are
different products, and none stands in for another.** The imbalance system price is what a party
pays or receives for the energy it was short or long in a settlement period, and is known only after
the period ends. The market index price is the half-hourly price of short-term trading on the EPEX
exchange, published with the traded volume beside it, so a battery can read it during the delivery
day. The `APXMIDP` price is not the day-ahead auction price. The N2EX day-ahead auction price is an
hourly price fixed the morning before delivery. Elexon's `MID` dataset has a second provider,
`N2EXMIDP`, which returned a price and a volume of exactly 0 in every period the project checked, so
it carries no information.

**Bid-offer data shows what batteries and other flexible units were paid to do, and is large.**
`BOD` holds the prices each balancing unit has submitted for being turned up or down, which the
project estimated at about 1.4 GB a year for all units. `BOALF` holds the instructions the system
operator accepted, with the time and power level. `BOAV` and `EBOCF` hold the accepted volumes and
the indicative cash flows, and the price an accepted bid or offer received is the cash flow divided
by the volume. Elexon marks these endpoints as quota-limited and publishes no rate limit. The
endpoints worked without a key.

**Ember's price data needs an API key and is monthly, so it is not recommended.**

### What the data looks like

**The imbalance system price, market index price, and N2EX day-ahead tables share one timestamp
convention: `time` is the UTC start of the period.** The imbalance system price and the market index
also carry Elexon's `settlement_date` and `settlement_period`. Settlement period 1 starts at 00:00
UK local time, so a settlement date has 48 periods, 46 on the day the clocks go forward, and 50 on
the day they go back. The N2EX day-ahead table has 24 rows on both clock-change days, which says
that NESO's `Delivery Period (GMT)` column is on a UTC grid. The hourly N2EX series also correlates
best with the hourly mean of the market index at a lag of 0 hours, with the N2EX series on the same
grid as the index and neither leading the other (0.90, against 0.87 at -1 hour and 0.86 at +1 hour,
over the 395 days downloaded). **These tables mark the start of each period, whereas the NGED power
feed marks the end of each half-hour,** so a join on `time` without shifting one side by 30 minutes
is off by one period. The rows below are three consecutive half-hours from the evening of 23 June
2026, in the project's downloaded copy.

Imbalance system price (the columns of `DISEBSP` the project keeps; the table also holds
`price_derivation_code`):

| `time` (UTC) | `settlement_date` | `settlement_period` | `system_sell_price_gbp_per_mwh` | `system_buy_price_gbp_per_mwh` | `net_imbalance_volume_mwh` | `total_accepted_offer_volume_mwh` | `total_accepted_bid_volume_mwh` |
|---|---|---|---|---|---|---|---|
| 2026-06-23 18:30 | 2026-06-23 | 40 | 770.05 | 770.05 | 597.1 | 1131.5 | -1006.0 |
| 2026-06-23 19:00 | 2026-06-23 | 41 | 789.04 | 789.04 | 617.4 | 1233.5 | -1082.8 |
| 2026-06-23 19:30 | 2026-06-23 | 42 | 800.00 | 800.00 | 748.3 | 1223.5 | -943.8 |

Market index price (`APXMIDP`), the same half-hours:

| `time` (UTC) | `settlement_date` | `settlement_period` | `price_gbp_per_mwh` | `volume_mwh` |
|---|---|---|---|---|
| 2026-06-23 18:30 | 2026-06-23 | 40 | 530.66 | 4313.35 |
| 2026-06-23 19:00 | 2026-06-23 | 41 | 545.21 | 5189.20 |
| 2026-06-23 19:30 | 2026-06-23 | 42 | 560.81 | 4971.75 |

N2EX day-ahead auction price, the two hours that cover them:

| `time` (UTC) | `delivery_date` | `price_gbp_per_mwh` |
|---|---|---|
| 2026-06-23 18:00 | 2026-06-23 | 248.39 |
| 2026-06-23 19:00 | 2026-06-23 | 250.08 |

Carbon intensity, the same half-hours (`time_end` is the UTC end of the half-hour; `index` is the
API's band):

| `time` (UTC) | `forecast_gco2_per_kwh` | `actual_gco2_per_kwh` | `index` |
|---|---|---|---|
| 2026-06-23 18:30 | 205 | 218 | high |
| 2026-06-23 19:00 | 209 | 218 | high |
| 2026-06-23 19:30 | 215 | 224 | high |

**Units are pounds sterling per megawatt-hour for prices and megawatt-hours for volumes.** The net
imbalance volume is positive when the system was short of energy. The accepted bid volume is
negative. The sell price and the buy price were equal in every one of the 18,960 periods downloaded,
because GB has used a single imbalance price since November 2015. The NESO source column is labelled
"Price (GBP)" with no energy unit, and the project reads its prices as pounds per megawatt-hour on
the strength of the column's description ("Auction price for the delivery period, expressed in the
requested currency (GBP)") and the series' agreement with the market index.

**Prices can be negative, and rare spikes dominate the ranges.** Across the 395 days downloaded, the
system price ranged from -97.92 to 800.00 pounds per megawatt-hour, and only one period, 19:30 UTC
on 23 June 2026, reached 800.00. The market index ranged from -102.92 to 560.81 and the N2EX
day-ahead price from -42.05 to 420.10. The N2EX table has 17 hours priced at exactly 0.0, which are
real prices and not missing values. The half-hourly system price and the market index had a
correlation of 0.85, and the system price differed from the index by 19.2 pounds per megawatt-hour
on average (mean absolute difference).

**Contains BMRS data © Elexon Limited copyright and database right 2026 (imbalance system price and
market index price); N2EX day-ahead price from the NESO data portal.**

![Line chart of three GB electricity prices over 22 to 28 June 2026: the hourly N2EX day-ahead
price, the half-hourly market index price, and the half-hourly imbalance system price. All three
rise and fall together, the system price reaching 800 pounds per megawatt-hour on 23 June and the
day-ahead price peaking at 420 on 24 June.](assets/gb_prices_one_week.svg)

**The day-ahead price follows a daily shape that a battery can plan around.** Over the 395 days
downloaded, the mean N2EX price was highest in the evening, from 17:00 to 19:00 UTC, peaking at the
hour starting 18:00 UTC (about 128 pounds per megawatt-hour, against about 127 at 17:00) and lowest
around the middle of the day, at 12:00 and 13:00 UTC (about 75). The interquartile range at 18:00
UTC runs from about 101 to 147 pounds per megawatt-hour. A second, lower rise appears at 06:00 and
07:00 UTC.

![Step chart of the mean N2EX day-ahead price by hour of day in UTC, with a shaded interquartile
range. The mean is about 75 pounds per megawatt-hour at midday, rises to about 128 at 18:00 UTC, and
falls back to about 88 by 23:00 UTC.](assets/gb_day_ahead_daily_profile.svg)

### Which series are known before delivery

**Elexon's feeds publish within minutes to hours of the moment a value is fixed, but a series only
helps a forecast if it is known before the power it influences.** NESO's copy of the N2EX prices
lagged the auction by more than a day in the check described below. The first group below is known
ahead of delivery. The second group is a price outcome that the project can use only as history.

Known ahead of delivery:

- **The N2EX day-ahead auction price** for the whole next day is fixed the morning before delivery.
  Nord Pool states that gate closure is at 09:50 GMT and that results are published at the latest at
  10:00 GMT ([Nord
  Pool](https://support.nordpoolgroup.com/support/solutions/articles/8000088463-about-the-n2ex-day-ahead-auction)).
  Whether "GMT" means UTC all year or UK clock time is not stated, and the project has not checked
  it. The timing of NESO's copy is in the table below.
- **Final physical notifications** (Elexon dataset `PN`), the stable export and import limits
  (`SEL`, `SIL`), and the maximum export limit (`MELS`) and maximum import limit (`MILS`) are what
  each balancing unit says it will do and can do. For a battery, the final physical notification is
  the unit's own plan for charging and discharging. Each unit's data is published only for periods
  whose gate has closed.
- **Submitted bid and offer prices** (`BOD`), per unit and per half-hour, show at what price each
  unit has said it will be turned up or down. They share the gate-closure horizon above.
- **National demand and wind forecasts**: Elexon's national demand forecast (`NDF`) and transmission
  system demand forecast (`TSDF`), the `WINDFOR` wind forecast, and the day-ahead wind and solar
  generation forecast (`DGWS`, from Elexon's `/datasets/DGWS` endpoint). These describe the conditions that drive the price, and not the price itself.
- **The carbon intensity forecast** from the Carbon Intensity API is documented as reaching up to 2
  days ahead, and a request for the 48-hour window returned 97 half-hourly values.

Known only after the fact, so history for a model and not an input to a forecast of the same period:

- **The imbalance system price** for a period appears about 19 minutes after the period ends and is
  written again about 24 hours later. The project has not established what the second write changes.
- **Accepted bid and offer volumes and cash flows** (`BOALF`, `BOAV`, `EBOCF`). `BOALF` shows an
  instruction soon after the system operator issues it, so a unit's current instruction is visible
  during the period it applies to. `BOAV` and `EBOCF` are settlement outputs.
- **The market index price** appears during the half-hour the row describes. The dataset has no
  publish-time field, so the delay is not measured.

**Whether a published price or notification predicts a battery's behaviour has not been tested
here.** Which series improve a forecast of a substation's power flow is an open question, and the
[Metrics and leaderboard](metrics-and-leaderboard.md) page owns how any such comparison would be
scored.

### When each series is published

**Publication times are stated in UTC unless a row says otherwise.** In UK summer time (BST), UTC is
one hour behind local clock time.

| Series | Time observed or stated | Status |
|---|---|---|
| N2EX day-ahead auction | Gate closure 09:50 GMT, results at the latest at 10:00 GMT, the day before delivery (Nord Pool's documentation) | Stated by Nord Pool; not checked against a live result |
| NESO's copy of the N2EX prices | The data portal resource's `metadata_modified` was 2026-10-07 11:30 UTC. At 12:17 UTC on 2026-10-08, the newest row was the delivery hour 21:00 to 22:00 UTC on 2026-10-08, although the auction for 2026-10-09 had closed by 10:00 UTC | Observed once; the update schedule is unverified |
| Imbalance system price | About 19 minutes after the settlement period ends (the `createdDateTime` of every period on 2026-10-08, to the nearest minute), then written again about 24 hours and 15 minutes after the period ends | Observed on 2026-10-08 and on seven earlier days (2025-10-26, 2026-01-15, 2026-08-01, 2026-09-20, 2026-10-03, 2026-10-06, 2026-10-07) |
| Later settlement runs | Elexon's settlement runs follow at 5, 16, about 36 to 40, about 81 to 85, about 151 to 155, and about 289 to 293 working days after the settlement day. Which runs change the imbalance price is not established here | Unverified: from a web search of Elexon pages |
| Market index price | During the half-hour the row describes: at 12:13 UTC on 2026-10-08 the row for the half-hour starting 12:00 UTC was present and the row for 12:30 UTC was not | Observed once |
| `PN`, `BOD`, `MELS`, `BOALF` | Within minutes; limited to periods whose gate has closed. For one battery unit queried at about 12:17 UTC on 2026-10-08, `PN`, `BOD`, and `MELS` reached 13:30 UTC and no further; a `BOALF` acceptance with `acceptanceTime` 12:00 UTC was present at 12:12 UTC | Observed once, for one unit |
| `NDF`, `TSDF` | `publishTime` 11:48 UTC on 2026-10-08, with the first forecast period at 12:00 UTC | Observed once; the cadence is unverified |
| `WINDFOR` | Latest `publishTime` 10:30 UTC on 2026-10-08, with an earlier publication at 03:30 UTC the same day; forecast periods reached 2026-10-10 | Observed once; the cadence is unverified |
| Day-ahead wind and solar forecast | `publishTime` 16:45 UTC on 2026-10-07, for 8 and 9 October | Observed once; the cadence is unverified |
| Carbon intensity forecast | A 48-hour forecast was returned at 12:13 UTC on 2026-10-08 | The update schedule is unverified |

**The EPEX GB day-ahead auctions are a separate series that the project has not downloaded, and they
are not free.** A web search of EPEX SPOT's published trading brochures (for example [the October
2022 brochure](https://www.epexspot.com/sites/default/files/2023-01/22-10-25_TradingBrochure.pdf))
found statements of an hourly GB day-ahead auction at 09:20 UK time and a half-hourly auction at
15:30 UK time, with results soon after. The project could not confirm these times on EPEX SPOT's
trading-products page, which refused an automated request, so treat them as unverified. Elexon's
`MID` market index (above) is the EPEX series the project has.
