# Data sources

The inputs to the forecasting system: NGED's power-flow data, supporting NGED files, and weather
data.

> **Status legend** — ✅ Ingested today · 🚧 Planned ingestion · 🔬 Research. The ECMWF ENS NWP and the
> NGED time-series JSON / metadata are ✅ ingested; the supporting NGED files and the extra weather
> datasets are 🚧 planned (needed for switching-event detection and capacity estimation), and
> NGED's published electricity-network model is 🔬 research. See the
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
results](../results/beam-diffuse-split.md#one-site-is-curtailed-and-it-is-the-noisiest-of-the-six).

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

| Source | Status | Description |
|---|---|---|
| **ECMWF ENS** (Dynamical.org) | ✅ | Main NWP source: 51-member ensemble, distributed as live-updating Zarrs. OCF converts gridded NWP to tabular via the H3 spatial index and stores as Delta Lake, stored as `Float32` rounded to a 13-bit significand, with zstd compression (~40 GB/year for all of GB; ~1 minute to download+convert one day). **The archive currently only extends back to 2024-04-01**; Dynamical.org are back-filling the operational archive from MARS to 2016-03-08 (51 members, 0.25°, 00Z inits only), but at ~0.8 TB/day against ~446 TB remaining the estimate is **~November 2027** — after v1.0, which is why we [extend the training history with ERA5](training-history.md) instead. Radiation: no direct component, which is what forces [DP forecasting of PV](disaggregation.md) (v2) to find the beam/diffuse split elsewhere — see [which sources carry which irradiance components](#which-sources-carry-which-irradiance-components). |
| **ERA5** (ECMWF global reanalysis) — *the project's reanalysis* | 🚧 (v0.5) | The **single reanalysis** we ingest, serving both **pre-training** and near-real-time **capacity estimation**. Covers 1940 to the present — far enough back to pre-train on the long power histories that predate the ENS archive (2024-04-01). Its 31 km resolution is coarser than CERRA, which is acceptable because weather anomalies are synoptic-scale and the high-resolution *solar* irradiance comes from CAMS regardless. Carries the [beam/diffuse split](#which-sources-carry-which-irradiance-components), which the live ENS feed does not. Its **ERA5T** near-real-time stream lands ~5 days behind real time, and final ERA5 overwrites it ~2–3 months later after quality control. Shares the ECMWF **IFS lineage** with the ENS forecasts, so systematic biases largely cancel when the two are combined. Ingest **2020 to present**, including the 2024+ ENS overlap, which is not optional — see [Extending the training history](training-history.md). [Which access route](#era5-which-access-route) is still open. |
| **CERRA** (Copernicus regional reanalysis for Europe) | 🔬 (deprioritised) | Higher-resolution (5.5 km) European reanalysis. Per the [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels), it now runs from **September 1984 to the present** — monthly updates, but **~3.5 months behind real time**. **Superseded by ERA5** for the active plan: that ~3.5-month latency rules it out for near-real-time capacity estimation, ERA5 reaches further back for pre-training, and we prefer to ingest a single reanalysis. Kept here because its 5.5 km resolution could still earn a place for fine-scale work (e.g. wind over complex terrain) if that ever proves decisive. Its [direct short-wave](#which-sources-carry-which-irradiance-components) is time-integrated from 3-hourly forecast cycles, so temporally coarser than SARAH-3. |
| **CM SAF** (Satellite Application Facility on Climate Monitoring) | 🔬 (v2 comparison) | SARAH-3 carries [every irradiance component](#which-sources-carry-which-irradiance-components) on a 0.05° grid at 30 minutes from 1983. Two reasons SARAH-3 is not the first ingest. Its climate data record ends 2020-12-31 and the Interim Climate Data Record extends that record, putting a version seam inside the 2019-onward history we train on. And its 30-minute values are **instantaneous snapshots**, whereas CAMS accumulates over the step, which is what a period-ending meter reading measures — under broken cloud an instantaneous sample and a 30-minute mean can differ a lot. Its gridded delivery would suit the H3 pipeline better than CAMS point requests, and comparing the two resolutions is not straightforward, because the CAMS point service interpolates to the requested location rather than publishing a grid. Latency is 2–5 days ([Pfeifroth et al. (2024)](https://doi.org/10.5194/essd-16-5243-2024)), immaterial offline. Worth a genuine head-to-head against CAMS in v2 — see [Correcting satellite irradiance over Great Britain](disaggregation.md#correcting-satellite-irradiance-over-great-britain). |
| **CAMS solar radiation** (Copernicus Atmosphere Monitoring Service) | 🚧 (v0.7) | The satellite-derived irradiance we ingest, used to estimate **solar PV** capacity. Used **offline only** — capacity estimation runs over history, and the production serving path takes no dependency on it. The CAMS Radiation Service carries [every irradiance component](#which-sources-carry-which-irradiance-components), under both clear sky and observed cloud, from 2004-02, under CC-BY-4.0, at steps of 1 minute, 15 minutes, 1 hour, 1 day, or 1 month — the beam/diffuse split the [DP solar model](../techniques/differentiable-physics.md#the-core-building-block-differentiablesolarplant) needs. Cloud information comes from Meteosat Second Generation; aerosol, ozone, and water vapour come from the CAMS global forecasting system, so aerosol optical depth is a 3-hourly analysis rather than SARAH-3's monthly climatology. Values are interpolated to the requested location rather than served on a grid. Chosen over SARAH-3 on **delivery and record continuity, not on measured accuracy over Great Britain** — see [CAMS: use the point API, not the gridded product](#cams-use-the-point-api-not-the-gridded-product) for the route and its traps, and [Correcting satellite irradiance over Great Britain](disaggregation.md#correcting-satellite-irradiance-over-great-britain) for what is known about this source's error and what v2 might do about it. |
| **ICON-EU** (Dynamical.org) | 🔬 (v0.9, uncertain) | Possible additional NWP source to test whether it improves skill over ECMWF ENS: a deterministic run from DWD, Germany's national weather service, on a ~6.5 km grid, 4 runs a day out to 5 days. Already carries the [beam/diffuse split](#which-sources-carry-which-irradiance-components) [the PV forward model](disaggregation.md#the-forward-model) needs, and holds the earliest roadmap slot of any source that does. Starts early 2026, so it can't enter the canonical CV folds directly — assessed via ad-hoc ablation first. |
| **AIFS-ENS** (ECMWF) | 🔬 (v2.1, uncertain) | ECMWF's machine-learned ensemble, now operational with the same 51 members and 15-day horizon as the physics ensemble, and more accurate than it on the majority of variables and lead times ([Lang et al. (2026)](https://doi.org/10.1038/s44387-026-00073-7)). Whether that translates into a better substation-load forecast is an open question. AIFS-ENS member *n* starts from the [same initial conditions](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v1) as ECMWF ENS member *n*, so the two can be fed [side by side](xgboost-improvements.md#several-nwp-sources-as-features-v21) rather than swapped. Unlike the physics ensemble, AIFS has [no direct-beam field](#which-sources-carry-which-irradiance-components) to ask for at all, in either the ensemble or the deterministic AIFS Single: on the 2026-09-19 00Z run, AIFS-ENS open data carries 29 parameters and AIFS Single 30, and `ssrd` and `strd` are the only radiation fields in either. **Both AIFS streams arrive on 6-hourly steps across the whole 15-day horizon — 61 steps, against the physics ensemble's 85** — where the physics ensemble is [3-hourly out to 144 hours](../architecture/nwp-variable-conventions.md#the-forecast-step-grid) and only then drops to 6-hourly. Twice the step width over days 0 to 6 costs most on irradiance, whose diurnal cycle varies faster than any other field we use: the [reconstructed solar day](../architecture/nwp-variable-conventions.md#period-ending-variables-are-interpolated-as-though-they-were-instantaneous) lags the true one by half a step width, so 3 hours at AIFS's steps against 1.5 hours at the physics ensemble's, and the modelled clear-sky peak falls from 816 W m⁻² to 756 at 3-hourly steps and 590 at 6-hourly. Same folds problem as ICON-EU: the archive starts mid-2025, so it is an ad-hoc ablation before it is a canonical source. |
| **WeatherNext 3** (Google DeepMind) | 🔬 (after v2, unlikely) | Google DeepMind's machine-learned 64-member ensemble, with single-level fields at 0.1° on hourly steps, running 15 days from the 00/06/12/18 UTC cycles and 48 hours from the hourly interim runs. 2 m temperature and dewpoint also arrive at 0.05°, from a neural-network output head trained directly on in-situ surface observations from airport stations, regional station networks, ships, and buoys ([Rasp et al. (2026)](https://arxiv.org/abs/2609.03582); [model specs](https://developers.google.com/weathernext/guides/models)). Carries the [beam/diffuse split](#which-sources-carry-which-irradiance-components) [the PV forward model](disaggregation.md#the-forward-model) needs. The paper reports a lower [continuous ranked probability score](../techniques/evaluation-metrics.md#crps-continuous-ranked-probability-score) than ECMWF ENS on surface solar radiation, scored on the 6-hour accumulations rather than the hourly fields. Every radiation evaluation in the paper scores against an ECMWF analysis, and none scores against a surface measurement, so any irradiance gain would have to be measured downstream on power. Two structural differences from AIFS-ENS: all 64 members start from one analysis, so the ensemble's spread comes from the model alone rather than from perturbed initial conditions, and no member pairs with an ECMWF ENS member. Data at least 1 hour old is CC-BY-4.0, real-time data falls under Google DeepMind's experimental terms, and access needs [a request form](https://developers.google.com/weathernext/guides/access-forecast). BigQuery and Earth Engine serve six summary statistics per variable (the mean, and the 10th, 25th, 50th, 75th, and 90th percentiles); the 64 members sit only in the Cloud Storage Zarr store. The folds problem is worse than ICON-EU's or AIFS-ENS's: the archive starts 2026-01-01, with 2024 and 2025 being backfilled. On current priorities we do not expect to reach this source inside the Network Innovation Allowance project. |
| **UKV and MOGREPS-UK** (Met Office, via AWS) | 🔬 (uncertain) | The Met Office's 2 km deterministic UK model and its 2.2 km UK ensemble. Both publish [every irradiance component](#which-sources-carry-which-irradiance-components) as its own field, so neither needs a subtraction to give [the PV forward model](disaggregation.md#the-forward-model) its beam/diffuse split, and both are free on AWS under British Crown copyright and CC BY-SA 4.0 ([UKV](https://registry.opendata.aws/met-office-uk-deterministic/), [MOGREPS-UK](https://registry.opendata.aws/met-office-uk-ensemble/)). Verified by listing the `met-office-atmospheric-model-data` and `met-office-uk-ensemble-model-data` buckets rather than from the documentation. Neither the horizons nor the archives suit the canonical folds — MOGREPS-UK is held as a 30-day rolling window — see [which feed carries a direct beam](#which-feed-carries-a-direct-beam-and-what-asking-for-one-would-cost). |

**ERA6 is a future upgrade, not a current option.** ECMWF began ERA6 production in March 2026, but
the phased release runs from late 2027 (first 20 years) into 2028, so it is out of scope for the
near-term milestones. When it lands it should drop in cleanly — same IFS family, a similar
ERA5T-style near-real-time fast track — and at ~14 km (2× finer than ERA5) it would close most of
the resolution gap with CERRA that motivates keeping CERRA on the list at all.

### Which sources carry which irradiance components

**The Met Office's UKV and MOGREPS-UK are the only forecast sources listed below that publish
global, direct, and diffuse short-wave as three separate fields in a free feed.** Every other free
forecast source makes the reader recover at least one component by arithmetic, and ECMWF's own
direct beam needs a licence. Both reanalyses and both satellite products carry a direct component,
where most of the forecast sources do not. That is why [capacity
estimation](capacity-estimation.md#irradiance-inputs) plans on having the beam/diffuse split, while
[forecasting](disaggregation.md#the-forward-model) has no route to the split today.

**Two rows below are not in the catalogue above**, because neither is a source we could ingest as
things stand: ECMWF's own ENS is the licensed feed behind the free one, and the Met Office's global
10 km model is the Met Office model Dynamical.org have on their tracker rather than one we have
asked for.

| Source | Kind | Global | Direct | Diffuse | Where it comes from |
|---|---|---|---|---|---|
| **ECMWF ENS** via Dynamical.org | Forecast | ✅ `ssrd` | ❌ | ❌ | Free ECMWF open data on AWS |
| **ECMWF ENS** from ECMWF | Forecast | ✅ `ssrd` | ✅ `fdir` | By subtraction | Licensed dissemination or a MARS subscription |
| **ECMWF AIFS**, both Single and ENS | Forecast | ✅ `ssrd` | ❌ | ❌ | Free ECMWF open data; no direct field exists to license |
| **ICON-EU** via Dynamical.org | Forecast | By addition | ✅ | ✅ | Free, already ingested by Dynamical.org |
| **UKV** (Met Office) | Forecast | ✅ | ✅ | ✅ | Free on AWS, CC BY-SA 4.0; all three components in every run back to the archive's start, 2024-09-19 |
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
not in the feed they read.

**`fdir` on ECMWF ENS is therefore not a route open to us.** Serving `fdir` would mean a licensed ECMWF
dissemination or a MARS subscription in place of, or alongside, the free bucket — a contract and a
recurring cost, not a storage decision. Dynamical.org build their catalogue from freely
redistributable data. Asking them to widen a variable list would be reasonable. Asking them to take
on a licensed feed is asking them to work outside that model, so a request for `fdir` is not worth
making. Dynamical.org already publish direct and diffuse short-wave for ICON-EU, which shows that a
missing variable was never the obstacle — the obstacle is which ECMWF feed the open bucket holds,
and which feed the bucket holds is ECMWF's decision rather than Dynamical.org's.

**The Met Office models are the request worth making, because their free feed already carries a
direct beam.** The one open Met Office request on Dynamical.org's issue tracker is for [the global
10 km deterministic model](https://github.com/dynamical-org/reformatters/issues/646), which publishes
global and direct short-wave and leaves diffuse to the same subtraction ECMWF would need. UKV and
MOGREPS-UK go further and publish diffuse as its own field, but neither appears on Dynamical.org's
tracker, so either would have to be asked for. None of the three Met Office models needs a new
licence, which is what makes them worth requesting at all where `fdir` on ENS is not.

**Every Met Office model brings a horizon problem, and two bring an archive problem.** The global
model reaches 168 hours, UKV reaches 120 hours on its 03 and 15 UTC runs and 54 hours on the rest,
and MOGREPS-UK reaches 126 hours. No Met Office model covers NGED's 14-day horizon. A Met Office
model would therefore sit alongside ECMWF ENS rather than replace the ECMWF feed, exactly as
ICON-EU would. MOGREPS-UK is held on AWS as a 30-day rolling window, which rules out backtesting
unless we archive the feed ourselves from the day we start.

**A backtest on the global 10 km model starts at its 2024-11-07 12 UTC run, not at the start of its
archive, because no earlier run carries global short-wave.** Listing every 6-hourly run from the
earliest on AWS, 2024-09-19 18 UTC, shows
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
steps the licensed catalogue carries out to 90 hours appear neither there nor anywhere else we
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

**`dsrp` is the more convenient field to ask ECMWF for, because `fdir` is a horizontal-plane flux and
`dsrp` is already direct normal irradiance.** The ENS catalogue linked above carries three direct
fields: `fdir` (paramId 228021, total-sky direct at the surface), `cdir` (paramId 228022, the
clear-sky equivalent), and `dsrp` (paramId 47, direct solar radiation into a plane facing the sun).
Deriving direct normal irradiance from `fdir` means dividing by the cosine of the solar zenith
angle, which is numerically unstable at low sun, and the sun is low over GB for much of the year.
`dsrp` needs no division, so a request naming only `fdir` leaves behind the field that needs none.

**`ssrd` − `fdir` is close to a diffuse pyranometer reading, but not equal to that reading.** [ECMWF's
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
whether its **subset** of surface and wave variables carries the radiation we need (`fdir` above
all, since the direct component is why we prefer ERA5 to the ENS feed here; further parameters are
added on request), and whether it serves **ERA5T** or only final ERA5, since near-real-time is half
of why we chose ERA5 over CERRA. The Zarr store is also explicitly a **beta** service, and the CDS
reserve the right to withdraw access to protect Data Store performance.

**Three fallbacks, in order.** [ARCO-ERA5](https://github.com/google-research/arco-era5) is a public
Google Cloud Zarr carrying ERA5T at about a week's lag — verify freshness via its
`valid_time_stop_era5t` / `last_updated` metadata. [Earthmover
Icechunk-ERA5](https://registry.opendata.aws/earthmover-era5/) is on AWS and updates daily, but is
paid; its free tier lags 3 months. The plain **CDS API** reaches about 5 days behind real time but
is not analysis-ready. Separately, a precomputed *mean* climatology for the [weather-abnormality
feature](xgboost-improvements.md#weather-abnormality-climatology-z-score-features) is available from
WeatherBench2 at `gs://weatherbench2/datasets/era5-hourly-climatology/`.

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

### What a CAMS-against-ERA5 comparison on the trial area's solar farms found

**The satellite retrieval beat the reanalysis by 4.1 points of mean absolute error on the six
metered solar farms, which is the largest effect anything in that experiment varied.** On the
126,784 hours both products cover, CAMS cut an XGBoost forecast's error from 10.12% to 6.07% of P99
output, and a fitted five-parameter physical model's from 10.37% to 6.73%. The largest contrast
from changing which beam/diffuse split the model saw was 0.124 points, and changing the model
family moved about 0.9. Which product feeds the model dominates both. The write-up is [Does a
weather product's beam/diffuse split help a PV forecast?](../results/beam-diffuse-split.md); the
code sits in a pull request kept for reference rather than merged,
[#785](https://github.com/openclimatefix/nged-substation-forecast/pull/785).

**That gap is what a 5 km cloud field at the meter's own coordinates delivers over a 31 km field
averaged across a cell the meter may sit 13 km from.** Both measurements are of these two products
on this fleet, and neither has been shown to hold for every product at those resolutions.

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
