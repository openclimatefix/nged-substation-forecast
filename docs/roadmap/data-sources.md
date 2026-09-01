# Data sources

The inputs to the forecasting system: NGED's power-flow data, supporting NGED files, and weather
data.

> **Status legend** — ✅ Ingested today · 🚧 Planned ingestion · 🔬 Research. The ECMWF ENS NWP and the
> NGED time-series JSON / metadata are ✅ ingested; the supporting NGED files and the extra weather
> datasets are 🚧 planned (needed for switching-event detection and capacity estimation). See the
> [roadmap index](index.md) for status conventions.

---

## Data from NGED

### Provided on NGED's AWS S3 bucket (live, updated every 6 hours)

| Source | Status | Description |
|---|---|---|
| **Time-series JSON files** | ✅ | Half-hourly power flow + metadata per substation / customer meter in the trial area. Ingested by OCF to produce operational forecasts. |
| **Curtailment (ANM set points)** | 🚧 | NGED-imposed curtailment. Crucial for distinguishing deliberate ANM ramp-downs from genuine faults / capacity loss. |

### Provided on SharePoint (mostly static reference / historical)

| File | Status | Description | Known issues |
|---|---|---|---|
| Historical time-series JSON | ✅ | Historical outputs of substations and customer meters. | See [data quality](#data-quality-availability). |
| **Monitor Direction.csv** | 🚧 | Metadata for all substations: meter (analogue) type and power-flow direction. | Lincoln Farm Solar Park (ID 30) has a different substation number vs. its metadata; other sources agree, so low risk. |
| **Primary Substation Interconnections.csv** | 🚧 | List of possible connections between primary substations (not all are in the trial area). | All trial-area substations have ≥ 1 connection; topology appears complete. |
| **Substations.csv** | 🚧 | For each substation/BSP/GSP: which BSP & GSP it connects to (names + IDs). | All trial-area substations valid. |
| **Switching Logs.xlsx** | 🚧 | History of every normally-open switching point between primaries, labelled by time-series ID. Primaries outside the trial area are labelled "Unknown". | **Extremely valuable** as the gold-standard *test set* for [switching-event detection](switching-events.md) — lets us validate the unsupervised method on the trial area (labels do **not** exist at scale). Some edges "collapse" into `[substation ID] – unknown`. Two edges present in Interconnections.csv are missing: 900016 (ID 10) ↔ 900019 (ID 13), and 900022 (ID 16) ↔ unknown (910026). Logs go back to ≥ 2019. |
| **MPAN to Substation Number.csv** | 🚧 | Associates each Embedded Capacity Register (ECR) generator to the substation it connects to. | All trial-area generators present, each with two MPANs (import + export). Three primaries appear with one MPAN each — looks like a data error. |
| **Peak Loads.xlsx** | 🚧 | Manually selected peak demand per trial-area substation, from 2024/25 (most recent survey). | Covers all 16 trial primaries. 12 have 2024/25 datapoints exceeding the reported peak; even at the 99th quantile, 3 (IDs 8, 13, 25: Horncastle, Wrangle T2, Warth Lane Skegness) show 2–4× the reported peak, while Stickney (ID 14) reports > 14 MVA peak but maxes at 6.8 MVA historically. Given the discrepancies, we use the **99th quantile of observed power** as the substation "capacity" proxy, at least initially. |

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
- **Unreliable meters**: some arrived labelled "analogue not working" or "analogue suspect".
- **False zeros**: substation data is prone to one-off drops to zero (telemetry faults), visible as
  an excess of exact zeros in the distribution vs. near-zero values.
- **Not-on assets**: Boston Biomass Generation (ID 19) has been pure noise since ~mid-2024 (not
  operational) — motivating the [building-blocks](forecast-building-blocks.md) delivery approach.
- **MVA / reverse flow**: primary data is disaggregated from metered generation where possible, but
  not always (e.g. Marsh Lane, ID 26, has two non-working solar meters). Combined with MVA metering
  (which reports absolute value), midday solar export "bounces" off zero and looks like extra load.
  See also the [MVA discussion in Net-demand disaggregation](disaggregation.md#apparent-power-mva-metering).

These data oddities are detected and reported back to NGED as warnings (see
[delivery tables, Table 2](delivery-tables.md#table-2-power_forecast_warnings)).

---

## NGED's network (context)

As of May 2026, NGED's full network (the v2 target scope) consists of:

- **1,161 primary substations** (33/11 kV & 66/11 kV)
- **271 bulk supply points (BSPs)** (132/33 kV & 132/66 kV)
- **52 grid supply points (GSPs)** (400/132 kV & 275/132 kV)
- **~1,500 generators** (industrial customer generators, not domestic):
    - 558 connect directly to GSP/BSP busbars at 33 kV or 132 kV (modelled by NGED as
      generation-only "substations"; mostly have telemetry; curtailable via ANM; comprising 329 solar,
      63 wind, 166 other).
    - ~1,000 are on the 11 kV network downstream of primaries; some metered, some not.
    - Power flow from metered generators connected to primaries is **already subtracted** from the
      primaries' power flow ("Disaggregated Demand").

The **Embedded Capacity Register** behind these figures has limits worth naming, because both
[capacity estimation](capacity-estimation.md) and [disaggregation](disaggregation.md) build on the register. The register records generation of 50 kW and above, and the capacity recorded is the export limit a site's
connection agreement permits rather than what the site can actually generate, which is why both
capacity candidates estimate *effective* capacity rather than reading nameplate capacity off the
register. Below 50 kW the register is silent, and most of the panels sit below 50 kW: of the 22,560 MW of
solar photovoltaic capacity installed in GB by the end of July 2026, 8,503 MW sits in arrays smaller
than 50 kW. Recovering solar on that scale, with no register to check the answer against, is what
[disaggregation](disaggregation.md) has to do. The figures come from the
[energy-forecasting
review](../background/energy-forecasting-review.md#8-disaggregating-unmetered-solar-and-wind-from-a-substations-net-flow),
which cites the Department for Energy Security and Net Zero's [solar deployment
statistics](https://www.gov.uk/government/statistics/solar-photovoltaics-deployment).

The **v1 trial area** is 32 of these time series — see the [roadmap index](index.md) for the
breakdown.

---

## Weather data

Issues: [#142](https://github.com/openclimatefix/nged-substation-forecast/issues/142) (CM SAF),
[#143](https://github.com/openclimatefix/nged-substation-forecast/issues/143) (reanalysis ingestion — ERA5)

| Source | Status | Description |
|---|---|---|
| **ECMWF ENS** (Dynamical.org) | ✅ | Main NWP source: 51-member ensemble, distributed as live-updating Zarrs. OCF converts gridded NWP to tabular via the H3 spatial index and stores as Delta Lake, quantised to 12-bit `Int16` with zstd compression (~40 GB/year for all of GB; ~1 minute to download+convert one day). **The archive currently only extends back to 2024-04-01**; Dynamical.org are back-filling the operational archive from MARS to 2016-03-08 (51 members, 0.25°, 00Z inits only), but at ~0.8 TB/day against ~446 TB remaining the estimate is **~November 2027** — after v1.0, which is why we [extend the training history with ERA5](training-history.md) instead. Radiation: **global short-wave (GHI) only, no direct component** — [DP forecasting of PV](disaggregation.md) (v2) therefore needs a differentiable GHI → DNI/DHI decomposition model, or `fdir` added to the upstream dataset. |
| **ERA5** (ECMWF global reanalysis) — *the project's reanalysis* | 🚧 (v0.5) | The **single reanalysis** we ingest, serving both **pre-training** and near-real-time **capacity estimation**. Ingest **2020 to present — the 2020–2023 gap *and* the 2024+ ENS overlap**; the overlap is not optional (it decorrelates era-specific feature values from date, and supplies the paired `ENS − ERA5` residual statistics) — see [Extending the training history](training-history.md). Shares the ECMWF **IFS lineage** with the ENS forecasts, so systematic biases largely cancel when the two are combined (pre-training domain shift; climatology anomalies). Covers **1940 to the present** — far enough back to pre-train on the long power histories that predate the ENS archive (2024-04-01) — and its **ERA5T** near-real-time stream lands **~5 days behind real time** (final ERA5 overwrites it ~2–3 months later after QC), fast enough for capacity estimation. 31 km resolution: coarser than CERRA, but weather anomalies are synoptic-scale and the high-resolution *solar* irradiance comes from CM SAF regardless. Radiation: global plus **direct** (`fdir`), giving the beam/diffuse split. Access options, best first: the **CDS's own ARCO Zarr store** — since [2026-06-30](https://forum.ecmwf.int/t/access-to-our-arco-era5-data-lake-zarr-store/15123) Copernicus publish ERA5 single levels (and ERA5-Land) as analysis-ready Zarr, opened straight from `xarray` with a CDS API key and offered **geo-chunked** (long periods over a small area — our access pattern) as well as time-chunked. Two things to check before committing to it, because neither is documented: whether its **subset** of surface and wave variables carries the radiation we need (`fdir` above all — the direct component is why we prefer ERA5 to the ENS feed here; further parameters are added on request), and whether it serves **ERA5T** or only final ERA5, since near-real-time is half of why we chose ERA5 over CERRA. It is also explicitly a **beta** service, and the CDS reserve the right to withdraw access to protect Data Store performance — so treat it as the leading candidate, not a settled decision. Fallbacks: **ARCO-ERA5** (public [Google Cloud Zarr](https://github.com/google-research/arco-era5), carrying ERA5T at ~1 week — verify freshness via its `valid_time_stop_era5t` / `last_updated` metadata), **[Earthmover Icechunk-ERA5](https://registry.opendata.aws/earthmover-era5/)** (AWS, daily-updating but paid; the free tier lags 3 months), or the plain **CDS API** (~5 days, but not analysis-ready). A precomputed *mean* climatology (for the [weather-abnormality feature](xgboost-improvements.md#weather-abnormality-climatology-z-score-features)) is available from **WeatherBench2** (`gs://weatherbench2/datasets/era5-hourly-climatology/`). |
| **CERRA** (Copernicus regional reanalysis for Europe) | 🔬 (deprioritised) | Higher-resolution (5.5 km) European reanalysis. Per the [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels), it now runs from **September 1984 to the present** — monthly updates, but **~3.5 months behind real time**. **Superseded by ERA5** for the active plan: that ~3.5-month latency rules it out for near-real-time capacity estimation, ERA5 reaches further back for pre-training, and we prefer to ingest a single reanalysis. Kept here because its 5.5 km resolution could still earn a place for fine-scale work (e.g. wind over complex terrain) if that ever proves decisive. Radiation: global plus time-integrated **direct** short-wave (diffuse by subtraction); accumulated fluxes from 3-hourly forecast cycles, so temporally coarser than SARAH-3. |
| **CM SAF** (Satellite Application Facility on Climate Monitoring) | 🚧 (v0.7) | High-resolution satellite-derived irradiance, used to estimate **solar PV** capacity. Used **offline only** — capacity estimation runs over history, and the production serving path takes no dependency on it. SARAH-3 provides global (SIS), **direct (SID) and direct-normal (DNI)** irradiance at 0.05° / 30-minute resolution from 1983 (diffuse = SIS − SID) — the beam/diffuse split the [DP solar model](../techniques/differentiable-physics.md#the-core-building-block-differentiablesolarplant) needs, at a resolution matching the half-hourly metering. One operational fact to confirm before v0.7 leans on SARAH-3: what the near-real-time **Interim Climate Data Record** (ICDR) actually delivers. [Pfeifroth et al. (2024)](https://doi.org/10.5194/essd-16-5243-2024) separate the committed latency from the typical latency — "The committed timeliness of the SARAH-3 ICDR is 5 d, but usually the SARAH-3 ICDR comes with a timeliness of only 2 d" — so check the Web User Interface for what the record is delivering now. |
| **CAMS solar radiation** (Copernicus Atmosphere Monitoring Service) | 🔬 (offline only, uncertain) | Worth *considering* as an alternative or a companion to SARAH-3, on the same offline-only footing. Carries **global, direct, diffuse, and direct-normal** irradiance under both clear sky and observed cloud, at time steps of 1 minute, 15 minutes, 1 hour, 1 day, or 1 month, from 2004-02, under CC-BY. The **sub-hourly** steps are the draw: they are what the [dynamic thermal model](../techniques/differentiable-physics.md) needs and what SARAH-3's 30 minutes cannot supply. The catch is delivery — the [Atmosphere Data Store catalogue](https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries)' request form takes **one latitude and longitude per request**, so CAMS arrives as point time series rather than the grid every other source here supplies and the H3 pipeline expects. Annoying rather than disqualifying: one request covers a location over a date range, so the count scales with sites and history chunks rather than with days, and the [v1 trial area](../index.md#scope) needs at most 32 requests — 6 for the solar farms alone — which is what makes a head-to-head test against SARAH-3 cheap to run before anything larger is committed to. Freshness is roughly one day, but that matters little for offline use. The all-sky product is limited to the Meteosat Second Generation and Himawari fields of view; Great Britain sits inside, with the same low-winter-sun degradation SARAH-3 documents. |
| **ICON-EU** (Dynamical.org) | 🔬 (v2, uncertain) | Possible additional NWP source to test whether it improves skill over ECMWF ENS. Starts early 2026, so it can't enter the canonical CV folds directly — assessed via ad-hoc ablation first. |
| **AIFS-ENS** (ECMWF) | 🔬 (v2, uncertain) | ECMWF's machine-learned ensemble, now operational with the same 51 members, 6-hourly steps and 15-day horizon as the physics ensemble, and more accurate than it on the majority of variables and lead times ([Lang et al. (2026)](https://doi.org/10.1038/s44387-026-00073-7)). Whether that translates into a better substation-load forecast is an open question, and the swap is cheap to test because the two ensembles share a shape. Same folds problem as ICON-EU: the archive starts mid-2025, so it is an ad-hoc ablation before it is a canonical source. |

**ERA6 is a future upgrade, not a current option.** ECMWF began ERA6 production in March 2026, but
the phased release runs from late 2027 (first 20 years) into 2028, so it is out of scope for the
near-term milestones. When it lands it should drop in cleanly — same IFS family, a similar
ERA5T-style near-real-time fast track — and at ~14 km (2× finer than ERA5) it would close most of
the resolution gap with CERRA that motivates keeping CERRA on the list at all.
