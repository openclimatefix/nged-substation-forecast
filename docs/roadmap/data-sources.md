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
  See also the [MVA discussion in Differentiable Physics](differentiable-physics.md).

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

The **v1 trial area** is 32 of these time series — see the [roadmap index](index.md) for the
breakdown.

---

## Weather data

| Source | Status | Description |
|---|---|---|
| **ECMWF ENS** (Dynamical.org) | ✅ | Main NWP source: 51-member ensemble, distributed as live-updating Zarrs. OCF converts gridded NWP to tabular via the H3 spatial index and stores as Delta Lake, quantised to 12-bit `Int16` with zstd compression (~40 GB/year for all of GB; ~1 minute to download+convert one day). **The archive currently only extends back to 2024-04-01**; Dynamical.org are actively back-filling earlier years, but this will take a while. |
| **CERRA** (Copernicus regional reanalysis for Europe) | 🚧 (v0.6) | High-resolution historical weather re-analysis. Useful for **pre-training** forecasting models and for estimating historical generator capacity. Covers 2020–2024, before the ECMWF dataset starts (April 2024). |
| **CM SAF** (Satellite Application Facility on Climate Monitoring) | 🚧 (v0.6) | High-resolution satellite-derived irradiance, used to estimate **solar PV** capacity. |
| **ICON-EU** (Dynamical.org) | 🔬 (v2, uncertain) | Possible additional NWP source to test whether it improves skill over ECMWF ENS. Starts early 2026, so it can't enter the canonical CV folds directly — assessed via ad-hoc ablation first. |
