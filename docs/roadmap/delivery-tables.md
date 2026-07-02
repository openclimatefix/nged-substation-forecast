# Delivery tables (Delta Lake on S3)

How OCF delivers forecasts and supporting data to NGED.

> **Status legend** — ✅ Implemented in code · 🚧 Planned · 🔬 Research (v2). See the
> [roadmap index](index.md) for the version timeline these statuses refer to.

---

## How forecasts are delivered

For v1 of the live service, OCF delivers live forecasts (and the supporting tables below) as
**Delta Lake** tables in an AWS S3 bucket. We are **not** building a REST API for v1 (see
[scope changes](index.md)); an API may be added later if it brings additional benefit.

- **Why Delta Lake?** It is just Parquet files plus a transaction log, giving ACID guarantees on
  cheap object storage. NGED never reads a half-written forecast: each update is atomic, so the
  latest forecast is either fully available or not visible at all. Delta also supports schema
  evolution and time-travel, and is readable from Excel, Polars, pandas, DuckDB, Power BI, etc.
- **Access**: the bucket holds forecasts about NGED's customers, so it is protected with S3
  authentication (mirroring how NGED protects their own time-series JSON bucket).
- **Update cadence**: every 6 hours, when a new forecast run is generated.
- **Reading it** is a one-liner — `scan_delta` is lazy and only fetches the partitions a query
  touches, so a single forecast comes back in a fraction of a second even though the full dataset is
  billions of rows:

  ```python
  dataframe = polars.scan_delta("s3://path-to-forecast")
  ```

There are **five** tables. This table tracks where each one stands today:

| # | Table | Status | Where the schema lives |
|---|---|---|---|
| 1 | `power_forecast` | ✅ Implemented (deterministic-ensemble flavour only) | `contracts.power_schemas.PowerForecast` |
| 2 | `power_forecast_warnings` | 🚧 Planned (partial in MVP) | not yet in code |
| 3 | `asset_health_history` | 🚧 Planned | not yet in code |
| 4 | `effective_capacity` | 🚧 MVP in v0.1 (P99 estimate); DP upgrade planned for v0.6 / v0.7 | `contracts.power_schemas.EffectiveCapacity` |
| 5 | `substation_switching` | 🚧 Planned (v0.6 / v0.7) | not yet in code |

> **Naming note.** The Milestone 1 report drafted these tables with the column name `timeseries_id`,
> but the codebase uses **`time_series_id`** everywhere. The implemented `PowerForecast` schema uses
> `time_series_id`, and the planned tables below should adopt the same name for consistency.

---

## Table 1 — `power_forecast`

> **Status: ✅ Implemented** as `contracts.power_schemas.PowerForecast`, but only the
> **deterministic-ensemble** representation of uncertainty (below). The two percentile-based
> representations are 🚧 planned.

Stores OCF's probabilistic power forecasts. The table extends ~14 days **forwards** in time (the
forecast horizon) and **backwards** to the start of the backtesting period.

The Milestone 1 report describes **three** ways of expressing uncertainty. We may deliver one or
several of these:

1. **Ensemble of deterministic forecasts** — one row per NWP ensemble member. ✅ **Implemented** and
   the likely MVP representation.
2. **Percentiles** — one row per `valid_time`, with a column per percentile. 🚧 Planned.
3. **Ensemble of percentile forecasts** — per-member *and* per-percentile. 🔬 Research.

### Fields common to all three representations

| Field | Data type | Notes |
|---|---|---|
| `time_series_id` | `int32` | The time-series ID provided by NGED. E.g. ID 1 is the raw MW flow for the "BOSTON" BSP; ID 10 is the disaggregated demand for the "SPILSBY" primary; ID 21 is the MW flow for a customer's PV site. |
| `valid_time` | `datetime` (UTC) | The time the forecast is *about*. |
| `nwp_init_time` | `datetime` (UTC), nullable | When the NWP model was initialised. Uniquely identifies the NWP ensemble member, and matters when NWP is delayed or when we deliberately use lagged NWP runs to grow the ensemble. Null for models that do not use NWP (e.g. persistence baselines). |
| `power_fcst_init_time` | `datetime` (UTC) | When OCF's power-forecast model was initialised (the "t0"). |
| `power_fcst_model_name` | `categorical` (string) | Name of OCF's model, e.g. `"xgboost_baseline"`. Set by the `BaseForecaster` subclass. |
| `power_fcst_model_version` | `int16` | Version of OCF's model. |

**Implemented extensions (beyond the report draft):** the live `PowerForecast` schema also carries
`ml_flow_experiment_id` (`int32`, nullable) and `fold_id` (`categorical`; the CV validation year, or
`"live"` for production forecasts). These let cross-validation rows and live rows share one table —
filter on `fold_id` to select the population you need. See
[metrics & leaderboard](metrics-and-leaderboard.md).

### Representation 1 — ensemble of deterministic forecasts ✅

| Field | Data type | Notes |
|---|---|---|
| `ensemble_member` | `int8` | ID of the NWP ensemble member driving this forecast. ECMWF ENS has 51 members. |
| `power_fcst` | `float32` | OCF's power forecast. **In this early version** the value is in physical units (MW for active power, MVA for apparent power). |

> **🚧 Planned change — normalise `power_fcst` to [−1, +1].** The intended long-term delivery
> contract is a normalised forecast in the range **[−1, +1]**, which NGED multiplies by a capacity
> to obtain MW/MVA (see [forecast building blocks](forecast-building-blocks.md)). We forecast raw
> MW/MVA for now because we are **not yet estimating capacity**; once capacity estimation lands
> (v0.6 / v0.7) we plan to switch `power_fcst` to the [−1, +1] scaled value. This is also noted in a
> comment on the `PowerForecast.power_fcst` field in `power_schemas.py`.

### Representation 2 — percentiles 🚧

One row per `valid_time`, with one column per percentile:

| Fields | Data type | Notes |
|---|---|---|
| `p1, p2, p5, p10, p20, p35, p50, p65, p80, p90, p95, p98, p99` | `float32` | The power forecast at each percentile. NGED is far more interested in the **tails** than the shoulders. Read `p50` as "50% chance the true power flow is below this value", `p99` as "99% chance below", etc. (Scaled to [−1, +1] once normalisation lands.) |

### Representation 3 — ensemble of percentile forecasts 🔬

Each ensemble member is itself a percentile forecast:

| Fields | Data type | Notes |
|---|---|---|
| `ensemble_member` | `int8` | ID of the NWP ensemble member. |
| `p1, p2, p5, p10, p20, p35, p50, p65, p80, p90, p95, p98, p99` | `float32` | The power forecast. |

---

## Table 2 — `power_forecast_warnings` 🚧

> **Status: 🚧 Planned.** Some warning types depend on switching-event detection and effective-capacity
> estimation, which do not exist yet — so this table will be **partial in the MVP** and complete by
> v1.0.

Tells NGED whenever we detect abnormal behaviour in the **most recent meter reading** for a
`time_series_id`. Such abnormality means the asset's actual performance may deviate from the
"normal operation" forecast over the next 14 days. Each warning is valid for one entire forecast run
and is refreshed every 6 hours.

> *Example:* a generator has run at only 60% of nominal capacity for the last month. The
> "normal operation" forecast still assumes 100% capacity, so we must flag that the forecast is
> deliberately deviating from reality.

Join `power_forecast_warnings` to `power_forecast` on `time_series_id` **and**
`power_fcst_init_time`.

| Field | Data type | Notes |
|---|---|---|
| `time_series_id` | `int32`, **primary key** | Time-series ID provided by NGED. |
| `power_fcst_init_time` | `datetime` (UTC), **primary key** | When OCF's model was initialised. |
| `last_observation_time` | `datetime` (UTC) | Time of the last valid meter reading available when this forecast was created. |
| `warning_type` | enum (see below) | Refers to the meter reading as of `last_observation_time`. |
| `warning_description` | `string` or null | Human-readable detail, e.g. *"The meter's value has been stuck at 2.1 MW for the previous 52 hours."* |

**`warning_type` enum values** (mostly mutually exclusive — there is a hierarchy: a meter error
blinds us to all other errors at that timestep; a generator/circuit fault blinds us to reduced
capacity):

- `HEALTHY`
- `MISSING VALUE`
- `STUCK TIMESERIES`
- `INVALID TIMESERIES VALUE` — note: *not* "invalid meter value"; NGED call meters "analogues"
- `GENERATOR OR CIRCUIT FAULT`
- `GENERATOR REDUCED CAPACITY`
- `SUBSTATION ABNORMAL RUNNING ARRANGEMENT`
- `STALE NWP` — the weather feed was delayed, forcing the forecast onto stale NWP
- `STALE POWER` — the live power data was delayed

**Self-assembly note:** NGED (or any user) can build their own forecast by multiplying the
(future) [−1, +1] `power_forecast` by a capacity derived from the
[`effective_capacity`](#table-4-effective_capacity) table's `effective_capacity_MW_mean` column —
its **maximum** over history (→ a "normal" forecast) or its **most recent** value (→ a
"prevailing conditions" forecast). See [forecast building blocks](forecast-building-blocks.md).

---

## Table 3 — `asset_health_history` 🚧

> **Status: 🚧 Planned.**

A complete **historical** record of the health of each `time_series_id`. The main use case: when an
NGED engineer sees a warning, they can manually investigate the recent behaviour of that time series.

| Field | Data type | Notes |
|---|---|---|
| `time_series_id` | `int32`, **primary key** | Time-series ID provided by NGED. |
| `time` | `datetime` (UTC), **primary key** | Timestamp from the original NGED time series. |
| `warning_type` | enum | Same vocabulary as Table 2, but only the *backward-looking* subset (no `STALE NWP` / `STALE POWER`): `HEALTHY`, `MISSING VALUE`, `STUCK TIMESERIES`, `INVALID TIMESERIES VALUE`, `GENERATOR OR CIRCUIT FAULT`, `GENERATOR REDUCED CAPACITY`, `SUBSTATION ABNORMAL RUNNING ARRANGEMENT`. |

Notes on specific flags:

- **`GENERATOR OR CIRCUIT FAULT`** is set whenever metered generation is zero when the generator
  *should* be generating (e.g. a solar farm reading zero at midday on a sunny day). It can also be
  set manually to represent, e.g., a maintenance schedule supplied by the generator.
- **`STUCK TIMESERIES`** / **`INVALID TIMESERIES VALUE`** are set for any historical timestep where
  the metered values are "stuck", or are physically implausible (e.g. 1 GW from a 1 MW solar farm).

---

## Table 4 — `effective_capacity` 🚧

> **Status: 🚧 MVP in v0.1** using P99 of observed power as a static capacity proxy.
> Schema lives in `contracts.power_schemas.EffectiveCapacity`.
> **Planned upgrade in v0.6 / v0.7** to differentiable-physics capacity estimation — see
> [Capacity estimation](capacity-estimation.md).

OCF's estimate of each generator's or substation's **effective capacity** at every half-hourly
timestep of the historical time series data. This table is **backward-looking only** — it does
not cover the forecast period.

**MVP approach (v0.1):** one row per `time_series_id`, `effective_capacity_mw` = P99 of
`|power|` over the full available observation history. This is a static scalar per series — a
robust capacity proxy that is less sensitive to outlier spikes than the maximum, and more
capacity-representative than the mean (which is dragged down by zero-output periods for PV/wind).
It is also the denominator used to normalise NMAE in the `forecast_metrics` table — see
[Normalising NMAE by `effective_capacity`](metrics-and-leaderboard.md#normalising-nmae-by-effective_capacity)
for why the MVP stores one scalar row per series (rather than repeating the value at every half-hour)
and how the metrics join evolves for the DP upgrade.

**DP upgrade (v0.6 / v0.7):** replace the static P99 with a time-varying estimate from the
differentiable-physics model. For generators, the prior comes from the Embedded Capacity Register
and is updated at each half-hour from the generator's power time series, absorbing PV-panel
degradation, partial inverter trips, etc., but **ignoring ANM** (a wind farm ANM-capped at 5 MW
with 10 MW physical capability has `effective_capacity_mw = 10`). For substations, the 99th
percentile of observed load over a rolling window, under normal running arrangement only. During a
switching event, effective capacity = last known normal-arrangement value plus the "switched
power" from [Table 5](#table-5-substation_switching). The schema is unchanged between MVP and DP;
the asset body changes to emit one row per `(time_series_id, time)`, and the metrics pipeline swaps
its `time_series_id`-only capacity join for a temporal as-of join (see
[Normalising NMAE by `effective_capacity`](metrics-and-leaderboard.md#normalising-nmae-by-effective_capacity)).

| Field | Data type | Notes |
|---|---|---|
| `time_series_id` | `int32` | NGED's time-series ID. |
| `time` | `datetime` (UTC), every half hour | The half-hourly timestep this estimate applies to. In the MVP, set to the end of the available observation history for that series. |
| `effective_capacity_mw` | `float32` | OCF's estimate of the effective capacity (MW) at this timestep. |

---

## Table 5 — `substation_switching` 🚧

> **Status: 🚧 Planned (v0.6 / v0.7).** Depends on switching-event detection.

Captures the amount of power OCF estimates has been **switched** from a "donor" substation to a
"recipient" substation. OCF estimates switching events purely from the power-flow time series and
the known electrical connections between substations — it has **no** access to NGED's operational
switch-control system. Represented as the mean + std of a Normal distribution.

A single donor can split power across multiple recipients (e.g. donor A loses 1 MW, emerging as
0.6 MW to B, 0.3 MW to C, 0.1 MW to D). The table therefore uses **one row per recipient**.

| Field | Data type | Notes |
|---|---|---|
| `donor_time_series_id` | `int32` | Time-series ID of the substation that has had power diverted *away* from it. |
| `recipient_time_series_id` | `int32` | Time-series ID of the substation power has been diverted *to*. |
| `time` | `datetime` (UTC) | Timestamp from the original NGED time series. |
| `switched_power_MW_mean` | `float32` (strictly positive) | Mean of OCF's estimate of power diverted from the donor to the recipient. |
| `switched_power_MW_std` | `float32` | Standard deviation of that estimate. |

> **Known simplification.** Switching events transfer *behaviour*, not a constant amount of power.
> v1 (the v0.6 statistical detector) estimates only the transferred magnitude. Reconstructing the
> latent demand under the normal running arrangement — and capturing the fact that a transferred
> slice can carry a different demand/PV/wind mix than its parent — is the job of the v2 mixture
> models. See [Switching events & latent demand](switching-events.md) (v2.5 / v2.6).
