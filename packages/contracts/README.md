# Contracts

Defines the "data contracts": the schemas defining the precise shape of each data source, the units
each column carries, and the sign convention each column's values follow.

The contracts package also owns the thin configuration layer that sits beside those schemas: the
cross-validation (CV) fold config and the `class_target`/`import_class` pair that turns a class into
a `_target_` string and back. A CV fold is one train-then-validate split of the history, and
`_target_` is the key a YAML config file uses to name the Python class to build. Both the fold
config and the class-path pair are model-agnostic and need nothing heavier than pydantic and PyYAML.

Two further modules sit here because every package needs them and neither is specific to machine
learning (ML). `contracts.settings` holds `Settings`, the single source of every setting the
pipeline reads: the data paths, the object-store credentials, the MLflow tracking URI, and the four
settings for Sentry, the error-reporting service (`sentry_dsn`, `sentry_environment`,
`sentry_traces_sample_rate`, and `sentry_monitor_forecasts`), resolved from the environment and the
workspace `.env` and reached through the cached `get_settings()`. `contracts.uri` holds the
local-or-remote path helpers those settings fields need, because a data-location field may be a
local path or an `s3://` URI, and `pathlib` mangles a URI.

## Light enough for any component to import

This package is designed to be lightweight. It defines the *shape* of the data using Patito and
Polars, plus the settings and object-store path helpers those shapes are read and written through
(`deltalake` and `obstore`), but it contains **no** ML-specific logic and no ML dependency such as
MLflow, XGBoost, or Dagster. That light dependency footprint is what lets any component in the
system (e.g., a data ingestion script or a dashboard) import these schemas without bringing in the
entire ML stack.

## Key data contracts

The five schemas below are the ones most callers touch. The package defines four more —
`EffectiveCapacity`, `Metrics`, `EligibleTimeSeries`, and `H3GridWeights` — for nine in all.

- **`PowerTimeSeries`**: Half-hourly power observations in MW (megawatts) or MVA (megavolt-amperes)
  per `time_series_id`, as received from National Grid Electricity Distribution (NGED), the
  distribution network operator whose network this project forecasts.
- **`TimeSeriesMetadata`**: Substation and customer meter metadata, including lat/lon, H3 index
  (the identifier of one cell of the H3 hexagonal grid the weather is aggregated onto),
  `substation_type` (`Primary`, `BSP` for bulk supply point, `GSP` for grid supply point,
  `EHV Customer` for an extra-high-voltage customer, or `HV Customer` for a high-voltage customer),
  and `time_series_type` (`PV` for solar photovoltaic, `Wind`, `BESS` for battery energy storage
  system, `Disaggregated Demand`, and 18 others — `LIST_OF_TIME_SERIES_TYPES` holds all 22).
- **`Nwp`**: Numerical weather prediction (NWP) data from the European Centre for Medium-Range
  Weather Forecasts (ECMWF) ensemble (ENS), in physical units (`Float32`), on disk and in memory
  alike. An ensemble forecast runs the weather model 51 times — one control run plus 50 runs from
  slightly perturbed starting conditions — and each run is one ensemble member, so the spread
  across the 51 members is what expresses the forecast's uncertainty. The on-disk copy is rounded
  to a 13-bit significand (a relative error of at most 2^-13 ≈ 1.2e-4, far below the forecast's own
  error) and laid out so that compression works well and a query can skip whole Parquet row groups,
  both by `delta_store.nwp`.
- **`AllFeatures`**: The final joined dataset passed to ML models. Primary key is `(time_series_id,
  power_fcst_init_time, valid_time[, ensemble_member])`, where the square brackets mark
  `ensemble_member` as a key column only when the frame carries one row per ensemble member.
  Includes NWP weather variables, power lag/rolling features (the power observed a given number of
  hours earlier, and its rolling mean over a given number of hours), and datetime features.
  `time_series_type` is the one metadata column it can carry, and only when a feature set asks for
  it.
- **`PowerForecast`**: ML model output schema. `power_fcst` is in MW (active power) or MVA (apparent
  power), with the unit given per `time_series_id` in `TimeSeriesMetadata`. A planned change will
  normalise the forecast to [−1, +1] for NGED to multiply by that series' `effective_capacity`, in
  the same MW or MVA, to recover a power — see [Forecast Building
  Blocks](https://openclimatefix.github.io/nged-substation-forecast/roadmap/forecast-building-blocks/).
  Includes `power_fcst_model_name`, `power_fcst_model_version`, `power_fcst_init_time`,
  `nwp_init_time`, `valid_time`, `time_series_id`, and `ensemble_member`.

## Sign convention

<!-- sign-convention:start --> Sign convention depends on `substation_type` in `TimeSeriesMetadata`,
whose five values (`BSP`, `EHV Customer`, `GSP`, `HV Customer`, `Primary`) partition into two
behavioural cases:

- **Substations** (`BSP`, `GSP`, `Primary`): positive = power flowing **towards end-users**;
  negative = excess generation flowing **back upstream**, into the network above the substation.
- **Customer meters** (`EHV Customer`, `HV Customer`): positive = the customer is **sending** power
  to NGED's distribution network; negative = the customer is **drawing** power from NGED's
  distribution network. A customer meter can sit at a demand site or a generation site, so this
  case is not "generators only".

<!-- sign-convention:end -->

## Design principles

- **The contract is the authoritative account of what the data means.** It says what the data
  *should* be, not what some current code path happens to produce. So when code and contract
  disagree, the code is the first suspect. A null the contract forbids usually means an upstream
  join kept a row it should have dropped, or a caller passed input it should have rejected. Widening
  a field to `| None`, or relaxing a range, so that a failing `validate()` passes buries that defect
  in the one place the rest of the system trusts. Fix the code instead. Change the contract only
  when you can say what the data now means and why that meaning is right. **Get the change agreed
  before making it**, including a widening that looks like a formality — every reader of `contracts`
  is relying on it to still mean what it said yesterday.
- **Column naming**: Column names are `snake_case` throughout, with any acronym or SI unit
  lower-cased inside the name (`effective_capacity_mw`, `h3_res_5`, `area_wkt`). Acronyms and SI
  units keep their standard capitalisation everywhere else — in a column's *values* (`MW`, `MVA`,
  `BSP`, `PV`), in class names, and in prose, where "DER" (distributed energy resource) and "MW"
  (megawatts) are both uppercase.
- **Semantic checks**: Range validation should be generous — the aim is to catch physically
  impossible values (e.g., 1 GW from a 1 MW solar farm), not possible-but-unlikely values.
- **Datetime ranges**: Timestamps on the columns where external data enters —
  `PowerTimeSeries.time`, `Nwp.init_time` and `Nwp.valid_time` — are bounded to
  `[MIN_PLAUSIBLE_DATETIME, MAX_PLAUSIBLE_DATETIME]` (2000-01-01 to 2100-01-01, inclusive), which
  rejects a corrupt feed or an epoch-unit mix-up without ever excluding a real reading. The check
  lives in each model's `validate` override via `check_datetime_bounds`, because Patito silently
  ignores `ge` and `le` — the greater-than-or-equal and less-than-or-equal bounds a field
  declares — on a datetime field. Patito derives its bounds checks from the JSON Schema
  `minimum`/`maximum` keywords, which JSON Schema (the validation standard, not NGED's JSON feed)
  defines for numbers only. Columns on our own *output*
  schemas (`PowerForecast`, `EffectiveCapacity`, `AllFeatures`, `Metrics`) have not opted in: they
  are computed from already-bounded inputs rather than received from outside.
- **Degrade, don't abort, at an ingestion boundary**: `validate()` stays strict everywhere — it is
  also used as a hard assertion in tests and R&D code, where a raise-on-violation contract must not
  silently change. But a single malformed row from an external feed should not abort ingestion of
  every other well-formed row in the batch, so `PowerTimeSeries.drop_implausible_rows()` filters out
  rows with an out-of-range or minute-misaligned `time` *before* `validate()` runs, returning the
  survivors plus a count of what was dropped. Only the NGED JSON ingestion path
  (`nged_data.read_nged_json`) calls it; the duplicate/sortedness checks in `validate()` are never
  relaxed, because those indicate a bug in our own pipeline rather than malformed external data.
- **No lookahead bias**: `AllFeatures` carries `power_fcst_init_time` (when we make the forecast) as
  a distinct field from `nwp_init_time` (when the NWP model ran). Power lag features are nullified
  by `_nullify_leaky_lags()` when the lag is shorter than or equal to the forecast lead time —
  the lead time being the gap between `power_fcst_init_time` and `valid_time`. A short lag is the
  dangerous one: a reading taken fewer hours before the target time than the lead time had not yet
  happened when the forecast was made, so using it would be reading the future.
