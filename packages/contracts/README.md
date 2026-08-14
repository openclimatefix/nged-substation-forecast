# Contracts

Defines the "data contracts": the schemas defining the precise shape of each data source and its semantics.

It also owns the thin configuration layer that sits beside those schemas: the CV fold config,
and the `class_target`/`import_class` pair that turns a class into a `_target_` string and
back. Both are model-agnostic and need nothing heavier than pydantic and PyYAML.

## Dependency Isolation

This package is designed to be extremely lightweight. It defines the *shape* of the data using Patito and Polars, but it does **not** contain any ML-specific logic or heavy dependencies like MLflow. This ensures that any component in the system (e.g., a data ingestion script or a dashboard) can import these schemas without bringing in the entire ML stack.

## Key Data Contracts

- **`PowerTimeSeries`**: Half-hourly power observations (MW or MVA) per `time_series_id`, as received from NGED.
- **`TimeSeriesMetadata`**: Substation and customer meter metadata, including lat/lon, H3 index, and asset type (primary substation, GSP, BSP, solar PV, wind, BESS, etc.).
- **`Nwp`**: ECMWF ENS NWP weather data in physical units (`Float32`), on disk and in memory alike. The on-disk copy is rounded to a 13-bit significand and laid out for compression and row-group pruning by `delta_store.nwp`.
- **`AllFeatures`**: The final joined dataset passed to ML models. Primary key is `(time_series_id, power_fcst_init_time, valid_time[, ensemble_member])`. Includes NWP weather variables, power lag/rolling features and datetime features. `time_series_type` is the one metadata column it can carry, and only when a feature set asks for it.
- **`PowerForecast`**: ML model output schema. `power_fcst` is in MW (active power) or MVA (apparent power), with the unit given per `time_series_id` in `TimeSeriesMetadata`. A planned change will normalise it to [−1, +1] for NGED to multiply by a capacity — see [Forecast Building Blocks](https://openclimatefix.github.io/nged-substation-forecast/roadmap/forecast-building-blocks/). Includes `power_fcst_model_name`, `power_fcst_model_version`, `power_fcst_init_time`, `nwp_init_time`, `valid_time`, `time_series_id`, and `ensemble_member`.

## Design Principles

- **The contract is the authoritative account of what the data means.** It says what the data
  *should* be, not what some current code path happens to produce. So when code and contract
  disagree, the code is the first suspect: a null the contract forbids usually means an upstream
  join kept a row it should have dropped, or a caller passed input it should have rejected.
  Widening a field to `| None`, or relaxing a range, so that a failing `validate()` passes buries
  that defect in the one place the rest of the system trusts. Fix the code instead, and change the
  contract only when you can say what the data now means and why that meaning is right. **Get the
  change agreed before making it**, including a widening that looks like a formality — every
  reader of `contracts` is relying on it to still mean what it said yesterday.
- **Column naming**: Prefer `snake_case`, except for acronyms or SI units. Capitalise "DER" (distributed energy resource) and use uppercase for "MW" (megawatts).
- **Semantic checks**: Range validation should be generous — the aim is to catch physically impossible values (e.g., 1 GW from a 1 MW solar farm), not possible-but-unlikely values.
- **Datetime ranges**: Timestamps on the columns where external data enters — `PowerTimeSeries.time`, `Nwp.init_time` and `Nwp.valid_time` — are bounded to `[MIN_PLAUSIBLE_DATETIME, MAX_PLAUSIBLE_DATETIME]` (2000-01-01 to 2100-01-01, inclusive), which rejects a corrupt feed or an epoch-unit mix-up without ever excluding a real reading. The check lives in each model's `validate` override via `check_datetime_bounds`, because Patito silently ignores `ge`/`le` on a datetime field — it derives its bounds checks from the JSON schema's `minimum`/`maximum`, which JSON Schema defines for numbers only. Columns on our own *output* schemas (`PowerForecast`, `EffectiveCapacity`, `AllFeatures`) have not opted in: they are computed from already-bounded inputs rather than received from outside.
- **Degrade, don't abort, at an ingestion boundary**: `validate()` stays strict everywhere — it is also used as a hard assertion in tests and R&D code, where a raise-on-violation contract must not silently change. But a single malformed row from an external feed should not abort ingestion of every other well-formed row in the batch, so `PowerTimeSeries.drop_implausible_rows()` filters out rows with an out-of-range or minute-misaligned `time` *before* `validate()` runs, returning the survivors plus a count of what was dropped. Only the NGED JSON ingestion path (`nged_data.read_nged_json`) calls it; the duplicate/sortedness checks in `validate()` are never relaxed, because those indicate a bug in our own pipeline rather than malformed external data.
- **No lookahead bias**: `AllFeatures` carries `power_fcst_init_time` (when we make the forecast) as a distinct field from `nwp_init_time` (when the NWP model ran). Power lag features are nullified by `nullify_leaky_lags()` when the lag is shorter than or equal to the forecast lead time.
