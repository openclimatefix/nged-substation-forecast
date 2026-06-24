## Contracts

Defines the "data contracts": the schemas defining the precise shape of each data source and its semantics.

## Dependency Isolation

This package is designed to be extremely lightweight. It defines the *shape* of the data using Patito and Polars, but it does **not** contain any ML-specific logic or heavy dependencies like MLflow. This ensures that any component in the system (e.g., a data ingestion script or a dashboard) can import these schemas without bringing in the entire ML stack.

## Key Data Contracts

- **`PowerTimeSeries`**: Half-hourly power observations (MW or MVA) per `time_series_id`, as received from NGED.
- **`TimeSeriesMetadata`**: Substation and customer meter metadata, including lat/lon, H3 index, and asset type (primary substation, GSP, BSP, solar PV, wind, BESS, etc.).
- **`NwpOnDisk`** / **`NwpInMemory`**: ECMWF ENS NWP weather data. Stored on disk as `Int16` (quantised to 12-bit range per `NwpScalingParams`) and converted back to `Float32` physical units in memory. The two schemas enforce that the right numeric type is used in each context.
- **`AllFeatures`**: The final joined dataset passed to ML models. Primary key is `(time_series_id, power_fcst_init_time, valid_time[, ensemble_member])`. Includes NWP weather variables, power lag/rolling features, static metadata columns, and datetime features.
- **`PowerForecast`**: ML model output schema. Power values are in the range [−1, +1] (normalised). Includes `power_fcst_model_name`, `power_fcst_model_version`, `power_fcst_init_time`, `nwp_init_time`, `valid_time`, `time_series_id`, and `ensemble_member`.

## Design Principles

- **Column naming**: Prefer `snake_case`, except for acronyms or SI units. Capitalise "DER" (distributed energy resource) and use uppercase for "MW" (megawatts).
- **Semantic checks**: Range validation should be generous — the aim is to catch physically impossible values (e.g., 1 GW from a 1 MW solar farm), not possible-but-unlikely values.
- **No lookahead bias**: `AllFeatures` carries `power_fcst_init_time` (when we make the forecast) as a distinct field from `nwp_init_time` (when the NWP model ran). Power lag features are nullified by `nullify_leaky_lags()` when the lag is shorter than or equal to the forecast lead time.
