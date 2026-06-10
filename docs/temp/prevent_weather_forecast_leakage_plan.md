# Implementation Plan: Prevent Weather Forecast Leakage (Once-a-Day Alignment MVP)

```yaml
---
status: "approved"
version: "v1.2"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/contracts", "packages/ml_core"]
---
```

## 1. Problem Statement

In time-series forecasting, **lookahead bias** and **target leakage** occur when a model is trained on features that contain information from the future that would not be available at the moment of live inference. 

In this project, we use Numerical Weather Prediction (NWP) forecasts as input features to predict power. NWP forecasts are initialized at specific times (e.g., `00:00 UTC`) and take several hours to be computed and published (the **publication delay**). 

### The Leakage Mechanism in the Current Code:
1. **Shortest Lead-Time Selection:** Currently, `_process_nwp_data` pre-aggregates the NWP data to select the forecast with the absolute shortest lead time for each `(time_series_id, valid_time)` combination across the entire dataset.
2. **Future Forecast Leakage:** When computing a weather lag (e.g., `temperature_2m_lag_24h`) for a forecast initialized at `power_fcst_init_time = 2026-06-10 00:00:00` predicting `valid_time = 2026-06-11 12:00:00`, the lagged target time is `2026-06-10 12:00:00`.
3. **The Leak:** Because the join does not enforce any temporal constraints on the weather forecast's initialization time (`nwp_init_time`), the pipeline pulls the forecast for `2026-06-10 12:00:00` with the shortest lead time (e.g., initialized at `2026-06-10 11:00:00`). At our forecast run time (`2026-06-10 00:00:00`), a forecast initialized at `11:00 UTC` does not exist yet. This leaks future weather forecast improvements and observations into the model, leading to overly optimistic backtest results that cannot be replicated in production.

---

## 2. Solutions Considered and Rejected

### Rejected Option A: Match `init_time` of the lag source to the `init_time` of the target row
* **Concept:** Force the weather lag to come from the exact same forecast run as the current weather feature.
* **Why Rejected:** An NWP forecast run initialized at `init_time` only contains forecasts for future times (`valid_time >= init_time`). If a weather lag goes into the past relative to `init_time` (e.g., `target_time = init_time - 12h`), the current forecast run has no data for that time, resulting in a high rate of `null` values and rendering historical weather lags useless.

### Rejected Option B: Temporal `join_asof` (Horizon Shuffling)
* **Concept:** Allow the lag to be pulled from any past forecast run as long as `nwp_init_time <= power_fcst_init_time - nwp_publication_delay_hours`. Use Polars' `join_asof` with `strategy="backward"` to find the latest available forecast.
* **Why Rejected for MVP:** While scientifically valid and highly flexible, implementing this requires complex pseudo-random lead-time sampling, deterministic hashing, and snapping logic to avoid multiplying the dataset size by the number of simulated forecast runs (e.g., 672 lead times for a 14-day forecast). It adds significant complexity before establishing a working baseline.

---

## 3. The Chosen Solution: Once-a-Day Alignment MVP

We will implement a highly simplified, elegant, and 100% scientifically valid design based on **Once-a-Day Alignment**:

1. **The Alignment Rule:** We assume the power forecast is run exactly once per day, precisely when the new NWP forecast becomes available:
   $$\text{power\_fcst\_init\_time} = \text{nwp\_init\_time} + \text{nwp\_publication\_delay\_hours}$$
2. **Standard Equi-Joins (No `join_asof`!):** Because of this exact equality, we can enforce the non-leakage constraint using standard, fast Polars equi-joins on `nwp_init_time` (renamed from `init_time`).
3. **No Row Multiplication:** The dataset size remains minimal ($1 \times N$), with exactly one row per `(time_series_id, valid_time, ensemble_member)`.
4. **Perfect Lead-Time Coverage:** Because the NWP run forecasts 14 days ahead, our training set naturally and uniformly covers all lead times from 30 minutes to 14 days.
5. **Production Compatibility:** A model trained this way will still work perfectly in production even if run 4 times a day, as it learns the general relationship between weather, lags, and lead times.

---

## 4. Detailed Step-by-Step Implementation Plan

### Step 1: Update `AllFeatures` Schema
Update the `AllFeatures` schema in `packages/contracts/src/contracts/ml_schemas.py` to rename `lead_time_hours` to `nwp_lead_time_hours` and add the two initialization time columns.

- **File:** `packages/contracts/src/contracts/ml_schemas.py`
- **Changes:**
  - Rename `lead_time_hours` to `nwp_lead_time_hours`:
    ```python
    nwp_lead_time_hours: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    ```
  - Add `power_fcst_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)`
  - Add `nwp_init_time: datetime | None = pt.Field(dtype=UTC_DATETIME_DTYPE, allow_missing=True)`

---

### Step 2: Modify `engineer_features` Signature & Derivation
Update `engineer_features` to accept `nwp_publication_delay_hours` (defaulting to 6) and remove any external `power_fcst_init_time` argument. Always generate `power_fcst_init_time` internally.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update signature and add a clear docstring:
    ```python
    def engineer_features(
        selected_features: set[str],
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpOnDisk] | None = None,
        nwp_publication_delay_hours: int = 6,
    ) -> pt.LazyFrame[AllFeatures]:
        """Engineer features.

        Args:
            selected_features: Set of features to engineer.
            power_time_series: Input power time series.
            time_series_metadata: Metadata for the time series.
            nwp: NWP weather forecast data.
            nwp_publication_delay_hours: The delay in hours between the initialization time
                of the NWP forecast and when it becomes publicly available for use in power
                forecasting. For example, if a weather forecast is initialized at 00:00 UTC,
                and nwp_publication_delay_hours is 6, then that forecast cannot be used by
                any power forecast initialized before 06:00 UTC. This prevents lookahead bias
                and target leakage during offline backtesting.
        """
    ```
  - Derive `power_fcst_init_time` and `nwp_init_time` internally:
    ```python
    # Convert Patito to Polars immediately for cleaner downstream code
    power_lf = pl.LazyFrame._from_pyldf(power_time_series._ldf).rename({"time": "valid_time"})
    metadata_lf = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf)
    nwp_lf = pl.LazyFrame._from_pyldf(nwp._ldf) if nwp is not None else None

    parsed_features = ParsedFeatures.from_strings(selected_features)

    if nwp_lf is None and parsed_features.requires_weather_data():
        raise ValueError("Weather features were requested but no NWP data was provided.")

    # Process NWP data
    processed_nwp, historical_weather = _process_nwp_data(nwp_lf, parsed_features)

    # Join Data
    raw_data = power_lf.join(metadata_lf, on="time_series_id", how="left")

    if processed_nwp is not None:
        # Standard join on valid_time
        raw_data = processed_nwp.join(raw_data, on=["time_series_id", "valid_time"], how="left")
        # Ensure unique rows
        raw_data = raw_data.unique(["time_series_id", "valid_time", "ensemble_member"])
        
        # Generate power_fcst_init_time and rename init_time to nwp_init_time
        raw_data = raw_data.with_columns(
            power_fcst_init_time=pl.col("init_time") + pl.duration(hours=nwp_publication_delay_hours),
            nwp_init_time=pl.col("init_time")
        )
    else:
        # Fallback if no NWP is provided
        raw_data = raw_data.with_columns(
            power_fcst_init_time=pl.col("valid_time"),
            nwp_init_time=pl.lit(None, dtype=pl.Datetime(time_unit="us", time_zone="UTC")),
            nwp_lead_time_hours=pl.lit(None, dtype=pl.Float32),
            ensemble_member=pl.lit(None, dtype=pl.UInt8)
        )
    ```
  - Update `base_cols` selection to include all required fields:
    ```python
    base_cols = [
        "valid_time",
        "time_series_id",
        "time_series_type",
        "power",
        "nwp_lead_time_hours",
        "ensemble_member",
        "power_fcst_init_time",
        "nwp_init_time",
    ]
    ```

---

### Step 3: Refactor NWP Processing Functions
Rename `calculate_lead_time` to `calculate_nwp_lead_time` and refactor `_process_nwp_data` to make it simpler and easier to read.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Rename and update `calculate_lead_time`:
    ```python
    def calculate_nwp_lead_time(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Calculates the NWP lead time in hours if 'init_time' is present in the schema."""
        if "init_time" in lf.collect_schema().names():
            return lf.with_columns(
                nwp_lead_time_hours=(
                    (pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600
                ).cast(pl.Float32)
            )
        return lf
    ```
  - Refactor `_process_nwp_data` to keep all available initialization times (no pre-aggregation to the "best" forecast):
    ```python
    def _process_nwp_data(
        nwp_lf: pl.LazyFrame | None,
        parsed_features: ParsedFeatures,
    ) -> tuple[pl.LazyFrame | None, pl.LazyFrame | None]:
        """Process NWP data and prepare historical weather for lag features."""
        if nwp_lf is None:
            return None, None

        processed_nwp = calculate_nwp_lead_time(nwp_lf)

        weather_lag_requested = any(lag_feat.base_col != "power" for lag_feat in parsed_features.lags)

        historical_weather = None
        if weather_lag_requested:
            # For weather lags, we use the control member (ensemble_member == 0)
            historical_weather = (
                processed_nwp.filter(pl.col("ensemble_member") == 0)
                .drop("ensemble_member")
            )

        return processed_nwp, historical_weather
    ```

---

### Step 4: Implement Non-Leaky Weather Lags in `apply_lag_feature`
Update `apply_lag_feature` to use standard equi-joins on `init_time` for weather lags, ensuring they are pulled from the exact same forecast run.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update implementation:
    ```python
    def apply_lag_feature(
        target_lf: pl.LazyFrame, source_lf: pl.LazyFrame, lag_feature: LagFeature
    ) -> pl.LazyFrame:
        """Applies a lag feature using a time-aware join.

        For power lags, we use an exact join on valid_time - lag_hours.
        For weather lags, we use an exact join on target_time and init_time to ensure
        the lag is pulled from the same forecast run, preventing lookahead bias.
        """
        lag_hours = lag_feature.hours
        base_col = lag_feature.base_col

        if base_col == "power":
            # Power lag: exact join on valid_time - lag_hours
            lf_with_target_time = target_lf.with_columns(
                target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
            )

            join_keys = ["time_series_id"]
            if "ensemble_member" in target_lf.collect_schema().names() and "ensemble_member" in source_lf.collect_schema().names():
                join_keys.append("ensemble_member")

            right_lf = source_lf.select(
                *join_keys,
                pl.col("valid_time"),
                pl.col(base_col).alias(lag_feature.string_repr),
            )

            return lf_with_target_time.join(
                right_lf,
                left_on=join_keys + ["target_time"],
                right_on=join_keys + ["valid_time"],
                how="left",
            ).drop("target_time")

        else:
            # Weather lag: exact join on target_time and init_time (no leakage!)
            lf_with_target_time = target_lf.with_columns(
                target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
            )

            join_keys = ["time_series_id", "init_time"]
            if "ensemble_member" in target_lf.collect_schema().names() and "ensemble_member" in source_lf.collect_schema().names():
                join_keys.append("ensemble_member")

            right_lf = source_lf.select(
                *join_keys,
                pl.col("valid_time"),
                pl.col(base_col).alias(lag_feature.string_repr),
            )

            return lf_with_target_time.join(
                right_lf,
                left_on=join_keys + ["target_time"],
                right_on=join_keys + ["valid_time"],
                how="left",
            ).drop("target_time")
    ```

---

### Step 5: Update `nullify_leaky_lags`
Update `nullify_leaky_lags` to use `nwp_lead_time_hours`.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update implementation:
    ```python
    def nullify_leaky_lags(
        lf: pl.LazyFrame, leaky_features: Sequence[LagFeature | RollingFeature]
    ) -> pl.LazyFrame:
        for feature in leaky_features:
            lf = lf.with_columns(
                pl.when(pl.col("nwp_lead_time_hours") >= feature.hours)
                .then(pl.lit(None))
                .otherwise(pl.col(feature.string_repr))
                .alias(feature.string_repr)
            )
        return lf
    ```

---

### Step 6: Update Configs & Tests
Update existing tests and configs to match the new signatures and column names.

- **File:** `conf/model/xgboost.yaml`
  - Rename `"lead_time_hours"` to `"nwp_lead_time_hours"`.
- **File:** `packages/contracts/tests/test_ml_schemas.py`
  - Rename `"lead_time_hours"` to `"nwp_lead_time_hours"`.
  - Add `"power_fcst_init_time"` and `"nwp_init_time"` to test cases.
- **File:** `packages/ml_core/tests/test_features.py`
  - Update `test_engineer_features_nwp_historical_best` to pass `nwp_publication_delay_hours=0` so that it continues to pass with its existing assertions.
  - Update `test_calculate_lead_time` and `test_nullify_leaky_lags` to use `nwp_lead_time_hours`.
  - Add `test_engineer_features_weather_lag_leakage_prevention` to verify that weather lags are pulled from the same forecast run and do not leak future forecasts.
