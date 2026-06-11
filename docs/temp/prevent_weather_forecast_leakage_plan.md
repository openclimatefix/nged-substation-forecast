# Implementation Plan: Prevent Weather Forecast Leakage (Hybrid Recommendation & Option B)

---
status: "approved"
version: "v1.3"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/contracts", "packages/ml_core", "conf"]
---

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

## 3. The Chosen Solution: Hybrid Recommendation & Option B (Expand the Primary Key)

We will implement a highly robust, elegant, and scientifically rigorous design based on the **Hybrid Recommendation** and **Option B (Expand the Primary Key)**:

1. **Primary Key Expansion (Option B):**
   - We expand the primary key of the final joined dataset (`AllFeatures`) to include `power_fcst_init_time`.
   - The new unique constraint/primary key is: `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)`.
   - This allows us to represent multiple forecast runs in the same dataset without row multiplication or silent deduplication.

2. **Explicit Temporal Alignment Join:**
   - We remove the silent `.unique()` deduplication step from `engineer_features`.
   - We perform an explicit temporal alignment join on `["time_series_id", "valid_time", "nwp_init_time"]` (where `nwp_init_time` is derived as `power_fcst_init_time - publication_delay`) to naturally prevent row multiplication.
   - We add a defensive check that fails loudly if duplicates are detected on the primary key `["time_series_id", "power_fcst_init_time", "valid_time", "ensemble_member"]`.

3. **Fix FLAW-001 (Over-Nullification of Lags):**
   - We ensure power lags are nullified based on `power_lead_time_hours = valid_time - power_fcst_init_time` (which is `nwp_lead_time_hours - delay`), not `nwp_lead_time_hours`. This prevents over-nullification of lags that are actually available at `power_fcst_init_time`.

4. **Fix FLAW-002 (Lagged NWP Selection Strategy):**
   - We implement a dual-strategy for weather lags:
     - If the lagged weather target time is in the future relative to `power_fcst_init_time` ($T_{\text{target}} > T_{\text{init}}$), use the exact same NWP run (`init_time`) and ensemble member as the weather used for `valid_time`.
     - If the lagged weather target time is in the past relative to `power_fcst_init_time` ($T_{\text{target}} \le T_{\text{init}}$), use the "freshest" NWP run (control member, ensemble 0) for that target time.

5. **Deterministic Signature & Derivation:**
   - We update `engineer_features` signature to accept `power_fcst_init_time: datetime | None = None` and `nwp_publication_delay_hours: int = 6`.
   - If `power_fcst_init_time` is provided (production/backtest mode), we use it.
   - If `power_fcst_init_time` is `None` (bulk training mode), we derive it internally using the Once-a-Day Alignment rule: `power_fcst_init_time = nwp_init_time + publication_delay`.

6. **Column Renaming:**
   - We rename `lead_time_hours` to `nwp_lead_time_hours` across all schemas, configs, and tests to distinguish it from power forecast lead time.

---

## 4. Detailed Step-by-Step Implementation Plan

### Step 1: Update `AllFeatures` and `Metrics` Schemas
Update the `AllFeatures` and `Metrics` schemas in `packages/contracts/src/contracts/ml_schemas.py` to rename `lead_time_hours` to `nwp_lead_time_hours`, add the two initialization time columns, and implement a custom `validate` method to enforce the primary key uniqueness constraint.

- **File:** `packages/contracts/src/contracts/ml_schemas.py`
- **Changes:**
  - Rename `lead_time_hours` to `nwp_lead_time_hours` in both `AllFeatures` and `Metrics`.
  - Add `power_fcst_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)` to `AllFeatures`.
  - Add `nwp_init_time: datetime | None = pt.Field(dtype=UTC_DATETIME_DTYPE, allow_missing=True)` to `AllFeatures`.
  - Implement a custom `validate` classmethod to enforce uniqueness on `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)`.
  - **Code Snippet:**
    ```python
    class AllFeatures(pt.Model):
        valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
        time_series_id: int = _get_time_series_id_dtype()
        time_series_type: str = pt.Field(dtype=pl.Enum(LIST_OF_TIME_SERIES_TYPES))
        ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)

        power: float = pt.Field(dtype=pl.Float32)
        
        # The NWP lead time in hours (valid_time - nwp_init_time).
        # Renamed from lead_time_hours to distinguish it from power forecast lead time.
        nwp_lead_time_hours: float = pt.Field(dtype=pl.Float32)

        # The initialization time of the power forecast run.
        # This is the temporal anchor for all feature engineering and lag nullification.
        power_fcst_init_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)

        # The initialization time of the NWP forecast run used for this row.
        # This is used to align weather features and weather lags.
        nwp_init_time: datetime | None = pt.Field(dtype=UTC_DATETIME_DTYPE, allow_missing=True)

        # Weather features...
        
        @classmethod
        def validate(
            cls,
            dataframe: pl.DataFrame,
            columns: Sequence[str] | None = None,
            allow_missing_columns: bool = False,
            allow_superfluous_columns: bool = False,
            drop_superfluous_columns: bool = False,
        ) -> pt.DataFrame[Self]:
            """Validate the given dataframe, ensuring uniqueness on the primary key.

            The primary key is (time_series_id, power_fcst_init_time, valid_time, ensemble_member).
            This uniqueness constraint is critical to prevent row multiplication during joins.
            """
            validated_df = super().validate(
                dataframe=dataframe,
                columns=columns,
                allow_missing_columns=allow_missing_columns,
                allow_superfluous_columns=allow_superfluous_columns,
                drop_superfluous_columns=drop_superfluous_columns,
            )

            # Validate uniqueness of (time_series_id, power_fcst_init_time, valid_time, ensemble_member)
            pk_cols = ["time_series_id", "power_fcst_init_time", "valid_time"]
            if "ensemble_member" in validated_df.columns:
                pk_cols.append("ensemble_member")

            if validated_df.select(pk_cols).is_duplicated().any():
                raise ValueError(f"Duplicate entries found for primary key {pk_cols}.")

            return validated_df
    ```

---

### Step 2: Update `SafeInputBaseColumn` and `ParsedFeatures`
Update `SafeInputBaseColumn` literal to include `"nwp_lead_time_hours"`, `"power_fcst_init_time"`, and `"nwp_init_time"`, and remove `"lead_time_hours"`.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update `SafeInputBaseColumn` literal:
    ```python
    SafeInputBaseColumn = Literal[
        "time_series_id",
        "time_series_type",
        "nwp_lead_time_hours",
        "ensemble_member",
        "power_fcst_init_time",
        "nwp_init_time",
    ]
    ```

---

### Step 3: Refactor NWP Processing Functions
Rename `calculate_lead_time` to `calculate_nwp_lead_time` and refactor `_process_nwp_data` to keep all available initialization times (no pre-aggregation to the "best" forecast).

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Rename and update `calculate_lead_time`:
    ```python
    def calculate_nwp_lead_time(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Calculates the NWP lead time in hours if 'init_time' is present in the schema.

        Also renames 'init_time' to 'nwp_init_time' to distinguish it from power forecast init time.
        """
        if "init_time" in lf.collect_schema().names():
            return lf.with_columns(
                nwp_lead_time_hours=(
                    (pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600
                ).cast(pl.Float32)
            ).rename({"init_time": "nwp_init_time"})
        return lf
    ```
  - Refactor `_process_nwp_data` to keep all available initialization times:
    ```python
    def _process_nwp_data(
        nwp_lf: pl.LazyFrame | None,
        parsed_features: ParsedFeatures,
    ) -> tuple[pl.LazyFrame | None, pl.LazyFrame | None]:
        """Process NWP data and prepare historical weather for lag features.

        Unlike the previous implementation, we do NOT pre-aggregate to the "best" forecast run.
        Instead, we keep all available forecast runs (init_times) to allow the caller to perform
        an explicit temporal alignment join. This naturally prevents row multiplication and lookahead bias.
        """
        if nwp_lf is None:
            return None, None

        processed_nwp = calculate_nwp_lead_time(nwp_lf)

        weather_lag_requested = any(lag_feat.base_col != "power" for lag_feat in parsed_features.lags)

        historical_weather = None
        if weather_lag_requested:
            # For weather lags, we use the control member (ensemble_member == 0)
            # and select the "freshest" forecast run (shortest lead time) for each valid_time.
            historical_weather = (
                processed_nwp.filter(pl.col("ensemble_member") == 0)
                .drop("ensemble_member")
                .group_by(["time_series_id", "valid_time"])
                .agg(pl.all().sort_by("nwp_lead_time_hours").first())
                .sort(["time_series_id", "valid_time"])
            )

        return processed_nwp, historical_weather
    ```

---

### Step 4: Update `engineer_features` Signature, Join, and Defensive Check
Update `engineer_features` to accept `power_fcst_init_time` and `nwp_publication_delay_hours`. Implement the unified join logic and the defensive check.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update signature and implement the unified join logic and defensive check:
    ```python
    def engineer_features(
        selected_features: set[str],
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpOnDisk] | None = None,
        power_fcst_init_time: datetime | None = None,
        nwp_publication_delay_hours: int = 6,
    ) -> pt.LazyFrame[AllFeatures]:
        """Engineer features.

        Args:
            selected_features: Set of features to engineer.
            power_time_series: Input power time series.
            time_series_metadata: Metadata for the time series.
            nwp: NWP weather forecast data.
            power_fcst_init_time: The initialization time of the power forecast run.
                If provided (production/backtest mode), we use it to filter/join the NWP data.
                If None (bulk training mode), we derive it internally using the Once-a-Day Alignment rule.
            nwp_publication_delay_hours: The delay in hours between the initialization time
                of the NWP forecast and when it becomes publicly available for use in power
                forecasting.
        """
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
        # First, join power observations with metadata
        power_with_metadata = power_lf.join(metadata_lf, on="time_series_id", how="left")

        if power_fcst_init_time is not None:
            # Production/backtest mode: we have a specific power_fcst_init_time
            # Derive the corresponding nwp_init_time
            nwp_init_time_val = power_fcst_init_time - timedelta(hours=nwp_publication_delay_hours)
            
            # Add these as columns to power_with_metadata
            power_with_metadata = power_with_metadata.with_columns(
                power_fcst_init_time=pl.lit(power_fcst_init_time),
                nwp_init_time=pl.lit(nwp_init_time_val)
            )
            
            if processed_nwp is not None:
                # Perform explicit temporal alignment join on ["time_series_id", "valid_time", "nwp_init_time"]
                # This naturally prevents row multiplication because we only match the specific NWP run
                raw_data = power_with_metadata.join(
                    processed_nwp,
                    on=["time_series_id", "valid_time", "nwp_init_time"],
                    how="left"
                )
            else:
                # Fallback if no NWP is provided
                raw_data = power_with_metadata.with_columns(
                    nwp_lead_time_hours=pl.lit(None, dtype=pl.Float32),
                    ensemble_member=pl.lit(None, dtype=pl.UInt8)
                )
        else:
            # Bulk training mode: power_fcst_init_time is None
            if processed_nwp is not None:
                # Derive power_fcst_init_time from nwp_init_time using Once-a-Day Alignment
                processed_nwp = processed_nwp.with_columns(
                    power_fcst_init_time=pl.col("nwp_init_time") + pl.duration(hours=nwp_publication_delay_hours)
                )
                
                # Join processed_nwp with power_with_metadata on ["time_series_id", "valid_time"]
                raw_data = processed_nwp.join(
                    power_with_metadata,
                    on=["time_series_id", "valid_time"],
                    how="left"
                )
            else:
                # Fallback if no NWP is provided
                raw_data = power_with_metadata.with_columns(
                    power_fcst_init_time=pl.col("valid_time"),
                    nwp_init_time=pl.lit(None, dtype=pl.Datetime(time_unit="us", time_zone="UTC")),
                    nwp_lead_time_hours=pl.lit(None, dtype=pl.Float32),
                    ensemble_member=pl.lit(None, dtype=pl.UInt8)
                )

        # Defensive check: fail loudly if duplicates are detected on the primary key
        pk_cols = ["time_series_id", "power_fcst_init_time", "valid_time"]
        if "ensemble_member" in raw_data.collect_schema().names():
            pk_cols.append("ensemble_member")

        is_duplicated = raw_data.select(pk_cols).is_duplicated().any().collect().item()
        if is_duplicated:
            raise ValueError(f"Duplicate rows detected on primary key: {pk_cols}")

        # Apply Features
        engineered_lf = _apply_post_join_features(
            raw_data,
            parsed_features,
            processed_nwp=processed_nwp,
            historical_weather=historical_weather
        )

        # Schema Assertion and Selection
        available_columns = engineered_lf.collect_schema().names()
        missing_cols = set(selected_features) - set(available_columns)
        if missing_cols:
            if "nwp_lead_time_hours" in missing_cols and "nwp_lead_time_hours" not in selected_features:
                missing_cols.remove("nwp_lead_time_hours")
            if missing_cols:
                raise ValueError(f"Feature engineering failed to create or find: {missing_cols}")

        base_cols = [
            "valid_time",
            "time_series_id",
            "time_series_type",
            "power",
            "power_fcst_init_time",
            "nwp_init_time",
        ]
        if "nwp_lead_time_hours" in engineered_lf.collect_schema().names():
            base_cols.append("nwp_lead_time_hours")
        if "ensemble_member" in engineered_lf.collect_schema().names():
            base_cols.append("ensemble_member")

        cols_to_select = list(set(base_cols + list(selected_features)))
        final_lf = engineered_lf.select(cols_to_select)

        return pt.LazyFrame.from_existing(final_lf).set_model(AllFeatures)
    ```

---

### Step 5: Update `apply_lag_feature` and `_apply_post_join_features`
Update `_apply_post_join_features` to pass `processed_nwp` and `historical_weather` to `apply_lag_feature`. Update `apply_lag_feature` to implement the dual-strategy for weather lags.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update `_apply_post_join_features` and `apply_lag_feature`:
    ```python
    def _apply_post_join_features(
        raw_data: pl.LazyFrame,
        parsed_features: ParsedFeatures,
        processed_nwp: pl.LazyFrame | None = None,
        historical_weather: pl.LazyFrame | None = None,
    ) -> pl.LazyFrame:
        """Applies requested features dynamically based on parsed feature configurations."""
        engineered_lf = raw_data

        # Static and Local Time
        if parsed_features.time_features:
            engineered_lf = apply_local_time_features(engineered_lf)

        if parsed_features.static_features:
            exprs = [STATIC_FEATURE_REGISTRY[f] for f in parsed_features.static_features]
            engineered_lf = engineered_lf.with_columns(exprs)

        # Lags
        for lag_feat in parsed_features.lags:
            if lag_feat.base_col == "power":
                engineered_lf = apply_lag_feature(
                    engineered_lf,
                    source_lf=engineered_lf,
                    lag_feature=lag_feat,
                )
            else:
                # Weather lag: use the dual-strategy
                engineered_lf = apply_lag_feature(
                    engineered_lf,
                    source_lf=processed_nwp,
                    lag_feature=lag_feat,
                    historical_weather_lf=historical_weather,
                )

        # Rolling Means
        for rolling_feat in parsed_features.rolling_means:
            if rolling_feat.base_col in engineered_lf.collect_schema().names():
                engineered_lf = apply_rolling_mean_feature(
                    engineered_lf, rolling_feat.base_col, rolling_feat.hours
                )

        # Nullify leaky lags
        leaky_features = parsed_features.get_leaky_features()
        if leaky_features and "power_fcst_init_time" in engineered_lf.collect_schema().names():
            engineered_lf = nullify_leaky_lags(engineered_lf, leaky_features)

        return engineered_lf


    def apply_lag_feature(
        target_lf: pl.LazyFrame,
        source_lf: pl.LazyFrame,
        lag_feature: LagFeature,
        historical_weather_lf: pl.LazyFrame | None = None,
    ) -> pl.LazyFrame:
        """Applies a lag feature using a time-aware lazy self-join.

        For power lags, we use an exact join on valid_time - lag_hours.
        For weather lags, we implement a dual-strategy:
          - If target_time > power_fcst_init_time (future relative to power forecast run time),
            we use the exact same NWP run (nwp_init_time) and ensemble member as the weather used for valid_time.
          - If target_time <= power_fcst_init_time (past relative to power forecast run time),
            we use the "freshest" NWP run (control member, ensemble 0) for that target time.
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
            # Weather lag: dual-strategy
            lf_with_target_time = target_lf.with_columns(
                target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
            )

            # Join 1: Same-Run Join (for target_time > power_fcst_init_time)
            join_keys_same = ["time_series_id", "nwp_init_time"]
            if "ensemble_member" in target_lf.collect_schema().names() and "ensemble_member" in source_lf.collect_schema().names():
                join_keys_same.append("ensemble_member")

            right_same_lf = source_lf.select(
                *join_keys_same,
                pl.col("valid_time").alias("target_time"),
                pl.col(base_col).alias(f"{lag_feature.string_repr}_same_run"),
            )

            lf_joined = lf_with_target_time.join(
                right_same_lf,
                on=join_keys_same + ["target_time"],
                how="left",
            )

            # Join 2: Freshest-Run Join (for target_time <= power_fcst_init_time)
            if historical_weather_lf is not None:
                right_freshest_lf = historical_weather_lf.select(
                    "time_series_id",
                    pl.col("valid_time").alias("target_time"),
                    pl.col(base_col).alias(f"{lag_feature.string_repr}_freshest_run"),
                )

                lf_joined = lf_joined.join(
                    right_freshest_lf,
                    on=["time_series_id", "target_time"],
                    how="left",
                )
            else:
                lf_joined = lf_joined.with_columns(
                    **{f"{lag_feature.string_repr}_freshest_run": pl.lit(None, dtype=pl.Float32)}
                )

            # Combine using the dual-strategy
            return lf_joined.with_columns(
                pl.when(pl.col("target_time") > pl.col("power_fcst_init_time"))
                .then(pl.col(f"{lag_feature.string_repr}_same_run"))
                .otherwise(pl.col(f"{lag_feature.string_repr}_freshest_run"))
                .alias(lag_feature.string_repr)
            ).drop([f"{lag_feature.string_repr}_same_run", f"{lag_feature.string_repr}_freshest_run", "target_time"])
    ```

---

### Step 6: Update `nullify_leaky_lags`
Update `nullify_leaky_lags` to calculate `power_lead_time_hours` dynamically and use it to nullify power lags.

- **File:** `packages/ml_core/src/ml_core/features.py`
- **Changes:**
  - Update `nullify_leaky_lags`:
    ```python
    def nullify_leaky_lags(
        lf: pl.LazyFrame, leaky_features: Sequence[LagFeature | RollingFeature]
    ) -> pl.LazyFrame:
        """Nullifies lagged features that would cause lookahead bias.

        During training, we must ensure that the model cannot access actual data that
        would not be available at inference time. If a requested lag is shorter than
        or equal to the forecast lead time, the feature is effectively a "future"
        value and must be nullified.

        To prevent over-nullification (FLAW-001), we calculate power_lead_time_hours
        relative to power_fcst_init_time, not nwp_init_time.
        """
        # Calculate power_lead_time_hours dynamically
        lf = lf.with_columns(
            power_lead_time_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
            ).cast(pl.Float32)
        )

        for feature in leaky_features:
            lf = lf.with_columns(
                pl.when(pl.col("power_lead_time_hours") >= feature.hours)
                .then(pl.lit(None))
                .otherwise(pl.col(feature.string_repr))
                .alias(feature.string_repr)
            )

        # Drop power_lead_time_hours to keep the schema clean
        return lf.drop("power_lead_time_hours")
    ```

---

### Step 7: Update Configs & Tests
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
  - **Code Snippet for New Test:**
    ```python
    def test_engineer_features_weather_lag_leakage_prevention():
        # Create dummy data to verify weather lag leakage prevention
        valid_time = datetime(2026, 6, 11, 12, 0)
        power_fcst_init_time = datetime(2026, 6, 10, 6, 0)
        nwp_init_time = datetime(2026, 6, 10, 0, 0)

        # NWP data has two runs:
        # 1. Run initialized at 2026-06-10 00:00:00 (the one we should use)
        # 2. Run initialized at 2026-06-10 12:00:00 (future run relative to power_fcst_init_time!)
        nwp_df = pl.DataFrame(
            {
                "time_series_id": ["ts1", "ts1", "ts1", "ts1"],
                "valid_time": [
                    valid_time,  # target time
                    valid_time - timedelta(hours=2),  # lag target time (2026-06-11 10:00:00)
                    valid_time - timedelta(hours=2),  # lag target time (2026-06-11 10:00:00)
                    valid_time - timedelta(hours=36),  # lag target time (2026-06-10 00:00:00)
                ],
                "ensemble_member": [0, 0, 0, 0],
                "init_time": [
                    nwp_init_time,  # run 1
                    nwp_init_time,  # run 1 (same run)
                    datetime(2026, 6, 10, 12, 0),  # run 2 (future run!)
                    nwp_init_time,  # run 1
                ],
                "temperature_2m": [
                    15.0,  # temp at valid_time
                    10.0,  # temp at lag target time in run 1
                    12.0,  # temp at lag target time in run 2 (future run!)
                    8.0,   # temp at lag target time (36h lag)
                ],
            }
        )

        power_df = pl.DataFrame(
            {
                "time_series_id": ["ts1"],
                "time": [valid_time],
                "power": [100.0],
            }
        )

        metadata_df = pl.DataFrame(
            {
                "time_series_id": ["ts1"],
                "time_series_type": ["substation"],
            }
        )

        # Run engineer_features in production/backtest mode
        engineered = engineer_features(
            power_time_series=pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
            time_series_metadata=pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
            nwp=pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpOnDisk),
            selected_features={"temperature_2m", "temperature_2m_lag_2h", "temperature_2m_lag_36h"},
            power_fcst_init_time=power_fcst_init_time,
            nwp_publication_delay_hours=6,
        ).collect()

        # Verify that temperature_2m_lag_2h is 10.0 (from run 1), NOT 12.0 (from run 2)
        # This proves that the same-run join was used for target_time > power_fcst_init_time
        assert engineered["temperature_2m_lag_2h"][0] == 10.0

        # Verify that temperature_2m_lag_36h is 8.0 (from freshest run)
        assert engineered["temperature_2m_lag_36h"][0] == 8.0
    ```

---

## 5. Code Commenting & Quality Standards

- **Rigorous Commenting:** All code comments must focus on the *why* (intent and rationale) rather than the *how* (obvious implementation). Comments must "connect the dots" across the codebase, explaining how new components relate to existing ones.
- **No FLAW IDs in Comments:** The review markdown files (and their FLAW-XXX IDs) are temporary and will be deleted after the PR is merged. The Builder is strictly forbidden from referencing FLAW-XXX IDs in code comments.

---

## 6. Review Responses & Rejections

* **FLAW-001 (Scientist):** ACCEPTED. We will calculate `power_lead_time_hours` relative to `power_fcst_init_time` instead of `nwp_lead_time_hours` to prevent over-nullification of lags that are actually available at `power_fcst_init_time`.
* **FLAW-002 (Scientist):** ACCEPTED. We will implement the dual-strategy for weather lags using a same-run join for future target times and a freshest-run join for past target times.
