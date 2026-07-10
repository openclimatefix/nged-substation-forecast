from pathlib import Path
from typing import Final, Self

import obstore
from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from contracts._uri import ObjectStoreOptions, uri_join

url_adapter = TypeAdapter(AnyHttpUrl)


PROJECT_ROOT: Final[Path] = Path(__file__).parents[4]


class DataQualitySettings(BaseSettings):
    """Settings for data quality thresholds in substation flow processing.

    These thresholds are used to identify problematic telemetry data:

    - `stuck_std_threshold`: When the rolling standard deviation falls below this value
      (across `stuck_window_periods`), the sensor is likely stuck. We replace such
      values with null to preserve the temporal grid. A value of 0.01 MW was chosen
      because substations with normal operation typically have much higher variability.

    - `max_mw_threshold`: Active power above this value is considered physically
      unrealistic for primary substations in the NGED portfolio. A threshold of 150 MW
      was chosen because typical primary substations operate in the tens of MW range,
      and values exceeding 100 MW are extremely rare anomalies.

    - `min_mw_threshold`: Active power below this value is potentially erroneous
      (negative values can occur at times of high renewable generation). A threshold of
      -50.0 MW was chosen to allow for reverse power flow during high renewable
      generation periods while still catching implausible extreme negative values.

    Centralizing these in Settings allows them to be configurable per environment
    (dev/staging/prod) while preventing logic drift between asset checks and data
    cleaning steps. All code that references these thresholds should import them from
    here, not define them locally.
    """

    stuck_std_threshold: float = 0.01
    stuck_window_periods: int = 48  # Each period is 30 minutes.
    max_mw_threshold: float = 150.0
    min_mw_threshold: float = -50.0


class Settings(BaseSettings):
    """Configuration settings for the NGED substation forecast project."""

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description=(
            "MLflow tracking URI. Centralized here for environment-specific configuration"
            " (e.g., local SQLite for development, remote server for production)."
            " The application entrypoint should read this setting and set the"
            " MLFLOW_TRACKING_URI environment variable accordingly, which MLflow will"
            " automatically pick up."
        ),
    )

    data_quality: DataQualitySettings = Field(
        default_factory=DataQualitySettings,
        description="Configurable thresholds for data quality checks.",
    )

    nged_s3_bucket_url: str = Field(
        ..., description="NGED S3 bucket URL. Typically stored in the .env file."
    )
    nged_s3_bucket_access_key: str = Field(
        ..., description="Access key for the NGED S3 bucket. Typically stored in the .env file."
    )
    nged_s3_bucket_secret: str = Field(
        ..., description="Secret key for the NGED S3 bucket. Typically stored in the .env file."
    )

    def get_nged_s3_store(self) -> obstore.store.S3Store:
        """Returns an initialized obstore.store.S3Store instance for the NGED bucket."""
        return obstore.store.S3Store.from_url(
            url=self.nged_s3_bucket_url,
            config={
                "aws_access_key_id": self.nged_s3_bucket_access_key,
                "aws_secret_access_key": self.nged_s3_bucket_secret,
            },
        )

    cv_config_path: Path = Field(
        default=PROJECT_ROOT / "conf" / "cv" / "default.yaml",
        description=(
            "Path to the canonical cross-validation fold definitions. These folds are the"
            " shared leaderboard evaluation protocol; read them from here, never hard-coded."
        ),
    )
    nwp_metadata_csv_path: Path = Field(
        default=PROJECT_ROOT / "metadata" / "nwp_metadata.csv",
        description=(
            "Static CSV of per-NWP-model metadata (H3 resolution, provider, ensemble flag),"
            " checked into the repo and read by NwpMetaData.load. A code-relative resource"
            " (like cv_config_path), so it stays a local Path even when data_path is a remote URI."
        ),
    )

    # --- Storage roots -------------------------------------------------------------------
    #
    # Two roots because nothing under local_artifacts_path is part of the S3-backed data plane:
    # the model cache is a node-local scratch cache used only by the (laptop) CV pipeline; the
    # production model is distributed via the container image, not shared storage (this dir is its
    # build-time staging area); and plot HTML is a local-dev convenience (the dashboard, reading
    # power_forecasts from data_path, is the deployed way to view forecasts). So the deployed
    # runtime reads its model from the image, data from S3, and writes forecasts to S3 — it never
    # uses local_artifacts_path as shared storage. See docs/live_service/setup.md.

    data_path: str = Field(
        default=str(PROJECT_ROOT / "data"),
        description=(
            "Root of the managed data tables (NWP, power, forecasts, metrics, …). A local path"
            " by default; may be a remote URI (e.g. 's3://bucket/data') so the data tables live"
            " on S3. Every *_data_path below that is left unset derives from this root."
        ),
    )
    local_artifacts_path: str = Field(
        default=str(PROJECT_ROOT / "data"),
        description=(
            "Root of the always-local artifacts (model cache, production model, plots). Kept"
            " separate from data_path because these back local-filesystem-only libraries and"
            " must stay local even when data_path is a remote URI."
        ),
    )

    # --- Object-store credentials for the data tables (used only when data_path is remote) --
    #
    # All empty by default. On AWS nothing needs setting: delta-rs' object_store discovers the
    # Fargate task's IAM-role credentials and region automatically. Populate these (via env)
    # only for a dev / MinIO / S3-compatible endpoint that needs an explicit endpoint + keys.
    # Separate from the nged_s3_bucket_* creds above, which authenticate reads of NGED's source
    # bucket (a different account/bucket from our own managed data tables).

    data_store_endpoint_url: str = Field(
        default="",
        description=(
            "S3-compatible endpoint URL for the data tables (e.g. a MinIO/dev endpoint). Empty on"
            " AWS, where the endpoint is inferred. When set, the store is also allowed to use"
            " plain HTTP (dev endpoints rarely have TLS)."
        ),
    )
    data_store_access_key_id: str = Field(
        default="", description="Access key for the data-table object store; empty on AWS (IAM)."
    )
    data_store_secret_access_key: str = Field(
        default="", description="Secret key for the data-table object store; empty on AWS (IAM)."
    )
    data_store_region: str = Field(
        default="", description="Region for the data-table object store; empty to auto-discover."
    )

    @property
    def storage_options(self) -> ObjectStoreOptions:
        """delta-rs / polars / obstore ``storage_options`` for the managed data tables.

        Empty on AWS — object_store auto-discovers the Fargate task's IAM-role credentials and
        region — and empty for a local ``data_path`` (delta-rs ignores it there). Populated from
        the ``data_store_*`` settings only for a dev/MinIO/S3-compatible endpoint. The ``aws_*``
        keys are the shared object_store aliases understood by delta-rs, polars cloud IO, and
        obstore alike, so one value feeds every IO site. Returned as an ``ObjectStoreOptions``
        ``TypedDict`` so ``ty`` checks each key here, where they are authored; widen it to a plain
        ``dict`` at each IO boundary with ``typeddict_to_dict``.
        """
        options: ObjectStoreOptions = {}
        if self.data_store_endpoint_url:
            options["aws_endpoint_url"] = self.data_store_endpoint_url
            options["aws_allow_http"] = "true"
        if self.data_store_access_key_id:
            options["aws_access_key_id"] = self.data_store_access_key_id
        if self.data_store_secret_access_key:
            options["aws_secret_access_key"] = self.data_store_secret_access_key
        if self.data_store_region:
            options["aws_region"] = self.data_store_region
        return options

    # --- Managed data tables (derive from data_path unless explicitly set) ----------------
    #
    # Each defaults to "" as a sentinel meaning "derive from data_path in _derive_unset_paths".
    # Set any one explicitly (e.g. NWP_DATA_PATH=s3://other-bucket/NWP) to override just that
    # table. After validation every field below is a concrete, non-empty path string.

    nged_data_path: str = ""
    """Directory holding the NGED power_time_series Delta table and metadata parquet."""
    nwp_data_path: str = ""
    """Delta table of NWP weather data."""
    power_forecasts_data_path: str = ""
    """Delta table of power forecasts (partitioned by experiment_name, fold_id)."""
    forecast_metrics_data_path: str = ""
    """Delta table of forecast evaluation metrics."""
    power_time_series_data_path: str = ""
    """Delta table of half-hourly power observations (under nged_data_path)."""
    metadata_path: str = ""
    """Parquet file of per-series substation metadata (under nged_data_path)."""
    eligible_time_series_data_path: str = Field(
        default="",
        description=(
            "Delta table of the canonical per-fold eligible time_series_id population, written"
            " by the eligible_time_series asset (partitioned by fold_id) and read by"
            " trained_cv_model and cv_power_forecasts so every experiment scores a fold on the"
            " identical, experiment-independent population."
        ),
    )
    effective_capacity_data_path: str = Field(
        default="",
        description=(
            "Delta table of per-series effective capacity (v0.1: full-history P99 of |power|),"
            " written by the effective_capacity asset and read by the metrics asset as the NMAE"
            " denominator."
        ),
    )
    h3_grid_weights_path: str = ""
    """Parquet file of fractional H3 cell overlap with the GB boundary."""

    # --- Always-local artifacts (derive from local_artifacts_path unless explicitly set) --

    plots_data_path: str = Field(
        default="",
        description=(
            "Directory where plot_power_forecast writes interactive forecast HTML files, one per"
            " materialisation (filename derived from experiment, fold, init time, and the plotted"
            " time_series_ids)."
        ),
    )
    model_cache_base_path: str = Field(
        default="",
        description=(
            "Root of the local-disk model cache, keyed by MLflow run ID, used by"
            " BaseForecaster.load_from_mlflow. Currently only the CV pipeline"
            " (cv_power_forecasts) reads a model back through this cache; production"
            " inference does not use it for v0.1 (the model is baked directly into the"
            " container image instead)."
        ),
    )
    production_model_path: str = Field(
        default="",
        description=(
            "Directory holding the current production model, written by the promoted_model"
            " asset (ml_core._production_helpers.fetch_model_artifacts) and read by"
            " live_forecasts via a plain BaseForecaster disk load — no MLflow at inference"
            " time. Later COPY'd into the container image at build time (issue #222)."
        ),
    )

    @model_validator(mode="after")
    def _derive_unset_paths(self) -> Self:
        """Fill any unset ("") path from its root, so callers always see a concrete path.

        The default layout lives here and nowhere else. A field set explicitly (e.g. via its
        env var) keeps its value; only the "" sentinels are derived.
        """
        self.nged_data_path = self.nged_data_path or uri_join(self.data_path, "NGED")
        self.nwp_data_path = self.nwp_data_path or uri_join(self.data_path, "NWP")
        self.power_forecasts_data_path = self.power_forecasts_data_path or uri_join(
            self.data_path, "power_forecasts"
        )
        self.forecast_metrics_data_path = self.forecast_metrics_data_path or uri_join(
            self.data_path, "forecast_metrics"
        )
        self.eligible_time_series_data_path = self.eligible_time_series_data_path or uri_join(
            self.data_path, "eligible_time_series"
        )
        self.effective_capacity_data_path = self.effective_capacity_data_path or uri_join(
            self.data_path, "effective_capacity"
        )
        self.h3_grid_weights_path = self.h3_grid_weights_path or uri_join(
            self.data_path, "h3_grid_weights.parquet"
        )
        # Derived from nged_data_path (itself derived above).
        self.power_time_series_data_path = self.power_time_series_data_path or uri_join(
            self.nged_data_path, "power_time_series.delta"
        )
        self.metadata_path = self.metadata_path or uri_join(self.nged_data_path, "metadata.parquet")
        # Always-local artifacts.
        self.plots_data_path = self.plots_data_path or uri_join(self.local_artifacts_path, "plots")
        self.model_cache_base_path = self.model_cache_base_path or uri_join(
            self.local_artifacts_path, "model_cache"
        )
        self.production_model_path = self.production_model_path or uri_join(
            self.local_artifacts_path, "production_model"
        )
        return self

    # Tell Pydantic to override defaults with fields set in the .env file.
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="forbid",
        env_file_encoding="utf-8",
        env_prefix="",
    )

    @field_validator("nged_s3_bucket_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that the S3 bucket URL is a valid URL."""
        url_adapter.validate_python(v)
        return v
