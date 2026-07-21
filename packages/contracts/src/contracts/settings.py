from functools import lru_cache
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
            " (like cv_config_path), so it stays a local Path even when the data tables are a"
            " remote URI."
        ),
    )

    # --- Storage roots -------------------------------------------------------------------
    #
    # data_path_internal and data_path_delivery hold the (S3-capable) data tables; local_
    # artifacts_path holds the always-local model cache and production model. Why they
    # are split, and what belongs in each:
    # https://openclimatefix.github.io/nged-substation-forecast/live_service/setup/

    data_path_internal: str = Field(
        default=str(PROJECT_ROOT / "data"),
        description=(
            "Root of OCF's own working data tables (NWP, power, forecast_metrics, …). A local"
            " path by default; may be a remote URI (e.g. 's3://bucket/data') so the data tables"
            " live on S3. Every *_data_path below that is left unset derives from this root,"
            " except the NGED-facing delivery tables, which derive from data_path_delivery"
            " instead."
        ),
    )
    data_path_delivery: str = Field(
        default=str(PROJECT_ROOT / "data"),
        description=(
            "Root of the NGED-facing delivery tables (power_forecasts, effective_capacity, …)."
            " Defaults to the same local path as data_path_internal, so local dev sees one"
            " directory and the split is invisible; on AWS this points at the separate,"
            " NGED-facing bucket."
        ),
    )
    local_artifacts_path: str = Field(
        default=str(PROJECT_ROOT / "data"),
        description=(
            "Root of the always-local artifacts (model cache, production model). Kept"
            " separate from the data-table roots because these back local-filesystem-only"
            " libraries and must stay local even when the data tables live on S3."
        ),
    )

    # --- Object-store credentials for the data tables (used only when a data-path root is remote)
    #
    # All empty by default; unset on AWS (object_store auto-discovers the IAM-role credentials),
    # set only for a dev/MinIO endpoint. The AWS/dev split, and how these differ from the
    # nged_s3_bucket_* source-bucket creds above:
    # https://openclimatefix.github.io/nged-substation-forecast/live_service/setup/

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
        region — and empty for a local data-path root (delta-rs ignores it there). Populated from
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

    # --- Managed data tables (derive from a root unless explicitly set) --------------------
    #
    # Each defaults to "" — a sentinel meaning "derive from data_path_internal or
    # data_path_delivery in _derive_unset_paths". The derive-from-root convention and per-table
    # overrides: https://openclimatefix.github.io/nged-substation-forecast/live_service/setup/

    nged_data_path: str = ""
    """Directory holding the NGED power_time_series Delta table and metadata parquet."""
    nwp_data_path: str = ""
    """Delta table of NWP weather data."""
    power_forecasts_data_path: str = ""
    """Delta table of power forecasts (partitioned by experiment_name, fold_id).

    An NGED-facing delivery table — derives from data_path_delivery, not data_path_internal.
    """
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
            " denominator. An NGED-facing delivery table — derives from data_path_delivery, not"
            " data_path_internal."
        ),
    )
    h3_grid_weights_path: str = ""
    """Parquet file of fractional H3 cell overlap with the GB boundary."""

    # --- Always-local artifacts (derive from local_artifacts_path unless explicitly set) --

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

    # --- Sentry observability (all optional; an empty DSN disables Sentry entirely) ---

    sentry_dsn: str = Field(
        default="",
        description=(
            "Sentry.io project DSN. Empty (the default) disables Sentry entirely — every Sentry"
            " code path becomes a no-op — so laptops and CI need no Sentry config. Set it in the"
            " .env file to enable error telemetry. Passed explicitly to sentry_sdk.init as the"
            " single source of truth (the SDK would otherwise also auto-read the SENTRY_DSN env"
            " var)."
        ),
    )
    sentry_environment: str = Field(
        default="local",
        description=(
            "Sentry environment tag, the dimension that separates deployments in Sentry's UI,"
            " alerts, and cron monitors. The always-on production box sets 'production'; each"
            " developer overrides the 'local' fallback with '<name>-laptop' (e.g. 'jacks-laptop')"
            " so laptop telemetry is cleanly filterable and the production-scoped missed-check-in"
            " alert never fires for a laptop."
        ),
    )
    sentry_traces_sample_rate: float = Field(
        default=0.0,
        description="Fraction of transactions sampled for Sentry performance tracing (off by default).",
    )
    sentry_monitor_forecasts: bool = Field(
        default=False,
        description=(
            "Emit the live_forecasts success heartbeat to Sentry's cron monitor (the"
            " missed-check-in alarm). True only on the always-on production deployment; left"
            " False on laptops so an intermittently-run laptop never registers a monitor"
            " environment that Sentry would then mark as missed."
        ),
    )

    @model_validator(mode="after")
    def _derive_unset_paths(self) -> Self:
        """Fill any unset ("") path from its root, so callers always see a concrete path.

        The default layout lives here and nowhere else. A field set explicitly (e.g. via its
        env var) keeps its value; only the "" sentinels are derived.
        """
        self.nged_data_path = self.nged_data_path or uri_join(self.data_path_internal, "NGED")
        self.nwp_data_path = self.nwp_data_path or uri_join(self.data_path_internal, "NWP")
        self.forecast_metrics_data_path = self.forecast_metrics_data_path or uri_join(
            self.data_path_internal, "forecast_metrics"
        )
        self.eligible_time_series_data_path = self.eligible_time_series_data_path or uri_join(
            self.data_path_internal, "eligible_time_series"
        )
        self.h3_grid_weights_path = self.h3_grid_weights_path or uri_join(
            self.data_path_internal, "h3_grid_weights.parquet"
        )
        # NGED-facing delivery tables (see docs/roadmap/delivery-tables.md) derive from
        # data_path_delivery instead, so a new delivery table can't silently land in the
        # internal bucket by inheriting the default derivation.
        self.power_forecasts_data_path = self.power_forecasts_data_path or uri_join(
            self.data_path_delivery, "power_forecasts"
        )
        self.effective_capacity_data_path = self.effective_capacity_data_path or uri_join(
            self.data_path_delivery, "effective_capacity"
        )
        # Derived from nged_data_path (itself derived above).
        self.power_time_series_data_path = self.power_time_series_data_path or uri_join(
            self.nged_data_path, "power_time_series.delta"
        )
        self.metadata_path = self.metadata_path or uri_join(self.nged_data_path, "metadata.parquet")
        # Always-local artifacts.
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


@lru_cache
def get_settings() -> Settings:
    """Return the shared, lazily-constructed ``Settings`` singleton.

    Prefer this over constructing ``Settings()`` at module import time. ``Settings`` has required
    fields (the ``nged_s3_bucket_*`` credentials) with no defaults, so instantiation reads ``.env``
    and raises ``ValidationError`` when those are absent. Deferring construction to first *use*
    keeps library modules (e.g. the ``contracts`` schemas) importable without live credentials —
    so type-checkers, doc builders, and tests that only touch in-memory frames don't need a
    populated ``.env`` — while still failing fast the moment settings are actually needed.

    Cached with ``lru_cache`` so every caller shares one instance (matching the previous
    module-level singletons). Call ``get_settings.cache_clear()`` in a test that needs to
    re-read the environment after changing it.
    """
    return Settings()
