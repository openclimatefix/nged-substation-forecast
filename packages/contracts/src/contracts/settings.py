from pathlib import Path
from typing import Final

import obstore
from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    production_model_run_id: str | None = Field(
        default=None,
        description=(
            "MLflow run ID of the model the production service serves."
            " Downloaded once into the local model cache (model_cache_base_path) and reused."
            " Set manually for now; a later champion_model asset will populate it"
            " automatically."
        ),
    )

    cv_config_path: Path = Field(
        default=PROJECT_ROOT / "conf" / "cv" / "default.yaml",
        description=(
            "Path to the canonical cross-validation fold definitions. These folds are the"
            " shared leaderboard evaluation protocol; read them from here, never hard-coded."
        ),
    )

    # Paths to the data we manage
    nged_data_path: Path = PROJECT_ROOT / "data" / "NGED"
    nwp_data_path: Path = PROJECT_ROOT / "data" / "NWP"
    power_forecasts_data_path: Path = PROJECT_ROOT / "data" / "power_forecasts"
    forecast_metrics_data_path: Path = PROJECT_ROOT / "data" / "forecast_metrics"
    plots_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "plots",
        description=(
            "Directory where plot_power_forecast writes interactive forecast HTML files, one per"
            " materialisation (filename derived from experiment, fold, init time, and the plotted"
            " time_series_ids)."
        ),
    )
    eligible_time_series_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "eligible_time_series",
        description=(
            "Delta table of the canonical per-fold eligible time_series_id population, written"
            " by the eligible_time_series asset (partitioned by fold_id) and read by"
            " trained_cv_model and cv_power_forecasts so every experiment scores a fold on the"
            " identical, experiment-independent population."
        ),
    )
    trained_ml_model_params_base_path: Path = PROJECT_ROOT / "data" / "trained_ML_model_params"
    h3_grid_weights_path: Path = PROJECT_ROOT / "data" / "h3_grid_weights.parquet"
    model_cache_base_path: Path = Field(
        default=PROJECT_ROOT / "data" / "model_cache",
        description=(
            "Root of the local-disk model cache, keyed by MLflow run ID."
            " Put this on a persistent volume so the production cache survives restarts"
            " during an MLflow outage."
        ),
    )

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
