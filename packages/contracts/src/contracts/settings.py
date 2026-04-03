from pathlib import Path

from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

url_adapter = TypeAdapter(AnyHttpUrl)


def find_project_root() -> Path:
    """Find the project root by looking for uv.lock."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "uv.lock").exists():
            return parent
    return current


PROJECT_ROOT = find_project_root()


class DataQualitySettings(BaseSettings):
    """Settings for data quality thresholds in substation flow processing.

    These thresholds are used to identify problematic telemetry data:
    - `stuck_std_threshold`: When the rolling standard deviation falls below this value
      (across a 48-period/24-hour window), the sensor is likely stuck. We replace such
      values with null to preserve the temporal grid. A value of 0.01 MW was chosen
      because substations with normal operation typically have much higher variability.

    - `max_mw_threshold`: Active power above this value is considered physically
      unrealistic for primary substations in the NGED portfolio. A threshold of 100.0 MW
      was chosen because typical primary substations operate in the tens of MW range,
      and values exceeding 100 MW are extremely rare anomalies.

    - `min_mw_threshold`: Active power below this value is potentially erroneous
      (negative values can occur at times of high renewable generation). A threshold of
      -20.0 MW was chosen to allow for reverse power flow during high renewable
      generation periods while still catching implausible extreme negative values.

    Centralizing these in Settings allows them to be configurable per environment
    (dev/staging/prod) while preventing logic drift between asset checks and data
    cleaning steps. All code that references these thresholds should import them from
    here, not define them locally.
    """

    stuck_std_threshold: float = 0.01
    stuck_window_periods: int = 48
    max_mw_threshold: float = 100.0
    min_mw_threshold: float = -20.0


class Settings(BaseSettings):
    """Configuration settings for the NGED substation forecast project."""

    # NGED Connected Data CKAN token
    nged_ckan_token: str = Field(...)

    # NWP Data Settings
    nwp_ensemble_member: int = Field(
        default=0,
        description=(
            "Which NWP ensemble member to use for training the ML model (typically 0, the "
            "control member)."
        ),
    )

    # ML Model Settings
    ml_model_ensemble_size: int = Field(
        default=10,
        description=(
            "Number of ML models to train in an ensemble (e.g. using different random seeds) "
            "to improve robustness and provide uncertainty estimates at inference time."
        ),
    )

    # MLflow Tracking URI
    # We centralize the MLflow tracking URI here to allow for environment-specific
    # configuration (e.g., local SQLite for development, remote server for production).
    # The application entrypoint should read this setting and set the MLFLOW_TRACKING_URI
    # environment variable accordingly, which MLflow will automatically pick up.
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking URI.",
    )

    # Data Quality Settings
    data_quality: DataQualitySettings = Field(
        default_factory=DataQualitySettings,
        description="Configurable thresholds for data quality checks.",
    )

    # S3 Storage
    nged_s3_bucket_url: str = Field(...)
    nged_s3_bucket_access_key: str = Field(...)
    nged_s3_bucket_secret: str = Field(...)

    # ECMWF Data Settings
    ecmwf_s3_bucket: str = Field(
        default="dynamical-ecmwf-ifs-ens",
        description="S3 bucket for ECMWF Icechunk store.",
    )
    ecmwf_s3_prefix: str = Field(
        default="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
        description="S3 prefix for ECMWF Icechunk store.",
    )

    # Paths
    nged_data_path: Path = Path("data/NGED")
    nwp_data_path: Path = Path("data/NWP")
    power_forecasts_data_path: Path = Path("data/power_forecasts")
    forecast_metrics_data_path: Path = Path("data/forecast_metrics")
    trained_ml_model_params_base_path: Path = Path("data/trained_ML_model_params")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
        env_file_encoding="utf-8",
        env_prefix="",
    )

    @field_validator("nged_s3_bucket_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that the S3 bucket URL is a valid URL."""
        url_adapter.validate_python(v)
        return v
