from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Find the project root by looking for uv.lock."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "uv.lock").exists():
            return parent
    return current


PROJECT_ROOT = find_project_root()


class Settings(BaseSettings):
    # NGED Connected Data
    # TODO: These should all be lowercase.
    NGED_CKAN_TOKEN: SecretStr = Field(...)

    # S3 Storage
    NGED_S3_BUCKET_URL: HttpUrl = Field(...)
    NGED_S3_BUCKET_ACCESS_KEY: SecretStr = Field(...)
    NGED_S3_BUCKET_SECRET: SecretStr = Field(...)

    # Paths
    NGED_DATA_PATH: Path = Path("data/NGED")
    NWP_DATA_PATH: Path = Path("data/NWP")
    POWER_FORECASTS_DATA_PATH: Path = Path("data/power_forecasts")
    FORECAST_METRICS_DATA_PATH: Path = Path("data/forecast_metrics")
    TRAINED_ML_MODEL_PARAMS_BASE_PATH: Path = Path("data/trained_ML_model_params")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
        env_file_encoding="utf-8",
        env_prefix="",
    )
