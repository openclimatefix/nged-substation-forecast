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


class Settings(BaseSettings):
    """Configuration settings for the NGED substation forecast project."""

    # NGED Connected Data
    nged_ckan_token: str = Field(...)

    # S3 Storage
    nged_s3_bucket_url: str = Field(...)
    nged_s3_bucket_access_key: str = Field(...)
    nged_s3_bucket_secret: str = Field(...)

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
