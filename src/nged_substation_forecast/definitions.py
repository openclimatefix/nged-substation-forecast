"""Dagster definitions for the NGED substation forecast project."""

from pathlib import Path

from dagster import Definitions, load_from_defs_folder
from contracts.config import Settings
from nged_substation_forecast.config_resource import NgedConfig


def create_defs() -> Definitions:
    """Create Dagster definitions."""
    settings = Settings()

    # Start with JSON-serialized settings (handles Path, HttpUrl, etc.)
    raw_settings = settings.model_dump(mode="json")

    # Override masked secrets with raw values
    for field_name, value in settings.model_dump().items():
        attr = getattr(settings, field_name)
        if hasattr(attr, "get_secret_value"):
            raw_settings[field_name] = attr.get_secret_value()

    config_resource = NgedConfig(**raw_settings)

    return Definitions(
        assets=load_from_defs_folder(path_within_project=Path(__file__).parent).assets,
        resources={"nged_config": config_resource},
    )


defs = create_defs()
