"""Dagster definitions for the NGED substation forecast project."""

from pathlib import Path

from dagster import Definitions, load_from_defs_folder
from contracts.config import Settings
from nged_substation_forecast.config_resource import NgedConfig


def create_defs() -> Definitions:
    """Create Dagster definitions."""
    settings = Settings()
    config_resource = NgedConfig(**settings.model_dump(mode="json"))

    return Definitions(
        assets=load_from_defs_folder(path_within_project=Path(__file__).parent).assets,
        resources={"nged_config": config_resource},
    )


defs = create_defs()
