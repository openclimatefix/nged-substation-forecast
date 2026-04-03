"""Dagster definitions for the NGED substation forecast project."""

from pathlib import Path

from dagster import Definitions, ResourceDefinition, load_from_defs_folder

from contracts.settings import Settings
from nged_substation_forecast.resources.h3_grid import H3GridResource


def create_defs() -> Definitions:
    """Create Dagster definitions."""
    settings = Settings()
    loaded_defs = load_from_defs_folder(path_within_project=Path(__file__).parent)

    return Definitions(
        assets=loaded_defs.assets,
        asset_checks=loaded_defs.asset_checks,
        jobs=loaded_defs.jobs,
        schedules=loaded_defs.schedules,
        sensors=loaded_defs.sensors,
        resources={
            "settings": ResourceDefinition.hardcoded_resource(settings),
            "h3_grid": H3GridResource(h3_grid_weights_path=str(settings.h3_grid_weights_path)),
        },
    )


defs = create_defs()
