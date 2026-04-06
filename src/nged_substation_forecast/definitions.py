"""Dagster definitions for the NGED substation forecast project."""

from pathlib import Path

from dagster import Definitions, ResourceDefinition, load_from_defs_folder
from geo.io_managers import CompositeIOManager
from upath import UPath

from contracts.settings import Settings


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
            "io_manager": CompositeIOManager(base_path=UPath("/home/jack/dagster_home/storage")),
        },
    )


defs = create_defs()
