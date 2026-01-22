"""Definitions for Dagster assets and resources."""

from dagster import load_assets_from_modules

from . import nged_assets

nged_data_assets = load_assets_from_modules([nged_assets])
