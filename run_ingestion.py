from dagster import build_asset_context
from contracts.settings import Settings
from nged_substation_forecast.defs.nged_assets import nged_sharepoint_json_asset

context = build_asset_context()
settings = Settings()
nged_sharepoint_json_asset(context, settings)
