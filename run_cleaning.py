from dagster import build_asset_context
from nged_substation_forecast.defs.cleaned_power_time_series_assets import cleaned_power_time_series
from nged_substation_forecast.defs.partitions import SIX_HOURLY_PARTITIONS
from contracts.settings import Settings

settings = Settings()
# Get partitions
partitions = SIX_HOURLY_PARTITIONS.get_partition_keys()

for partition in partitions:
    print(f"Cleaning partition: {partition}")
    context = build_asset_context(partition_key=partition)
    cleaned_power_time_series(context, settings)
