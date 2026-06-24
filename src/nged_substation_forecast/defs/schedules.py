"""Dagster schedules for the NGED Substation Forecast project."""

from dagster import AssetSelection, ScheduleDefinition, define_asset_job

# Define a job that targets the power_time_series_and_metadata asset
power_time_series_and_metadata_job = define_asset_job(
    name="power_time_series_and_metadata_job",
    selection=AssetSelection.assets("power_time_series_and_metadata"),
)

# Define a schedule that runs the job every hour
power_time_series_and_metadata_schedule = ScheduleDefinition(
    name="power_time_series_and_metadata_schedule",
    job=power_time_series_and_metadata_job,
    cron_schedule="0 * * * *",
)
