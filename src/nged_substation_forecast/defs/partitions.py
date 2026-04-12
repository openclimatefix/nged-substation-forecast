import dagster as dg

DAILY_PARTITIONS = dg.DailyPartitionsDefinition(start_date="2026-01-26", end_offset=1)
SIX_HOURLY_PARTITIONS = dg.TimeWindowPartitionsDefinition(
    cron_schedule="0 */6 * * *",
    start="2026-01-26-00:00",
    fmt="%Y-%m-%d-%H:%M",
    timezone="UTC",
)
model_partitions = dg.DynamicPartitionsDefinition(name="model_partitions")
