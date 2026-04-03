import dagster as dg

DAILY_PARTITIONS = dg.DailyPartitionsDefinition(start_date="2026-01-26", end_offset=1)
model_partitions = dg.DynamicPartitionsDefinition(name="model_partitions")
