import dagster as dg

model_partitions = dg.DynamicPartitionsDefinition(name="model_partitions")
