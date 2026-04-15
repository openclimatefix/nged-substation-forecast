import dagster as dg
import polars as pl
from upath import UPath
from datetime import timedelta


class DeltaPolarsIOManager(dg.ConfigurableIOManager):
    """
    Universal IO Manager for reading and writing Polars DataFrames to Delta Lake.

    Write Strategies (defined in asset metadata 'delta_write_mode'):
    - 'incremental_append': Finds the max timestamp per entity and appends only new rows.
    - 'overwrite': Performs a partition overwrite (or full overwrite if unpartitioned).
    """

    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: pl.DataFrame) -> None:
        # Allow assets to override the physical table name (useful for multiple assets writing to one table)
        table_name = context.metadata.get("delta_table_name", context.asset_key.path[-1])
        target_path = str(UPath(self.base_path) / table_name)

        write_mode = context.metadata.get("delta_write_mode", "overwrite")

        # Clean switch statement for write strategies
        if write_mode == "incremental_append":
            self._write_incremental_append(context, obj, target_path)
        elif write_mode == "overwrite":
            self._write_overwrite(context, obj, target_path)
        else:
            raise ValueError(f"Unknown delta_write_mode: {write_mode}")

    def _write_incremental_append(
        self, context: dg.OutputContext, obj: pl.DataFrame, target_path: str
    ) -> None:
        """
        Appends only new data by checking the 'High Water Mark' per entity.

        WHY INCREMENTAL APPEND? MERGE is computationally expensive as it requires
        rewriting Parquet files. Append is blazing fast. By checking the max timestamp
        *per time_series_id*, we safely handle asynchronous reporting delays without
        dropping data.
        """
        append_keys = context.metadata.get("delta_append_keys")
        if not append_keys or len(append_keys) != 2:
            raise ValueError("delta_append_keys must be provided as [entity_id_col, timestamp_col]")

        entity_col, time_col = append_keys

        if UPath(target_path).exists():
            # Find the max timestamp for EACH entity
            max_times = (
                pl.scan_delta(target_path)
                .group_by(entity_col)
                .agg(pl.col(time_col).max().alias("max_time"))
                .collect()
            )

            # Keep only rows newer than the max_time (or new entities)
            obj = obj.join(max_times, on=entity_col, how="left")
            obj = obj.filter(
                pl.col("max_time").is_null() | (pl.col(time_col) > pl.col("max_time"))
            ).drop("max_time")

        if not obj.is_empty():
            obj.write_delta(target_path, mode="append")
            context.log.info(f"Appended {len(obj)} new rows to {target_path}")
        else:
            context.log.info("No new rows to append.")

    def _write_overwrite(
        self, context: dg.OutputContext, obj: pl.DataFrame, target_path: str
    ) -> None:
        """
        Overwrites the target table or specific partitions.

        WHY OVERWRITE? Used for derived assets (like cleaned data). If our cleaning
        logic changes and drops a row, OVERWRITE ensures that row is actually removed
        from the database.
        """
        write_options = {}

        if context.has_asset_partitions:
            start, end = context.asset_partitions_time_window
            time_col = context.metadata.get("delta_partition_time_col", "period_end_time")
            predicate = f"{time_col} >= '{start}' AND {time_col} < '{end}'"
            write_options["predicate"] = predicate

        partition_by = context.metadata.get("delta_partition_by")
        if partition_by:
            write_options["partition_by"] = partition_by

        obj.write_delta(
            target_path,
            mode="overwrite",
            delta_write_options=write_options if write_options else None,
        )
        context.log.info(f"Overwrote data in {target_path}")

    def load_input(self, context: dg.InputContext) -> pl.DataFrame:
        # Allow upstream assets to have written to a different physical table name
        table_name = context.upstream_output.metadata.get(
            "delta_table_name", context.asset_key.path[-1]
        )
        source_path = str(UPath(self.base_path) / table_name)

        lf = pl.scan_delta(source_path)

        if context.has_asset_partitions:
            start, end = context.asset_partitions_time_window

            # Support lookback for rolling features (e.g., rolling std dev in cleaning)
            lookback_days = context.upstream_output.metadata.get("delta_read_lookback_days", 0)
            if lookback_days > 0:
                start = start - timedelta(days=lookback_days)

            time_col = context.upstream_output.metadata.get(
                "delta_partition_time_col", "period_end_time"
            )
            lf = lf.filter(pl.col(time_col).is_between(start, end, closed="left"))

        return lf.collect()
