"""Test helper: write the cleaned power table that the readers of observed power scan."""

from pathlib import Path

import polars as pl
from contracts.power_schemas import DROP_REASONS, TimeSeriesMetadata


def write_metadata(path: str | Path, substation_types: dict[int, str]) -> None:
    """Write a minimal valid ``TimeSeriesMetadata`` parquet.

    Args:
        path: Where to write the parquet.
        substation_types: The ``time_series_id`` of each series, mapped to its ``substation_type``.
    """
    rows = [
        {
            "time_series_id": time_series_id,
            "time_series_name": f"Substation {time_series_id}",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": time_series_id,
            "substation_type": substation_type,
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        }
        for time_series_id, substation_type in substation_types.items()
    ]
    TimeSeriesMetadata.DataFrame(rows).cast().validate().write_parquet(path)


def write_cleaned_copy(
    raw_power_path: str | Path,
    flag_where: pl.Expr | None = None,
) -> None:
    """Write ``cleaned_power_time_series.delta`` beside a raw ``power_time_series.delta``.

    Every raw row is copied across, overwriting any cleaned table already there. Rows for which
    ``flag_where`` is true carry the first drop reason in ``DROP_REASONS``, and every other row
    carries a null ``drop_reason``.

    Args:
        raw_power_path: The raw ``power_time_series.delta`` table the test has written.
        flag_where: A boolean expression over the raw columns that is true for a row to flag, or
            ``None`` to flag nothing.
    """
    raw_path = Path(raw_power_path)
    flagged = flag_where if flag_where is not None else pl.lit(False)
    cleaned = pl.read_delta(str(raw_path)).with_columns(
        drop_reason=pl.when(flagged)
        .then(pl.lit(DROP_REASONS[0]))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    cleaned.write_delta(
        str(raw_path.parent / "cleaned_power_time_series.delta"),
        mode="overwrite",
        delta_write_options={"partition_by": ["time_series_id"]},
    )
