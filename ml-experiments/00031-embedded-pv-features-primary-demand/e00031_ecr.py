"""The Embedded Capacity Register's solar capacity under each demand series.

A Primary series is "Disaggregated Demand": NGED has already subtracted its metered generators, so
the register rows that have their own meter are not in the series. The register is therefore a
prior for a Primary. The prior counts only the rows on the 50 kW to 1 MW sheet, because the
metadata table lists none of the larger solar farms under a Primary, and the rows of 1 MW or more
are assumed to be metered; NGED's list of meters is not available to confirm that. A BSP or GSP
series is "Raw Flow": the metered generators below it are not subtracted, so every connected
solar row below the substation is a floor on the PV in the series.
"""

from pathlib import Path
from typing import Final

import e00031_ecr_reader as reader
import polars as pl
from studies.name_matching import normalise
from studies.sources import REPO_DATA_DIR

HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
SMALL_SHEET: Final[str] = "Register Part 1 50kW - <1MW"


def _solar_rows() -> pl.DataFrame:
    """Return the connected solar rows of the register with their normalised names."""
    ecr = reader.read_ecr()
    return ecr.filter(
        (pl.col("connection_status") == "Connected")
        & (pl.col("energy_source_1") == "Solar")
        & pl.col("connected_registered_capacity_mw").is_not_null()
    ).with_columns(
        primary_key=pl.col("primary_substation")
        .fill_null("")
        .map_elements(normalise, return_dtype=pl.String),
        bsp_key=pl.col("bulk_supply_point")
        .fill_null("")
        .map_elements(normalise, return_dtype=pl.String),
        gsp_key=pl.col("grid_supply_point")
        .fill_null("")
        .map_elements(normalise, return_dtype=pl.String),
    )


def ecr_table(*, series: pl.DataFrame, decisions_path: Path | None = None) -> pl.DataFrame:
    """Return the register's solar capacity under each series, in MW and as a share of capacity.

    Args:
        series: `common.series_table()`.
        decisions_path: Where to write the per-row decision list (private), or None.

    Returns:
        One row per series with `site`, `role` (`prior` for a Primary, `floor` for a BSP or GSP),
        `ecr_mw`, and `ecr_share` (`ecr_mw` over the series' effective capacity).
    """
    rows = _solar_rows()
    rows = rows.with_columns(own_meter=pl.col("sheet") != SMALL_SHEET)
    hierarchy = pl.read_csv(HIERARCHY_DIR / "Substations.csv")
    metadata = pl.read_parquet(METADATA_PATH).select(
        "time_series_id", "time_series_name", "substation_number"
    )
    named = series.join(metadata, on="time_series_id", how="left", validate="1:1").with_columns(
        base_key=pl.col("time_series_name").map_elements(
            lambda name: normalise(name.removesuffix(" T1").removesuffix(" T2")),
            return_dtype=pl.String,
        )
    )
    out = []
    for row in named.iter_rows(named=True):
        if row["substation_type"] == "Primary":
            matched = rows.filter((pl.col("primary_key") == row["base_key"]) & ~pl.col("own_meter"))
            shared = named.filter(pl.col("base_key") == row["base_key"])["effective_capacity_mw"]
            share = row["effective_capacity_mw"] / float(shared.sum())
            role = "prior"
            mw = float(matched["connected_registered_capacity_mw"].sum()) * share
        else:
            column = "bsp_key" if row["substation_type"] == "BSP" else "gsp_key"
            under = hierarchy.filter(
                pl.col("substation_number") == row["substation_number"]
            )  # the substation's own row names its BSP and GSP
            name_column = "bsp_name" if row["substation_type"] == "BSP" else "gsp_name"
            key = normalise(under[name_column][0]) if under.height else row["base_key"]
            matched = rows.filter(pl.col(column) == key)
            role = "floor"
            mw = float(matched["connected_registered_capacity_mw"].sum())
        out.append(
            {
                "site": row["site"],
                "role": role,
                "ecr_mw": mw,
                "ecr_share": mw / row["effective_capacity_mw"],
                "rows": matched.height,
            }
        )
    if decisions_path is not None:
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        rows.select(
            "primary_substation",
            "bulk_supply_point",
            "grid_supply_point",
            "sheet",
            "connected_registered_capacity_mw",
            "own_meter",
        ).write_csv(decisions_path)
    return pl.DataFrame(out)


def directional_sites(*, series: pl.DataFrame) -> frozenset[str]:
    """Return the sites whose meter reads a signed flow, so PV above demand does not fold.

    Args:
        series: `common_series_table`-style frame with `time_series_id` and `site`.

    Returns:
        The site labels whose "Directional Monitoring?" flag is "Yes".
    """
    direction = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        time_series_name="TimeSeriesName", directional="Directional Monitoring?"
    )
    metadata = pl.read_parquet(METADATA_PATH).select("time_series_id", "time_series_name")
    joined = series.join(metadata, on="time_series_id", how="left", validate="1:1").join(
        direction, on="time_series_name", how="left", validate="m:1"
    )
    return frozenset(joined.filter(pl.col("directional") == "Yes")["site"].to_list())
