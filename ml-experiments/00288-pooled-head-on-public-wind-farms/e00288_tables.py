"""Assemble the head tables of experiment 00288: the public farms and W1 to W3 in one schema.

Every row is one case (farm, run, valid time) with R0's 11 base quantiles, the seven member
summaries, the lead, and the farm's static inputs (log capacity, turbine rating, hub height).
Public-farm rows come from `e00288_base.py`; W1 to W3's rows are experiment 259's stored R0 rows and
summaries, which score the same 53 Tuesday runs. W1 to W3 take the pool's median turbine rating and
hub height, because their own values are private.
"""

from typing import Final

import e00288_common as common
import e00288_wsites as wsites
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BASE_DIR: Final = common.WORK_DIR / "base"
CLEAN_DIR: Final = common.WORK_DIR / "clean"
W259_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00259-wind-plant-plus-quantile-offset-head" / "work/base"
)
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "ws_mean",
    "ws_std",
    "ws_p10",
    "ws_p90",
    "dir_sin",
    "dir_cos",
    "temp_mean",
)
STATIC_COLUMNS: Final[tuple[str, ...]] = ("log_capacity_mw", "turbine_mw", "hub_height_m")
Q: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
SCHEMA_COLUMNS: Final[tuple[str, ...]] = (
    "group",
    "farm",
    "fold",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    "lead_day",
    "actual_norm",
    "masked",
    *Q,
    *SUMMARY_COLUMNS,
    *STATIC_COLUMNS,
)


def pool_table(*, set_name: str, tag: str = "all", seed: int = 0) -> pl.DataFrame:
    """Return the public farms' cases of a set with R0's quantiles and static inputs."""
    farms = pl.read_parquet(CLEAN_DIR / "farms.parquet")
    cases = pl.read_parquet(BASE_DIR / f"cases_{set_name}.parquet")
    base = pl.read_parquet(BASE_DIR / f"R0_{set_name}_h{tag}_seed{seed}.parquet")
    joined = cases.join(
        base, on=["farm", "power_fcst_init_time", "valid_time"], how="inner", validate="1:1"
    ).join(
        farms.select(
            pl.col("elexon_bmu_id").alias("farm"),
            "fold",
            log_capacity_mw=pl.col("capacity_mw").log(),
            turbine_mw=pl.col("turbine_mw"),
            hub_height_m=pl.col("hub_height_m"),
        ),
        on="farm",
        validate="m:1",
    )
    return joined.with_columns(group=pl.lit("pool"), masked=pl.lit(value=False)).select(
        SCHEMA_COLUMNS
    )


def w_table(*, set_name: str, seed: int, pool_farms: pl.DataFrame) -> pl.DataFrame:
    """Return W1 to W3's cases of a set with experiment 259's R0 quantiles and static inputs."""
    summaries = pl.read_parquet(W259_DIR / f"summaries_{set_name}.parquet")
    base = pl.read_parquet(W259_DIR / f"R0_{set_name}_seed{seed}.parquet").drop("seed")
    sites = wsites.site_table().select("time_series_id", "site", "effective_capacity_mw")
    joined = summaries.drop("site").join(
        base, on=list(common.KEY_COLUMNS), how="inner", validate="1:1"
    )
    joined = joined.join(sites, on="time_series_id", validate="m:1")
    return joined.select(
        pl.lit("W").alias("group"),
        pl.col("site").alias("farm"),
        pl.lit(-1, dtype=pl.Int32).alias("fold"),
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        pl.col("lead_day").cast(pl.Int32),
        "actual_norm",
        "masked",
        *Q,
        *SUMMARY_COLUMNS,
        pl.col("effective_capacity_mw").log().alias("log_capacity_mw"),
        pl.lit(float(pool_farms["turbine_mw"].median())).alias("turbine_mw"),
        pl.lit(float(pool_farms["hub_height_m"].median())).alias("hub_height_m"),
    )


def static_matrix(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the static inputs as shape (cases, 3)."""
    return frame.select(STATIC_COLUMNS).to_numpy().astype(np.float64)
