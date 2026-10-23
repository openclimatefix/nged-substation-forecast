# ruff: noqa: E501
"""Inputs of experiment 255: the pilot node table, the node forecasts, and the six farms' forecasts.

The node table joins generators to substations, so it stays private. Farms appear only as labels A
to F; the label-to-series table is read from `studies.pv_dataset.pv_sites` and never printed.
"""

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from contracts.settings import PROJECT_ROOT
from nged_data.storage import scan_cleaned_power
from studies.node_curtailment import NodeTable, node_table_pilot
from studies.pv_dataset import pv_sites
from studies.solar import zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for
from studies.tail_calibration import read_arms

EXPERIMENT_ID: Final[int] = 255
SLUG: Final[str] = "lsc-curtailment-metric-v2-member-wise-ens"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PRIVATE_DIR: Final[Path] = DATA_DIR / "private"
RESULTS_DIR: Final[Path] = DATA_DIR / "results"
E84_PRIVATE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00084-substation-limit-curtailment-cost-worked-example" / "private"
)
E224_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
E243_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00243-metered-farm-architecture-comparison"

SOURCE_EXPERIMENT_ID: Final[int] = 48
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 9, 1, tzinfo=dt.UTC)
"""First valid-time instant of the scored rows; scaled limits use only flows strictly before it."""
SCORED_END: Final[dt.datetime] = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
"""The holdout starts here, so no flow on or after it is read."""
MIN_ELEVATION_DEG: Final[float] = 5.0
RUN_LEAD_MIN_HOURS: Final[float] = 12.0
"""A farm forecast is the latest 00 UTC run that is at least this many hours before the valid time."""

METADATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "NGED" / "metadata.parquet"
SUBSTATIONS_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "NGED" / "substation_hierarchy_2026-10-15" / "Substations.csv"
)
CLEANED_POWER_PATH: Final[Path] = PROJECT_ROOT / "data" / "NGED" / "cleaned_power_time_series.delta"
CAPACITY_PATH: Final[Path] = (
    study_dir_for(study="quick_wins_fixed_pipeline_1156")
    / "frames"
    / "combined"
    / "fold_a"
    / "capacity.parquet"
)

BOSTON: Final[str] = "BOSTON 33kV S STN"
SKEGNESS: Final[str] = "SKEGNESS 33kV S STN"
BICKER_FEN: Final[str] = "BICKER FEN 132kV S STN"
WALPOLE: Final[str] = "WALPOLE 132kV S STN"
SCORED_NODES: Final[tuple[str, ...]] = (BOSTON, SKEGNESS)
EXPORT_LIMITS_MW: Final[Mapping[str, tuple[float, ...]]] = {
    BOSTON: (45.0, 45.1),
    SKEGNESS: (29.8, 31.3),
    BICKER_FEN: (263.5,),
    WALPOLE: (73.4,),
}
SIGNS: Final[Mapping[str, tuple[int | None, str]]] = {
    BOSTON: (-1, "our data: a positive reading is import"),
    SKEGNESS: (-1, "our data: a positive reading is import"),
    BICKER_FEN: (1, "UNCONFIRMED"),
    WALPOLE: (None, "unknown"),
}
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
NODE_QUANTILES: Final[tuple[str, ...]] = QUANTILES[1:-1]
"""The nine levels experiment 48 stored for nodes and the other generators."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)


def read_capacity_mw() -> dict[int, float]:
    """Return each series' effective capacity in MW, as experiment 48 normalised by."""
    capacity = pl.read_parquet(CAPACITY_PATH)
    return {
        int(s): float(c)
        for s, c in zip(capacity["time_series_id"], capacity["effective_capacity_mw"], strict=True)
    }


def read_arm_frames(*, names: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Read experiment 48's named prediction sets, dropping the label columns."""
    raw = read_arms(experiment_id=SOURCE_EXPERIMENT_ID, names=names, row_set_id=ROW_SET_ID)
    return {
        name: frame.drop("site", "group", "lead_hours", strict=False).with_columns(
            pl.col("time_series_id").cast(pl.Int64), pl.col("seed").cast(pl.Int64)
        )
        for name, frame in raw.items()
    }


def build_pilot_table(*, capacity_mw: Mapping[int, float], forecast_ids: set[int]) -> NodeTable:
    """Build the pilot node and generator tables from NGED's private files."""
    return node_table_pilot(
        metadata=pl.read_parquet(METADATA_PATH),
        substations=pl.read_csv(SUBSTATIONS_PATH),
        transformer_export_limits_mw=EXPORT_LIMITS_MW,
        signs=SIGNS,
        capacity_mw=capacity_mw,
        forecast_ids=forecast_ids,
    )


def scoring_table(*, table: NodeTable) -> NodeTable:
    """Return the table of the two scored bulk supply points, with parent links cleared."""
    keep = list(SCORED_NODES)
    return NodeTable(
        nodes=table.nodes.filter(pl.col("node_id").is_in(keep)).with_columns(
            parent_node_id=pl.lit(None, dtype=pl.String)
        ),
        limits=table.limits.filter(pl.col("node_id").is_in(keep)),
        generators=table.generators.filter(pl.col("node_id").is_in(keep)),
    )


def read_flows(*, table: NodeTable) -> pl.DataFrame:
    """Return the metered flows of the table's nodes from cleaned, unflagged power, before the holdout."""
    series = table.nodes["time_series_id"].to_list()
    unflagged = scan_cleaned_power(str(CLEANED_POWER_PATH))
    plain = pl.LazyFrame._from_pyldf(unflagged._ldf)  # strip the Patito model, zero-copy
    return (
        plain.filter(pl.col("time_series_id").is_in(series), pl.col("time") < SCORED_END)
        .select(
            pl.col("time_series_id").cast(pl.Int64),
            pl.col("time").alias("valid_time"),
            flow_mw=pl.col("power").cast(pl.Float64),
        )
        .collect()
    )


def farm_sites() -> pl.DataFrame:
    """Return `site`, `time_series_id` (Int64), `latitude`, `longitude`, `effective_capacity_mw`."""
    return pv_sites().with_columns(
        pl.col("time_series_id").cast(pl.Int64), pl.col("effective_capacity_mw").cast(pl.Float64)
    )


def read_farm_power_mw() -> pl.DataFrame:
    """Return `time_series_id`, `valid_time`, `output_mw` of the farms' cleaned, unflagged power."""
    path = (
        ML_EXPERIMENTS_DATA_DIR
        / "00179-metered-solar-farm-exploration"
        / "cleaned_solar_farms.parquet"
    )
    return (
        pl.read_parquet(path)
        .filter(pl.col("drop_reason").is_null())
        .select(
            pl.col("time_series_id").cast(pl.Int64),
            valid_time="time",
            output_mw=pl.col("power").cast(pl.Float64),
        )
    )


def is_daylight(*, valid_times: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return a mask of the half-hours (labelled by their end) with the sun above the horizon limit."""
    midpoints = valid_times.dt.offset_by("-15m")
    return (
        90.0 - zenith(stamps=midpoints, latitude=latitude, longitude=longitude)
    ) > MIN_ELEVATION_DEG


def _with_run_time(*, frame: pl.DataFrame, time_column: str) -> pl.DataFrame:
    """Add `run_time`: the latest 00 UTC run at least `RUN_LEAD_MIN_HOURS` before the valid time."""
    return frame.with_columns(
        run_time=pl.col(time_column).dt.offset_by(f"-{int(RUN_LEAD_MIN_HOURS)}h").dt.truncate("1d")
    )


def read_farm_forecasts() -> dict[str, pl.DataFrame]:
    """Return each farm arm's stored forecast on the rows every arm holds.

    Rows are `farm` (label), `valid_time`, `actual` (fraction of the farm's effective capacity),
    `lead_hours`, and `q01` to `q99`, in fractions of capacity. The H5 and XGBoost forecasts are the
    mean over the five seeds of each quantile. The rows kept are those that every arm holds with a
    finite median, which are experiment 243's common rows, less the few rows where a floor is
    missing.
    """
    yearly = pl.read_parquet(E224_DIR / "predictions" / "scored_year.parquet").filter(
        pl.col("band") == 0
    )
    pieces: dict[str, pl.DataFrame] = {}
    for arm, name in (
        ("floor_heuristic", "floor_heuristic"),
        ("floor_climatology", "floor_climatology"),
        ("plant_two_number", "a3_two_number"),
    ):
        pieces[arm] = yearly.filter(pl.col("arm") == name).select(
            *QUANTILES,
            farm="time_series_id",
            valid_time="valid_time",
            actual="actual",
            lead_hours="lead_hours",
        )
    for arm, name, setting in (
        ("xgboost", "xgboost", "own_history"),
        ("h5", "h5_quantile", "own_history"),
        ("h5_zero_history", "h5_quantile", "zero_history"),
    ):
        pieces[arm] = _mean_over_seeds(name=name, setting=setting, seeds=(0,))
    pieces["h5_mean5"] = _mean_over_seeds(name="h5_quantile", setting="own_history", seeds=SEEDS)
    keys: pl.DataFrame | None = None
    for frame in pieces.values():
        ok = frame.filter(pl.col("q50").is_finite()).select("farm", "valid_time")
        keys = ok if keys is None else keys.join(ok, on=["farm", "valid_time"], how="inner")
    if keys is None:
        msg = "no farm forecast arms"
        raise ValueError(msg)
    return {
        arm: frame.join(keys, on=["farm", "valid_time"], how="inner").sort("farm", "valid_time")
        for arm, frame in pieces.items()
    }


def _mean_over_seeds(*, name: str, setting: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an experiment 243 arm with each quantile averaged over its stored seeds.

    H5 holds five seeds; XGBoost holds one, because it has no random draw to vary.
    """
    files = [
        f
        for k in seeds
        for f in sorted((E243_DIR / "predictions" / setting).glob(f"{name}_s{k}_*.parquet"))
    ]
    frame = pl.concat([pl.read_parquet(f) for f in files])
    return (
        frame.group_by("time_series_id", "time")
        .agg(
            pl.col("actual").first().cast(pl.Float64),
            pl.col("lead_hours").first().cast(pl.Float64),
            *(pl.col(q).cast(pl.Float64).mean() for q in QUANTILES),
        )
        .rename({"time_series_id": "farm", "time": "valid_time"})
    )


def seed_forecasts(*, name: str, setting: str, seed: int) -> pl.DataFrame:
    """Return one seed of an experiment 243 arm, with the columns of `read_farm_forecasts`."""
    files = sorted((E243_DIR / "predictions" / setting).glob(f"{name}_s{seed}_*.parquet"))
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .select(
            *(pl.col(q).cast(pl.Float64) for q in QUANTILES),
            farm="time_series_id",
            valid_time="time",
            actual=pl.col("actual").cast(pl.Float64),
            lead_hours=pl.col("lead_hours").cast(pl.Float64),
        )
        .sort("farm", "valid_time")
    )
