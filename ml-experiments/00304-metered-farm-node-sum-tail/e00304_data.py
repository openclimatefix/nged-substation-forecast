"""Inputs of experiment 304: the farms' stored quantile forecasts joined to ensemble members.

Farms appear only as labels A to F. The two nodes hold farms {A, C, E} and {B, D, F}; the table
that ties each farm to its bulk supply point is private and is not read here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 304
SLUG: Final[str] = "metered-farm-node-sum-tail"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
E224_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
E287_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00287-metered-farm-spread-binned-plant-heads" / "predictions"
)
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
NODES: Final[dict[str, tuple[str, ...]]] = {"node 1": ("A", "C", "E"), "node 2": ("B", "D", "F")}
"""Node labels and the farms beneath each, as labels only."""
ARMS: Final[dict[str, str]] = {
    "p2": "spread-binned plant (experiment 270's P2)",
    "blend": "per-quantile blend (experiment 287)",
    "xgb_x2": "XGBoost (experiment 287)",
}
"""The stored seed-0 own-history prediction sets used as each farm's marginal."""
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
QUANTILE_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{round(100 * lv):02d}" for lv in LEVELS)
OUTPUT_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.50, 0.95, 0.99)
CAPACITY_CAP_FRACTION: Final[float] = 1.2
"""No farm's output exceeds this fraction of its effective capacity."""


@dataclass
class FarmRows:
    """One farm's scored rows for one arm, sorted by (time, init_time).

    Attributes:
        quantiles: `(n, 11)` quantiles in MW at `LEVELS`.
        actual: `(n,)` observed output in MW.
        members: `(n, 51)` the ensemble members' unit plant responses (rank-ordered only).
        csi: `(n,)` clear-sky index, the ensemble-mean irradiance over the clear-sky irradiance.
        keys: Frame with `time`, `init_time` and `lead_hours`.
        capacity_mw: The farm's effective capacity.
    """

    quantiles: np.ndarray
    actual: np.ndarray
    members: np.ndarray
    csi: np.ndarray
    keys: pl.DataFrame
    capacity_mw: float


def capacities() -> dict[str, float]:
    """Return each farm's effective capacity in MW by label."""
    sites = pv_sites()
    return dict(zip(sites["site"].to_list(), sites["effective_capacity_mw"].to_list(), strict=True))


def read_farm(*, site: str, arm: str, capacity_mw: float) -> pl.DataFrame:
    """Return a farm's arm rows joined to its member responses and clear-sky index."""
    predictions = pl.read_parquet(E287_DIR / "own_history" / f"{arm}_s0_{site}.parquet")
    members = pl.read_parquet(E224_DIR / "members" / f"{site}_b0.parquet")
    unit = np.load(E224_DIR / "members" / f"{site}_b0_unit.npy")
    if unit.shape[0] != members.height:
        msg = f"farm {site}: {unit.shape[0]} member rows but {members.height} table rows"
        raise ValueError(msg)
    members = members.select("time", "init_time", "ghi_mean").with_columns(
        member_row=pl.int_range(pl.len(), dtype=pl.Int64)
    )
    sky = (
        pl.read_parquet(FARM_ROWS)
        .filter(pl.col("site") == site)
        .select("time", "clear_sky")
        .unique("time")
    )
    joined = (
        predictions.select("time", "init_time", "lead_hours", "actual", *QUANTILE_COLUMNS)
        .join(members, on=["time", "init_time"], how="inner", validate="1:1")
        .join(sky, on="time", how="left")
        .sort("time", "init_time")
    )
    if joined.height != predictions.height:
        msg = f"farm {site} arm {arm}: member join lost {predictions.height - joined.height} rows"
        raise ValueError(msg)
    csi = (
        (joined["ghi_mean"] / joined["clear_sky"].clip(lower_bound=1.0))
        .clip(0.0, 1.5)
        .fill_null(0.0)
    )
    return joined.with_columns(
        csi=csi,
        member_responses=pl.Series(unit[joined["member_row"].to_numpy()].tolist()),
        capacity_mw=pl.lit(capacity_mw),
    )


def node_rows(*, node: str, arm: str) -> list[FarmRows]:
    """Return the node's farms on the half-hours where all of them have a stored forecast."""
    caps = capacities()
    frames = {site: read_farm(site=site, arm=arm, capacity_mw=caps[site]) for site in NODES[node]}
    keys = next(iter(frames.values())).select("time", "init_time")
    for frame in frames.values():
        keys = keys.join(frame.select("time", "init_time"), on=["time", "init_time"], how="inner")
    out = []
    for site in NODES[node]:
        frame = (
            frames[site].join(keys, on=["time", "init_time"], how="inner").sort("time", "init_time")
        )
        capacity = caps[site]
        out.append(
            FarmRows(
                quantiles=capacity * frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64),
                actual=capacity * frame["actual"].to_numpy().astype(np.float64),
                members=np.asarray(frame["member_responses"].to_list(), dtype=np.float64),
                csi=frame["csi"].to_numpy(),
                keys=frame.select("time", "init_time", "lead_hours"),
                capacity_mw=capacity,
            )
        )
    first = out[0].keys.select("time", "init_time")
    for rows in out[1:]:
        if not first.equals(rows.keys.select("time", "init_time")):
            msg = f"{node}: farms' rows are not aligned"
            raise ValueError(msg)
    return out
