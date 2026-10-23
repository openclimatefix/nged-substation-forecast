"""Shared constants and the model-curve frame for experiment 282 (audit of the solar plant curve).

Farms appear only as labels A to F. The frame joins experiment 218's farm rows and predictions to
the solar term's curve (experiment 139's engine: the fleet plant's shape, hard-clipped at DC:AC
1.15, times the 0.4% per kelvin temperature loss, times one fitted capacity per farm). Everything is
private and written under `data/ml-experiments/00282-*/`.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.pv_physics import PlantParameters, SunAndSky, plane_of_array_w_m2, plant_power_mw
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
"""Copy of `studies.dp_features.FLEET`, which cannot be imported here without torch."""
DERATING_PER_KELVIN: Final[float] = 0.004
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
"""Copies of the constants of `studies.dp_gam`, which cannot be imported here without torch."""
SLUG: Final[str] = "00282-pv-plant-curve-audit"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
E218_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit"
ROWS_PATH: Final[Path] = E218_DIR / "farm_rows_cams.parquet"
PREDICTIONS_PATH: Final[Path] = E218_DIR / "predictions.parquet"
FRAME_PATH: Final[Path] = DATA_DIR / "curve_frame.parquet"
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
FOLD: Final[int] = 1
SCORED_START: Final[str] = "2025-09-01"
BIN_WIDTH_W_M2: Final[float] = 100.0
MIN_BIN_ROWS: Final[int] = 100
DEPARTURE_POINTS: Final[float] = 5.0
STOP_POINTS: Final[float] = 3.0
BOOTSTRAP_DRAWS: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 282
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}
CLEAR_INDEX: Final[float] = 0.9
LOW_SUN_DEG: Final[float] = 15.0
MIDDAY_DEG: Final[float] = 40.0
MODELS: Final[tuple[str, ...]] = ("plant_fleet", "plant_per_farm", "stack_term")


def sky_of(frame: pl.DataFrame) -> SunAndSky:
    """Return the frame's sun and sky as the physics module's record."""
    return SunAndSky(
        half_hour_end_time=frame["time"].to_numpy(),
        ghi_w_m2=frame["ghi_w_m2"].to_numpy(),
        dni_w_m2=frame["dni_w_m2"].to_numpy(),
        dhi_w_m2=frame["dhi_w_m2"].to_numpy(),
        zenith_deg=frame["zenith_deg"].to_numpy(),
        sun_azimuth_deg=frame["sun_azimuth_deg"].to_numpy(),
    )


def stack_unit_shape(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the stack's derated unit shape per MW of AC capacity for each row of one farm."""
    unit = plant_power_mw(sky=sky_of(frame), parameters=FLEET)
    cell = frame["temperature_c"].to_numpy() + CELL_TEMPERATURE_RISE_K * unit
    return unit * np.clip(
        1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0, None
    )


def build_frame() -> pl.DataFrame:
    """Join rows, 218's predictions and the stack's curve into one frame of fold 1 rows."""
    rows = pl.read_parquet(ROWS_PATH)
    predictions = (
        pl.read_parquet(PREDICTIONS_PATH)
        .filter(pl.col("fold") == FOLD)
        .select("site", "time", "scored", "arm1_fleet_shape", "arm2_per_farm")
    )
    parts = []
    for farm in FARMS:
        frame = rows.filter(pl.col("site") == farm).sort("time")
        shape = stack_unit_shape(frame=frame)
        poa = plane_of_array_w_m2(
            sky=sky_of(frame), tilt_deg=FLEET.tilt_deg, azimuth_deg=FLEET.azimuth_deg, tracker=False
        )
        parts.append(frame.with_columns(stack_shape=shape, poa_fleet_w_m2=poa))
    joined = pl.concat(parts).join(predictions, on=["site", "time"], how="inner", validate="1:1")
    tuning = ~joined["scored"] & (
        joined["time"] < pl.lit(SCORED_START).str.to_datetime(time_zone="UTC")
    )
    capacity = (
        joined.filter(tuning)
        .group_by("site")
        .agg(
            capacity=(pl.col("output").cast(pl.Float64) * pl.col("stack_shape")).sum()
            / (pl.col("stack_shape") ** 2).sum()
        )
    )
    return (
        joined.join(capacity, on="site", validate="m:1")
        .with_columns(
            plant_fleet=pl.col("arm1_fleet_shape"),
            plant_per_farm=pl.col("arm2_per_farm"),
            stack_term=pl.col("capacity") * pl.col("stack_shape"),
            output=pl.col("output").cast(pl.Float64),
            clear_index=pl.col("ghi_w_m2") / pl.col("clear_sky"),
            hour_of_day=pl.col("time").dt.hour() + pl.col("time").dt.minute() / 60.0,
            calendar_month=pl.col("time").dt.month(),
            poa_bin=(pl.col("poa_fleet_w_m2") / BIN_WIDTH_W_M2).floor() * BIN_WIDTH_W_M2,
        )
        .sort("site", "time")
    )
