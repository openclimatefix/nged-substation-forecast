"""The empirical solar shape of experiment 280: tables learned from the metered farms' output.

Copied from experiment 269's common module so this folder imports only itself and the reviewed
packages. For each substation, the script averages the learned response tables of the six metered
farms (farms A to F, anonymous); each table maps the satellite irradiance and the sun's height to
a unit output. The tables are saved under the private `work/` folder and read by `e00280_common`.

**Privacy.** The farms' output enters only through the averaged table, in units of the farm's own
99th percentile.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import (
    CELL_TEMPERATURE_RISE_K,
    DERATING_PER_KELVIN,
    REFERENCE_CELL_TEMPERATURE_C,
)
from studies.pv_plant_fit import ResponseGrid, fit_response_table
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
INJECTION_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
TABLE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00280-pv-empirical-shape-and-capacity-in-arm-d" / "work" / "tables"
)
SHAPE_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""Farm half-hours at or after this instant, the start of the scored year, never enter a table."""
ALL_FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
DERATING_ITERATIONS: Final[int] = 12
RESPONSE_GRID: Final[ResponseGrid] = ResponseGrid(
    irradiance_w_m2=np.arange(0.0, 1101.0, 50.0), elevation_deg=np.arange(5.0, 66.0, 5.0)
)
RESPONSE_MIN_ELEVATION_DEG: Final[float] = 0.0
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")


def load_cache() -> FeatureCache:
    """Return the feature cache of experiment 106, which holds data to 2026-07-01."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def farm_norm(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return a farm's power divided by its 99th percentile, on the cache's grid.

    The 99th percentile is over the farm's daylight half-hours in all history, as in experiments 39
    and 129. A half-hour returns `NaN` when the farm's frame has no row for it while the sun is up,
    or has a daylight row (zenith under 75 degrees) with no reading or a dropout (power at or below
    zero while the irradiance exceeds `FARM_DROPOUT_GHI_W_M2`), so that the row drops out of
    training. Remaining negative readings are set to zero. A half-hour with the sun below 75 degrees
    of zenith returns 0, and so does a half-hour with no frame row while the sun is down.

    Args:
        cache: The cache, for the grid.
        farm: The farm label (A to F).

    Returns:
        The normalised power on the grid.
    """
    everything = pl.read_parquet(INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    frame = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG)
    power = frame.filter(pl.col("power").is_not_null())
    p99 = float(np.nanpercentile(power["power"].to_numpy(), 99))
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI_W_M2)
    daylight = frame.select(
        "time",
        norm=pl.when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(daylight, on="time", how="left", validate="1:1", maintain_order="left")
    norm = joined["norm"].fill_null(0.0).to_numpy().astype(np.float64)
    in_frame = grid.join(everything.select("time"), on="time", how="semi")["time"]
    missing = ~np.isin(cache.times_series().to_numpy(), in_frame.to_numpy())
    sun_up = np.max(
        [np.nan_to_num(site.pv_unit_shape, nan=0.0) for site in cache.sites.values()], axis=0
    )
    norm[missing & (sun_up > SUN_UP_UNIT_SHAPE)] = np.nan
    return norm


def farm_sky(*, cache: FeatureCache, farm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's CAMS global irradiance and sun elevation on the cache's grid.

    Both come from the farm's row of experiment 39's frame. A half-hour with no row returns `NaN`.

    Args:
        cache: The cache, for the grid.
        farm: The farm label (A to F).

    Returns:
        `(ghi in W/m2, elevation in degrees)`.
    """
    frame = (
        pl.read_parquet(INJECTION_FRAME_PATH)
        .filter(pl.col("site") == farm, pl.col("kind") == "farm")
        .select("time", "ghi_u", elevation=90.0 - pl.col("zenith"))
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(frame, on="time", how="left", validate="1:1", maintain_order="left")
    return (
        joined["ghi_u"].to_numpy().astype(np.float64),
        joined["elevation"].to_numpy().astype(np.float64),
    )


def reference_unit_shape(*, cache: FeatureCache, site: str, reference: str) -> np.ndarray:
    """Return the Primary's unit shape built from a reference farm's measured output.

    The farm's normalised output already carries its own heat loss, and the fit multiplies the unit
    shape by a temperature derating, so the derating is divided out first: `u` solves
    `u * (1 - 0.004 * (T + 25 u - 25)) = r` for the farm's normalised output `r` and the Primary's
    temperature `T`, by fixed-point steps. Half-hours at which the reference farm has a dropout or
    no reading stay `NaN`, so that the fit drops them.

    Args:
        cache: The cache, for the grid and the Primary's temperature.
        site: The Primary's label.
        reference: The reference farm's label.

    Returns:
        The shape on the cache's grid.
    """
    measured = farm_norm(cache=cache, farm=reference)
    temperature = cache.sites[site].temperature_c
    unit = measured.copy()
    for _ in range(DERATING_ITERATIONS):
        cell = temperature + CELL_TEMPERATURE_RISE_K * unit
        factor = np.clip(
            1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0.2, None
        )
        unit = measured / factor
    return unit


def learned_values(*, cache: FeatureCache, site: str, farm: str) -> np.ndarray:
    """Fit a farm's learned response table at a substation and return its values.

    The farm's unit shape (output with the temperature derating divided out at the substation's
    temperature) is fitted against the farm's own satellite irradiance and sun elevation, as a
    smooth table that is monotone in irradiance, with no anchor to a prior.

    Args:
        cache: The cache.
        site: The substation's label, for the temperature.
        farm: The farm label (A to F).

    Only farm half-hours before `SHAPE_CUT` are used.

    Returns:
        The table, shape (irradiance nodes, elevation nodes).
    """
    unit = reference_unit_shape(cache=cache, site=site, reference=farm)
    ghi, elevation = farm_sky(cache=cache, farm=farm)
    before = cache.times_series().to_numpy() < np.datetime64(SHAPE_CUT.replace(tzinfo=None))
    rows = np.isfinite(unit) & np.isfinite(ghi) & (elevation > RESPONSE_MIN_ELEVATION_DEG) & before
    table = fit_response_table(
        grid=RESPONSE_GRID,
        irradiance_w_m2=ghi[rows],
        elevation_deg=elevation[rows],
        output=unit[rows],
        prior_output=np.zeros(int(rows.sum())),
        penalty=0.0,
    )
    return table.values


def table_path(*, site: str) -> Path:
    """Return the file that holds a substation's averaged table."""
    return TABLE_DIR / f"{site}.npy"


def main() -> None:
    """Write the average of the six farms' learned tables at each substation."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    for site in SITES:
        values = np.mean(
            [learned_values(cache=cache, site=site, farm=farm) for farm in ALL_FARMS], axis=0
        )
        np.save(table_path(site=site), values)
        print(f"{site}: table max {values.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
