"""Fixtures shared by the tests of `packages/studies`.

`dp_synthetic_inputs` writes small input files in the schemas that experiment 103 and the weather
files use, and `dp_synthetic_cache` builds the differentiable-physics feature cache from them with
the real builder. Its one substation, `S1`, has a temperature that follows the seasons, an
irradiance that follows the sun, and placeholder power; the tests replace the power with a known
truth.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.dp_features import DpInputPaths, FeatureCache, build_feature_cache, grid_end_times
from studies.pv_physics import solar_position

SYNTHETIC_SITE: Final[str] = "S1"
SYNTHETIC_LATITUDE: Final[float] = 53.0
SYNTHETIC_LONGITUDE: Final[float] = 0.0
SYNTHETIC_EFFECTIVE_CAPACITY_MW: Final[float] = 10.0
SYNTHETIC_FIRST_ROW: Final[datetime] = datetime(2021, 1, 1, 0, 30, tzinfo=UTC)


@pytest.fixture(scope="session")
def dp_synthetic_inputs(tmp_path_factory: pytest.TempPathFactory) -> DpInputPaths:
    """Write the synthetic input files and return their paths."""
    directory: Path = tmp_path_factory.mktemp("dp_inputs")
    generator = np.random.default_rng(1)
    grid = grid_end_times()
    day_of_year = grid.dt.ordinal_day().to_numpy()
    hour = grid.dt.hour().to_numpy() + grid.dt.minute().to_numpy() / 60.0
    noise = np.convolve(generator.normal(size=len(grid)), np.ones(48) / 12.0, mode="same")
    temperature = (
        10.0
        + 8.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25)
        + 3.0 * np.sin(2 * np.pi * (hour - 9) / 24)
        + noise
    )
    pl.DataFrame(
        {
            "time_series_id": np.full(len(grid), 1, dtype=np.int32),
            "valid_time": grid,
            "temperature_2m": temperature.astype(np.float32),
        }
    ).write_parquet(directory / "era5.parquet")
    hourly = pl.datetime_range(
        datetime(2019, 9, 1, 1, tzinfo=UTC),
        datetime(2026, 7, 1, 0, tzinfo=UTC),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    zenith, _ = solar_position(
        stamps=hourly.dt.offset_by("-30m"),
        latitude=SYNTHETIC_LATITUDE,
        longitude=SYNTHETIC_LONGITUDE,
    )
    elevation = 90.0 - zenith
    days = hourly.dt.date().rank("dense").to_numpy()
    daily_cloud = generator.uniform(0.2, 1.0, size=int(days.max()) + 1)
    ghi = 900.0 * np.clip(np.sin(np.radians(elevation)), 0.0, None) * daily_cloud[days]
    pl.DataFrame(
        {
            "site": SYNTHETIC_SITE,
            "time": hourly,
            "ghi_w_m2": ghi.astype(np.float32),
        }
    ).write_parquet(directory / "cams.parquet")
    row_times = pl.datetime_range(
        SYNTHETIC_FIRST_ROW,
        datetime(2026, 6, 30, 23, 30, tzinfo=UTC),
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    n_rows = len(row_times)
    pl.DataFrame(
        {
            "time_series_id": np.full(n_rows, 1, dtype=np.int32),
            "site": SYNTHETIC_SITE,
            "time": row_times,
            "power": np.full(n_rows, 5.0, dtype=np.float32),
            "present": np.ones(n_rows, dtype=bool),
            "in_pool": np.ones(n_rows, dtype=bool),
            "default_rows": np.ones(n_rows, dtype=bool),
            "switching_mask": np.zeros(n_rows, dtype=bool),
            "lockdown": np.zeros(n_rows, dtype=bool),
            "dark": np.zeros(n_rows, dtype=bool),
        }
    ).write_parquet(directory / "rows.parquet")
    pl.DataFrame(
        {
            "site": [SYNTHETIC_SITE],
            "time_series_id": np.array([1], dtype=np.int32),
            "group": ["primary demand"],
            "units": ["MW"],
            "latitude": np.array([SYNTHETIC_LATITUDE], dtype=np.float32),
            "longitude": np.array([SYNTHETIC_LONGITUDE], dtype=np.float32),
            "effective_capacity_mw": np.array([SYNTHETIC_EFFECTIVE_CAPACITY_MW], dtype=np.float32),
        }
    ).write_parquet(directory / "series_table.parquet")
    pl.DataFrame(
        {
            "site": [SYNTHETIC_SITE],
            "time_series_id": np.array([1], dtype=np.int64),
            "role": ["mw_long"],
            "first_usable": [SYNTHETIC_FIRST_ROW],
        }
    ).write_parquet(directory / "sites.parquet")
    return DpInputPaths(
        rows=directory / "rows.parquet",
        series_table=directory / "series_table.parquet",
        sites=directory / "sites.parquet",
        era5=directory / "era5.parquet",
        cams=directory / "cams.parquet",
    )


@pytest.fixture(scope="session")
def dp_synthetic_cache(dp_synthetic_inputs: DpInputPaths) -> FeatureCache:
    """Build the feature cache from the synthetic inputs with the real builder."""
    return build_feature_cache(paths=dp_synthetic_inputs)
