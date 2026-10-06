"""Build the rows of the UKV-on-CEDA against ERA5 study: station-hours and two sets of fit rows.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1024>. The question is whether
the main work should take past wind speed and past air temperature from the Met Office's UKV as
archived by CEDA (UKV-CEDA) or from ERA5. Set A scores each product against Met Office station
readings (`ukv_ceda_station_scores.py`). Set B fits one XGBoost model per metered generator and per
product (`ukv_ceda_vs_era5_fit.py`). This script builds the rows both sets read, and holds the
constants every sibling script imports: the windows, the margins, and one function per arm type that
returns the arm's feature columns.

**Rows are the intersection across every arm, decided from the target and from availability, and
never from a product's values.** An hour stays only if the target exists and every arm's input
exists. The wind rows drop the hours `wind_product_frames.common_rows` drops. The solar rows drop
the hours `solar_product_frames.common_rows` drops, and carry the export-cap `constrained` flag that
the fit loop reads. The 12 days 2021-12-01 to 2021-12-12, on which Open-Meteo's mirror of ERA5
temperature disagrees with the Copernicus archive, are dropped from every arm of both sets. The
months 2019-12 and 2026-01, in which a UKV physics version changed part-way, are dropped from every
arm too.

**UKV-CEDA is read from the freshest run at or before each hour, and an incomplete run drops the
hour.** `studies.ukv_ceda_stores` maps each instant to its run. Wind is the instantaneous value at
the hour's label. Temperature in the solar rows is the mean of the instants at the hour's two ends,
for both products, because the power hour ends at its label. A solar hour therefore needs both
instants' runs to be complete.

**Eras.** `studies.ukv_ceda_stores.with_ukv_eras` labels the three UKV eras, drops the straddling
months, and cuts folds inside each era. The build stops unless `era_code` holds exactly 0, 1 and 2.

**Privacy.** Nothing here prints or writes a generator's name, identifier or coordinates, or a
station's identifier or coordinates. Stations are labelled S1 upwards, generators A to F and W1 to
W3, and every distance printed is pooled over stations.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_era5_build.py`. `--check-only` runs the
coverage check on the real data without reading any UKV value or writing a file, and exits non-zero
if a guard fails. `--dry-run` builds one month (`--dry-run-month`) with real values and writes
nothing. A fresh run stops (`refuse_to_overwrite`) while an output exists.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, NamedTuple

import icechunk
import numpy as np
import polars as pl
import zarr
from cerra_past_solar import check_column_counts
from deltalake import DeltaTable
from studies.arm_runner import add_time_features, dataset_path_for
from studies.blending import climatology_permutation
from studies.export_cap import with_export_cap
from studies.grid_sampling import distance_matrix_km, nearest_cells
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.midas import read_hourly_weather, read_station_metadata
from studies.neighbouring_hours import with_neighbouring_hours
from studies.pv_dataset import (
    CAPACITY_DELTA_URI,
    OPEN_METEO_PATH,
    POWER_DELTA_URI,
    nearest_era5_cell,
    pv_sites,
    read_era5,
    wind_sites,
)
from studies.solar_product_frames import common_rows as solar_common_rows
from studies.sources import (
    ERA5_PRODUCT_DIR,
    ERA5_WIND_2019_2023_PRODUCT_DIR,
    MIDAS_OPEN_PRODUCT_DIR,
    UKV_CEDA_PART2_PRODUCT_DIR,
    UKV_CEDA_PART3_PRODUCT_DIR,
    UKV_CEDA_PRODUCT_DIR,
    UKV_VS_ERA5_DIR,
)
from studies.ukv_ceda_profiles import (
    DEFAULT_PROFILE,
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_PARTIAL,
)
from studies.ukv_ceda_stores import (
    ERA_FIRST_MONTHS,
    STRADDLING_MONTHS,
    ArrayGroup,
    Store,
    make_stores,
    read_at_instants,
    ukv_fold_designs,
    usable_hours,
    with_ukv_eras,
)
from studies.wind_product_frames import common_rows as wind_common_rows
from studies.wind_product_frames import wind_hourly_power

_LOG: Final[logging.Logger] = logging.getLogger("ukv_ceda_vs_era5_build")

OUTPUT_DIR: Final[Path] = UKV_VS_ERA5_DIR
"""The write-once folder every script of the study writes to."""

STORE_DIRS: Final[tuple[Path, Path, Path]] = (
    UKV_CEDA_PRODUCT_DIR,
    UKV_CEDA_PART2_PRODUCT_DIR,
    UKV_CEDA_PART3_PRODUCT_DIR,
)
"""The three UKV-on-CEDA stores of 6-hourly runs, in date order."""

STATION_HOURS_NAME: Final[str] = "station_hours.parquet"
WIND_ROWS_NAME: Final[str] = "wind_rows.parquet"
WIND_KEEP_ZERO_ROWS_NAME: Final[str] = "wind_rows_keep_zero_hours.parquet"
WIND_HOUR_STARTING_ROWS_NAME: Final[str] = "wind_rows_hour_starting.parquet"
"""The wind rows with the power of the hour starting at the label added, for the power-hour scan."""
WIND_MATCHED_ROWS_NAME: Final[str] = "wind_rows_matched_10m.parquet"
"""The wind rows with ERA5's 10 m direction added, for the post hoc matched-height pair."""
SOLAR_ROWS_NAME: Final[str] = "solar_rows.parquet"
STAMP_NAME: Final[str] = "build.json"
README_NAME: Final[str] = "README.md"

HOURLY_WEATHER_PATH: Final[Path] = MIDAS_OPEN_PRODUCT_DIR / "uk_hourly_weather_obs.parquet"
STATION_METADATA_PATH: Final[Path] = (
    MIDAS_OPEN_PRODUCT_DIR
    / "_station_metadata"
    / "midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv"
)
ERA5_WIND_NATIVE_PATH: Final[Path] = ERA5_PRODUCT_DIR / "wind_native_cds.parquet"
"""ERA5 native wind at the wind farms' 3 by 3 blocks, 2024-01 to 2026-09."""
ERA5_WIND_CDS_PATH: Final[Path] = ERA5_WIND_2019_2023_PRODUCT_DIR / "wind_era5_cds.parquet"
ERA5_CELL_GROUPS_PATH: Final[Path] = ERA5_WIND_2019_2023_PRODUCT_DIR / "cell_groups.parquet"
"""Which ERA5 cells belong to which wind farm or station block, and each cell's position."""

WINDOW_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
STATION_WINDOW_END: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
"""The first instant after the MIDAS Open download, which ends on 2025-12-31."""
COVERAGE_WINDOW_END: Final[datetime] = datetime(2026, 9, 11, tzinfo=UTC)
"""The first instant after the last hour of the Open-Meteo ERA5 download."""

DROPPED_ERA5_TEMPERATURE_DAYS: Final[tuple[datetime, datetime]] = (
    datetime(2021, 12, 1, tzinfo=UTC),
    datetime(2021, 12, 13, tzinfo=UTC),
)
"""Half-open range of the 12 days on which Open-Meteo's ERA5 temperature is wrong by up to 2 C."""

FIRST_LATE_ERA_MONTH: Final[str] = ERA_FIRST_MONTHS[0]
"""The first month of the second UKV era, 2020-01, after the 2019-12-04 physics change."""

EARLY_END_MONTH: Final[str] = "2021-01"
"""The first month of the late window. The early window is every earlier month, from 2019-09."""

MIN_STATION_COVERAGE: Final[float] = 0.9
"""A station is scored only if it has both a wind speed and a temperature at this share of hours."""

PLANNED_STATION_COUNT: Final[int] = 4
"""How many stations the plan found inside the UKV-CEDA crop."""

CELL_DISTANCE_LIMIT_KM: Final[float] = 2.0
"""A point farther than this from its nearest UKV-CEDA cell lies outside the crop."""

ERA5_HALF_CELL_DEG: Final[float] = 0.125
"""A point farther than this from its nearest ERA5 cell centre lies outside the 0.25 degree grid."""

NEAR_GENERATOR_KM: Final[float] = 40.0
"""A station farther than this from every generator is context for set A only."""

MAX_MONTH_LOSS_SHARE: Final[float] = 0.25
"""The build stops if one month loses more than this share of hours to incomplete UKV-CEDA runs."""

MIN_SCORED_HOURS_SHOWN: Final[int] = 2000
"""A generator-year with fewer scored hours is not shown separately."""

MARGIN_STATION_SHARE: Final[float] = 0.05
"""The set A margin, as a share of ERA5's bias-removed error on the same rows."""

MARGIN_WIND_PP: Final[float] = 0.16
"""The set B wind margin, in percentage points of capacity."""

MARGIN_SOLAR_PP: Final[float] = 0.06
"""The set B solar margin, in percentage points of capacity."""

SHUFFLE_SEED: Final[int] = 1024
"""Seeds the control shuffles. Product `i` of a frame is shuffled under `SHUFFLE_SEED + i`."""

SHUFFLE_GROUPS: Final[tuple[str, ...]] = ("site", "month", "hour_of_day")
"""A shuffled value stays within one site, one year-month and one hour of day."""

SHUFFLED_SUFFIX: Final[str] = "_shuffled"

KELVIN: Final[float] = 273.15

UKV_WIND_VARIABLES: Final[tuple[str, str, str]] = (
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_925hpa",
)
UKV_TEMPERATURE_VARIABLE: Final[str] = "temperature_1p5m"

WIND_SHARED_FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "day_of_year", "era_code")
SOLAR_SHARED_FEATURES: Final[tuple[str, ...]] = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "extraterrestrial_horizontal_w_m2",
    "hour_of_day",
    "day_of_year",
    "era_code",
)
SOLAR_IRRADIANCE_FEATURES: Final[tuple[str, ...]] = ("ghi_cams", "bhi_cams", "dhi_cams")
"""CAMS global, beam and diffuse irradiance, which both solar arms read."""

PRODUCTS: Final[tuple[str, str]] = ("era5", "ukv_ceda")
"""The two products, in the order a contrast reads: the reference, then the treatment."""


def wind_columns(*, product: str) -> tuple[str, str, str, str]:
    """Return a product's four wind columns: the arm's speed, sine, cosine, and a second speed.

    ERA5's arm reads its 100 m speed and direction and its 10 m speed. UKV-CEDA has no 100 m wind,
    so its arm reads the 10 m speed and direction and the 925 hPa speed.

    Args:
        product: `era5` or `ukv_ceda`.

    Returns:
        Four frame column names.

    Raises:
        ValueError: If the product is not one of `PRODUCTS`.
    """
    if product == "era5":
        return ("era5_speed_100m", "era5_sin_100m", "era5_cos_100m", "era5_speed_10m")
    if product == "ukv_ceda":
        return (
            "ukv_ceda_speed_10m",
            "ukv_ceda_sin_10m",
            "ukv_ceda_cos_10m",
            "ukv_ceda_speed_925hpa",
        )
    msg = f"unknown product {product!r}"
    raise ValueError(msg)


def wind_arm_columns(*, product: str, shuffled: bool = False) -> tuple[str, ...]:
    """Return a wind arm's seven feature columns.

    Args:
        product: `era5` or `ukv_ceda`.
        shuffled: Whether the arm reads the product's shuffled copy, for the negative control.

    Returns:
        The three shared columns, then the product's four.
    """
    product_columns = wind_columns(product=product)
    if shuffled:
        product_columns = tuple(f"{column}{SHUFFLED_SUFFIX}" for column in product_columns)
    return (*WIND_SHARED_FEATURES, *product_columns)


def temperature_column(*, product: str, shuffled: bool = False) -> str:
    """Return a product's temperature column in the solar rows.

    Args:
        product: `era5` or `ukv_ceda`.
        shuffled: Whether to return the shuffled copy.

    Returns:
        The column name.
    """
    if product not in PRODUCTS:
        msg = f"unknown product {product!r}"
        raise ValueError(msg)
    return f"{product}_temp{SHUFFLED_SUFFIX if shuffled else ''}"


def solar_arm_columns(*, product: str, shuffled: bool = False) -> tuple[str, ...]:
    """Return a solar arm's ten feature columns.

    The two arms differ only in the temperature column. The shuffled arm shuffles the temperature
    only and leaves the CAMS irradiance intact, so the contrast measures temperature alone.

    Args:
        product: `era5` or `ukv_ceda`.
        shuffled: Whether the arm reads the product's shuffled temperature.

    Returns:
        The six shared columns, the product's temperature, and the three CAMS irradiances.
    """
    return (
        *SOLAR_SHARED_FEATURES,
        temperature_column(product=product, shuffled=shuffled),
        *SOLAR_IRRADIANCE_FEATURES,
    )


def matched_wind_arm_columns(*, product: str) -> tuple[str, ...]:
    """Return a post hoc matched-height wind arm's six feature columns: the 10 m wind alone.

    Both products give their 10 m speed and the sine and cosine of their 10 m direction, so the pair
    differs in product and served lead but not in height.

    Args:
        product: `era5` or `ukv_ceda`.

    Returns:
        The three shared columns, then the 10 m speed, sine and cosine.

    Raises:
        ValueError: If the product is not one of `PRODUCTS`.
    """
    if product not in PRODUCTS:
        msg = f"unknown product {product!r}"
        raise ValueError(msg)
    return (
        *WIND_SHARED_FEATURES,
        f"{product}_speed_10m",
        f"{product}_sin_10m",
        f"{product}_cos_10m",
    )


def check_arm_widths() -> None:
    """Raise unless the two arms of every contrast carry the same number of distinct columns."""
    for arms in (
        {product: matched_wind_arm_columns(product=product) for product in PRODUCTS},
        {product: wind_arm_columns(product=product) for product in PRODUCTS},
        {product: solar_arm_columns(product=product) for product in PRODUCTS},
        {product: wind_arm_columns(product=product, shuffled=True) for product in PRODUCTS},
        {product: solar_arm_columns(product=product, shuffled=True) for product in PRODUCTS},
    ):
        check_column_counts(features=arms, contrasts=[("ukv_ceda", "era5")])


ContrastReadingType = Literal[
    "ukv_clearly_better", "era5_clearly_better", "small_not_clear", "no_clear_difference"
]
"""How a contrast reads against its margin."""


def contrast_reading(
    *, difference: float, lower: float, upper: float, margin: float
) -> ContrastReadingType:
    """Read one contrast, UKV-CEDA minus ERA5, against its margin.

    A contrast clearly favours a product only when its 95% interval lies wholly on one side of zero
    and the point estimate is beyond the margin. A negative difference favours UKV-CEDA.

    Args:
        difference: The point estimate, UKV-CEDA minus ERA5.
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        margin: The margin, a positive number in the contrast's own unit.

    Returns:
        `ukv_clearly_better`, `era5_clearly_better`, `small_not_clear` where the interval excludes
        zero but the estimate is within the margin, or `no_clear_difference`.
    """
    if upper < 0.0 and difference < -margin:
        return "ukv_clearly_better"
    if lower > 0.0 and difference > margin:
        return "era5_clearly_better"
    if upper < 0.0 or lower > 0.0:
        return "small_not_clear"
    return "no_clear_difference"


# --- Reading UKV-CEDA -----------------------------------------------------------------------------


class UkvStores(NamedTuple):
    """The three stores, opened read-only, and the cell centres they share.

    Attributes:
        stores: The stores, in date order.
        latitude: Each cell's latitude. Private: never printed.
        longitude: Each cell's longitude. Private: never printed.
        snapshot_ids: The Icechunk snapshot read from each store.
    """

    stores: tuple[Store, ...]
    latitude: np.ndarray
    longitude: np.ndarray
    snapshot_ids: tuple[str, ...]


def _array(*, group: zarr.Group, name: str) -> zarr.Array:
    member = group[name]
    if not isinstance(member, zarr.Array):
        msg = f"{name} is not an array"
        raise TypeError(msg)
    return member


def open_ukv_stores(*, store_dirs: Sequence[Path] = STORE_DIRS) -> UkvStores:
    """Open each store read-only on its main branch, and check the three share one set of cells.

    Args:
        store_dirs: The product folders, each holding `store`.

    Returns:
        The stores and their shared cell centres.

    Raises:
        ValueError: If a store holds another product, or the stores' cell centres differ.
    """
    groups: list[ArrayGroup] = []
    statuses: list[np.ndarray] = []
    snapshots: list[str] = []
    centres: list[tuple[np.ndarray, np.ndarray]] = []
    for store_dir in store_dirs:
        repository = icechunk.Repository.open(
            storage=icechunk.local_filesystem_storage(str(store_dir / "store"))
        )
        session = repository.readonly_session(branch="main")
        group = zarr.open_group(session.store, mode="r")
        if group.attrs.get("product") != DEFAULT_PROFILE.product_name:
            msg = f"{store_dir.name} holds {group.attrs.get('product')!r}"
            raise ValueError(msg)
        groups.append(group)
        statuses.append(np.asarray(_array(group=group, name="status")[:], dtype=np.int8))
        snapshots.append(session.snapshot_id)
        centres.append(
            (
                np.asarray(_array(group=group, name="cell_latitude")[:]),
                np.asarray(_array(group=group, name="cell_longitude")[:]),
            )
        )
    for latitude, longitude in centres[1:]:
        if not (
            np.array_equal(latitude, centres[0][0]) and np.array_equal(longitude, centres[0][1])
        ):
            msg = "the three stores do not share one set of cells"
            raise ValueError(msg)
    return UkvStores(
        stores=make_stores(groups=groups, statuses=statuses),
        latitude=centres[0][0],
        longitude=centres[0][1],
        snapshot_ids=tuple(snapshots),
    )


def nearest_ukv_cells(*, ukv: UkvStores, points: pl.DataFrame) -> pl.DataFrame:
    """Find each point's nearest UKV-CEDA cell, and check the point lies inside the crop.

    Args:
        ukv: The opened stores.
        points: Rows with `site`, `latitude` and `longitude`.

    Returns:
        `site`, `cell_id` (a flat index into the stores' cell axis) and `distance_km`.

    Raises:
        ValueError: If a point lies farther than `CELL_DISTANCE_LIMIT_KM` from every cell.
    """
    cells = pl.DataFrame(
        {
            "cell_id": np.arange(len(ukv.latitude)),
            "latitude": ukv.latitude,
            "longitude": ukv.longitude,
        }
    )
    nearest = nearest_cells(sites=points, cells=cells)
    outside = nearest.filter(pl.col("distance_km") > CELL_DISTANCE_LIMIT_KM)
    if outside.height:
        msg = f"{outside.height} points lie outside the UKV-CEDA crop"
        raise ValueError(msg)
    return nearest


def ukv_at(
    *,
    ukv: UkvStores,
    hours: pl.Series,
    offsets_hours: Sequence[int],
    variables: Sequence[str],
    cells: np.ndarray,
    read_values: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Read UKV-CEDA at each hour, averaging the instants an hour is built from.

    Args:
        ukv: The opened stores.
        hours: The hours' labels, as a UTC datetime series.
        offsets_hours: The instants each hour reads, as offsets from its label.
        variables: The store arrays to read.
        cells: Flat cell indices, one column of each value array per cell.
        read_values: Whether to read values. When false, only run availability is decided, and
            every value is zero, which is enough to decide which hours survive.

    Returns:
        The usable flag per hour, each variable's mean over the instants (shape hours by cells),
        and the lead in hours of the hour's label instant.
    """
    usable = usable_hours(stores=ukv.stores, hours=hours, offsets_hours=offsets_hours)
    label = read_at_instants(
        stores=ukv.stores,
        instants=hours,
        variables=variables if read_values else (),
        cells=cells,
    )
    means = {name: np.zeros((len(hours), len(cells))) for name in variables}
    if read_values:
        for offset in offsets_hours:
            reads = (
                label
                if offset == 0
                else read_at_instants(
                    stores=ukv.stores,
                    instants=hours.dt.offset_by(f"{offset}h"),
                    variables=variables,
                    cells=cells,
                )
            )
            for name in variables:
                means[name] += reads.values[name] / len(offsets_hours)
    return usable, means, label.lead_hours


# --- Reading ERA5 ---------------------------------------------------------------------------------


def wind_from_components(*, u: pl.Expr, v: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Return the wind speed and the direction it blows from, in degrees, from its components.

    Args:
        u: The eastward component.
        v: The northward component.

    Returns:
        The speed, and the direction clockwise from north that the wind blows from (a wind from the
        west, with `u` positive and `v` zero, is 270).
    """
    speed = (u.cast(pl.Float64) ** 2 + v.cast(pl.Float64) ** 2).sqrt()
    direction = (270.0 - pl.arctan2(v.cast(pl.Float64), u.cast(pl.Float64)).degrees()) % 360.0
    return speed, direction


def era5_wind_by_cell() -> pl.DataFrame:
    """Read ERA5's native wind at every cell of every wind-farm and station block.

    The 2019-09 to 2023-12 download is keyed by cell. The 2024-01 to 2026-09 download is keyed by
    wind farm and offset within the farm's block, and `cell_groups.parquet` says which cell each
    (farm, offset) is. Both are returned keyed by the cell's position, so a station's cell finds
    the 2024 onward values wherever a wind-farm block covers it.

    Returns:
        `lat_q`, `lon_q` (the cell in whole quarter degrees), `time` (UTC), `speed_10m`,
        `speed_100m`, `direction_100m` and `direction_10m` (degrees the wind blows from).
    """
    groups = pl.read_parquet(ERA5_CELL_GROUPS_PATH)
    positions = groups.select("cell_id", "lat_q", "lon_q").unique()
    early = pl.read_parquet(ERA5_WIND_CDS_PATH).join(positions, on="cell_id")
    wind_blocks = groups.filter(pl.col("kind") == "wind").select(
        site="label", dy="dy", dx="dx", lat_q="lat_q", lon_q="lon_q"
    )
    late = (
        pl.read_parquet(ERA5_WIND_NATIVE_PATH)
        .cast({"dy": pl.Int8, "dx": pl.Int8})
        .with_columns(time=pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"))
        .join(wind_blocks, on=["site", "dy", "dx"])
    )
    stacked = pl.concat(
        [
            frame.select("lat_q", "lon_q", "time", "u10", "v10", "u100", "v100")
            for frame in (early, late)
        ]
    )
    speed_10m, direction_10m = wind_from_components(u=pl.col("u10"), v=pl.col("v10"))
    speed_100m, direction_100m = wind_from_components(u=pl.col("u100"), v=pl.col("v100"))
    wind = stacked.select(
        "lat_q",
        "lon_q",
        "time",
        speed_10m=speed_10m,
        speed_100m=speed_100m,
        direction_100m=direction_100m,
        direction_10m=direction_10m,
    ).unique(subset=["lat_q", "lon_q", "time"], keep="first")
    return wind.sort("lat_q", "lon_q", "time")


def block_centres(*, kind: str) -> pl.DataFrame:
    """Return the cell at the centre of each wind-farm or station block.

    Args:
        kind: `wind` or `station`.

    Returns:
        `label`, `lat_q` and `lon_q`, one row per block.
    """
    return (
        pl.read_parquet(ERA5_CELL_GROUPS_PATH)
        .filter(pl.col("kind") == kind, pl.col("dy") == 0, pl.col("dx") == 0)
        .select("label", "lat_q", "lon_q")
    )


def era5_temperature_by_cell() -> pl.DataFrame:
    """Read Open-Meteo's ERA5 2 m air temperature, which is an instant at its label.

    Returns:
        `latitude`, `longitude`, `time` and `temp_c`.
    """
    return read_era5(source="open-meteo").select("latitude", "longitude", "time", "temp_c")


def with_era5_cells(*, points: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Attach the ERA5 cell centre nearest each point, and check the point lies inside the grid.

    Args:
        points: Rows with `site`, `latitude` and `longitude`.
        era5: ERA5 temperature rows, for the grid's cell centres.

    Returns:
        `points` with `cell_latitude` and `cell_longitude`.

    Raises:
        ValueError: If a point is farther than half a cell from its nearest centre, which means the
            point lies outside the downloaded cells.
    """
    placed = nearest_era5_cell(sites=points, era5=era5)
    outside = placed.filter(
        ((pl.col("latitude") - pl.col("cell_latitude")).abs() > ERA5_HALF_CELL_DEG)
        | ((pl.col("longitude") - pl.col("cell_longitude")).abs() > ERA5_HALF_CELL_DEG)
    )
    if outside.height:
        msg = f"{outside.height} points lie outside the ERA5 temperature cells"
        raise ValueError(msg)
    return placed


def era5_temperature_at(*, points: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Return each point's ERA5 temperature at its nearest cell.

    Args:
        points: Rows with `site`, `latitude` and `longitude`.
        era5: `era5_temperature_by_cell`'s result.

    Returns:
        `site`, `time` and `temp_c`.
    """
    placed = with_era5_cells(points=points, era5=era5)
    return placed.join(
        era5,
        left_on=["cell_latitude", "cell_longitude"],
        right_on=["latitude", "longitude"],
    ).select("site", "time", "temp_c")


# --- Set A: station-hours -------------------------------------------------------------------------


class Stations(NamedTuple):
    """The stations inside the crop that set A scores.

    Attributes:
        points: `site` (S1 upwards), `latitude`, `longitude`, `src_id`, `ukv_cell`. Private.
        n_in_crop: How many hourly stations lie inside the crop.
        coverage: Each station's share of hours with both a wind speed and a temperature.
    """

    points: pl.DataFrame
    n_in_crop: int
    coverage: pl.DataFrame


def station_readings() -> pl.DataFrame:
    """Read the hourly stations' wind speed and air temperature on whole hours from 2019-09.

    Returns:
        `src_id`, `time`, `wind_speed_m_s` and `air_temperature`.
    """
    return read_hourly_weather(
        path=HOURLY_WEATHER_PATH, columns=["wind_speed_m_s", "air_temperature"]
    ).filter(
        pl.col("time") >= WINDOW_START,
        pl.col("time") < STATION_WINDOW_END,
        pl.col("time").dt.minute() == 0,
        pl.col("time").dt.second() == 0,
    )


def choose_stations(*, ukv: UkvStores, readings: pl.DataFrame) -> Stations:
    """Choose the stations inside the UKV-CEDA crop that report wind and temperature.

    The rule reads no score. A station is chosen when its nearest UKV-CEDA cell lies within
    `CELL_DISTANCE_LIMIT_KM` and it has both a wind speed and a temperature at no less than
    `MIN_STATION_COVERAGE` of the hours of the window.

    Args:
        ukv: The opened stores.
        readings: `station_readings`'s result.

    Returns:
        The stations, labelled S1 upwards in an order that does not follow the station identifier.
    """
    metadata = read_station_metadata(path=STATION_METADATA_PATH).filter(
        pl.col("src_id").is_in(readings["src_id"].unique().to_list())
    )
    points = metadata.select(site="src_id", latitude="latitude", longitude="longitude")
    cells = pl.DataFrame(
        {
            "cell_id": np.arange(len(ukv.latitude)),
            "latitude": ukv.latitude,
            "longitude": ukv.longitude,
        }
    )
    nearest = nearest_cells(sites=points, cells=cells)
    in_crop = nearest.filter(pl.col("distance_km") <= CELL_DISTANCE_LIMIT_KM)
    n_hours = int((STATION_WINDOW_END - WINDOW_START).total_seconds() // 3600)
    coverage = (
        readings.filter(
            pl.col("src_id").is_in(in_crop["site"].to_list()),
            pl.col("wind_speed_m_s").is_not_null(),
            pl.col("air_temperature").is_not_null(),
        )
        .group_by("src_id")
        .agg(coverage=pl.len() / n_hours)
    )
    chosen = coverage.filter(pl.col("coverage") >= MIN_STATION_COVERAGE)
    order = np.random.default_rng(SHUFFLE_SEED).permutation(chosen.height) + 1
    labelled = (
        chosen.sort("src_id")
        .with_columns(index=pl.Series(order))
        .select("src_id", site=pl.format("S{}", pl.col("index")), coverage="coverage")
        .join(metadata.select("src_id", "latitude", "longitude"), on="src_id")
        .join(
            in_crop.select(src_id="site", ukv_cell="cell_id"),
            on="src_id",
        )
        .sort("site")
    )
    return Stations(
        points=labelled, n_in_crop=in_crop.height, coverage=labelled.select("site", "coverage")
    )


def station_hours(
    *,
    ukv: UkvStores,
    stations: Stations,
    readings: pl.DataFrame,
    wind: pl.DataFrame,
    temperature: pl.DataFrame,
    hours: pl.Series,
    read_values: bool,
    drop_months: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """Build set A: one row per station and hour with the station, ERA5, and UKV-CEDA values.

    Args:
        ukv: The opened stores.
        stations: `choose_stations`'s result.
        readings: `station_readings`'s result.
        wind: `era5_wind_by_cell`'s result.
        temperature: `era5_temperature_by_cell`'s result.
        hours: The hours to build, as a UTC datetime series.
        read_values: Whether to read UKV-CEDA values, as `ukv_at` says.
        drop_months: The months that lose more than `MAX_MONTH_LOSS_SHARE` of their hours to
            incomplete UKV-CEDA runs, dropped from every arm.

    Returns:
        `site`, `time`, `month`, `era_code`, `lead_hours`, `station_wind_m_s`, `station_temp_c`,
        `era5_wind_m_s`, `era5_temp_c`, `ukv_wind_m_s` and `ukv_temp_c`. A value that a product or a
        station does not have at the hour is null. The straddling months and the dropped ERA5
        temperature days are removed.
    """
    points = stations.points
    usable, means, leads = ukv_at(
        ukv=ukv,
        hours=hours,
        offsets_hours=(0,),
        variables=(UKV_WIND_VARIABLES[0], UKV_TEMPERATURE_VARIABLE),
        cells=points["ukv_cell"].to_numpy(),
        read_values=read_values,
    )
    ukv_rows = pl.concat(
        pl.DataFrame(
            {
                "site": site,
                "time": hours,
                "lead_hours": pl.Series(leads, dtype=pl.Int64),
                "ukv_usable": usable,
                "ukv_wind_m_s": means[UKV_WIND_VARIABLES[0]][:, column],
                "ukv_temp_c": means[UKV_TEMPERATURE_VARIABLE][:, column] - KELVIN,
            }
        )
        for column, site in enumerate(points["site"])
    ).with_columns(
        ukv_wind_m_s=pl.when(pl.col("ukv_usable")).then(pl.col("ukv_wind_m_s")),
        ukv_temp_c=pl.when(pl.col("ukv_usable")).then(pl.col("ukv_temp_c")),
    )
    station_cells = block_centres(kind="station").join(
        points.select("src_id", "site"), left_on="label", right_on="src_id"
    )
    era5_wind = station_cells.join(wind, on=["lat_q", "lon_q"]).select(
        "site", "time", era5_wind_m_s="speed_10m"
    )
    era5_temp = era5_temperature_at(
        points=points.select("site", "latitude", "longitude"), era5=temperature
    ).rename({"temp_c": "era5_temp_c"})
    observed = readings.join(points.select("src_id", "site"), on="src_id").select(
        "site", "time", station_wind_m_s="wind_speed_m_s", station_temp_c="air_temperature"
    )
    low, high = DROPPED_ERA5_TEMPERATURE_DAYS
    rows = (
        ukv_rows.drop("ukv_usable")
        .join(observed, on=["site", "time"], how="left")
        .join(era5_wind, on=["site", "time"], how="left")
        .join(era5_temp, on=["site", "time"], how="left")
        .filter(~pl.col("time").is_between(low, high, closed="left"))
    )
    return (
        add_time_features(dataset=rows)
        .filter(~pl.col("month").is_in([*STRADDLING_MONTHS, *drop_months]))
        .with_columns(
            era_code=sum((pl.col("month") >= month).cast(pl.Int8) for month in ERA_FIRST_MONTHS)
        )
        .select(
            "site",
            "time",
            "month",
            "era_code",
            "lead_hours",
            "station_wind_m_s",
            "station_temp_c",
            "era5_wind_m_s",
            "era5_temp_c",
            "ukv_wind_m_s",
            "ukv_temp_c",
        )
        .sort("site", "time")
    )


# --- Set B: wind and solar rows -------------------------------------------------------------------


def _shuffled(*, frame: pl.DataFrame, groups: Sequence[Sequence[str]]) -> pl.DataFrame:
    """Add shuffled copies of column groups within site, year-month and hour of day.

    Args:
        frame: Rows with `site`, `month` and `hour_of_day`.
        groups: The column groups to shuffle, each jointly under one permutation.

    Returns:
        `frame` with a `_shuffled` copy of every column in `groups`.
    """
    return climatology_permutation(
        frame=frame,
        column_groups=groups,
        by=SHUFFLE_GROUPS,
        seed=SHUFFLE_SEED,
        suffix=SHUFFLED_SUFFIX,
    )


def wind_rows(
    *,
    ukv: UkvStores,
    wind: pl.DataFrame,
    sites: pl.DataFrame,
    read_values: bool,
    drop_zero_hours: bool = True,
    drop_months: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """Build the wind farms' fit rows, with the shuffled control columns, eras and folds.

    Args:
        ukv: The opened stores.
        wind: `era5_wind_by_cell`'s result.
        sites: The wind roster with `site`, `latitude`, `longitude`, `effective_capacity_mw`.
        read_values: Whether to read UKV-CEDA values, as `ukv_at` says.
        drop_zero_hours: Whether to drop every hour holding an exactly zero half-hour. False builds
            the exploratory rows that keep them.
        drop_months: The months that lose more than `MAX_MONTH_LOSS_SHARE` of their hours to
            incomplete UKV-CEDA runs, dropped from every arm.

    Returns:
        One row per (site, hour) with the centred and the hour-ending power, both products' wind
        columns, `constrained`, `cap_mw`, `month`, `era_code`, `fold`, and the shuffled columns.
    """
    centred = wind_hourly_power(sites=sites, centred=True)
    ending = wind_hourly_power(sites=sites, centred=False).select(
        "site",
        "time",
        power_hour_ending_mw="power_mw",
        zero_hour_ending="has_zero_half_hour",
    )
    power = (
        centred.join(ending, on=["site", "time"], how="inner")
        .with_columns(has_zero_half_hour=pl.col("has_zero_half_hour") | pl.col("zero_hour_ending"))
        .drop("zero_hour_ending")
        .join(sites.select("site", "effective_capacity_mw"), on="site")
    )
    centres = block_centres(kind="wind").rename({"label": "site"})
    speed_100m, sine, cosine, speed_10m = wind_columns(product="era5")
    era5 = centres.join(wind, on=["lat_q", "lon_q"]).select(
        "site",
        "time",
        pl.col("speed_100m").alias(speed_100m),
        pl.col("direction_100m").radians().sin().alias(sine),
        pl.col("direction_100m").radians().cos().alias(cosine),
        pl.col("speed_10m").alias(speed_10m),
    )
    hours = power["time"].unique().sort()
    cells = nearest_ukv_cells(ukv=ukv, points=sites.select("site", "latitude", "longitude"))
    usable, means, _ = ukv_at(
        ukv=ukv,
        hours=hours,
        offsets_hours=(0,),
        variables=UKV_WIND_VARIABLES,
        cells=cells["cell_id"].to_numpy(),
        read_values=read_values,
    )
    ukv_speed, ukv_sin, ukv_cos, ukv_upper = wind_columns(product="ukv_ceda")
    ukv_rows = (
        pl.concat(
            pl.DataFrame(
                {
                    "site": site,
                    "time": hours.filter(pl.Series(usable)),
                    ukv_speed: means[UKV_WIND_VARIABLES[0]][usable, column],
                    "direction": means[UKV_WIND_VARIABLES[1]][usable, column],
                    ukv_upper: means[UKV_WIND_VARIABLES[2]][usable, column],
                }
            )
            for column, site in enumerate(cells["site"])
        )
        .with_columns(
            pl.col("direction").radians().sin().alias(ukv_sin),
            pl.col("direction").radians().cos().alias(ukv_cos),
        )
        .drop("direction")
    )
    joined = power.join(era5, on=["site", "time"], how="inner").join(
        ukv_rows, on=["site", "time"], how="inner"
    )
    return _finish(
        frame=wind_common_rows(frame=joined, drop_zero_hours=drop_zero_hours),
        shuffle_groups=[wind_columns(product=product) for product in PRODUCTS],
        drop_months=drop_months,
    )


def solar_rows(
    *,
    ukv: UkvStores,
    temperature: pl.DataFrame,
    sites: pl.DataFrame,
    read_values: bool,
    drop_months: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """Build the solar farms' fit rows, with the shuffled temperature columns, eras and folds.

    Args:
        ukv: The opened stores.
        temperature: `era5_temperature_by_cell`'s result.
        sites: The solar roster with `site`, `latitude`, `longitude`.
        read_values: Whether to read UKV-CEDA values, as `ukv_at` says.
        drop_months: The months that lose more than `MAX_MONTH_LOSS_SHARE` of their hours to
            incomplete UKV-CEDA runs, dropped from every arm.

    Returns:
        One row per (site, daytime hour) with the power, the geometry, CAMS irradiance, both
        products' temperature, `constrained`, `cap_mw`, `month`, `era_code`, `fold`, and the
        shuffled temperature columns.
    """
    base = pl.read_parquet(dataset_path_for(source="cams_allhours")).select(
        "site",
        "time",
        "power_mw",
        "effective_capacity_mw",
        "solar_zenith_deg",
        "solar_azimuth_deg",
        "extraterrestrial_horizontal_w_m2",
        ghi_cams="ghi_w_m2",
        bhi_cams="bhi_w_m2",
        dhi_cams="dhi_w_m2",
    )
    era5_series = era5_temperature_at(
        points=sites.select("site", "latitude", "longitude"), era5=temperature
    )
    era5 = with_neighbouring_hours(
        frame=base.select("site", "time"),
        source=era5_series,
        columns={"temp_previous": ("temp_c", -1), "temp_label": ("temp_c", 0)},
    ).select(
        "site",
        "time",
        era5_temp=(pl.col("temp_previous") + pl.col("temp_label")) / 2.0,
    )
    hours = base["time"].unique().sort()
    cells = nearest_ukv_cells(ukv=ukv, points=sites.select("site", "latitude", "longitude"))
    usable, means, _ = ukv_at(
        ukv=ukv,
        hours=hours,
        offsets_hours=(-1, 0),
        variables=(UKV_TEMPERATURE_VARIABLE,),
        cells=cells["cell_id"].to_numpy(),
        read_values=read_values,
    )
    ukv_rows = pl.concat(
        pl.DataFrame(
            {
                "site": site,
                "time": hours.filter(pl.Series(usable)),
                temperature_column(product="ukv_ceda"): means[UKV_TEMPERATURE_VARIABLE][
                    usable, column
                ]
                - KELVIN,
            }
        )
        for column, site in enumerate(cells["site"])
    )
    joined = (
        base.join(
            era5.rename({"era5_temp": temperature_column(product="era5")}), on=["site", "time"]
        )
        .join(ukv_rows, on=["site", "time"], how="inner")
        .sort("site", "time")
    )
    return _finish(
        frame=with_export_cap(dataset=solar_common_rows(frame=joined)),
        shuffle_groups=[(temperature_column(product=product),) for product in PRODUCTS],
        drop_months=drop_months,
    )


def _finish(
    *,
    frame: pl.DataFrame,
    shuffle_groups: Sequence[Sequence[str]],
    drop_months: frozenset[str],
) -> pl.DataFrame:
    """Drop the ERA5 temperature days and the lossy months, label eras and folds, and shuffle.

    Args:
        frame: Common rows with every arm column.
        shuffle_groups: The column groups to shuffle for the negative control.
        drop_months: The months that lose more than `MAX_MONTH_LOSS_SHARE` of their hours to
            incomplete UKV-CEDA runs, dropped from every arm.

    Returns:
        The rows ready to fit, sorted by site and time, because the shuffled controls and XGBoost's
        row subsampling both depend on row order. The straddling months are dropped by
        `with_ukv_eras`.
    """
    low, high = DROPPED_ERA5_TEMPERATURE_DAYS
    kept = (
        add_time_features(
            dataset=frame.filter(~pl.col("time").is_between(low, high, closed="left"))
        )
        .filter(~pl.col("month").is_in(list(drop_months)))
        .sort("site", "time")
    )
    cut, _ = with_ukv_eras(frame=kept)
    return _shuffled(frame=cut, groups=shuffle_groups)


def check_rows(
    *, wind: pl.DataFrame, solar: pl.DataFrame, wind_keep_zero_hours: pl.DataFrame
) -> None:
    """Raise unless every arm's columns are present on every row of both fit row sets.

    Args:
        wind: The wind rows.
        solar: The solar rows.
        wind_keep_zero_hours: The wind rows that keep the hours holding an exactly zero half-hour.
    """
    check_arm_widths()
    for frame, arms in (
        (wind_keep_zero_hours, [wind_arm_columns(product=p) for p in PRODUCTS]),
        (wind, [wind_arm_columns(product=p, shuffled=s) for p in PRODUCTS for s in (False, True)]),
        (
            solar,
            [solar_arm_columns(product=p, shuffled=s) for p in PRODUCTS for s in (False, True)],
        ),
    ):
        check_no_missing(
            frame=frame,
            columns=["power_mw", "effective_capacity_mw", *(c for arm in arms for c in arm)],
        )


# --- The coverage check ---------------------------------------------------------------------------


def _all_hours() -> pl.Series:
    return pl.datetime_range(
        WINDOW_START, COVERAGE_WINDOW_END, interval="1h", time_zone="UTC", eager=True, closed="left"
    )


def run_status_lines(*, ukv: UkvStores) -> list[str]:
    """Count the runs of each store by status, and the runs CEDA holds no directory for.

    Args:
        ukv: The opened stores.

    Returns:
        Markdown lines.
    """
    lines = ["#### Runs by status", ""]
    totals = {"complete": 0, "partial": 0, "missing": 0, "unlisted": 0}
    unlisted_slots: list[int] = []
    for position, store in enumerate(ukv.stores, start=1):
        owned = store.statuses[store.first_slot : store.end_slot]
        counts = {
            "complete": int((owned == STATUS_COMPLETE).sum()),
            "partial": int((owned == STATUS_PARTIAL).sum()),
            "missing": int((owned == STATUS_MISSING).sum()),
            "unlisted": int((owned == 0).sum()),
        }
        for name, count in counts.items():
            totals[name] += count
        unlisted_slots += [store.first_slot + int(i) for i in np.flatnonzero(owned == 0)]
        lines.append(f"- Store {position}: {counts}.")
    days = {slot * DEFAULT_PROFILE.cycle_hours // 24 for slot in unlisted_slots}
    n_runs = sum(totals.values())
    lost = n_runs - totals["complete"]
    lines += [
        (
            f"- All stores: {totals}, {n_runs:,} runs, of which {lost:,} ({lost / n_runs:.2%}) are "
            f"partial, missing or unlisted. The unlisted runs fall on {len(days)} days."
        ),
        (
            "- Not re-checked against CEDA here: re-listing the unlisted days and the partial runs "
            "needs the fetcher and a CEDA token, so these counts come from the stores' own status "
            "arrays."
        ),
    ]
    return lines


def hours_lost_lines(*, ukv: UkvStores) -> tuple[list[str], dict[str, float]]:
    """Count, for each month, the share of hours lost to incomplete runs, for wind and for solar.

    Args:
        ukv: The opened stores.

    Returns:
        Markdown lines, and each month that loses more than `MAX_MONTH_LOSS_SHARE` of its hours by
        either rule, with the share lost.
    """
    hours = _all_hours()
    frame = pl.DataFrame(
        {
            "month": hours.dt.strftime("%Y-%m"),
            "wind_lost": ~usable_hours(stores=ukv.stores, hours=hours, offsets_hours=(0,)),
            "solar_lost": ~usable_hours(stores=ukv.stores, hours=hours, offsets_hours=(-1, 0)),
        }
    )
    monthly = (
        frame.group_by("month")
        .agg(wind=pl.col("wind_lost").mean(), solar=pl.col("solar_lost").mean(), hours=pl.len())
        .sort("month")
    )
    worst = monthly.select("month", share=pl.max_horizontal("wind", "solar")).filter(
        pl.col("share") > MAX_MONTH_LOSS_SHARE
    )
    shown = monthly.filter((pl.col("wind") > 0) | (pl.col("solar") > 0))
    lines = [
        "#### Share of hours lost to incomplete UKV-CEDA runs, by month",
        "",
        (
            f"Months with no hour lost: {monthly.height - shown.height} of {monthly.height}. "
            f"Months losing more than {MAX_MONTH_LOSS_SHARE:.0%}, which the build drops from every "
            f"arm and every set: {worst.height}."
        ),
        "",
        "| Month | Wind (one instant) | Solar (two instants) |",
        "|---|---|---|",
        *(
            f"| {row['month']} | {row['wind']:.1%} | {row['solar']:.1%} |"
            for row in shown.iter_rows(named=True)
        ),
    ]
    return lines, dict(worst.iter_rows())


def era5_coverage_lines(
    *, wind: pl.DataFrame, temperature: pl.DataFrame, stations: Stations
) -> list[str]:
    """Print each ERA5 product's coverage at the generators and at each station.

    Args:
        wind: `era5_wind_by_cell`'s result.
        temperature: `era5_temperature_by_cell`'s result.
        stations: The chosen stations.

    Returns:
        Markdown lines.
    """
    station_cells = block_centres(kind="station").join(
        stations.points.select("src_id", "site"), left_on="label", right_on="src_id"
    )
    lines = ["#### ERA5 coverage", ""]
    for noun, centres, key in (
        ("wind farms", block_centres(kind="wind"), "label"),
        ("stations", station_cells, "site"),
    ):
        spans = (
            centres.join(wind, on=["lat_q", "lon_q"])
            .group_by(key)
            .agg(first=pl.col("time").min(), last=pl.col("time").max())
        )
        lines.append(
            f"- ERA5 native wind at the {noun}: {spans.height} of {centres.height} blocks have "
            f"data, from {spans['first'].min():%Y-%m-%d} to {spans['last'].max():%Y-%m-%d}; the "
            f"shortest block's record ends on {spans['last'].min():%Y-%m-%d}."
        )
    with_2024 = station_cells.join(
        wind.filter(pl.col("time") >= datetime(2024, 1, 1, tzinfo=UTC))
        .select("lat_q", "lon_q")
        .unique(),
        on=["lat_q", "lon_q"],
    )
    lines.append(
        f"- Stations whose ERA5 wind cell has values from 2024-01: {with_2024.height} of "
        f"{station_cells.height}."
    )
    n_cells = temperature.select(pl.struct("latitude", "longitude").n_unique()).item()
    lines.append(
        f"- ERA5 temperature: {n_cells} cells, {temperature['time'].min():%Y-%m-%d} to "
        f"{temperature['time'].max():%Y-%m-%d}."
    )
    placed = with_era5_cells(
        points=stations.points.select("site", "latitude", "longitude"), era5=temperature
    )
    same_cell = placed.join(station_cells, on="site").filter(
        (pl.col("cell_latitude") == pl.col("lat_q") * 0.25)
        & (pl.col("cell_longitude") == pl.col("lon_q") * 0.25)
    )
    lines.append(
        f"- Stations whose ERA5 temperature cell is the cell of their ERA5 wind block: "
        f"{same_cell.height} of {stations.points.height}."
    )
    return lines


def station_lines(
    *, stations: Stations, readings: pl.DataFrame, generators: pl.DataFrame
) -> list[str]:
    """Print the station counts, coverage per year, and pooled distances, with no identifier.

    Args:
        stations: The chosen stations.
        readings: `station_readings`'s result.
        generators: Every generator's `latitude` and `longitude`.

    Returns:
        Markdown lines.
    """
    points = stations.points
    distance = distance_matrix_km(sites=points, cells=generators).min(axis=1)
    window = pl.datetime_range(
        WINDOW_START, STATION_WINDOW_END, interval="1h", time_zone="UTC", eager=True, closed="left"
    )
    hours_per_year = (
        pl.DataFrame({"year": window.dt.year()}).group_by("year").agg(window_hours=pl.len())
    )
    per_year = (
        readings.join(points.select("src_id", "site"), on="src_id")
        .filter(pl.col("wind_speed_m_s").is_not_null(), pl.col("air_temperature").is_not_null())
        .group_by("site", year=pl.col("time").dt.year())
        .agg(hours=pl.len())
        .join(hours_per_year, on="year")
        .with_columns(share=pl.col("hours") / pl.col("window_hours"))
        .group_by("year")
        .agg(lowest=pl.col("share").min(), highest=pl.col("share").max())
        .sort("year")
    )
    per_station = ", ".join(
        f"{row['site']} {row['coverage']:.1%}" for row in stations.coverage.iter_rows(named=True)
    )
    return [
        "#### Stations",
        "",
        (
            f"- Hourly stations inside the UKV-CEDA crop: {stations.n_in_crop}. Stations with "
            f"wind speed and temperature at {MIN_STATION_COVERAGE:.0%} or more of hours, which "
            f"set A scores: {points.height} (the plan found {PLANNED_STATION_COUNT})."
        ),
        f"- Share of hours with both readings, per station (S1 upwards): {per_station}.",
        (
            f"- Distance to the nearest generator, pooled over stations: {distance.min():.0f} km "
            f"to {distance.max():.0f} km. Stations farther than {NEAR_GENERATOR_KM:.0f} km from "
            f"every generator: {int((distance > NEAR_GENERATOR_KM).sum())} of {points.height}."
        ),
        "- Share of hours with both readings, by year, lowest to highest station: "
        + "; ".join(
            f"{row['year']} {row['lowest']:.0%} to {row['highest']:.0%}"
            for row in per_year.iter_rows(named=True)
        )
        + ".",
    ]


def row_set_lines(
    *,
    wind: pl.DataFrame,
    solar: pl.DataFrame,
    offsets: Mapping[str, Mapping[int, int]],
    n_designs: Mapping[str, int],
) -> list[str]:
    """Print the fit row sets' size by generator-year, era, and window, with no value.

    Args:
        wind: The wind rows.
        solar: The solar rows.
        offsets: The chosen fold rotation of each row set.
        n_designs: How many rotations cover every calendar month, per row set.

    Returns:
        Markdown lines.
    """
    lines = ["#### Fit rows", ""]
    for name, frame in (("wind", wind), ("solar", solar)):
        by_year = (
            frame.group_by("site", year=pl.col("time").dt.year())
            .agg(hours=pl.len())
            .sort("site", "year")
        )
        small = by_year.filter(pl.col("hours") < MIN_SCORED_HOURS_SHOWN)
        early = frame.filter(pl.col("month") < EARLY_END_MONTH)
        early_months = early.group_by("site").agg(months=pl.col("month").n_unique())
        enough = early_months.filter(pl.col("months") >= 6).height
        months_per_era = dict(
            sorted(frame.group_by("era_code").agg(m=pl.col("month").n_unique()).iter_rows())
        )
        late_months = frame.filter(pl.col("month") >= EARLY_END_MONTH)["month"].n_unique()
        lines += [
            (
                f"- {name.capitalize()} rows: {frame.height:,}, {frame['site'].n_unique()} "
                f"generators, {frame['month'].n_unique()} scored months "
                f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})."
            ),
            (
                f"  - Scored months per era: {months_per_era}. Early window (before "
                f"{EARLY_END_MONTH}): {early['month'].n_unique()} months, {early.height:,} rows; "
                f"late window: {late_months} months."
            ),
            f"  - Generators with at least 6 early-window months: {enough} of "
            f"{frame['site'].n_unique()}. Per-generator early-window rows: "
            + ", ".join(
                f"{site} {count:,}"
                for site, count in sorted(early.group_by("site").len().iter_rows())
            )
            + ".",
            (
                f"  - Generator-years below {MIN_SCORED_HOURS_SHOWN:,} scored hours (not shown "
                f"separately): {small.height} of {by_year.height}."
            ),
            (
                f"  - Fold rotation by era: {dict(offsets[name])}; rotations covering every "
                f"calendar month: {n_designs[name]}."
            ),
        ]
    return lines


def _generators() -> pl.DataFrame:
    return pl.concat(
        [frame.select("latitude", "longitude") for frame in (pv_sites(), wind_sites())]
    )


def coverage_check(*, read_values: bool = False) -> tuple[list[str], list[str]]:
    """Run the coverage check, with no value-based filter, and report which guards failed.

    Args:
        read_values: Whether to read UKV-CEDA values while building the row sets.

    Returns:
        Markdown lines, and the failed guards (empty when the check passes).
    """
    failures: list[str] = []
    ukv = open_ukv_stores()
    lines = ["### Coverage check", "", *run_status_lines(ukv=ukv), ""]
    loss_lines, worst = hours_lost_lines(ukv=ukv)
    lines += [*loss_lines, ""]
    drop_months = frozenset(worst)
    lines += [
        "Dropped from every arm and every set, as a planned-guard outcome: "
        + (", ".join(f"{month} ({share:.0%})" for month, share in sorted(worst.items())) or "none"),
        "",
    ]
    readings = station_readings()
    stations = choose_stations(ukv=ukv, readings=readings)
    wind = era5_wind_by_cell()
    temperature = era5_temperature_by_cell()
    if stations.points.height != PLANNED_STATION_COUNT:
        failures.append(f"{stations.points.height} stations are scored, the plan found 4")
    lines += [
        *era5_coverage_lines(wind=wind, temperature=temperature, stations=stations),
        "",
        *station_lines(stations=stations, readings=readings, generators=_generators()),
        "",
    ]
    wind_frame = wind_rows(
        ukv=ukv,
        wind=wind,
        sites=wind_sites(),
        read_values=read_values,
        drop_months=drop_months,
    )
    keep_zero_frame = wind_rows(
        ukv=ukv,
        wind=wind,
        sites=wind_sites(),
        read_values=read_values,
        drop_zero_hours=False,
        drop_months=drop_months,
    )
    solar_frame = solar_rows(
        ukv=ukv,
        temperature=temperature,
        sites=pv_sites(),
        read_values=read_values,
        drop_months=drop_months,
    )
    designs = {
        "wind": ukv_fold_designs(frame=wind_frame),
        "solar": ukv_fold_designs(frame=solar_frame),
    }
    _, wind_offsets = with_ukv_eras(frame=wind_frame)
    _, solar_offsets = with_ukv_eras(frame=solar_frame)
    lines += row_set_lines(
        wind=wind_frame,
        solar=solar_frame,
        offsets={"wind": wind_offsets, "solar": solar_offsets},
        n_designs={name: len(found) for name, found in designs.items()},
    )
    check_rows(wind=wind_frame, solar=solar_frame, wind_keep_zero_hours=keep_zero_frame)
    lines += [
        "",
        "- Arm feature columns, by arm type: "
        + "; ".join(
            f"{kind} {product}: {len(columns)}"
            for kind, function in (("wind", wind_arm_columns), ("solar", solar_arm_columns))
            for product in PRODUCTS
            for columns in (function(product=product),)
        )
        + ".",
    ]
    return lines, failures


# --- Writing --------------------------------------------------------------------------------------


INPUT_FILES: Final[tuple[Path, ...]] = (
    ERA5_WIND_CDS_PATH,
    ERA5_WIND_NATIVE_PATH,
    ERA5_CELL_GROUPS_PATH,
    OPEN_METEO_PATH,
    dataset_path_for(source="cams_allhours"),
    HOURLY_WEATHER_PATH,
    STATION_METADATA_PATH,
)
"""Every file the build reads apart from the stores and the two Delta tables, all hashed."""


def _file_hash(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class BuildStamp:
    """What the build records about its inputs, so a later run can tell what it read."""

    status_hashes: list[str]
    snapshot_ids: list[str]
    input_hashes: dict[str, str]
    rows: dict[str, int]
    dropped_months: dict[str, float]
    delta_versions: dict[str, int]

    def to_json(self) -> str:
        """Render the stamp as JSON."""
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


README_TEXT: Final[str] = """# UKV-CEDA against ERA5, rows and results

Private: this folder holds per-generator and per-station values and is never published.

- `station_hours.parquet`, `wind_rows.parquet`, `wind_rows_keep_zero_hours.parquet`,
  `solar_rows.parquet`, `build.json` (`ukv_ceda_vs_era5_build.py`): set A rows (one per anonymous
  station and hour), the three sets of fit rows with every arm's columns, and the hashes of the
  inputs with the row counts.
- `station_intervals.parquet`, `station_report.md` (`ukv_ceda_station_scores.py`): set A intervals
  and tables.
- `verify.md` (`ukv_ceda_vs_era5_verify.py`): column counts, lag scan, monthly steps, and leads.
- `losses_<domain>.parquet`, `losses_<domain>.fingerprint` (`ukv_ceda_vs_era5_fit.py`): per-row
  losses with the actual value, the out-of-fold prediction, the site, time, fold, seed, arm and
  setting, and the hash of the rows, jobs, seeds and settings.
- `intervals.parquet`, `report.md`, `decision.md` (`ukv_ceda_vs_era5_fit.py`): set B intervals for
  every scope, every table the page quotes, and the decision rule applied.
"""


def with_era5_10m_direction(*, frame: pl.DataFrame, wind: pl.DataFrame) -> pl.DataFrame:
    """Add ERA5's 10 m wind direction, as a sine and a cosine, to the wind rows.

    Args:
        frame: The wind rows, carrying `site` and `time`.
        wind: `era5_wind_by_cell`'s result.

    Returns:
        `frame`, in its own row order, with `era5_sin_10m` and `era5_cos_10m`.

    Raises:
        ValueError: If a row has no ERA5 10 m direction.
    """
    direction = (
        block_centres(kind="wind")
        .rename({"label": "site"})
        .join(wind, on=["lat_q", "lon_q"])
        .select(
            "site",
            "time",
            era5_sin_10m=pl.col("direction_10m").radians().sin(),
            era5_cos_10m=pl.col("direction_10m").radians().cos(),
        )
    )
    joined = frame.join(direction, on=["site", "time"], how="left", maintain_order="left")
    check_no_missing(frame=joined, columns=["era5_sin_10m", "era5_cos_10m"])
    return joined


def with_hour_starting_power(*, frame: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Add the power of the hour that starts at each row's label, and keep the rows that have it.

    The hour starting at `T` is the hour ending at `T + 1 hour`, so it is read from the hour-ending
    series shifted back by an hour. The hours holding an exactly zero half-hour are dropped, as for
    the centred and the hour-ending powers, so one row set carries all three conventions.

    Args:
        frame: The wind rows, carrying `site`, `time` and the centred and hour-ending power.
        sites: The wind roster.

    Returns:
        `frame`, in its own row order, restricted to the rows with an hour-starting power, with
        `power_hour_starting_mw`.
    """
    starting = (
        wind_hourly_power(sites=sites, centred=False)
        .filter(~pl.col("has_zero_half_hour"))
        .select(
            "site",
            time=pl.col("time").dt.offset_by("-1h"),
            power_hour_starting_mw="power_mw",
        )
    )
    return frame.join(starting, on=["site", "time"], how="inner", maintain_order="left")


def dropped_months_text(*, dropped_months: dict[str, float]) -> str:
    """Describe the months the build dropped, and why, for the README.

    Args:
        dropped_months: Each dropped month with the share of its hours lost.

    Returns:
        Markdown that names each month and the planned rule that dropped it.
    """
    listed = ", ".join(f"{month} ({share:.0%})" for month, share in sorted(dropped_months.items()))
    return (
        "\n## Months dropped from every arm and every set\n\n"
        f"The plan stops the build if a month loses more than {MAX_MONTH_LOSS_SHARE:.0%} of its "
        "hours to incomplete, missing or unlisted UKV-CEDA runs. The coverage check found "
        f"{len(dropped_months)} such months, and the same-rows rule drops each from every arm of "
        f"both sets, never filling it from an older run: {listed or 'none'}. This is an "
        "availability rule decided before any result, not a choice made after seeing one. Two "
        "more months are dropped for a physics change inside them (2019-12 and 2026-01).\n"
    )


def write_outputs(
    *,
    output_dir: Path,
    ukv: UkvStores,
    station_rows: pl.DataFrame,
    wind: pl.DataFrame,
    wind_keep_zero_hours: pl.DataFrame,
    solar: pl.DataFrame,
    dropped_months: dict[str, float],
) -> None:
    """Write the four row sets, the stamp and the README into a new write-once folder.

    Args:
        output_dir: The folder to write.
        ukv: The opened stores, for the status hashes and snapshots.
        station_rows: Set A.
        wind: The wind rows.
        wind_keep_zero_hours: The wind rows that keep the hours holding an exactly zero half-hour.
        solar: The solar rows.
        dropped_months: Each month dropped from every arm and set, with the share of its hours
            lost to incomplete UKV-CEDA runs.

    Raises:
        FileExistsError: If an output already exists, before anything is written.
    """
    paths = {
        name: output_dir / name
        for name in (
            STATION_HOURS_NAME,
            WIND_ROWS_NAME,
            WIND_KEEP_ZERO_ROWS_NAME,
            SOLAR_ROWS_NAME,
            STAMP_NAME,
            README_NAME,
        )
    }
    refuse_to_overwrite(paths=paths.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    station_rows.write_parquet(paths[STATION_HOURS_NAME])
    wind.write_parquet(paths[WIND_ROWS_NAME])
    wind_keep_zero_hours.write_parquet(paths[WIND_KEEP_ZERO_ROWS_NAME])
    solar.write_parquet(paths[SOLAR_ROWS_NAME])
    stamp = BuildStamp(
        status_hashes=[
            hashlib.sha256(store.statuses.tobytes()).hexdigest() for store in ukv.stores
        ],
        snapshot_ids=list(ukv.snapshot_ids),
        input_hashes={path.name: _file_hash(path=path) for path in INPUT_FILES},
        delta_versions={
            "power_time_series": DeltaTable(POWER_DELTA_URI).version(),
            "effective_capacity": DeltaTable(CAPACITY_DELTA_URI).version(),
        },
        rows={
            "station_hours": station_rows.height,
            "wind": wind.height,
            "wind_keep_zero_hours": wind_keep_zero_hours.height,
            "solar": solar.height,
        },
        dropped_months=dropped_months,
    )
    paths[STAMP_NAME].write_text(stamp.to_json())
    paths[README_NAME].write_text(README_TEXT + dropped_months_text(dropped_months=dropped_months))


def month_hours(*, month: str) -> pl.Series:
    """Return every hour of one `%Y-%m` month, as a UTC datetime series.

    Args:
        month: The month, such as `2024-06`.

    Returns:
        The month's hours.
    """
    first = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    following = first.replace(year=first.year + (first.month == 12), month=first.month % 12 + 1)
    return pl.datetime_range(
        first, following, interval="1h", time_zone="UTC", eager=True, closed="left"
    )


def dry_run(*, month: str) -> list[str]:
    """Build set A for one month with real UKV-CEDA values, and write nothing.

    Args:
        month: The month to build.

    Returns:
        Markdown lines about the rows built.
    """
    ukv = open_ukv_stores()
    readings = station_readings()
    stations = choose_stations(ukv=ukv, readings=readings)
    rows = station_hours(
        ukv=ukv,
        stations=stations,
        readings=readings,
        wind=era5_wind_by_cell(),
        temperature=era5_temperature_by_cell(),
        hours=month_hours(month=month),
        read_values=True,
    )
    both = rows.drop_nulls(["station_wind_m_s", "ukv_wind_m_s", "station_temp_c", "ukv_temp_c"])
    return [
        (
            f"--dry-run {month}: {rows.height:,} station-hours built, {both.height:,} with a "
            "station reading and UKV-CEDA wind and temperature."
        ),
        f"- Nulls per column: {dict(zip(rows.columns, rows.null_count().row(0), strict=True))}.",
        (
            f"- Station wind minus UKV-CEDA wind, mean: "
            f"{(both['ukv_wind_m_s'] - both['station_wind_m_s']).mean():+.2f} m/s; temperature: "
            f"{(both['ukv_temp_c'] - both['station_temp_c']).mean():+.2f} C."
        ),
        f"- UKV-CEDA leads present: {sorted(rows['lead_hours'].unique().to_list())}.",
    ]


def main() -> int:
    """Run the coverage check, a one-month dry run, or the full build."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Coverage check; write nothing.")
    mode.add_argument("--dry-run", action="store_true", help="Build one month; write nothing.")
    mode.add_argument(
        "--matched-10m",
        action="store_true",
        help="Add the matched-height wind rows to an existing build; write one new file.",
    )
    mode.add_argument(
        "--hour-starting",
        action="store_true",
        help="Add the hour-starting power to an existing build; write one new file.",
    )
    parser.add_argument("--dry-run-month", default="2024-06")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    if arguments.check_only:
        lines, failures = coverage_check()
        sys.stdout.write("\n".join(lines) + "\n")
        if failures:
            sys.stdout.write("\nGUARDS FAILED:\n" + "\n".join(f"- {f}" for f in failures) + "\n")
            return 1
        sys.stdout.write("\nAll guards passed.\n")
        return 0
    if arguments.dry_run:
        sys.stdout.write("\n".join(dry_run(month=arguments.dry_run_month)) + "\n")
        return 0
    if arguments.matched_10m:
        path = arguments.output_dir / WIND_MATCHED_ROWS_NAME
        refuse_to_overwrite(paths=[path])
        rows = pl.read_parquet(arguments.output_dir / WIND_ROWS_NAME)
        matched = with_era5_10m_direction(frame=rows, wind=era5_wind_by_cell())
        check_no_missing(
            frame=matched,
            columns=[c for p in PRODUCTS for c in matched_wind_arm_columns(product=p)],
        )
        matched.write_parquet(path)
        sys.stdout.write(f"Wrote {matched.height:,} matched wind rows to {path}.\n")
        return 0

    if arguments.hour_starting:
        path = arguments.output_dir / WIND_HOUR_STARTING_ROWS_NAME
        refuse_to_overwrite(paths=[path])
        rows = pl.read_parquet(arguments.output_dir / WIND_ROWS_NAME)
        starting = with_hour_starting_power(frame=rows, sites=wind_sites())
        check_no_missing(frame=starting, columns=["power_mw", "power_hour_starting_mw"])
        starting.write_parquet(path)
        sys.stdout.write(
            f"Wrote {starting.height:,} of {rows.height:,} wind rows with an hour-starting power "
            f"to {path}.\n"
        )
        return 0

    ukv = open_ukv_stores()
    _, lossy_months = hours_lost_lines(ukv=ukv)
    drop_months = frozenset(lossy_months)
    readings = station_readings()
    stations = choose_stations(ukv=ukv, readings=readings)
    temperature = era5_temperature_by_cell()
    wind = era5_wind_by_cell()
    station_rows = station_hours(
        ukv=ukv,
        stations=stations,
        readings=readings,
        wind=wind,
        temperature=temperature,
        hours=pl.datetime_range(
            WINDOW_START,
            STATION_WINDOW_END,
            interval="1h",
            time_zone="UTC",
            eager=True,
            closed="left",
        ),
        read_values=True,
        drop_months=drop_months,
    )
    wind_frame = wind_rows(
        ukv=ukv, wind=wind, sites=wind_sites(), read_values=True, drop_months=drop_months
    )
    keep_zero_frame = wind_rows(
        ukv=ukv,
        wind=wind,
        sites=wind_sites(),
        read_values=True,
        drop_zero_hours=False,
        drop_months=drop_months,
    )
    solar_frame = solar_rows(
        ukv=ukv,
        temperature=temperature,
        sites=pv_sites(),
        read_values=True,
        drop_months=drop_months,
    )
    check_rows(wind=wind_frame, solar=solar_frame, wind_keep_zero_hours=keep_zero_frame)
    write_outputs(
        output_dir=arguments.output_dir,
        ukv=ukv,
        station_rows=station_rows,
        wind=wind_frame,
        wind_keep_zero_hours=keep_zero_frame,
        solar=solar_frame,
        dropped_months=lossy_months,
    )
    sys.stdout.write(
        f"Wrote {station_rows.height:,} station-hours, {wind_frame.height:,} wind rows and "
        f"{solar_frame.height:,} solar rows to {arguments.output_dir}.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
