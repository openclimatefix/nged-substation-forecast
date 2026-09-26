"""Archive the Met Office UKV forecasts held at CEDA into a local Icechunk store, cropped.

One-off throwaway script for the forecast study. CEDA archives every Met Office UKV run as whole
GRIB files on the 2 km Ordnance Survey national grid (548 by 704 cells): no server-side crop and
no `.idx` files exist, so each file is downloaded whole, the trial-area box is cropped locally, and
the raw file is deleted at once. The archive keeps four runs a day (00, 06, 12, and 18 UTC), from
`SLOT_EPOCH` (2019-09-01), and each run needs seven files: `Wholesale1`, `Wholesale2`,
`Wholesale3`, and `Wholesale4` (leads 0 to 36 hours) plus `Wholesale1T54`, `Wholesale2T54`, and
`Wholesale3T54` (leads 37 to 54 hours). CEDA holds `T120` files (leads 55 to 120 hours) for the 03
and 15 UTC runs only, so those leads never exist for the runs archived here.

**The store layout follows the ensemble archive of the Data archivist** (`nwp_archivist.store`): one
Icechunk repository under `<product dir>/store`, one Zarr array per variable with dimensions
`(init_time, step, cell)` and chunks `(1, n_steps, n_cells)`, and bookkeeping arrays along
`init_time` (`status`, `files_expected`, `files_received`, `archived_at`, `code_version`) written in
the same atomic Icechunk commit as the run's data. `init_time` is a fixed 6-hourly slot grid from
`SLOT_EPOCH`, so a missing run is a NaN slot with status 3 and never a shifted row. `step` is
hourly from 0 to `MAX_STEP_HOURS` hours, and a variable is NaN at every lead the source does not
serve. The cropped cells are the bounding rectangle, on the 2D grid, of every cell inside the box,
flattened row-major to one `cell` axis. The rectangle's `(rows, columns)` shape is an attribute of
the root group, and `cell_latitude`, `cell_longitude`, `cell_row`, and `cell_column` describe each
cell. **These four arrays and `_grid_cells.parquet` reveal the private trial-area box, so they stay
in the private store, and no log line, README, or lineage note carries a coordinate or a cell
count.**

**Streaming and resumption.** One raw file is on disk at a time, in `<product dir>/_scratch/` on the
data disk (never `/tmp`, which is tmpfs). Each file is cropped to `.npy` files in a per-run cache
directory as soon as it lands, the run is committed once every file is cached, and the cache is
deleted after the commit. A re-run skips every file already cached and every run whose status is
complete or missing. Pass `--retry-partial` to fetch partial and missing runs again. A lock file
stops two writers. The download is one stream with a delay between files, retries with backoff, and
an HTTP Range resume of an interrupted file, and it stops below `--min-free-gb` of free disk. A day
whose directory listing fails or is empty is skipped and never recorded, so a later run retries it.

Set `CEDA_TOKEN` in the environment (a CEDA access token). The script never prints or stores it, and
never follows a redirect: a redirect means the token was rejected. Run it with `uv run --with
icechunk --with zarr --with eccodes python studies/weather_downloads/fetch_ukv_ceda.py --start
2026-09-20 --end 2026-09-20 --store-dir <scratch dir>` for a one-day trial. Then check the store
with `validate_ukv_ceda.py`.
"""

import argparse
import fcntl
import json
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import icechunk
import numpy as np
import polars as pl
import requests
import zarr
import zarr.errors
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box
from pyproj import Transformer
from zarr.codecs import Crc32cCodec, ZstdCodec

PRODUCT_NAME: Final[str] = "UKV-CEDA"
CODE_VERSION: Final[str] = "fetch_ukv_ceda-1"
BASE_URL: Final[str] = "https://dap.ceda.ac.uk/badc/ukmo-nwp/data/ukv-grib"
CATALOGUE_URL: Final[str] = "https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f"

SLOT_EPOCH: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""The first slot of the `init_time` axis. A run's slot is its offset from here in whole cycles."""
CYCLE_HOURS: Final[int] = 6
RUN_HOURS: Final[tuple[int, ...]] = (0, 6, 12, 18)
"""The runs archived. CEDA holds eight runs a day, and `T120` files only for 03 and 15 UTC."""
PUBLICATION_LAG: Final[timedelta] = timedelta(days=4)
"""A run newer than this is not yet published, so it is neither fetched nor recorded."""

MAX_STEP_HOURS: Final[int] = 54
N_STEPS: Final[int] = MAX_STEP_HOURS + 1
PLAIN_LAST_STEP: Final[int] = 36
HOURLY_LAST_STEP: Final[int] = 48
T54_STEPS: Final[tuple[int, ...]] = (*range(37, HOURLY_LAST_STEP + 1), 51, 54)
"""Leads in a `T54` file: hourly to 48 hours, then 3-hourly."""

SIGNIFICAND_BITS: Final[int] = 13
"""The significand bits kept when rounding, the same as `delta_store.nwp.NWP_SIGNIFICAND_BITS`."""
INIT_TIMES_PER_MANIFEST: Final[int] = 64
_INIT_TIME_CHUNK: Final[int] = 1024
_MAIN_BRANCH: Final[str] = "main"

STATUS_COMPLETE: Final[int] = 1
STATUS_PARTIAL: Final[int] = 2
STATUS_MISSING: Final[int] = 3
STATUS_NAMES: Final[dict[int, str]] = {
    STATUS_COMPLETE: "complete",
    STATUS_PARTIAL: "partial",
    STATUS_MISSING: "missing",
}

GRID_NI: Final[int] = 548
GRID_NJ: Final[int] = 704
GRID_X1_M: Final[float] = -238000.0
GRID_Y1_M: Final[float] = 1222000.0
GRID_SPACING_M: Final[float] = 2000.0
GRID_KEYS: Final[dict[str, float]] = {
    "Ni": GRID_NI,
    "Nj": GRID_NJ,
    "X1InGridLengths": GRID_X1_M,
    "Y1InGridLengths": GRID_Y1_M,
    "DiInMetres": GRID_SPACING_M,
    "DjInMetres": GRID_SPACING_M,
    "XRInMetres": 400000.0,
    "YRInMetres": -100000.0,
    "scanningMode": 64,
}
"""The grid keys checked in every file. Rows run north to south, though `scanningMode` says south
to north: the grid is checked against the temperature and shortwave gradients, which put the first
row in the north. The keys for the latitude and longitude of the true origin are left out on
purpose, because CEDA documents them as wrongly encoded before 2020-01-15."""

MISSING_VALUE: Final[float] = 1e30
"""Passed to eccodes as `missingValue` so that a bitmap-masked cell decodes as this, not as 9999."""

REQUEST_DELAY_S: Final[float] = 1.0
MAX_ATTEMPTS: Final[int] = 8
MAX_BACKOFF_S: Final[float] = 150.0
"""Attempt `n` waits `5 * 2**(n-1)` seconds up to this cap, so 8 attempts span about 8 minutes."""
DOWNLOAD_CHUNK_BYTES: Final[int] = 4 * 1024 * 1024
DEFAULT_MIN_FREE_GB: Final[float] = 100.0

_LISTING_ROW: Final[re.Pattern[str]] = re.compile(
    r'<a href="([^"]+\.grib)">[^<]*</a>\s+\S+\s+\S+\s+(\d+)'
)
_FILE_NAME_TEMPLATE: Final[str] = "{init:%Y%m%d%H}00_u1096_ng_umqv_{tag}.grib"


@dataclass(frozen=True)
class FieldSpec:
    """One GRIB field kept in the store, identified by its GRIB2 parameter and level keys.

    Attributes:
        variable: The Zarr array name.
        wholesale: The `Wholesale` group (1 to 4) whose files carry the field.
        discipline: GRIB2 discipline.
        category: GRIB2 parameter category.
        number: GRIB2 parameter number.
        first_surface_type: GRIB2 `typeOfFirstFixedSurface`.
        first_surface_value: GRIB2 `scaledValueOfFirstFixedSurface` (a height in metres, or a
            pressure in pascals for `first_surface_type` 100).
        second_surface_type: GRIB2 `typeOfSecondFixedSurface`, 255 when absent.
        second_surface_value: GRIB2 `scaledValueOfSecondFixedSurface`, `NO_SURFACE` when absent.
        template: GRIB2 `productDefinitionTemplateNumber`: 0 is an instantaneous value, 8 a value
            over the interval since the previous step, and 5 a probability.
        units: The units as served.
        description: A one-line description for the README.
        invalid_below: Values below this are a "no value" flag and are stored as NaN, or `None`.
        process: GRIB2 `typeOfStatisticalProcessing` that an interval-valued field must carry
            (1 accumulation, 2 maximum), or `None` for an instantaneous field.
        lower_limit: The probability threshold, as `scaledValueOfLowerLimit`, that a probability
            field must carry, or `None`.
    """

    variable: str
    wholesale: int
    discipline: int
    category: int
    number: int
    first_surface_type: int
    first_surface_value: int
    second_surface_type: int
    second_surface_value: int
    template: int
    units: str
    description: str
    invalid_below: float | None = None
    process: int | None = None
    lower_limit: int | None = None

    @property
    def key(self) -> tuple[int, ...]:
        """The tuple of GRIB keys that identifies this field within a file."""
        return (
            self.discipline,
            self.category,
            self.number,
            self.first_surface_type,
            self.first_surface_value,
            self.second_surface_type,
            self.second_surface_value,
            self.template,
        )

    @property
    def interval_valued(self) -> bool:
        """Whether the field is over the interval since the previous step, so it has no lead 0."""
        return self.template == 8

    @property
    def tags(self) -> tuple[str, ...]:
        """The file tags that hold the field: the plain file, and the `T54` file if one exists."""
        plain = f"Wholesale{self.wholesale}"
        return (plain,) if self.wholesale == 4 else (plain, f"{plain}T54")

    def expected_steps(self, *, tag: str) -> tuple[int, ...]:
        """The leads, in hours, that the file `tag` serves for this field."""
        if tag.endswith("T54"):
            return T54_STEPS
        first = 1 if self.interval_valued else 0
        return tuple(range(first, PLAIN_LAST_STEP + 1))


NO_SURFACE: Final[int] = 2147483647


def _field(
    variable: str,
    wholesale: int,
    parameter: tuple[int, int, int],
    surface: tuple[int, int],
    units: str,
    description: str,
    *,
    second: tuple[int, int] = (255, NO_SURFACE),
    template: int = 0,
    invalid_below: float | None = None,
    process: int | None = None,
    lower_limit: int | None = None,
) -> FieldSpec:
    """Build a `FieldSpec` from grouped keys, so that the table below stays one line per field."""
    return FieldSpec(
        variable=variable,
        wholesale=wholesale,
        discipline=parameter[0],
        category=parameter[1],
        number=parameter[2],
        first_surface_type=surface[0],
        first_surface_value=surface[1],
        second_surface_type=second[0],
        second_surface_value=second[1],
        template=template,
        units=units,
        description=description,
        invalid_below=invalid_below,
        process=process,
        lower_limit=lower_limit,
    )


_HEIGHT: Final[int] = 103
_GROUND: Final[int] = 1
_PRESSURE: Final[int] = 100

FIELDS: Final[tuple[FieldSpec, ...]] = (
    _field("temperature_1p5m", 1, (0, 0, 0), (_HEIGHT, 1), "K", "Screen-level temperature"),
    _field("temperature_0m", 1, (0, 0, 0), (_HEIGHT, 0), "K", "Temperature at height 0 m"),
    _field("dew_point_1p5m", 1, (0, 0, 6), (_HEIGHT, 1), "K", "Screen-level dew point"),
    _field("relative_humidity_1p5m", 1, (0, 1, 1), (_HEIGHT, 1), "%", "Screen-level humidity"),
    _field("visibility_1p5m", 1, (0, 19, 0), (_HEIGHT, 1), "m", "Screen-level visibility"),
    _field(
        "visibility_below_1km_probability",
        1,
        (0, 19, 0),
        (_HEIGHT, 1),
        "fraction",
        "Probability that visibility is below 1000 m",
        template=5,
        lower_limit=1000,
    ),
    _field("precipitation_rate", 1, (0, 1, 7), (_GROUND, 0), "kg m-2 s-1", "Precipitation rate"),
    _field(
        "precipitation_amount",
        1,
        (0, 1, 8),
        (_GROUND, 0),
        "kg m-2",
        "Precipitation accumulated since the previous served step",
        template=8,
        process=1,
    ),
    _field(
        "param_0_1_230",
        1,
        (0, 1, 230),
        (_GROUND, 0),
        "unknown",
        "Unidentified precipitation-category parameter, zero in the early leads of a dry run",
    ),
    _field("wind_speed_10m", 1, (0, 2, 1), (_HEIGHT, 10), "m s-1", "10 m wind speed"),
    _field("wind_direction_10m", 1, (0, 2, 0), (_HEIGHT, 10), "degrees", "10 m wind direction"),
    _field("pressure_msl", 1, (0, 3, 1), (101, 0), "Pa", "Pressure reduced to mean sea level"),
    _field("cloud_total", 2, (0, 6, 1), (10, 0), "%", "Total cloud cover, whole atmosphere"),
    _field(
        "cloud_low",
        2,
        (0, 6, 3),
        (_HEIGHT, 0),
        "%",
        "Low cloud cover, 0 to 1524 m",
        second=(_HEIGHT, 1524),
    ),
    _field(
        "cloud_very_low",
        2,
        (0, 6, 3),
        (_HEIGHT, 0),
        "%",
        "Very low cloud cover, 0 to 305 m",
        second=(_HEIGHT, 305),
    ),
    _field(
        "cloud_medium",
        2,
        (0, 6, 4),
        (_HEIGHT, 1524),
        "%",
        "Medium cloud cover, 1524 to 4572 m",
        second=(_HEIGHT, 4572),
    ),
    _field(
        "cloud_high",
        2,
        (0, 6, 5),
        (_HEIGHT, 4572),
        "%",
        "High cloud cover, 4572 to 30000 m",
        second=(_HEIGHT, 30000),
    ),
    _field("cloud_base_height", 2, (0, 6, 11), (2, 0), "m", "Height of the lowest cloud base"),
    _field(
        "cloud_param_0_6_26",
        2,
        (0, 6, 26),
        (_GROUND, 0),
        "unknown",
        "Unidentified cloud-category parameter, 2 to 1800 m where valid",
    ),
    _field(
        "convective_cloud_top_height",
        2,
        (0, 6, 27),
        (_GROUND, 0),
        "m",
        "Height of convective cloud top, NaN where no convective cloud",
        invalid_below=-32000.0,
    ),
    _field("snow_depth", 2, (0, 1, 11), (_GROUND, 0), "m", "Snow depth"),
    _field(
        "shortwave_down",
        2,
        (0, 4, 7),
        (_GROUND, 0),
        "W m-2",
        "Downward shortwave flux at the surface, instantaneous",
    ),
    _field(
        "longwave_down",
        2,
        (0, 5, 3),
        (_GROUND, 0),
        "W m-2",
        "Downward longwave flux at the surface, instantaneous",
    ),
    *(
        _field(
            f"{name}_{level // 100}hpa",
            3,
            parameter,
            (_PRESSURE, level),
            units,
            f"{label} at {level // 100} hPa, NaN where the level is below the ground",
        )
        for level in (100000, 92500)
        for name, parameter, units, label in (
            ("wind_speed", (0, 2, 1), "m s-1", "Wind speed"),
            ("wind_direction", (0, 2, 0), "degrees", "Wind direction"),
            ("geopotential_height", (0, 3, 5), "gpm", "Geopotential height"),
        )
    ),
    _field("gust_10m", 4, (0, 2, 22), (_HEIGHT, 10), "m s-1", "10 m wind gust, instantaneous"),
    _field(
        "gust_10m_max",
        4,
        (0, 2, 22),
        (_HEIGHT, 10),
        "m s-1",
        "Maximum 10 m wind gust since the previous step",
        template=8,
        process=2,
    ),
)

FILE_TAGS: Final[tuple[str, ...]] = tuple(
    sorted(
        {tag for spec in FIELDS for tag in spec.tags}, key=lambda tag: (tag.endswith("T54"), tag)
    )
)
"""The seven files a run needs: four plain files, then three `T54` files."""

_SPECS_BY_TAG: Final[dict[str, tuple[FieldSpec, ...]]] = {
    tag: tuple(spec for spec in FIELDS if tag in spec.tags) for tag in FILE_TAGS
}


class CedaAuthError(RuntimeError):
    """CEDA answered with a redirect or an authorisation failure, so the token was not accepted."""


class FileAbsentError(RuntimeError):
    """CEDA does not hold the requested file."""


@dataclass(frozen=True)
class CellGrid:
    """The cropped cells: the 2D rectangle that contains every cell inside the private box.

    Attributes:
        row_start: First row of the rectangle on the full 548 by 704 grid, counting from the north.
        col_start: First column of the rectangle.
        n_rows: Rows in the rectangle.
        n_cols: Columns in the rectangle.
        latitude: Latitude of each cell in degrees, flattened row-major, length `n_rows * n_cols`.
        longitude: Longitude of each cell in degrees east, in the same order.
    """

    row_start: int
    col_start: int
    n_rows: int
    n_cols: int
    latitude: np.ndarray
    longitude: np.ndarray

    @property
    def n_cells(self) -> int:
        """The number of cells on the flattened `cell` axis."""
        return self.n_rows * self.n_cols

    def crop(self, full_grid_values: np.ndarray) -> np.ndarray:
        """Crop a flat array of every full-grid cell to the rectangle, flattened row-major."""
        grid = full_grid_values.reshape(GRID_NJ, GRID_NI)
        rows = slice(self.row_start, self.row_start + self.n_rows)
        cols = slice(self.col_start, self.col_start + self.n_cols)
        return grid[rows, cols].reshape(-1)


def full_grid_latitude_longitude() -> tuple[np.ndarray, np.ndarray]:
    """Compute the latitude and longitude of every cell on the full grid, as `(nj, ni)` arrays.

    The grid is the Ordnance Survey national grid (`EPSG:27700`) at 2 km spacing. Its first row is
    the northernmost.
    """
    easting = GRID_X1_M + GRID_SPACING_M * np.arange(GRID_NI)
    northing = GRID_Y1_M - GRID_SPACING_M * np.arange(GRID_NJ)
    eastings, northings = np.meshgrid(easting, northing)
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(eastings, northings)
    return np.asarray(latitude), np.asarray(longitude)


def build_cell_grid() -> CellGrid:
    """Find the rectangle of the full grid that contains every cell inside the private box."""
    box = load_trial_area_box()
    latitude, longitude = full_grid_latitude_longitude()
    inside = (
        (latitude >= box.lat_min)
        & (latitude <= box.lat_max)
        & (longitude >= box.lon_min)
        & (longitude <= box.lon_max)
    )
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        message = "no grid cell lies inside the trial-area box"
        raise RuntimeError(message)
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    col_slice = slice(int(cols[0]), int(cols[-1]) + 1)
    return CellGrid(
        row_start=row_slice.start,
        col_start=col_slice.start,
        n_rows=row_slice.stop - row_slice.start,
        n_cols=col_slice.stop - col_slice.start,
        latitude=latitude[row_slice, col_slice].reshape(-1),
        longitude=longitude[row_slice, col_slice].reshape(-1),
    )


def round_significand(values: np.ndarray, *, keep_bits: int = SIGNIFICAND_BITS) -> np.ndarray:
    """Round `float32` values to `keep_bits` significand bits, to nearest.

    This is Veltkamp splitting evaluated in `float32` with numpy, the rule that
    `nwp_archivist.store.round_significand` uses and that `delta_store.precision` documents. It is
    bit-identical to `delta_store.precision.round_to_significand_bits`, including its pass-through
    of NaN, infinity, and values too large for the splitting to avoid overflow.

    Args:
        values: A `float32` array.
        keep_bits: The significand bits to keep, from 2 to 22 inclusive.

    Returns:
        A new `float32` array of the same shape.

    Raises:
        TypeError: If `values` is not `float32`.
        ValueError: If `keep_bits` is outside the range the splitting supports.
    """
    if values.dtype != np.float32:
        message = f"expected float32, got {values.dtype}"
        raise TypeError(message)
    shift = 24 - keep_bits
    if not 2 <= shift <= 22:
        message = f"keep_bits={keep_bits} is outside the range Veltkamp splitting supports"
        raise ValueError(message)
    splitter = np.float32(2**shift + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = values * splitter
        rounded = scaled - (scaled - values)
    return np.where(np.isfinite(scaled), rounded, values)


def slot_for(init_time: datetime) -> int:
    """The index of a run on the `init_time` axis."""
    offset = init_time - SLOT_EPOCH
    if offset % timedelta(hours=CYCLE_HOURS):
        message = f"{init_time.isoformat()} is not on the {CYCLE_HOURS}-hourly slot grid"
        raise ValueError(message)
    return offset // timedelta(hours=CYCLE_HOURS)


def _init_time_seconds(slots: range) -> np.ndarray:
    """The `init_time` coordinate in seconds since 1970, for a range of slots."""
    epoch_seconds = int(SLOT_EPOCH.timestamp())
    return np.array([epoch_seconds + slot * CYCLE_HOURS * 3600 for slot in slots], dtype=np.int64)


def _array(group: zarr.Group, name: str) -> zarr.Array:
    """Fetch an array that the layout guarantees exists."""
    member = group[name]
    if not isinstance(member, zarr.Array):
        message = f"{name} is not an array"
        raise TypeError(message)
    return member


def _write_static(
    group: zarr.Group, name: str, values: np.ndarray, *, dims: tuple[str, ...]
) -> None:
    """Write a small array whole, as one chunk."""
    array = group.create_array(
        name,
        shape=values.shape,
        chunks=values.shape,
        dtype=values.dtype,
        dimension_names=dims,
        compressors=(ZstdCodec(level=3), Crc32cCodec()),
    )
    array[:] = values


def _repository_config() -> icechunk.RepositoryConfig:
    """The repository configuration, which every open must repeat because it is not persisted."""
    config = icechunk.RepositoryConfig.default()
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict(
            {
                icechunk.ManifestSplitCondition.AnyArray(): {
                    icechunk.ManifestSplitDimCondition.DimensionName("init_time"): (
                        INIT_TIMES_PER_MANIFEST
                    )
                }
            }
        )
    )
    return config


@dataclass(frozen=True)
class RunResult:
    """Everything one Icechunk commit records about a run.

    Attributes:
        init_time: The run's initialisation time.
        status: `STATUS_COMPLETE`, `STATUS_PARTIAL`, or `STATUS_MISSING`.
        files_expected: How many files the run should contain.
        files_received: How many of them arrived and were cropped.
        blocks: For each variable found, an `(N_STEPS, n_cells)` `float32` array, NaN at the leads
            not served.
    """

    init_time: datetime
    status: int
    files_expected: int
    files_received: int
    blocks: dict[str, np.ndarray]


class UkvStore:
    """The Icechunk repository of the product."""

    def __init__(self, *, repository: icechunk.Repository) -> None:
        """Wrap an opened repository."""
        self.repository = repository

    @classmethod
    def open(cls, *, store_path: Path) -> UkvStore:
        """Open the repository at `store_path`, creating an empty one if there is none."""
        store_path.mkdir(parents=True, exist_ok=True)
        repository = icechunk.Repository.open_or_create(
            storage=icechunk.local_filesystem_storage(str(store_path)),
            config=_repository_config(),
        )
        return cls(repository=repository)

    def _read_group(self) -> zarr.Group | None:
        """Open the archive for reading, or return `None` while the repository is empty."""
        session = self.repository.readonly_session(branch=_MAIN_BRANCH)
        try:
            return zarr.open_group(session.store, mode="r")
        except zarr.errors.GroupNotFoundError:
            return None

    def statuses(self) -> np.ndarray:
        """The status of every slot: 0 for never archived, otherwise a `STATUS_*` code."""
        group = self._read_group()
        if group is None or "status" not in group:
            return np.zeros(0, dtype=np.int8)
        return np.asarray(_array(group, "status")[:], dtype=np.int8)

    def initialise(self, grid: CellGrid) -> None:
        """Create the layout from the cell grid, unless the repository already has one."""
        group = self._read_group()
        if group is not None and "status" in group:
            stored = _array(group, "cell_latitude")[:]
            if not np.array_equal(stored, grid.latitude.astype(np.float64)):
                message = "the store's cells differ from the trial-area box's cells"
                raise RuntimeError(message)
            return
        session = self.repository.writable_session(branch=_MAIN_BRANCH)
        root = zarr.open_group(session.store, mode="w")
        _write_layout(root, grid)
        session.commit("Create the UKV-CEDA archive layout", allow_empty=True)

    def commit_run(self, run: RunResult) -> str:
        """Write one run into its slot and commit it, in a single Icechunk commit.

        Args:
            run: The run's status, file counts, and cropped data.

        Returns:
            The identifier of the new snapshot.
        """
        slot = slot_for(run.init_time)
        session = self.repository.writable_session(branch=_MAIN_BRANCH)
        group = zarr.open_group(session.store, mode="r+")
        _grow(group, slot + 1)
        for variable, block in run.blocks.items():
            _array(group, variable)[slot] = round_significand(block)
        _array(group, "status")[slot] = run.status
        _array(group, "files_expected")[slot] = run.files_expected
        _array(group, "files_received")[slot] = run.files_received
        _array(group, "archived_at")[slot] = int(datetime.now(UTC).timestamp())
        _array(group, "code_version")[slot] = CODE_VERSION
        message = (
            f"{PRODUCT_NAME} {run.init_time:%Y-%m-%dT%H:%MZ} {STATUS_NAMES[run.status]} "
            f"{run.files_received}/{run.files_expected} files"
        )
        return session.commit(message, allow_empty=True)


_ALONG_INIT_TIME: Final[tuple[str, ...]] = (
    "init_time",
    "status",
    "files_expected",
    "files_received",
    "archived_at",
    "code_version",
)


def _grow(group: zarr.Group, n_slots: int) -> None:
    """Extend every array along `init_time` to at least `n_slots`, filling the coordinate."""
    current = _array(group, "status").shape[0]
    if n_slots <= current:
        return
    for name in (*_ALONG_INIT_TIME, *(spec.variable for spec in FIELDS)):
        array = _array(group, name)
        array.resize((n_slots, *array.shape[1:]))
    _array(group, "init_time")[current:n_slots] = _init_time_seconds(range(current, n_slots))


def _write_layout(group: zarr.Group, grid: CellGrid) -> None:
    """Create every array, empty along `init_time`, and store the grid. The caller commits."""
    group.attrs.update(
        {
            "product": PRODUCT_NAME,
            "provider": "Met Office, via the CEDA archive",
            "licence": "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)",
            "catalogue_record": CATALOGUE_URL,
            "cycle_hours": CYCLE_HOURS,
            "slot_epoch": SLOT_EPOCH.isoformat(),
            "rect_shape": [grid.n_rows, grid.n_cols],
            "full_grid_shape": [GRID_NJ, GRID_NI],
            "status_codes": {str(code): name for code, name in STATUS_NAMES.items()},
            "step_note": (
                "step is the lead in hours. A variable sits at the leads its files serve and is "
                "NaN at every other lead."
            ),
            "reshape_note": (
                "Reshape the cell axis to rect_shape (rows from the north, then columns from the "
                "west) to recover the 2D field."
            ),
            "variables": {spec.variable: spec.units for spec in FIELDS},
        }
    )
    time_attrs = {"units": "seconds since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"}

    def along_init_time(name: str, dtype: str, attrs: dict[str, Any] | None = None) -> None:
        group.create_array(
            name,
            shape=(0,),
            chunks=(_INIT_TIME_CHUNK,),
            dtype=dtype,
            fill_value=0 if dtype != "str" else None,
            dimension_names=("init_time",),
            attributes=attrs or {},
        )

    along_init_time("init_time", "int64", time_attrs)
    along_init_time("status", "int8", {"description": "0 means not archived; see status_codes"})
    along_init_time("files_expected", "int32")
    along_init_time("files_received", "int32")
    along_init_time("archived_at", "int64", time_attrs)
    along_init_time("code_version", "str")
    _write_static(group, "step", np.arange(N_STEPS, dtype=np.int32), dims=("step",))
    _array(group, "step").attrs["units"] = "hours"
    cell_rows = grid.row_start + np.repeat(np.arange(grid.n_rows), grid.n_cols)
    cell_cols = grid.col_start + np.tile(np.arange(grid.n_cols), grid.n_rows)
    _write_static(group, "cell_row", cell_rows.astype(np.int32), dims=("cell",))
    _write_static(group, "cell_column", cell_cols.astype(np.int32), dims=("cell",))
    _write_static(group, "cell_latitude", grid.latitude.astype(np.float64), dims=("cell",))
    _write_static(group, "cell_longitude", grid.longitude.astype(np.float64), dims=("cell",))
    for spec in FIELDS:
        group.create_array(
            spec.variable,
            shape=(0, N_STEPS, grid.n_cells),
            chunks=(1, N_STEPS, grid.n_cells),
            dtype="float32",
            fill_value=float("nan"),
            dimension_names=("init_time", "step", "cell"),
            compressors=(ZstdCodec(level=3), Crc32cCodec()),
            attributes={
                "units": spec.units,
                "description": spec.description,
                "grib_key": list(spec.key),
                "interval_valued": spec.interval_valued,
            },
        )


class TooManyRequestsError(requests.RequestException):
    """CEDA answered 429. `retry_after` is the seconds it asked for, or `None`."""

    def __init__(self, retry_after: float | None) -> None:
        """Record how long CEDA asked the client to wait."""
        super().__init__("HTTP 429")
        self.retry_after = retry_after


def _retry_after_seconds(response: requests.Response) -> float | None:
    """Read a `Retry-After` header given in seconds, or return `None`."""
    try:
        return float(response.headers["Retry-After"])
    except KeyError, ValueError:
        return None


def _retry[T](operation: Callable[[], T], *, what: str) -> T:
    """Run `operation`, retrying network failures with exponential backoff.

    Only the exception type and the attempt number are logged, so that a URL never reaches a log.
    `CedaAuthError` and `FileAbsentError` are not network failures and pass straight through. A 429
    waits at least as long as its `Retry-After` header asked.

    Args:
        operation: The call to make.
        what: A short label for the log line, such as a file tag.

    Returns:
        Whatever `operation` returns.

    Raises:
        requests.RequestException: If every attempt failed.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return operation()
        except requests.RequestException as error:
            print(f"{what}: attempt {attempt}/{MAX_ATTEMPTS} failed with {type(error).__name__}")
            if attempt == MAX_ATTEMPTS:
                raise
            delay = min(5 * 2 ** (attempt - 1), MAX_BACKOFF_S)
            if isinstance(error, TooManyRequestsError) and error.retry_after is not None:
                delay = max(delay, error.retry_after)
            time.sleep(delay)
    message = "unreachable"
    raise AssertionError(message)


def list_day(session: requests.Session, *, day: date) -> dict[str, int] | None:
    """List the GRIB files CEDA holds for a day, with their sizes in bytes.

    Args:
        session: An HTTP session. The listing is anonymous.
        day: The UTC calendar day.

    Returns:
        A mapping from file name to size, or `None` if CEDA has no directory for the day.
    """

    def fetch() -> dict[str, int] | None:
        response = session.get(f"{BASE_URL}/{day:%Y/%m/%d}/", timeout=(30, 60))
        if response.status_code == 404:
            return None
        if response.status_code == 429:
            raise TooManyRequestsError(_retry_after_seconds(response))
        response.raise_for_status()
        return {name: int(size) for name, size in _LISTING_ROW.findall(response.text)}

    return _retry(fetch, what=f"listing {day.isoformat()}")


def download(
    session: requests.Session, *, token: str, day: date, name: str, size: int, destination: Path
) -> None:
    """Download one file to `destination`, resuming an interrupted transfer with an HTTP Range.

    The bytes land in `<destination>.partial`. A retry, or a later run after a crash, asks CEDA for
    the rest of the file with a `Range` header, and starts again if CEDA does not answer 206. The
    file is renamed into place only when its size equals the listing's.

    Args:
        session: An HTTP session.
        token: The CEDA access token, sent as a bearer header and never logged.
        day: The UTC calendar day the file is filed under.
        name: The file name.
        size: The size in bytes that the directory listing gave.
        destination: Where to write the file.

    Raises:
        CedaAuthError: If CEDA redirected the request or refused the token.
        FileAbsentError: If CEDA answered 404.
        requests.ConnectionError: If the transfer ended with fewer bytes than `size`.
    """
    partial = destination.with_suffix(".partial")

    def fetch() -> None:
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            partial.unlink()
            have = 0
        if have < size:
            _transfer(
                session,
                token=token,
                url=f"{BASE_URL}/{day:%Y/%m/%d}/{name}",
                partial=partial,
                have=have,
            )
        if partial.stat().st_size != size:
            message = f"received {partial.stat().st_size} of {size} bytes"
            raise requests.ConnectionError(message)
        partial.rename(destination)

    _retry(fetch, what=name)


def _transfer(session: requests.Session, *, token: str, url: str, partial: Path, have: int) -> None:
    """Append the rest of a file, from byte `have`, to `partial`."""
    headers = {"Authorization": f"Bearer {token}"}
    if have:
        headers["Range"] = f"bytes={have}-"
    response = session.get(
        url, headers=headers, stream=True, allow_redirects=False, timeout=(30, 120)
    )
    with response:
        if response.status_code == 404:
            raise FileAbsentError(url.rsplit("/", 1)[-1])
        if response.is_redirect or response.status_code in (401, 403):
            message = f"CEDA answered {response.status_code}: the token was not accepted"
            raise CedaAuthError(message)
        if response.status_code == 429:
            raise TooManyRequestsError(_retry_after_seconds(response))
        response.raise_for_status()
        resumed = have > 0 and response.status_code == 206
        with partial.open("ab" if resumed else "wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)


def _scaled_key(handle: int, key: str) -> int:
    """Read a GRIB integer key, or return -1 where the message lacks it."""
    import eccodes  # ty: ignore[unresolved-import]

    try:
        return int(eccodes.codes_get_long(handle, key))
    except eccodes.KeyValueNotFoundError:
        return -1


def extract_file(path: Path, *, tag: str, grid: CellGrid) -> tuple[dict[str, np.ndarray], dict]:
    """Crop every kept field of one GRIB file.

    Args:
        path: The raw GRIB file.
        tag: The file's tag, such as `Wholesale1T54`.
        grid: The cells to keep.

    Returns:
        The arrays and a report. Each array is `(N_STEPS, n_cells)` `float32`, NaN at the leads the
        file does not carry. The report maps `found` to each variable's list of leads and
        `problems` to a list of one-line descriptions of what was expected but absent.

    Raises:
        RuntimeError: If the file's grid differs from `GRID_KEYS`.
    """
    import eccodes  # ty: ignore[unresolved-import]

    specs = {spec.key: spec for spec in _SPECS_BY_TAG[tag]}
    arrays: dict[str, np.ndarray] = {}
    found: dict[str, list[int]] = {}
    faults: list[str] = []
    with path.open("rb") as handle:
        while (message := eccodes.codes_grib_new_from_file(handle)) is not None:
            try:
                _check_grid(message)
                key = tuple(
                    _scaled_key(message, name)
                    for name in (
                        "discipline",
                        "parameterCategory",
                        "parameterNumber",
                        "typeOfFirstFixedSurface",
                        "scaledValueOfFirstFixedSurface",
                        "typeOfSecondFixedSurface",
                        "scaledValueOfSecondFixedSurface",
                        "productDefinitionTemplateNumber",
                    )
                )
                spec = specs.get(key)
                if spec is None:
                    continue
                step = _scaled_key(message, "endStep")
                if step > MAX_STEP_HOURS:
                    continue
                fault = _semantic_fault(message, spec, step=step)
                if fault is None and step in found.get(spec.variable, []):
                    fault = f"{spec.variable} has a second message at lead {step}"
                if fault is not None:
                    faults.append(f"{tag}: {fault}")
                    continue
                eccodes.codes_set(message, "missingValue", MISSING_VALUE)
                values = eccodes.codes_get_values(message).astype(np.float32)
                values[values >= MISSING_VALUE / 10] = np.nan
                if spec.invalid_below is not None:
                    values[values < spec.invalid_below] = np.nan
                block = arrays.setdefault(
                    spec.variable, np.full((N_STEPS, grid.n_cells), np.nan, dtype=np.float32)
                )
                block[step] = grid.crop(values)
                found.setdefault(spec.variable, []).append(step)
            finally:
                eccodes.codes_release(message)
    problems = _problems(found, tag=tag) + sorted(set(faults))
    return arrays, {
        "found": {var: sorted(steps) for var, steps in found.items()},
        "problems": problems,
    }


def _semantic_fault(message: int, spec: FieldSpec, *, step: int) -> str | None:
    """Describe how a message's statistical process, interval, or threshold differs from `spec`.

    An interval-valued field must carry its statistical process (accumulation or maximum) and cover
    the interval since the previous served lead: 1 hour up to lead 48 and 3 hours after.
    """
    if spec.process is not None:
        process = _scaled_key(message, "typeOfStatisticalProcessing")
        if process != spec.process:
            return f"{spec.variable} has statistical process {process}, expected {spec.process}"
        length = step - _scaled_key(message, "startStep")
        expected = 1 if step <= HOURLY_LAST_STEP else 3
        if length != expected:
            return f"{spec.variable} covers {length} h at lead {step}, expected {expected}"
    if spec.lower_limit is not None:
        limit = _scaled_key(message, "scaledValueOfLowerLimit")
        if limit != spec.lower_limit:
            return f"{spec.variable} has threshold {limit}, expected {spec.lower_limit}"
    return None


def _check_grid(message: int) -> None:
    """Raise if a message's grid keys differ from `GRID_KEYS`."""
    import eccodes  # ty: ignore[unresolved-import]

    for key, expected in GRID_KEYS.items():
        if float(eccodes.codes_get(message, key)) != expected:
            text = f"the file's grid differs from the archived grid in key {key}"
            raise RuntimeError(text)


def _problems(found: dict[str, list[int]], *, tag: str) -> list[str]:
    """Describe every kept field of the file `tag` that is absent, or lacks a lead."""
    problems: list[str] = []
    for spec in _SPECS_BY_TAG[tag]:
        expected = set(spec.expected_steps(tag=tag))
        seen = set(found.get(spec.variable, []))
        if not seen:
            problems.append(f"{tag}: {spec.variable} absent")
        elif seen != expected:
            problems.append(f"{tag}: {spec.variable} has {len(expected - seen)} leads missing")
    return problems


def _cache_paths(run_dir: Path, tag: str) -> tuple[Path, Callable[[str], Path]]:
    """The cache's done-marker path for a file, and a function giving each variable's array path."""
    return run_dir / f"{tag}.json", lambda variable: run_dir / f"{tag}__{variable}.npy"


def cache_file(run_dir: Path, *, tag: str, arrays: dict[str, np.ndarray], report: dict) -> None:
    """Write a file's cropped arrays and then its done-marker, so an interrupted write is redone."""
    run_dir.mkdir(parents=True, exist_ok=True)
    marker, array_path = _cache_paths(run_dir, tag)
    for variable, block in arrays.items():
        np.save(array_path(variable), block)
    partial = marker.with_suffix(".partial")
    partial.write_text(json.dumps(report))
    partial.rename(marker)


def merge_run(run_dir: Path, *, init_time: datetime, files_expected: int) -> RunResult:
    """Combine a run's cached files into one `RunResult`.

    Args:
        run_dir: The run's cache directory.
        init_time: The run's initialisation time.
        files_expected: How many files the run should have.

    Returns:
        The run, `STATUS_COMPLETE` when every file was cached and none reported a problem, and
        `STATUS_PARTIAL` otherwise.
    """
    blocks: dict[str, np.ndarray] = {}
    received = 0
    problems: list[str] = []
    for tag in FILE_TAGS:
        marker, array_path = _cache_paths(run_dir, tag)
        if not marker.exists():
            problems.append(f"{tag}: file not received")
            continue
        received += 1
        report = json.loads(marker.read_text())
        problems.extend(report["problems"])
        for variable, steps in report["found"].items():
            cached = np.load(array_path(variable))
            block = blocks.setdefault(variable, np.full_like(cached, np.nan))
            block[steps] = cached[steps]
    status = STATUS_COMPLETE if received == files_expected and not problems else STATUS_PARTIAL
    if problems:
        _log_ledger(run_dir.parent.parent, init_time=init_time, problems=problems)
    return RunResult(
        init_time=init_time,
        status=status,
        files_expected=files_expected,
        files_received=received,
        blocks=blocks,
    )


def _log_ledger(product_dir: Path, *, init_time: datetime, problems: list[str]) -> None:
    """Append one ledger line per problem. A line names a run and a file, never a coordinate."""
    with (product_dir / "ledger.txt").open("a") as handle:
        for problem in problems:
            handle.write(f"{init_time:%Y-%m-%dT%HZ} partial: {problem}\n")


def _free_gb(path: Path) -> float:
    """The free space, in GB, of the filesystem holding `path`."""
    return shutil.disk_usage(path).free / 1e9


def _directory_bytes(path: Path) -> int:
    """The total size of every file under `path`."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


@dataclass
class RunTimings:
    """Counters for one run, printed in its summary line.

    Attributes:
        raw_bytes: Bytes downloaded.
        download_seconds: Seconds spent downloading.
    """

    raw_bytes: int = 0
    download_seconds: float = 0.0


def fetch_run(
    http: requests.Session,
    *,
    token: str,
    init_time: datetime,
    listing: dict[str, int],
    grid: CellGrid,
    product_dir: Path,
    min_free_gb: float,
    timings: RunTimings,
) -> RunResult:
    """Download and crop every file of one run, then return the merged run.

    Args:
        http: An HTTP session.
        token: The CEDA access token.
        init_time: The run's initialisation time.
        listing: The day's file listing, which holds at least one `.grib` file.
        grid: The cells to keep.
        product_dir: The product directory, which holds `_scratch/`.
        min_free_gb: Stop when less than this much disk is free.
        timings: Counters this call adds to.

    Returns:
        The merged run. Its status is `STATUS_MISSING` if CEDA holds none of the run's files.

    Raises:
        SystemExit: If free disk falls below `min_free_gb`.
    """
    run_dir = product_dir / "_scratch" / f"{init_time:%Y%m%dT%H}"
    names = {tag: _FILE_NAME_TEMPLATE.format(init=init_time, tag=tag) for tag in FILE_TAGS}
    if not any(name in listing for name in names.values()):
        return RunResult(init_time, STATUS_MISSING, len(FILE_TAGS), 0, {})
    for tag, name in names.items():
        if _cache_paths(run_dir, tag)[0].exists() or name not in listing:
            continue
        if _free_gb(product_dir) < min_free_gb:
            print(f"stopping: less than {min_free_gb} GB free")
            raise SystemExit(1)
        raw = product_dir / "_scratch" / name
        started = time.monotonic()
        try:
            download(
                http,
                token=token,
                day=init_time.date(),
                name=name,
                size=listing[name],
                destination=raw,
            )
        except FileAbsentError:
            print(f"{init_time:%Y-%m-%dT%HZ} {tag}: listed but CEDA answered 404")
            continue
        timings.download_seconds += time.monotonic() - started
        timings.raw_bytes += listing[name]
        try:
            arrays, report = extract_file(raw, tag=tag, grid=grid)
        finally:
            raw.unlink()
        cache_file(run_dir, tag=tag, arrays=arrays, report=report)
        time.sleep(REQUEST_DELAY_S)
    return merge_run(run_dir, init_time=init_time, files_expected=len(FILE_TAGS))


def run_times(*, start: date, end: date) -> Iterator[datetime]:
    """Yield every archived run from the start of `start` to the end of `end`, oldest first."""
    day = start
    while day <= end:
        for hour in RUN_HOURS:
            yield datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
        day += timedelta(days=1)


def archive(args: argparse.Namespace) -> int:
    """Archive every requested run that the store does not hold yet.

    Args:
        args: The parsed command line.

    Returns:
        The process exit code.
    """
    token = _token()
    product_dir: Path = args.store_dir
    scratch = product_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with (product_dir / ".lock").open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another writer holds the lock on this store")
            return 1
        return _archive_locked(args, token=token, product_dir=product_dir)


def _token() -> str:
    """Read the CEDA token from the environment, exiting with a message if it is unset."""
    import os

    token = os.environ.get("CEDA_TOKEN")
    if not token:
        print("CEDA_TOKEN is not set")
        raise SystemExit(1)
    return token


def _archive_locked(args: argparse.Namespace, *, token: str, product_dir: Path) -> int:
    """Run the archive loop while holding the store lock."""
    grid = build_cell_grid()
    store = UkvStore.open(store_path=product_dir / "store")
    store.initialise(grid)
    http = requests.Session()
    newest = datetime.now(UTC) - PUBLICATION_LAG
    end = args.end or newest.date()
    listings: dict[date, dict[str, int] | None] = {}
    done = 0
    for init_time in run_times(start=args.start, end=end):
        if init_time > newest:
            continue
        status = _status_at(store.statuses(), init_time)
        skip = (
            (STATUS_COMPLETE,)
            if args.retry_partial
            else (STATUS_COMPLETE, STATUS_MISSING, STATUS_PARTIAL)
        )
        if status in skip:
            continue
        if args.max_runs is not None and done >= args.max_runs:
            break
        day = init_time.date()
        if day not in listings:
            listings.clear()
            listings[day] = _usable_listing(http, day=day)
        listing = listings[day]
        if listing is None:
            print(f"{day.isoformat()}: no usable directory listing, skipping the day")
            continue
        started = time.monotonic()
        timings = RunTimings()
        run = fetch_run(
            http,
            token=token,
            init_time=init_time,
            listing=listing,
            grid=grid,
            product_dir=product_dir,
            min_free_gb=args.min_free_gb,
            timings=timings,
        )
        before = _directory_bytes(product_dir / "store") if args.measure else 0
        store.commit_run(run)
        shutil.rmtree(product_dir / "_scratch" / f"{init_time:%Y%m%dT%H}", ignore_errors=True)
        done += 1
        _print_summary(
            run,
            started=started,
            timings=timings,
            store_bytes=(
                _directory_bytes(product_dir / "store") - before if args.measure else None
            ),
        )
    write_documents(product_dir=product_dir, store=store, grid=grid)
    return 0


def _usable_listing(session: requests.Session, *, day: date) -> dict[str, int] | None:
    """List a day, or return `None` if the listing failed or holds no `.grib` file.

    A listing that fails, is absent, or parses to nothing is a transient fault, not evidence that
    CEDA lacks the day's runs, so the caller records nothing and a later run tries the day again.
    """
    try:
        listing = list_day(session, day=day)
    except requests.RequestException:
        return None
    return listing or None


def _status_at(statuses: np.ndarray, init_time: datetime) -> int:
    """The stored status of a run's slot, 0 if the slot does not exist yet."""
    slot = slot_for(init_time)
    return int(statuses[slot]) if slot < len(statuses) else 0


def _print_summary(
    run: RunResult, *, started: float, timings: RunTimings, store_bytes: int | None
) -> None:
    """Print one line for a committed run: its status, sizes, and speed."""
    elapsed = time.monotonic() - started
    speed = timings.raw_bytes / 1e6 / timings.download_seconds if timings.download_seconds else 0.0
    stored = "" if store_bytes is None else f" stored={store_bytes / 1e6:.1f}MB"
    print(
        f"{run.init_time:%Y-%m-%dT%HZ} {STATUS_NAMES[run.status]} "
        f"files={run.files_received}/{run.files_expected} raw={timings.raw_bytes / 1e6:.0f}MB"
        f"{stored} total={elapsed:.0f}s download={speed:.1f}MB/s"
    )


def write_documents(*, product_dir: Path, store: UkvStore, grid: CellGrid) -> None:
    """Write `_grid_cells.parquet`, the lineage note, and the README beside the store."""
    cells = pl.DataFrame(
        {
            "cell": np.arange(grid.n_cells, dtype=np.int32),
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        }
    )
    cells.write_parquet(product_dir / "_grid_cells.parquet")
    statuses = store.statuses()
    counts = {name: int((statuses == code).sum()) for code, name in STATUS_NAMES.items()}
    write_lineage_note(
        product_dir=product_dir,
        source_address=BASE_URL,
        request_description=(
            "Met Office UKV GRIB files from the CEDA archive: Wholesale1 to Wholesale4 and the "
            "T54 files of Wholesale1 to Wholesale3, for the 00, 06, 12, and 18 UTC runs, cropped "
            "to the private trial-area box."
        ),
        variables=[spec.variable for spec in FIELDS],
        extra={
            "run_counts_by_status": counts,
            "code_version": CODE_VERSION,
            "note": "The cell coordinates are in _grid_cells.parquet and the store, both private.",
        },
    )
    write_readme(
        product_dir=product_dir,
        product_name="Met Office UKV 2 km, from the CEDA archive",
        source_web_page=CATALOGUE_URL,
        script_path="studies/weather_downloads/fetch_ukv_ceda.py",
        columns=readme_columns(),
        missing_value_convention=(
            "A variable is NaN at every lead its files do not serve. A slot that was never "
            "archived, or that CEDA does not hold, has status 0 or 3 and every variable NaN. "
            "Below-ground pressure-level cells, and convective cloud top where no convective "
            f"cloud exists, are NaN. Run counts by status: {counts}."
        ),
        gotchas=readme_gotchas(),
        external_docs={
            "CEDA UKV catalogue record and licence": CATALOGUE_URL,
            "Met Office UKV file format description": (
                "https://dap.ceda.ac.uk/badc/ukmo-nwp/doc/NWP_UKV_Information.pdf"
            ),
        },
        lineage_filenames=["lineage.json"],
    )


def readme_columns() -> dict[str, str]:
    """Describe every array in the store, one line each, for the README."""
    columns = {
        "init_time": "Slot coordinate, seconds since 1970-01-01 UTC, 6-hourly from 2019-09-01 00Z.",
        "step": "Lead time in hours, 0 to 54, hourly.",
        "status": "0 never archived, 1 complete, 2 partial, 3 missing on CEDA.",
        "files_expected / files_received": "Files the run should have, and files that arrived.",
        "archived_at / code_version": "When the run was committed, and by which script version.",
        "cell_row, cell_column, cell_latitude, cell_longitude": (
            "Private: position of each cell on the full grid and in degrees."
        ),
    }
    for spec in FIELDS:
        columns[spec.variable] = f"{spec.description}, {spec.units}. GRIB2 key {spec.key}."
    return columns


def readme_gotchas() -> list[str]:
    """The traps a reader of the store would otherwise rediscover, for the README."""
    return [
        (
            "Licence: CC BY-NC-SA 4.0, so non-commercial use only, and adaptations must be shared "
            "alike. Any commercial or production use needs the maintainer's confirmation first. "
            "The licence file asks users to cite the data as: Met Office (2016): NWP-UKV: Met "
            "Office UK Atmospheric High Resolution Model data. Centre for Environmental Data "
            f"Analysis, date of citation. {CATALOGUE_URL}"
        ),
        (
            "A GRIB2 key in the column list is (discipline, parameter category, parameter number, "
            "type of first fixed surface, its value, type of second fixed surface, its value, "
            "product definition template). Type 100 is a pressure in pascals, 103 a height in "
            "metres, 1 the ground, and 255 no second surface."
        ),
        (
            "The archive carries one downward shortwave field, the total (0/4/7). No direct or "
            "diffuse shortwave exists in any Wholesale group, so this product cannot give a "
            "beam and diffuse split."
        ),
        (
            "Fields not kept, identical in the 2019-09-01 00Z and 2026-09-20 00Z files inspected: "
            "in Wholesale2, 0/0/210 (unidentified, 800 to 3800 in the 2019 file) and 0/3/6 (height "
            "of the 0 degrees C isotherm); in Wholesale3, temperature and relative humidity at all "
            "15 pressure levels (30 to 1000 hPa), and wind and geopotential height at the 13 "
            "levels other than 1000 and 925 hPa; and all of Wholesale5. Every field of "
            "Wholesale1 and Wholesale4 is kept."
        ),
        (
            "CEDA UKV differs statistically from the live UKV feed: do not train on one and "
            "infer on the other."
        ),
        (
            "Runs at 00, 06, 12, and 18 UTC reach 54 hours at most, because CEDA holds T120 "
            "files (leads 55 to 120 hours) for the 03 and 15 UTC runs only. Only the 00, 06, 12, "
            "and 18 UTC runs are archived, so no lead beyond 54 hours exists here."
        ),
        (
            "Leads are hourly to 48 hours and 3-hourly to 54 hours (51 and 54), so steps 49, 50, "
            "52, and 53 are NaN. Wind gusts exist to lead 36 hours only."
        ),
        (
            "Downward shortwave and longwave flux are instantaneous values, not means since the "
            "start of the run, and lead 0 is served."
        ),
        (
            "Precipitation amount and maximum gust are over the interval since the previous "
            "served step: 1 hour to lead 48, then 3 hours at leads 51 and 54. They have no "
            "lead 0. The fetch script checks every such message: precipitation must carry "
            "statistical process 1 (accumulation), gust must carry 2 (maximum), and the "
            "interval must be the expected length, else the run is partial and the message is "
            "not stored."
        ),
        (
            "Screen-level fields are coded at level 1 (m) in the GRIB files, though the Met "
            "Office documents screen level as 1.5 m."
        ),
        (
            "Cloud fractions are percentages, 0 to 100. Wind and geopotential height at 1000 hPa "
            "and 925 hPa replace hub-height wind: the files hold no model-level wind."
        ),
        (
            "Two parameters (0/1/230 and 0/6/26) have no eccodes name. Wholesale5 (0/17/194, "
            "integer-valued 2 to 5) is not archived and is unidentified."
        ),
        (
            "CEDA corrected the transverse Mercator metadata in files from 2020-01-15: earlier "
            "files carry a wrongly encoded origin latitude and longitude. The grid itself is "
            "identical, and the script computes cell positions from the Ordnance Survey national "
            "grid."
        ),
        (
            "The first row of the grid is the northernmost, although the GRIB scanning mode says "
            "otherwise. Values are packed at 8 bits in the source files."
        ),
    ]


def _parse_date(text: str) -> date:
    """Parse a `YYYY-MM-DD` command-line date."""
    return date.fromisoformat(text)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--start", type=_parse_date, default=SLOT_EPOCH.date())
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--retry-partial", action="store_true")
    parser.add_argument("--measure", action="store_true", help="print the stored bytes per run")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--store-dir", type=Path, default=WEATHER_DOWNLOADS_DIR / PRODUCT_NAME)
    return parser


if __name__ == "__main__":
    sys.exit(archive(build_parser().parse_args()))
