"""Pure helpers for the ERA5 solar-variables download.

They hold the variable list, the request plan, and the reading of a downloaded archive into tidy
frames.

One-off throwaway module for the study of which ERA5 variables help explain the sunlight reaching a
solar farm. It is shared by `fetch_era5_solar_variables.py` and
`validate_era5_solar_variables.py`. Nothing here contacts the network or reads private data, so
every function is testable on its own.

**The variables the study asks for are grouped into four tiers.** A tier is the unit a user selects
with `--tier`, so the first results do not wait for the whole queue. Within a tier, accumulated
fields and instantaneous fields go in separate requests, because the Climate Data Store (CDS) stores
them under different GRIB step types and would otherwise split one request across several files
whose time axes differ.
"""

import math
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, Protocol

import numpy as np
import polars as pl
import xarray as xr
from era5_cells import Chunk, half_year_chunks
from studies.era5_grid import (
    AREA,
    FIRST_MONTH_OF_FIRST_YEAR,
    GRID_LATITUDES,
    GRID_LONGITUDES,
    PUBLISHED_FIRST_YEAR,
    PUBLISHED_LAST_DATE,
)

VariableKindType = Literal["instantaneous", "accumulation"]
"""An accumulation is a total over the hour ending at the time stamp; every other field is a
snapshot at the time stamp."""

TierType = Literal["tier1a", "tier1b", "tier2", "tier3"]
SeamFamilyType = Literal["cloud", "analysed", "solar_accumulation", "other_accumulation", "none"]
"""Which seams in the hour-of-day profile of a variable's hour-to-hour change are checked for a
step: see `SEAM_HOURS` and `SEAM_RULES`."""
SeamRuleType = Literal["median_of_others", "larger_neighbour"]

CDS_DATASET: Final[str] = "reanalysis-era5-single-levels"

CDS_FIELD_LIMIT: Final[int] = 121_000
"""The largest request, in the Climate Data Store's own field count, that the store accepts."""

PLANNING_FIELD_LIMIT: Final[int] = 110_000
"""The size the planner aims under, which leaves about 9% for a miscount."""

COST_FIELDS_PER_VARIABLE_HOUR: Final[int] = 6
"""How many of the store's fields one variable-hour counts for.

Measured twice, both exact: the store accepted 78,192 for 3 variables over January to June 2021
(4,344 hours each) and 104,832 for 4 variables over January to June 2020 (4,368 hours each), which
is 6 per variable-hour in both cases. The count does not depend on the area. The plan's earlier
estimate of about 62,000 fields for one variable over the whole span counted a variable-hour as one
field, which understates the store's count by this factor.
"""

MAX_HALF_YEAR_HOURS: Final[int] = 184 * 24
"""The hours in the longest half-year chunk, July to December."""

MAX_VARIABLES_PER_REQUEST: Final[int] = PLANNING_FIELD_LIMIT // (
    COST_FIELDS_PER_VARIABLE_HOUR * MAX_HALF_YEAR_HOURS
)
"""Four: the most variables whose longest half-year chunk stays under `PLANNING_FIELD_LIMIT`."""

FIRST_HOUR: Final[datetime] = datetime(
    PUBLISHED_FIRST_YEAR, FIRST_MONTH_OF_FIRST_YEAR, 1, tzinfo=UTC
)
LAST_HOUR: Final[datetime] = datetime.fromisoformat(PUBLISHED_LAST_DATE).replace(
    hour=23, tzinfo=UTC
)
"""The first and last hourly stamps of the published span, the same span as the held ERA5 copy."""

SPAN_FIRST_MONTH: Final[tuple[int, int]] = (FIRST_HOUR.year, FIRST_HOUR.month)
SPAN_LAST_MONTH: Final[tuple[int, int]] = (LAST_HOUR.year, LAST_HOUR.month)
"""The months the requests cover. Whole months are requested and trimmed to the span afterwards."""

PILOT_MONTH: Final[tuple[int, int]] = (2025, 6)

CELL_COUNT: Final[int] = len(GRID_LATITUDES) * len(GRID_LONGITUDES)
"""The 20 cells of the public box that the held ERA5 copy also covers."""


@dataclass(frozen=True)
class Variable:
    """One ERA5 variable the study asks for.

    Attributes:
        short_name: The ERA5 short name, which is also the netCDF variable name and the parquet
            file stem.
        cds_name: The name the CDS request form takes.
        kind: Whether the field is an accumulation over the hour ending at the stamp or a snapshot.
        tier: The tier the variable is fetched in.
        expected_units: The unit the CDS documents for the field, used to label tables and to
            derive the physical limits below. The unit read from the downloaded file is recorded
            beside it.
        minimum: The physical lower limit in `expected_units`, or `None` for no limit.
        maximum: The physical upper limit in `expected_units`, or `None` for no limit.
        seam_family: Which seams in the hour-of-day profile are checked for a step.
        nan_means_no_cloud: Whether the field is missing where there is no cloud, so a NaN is a
            value and not a defect.
    """

    short_name: str
    cds_name: str
    kind: VariableKindType
    tier: TierType
    expected_units: str
    minimum: float | None
    maximum: float | None
    seam_family: SeamFamilyType
    nan_means_no_cloud: bool = False


VARIABLES: Final[tuple[Variable, ...]] = (
    Variable(
        short_name="tcc",
        cds_name="total_cloud_cover",
        kind="instantaneous",
        tier="tier1a",
        expected_units="(0 - 1)",
        minimum=0,
        maximum=1,
        seam_family="cloud",
    ),
    Variable(
        short_name="lcc",
        cds_name="low_cloud_cover",
        kind="instantaneous",
        tier="tier1a",
        expected_units="(0 - 1)",
        minimum=0,
        maximum=1,
        seam_family="cloud",
    ),
    Variable(
        short_name="mcc",
        cds_name="medium_cloud_cover",
        kind="instantaneous",
        tier="tier1a",
        expected_units="(0 - 1)",
        minimum=0,
        maximum=1,
        seam_family="cloud",
    ),
    Variable(
        short_name="hcc",
        cds_name="high_cloud_cover",
        kind="instantaneous",
        tier="tier1a",
        expected_units="(0 - 1)",
        minimum=0,
        maximum=1,
        seam_family="cloud",
    ),
    Variable(
        short_name="tclw",
        cds_name="total_column_cloud_liquid_water",
        kind="instantaneous",
        tier="tier1b",
        expected_units="kg m**-2",
        minimum=0,
        maximum=50,
        seam_family="cloud",
    ),
    Variable(
        short_name="tciw",
        cds_name="total_column_cloud_ice_water",
        kind="instantaneous",
        tier="tier1b",
        expected_units="kg m**-2",
        minimum=0,
        maximum=50,
        seam_family="cloud",
    ),
    Variable(
        short_name="tcslw",
        cds_name="total_column_supercooled_liquid_water",
        kind="instantaneous",
        tier="tier1b",
        expected_units="kg m**-2",
        minimum=0,
        maximum=10,
        seam_family="cloud",
    ),
    Variable(
        short_name="cbh",
        cds_name="cloud_base_height",
        kind="instantaneous",
        tier="tier1b",
        expected_units="m",
        minimum=0,
        maximum=20000,
        seam_family="cloud",
        nan_means_no_cloud=True,
    ),
    Variable(
        short_name="ssrdc",
        cds_name="surface_solar_radiation_downward_clear_sky",
        kind="accumulation",
        tier="tier1b",
        expected_units="J m**-2",
        minimum=0,
        maximum=5400000.0,
        seam_family="solar_accumulation",
    ),
    Variable(
        short_name="cdir",
        cds_name="clear_sky_direct_solar_radiation_at_surface",
        kind="accumulation",
        tier="tier1b",
        expected_units="J m**-2",
        minimum=0,
        maximum=5400000.0,
        seam_family="solar_accumulation",
    ),
    Variable(
        short_name="strd",
        cds_name="surface_thermal_radiation_downwards",
        kind="accumulation",
        tier="tier1b",
        expected_units="J m**-2",
        minimum=0,
        maximum=2000000.0,
        seam_family="solar_accumulation",
    ),
    Variable(
        short_name="d2m",
        cds_name="2m_dewpoint_temperature",
        kind="instantaneous",
        tier="tier1b",
        expected_units="K",
        minimum=200,
        maximum=320,
        seam_family="analysed",
    ),
    Variable(
        short_name="tcwv",
        cds_name="total_column_water_vapour",
        kind="instantaneous",
        tier="tier1b",
        expected_units="kg m**-2",
        minimum=0,
        maximum=100,
        seam_family="analysed",
    ),
    Variable(
        short_name="blh",
        cds_name="boundary_layer_height",
        kind="instantaneous",
        tier="tier1b",
        expected_units="m",
        minimum=0,
        maximum=8000,
        seam_family="none",
    ),
    Variable(
        short_name="u10",
        cds_name="10m_u_component_of_wind",
        kind="instantaneous",
        tier="tier1b",
        expected_units="m s**-1",
        minimum=-60,
        maximum=60,
        seam_family="analysed",
    ),
    Variable(
        short_name="v10",
        cds_name="10m_v_component_of_wind",
        kind="instantaneous",
        tier="tier1b",
        expected_units="m s**-1",
        minimum=-60,
        maximum=60,
        seam_family="analysed",
    ),
    Variable(
        short_name="sd",
        cds_name="snow_depth",
        kind="instantaneous",
        tier="tier2",
        expected_units="m of water equivalent",
        minimum=0,
        maximum=20,
        seam_family="analysed",
    ),
    Variable(
        short_name="sf",
        cds_name="snowfall",
        kind="accumulation",
        tier="tier2",
        expected_units="m of water equivalent",
        minimum=0,
        maximum=0.2,
        seam_family="other_accumulation",
    ),
    Variable(
        short_name="asn",
        cds_name="snow_albedo",
        kind="instantaneous",
        tier="tier2",
        expected_units="(0 - 1)",
        minimum=0,
        maximum=1,
        seam_family="none",
    ),
    Variable(
        short_name="fal",
        cds_name="forecast_albedo",
        kind="instantaneous",
        tier="tier2",
        expected_units="(0 - 1)",
        minimum=0,
        maximum=1,
        seam_family="none",
    ),
    Variable(
        short_name="tp",
        cds_name="total_precipitation",
        kind="accumulation",
        tier="tier2",
        expected_units="m",
        minimum=0,
        maximum=0.5,
        seam_family="other_accumulation",
    ),
    Variable(
        short_name="cape",
        cds_name="convective_available_potential_energy",
        kind="instantaneous",
        tier="tier2",
        expected_units="J kg**-1",
        minimum=0,
        maximum=10000,
        seam_family="none",
    ),
    Variable(
        short_name="cin",
        cds_name="convective_inhibition",
        kind="instantaneous",
        tier="tier2",
        expected_units="J kg**-1",
        minimum=0,
        maximum=5000,
        seam_family="none",
        nan_means_no_cloud=True,
    ),
    Variable(
        short_name="skt",
        cds_name="skin_temperature",
        kind="instantaneous",
        tier="tier2",
        expected_units="K",
        minimum=200,
        maximum=340,
        seam_family="analysed",
    ),
    Variable(
        short_name="sp",
        cds_name="surface_pressure",
        kind="instantaneous",
        tier="tier2",
        expected_units="Pa",
        minimum=45000,
        maximum=110000,
        seam_family="analysed",
    ),
    Variable(
        short_name="i10fg",
        cds_name="instantaneous_10m_wind_gust",
        kind="instantaneous",
        tier="tier2",
        expected_units="m s**-1",
        minimum=0,
        maximum=120,
        seam_family="none",
    ),
    Variable(
        short_name="tcrw",
        cds_name="total_column_rain_water",
        kind="instantaneous",
        tier="tier3",
        expected_units="kg m**-2",
        minimum=0,
        maximum=50,
        seam_family="cloud",
    ),
    Variable(
        short_name="tcsw",
        cds_name="total_column_snow_water",
        kind="instantaneous",
        tier="tier3",
        expected_units="kg m**-2",
        minimum=0,
        maximum=50,
        seam_family="cloud",
    ),
    Variable(
        short_name="tco3",
        cds_name="total_column_ozone",
        kind="instantaneous",
        tier="tier3",
        expected_units="kg m**-2",
        minimum=0,
        maximum=0.05,
        seam_family="none",
    ),
    Variable(
        short_name="uvb",
        cds_name="downward_uv_radiation_at_the_surface",
        kind="accumulation",
        tier="tier3",
        expected_units="J m**-2",
        minimum=0,
        maximum=800000.0,  # about 15% of the 5.4e6 clear-sky ceiling; sunny summer noon is ~2.5e5
        seam_family="solar_accumulation",
    ),
    Variable(
        short_name="deg0l",
        cds_name="zero_degree_level",
        kind="instantaneous",
        tier="tier3",
        expected_units="m",
        minimum=0,
        maximum=8000,
        seam_family="none",
    ),
)
"""Every new variable, in the order of the tiers. `ssrd`, `fdir`, and `t2m` are already held in
`beam_diffuse/` and are not fetched again."""

VARIABLES_BY_NAME: Final[dict[str, Variable]] = {v.short_name: v for v in VARIABLES}
TIERS: Final[tuple[TierType, ...]] = ("tier1a", "tier1b", "tier2", "tier3")

SEAM_HOURS: Final[dict[SeamFamilyType, tuple[int, ...]]] = {
    "cloud": (6, 7, 9, 10, 18, 19, 21, 22),
    "analysed": (9, 10, 21, 22),
    "solar_accumulation": (7, 19),
    "other_accumulation": (7, 19),
    "none": (),
}
"""The UTC hours of the hour-of-day profile checked for a step, by family.

Cloud and cloud-water fields come from the 06 and 18 UTC forecasts, so a step shows at 06 or 07 and
at 18 or 19. They are also checked at 09, 10, 21, and 22 UTC, the 4D-Var window boundaries and the
hour after, because the exact hour of a step depends on how the stamps are labelled. Analysed
fields are checked at those four hours. ERA5 accumulations change forecast run at 07 and 19 UTC.
"""

SEAM_RULES: Final[dict[SeamFamilyType, SeamRuleType]] = {
    "cloud": "median_of_others",
    "analysed": "median_of_others",
    "solar_accumulation": "larger_neighbour",
    "other_accumulation": "median_of_others",
    "none": "median_of_others",
}
"""How a seam hour is compared. The median rule suits a field with no strong daily cycle. A solar
accumulation follows the sun, so the hour-to-hour change is near zero at night and large at
sunrise, and the median of the other hours is no baseline. A seam bump there is instead a value
more than the factor above both neighbouring hours."""

STEP_FACTOR: Final[float] = 3.0
"""A seam hour is flagged when its value exceeds this multiple of the baseline."""


def variables_in(*, tiers: Sequence[TierType]) -> list[Variable]:
    """Return the variables of the given tiers, in registry order."""
    return [variable for variable in VARIABLES if variable.tier in tiers]


@dataclass(frozen=True)
class PlannedChunk:
    """One request: a group of variables of one kind over one calendar half-year.

    Attributes:
        tier: The tier the variables belong to.
        kind: Whether the group holds accumulations or instantaneous fields; never both.
        group_index: The group's position among the tier's groups of this kind.
        variables: The ERA5 short names requested together.
        period: The calendar half-year, never crossing a year boundary.
    """

    tier: TierType
    kind: VariableKindType
    group_index: int
    variables: tuple[str, ...]
    period: Chunk

    @property
    def chunk_id(self) -> str:
        """Return a filename-safe id such as `tier1a_instantaneous_g0_2019_09_12`."""
        return f"{self.tier}_{self.kind}_g{self.group_index}_{self.period.name}"

    @property
    def variable_hours(self) -> int:
        """Return how many variable-hours the request asks for."""
        return len(self.variables) * self.period.n_hours

    @property
    def cost_fields(self) -> int:
        """Return the request's size in the store's own field count."""
        return self.variable_hours * COST_FIELDS_PER_VARIABLE_HOUR


def balanced_groups(*, names: Sequence[str], max_size: int) -> list[tuple[str, ...]]:
    """Split `names` into the fewest groups of at most `max_size`, as even in size as possible.

    Args:
        names: The items to group, in order.
        max_size: The largest group allowed.

    Returns:
        The groups, in order; empty if `names` is empty.
    """
    if not names:
        return []
    n_groups = math.ceil(len(names) / max_size)
    base, extra = divmod(len(names), n_groups)
    groups = []
    start = 0
    for index in range(n_groups):
        size = base + (1 if index < extra else 0)
        groups.append(tuple(names[start : start + size]))
        start += size
    return groups


def plan_chunks(
    *,
    tiers: Sequence[TierType],
    first_month: tuple[int, int] = SPAN_FIRST_MONTH,
    last_month: tuple[int, int] = SPAN_LAST_MONTH,
) -> list[PlannedChunk]:
    """Return every request for the given tiers, group by group, each group in time order.

    A group finishes for every half-year before the next group starts, so stopping partway leaves
    complete variables rather than a little of each.

    Args:
        tiers: The tiers to plan.
        first_month: The first `(year, month)` to request.
        last_month: The last `(year, month)` to request.

    Returns:
        The planned requests.

    Raises:
        ValueError: If a request would exceed the store's field limit.
    """
    periods = half_year_chunks(first=first_month, last=last_month)
    chunks = []
    for tier in tiers:
        for kind in ("instantaneous", "accumulation"):
            names = [v.short_name for v in variables_in(tiers=[tier]) if v.kind == kind]
            for group_index, group in enumerate(
                balanced_groups(names=names, max_size=MAX_VARIABLES_PER_REQUEST)
            ):
                chunks.extend(
                    PlannedChunk(
                        tier=tier,
                        kind=kind,
                        group_index=group_index,
                        variables=group,
                        period=period,
                    )
                    for period in periods
                )
    too_big = [c.chunk_id for c in chunks if c.cost_fields > CDS_FIELD_LIMIT]
    if too_big:
        msg = f"requests over the {CDS_FIELD_LIMIT} field limit: {too_big}"
        raise ValueError(msg)
    return chunks


def pilot_chunks(*, month: tuple[int, int] = PILOT_MONTH) -> list[PlannedChunk]:
    """Return the pilot: one month of three groups that between them cover every kind of request.

    The groups are the first `tier1a` instantaneous group (cloud covers, which every cloud-NaN
    split needs), the first `tier1b` accumulation group (the largest accumulations, whose hour
    convention is the riskiest), and the first `tier2` instantaneous group (snow and convective
    fields).

    Args:
        month: The `(year, month)` to fetch.

    Returns:
        Three chunks, in that order.
    """
    wanted = (("tier1a", "instantaneous"), ("tier1b", "accumulation"), ("tier2", "instantaneous"))
    chunks = plan_chunks(tiers=TIERS, first_month=month, last_month=month)
    return [
        next(c for c in chunks if (c.tier, c.kind, c.group_index) == (tier, kind, 0))
        for tier, kind in wanted
    ]


def request_body(*, chunk: PlannedChunk) -> dict[str, object]:
    """Return the CDS request body for one chunk, over the public ERA5 box.

    Args:
        chunk: The chunk to request.

    Returns:
        The body `cdsapi.Client.retrieve` takes. It holds only the public `AREA`.
    """
    return {
        "product_type": ["reanalysis"],
        "variable": [VARIABLES_BY_NAME[name].cds_name for name in chunk.variables],
        "year": [str(chunk.period.year)],
        "month": [f"{month:02d}" for month in chunk.period.months],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": list(AREA),
        "data_format": "netcdf",
        "download_format": "zip",
    }


def expected_hours(*, first_hour: datetime, last_hour: datetime) -> int:
    """Return how many hourly stamps lie in `[first_hour, last_hour]`."""
    return int((last_hour - first_hour).total_seconds() // 3600) + 1


def expected_rows(*, first_hour: datetime, last_hour: datetime) -> int:
    """Return the row count of one variable's table: the 20 cells times the hours."""
    return CELL_COUNT * expected_hours(first_hour=first_hour, last_hour=last_hour)


def chunk_is_valid(*, path: Path) -> bool:
    """Return whether `path` holds a complete, readable archive with at least one netCDF member.

    An empty file, a truncated download, and a non-archive all return `False`, so the resume logic
    fetches them again instead of skipping them for good.

    Args:
        path: The archive to test.

    Returns:
        `True` if the file exists, is not empty, passes the archive's own checksum test, and holds
        a `.nc` member.
    """
    if not path.exists() or path.stat().st_size == 0 or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None and any(
                name.endswith(".nc") for name in archive.namelist()
            )
    except zipfile.BadZipFile, OSError:
        return False


class ChunkLike(Protocol):
    """What `run_chunks` needs to know about a request."""

    @property
    def chunk_id(self) -> str:
        """Return a filename-safe id."""
        ...

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the variables requested."""
        ...

    @property
    def variable_hours(self) -> int:
        """Return the request's size in variable-hours."""
        ...

    @property
    def cost_fields(self) -> int:
        """Return the request's size in the store's field count."""
        ...


class ChunkRecord(NamedTuple):
    """What one chunk's run left behind.

    Attributes:
        chunk_id: The chunk's id.
        variables: The short names requested.
        variable_hours: The request's size in variable-hours.
        cost_fields: The request's size in the store's field count.
        bytes: The archive's size on disk.
        seconds: The wall time the request took; `None` if the chunk was already on disk.
        skipped: Whether the chunk was already on disk and valid.
    """

    chunk_id: str
    variables: tuple[str, ...]
    variable_hours: int
    cost_fields: int
    bytes: int
    seconds: float | None
    skipped: bool


def run_chunks[ChunkT: ChunkLike](
    *,
    chunks: Sequence[ChunkT],
    chunk_dir: Path,
    download: Callable[[ChunkT, Path], None],
    log: Callable[[str], None],
    on_record: Callable[[ChunkRecord], None] = lambda _: None,
) -> list[ChunkRecord]:
    """Fetch each chunk one after another, skipping those already on disk and valid.

    Each chunk is written to a `.partial` name and renamed once it passes `chunk_is_valid`, so an
    interrupted run never leaves a truncated archive under a final name. One failure raises and
    stops the loop; a re-run resumes from the cache. One request runs at a time because the store
    runs one job per account.

    Args:
        chunks: The chunks to fetch, in order.
        chunk_dir: Where the archives go.
        download: Writes one chunk to the path it is given. Called with `(chunk, destination)`.
        log: Receives one line per chunk: the id, the fields, the bytes, and the seconds.
        on_record: Called with each chunk's record as soon as the chunk is finished or skipped, so
            a caller can persist the record before the next request starts.

    Returns:
        One record per chunk.

    Raises:
        RuntimeError: If a download leaves an archive that is not valid.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for chunk in chunks:
        path = chunk_dir / f"{chunk.chunk_id}.zip"
        if chunk_is_valid(path=path):
            record = ChunkRecord(
                chunk_id=chunk.chunk_id,
                variables=chunk.variables,
                variable_hours=chunk.variable_hours,
                cost_fields=chunk.cost_fields,
                bytes=path.stat().st_size,
                seconds=None,
                skipped=True,
            )
            log(f"{chunk.chunk_id}: already valid on disk, skipping")
            records.append(record)
            on_record(record)
            continue
        partial = path.with_suffix(".zip.partial")
        partial.unlink(missing_ok=True)
        started = time.monotonic()
        download(chunk, partial)
        seconds = time.monotonic() - started
        if not chunk_is_valid(path=partial):
            msg = f"{chunk.chunk_id}: the download is not a valid archive; left at {partial.name}"
            raise RuntimeError(msg)
        partial.rename(path)
        record = ChunkRecord(
            chunk_id=chunk.chunk_id,
            variables=chunk.variables,
            variable_hours=chunk.variable_hours,
            cost_fields=chunk.cost_fields,
            bytes=path.stat().st_size,
            seconds=seconds,
            skipped=False,
        )
        log(
            f"{chunk.chunk_id}: {chunk.variable_hours} variable-hours "
            f"({chunk.cost_fields} store fields), {record.bytes} bytes, {seconds:.1f} s"
        )
        records.append(record)
        on_record(record)
    return records


def retrieve_with_cleanup(
    *,
    client: Any,
    collection: str,
    request: dict[str, object],
    destination: Path,
    log: Callable[[str], None],
    max_wait_seconds: float = 12 * 3600,
    poll_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Submit one request, wait for it, download it, and delete the remote job if anything stops us.

    The store runs one job per account at a time, so a job left queued after an interrupted run
    blocks every later request until it finishes. This function deletes the job when the wait is
    interrupted (`KeyboardInterrupt`, or `SIGTERM` turned into one by the caller), when the wait
    exceeds `max_wait_seconds`, and when the job or the download fails.

    **A hard kill leaves the job queued.** `SIGKILL`, an out-of-memory kill, and a power cut run no
    cleanup. The job id is logged at submission, so the job can then be deleted by hand with
    `ecmwf.datastores.Client().delete(job_id)`.

    Args:
        client: An `ecmwf.datastores.Client`.
        collection: The dataset id.
        request: The request body.
        destination: Where the downloaded file goes.
        log: Receives the job id and the delete outcome. Never receives the key.
        max_wait_seconds: How long to wait for the job before giving up.
        poll_seconds: The pause between status checks.
        sleep: The pause function, replaceable in tests.

    Raises:
        TimeoutError: If the job is not ready within `max_wait_seconds`.
    """
    remote = client.submit(collection, request)
    log(f"job {remote.request_id} submitted")
    waited = 0.0
    try:
        while not remote.results_ready:
            if waited >= max_wait_seconds:
                msg = f"job {remote.request_id} not ready after {max_wait_seconds:.0f} s"
                raise TimeoutError(msg)  # noqa: TRY301
            sleep(poll_seconds)
            waited += poll_seconds
        remote.get_results().download(str(destination))
    except BaseException:
        try:
            remote.delete()
            log(f"job {remote.request_id} deleted after an interruption or failure")
        except Exception as error:  # noqa: BLE001
            log(f"job {remote.request_id} could not be deleted ({type(error).__name__})")
        raise


def expver_label(*, value: object) -> str:
    """Return an `expver` value as the four-character label ERA5 uses, such as `0001`.

    Args:
        value: The value as the file holds it: a string such as `0001`, an integer such as `1`, or
            bytes.

    Returns:
        The zero-padded label.
    """
    text = value.decode() if isinstance(value, bytes) else str(value)
    return f"{int(float(text.strip())):04d}"


def _time_name(*, dataset: xr.Dataset) -> str:
    return "valid_time" if "valid_time" in dataset.variables else "time"


def collapse_expver(*, dataset: xr.Dataset) -> tuple[xr.Dataset, np.ndarray]:
    """Remove the `expver` information from `dataset` and return it as one label per hour.

    ERA5 mixes the final release (`expver` 0001) with the preliminary release for the latest
    months (0005). A file can carry `expver` in three layouts, and all three are handled:

    - a dimension of the data variables, with all values missing under one label at each hour (the
      layout the legacy archive used, and the one CDS uses when a request spans both releases);
    - a variable along the time axis, holding one label per hour;
    - a scalar.

    Args:
        dataset: One netCDF file from a chunk.

    Returns:
        The dataset without `expver`, and an array of one label per time stamp.

    Raises:
        ValueError: If the file carries no `expver`, carries it in a layout not listed above, or
            holds data for an hour under no label or under two.
    """
    time_name = _time_name(dataset=dataset)
    n_times = dataset.sizes[time_name]
    if "expver" not in dataset.variables:
        msg = "the file carries no `expver`, so which release each hour is cannot be recorded"
        raise ValueError(msg)
    expver = dataset["expver"]
    if "expver" in dataset.dims:
        return _collapse_expver_dimension(dataset=dataset, time_name=time_name)
    labels = np.array([expver_label(value=v) for v in expver.values.ravel()])
    if expver.ndim == 0:
        per_hour = np.full(n_times, labels[0])
    elif expver.dims == (time_name,):
        per_hour = labels
    else:
        msg = f"unrecognised `expver` layout with dimensions {expver.dims}"
        raise ValueError(msg)
    return dataset.drop_vars("expver"), per_hour


def _collapse_expver_dimension(
    *, dataset: xr.Dataset, time_name: str
) -> tuple[xr.Dataset, np.ndarray]:
    """Pick, hour by hour, the `expver` slice that holds data, for a file with an `expver` axis."""
    labels = [expver_label(value=v) for v in dataset["expver"].values]
    n_times = dataset.sizes[time_name]
    names = [name for name in dataset.data_vars if "expver" in dataset[name].dims]
    stacked = {name: dataset[name].transpose("expver", time_name, ...) for name in names}
    has_data = np.zeros((len(labels), n_times), dtype=bool)
    for array in stacked.values():
        flat = array.values.reshape(len(labels), n_times, -1)
        has_data |= (~np.isnan(flat)).any(axis=2)
    labels_per_hour = has_data.sum(axis=0)
    if (labels_per_hour != 1).any():
        msg = (
            f"{int((labels_per_hour != 1).sum())} hours hold data under no `expver` label or "
            "under more than one"
        )
        raise ValueError(msg)
    chosen = has_data.argmax(axis=0)
    collapsed = {}
    for name, array in stacked.items():
        picked = array.values[chosen, np.arange(n_times)]
        dims = array.dims[1:]
        collapsed[name] = xr.DataArray(
            picked,
            dims=dims,
            coords={d: array.coords[d] for d in dims if d in array.coords},
            attrs=array.attrs,
        )
    kept = {name: dataset[name] for name in dataset.data_vars if "expver" not in dataset[name].dims}
    result = xr.Dataset({**kept, **collapsed}).drop_vars("expver", errors="ignore")
    return result, np.array(labels)[chosen]


class ParsedArchive(NamedTuple):
    """The contents of one chunk archive.

    Attributes:
        frames: One tidy frame per variable, keyed by the netCDF variable name.
        units: The `units` attribute of each variable, as the file states it.
    """

    frames: dict[str, pl.DataFrame]
    units: dict[str, str]


def dataset_to_frames(*, dataset: xr.Dataset, has_expver: bool = True) -> ParsedArchive:
    """Turn one netCDF file into a tidy frame per variable.

    Args:
        dataset: One file of a chunk archive.
        has_expver: Whether the file carries `expver`, as ERA5 files do. A product without it,
            such as the CAMS reanalysis, gets frames without an `expver` column.

    Returns:
        Frames with `time` (UTC, the stamp the file labels the value with), `latitude`,
        `longitude` (`Float64`, degrees), `value` (`Float32`, in the file's own unit, NaN kept),
        and, if `has_expver`, `expver`.
    """
    time_name = _time_name(dataset=dataset)
    if has_expver:
        collapsed, expver_per_hour = collapse_expver(dataset=dataset)
    else:
        collapsed, expver_per_hour = dataset, np.full(dataset.sizes[time_name], "")
    times = collapsed[time_name].values.astype("datetime64[us]")
    latitudes = collapsed["latitude"].values.astype(np.float64).round(4)
    longitudes = collapsed["longitude"].values.astype(np.float64).round(4)
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    time_grid, lat_grid, lon_grid = np.meshgrid(times, latitudes, longitudes, indexing="ij")
    expver_grid = np.broadcast_to(expver_per_hour[:, None, None], time_grid.shape)
    frames = {}
    units = {}
    for name in collapsed.data_vars:
        array = collapsed[name].transpose(time_name, "latitude", "longitude")
        columns = {
            "time": time_grid.ravel(),
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "value": array.values.astype(np.float32).ravel(),
        }
        if has_expver:
            columns["expver"] = expver_grid.ravel()
        frames[str(name)] = pl.DataFrame(columns).with_columns(
            pl.col("time").cast(pl.Datetime("us", "UTC"))
        )
        units[str(name)] = str(array.attrs.get("units", ""))
    return ParsedArchive(frames=frames, units=units)


def check_expver_agreement(*, frames: dict[str, pl.DataFrame]) -> None:
    """Raise if two variables, or two hours' rows, disagree on the release of one hour.

    The check applies within one archive, and the planner never puts an accumulation and an
    instantaneous field in the same archive. The restriction matters on real data: an accumulation
    stamped 00 to 06 UTC on the first day of a month comes from the previous day's forecast, so it
    can carry a different `expver` from an instantaneous field at the same stamp. Comparing
    `expver` across the two kinds would raise on correct data, and so would a study build that
    requires equal `expver` across all variables for one hour; it should compare only within a kind.

    Args:
        frames: The frames of one chunk, keyed by variable.

    Raises:
        ValueError: If any hour carries more than one `expver` across the frames.
    """
    labels = pl.concat([frame.select("time", "expver").unique() for frame in frames.values()])
    disagreeing = (
        labels.unique()
        .group_by("time")
        .agg(pl.col("expver").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if disagreeing.height:
        msg = f"{disagreeing.height} hours carry different `expver` labels across variables"
        raise ValueError(msg)


def read_archive(*, path: Path, scratch_dir: Path, has_expver: bool = True) -> ParsedArchive:
    """Unpack one chunk archive and parse every netCDF file in it.

    Reading a netCDF file needs the `netCDF4` or `h5netcdf` package, which the workspace does not
    install; run with `uv run --with netCDF4`.

    Args:
        path: The archive.
        scratch_dir: A directory on a real disk (not `/tmp`) that the members are extracted into.
        has_expver: Whether the files carry `expver`; see `dataset_to_frames`.

    Returns:
        The frames and units of every variable in every member, with `expver` checked to agree.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pl.DataFrame] = {}
    units: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".nc"):
                continue
            extracted = Path(archive.extract(member, scratch_dir))
            try:
                with xr.open_dataset(extracted) as dataset:
                    parsed = dataset_to_frames(dataset=dataset, has_expver=has_expver)
            finally:
                extracted.unlink(missing_ok=True)
            duplicated = set(frames) & set(parsed.frames)
            if duplicated:
                msg = f"{path.name}: {sorted(duplicated)} appear in more than one member"
                raise ValueError(msg)
            frames |= parsed.frames
            units |= parsed.units
    if has_expver:
        check_expver_agreement(frames=frames)
    return ParsedArchive(frames=frames, units=units)


def check_cells(*, frame: pl.DataFrame) -> None:
    """Raise unless the frame covers exactly the 20 public cells.

    Args:
        frame: A tidy frame.

    Raises:
        ValueError: If the cells differ from `GRID_LATITUDES` times `GRID_LONGITUDES`.
    """
    cells = set(frame.select("latitude", "longitude").unique().iter_rows())
    expected = {(lat, lon) for lat in GRID_LATITUDES for lon in GRID_LONGITUDES}
    if cells != expected:
        msg = f"the frame's cells differ from the public box: {len(cells ^ expected)} differ"
        raise ValueError(msg)


def trim_to_span(*, frame: pl.DataFrame, first_hour: datetime, last_hour: datetime) -> pl.DataFrame:
    """Keep the rows stamped within `[first_hour, last_hour]`."""
    return frame.filter(pl.col("time").is_between(first_hour, last_hour))


def last_final_month(*, frame: pl.DataFrame) -> str | None:
    """Return the last month up to which every hour is final ERA5 (`expver` 0001).

    Args:
        frame: A tidy frame with `time` and `expver`.

    Returns:
        The `YYYY-MM` of the month before the first hour with another label, or the frame's last
        month if every hour is final, or `None` if the very first hour is not final.
    """
    months = (
        frame.select("time", "expver")
        .unique()
        .group_by(pl.col("time").dt.strftime("%Y-%m").alias("month"))
        .agg(is_final=(pl.col("expver") == "0001").all())
        .sort("month")
    )
    flags = months["is_final"].to_list()
    labels = months["month"].to_list()
    if all(flags):
        return labels[-1] if labels else None
    first_not_final = flags.index(False)
    return labels[first_not_final - 1] if first_not_final else None


def hour_profile(*, frame: pl.DataFrame) -> list[float | None]:
    """Return the mean absolute hour-to-hour change by UTC hour of day.

    The change at hour `h` is `|value(h) - value(h - 1)|` for a cell, over pairs of consecutive
    hourly stamps with both values present. Averaging over every cell and day gives one number per
    hour of day, and a step where the forecast run or the analysis window changes shows as a bump.

    Args:
        frame: A tidy frame with `time`, `latitude`, `longitude`, and `value`.

    Returns:
        24 numbers, index 0 for the change into 00 UTC; `None` where an hour has no valid pair.
    """
    cell = ["latitude", "longitude"]
    ordered = frame.sort(*cell, "time").with_columns(
        previous_time=pl.col("time").shift(1).over(cell),
        change=(pl.col("value") - pl.col("value").shift(1).over(cell)).abs(),
    )
    pairs = ordered.filter(
        (pl.col("time") - pl.col("previous_time") == pl.duration(hours=1))
        & pl.col("change").is_not_nan()
        & pl.col("change").is_not_null()
    )
    means = pairs.group_by(pl.col("time").dt.hour().alias("hour")).agg(pl.col("change").mean())
    by_hour = dict(means.iter_rows())
    return [by_hour.get(hour) for hour in range(24)]


def step_flags(
    *,
    profile: Sequence[float | None],
    seam_hours: Sequence[int],
    rule: SeamRuleType,
    factor: float = STEP_FACTOR,
) -> list[int]:
    """Return the seam hours whose profile value is a step.

    With the `median_of_others` rule an hour is flagged when its value exceeds `factor` times the
    median of the other hours. With the `larger_neighbour` rule it is flagged when it exceeds
    `factor` times the larger of the hours either side. A baseline of zero flags any positive value.

    Args:
        profile: The 24 values from `hour_profile`.
        seam_hours: The hours to test.
        rule: How to form the baseline.
        factor: The multiple of the baseline that counts as a step.

    Returns:
        The flagged hours, in the order given.
    """
    flagged = []
    for hour in seam_hours:
        value = profile[hour]
        if value is None:
            continue
        if rule == "median_of_others":
            others = [v for index, v in enumerate(profile) if index != hour and v is not None]
            baseline = float(np.median(others)) if others else 0.0
        else:
            neighbours = [profile[(hour - 1) % 24], profile[(hour + 1) % 24]]
            baseline = max((v for v in neighbours if v is not None), default=0.0)
        if value > factor * baseline:
            flagged.append(hour)
    return flagged


TCC_THRESHOLD: Final[float] = 0.05
"""Total cloud cover below this counts as clear sky for the missing-value split."""


def nan_share_by_tcc(
    *, frame: pl.DataFrame, tcc: pl.DataFrame, threshold: float = TCC_THRESHOLD
) -> dict[str, dict[str, float | int | None]]:
    """Return the share of missing values in `frame`, split by whether `tcc` is below `threshold`.

    A value counts as missing if it is NaN or null.

    Args:
        frame: A tidy frame of the variable to test.
        tcc: A tidy frame of total cloud cover on the same cells and hours.
        threshold: The cloud cover that separates clear from cloudy.

    Returns:
        For `tcc_below`, `tcc_at_or_above`, and `tcc_missing`: the rows, the missing rows, and the
        share (`None` for an empty split).
    """
    key = ["time", "latitude", "longitude"]
    joined = frame.select(*key, "value").join(
        tcc.select(*key, pl.col("value").alias("tcc")), on=key, how="left"
    )
    missing_value = pl.col("value").is_nan() | pl.col("value").is_null()
    split = (
        pl.when(pl.col("tcc").is_null() | pl.col("tcc").is_nan())
        .then(pl.lit("tcc_missing"))
        .when(pl.col("tcc") < threshold)
        .then(pl.lit("tcc_below"))
        .otherwise(pl.lit("tcc_at_or_above"))
    )
    counts = {
        name: (rows, nans)
        for name, rows, nans in joined.group_by(split.alias("split"))
        .agg(pl.len().alias("rows"), missing_value.sum().alias("nans"))
        .iter_rows()
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for name in ("tcc_below", "tcc_at_or_above", "tcc_missing"):
        rows, nans = counts.get(name, (0, 0))
        result[name] = {"rows": rows, "nan_rows": nans, "share": nans / rows if rows else None}
    return result
