r"""Download NORA3 hourly wind speed and direction at 50 m and 100 m, cut to the trial-area box.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. MET Norway serves NORA3
from its THREDDS server over OPeNDAP, on its own 3 km Lambert-conformal grid
(`nora3_subset_atmos/wind_hourly_v2_agg/nora3_wind_hourly.ncml`). The cut to the trial-area box
happens server-side: each request slices the grid's own `x` and `y` index ranges, so no
whole-domain file is downloaded. `paths.TrialAreaBox`'s latitude and longitude bounds are converted
once, locally, to a grid index range (`_box_index_range`). The bounds, and the index range derived
from them, are as private as the box itself: they go to the OPeNDAP call and nowhere else. No log
line, error message, lineage note, or README carries them.

The script fetches `wind_speed` and `wind_direction` at 50 m and 100 m (`height` indices 2 and 3 of
the served `[10, 20, 50, 100, 250, 500, 750]`, checked against the served `height` array at run
time). It requires `pydap`, which is not a workspace dependency, so run it as

```bash
uv run --with pydap python -u studies/weather_downloads/fetch_nora3.py \
    --start-month 2015-01 --end-month 2025-01
```

The script fetches and checkpoints one whole calendar month at a time (see the `data-download`
skill). It writes each month to `_month_cache/YYYY-MM.parquet` as soon as the month lands and skips
cached months on the next run. Before a month is written, the script reads the served `time` values
for that month and asserts that they equal the requested hours exactly, so a time-axis offset or gap
cannot mislabel rows. A month that extends past the last hour the aggregated dataset serves is
skipped, because the aggregate ends there; MET Norway publishes later months as individual monthly
files under `nora3_subset_atmos/wind_hourly_v2/`, which this script does not fetch. A run whose
range passes the aggregate's end prints a warning and exits 1 after writing everything it could.
Every other failure aborts the run. The lineage note lists the months combined and the months
beyond the aggregate. The aggregate's last hour is recorded in `lineage.json` as
`last_served_hour_utc`.

The combined output is one canonical file, `NORA3_wind.parquet`, rebuilt from every month in the
month cache on every run, whatever range the run requested. Validate it with `validate_nora3.py`.
"""

import argparse
import calendar
import sys
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from delta_store.nwp import NWP_SIGNIFICAND_BITS
from delta_store.precision import round_to_significand_bits
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box
from pydap.client import open_url  # ty: ignore[unresolved-import]
from pyproj import Transformer

CATALOG_URL: Final[str] = (
    "https://thredds.met.no/thredds/dodsC/nora3_subset_atmos/wind_hourly_v2_agg/nora3_wind_hourly.ncml"
)
LAMBERT_PROJ4: Final[str] = (
    "+proj=lcc +lat_1=66.3 +lat_2=66.3 +lat_0=66.3 +lon_0=-42.0 +R=6371000 +units=m +no_defs"
)
"""From the dataset's own `projection_lambert` attributes (`standard_parallel=66.3`,
`longitude_of_central_meridian=-42.0`, `latitude_of_projection_origin=66.3`,
`earth_radius=6371000`), read via the catalog's `.das` metadata."""

GRID_X0_M: Final[float] = 778360.9
GRID_Y0_M: Final[float] = -1270477.0
GRID_SPACING_M: Final[float] = 3000.0
"""The grid's first `x` and `y` coordinate and spacing, read once from the catalog's own `x` and
`y` arrays. `_open_dataset` asserts the served axes still match them."""

HEIGHT_INDICES: Final[tuple[int, int]] = (2, 3)
"""Indices into the served `height` dimension for 50 m and 100 m."""

HEIGHTS_M: Final[tuple[int, int]] = (50, 100)
"""The heights, in metres, that `HEIGHT_INDICES` must select."""

EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)
"""`time`'s units are seconds since this epoch."""

FILL_VALUE: Final[int] = -32767
"""The `_FillValue` the catalog's `.das` gives for both variables."""

SPEED_SCALE_FACTOR: Final[float] = 0.01
DIRECTION_SCALE_FACTOR: Final[float] = 0.1
"""The `.das` scale factors (`add_offset=0.0` for both). The two variables do not share one."""

ADD_OFFSET: Final[float] = 0.0
"""The `.das` `add_offset` for both variables."""

OUTPUT_NAME: Final[str] = "NORA3_wind.parquet"
"""The one canonical combined file, so runs over different date ranges never leave overlapping
files behind."""


def _open_dataset() -> object:
    """Open the OPeNDAP dataset and assert its grid axes and heights match this script's constants.

    Returns:
        The opened `pydap` dataset.

    Raises:
        RuntimeError: If the dataset's `x`, `y`, or `height` axis, or a variable's `scale_factor`,
            `add_offset`, or `_FillValue`, differs from the constants above.
    """
    dataset = open_url(CATALOG_URL)
    x_first = np.asarray(dataset["x"][0:2].data)
    y_first = np.asarray(dataset["y"][0:2].data)
    if not (
        np.allclose(x_first, [GRID_X0_M, GRID_X0_M + GRID_SPACING_M], atol=0.5)
        and np.allclose(y_first, [GRID_Y0_M, GRID_Y0_M + GRID_SPACING_M], atol=0.5)
    ):
        raise RuntimeError("The served x/y axes no longer match GRID_X0_M/GRID_Y0_M/GRID_SPACING_M")
    for name, scale_factor in (
        ("wind_speed", SPEED_SCALE_FACTOR),
        ("wind_direction", DIRECTION_SCALE_FACTOR),
    ):
        attributes = dataset[name].attributes
        if (
            attributes.get("scale_factor") != scale_factor
            or attributes.get("add_offset") != ADD_OFFSET
            or attributes.get("_FillValue") != FILL_VALUE
        ):
            raise RuntimeError(
                f"{name}'s served scale_factor, add_offset, or _FillValue differs from the "
                "constants in this script"
            )
    heights = np.asarray(dataset["height"][:].data)[list(HEIGHT_INDICES)]
    if not np.array_equal(heights, HEIGHTS_M):
        raise RuntimeError(f"HEIGHT_INDICES select {heights.tolist()} m, expected {HEIGHTS_M}")
    return dataset


def _box_index_range(*, n_x: int, n_y: int) -> tuple[int, int, int, int]:
    """Convert the trial-area box to the grid's own `x` and `y` index range.

    The returned indices are as private as the box: the caller may pass them to the OPeNDAP call
    and must never log or print them.

    Args:
        n_x: Size of the served `x` axis.
        n_y: Size of the served `y` axis.

    Returns:
        `(ix0, ix1, iy0, iy1)`, inclusive grid index bounds, one cell wider than the box each side.

    Raises:
        ValueError: If any bound lies outside the grid. The message names no index.
    """
    box = load_trial_area_box()
    transformer = Transformer.from_crs("EPSG:4326", LAMBERT_PROJ4, always_xy=True)
    corners = [
        (box.lon_min, box.lat_min),
        (box.lon_min, box.lat_max),
        (box.lon_max, box.lat_min),
        (box.lon_max, box.lat_max),
    ]
    xs, ys = zip(*(transformer.transform(lon, lat) for lon, lat in corners), strict=True)
    ix0 = int((min(xs) - GRID_X0_M) / GRID_SPACING_M) - 1
    ix1 = int((max(xs) - GRID_X0_M) / GRID_SPACING_M) + 1
    iy0 = int((min(ys) - GRID_Y0_M) / GRID_SPACING_M) - 1
    iy1 = int((max(ys) - GRID_Y0_M) / GRID_SPACING_M) + 1
    if not (0 <= ix0 <= ix1 < n_x and 0 <= iy0 <= iy1 < n_y):
        raise ValueError("The trial-area box's grid index range falls outside the NORA3 grid")
    return ix0, ix1, iy0, iy1


def _month_hours(*, year: int, month: int) -> np.ndarray:
    """Return the epoch-second timestamps of every hour in a calendar month (UTC).

    Args:
        year: Calendar year.
        month: Calendar month, 1 to 12.

    Returns:
        One `float64` per hour, spaced 3600 s apart, starting at the month's first hour.
    """
    first = int((datetime(year, month, 1, tzinfo=UTC) - EPOCH).total_seconds())
    n_hours = calendar.monthrange(year, month)[1] * 24
    return first + 3600.0 * np.arange(n_hours)


def _months(*, start_month: str, end_month: str) -> list[tuple[int, int]]:
    """Return every `(year, month)` from `start_month` to `end_month` inclusive.

    Args:
        start_month: First month, as `YYYY-MM`.
        end_month: Last month, as `YYYY-MM`.

    Returns:
        The months in chronological order.
    """
    start = datetime.strptime(start_month, "%Y-%m").replace(tzinfo=UTC)
    end = datetime.strptime(end_month, "%Y-%m").replace(tzinfo=UTC)
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _read_scaled(*, dataset: object, name: str, slices: tuple, scale_factor: float) -> np.ndarray:
    """Read one variable slice, mask the fill value to NaN, and apply the scale factor.

    Args:
        dataset: The opened `pydap` dataset.
        name: Variable name, `wind_speed` or `wind_direction`.
        slices: The `(time, height, y, x)` slices to read.
        scale_factor: The variable's `scale_factor`.

    Returns:
        A `float32` array shaped `(time, height, y, x)`.
    """
    raw = np.asarray(dataset[name].array[slices])  # ty: ignore[not-subscriptable]
    masked = np.where(raw == FILL_VALUE, np.nan, raw)
    return masked.astype(np.float32) * np.float32(scale_factor)


def fetch_month(
    *, dataset: object, year: int, month: int, index_range: tuple[int, int, int, int]
) -> pl.DataFrame:
    """Fetch wind speed and direction at 50 m and 100 m for one calendar month.

    Args:
        dataset: The opened `pydap` dataset.
        year: Calendar year.
        month: Calendar month, 1 to 12.
        index_range: `(ix0, ix1, iy0, iy1)` from `_box_index_range`.

    Returns:
        One row per (time, height_m, y_index, x_index). `y_index` and `x_index` are the grid's own
        integer indices, not coordinates.

    Raises:
        RuntimeError: If the served `time` values differ from the requested hours, or if the
            OPeNDAP request fails. The message names no index or URL.
    """
    ix0, ix1, iy0, iy1 = index_range
    h0, h1 = HEIGHT_INDICES
    hours = _month_hours(year=year, month=month)
    t0 = int(hours[0] // 3600)
    t1 = t0 + len(hours)
    try:
        served_time = np.asarray(dataset["time"][t0:t1].data)  # ty: ignore[not-subscriptable]
        slices = (slice(t0, t1), slice(h0, h1 + 1), slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))
        speed = _read_scaled(
            dataset=dataset, name="wind_speed", slices=slices, scale_factor=SPEED_SCALE_FACTOR
        )
        direction = _read_scaled(
            dataset=dataset,
            name="wind_direction",
            slices=slices,
            scale_factor=DIRECTION_SCALE_FACTOR,
        )
    except Exception as error:  # noqa: BLE001
        # An OPeNDAP or HTTP error commonly embeds the request URL, which carries the index range.
        # Raise a new error naming only the month and the error type, and drop the original chain.
        raise RuntimeError(
            f"NORA3 {year}-{month:02d}: request failed ({type(error).__name__}, details withheld)"
        ) from None
    if not np.array_equal(served_time, hours):
        raise RuntimeError(f"NORA3 {year}-{month:02d}: served time axis differs from the request")

    n_time, n_height, n_y, n_x = speed.shape
    height_grid, y_grid, x_grid = np.meshgrid(
        np.array(HEIGHTS_M), np.arange(iy0, iy1 + 1), np.arange(ix0, ix1 + 1), indexing="ij"
    )
    time_column = np.repeat(
        (hours * 1000).astype("int64").astype("datetime64[ms]"), n_height * n_y * n_x
    )
    frame = pl.DataFrame(
        {
            "time": time_column,
            "height_m": np.tile(height_grid.ravel(), n_time).astype(np.int16),
            "y_index": np.tile(y_grid.ravel(), n_time).astype(np.int16),
            "x_index": np.tile(x_grid.ravel(), n_time).astype(np.int16),
            "wind_speed_m_s": speed.ravel(),
            "wind_direction_deg": direction.ravel(),
        }
    )
    return frame.with_columns(
        round_to_significand_bits(pl.col("wind_speed_m_s"), keep_bits=NWP_SIGNIFICAND_BITS),
        round_to_significand_bits(pl.col("wind_direction_deg"), keep_bits=NWP_SIGNIFICAND_BITS),
    )


def main() -> int:
    """Fetch the requested months, one at a time, and write the combined file and lineage note.

    Returns:
        The process exit code: 0 on success.
    """
    parser = argparse.ArgumentParser(description="Download NORA3 wind over the trial-area box.")
    parser.add_argument("--start-month", required=True, help="First month, YYYY-MM.")
    parser.add_argument("--end-month", required=True, help="Last month, YYYY-MM.")
    arguments = parser.parse_args()

    output_dir = WEATHER_DOWNLOADS_DIR / "NORA3"
    month_cache_dir = output_dir / "_month_cache"
    month_cache_dir.mkdir(parents=True, exist_ok=True)
    months = _months(start_month=arguments.start_month, end_month=arguments.end_month)

    dataset = _open_dataset()
    n_time = dataset["time"].shape[0]  # ty: ignore[not-subscriptable]
    last_served_hour = float(np.asarray(dataset["time"][n_time - 1 : n_time].data)[0])  # ty: ignore[not-subscriptable]
    index_range = _box_index_range(
        n_x=dataset["x"].shape[0],  # ty: ignore[not-subscriptable]
        n_y=dataset["y"].shape[0],  # ty: ignore[not-subscriptable]
    )

    beyond_aggregate: list[str] = []
    for year, month in months:
        label = f"{year}-{month:02d}"
        month_path = month_cache_dir / f"{label}.parquet"
        if month_path.exists():
            print(f"NORA3 {label}: already cached, skipping")
            continue
        if _month_hours(year=year, month=month)[-1] > last_served_hour:
            print(f"NORA3 {label}: beyond the end of the aggregated dataset, skipping")
            beyond_aggregate.append(label)
            continue
        frame = fetch_month(dataset=dataset, year=year, month=month, index_range=index_range)
        partial = month_path.with_suffix(".parquet.partial")
        frame.write_parquet(partial)
        partial.rename(month_path)
        print(f"NORA3 {label}: fetched and cached")

    labels = sorted(path.stem for path in month_cache_dir.glob("*.parquet"))
    if not labels:
        print("NORA3: no month is cached, nothing written")
        return 1
    combined = pl.concat(
        [pl.read_parquet(month_cache_dir / f"{label}.parquet") for label in labels]
    )
    output_path = output_dir / OUTPUT_NAME
    combined.write_parquet(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"NORA3: wrote the combined file from {len(labels)} cached months to {output_path}")

    write_lineage_note(
        product_dir=output_dir,
        source_address=CATALOG_URL,
        request_description=(
            "NORA3 hourly wind_speed and wind_direction at 50 m and 100 m, OPeNDAP index-range "
            "slice of the grid's own x/y axes to the trial-area box (one grid cell's margin "
            "around the NGED generator roster's own extent), whole calendar months"
        ),
        variables=["wind_speed", "wind_direction"],
        extra={
            "months_requested": [arguments.start_month, arguments.end_month],
            "months_combined": labels,
            "months_beyond_aggregate": beyond_aggregate,
            "last_served_hour_utc": datetime.fromtimestamp(last_served_hour, tz=UTC),
            "heights_m": list(HEIGHTS_M),
            "rows": combined.height,
            "size_mb": round(size_mb, 3),
        },
    )
    write_readme(
        product_dir=output_dir,
        product_name="NORA3 (NORwegian hindcast Archive, 3 km) hourly wind",
        source_web_page="https://thredds.met.no/thredds/catalog/nora3_subset_atmos/catalog.html",
        script_path="studies/weather_downloads/fetch_nora3.py",
        lineage_filenames=["lineage.json"],
        columns={
            "time": "Timezone-naive (implicitly UTC) hourly timestamp of the instantaneous value.",
            "height_m": "Height above ground, metres: 50 or 100.",
            "y_index": "Row index into NORA3's native 3 km Lambert-conformal grid, not a "
            "coordinate.",
            "x_index": "Column index into the same grid, not a coordinate.",
            "wind_speed_m_s": "Wind speed, m/s, rounded to a 13-bit significand.",
            "wind_direction_deg": "Direction the wind blows from, degrees clockwise from north, "
            "rounded to a 13-bit significand.",
        },
        missing_value_convention="A served fill value (-32767) becomes NaN. Rows are never "
        "dropped, so a missing reading is a NaN in the value column.",
        gotchas=[
            "The grid is projected: `x_index` and `y_index` are not longitude and latitude.",
            "Wind direction is circular: average vectors, not degrees.",
            (
                "The aggregated dataset ends at `last_served_hour_utc` in `lineage.json`. Months "
                "after it are not fetched and are listed under `months_beyond_aggregate`."
            ),
        ],
        external_docs={
            "NORA3 wind dataset paper": "https://doi.org/10.1175/JAMC-D-21-0029.1",
        },
    )
    if beyond_aggregate:
        print(
            f"NORA3: WARNING: {len(beyond_aggregate)} requested month(s) lie beyond the end of "
            f"the aggregated dataset ({beyond_aggregate[0]} onwards) and were NOT fetched. "
            "Exiting non-zero."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
