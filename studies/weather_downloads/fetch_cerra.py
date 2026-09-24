"""Download CERRA (deterministic reanalysis), cut to the trial-area box, 6 months at a time.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. CERRA has no `area`
request parameter on either dataset this script uses (confirmed against both datasets'
`form.json`, see the `data-download` skill), so every request returns the whole of Europe on
CERRA's own Lambert-conformal grid regardless of what is asked for. The returned NetCDF carries
`latitude`/`longitude` as 2D `(y, x)` fields rather than a regular lat/lon axis, so cropping needs
a boolean mask over those 2D fields (`_cells_in_box`) rather than a simple slice — the same shape
of problem `fetch_icon_dream.py` solves for DWD's unstructured grid, with a regular 2D mask in
place of a cell-index array. The whole-domain NetCDF is downloaded to scratch, masked down to the
trial-area box, and deleted — never kept.

Fetches `data_type=reanalysis` (deterministic) only, never `ensemble_members`.

Two datasets: `reanalysis-cerra-single-levels` for the solar candidate
(`surface_solar_radiation_downwards`,
`time_integrated_surface_direct_short_wave_radiation_flux`) and `reanalysis-cerra-height-levels`
for wind (`wind_speed` at every height in `HEIGHT_LEVELS`). The height band is deliberately
generous — the roster's onshore wind generators span a range of hub heights, and the cropped
output is cheap regardless of how many heights are kept (see the `data-download` skill's sizing
discussion) — so a later study has more than one vertical level to compare, the same choice
`fetch_icon_dream.py` makes for `WS`.

Requests are chunked to six months, matching `studies/beam_diffuse_split/fetch_era5.py`: the
Climate Data Store prices a request by field count (`variable x time x height`) and queues one job
per account at a time, so queue wait rather than data volume sets the wall-clock time. A six-month
chunk stays comfortably under the store's per-request field ceiling; each whole-domain scratch
download is still on the order of 15 GB (measured: about 10 MB per field, roughly 1,460 fields in a
six-month chunk), which is why the scratch directory lives under `data/`, on real disk, rather than
under `/tmp`'s tmpfs — see below.

**Checkpoints one chunk at a time — see the `data-download` skill.** Each chunk is written
atomically to `_chunk_cache/<variable_or_height>/<start>_<end>.parquet` as soon as it is fetched;
a failed or interrupted chunk is retried by re-running the same command, which skips every chunk
already cached.

Run it with `uv run --with cdsapi --with netCDF4 python studies/weather_downloads/fetch_cerra.py
--start-date 2019-09-01 --end-date 2026-06-30`. **Check both datasets' own `constraints.json` (the
`data-download` skill explains the catalogue links) for how far CERRA actually reaches before
passing a later `--end-date`** — a request for a month CDS has not published yet does not fail
loudly at request time, so an end date past real coverage risks a chunk cached as "done" that is
actually short.

Requires a Copernicus Climate Data Store token in `~/.cdsapirc`, and the account must have accepted
the CERRA licence on both the single-levels and height-levels dataset pages.

**The solar variables are `product_type=forecast`, not `analysis`.** CDS has no `analysis` product
for `surface_solar_radiation_downwards` or
`time_integrated_surface_direct_short_wave_radiation_flux` on `reanalysis-cerra-single-levels` —
confirmed against the dataset's own `constraints.json` and its unauthenticated `/costing` endpoint,
which reports `request_is_valid: false` for an `analysis` request of either variable. Both are
fetched instead as a 3-hour forecast accumulation
(`leadtime_hour=["3"]`), which is the shortest lead CERRA serves and yields one value per 3-hour
CERRA analysis time, the same cadence `wind_speed`'s `analysis` product gives. The values are
therefore an energy accumulated over each 3-hour window (J/m²), not an instantaneous flux — this is
recorded in the lineage note, and any later use has to divide by the window length to reach a mean
flux in W/m².
"""

import argparse
import calendar
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import cdsapi  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xarray as xr
from delta_store.nwp import NWP_SIGNIFICAND_BITS
from delta_store.precision import round_to_significand_bits
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box

SINGLE_LEVELS_DATASET: Final[str] = "reanalysis-cerra-single-levels"
HEIGHT_LEVELS_DATASET: Final[str] = "reanalysis-cerra-height-levels"
SOLAR_VARIABLES: Final[tuple[str, ...]] = (
    "surface_solar_radiation_downwards",
    "time_integrated_surface_direct_short_wave_radiation_flux",
)
WIND_VARIABLE: Final[str] = "wind_speed"
HEIGHT_LEVELS: Final[tuple[str, ...]] = ("100_m", "75_m", "50_m", "150_m")
"""A generous band bracketing GB onshore turbine hub heights, not just the issue's original 75/100
m — see the module docstring. Fetched in this order, 100 m first, so the most useful height lands
first."""

SOLAR_LEADTIME_HOURS: Final[int] = 3
"""The single value both `_build_request`'s `leadtime_hour` and `crop_one_chunk`'s
`window_start_offset_hours` must agree on for the solar case — one named constant, so changing the
lead time can't silently desync the request from the boundary-timestamp filter."""

CERRA_TIMES: Final[tuple[str, ...]] = tuple(f"{hour:02d}:00" for hour in range(0, 24, 3))
"""CERRA's own 3-hourly analysis times."""

_SCRATCH_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR.parent.parent / "_scratch" / "cerra"
"""Where a whole-domain NetCDF lands transiently before being cropped and deleted. Under `data/`,
not `/tmp`: `/tmp` on this machine is tmpfs, and a whole-Europe file (~15 GB per six-month chunk,
see the module docstring) belongs on real disk — see the `data-download` skill."""


def _cells_in_box(*, latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the `(y_index, x_index)` pairs whose grid cell falls inside the trial-area box.

    Args:
        latitude: The file's 2D `latitude(y, x)` field.
        longitude: The file's 2D `longitude(y, x)` field.

    Returns:
        `(y_indices, x_indices)`, parallel arrays of integer grid indices, in no particular order.
    """
    box = load_trial_area_box()
    in_box = (
        (latitude >= box.lat_min)
        & (latitude <= box.lat_max)
        & (longitude >= box.lon_min)
        & (longitude <= box.lon_max)
    )
    return np.nonzero(in_box)


def _six_month_chunks(*, start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split `[start_date, end_date]` into Jan-Jun/Jul-Dec half-year chunks, first and last clipped.

    Aligning every chunk to a calendar half-year, rather than to a rolling six months from
    `start_date`, guarantees each chunk sits inside a single year — which is what lets
    `_years_months_days` below build a plain year/month list instead of a year x month cross
    product. A request built from `year=[2019, 2020], month=[09..12, 01, 02]` asks CDS for every
    combination — including 2020-09 through 2020-12 and 2019-01 through 2019-02, months nobody
    wanted — so a chunk spanning two years would silently pull two to four times the intended data.

    Args:
        start_date: First day to cover, `YYYY-MM-DD`.
        end_date: Last day to cover, `YYYY-MM-DD`.

    Returns:
        `(chunk_start, chunk_end)` date strings, in chronological order, each within one year and
        one calendar half.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    chunks: list[tuple[str, str]] = []
    chunk_start = start
    while chunk_start <= end:
        half_end_month = 6 if chunk_start.month <= 6 else 12
        last_day = calendar.monthrange(chunk_start.year, half_end_month)[1]
        chunk_end = min(end, datetime(chunk_start.year, half_end_month, last_day, tzinfo=UTC))
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def _years_months_days(*, start_date: str, end_date: str) -> tuple[list[str], list[str], list[str]]:
    """Return the CDS `year`/`month`/`day` request lists spanning `[start_date, end_date]`.

    Requires `start_date` and `end_date` to fall in the same year — true for every chunk
    `_six_month_chunks` produces, and asserted here so a caller that stops using that function does
    not silently reintroduce the year x month cross-product problem `_six_month_chunks` explains.

    CDS wants explicit year/month/day lists rather than a range, and over-requesting days (e.g. day
    31 in a 30-day month) is accepted and simply returns nothing for the missing dates, so a single
    day list covers every month in the chunk without per-month splitting.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if start.year != end.year:
        raise ValueError(f"chunk spans more than one year: {start_date} to {end_date}")
    years = [str(start.year)]
    months = [f"{m:02d}" for m in range(start.month, end.month + 1)]
    days = [f"{d:02d}" for d in range(1, 32)]
    return years, months, days


def _build_request(
    *, variable: str, height_level: str | None, start_date: str, end_date: str
) -> dict[str, object]:
    """Build one CDS request for a solar (`height_level is None`) or wind chunk."""
    years, months, days = _years_months_days(start_date=start_date, end_date=end_date)
    request: dict[str, object] = {
        "variable": [variable],
        "data_type": ["reanalysis"],
        "year": years,
        "month": months,
        "day": days,
        "time": list(CERRA_TIMES),
        "data_format": "netcdf",
    }
    if height_level is not None:
        # Wind: an analysis product on height_level, no level_type field on this dataset's form.
        request["height_level"] = [height_level]
        request["product_type"] = ["analysis"]
    else:
        # Solar: no analysis product on CDS for either solar variable (see the module docstring),
        # so this is a 3-hour forecast accumulation instead — the shortest lead CERRA serves.
        request["level_type"] = ["surface_or_atmosphere"]
        request["product_type"] = ["forecast"]
        request["leadtime_hour"] = [str(SOLAR_LEADTIME_HOURS)]
    return request


def download_one_chunk(*, dataset: str, request: dict[str, object], scratch_path: Path) -> None:
    """Submit one CDS request and download it to `scratch_path`. Raises on any CDS/network failure.

    Kept separate from cropping so a caller can catch exactly this step's failures (network,
    queueing, an invalid request) and retry later, while a bug in the crop step below is left to
    raise and stop the run — see the `data-download` skill on scoping a resumable catch.
    """
    _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True, progress=False)
    client.retrieve(dataset, request, str(scratch_path))


def crop_one_chunk(
    *,
    scratch_path: Path,
    start_date: str,
    end_date: str,
    unit_column: str,
    window_start_offset_hours: int,
) -> pl.DataFrame:
    """Crop one already-downloaded whole-domain NetCDF to the trial-area box, then delete it.

    Any unexpected structure here (a variable count that is not 1, a crop that matches zero cells,
    a time range that filters down to nothing) raises rather than being treated as a retryable
    download failure — those are bugs in this function's assumptions, not a flaky network, and
    silently caching an empty or wrong result would be worse than stopping the run.

    Args:
        scratch_path: The whole-domain NetCDF `download_one_chunk` just wrote.
        start_date: First day actually requested, `YYYY-MM-DD` (the day list over-requests).
        end_date: Last day actually requested, `YYYY-MM-DD`.
        unit_column: Name for the value column.
        window_start_offset_hours: How many hours before `valid_time` this value's own averaging
            window began — `0` for an `analysis` product, whose `valid_time` marks the instant
            itself, and `3` for the `leadtime_hour=3` solar forecast, whose `valid_time` marks the
            *end* of the 3-hour accumulation window. Filtering on `valid_time` directly for the
            solar case would drop the 00:00 value at every chunk boundary: a request for
            `[start_date, end_date]` also returns the accumulation ending at `end_date + 1` day's
            00:00, which the naive `valid_time < end_date + 1 day` bound excludes even though its
            averaging window falls entirely inside the requested range. Filtering on the window's
            *start* (`valid_time - window_start_offset_hours`) is what the requested date range
            actually means for an accumulated product.

    Returns:
        One row per `(valid_time, y_index, x_index)`, with `y_index`/`x_index` grid indices rather
        than coordinates.
    """
    try:
        with xr.open_dataset(scratch_path) as ds:
            if len(ds.data_vars) != 1:
                raise ValueError(f"expected exactly one data variable, got {list(ds.data_vars)}")
            variable_name = next(iter(ds.data_vars))
            data_array = ds[variable_name]
            # A forecast product's `step` dimension is size 1 here (a single leadtime_hour was
            # requested); squeeze it rather than assume it is absent, so the crop works whether the
            # source was an `analysis` or a `forecast` product.
            if "step" in data_array.dims:
                data_array = data_array.squeeze("step", drop=True)
            valid_time_array = ds["valid_time"]
            if "step" in valid_time_array.dims:
                valid_time_array = valid_time_array.squeeze("step", drop=True)
            # A wind file may carry the single requested height as a size-1 dimension; drop every
            # size-1 dimension except y/x so the positional crop below sees exactly (time, y, x).
            data_array = data_array.squeeze(
                [
                    dim
                    for dim in data_array.dims
                    if dim not in ("y", "x") and data_array.sizes[dim] == 1
                ],
                drop=True,
            )
            if data_array.ndim != 3 or data_array.dims[-2:] != ("y", "x"):
                raise ValueError(f"expected dims (time, y, x), got {data_array.dims}")

            latitude = ds["latitude"].to_numpy()
            # CERRA's longitude convention was not confirmed against a live file in review;
            # normalise to [-180, 180) so a box straddling the prime meridian, or a grid served in
            # [0, 360), matches correctly either way.
            longitude = ((ds["longitude"].to_numpy() + 180) % 360) - 180
            y_indices, x_indices = _cells_in_box(latitude=latitude, longitude=longitude)
            if y_indices.size == 0:
                raise ValueError("trial-area box matched zero CERRA grid cells")
            y0, y1 = int(y_indices.min()), int(y_indices.max())
            x0, x1 = int(x_indices.min()), int(x_indices.max())

            # Select the small bounding window before loading, so the multi-GB whole-domain array
            # is never materialised in full — only the window around the trial-area box is.
            windowed = data_array.isel(y=slice(y0, y1 + 1), x=slice(x0, x1 + 1))
            values = windowed.to_numpy()[:, y_indices - y0, x_indices - x0]
            valid_time = valid_time_array.to_numpy()

            window_start = valid_time - np.timedelta64(window_start_offset_hours, "h")
            in_range = (window_start >= np.datetime64(start_date)) & (
                window_start < np.datetime64(end_date) + np.timedelta64(1, "D")
            )
            if not in_range.any():
                raise ValueError(f"no valid_time fell inside {start_date}..{end_date}")
            values = values[in_range]
            valid_time = valid_time[in_range]
    finally:
        scratch_path.unlink(missing_ok=True)

    n_time, n_cell = values.shape
    frame = pl.DataFrame(
        {
            "valid_time": np.repeat(valid_time, n_cell),
            "y_index": np.tile(y_indices, n_time),
            "x_index": np.tile(x_indices, n_time),
            unit_column: values.reshape(-1),
        }
    ).with_columns(pl.col(unit_column).cast(pl.Float32))
    return frame.with_columns(
        round_to_significand_bits(pl.col(unit_column), keep_bits=NWP_SIGNIFICAND_BITS)
    )


def _run_variable(
    *,
    dataset: str,
    variable: str,
    height_level: str | None,
    unit_column: str,
    chunks: list[tuple[str, str]],
    output_dir: Path,
    window_start_offset_hours: int,
) -> dict[str, object] | None:
    """Fetch every chunk for one (variable, height_level) pair, checkpointed, then combine.

    Args:
        dataset: The CDS dataset name to request from.
        variable: The CDS variable name to request.
        height_level: The wind height level to request (e.g. `"100_m"`), or `None` for a
            surface-level solar variable.
        unit_column: Name for the value column in the combined frame.
        chunks: The `(chunk_start, chunk_end)` date pairs to fetch, one request per chunk.
        output_dir: Where this model's checkpoint cache and combined output live.
        window_start_offset_hours: Passed through to `crop_one_chunk` — see its docstring.

    Returns:
        A summary (`label`, `rows`, `chunks_requested`, `chunks_missing`) for the README's
        completeness claim, or `None` if no chunk succeeded and nothing was written.
    """
    label = f"{variable}_{height_level or 'surface'}"
    chunk_cache_dir = output_dir / "_chunk_cache" / label
    chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start, chunk_end in chunks:
        chunk_path = chunk_cache_dir / f"{chunk_start}_{chunk_end}.parquet"
        if chunk_path.exists():
            print(f"{label} {chunk_start}..{chunk_end}: already cached, skipping")
            continue
        request = _build_request(
            variable=variable, height_level=height_level, start_date=chunk_start, end_date=chunk_end
        )
        scratch_path = _SCRATCH_DIR / f"{label}_{chunk_start}_{chunk_end}.nc"
        try:
            download_one_chunk(dataset=dataset, request=request, scratch_path=scratch_path)
        # Only the download step is caught: a failed or timed-out request must not abandon chunks
        # already cached, and re-running this command retries only the chunks still missing from
        # chunk_cache_dir. A bug in the crop step below (an unexpected variable count, a zero-cell
        # crop) is deliberately left to raise, per the `data-download` skill.
        except Exception as error:  # noqa: BLE001
            scratch_path.unlink(missing_ok=True)  # a half-written download must not linger
            print(f"{label} {chunk_start}..{chunk_end}: download failed ({error!r}), skipping")
            continue
        frame = crop_one_chunk(
            scratch_path=scratch_path,
            start_date=chunk_start,
            end_date=chunk_end,
            unit_column=unit_column,
            window_start_offset_hours=window_start_offset_hours,
        )
        partial = chunk_path.with_suffix(".parquet.partial")
        frame.write_parquet(partial)
        partial.rename(chunk_path)
        print(f"{label} {chunk_start}..{chunk_end}: {frame.height} rows")

    cached_paths = [
        p for start, end in chunks if (p := chunk_cache_dir / f"{start}_{end}.parquet").exists()
    ]
    if not cached_paths:
        print(f"{label}: no chunk succeeded, nothing written")
        return None
    combined = pl.concat([pl.read_parquet(path) for path in cached_paths])
    output_path = output_dir / f"{label}.parquet"
    combined.write_parquet(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"{label}: wrote {combined.height} rows to {output_path}, {size_mb:.3f} MB")

    chunks_cached = [
        f"{s}_{e}" for s, e in chunks if (chunk_cache_dir / f"{s}_{e}.parquet").exists()
    ]
    chunks_requested = [f"{s}_{e}" for s, e in chunks]
    chunks_missing = [chunk for chunk in chunks_requested if chunk not in chunks_cached]

    write_lineage_note(
        product_dir=output_dir,
        filename=f"lineage_{label}.json",
        source_address=f"https://cds.climate.copernicus.eu/datasets/{dataset}",
        request_description=(
            f"CERRA {dataset} {variable}"
            + (f" at height_level={height_level}, product_type=analysis" if height_level else "")
            + (
                " (surface), product_type=forecast, leadtime_hour=3 — values are a 3-hour "
                "accumulation (e.g. J/m2), not an instantaneous flux, because CDS has no "
                "analysis product for this variable"
                if not height_level
                else ""
            )
            + ", data_type=reanalysis (deterministic, never ensemble_members). Whole-domain "
            "NetCDF downloaded in chunks then cropped by a lat/lon mask to the trial-area box "
            "(a few grid cells' margin around the NGED generator roster's own extent); values "
            f"cast to Float32 and rounded to {NWP_SIGNIFICAND_BITS} significand bits before "
            "writing."
        ),
        variables=[variable],
        extra={
            "n_rows_written": combined.height,
            "chunks_requested": chunks_requested,
            "chunks_cached": chunks_cached,
            "chunks_missing": chunks_missing,
        },
    )
    return {
        "label": label,
        "rows": combined.height,
        "chunks_requested": len(chunks_requested),
        "chunks_missing": chunks_missing,
    }


def main() -> int:
    """Fetch every solar and wind (variable, height) pair over the requested date range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    arguments = parser.parse_args()

    chunks = _six_month_chunks(start_date=arguments.start_date, end_date=arguments.end_date)
    output_dir = WEATHER_DOWNLOADS_DIR / "CERRA"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        _run_variable(
            dataset=SINGLE_LEVELS_DATASET,
            variable=variable,
            height_level=None,
            unit_column=f"{variable}_value",
            chunks=chunks,
            output_dir=output_dir,
            # Solar is a forecast accumulation: valid_time marks the END of its window, not its
            # start — see crop_one_chunk's docstring.
            window_start_offset_hours=SOLAR_LEADTIME_HOURS,
        )
        for variable in SOLAR_VARIABLES
    ]
    summaries.extend(
        _run_variable(
            dataset=HEIGHT_LEVELS_DATASET,
            variable=WIND_VARIABLE,
            height_level=height_level,
            unit_column="wind_speed_m_s",
            chunks=chunks,
            output_dir=output_dir,
            # Wind is an analysis product: valid_time is the instant itself.
            window_start_offset_hours=0,
        )
        for height_level in HEIGHT_LEVELS
    )
    written = [summary for summary in summaries if summary is not None]
    missing_chunk_labels = [
        str(summary["label"]) for summary in written if summary["chunks_missing"]
    ]
    if missing_chunk_labels:
        missing_value_convention = (
            "Polars null never appears — a whole chunk that failed to download is simply "
            "absent as rows, not represented as a null value. "
            f"Chunk(s) failed to download for {', '.join(missing_chunk_labels)}; see each "
            "`lineage_<label>.json`'s `chunks_missing` field for exactly which date ranges are "
            "absent from that file, and re-run this script to retry them."
        )
    else:
        missing_value_convention = (
            "Polars null never appears. Every requested six-month chunk downloaded and cropped "
            "successfully for every variable/height — see each `lineage_<label>.json`'s "
            "`n_rows_written` field for the exact row count and `chunks_requested`/"
            "`chunks_cached` to confirm no chunk is missing."
        )
    lineage_filenames = [f"lineage_{summary['label']}.json" for summary in written]

    write_readme(
        product_dir=output_dir,
        product_name="CERRA (Copernicus Regional Reanalysis for Europe)",
        source_web_page="https://confluence.ecmwf.int/display/CKB/CERRA%3A+product+user+guide",
        script_path="studies/weather_downloads/fetch_cerra.py",
        lineage_filenames=lineage_filenames,
        columns={
            "valid_time": "Timezone-naive (implicitly UTC). For the two solar files, marks the "
            "END of a 3-hour forecast accumulation (leadtime_hour=3), NOT the instant itself — "
            "see gotchas. For the wind files (height_level != surface), it is the instant itself "
            "(an analysis product).",
            "y_index": "Row index into CERRA's native Lambert-conformal grid — not a coordinate.",
            "x_index": "Column index into CERRA's native Lambert-conformal grid — not a "
            "coordinate.",
            "surface_solar_radiation_downwards_value": "Global downward shortwave radiation, a "
            "3-hour accumulation in J/m^2 — NOT an instantaneous W/m^2 flux. Divide by the "
            "window length in seconds (10,800) to reach a mean flux. Present only in the "
            "`surface_solar_radiation_downwards_surface.parquet` file.",
            "time_integrated_surface_direct_short_wave_radiation_flux_value": "Direct downward "
            "shortwave radiation, a 3-hour accumulation in J/m^2 — NOT an instantaneous W/m^2 "
            "flux. Divide by the window length in seconds (10,800) to reach a mean flux. "
            "Present only in the "
            "`time_integrated_surface_direct_short_wave_radiation_flux_surface.parquet` file.",
            "wind_speed_m_s": "Wind speed, m/s, an analysis value at each of "
            f"{', '.join(HEIGHT_LEVELS)} above ground (one file per height, named in the "
            "filename and in lineage_wind_speed_<height>.json).",
        },
        missing_value_convention=missing_value_convention,
        gotchas=[
            (
                "Solar variables (`surface_solar_radiation_downwards`, "
                "`time_integrated_surface_direct_short_wave_radiation_flux`) have no CDS "
                "`analysis` product — they are fetched as a `forecast` product at the "
                "shortest lead CDS serves (`leadtime_hour=3`), so `valid_time` marks the end "
                "of a 3-hour accumulation window, not an instant. Wind (`wind_speed`) IS an "
                "analysis product."
            ),
            (
                "Only the deterministic reanalysis is fetched (`data_type=reanalysis`), "
                "never CDS's `ensemble_members` option."
            ),
            (
                "CERRA has no server-side area crop on either dataset used here: every "
                "request returns the whole of Europe, cropped locally to the trial-area box "
                "after download and the whole-domain file deleted — see the module docstring "
                "for why."
            ),
            "Wind heights kept are a generous band ("
            + ", ".join(HEIGHT_LEVELS)
            + ") bracketing GB onshore turbine hub heights, not narrowed to a single height.",
        ],
        external_docs={
            "CERRA single-levels dataset": (
                "https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels"
            ),
            "CERRA height-levels dataset": (
                "https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels"
            ),
            "CERRA product documentation": (
                "https://confluence.ecmwf.int/display/CKB/CERRA%3A+product+user+guide"
            ),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
