"""Download CAMS global reanalysis (EAC4) aerosol optical depth at 550 nm over the public ERA5 box.

One-off throwaway script for the study of which variables explain the sunlight reaching a solar
farm. ERA5 carries no aerosol optical depth, and the CAMS global reanalysis (EAC4) does, so the
study's last rung adds EAC4's total and dust aerosol optical depth at 550 nm.

**EAC4 is 3-hourly (00, 03, ..., 21 UTC) on a 0.75 degree grid, coarser than ERA5's 0.25 degree
grid.** The request box is the smallest box on EAC4's grid lines that encloses the ERA5 box in
`studies.era5_grid.AREA`, so every ERA5 cell sits inside it. `eac4_cells` computes the cells that
box holds, and the written README states the cells the files actually contained.

**The last month is read from the Atmosphere Data Store (ADS) catalogue, not hard-coded.** The
script reads the collection's `constraints` link from
`https://ads.atmosphere.copernicus.eu/api/catalogue/v1/collections/cams-global-reanalysis-eac4`
(the catalogue is public and needs no key) and takes the latest year and month that any constraint
lists. `parse_latest_month` also accepts date ranges. If the constraints list no date, the script
falls back to the `maxEnd` of the date-range widget in the collection's `form.json`. The EAC4
archive ends some months behind real time, so the end month moves as the store is extended.

**The request asks for `data_format` `netcdf_zip`.** The EAC4 form offers only `grib` and
`netcdf_zip` (checked against the live `form.json`) and has no `download_format`, so each
download is a zip holding netCDF files, which the same reader handles as the ERA5 archives.

**Requests are one calendar year each, about 7 requests and a few megabytes.** A request for the
two variables over one year asks for 2 x 8 x 366 = 5,856 variable-times. Each chunk is written to
`_chunks/` as soon as it arrives, and a re-run skips every chunk that is already there and valid.

Run it with `uv run --with cdsapi --with netCDF4 python
studies/weather_downloads/fetch_cams_eac4_aod.py`, adding `--dry-run` to list the requests without
contacting the store (it then plans the first year only, because the end month needs the catalogue,
unless `--end-month YYYY-MM` or `--constraints-file PATH` supplies it). The key is read from
`~/.cdsapirc`, as `fetch_cams.py` does, and is never logged or written.
"""

import argparse
import calendar
import json
import logging
import math
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import polars as pl
from era5_cells import Chunk
from era5_solar_variables import (
    COST_FIELDS_PER_VARIABLE_HOUR,
    FIRST_HOUR,
    LAST_HOUR,
    ChunkRecord,
    read_archive,
    retrieve_with_cleanup,
    run_chunks,
)
from lineage import write_lineage_note, write_readme
from studies.era5_grid import AREA
from studies.sources import CAMS_EAC4_AOD_PRODUCT_DIR, SCRATCH_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_cams_eac4_aod")

SCRIPT_PATH: Final[str] = "studies/weather_downloads/fetch_cams_eac4_aod.py"
ADS_URL: Final[str] = "https://ads.atmosphere.copernicus.eu/api"
DATASET: Final[str] = "cams-global-reanalysis-eac4"
COLLECTION_URL: Final[str] = f"{ADS_URL}/catalogue/v1/collections/{DATASET}"
SOURCE_PAGE: Final[str] = f"https://ads.atmosphere.copernicus.eu/datasets/{DATASET}"
OUTPUT_DIR: Final[Path] = CAMS_EAC4_AOD_PRODUCT_DIR
CHUNK_DIR: Final[Path] = OUTPUT_DIR / "_chunks"
OUTPUT_PATH: Final[Path] = OUTPUT_DIR / "eac4_aod.parquet"

GRID_STEP_DEG: Final[float] = 0.75
"""EAC4's grid spacing in latitude and longitude."""

VARIABLES: Final[dict[str, str]] = {
    "aod550": "total_aerosol_optical_depth_550nm",
    "duaod550": "dust_aerosol_optical_depth_550nm",
}
"""The netCDF variable name of each field against the name the request form takes."""

TIMES: Final[tuple[str, ...]] = tuple(f"{hour:02d}:00" for hour in range(0, 24, 3))
"""The eight analysis times of day, UTC."""


def eac4_area() -> tuple[float, float, float, float]:
    """Return the smallest box on EAC4's grid lines that encloses the ERA5 box.

    Returns:
        The box as (north, west, south, east) in degrees, from the public `AREA`.
    """
    north, west, south, east = AREA
    return (
        math.ceil(north / GRID_STEP_DEG) * GRID_STEP_DEG,
        math.floor(west / GRID_STEP_DEG) * GRID_STEP_DEG,
        math.floor(south / GRID_STEP_DEG) * GRID_STEP_DEG,
        math.ceil(east / GRID_STEP_DEG) * GRID_STEP_DEG,
    )


def eac4_cells() -> list[tuple[float, float]]:
    """Return the (latitude, longitude) of every EAC4 cell inside `eac4_area`, north to south."""
    north, west, south, east = eac4_area()
    n_lat = round((north - south) / GRID_STEP_DEG)
    n_lon = round((east - west) / GRID_STEP_DEG)
    return [
        (round(north - i * GRID_STEP_DEG, 4), round(west + j * GRID_STEP_DEG, 4))
        for i in range(n_lat + 1)
        for j in range(n_lon + 1)
    ]


def _dates_in(*, value: object) -> list[date]:
    """Return the end dates in a constraint's `date` entry: `A/B` ranges or single dates."""
    items = value if isinstance(value, list) else [value]
    dates = []
    for item in items:
        text = str(item).split("/")[-1].strip()
        digits = text.replace("-", "")
        if len(digits) == 8 and digits.isdigit():
            dates.append(date(int(digits[:4]), int(digits[4:6]), int(digits[6:])))
    return dates


def parse_latest_month(*, constraints: object) -> tuple[int, int]:
    """Return the latest `(year, month)` that the dataset's constraints list.

    The Climate and Atmosphere Data Stores publish constraints as a list of entries. Each entry
    lists allowed values by field, and the allowed combinations are the cross product of one
    entry's values. An entry therefore ends at its largest year and, within it, its largest month.
    An entry with a `date` field ends at the end of its last range.

    Args:
        constraints: The decoded `constraints.json`: a list of entries, or one entry.

    Returns:
        The latest year and month over all entries.

    Raises:
        ValueError: If no entry lists a year and month or a date.
    """
    entries = constraints if isinstance(constraints, list) else [constraints]
    latest: list[tuple[int, int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        years = [int(y) for y in entry.get("year", [])]
        months = [int(m) for m in entry.get("month", [])]
        if years and months:
            latest.append((max(years), max(months)))
        latest.extend((d.year, d.month) for d in _dates_in(value=entry.get("date", [])))
    if not latest:
        msg = "the constraints list no year and month and no date"
        raise ValueError(msg)
    return max(latest)


@dataclass(frozen=True)
class Eac4Chunk:
    """One request: both aerosol fields over the months of one calendar year.

    Attributes:
        period: The year and the first and last month requested.
    """

    period: Chunk

    @property
    def chunk_id(self) -> str:
        """Return a filename-safe id such as `eac4_aod_2019_09_12`."""
        return f"eac4_aod_{self.period.name}"

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the two netCDF variable names."""
        return tuple(VARIABLES)

    @property
    def n_days(self) -> int:
        """Return how many days the request covers."""
        return self.period.n_hours // 24

    @property
    def variable_hours(self) -> int:
        """Return the variable-times asked for: variables x times of day x days."""
        return len(VARIABLES) * len(TIMES) * self.n_days

    @property
    def cost_fields(self) -> int:
        """Return a conservative store field count, using ERA5's 6 per variable-time."""
        return self.variable_hours * COST_FIELDS_PER_VARIABLE_HOUR


def plan_chunks(*, first_month: tuple[int, int], last_month: tuple[int, int]) -> list[Eac4Chunk]:
    """Return one chunk per calendar year from `first_month` to `last_month` inclusive."""
    chunks = []
    for year in range(first_month[0], last_month[0] + 1):
        first = first_month[1] if year == first_month[0] else 1
        last = last_month[1] if year == last_month[0] else 12
        chunks.append(Eac4Chunk(period=Chunk(year=year, first_month=first, last_month=last)))
    return chunks


def request_body(*, chunk: Eac4Chunk) -> dict[str, object]:
    """Return the ADS request body for one chunk, over the enclosing EAC4 box."""
    period = chunk.period
    last_day = calendar.monthrange(period.year, period.last_month)[1]
    return {
        "variable": list(VARIABLES.values()),
        "date": [
            (
                f"{period.year}-{period.first_month:02d}-01/"
                f"{period.year}-{period.last_month:02d}-{last_day:02d}"
            )
        ],
        "time": list(TIMES),
        "data_format": "netcdf_zip",
        "area": list(eac4_area()),
    }


def parse_form_end(*, form: object) -> tuple[int, int]:
    """Return the `(year, month)` of the date-range widget's `maxEnd` in a `form.json`.

    Args:
        form: The decoded `form.json`: a list of widgets.

    Returns:
        The year and month of the latest date the store offers.

    Raises:
        ValueError: If no widget carries a `maxEnd` date.
    """
    widgets = form if isinstance(form, list) else []
    for widget in widgets:
        details = widget.get("details", {}) if isinstance(widget, dict) else {}
        ends = _dates_in(value=details.get("maxEnd", []))
        if ends:
            return (ends[0].year, ends[0].month)
    msg = "the form lists no `maxEnd` date"
    raise ValueError(msg)


def read_end_month(*, constraints_file: Path | None) -> tuple[tuple[int, int], str]:
    """Return the latest month the store lists, and where it was read from.

    Args:
        constraints_file: A saved `constraints.json` to read instead of the store, or `None` to read
            the public catalogue.

    Returns:
        The `(year, month)` and a one-line account of the source.
    """
    if constraints_file is not None:
        constraints = json.loads(constraints_file.read_text())
        return parse_latest_month(constraints=constraints), f"file {constraints_file.name}"
    import requests

    collection = requests.get(COLLECTION_URL, timeout=60)
    collection.raise_for_status()
    links = {link.get("rel"): link.get("href") for link in collection.json().get("links", [])}
    href = links.get("constraints") or f"{COLLECTION_URL}/constraints.json"
    response = requests.get(href, timeout=60)
    response.raise_for_status()
    try:
        return parse_latest_month(constraints=response.json()), f"catalogue link {href}"
    except ValueError:
        # The constraints may not list dates; the form's date-range widget always states its end.
        form = requests.get(f"{COLLECTION_URL}/form.json", timeout=60)
        form.raise_for_status()
        return parse_form_end(form=form.json()), f"{COLLECTION_URL}/form.json (maxEnd)"


def _download(chunk: Eac4Chunk, destination: Path) -> None:
    """Submit one request to the ADS and write the zipped netCDF to `destination`.

    The remote job is deleted if the wait is interrupted or fails; see `retrieve_with_cleanup`.
    """
    from ecmwf.datastores import Client  # ty: ignore[unresolved-import]  # `--with cdsapi` only

    retrieve_with_cleanup(
        client=Client(url=ADS_URL, key=_api_key(), progress=False),
        collection=DATASET,
        request=request_body(chunk=chunk),
        destination=destination,
        log=_LOG.info,
    )


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    """Turn `SIGTERM` into `KeyboardInterrupt`, so the running job is deleted on the way out."""
    raise KeyboardInterrupt


def _api_key() -> str:
    """Read the Copernicus API key from `~/.cdsapirc`, which both data stores accept.

    Returns:
        The key. It must never be logged or written.

    Raises:
        RuntimeError: If the file holds no `key:` line.
    """
    for line in Path.home().joinpath(".cdsapirc").read_text().splitlines():
        if line.startswith("key:"):
            return line.split(":", 1)[1].strip()
    msg = "no key found in ~/.cdsapirc"
    raise RuntimeError(msg)


def assemble(*, chunks: Sequence[Eac4Chunk]) -> pl.DataFrame:
    """Join every chunk into one wide table, one row per (time, cell).

    Args:
        chunks: The chunks, all on disk.

    Returns:
        `time`, `latitude`, `longitude`, and one `Float32` column per variable, sorted.

    Raises:
        ValueError: If a chunk lacks a variable, a (time, cell) repeats, or the row count differs
            from cells x 8 times of day x days.
    """
    parts = []
    for chunk in chunks:
        archive = read_archive(
            path=CHUNK_DIR / f"{chunk.chunk_id}.zip",
            scratch_dir=SCRATCH_DIR / "cams_eac4_aod",
            has_expver=False,
        )
        missing = set(VARIABLES) - set(archive.frames)
        if missing:
            msg = f"{chunk.chunk_id}: the archive lacks {sorted(missing)}"
            raise ValueError(msg)
        key = ["time", "latitude", "longitude"]
        wide = archive.frames["aod550"].rename({"value": "aod550"})
        wide = wide.join(
            archive.frames["duaod550"].rename({"value": "duaod550"}),
            on=key,
            how="full",
            coalesce=True,
        )
        parts.append(wide)
    frame = pl.concat(parts).sort("time", "latitude", "longitude")
    duplicates = frame.height - frame.select("time", "latitude", "longitude").unique().height
    n_cells = frame.select("latitude", "longitude").unique().height
    wanted = n_cells * len(TIMES) * sum(c.n_days for c in chunks)
    if duplicates or frame.height != wanted:
        msg = f"{frame.height} rows, {duplicates} duplicates, expected {wanted}"
        raise ValueError(msg)
    return frame


def write_documentation(
    *, frame: pl.DataFrame, end_month: tuple[int, int], source: str, log: Sequence[ChunkRecord]
) -> None:
    """Write `lineage.json` and `README.md` from the measured frame."""
    cells = sorted(frame.select("latitude", "longitude").unique().iter_rows(), reverse=True)
    expected_cells = sorted(eac4_cells(), reverse=True)
    nan_counts = {c: int(frame[c].is_nan().sum()) for c in VARIABLES}
    write_lineage_note(
        product_dir=OUTPUT_DIR,
        source_address=ADS_URL,
        request_description=(
            f"CAMS global reanalysis (EAC4), `{DATASET}`, total and dust aerosol optical depth at "
            f"550 nm at {', '.join(TIMES)} UTC, netCDF in zip, one request per calendar year, "
            "over the smallest box on the 0.75 degree grid that encloses the public ERA5 box."
        ),
        variables=list(VARIABLES),
        extra={
            "first_month": f"{FIRST_HOUR.year}-{FIRST_HOUR.month:02d}",
            "last_month": f"{end_month[0]}-{end_month[1]:02d}",
            "last_month_read_from": source,
            "rows": frame.height,
            "cells_returned": cells,
            "cells_expected": expected_cells,
            "chunks": [record._asdict() for record in log],
        },
    )
    write_readme(
        product_dir=OUTPUT_DIR,
        product_name="CAMS global reanalysis (EAC4) aerosol optical depth at 550 nm",
        source_web_page=SOURCE_PAGE,
        script_path=SCRIPT_PATH,
        lineage_filenames=["lineage.json"],
        columns={
            "time": f"Valid time, UTC, {frame.schema['time']}; 3-hourly, 00, 03, ..., 21 UTC.",
            "latitude": f"Cell latitude, degrees north, {frame.schema['latitude']}.",
            "longitude": f"Cell longitude, degrees east, {frame.schema['longitude']}.",
            "aod550": f"Total aerosol optical depth at 550 nm, dimensionless, "
            f"{frame.schema['aod550']}.",
            "duaod550": f"Dust aerosol optical depth at 550 nm, dimensionless, "
            f"{frame.schema['duaod550']}.",
        },
        missing_value_convention=(
            f"No missing values expected. NaN counts in the written table: {nan_counts}."
        ),
        gotchas=[
            (
                f"The grid is 0.75 degrees, so the box holds {len(cells)} cells, where the ERA5 "
                f"box holds 20 cells at 0.25 degrees. The cells in the file are "
                f"{'the expected set' if cells == expected_cells else 'NOT the expected set'}."
            ),
            (
                f"The last month, {end_month[0]}-{end_month[1]:02d}, was read from the store's "
                f"constraints ({source}) when the script ran, and moves as the store is extended."
            ),
            (
                f"The EAC4 listing ends {end_month[0]}-{end_month[1]:02d}, which is "
                f"{(LAST_HOUR.year - end_month[0]) * 12 + LAST_HOUR.month - end_month[1]} months "
                f"before the ERA5 span ends ({LAST_HOUR:%Y-%m}); the study's aerosol rung can use "
                "only the months both cover."
            ),
            (
                "The values are 3-hourly stamps, so joining them to an hour-ending ERA5 stamp "
                "needs interpolation in time."
            ),
        ],
        external_docs={
            "CAMS global reanalysis (EAC4)": SOURCE_PAGE,
            "Inness et al. (2019)": "https://doi.org/10.5194/acp-19-3515-2019",
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, fetch, and assemble the EAC4 aerosol table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list the plan; contact nothing")
    parser.add_argument("--end-month", help="YYYY-MM, instead of reading the catalogue")
    parser.add_argument("--constraints-file", type=Path, help="a saved constraints.json")
    args = parser.parse_args(argv)

    first_month = (FIRST_HOUR.year, FIRST_HOUR.month)
    if args.end_month:
        year, month = args.end_month.split("-")
        end_month, source = (int(year), int(month)), "the --end-month argument"
    elif args.constraints_file or not args.dry_run:
        end_month, source = read_end_month(constraints_file=args.constraints_file)
    else:
        print(
            f"end month would be read from {COLLECTION_URL} (constraints link); planning 2019 only"
        )
        end_month, source = (first_month[0], 12), "dry run, not read"
    chunks = plan_chunks(first_month=first_month, last_month=end_month)
    print(f"EAC4 box (N, W, S, E): {eac4_area()}; {len(eac4_cells())} cells")
    for chunk in chunks:
        print(
            f"{chunk.chunk_id}: {chunk.n_days} days, {chunk.variable_hours} variable-times "
            f"({chunk.cost_fields} fields at the ERA5 rate of {COST_FIELDS_PER_VARIABLE_HOUR})"
        )
    if args.dry_run:
        print("first request body:")
        print(json.dumps(request_body(chunk=chunks[0])))
        return 0

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    records = run_chunks(chunks=chunks, chunk_dir=CHUNK_DIR, download=_download, log=_LOG.info)
    frame = assemble(chunks=chunks)
    partial = OUTPUT_PATH.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(OUTPUT_PATH)
    write_documentation(frame=frame, end_month=end_month, source=source, log=records)
    _LOG.info("wrote %d rows to %s at %s", frame.height, OUTPUT_PATH.name, datetime.now(UTC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
