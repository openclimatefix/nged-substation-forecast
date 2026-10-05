"""Download native ERA5 10 m and 100 m wind for 2019-09 to 2023-12 from the Climate Data Store.

One-off throwaway script for the downloads in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. It extends the 2024 to
2026 wind in `data/studies/weather/ERA5/wind_native_cds.parquet` (made by `fetch_era5_wind.py`) back
to the start of the metered generators' history, from the same product by the same route.

**Three groups of cells are kept.** The trial-area box (every 0.25 degree cell inside it), a 3 x 3
block of cells around each MIDAS Open station that reports wind speed, and a 3 x 3 block around each
metered wind farm. Cells shared by two groups are stored once, keyed by an integer `cell_id`; the
group memberships go in `cell_groups.parquet`. No coordinate is printed, logged, or written to a
filename, and the output folder is under the gitignored `data/` tree.

**Requests are half-yearly.** The Climate Data Store prices a request by fields, not area: a
6-month request for the four variables costs 104,832 of its 121,000 limit wherever the area is, so
half-years are the largest chunk allowed. The cost is checked with the dataset's `costing_api`
before every submission. The requested rectangles come from `era5_cells.cluster_cells`, one
request per cluster of nearby cells; `--link-cells` sets how far apart two cells may be and still
share a request. A large value gives one pooled request per chunk, cropped locally.

**The output folder is write-once.** It is `ERA5-WIND-2019-2023` and is never reused for another
product. Every (chunk, area) result is checkpointed to its own parquet with a `.partial` rename, a
re-run skips what is cached, and `status.json` records each chunk's state. A failed chunk aborts the
run, because every chunk is final ERA5 and so none is "not published yet".

**A second request re-fetches 2024-01** for the wind-farm cells only, to confirm bit-equality with
the on-disk `wind_native_cds.parquet` (guards the cell indexing and the u/v naming).

Run with `uv run --with cdsapi --with netCDF4 python
studies/weather_downloads/fetch_era5_wind_2019_2023.py`. Options: `--dry-run` prints the plan and
each request's cost without fetching; `--trial` fetches only 2019-09 into `_trial/` and prints the
time taken. Credentials are read by `cdsapi` from `~/.cdsapirc`.
"""

import argparse
import json
import logging
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.request import Request, urlopen

import cdsapi  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xarray as xr
from era5_cells import (
    GRID_STEP_DEG,
    Cell,
    Chunk,
    block_cells,
    bounding_area,
    box_cells,
    cluster_cells,
    half_year_chunks,
    quarter_degree_index,
    request_body,
)
from fetch_midas_open import _FILENAME_VERSION_TAG, STATION_METADATA_DIR, _badc_table
from lineage import write_lineage_note, write_readme
from studies.pv_dataset import wind_sites
from studies.sources import (
    ERA5_PRODUCT_DIR,
    ERA5_WIND_2019_2023_PRODUCT_DIR,
    MIDAS_OPEN_PRODUCT_DIR,
)
from studies.trial_area import load_trial_area_box

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("fetch_era5_wind_2019_2023")

SCRIPT_PATH: Final[str] = "studies/weather_downloads/fetch_era5_wind_2019_2023.py"
PRODUCT_DIR: Final[Path] = ERA5_WIND_2019_2023_PRODUCT_DIR
CHUNK_DIR: Final[Path] = PRODUCT_DIR / "_chunks"
SCRATCH_DIR: Final[Path] = PRODUCT_DIR / "_scratch"
TRIAL_DIR: Final[Path] = PRODUCT_DIR / "_trial"
OUTPUT_PATH: Final[Path] = PRODUCT_DIR / "wind_era5_cds.parquet"
OVERLAP_PATH: Final[Path] = PRODUCT_DIR / "overlap_2024_01.parquet"
CELLS_PATH: Final[Path] = PRODUCT_DIR / "cells.parquet"
GROUPS_PATH: Final[Path] = PRODUCT_DIR / "cell_groups.parquet"
STATUS_PATH: Final[Path] = PRODUCT_DIR / "status.json"
OLD_WIND_PATH: Final[Path] = ERA5_PRODUCT_DIR / "wind_native_cds.parquet"
MIDAS_WEATHER_PATH: Final[Path] = MIDAS_OPEN_PRODUCT_DIR / "uk_hourly_weather_obs.parquet"

DATASET: Final[str] = "reanalysis-era5-single-levels"
COSTING_URL: Final[str] = (
    f"https://cds.climate.copernicus.eu/api/retrieve/v1/processes/{DATASET}/costing"
)
VARIABLES: Final[dict[str, str]] = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
}
VALUE_COLUMNS: Final[tuple[str, ...]] = ("u10", "v10", "u100", "v100")
FIRST_MONTH: Final[tuple[int, int]] = (2019, 9)
LAST_MONTH: Final[tuple[int, int]] = (2023, 12)
OVERLAP_CHUNK: Final[Chunk] = Chunk(year=2024, first_month=1, last_month=1)
OVERLAP_KEY: Final[str] = f"{OVERLAP_CHUNK.name}_a00"
EXPECTED_STATIONS: Final[int] = 18
"""How many MIDAS Open hourly-weather stations report a wind speed (see the MIDAS README)."""
POOLED_LINK_CELLS: Final[int] = 10_000
DEFAULT_LINK_CELLS: Final[int] = 8
"""Two cells up to 2 degrees apart share a request, which keeps nearby stations together."""

ChunkStateType = Literal["pending", "cached", "requested", "downloaded", "failed"]


@dataclass(frozen=True)
class Area:
    """One request rectangle and the cells (and only those cells) it is kept for."""

    index: int
    cells: frozenset[Cell]


# --------------------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------------------


def _station_centres() -> list[tuple[str, Cell]]:
    """Return `(src_id, nearest cell)` for every MIDAS station that reports wind speed."""
    wind = (
        pl.scan_parquet(MIDAS_WEATHER_PATH)
        .filter(pl.col("wind_speed_m_s").is_not_null())
        .select("src_id")
        .unique()
        .collect()["src_id"]
        .to_list()
    )
    if len(wind) != EXPECTED_STATIONS:
        msg = f"expected {EXPECTED_STATIONS} stations with wind speed, found {len(wind)}"
        raise RuntimeError(msg)
    path = STATION_METADATA_DIR / (
        f"midas-open_uk-hourly-weather-obs_{_FILENAME_VERSION_TAG}_station-metadata.csv"
    )
    table = pl.read_csv(
        _badc_table(text=path.read_text(encoding="utf-8"), label="station metadata").encode(),
        schema_overrides={"src_id": pl.String},
    ).with_columns(pl.col("src_id").str.strip_chars().str.zfill(5))
    located = table.filter(pl.col("src_id").is_in(wind)).select(
        "src_id", "station_latitude", "station_longitude"
    )
    if located.height != EXPECTED_STATIONS:
        msg = "a wind-reporting station is missing from the station metadata"
        raise RuntimeError(msg)
    return [
        (
            src_id,
            (
                quarter_degree_index(degrees=lat),
                quarter_degree_index(degrees=lon),
            ),
        )
        for src_id, lat, lon in located.sort("src_id").iter_rows()
    ]


def build_group_table() -> pl.DataFrame:
    """Return one row per (group, cell): `kind`, `label`, `dy`, `dx`, `lat_q`, `lon_q`.

    `kind` is `box`, `station`, or `wind`. The box rows have null `dy` and `dx`. Coordinates are
    held as integer quarter-degree indices and only ever written under `data/`.
    """
    rows: list[dict[str, object]] = []
    box = load_trial_area_box()
    rows.extend(
        {"kind": "box", "label": "BOX", "dy": None, "dx": None, "lat_q": lat_q, "lon_q": lon_q}
        for lat_q, lon_q in box_cells(
            lat_min=box.lat_min, lat_max=box.lat_max, lon_min=box.lon_min, lon_max=box.lon_max
        )
    )
    centres: list[tuple[Literal["station", "wind"], str, Cell]] = [
        ("station", src_id, centre) for src_id, centre in _station_centres()
    ]
    wind = wind_sites().select("site", "latitude", "longitude").sort("site")
    centres.extend(
        (
            "wind",
            site,
            (quarter_degree_index(degrees=lat), quarter_degree_index(degrees=lon)),
        )
        for site, lat, lon in wind.iter_rows()
    )
    for kind, label, centre in centres:
        rows.extend(
            {"kind": kind, "label": label, "dy": dy, "dx": dx, "lat_q": cell[0], "lon_q": cell[1]}
            for dy, dx, cell in block_cells(centre=centre)
        )
    return pl.DataFrame(
        rows,
        schema={
            "kind": pl.String,
            "label": pl.String,
            "dy": pl.Int8,
            "dx": pl.Int8,
            "lat_q": pl.Int16,
            "lon_q": pl.Int16,
        },
    )


def assign_cell_ids(*, groups: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return `(cells, groups_with_ids)`, numbering each distinct cell northernmost row first.

    Args:
        groups: The frame from `build_group_table`.

    Returns:
        `cells` with `cell_id`, `lat_q`, `lon_q`, and `groups` with `cell_id` added.
    """
    cells = (
        groups.select("lat_q", "lon_q")
        .unique()
        .sort(["lat_q", "lon_q"], descending=[True, False])
        .with_row_index(name="cell_id")
        .with_columns(pl.col("cell_id").cast(pl.Int32))
    )
    with_ids = groups.join(cells, on=["lat_q", "lon_q"], how="left").select(
        "cell_id", "kind", "label", "dy", "dx", "lat_q", "lon_q"
    )
    return cells, with_ids.sort("kind", "label", "dy", "dx", nulls_last=False)


def plan_areas(*, cells: pl.DataFrame, link_cells: int) -> list[Area]:
    """Return the request rectangles for `cells`, one per cluster of nearby cells."""
    cell_set: set[Cell] = {(int(a), int(b)) for a, b in cells.select("lat_q", "lon_q").iter_rows()}
    return [
        Area(index=index, cells=frozenset(group))
        for index, group in enumerate(cluster_cells(cells=cell_set, link_cells=link_cells))
    ]


# --------------------------------------------------------------------------------------------------
# Status file
# --------------------------------------------------------------------------------------------------


def _write_json_atomically(*, path: Path, payload: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, default=str))
    partial.rename(path)


def _set_state(
    *, status: list[dict[str, object]], key: str, state: ChunkStateType, **fields: object
) -> None:
    """Update one entry of the status array and rewrite `status.json`."""
    for entry in status:
        if entry["key"] == key:
            entry |= {"state": state, **fields}
    _write_json_atomically(path=STATUS_PATH, payload=status)
    _LOG.info("%s: %s", key, state)


# --------------------------------------------------------------------------------------------------
# Request, read, crop
# --------------------------------------------------------------------------------------------------


def request_cost(*, body: dict[str, object]) -> tuple[float, float]:
    """Return `(cost, limit)` from the dataset's costing endpoint for one request body."""
    request = Request(
        COSTING_URL,
        data=json.dumps({"inputs": body}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return float(answer["cost"]), float(answer["limit"])


def _read_zip(*, zip_path: Path, cells: pl.DataFrame, chunk: Chunk) -> pl.DataFrame:
    """Return `cell_id`, `time` and the four wind columns for `cells`, checked for completeness.

    Args:
        zip_path: The downloaded zip of netCDF files.
        cells: The cells to keep, with `cell_id`, `lat_q` and `lon_q`.
        chunk: The months the zip should hold.

    Returns:
        One row per kept cell and hour, sorted by `cell_id` then `time`.

    Raises:
        RuntimeError: If a kept cell or hour is missing, or any value is NaN.
    """
    extract_dir = SCRATCH_DIR / "_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    long: pl.DataFrame | None = None
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            extracted = Path(archive.extract(name, extract_dir))
            with xr.open_dataset(extracted) as ds:
                time_name = "valid_time" if "valid_time" in ds.coords else "time"
                names = [column for column in VALUE_COLUMNS if column in ds.data_vars]
                frame = _dataset_to_frame(ds=ds, time_name=time_name, names=names)
            extracted.unlink()
            long = (
                frame
                if long is None
                else long.join(frame, on=["time", "lat_q", "lon_q"], how="full", coalesce=True)
            )
    if long is None:
        msg = f"{zip_path.name}: the zip holds no files"
        raise RuntimeError(msg)
    kept = (
        long.join(cells, on=["lat_q", "lon_q"], how="inner")
        .select("cell_id", "time", *VALUE_COLUMNS)
        .sort("cell_id", "time")
    )
    expected_rows = cells.height * chunk.n_hours
    if kept.height != expected_rows:
        msg = f"{zip_path.name}: kept {kept.height} rows, expected {expected_rows}"
        raise RuntimeError(msg)
    has_bad = kept.select(
        pl.any_horizontal(pl.col(*VALUE_COLUMNS).is_nan(), pl.col(*VALUE_COLUMNS).is_null()).any()
    ).item()
    if has_bad:
        msg = f"{zip_path.name}: NaN or null in the kept cells"
        raise RuntimeError(msg)
    return kept


def _dataset_to_frame(*, ds: xr.Dataset, time_name: str, names: list[str]) -> pl.DataFrame:
    """Flatten a (time, latitude, longitude) dataset to a long frame of quarter-degree indices."""
    for name in names:
        if set(ds[name].dims) != {time_name, "latitude", "longitude"}:
            msg = f"{name}: unexpected dimensions {ds[name].dims}"
            raise RuntimeError(msg)
    times = ds[time_name].values
    lat_q = np.rint(ds["latitude"].values / GRID_STEP_DEG).astype(np.int16)
    lon_q = np.rint(ds["longitude"].values / GRID_STEP_DEG).astype(np.int16)
    grid_t, grid_lat, grid_lon = np.meshgrid(times, lat_q, lon_q, indexing="ij")
    columns: dict[str, np.ndarray] = {
        "time": grid_t.ravel(),
        "lat_q": grid_lat.ravel(),
        "lon_q": grid_lon.ravel(),
    }
    for name in names:
        columns[name] = (
            ds[name].transpose(time_name, "latitude", "longitude").values.astype(np.float32).ravel()
        )
    frame = pl.DataFrame(columns)
    return frame.with_columns(pl.col("time").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def _fetch_one(
    *,
    chunk: Chunk,
    area: Area,
    cells: pl.DataFrame,
    cache_dir: Path,
    status: list[dict[str, object]],
) -> Path:
    """Fetch one (chunk, area), checkpoint its cropped parquet, and return the parquet path."""
    key = f"{chunk.name}_a{area.index:02d}"
    path = cache_dir / f"{key}.parquet"
    if path.exists():
        _set_state(status=status, key=key, state="cached")
        return path
    wanted = pl.DataFrame(
        sorted(area.cells), schema={"lat_q": pl.Int16, "lon_q": pl.Int16}, orient="row"
    )
    area_cells = cells.join(wanted, on=["lat_q", "lon_q"], how="semi")
    body = request_body(
        chunk=chunk, variables=list(VARIABLES), area=bounding_area(cells=set(area.cells))
    )
    cost, limit = request_cost(body=body)
    if cost > limit:
        msg = f"{key}: cost {cost:.0f} exceeds the limit {limit:.0f}"
        raise RuntimeError(msg)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SCRATCH_DIR / f"{key}.zip"
    started = time.monotonic()
    _set_state(status=status, key=key, state="requested", cost=cost)
    try:
        partial = zip_path.with_suffix(".zip.partial")
        cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, body, str(partial))
        partial.rename(zip_path)
        minutes = (time.monotonic() - started) / 60
        _set_state(
            status=status,
            key=key,
            state="downloaded",
            minutes=round(minutes, 2),
            zip_mb=round(zip_path.stat().st_size / 1e6, 1),
        )
        frame = _read_zip(zip_path=zip_path, cells=area_cells, chunk=chunk)
    except Exception as error:
        _set_state(status=status, key=key, state="failed", error=type(error).__name__)
        raise
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial_parquet = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial_parquet)
    partial_parquet.rename(path)
    zip_path.unlink()
    _set_state(status=status, key=key, state="cached", minutes=round(minutes, 2))
    return path


# --------------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------------


def _prepare_folder(*, groups: pl.DataFrame, cells: pl.DataFrame) -> None:
    """Create the write-once folder, or check that it holds the same cell plan."""
    if OUTPUT_PATH.exists():
        msg = f"{OUTPUT_PATH.name} already exists; the folder is write-once"
        raise RuntimeError(msg)
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    if GROUPS_PATH.exists():
        if not pl.read_parquet(GROUPS_PATH).equals(groups):
            msg = "the cell plan differs from the one on disk; refusing to mix chunks"
            raise RuntimeError(msg)
        return
    if any(PRODUCT_DIR.iterdir()):
        msg = f"{PRODUCT_DIR.name} holds files but no cell plan; it is not ours to reuse"
        raise RuntimeError(msg)
    cells.write_parquet(CELLS_PATH)
    groups.write_parquet(GROUPS_PATH)


def _plan_status(
    *, chunks: list[Chunk], areas: list[Area], include_overlap: bool
) -> list[dict[str, object]]:
    """Return the status array: earlier runs' entries are kept, new keys start `pending`."""
    previous: list[dict[str, object]] = (
        json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else []
    )
    keys = [f"{chunk.name}_a{area.index:02d}" for chunk in chunks for area in areas]
    if include_overlap:
        keys.append(OVERLAP_KEY)
    known = {entry["key"]: entry for entry in previous}
    return [known.get(key, {"key": key, "state": "pending"}) for key in keys]


def _fetch_overlap(
    *, cells: pl.DataFrame, groups: pl.DataFrame, status: list[dict[str, object]]
) -> None:
    """Fetch 2024-01 for the wind-farm cells over one pooled rectangle, once."""
    if OVERLAP_PATH.exists():
        _set_state(status=status, key=OVERLAP_KEY, state="cached")
        return
    wind_ids = groups.filter(pl.col("kind") == "wind")["cell_id"].unique()
    wind_cells = cells.filter(pl.col("cell_id").is_in(wind_ids))
    area = plan_areas(cells=wind_cells, link_cells=POOLED_LINK_CELLS)[0]
    path = _fetch_one(
        chunk=OVERLAP_CHUNK,
        area=area,
        cells=wind_cells,
        cache_dir=CHUNK_DIR / "_overlap",
        status=status,
    )
    path.rename(OVERLAP_PATH)


def _combine(*, paths: list[Path]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(path) for path in paths]).sort("cell_id", "time")


def _write_docs(*, frame: pl.DataFrame, groups: pl.DataFrame) -> None:
    """Write the lineage note and README, computing every stated fact from the written frame."""
    kinds = groups.group_by("kind").agg(pl.col("label").n_unique().alias("labels"))
    write_lineage_note(
        product_dir=PRODUCT_DIR,
        filename="lineage.json",
        source_address=f"https://cds.climate.copernicus.eu/datasets/{DATASET}",
        request_description=(
            "ERA5 hourly single levels, 10 m and 100 m u/v wind, netCDF in a zip, requested in "
            "calendar half-years (2019-09 to 2019-12 first) for rectangles covering the trial-area "
            "box and a 3 x 3 block of 0.25 degree cells around each MIDAS wind-reporting station "
            "and each metered wind farm, then cropped to those cells."
        ),
        variables=list(VARIABLES),
        extra={
            "first_time": str(frame["time"].min()),
            "last_time": str(frame["time"].max()),
            "rows": frame.height,
            "cells": frame["cell_id"].n_unique(),
            "labels_per_kind": dict(kinds.iter_rows()),
            "status": json.loads(STATUS_PATH.read_text()),
        },
    )
    nulls = frame.null_count().row(0, named=True)
    write_readme(
        product_dir=PRODUCT_DIR,
        product_name="ERA5 native wind 2019-09 to 2023-12 from the Climate Data Store",
        source_web_page=f"https://cds.climate.copernicus.eu/datasets/{DATASET}",
        script_path=SCRIPT_PATH,
        lineage_filenames=["lineage.json"],
        columns={
            "cell_id": f"Integer index of a 0.25 degree ERA5 cell ({frame.schema['cell_id']}); "
            "joins to `cells.parquet` and `cell_groups.parquet`.",
            "time": f"UTC instant of the value ({frame.schema['time']}); ERA5 winds are "
            "instantaneous, not averaged.",
            "u10": f"Eastward wind at 10 m, m/s ({frame.schema['u10']}).",
            "v10": f"Northward wind at 10 m, m/s ({frame.schema['v10']}).",
            "u100": f"Eastward wind at 100 m, m/s ({frame.schema['u100']}).",
            "v100": f"Northward wind at 100 m, m/s ({frame.schema['v100']}).",
            "cells.parquet: lat_q, lon_q": "Cell latitude and longitude in whole quarter degrees "
            "(integer index times 0.25 degrees). Private store only; never publish them.",
            "cell_groups.parquet: kind": "`box` (trial-area box), `station` (MIDAS station block) "
            "or `wind` (metered wind-farm block). A cell can belong to several groups.",
            "cell_groups.parquet: label": "`BOX`, the MIDAS `src_id`, or the anonymous wind label "
            "(`W1` to `W3`).",
            "cell_groups.parquet: dy, dx": "Offset in cells north and east of the block's centre "
            "cell (the cell nearest the site); null for `box`.",
            "overlap_2024_01.parquet": "The same columns for 2024-01 over the wind-farm cells, "
            "re-fetched to check bit-equality with `ERA5/wind_native_cds.parquet`.",
        },
        missing_value_convention=(
            f"No missing values: the fetch refuses a chunk with any NaN or null, and the written "
            f"file has {sum(nulls.values())} nulls in total."
        ),
        gotchas=[
            (
                "Speed is sqrt(u^2 + v^2); the direction the wind blows from is "
                "(270 - atan2(v, u) in degrees) mod 360."
            ),
            "ERA5 latitude is stored north to south; `cell_id` numbers cells northernmost first.",
            "The hours are final ERA5, not ERA5T, so the values will not be revised.",
            (
                "Run `validate_era5_wind_2019_2023.py` before using the data; its measured "
                "numbers are in `validation.json` next to this file."
            ),
        ],
        external_docs={
            "ERA5 hourly data on single levels": f"https://cds.climate.copernicus.eu/datasets/{DATASET}",
            "Hersbach et al. (2020)": "https://doi.org/10.1002/qj.3803",
        },
    )


def main() -> int:
    """Plan, fetch every chunk, fetch the overlap month, then combine and document."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the plan and costs only")
    parser.add_argument("--trial", action="store_true", help="fetch 2019-09 only, into _trial/")
    parser.add_argument("--link-cells", type=int, default=DEFAULT_LINK_CELLS)
    args = parser.parse_args()

    groups = build_group_table()
    cells, groups = assign_cell_ids(groups=groups)
    areas = plan_areas(cells=cells, link_cells=args.link_cells)
    chunks = half_year_chunks(first=FIRST_MONTH, last=LAST_MONTH)
    if args.trial:
        chunks = [Chunk(year=2019, first_month=9, last_month=9)]
    _LOG.info(
        "%d chunks x %d areas = %d requests", len(chunks), len(areas), len(chunks) * len(areas)
    )
    if args.dry_run:
        for chunk in chunks:
            cost, limit = request_cost(
                body=request_body(chunk=chunk, variables=list(VARIABLES), area=[1, 0, 0, 1])
            )
            _LOG.info("%s: cost %.0f of limit %.0f", chunk.name, cost, limit)
        return 0

    _prepare_folder(groups=groups, cells=cells)
    cache_dir = TRIAL_DIR if args.trial else CHUNK_DIR
    status = _plan_status(chunks=chunks, areas=areas, include_overlap=not args.trial)
    _write_json_atomically(path=STATUS_PATH, payload=status)
    paths_by_chunk: list[Path] = []
    for chunk in chunks:
        paths_by_chunk.extend(
            _fetch_one(chunk=chunk, area=area, cells=cells, cache_dir=cache_dir, status=status)
            for area in areas
        )
    if args.trial:
        _LOG.info("trial done; see status.json for the minutes taken")
        return 0
    _fetch_overlap(cells=cells, groups=groups, status=status)
    frame = _combine(paths=paths_by_chunk).with_columns(pl.col("cell_id").cast(pl.Int32))
    partial = OUTPUT_PATH.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(OUTPUT_PATH)
    _LOG.info("wrote the combined parquet (%d rows)", frame.height)
    _write_docs(frame=frame, groups=groups)
    return 0


if __name__ == "__main__":
    np.seterr(all="raise")
    sys.exit(main())
