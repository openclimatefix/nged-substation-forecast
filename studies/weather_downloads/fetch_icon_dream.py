"""Download ICON-DREAM-EU whole-domain monthly GRIB, cut to the trial-area box, delete the rest.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. ICON-DREAM-EU is served
from DWD's open-data HTTPS index as one whole-Europe GRIB file per variable per month, on ICON's
native unstructured triangular grid (no lat/lon axes in the GRIB itself). Cropping therefore needs
`ICON-DREAM-EU_grid.nc`, the separate grid-description file carrying each cell's centre `clon`/
`clat` in radians: `_cells_in_box` reads it once, converts to degrees, and returns the small set of
cell indices inside the trial-area box (126 of the grid's 659,156 cells, measured against the
current box). Each monthly GRIB is downloaded whole, opened with `cfgrib` (which needs the system
`eccodes` library; installed for this one-off run with `uv run --with cfgrib --with eccodes`, not
added to `pyproject.toml`), sliced to those cell indices, written as parquet, and the whole-domain
GRIB is deleted — never kept.

**Wind (`WS`) is far larger than the issue's "tens of GB" estimate, but only in transit, not on
disk.** One month of `WS` (wind speed on the 10 lowest model levels) is measured at 6.97 GB
whole-domain, where `ASWDIR_S`/`ASWDIFD_S` (single-level surface radiation) are about 0.2 GB each.
Network transfer is unmetered here, so the whole-domain bytes passing through and being deleted are
not a real cost; what matters is the small cropped file that survives. `fetch_one_month` keeps every
one of the 10 levels (not just a narrow hub-height band) precisely because the cropped output is
cheap regardless of level count — a few hundred MB for the full range below — and a generous level
set is what lets a later study compare wind power prediction using more than one vertical level.

**Vertical interpolation to a fixed hub height is out of scope here.** `WS` is served on model
levels, referenced to height-above-ground through `HHL` in
`ICON-DREAM-EU_constant_fields.grb`, and turning that into "wind speed at 100 m" is new, tested code
that belongs in `packages/studies/`, not in this download script. This script keeps the raw
per-level values so that interpolation step has every level to work from.

**Date range defaults to the metered wind roster's own coverage, not the product's full span.**
ICON-DREAM-EU runs 2010-01 to 2026-08, but NGED's metered wind generators only start 2019-09-17, so
`main`'s `--start-year-month`/`--end-year-month` are meant to be passed as `201909`/`202609` (or
later, once measured) rather than the product's full range — there is no metered wind to compare
years before that against.

Run it with `uv run --with cfgrib --with eccodes python3 -u
studies/weather_downloads/fetch_icon_dream.py --start-year-month 201909 --end-year-month 202609`,
unbuffered (`-u`) so a redirected log stays readable — see the `data-download` skill for why.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import requests
import xarray as xr
from delta_store.precision import round_to_significand_bits
from lineage import write_lineage_note, write_readme
from paths import WEATHER_DOWNLOADS_DIR, load_trial_area_box

BASE_URL: Final[str] = "https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-EU"
GRID_URL: Final[str] = f"{BASE_URL}/invariant/ICON-DREAM-EU_grid.nc"
SOLAR_VARIABLES: Final[tuple[str, ...]] = ("ASWDIR_S", "ASWDIFD_S")
"""Direct and diffuse downward shortwave radiation at the surface, DWD's own GRIB short names."""
WIND_VARIABLE: Final[str] = "WS"
"""Wind speed, DWD's own GRIB short name, served on the 10 lowest full model levels."""
VARIABLE_UNITS: Final[dict[str, str]] = {
    "ASWDIR_S": "aswdir_s_w_m2",
    "ASWDIFD_S": "aswdifd_s_w_m2",
    WIND_VARIABLE: "ws_m_s",
}
"""Each variable's value-column name, shared between the fetch loop and `_write_docs`."""
NWP_SIGNIFICAND_BITS: Final[int] = 13
"""Same choice `delta_store.nwp` makes for production NWP storage: see that module for the
measured rationale. Applied here to the two continuous value columns before writing parquet."""

_SCRATCH_DIR: Final[Path] = Path("/tmp/icon_dream_scratch")
"""Where a whole-domain GRIB lands transiently before being cropped and deleted. Not under `data/`:
nothing here is meant to survive the run that downloads it."""


def _download(*, url: str, destination: Path) -> None:
    """Stream one file to disk, overwriting any previous scratch copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)


def _cells_in_box() -> np.ndarray:
    """Return the ICON grid's cell indices whose centre falls inside the trial-area box.

    Downloads `ICON-DREAM-EU_grid.nc` (about 400 MB) to scratch, reads only `clon`/`clat`, and
    deletes it. The result depends only on the box and the fixed ICON-DREAM-EU grid, so a caller
    fetching several months should call this once and reuse the array.

    Returns:
        Cell indices into the GRIB's `values` dimension, in ascending order.
    """
    grid_path = _SCRATCH_DIR / "ICON-DREAM-EU_grid.nc"
    _download(url=GRID_URL, destination=grid_path)
    box = load_trial_area_box()
    with xr.open_dataset(grid_path) as grid:
        cell_lon_deg = np.degrees(grid["clon"].to_numpy())
        cell_lat_deg = np.degrees(grid["clat"].to_numpy())
    grid_path.unlink()
    in_box = (
        (cell_lon_deg >= box.lon_min)
        & (cell_lon_deg <= box.lon_max)
        & (cell_lat_deg >= box.lat_min)
        & (cell_lat_deg <= box.lat_max)
    )
    return np.nonzero(in_box)[0]


def fetch_one_month(
    *, variable: str, year_month: str, cell_indices: np.ndarray, unit_column: str
) -> pl.DataFrame:
    """Download, crop, and delete one variable's whole-domain GRIB for one month.

    Args:
        variable: DWD's GRIB short name, e.g. `ASWDIR_S` or `WS`.
        year_month: `YYYYMM`.
        cell_indices: The result of `_cells_in_box`.
        unit_column: Name for the value column, e.g. `aswdir_s_w_m2` or `ws_m_s`.

    Returns:
        One row per (valid_time, cell_id) for a single-level variable, or per
        (valid_time, cell_id, model_level) for a multi-level one such as `WS` — `model_level` is
        present as a column only when the source GRIB carries that dimension. `cell_id` is an index
        into the ICON grid rather than a coordinate. `model_level` is cfgrib's own
        `generalVerticalLayer` coordinate value (65-74 for the 10 levels `WS` serves), read directly
        from the file rather than re-numbered 0-9 — a 0-9 position is ambiguous about which end is
        the ground without also carrying DWD's own numbering convention, and ICON numbers levels
        top-down, so the *largest* `generalVerticalLayer` value (74) is the one nearest the surface,
        not the smallest. Turning a level value into a height-above-ground needs `HHL` in
        `ICON-DREAM-EU_constant_fields.grb`, which this script does not read. `valid_time` here is
        DWD's own reanalysis assembly of overlapping short-range forecast steps into an hourly
        series; this frame keeps every (time, step) value as reported rather than resolving any
        overlap, which is a study-time decision, not a download-time one.
    """
    import cfgrib  # ty: ignore[unresolved-import]

    grib_path = _SCRATCH_DIR / f"{variable}_{year_month}.grb"
    url = f"{BASE_URL}/hourly/{variable}/ICON-DREAM-EU_{year_month}_{variable}_hourly.grb"
    _download(url=url, destination=grib_path)
    try:
        dataset = cfgrib.open_dataset(str(grib_path))
        if len(dataset.data_vars) != 1:
            raise ValueError(f"expected exactly one data variable, got {list(dataset.data_vars)}")
        # cfgrib names the data variable from the GRIB's own short name, which is not always
        # DWD's own variable name in uppercase (e.g. `WS` decodes as `ws`) — read it back rather
        # than assuming `variable` itself is the key.
        data_array = dataset[next(iter(dataset.data_vars))].isel(values=cell_indices)
        # Single-level variables (e.g. ASWDIR_S) carry (time, step, values); a multi-level one
        # (e.g. WS) carries an extra level dimension between `step` and `values`. Anything else is
        # a surprise worth stopping on rather than silently reshaping the wrong axis as a level.
        single_level_dims = ("time", "step", "values")
        multi_level_dims = ("time", "step", "generalVerticalLayer", "values")
        if data_array.dims not in (single_level_dims, multi_level_dims):
            raise ValueError(f"unexpected dimension order {data_array.dims}")
        valid_time = dataset["valid_time"].to_numpy()
        values = data_array.to_numpy()
        has_level_dim = values.ndim == 4
        if has_level_dim:
            n_time, n_step, n_level, n_cell = values.shape
            # DWD's own generalVerticalLayer values (65-74 for WS), not a synthetic 0-9 position —
            # see the docstring above for why an arbitrary index would be ambiguous.
            level_values = data_array["generalVerticalLayer"].to_numpy().astype(np.int64)
            columns = {
                "valid_time": np.repeat(valid_time.ravel(), n_level * n_cell),
                "model_level": np.tile(np.repeat(level_values, n_cell), n_time * n_step),
                "cell_id": np.tile(cell_indices, n_time * n_step * n_level),
                unit_column: values.reshape(-1),
            }
        else:
            n_time, n_step, n_cell = values.shape
            columns = {
                "valid_time": np.repeat(valid_time.ravel(), n_cell),
                "cell_id": np.tile(cell_indices, n_time * n_step),
                unit_column: values.reshape(-1),
            }
    finally:
        grib_path.unlink(missing_ok=True)
        # cfgrib writes a sidecar .idx file next to the GRIB it indexes; clean it up too, or these
        # accumulate on tmpfs (harmless but wasteful) across a multi-year backfill.
        for idx_path in _SCRATCH_DIR.glob(f"{grib_path.name}.*.idx"):
            idx_path.unlink(missing_ok=True)

    frame = pl.DataFrame(columns).with_columns(pl.col(unit_column).cast(pl.Float32))
    return frame.with_columns(
        round_to_significand_bits(pl.col(unit_column), keep_bits=NWP_SIGNIFICAND_BITS)
    )


def _write_docs(
    *,
    output_dir: Path,
    variable: str,
    cell_indices: np.ndarray,
    arguments: argparse.Namespace,
) -> None:
    """Write this variable's lineage note and README, given its already-combined parquet."""
    write_lineage_note(
        product_dir=output_dir,
        source_address=f"{BASE_URL}/hourly/{variable}/",
        request_description=(
            f"ICON-DREAM-EU {variable}, whole-domain monthly GRIB downloaded then cropped to "
            f"{cell_indices.size} ICON grid cells inside the trial-area box (a few grid cells' "
            f"margin around the NGED generator roster's own extent) using "
            f"ICON-DREAM-EU_grid.nc; the whole-domain file is deleted immediately after "
            f"cropping. Values cast to Float32 and rounded to {NWP_SIGNIFICAND_BITS} "
            f"significand bits before writing, matching delta_store.nwp's production "
            f"convention. {arguments.start_year_month} to {arguments.end_year_month}"
        ),
        variables=[variable],
        filename=f"lineage_{variable}.json",
        extra={
            "year_month_range_fetched": [arguments.start_year_month, arguments.end_year_month],
            "n_cells_in_box": int(cell_indices.size),
            "note": (
                "All 10 of DWD's served model levels are kept for WS (not narrowed to a "
                "hub-height band), so a later multi-level wind-power study has every level "
                "to draw on. model_level is DWD's own generalVerticalLayer value (65-74), "
                "not a re-numbered 0-9 position and not a height-above-ground; ICON numbers "
                "levels top-down, so 74 (the LARGEST value) is nearest the surface and 65 "
                "is the highest of the 10. The combined parquet also carries 3 NaN padding "
                "rows per (cell, level) per month (cfgrib pads the time x step grid at each "
                "month boundary): filter with is_not_nan(), not is_not_null(), before any "
                "join, dedupe, or mean, since a naive unique() can keep the NaN row instead "
                "of the real one."
                if variable == WIND_VARIABLE
                else (
                    "IMPORTANT: the stored value at each hour is a running mean since the "
                    "nearest 00/03/06/.../21 UTC forecast start (DWD stepType=avg, 'mean "
                    "over forecast time'), NOT an hourly mean. At hour h, the mean spans "
                    "(h mod 3) hours, or 3 if h mod 3 == 0. Recover an hourly mean at step "
                    "k (1, 2, or 3) as k*A_k - (k-1)*A_{k-1}, where A_k is the stored value "
                    "and A_{k-1} the previous hour's row for the same cell from the same "
                    "forecast run (A_0 = 0); this de-averaging is a study-time step, not "
                    "applied here. The combined parquet also carries 3 NaN padding rows per "
                    "cell per month (cfgrib pads the time x step grid at each month "
                    "boundary): filter with is_not_nan(), not is_not_null(), before any "
                    "join, dedupe, or mean, since a naive unique() can keep the NaN row "
                    "instead of the real one. valid_time is timezone-naive (implicitly UTC). "
                    "VERIFIED by the data-validation skill checklist against SARAH-3: after "
                    "de-averaging, de-averaged global correlates at 0.89 (0.86 before "
                    "de-averaging), no night-time irradiance, no step change across years. "
                    "De-averaging can produce small negative values from the upstream "
                    "source's own averaging, not from this script's 13-bit rounding (worth "
                    "~0.03 W/m2 on its own) -- clip de-averaged values at 0 at study time. "
                    "Measured separately per variable (not shared, since the two differ by "
                    "roughly 40x): ASWDIR_S de-averages to 133,330 of 7.73M real rows "
                    "negative, minimum -10.03 W/m2; ASWDIFD_S de-averages to 93,572 "
                    "negative, minimum -0.023 W/m2 (consistent with rounding noise alone)."
                )
            ),
        },
    )
    write_readme(
        product_dir=output_dir,
        product_name=f"ICON-DREAM-EU {variable}",
        source_web_page="https://www.dwd.de/EN/ourservices/reanalysis_icon_dream/reanalysis_icon_dream.html",
        script_path="studies/weather_downloads/fetch_icon_dream.py",
        lineage_filenames=[f"lineage_{variable}.json"],
        columns=(
            {
                "valid_time": "Timezone-naive (implicitly UTC). DWD's reanalysis assembly "
                "of overlapping short-range forecast steps into an hourly series — every "
                "(time, step) value is kept as reported, with no overlap resolved.",
                "cell_id": "Index into ICON-DREAM-EU's native unstructured grid "
                "(ICON-DREAM-EU_grid.nc) — not a coordinate.",
                "model_level": "DWD's own generalVerticalLayer value (65-74 for WS). ICON "
                "numbers levels top-down, so the LARGEST value (74) is nearest the "
                "surface, not the smallest.",
                "ws_m_s": "Wind speed, m/s, an analysis value (the instant itself, not "
                "averaged) at each of the 10 model_level values.",
            }
            if variable == WIND_VARIABLE
            else {
                "valid_time": "Timezone-naive (implicitly UTC). Marks the END of a running "
                "mean since the nearest 00/03/06/.../21 UTC forecast start — NOT an hourly "
                "mean. See gotchas.",
                "cell_id": "Index into ICON-DREAM-EU's native unstructured grid "
                "(ICON-DREAM-EU_grid.nc) — not a coordinate.",
                VARIABLE_UNITS[variable]: f"{'Direct' if variable == 'ASWDIR_S' else 'Diffuse'} "
                "downward shortwave radiation flux at the surface, W/m^2, as a running mean "
                "(see gotchas) — not an hourly mean.",
            }
        ),
        missing_value_convention=(
            "Float NaN (not Polars null): cfgrib pads the (time, step) grid to a rectangle "
            "at each month's boundary, leaving 3 NaN rows per (cell[, level]) per month. "
            "Filter with `is_not_nan()`, not `is_not_null()`, before any join, dedupe, or "
            "mean — a naive `unique()` can keep the NaN row instead of the real one."
        ),
        gotchas=(
            [
                (
                    "All 10 of DWD's served model levels are kept (not narrowed to a "
                    "hub-height band) for a later multi-level wind-power study."
                ),
                (
                    "model_level is DWD's own level index (65-74), not a 0-based position "
                    "and not a height-above-ground — see the columns section above."
                ),
            ]
            if variable == WIND_VARIABLE
            else [
                (
                    "The stored value is a RUNNING MEAN since the nearest 3-hourly forecast "
                    "start, not an hourly mean — de-averaging is required before use. See "
                    f"`lineage_{variable}.json` for the exact formula and verification "
                    "against SARAH-3."
                ),
                (
                    "De-averaging can produce small negative values from the upstream "
                    "source's own averaging (not from this script's rounding) — clip at 0."
                ),
            ]
        ),
        external_docs={
            "DWD ICON-DREAM-EU documentation": (
                "https://www.dwd.de/EN/ourservices/reanalysis_icon_dream/reanalysis_icon_dream.html"
            ),
            "DWD ICON-DREAM-EU parameter table": (f"{BASE_URL}/ParameterTables_ICON.pdf"),
        },
        filename=f"README_{variable}.md",
    )


def main() -> int:
    """Fetch every whole month of `SOLAR_VARIABLES` in `[--start-year-month, --end-year-month]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year-month", required=True, help="YYYYMM")
    parser.add_argument("--end-year-month", required=True, help="YYYYMM")
    arguments = parser.parse_args()

    cell_indices = _cells_in_box()
    print(f"{cell_indices.size} ICON-DREAM-EU cells fall inside the trial-area box")

    start_year = int(arguments.start_year_month[:4])
    start_month = int(arguments.start_year_month[4:])
    end_year = int(arguments.end_year_month[:4])
    end_month = int(arguments.end_year_month[4:])
    year_months: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        year_months.append(f"{year:04d}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1

    for variable in (*SOLAR_VARIABLES, WIND_VARIABLE):
        output_dir = WEATHER_DOWNLOADS_DIR / "ICON-DREAM-EU"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Every month is checkpointed to its own parquet as soon as it is cropped, so a crash
        # partway through a variable (a missing month, a network drop, a server error) loses at
        # most one month's work rather than the whole run: re-running skips whatever is cached.
        month_cache_dir = output_dir / "_month_cache" / variable
        month_cache_dir.mkdir(parents=True, exist_ok=True)
        for year_month in year_months:
            month_path = month_cache_dir / f"{year_month}.parquet"
            if month_path.exists():
                print(f"{variable} {year_month}: already cached, skipping")
                continue
            try:
                frame = fetch_one_month(
                    variable=variable,
                    year_month=year_month,
                    cell_indices=cell_indices,
                    unit_column=VARIABLE_UNITS[variable],
                )
            # The newest requested month may not be published yet (DWD lags behind "now" by an
            # unpublished amount), and losing every earlier month's work over one missing month at
            # the end is worse than skipping it: only a 404 is treated as "not yet published"; any
            # other failure still aborts the run.
            except requests.exceptions.HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    print(f"{variable} {year_month}: not published yet (404), skipping")
                    continue
                raise
            partial = month_path.with_suffix(".parquet.partial")
            frame.write_parquet(partial)
            partial.rename(month_path)
            print(f"{variable} {year_month}: {frame.height} rows")
        # Combine only the months this run asked for, not every file the cache directory happens
        # to hold — a directory reused across a differently-scoped run must not silently mix in.
        cached_paths = [
            p for ym in year_months if (p := month_cache_dir / f"{ym}.parquet").exists()
        ]
        if not cached_paths:
            print(f"{variable}: no month succeeded, nothing written")
            continue
        combined = pl.concat([pl.read_parquet(path) for path in cached_paths])
        filename = f"{variable}_{arguments.start_year_month}_{arguments.end_year_month}.parquet"
        output_path = output_dir / filename
        combined.write_parquet(output_path)
        size_mb = output_path.stat().st_size / 1e6
        print(f"{variable}: wrote {combined.height} rows to {output_path}, {size_mb:.3f} MB")

        _write_docs(
            output_dir=output_dir,
            variable=variable,
            cell_indices=cell_indices,
            arguments=arguments,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
