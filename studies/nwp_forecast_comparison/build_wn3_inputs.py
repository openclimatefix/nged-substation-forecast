"""Read the WeatherNext 3 (WN3) trial-area subset and build the WN3 arm inputs.

One-off throwaway script for the WN3 arms of
<https://github.com/openclimatefix/nged-substation-forecast/issues/974>. It has two stages.

`--read-store` opens the WN3 Icechunk repository (`--bucket`, branch `main`) read-only and copies
the 00 UTC runs' six variables that the arms use (2 m temperature, total solar radiation, and the
eastward and northward wind at 10 m and 100 m) over a box around the private generator roster to
`--weather-dir/WeatherNext3_trial_area/`: `trial_area.zarr` and `_grid_cells.parquet`. The box is
the roster's extent plus `PAD_DEGREES` on every side, so every site's H3 resolution-5 hexagon lies
inside it. It is computed at run time, is never printed, and is never written into a committed
file. A missing chunk reads as `NaN`, so the read fails if any copied variable is all `NaN`, and it
prints each variable's `NaN` share. Every later stage reads the local copy.

`--build` writes `<domain>_wn3_inputs.parquet` into a new write-once `--output-dir`, on the
published inputs' own `(site, time)` keys:

- `wn3_mean_day<N>_*` at `--days`, WN3's ensemble mean read as ENS is: the 00 UTC run of day
  `D - N` for a target hour on day `D`. The store is hourly already, so each hour's value passes
  through. A solar hour is labelled by its end and reads the radiation of the hour ending at that
  lead, in W m-2 (the store's J m-2 divided by 3600), and the temperature is the mean of the two
  hourly values around the hour's midpoint, in degrees Celsius. A wind hour reads the hour's own
  lead. Each site's value is the H3 resolution-5 overlap-weighted mean of the crop's 0.1 degree
  cells, taken on the eastward and northward components and turned into speed and direction after
  that.
- `ens_mean_day<N>_*` at days 4, 5, 7, 10 and 14, the same-rows ENS reference, built by
  `build_forecast_inputs._ens_extra_frame` as the published extra-lead inputs are.
- Wind only, `ens_meanvec_day<N>_*` at every built day, the matched ENS reference whose speed is
  the length of the mean of the members' wind vectors, as WN3's is. The store holds only the
  ensemble-mean components, so WN3's speed is the length of the mean vector, which is lower than
  the mean of the members' speeds whenever the members disagree in direction. The ENS mean
  averages the members' speeds.
- `<arm>_init_time` for each WN3 arm and for each `ens_meanvec_day<N>`, which `fit_aifs.py --wn3`
  checks against the run it should read.

Every output row carries only the anonymised `site` label; no generator name, id or coordinate is
printed.

Run it with `uv run python studies/nwp_forecast_comparison/build_wn3_inputs.py --read-store
--bucket BUCKET`, then `uv run python studies/nwp_forecast_comparison/build_wn3_inputs.py --build
--output-dir DIR`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_forecast_inputs import (
    DomainType,
    _ens_extra_frame,
    _repo_data_dir,
    aifs_site_weights,
    ens_members,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "weather_downloads"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
import ens_forecast_horizons as efh

# `fetch_weathernext3` sets `ICECHUNK_LOG` before `icechunk` is imported below it.
import fetch_weathernext3 as fetch
import icechunk
import zarr
from studies.guards import refuse_to_overwrite
from studies.resample import interpolate_linear, wind_components

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

WN3_DIR_NAME: Final[str] = "WeatherNext3_trial_area"
"""Under `data/studies/weather/`, the folder of the local copy."""

ZARR_NAME: Final[str] = "trial_area.zarr"

GRID_CELLS_NAME: Final[str] = "_grid_cells.parquet"

GRID_DEGREES: Final[float] = 0.1
"""The WN3 grid's cell size, as `compute_h3_grid_weights` bins by."""

PAD_DEGREES: Final[float] = 0.25
"""How far the copied box extends beyond the roster's extent. An H3 resolution-5 hexagon reaches
about 0.13 degrees from its centre, and the 0.1 degree grid adds up to a cell more."""

FIRST_INIT: Final[np.datetime64] = np.datetime64("2026-01-01T00", "h")
"""The store's first run."""

TEMPERATURE: Final[str] = "temperature_2m_mean"
RADIATION: Final[str] = "surface_solar_radiation_downwards_1hr_mean"
WIND_COMPONENTS: Final[dict[str, tuple[str, str]]] = {
    "100m": ("u_component_of_wind_100m_mean", "v_component_of_wind_100m_mean"),
    "10m": ("u_component_of_wind_10m_mean", "v_component_of_wind_10m_mean"),
}
"""The store's variable for each quantity the arms read, and each height's eastward and northward
components. Direct radiation, the seventh stored variable, is not read."""

COPIED_VARIABLES: Final[tuple[str, ...]] = (
    TEMPERATURE,
    RADIATION,
    *(name for pair in WIND_COMPONENTS.values() for name in pair),
)
"""The six variables the read copies."""

J_PER_M2_TO_W_PER_M2: Final[float] = 1.0 / 3600.0
"""The store's radiation is J m-2 over an hour; dividing by 3600 s gives the mean W m-2."""

KELVIN: Final[float] = 273.15

N_LEADS: Final[int] = 360
"""The store's leads, hourly from 1 to 360."""

WN3_DAYS: Final[tuple[int, ...]] = (1, 2, 7, 14)
"""The lead days built unless `--days` names others."""

ENS_EXTRA_DAYS: Final[tuple[int, ...]] = (4, 5, 7, 10, 14)
"""The days whose same-rows ENS mean `_ens_extra_frame` builds beside the WN3 arms. Days 0 to 3 are
in the published inputs already."""

PHYSICAL_RANGES: Final[dict[str, tuple[float, float]]] = {
    "ghi": (0.0, 1400.0),
    "temp": (-30.0, 45.0),
    "speed_100m": (0.0, 70.0),
    "speed_10m": (0.0, 70.0),
}
"""The physical range of each built column that every value must lie inside: hourly-mean global
horizontal irradiance in W m-2 (the solar constant is 1,361), 2 m temperature in degrees C at a
Great Britain site, and wind speed in m s-1. The identity check compares the transform with itself,
so a wrong unit passes it, and this range does not."""

MIN_PEAK_GHI: Final[float] = 300.0
"""The lowest peak a WN3 arm's irradiance may have over a run of months: a smaller peak means the
values are still in the wrong unit (kJ or MJ, say)."""

CHECK_ROWS: Final[int] = 500
"""How many built rows `check_against_store` recomputes from the local copy, cell by cell."""

CHECK_TOLERANCE: Final[float] = 1e-4
"""The relative difference the identity check allows (the copy is `Float32`)."""


def local_dir(*, weather_dir: Path) -> Path:
    """Return the folder holding the local copy."""
    return weather_dir / WN3_DIR_NAME


def _snapped_slice(*, axis: np.ndarray, low: float, high: float) -> slice:
    """Return the slice of a stored axis covering `low` to `high`, widened to whole cells."""
    inside = np.flatnonzero((axis >= low - GRID_DEGREES / 2) & (axis <= high + GRID_DEGREES / 2))
    if not inside.size:
        msg = "no stored cell lies inside the roster's box"
        raise ValueError(msg)
    return slice(int(inside[0]), int(inside[-1]) + 1)


def log_run_provenance(
    *, init_hours: np.ndarray, written: np.ndarray, source_hours: np.ndarray
) -> None:
    """Log each 00 UTC run's `run_written` flag beside its `init_time` and `source_init_time`.

    LOOKAHEAD RECORD STEP. The page's lookahead section says the publication latency of a run is
    not stated in Google's documentation. Once the store may be read, this output is the evidence
    that the runs were written in real time: a run whose `source_init_time` differs from its
    `init_time`, or that is not written, is not a plain real-time 00 UTC run. Read the log before
    any fit, and copy what it shows into the page's lookahead section.

    Args:
        init_hours: The store's `init_time`, in integer hours since the epoch.
        written: The store's `run_written` flags.
        source_hours: The store's `source_init_time`, in the same unit.

    Raises:
        ValueError: After logging every run, if no run is written, a 00 UTC run before the last
            written one is not written, or a written run has a `source_init_time` different from
            its `init_time`.
    """
    init = init_hours.astype("datetime64[h]")
    midnight = init.astype("datetime64[D]").astype("datetime64[h]") == init
    positions = np.flatnonzero(midnight & (init >= FIRST_INIT))
    is_written = written[positions].astype(bool)
    if not is_written.any():
        msg = "no 00 UTC run of 2026 is written in the store"
        raise ValueError(msg)
    last = int(np.flatnonzero(is_written).max())
    for position in positions[: last + 1]:
        _LOG.info(
            "run %s: run_written=%s source_init_time=%s",
            init[position],
            bool(written[position]),
            source_hours[position].astype("datetime64[h]"),
        )
    done = positions[is_written]
    differs = int((source_hours[done] != init_hours[done]).sum())
    # The store's init_time axis is allocated years ahead, so runs after the last written one are
    # not gaps; an unwritten run before the last written one is.
    gaps = int((~is_written[: last + 1]).sum())
    _LOG.info(
        "%d 00 UTC runs written, %d gaps before the last written run, %d with source_init_time "
        "different from init_time",
        len(done),
        gaps,
        differs,
    )
    if gaps or differs:
        msg = (
            f"{gaps} 00 UTC runs are unwritten before the last written run and {differs} have a "
            "source_init_time different from their init_time: the runs may not be plain "
            "real-time runs. Do not fit until the difference is explained."
        )
        raise ValueError(msg)


def read_trial_area(*, bucket: str, weather_dir: Path) -> None:
    """Copy the 00 UTC runs of the arms' six variables, over the roster's box, to local disk.

    The copy is one variable at a time, so memory holds one variable's box.

    Args:
        bucket: The Cloud Storage bucket holding the Icechunk repository.
        weather_dir: The folder whose `WeatherNext3_trial_area` subfolder receives the copy.

    Raises:
        FileExistsError: If the copy already exists.
        ValueError: If a copied variable is entirely `NaN`, or the box holds no stored cell.
    """
    target = local_dir(weather_dir=weather_dir)
    refuse_to_overwrite(paths=[target / ZARR_NAME, target / GRID_CELLS_NAME])
    roster = pl.concat([efh.site_roster(domain="solar"), efh.site_roster(domain="wind")])
    storage = icechunk.gcs_storage(bucket=bucket, prefix=fetch.STORE_PREFIX, from_env=True)
    session = icechunk.Repository.open(storage).readonly_session(branch=fetch.MAIN_BRANCH)
    root = zarr.open_group(session.store, mode="r")
    latitude = np.asarray(root.get_array(fetch.LATITUDE)[:])
    longitude = np.asarray(root.get_array(fetch.LONGITUDE)[:])
    lat_slice = _snapped_slice(
        axis=latitude,
        low=float(roster["latitude"].to_numpy().min()) - PAD_DEGREES,
        high=float(roster["latitude"].to_numpy().max()) + PAD_DEGREES,
    )
    lon_slice = _snapped_slice(
        axis=longitude,
        low=float(roster["longitude"].to_numpy().min()) - PAD_DEGREES,
        high=float(roster["longitude"].to_numpy().max()) + PAD_DEGREES,
    )
    init_hours = np.asarray(root.get_array(fetch.INIT_TIME)[:])
    written = np.asarray(root.get_array(fetch.RUN_WRITTEN)[:])
    log_run_provenance(
        init_hours=init_hours,
        written=written,
        source_hours=np.asarray(root.get_array(fetch.SOURCE_INIT_TIME)[:]),
    )
    init = init_hours.astype("datetime64[h]")
    at_midnight = init.astype("datetime64[D]").astype("datetime64[h]") == init
    positions = np.flatnonzero(at_midnight & written & (init >= FIRST_INIT))
    _LOG.info("%d 00 UTC runs copied", len(positions))
    coords = {
        fetch.INIT_TIME: init[positions].astype("datetime64[ns]"),
        fetch.LEAD_TIME: np.arange(1, N_LEADS + 1),
        fetch.LATITUDE: latitude[lat_slice],
        fetch.LONGITUDE: longitude[lon_slice],
    }
    target.mkdir(parents=True, exist_ok=True)
    for number, name in enumerate(COPIED_VARIABLES):
        values = root.get_array(name).get_orthogonal_selection(
            (positions, slice(None), lat_slice, lon_slice)
        )
        if bool(np.isnan(values).all()):
            msg = f"{name} is entirely NaN in the copied box"
            raise ValueError(msg)
        _LOG.info("%s: NaN share %.4f", name, float(np.isnan(values).mean()))
        xr.Dataset({name: (tuple(coords), values)}, coords=coords).to_zarr(
            target / ZARR_NAME, mode="w-" if number == 0 else "a"
        )
    lat_index, lon_index = np.meshgrid(
        np.arange(len(coords[fetch.LATITUDE])),
        np.arange(len(coords[fetch.LONGITUDE])),
        indexing="ij",
    )
    pl.DataFrame(
        {
            "lat_index": lat_index.ravel().astype(np.int16),
            "lon_index": lon_index.ravel().astype(np.int16),
            "latitude": coords[fetch.LATITUDE][lat_index.ravel()],
            "longitude": coords[fetch.LONGITUDE][lon_index.ravel()],
        }
    ).write_parquet(target / GRID_CELLS_NAME)


def open_local(*, weather_dir: Path) -> xr.Dataset:
    """Open the local copy, loaded into memory.

    Args:
        weather_dir: The directory holding the local copy of the store.

    Returns:
        The six variables, with dimensions `(init_time, lead_time, latitude, longitude)`.
    """
    return xr.open_zarr(local_dir(weather_dir=weather_dir) / ZARR_NAME, chunks=None).load()


def site_cube(
    *, dataset: xr.Dataset, weights: pl.DataFrame, sites: list[str], name: str
) -> np.ndarray:
    """Return one variable's overlap-weighted mean at each site, for every run and lead.

    A `NaN` in any cell under a site's hexagon makes that site's value `NaN`; a `NaN` in a cell
    outside every hexagon does not.

    Args:
        dataset: `open_local`'s result.
        weights: `aifs_site_weights`'s result: `site`, `lat_index`, `lon_index` and `weight`.
        sites: The sites, in the order of the result's last axis.
        name: The variable.

    Returns:
        Shape (n_runs, n_leads, n_sites), `Float64`.
    """
    n_lat, n_lon = dataset.sizes[fetch.LATITUDE], dataset.sizes[fetch.LONGITUDE]
    matrix = np.zeros((n_lat * n_lon, len(sites)))
    site_position = {site: position for position, site in enumerate(sites)}
    for site, lat_index, lon_index, weight in weights.select(
        "site", "lat_index", "lon_index", "weight"
    ).iter_rows():
        matrix[lat_index * n_lon + lon_index, site_position[site]] += weight
    covered = (matrix > 0).astype(np.float64)
    values = dataset[name].to_numpy()
    cube = np.empty((values.shape[0], values.shape[1], len(sites)))
    for run in range(values.shape[0]):
        flat = values[run].reshape(values.shape[1], n_lat * n_lon).astype(np.float64)
        bad = np.isnan(flat)
        cube[run] = np.nan_to_num(flat) @ matrix
        cube[run][(bad.astype(np.float64) @ covered) > 0] = np.nan
    return cube


def _row_positions(
    *, keys: pl.DataFrame, sites: list[str], runs: np.ndarray, domain: DomainType, day: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return, for each key row, the run it reads and the run, lead and site positions.

    Args:
        keys: `site` and `time` (UTC).
        sites: The sites, in the order of the cubes' last axis.
        runs: The store copy's runs, `datetime64[h]`, ascending.
        domain: `solar` or `wind`.
        day: The band's day.

    Returns:
        The run each row reads (`datetime64[h]`), its run position, its lead position (the lead
        minus 1, since the store's leads start at 1), and its site position. The run and lead
        positions are clamped inside the arrays; `_has_value` says which rows have a value.
    """
    time = keys["time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[h]")
    hour_of_day = time - np.timedelta64(1, "h") if domain == "solar" else time
    run = hour_of_day.astype("datetime64[D]").astype("datetime64[h]") - np.timedelta64(
        24 * day, "h"
    )
    lead = ((time - run) / np.timedelta64(1, "h")).astype(np.int64)
    run_position = np.clip(np.searchsorted(runs, run), 0, len(runs) - 1)
    site_position = keys["site"].replace_strict(
        {site: position for position, site in enumerate(sites)}, return_dtype=pl.Int64
    )
    return run, run_position, lead - 1, site_position.to_numpy()


def _has_value(
    *,
    run: np.ndarray,
    run_position: np.ndarray,
    lead_position: np.ndarray,
    runs: np.ndarray,
    domain: DomainType,
) -> np.ndarray:
    """Return which rows have a copied run and every stored lead the domain reads."""
    # Solar also reads the lead one hour earlier (for the temperature), so it needs the store's
    # second lead onward, and wind needs the first. From day 1 every band's lead is at least 24
    # hours, far above the bound. A day-0 row at the run's first hour (wind at 00:00, solar at
    # 01:00 UTC) has no stored lead, so it is left null.
    first_position = 1 if domain == "solar" else 0
    return (
        (runs[run_position] == run) & (lead_position >= first_position) & (lead_position < N_LEADS)
    )


def _masked(*, values: np.ndarray, present: np.ndarray) -> pl.Series:
    """Return `values` as a series with a null wherever a row has no value."""
    return pl.Series(np.where(present, values, np.nan)).fill_nan(None)


def wn3_arm_frame(
    *,
    cubes: dict[str, np.ndarray],
    runs: np.ndarray,
    sites: list[str],
    keys: pl.DataFrame,
    domain: DomainType,
    day: int,
) -> pl.DataFrame:
    """Return one day's `wn3_mean_day<N>` columns and run stamp on `keys`.

    Args:
        cubes: `site_cube`'s result for each variable read, by store name.
        runs: The copy's runs, `datetime64[h]`.
        sites: The sites, in the cubes' last-axis order.
        keys: `site` and `time` for every row the study might score.
        domain: `solar` or `wind`.
        day: The band's day: an hour on day `D` reads the 00 UTC run of day `D - day`.

    Returns:
        `keys` with the arm's columns and `<arm>_init_time`, null where the run or lead is not in
        the copy.
    """
    run, run_position, lead_position, site_position = _row_positions(
        keys=keys, sites=sites, runs=runs, domain=domain, day=day
    )
    present = _has_value(
        run=run, run_position=run_position, lead_position=lead_position, runs=runs, domain=domain
    )
    arm = f"wn3_mean_day{day}"

    def at(name: str, *, lead_shift: int = 0) -> np.ndarray:
        lead = np.clip(lead_position - lead_shift, 0, N_LEADS - 1)
        return cubes[name][run_position, lead, site_position]

    if domain == "solar":
        fields = {
            "ghi": at(RADIATION) * J_PER_M2_TO_W_PER_M2,
            "temp": (at(TEMPERATURE) + at(TEMPERATURE, lead_shift=1)) / 2.0 - KELVIN,
        }
    else:
        east, north = (at(name) for name in WIND_COMPONENTS["100m"])
        east_10m, north_10m = (at(name) for name in WIND_COMPONENTS["10m"])
        speed = np.hypot(east, north)
        with np.errstate(invalid="ignore", divide="ignore"):
            fields = {
                "speed_100m": speed,
                "sin_100m": -east / speed,
                "cos_100m": -north / speed,
                "speed_10m": np.hypot(east_10m, north_10m),
            }
    reduced = keys.select("site", "time").with_columns(
        **{field: _masked(values=values, present=present) for field, values in fields.items()}
    )
    stamp = pl.Series(np.where(present, run, np.datetime64("NaT", "h")).astype("datetime64[us]"))
    return efh.prefixed(frame=reduced, arm=arm, domain=domain).with_columns(
        stamp.dt.replace_time_zone("UTC").alias(f"{arm}_init_time")
    )


def _label_read(
    *, dataset: xr.Dataset, cells: pl.DataFrame, run: np.datetime64, lead: int, name: str
) -> float:
    """Return a site's weighted mean of one variable at one run and lead, cell by cell by label."""
    moment = dataset[name].sel({fetch.INIT_TIME: run, fetch.LEAD_TIME: lead})
    return sum(
        weight * float(moment.isel({fetch.LATITUDE: lat_index, fetch.LONGITUDE: lon_index}))
        for lat_index, lon_index, weight in cells.select(
            "lat_index", "lon_index", "weight"
        ).iter_rows()
    )


def check_against_store(
    *,
    built: pl.DataFrame,
    dataset: xr.Dataset,
    weights: pl.DataFrame,
    domain: DomainType,
    day: int,
) -> None:
    """Raise unless sampled built values equal a recomputation from the local copy.

    A different path from `site_cube`'s matrix product and `wn3_arm_frame`'s gathers: each sampled
    row's run is the stamp the row carries, and its value is read cell by cell by label. It checks
    the radiation (solar) and the 100 m speed (wind).

    Args:
        built: `wn3_arm_frame`'s result.
        dataset: `open_local`'s result.
        weights: `aifs_site_weights`'s result.
        domain: `solar` or `wind`.
        day: The band's day.

    Raises:
        ValueError: If fewer than `CHECK_ROWS` rows could be sampled, or any sampled row differs
            from the recomputation by more than `CHECK_TOLERANCE` (relative).
    """
    arm = f"wn3_mean_day{day}"
    column = f"{arm}_ghi" if domain == "solar" else f"{arm}_speed_100m"
    valid = built.filter(pl.col(column).is_not_null())
    if valid.height < CHECK_ROWS:
        msg = f"{arm}: only {valid.height} rows to sample for the identity check"
        raise ValueError(msg)
    sample = valid.sample(n=CHECK_ROWS, seed=day, shuffle=True)
    cells = {site: frame for (site,), frame in weights.group_by("site")}
    worst = 0.0
    for site, time, init, value in sample.select(
        "site", "time", f"{arm}_init_time", column
    ).iter_rows():
        run = np.datetime64(init.replace(tzinfo=None), "ns")
        hour = np.datetime64(time.replace(tzinfo=None), "h")
        lead = int((hour - run.astype("datetime64[h]")) / np.timedelta64(1, "h"))
        if domain == "solar":
            expected = (
                _label_read(dataset=dataset, cells=cells[site], run=run, lead=lead, name=RADIATION)
                * J_PER_M2_TO_W_PER_M2
            )
        else:
            east, north = (
                _label_read(dataset=dataset, cells=cells[site], run=run, lead=lead, name=name)
                for name in WIND_COMPONENTS["100m"]
            )
            expected = float(np.hypot(east, north))
        worst = max(worst, abs(value - expected) / max(abs(expected), 1.0))
    if worst > CHECK_TOLERANCE:
        msg = f"{arm}: built values differ from the local copy by up to {worst:.2e} (relative)"
        raise ValueError(msg)
    _LOG.info("%s: identity check against the local copy passed (worst %.1e)", arm, worst)


def check_physical_range(*, built: pl.DataFrame, domain: DomainType, day: int) -> None:
    """Raise unless every value of a WN3 arm's columns lies in a physical range.

    Args:
        built: `wn3_arm_frame`'s result.
        domain: `solar` or `wind`.
        day: The band's day.

    Raises:
        ValueError: If a built value lies outside `PHYSICAL_RANGES`, or the peak irradiance is
            below `MIN_PEAK_GHI`.
    """
    arm = f"wn3_mean_day{day}"
    names = ("ghi", "temp") if domain == "solar" else ("speed_100m", "speed_10m")
    for name in names:
        low, high = PHYSICAL_RANGES[name]
        values = built[f"{arm}_{name}"].drop_nulls()
        if values.min() < low or values.max() > high:  # ty: ignore[unsupported-operator]
            msg = (
                f"{arm}_{name}: values run {values.min()} to {values.max()}, "
                f"outside {low} to {high}"
            )
            raise ValueError(msg)
    if domain == "solar" and built[f"{arm}_ghi"].max() < MIN_PEAK_GHI:  # ty: ignore[unsupported-operator]
        msg = f"{arm}_ghi: the peak is {built[f'{arm}_ghi'].max()}, so the unit is probably wrong"
        raise ValueError(msg)
    _LOG.info("%s: physical-range check passed", arm)


def check_band_complete(*, built: pl.DataFrame, domain: DomainType, day: int) -> None:
    """Raise if a row that reads a stored run and lead has a missing value in the band.

    `wn3_arm_frame` leaves a value null where the run is absent from the copy or the lead is not
    stored, and stamps the run only on the rows that have both. A stamped row with a null value
    therefore reads a run whose band has a hole: a `NaN` in the store, which the weighted mean
    carries through. The hole would otherwise drop the row from the fit without a word, as a run
    missing from the copy does.

    Args:
        built: `wn3_arm_frame`'s result.
        domain: `solar` or `wind`.
        day: The band's day.

    Raises:
        ValueError: If a stamped row has a null in the arm's irradiance and temperature (solar) or
            its two wind speeds (wind).
    """
    arm = f"wn3_mean_day{day}"
    names = ("ghi", "temp") if domain == "solar" else ("speed_100m", "speed_10m")
    holes = built.filter(
        pl.col(f"{arm}_init_time").is_not_null(),
        pl.any_horizontal(pl.col(f"{arm}_{name}").is_null() for name in names),
    )
    if holes.height:
        first = holes["time"].min()
        msg = (
            f"{arm}: {holes.height} rows read a stored run and lead but hold no value, the first "
            f"at {first}, so the store has a hole inside the day-{day} band"
        )
        raise ValueError(msg)
    _LOG.info("%s: band-completeness check passed", arm)


def ens_vector_mean_frame(*, extract: pl.DataFrame, day: int) -> pl.DataFrame:
    """Return ENS's wind arm whose speed is the length of the mean of the members' wind vectors.

    The upsampling is the published ENS mean's linear interpolation of each member's components,
    at both heights; only the reduction across members differs.

    Args:
        extract: `ens_forecast_horizons.members`' result for the wind sites.
        day: The band's day.

    Returns:
        `site`, `time`, `ens_meanvec_day<N>_*` and `ens_meanvec_day<N>_init_time`.
    """
    steps = efh.band_steps(members=extract, day=day, domain="wind")
    targets = efh.target_leads(day=day, domain="wind")
    mean: dict[str, np.ndarray] = {}
    for height in ("100m", "10m"):
        east, north = wind_components(
            speed=steps.values[f"speed_{height}"],
            direction_deg=steps.values[f"direction_{height}"],
        )
        for label, component in (("east", east), ("north", north)):
            hourly = interpolate_linear(values=component, x=steps.leads, targets=targets)
            mean[f"{label}_{height}"] = hourly.reshape(-1, steps.ensemble_size, len(targets)).mean(
                axis=1
            )
    speed = np.hypot(mean["east_100m"], mean["north_100m"])
    fields = {
        "speed_100m": speed,
        "sin_100m": -mean["east_100m"] / speed,
        "cos_100m": -mean["north_100m"] / speed,
        "speed_10m": np.hypot(mean["east_10m"], mean["north_10m"]),
    }
    runs = steps.keys.filter(pl.col("ensemble_member") == 0).select("site", "init_time")
    arm = f"ens_meanvec_day{day}"
    rows = runs.select(pl.all().repeat_by(len(targets)).explode(empty_as_null=True)).with_columns(
        lead=pl.Series(np.tile(targets, runs.height)).cast(pl.Int64),
        **{name: pl.Series(values.reshape(-1)) for name, values in fields.items()},
    )
    init_time = pl.col("init_time").dt.replace_time_zone("UTC", non_existent="raise")
    rows = rows.select(
        "site",
        *fields,
        time=init_time + pl.duration(hours=pl.col("lead")),
        init_time=init_time,
    )
    return efh.prefixed(frame=rows, arm=arm, domain="wind").join(
        rows.select("site", "time", **{f"{arm}_init_time": "init_time"}),
        on=["site", "time"],
        how="left",
    )


def build_domain(
    *,
    domain: DomainType,
    published_dir: Path,
    output_dir: Path,
    weather_dir: Path,
    days: tuple[int, ...],
) -> pl.DataFrame:
    """Build one technology's WN3 inputs on the published inputs' own `(site, time)` keys.

    Nothing is written, so `main` can build both technologies before it writes either.

    Args:
        domain: `solar` or `wind`.
        published_dir: The folder holding the published `<domain>_forecast_inputs.parquet`.
        output_dir: The new folder to write `<domain>_wn3_inputs.parquet` into.
        weather_dir: The folder holding the local copy.
        days: The bands to build.

    Returns:
        The built frame.

    Raises:
        ValueError: If `output_dir` is `published_dir`, `days` is empty or holds a day below 0, an
            identity check fails, or a built column is null on every row.
        FileExistsError: If the output file already exists.
    """
    if output_dir.resolve() == published_dir.resolve():
        msg = f"the WN3 output must not be the published folder {published_dir}"
        raise ValueError(msg)
    if not days or min(days) < 0:
        msg = f"days must be a non-empty tuple of days from 0, got {days}"
        raise ValueError(msg)
    output_path = output_dir / f"{domain}_wn3_inputs.parquet"
    refuse_to_overwrite(paths=[output_path])
    keys = pl.read_parquet(published_dir / f"{domain}_forecast_inputs.parquet").select(
        "site", "time"
    )
    sites = sorted(keys["site"].unique().to_list())
    weights = aifs_site_weights(
        path=local_dir(weather_dir=weather_dir),
        domain=domain,
        sites=sites,
        spatial="h3",
        grid_degrees=GRID_DEGREES,
    )
    dataset = open_local(weather_dir=weather_dir)
    runs = dataset[fetch.INIT_TIME].to_numpy().astype("datetime64[h]")
    names = (
        (TEMPERATURE, RADIATION)
        if domain == "solar"
        else tuple(name for pair in WIND_COMPONENTS.values() for name in pair)
    )
    cubes = {
        name: site_cube(dataset=dataset, weights=weights, sites=sites, name=name) for name in names
    }
    frame = keys
    for day in days:
        arm_frame = wn3_arm_frame(
            cubes=cubes, runs=runs, sites=sites, keys=keys, domain=domain, day=day
        )
        check_physical_range(built=arm_frame, domain=domain, day=day)
        check_band_complete(built=arm_frame, domain=domain, day=day)
        check_against_store(
            built=arm_frame, dataset=dataset, weights=weights, domain=domain, day=day
        )
        frame = frame.join(arm_frame, on=["site", "time"], how="left")
    extra = _ens_extra_frame(
        keys=keys,
        domain=domain,
        mean_days=tuple(day for day in days if day in ENS_EXTRA_DAYS),
        control_days=(),
    )
    frame = frame.join(extra, on=["site", "time"], how="left")
    if domain == "wind":
        extract = ens_members(sites=sites)
        for day in days:
            frame = frame.join(
                ens_vector_mean_frame(extract=extract, day=day), on=["site", "time"], how="left"
            )
    empty = [
        column
        for column in frame.columns[keys.width :]
        if frame[column].null_count() == frame.height
    ]
    if empty:
        msg = f"{domain}: columns null on every row: {empty}"
        raise ValueError(msg)
    return frame


def write_domain(*, frame: pl.DataFrame, domain: DomainType, output_dir: Path) -> Path:
    """Write one technology's built frame into the write-once `output_dir`.

    Args:
        frame: `build_domain`'s result.
        domain: `solar` or `wind`.
        output_dir: The new folder.

    Returns:
        The written file's path.

    Raises:
        FileExistsError: If the output file already exists.
    """
    output_path = output_dir / f"{domain}_wn3_inputs.parquet"
    refuse_to_overwrite(paths=[output_path])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    _LOG.info("%s: wrote %d rows, %d columns to %s", domain, frame.height, frame.width, output_path)
    return output_path


def main() -> int:
    """Read the WN3 trial-area copy, or build the WN3 arm inputs from it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--read-store", action="store_true", help="Copy the trial area to disk.")
    stage.add_argument("--build", action="store_true", help="Build the arm inputs.")
    parser.add_argument("--bucket", help="With --read-store: the Cloud Storage bucket.")
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / "weather",
        help="The folder holding (or receiving) the WeatherNext3_trial_area copy.",
    )
    parser.add_argument(
        "--published-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / "nwp_forecast_comparison",
        help="With --build: the folder holding the published arm-input parquets.",
    )
    parser.add_argument("--output-dir", type=Path, help="With --build: the new folder to write.")
    parser.add_argument(
        "--days", type=int, nargs="+", default=list(WN3_DAYS), help="With --build: the bands."
    )
    args = parser.parse_args()
    if args.read_store:
        if not args.bucket:
            parser.error("--read-store needs --bucket")
        read_trial_area(bucket=args.bucket, weather_dir=args.weather_dir)
        return 0
    if args.output_dir is None:
        parser.error("--build needs --output-dir")
    domains: tuple[DomainType, ...] = ("solar", "wind")
    built = {
        domain: build_domain(
            domain=domain,
            published_dir=args.published_dir,
            output_dir=args.output_dir,
            weather_dir=args.weather_dir,
            days=tuple(args.days),
        )
        for domain in domains
    }
    for domain, frame in built.items():
        write_domain(frame=frame, domain=domain, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
