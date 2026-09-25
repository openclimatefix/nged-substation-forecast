"""Compare Open-Meteo's ICON-EU with DWD's ICON-EU as republished by Dynamical.org, at nine sites.

One-off throwaway script. The wind and radiation rankings in the past-weather and forecast studies
rest on Open-Meteo's processing of DWD ICON products. If Open-Meteo's ICON-EU matches DWD's own
values as republished by Dynamical.org, Open-Meteo's ICON-D2 is treated as safe too. The script is
read-only analysis: no model is fitted.

**Sources.** Dynamical.org's `dwd-icon-eu-forecast-5-day` (Icechunk Zarr on AWS Open Data, CC BY
4.0, 0.0625 degrees, runs at 00, 06, 12, and 18 UTC, hourly to 78 h) is opened with
`dynamical_catalog.open`, as `fetch_dynamical_zarr.py` does. Open-Meteo's values come from its
Single Runs API (`customer-single-runs-api.open-meteo.com`, model id `icon_eu`), which serves one
named model run per request, so a run is matched by its exact `init_time`. Open-Meteo's ICON-EU also
runs at 03, 09, 15, and 21 UTC; the Dynamical.org store holds no such runs, so they are not sampled.
Each run costs one request per site group, two in total (six photovoltaic sites with
`cell_selection=nearest`, three wind sites with `cell_selection=land`, the settings the studies'
own downloads used), so nine locations by four variables by 79 hours weighs nine calls.

**Sites.** The nine study sites, labelled `A`-`F` and `W1`-`W3` only. Coordinates are read from the
private roster at run time and held in memory. The nearest grid cell of the Dynamical.org store to
each site is chosen in memory as well, and neither the coordinates nor the cell indices reach a file
or a log line.

**Stages.** `runs` samples the runs and fetches both sides for each, checkpointing one parquet file
per run. `stitch` answers the freshest-run question against Open-Meteo's Previous Runs API.
`analyse` validates the cached data and writes the pairs, `summary.csv`, and the README. `all`
runs the three in order. `--max-runs` restricts `runs` to the last few sampled runs, for a trial.

Run it with `uv run python studies/weather_downloads/compare_icon_eu_dynamical_openmeteo.py --stage
all`, after exporting `OPEN_METEO_TOKEN` into the environment.
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import dynamical_catalog
import numpy as np
import polars as pl
import xarray as xr
from fetch_open_meteo_previous_runs import _get_json, _pv_sites, _wind_sites
from paths import REPO_DATA_DIR, open_meteo_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("compare_icon_eu")

StageType = Literal["runs", "stitch", "analyse", "all"]

OUTPUT_DIR: Final[Path] = REPO_DATA_DIR / "studies" / "icon_eu_compare"
RUN_CACHE_DIR: Final[Path] = OUTPUT_DIR / "_run_cache"
DYNAMICAL_CACHE_DIR: Final[Path] = OUTPUT_DIR / "_dynamical_cache"
STITCH_CACHE_DIR: Final[Path] = OUTPUT_DIR / "_stitch_cache"

DATASET_ID: Final[str] = "dwd-icon-eu-forecast-5-day"
SINGLE_RUNS_URL: Final[str] = "https://customer-single-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL: Final[str] = "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL: Final[str] = "icon_eu"
PREVIOUS_RUNS_MODEL: Final[str] = "dwd_icon_eu"
"""The Previous Runs API's identifier for the same model; a probe showed the Single Runs API
accepts both names and returns identical values."""

SINGLE_RUNS_FIRST_DATE: Final[datetime] = datetime(2026, 4, 2)  # noqa: DTZ001  (naive UTC, as numpy holds it)
"""Open-Meteo's Single Runs archive holds most models from this date, per its docs page."""

LAST_SAMPLED_RUN: Final[datetime] = datetime(2026, 9, 24, 18)  # noqa: DTZ001
"""The newest run the sample may hold: the latest 18 UTC run complete on 2026-09-25, when the
sample was pinned. Pinning it keeps the sample the same however late the script is re-run."""

LAST_STITCH_DAY: Final[str] = "2026-09-23"
"""The newest day the freshest-run test may sample, pinned for the same reason."""

FIRST_STITCH_DAY: Final[str] = "2026-04-06"
"""The first day the freshest-run test may sample: four days after the Single Runs archive starts,
so every candidate run for the day lies inside the overlap."""

INSTANT_SCORED_HOURS: Final[frozenset[int]] = frozenset({0, 1, 2})
"""Valid hours (modulo 6) whose freshest Open-Meteo run is a 00, 06, 12, or 18 UTC run, so one the
Dynamical.org store holds. Open-Meteo also runs at 03, 09, 15, and 21 UTC, which the store lacks,
and serves those runs' first leads for the other valid hours. Applies to 2 m temperature and 10 m
wind, which are instants."""

RADIATION_SCORED_HOURS: Final[frozenset[int]] = frozenset({1, 2, 3})
"""The same hours for radiation, shifted by one because radiation is stamped at the end of its
hour."""

MAX_LEAD_HOURS: Final[int] = 78
N_LEADS: Final[int] = MAX_LEAD_HOURS + 1
"""Hourly leads 0 to 78 h, the hourly part of the Dynamical.org store's lead axis."""

LEAD_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("0-6", 0, 6),
    ("6-24", 6, 24),
    ("24-48", 24, 48),
    ("48-78", 48, 79),
)
"""Half-open lead-hour ranges; the last one includes lead 78 h."""

SHIFTS_HOURS: Final[tuple[int, ...]] = (-1, 0, 1)
"""Offsets tested between the two sides' timestamps; the offset with the smallest RMS wins."""

OPEN_METEO_REQUEST_SLEEP_SECONDS: Final[float] = 0.5
REQUEST_HOURLY: Final[tuple[str, ...]] = (
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
)
VARIABLES: Final[tuple[str, ...]] = REQUEST_HOURLY
"""The four compared variables, named as Open-Meteo names them."""

RADIATION_VARIABLES: Final[tuple[str, ...]] = ("direct_radiation", "diffuse_radiation")
DYNAMICAL_RADIATION: Final[dict[str, str]] = {
    "direct_radiation": "downward_direct_short_wave_radiation_flux_surface",
    "diffuse_radiation": "downward_diffuse_short_wave_radiation_flux_surface",
}
DYNAMICAL_LOADED: Final[tuple[str, ...]] = (
    *DYNAMICAL_RADIATION.values(),
    "temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
)

MATCH_TOLERANCE: Final[dict[str, float]] = {
    "direct_radiation": 1.0,
    "diffuse_radiation": 1.0,
    "temperature_2m": 0.1,
    "wind_speed_10m": 0.1,
}
"""One rounding step of Open-Meteo's served value (integer W/m2, 0.1 degrees Celsius, 0.1 m/s), the
smallest difference the freshest-run test can call a mismatch."""

DAYLIGHT_THRESHOLD_W_M2: Final[float] = 10.0
"""Radiation pairs whose mean of the two sides is above this are the `daylight` subset."""

PLAUSIBLE_RANGE: Final[dict[str, tuple[float, float]]] = {
    "direct_radiation": (0.0, 1400.0),
    "diffuse_radiation": (0.0, 1400.0),
    "temperature_2m": (-40.0, 50.0),
    "wind_speed_10m": (0.0, 60.0),
}
"""Physical bounds checked by the validation step, in each variable's served unit."""


def _require_api_key() -> str:
    """Return the Open-Meteo API key from the environment, never logging it.

    Returns:
        The key.

    Raises:
        RuntimeError: If `OPEN_METEO_TOKEN` is not exported. The customer hosts need it.
    """
    key = open_meteo_api_key()
    if not key:
        msg = "export OPEN_METEO_TOKEN into the environment first (its value is never logged)"
        raise RuntimeError(msg)
    return key


def _sites() -> pl.DataFrame:
    """Return the nine study sites with the `cell_selection` each group is queried with.

    Returns:
        One row per site: `site`, `latitude`, `longitude`, `cell_selection`. Held in memory only.
    """
    return pl.concat(
        [
            _pv_sites().with_columns(cell_selection=pl.lit("nearest")),
            _wind_sites().with_columns(cell_selection=pl.lit("land")),
        ]
    )


# ---------------------------------------------------------------------------------------------
# Dynamical.org side
# ---------------------------------------------------------------------------------------------


def _open_store() -> xr.Dataset:
    """Open the Dynamical.org ICON-EU store lazily, keeping the loaded variables only."""
    return dynamical_catalog.open(DATASET_ID, chunks=None)[list(DYNAMICAL_LOADED)]


def _nearest_cell_indexers(*, store: xr.Dataset, sites: pl.DataFrame) -> dict[str, xr.DataArray]:
    """Return the `isel` indexers picking each site's nearest grid cell, in memory only.

    Args:
        store: The opened store.
        sites: The roster, carrying `site`, `latitude`, and `longitude`.

    Returns:
        Indexers for `latitude` and `longitude`, both along a `site` dimension.
    """
    latitudes = store["latitude"].to_numpy()
    longitudes = store["longitude"].to_numpy()
    lat_index = [int(np.argmin(np.abs(latitudes - value))) for value in sites["latitude"]]
    lon_index = [int(np.argmin(np.abs(longitudes - value))) for value in sites["longitude"]]
    return {
        "latitude": xr.DataArray(lat_index, dims="site"),
        "longitude": xr.DataArray(lon_index, dims="site"),
    }


def _init_times(*, store: xr.Dataset) -> np.ndarray:
    """Return every `init_time` in the store, ascending.

    The store's `ingested_forecast_length` coordinate holds no value, so a run's completeness is
    judged from the loaded values instead: see `_is_complete`.
    """
    return store["init_time"].to_numpy()


def _is_complete(*, frame: pl.DataFrame, prefix: str = "") -> bool:
    """Return whether one side of a loaded run holds a value at every site and lead.

    A missing value is `NaN` in the Dynamical.org store and null here. Radiation has no lead 0 on
    the Dynamical.org side, so one null per site is allowed there.

    Args:
        frame: One run's frame.
        prefix: `om_` to check Open-Meteo's columns, empty for the Dynamical.org columns.

    Returns:
        Whether every variable's null count is exactly what a complete run holds.
    """
    n_sites = frame["site"].n_unique()
    return all(
        frame[f"{prefix}{name}"].null_count() <= (n_sites if name in RADIATION_VARIABLES else 0)
        for name in VARIABLES
    )


def _dynamical_frame(
    *,
    store: xr.Dataset,
    indexers: dict[str, xr.DataArray],
    init_time: np.datetime64,
    labels: list[str],
) -> pl.DataFrame:
    """Load one run at the nine sites' cells, as a frame keyed by site label and valid time.

    Args:
        store: The opened store.
        indexers: The nearest-cell indexers from `_nearest_cell_indexers`.
        init_time: The run to load.
        labels: The site labels, in the order the indexers were built.

    Returns:
        Columns `site`, `init_time`, `valid_time`, and one column per compared variable, named as
        Open-Meteo names it. Radiation is the mean over the hour ending at `valid_time`, so lead 0
        is null. Wind speed is `hypot(u, v)`.

    Raises:
        ValueError: If the lead axis is not hourly from 0 to 78 h.
    """
    run = store.sel(init_time=init_time).isel(lead_time=slice(0, N_LEADS)).isel(indexers).load()
    leads = run["lead_time"].to_numpy()
    if not np.array_equal(leads, np.arange(N_LEADS) * np.timedelta64(1, "h")):
        msg = "the lead axis is not hourly from 0 to 78 h"
        raise ValueError(msg)
    valid_times = (init_time + leads).astype("datetime64[us]")
    columns: dict[str, np.ndarray] = {
        "site": np.repeat(labels, N_LEADS),
        "init_time": np.full(len(labels) * N_LEADS, init_time.astype("datetime64[us]")),
        "valid_time": np.tile(valid_times, len(labels)),
        "temperature_2m": run["temperature_2m"].transpose("site", "lead_time").to_numpy().ravel(),
        "wind_speed_10m": np.hypot(
            run["wind_u_10m"].transpose("site", "lead_time").to_numpy(),
            run["wind_v_10m"].transpose("site", "lead_time").to_numpy(),
        ).ravel(),
    }
    for name, source in DYNAMICAL_RADIATION.items():
        columns[name] = run[source].transpose("site", "lead_time").to_numpy().ravel()
    return (
        pl.DataFrame(columns)
        .cast(dict.fromkeys(VARIABLES, pl.Float64) | {"site": pl.Utf8})
        .with_columns(pl.col("init_time", "valid_time").cast(pl.Datetime("us")))
        .with_columns(pl.col(*VARIABLES).fill_nan(None))
    )


def _run_stamp(*, init_time: np.datetime64) -> str:
    """Return a filename-safe label for a run, e.g. `20260801T0600`."""
    return str(init_time.astype("datetime64[m]")).replace("-", "").replace(":", "")


def _cached_dynamical_frame(
    *,
    store: xr.Dataset,
    indexers: dict[str, xr.DataArray],
    init_time: np.datetime64,
    labels: list[str],
) -> pl.DataFrame:
    """Return one run's Dynamical.org frame, loading and checkpointing it on first use."""
    path = DYNAMICAL_CACHE_DIR / f"{_run_stamp(init_time=init_time)}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    frame = _dynamical_frame(store=store, indexers=indexers, init_time=init_time, labels=labels)
    _write_atomically(frame=frame, path=path)
    return frame


def _write_atomically(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file through a `.partial` name, so a killed write leaves no cached stub."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(path)


# ---------------------------------------------------------------------------------------------
# Open-Meteo side
# ---------------------------------------------------------------------------------------------


def _open_meteo_blocks(
    *, url: str, sites: pl.DataFrame, extra_query: str, variables: tuple[str, ...], model: str
) -> pl.DataFrame:
    """Request one site group from an Open-Meteo customer host and return it keyed by site label.

    Args:
        url: The customer endpoint.
        sites: One `cell_selection` group of the roster.
        extra_query: The query string tail selecting the run or the date range, without `&`.
        variables: The hourly variables, with any `_previous_dayN` suffixes.
        model: The `models=` value.

    Returns:
        Columns `site`, `time` (naive UTC), and one column per variable. Coordinates never enter.

    Raises:
        RuntimeError: If the response does not carry one block per site.
    """
    cell_selection = sites["cell_selection"][0]
    request = (
        f"{url}?latitude={','.join(str(value) for value in sites['latitude'])}"
        f"&longitude={','.join(str(value) for value in sites['longitude'])}"
        f"&{extra_query}&hourly={','.join(variables)}&models={model}&timezone=UTC"
        f"&wind_speed_unit=ms&cell_selection={cell_selection}&apikey={_require_api_key()}"
    )
    payload = _get_json(url=request)
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != sites.height:
        msg = f"asked for {sites.height} sites and got {len(blocks)} blocks"
        raise RuntimeError(msg)
    time.sleep(OPEN_METEO_REQUEST_SLEEP_SECONDS)
    return pl.concat(
        pl.DataFrame(
            {"site": site, "time": block["hourly"]["time"]}
            | {name: block["hourly"][name] for name in variables},
            schema_overrides=dict.fromkeys(variables, pl.Float64),
        )
        for site, block in zip(sites["site"], blocks, strict=True)
    ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us"))


def _single_run_frame(*, sites: pl.DataFrame, init_time: np.datetime64) -> pl.DataFrame:
    """Fetch one ICON-EU run from Open-Meteo's Single Runs API at every site.

    Args:
        sites: The roster with `cell_selection`.
        init_time: The run to request.

    Returns:
        Columns `site`, `valid_time`, and one `om_<variable>` column per compared variable, 79
        hourly rows per site starting at the run's own `init_time`.

    Raises:
        RuntimeError: If the API refuses the request, such as for a run it does not hold.
    """
    run_text = str(init_time.astype("datetime64[m]"))
    frames = [
        _open_meteo_blocks(
            url=SINGLE_RUNS_URL,
            sites=group,
            extra_query=f"run={run_text}&forecast_hours={N_LEADS}",
            variables=REQUEST_HOURLY,
            model=OPEN_METEO_MODEL,
        )
        for group in sites.partition_by("cell_selection")
    ]
    return (
        pl.concat(frames)
        .rename({"time": "valid_time"})
        .rename({name: f"om_{name}" for name in VARIABLES})
    )


# ---------------------------------------------------------------------------------------------
# Stage 1: sampled runs
# ---------------------------------------------------------------------------------------------


def _sample_runs(*, store: xr.Dataset, n_runs: int, seed: int) -> list[np.datetime64]:
    """Sample `n_runs` overlap runs, evenly over the four cycles, with a fixed seed.

    The overlap runs from Open-Meteo's Single Runs start to `LAST_SAMPLED_RUN`.

    Args:
        store: The opened store.
        n_runs: How many runs to sample in total.
        seed: The seed of the random generator.

    Returns:
        The sampled `init_time`s, ascending.
    """
    latest = np.datetime64(LAST_SAMPLED_RUN, "us")
    complete = _init_times(store=store)
    overlap = complete[
        (complete >= np.datetime64(SINGLE_RUNS_FIRST_DATE, "us")) & (complete <= latest)
    ]
    hours = overlap.astype("datetime64[h]").astype(np.int64) % 24
    generator = np.random.default_rng(seed)
    picked: list[np.datetime64] = []
    for cycle_hour, count in zip((0, 6, 12, 18), _split_evenly(total=n_runs, parts=4), strict=True):
        pool = generator.permutation(overlap[hours == cycle_hour])
        picked.extend(pool[:count])
    return sorted(picked)


def _split_evenly(*, total: int, parts: int) -> list[int]:
    """Split `total` into `parts` counts differing by at most one, larger counts first."""
    return [total // parts + (1 if index < total % parts else 0) for index in range(parts)]


def _run_runs_stage(*, n_runs: int, seed: int, max_runs: int | None) -> None:
    """Fetch both sides for each sampled run, checkpointing one parquet file per run.

    A run either side lacks is recorded in a `.skipped` marker file, so a re-run does not retry it.
    """
    sites = _sites()
    labels = sites["site"].to_list()
    store = _open_store()
    indexers = _nearest_cell_indexers(store=store, sites=sites)
    sampled = _sample_runs(store=store, n_runs=n_runs, seed=seed)
    if max_runs is not None:
        sampled = sampled[-max_runs:]
    _LOG.info("sampled %d runs, first %s, last %s", len(sampled), sampled[0], sampled[-1])
    for init_time in sampled:
        stamp = _run_stamp(init_time=init_time)
        target = RUN_CACHE_DIR / f"{stamp}.parquet"
        marker = RUN_CACHE_DIR / f"{stamp}.skipped"
        if target.exists() or marker.exists():
            _LOG.info("%s: already done, skipping", stamp)
            continue
        RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dynamical = _cached_dynamical_frame(
            store=store, indexers=indexers, init_time=init_time, labels=labels
        )
        if not _is_complete(frame=dynamical):
            marker.write_text("dynamical_run_incomplete")
            _LOG.warning("%s: the Dynamical.org run is incomplete, skipped", stamp)
            continue
        try:
            open_meteo = _single_run_frame(sites=sites, init_time=init_time)
        except RuntimeError as refusal:
            if "not available" not in str(refusal):
                raise
            marker.write_text("open_meteo_run_not_available")
            _LOG.warning("%s: Open-Meteo does not hold this run, skipped", stamp)
            continue
        if not _is_complete(frame=open_meteo, prefix="om_"):
            marker.write_text("open_meteo_run_null_filled")
            _LOG.warning("%s: Open-Meteo returned nulls for this run, skipped", stamp)
            continue
        joined = dynamical.join(open_meteo, on=["site", "valid_time"], how="full", coalesce=True)
        _write_atomically(frame=joined.sort("site", "valid_time"), path=target)
        _LOG.info("%s: fetched", stamp)


# ---------------------------------------------------------------------------------------------
# Stage 2: freshest-run question
# ---------------------------------------------------------------------------------------------


def _stitch_days(*, n_days: int, seed: int) -> list[np.datetime64]:
    """Sample `n_days` days between `FIRST_STITCH_DAY` and `LAST_STITCH_DAY`, fixed by the seed."""
    days = np.arange(
        np.datetime64(FIRST_STITCH_DAY), np.datetime64(LAST_STITCH_DAY) + np.timedelta64(1, "D")
    )
    return sorted(np.random.default_rng(seed).permutation(days)[:n_days])


def _run_stitch_stage(*, n_days: int, seed: int) -> None:
    """Compare Open-Meteo's stitched day-0 series with every Dynamical.org run covering each hour.

    Open-Meteo's Previous Runs API serves, at each valid hour, the value from the freshest model run
    it held (`<variable>` with no suffix). For `n_days` sampled days, every Dynamical.org run
    whose leads 0 to 78 h include the day is loaded, and each hour's Open-Meteo value is tested
    against each candidate run's value at that hour.
    """
    sites = _sites()
    labels = sites["site"].to_list()
    store = _open_store()
    indexers = _nearest_cell_indexers(store=store, sites=sites)
    complete = _init_times(store=store)
    for day in _stitch_days(n_days=n_days, seed=seed):
        target = STITCH_CACHE_DIR / f"{day}.parquet"
        if target.exists():
            _LOG.info("stitch %s: already cached, skipping", day)
            continue
        day_start = day.astype("datetime64[us]")
        candidates = complete[
            (complete > day_start - np.timedelta64(MAX_LEAD_HOURS, "h"))
            & (complete < day_start + np.timedelta64(24, "h"))
        ]
        loaded = [
            _cached_dynamical_frame(
                store=store, indexers=indexers, init_time=init_time, labels=labels
            )
            for init_time in candidates
        ]
        kept = [frame for frame in loaded if _is_complete(frame=frame)]
        _LOG.info(
            "stitch %s: %d Dynamical.org candidate runs in the store (%d expected at 6-hourly "
            "spacing), %d dropped as incomplete",
            day,
            len(loaded),
            (MAX_LEAD_HOURS + 24) // 6,
            len(loaded) - len(kept),
        )
        dynamical = pl.concat(kept).filter(pl.col("valid_time").dt.date() == day.astype(object))
        served = pl.concat(
            _open_meteo_blocks(
                url=PREVIOUS_RUNS_URL,
                sites=group,
                extra_query=f"start_date={day}&end_date={day}",
                variables=REQUEST_HOURLY,
                model=PREVIOUS_RUNS_MODEL,
            )
            for group in sites.partition_by("cell_selection")
        ).rename({"time": "valid_time"}, strict=True)
        candidate_long = dynamical.unpivot(
            on=list(VARIABLES),
            index=["site", "init_time", "valid_time"],
            variable_name="variable",
            value_name="dynamical",
        )
        served_long = served.unpivot(
            on=list(VARIABLES),
            index=["site", "valid_time"],
            variable_name="variable",
            value_name="open_meteo",
        )
        joined = candidate_long.join(served_long, on=["site", "valid_time", "variable"])
        _write_atomically(frame=joined, path=target)
        _LOG.info("stitch %s: %d candidate rows", day, joined.height)


def _freshest_run_verdicts(*, stitched: pl.DataFrame) -> pl.DataFrame:
    """Classify each scorable (site, variable, valid hour) by the run Open-Meteo's value matches.

    **Only hours whose freshest Open-Meteo run the store holds are scored.** Open-Meteo also runs
    ICON-EU at 03, 09, 15, and 21 UTC, which the Dynamical.org store lacks, so at other valid hours
    the freshest Open-Meteo run is one no candidate can match. `INSTANT_SCORED_HOURS` and
    `RADIATION_SCORED_HOURS` keep the hours where it is a run the store holds.

    **An hour is informative only if the freshest candidate differs from the second-freshest by
    more than two match tolerances**, one rounding step of Open-Meteo's served value each. Without
    that gap a match to the freshest run cannot be told from a match to the previous one. The
    freshest candidate is the one with the shortest lead; radiation has no lead 0, so its freshest
    candidate has lead 1 h at the earliest.

    Args:
        stitched: Rows of `site`, `init_time`, `valid_time`, `variable`, `dynamical`, `open_meteo`.

    Returns:
        One row per scorable informative hour with `variable`, `hour_in_cycle` (the valid hour
        modulo 6), `verdict` (`freshest`, `older`, or `none_match`), and `lag_hours`, the freshest
        candidate's `init_time` minus the best candidate's, in hours (0 when the freshest matches).
    """
    tolerance = pl.col("variable").replace_strict(MATCH_TOLERANCE, return_dtype=pl.Float64)
    hour_in_cycle = pl.col("valid_time").dt.hour() % 6
    is_scored = (
        pl.when(pl.col("variable").is_in(RADIATION_VARIABLES))
        .then(hour_in_cycle.is_in(RADIATION_SCORED_HOURS))
        .otherwise(hour_in_cycle.is_in(INSTANT_SCORED_HOURS))
    )
    rows = (
        stitched.drop_nulls("dynamical")
        .with_columns(
            tolerance=tolerance,
            error=(pl.col("open_meteo") - pl.col("dynamical")).abs(),
            init_hour=pl.col("init_time").dt.epoch("s") / 3600,
            hour_in_cycle=hour_in_cycle,
        )
        .filter(is_scored)
    )
    by_freshness = pl.col("dynamical").sort_by("init_hour", descending=True)
    return (
        rows.group_by("site", "variable", "valid_time")
        .agg(
            tolerance=pl.col("tolerance").first(),
            hour_in_cycle=pl.col("hour_in_cycle").first(),
            freshest_value=by_freshness.first(),
            second_value=by_freshness.get(1, null_on_oob=True),
            freshest_hour=pl.col("init_hour").max(),
            freshest_error=pl.col("error").sort_by("init_hour").last(),
            best_error=pl.col("error").min(),
            best_hour=pl.col("init_hour").filter(pl.col("error") == pl.col("error").min()).max(),
        )
        .filter((pl.col("freshest_value") - pl.col("second_value")).abs() > 2 * pl.col("tolerance"))
        .with_columns(
            verdict=pl.when(pl.col("freshest_error") <= pl.col("tolerance"))
            .then(pl.lit("freshest"))
            .when(pl.col("best_error") <= pl.col("tolerance"))
            .then(pl.lit("older"))
            .otherwise(pl.lit("none_match")),
            lag_hours=(pl.col("freshest_hour") - pl.col("best_hour")).cast(pl.Int64),
        )
        .select("variable", "hour_in_cycle", "verdict", "lag_hours")
    )


def _freshest_run_summary(*, verdicts: pl.DataFrame) -> pl.DataFrame:
    """Summarise verdicts per variable and hour within the cycle, with an `all` row per variable.

    Args:
        verdicts: The frame from `_freshest_run_verdicts`.

    Returns:
        Per group: hour count, the shares of each verdict, and the median lag of `older` hours.
    """
    aggregations = [
        pl.len().alias("n_informative_hours"),
        (pl.col("verdict") == "freshest").mean().alias("share_freshest"),
        (pl.col("verdict") == "older").mean().alias("share_older"),
        (pl.col("verdict") == "none_match").mean().alias("share_none_match"),
        pl.col("lag_hours")
        .filter(pl.col("verdict") == "older")
        .median()
        .alias("median_lag_hours_when_older"),
    ]
    per_hour = verdicts.group_by("variable", "hour_in_cycle").agg(aggregations)
    overall = (
        verdicts.group_by("variable").agg(aggregations).with_columns(hour_in_cycle=pl.lit(None))
    )
    return (
        pl.concat([per_hour, overall.select(per_hour.columns)])
        .with_columns(
            hour_in_cycle=pl.col("hour_in_cycle").cast(pl.Utf8).fill_null("all"),
        )
        .sort("variable", "hour_in_cycle")
    )


# ---------------------------------------------------------------------------------------------
# Stage 3: validation, pairs, summary, README
# ---------------------------------------------------------------------------------------------


def _read_runs(*, sampled: list[np.datetime64]) -> tuple[pl.DataFrame, dict[str, int]]:
    """Read the cached runs among the sampled ones, and count what the sample lost.

    Runs cached under other stamps, such as a trial with a different sample, are ignored.

    Args:
        sampled: The sampled `init_time`s.

    Returns:
        The runs' frame, and counts of the sampled runs by outcome: `compared`, one key per skip
        reason, and `not_fetched`.

    Raises:
        RuntimeError: If no sampled run is cached.
    """
    counts = {"compared": 0, "not_fetched": 0}
    paths: list[Path] = []
    for init_time in sampled:
        stamp = _run_stamp(init_time=init_time)
        marker = RUN_CACHE_DIR / f"{stamp}.skipped"
        if (RUN_CACHE_DIR / f"{stamp}.parquet").exists():
            paths.append(RUN_CACHE_DIR / f"{stamp}.parquet")
            counts["compared"] += 1
        elif marker.exists():
            reason = marker.read_text()
            counts[reason] = counts.get(reason, 0) + 1
        else:
            counts["not_fetched"] += 1
    if not paths:
        msg = f"none of the {len(sampled)} sampled runs is cached; run the `runs` stage first"
        raise RuntimeError(msg)
    return pl.concat(pl.read_parquet(path) for path in paths), counts


def _validate_runs(*, runs: pl.DataFrame) -> None:
    """Raise if the cached runs break a shape, completeness, or plausibility expectation.

    Checks, per run, that every site has 79 hourly rows and that each side is complete
    (`_is_complete`); and that every value of each side lies inside `PLAUSIBLE_RANGE`.

    Args:
        runs: The compared runs, from `_read_runs`.

    Raises:
        ValueError: On the first failed expectation, naming it and its measured value.
    """
    counts = runs.group_by("init_time", "site").len()["len"]
    if counts.min() != N_LEADS or counts.max() != N_LEADS:
        msg = (
            f"expected {N_LEADS} hourly rows per run and site, found {counts.min()}-{counts.max()}"
        )
        raise ValueError(msg)
    for init_time, run in runs.partition_by("init_time", as_dict=True).items():
        for prefix in ("", "om_"):
            if not _is_complete(frame=run, prefix=prefix):
                msg = f"the run at {init_time} has nulls on the {prefix or 'dynamical_'} side"
                raise ValueError(msg)
    for name in VARIABLES:
        low, high = PLAUSIBLE_RANGE[name]
        for side, column in (("dynamical", name), ("open_meteo", f"om_{name}")):
            values = runs[column].drop_nulls().to_numpy()
            lowest, highest = float(np.min(values)), float(np.max(values))
            if lowest < low - 1e-6 or highest > high:
                msg = f"{side} {name} spans {lowest:.2f} to {highest:.2f}, outside {low} to {high}"
                raise ValueError(msg)


def _shift_open_meteo(*, runs: pl.DataFrame, hours: int) -> pl.DataFrame:
    """Pair each Dynamical.org hour with Open-Meteo's value `hours` later, per run and site."""
    ordered = runs.sort("init_time", "site", "valid_time")
    return ordered.with_columns(
        pl.col(f"om_{name}").shift(-hours).over("init_time", "site") for name in VARIABLES
    )


def _pairs_at_shift(*, runs: pl.DataFrame, hours: int) -> pl.DataFrame:
    """Return the long pairs frame at one timestamp offset between the sides."""
    shifted = _shift_open_meteo(runs=runs, hours=hours)
    long_frames = [
        shifted.select(
            "site",
            "init_time",
            "valid_time",
            lead_hours=((pl.col("valid_time") - pl.col("init_time")).dt.total_hours()),
            variable=pl.lit(name),
            dynamical=pl.col(name),
            open_meteo=pl.col(f"om_{name}"),
        )
        for name in VARIABLES
    ]
    return (
        pl.concat(long_frames)
        .drop_nulls(["dynamical", "open_meteo"])
        .with_columns(pl.col("lead_hours").cast(pl.Int64))
    )


def _shift_table(*, runs: pl.DataFrame) -> pl.DataFrame:
    """Return the RMS difference per variable at each tested offset."""
    tables = []
    for hours in SHIFTS_HOURS:
        pairs = _pairs_at_shift(runs=runs, hours=hours)
        tables.append(
            pairs.group_by("variable")
            .agg(
                rms=((pl.col("open_meteo") - pl.col("dynamical")).pow(2).mean()).sqrt(),
                n=pl.len(),
            )
            .with_columns(shift_hours=pl.lit(hours))
        )
    return pl.concat(tables).sort("variable", "shift_hours")


def _best_shifts(*, shift_table: pl.DataFrame) -> dict[str, int]:
    """Return, for each variable, the offset with the smallest RMS difference."""
    best = shift_table.sort("rms").group_by("variable", maintain_order=True).first()
    return {str(row["variable"]): int(row["shift_hours"]) for row in best.to_dicts()}


def _chosen_pairs(*, runs: pl.DataFrame, shifts: dict[str, int]) -> pl.DataFrame:
    """Return the long pairs frame with each variable at its own best offset."""
    return pl.concat(
        _pairs_at_shift(runs=runs, hours=hours).filter(
            pl.col("variable").is_in([name for name, chosen in shifts.items() if chosen == hours])
        )
        for hours in sorted(set(shifts.values()))
    ).sort("variable", "site", "init_time", "valid_time")


def _metrics(*, pairs: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return count and difference statistics of `open_meteo - dynamical` per group.

    `ratio_of_means` is null where the Dynamical.org mean is not above the daylight threshold
    (radiation) or above zero (other variables), and `correlation` is null where either side is
    constant, as in a night-only lead bucket.
    """
    difference = pl.col("open_meteo") - pl.col("dynamical")
    floor = (
        pl.when(pl.col("variable").first().is_in(RADIATION_VARIABLES))
        .then(pl.lit(DAYLIGHT_THRESHOLD_W_M2))
        .otherwise(pl.lit(0.0))
    )
    return pairs.group_by(by).agg(
        n=pl.len(),
        mean_abs_diff=difference.abs().mean(),
        rms_diff=difference.pow(2).mean().sqrt(),
        mean_signed_diff=difference.mean(),
        max_abs_diff=difference.abs().max(),
        correlation=pl.corr("open_meteo", "dynamical").fill_nan(None),
        ratio_of_means=pl.when(pl.col("dynamical").mean() > floor)
        .then(pl.col("open_meteo").mean() / pl.col("dynamical").mean())
        .otherwise(None),
    )


def _summary(*, pairs: pl.DataFrame) -> pl.DataFrame:
    """Return metrics per variable overall, per site, and per lead bucket, and for daylight."""
    bucketed = pairs.with_columns(
        lead_bucket=pl.when(pl.col("lead_hours") < LEAD_BUCKETS[0][2])
        .then(pl.lit(LEAD_BUCKETS[0][0]))
        .when(pl.col("lead_hours") < LEAD_BUCKETS[1][2])
        .then(pl.lit(LEAD_BUCKETS[1][0]))
        .when(pl.col("lead_hours") < LEAD_BUCKETS[2][2])
        .then(pl.lit(LEAD_BUCKETS[2][0]))
        .otherwise(pl.lit(LEAD_BUCKETS[3][0])),
        all=pl.lit("all"),
    )
    parts = []
    for subset, frame in (
        ("all", bucketed),
        (
            "daylight",
            bucketed.filter(
                pl.col("variable").is_in(RADIATION_VARIABLES)
                & ((pl.col("dynamical") + pl.col("open_meteo")) / 2 > DAYLIGHT_THRESHOLD_W_M2)
            ),
        ),
    ):
        parts.extend(
            _metrics(pairs=frame, by=["variable", kind])
            .rename({kind: "group"})
            .with_columns(subset=pl.lit(subset), grouping=pl.lit(kind))
            for kind in ("all", "site", "lead_bucket")
        )
    return (
        pl.concat(parts)
        .select(
            "variable",
            "subset",
            "grouping",
            "group",
            "n",
            "mean_abs_diff",
            "rms_diff",
            "mean_signed_diff",
            "max_abs_diff",
            "correlation",
            "ratio_of_means",
        )
        .sort("variable", "subset", "grouping", "group")
    )


def _markdown_table(*, frame: pl.DataFrame) -> str:
    """Render a small frame as a Markdown table, floats to four significant figures."""
    header = (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join("---" for _ in frame.columns)
        + " |"
    )
    lines = [
        "| "
        + " | ".join(f"{cell:.4g}" if isinstance(cell, float) else str(cell) for cell in row)
        + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, *lines])


def _write_readme(
    *,
    runs: pl.DataFrame,
    outcomes: dict[str, int],
    first_run: str,
    shift_table: pl.DataFrame,
    shifts: dict[str, int],
    summary: pl.DataFrame,
    freshest: pl.DataFrame | None,
) -> None:
    """Write the generated README, stating sources, licence, timing conventions, and findings."""
    overall = summary.filter((pl.col("grouping") == "all") & (pl.col("subset") == "all")).drop(
        "subset", "grouping", "group"
    )
    n_sampled = sum(outcomes.values())
    n_runs = outcomes["compared"]
    outcome_text = ", ".join(f"{count} {reason}" for reason, count in sorted(outcomes.items()))
    freshest_text = (
        _markdown_table(frame=freshest)
        if freshest is not None
        else "The `stitch` stage has not been run."
    )
    shifts_text = ", ".join(f"{name} {hours:+d} h" for name, hours in sorted(shifts.items()))
    text = f"""# Open-Meteo ICON-EU against Dynamical.org ICON-EU

Generated by `studies/weather_downloads/compare_icon_eu_dynamical_openmeteo.py`. Nothing here
holds a coordinate or a grid-cell index: sites are labelled `A`-`F` (photovoltaic) and `W1`-`W3`
(wind).

## Sources and licence

- Dynamical.org `dwd-icon-eu-forecast-5-day`, DWD ICON-EU republished on a 0.0625 degree grid, runs
  at 00, 06, 12, and 18 UTC, hourly to 78 h. Licence CC BY 4.0. Attribution: DWD ICON-EU data
  processed by dynamical.org. The store's first run is {first_run}.
  <https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/>
- Open-Meteo's Single Runs API, model `icon_eu`, one named run per request, archive from 2 April
  2026. Two requests per run (one per `cell_selection` group), weighing nine calls.
  <https://open-meteo.com/en/docs/single-runs-api>. The Previous Runs API (`dwd_icon_eu`) serves the
  stitched series for the freshest-run question.

## Design

- {n_sampled} runs were sampled with a fixed seed, evenly over the four cycles, from the overlap of
  the two archives, ending at the pinned run `{LAST_SAMPLED_RUN:%Y-%m-%d %H:%M}` UTC. {n_runs} are
  compared. Outcomes: {outcome_text}. A run is skipped when Dynamical.org's values are incomplete
  or Open-Meteo lacks the run or returns nulls for it.
- Each run compares leads 0 to 78 h at each site's nearest Dynamical.org grid cell. Open-Meteo is
  queried at the site's own coordinates with `cell_selection=nearest` (photovoltaic sites) or `land`
  (wind sites), and `wind_speed_unit=ms`, so a cell chosen by Open-Meteo can differ from the
  nearest cell of the Dynamical.org grid.
- Differences are `open_meteo - dynamical`. `ratio_of_means` is the mean Open-Meteo value over the
  mean Dynamical.org value, which exposes a scaling.

## Variables and timing conventions

- Direct and diffuse radiation are horizontal-plane fluxes in W/m2. Dynamical.org's are DWD's
  `ASWDIR_S` and `ASWDIFD_S`, the average flux since the previous forecast step, stamped at the end
  of the hour, so lead 0 is null. Open-Meteo derives its `direct_radiation` and `diffuse_radiation`
  from the same DWD fields and serves a mean over the hour ending at the stamp.
- Temperature at 2 m (degrees Celsius) and 10 m wind speed (m/s) are instants on both sides.
  Dynamical.org's wind speed is `hypot(u, v)` of its 10 m components.
- Alignment was tested at Open-Meteo offsets of -1, 0, and +1 h. The chosen offset per variable is
  the one with the smallest RMS difference: {shifts_text}.

{_markdown_table(frame=shift_table)}

## Rounding noise floor

Open-Meteo serves rounded values, and Dynamical.org's are unrounded, so a perfect match still has
a difference. Rounding to a step of `h` gives an expected RMS difference of `h / sqrt(12)`: 0.029
m/s for wind (step 0.1 m/s), 0.029 degrees Celsius for temperature (step 0.1), and 0.29 W/m2 for
radiation (step 1). An RMS difference near these values means a match to within rounding.

## Results over all pairs

{_markdown_table(frame=overall)}

`ratio_of_means` is blank where the Dynamical.org mean is not above 10 W/m2 (radiation) or 0
(other variables), and `correlation` is blank where either side is constant.
`summary.csv` holds the same statistics per site, per lead bucket (0-6, 6-24, 24-48, and 48-78 h),
and for a daylight subset of the radiation pairs.

## Freshest-run question

For sampled days, Open-Meteo's Previous Runs day-0 series is compared with the value each
Dynamical.org run covering that hour holds. An hour counts only if the freshest candidate run
differs from the second-freshest by more than two rounding steps of Open-Meteo's served value (0.1
m/s wind, 1 W/m2 radiation, 0.1 degrees Celsius temperature), because otherwise a match to the
freshest run cannot be told from a match to the previous one. `freshest` means the freshest
Dynamical.org run matches within one rounding step, `older` means only an older run does, and
`none_match` means no Dynamical.org run does. Open-Meteo also runs ICON-EU at 03, 09, 15, and 21
UTC, which the Dynamical.org store lacks. **Only the hours whose freshest Open-Meteo run the store
holds are scored**: valid hours 0, 1, and 2 modulo 6 for temperature and wind, and hours 1, 2, and
3 modulo 6 for radiation, which is stamped at the end of its hour. At other hours the freshest
Open-Meteo run is one no candidate can match. `hour_in_cycle` is the valid hour modulo 6.

{freshest_text}

## Files

- `pairs.parquet`: private, never to be published. Matched pairs keyed by `site`, `init_time`,
  `valid_time`, and `variable`, with `lead_hours`, `dynamical`, and `open_meteo`, at each
  variable's chosen offset.
- `summary.csv`: the statistics above.
- `freshest_run_summary.csv`: the freshest-run table (after the `stitch` stage).

Everything here stays under the private `data/` tree. `pairs.parquet` holds each site's time
series under an anonymous label and must not be published.
"""
    (OUTPUT_DIR / "README.md").write_text(text)


def _run_analyse_stage(*, n_runs: int, n_days: int, seed: int) -> None:
    """Validate the cached runs, then write the pairs, summary, and README."""
    store = _open_store()
    sampled = _sample_runs(store=store, n_runs=n_runs, seed=seed)
    runs, outcomes = _read_runs(sampled=sampled)
    if outcomes["not_fetched"]:
        _LOG.warning("%d sampled runs have not been fetched yet", outcomes["not_fetched"])
    _validate_runs(runs=runs)
    shift_table = _shift_table(runs=runs)
    shifts = _best_shifts(shift_table=shift_table)
    pairs = _chosen_pairs(runs=runs, shifts=shifts)
    summary = _summary(pairs=pairs)
    _write_atomically(frame=pairs, path=OUTPUT_DIR / "pairs.parquet")
    summary.write_csv(OUTPUT_DIR / "summary.csv")
    stitch_paths = [
        path
        for day in _stitch_days(n_days=n_days, seed=seed)
        if (path := STITCH_CACHE_DIR / f"{day}.parquet").exists()
    ]
    freshest = None
    if stitch_paths:
        verdicts = _freshest_run_verdicts(
            stitched=pl.concat(pl.read_parquet(p) for p in stitch_paths)
        )
        freshest = _freshest_run_summary(verdicts=verdicts)
        freshest.write_csv(OUTPUT_DIR / "freshest_run_summary.csv")
    first_run = str(store["init_time"].to_numpy().min().astype("datetime64[m]"))
    _write_readme(
        runs=runs,
        outcomes=outcomes,
        first_run=first_run,
        shift_table=shift_table,
        shifts=shifts,
        summary=summary,
        freshest=freshest,
    )
    (OUTPUT_DIR / "lineage.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "dynamical_dataset": DATASET_ID,
                "open_meteo_endpoints": ["single-runs-api", "previous-runs-api"],
                "run_outcomes": outcomes,
                "stitch_days_used": len(stitch_paths),
                "chosen_shift_hours": shifts,
            },
            indent=2,
        )
    )
    _LOG.info("wrote %d pairs, summary.csv, and README.md to %s", pairs.height, OUTPUT_DIR)


def main() -> int:
    """Run the requested stage or stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("runs", "stitch", "analyse", "all"), required=True)
    parser.add_argument("--n-runs", type=int, default=40)
    parser.add_argument("--stitch-days", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int, help="Fetch only the last N sampled runs (trial).")
    arguments = parser.parse_args()
    stage: StageType = arguments.stage
    if stage in ("runs", "all"):
        _run_runs_stage(n_runs=arguments.n_runs, seed=arguments.seed, max_runs=arguments.max_runs)
    if stage in ("stitch", "all"):
        _run_stitch_stage(n_days=arguments.stitch_days, seed=arguments.seed)
    if stage in ("analyse", "all"):
        _run_analyse_stage(
            n_runs=arguments.n_runs, n_days=arguments.stitch_days, seed=arguments.seed
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
