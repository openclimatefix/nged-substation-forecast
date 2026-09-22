"""Join NGED PV power to ERA5 irradiance and write the one frame every arm of the experiment reads.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. It is deliberately outside
the Dagster asset graph, builds its frames locally rather than routing ERA5 through
`contracts.weather_schemas.Nwp`, and writes nothing the rest of the repo reads.

The whole experiment turns on every arm seeing identical rows, so all the filtering, resampling and
solar-geometry work happens here, once, before the arms exist. `run_experiment.py` then chooses
which irradiance columns to show the model and changes nothing else.

Run it with `uv run --no-project` plus `--with pvlib --with polars --with deltalake --with
xarray --with netcdf4`.

Three conventions have to line up and are the easiest thing to get wrong:

- **NGED power is period-ending**: the reading stamped `T` is the mean MW over `(T - 30 min, T]`.
- **ERA5 `ssrd` and `fdir` are accumulations in J m⁻² over the preceding hour**, stamped at the end
  of that hour, so the value at `T` covers `(T - 1 h, T]`. Dividing by 3600 gives the mean W m⁻²
  over the same window.
- So an hourly row stamped `T` pairs the ERA5 hour ending `T` with the mean of the two NGED
  half-hours ending `T - 30 min` and `T`, and solar geometry is evaluated at the window's midpoint,
  `T - 30 min`.

`fdir` is the direct beam's flux onto a *horizontal* plane, not direct normal irradiance, so the
diffuse flux onto a horizontal plane is `ssrd - fdir` with no cosine anywhere.
"""

import argparse
import logging
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl

# pvlib is not a workspace dependency; this throwaway script is run with `uv run --with pvlib`.
import pvlib  # ty: ignore[unresolved-import]
import xarray as xr
from era5_grid import LAST_DATE
from sources import (
    OPEN_METEO_MODELS,
    PER_SITE_SOURCES,
    REPO_DATA_DIR,
    SOURCE_CHOICES,
    PointTemporalType,
    SourceType,
    point_output_path_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_dataset")

ERA5_DIR: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse"
POWER_DELTA_URI: Final[str] = str(REPO_DATA_DIR / "NGED" / "power_time_series.delta")
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
CAPACITY_DELTA_URI: Final[str] = str(REPO_DATA_DIR / "effective_capacity")
OPEN_METEO_PATH: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_open_meteo.parquet"
CAMS_PATH: Final[Path] = REPO_DATA_DIR / "CAMS" / "beam_diffuse_cams.parquet"
AlignmentType = Literal["as-labelled", "shifted", "piecewise"]
"""How the power stamps are read against ERA5.

`as-labelled` takes `PowerTimeSeries.time` at its word: the reading stamped `T` is the mean over
`(T - 30 min, T]`. `shifted` moves every power stamp 30 minutes earlier before the join.
`piecewise` moves only the stamps before `ALIGNMENT_FIXED_AT`, which is the correct treatment:
NGED corrected the feed at that instant, so the stamps before it are half an hour late and the
stamps after it are not.

**Three independent tests say the stamps arrive half an hour late, which is why the second reading
exists.** Against the sun's own horizon crossings the first and last generating half-hour of a clear
day both fall 30 minutes later than they should — a truncation at low sun would move those two edges
towards each other rather than shifting both the same way. The power-weighted centroid of a clear
day runs 0.45 hours late. And the correlation with ERA5 global irradiance is maximised at a
30-minute shift for all six meters, in every year. It is not a daylight-saving bug: the centroid
offset is flat across the March and October boundaries, where a local-time error would step by a
whole hour.

The experiment is run both ways rather than one, because a half-hour misalignment blunts the beam
component more than the diffuse one — beam is the sharper, faster-varying signal — so it penalises
exactly the arm under test. That is an asymmetric error, and a paired design gives no protection
against it. Whether the contract or the feed is at fault is a question for NGED and is not settled
here.
"""

ALIGNMENT_FIXED_AT: Final[datetime] = datetime(2026, 3, 26, 8, 30, tzinfo=UTC)
"""The instant NGED corrected the half-hourly stamps.

Every reading stamped before this instant is half an hour late: its value is the mean over
`(T - 60 min, T - 30 min]` rather than the `(T - 30 min, T]` the contract states. Readings from this
instant on are correct as stamped. NGED reported the correction, and three independent measurements
agree with it: the power-weighted centroid of a clear day's output steps from 45 minutes after solar
noon to 15 minutes at this stamp, the feed stops publishing exact-zero rows at the same instant, and
the published rows per day drop from 48 to 26.
"""


def output_path_for(*, dataset_name: str, alignment: AlignmentType) -> Path:
    """Return where one built frame is written.

    Args:
        dataset_name: The irradiance source, plus any suffix distinguishing a variant build from
            the main one for the same source.
        alignment: Which stamp alignment the frame was built under.

    Returns:
        The parquet path every arm of that run reads.
    """
    return REPO_DATA_DIR / "ERA5" / f"beam_diffuse_dataset_{dataset_name}_{alignment}.parquet"


MIN_YEARS_OF_READINGS: Final[float] = 1.0
"""A PV series with less than this much history is dropped.

The roster holds seven PV series, one of which is a single row; the threshold exists to drop that
one without naming it.
"""

OUTAGE_RUN_HOURS: Final[int] = 24
"""A run of exactly-zero hourly power at least this long is treated as an outage and dropped.

Twenty-four hours necessarily spans a daylight period, so a PV site cannot produce such a run by
physics. Reading only the power column does not by itself make the filter neutral between the arms,
because power is a function of irradiance: a short, heavily overcast winter day whose output is
exactly zero joins the long nights either side and can clear 24 hours, and an overcast day is the
high-diffuse regime one arm exists to exploit. What makes it acceptable is the size of the tilt it
introduces, which `main` logs: the filter removes about 1.2% of daylight rows, a little over 3% of
December rows against 1.5% of June rows.
"""

FALSE_ZERO_IRRADIANCE_W_M2: Final[float] = 100.0
"""An hour with an exactly-zero half-hour and more global irradiance than this is dropped.

A grid-connected PV site under 100 W m⁻² of global horizontal irradiance produces a small positive
output, never an exact zero, so an exact zero there is a meter dropout. NGED's own delivery spec
names the condition: a `GENERATOR OR CIRCUIT FAULT` warning is raised "whenever metered generation
is zero when the generator should be generating (e.g. a solar farm reading zero at midday on a sunny
day)". Estimating each dropped reading's expected output from its own site's median capacity factor
in the same irradiance and elevation bin puts the mean at 0.37 to 0.76 of capacity: these are holes,
not dim conditions.

**The test is "either half-hour reads zero", not "the hourly mean reads zero", because the average
of one dropped half-hour and one real one is the dangerous case** — it reads plausible and is wrong
by about half the site's output. That corrupted-average case is roughly two thirds of what this rule
removes.

The rule has to be blind to the beam/diffuse split, or it would bias the very comparison the
experiment makes. Matched within irradiance bins, the rows it removes differ from the rows it keeps
by 0.010 in diffuse fraction, against a 0.24-to-0.75 range across irradiance bins — the most neutral
of every variant tested. Conditioning on irradiance alone makes the *unmatched* comparison
misleading in both directions, so neutrality has to be judged within bins.
"""

IMPLAUSIBLE_CAPACITY_MULTIPLE: Final[float] = 1.5
"""Readings above this multiple of the site's effective capacity are dropped as meter spikes.

Nothing in seven years reaches it, so the rule removes no rows today and is kept only as a guard.
A tighter threshold would not be free: `effective_capacity_mw` is the 99th percentile of the
absolute power, so about 1% of readings must exceed it arithmetically, and cutting at 1.1 times
capacity would delete genuine high-output rows with a matched diffuse-fraction bias of 0.023.
"""

MIN_CAMS_RELIABILITY: Final[float] = 0.9
"""Hours where the CAMS service flags less than this fraction of its inputs as reliable are dropped.

The flag covers the satellite retrieval the global, beam and diffuse fluxes are all derived from, so
dropping an hour removes it from every arm at once and cannot favour one of them.
"""

MIN_SOLAR_ELEVATION_DEGREES: Final[float] = 0.0
"""Rows whose window-midpoint sun is below this elevation are dropped.

Zero degrees means "the sun is above the horizon", which is the one definition of daylight nobody
can argue was chosen to flatter a result. It leaves the twilight hours in, where power is near zero
and every arm is equally right, so every percentage difference this experiment reports is diluted by
rows no arm can get wrong. Raising the threshold would sharpen the contrast and would also be a
choice made with the answer in view, which is the worse trade.
"""

CONTROL_TILT_DEGREES_BY_SITE: Final[tuple[float, ...]] = (15.0, 22.0, 28.0, 35.0, 41.0, 47.0)
"""Tilt of each site's notional array in the synthetic control target, in `SITE_LABELS` order.

**Every value is deliberately away from the 30 degrees the physical instrument's optimiser starts
at**, so a fit that never moved would score badly rather than well. A control whose answer is the
starting point measures nothing.
"""

CONTROL_AZIMUTH_DEGREES_BY_SITE: Final[tuple[float, ...]] = (
    148.0,
    163.0,
    172.0,
    191.0,
    205.0,
    219.0,
)
"""Azimuth of each site's notional array, in `SITE_LABELS` order.

Spread either side of due south, and for the same reason as the tilts: 180 degrees is where the
physical instrument's optimiser starts.
"""

GROUND_ALBEDO: Final[float] = 0.2
"""Ground reflectance for the synthetic control target's transposition."""

REFERENCE_PLANE_OF_ARRAY_W_M2: Final[float] = 1000.0
"""Plane-of-array irradiance at which the synthetic array is taken to produce its full capacity."""

CONTROL_NOISE_FRACTION_OF_CAPACITY: Final[float] = 0.02
"""Noise added to the synthetic control target, as a fraction of the site's capacity.

A noise-free target would only show that the pipeline can detect an enormous effect, and too much
noise makes the control's own arm-to-arm difference smaller than the differences it is meant to
certify. Two per cent of capacity leaves the control's difference comfortably larger than anything
the real meters produce, which is what makes it usable as a detection threshold.
"""

CONTROL_NOISE_SEED: Final[int] = 5150

MIN_CONTROL_COS_ZENITH: Final[float] = 0.05
"""The floor on the cosine of the solar zenith angle when the control's beam is transposed.

The same floor `physics_model.MIN_COS_ZENITH` applies, for the same reason: near the horizon the
ratio of two vanishing quantities is numerical noise.
"""

JOULES_PER_HOUR_TO_WATTS: Final[float] = 3600.0
KELVIN_TO_CELSIUS_OFFSET: Final[float] = 273.15
SOLAR_CONSTANT_W_M2: Final[float] = 1361.0

SITE_LABELS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
"""Anonymous site labels.

These metered generators' output is commercially sensitive, so nothing downstream of this script
ever sees a `time_series_id`, a site name or a coordinate.
"""

LABEL_PERMUTATION_SEED: Final[int] = 784
"""Seed for the shuffle that assigns `SITE_LABELS` to sites.

Assigning labels in `time_series_id` order would put the mapping one sort away for anyone who
holds NGED's own roster and reads a published per-site table. The shuffle removes that sort, and
nothing more: the seed and the shuffle both sit in this public file, so a roster-holder who runs
`_pv_sites` reproduces the mapping exactly. **What protects the mapping is that the roster is
private, not that the labels are shuffled.** The seed is fixed so a re-run reproduces the same
labels.
"""


def _read_era5(*, source: SourceType) -> pl.DataFrame:
    """Read one ERA5 download into a long frame of hourly fluxes, trimmed to the shared span.

    Both sources are trimmed at `era5_grid.LAST_DATE`, because the Copernicus request is made in
    whole months and would otherwise run a few days past where the mirror stops.

    Args:
        source: Which download to read.

    Returns:
        One row per (time, latitude, longitude) with `ghi_w_m2`, `bhi_w_m2` and `temp_c`.
    """
    era5 = _read_open_meteo() if source == "open-meteo" else _read_cds_archives()
    return era5.filter(
        pl.col("time")
        <= pl.lit(f"{LAST_DATE} 23:00:00").str.to_datetime().dt.replace_time_zone("UTC")
    )


def _read_open_meteo() -> pl.DataFrame:
    """Read the Open-Meteo long frame `fetch_era5_open_meteo.py` wrote."""
    if not OPEN_METEO_PATH.exists():
        msg = f"{OPEN_METEO_PATH} missing; run fetch_era5_open_meteo.py first"
        raise FileNotFoundError(msg)
    return pl.read_parquet(OPEN_METEO_PATH).sort("time", "latitude", "longitude")


def _read_cams(*, min_reliability: float) -> pl.DataFrame:
    """Read the CAMS per-site frame `fetch_cams.py` wrote, trimmed and quality-filtered.

    Args:
        min_reliability: Hours the service flags below this fraction are dropped. Passing zero
            keeps every hour, which is what measures whether the filter changed the answer.

    Returns:
        One row per (site, time) with `ghi_w_m2` and `bhi_w_m2`.
    """
    if not CAMS_PATH.exists():
        msg = f"{CAMS_PATH} missing; run fetch_cams.py first"
        raise FileNotFoundError(msg)
    cams = pl.read_parquet(CAMS_PATH)
    reliable = cams.filter(pl.col("reliability") >= min_reliability)
    _LOG.info(
        "CAMS: %d of %d hours pass the reliability flag",
        reliable.height,
        cams.height,
    )
    return reliable.select("site", "time", "ghi_w_m2", "bhi_w_m2").sort("site", "time")


def _read_open_meteo_point(*, source: SourceType, temporal: PointTemporalType) -> pl.DataFrame:
    """Read one per-site frame `fetch_open_meteo_point.py` wrote.

    Every model the fetcher serves writes the same two default columns under the same names, so one
    reader covers them; only the path differs, and `point_output_path_for` owns that. Only a model
    whose own radiation is instantaneous also carries the `_instant` pair, because for an
    accumulated model the hourly column is the native quantity and a snapshot reconstructed from it
    would carry no information the hourly column lacks.

    Args:
        source: Which model's download to read.
        temporal: Which pair of columns to feed the arms. `hourly` is the default served
            column, a backward-looking mean over the hour ending at the label; `instant` is
            the snapshot at that label.

    Returns:
        One row per (site, time) with `ghi_w_m2` and `bhi_w_m2`.

    Raises:
        FileNotFoundError: If that model has not been downloaded.
        ValueError: If `instant` columns are asked of a model that publishes accumulated
            radiation, which has none.
    """
    if temporal == "instant" and OPEN_METEO_MODELS[source].native_radiation != "instantaneous":
        msg = (
            f"{source} publishes accumulated radiation, so its download carries no _instant "
            f"columns; build it with --point-temporal hourly"
        )
        raise ValueError(msg)
    path = point_output_path_for(source=source)
    if not path.exists():
        msg = f"{path} missing; run fetch_open_meteo_point.py --model {source} first"
        raise FileNotFoundError(msg)
    suffix = "_instant" if temporal == "instant" else ""
    frame = pl.read_parquet(path)
    return frame.select(
        "site",
        "time",
        ghi_w_m2=pl.col(f"ghi{suffix}_w_m2"),
        bhi_w_m2=pl.col(f"bhi{suffix}_w_m2"),
    ).sort("site", "time")


def _read_cds_archives() -> pl.DataFrame:
    """Read every downloaded Copernicus archive into one long frame of hourly fluxes.

    Returns:
        One row per (time, latitude, longitude) with `ghi_w_m2`, `bhi_w_m2` and `temp_c`.
    """
    archives = sorted(ERA5_DIR.glob("era5_*.zip"))
    if not archives:
        msg = f"no ERA5 archives in {ERA5_DIR}; run fetch_era5.py first"
        raise FileNotFoundError(msg)
    _LOG.info("reading %d ERA5 monthly archives", len(archives))

    with tempfile.TemporaryDirectory(dir=REPO_DATA_DIR) as scratch:
        frames = [
            _read_one_era5_archive(archive=archive, scratch=Path(scratch)) for archive in archives
        ]

    era5 = pl.concat(frames).unique(subset=["time", "latitude", "longitude"], keep="first")
    return era5.sort("time", "latitude", "longitude")


def _read_one_era5_archive(*, archive: Path, scratch: Path) -> pl.DataFrame:
    """Unpack one monthly CDS archive and return its hourly fluxes as a long frame.

    Args:
        archive: The `.zip` CDS returned for one calendar month.
        scratch: A directory the archive's members may be extracted into.

    Returns:
        One row per (time, latitude, longitude).
    """
    member_dir = scratch / archive.stem
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(member_dir)

    # CDS splits a mixed request into one netCDF per GRIB stepType: t2m is instantaneous while ssrd
    # and fdir are accumulations, so they cannot share a file.
    datasets = [xr.open_dataset(path) for path in sorted(member_dir.glob("*.nc"))]
    merged = xr.merge([ds.drop_vars("expver", errors="ignore") for ds in datasets])

    times = merged.valid_time.values.astype("datetime64[us]")
    latitudes = merged.latitude.values.astype(np.float64)
    longitudes = merged.longitude.values.astype(np.float64)
    time_grid, lat_grid, lon_grid = np.meshgrid(times, latitudes, longitudes, indexing="ij")
    long_frame = pl.DataFrame(
        {
            "time": time_grid.ravel(),
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "ghi_w_m2": merged.ssrd.values.ravel() / JOULES_PER_HOUR_TO_WATTS,
            "bhi_w_m2": merged.fdir.values.ravel() / JOULES_PER_HOUR_TO_WATTS,
            "temp_c": merged.t2m.values.ravel() - KELVIN_TO_CELSIUS_OFFSET,
        }
    )
    for dataset in datasets:
        dataset.close()
    return long_frame.with_columns(pl.col("time").cast(pl.Datetime("us", "UTC")))


def _pv_sites() -> pl.DataFrame:
    """Return the usable PV sites, labelled `A`–`F`.

    Returns:
        One row per site with `time_series_id`, `site`, `latitude`, `longitude` and
        `effective_capacity_mw`.
    """
    metadata = pl.read_parquet(METADATA_PATH).filter(pl.col("time_series_type") == "PV")
    row_counts = (
        pl.scan_delta(POWER_DELTA_URI)
        .group_by("time_series_id")
        .agg(pl.len().alias("n_rows"))
        .collect()
    )
    capacity = (
        pl.scan_delta(CAPACITY_DELTA_URI)
        .sort("time")
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").last())
        .collect()
    )

    min_rows = int(MIN_YEARS_OF_READINGS * 365.25 * 48)
    sites = (
        metadata.select("time_series_id", "latitude", "longitude")
        .join(row_counts, on="time_series_id", how="inner")
        .join(capacity, on="time_series_id", how="inner")
        .filter(pl.col("n_rows") >= min_rows)
        .sort("time_series_id")
    )
    if sites.height != len(SITE_LABELS):
        msg = f"expected {len(SITE_LABELS)} usable PV sites, found {sites.height}"
        raise ValueError(msg)
    shuffled = np.random.default_rng(LABEL_PERMUTATION_SEED).permutation(list(SITE_LABELS))
    return sites.with_columns(site=pl.Series(shuffled, dtype=pl.Utf8)).drop("n_rows")


def _hourly_power(*, sites: pl.DataFrame, alignment: AlignmentType) -> pl.DataFrame:
    """Aggregate the half-hourly PV readings onto the ERA5 hourly, period-ending grid.

    An hour ending at `T` is the mean of the two half-hours ending at `T - 30 min` and `T`, and is
    produced only when both are present, so a partly-missing hour is dropped rather than silently
    becoming a half-hour mean.

    Args:
        sites: The site roster from `_pv_sites`.
        alignment: Whether to take the power stamps at face value or move them 30 minutes earlier
            first. See `AlignmentType`.

    Returns:
        One row per (site, time) with `power_mw` and `has_zero_half_hour`.
    """
    half_hourly = (
        pl.scan_delta(POWER_DELTA_URI)
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .collect()
        .join(sites.select("time_series_id", "site"), on="time_series_id")
        .rename({"power": "power_mw"})
        .select("site", "time", "power_mw")
    )
    # Both half-hours of the window (T-1h, T] carry the stamp of their own end, so the later one is
    # already stamped T and the earlier one has to be rolled forward by 30 minutes. Under the
    # shifted reading every stamp is half an hour late, which cancels that roll-forward exactly.
    # NGED corrected the feed mid-record, so a single global offset is wrong either side of
    # ALIGNMENT_FIXED_AT. Correct the stamp first, then one roll-forward serves every row.
    if alignment == "piecewise":
        corrected = (
            pl.when(pl.col("time") < pl.lit(ALIGNMENT_FIXED_AT))
            .then(pl.col("time").dt.offset_by("-30m"))
            .otherwise(pl.col("time"))
        )
        hour_end = corrected.dt.offset_by("30m").dt.truncate("1h")
    else:
        offset = "0m" if alignment == "shifted" else "30m"
        hour_end = pl.col("time").dt.offset_by(offset).dt.truncate("1h")
    return (
        half_hourly.with_columns(hour_end=hour_end)
        .group_by("site", "hour_end")
        .agg(
            power_mw=pl.col("power_mw").mean(),
            n_half_hours=pl.len(),
            has_zero_half_hour=(pl.col("power_mw") == 0.0).any(),
        )
        .filter(pl.col("n_half_hours") == 2)
        .drop("n_half_hours")
        .rename({"hour_end": "time"})
        .sort("site", "time")
    )


def _drop_outages_and_spikes(*, power: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Remove multi-day zero runs and physically impossible meter spikes.

    Whatever these filters remove is removed before the arms exist, so every arm trains and is
    scored on exactly the same rows.

    Args:
        power: Hourly power from `_hourly_power`.
        sites: The site roster, for each site's effective capacity.

    Returns:
        `power` with the offending rows removed.
    """
    with_capacity = power.join(sites.select("site", "effective_capacity_mw"), on="site")
    spike_free = with_capacity.filter(
        pl.col("power_mw") <= IMPLAUSIBLE_CAPACITY_MULTIPLE * pl.col("effective_capacity_mw")
    )

    # A zero run is only an outage if its rows are consecutive *hours*; a gap in the readings ends
    # the run, because we cannot tell what happened across it.
    is_zero = pl.col("power_mw") == 0.0
    contiguous = pl.col("time").diff().over("site") == pl.duration(hours=1)
    run_id = (~(is_zero & contiguous.fill_null(value=False))).cum_sum().over("site")
    labelled = spike_free.sort("site", "time").with_columns(_zero=is_zero, _run_id=run_id)
    run_lengths = labelled.filter("_zero").group_by("site", "_run_id").agg(run_hours=pl.len())
    outage_runs = run_lengths.filter(pl.col("run_hours") >= OUTAGE_RUN_HOURS)

    # A run id is shared by the zero rows of an outage *and* by the non-zero row immediately before
    # it, because the id increments on the first row that is not a contiguous zero. Requiring
    # `_zero` as well as a matching run id keeps that last good reading.
    doomed = labelled.filter("_zero").join(outage_runs, on=["site", "_run_id"], how="semi")
    return (
        labelled.join(doomed.select("site", "time"), on=["site", "time"], how="anti")
        .drop("_zero", "_run_id")
        .sort("site", "time")
    )


def _drop_false_zeros(*, joined: pl.DataFrame) -> pl.DataFrame:
    """Drop hours whose meter read exactly zero while the sun was well up.

    Runs after the ERA5 join because it is the only filter that needs irradiance. It reads global
    horizontal irradiance alone and never the beam or the diffuse component, which is what keeps it
    from favouring an arm — see `FALSE_ZERO_IRRADIANCE_W_M2`.

    Args:
        joined: Hourly power already joined to ERA5.

    Returns:
        `joined` with the false zeros removed, and without the flag column.
    """
    kept = joined.filter(
        ~(pl.col("has_zero_half_hour") & (pl.col("ghi_w_m2") > FALSE_ZERO_IRRADIANCE_W_M2))
    )
    _LOG.info(
        "false-zero filter removed %d of %d hours (%.2f%%)",
        joined.height - kept.height,
        joined.height,
        100.0 * (joined.height - kept.height) / joined.height,
    )
    return kept.drop("has_zero_half_hour")


def _nearest_era5_cell(*, sites: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Attach each site to the ERA5 grid cell whose centre is nearest it.

    Nearest-cell rather than interpolation on purpose. Interpolating between cells would smooth the
    beam field, and the beam field's small-scale structure is exactly what the experiment is asking
    about — blunting it would bias the answer towards "no information".

    Args:
        sites: The site roster.
        era5: The long ERA5 frame, for its grid coordinates.

    Returns:
        `sites` with `cell_latitude` and `cell_longitude` added.
    """
    latitudes = np.sort(era5["latitude"].unique().to_numpy())
    longitudes = np.sort(era5["longitude"].unique().to_numpy())
    site_lat = sites["latitude"].to_numpy().astype(np.float64)
    site_lon = sites["longitude"].to_numpy().astype(np.float64)
    nearest_lat = latitudes[np.abs(site_lat[:, None] - latitudes[None, :]).argmin(axis=1)]
    nearest_lon = longitudes[np.abs(site_lon[:, None] - longitudes[None, :]).argmin(axis=1)]
    return sites.with_columns(
        cell_latitude=pl.Series(nearest_lat),
        cell_longitude=pl.Series(nearest_lon),
    )


def _add_solar_geometry(*, joined: pl.DataFrame) -> pl.DataFrame:
    """Add solar position and extraterrestrial irradiance at each window's midpoint.

    Solar geometry is deterministic given time and place, so it is a legitimate feature for every
    arm and makes the global-irradiance-only arm as strong as it can be.

    Args:
        joined: Rows carrying `time`, `latitude`, `longitude`.

    Returns:
        `joined` with `solar_zenith_deg`, `solar_elevation_deg`, `solar_azimuth_deg` and
        `extraterrestrial_horizontal_w_m2` added.
    """
    frames: list[pl.DataFrame] = []
    for _site_key, site_rows in joined.sort("site", "time").group_by(["site"], maintain_order=True):
        # The fluxes are means over (T - 1 h, T], so the representative sun position is the one at
        # the window's midpoint.
        midpoints = site_rows["time"].dt.offset_by("-30m").to_numpy()
        position = pvlib.solarposition.get_solarposition(
            time=midpoints,
            latitude=float(site_rows["latitude"][0]),
            longitude=float(site_rows["longitude"][0]),
        )
        zenith = position["apparent_zenith"].to_numpy().astype(np.float64)
        azimuth = position["azimuth"].to_numpy().astype(np.float64)
        day_of_year = site_rows["time"].dt.ordinal_day().to_numpy()
        extraterrestrial_normal = pvlib.irradiance.get_extra_radiation(
            datetime_or_doy=day_of_year, solar_constant=SOLAR_CONSTANT_W_M2
        )
        frames.append(
            site_rows.with_columns(
                solar_zenith_deg=pl.Series(zenith),
                solar_elevation_deg=pl.Series(90.0 - zenith),
                solar_azimuth_deg=pl.Series(azimuth),
                extraterrestrial_horizontal_w_m2=pl.Series(
                    np.asarray(extraterrestrial_normal)
                    * np.clip(np.cos(np.radians(zenith)), 0.0, None)
                ),
            )
        )
    return pl.concat(frames)


def _add_separation_models(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the irradiance columns each arm draws on, including the separation-model estimates.

    **Every arm's beam column is a flux onto a horizontal plane, never a direct-normal one.** A
    separation model returns direct normal irradiance, which is beam-on-horizontal divided by the
    cosine of the solar zenith — a much larger number at low sun, where beam-on-horizontal is near
    zero. Feeding one arm normal components and another horizontal ones would let the arms differ
    in how the quantity is encoded as well as in what it knows, and a tree cannot divide, so it
    would have to rediscover the cosine across many splits. Converting the separation models' output
    back to the horizontal plane removes that confound.

    Erbs is the separation model the [Horat, Klerings and Lerch
    (2024)](https://arxiv.org/abs/2406.04424) chain uses, and is the primary. DISC is the
    sensitivity, in place of the DIRINT model the issue suggested, because DISC is like Erbs a
    per-row function of global irradiance and solar geometry. DIRINT's `delta_kt_prime` term reads
    the neighbouring hours' global irradiance, which the global-only arm never sees, so a DIRINT arm
    could beat that arm by smuggling in temporal structure rather than by carrying the split.

    Args:
        frame: Rows carrying `ghi_w_m2`, `bhi_w_m2` and `solar_zenith_deg`.

    Returns:
        `frame` with the separation-model and true-split columns added.
    """
    ghi = frame["ghi_w_m2"].to_numpy().astype(np.float64)
    bhi = frame["bhi_w_m2"].to_numpy().astype(np.float64)
    zenith = frame["solar_zenith_deg"].to_numpy().astype(np.float64)
    day_of_year = frame["time"].dt.ordinal_day().to_numpy()

    erbs = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=day_of_year)
    disc = pvlib.irradiance.disc(ghi=ghi, solar_zenith=zenith, datetime_or_doy=day_of_year)
    cos_zenith = np.clip(np.cos(np.radians(zenith)), 0.0, None)

    erbs_bhi = np.nan_to_num(np.asarray(erbs["dni"]), nan=0.0) * cos_zenith
    disc_bhi = np.nan_to_num(np.asarray(disc["dni"]), nan=0.0) * cos_zenith

    # The direct fraction is undefined when there is no global irradiance at all, and a ratio taken
    # near zero is numerical noise rather than a physical quantity, so both are pinned to zero.
    direct_fraction = np.where(ghi > 1.0, np.clip(bhi / np.maximum(ghi, 1e-9), 0.0, 1.0), 0.0)

    return frame.with_columns(
        erbs_bhi_w_m2=pl.Series(np.clip(erbs_bhi, 0.0, ghi)),
        erbs_dhi_w_m2=pl.Series(np.clip(ghi - erbs_bhi, 0.0, None)),
        disc_bhi_w_m2=pl.Series(np.clip(disc_bhi, 0.0, ghi)),
        disc_dhi_w_m2=pl.Series(np.clip(ghi - disc_bhi, 0.0, None)),
        dhi_w_m2=pl.Series(np.maximum(ghi - bhi, 0.0)),
        direct_fraction=pl.Series(direct_fraction),
    )


def _add_synthetic_control_target(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add a synthetic power target that the beam/diffuse split is guaranteed to help predict.

    **This is the experiment's positive control, and without it a null result would be
    uninterpretable.** Finding no arm-to-arm difference on the real meters says "we did not detect
    an effect", which only becomes "there is no effect to detect" once the instrument has been shown
    to detect one. The synthetic target is the plane-of-array irradiance a tilted array would see,
    scaled to the site's capacity and buried in noise: transposition needs the beam and the diffuse
    separately, so the split *must* help here. An arm that cannot beat the global-only arm on this
    target cannot be trusted to have looked properly at the real one.

    **Two choices stop the control flattering the physical instrument.** Each site gets its own
    tilt and azimuth, none of them the values that instrument's optimiser starts from, so a fit that
    never moved would score badly. And the sky diffuse is transposed by the Hay-Davies model, which
    puts part of the diffuse in a circumsolar band around the sun, while the physical instrument
    assumes an isotropic sky. The instrument's hypothesis class therefore does not contain the
    target, which is the situation a real meter puts it in.

    Beam on the tilted plane is the beam's horizontal flux times the ratio of the cosine of the
    angle of incidence to the cosine of the solar zenith. The cosine of the zenith is floored,
    because it goes to zero at sunset while beam-on-horizontal goes to zero with it, and the ratio
    of two vanishing quantities is numerical noise.

    Args:
        frame: Rows carrying the true split, solar geometry, `site` and `effective_capacity_mw`.

    Returns:
        `frame` with `synthetic_power_mw` added.
    """
    site_index = np.array(
        [SITE_LABELS.index(label) for label in frame["site"].to_list()], dtype=np.int64
    )
    tilt = np.radians(np.asarray(CONTROL_TILT_DEGREES_BY_SITE)[site_index])
    surface_azimuth = np.radians(np.asarray(CONTROL_AZIMUTH_DEGREES_BY_SITE)[site_index])

    zenith = np.radians(frame["solar_zenith_deg"].to_numpy().astype(np.float64))
    azimuth = np.radians(frame["solar_azimuth_deg"].to_numpy().astype(np.float64))
    cos_incidence = np.clip(
        np.cos(zenith) * np.cos(tilt)
        + np.sin(zenith) * np.sin(tilt) * np.cos(azimuth - surface_azimuth),
        0.0,
        None,
    )
    beam_ratio = cos_incidence / np.maximum(np.cos(zenith), MIN_CONTROL_COS_ZENITH)

    beam_horizontal = frame["bhi_w_m2"].to_numpy().astype(np.float64)
    diffuse_horizontal = frame["dhi_w_m2"].to_numpy().astype(np.float64)
    global_horizontal = frame["ghi_w_m2"].to_numpy().astype(np.float64)
    extraterrestrial = frame["extraterrestrial_horizontal_w_m2"].to_numpy().astype(np.float64)

    # Hay-Davies weights the circumsolar part of the diffuse by how much beam survived the
    # atmosphere, which on a horizontal plane is the beam's share of the top-of-atmosphere flux.
    anisotropy = np.where(
        extraterrestrial > 1.0,
        np.clip(beam_horizontal / np.maximum(extraterrestrial, 1e-9), 0.0, 1.0),
        0.0,
    )
    sky_diffuse = diffuse_horizontal * (
        anisotropy * beam_ratio + (1.0 - anisotropy) * (1.0 + np.cos(tilt)) / 2.0
    )
    beam_on_plane = beam_horizontal * beam_ratio
    ground_reflected = global_horizontal * GROUND_ALBEDO * (1.0 - np.cos(tilt)) / 2.0
    plane_of_array = beam_on_plane + sky_diffuse + ground_reflected

    capacity = frame["effective_capacity_mw"].to_numpy().astype(np.float64)
    generator = np.random.default_rng(CONTROL_NOISE_SEED)
    noise = generator.normal(0.0, CONTROL_NOISE_FRACTION_OF_CAPACITY * capacity)
    synthetic = np.clip(
        capacity * plane_of_array / REFERENCE_PLANE_OF_ARRAY_W_M2 + noise, 0.0, None
    )
    return frame.with_columns(synthetic_power_mw=pl.Series(synthetic))


def main() -> int:
    """Build the joined frame for the irradiance source named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="open-meteo")
    parser.add_argument(
        "--alignment", choices=("as-labelled", "shifted", "piecewise"), default="piecewise"
    )
    parser.add_argument(
        "--min-cams-reliability",
        type=float,
        default=MIN_CAMS_RELIABILITY,
        help="Drop CAMS hours flagged below this fraction. Zero keeps every hour.",
    )
    parser.add_argument(
        "--point-temporal",
        choices=("hourly", "instant"),
        default="hourly",
        help=(
            "Which UKV columns the arms see. Pair it with --suffix, or the variant build "
            "overwrites the main one."
        ),
    )
    parser.add_argument(
        "--first-date",
        default=None,
        help=(
            "Drop rows before this YYYY-MM-DD. Pair it with --suffix. What it exists for is "
            "running a source over the span another source can be checked against."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Appended to the output filename, so a variant build does not overwrite the main one.",
    )
    arguments = parser.parse_args()
    source: SourceType = arguments.source
    alignment: AlignmentType = arguments.alignment

    sites = _pv_sites()
    _LOG.info(
        "using %d PV sites, irradiance source %s, power stamps %s", sites.height, source, alignment
    )

    gridded = _read_era5(source="open-meteo" if source in PER_SITE_SOURCES else source)
    _LOG.info(
        "gridded fields: %d rows, %s to %s",
        gridded.height,
        gridded["time"].min(),
        gridded["time"].max(),
    )

    power = _drop_outages_and_spikes(
        power=_hourly_power(sites=sites, alignment=alignment), sites=sites
    )
    _LOG.info("hourly power after outage and spike filtering: %d rows", power.height)

    sites_with_cells = _nearest_era5_cell(sites=sites, era5=gridded)
    _LOG.info(
        "the %d sites resolve to %d distinct ERA5 grid cells",
        sites_with_cells.height,
        sites_with_cells.select("cell_latitude", "cell_longitude").n_unique(),
    )
    # Every source needs the gridded frame's air temperature, which is a shared feature rather than
    # an irradiance one, so the gridded join runs for the CAMS build too and only its two irradiance
    # columns are then replaced.
    joined = (
        power.join(sites_with_cells, on=["site", "effective_capacity_mw"])
        .join(
            gridded,
            left_on=["time", "cell_latitude", "cell_longitude"],
            right_on=["time", "latitude", "longitude"],
            how="inner",
        )
        .drop("time_series_id")
    )
    if source in PER_SITE_SOURCES:
        per_site = (
            _read_cams(min_reliability=arguments.min_cams_reliability)
            if source == "cams"
            else _read_open_meteo_point(source=source, temporal=arguments.point_temporal)
        )
        joined = joined.drop("ghi_w_m2", "bhi_w_m2").join(
            per_site, on=["site", "time"], how="inner"
        )
    _LOG.info("after joining irradiance: %d rows", joined.height)

    with_geometry = _add_solar_geometry(joined=_drop_false_zeros(joined=joined))
    daylight = with_geometry.filter(pl.col("solar_elevation_deg") > MIN_SOLAR_ELEVATION_DEGREES)
    _LOG.info("daylight rows: %d", daylight.height)

    dataset = _add_synthetic_control_target(frame=_add_separation_models(frame=daylight)).drop(
        "cell_latitude", "cell_longitude", "latitude", "longitude"
    )
    # The filter runs after the synthetic control target, so a shorter span keeps the same seeded
    # noise on the rows it shares with the full build rather than redrawing it.
    if arguments.first_date is not None:
        before = dataset.height
        dataset = dataset.filter(
            pl.col("time")
            >= pl.lit(f"{arguments.first_date} 00:00:00")
            .str.to_datetime()
            .dt.replace_time_zone("UTC")
        )
        _LOG.info("first-date filter kept %d of %d rows", dataset.height, before)
    output_path = output_path_for(dataset_name=f"{source}{arguments.suffix}", alignment=alignment)
    dataset.write_parquet(output_path)
    _LOG.info("wrote %d rows to %s", dataset.height, output_path)
    _LOG.info(
        "span %s to %s; per-site rows %s",
        dataset["time"].min(),
        dataset["time"].max(),
        dataset.group_by("site").len().sort("site").to_dicts(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
