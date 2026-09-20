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

import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

# pvlib is not a workspace dependency; this throwaway script is run with `uv run --with pvlib`.
import pvlib  # ty: ignore[unresolved-import]
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("build_dataset")

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")
ERA5_DIR: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse"
POWER_DELTA_URI: Final[str] = str(REPO_DATA_DIR / "NGED" / "power_time_series.delta")
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
CAPACITY_DELTA_URI: Final[str] = str(REPO_DATA_DIR / "effective_capacity")
OUTPUT_PATH: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_dataset.parquet"

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

IMPLAUSIBLE_CAPACITY_MULTIPLE: Final[float] = 1.5
"""Readings above this multiple of the site's effective capacity are dropped as meter spikes."""

MIN_SOLAR_ELEVATION_DEGREES: Final[float] = 0.0
"""Rows whose window-midpoint sun is below this elevation are dropped.

Zero degrees means "the sun is above the horizon", which is the one definition of daylight nobody
can argue was chosen to flatter a result. It leaves the twilight hours in, where power is near zero
and every arm is equally right, so every percentage difference this experiment reports is diluted by
rows no arm can get wrong. Raising the threshold would sharpen the contrast and would also be a
choice made with the answer in view, which is the worse trade.
"""

CONTROL_TILT_DEGREES: Final[float] = 30.0
"""Tilt of the notional array the synthetic control target is built from."""

CONTROL_AZIMUTH_DEGREES: Final[float] = 180.0
"""Azimuth of that notional array: due south, the usual choice for a GB fixed-tilt farm."""

GROUND_ALBEDO: Final[float] = 0.2
"""Ground reflectance for the synthetic control target's transposition."""

REFERENCE_PLANE_OF_ARRAY_W_M2: Final[float] = 1000.0
"""Plane-of-array irradiance at which the synthetic array is taken to produce its full capacity."""

CONTROL_NOISE_FRACTION_OF_CAPACITY: Final[float] = 0.05
"""Noise added to the synthetic control target, as a fraction of the site's capacity.

A noise-free target would only show that the pipeline can detect an enormous effect. Five per cent
of capacity is the order of a competent irradiance-driven PV forecast's error, so clearing the bar
on this target says the pipeline can find a split effect at a realistic signal-to-noise ratio.
"""

CONTROL_NOISE_SEED: Final[int] = 5150

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

Assigning labels in `time_series_id` order would not anonymise anything: `time_series_id` comes
from NGED's own feed, so anyone holding the same roster could sort it and read the mapping straight
off a published per-site table. Shuffling breaks that. The seed is fixed so a re-run reproduces the
same labels.
"""


def _read_era5() -> pl.DataFrame:
    """Read every downloaded ERA5 month into one long frame of hourly fluxes.

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


def _hourly_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the half-hourly PV readings onto the ERA5 hourly, period-ending grid.

    An hour ending at `T` is the mean of the two half-hours ending at `T - 30 min` and `T`, and is
    produced only when both are present, so a partly-missing hour is dropped rather than silently
    becoming a half-hour mean.

    Args:
        sites: The site roster from `_pv_sites`.

    Returns:
        One row per (site, time) with `power_mw`.
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
    # already stamped T and the earlier one has to be rolled forward by 30 minutes.
    hour_end = pl.col("time").dt.offset_by("30m").dt.truncate("1h")
    return (
        half_hourly.with_columns(hour_end=hour_end)
        .group_by("site", "hour_end")
        .agg(power_mw=pl.col("power_mw").mean(), n_half_hours=pl.len())
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
    to detect one. The synthetic target is the plane-of-array irradiance a south-facing array tilted
    at `CONTROL_TILT_DEGREES` would see, scaled to the site's capacity and buried in noise:
    transposition needs the beam and the diffuse separately, so the split *must* help here. An arm
    that cannot beat the global-only arm on this target cannot be trusted to have looked properly at
    the real one.

    Beam on the tilted plane is the beam's horizontal flux times the ratio of the cosine of the
    angle of incidence to the cosine of the solar zenith. The cosine of the zenith is floored,
    because it goes to zero at sunset while beam-on-horizontal goes to zero with it, and the ratio
    of two vanishing quantities is numerical noise.

    Args:
        frame: Rows carrying the true split, solar geometry and `effective_capacity_mw`.

    Returns:
        `frame` with `synthetic_power_mw` added.
    """
    zenith = np.radians(frame["solar_zenith_deg"].to_numpy().astype(np.float64))
    azimuth = np.radians(frame["solar_azimuth_deg"].to_numpy().astype(np.float64))
    tilt = np.radians(CONTROL_TILT_DEGREES)
    surface_azimuth = np.radians(CONTROL_AZIMUTH_DEGREES)

    cos_incidence = np.clip(
        np.cos(zenith) * np.cos(tilt)
        + np.sin(zenith) * np.sin(tilt) * np.cos(azimuth - surface_azimuth),
        0.0,
        None,
    )
    beam_on_plane = frame["bhi_w_m2"].to_numpy() * cos_incidence / np.maximum(np.cos(zenith), 0.05)
    sky_diffuse = frame["dhi_w_m2"].to_numpy() * (1.0 + np.cos(tilt)) / 2.0
    ground_reflected = frame["ghi_w_m2"].to_numpy() * GROUND_ALBEDO * (1.0 - np.cos(tilt)) / 2.0
    plane_of_array = beam_on_plane + sky_diffuse + ground_reflected

    capacity = frame["effective_capacity_mw"].to_numpy().astype(np.float64)
    generator = np.random.default_rng(CONTROL_NOISE_SEED)
    noise = generator.normal(0.0, CONTROL_NOISE_FRACTION_OF_CAPACITY * capacity)
    synthetic = np.clip(
        capacity * plane_of_array / REFERENCE_PLANE_OF_ARRAY_W_M2 + noise, 0.0, None
    )
    return frame.with_columns(synthetic_power_mw=pl.Series(synthetic))


def main() -> int:
    """Build the joined frame and write it to parquet."""
    sites = _pv_sites()
    _LOG.info("using %d PV sites", sites.height)

    era5 = _read_era5()
    _LOG.info("ERA5: %d rows, %s to %s", era5.height, era5["time"].min(), era5["time"].max())

    power = _drop_outages_and_spikes(power=_hourly_power(sites=sites), sites=sites)
    _LOG.info("hourly power after outage and spike filtering: %d rows", power.height)

    sites_with_cells = _nearest_era5_cell(sites=sites, era5=era5)
    _LOG.info(
        "the %d sites resolve to %d distinct ERA5 grid cells",
        sites_with_cells.height,
        sites_with_cells.select("cell_latitude", "cell_longitude").n_unique(),
    )
    joined = (
        power.join(sites_with_cells, on=["site", "effective_capacity_mw"])
        .join(
            era5,
            left_on=["time", "cell_latitude", "cell_longitude"],
            right_on=["time", "latitude", "longitude"],
            how="inner",
        )
        .drop("time_series_id")
    )
    _LOG.info("after joining ERA5: %d rows", joined.height)

    with_geometry = _add_solar_geometry(joined=joined)
    daylight = with_geometry.filter(pl.col("solar_elevation_deg") > MIN_SOLAR_ELEVATION_DEGREES)
    _LOG.info("daylight rows: %d", daylight.height)

    dataset = _add_synthetic_control_target(frame=_add_separation_models(frame=daylight)).drop(
        "cell_latitude", "cell_longitude", "latitude", "longitude"
    )
    dataset.write_parquet(OUTPUT_PATH)
    _LOG.info("wrote %d rows to %s", dataset.height, OUTPUT_PATH)
    _LOG.info(
        "span %s to %s; per-site rows %s",
        dataset["time"].min(),
        dataset["time"].max(),
        dataset.group_by("site").len().sort("site").to_dicts(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
