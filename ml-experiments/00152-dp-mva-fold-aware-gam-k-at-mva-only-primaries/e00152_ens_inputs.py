"""The ensemble inputs of experiment 152: arrays per site, setup, and calibration.

Copied from experiments 111 and 136 so this folder imports only itself. The five MW decision
Primaries read experiment 111's stored arrays and calibrations. The eight MVA-only Primaries have
no stored arrays, so `build_store` cuts them from the same ensemble frames, and their calibration
is fitted here on the tuning months with the function experiment 111 used.

    uv run python ml-experiments/00152-*/e00152_ens_inputs.py [--workers 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00152_common as common
import h3
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.dp_ens_weather import (
    HORIZON_HALF_HOURS,
    AnalysisProxy,
    Calibration,
    EnsSiteRun,
    WeatherContext,
    ghi_climatology,
    lead_band_of_hours,
    season_of_months,
)
from studies.dp_features import FeatureCache, default_input_paths
from studies.sources import REPO_DATA_DIR

N_MEMBERS: Final[int] = 51
N_TARGETS: Final[int] = 702
"""Half-hours in a frame: 351 hours after the origin."""
CLIMATOLOGY_BEFORE: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
"""CAMS irradiance after this date never enters the climatology the solar shrink pulls towards."""


ARCHIVE_DIR: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
H3_RESOLUTION: Final[int] = 5
HORIZON_HOURS: Final[int] = 351
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "h3_index",
    "ensemble_member",
    "valid_time",
    "temperature_2m",
    "downward_short_wave_radiation_flux_surface",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)


def partition_path(*, run: datetime) -> Path:
    """Return the archive folder of a run, whose name percent-encodes the init time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    return ARCHIVE_DIR / f"init_time={stamp}"


def site_table(*, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return `site`, `time_series_id`, `latitude`, `longitude`, and `h3_res_5` of some sites.

    The coordinates stay inside the process; only the arrays leave.
    """
    series = pl.read_parquet(default_input_paths().series_table)
    kept = series.filter(pl.col("site").is_in(list(sites))).select(
        "site", "time_series_id", "latitude", "longitude"
    )
    return kept.with_columns(
        h3_res_5=pl.struct("latitude", "longitude").map_elements(
            lambda row: h3.str_to_int(
                h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION)
            ),
            return_dtype=pl.Int64,
        ),
        time_series_id=pl.col("time_series_id").cast(pl.Int32),
    )


def read_run(*, run: datetime, sites: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return one run's temperature and shortwave per site, shape (members, targets) each.

    The run is cut to the sites' H3 cells and resampled to the half-hourly grid by production's
    `_upsample_nwp_to_half_hourly`, as experiment 111's frame builder did, and cut to valid times
    from the run's `power_fcst_init_time` to 351 hours later.

    Raises:
        ValueError: If a site lacks a member or a half-hour, or a key repeats.
    """
    cells = sites["h3_res_5"].unique().to_list()
    targets = sites.select("time_series_id", "h3_res_5", "latitude", "longitude")
    native = (
        pl.scan_parquet(partition_path(run=run) / "*.parquet")
        .select(SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            targets.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = targets.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = (
        _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates)
        .collect()
        .rename({"downward_short_wave_radiation_flux_surface": "shortwave"})
    )
    origin = run + common.PUBLICATION_DELAY
    frame = half_hourly.with_columns(
        lead_hours=(
            pl.col("valid_time") - pl.lit(origin).cast(pl.Datetime("us", "UTC"))
        ).dt.total_minutes()
        / 60.0
    ).filter((pl.col("lead_hours") > 0.0) & (pl.col("lead_hours") <= HORIZON_HOURS))
    if frame.select("time_series_id", "ensemble_member", "valid_time").is_duplicated().any():
        msg = f"the {run:%Y-%m-%d} frame repeats a key"
        raise ValueError(msg)
    out = {}
    for row in sites.iter_rows(named=True):
        rows = frame.filter(pl.col("time_series_id") == row["time_series_id"]).sort(
            "ensemble_member", "valid_time"
        )
        if rows.height != N_MEMBERS * N_TARGETS:
            msg = (
                f"{row['site']} on {run:%Y-%m-%d} has {rows.height} rows, "
                f"not {N_MEMBERS * N_TARGETS}"
            )
            raise ValueError(msg)
        out[row["site"]] = (
            rows["temperature_2m"].to_numpy().reshape(N_MEMBERS, N_TARGETS),
            rows["shortwave"].to_numpy().reshape(N_MEMBERS, N_TARGETS),
        )
    return out


_SITES: list[pl.DataFrame] = []


def _initialise_worker() -> None:
    _SITES.append(site_table(sites=common.MVA_ONLY_PRIMARIES))


def _read(run: datetime) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return read_run(run=run, sites=_SITES[0])


def build_store(*, workers: int) -> None:
    """Read every run and write one array file per MVA-only Primary under `OWN_STORE_DIR`."""
    runs = common.run_dates()
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise_worker) as pool:
        results = list(pool.map(_read, runs, chunksize=8))
    common.OWN_STORE_DIR.mkdir(parents=True, exist_ok=True)
    origins = np.array(
        [np.datetime64(common.origin_of(run=run).replace(tzinfo=None), "us") for run in runs]
    )
    for site in common.MVA_ONLY_PRIMARIES:
        temperature = np.stack([result[site][0] for result in results])
        shortwave = np.stack([result[site][1] for result in results])
        target = common.OWN_STORE_DIR / f"{site}.npz"
        temporary = target.with_name(f"{target.stem}.tmp.npz")
        np.savez(temporary, origins=origins, temperature=temperature, shortwave=shortwave)
        temporary.rename(target)
        print(f"{site}: {temperature.shape}, nulls {int(np.isnan(shortwave).sum())}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


@dataclass(frozen=True)
class SiteSetup:
    """A substation's ensemble runs and the weather layer built from them.

    Attributes:
        site: The substation's label.
        run_dates: The 00:00 UTC initialisation time of each run.
        origins: Each run's `power_fcst_init_time`.
        runs: Each run's ensemble weather on the cache's grid.
        context: The calibration, climatology, and analysis proxy.
    """

    site: str
    run_dates: list[datetime]
    origins: list[datetime]
    runs: list[EnsSiteRun]
    context: WeatherContext

    def with_calibration(self, *, calibration: Calibration) -> SiteSetup:
        """Return the setup with another calibration."""
        return replace(self, context=replace(self.context, calibration=calibration))

    def indices(self, *, periods: set[str]) -> list[int]:
        """Return the runs that belong to the named periods."""
        return [
            index
            for index, run in enumerate(self.run_dates)
            if common.period_of_run(run=run) in periods
        ]


def store_file(*, site: str):  # noqa: ANN201
    """Return the array file of a site: experiment 111's for MW sites, this folder's for MVA."""
    own = common.OWN_STORE_DIR / f"{site}.npz"
    return own if own.exists() else common.ENS_SOURCE_DIR / "ens_store" / f"{site}.npz"


def load_setup(*, cache: FeatureCache, site: str) -> SiteSetup:
    """Load a site's runs and build its proxy and climatology, with an identity calibration."""
    with np.load(store_file(site=site)) as stored:
        arrays = {name: stored[name] for name in stored.files}
    run_dates = common.run_dates()
    origins = [common.origin_of(run=run) for run in run_dates]
    positions = np.searchsorted(cache.end_times, arrays["origins"])
    runs = [
        EnsSiteRun(
            origin_position=int(position),
            temperature_c=arrays["temperature"][index],
            shortwave_w_m2=arrays["shortwave"][index],
        )
        for index, position in enumerate(positions)
    ]
    context = WeatherContext(
        site=site,
        calibration=Calibration.identity(),
        climatology=ghi_climatology(cache=cache, site=site, before=CLIMATOLOGY_BEFORE),
        proxy=AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )
    return SiteSetup(site=site, run_dates=run_dates, origins=origins, runs=runs, context=context)


def fit_calibration(*, cache: FeatureCache, setup: SiteSetup) -> Calibration:
    """Fit the calibration of a substation on the tuning months' runs only (experiment 111's)."""
    features = cache.sites[setup.site]
    names = ("origin", "season", "lead", "band", "ens_t", "era5_t", "ens_g", "cams_g")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in names}
    lead = lead_band_of_hours(lead_hours=np.arange(1, HORIZON_HALF_HOURS + 1) / 2.0)
    for index in setup.indices(periods={"tuning"}):
        run = setup.runs[index]
        rows = np.arange(run.origin_position + 1, run.origin_position + 1 + HORIZON_HALF_HOURS)
        months = cache.calendar.local_date[rows].astype("datetime64[M]").astype(np.int64) % 12 + 1
        pieces["origin"].append(
            np.full(len(rows), np.datetime64(setup.origins[index].replace(tzinfo=None), "us"))
        )
        pieces["season"].append(season_of_months(months=months))
        pieces["lead"].append(lead)
        pieces["band"].append(cache.calendar.band[rows].astype(np.int64))
        pieces["ens_t"].append(np.nanmean(run.temperature_c[:, :HORIZON_HALF_HOURS], axis=0))
        pieces["era5_t"].append(features.temperature_c[rows])
        pieces["ens_g"].append(np.nanmean(run.shortwave_w_m2[:, :HORIZON_HALF_HOURS], axis=0))
        pieces["cams_g"].append(features.ghi_w_m2[rows])
    joined = {name: np.concatenate(values) for name, values in pieces.items()}
    return Calibration.fit(
        origin_time=joined["origin"],
        allowed_months=common.TUNING_MONTHS,
        season=joined["season"],
        lead_band=joined["lead"],
        hour_band=joined["band"],
        ens_mean_temperature_c=joined["ens_t"],
        target_temperature_c=joined["era5_t"],
        ens_mean_ghi=joined["ens_g"],
        target_ghi=joined["cams_g"],
    )


def calibrated_setup(*, cache: FeatureCache, site: str) -> SiteSetup:
    """Return a site's setup with its calibration: experiment 111's if stored, else fitted here."""
    setup = load_setup(cache=cache, site=site)
    stored = common.ENS_SOURCE_DIR / "calibration" / f"{site}.npz"
    own = common.OWN_CALIBRATION_DIR / f"{site}.npz"
    if stored.exists():
        with np.load(stored) as arrays:
            calibration = Calibration(
                temperature_shift_c=arrays["shift"],
                irradiance_ratio=arrays["ratio"],
                solar_shrink=np.ones(4),
            )
    elif own.exists():
        with np.load(own) as arrays:
            calibration = Calibration(
                temperature_shift_c=arrays["shift"],
                irradiance_ratio=arrays["ratio"],
                solar_shrink=np.ones(4),
            )
    else:
        calibration = fit_calibration(cache=cache, setup=setup)
        common.OWN_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            own,
            shift=calibration.temperature_shift_c,
            ratio=calibration.irradiance_ratio,
            shrink=calibration.solar_shrink,
        )
    return setup.with_calibration(calibration=calibration)


def main() -> None:
    """Build the ensemble arrays of the MVA-only Primaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    build_store(workers=parser.parse_args().workers)


if __name__ == "__main__":
    main()
