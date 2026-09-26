"""Score Open-Meteo's ensemble-mean products, for solar and wind power, by mean absolute error.

One-off script for the comparison in
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. **The comparison is
descriptive.** It states each product's mean absolute error over one 88-day summer window and puts
no interval or significance test on any difference, because a window that short holds too few
independent weather episodes for a month-resampled interval.

**What an arm is.** An arm is one weather product's values, given to an XGBoost model that predicts
one generator's hourly power. Each arm is fitted for each generator on its own history, scored on
weeks the fit never saw, and given the same rows and folds as every other arm.

- Solar arms read a product's global horizontal irradiance (`shortwave_radiation`): the four
  Open-Meteo ensemble means, this repository's own ECMWF ENS member mean, and CAMS, the satellite
  retrieval the earlier past-solar study found closest to the meters. CAMS is the reference, not a
  forecast.
- Wind arms read a product's wind speed. ERA5, an analysis, is the reference. The "10 m" design
  gives every product its 10 m speed, the one wind field all six products carry. The "hub height"
  design adds the 100 m speed for the products that carry it: ERA5, ICON-D2-EPS, and the two ECMWF
  ENS products. Open-Meteo serves no 100 m wind for the MOGREPS-UK and ICON-EU-EPS means, and this
  script does not scale a 10 m speed up to 100 m. Wind direction is left out of both designs,
  because Open-Meteo's ensemble mean of a direction is a naive average of member directions, which
  is wrong near north.
- The `no_weather` arm is given the shared features alone. It shows how much each weather product
  adds, and has fewer columns than the other arms by design.

**Shared features and the equal-columns rule.** Every solar arm reads the same five shared columns
(solar zenith, solar azimuth, extraterrestrial irradiance, ERA5 air temperature, and hour of day)
and one irradiance column. Every wind arm of one design reads the same shared column (hour of day)
and the same number of wind columns. `jobs` raises otherwise. XGBoost's column subsampling stays at
1 (`studies.cross_validation` never sets it). Day of year is left out because, with folds cut by
week from June to September, it would identify the scored week.

**Rows.** A row is kept only if every arm's input covers it. The target rules come from the target
and never from an arm's input. The solar rows drop multi-day zero runs, metering spikes above 1.5
times capacity, and daytime hours where the meter reads exactly zero while ERA5 reports the sun
above 100 W/m2 (`build_dataset.FALSE_ZERO_IRRADIANCE_W_M2`; ERA5 is not an arm here). The wind rows
drop every hour holding an exactly-zero half-hour, as the wind study does. Solar rows are hours with
the sun above the horizon at the hour's midpoint. Solar power is the hour ending at its label and
wind power the hour centred on its label, as in the earlier studies (`studies.power`).

**Folds.** Leave-one-week-out: 7-day blocks counted from `WINDOW_START`, the last block absorbing
any remainder (`studies.cross_validation.assign_week_folds`). Every arm, generator, and setting
shares them. Because the days beside a scored week stay in the training rows, the scheme measures
interpolation between weather episodes in one season, not forecasting into a new season.

**Two hyperparameter settings.** Every arm is fitted at `PRIMARY_HYPER_PARAMETERS` and at
`SENSITIVITY_HYPER_PARAMETERS`, three seeds each, and the report shows both.

**What the products are.** Each Open-Meteo product is one stitched series: for each hour, the mean
from the newest ensemble run that covers it, with no run time attached, so the leads are short and
mixed and no lead-resolved score is possible. The local ECMWF ENS arm is built the same way from the
repository's own member table: the mean of the 51 members, the newest run for each valid time
(leads 3 to 69 hours, from runs at 00 UTC), radiation held over its 3-hour step, and wind
interpolated linearly between steps (`studies.stitched_ensemble`). The Open-Meteo ECMWF ENS mean and
the local mean come from the same forecasts on different grids and different stitching, so a gap
between them measures the pipeline.

**Weather-error check.** The report also gives each product's own mean absolute error and bias
against the reference (CAMS irradiance in W/m2, ERA5 wind speed in km/h) on the same rows.

**Data.** The four Open-Meteo means are under
`data/studies/weather/OPEN-METEO-ENSEMBLE-MEANS/`. CAMS and ERA5 come from the original downloads,
which end on 2026-09-10 and 2026-09-11, extended by the refreshed downloads that start on
2026-08-20. Where the two overlap the refreshed download is used, and the report counts the overlap
rows on which the values differ. Only anonymised site labels (`A` to `F`, `W1` to `W3`) reach any
output; the site roster's coordinates and identifiers stay inside `build_dataset`.

Run it with `uv run python studies/open_meteo_ensemble_means/ensemble_means_mae.py`. It writes to a
new folder, `data/studies/open_meteo_ensemble_means/`, and refuses to overwrite: a re-run first
moves the existing files to a `superseded/` subfolder. `--report-only` rebuilds the tables and
`report.md` from the saved frames and losses.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import polars as pl
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    assign_week_folds,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.stitched_ensemble import (
    hold_backward_mean_hourly,
    interpolate_instants_hourly,
    newest_run_member_means,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
from build_dataset import (
    _add_solar_geometry,
    _drop_false_zeros,
    _drop_outages_and_spikes,
    _pv_sites,
    _wind_sites,
    nearest_era5_cell,
)
from build_dataset import _hourly_power as _solar_hourly_power
from export_cap import with_export_cap
from run_experiment import Job, run_all
from sources import STUDIES_DATA_DIR, WEATHER_DATA_DIR
from wind_products import _hourly_power as _wind_hourly_power

_LOG: Final[logging.Logger] = logging.getLogger("ensemble_means_mae")

DomainType = Literal["solar", "wind"]
DesignType = Literal["solar", "wind_10m", "wind_hub"]

WINDOW_START: Final[datetime] = datetime(2026, 6, 25, tzinfo=UTC)
"""The first hour of the Open-Meteo ensemble-mean archive, and the first day of the first fold."""

OUTPUT_DIR: Final[Path] = STUDIES_DATA_DIR / "open_meteo_ensemble_means"
"""Where this script writes, apart from a `superseded/` folder for re-runs."""

ENSEMBLE_MEANS_DIR: Final[Path] = WEATHER_DATA_DIR / "OPEN-METEO-ENSEMBLE-MEANS"
"""One folder per Open-Meteo ensemble-mean product, each holding a parquet named for the folder."""

OPEN_METEO_PRODUCTS: Final[dict[str, str]] = {
    "mogreps_uk_mean": "UKMO-UK-ENSEMBLE-MEAN-2KM",
    "icon_d2_eps_mean": "ICON-D2-EPS-ENSEMBLE-MEAN",
    "icon_eu_eps_mean": "ICON-EU-EPS-ENSEMBLE-MEAN",
    "ecmwf_ens_025_mean": "ECMWF-IFS-ENS-MEAN-025",
}
"""Each Open-Meteo ensemble-mean product's arm key, and the folder that holds it."""

LOCAL_ENS: Final[str] = "ecmwf_ens_local_mean"
"""The arm key of the mean of this repository's own ECMWF ENS members, stitched across runs."""

SOLAR_REFERENCE: Final[str] = "cams"
WIND_REFERENCE: Final[str] = "era5"
NO_WEATHER: Final[str] = "no_weather"

SOLAR_PRODUCTS: Final[tuple[str, ...]] = (*OPEN_METEO_PRODUCTS, LOCAL_ENS, SOLAR_REFERENCE)
"""Every product a solar arm reads, the reference last."""

WIND_PRODUCTS: Final[tuple[str, ...]] = (*OPEN_METEO_PRODUCTS, LOCAL_ENS, WIND_REFERENCE)
"""Every product a wind arm reads, the reference last."""

HUB_HEIGHT_PRODUCTS: Final[tuple[str, ...]] = (
    "icon_d2_eps_mean",
    "ecmwf_ens_025_mean",
    LOCAL_ENS,
    WIND_REFERENCE,
)
"""The products that carry a 100 m wind speed. `_read_open_meteo_mean` checks the four Open-Meteo
means against this list, so a product that gains or loses the field fails the run."""

SOLAR_SHARED_FEATURES: Final[tuple[str, ...]] = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "extraterrestrial_horizontal_w_m2",
    "temp_c",
    "hour_of_day",
)
WIND_SHARED_FEATURES: Final[tuple[str, ...]] = ("hour_of_day",)

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""Each row's absolute error, held to the export cap in force, over its own row's capacity."""

PERCENT: Final[float] = 100.0

MAX_CONCURRENT_FITS: Final[int] = 2
"""Two fits at a time, each on 4 cores (`studies.cross_validation.THREADS_PER_FIT`): 8 threads."""

EXPECTED_MEMBERS: Final[int] = 51
"""The members in each ECMWF ENS run."""

LOCAL_ENS_HORIZONS: Final[tuple[str, ...]] = ("T+3", "day+1", "day+2")
"""The lead bands with a 3-hourly step, leads 3 to 69 hours, which chain with no gap."""

LOCAL_ENS_MAX_LEAD_HOURS: Final[int] = 69
LOCAL_ENS_RUN_LOOKBACK_DAYS: Final[int] = 3
"""How many days before `WINDOW_START` a run may start and still supply the first hours."""

MS_TO_KM_PER_H: Final[float] = 3.6

ENS_DIR: Final[Path] = WEATHER_DATA_DIR / "ENS"
CAMS_PATHS: Final[tuple[Path, ...]] = (
    WEATHER_DATA_DIR / "CAMS" / "beam_diffuse_cams.parquet",
    WEATHER_DATA_DIR / "CAMS" / "beam_diffuse_cams_2026-08-20_2026-09-21.parquet",
)
ERA5_GRID_PATHS: Final[tuple[Path, ...]] = (
    WEATHER_DATA_DIR / "ERA5" / "beam_diffuse_open_meteo.parquet",
    WEATHER_DATA_DIR / "ERA5" / "beam_diffuse_open_meteo_2026-08-20_2026-09-21.parquet",
)
ERA5_WIND_PATHS: Final[tuple[Path, ...]] = (
    WEATHER_DATA_DIR / "ERA5" / "wind_era5.parquet",
    WEATHER_DATA_DIR / "ERA5" / "wind_era5_2026-08-20_2026-09-21.parquet",
)
"""Each pair is the original download, then the refreshed one, which wins where they overlap."""

ANONYMITY_FORBIDDEN_COLUMNS: Final[frozenset[str]] = frozenset(
    {"latitude", "longitude", "cell_latitude", "cell_longitude", "time_series_id"}
)
"""Columns that would identify a generator, which no saved frame may carry."""


def _read_extended(*, paths: Sequence[Path], key: Sequence[str]) -> tuple[pl.DataFrame, int]:
    """Read an original download and its refresh, the refresh winning where they overlap.

    Args:
        paths: The original download, then each refreshed download.
        key: The columns that identify a row.

    Returns:
        The stacked rows from `WINDOW_START` on, sorted by `key`, and the number of overlap rows
        whose values differ between the downloads.
    """
    stacked = pl.concat(
        [
            pl.read_parquet(path).with_columns(_source=pl.lit(index))
            for index, path in enumerate(paths)
        ]
    ).filter(pl.col("time") >= WINDOW_START)
    values = [name for name in stacked.columns if name not in (*key, "_source")]
    overlaps = stacked.filter(pl.struct(*key).is_duplicated())
    differing = (
        overlaps.group_by(*key)
        .agg([pl.col(name).n_unique() for name in values])
        .filter(pl.any_horizontal([pl.col(name) > 1 for name in values]))
        .height
    )
    merged = (
        stacked.sort(*key, "_source")
        .unique(subset=list(key), keep="last", maintain_order=True)
        .drop("_source")
        .sort(*key)
    )
    return merged, differing


def _read_open_meteo_mean(*, product: str, columns: dict[str, str]) -> pl.DataFrame:
    """Read one Open-Meteo ensemble-mean product, renaming the named columns.

    Args:
        product: A key of `OPEN_METEO_PRODUCTS`.
        columns: Each source column to keep, and the name to give it.

    Returns:
        One row per (site, hour), from `WINDOW_START`, with a timezone-aware `time`.

    Raises:
        ValueError: If the product's 100 m wind is present where `HUB_HEIGHT_PRODUCTS` says it is
            absent, or absent where it says present.
    """
    directory = OPEN_METEO_PRODUCTS[product]
    raw = pl.read_parquet(ENSEMBLE_MEANS_DIR / directory / f"{directory}.parquet")
    has_100m = raw["wind_speed_100m"].null_count() < raw.height
    if has_100m != (product in HUB_HEIGHT_PRODUCTS):
        msg = (
            f"{product}: a 100 m wind speed is {'present' if has_100m else 'absent'}, unexpectedly"
        )
        raise ValueError(msg)
    return (
        raw.select(
            "site",
            pl.col("time").dt.replace_time_zone("UTC"),
            *(pl.col(source).alias(name) for source, name in columns.items()),
        )
        .filter(pl.col("time") >= WINDOW_START)
        .sort("site", "time")
    )


def _local_ens_members(*, path: Path, value_columns: Sequence[str]) -> pl.DataFrame:
    """Read this repository's ENS member table over the lead bands that chain in 3-hour steps.

    Args:
        path: The member parquet under `data/studies/weather/ENS/`.
        value_columns: The columns to keep beside the keys.

    Returns:
        One row per (site, run, valid time, member) for the runs that can reach the window.
    """
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("horizon").is_in(LOCAL_ENS_HORIZONS),
            pl.col("init_time") >= WINDOW_START - timedelta(days=LOCAL_ENS_RUN_LOOKBACK_DAYS),
        )
        .select("site", "init_time", "valid_time", "lead_hours", "ensemble_member", *value_columns)
        .collect()
    )


def _local_ens_solar() -> pl.DataFrame:
    """Build the local ENS member mean of global irradiance, stitched and held over each step.

    Returns:
        One row per (site, hour ending at `time`) with `ghi_ecmwf_ens_local_mean`.
    """
    steps = newest_run_member_means(
        members=_local_ens_members(
            path=ENS_DIR / "beam_diffuse_ens.parquet", value_columns=["ghi_w_m2"]
        ),
        value_columns=["ghi_w_m2"],
        expected_members=EXPECTED_MEMBERS,
        max_lead_hours=LOCAL_ENS_MAX_LEAD_HOURS,
    )
    return hold_backward_mean_hourly(steps=steps, value_columns=["ghi_w_m2"]).rename(
        {"ghi_w_m2": f"ghi_{LOCAL_ENS}"}
    )


def _local_ens_wind() -> pl.DataFrame:
    """Build the local ENS member mean of wind speed in km/h, stitched and interpolated hourly.

    Returns:
        One row per (site, hour) with `speed10_ecmwf_ens_local_mean` and
        `speed100_ecmwf_ens_local_mean`.
    """
    columns = ["speed_10m_ms", "speed_100m_ms"]
    steps = newest_run_member_means(
        members=_local_ens_members(
            path=ENS_DIR / "beam_diffuse_ens_wind.parquet", value_columns=columns
        ),
        value_columns=columns,
        expected_members=EXPECTED_MEMBERS,
        max_lead_hours=LOCAL_ENS_MAX_LEAD_HOURS,
    )
    return interpolate_instants_hourly(steps=steps, value_columns=columns).select(
        "site",
        "time",
        pl.col("speed_10m_ms").mul(MS_TO_KM_PER_H).alias(f"speed10_{LOCAL_ENS}"),
        pl.col("speed_100m_ms").mul(MS_TO_KM_PER_H).alias(f"speed100_{LOCAL_ENS}"),
    )


def _joined_with_funnel(
    *, frame: pl.DataFrame, other: pl.DataFrame, name: str, funnel: list[tuple[str, int]]
) -> pl.DataFrame:
    """Inner-join one more product's columns and record how many rows survive.

    Args:
        frame: The rows so far.
        other: One product's columns, keyed by `site` and `time`.
        name: The product, for the funnel.
        funnel: Each step's name and row count so far, appended to.

    Returns:
        The joined rows.
    """
    joined = frame.join(other, on=["site", "time"], how="inner")
    funnel.append((f"joined {name}", joined.height))
    return joined


def build_solar_frame() -> tuple[pl.DataFrame, list[tuple[str, int]], dict[str, int]]:
    """Build the solar rows: power, features, folds, and every product's irradiance.

    Returns:
        The rows sorted by site and time, the row funnel, and the count of overlap rows on which the
        original and refreshed downloads differ, by download name.
    """
    funnel: list[tuple[str, int]] = []
    sites = _pv_sites()
    era5, era5_differing = _read_extended(
        paths=ERA5_GRID_PATHS, key=["time", "latitude", "longitude"]
    )
    cams, cams_differing = _read_extended(paths=CAMS_PATHS, key=["site", "time"])
    power = _drop_outages_and_spikes(power=_solar_hourly_power(sites=sites), sites=sites).filter(
        pl.col("time") >= WINDOW_START
    )
    funnel.append(("hourly power after outage and spike rules", power.height))
    cells = nearest_era5_cell(sites=sites, era5=era5)
    joined = (
        power.join(cells, on=["site", "effective_capacity_mw"])
        .join(
            era5.select("time", "latitude", "longitude", "ghi_w_m2", "temp_c"),
            left_on=["time", "cell_latitude", "cell_longitude"],
            right_on=["time", "latitude", "longitude"],
            how="inner",
        )
        .drop("time_series_id", "cell_latitude", "cell_longitude")
    )
    funnel.append(("joined ERA5 grid cells", joined.height))
    kept = _drop_false_zeros(joined=joined).drop("ghi_w_m2")
    funnel.append(("after the false-zero rule", kept.height))
    daylight = _add_solar_geometry(joined=kept).filter(pl.col("solar_elevation_deg") > 0.0)
    funnel.append(("sun above the horizon", daylight.height))
    frame = with_export_cap(
        dataset=daylight.drop("latitude", "longitude").with_columns(
            hour_of_day=pl.col("time").dt.hour(), month=pl.col("time").dt.strftime("%Y-%m")
        )
    )
    frame = _joined_with_funnel(
        frame=frame,
        other=cams.select("site", "time", pl.col("ghi_w_m2").alias(f"ghi_{SOLAR_REFERENCE}")),
        name=SOLAR_REFERENCE,
        funnel=funnel,
    )
    for product in OPEN_METEO_PRODUCTS:
        frame = _joined_with_funnel(
            frame=frame,
            other=_read_open_meteo_mean(
                product=product, columns={"shortwave_radiation": f"ghi_{product}"}
            ),
            name=product,
            funnel=funnel,
        )
    frame = _joined_with_funnel(
        frame=frame, other=_local_ens_solar(), name=LOCAL_ENS, funnel=funnel
    )
    rows = assign_week_folds(dataset=frame.sort("site", "time"), first_day=WINDOW_START)
    return rows, funnel, {"ERA5 grid": era5_differing, "CAMS": cams_differing}


def build_wind_frame() -> tuple[pl.DataFrame, list[tuple[str, int]], dict[str, int]]:
    """Build the wind rows: power, features, folds, and every product's 10 m and 100 m speeds.

    Returns:
        The rows sorted by site and time, the row funnel, and the count of overlap rows on which the
        original and refreshed ERA5 wind downloads differ.
    """
    funnel: list[tuple[str, int]] = []
    sites = _wind_sites()
    era5, era5_differing = _read_extended(paths=ERA5_WIND_PATHS, key=["site", "time"])
    power = (
        _wind_hourly_power(sites=sites, centred=True)
        .join(sites.select("site", "effective_capacity_mw"), on="site")
        .filter(pl.col("time") >= WINDOW_START)
    )
    funnel.append(("hourly power", power.height))
    frame = power.filter(~pl.col("has_zero_half_hour")).with_columns(
        hour_of_day=pl.col("time").dt.hour(),
        month=pl.col("time").dt.strftime("%Y-%m"),
        constrained=pl.lit(value=False),
        cap_mw=pl.lit(None, dtype=pl.Float64),
    )
    funnel.append(("no exactly-zero half-hour", frame.height))
    frame = _joined_with_funnel(
        frame=frame,
        other=era5.select(
            "site",
            "time",
            pl.col("wind_speed_10m").alias(f"speed10_{WIND_REFERENCE}"),
            pl.col("wind_speed_100m").alias(f"speed100_{WIND_REFERENCE}"),
        ),
        name=WIND_REFERENCE,
        funnel=funnel,
    )
    for product in OPEN_METEO_PRODUCTS:
        columns = {"wind_speed_10m": f"speed10_{product}"}
        if product in HUB_HEIGHT_PRODUCTS:
            columns["wind_speed_100m"] = f"speed100_{product}"
        frame = _joined_with_funnel(
            frame=frame,
            other=_read_open_meteo_mean(product=product, columns=columns),
            name=product,
            funnel=funnel,
        )
    frame = _joined_with_funnel(frame=frame, other=_local_ens_wind(), name=LOCAL_ENS, funnel=funnel)
    rows = assign_week_folds(dataset=frame.sort("site", "time"), first_day=WINDOW_START)
    return rows.drop("has_zero_half_hour"), funnel, {"ERA5 wind": era5_differing}


def _feature_columns(*, design: DesignType, product: str) -> tuple[str, ...]:
    """Return the weather columns one product's arm reads in one design.

    Args:
        design: The design.
        product: The product.

    Returns:
        The column names.
    """
    if design == "solar":
        return (f"ghi_{product}",)
    if design == "wind_10m":
        return (f"speed10_{product}",)
    return (f"speed100_{product}", f"speed10_{product}")


def arm_features(*, design: DesignType) -> dict[str, tuple[str, ...]]:
    """Return every arm of one design, and the feature columns each is shown.

    Args:
        design: The design.

    Returns:
        Each arm's name and its columns, the shared features first, the `no_weather` arm last.
    """
    shared = SOLAR_SHARED_FEATURES if design == "solar" else WIND_SHARED_FEATURES
    products = {
        "solar": SOLAR_PRODUCTS,
        "wind_10m": WIND_PRODUCTS,
        "wind_hub": HUB_HEIGHT_PRODUCTS,
    }[design]
    arms = {
        product: (*shared, *_feature_columns(design=design, product=product))
        for product in products
    }
    arms[NO_WEATHER] = shared
    return arms


def jobs(*, design: DesignType) -> list[Job]:
    """Return every arm's job at both hyperparameter settings.

    Args:
        design: The design.

    Returns:
        One job per (arm, setting), the target `power_mw`, no quantile model.

    Raises:
        ValueError: Unless every weather arm of the design carries the same number of columns.
    """
    arms = arm_features(design=design)
    counts = {len(columns) for arm, columns in arms.items() if arm != NO_WEATHER}
    if len(counts) != 1:
        msg = f"{design}: every weather arm should carry the same number of columns, found {counts}"
        raise ValueError(msg)
    settings = {"primary": PRIMARY_HYPER_PARAMETERS, "sensitivity": SENSITIVITY_HYPER_PARAMETERS}
    return [
        (f"{design}:{arm}", setting, "power_mw", columns, hyper_parameters, False)
        for setting, hyper_parameters in settings.items()
        for arm, columns in arms.items()
    ]


def _check_anonymous(*, frame: pl.DataFrame) -> None:
    """Raise if a frame about to be saved carries a column that identifies a generator.

    Args:
        frame: The frame.

    Raises:
        ValueError: Naming the offending columns.
    """
    found = ANONYMITY_FORBIDDEN_COLUMNS.intersection(frame.columns)
    if found:
        msg = f"refusing to save identifying columns {sorted(found)}"
        raise ValueError(msg)


def _check_same_rows(*, losses: pl.DataFrame) -> None:
    """Raise unless every arm at every setting was scored on the same site-hours and seeds.

    Args:
        losses: Every job's losses, with `arm` and `setting`.

    Raises:
        ValueError: If two arms' scored rows differ.
    """
    keys = losses.group_by("arm", "setting").agg(
        rows=pl.struct("site", "time", "seed").sort_by("site", "time", "seed").hash().sum(),
        n=pl.len(),
    )
    if keys.select("rows", "n").n_unique() != 1:
        msg = f"arms were scored on different rows:\n{keys}"
        raise ValueError(msg)


def _absolute_table(
    *, losses: pl.DataFrame, reference: str | None, design: DesignType
) -> pl.DataFrame:
    """Summarise each arm's error, its seed range, and its weeks against the reference.

    Args:
        losses: One design's losses.
        reference: The reference arm's name, or `None` for a design with no reference.
        design: The design, to label the rows.

    Returns:
        One row per (arm, setting): `mae_pct` and `mae_pct_seed_min` and `_max`, the mean across
        rows of the capped absolute error as a percentage of capacity; `n_rows`; `weeks_better`,
        the number of held-out weeks in which the arm's error is below the reference's, and
        `n_weeks`; and `minus_reference_points`.
    """
    per_seed = losses.group_by("arm", "setting", "seed").agg(mae=pl.col(METRIC).mean() * PERCENT)
    summary = (
        losses.group_by("arm", "setting")
        .agg(mae_pct=pl.col(METRIC).mean() * PERCENT, n_rows=pl.len())
        .join(
            per_seed.group_by("arm", "setting").agg(
                mae_pct_seed_min=pl.col("mae").min(), mae_pct_seed_max=pl.col("mae").max()
            ),
            on=["arm", "setting"],
        )
    )
    per_week = losses.group_by("arm", "setting", "fold").agg(week_mae=pl.col(METRIC).mean())
    if reference is None:
        return summary.with_columns(design=pl.lit(design)).sort("setting", "mae_pct")
    reference_weeks = per_week.filter(pl.col("arm") == f"{design}:{reference}").select(
        "setting", "fold", reference_mae=pl.col("week_mae")
    )
    weeks = (
        per_week.join(reference_weeks, on=["setting", "fold"])
        .group_by("arm", "setting")
        .agg(
            weeks_better=(pl.col("week_mae") < pl.col("reference_mae")).sum(),
            n_weeks=pl.len(),
        )
    )
    reference_mae = summary.filter(pl.col("arm") == f"{design}:{reference}").select(
        "setting", reference_mae_pct=pl.col("mae_pct")
    )
    return (
        summary.join(weeks, on=["arm", "setting"])
        .join(reference_mae, on="setting")
        .with_columns(
            minus_reference_points=pl.col("mae_pct") - pl.col("reference_mae_pct"),
            design=pl.lit(design),
        )
        .drop("reference_mae_pct")
        .sort("setting", "mae_pct")
    )


def _per_site_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Summarise each arm's mean absolute error at each anonymised generator.

    Args:
        losses: One design's losses.

    Returns:
        One row per (arm, setting, site) with `mae_pct`.
    """
    return (
        losses.group_by("arm", "setting", "site")
        .agg(mae_pct=pl.col(METRIC).mean() * PERCENT)
        .sort("setting", "site", "mae_pct")
    )


def _weather_error_table(*, frame: pl.DataFrame, design: DesignType) -> pl.DataFrame:
    """Measure each product's own error against the reference, on the rows every arm shares.

    Args:
        frame: The rows.
        design: `solar`, `wind_10m`, or `wind_hub`.

    Returns:
        One row per (variable, product): the mean absolute error, the mean signed error (product
        minus reference), and the reference's own mean, in the variable's unit.
    """
    reference = SOLAR_REFERENCE if design == "solar" else WIND_REFERENCE
    products = SOLAR_PRODUCTS if design == "solar" else WIND_PRODUCTS
    variables = {"solar": {"ghi": "W/m2"}, "wind_10m": {"speed10": "km/h"}}.get(
        design, {"speed100": "km/h"}
    )
    rows = []
    for prefix, unit in variables.items():
        reference_column = f"{prefix}_{reference}"
        for product in products:
            column = f"{prefix}_{product}"
            if product == reference or column not in frame.columns:
                continue
            difference = pl.col(column) - pl.col(reference_column)
            rows.append(
                frame.select(
                    variable=pl.lit(prefix),
                    unit=pl.lit(unit),
                    product=pl.lit(product),
                    reference=pl.lit(reference),
                    mean_absolute_error=difference.abs().mean(),
                    mean_signed_error=difference.mean(),
                    reference_mean=pl.col(reference_column).mean(),
                    n_rows=pl.len(),
                )
            )
    return pl.concat(rows).sort("variable", "mean_absolute_error")


def _distinct_series(*, product: str, column: str, sites: Sequence[str]) -> int:
    """Count the distinct weather series a product serves across some sites.

    Sites in one grid cell get identical series, so the count is at most the number of cells.

    Args:
        product: A key of `OPEN_METEO_PRODUCTS`.
        column: The source column to compare.
        sites: The anonymised site labels to count over.

    Returns:
        The number of distinct series among `sites`.
    """
    directory = OPEN_METEO_PRODUCTS[product]
    raw = pl.read_parquet(ENSEMBLE_MEANS_DIR / directory / f"{directory}.parquet")
    series = (
        raw.filter(pl.col("site").is_in(list(sites)))
        .sort("site", "time")
        .group_by("site")
        .agg(pl.col(column))[column]
    )
    return len({tuple(values) for values in series.to_list()})


def _markdown_table(*, frame: pl.DataFrame, decimals: int = 3) -> list[str]:
    """Render a frame as a markdown table, rounding floats.

    Args:
        frame: The frame.
        decimals: How many decimals a float keeps.

    Returns:
        Markdown lines.
    """
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = [
        "| "
        + " | ".join(
            f"{value:.{decimals}f}" if isinstance(value, float) else str(value) for value in row
        )
        + " |"
        for row in frame.iter_rows()
    ]
    return [header, rule, *body]


def _report(
    *,
    frames: dict[DomainType, pl.DataFrame],
    funnels: dict[DomainType, list[tuple[str, int]]],
    differing: dict[str, int],
    tables: dict[DesignType, dict[str, pl.DataFrame]],
    weather_errors: dict[DesignType, pl.DataFrame],
) -> str:
    """Render every number the study page quotes, as markdown.

    Args:
        frames: Each domain's rows.
        funnels: Each domain's row funnel.
        differing: The overlap rows that differ between an original and its refreshed download.
        tables: Each design's `absolute` and `per_site` tables.
        weather_errors: Each design's weather-error table.

    Returns:
        The report.
    """
    lines = ["# Open-Meteo ensemble means: power mean absolute error", ""]
    lines += ["## Data checks", ""]
    for domain, frame in frames.items():
        first, last = frame["time"].min(), frame["time"].max()
        lines += [
            (
                f"### {domain}: {frame.height} rows, {frame['site'].n_unique()} sites, "
                f"{first} to {last}"
            ),
            "",
            *_markdown_table(
                frame=frame.group_by("site")
                .agg(
                    rows=pl.len(),
                    constrained_rows=pl.col("constrained").sum(),
                    weeks=pl.col("fold").n_unique(),
                )
                .sort("site")
            ),
            "",
            "Row funnel:",
            "",
            *(f"- {name}: {count}" for name, count in funnels[domain]),
            "",
        ]
    lines += [
        "Overlap rows whose values differ between an original and its refreshed download: "
        + ", ".join(f"{name} {count}" for name, count in differing.items()),
        "",
        "Distinct weather series among sites, by Open-Meteo product (one grid cell, one series):",
        "",
    ]
    solar_sites = sorted(frames["solar"]["site"].unique().to_list())
    wind_sites = sorted(frames["wind"]["site"].unique().to_list())
    for product in OPEN_METEO_PRODUCTS:
        solar_count = _distinct_series(
            product=product, column="shortwave_radiation", sites=solar_sites
        )
        wind_count = _distinct_series(product=product, column="wind_speed_10m", sites=wind_sites)
        lines.append(
            f"- {product}: {solar_count} among {len(solar_sites)} solar sites, "
            f"{wind_count} among {len(wind_sites)} wind sites"
        )
    lines += ["", "## Every arm's feature columns", ""]
    for design in tables:
        for arm, columns in arm_features(design=design).items():
            lines.append(
                f"- `{design}:{arm}` ({len(columns)} columns): "
                + ", ".join(f"`{c}`" for c in columns)
            )
    lines.append("")
    for design, design_tables in tables.items():
        lines += [f"## {design}: mean absolute error, % of capacity", ""]
        for setting in ("primary", "sensitivity"):
            lines += [f"### {setting} setting", ""]
            lines += _markdown_table(
                frame=design_tables["absolute"]
                .filter(pl.col("setting") == setting)
                .drop("setting", "design")
            )
            lines.append("")
        lines += [f"### {design}: per generator, primary setting", ""]
        lines += _markdown_table(
            frame=design_tables["per_site"].filter(pl.col("setting") == "primary").drop("setting")
        )
        lines += ["", f"### {design}: each product's weather error against the reference", ""]
        lines += _markdown_table(frame=weather_errors[design])
        lines.append("")
    return "\n".join(lines)


def _analyse(
    *,
    frames: dict[DomainType, pl.DataFrame],
    losses: dict[DesignType, pl.DataFrame],
) -> tuple[dict[DesignType, dict[str, pl.DataFrame]], dict[DesignType, pl.DataFrame]]:
    """Summarise every design's losses and weather errors.

    Args:
        frames: Each domain's rows.
        losses: Each design's losses.

    Returns:
        Each design's `absolute` and `per_site` tables, and each design's weather-error table.
    """
    references: dict[DesignType, str] = {
        "solar": SOLAR_REFERENCE,
        "wind_10m": WIND_REFERENCE,
        "wind_hub": WIND_REFERENCE,
    }
    tables: dict[DesignType, dict[str, pl.DataFrame]] = {}
    weather_errors: dict[DesignType, pl.DataFrame] = {}
    for design, design_losses in losses.items():
        _check_same_rows(losses=design_losses)
        tables[design] = {
            "absolute": _absolute_table(
                losses=design_losses, reference=references[design], design=design
            ),
            "per_site": _per_site_table(losses=design_losses),
        }
        domain: DomainType = "solar" if design == "solar" else "wind"
        weather_errors[design] = _weather_error_table(frame=frames[domain], design=design)
    return tables, weather_errors


def _domain_of(*, design: DesignType) -> DomainType:
    """Return the domain a design belongs to."""
    return "solar" if design == "solar" else "wind"


DESIGNS: Final[tuple[DesignType, ...]] = ("solar", "wind_10m", "wind_hub")


def _paths() -> dict[str, Path]:
    """Return every file the script writes, by name."""
    names = ["report.md", "run_facts.json", "mae_by_arm.parquet", "mae_by_site.parquet"]
    names += ["weather_error.parquet"]
    names += [f"frame_{domain}.parquet" for domain in ("solar", "wind")]
    names += [f"losses_{design}.parquet" for design in DESIGNS]
    return {name: OUTPUT_DIR / name for name in names}


def _write_results(
    *,
    frames: dict[DomainType, pl.DataFrame],
    funnels: dict[DomainType, list[tuple[str, int]]],
    differing: dict[str, int],
    losses: dict[DesignType, pl.DataFrame],
    only_report: bool,
) -> None:
    """Write the tables and the report, and the frames and losses unless only the report is wanted.

    Args:
        frames: Each domain's rows.
        funnels: Each domain's row funnel.
        differing: The overlap rows that differ between an original and its refreshed download.
        losses: Each design's losses.
        only_report: Whether the frames and losses are already saved.
    """
    paths = _paths()
    derived = ["report.md", "mae_by_arm.parquet", "mae_by_site.parquet", "weather_error.parquet"]
    refuse_to_overwrite(paths=[paths[name] for name in (derived if only_report else paths)])
    tables, weather_errors = _analyse(frames=frames, losses=losses)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not only_report:
        facts = {"funnels": funnels, "differing": differing}
        paths["run_facts.json"].write_text(json.dumps(facts, indent=2))
        for domain, frame in frames.items():
            _check_anonymous(frame=frame)
            frame.write_parquet(paths[f"frame_{domain}.parquet"])
        for design, design_losses in losses.items():
            design_losses.write_parquet(paths[f"losses_{design}.parquet"])
    pl.concat([tables[design]["absolute"] for design in DESIGNS]).write_parquet(
        paths["mae_by_arm.parquet"]
    )
    pl.concat(
        [tables[design]["per_site"].with_columns(design=pl.lit(design)) for design in DESIGNS]
    ).write_parquet(paths["mae_by_site.parquet"])
    pl.concat(
        [weather_errors[design].with_columns(design=pl.lit(design)) for design in DESIGNS]
    ).write_parquet(paths["weather_error.parquet"])
    report = _report(
        frames=frames,
        funnels=funnels,
        differing=differing,
        tables=tables,
        weather_errors=weather_errors,
    )
    paths["report.md"].write_text(report)
    _LOG.info("wrote %s", OUTPUT_DIR)
    print(report)


def _fit(*, frames: dict[DomainType, pl.DataFrame]) -> dict[DesignType, pl.DataFrame]:
    """Fit every design's jobs, checking the columns each arm reads first.

    Args:
        frames: Each domain's rows.

    Returns:
        Each design's losses, `arm` labelled `<design>:<arm>`.
    """
    losses: dict[DesignType, pl.DataFrame] = {}
    for design in DESIGNS:
        frame = frames[_domain_of(design=design)]
        design_jobs = jobs(design=design)
        check_no_missing(frame=frame, columns=[column for job in design_jobs for column in job[3]])
        losses[design] = run_all(dataset=frame, jobs=design_jobs, max_workers=MAX_CONCURRENT_FITS)
    return losses


def main() -> int:
    """Build the rows, fit every arm, and write the tables and the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the tables and the report from the saved frames and losses.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    paths = _paths()
    if arguments.report_only:
        frames: dict[DomainType, pl.DataFrame] = {
            "solar": pl.read_parquet(paths["frame_solar.parquet"]),
            "wind": pl.read_parquet(paths["frame_wind.parquet"]),
        }
        saved = {design: pl.read_parquet(paths[f"losses_{design}.parquet"]) for design in DESIGNS}
        facts = json.loads(paths["run_facts.json"].read_text())
        funnels: dict[DomainType, list[tuple[str, int]]] = facts["funnels"]
        differing: dict[str, int] = facts["differing"]
        _write_results(
            frames=frames, funnels=funnels, differing=differing, losses=saved, only_report=True
        )
        return 0
    refuse_to_overwrite(paths=paths.values())
    solar, solar_funnel, solar_differing = build_solar_frame()
    wind, wind_funnel, wind_differing = build_wind_frame()
    built: dict[DomainType, pl.DataFrame] = {"solar": solar, "wind": wind}
    built_funnels: dict[DomainType, list[tuple[str, int]]] = {
        "solar": solar_funnel,
        "wind": wind_funnel,
    }
    _write_results(
        frames=built,
        funnels=built_funnels,
        differing=solar_differing | wind_differing,
        losses=_fit(frames=built),
        only_report=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
