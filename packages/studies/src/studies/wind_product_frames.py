"""Build the wind past-weather study's common rows, which every later wind study starts from.

Written for the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/826>. It
holds each wind product's download location, the hourly power a wind generator is scored on, the
feature columns of every arm, and the common rows. The blending study and the ENS forecast study
read these rows, so they live here and not in a script. Scripts in
`studies/nwp_forecast_comparison/`, `studies/open_meteo_ensemble_means/`, and
`studies/past_weather/` import it.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from studies.arm_runner import Job
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SENSITIVITY_HYPER_PARAMETERS
from studies.era5_grid import suffixed
from studies.neighbouring_hours import with_neighbouring_hours
from studies.power import hourly_from_half_hourly
from studies.pv_dataset import POWER_DELTA_URI
from studies.solar_product_frames import UPGRADE_DAY
from studies.sources import HISTORICAL_FORECAST_URL, site_points_dir_for

ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
"""Open-Meteo's reanalysis endpoint, which serves ERA5 with the same query shape."""


PRODUCTS: Final[dict[str, tuple[str, str]]] = {
    "era5": ("era5", ARCHIVE_URL),
    "ukv": ("ukmo_uk_deterministic_2km", HISTORICAL_FORECAST_URL),
    "icon_d2": ("icon_d2", HISTORICAL_FORECAST_URL),
    "icon_eu": ("icon_eu", HISTORICAL_FORECAST_URL),
    "icon_global": ("icon_global", HISTORICAL_FORECAST_URL),
}
"""Each product's `models=` value and endpoint."""


def output_path_for(*, product: str) -> Path:
    """Return where one product's wind download is written.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        The parquet path.
    """
    return suffixed(
        site_points_dir_for(product=product.upper().replace("_", "-")) / f"wind_{product}.parquet"
    )


OUTPUT_DIR_NAME: Final[str] = "beam_diffuse_wind_products"
"""The results directory under `STUDY_DATA_DIR`."""


SHARED_FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "day_of_year", "era_code")
"""Features every arm gets, on top of its product's wind."""


def wind_hourly_power(*, sites: pl.DataFrame, centred: bool = True) -> pl.DataFrame:
    """Average each wind generator's half-hours onto an hour centred on its label.

    Shifting every stamp back 30 minutes before `hourly_from_half_hourly` means the hour labelled T
    holds the half-hours ending at T and at T + 30 min, which span T - 30 min to T + 30 min.

    Args:
        sites: The wind roster.
        centred: Whether to centre the hour on its label; False gives the solar study's hour,
            ending at the label, for the `hour_ending` setting.

    Returns:
        One row per (site, time) with `power_mw` and `has_zero_half_hour`.
    """
    half_hourly = (
        pl.scan_delta(POWER_DELTA_URI)
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .collect()
        .join(sites.select("time_series_id", "site"), on="time_series_id")
        .select(
            "site",
            time=pl.col("time").dt.offset_by("-30m" if centred else "0m"),
            power_mw=pl.col("power"),
        )
    )
    return hourly_from_half_hourly(half_hourly=half_hourly)


def wind_columns(*, product: str) -> tuple[str, str, str, str]:
    """Return one product's four wind feature names.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        The hub-height speed, that height's direction as sine and cosine, and the 10 m speed.
    """
    return (
        f"speed_hub_{product}",
        f"direction_sin_{product}",
        f"direction_cos_{product}",
        f"speed_10m_{product}",
    )


def served_100m_columns(*, product: str) -> tuple[str, str, str, str]:
    """Return one product's served 100 m speed and direction, and its 10 m speed.

    This is the arm the plan specified before the first run, for every product.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        The 100 m speed, the 100 m direction's sine and cosine, and the 10 m speed.
    """
    return (
        f"speed_100m_{product}",
        f"sin_100m_{product}",
        f"cos_100m_{product}",
        f"speed_10m_{product}",
    )


UKV_80M_COLUMNS: Final[tuple[str, str, str, str]] = (
    "speed_80m_ukv",
    "sin_80m_ukv",
    "cos_80m_ukv",
    "speed_10m_ukv",
)
"""UKV's served 80 m speed and direction and its 10 m speed, for the check arm `ukv_80m`."""


STEP_DATES: Final[tuple[datetime, datetime]] = (
    datetime(2025, 6, 2, tzinfo=UTC),
    datetime(2026, 6, 2, tzinfo=UTC),
)
"""The two days ICON global's served wind steps at one generator, relative to ICON-EU's.

The `_step` arms are told which of the three periods each hour falls in, which measures how much of
ICON global's deficit the steps explain. ERA5 and ICON-EU get the same flag, so ICON global told the
steps is compared with rivals that carry the same extra column.
"""


STEP_ARM_PRODUCTS: Final[tuple[str, ...]] = ("era5", "icon_eu", "icon_global")
"""The products given a `_step` arm: ICON global, and the two rivals it is compared with."""


ROW_SET_SETTINGS: Final[tuple[str, ...]] = ("hour_ending", "keep_zero_hours")
"""Settings fitted on a different row set from the main one, each for every product's wind arm.

`hour_ending` builds the power hour as the solar study does, ending at the label, and
`keep_zero_hours` keeps the hours holding an exactly-zero half-hour. Both are post hoc checks.
"""


def hub_height_m(*, product: str) -> int:
    """Return the height in metres of the wind a product's arm is shown.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        80 for the ICON products, whose native heights are 80 m and 120 m, and 100 otherwise.
    """
    return 80 if product.startswith("icon") else 100


HUB_OFFSETS_HOURS: Final[tuple[int, ...]] = (-2, -1, 1, 2)
"""The neighbouring hours of hub-height speed a context arm is shown."""


SURFACE_OFFSETS_HOURS: Final[tuple[int, ...]] = (-1, 1)
"""The neighbouring hours of 10 m speed a context arm is shown."""


def offset_label(offset_hours: int) -> str:
    """Return an offset's name stem, such as `minus2h` or `plus1h`.

    Args:
        offset_hours: The offset from the row's hour.

    Returns:
        The stem.
    """
    return f"{'minus' if offset_hours < 0 else 'plus'}{abs(offset_hours)}h"


def context_columns(*, product: str) -> tuple[str, ...]:
    """Return one product's neighbouring-hour speed columns, which `with_wind_context` adds.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        The hub-height speed at each of `HUB_OFFSETS_HOURS`, then the 10 m speed at each of
        `SURFACE_OFFSETS_HOURS`.
    """
    return (
        *(f"speed_hub_{offset_label(offset)}_{product}" for offset in HUB_OFFSETS_HOURS),
        *(f"speed_10m_{offset_label(offset)}_{product}" for offset in SURFACE_OFFSETS_HOURS),
    )


def with_wind_context(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add every product's hub-height and 10 m speeds in the hours around each row.

    The neighbours are read from each product's own download, not from the scored rows, which
    exclude hours by the target. Each download is checked to reproduce the frame's own hub-height
    and 10 m speed columns at offset zero first, so a neighbour cannot come from a series labelled
    differently.

    Args:
        frame: The common rows, carrying `speed_hub_<product>` and `speed_10m_<product>` for every
            product, as `wind_columns` names them.

    Returns:
        `frame`, in its own row order, with `context_columns(product=...)` for every product.

    Raises:
        ValueError: If a download does not reproduce the frame's columns at offset zero.
    """
    for product in PRODUCTS:
        hub = f"wind_speed_{hub_height_m(product=product)}m"
        offsets = [(hub, offset) for offset in HUB_OFFSETS_HOURS]
        offsets += [("wind_speed_10m", offset) for offset in SURFACE_OFFSETS_HOURS]
        own_hub, _, _, own_surface = wind_columns(product=product)
        hub_at_zero, surface_at_zero = f"{own_hub}_at_zero", f"{own_surface}_at_zero"
        zero_columns = {hub_at_zero: (hub, 0), surface_at_zero: ("wind_speed_10m", 0)}
        frame = with_neighbouring_hours(
            frame=frame,
            source=pl.read_parquet(output_path_for(product=product)),
            columns={
                **zero_columns,
                **dict(zip(context_columns(product=product), offsets, strict=True)),
            },
        )
        for own, at_zero in ((own_hub, hub_at_zero), (own_surface, surface_at_zero)):
            if not frame[at_zero].cast(pl.Float64).equals(frame[own].cast(pl.Float64)):
                msg = f"the {product} download does not reproduce {own} at offset zero"
                raise ValueError(msg)
        frame = frame.drop(*zero_columns)
    return frame


def joined(*, sites: pl.DataFrame, centred: bool = True) -> pl.DataFrame:
    """Join the hourly power to every product's wind on the site-hours all of them cover.

    Args:
        sites: The wind roster.
        centred: Passed to `wind_hourly_power`.

    Returns:
        One row per common site-hour with the power, the capacity, and every product's wind.
    """
    frame = wind_hourly_power(sites=sites, centred=centred).join(
        sites.select("site", "effective_capacity_mw"), on="site"
    )
    for product in PRODUCTS:
        speed, sine, cosine, surface = wind_columns(product=product)
        speed_100m, sine_100m, cosine_100m, _ = served_100m_columns(product=product)
        hub = hub_height_m(product=product)
        wind = pl.read_parquet(output_path_for(product=product)).select(
            "site",
            "time",
            pl.col(f"wind_speed_{hub}m").alias(speed),
            pl.col(f"wind_direction_{hub}m").radians().sin().alias(sine),
            pl.col(f"wind_direction_{hub}m").radians().cos().alias(cosine),
            pl.col("wind_speed_10m").alias(surface),
            pl.col("wind_speed_100m").alias(speed_100m),
            pl.col("wind_direction_100m").radians().sin().alias(sine_100m),
            pl.col("wind_direction_100m").radians().cos().alias(cosine_100m),
        )
        frame = frame.join(wind, on=["site", "time"], how="inner")
    speed_80m, sine_80m, cosine_80m, _ = UKV_80M_COLUMNS
    ukv_80m = pl.read_parquet(output_path_for(product="ukv")).select(
        "site",
        "time",
        pl.col("wind_speed_80m").alias(speed_80m),
        pl.col("wind_direction_80m").radians().sin().alias(sine_80m),
        pl.col("wind_direction_80m").radians().cos().alias(cosine_80m),
    )
    first, second = STEP_DATES
    step = (pl.col("time") >= first).cast(pl.Int8) + (pl.col("time") >= second).cast(pl.Int8)
    return (
        frame.join(ukv_80m, on=["site", "time"], how="inner")
        .with_columns(step_period=step)
        .sort("site", "time")
    )


def common_rows(*, frame: pl.DataFrame, drop_zero_hours: bool = True) -> pl.DataFrame:
    """Drop the rows no product should be scored on, by rules no product's values decide.

    Args:
        frame: The joined frame.
        drop_zero_hours: Whether to drop every hour holding an exactly-zero half-hour; False
            keeps them, for the `keep_zero_hours` setting.

    Returns:
        The frame without zero-half-hour hours and the post-upgrade tail of January 2026, with the
        `constrained` and `cap_mw` columns the fit loop reads. NGED has confirmed that the only
        generator under active network management in the trial area is solar, so no wind hour is
        constrained.
    """
    february = datetime(2026, 2, 1, tzinfo=UTC)
    zero_rule = ~pl.col("has_zero_half_hour") if drop_zero_hours else pl.lit(value=True)
    return frame.filter(
        zero_rule,
        ~pl.col("time").is_between(UPGRADE_DAY, february, closed="left"),
    ).with_columns(constrained=pl.lit(value=False), cap_mw=pl.lit(None, dtype=pl.Float64))


def jobs() -> list[Job]:
    """Return every product's wind arm at both settings and its served-100 m arm, and the checks.

    Returns:
        Three jobs per product, the UKV 80 m arm, the three step-period arms, and one job per
        product in each of `ROW_SET_SETTINGS`.
    """
    jobs: list[Job] = []
    for product in PRODUCTS:
        wind = (*SHARED_FEATURES, *wind_columns(product=product))
        jobs += [
            (f"{product}_wind", "pooled", "power_mw", wind, PRIMARY_HYPER_PARAMETERS, False),
            (
                f"{product}_wind",
                "sensitivity",
                "power_mw",
                wind,
                SENSITIVITY_HYPER_PARAMETERS,
                False,
            ),
            (
                f"{product}_100m",
                "pooled",
                "power_mw",
                (*SHARED_FEATURES, *served_100m_columns(product=product)),
                PRIMARY_HYPER_PARAMETERS,
                False,
            ),
        ]
    jobs.append(
        (
            "ukv_80m",
            "pooled",
            "power_mw",
            (*SHARED_FEATURES, *UKV_80M_COLUMNS),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
    )
    jobs += [
        (
            f"{product}_step",
            "pooled",
            "power_mw",
            (*SHARED_FEATURES, *wind_columns(product=product), "step_period"),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for product in STEP_ARM_PRODUCTS
    ]
    jobs += [
        (
            f"{product}_wind",
            setting,
            "power_mw",
            (*SHARED_FEATURES, *wind_columns(product=product)),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for setting in ROW_SET_SETTINGS
        for product in PRODUCTS
    ]
    return jobs
