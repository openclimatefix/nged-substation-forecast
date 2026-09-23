"""Score five weather products as descriptions of past wind, on one common row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>.

**Every product gets one arm with the same four wind columns** — its native hub-height speed, that
height's direction as sine and cosine, and its 10 m speed — plus the hour of day, the day of the
year, and the UKV era, fitted per generator by the tested out-of-fold loop from
`studies.cross_validation`. The hub height is 100 m for ERA5 and UKV and 80 m for the ICON products,
whose served 100 m value is their 120 m speed rescaled. So a contrast between two arms is a contrast
between the products' wind. A second arm per product, shown the served 100 m speed and direction and
the 10 m speed, and a second hyperparameter setting are sensitivity checks. The products are ERA5,
UKV, ICON-D2, ICON-EU, and ICON global, downloaded by `fetch_wind_point.py`.

**The power hour is centred on the label, unlike the solar study's.** Open-Meteo's wind is an
instantaneous value at the label, where its radiation is a mean over the hour ending there, so the
hour labelled T is built from the half-hours ending at T and at T + 30 min. An offset scan in the
plan review found every product scoring best with the hour centred this way.

**An hour holding an exactly-zero half-hour is dropped, whatever any product says.** From April 2026
the feed publishes no exact zeros at two of the generators: their calm half-hours are missing
instead, and `hourly_from_half_hourly` already drops an hour with a missing half-hour. Dropping the
zeros makes the earlier period match. Most dropped hours are calm, so behaviour near cut-in is
under-sampled; dropping no rows at all moves every contrast by 0.03 points or less.

**The folds are cut inside each era of the UKV record, and every arm is told the era**, as in
`weather_products.py`.

Run it with `uv run python studies/beam_diffuse_split/wind_products.py`, after
`fetch_wind_point.py`. With `--fit-missing` it keeps the losses a full run saved and fits only the
arms they lack.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Final

import polars as pl
from build_dataset import POWER_DELTA_URI, _wind_sites
from fetch_wind_point import PRODUCTS, output_path_for
from run_experiment import Job, _add_time_features, run_all
from sources import STUDY_DATA_DIR
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SENSITIVITY_HYPER_PARAMETERS
from studies.neighbouring_hours import with_neighbouring_hours
from studies.power import hourly_from_half_hourly
from weather_products import (
    CONTRAST_HEADER,
    UPGRADE_DAY,
    _contrast_line,
    _mae,
    _scope,
    geometry_lines,
    with_eras,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR_NAME: Final[str] = "beam_diffuse_wind_products"
"""The results directory under `STUDY_DATA_DIR`."""

SHARED_FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "day_of_year", "era_code")
"""Features every arm gets, on top of its product's wind."""

DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_eu_wind", "era5_wind"),
    ("ukv_wind", "era5_wind"),
    ("icon_eu_wind", "ukv_wind"),
    ("icon_d2_wind", "icon_eu_wind"),
)
"""The four contrasts the recommendations rest on, named before the run.

Whether a Great-Britain-wide weather model beats the reanalysis, twice; which of the two
Great-Britain-wide models is better; and whether the regional model adds anything. Every other
contrast in the report is exploratory.
"""

REPORTED_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *DECIDING_CONTRASTS,
    ("icon_d2_wind", "ukv_wind"),
)
"""The deciding contrasts, plus ICON-D2 against UKV, reported in every block."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_global_wind", "icon_eu_wind"),
    ("icon_d2_wind", "era5_wind"),
    ("icon_global_wind", "era5_wind"),
)
"""Contrasts reported for context, not relied on."""


def _hourly_power(*, sites: pl.DataFrame, centred: bool = True) -> pl.DataFrame:
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


def _wind_columns(*, product: str) -> tuple[str, str, str, str]:
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


def _served_100m_columns(*, product: str) -> tuple[str, str, str, str]:
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

STEP_SITE: Final[str] = "W3"
"""The generator at which ICON global's served wind steps, relative to ICON-EU's."""

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


def _hub_height_m(*, product: str) -> int:
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


def _offset_label(offset_hours: int) -> str:
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
        *(f"speed_hub_{_offset_label(offset)}_{product}" for offset in HUB_OFFSETS_HOURS),
        *(f"speed_10m_{_offset_label(offset)}_{product}" for offset in SURFACE_OFFSETS_HOURS),
    )


def with_wind_context(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add every product's hub-height and 10 m speeds in the hours around each row.

    The neighbours are read from each product's own download, not from the scored rows, which
    exclude hours by the target. Each download is checked to reproduce the frame's own hub-height
    and 10 m speed columns at offset zero first, so a neighbour cannot come from a series labelled
    differently.

    Args:
        frame: The common rows, carrying `speed_hub_<product>` and `speed_10m_<product>` for every
            product, as `_wind_columns` names them.

    Returns:
        `frame`, in its own row order, with `context_columns(product=...)` for every product.

    Raises:
        ValueError: If a download does not reproduce the frame's columns at offset zero.
    """
    for product in PRODUCTS:
        hub = f"wind_speed_{_hub_height_m(product=product)}m"
        offsets = [(hub, offset) for offset in HUB_OFFSETS_HOURS]
        offsets += [("wind_speed_10m", offset) for offset in SURFACE_OFFSETS_HOURS]
        own_hub, _, _, own_surface = _wind_columns(product=product)
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
        centred: Passed to `_hourly_power`.

    Returns:
        One row per common site-hour with the power, the capacity, and every product's wind.
    """
    frame = _hourly_power(sites=sites, centred=centred).join(
        sites.select("site", "effective_capacity_mw"), on="site"
    )
    for product in PRODUCTS:
        speed, sine, cosine, surface = _wind_columns(product=product)
        speed_100m, sine_100m, cosine_100m, _ = _served_100m_columns(product=product)
        hub = _hub_height_m(product=product)
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
        wind = (*SHARED_FEATURES, *_wind_columns(product=product))
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
                (*SHARED_FEATURES, *_served_100m_columns(product=product)),
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
            (*SHARED_FEATURES, *_wind_columns(product=product), "step_period"),
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
            (*SHARED_FEATURES, *_wind_columns(product=product)),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for setting in ROW_SET_SETTINGS
        for product in PRODUCTS
    ]
    return jobs


def _mean_speed_m_s(*, frame: pl.DataFrame, product: str) -> float:
    """Return one product's mean served 100 m wind speed in m/s, from Open-Meteo's km/h.

    Args:
        frame: The common rows.
        product: A key of `PRODUCTS`.

    Returns:
        The mean speed.
    """
    return float(frame.select(pl.col(f"speed_100m_{product}").mean()).item()) / 3.6


def _scoped(*, losses: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Restrict the losses to an era scope from `weather_products`, or to a half of the year.

    Args:
        losses: Per-row losses carrying `month` and `time`.
        scope: `winter` (October to March), `summer` (April to September), or a scope
            `weather_products._scope` accepts.

    Returns:
        The rows belonging to that scope.
    """
    month = pl.col("time").dt.month()
    if scope == "winter":
        return losses.filter((month >= 10) | (month <= 3))
    if scope == "summer":
        return losses.filter(month.is_between(4, 9))
    return _scope(losses=losses, scope=scope)


def _renamed(*, losses: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Rename one family of arms to the `_wind` names the contrasts use.

    Args:
        losses: Per-row losses for one setting.
        suffix: The arm suffix to keep, such as `_100m`.

    Returns:
        Those arms' losses, renamed to end in `_wind`.
    """
    return losses.filter(pl.col("arm").str.ends_with(suffix)).with_columns(
        arm=pl.col("arm").str.replace(f"{suffix}$", "_wind")
    )


def _check_arms(*, losses: pl.DataFrame, wind: pl.DataFrame, sites: list[str]) -> list[str]:
    """Report the post hoc checks: UKV at 80 m, lead-matched contrasts, and ICON global's steps.

    Args:
        losses: The pooled setting's losses, every arm.
        wind: The `_wind` arms.
        sites: The site labels.

    Returns:
        Markdown lines.
    """
    ukv_80m = pl.concat(
        [
            wind.filter(pl.col("arm") != "ukv_wind"),
            losses.filter(pl.col("arm") == "ukv_80m").with_columns(arm=pl.lit("ukv_wind")),
        ],
    )
    lines = [
        "",
        "#### Checks added after the first run (exploratory)",
        "",
        (
            f"MAE: ukv_80m {_mae(losses=losses, arm='ukv_80m'):.3f}, "
            f"era5_step {_mae(losses=losses, arm='era5_step'):.3f}, "
            f"icon_eu_step {_mae(losses=losses, arm='icon_eu_step'):.3f}, "
            f"icon_global_step {_mae(losses=losses, arm='icon_global_step'):.3f}."
        ),
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=ukv_80m, treatment=t, reference=r, label="UKV at 80 m")
        for t, r in (("icon_d2_wind", "ukv_wind"), ("icon_eu_wind", "ukv_wind"))
    ]
    three_hour = wind.with_columns(icon_lead=pl.col("time").dt.hour() % 3)
    lines += [
        _contrast_line(
            losses=three_hour.filter(pl.col("icon_lead") == lead),
            treatment=t,
            reference=r,
            label=f"ICON lead {lead} h",
        )
        for t, r in (("icon_d2_wind", "ukv_wind"), ("icon_eu_wind", "ukv_wind"))
        for lead in (0, 1, 2)
    ]
    lines += [
        _contrast_line(
            losses=_scoped(losses=three_hour, scope=era).filter(pl.col("icon_lead") == lead),
            treatment="icon_d2_wind",
            reference="ukv_wind",
            label=f"ICON lead {lead} h, {era}",
        )
        for era in ("pre", "post")
        for lead in (0, 1, 2)
    ]
    six_hour = wind.with_columns(global_lead=pl.col("time").dt.hour() % 6)
    lines += [
        _contrast_line(
            losses=six_hour.filter(condition),
            treatment="icon_global_wind",
            reference="icon_eu_wind",
            label=label,
        )
        for label, condition in (
            ("ICON global lead 0–2 h, equal to ICON-EU's", pl.col("global_lead") < 3),
            ("ICON global lead 3–5 h", pl.col("global_lead") >= 3),
        )
    ]
    lines += [
        _contrast_line(
            losses=wind.filter(pl.col("site") == site),
            treatment="icon_global_wind",
            reference="icon_eu_wind",
            label=f"site {site}",
        )
        for site in sites
    ]
    step = losses.filter(pl.col("arm").str.ends_with("_step"))
    lines += [
        _contrast_line(
            losses=scoped, treatment="icon_global_step", reference="icon_eu_step", label=label
        )
        for label, scoped in (
            ("told the step period, all sites", step),
            *(
                (f"told the step period, site {site}", step.filter(pl.col("site") == site))
                for site in sites
            ),
        )
    ]
    others = [site for site in sites if site != STEP_SITE]
    pooled_label = "sites " + " and ".join(others)
    lines += [
        _contrast_line(
            losses=wind.filter(pl.col("site").is_in(others)),
            treatment="icon_global_wind",
            reference="icon_eu_wind",
            label=pooled_label,
        ),
        _contrast_line(
            losses=step.filter(pl.col("site").is_in(others)),
            treatment="icon_global_step",
            reference="icon_eu_step",
            label=f"told the step period, {pooled_label}",
        ),
    ]
    lines += [
        _contrast_line(
            losses=scoped, treatment="icon_global_step", reference="era5_step", label=label
        )
        for label, scoped in (
            ("told the step period, all sites", step),
            *(
                (f"told the step period, site {site}", step.filter(pl.col("site") == site))
                for site in sites
            ),
        )
    ]
    return lines


def _step_ratio_lines() -> list[str]:
    """Report ICON global's mean wind speed over ICON-EU's at each site, either side of each step.

    Each ratio is of the whole period's means, read from the downloads `fetch_wind_point.py` wrote.

    Returns:
        Markdown lines.
    """
    heights = (10, 80)
    speeds = [f"wind_speed_{height}m" for height in heights]
    joined = (
        pl.read_parquet(output_path_for(product="icon_global"))
        .select("site", "time", *speeds)
        .join(
            pl.read_parquet(output_path_for(product="icon_eu")).select("site", "time", *speeds),
            on=["site", "time"],
            suffix="_eu",
        )
        .with_columns(period=sum(pl.col("time") >= date for date in STEP_DATES))
    )
    lines = [
        "#### ICON global's mean wind speed over ICON-EU's, by step period",
        "",
        "| Site | Height | Before 2 June 2025 | Between | From 2 June 2026 |",
        "|---|---|---|---|---|",
    ]
    for site in sorted(joined["site"].unique().to_list()):
        for height in heights:
            speed = f"wind_speed_{height}m"
            ratios = (
                joined.filter(pl.col("site") == site)
                .group_by("period")
                .agg(ratio=pl.col(speed).mean() / pl.col(f"{speed}_eu").mean())
                .sort("period")["ratio"]
                .to_list()
            )
            cells = " | ".join(f"{ratio:.3f}" for ratio in ratios)
            lines.append(f"| {site} | {height} m | {cells} |")
    return lines


def _row_set_lines(*, losses: pl.DataFrame, wind: pl.DataFrame) -> list[str]:
    """Report the post hoc row-set checks: the solar study's power hour, and keeping zero hours.

    Args:
        losses: Every arm's losses, every setting.
        wind: The main setting's `_wind` arms.

    Returns:
        Markdown lines.
    """
    by_setting = {
        setting: losses.filter(pl.col("setting") == setting) for setting in ROW_SET_SETTINGS
    }
    lines = [
        "#### The power hour and the zero rule (post hoc)",
        "",
        "| Product | Main | Hour ending at the label | Zero hours kept |",
        "|---|---|---|---|",
    ]
    for product in PRODUCTS:
        arm = f"{product}_wind"
        cells = " | ".join(f"{_mae(losses=scoped, arm=arm):.3f}" for scoped in by_setting.values())
        lines.append(f"| {product} | {_mae(losses=wind, arm=arm):.3f} | {cells} |")
    lines += ["", *CONTRAST_HEADER]
    for setting, scoped in by_setting.items():
        lines += [
            _contrast_line(losses=scoped, treatment=t, reference=r, label=setting)
            for t, r in REPORTED_CONTRASTS
        ]
    lines.append("")
    for setting, scoped in by_setting.items():
        largest = max(
            abs(
                _mae(losses=scoped, arm=t)
                - _mae(losses=scoped, arm=r)
                - (_mae(losses=wind, arm=t) - _mae(losses=wind, arm=r))
            )
            for t, r in REPORTED_CONTRASTS
        )
        lines.append(f"Largest change in a reported contrast's estimate, {setting}: {largest:.3f}.")
    return lines


def _report(*, frame: pl.DataFrame, losses: pl.DataFrame, sites_roster: pl.DataFrame) -> str:
    """Assemble the markdown report.

    Args:
        frame: The common rows.
        losses: Every arm's losses, at every setting.
        sites_roster: The wind roster, for the distances.

    Returns:
        The report.
    """
    sites = sorted(frame["site"].unique().to_list())
    pooled = losses.filter(pl.col("setting") == "pooled")
    wind = _renamed(losses=pooled, suffix="_wind")
    served_100m = _renamed(losses=pooled, suffix="_100m")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    lines = [
        (
            f"### Five weather products on {frame.height:,} common site-hours of wind "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        "| Product | Hub height shown | All sites | "
        + " | ".join(sites)
        + " | Served 100 m and 10 m | Second setting |",
        "|---" * (len(sites) + 5) + "|",
    ]
    for product in PRODUCTS:
        arm = f"{product}_wind"
        per_site = [
            f"{_mae(losses=wind.filter(pl.col('site') == site), arm=arm):.3f}" for site in sites
        ]
        lines.append(
            f"| {product} | {_hub_height_m(product=product)} m "
            f"| {_mae(losses=wind, arm=arm):.3f} | "
            + " | ".join(per_site)
            + f" | {_mae(losses=served_100m, arm=arm):.3f} "
            f"| {_mae(losses=sensitivity, arm=arm):.3f} |"
        )
    lines += ["", "Mean absolute error as a percentage of each site's P99 output.", ""]
    lines += ["#### Deciding contrasts, named before the run", "", *CONTRAST_HEADER]
    for treatment, reference in REPORTED_CONTRASTS:
        lines.append(
            _contrast_line(losses=wind, treatment=treatment, reference=reference, label="all")
        )
        lines += [
            _contrast_line(
                losses=wind.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in sites
        ]
    lines += [
        "",
        "The last block, ICON-D2 against UKV, is exploratory.",
        "",
        "#### By era and by half of the year (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=_scoped(losses=wind, scope=scope), treatment=t, reference=r, label=scope
        )
        for scope in ("pre", "pre_matched", "post", "winter", "summer")
        for t, r in (*REPORTED_CONTRASTS, ("icon_d2_wind", "era5_wind"))
    ]
    lines += [
        "",
        (
            "The post scope holds eight months, so its intervals rest on eight clusters and "
            "under-cover; read its fold-sign counts alongside them."
        ),
        "",
        "#### Sensitivity: the served 100 m arm the plan specified, and the second setting",
        "",
        *CONTRAST_HEADER,
    ]
    for label, scoped in (("served 100 m and 10 m", served_100m), ("second setting", sensitivity)):
        lines += [
            _contrast_line(losses=scoped, treatment=t, reference=r, label=label)
            for t, r in REPORTED_CONTRASTS
        ]
    lines += _check_arms(losses=pooled, wind=wind, sites=sites)
    lines += ["", "#### Other contrasts (exploratory)", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=wind, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
    ]
    lines += [
        _contrast_line(
            losses=wind.filter(pl.col("site") == site),
            treatment="icon_d2_wind",
            reference="era5_wind",
            label=f"site {site}",
        )
        for site in sites
    ]
    lines += [
        _contrast_line(
            losses=scoped, treatment="icon_global_wind", reference="era5_wind", label=label
        )
        for label, scoped in (
            *((scope, _scoped(losses=wind, scope=scope)) for scope in ("winter", "summer")),
            *((f"site {site}", wind.filter(pl.col("site") == site)) for site in sites),
        )
    ]
    lines += [
        "",
        "Mean served 100 m wind speed, all sites: "
        + ", ".join(
            f"{product} {_mean_speed_m_s(frame=frame, product=product):.2f} m/s"
            for product in PRODUCTS
        )
        + ".",
    ]
    lines += ["", *_row_set_lines(losses=losses, wind=wind)]
    lines += ["", *_step_ratio_lines()]
    lines += ["", *geometry_lines(sites=sites_roster, noun="wind farms")]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Fit every arm, bootstrap every contrast, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-missing",
        action="store_true",
        help="Keep the losses already on disk and fit only the jobs they lack.",
    )
    arguments = parser.parse_args()

    sites = _wind_sites()
    frame = with_eras(frame=_add_time_features(dataset=common_rows(frame=joined(sites=sites))))
    frames = {
        "hour_ending": with_eras(
            frame=_add_time_features(dataset=common_rows(frame=joined(sites=sites, centred=False)))
        ),
        "keep_zero_hours": with_eras(
            frame=_add_time_features(
                dataset=common_rows(frame=joined(sites=sites), drop_zero_hours=False)
            )
        ),
    }
    by_site = frame.group_by("site", "era").agg(pl.len(), pl.col("month").n_unique()).sort("site")
    _LOG.info("common rows: %d\n%s", frame.height, by_site)

    output_dir = STUDY_DATA_DIR / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "losses.parquet"
    all_jobs = jobs()
    saved = pl.read_parquet(path) if arguments.fit_missing else None
    done = set(saved.select("arm", "setting").unique().iter_rows()) if saved is not None else set()
    missing = [job for job in all_jobs if (job[0], job[1]) not in done]
    _LOG.info("fitting %d jobs of %d", len(missing), len(all_jobs))
    parts = [] if saved is None else [saved]
    for key, dataset in (("main", frame), *frames.items()):
        chosen = [job for job in missing if (job[1] if job[1] in frames else "main") == key]
        if chosen:
            parts.append(run_all(dataset=dataset, jobs=chosen))
    losses = pl.concat(parts)
    losses.write_parquet(path)

    report = _report(frame=frame, losses=losses, sites_roster=sites)
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
