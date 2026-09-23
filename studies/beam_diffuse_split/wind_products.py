"""Score five weather products as descriptions of past wind, on one common row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>.

**Every product gets one arm with the same four wind columns** — its native hub-height speed, that
height's direction as sine and cosine, and its 10 m speed — plus the hour of day, the day of the
year, and the UKV era, fitted per generator by the tested out-of-fold loop from
`studies.cross_validation`. The hub height is 100 m for ERA5 and UKV and 80 m for the ICON products,
whose served 100 m value is their 120 m speed rescaled. So a contrast between two arms is a contrast
between the products' wind. A second arm per product, shown the served 100 m speed and direction
alone, and a second hyperparameter setting are sensitivity checks. The products are ERA5, UKV,
ICON-D2, ICON-EU, and ICON global, downloaded by `fetch_wind_point.py`.

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
`fetch_wind_point.py`.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Final

import polars as pl
from build_dataset import POWER_DELTA_URI, _wind_sites
from fetch_wind_point import PRODUCTS, output_path_for
from run_experiment import Job, _add_time_features, _run_all
from sources import STUDY_DATA_DIR
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SENSITIVITY_HYPER_PARAMETERS
from studies.power import hourly_from_half_hourly
from weather_products import (
    CONTRAST_HEADER,
    UPGRADE_DAY,
    _contrast_line,
    _mae,
    _scope,
    _with_eras,
)

_LOG = logging.getLogger(__name__)

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

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_d2_wind", "ukv_wind"),
    ("icon_global_wind", "icon_eu_wind"),
    ("icon_d2_wind", "era5_wind"),
    ("icon_global_wind", "era5_wind"),
)
"""Contrasts reported for context, not relied on."""


def _hourly_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Average each wind generator's half-hours onto an hour centred on its label.

    Shifting every stamp back 30 minutes before `hourly_from_half_hourly` means the hour labelled T
    holds the half-hours ending at T and at T + 30 min, which span T - 30 min to T + 30 min.

    Args:
        sites: The wind roster.

    Returns:
        One row per (site, time) with `power_mw` and `has_zero_half_hour`.
    """
    half_hourly = (
        pl.scan_delta(POWER_DELTA_URI)
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .collect()
        .join(sites.select("time_series_id", "site"), on="time_series_id")
        .select("site", time=pl.col("time").dt.offset_by("-30m"), power_mw=pl.col("power"))
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


def _served_100m_columns(*, product: str) -> tuple[str, str, str]:
    """Return one product's served 100 m speed and direction feature names.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        The 100 m speed and the 100 m direction's sine and cosine.
    """
    return (f"speed_100m_{product}", f"sin_100m_{product}", f"cos_100m_{product}")


def _hub_height_m(*, product: str) -> int:
    """Return the height in metres a product's wind arm is shown: its native one nearest 100 m.

    Args:
        product: A key of `PRODUCTS`.

    Returns:
        80 for the ICON products, 100 otherwise.
    """
    return 80 if product.startswith("icon") else 100


def _joined(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Join the hourly power to every product's wind on the site-hours all of them cover.

    Args:
        sites: The wind roster.

    Returns:
        One row per common site-hour with the power, the capacity, and every product's wind.
    """
    frame = _hourly_power(sites=sites).join(
        sites.select("site", "effective_capacity_mw"), on="site"
    )
    for product in PRODUCTS:
        speed, sine, cosine, surface = _wind_columns(product=product)
        speed_100m, sine_100m, cosine_100m = _served_100m_columns(product=product)
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
    return frame.sort("site", "time")


def _common_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows no product should be scored on, by rules no product's values decide.

    Args:
        frame: The joined frame.

    Returns:
        The frame without zero-half-hour hours and the post-upgrade tail of January 2026, with the
        `constrained` and `cap_mw` columns the fit loop reads. NGED has confirmed that the only
        generator under active network management in the trial area is solar, so no wind hour is
        constrained.
    """
    february = datetime(2026, 2, 1, tzinfo=UTC)
    return frame.filter(
        ~pl.col("has_zero_half_hour"),
        ~pl.col("time").is_between(UPGRADE_DAY, february, closed="left"),
    ).with_columns(constrained=pl.lit(value=False), cap_mw=pl.lit(None, dtype=pl.Float64))


def _jobs() -> list[Job]:
    """Return every product's wind arm at both settings, and its served-100 m arm.

    Returns:
        Three jobs per product.
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
    return jobs


def _mean_speed_m_s(*, frame: pl.DataFrame, product: str) -> float:
    """Return one product's mean hub-height wind speed in m/s, from Open-Meteo's km/h.

    Args:
        frame: The common rows.
        product: A key of `PRODUCTS`.

    Returns:
        The mean speed.
    """
    return float(frame.select(pl.col(f"speed_hub_{product}").mean()).item()) / 3.6


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


def _report(*, frame: pl.DataFrame, losses: pl.DataFrame) -> str:
    """Assemble the markdown report.

    Args:
        frame: The common rows.
        losses: Every arm's losses, at both settings.

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
        + " | Served 100 m only | Second setting |",
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
    for treatment, reference in (*DECIDING_CONTRASTS, ("icon_d2_wind", "ukv_wind")):
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
        for t, r in (*DECIDING_CONTRASTS, ("icon_d2_wind", "ukv_wind"))
    ]
    lines += [
        "",
        (
            "The post scope holds eight months, so its intervals rest on eight clusters and "
            "under-cover; read its fold-sign counts alongside them."
        ),
        "",
        "#### Sensitivity: the served 100 m wind alone, and the second hyperparameter setting",
        "",
        *CONTRAST_HEADER,
    ]
    for label, scoped in (("served 100 m only", served_100m), ("second setting", sensitivity)):
        lines += [
            _contrast_line(losses=scoped, treatment=t, reference=r, label=label)
            for t, r in (*DECIDING_CONTRASTS, ("icon_d2_wind", "ukv_wind"))
        ]
    lines += ["", "#### Other contrasts (exploratory)", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=wind, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
        if (t, r) != ("icon_d2_wind", "ukv_wind")
    ]
    lines += [
        "",
        "Mean wind speed at the hub height shown, all sites: "
        + ", ".join(
            f"{product} {_mean_speed_m_s(frame=frame, product=product):.2f} m/s"
            for product in PRODUCTS
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Fit every arm, bootstrap every contrast, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()

    sites = _wind_sites()
    frame = _with_eras(frame=_add_time_features(dataset=_common_rows(frame=_joined(sites=sites))))
    by_site = frame.group_by("site", "era").agg(pl.len(), pl.col("month").n_unique()).sort("site")
    _LOG.info("common rows: %d\n%s", frame.height, by_site)

    output_dir = STUDY_DATA_DIR / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    losses = _run_all(dataset=frame, jobs=_jobs())
    losses.write_parquet(output_dir / "losses.parquet")

    report = _report(frame=frame, losses=losses)
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
