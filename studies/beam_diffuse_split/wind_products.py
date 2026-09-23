"""Score five weather products as descriptions of past wind, on one common row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>.

**Every product gets one arm with the same four wind columns** — 100 m speed, 100 m direction as
sine and cosine, and 10 m speed — plus the hour of day, the day of the year, and the UKV era, fitted
per generator by the tested out-of-fold loop from `studies.cross_validation`. So a contrast between
two arms is a contrast between the products' wind. The products are ERA5, UKV, ICON-D2, ICON-EU,
and ICON global, downloaded by `fetch_wind_point.py`.

**The power hour is centred on the label, unlike the solar study's.** Open-Meteo's wind is an
instantaneous value at the label, where its radiation is a mean over the hour ending there, so the
hour labelled T is built from the half-hours ending at T and at T + 30 min. An offset scan in the
plan review found every product scoring best with the hour centred this way.

**An hour holding an exactly-zero half-hour is dropped, whatever any product says.** At one
generator the zeros are disconnections rather than calm: half of them fall where ERA5 reads above
6 m/s. The rule reads the power column alone, and it also removes some genuine calm hours, so
behaviour near cut-in is under-sampled.

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
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS
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
        The 100 m speed, the 100 m direction's sine and cosine, and the 10 m speed.
    """
    return (
        f"speed_100m_{product}",
        f"direction_sin_{product}",
        f"direction_cos_{product}",
        f"speed_10m_{product}",
    )


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
        wind = pl.read_parquet(output_path_for(product=product)).select(
            "site",
            "time",
            pl.col("wind_speed_100m").alias(speed),
            pl.col("wind_direction_100m").radians().sin().alias(sine),
            pl.col("wind_direction_100m").radians().cos().alias(cosine),
            pl.col("wind_speed_10m").alias(surface),
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
    """Return one arm per product, every arm shown the same number of columns.

    Returns:
        One job per product.
    """
    return [
        (
            f"{product}_wind",
            "pooled",
            "power_mw",
            (*SHARED_FEATURES, *_wind_columns(product=product)),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for product in PRODUCTS
    ]


def _mean_speed_m_s(*, frame: pl.DataFrame, product: str) -> float:
    """Return one product's mean 100 m wind speed in m/s, from Open-Meteo's km/h.

    Args:
        frame: The common rows.
        product: A key of `PRODUCTS`.

    Returns:
        The mean speed.
    """
    return float(frame.select(pl.col(f"speed_100m_{product}").mean()).item()) / 3.6


def _report(*, frame: pl.DataFrame, losses: pl.DataFrame) -> str:
    """Assemble the markdown report.

    Args:
        frame: The common rows.
        losses: The pooled run's losses.

    Returns:
        The report.
    """
    sites = sorted(frame["site"].unique().to_list())
    lines = [
        (
            f"### Five weather products on {frame.height:,} common site-hours of wind "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        "| Product | All sites | " + " | ".join(sites) + " |",
        "|---" * (len(sites) + 2) + "|",
    ]
    for product in PRODUCTS:
        arm = f"{product}_wind"
        per_site = [
            f"{_mae(losses=losses.filter(pl.col('site') == site), arm=arm):.3f}" for site in sites
        ]
        lines.append(
            f"| {product} | {_mae(losses=losses, arm=arm):.3f} | " + " | ".join(per_site) + " |"
        )
    lines += ["", "Mean absolute error as a percentage of each site's P99 output.", ""]
    lines += ["#### Deciding contrasts, named before the run", "", *CONTRAST_HEADER]
    for treatment, reference in DECIDING_CONTRASTS:
        lines.append(
            _contrast_line(losses=losses, treatment=treatment, reference=reference, label="all")
        )
        lines += [
            _contrast_line(
                losses=losses.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in sites
        ]
    lines += ["", "#### Exploratory contrasts", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=_scope(losses=losses, scope=scope), treatment=t, reference=r, label=scope
        )
        for scope in ("all", "pre", "post")
        for t, r in (*DECIDING_CONTRASTS, *EXPLORATORY_CONTRASTS)
        if scope != "all" or (t, r) in EXPLORATORY_CONTRASTS
    ]
    lines += [
        "",
        (
            "The post scope holds eight months, so its intervals rest on eight clusters and "
            "under-cover; read its fold-sign counts alongside them."
        ),
        "",
        "Mean wind speed at 100 m, all sites: "
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
