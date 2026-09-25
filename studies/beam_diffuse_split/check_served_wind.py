"""Check three properties of the wind Open-Meteo serves, which the wind write-up relies on.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/826>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/wind/>.

**Which grid cell ICON global is read from.** The study reads every product from its nearest land
cell. This script compares ICON global's nearest cell with its nearest land cell at each
generator, either side of the two step dates in `wind_products.STEP_DATES`.

**How the served 100 m wind is built for each ICON product.** Open-Meteo derives it from the 120 m
speed; the ratio of the two shows how.

**When each ICON product's 80 m wind starts in the archive.**

The script sends each generator's coordinates to Open-Meteo, read at run time from the private
roster, and writes only the anonymous labels. Run it with
`uv run python studies/beam_diffuse_split/check_served_wind.py`; it writes
`served_wind_checks.md` beside the wind study's report.
"""

import argparse
import logging
import sys
from typing import Final

import polars as pl
from build_dataset import _wind_sites
from fetch_open_meteo_point import fetch_point_frame
from sources import HISTORICAL_FORECAST_URL, STUDY_DATA_DIR
from wind_products import OUTPUT_DIR_NAME, STEP_DATES

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

WINDOW: Final[tuple[tuple[str, str], ...]] = (
    ("2024-08-12", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-09-10"),
)
"""The study's window, in the chunks one request can carry."""

ICON_MODELS: Final[tuple[str, ...]] = ("icon_d2", "icon_eu", "icon_global")
"""Open-Meteo's names for the three ICON products."""

HEIGHT_CHECK_DATES: Final[tuple[str, str]] = ("2026-06-01", "2026-06-30")
"""One month, enough to show whether the served 100 m wind is a fixed multiple of the 120 m."""

ARCHIVE_START_DATES: Final[tuple[str, str]] = ("2022-06-01", "2023-02-28")
"""A span bracketing the start of Open-Meteo's ICON 80 m wind."""


def _cell_lines(*, sites: pl.DataFrame) -> list[str]:
    """Compare ICON global's nearest cell with its nearest land cell, per step period.

    Args:
        sites: The wind roster.

    Returns:
        Markdown lines.
    """
    frames = {}
    for selection in ("nearest", "land"):
        frames[selection] = pl.concat(
            fetch_point_frame(
                sites=sites,
                variables=("wind_speed_10m",),
                models_parameter="icon_global",
                first_date=first,
                last_date=last,
                base_url=HISTORICAL_FORECAST_URL,
                cell_selection=selection,
            )
            for first, last in WINDOW
        ).rename({"wind_speed_10m": selection})
    joined = (
        frames["nearest"]
        .join(frames["land"], on=["site", "time"])
        .with_columns(period=sum(pl.col("time") >= date for date in STEP_DATES))
    )
    summary = (
        joined.group_by("site", "period")
        .agg(
            ratio=pl.col("nearest").mean() / pl.col("land").mean(),
            identical=(pl.col("nearest") == pl.col("land")).mean(),
        )
        .sort("site", "period")
    )
    lines = [
        "#### ICON global's nearest cell against its nearest land cell, 10 m wind",
        "",
        "| Site | Period | Mean speed, nearest over land | Share of hours identical |",
        "|---|---|---|---|",
    ]
    names = ("before 2 June 2025", "between", "from 2 June 2026")
    lines += [
        f"| {row['site']} | {names[row['period']]} | {row['ratio']:.3f} | {row['identical']:.3f} |"
        for row in summary.iter_rows(named=True)
    ]
    return lines


def _height_lines(*, sites: pl.DataFrame) -> list[str]:
    """Report each ICON product's served 100 m speed over its 120 m speed.

    Args:
        sites: The wind roster.

    Returns:
        Markdown lines.
    """
    first, last = HEIGHT_CHECK_DATES
    lines = [
        f"#### Served 100 m speed over 120 m speed, {first} to {last}, hours above 10 km/h",
        "",
        "| Product | 5th percentile | Median | 95th percentile |",
        "|---|---|---|---|",
    ]
    for model in ICON_MODELS:
        frame = fetch_point_frame(
            sites=sites,
            variables=("wind_speed_100m", "wind_speed_120m"),
            models_parameter=model,
            first_date=first,
            last_date=last,
            cell_selection="land",
        ).filter(pl.col("wind_speed_120m") > 10.0)
        ratio = frame["wind_speed_100m"] / frame["wind_speed_120m"]
        quantiles = " | ".join(f"{ratio.quantile(q):.3f}" for q in (0.05, 0.5, 0.95))
        lines.append(f"| {model} | {quantiles} |")
    return lines


def _archive_start_lines(*, sites: pl.DataFrame) -> list[str]:
    """Report the first hour each ICON product's 80 m wind is served at every generator.

    Args:
        sites: The wind roster.

    Returns:
        Markdown lines.
    """
    first, last = ARCHIVE_START_DATES
    lines = [
        f"#### First hour with 80 m wind at every generator, searching {first} to {last}",
        "",
        "| Product | First hour |",
        "|---|---|",
    ]
    for model in ICON_MODELS:
        frame = fetch_point_frame(
            sites=sites,
            variables=("wind_speed_80m",),
            models_parameter=model,
            first_date=first,
            last_date=last,
            cell_selection="land",
        )
        start = frame.group_by("site").agg(pl.col("time").min())["time"].max()
        lines.append(f"| {model} | {start:%Y-%m-%d %H:%M} |")
    return lines


def main() -> int:
    """Run the three checks and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    sites = _wind_sites()
    lines = [
        *_cell_lines(sites=sites),
        "",
        *_height_lines(sites=sites),
        "",
        *_archive_start_lines(sites=sites),
    ]
    path = STUDY_DATA_DIR / OUTPUT_DIR_NAME / "served_wind_checks.md"
    report = "\n".join(lines) + "\n"
    path.write_text(report)
    sys.stdout.write(report)
    _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
