"""Check the two readings the extra-lead arms rest on, from the data on disk, before any fit.

One-off throwaway script for the extra lead days of
<https://github.com/openclimatefix/nged-substation-forecast/issues/912>. It reads and never fits.
It writes two Markdown files under `<output-dir>/verification/`:

1. `gefs_radiation_window.md`: which window GEFS's radiation averages over at 6-hourly leads
   beyond 240 h. At leads to 240 h the documented windows are 3 hours at leads that are odd
   multiples of 3 and 6 hours at multiples of 6, so every valid hour of a 00 UTC run at a multiple
   of 6 hours is a 6-hour window. The check compares the mean radiation at each valid hour beyond
   240 h with the mean at the same valid hour at leads to 240 h (a 6-hour window), and with the mean
   3-hour window ending at that hour, which the 3-hourly leads give exactly. It stops with an error
   unless the 6-hour reading is closer at every valid hour where the two windows differ by at least
   15%. It also compares the mean at each of the first 6-hourly leads (246 to 270 h) with the mean
   24 hours earlier at the same valid hour, and fails if any daylight lead differs by more than 10%.
2. `day0_matches_past_series.md`: whether Open-Meteo's unsuffixed Previous Runs series, which the
   extra-lead build reads as day 0 for ICON-D2 and ICON-EU, equals the series the past-weather
   studies read (`beam_diffuse_<product>.parquet`, `wind_<product>.parquet`) on the hours both hold.

The GEFS check pools every grid cell of the crop, every member and every 00 UTC run of the month
cache, so it prints no site and no coordinate.

Run it with `uv run python studies/nwp_forecast_comparison/verify_extra_leads.py --output-dir DIR`.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Final

import polars as pl
from contracts.settings import PROJECT_ROOT

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

GEFS_DIR_NAME: Final[str] = "GEFS_window_2024-11-01_None"
"""Under `data/studies/weather/`, the finished GEFS download."""

FINE_LAST_LEAD_HOURS: Final[int] = 240
"""GEFS steps every 3 hours to this lead and every 6 hours beyond it."""

VALID_HOURS: Final[tuple[int, ...]] = (0, 6, 12, 18)
"""The valid hours (UTC) of a 00 UTC run's 6-hourly leads."""

MIN_WINDOW_DIFFERENCE: Final[float] = 0.15
"""A valid hour tells the 3-hour window from the 6-hour window only if the two windows' mean
radiation differs by at least this share of the 6-hour window's."""

MIN_MEAN_W_M2: Final[float] = 20.0
"""Below this mean radiation a valid hour is night-time and carries no evidence either way."""

MAX_RATIO_ERROR: Final[float] = 0.10
"""How far beyond-240-h mean radiation may sit from the 6-hour reading's, as a share of it."""

PRODUCTS: Final[dict[str, str]] = {"ICON-D2": "icon-d2", "ICON-EU": "icon-eu"}
"""Each product's directory under `data/studies/weather/`, and the slug in its file names."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    Duplicated from `verify_previous_runs_leads._repo_data_dir`, because study scripts cannot
    import one another's private helpers.

    Returns:
        The directory holding `studies/` and the rest of the shared downloads.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def gefs_window_table(*, files: list[Path]) -> pl.DataFrame:
    """Return the mean radiation at each valid hour under each candidate window.

    Args:
        files: The GEFS month-cache parquets.

    Returns:
        One row per valid hour of `VALID_HOURS`, with `six_hour_fine` (mean radiation of the 6-hour
        windows ending at that hour at leads 24 to 240 h), `three_hour_fine` (the mean of the
        3-hour window ending at that hour, from the 3-hourly leads: twice the 6-hour window's mean
        less the mean of the 3-hour window ending three hours earlier), `beyond` (the mean at
        6-hourly leads beyond 240 h), and the counts behind each mean.
    """
    lead_hours = (pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32)
    radiation = "downward_short_wave_radiation_flux_surface"
    frame = (
        pl.scan_parquet(files)
        .filter(pl.col("init_time").dt.hour() == 0)
        .select(lead_hours=lead_hours, value=pl.col(radiation))
        .filter(pl.col("value").is_not_nan(), pl.col("lead_hours") > 0)
        .with_columns(valid_hour=pl.col("lead_hours") % 24)
    )
    fine = pl.col("lead_hours").is_between(24, FINE_LAST_LEAD_HOURS)
    means = (
        frame.group_by("valid_hour", (pl.col("lead_hours") > FINE_LAST_LEAD_HOURS).alias("beyond"))
        .agg(mean=pl.col("value").mean(), n=pl.len())
        .collect()
    )
    fine_means = (
        frame.filter(fine).group_by("valid_hour").agg(mean=pl.col("value").mean()).collect()
    )
    fine_by_hour = dict(zip(fine_means["valid_hour"], fine_means["mean"], strict=True))
    beyond = means.filter(pl.col("beyond"))
    beyond_by_hour = dict(zip(beyond["valid_hour"], beyond["mean"], strict=True))
    counts = dict(zip(beyond["valid_hour"], beyond["n"], strict=True))
    rows = []
    for hour in VALID_HOURS:
        six = fine_by_hour[hour]
        first_half = fine_by_hour[(hour - 3) % 24]
        rows.append(
            {
                "valid_hour": hour,
                "six_hour_fine": six,
                "three_hour_fine": 2.0 * six - first_half,
                "beyond": beyond_by_hour[hour],
                "n_beyond": counts[hour],
            }
        )
    return pl.DataFrame(rows)


BOUNDARY_LEADS: Final[tuple[int, ...]] = (246, 252, 258, 264, 270)
"""The first 6-hourly leads, which day 10's band reads; each is compared with the lead 24 hours
earlier, which has the same valid hour and a documented 6-hour window."""


def gefs_boundary_table(*, files: list[Path]) -> pl.DataFrame:
    """Return the mean radiation at each of the first 6-hourly leads and 24 hours before it.

    Args:
        files: The GEFS month-cache parquets.

    Returns:
        One row per lead of `BOUNDARY_LEADS`, with `lead_hours`, `mean` and `mean_day_before`
        (the mean 24 hours earlier, a 6-hour window).
    """
    lead_hours = (pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32)
    wanted = [*BOUNDARY_LEADS, *(lead - 24 for lead in BOUNDARY_LEADS)]
    means = (
        pl.scan_parquet(files)
        .filter(pl.col("init_time").dt.hour() == 0)
        .select(lead_hours=lead_hours, value=pl.col("downward_short_wave_radiation_flux_surface"))
        .filter(pl.col("value").is_not_nan(), pl.col("lead_hours").is_in(wanted))
        .group_by("lead_hours")
        .agg(mean=pl.col("value").mean())
        .collect()
    )
    by_lead = dict(zip(means["lead_hours"], means["mean"], strict=True))
    return pl.DataFrame(
        [
            {"lead_hours": lead, "mean": by_lead[lead], "mean_day_before": by_lead[lead - 24]}
            for lead in BOUNDARY_LEADS
        ]
    )


def gefs_boundary_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the boundary leads whose mean is not that of the same valid hour a day earlier.

    Args:
        table: `gefs_boundary_table`'s result.

    Returns:
        One line per failing lead; night-time leads (a mean under `MIN_MEAN_W_M2` a day earlier)
        are skipped. A lead fails if its mean is more than `MAX_RATIO_ERROR` from the mean a day
        earlier.
    """
    failures = []
    for row in table.iter_rows(named=True):
        previous = row["mean_day_before"]
        if previous >= MIN_MEAN_W_M2 and abs(row["mean"] - previous) > MAX_RATIO_ERROR * previous:
            failures.append(
                f"lead {row['lead_hours']} h: mean {row['mean']:.1f} W/m2 is more than "
                f"{MAX_RATIO_ERROR:.0%} from {previous:.1f} a day earlier"
            )
    return failures


def gefs_window_verdict(*, table: pl.DataFrame) -> list[str]:
    """List the valid hours at which the beyond-240-h radiation is not the 6-hour window.

    Args:
        table: `gefs_window_table`'s result.

    Returns:
        One line per failing valid hour; empty when the 6-hour reading holds. A valid hour fails if
        its beyond-240-h mean is more than `MAX_RATIO_ERROR` from the 6-hour mean, or, where the two
        windows differ by at least `MIN_WINDOW_DIFFERENCE`, closer to the 3-hour mean. Night-time
        hours (a 6-hour mean under `MIN_MEAN_W_M2`) are skipped.
    """
    failures: list[str] = []
    for row in table.iter_rows(named=True):
        six, three, beyond = row["six_hour_fine"], row["three_hour_fine"], row["beyond"]
        if six < MIN_MEAN_W_M2:
            continue
        if abs(beyond - six) > MAX_RATIO_ERROR * six:
            failures.append(
                f"valid hour {row['valid_hour']:02d} UTC: beyond-240-h mean {beyond:.1f} W/m2 is "
                f"more than {MAX_RATIO_ERROR:.0%} from the 6-hour mean {six:.1f}"
            )
        elif abs(three - six) >= MIN_WINDOW_DIFFERENCE * six and abs(beyond - three) < abs(
            beyond - six
        ):
            failures.append(
                f"valid hour {row['valid_hour']:02d} UTC: beyond-240-h mean {beyond:.1f} W/m2 is "
                f"closer to the 3-hour mean {three:.1f} than to the 6-hour mean {six:.1f}"
            )
    return failures


DAY0_COLUMNS: Final[tuple[tuple[str, str, str], ...]] = (
    ("shortwave_radiation", "beam_diffuse_{slug}.parquet", "ghi_w_m2"),
    ("wind_speed_100m", "wind_{underscored}.parquet", "wind_speed_100m"),
    ("wind_speed_10m", "wind_{underscored}.parquet", "wind_speed_10m"),
    ("wind_direction_100m", "wind_{underscored}.parquet", "wind_direction_100m"),
)
"""Each unsuffixed Previous Runs column the build reads as day 0, the past studies' file that holds
the same variable (with `{slug}` and `{underscored}` filled in), and its column there. Both series
are in the same units (radiation in W/m2, speed in km/h, direction in degrees). The past studies
hold no 2 m temperature for these products, so that column is not checked."""

DAY0_TOLERANCE: Final[float] = 0.01
"""How close the two series must be on an hour to count as equal, in each column's own unit."""


def day0_comparison(
    *, weather_dir: Path, product: str, slug: str
) -> list[dict[str, str | float | int]]:
    """Compare a product's unsuffixed Previous Runs columns with the past studies' series.

    Args:
        weather_dir: `data/studies/weather/`.
        product: The product's directory name.
        slug: The product's slug in its file names.

    Returns:
        One record per column of `DAY0_COLUMNS`, with `column`, the number of shared non-null
        (site, time) hours, the largest absolute difference, and the share of those hours within
        `DAY0_TOLERANCE`.
    """
    combined = pl.read_parquet(weather_dir / product / "previous_runs" / "combined.parquet")
    records: list[dict[str, str | float | int]] = []
    for column, file_pattern, past_column in DAY0_COLUMNS:
        past = pl.read_parquet(
            weather_dir
            / product
            / file_pattern.format(slug=slug, underscored=slug.replace("-", "_"))
        ).select("site", "time", past=past_column)
        shared = (
            combined.select("site", "time", column)
            .join(past, on=["site", "time"], how="inner")
            .drop_nulls([column, "past"])
        )
        difference = (shared[column] - shared["past"]).abs()
        records.append(
            {
                "column": column,
                "hours": shared.height,
                "max_abs_difference": float(difference.max()),  # ty: ignore[invalid-argument-type]
                "share_within_tolerance": float((difference <= DAY0_TOLERANCE).mean()),  # ty: ignore[invalid-argument-type]
            }
        )
    return records


def main() -> int:
    """Run both checks and write their Markdown files; exit non-zero if the GEFS check fails."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    weather_dir = _repo_data_dir() / "studies" / "weather"
    verification = args.output_dir / "verification"
    verification.mkdir(parents=True, exist_ok=True)

    cache = weather_dir / GEFS_DIR_NAME / "_month_cache"
    cache_files = sorted(cache.glob("*.parquet"))
    table = gefs_window_table(files=cache_files)
    boundary = gefs_boundary_table(files=cache_files)
    failures = [
        *gefs_window_verdict(table=table),
        *gefs_boundary_verdict(table=boundary),
    ]
    lines = [
        "# GEFS radiation window beyond 240 h",
        "",
        (
            "Mean radiation (W/m2) of 00 UTC runs, pooled over every grid cell of the crop, every "
            "member and every cached month, by valid hour."
        ),
        "",
        (
            "| Valid hour (UTC) | 6-hour window, leads 24 to 240 h "
            "| 3-hour window, leads 24 to 240 h | Leads beyond 240 h | Rows beyond 240 h |"
        ),
        "|---|---|---|---|---|",
        *(
            f"| {row['valid_hour']:02d} | {row['six_hour_fine']:.1f} | {row['three_hour_fine']:.1f}"
            f" | {row['beyond']:.1f} | {row['n_beyond']} |"
            for row in table.iter_rows(named=True)
        ),
        "",
        (
            "Mean radiation (W/m2) at the first 6-hourly leads, and 24 hours earlier at the same "
            "valid hour:"
        ),
        "",
        "| Lead (h) | Mean | Mean 24 h earlier |",
        "|---|---|---|",
        *(
            f"| {row['lead_hours']} | {row['mean']:.1f} | {row['mean_day_before']:.1f} |"
            for row in boundary.iter_rows(named=True)
        ),
        "",
        "**Verdict:** "
        + (
            "the values beyond 240 h are 6-hour window means."
            if not failures
            else "NOT 6-hour window means: " + "; ".join(failures)
        ),
    ]
    (verification / "gefs_radiation_window.md").write_text("\n".join(lines) + "\n")

    day0 = [
        "# Day 0 of Open-Meteo's Previous Runs against the past studies' series",
        "",
        (
            "| Product | Column | Shared hours | Largest absolute difference "
            f"| Share within {DAY0_TOLERANCE} |"
        ),
        "|---|---|---|---|---|",
    ]
    for product, slug in PRODUCTS.items():
        day0.extend(
            f"| {product} | {record['column']} | {record['hours']} "
            f"| {record['max_abs_difference']:.4f} | {record['share_within_tolerance']:.4f} |"
            for record in day0_comparison(weather_dir=weather_dir, product=product, slug=slug)
        )
    (verification / "day0_matches_past_series.md").write_text("\n".join(day0) + "\n")
    _LOG.info("wrote %s", verification)
    if failures:
        _LOG.error("GEFS radiation window check failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
