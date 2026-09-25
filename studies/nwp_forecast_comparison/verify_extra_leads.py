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
   15%.
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

import numpy as np
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

DAY0_TOLERANCE_W_M2: Final[float] = 0.5
"""How close the two radiation series must be on an hour to count as equal."""

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


def day0_comparison(*, weather_dir: Path, product: str, slug: str) -> dict[str, float | int]:
    """Compare a product's unsuffixed Previous Runs series with the past studies' series.

    Args:
        weather_dir: `data/studies/weather/`.
        product: The product's directory name.
        slug: The product's slug in its file names.

    Returns:
        The number of shared (site, time) hours, the largest absolute radiation difference, the
        share of hours within `DAY0_TOLERANCE_W_M2`, and the largest absolute wind-speed ratio
        deviation from the median ratio (Previous Runs serves km/h, the past-wind file m/s).
    """
    combined = pl.read_parquet(
        weather_dir / product / "previous_runs" / "combined.parquet",
        columns=["site", "time", "shortwave_radiation", "wind_speed_100m"],
    )
    past_radiation = pl.read_parquet(weather_dir / product / f"beam_diffuse_{slug}.parquet").select(
        "site", "time", "ghi_w_m2"
    )
    past_wind = pl.read_parquet(weather_dir / product / f"wind_{slug.replace('-', '_')}.parquet")
    radiation = combined.join(past_radiation, on=["site", "time"], how="inner").drop_nulls(
        ["shortwave_radiation", "ghi_w_m2"]
    )
    difference = (radiation["shortwave_radiation"] - radiation["ghi_w_m2"]).abs()
    wind = (
        combined.join(
            past_wind.select("site", "time", "wind_speed_100m"),
            on=["site", "time"],
            how="inner",
            suffix="_past",
        )
        .drop_nulls(["wind_speed_100m", "wind_speed_100m_past"])
        .filter(pl.col("wind_speed_100m_past") > 1.0)
        .with_columns(ratio=pl.col("wind_speed_100m") / pl.col("wind_speed_100m_past"))
    )
    median_ratio = float(wind["ratio"].median())  # ty: ignore[invalid-argument-type]
    return {
        "hours": radiation.height,
        "max_abs_radiation_difference": float(difference.max()),  # ty: ignore[invalid-argument-type]
        "share_within_tolerance": float((difference <= DAY0_TOLERANCE_W_M2).mean()),  # ty: ignore[invalid-argument-type]
        "median_wind_ratio": median_ratio,
        "max_wind_ratio_deviation": float(np.abs(wind["ratio"].to_numpy() - median_ratio).max()),
    }


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
    table = gefs_window_table(files=sorted(cache.glob("*.parquet")))
    failures = gefs_window_verdict(table=table)
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
            "| Product | Shared hours | Largest radiation difference (W/m2) "
            f"| Share within {DAY0_TOLERANCE_W_M2} W/m2 | Median wind ratio (km/h over m/s) "
            "| Largest deviation from that ratio |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for product, slug in PRODUCTS.items():
        result = day0_comparison(weather_dir=weather_dir, product=product, slug=slug)
        day0.append(
            f"| {product} | {result['hours']} | {result['max_abs_radiation_difference']:.3f} | "
            f"{result['share_within_tolerance']:.4f} | {result['median_wind_ratio']:.3f} | "
            f"{result['max_wind_ratio_deviation']:.3f} |"
        )
    (verification / "day0_matches_past_series.md").write_text("\n".join(day0) + "\n")
    _LOG.info("wrote %s", verification)
    if failures:
        _LOG.error("GEFS radiation window check failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
