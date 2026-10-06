"""Print the evidence that Open-Meteo's 10 m wind speed is built differently in two spans.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1051>. The first fit of the study
found that Open-Meteo's 10 m speed, against CEDA's at CEDA's lead 0, reads about 0.97 in most months
and higher in 2024-11, 2025-01, and 2025-02. This script prints the evidence behind the build's
`OPEN_METEO_WIND_STEP_DAYS` into `wind_step_report.md`, and writes the daily series that the
page's figure draws.

**What it checks.** The median ratio of Open-Meteo's 10 m speed to CEDA's at lead 0, by month and
inside and outside the spans; the same ratios against ERA5's 10 m speed at the three wind farms,
which shows whether CEDA or Open-Meteo moved; Open-Meteo's 100 m to 10 m speed ratio, which falls in
the spans; the ratios of Open-Meteo's temperature, direction, and irradiance, which do not step; the
six-hour blocks around each edge; and that `combined.parquet` equals the `site_points/` extract
that the other studies read, hour for hour, inside the spans.

**The spans are decided from the served series and not from any power result.** The edges are whole
UTC days that contain each change, so a few hours at each edge are dropped that the series did not
change, and a few hours inside the span at the edge may be less affected than the rest.

Nothing here prints a generator's name, identifier or coordinates. Run it with
`uv run python studies/past_weather/ukv_ceda_vs_openmeteo_wind_steps.py`. A fresh run stops
(`refuse_to_overwrite`) while an output exists.
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.guards import refuse_to_overwrite
from studies.sources import UKV_VS_ERA5_DIR, site_points_dir_for
from ukv_ceda_vs_openmeteo_build import (
    MODEL_FREE_NAME,
    OPEN_METEO_PATH,
    OPEN_METEO_WIND_STEP_DAYS,
    OUTPUT_DIR,
    in_wind_step_days,
    lead_zero,
)

REPORT_NAME: Final[str] = "wind_step_report.md"
DAILY_NAME: Final[str] = "wind_step_daily.parquet"

MIN_SPEED_M_S: Final[float] = 1.0
"""Hours at or below this CEDA speed are left out of the ratios, where a ratio is unstable."""

MIN_SPEED_100M_M_S: Final[float] = 3.0
"""Hours at or below this Open-Meteo 10 m speed (km/h converted) are left out of the 100 m ratio."""

WINDOW_FIRST: Final[str] = "2024-10-01"
WINDOW_LAST: Final[str] = "2025-03-31"
"""The days the daily series covers, around the two spans."""


def month_ratios(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the median Open-Meteo to CEDA 10 m speed ratio at lead 0, by month.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One row per month with `ratio`, the two mean speeds, and `n`, pooled over the nine sites.
    """
    return (
        lead_zero(frame=frame)
        .filter(pl.col("ceda_speed_10m_m_s") > MIN_SPEED_M_S)
        .group_by("month")
        .agg(
            ratio=(pl.col("om_speed_10m_m_s") / pl.col("ceda_speed_10m_m_s")).median(),
            ceda_mean=pl.col("ceda_speed_10m_m_s").mean(),
            om_mean=pl.col("om_speed_10m_m_s").mean(),
            n=pl.len(),
        )
        .sort("month")
    )


def span_ratios(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Compare inside and outside the spans: speed, temperature, direction, and irradiance.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        Two rows (`inside` false and true) with the median speed ratio, the mean temperature offset,
        the median direction difference at speeds above 3 m/s, the median irradiance ratio with the
        sun above 10 degrees, and `n`.
    """
    return (
        lead_zero(frame=frame)
        .group_by("om_wind_step")
        .agg(
            speed_ratio=(pl.col("om_speed_10m_m_s") / pl.col("ceda_speed_10m_m_s"))
            .filter(pl.col("ceda_speed_10m_m_s") > MIN_SPEED_M_S)
            .median(),
            temperature_offset_k=(pl.col("om_temp_c") - pl.col("ceda_temp_c")).mean(),
            direction_difference_deg=(
                (pl.col("om_direction_10m_deg") - pl.col("ceda_direction_10m_deg") + 180) % 360
                - 180
            )
            .filter(pl.col("ceda_speed_10m_m_s") > 3.0)
            .median(),
            irradiance_ratio=(pl.col("ceda_ghi") / pl.col("om_ghi"))
            .filter((pl.col("om_ghi") > 100.0) & (pl.col("sun_elevation_deg") > 10.0))
            .median(),
            n=pl.len(),
        )
        .sort("om_wind_step")
    )


def era5_ratios(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return both archives' median 10 m speed against ERA5's at the wind farms, by month.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One row per month with `ceda_over_era5` and `om_over_era5`, at lead 0 and CEDA speeds above
        1 m/s. A steady CEDA ratio beside a stepping Open-Meteo ratio puts the step in Open-Meteo.
    """
    era5 = pl.read_parquet(UKV_VS_ERA5_DIR / "wind_rows.parquet").select(
        "site", "time", "era5_speed_10m"
    )
    joined = lead_zero(frame=frame).join(era5, on=["site", "time"])
    return (
        joined.filter(pl.col("ceda_speed_10m_m_s") > MIN_SPEED_M_S)
        .group_by("month")
        .agg(
            ceda_over_era5=(pl.col("ceda_speed_10m_m_s") / pl.col("era5_speed_10m")).median(),
            om_over_era5=(pl.col("om_speed_10m_m_s") / pl.col("era5_speed_10m")).median(),
        )
        .sort("month")
    )


def height_ratios(*, path: Path = OPEN_METEO_PATH) -> pl.DataFrame:
    """Return Open-Meteo's median 100 m to 10 m speed ratio inside and outside the spans.

    Args:
        path: Open-Meteo's `combined.parquet`.

    Returns:
        Two rows (`om_wind_step` false and true) with `ratio_100m_to_10m` and `n`, at the three
        wind farms and 10 m speeds above 3 km/h.
    """
    return (
        pl.read_parquet(path, columns=["site", "time", "wind_speed_10m", "wind_speed_100m"])
        .filter(pl.col("site").str.starts_with("W") & (pl.col("wind_speed_10m") > 3.0))
        .filter(
            (pl.col("time") >= pl.lit(WINDOW_FIRST).str.to_datetime(time_zone="UTC"))
            & (pl.col("time") < pl.lit("2025-04-01").str.to_datetime(time_zone="UTC"))
        )
        .with_columns(om_wind_step=in_wind_step_days())
        .group_by("om_wind_step")
        .agg(
            ratio_100m_to_10m=(pl.col("wind_speed_100m") / pl.col("wind_speed_10m")).median(),
            n=pl.len(),
        )
        .sort("om_wind_step")
    )


def equal_to_site_points() -> tuple[int, int]:
    """Count the wind rows of the step spans that `combined.parquet` and `site_points/` share.

    Returns:
        The rows the two files share inside the spans, and how many of them hold different values
        for the 10 m speed.
    """
    combined = pl.read_parquet(OPEN_METEO_PATH, columns=["site", "time", "wind_speed_10m"]).filter(
        pl.col("site").str.starts_with("W")
    )
    points = pl.read_parquet(site_points_dir_for(product="UKV") / "wind_ukv.parquet").select(
        "site", "time", points_speed="wind_speed_10m"
    )
    joined = combined.join(points, on=["site", "time"]).filter(in_wind_step_days())
    different = joined.filter(pl.col("wind_speed_10m") != pl.col("points_speed")).height
    return joined.height, different


def daily_ratio(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the daily median ratio of Open-Meteo's to CEDA's 10 m speed, around the spans.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        One row per UTC day from `WINDOW_FIRST` to `WINDOW_LAST` with `ratio`, `n`, and whether the
        day lies in a span. Pooled over the nine sites, so no series belongs to one generator.
    """
    return (
        lead_zero(frame=frame)
        .filter(pl.col("ceda_speed_10m_m_s") > MIN_SPEED_M_S)
        .with_columns(day=pl.col("time").dt.date())
        .filter(
            (pl.col("day") >= pl.lit(WINDOW_FIRST).str.to_date())
            & (pl.col("day") <= pl.lit(WINDOW_LAST).str.to_date())
        )
        .group_by("day")
        .agg(
            ratio=(pl.col("om_speed_10m_m_s") / pl.col("ceda_speed_10m_m_s")).median(),
            n=pl.len(),
        )
        .sort("day")
        .with_columns(in_span=in_wind_step_days(day=pl.col("day")))
    )


def edge_blocks(*, frame: pl.DataFrame) -> list[str]:
    """Print the six-hour blocks of the speed ratio one day either side of each edge.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        Markdown lines, one per edge.
    """
    lines: list[str] = []
    for first, last in OPEN_METEO_WIND_STEP_DAYS:
        for label, day in (("start", first), ("end", last)):
            low = pl.lit(day - timedelta(days=1))
            high = pl.lit(day + timedelta(days=1))
            blocks = (
                lead_zero(frame=frame)
                .with_columns(date=pl.col("time").dt.date())
                .filter((pl.col("date") >= low) & (pl.col("date") <= high))
                .filter(pl.col("ceda_speed_10m_m_s") > MIN_SPEED_M_S)
                .group_by(pl.col("time").dt.truncate("6h").alias("block"))
                .agg(ratio=(pl.col("om_speed_10m_m_s") / pl.col("ceda_speed_10m_m_s")).median())
                .sort("block")
            )
            cells = ", ".join(
                f"{row['block']:%m-%d %H} {row['ratio']:.3f}"
                for row in blocks.iter_rows(named=True)
            )
            lines.append(f"- {label} of the span ending or starting {day}: {cells}.")
    return lines


def report_text(*, frame: pl.DataFrame) -> str:
    """Render the evidence tables.

    Args:
        frame: `model_free_hours`'s result.

    Returns:
        The report, in Markdown.
    """
    months = month_ratios(frame=frame).join(era5_ratios(frame=frame), on="month")
    inside = span_ratios(frame=frame)
    heights = height_ratios()
    shared, different = equal_to_site_points()
    lines = [
        "# Open-Meteo's 10 m wind speed: two spans built differently",
        "",
        "The spans (UTC days, inclusive): "
        + "; ".join(f"{first} to {last}" for first, last in OPEN_METEO_WIND_STEP_DAYS)
        + ".",
        "",
        "## Open-Meteo's speed over CEDA's at CEDA lead 0, and both over ERA5's, by month",
        "",
        (
            "| Month | Open-Meteo over CEDA | CEDA mean (m/s) | Open-Meteo mean (m/s) | "
            "CEDA over ERA5 | Open-Meteo over ERA5 | Rows |"
        ),
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r['month']} | {r['ratio']:.3f} | {r['ceda_mean']:.3f} | {r['om_mean']:.3f} | "
            f"{r['ceda_over_era5']:.3f} | {r['om_over_era5']:.3f} | {r['n']:,} |"
            for r in months.iter_rows(named=True)
        ),
        "",
        "## Inside and outside the spans, at CEDA lead 0",
        "",
        (
            "| Inside a span | Speed ratio (Open-Meteo over CEDA) | Temperature offset (K) | "
            "Direction difference (degrees) | Irradiance ratio (CEDA over Open-Meteo) | Rows |"
        ),
        "|---|---|---|---|---|---|",
        *(
            f"| {r['om_wind_step']} | {r['speed_ratio']:.3f} | {r['temperature_offset_k']:+.3f} | "
            f"{r['direction_difference_deg']:+.1f} | {r['irradiance_ratio']:.3f} | {r['n']:,} |"
            for r in inside.iter_rows(named=True)
        ),
        "",
        "## Open-Meteo's 100 m to 10 m speed ratio at the wind farms, 2024-10 to 2025-03",
        "",
        "| Inside a span | Median ratio | Rows |",
        "|---|---|---|",
        *(
            f"| {r['om_wind_step']} | {r['ratio_100m_to_10m']:.3f} | {r['n']:,} |"
            for r in heights.iter_rows(named=True)
        ),
        "",
        "## The edges, in six-hour blocks of the speed ratio",
        "",
        *edge_blocks(frame=frame),
        "",
        "## Does `combined.parquet` match the `site_points/` extract?",
        "",
        (
            f"Inside the spans the two files share {shared:,} wind rows, and {different:,} of "
            "them hold different 10 m speeds."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Read the model-free hours, print the evidence, and write the daily series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    paths = (directory / REPORT_NAME, directory / DAILY_NAME)
    refuse_to_overwrite(paths=paths)
    frame = pl.read_parquet(directory / MODEL_FREE_NAME)
    paths[0].write_text(report_text(frame=frame))
    daily_ratio(frame=frame).write_parquet(paths[1])
    sys.stdout.write(f"Wrote {paths[0]} and {paths[1]}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
