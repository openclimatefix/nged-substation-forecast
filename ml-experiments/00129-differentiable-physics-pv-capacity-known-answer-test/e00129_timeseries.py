"""Time-series figure of experiment 129: Primary power beside the estimated solar power.

For three Primaries and three days each, the figure shows the metered net power and the estimated
solar power, which is arm 2's latest-year capacity estimate times the unit PV shape (before the
panels' temperature derating). The three days at a Primary are chosen by a rule fixed before any
plot was drawn: among the days from 2026-05-01 to 2026-07-31 with a full day of readings outside
the switching masks, the day with the lowest, the median, and the highest mean CAMS irradiance
between 06:00 and 18:00 UTC.

    E00129_THROUGH_JULY=1 uv run --no-sync python ml-experiments/00129-*/e00129_timeseries.py
"""

from datetime import UTC, datetime
from typing import Final

import altair as alt
import e00129_charts as charts
import e00129_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_features import FeatureCache

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D14")
FIRST_DAY: Final[datetime] = datetime(2026, 5, 1, tzinfo=UTC)
LAST_DAY: Final[datetime] = datetime(2026, 7, 31, tzinfo=UTC)
SCENARIOS: Final[tuple[str, ...]] = ("Cloudiest day", "Median day", "Sunniest day")
HALF_HOURS_PER_DAY: Final[int] = 48


def chosen_days(*, cache: FeatureCache, site: str) -> dict[str, np.datetime64]:
    """Return the cloudiest, median, and sunniest full day at a Primary, by the rule above."""
    features = cache.sites[site]
    starts = cache.end_times - np.timedelta64(30, "m")
    days = starts.astype("datetime64[D]")
    hours = (starts - days.astype("datetime64[us]")) / np.timedelta64(1, "h")
    frame = (
        pl.DataFrame(
            {
                "day": days,
                "ghi": np.where((hours >= 6) & (hours < 18), features.ghi_w_m2, np.nan),
                "ok": features.present & ~features.switching_mask,
            }
        )
        .with_columns(ghi=pl.col("ghi").fill_nan(None))
        .filter(
            pl.col("day") >= FIRST_DAY.date(),
            pl.col("day") <= LAST_DAY.date(),
        )
    )
    daily = (
        frame.group_by("day")
        .agg(ghi=pl.col("ghi").mean(), ok=pl.col("ok").all(), n=pl.len())
        .filter(pl.col("ok"), pl.col("n") == HALF_HOURS_PER_DAY)
        .sort("ghi")
    )
    ghi = daily["ghi"].to_list()
    median_index = int(np.argmin(np.abs(np.array(ghi) - np.median(ghi))))
    picks = (0, median_index, daily.height - 1)
    return {
        scenario: np.datetime64(daily["day"][position])
        for scenario, position in zip(SCENARIOS, picks, strict=True)
    }


def series_rows(*, cache: FeatureCache) -> list[dict]:
    """Return one row per site, scenario, and half-hour."""
    estimates = {
        row["site"]: row["estimate_mw"]
        for row in pl.read_parquet(common.WORK_DIR / "real_arm2_to_july.parquet")
        .filter(pl.col("season_year") == common.SEASON_YEARS[-1])
        .iter_rows(named=True)
    }
    rows = []
    for site in SITES:
        features = cache.sites[site]
        scale = features.table.effective_capacity_mw / 100.0
        starts = cache.end_times - np.timedelta64(30, "m")
        for scenario, day in chosen_days(cache=cache, site=site).items():
            index = np.flatnonzero(starts.astype("datetime64[D]") == day)
            rows.extend(
                {
                    "site": site,
                    "scenario": scenario,
                    "panel": f"{scenario}, {np.datetime_as_string(day)}",
                    "hour": float(
                        (starts[i] - day.astype("datetime64[us]")) / np.timedelta64(1, "h")
                    ),
                    "net_power_mw": float(features.y[i] * scale),
                    "solar_mw": float(np.nan_to_num(features.pv_unit_shape[i]) * estimates[site]),
                }
                for i in index
            )
    return rows


def figure_timeseries(*, cache: FeatureCache) -> alt.Chart:
    """Return the net power and estimated solar power, per site (rows) and scenario (columns)."""
    table = pl.DataFrame(series_rows(cache=cache)).unpivot(
        index=["site", "scenario", "panel", "hour"],
        on=["net_power_mw", "solar_mw"],
        variable_name="series",
        value_name="mw",
    )
    table = table.with_columns(
        series=pl.col("series").replace(
            {
                "net_power_mw": "Metered net power",
                "solar_mw": "Estimated solar power",
            }
        )
    )
    order = [
        "Metered net power",
        "Estimated solar power",
    ]
    chart = (
        alt.Chart(table)
        .mark_line(clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hour of the day (UTC)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y("mw:Q", title="Power (MW)"),
            color=alt.Color(
                "series:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=170, height=110)
    )
    return (
        chart.facet(
            column=alt.Column("scenario:N", sort=list(SCENARIOS), title=None),
            row=alt.Row("site:N", sort=list(SITES), title=None),
        )
        .resolve_scale(y="independent")
        .properties(
            title=charts.chart_title(
                text="Primary power beside estimated solar power on cloudy, median, and sunny days",
                subtitle=(
                    "Blue: power metered at the Primary, with metered generators already"
                    " subtracted. Orange: arm 2's 2025/26 capacity estimate times the CAMS-driven"
                    " solar shape, before temperature derating. Days between May and July 2026,"
                    " chosen at each Primary by mean daylight irradiance. Each row has its own"
                    " scale."
                ),
            )
        )
    )


def main() -> None:
    """Draw the figure into the docs folder."""
    charts.save_chart(figure_timeseries(cache=common.load_cache()), name="figure_04_timeseries")


if __name__ == "__main__":
    main()
