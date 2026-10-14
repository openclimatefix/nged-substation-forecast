"""Score the holiday and Christmas columns on special-day windows rather than on the whole year.

A column can help a lot on the few weeks that are hard to forecast and little when averaged over a
year. This script reads the Christmas-control predictions (three folds, three fitting seeds, every
substation type), cuts the valid times into calendar windows (Christmas, Easter, and the summer
school holidays), and reports each column's change in mean absolute error against the weather
block plus clock plus COVID indicator inside each window, with a 95% interval that resamples whole
days. Each window holds three occurrences, one per fold, and neighbouring days are correlated, so
the intervals describe these three years only. Figure 12 draws the Primary changes.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_special_days.py
"""

from datetime import date, timedelta
from typing import Final

import altair as alt
import e00006_charts as charts
import e00006_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from dateutil.easter import easter

BASE: Final[str] = "W+clock+covid_flag"
ARMS: Final[dict[str, str]] = {
    "W+clock+covid_flag+xmas": "Days to Christmas",
    "W+clock+covid_flag+xmas_window": "Days to Christmas, 14-day window",
    "W+clock+covid_flag+hol_flags": "Holiday flags",
    "W+clock+covid_flag+doy1": "One harmonic of time of year",
    "W+clock+covid_flag+season_half": "Half-year flag",
}
FOLDS: Final[tuple[str, ...]] = ("select_2023", "select_2024", "confirm_2025")
N_RESAMPLES: Final[int] = 4000
SEED: Final[int] = 0


def windows_of(*, year: int) -> dict[str, tuple[date, date]]:
    """Return each window's first and last local date for the occurrence that starts in a year.

    The Christmas windows start in December of `year`, and the Easter and summer windows fall in
    `year + 1` for a fold that starts in July of `year`.
    """
    easter_day = easter(year + 1)
    return {
        "Christmas week (24 Dec to 1 Jan)": (date(year, 12, 24), date(year + 1, 1, 1)),
        "Christmas weeks (15 Dec to 7 Jan)": (date(year, 12, 15), date(year + 1, 1, 7)),
        "Easter long weekend (Thursday to Tuesday)": (
            easter_day - timedelta(days=3),
            easter_day + timedelta(days=2),
        ),
        "Easter fortnight (7 days either side)": (
            easter_day - timedelta(days=7),
            easter_day + timedelta(days=7),
        ),
        "Summer school holidays (22 Jul to 31 Aug)": (date(year, 7, 22), date(year, 8, 31)),
    }


def window_name(*, fold: str, local_date: date) -> list[str]:
    """Return the windows a local date falls in, for one fold."""
    year = {"select_2023": 2023, "select_2024": 2024, "confirm_2025": 2025}[fold]
    return [n for n, (a, b) in windows_of(year=year).items() if a <= local_date <= b]


def daily_errors() -> pl.DataFrame:
    """Return the mean absolute error of each arm per type, fold, local day, and window."""
    pieces = []
    types = report.series_types()
    for fold in FOLDS:
        rows = report.read_seeds(stage="christmas_check", fold_name=fold).join(
            types.select("time_series_id", "substation_type"), on="time_series_id"
        )
        arms = [BASE, *ARMS]
        local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
        frame = rows.with_columns(
            local_date=(local - pl.duration(minutes=1)).dt.date(),
            **{
                f"err_{i}": (pl.col(a) - pl.col("target")).abs() * 100.0 for i, a in enumerate(arms)
            },
        )
        daily = frame.group_by("substation_type", "local_date", "seed").agg(
            [pl.col(f"err_{i}").mean().alias(a) for i, a in enumerate(arms)]
        )
        daily = daily.group_by("substation_type", "local_date").agg(pl.col(arms).mean())
        pieces.append(daily.with_columns(fold=pl.lit(fold)))
    return pl.concat(pieces)


def window_table(*, daily: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's change against the base in each window, by type, with 95% intervals."""
    rng = np.random.default_rng(SEED)
    out = []
    members = daily.with_columns(
        window=pl.struct("fold", "local_date").map_elements(
            lambda s: window_name(fold=s["fold"], local_date=s["local_date"]),
            return_dtype=pl.List(pl.String),
        )
    )
    inside = members.explode("window").drop_nulls("window")
    outside = members.filter(pl.col("window").list.len() == 0).with_columns(
        window=pl.lit("Every other day")
    )
    for (substation_type, window), frame in pl.concat([inside, outside]).group_by(
        "substation_type", "window"
    ):
        base = frame[BASE].to_numpy()
        for arm in ARMS:
            diff = frame[arm].to_numpy() - base
            idx = rng.integers(0, len(diff), size=(N_RESAMPLES, len(diff)))
            draws = diff[idx].mean(axis=1)
            out.append(
                {
                    "substation_type": substation_type,
                    "window": window,
                    "arm": ARMS[arm],
                    "days": len(diff),
                    "base_error": float(base.mean()),
                    "change": float(diff.mean()),
                    "lower": float(np.quantile(draws, 0.025)),
                    "upper": float(np.quantile(draws, 0.975)),
                }
            )
    return pl.DataFrame(out)


def christmas_by_year(*, daily: pl.DataFrame) -> pl.DataFrame:
    """Return the change in Christmas-week error per fold for Primary, one row per arm."""
    rows = []
    for fold in FOLDS:
        year = {"select_2023": 2023, "select_2024": 2024, "confirm_2025": 2025}[fold]
        start, end = windows_of(year=year)["Christmas weeks (15 Dec to 7 Jan)"]
        frame = daily.filter(
            pl.col("substation_type") == "Primary",
            pl.col("fold") == fold,
            pl.col("local_date").is_between(start, end),
        )
        for arm, label in ARMS.items():
            rows.append(
                {
                    "fold": fold,
                    "arm": label,
                    "base": frame[BASE].to_numpy().mean(),
                    "change": (frame[arm].to_numpy() - frame[BASE].to_numpy()).mean(),
                }
            )
    return pl.DataFrame(rows)


def figure_12(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the Primary change in error by window and column."""
    primary = table.filter(
        pl.col("substation_type") == "Primary", pl.col("window") != "Every other day"
    )
    order = list(windows_of(year=2023))
    colours = {
        "Days to Christmas": ocf.BRAND_ORANGE,
        "Days to Christmas, 14-day window": ocf.DATA_SKY,
        "Holiday flags": ocf.DATA_BLUE,
        "One harmonic of time of year": ocf.DATA_PURPLE,
        "Half-year flag": "#999999",
    }
    base = alt.Chart(primary).encode(
        y=alt.Y("window:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("arm:N", sort=list(colours)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=320),
        ),
    )
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title="Change in mean absolute error (points of capacity)"),
        x2="upper:Q",
    )
    dots = base.mark_circle(size=60).encode(x="change:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222222").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(rules, dots, zero).properties(
        width=charts.WIDTH,
        height=300,
        title=charts._title(
            text="The Christmas columns cut the error at Christmas and are neutral elsewhere",
            lines=[
                (
                    "Change in the mean absolute error of the Primary substations against the "
                    "weather block plus clock plus COVID indicator, inside each window, over the "
                    "three folds (one occurrence per fold). More negative is better. Dot: "
                    "estimate. Line: 95% interval resampling whole days."
                )
            ],
        ),
    )


def main() -> None:
    """Print the tables the write-up quotes and draw figure 12."""
    daily = daily_errors()
    table = window_table(daily=daily)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(60)
    for substation_type in ("Primary", "BSP", "GSP"):
        print(substation_type)
        print(
            table.filter(pl.col("substation_type") == substation_type)
            .sort("window", "arm")
            .drop("substation_type")
            .with_columns(pl.col("base_error", "change", "lower", "upper").round(2))
        )
    print("Primary Christmas weeks by year")
    print(christmas_by_year(daily=daily).with_columns(pl.col("base", "change").round(2)))
    chart = figure_12(table=table)
    chart.save(charts.OUT_DIR / "figure_12.svg")
    chart.save(report.RESULTS_DIR / "figure_12.png", scale_factor=2)


if __name__ == "__main__":
    main()
