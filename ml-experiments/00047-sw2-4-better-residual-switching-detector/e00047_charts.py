"""Draw experiment 47's figures as SVG: raw power around bank holidays first, then the comparisons.

The series are Primary substations, not NGED generators. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00047-*/e00047_charts.py
"""

from __future__ import annotations

import argparse
import subprocess
import textwrap
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00047_common import GRID_LENGTH, GRID_START, WORK_DIR
from e00047_detect import SETTINGS, load_arm_panel
from e00047_fit import FITS_DIR
from e00047_inject import DURATION_BANDS, SIZE_BANDS
from e00047_report import (
    CONTRASTS,
    LABELS,
    SEEDS,
    event_table,
    paired_interval,
    seed_mean_found,
)

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00047-sw2-4-better-residual-switching-detector"
)
WIDTH: Final[int] = 680
PLOT_WIDTH: Final[int] = 520
WRAP: Final[int] = 100
ARMS: Final[dict[str, str]] = {
    "H0.5": "H0.5 (baseline)",
    "HOL": "H0.5 + days to nearest bank holiday",
    "PLA": "Placebo (holidays 28 days later)",
}
ARM_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
HOLIDAY_EXAMPLES: Final[tuple[date, ...]] = (date(2023, 5, 1), date(2024, 4, 1), date(2025, 8, 25))
EXAMPLE_SERIES: Final[int] = 10
SEED_SETS: Final[dict[str, tuple[int, ...]]] = {"H0.5": SEEDS, "HOL": SEEDS, "PLA": (0,)}


def _colour(*, title: str | None = None, arms: tuple[str, ...] = tuple(ARMS)) -> alt.Color:
    """Colour by arm, with a legend above the plot that names only the given arms."""
    domain = [ARMS[a] for a in arms]
    colours = [ARM_COLOURS[list(ARMS).index(a)] for a in arms]
    return alt.Color(
        "arm:N",
        title=title,
        scale=alt.Scale(domain=domain, range=colours),
        legend=alt.Legend(orient="top", columns=1, labelLimit=400),
    )


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _names(*, out: Path) -> list[str]:
    return sorted(p.stem.removeprefix("thresholds_") for p in out.glob("thresholds_*.json"))


def _arm_label(name: str, /) -> str:
    return ARMS[name.split("_s", maxsplit=1)[0]]


def figure_raw_holidays(*, fits: Path, fallback: bool = False) -> alt.TopLevelMixin:
    """Measured power, then both residuals, for 3 days either side of three bank holidays."""
    panels = {
        arm: load_arm_panel(name=name, fits_dir=fits, fallback=fallback)
        for arm, name in (("H0.5", "H0.5_s0"), ("HOL", "HOL_s0"))
    }
    power = panels["H0.5"].power[EXAMPLE_SERIES]
    charts = []
    for holiday in HOLIDAY_EXAMPLES:
        midnight = datetime(holiday.year, holiday.month, holiday.day, tzinfo=UTC)
        centre = int((midnight - GRID_START).total_seconds() // 1800)
        lo, hi = max(centre - 3 * 48, 0), min(centre + 4 * 48, GRID_LENGTH)
        hours = (np.arange(lo, hi) - centre) / 2
        x = alt.X(
            "hours:Q",
            title="Hours from midnight at the start of the bank holiday",
            axis=alt.Axis(values=[-72, -48, -24, 0, 24, 48, 72, 96]),
        )
        raw = pl.DataFrame({"hours": hours, "mw": power[lo:hi]})
        top = (
            alt.Chart(raw)
            .mark_line(strokeWidth=1.2, color=ocf.DATA_BLUE, clip=True)
            .encode(x=x, y=alt.Y("mw:Q", title="Power (MW)"))  # ty: ignore[unresolved-attribute]
            .properties(width=PLOT_WIDTH, height=90)
        )
        residual = pl.concat(
            [
                pl.DataFrame(
                    {
                        "hours": hours,
                        "residual": panels[arm].residual[EXAMPLE_SERIES][lo:hi],
                        "arm": ARMS[arm],
                    }
                )
                for arm in panels
            ]
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [0.0, 24.0]}))
            .encode(x="x:Q")
            .mark_rule(strokeDash=[4, 3], color="#444444")
        )
        bottom = alt.layer(
            alt.Chart(residual)
            .mark_line(strokeWidth=1.2, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x,
                y=alt.Y("residual:Q", title="Residual (MW)"),
                color=_colour(arms=("H0.5", "HOL")),
            ),
            rule,  # ty: ignore[invalid-argument-type]
        ).properties(width=PLOT_WIDTH, height=110)
        charts.append(alt.vconcat(top, bottom).properties(title=f"{holiday:%B %Y}: {holiday:%A}"))
    return alt.vconcat(*charts).properties(
        title=_title(
            text="Power and residuals around three bank holidays at one Primary substation",
            lines=[
                (
                    f"Primary substation, series {EXAMPLE_SERIES}. Top panels: measured power. "
                    "Bottom panels: measured power minus the weather model's expectation (the "
                    "residual) "
                    "for the baseline and the arm with the bank-holiday column, seed 0. Dashed "
                    "lines "
                    "bound the holiday's 24 hours (UTC)."
                )
            ],
        )
    )


def figure_spread(*, out: Path) -> alt.TopLevelMixin:
    """Residual spread in points of capacity per series, one dot per arm (seed mean)."""
    spread = pl.read_parquet(out / "diag_spread.parquet").with_columns(
        arm=pl.col("name").map_elements(_arm_label, return_dtype=pl.String),
        series_label=pl.format("Series {}", pl.col("series")),
    )
    data = spread.group_by("arm", "series_label").agg(pl.col("mad_points_of_capacity").mean())
    return (
        alt.Chart(data)
        .mark_point(filled=True, size=60, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("series_label:N", title=None, sort=None),
            x=alt.X("mad_points_of_capacity:Q", title="Residual spread (points of capacity)"),
            color=_colour(),
        )
        .properties(
            width=PLOT_WIDTH,
            height=330,
            title=_title(
                text="Spread of the residual per Primary substation and arm",
                lines=[
                    (
                        "Median absolute deviation of the residual per Primary substation, as a "
                        "share of "
                        "the substation's capacity, in points; mean over seeds. Smaller is tighter."
                    )
                ],
            ),
        )
    )


def figure_autocorrelation(*, out: Path) -> alt.TopLevelMixin:
    """Mean autocorrelation of the residual at lag 1, lag 48, and of its daily mean at lag 1."""
    spread = pl.read_parquet(out / "diag_spread.parquet").with_columns(
        arm=pl.col("name").map_elements(_arm_label, return_dtype=pl.String)
    )
    long = (
        spread.group_by("arm")
        .agg(
            pl.col("lag_1").mean(),
            pl.col("lag_48").mean(),
            pl.col("daily_lag_1").mean(),
        )
        .unpivot(index="arm", variable_name="metric", value_name="correlation")
        .with_columns(
            metric=pl.col("metric").replace(
                {
                    "lag_1": "Half-hour to half-hour (lag 1)",
                    "lag_48": "Same time previous day (lag 48)",
                    "daily_lag_1": "Daily mean to previous daily mean",
                }
            )
        )
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("metric:N", title=None, sort=None),
            yOffset=alt.YOffset("arm:N", scale=alt.Scale(domain=list(ARMS.values()))),
            x=alt.X(
                "correlation:Q", title="Mean autocorrelation of the residual over the 16 series"
            ),
            color=_colour(),
        )
        .properties(
            width=PLOT_WIDTH,
            height=160,
            title=_title(
                text="Autocorrelation of the residual by arm",
                lines=[
                    (
                        "A residual that stays high for days can look like a step. "
                        "Mean over the 16 Primary substations and over seeds."
                    )
                ],
            ),
        )
    )


def figure_shares(*, out: Path) -> alt.TopLevelMixin:
    """Share of squared residual inside logged closures and labelled visible transfers."""
    shares = pl.read_parquet(out / "diag_shares.parquet").with_columns(
        arm=pl.col("name").map_elements(_arm_label, return_dtype=pl.String)
    )
    data = shares.group_by("arm", "region").agg(
        pl.col("share_of_squared_residual").mean(), pl.col("share_of_half_hours").mean()
    )
    base = alt.Chart(data).encode(y=alt.Y("region:N", title=None))
    bars = base.encode(
        yOffset=alt.YOffset("arm:N", scale=alt.Scale(domain=list(ARMS.values()))),
        x=alt.X("share_of_squared_residual:Q", title="Share of the residual's squared sum"),
        color=_colour(),
    ).mark_bar()
    ticks = base.encode(x="share_of_half_hours:Q").mark_tick(color="black", thickness=2, size=44)
    return alt.layer(bars, ticks).properties(  # ty: ignore[invalid-argument-type]
        width=PLOT_WIDTH,
        height=140,
        title=_title(
            text="Share of the squared residual inside logged closures and labelled transfers",
            lines=[
                (
                    "Bars: share of the squared residual inside logged closures and inside "
                    "labelled "
                    "visible transfers. Black tick: those regions' share of all half-hours. A bar "
                    "beyond its tick means larger residual inside the region."
                )
            ],
        ),
    )


def figure_operating_points(*, out: Path) -> alt.TopLevelMixin:
    """Odd-month unexplained alarms against threshold, per arm (seed 0), a panel per setting."""
    frames = [
        pl.read_parquet(out / f"curve_{name}.parquet").with_columns(arm=pl.lit(_arm_label(name)))
        for name in _names(out=out)
        if name.endswith("_s0")
    ]
    data = pl.concat(frames).with_columns(setting_label=pl.col("setting").replace(LABELS))
    panels = []
    for setting in SETTINGS:
        part = data.filter(pl.col("setting") == setting)
        target = part["target"][0]
        line = (
            alt.Chart(part)
            .mark_line(point=True, strokeWidth=1.5)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("threshold:Q", title="Threshold (units of the odd-month noise scale)"),
                y=alt.Y("odd_unexplained:Q", title="Odd-month unexplained alarms"),
                color=_colour(),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [target]}))
            .mark_rule(strokeDash=[4, 3], color="#444444")
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(line, rule).properties(
                width=PLOT_WIDTH, height=110, title=f"{LABELS[setting]} (target {target})"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Unexplained alarms in odd months against threshold, by arm",
            lines=[
                (
                    "Unexplained alarms (more than 24 hours from a logged closure) in odd months "
                    "against the detector threshold, seed 0. Dashed line: the baseline's count at "
                    "experiment 43's threshold, which every arm's threshold is matched to."
                )
            ],
        )
    )


def figure_alarms_by_holiday_distance(*, out: Path) -> alt.TopLevelMixin:
    """Unexplained D0 24-hour alarms by distance to the nearest bank holiday, per arm."""
    rows = []
    for name in _names(out=out):
        alarms = pl.read_parquet(out / f"alarms_matched_{name}.parquet").filter(
            pl.col("setting") == "D0_48", ~pl.col("logged_matched")
        )
        counts = alarms.group_by(pl.col("holiday_distance").abs().clip(0, 7).alias("days")).len()
        rows.append(counts.with_columns(arm=pl.lit(_arm_label(name)), name=pl.lit(name)))
    data = (
        pl.concat(rows)
        .group_by("arm", "days")
        .agg(count=pl.col("len").sum() / pl.col("name").n_unique())
        .with_columns(days=pl.col("days").cast(pl.Int64))
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "days:O",
                title="Days from the nearest bank holiday outside Christmas (7 means 7 or more)",
            ),
            xOffset=alt.XOffset("arm:N", scale=alt.Scale(domain=list(ARMS.values()))),
            y=alt.Y("count:Q", title="Unexplained alarms, mean over seeds"),
            color=_colour(),
        )
        .properties(
            width=PLOT_WIDTH,
            height=220,
            title=_title(
                text="Unexplained alarms by distance to the nearest bank holiday",
                lines=[
                    (
                        "D0 detector, 24-hour window, all months, at each arm's matched threshold; "
                        "alarms more than 24 hours from a logged closure."
                    )
                ],
            ),
        )
    )


def figure_recall_by_band(*, out: Path) -> alt.TopLevelMixin:
    """Share of injected transfers found by band, D0 at 24 hours, even months, per arm."""
    events = event_table(out=out)
    rows = []
    for arm, seeds in SEED_SETS.items():
        found = seed_mean_found(out=out, arm=arm, events=events, setting="D0_48", seeds=seeds)
        table = events.join(found, on="event_id").filter(pl.col("even"))
        for size in SIZE_BANDS:
            for duration in DURATION_BANDS:
                band = table.filter(
                    pl.col("size_band") == size, pl.col("duration_band") == duration
                )
                rows.append(
                    {
                        "arm": ARMS[arm],
                        "band": f"{size} of receiving load, {duration}",
                        "found": float(band["found"].mean()),  # ty: ignore[invalid-argument-type]
                    }
                )
    order = [f"{s} of receiving load, {d}" for s in SIZE_BANDS for d in DURATION_BANDS]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=60, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("band:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            yOffset=alt.YOffset("arm:N", scale=alt.Scale(domain=list(ARMS.values()))),
            x=alt.X(
                "found:Q",
                title="Share of injected transfers found (larger is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=_colour(),
        )
        .properties(
            width=PLOT_WIDTH - 120,
            height=320,
            title=_title(
                text="Share of injected transfers found, by size and duration band",
                lines=[
                    (
                        "D0 detector, 24-hour window, transfers starting in even months, mean over "
                        "seeds (the placebo has one seed)."
                    )
                ],
            ),
        )
    )


def figure_contrasts(*, out: Path) -> alt.TopLevelMixin:
    """C1 to C4 with their 98.75% intervals, and the placebo against the baseline for C1 to C4."""
    events = event_table(out=out)
    even_headline = events.filter(pl.col("headline"), pl.col("even")).select(
        "event_id", "year_month"
    )
    rows = []
    comparisons = (
        ("New arm minus baseline (seeds 0 to 2)", "HOL", SEEDS, "H0.5", SEEDS),
        ("Placebo minus baseline (seed 0)", "PLA", (0,), "H0.5", (0,)),
    )
    for label, arm, seeds, ref, ref_seeds in comparisons:
        for index, (contrast, setting) in enumerate(CONTRASTS.items()):
            new = seed_mean_found(out=out, arm=arm, events=events, setting=setting, seeds=seeds)
            reference = seed_mean_found(
                out=out, arm=ref, events=events, setting=setting, seeds=ref_seeds
            )
            mean, low, high = paired_interval(
                population=even_headline, new=new, reference=reference, seed=4700 + index
            )
            rows.append(
                {
                    "comparison": label,
                    "contrast": f"{contrast}: {LABELS[setting]}",
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    data = pl.DataFrame(rows)
    domain = [c[0] for c in comparisons]
    colour = alt.Color(
        "comparison:N",
        title=None,
        scale=alt.Scale(domain=domain, range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE]),
        legend=alt.Legend(orient="top", columns=1, labelLimit=400),
    )
    offset = alt.YOffset("comparison:N", scale=alt.Scale(domain=domain))
    y = alt.Y("contrast:N", title=None, axis=alt.Axis(labelLimit=200))
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#444444").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    bar = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X("low:Q", title="Change in share of transfers found (larger is better)"),
            x2="high:Q",
            color=colour,
        )
    )
    dot = (
        alt.Chart(data)
        .mark_point(filled=True, size=60, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, yOffset=offset, x="mean:Q", color=colour
        )
    )
    return alt.layer(zero, bar, dot).properties(
        width=PLOT_WIDTH - 90,
        height=220,
        title=_title(
            text="Change in the share of injected transfers found, against the baseline",
            lines=[
                (
                    "Dot: mean paired difference over the headline band (25% to 40% of the "
                    "receiving "
                    "load, over 1 day) in even months. Line: 98.75% interval from resampling "
                    "whole calendar months. The vertical line is no change; the verdict needs "
                    "the interval of C1 wholly above it."
                )
            ],
        ),
    )


def figure_median_error(*, out: Path) -> alt.TopLevelMixin:
    """Mean absolute error of the median by scope, per arm."""
    mae = pl.read_parquet(out / "diag_mae.parquet").with_columns(
        arm=pl.col("name").map_elements(_arm_label, return_dtype=pl.String)
    )
    data = mae.group_by("arm", "scope").agg(pl.col("mae_points").mean())
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "scope:N",
                title=None,
                sort=["all rows", "bank holidays and the day either side", "bank holidays"],
            ),
            yOffset=alt.YOffset("arm:N", scale=alt.Scale(domain=list(ARMS.values()))),
            x=alt.X(
                "mae_points:Q",
                title="Mean absolute error of the median (points of capacity; smaller is better)",
            ),
            color=_colour(),
        )
        .properties(
            width=PLOT_WIDTH - 100,
            height=170,
            title=_title(
                text="Error of the median forecast, all rows and on bank holidays",
                lines=[
                    (
                        "Mean absolute error over the 16 Primary substations, January 2022 to June "
                        "2026, mean over seeds. Bank holidays include Christmas Day, Boxing Day, "
                        "and New "
                        "Year's Day."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write every figure that its inputs allow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    parser.add_argument("--fits", type=Path, default=FITS_DIR)
    parser.add_argument("--figures", type=Path, default=OUT_DIR)
    parser.add_argument("--fallback", action="store_true", help="smoke test: fill missing months")
    parser.add_argument("--optimise", action="store_true", help="run svgo over the written SVGs")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    out = args.out
    figures = {
        "figure_1_raw_power_around_bank_holidays": lambda: figure_raw_holidays(
            fits=args.fits, fallback=args.fallback
        ),
        "figure_2_residual_spread": lambda: figure_spread(out=out),
        "figure_3_residual_autocorrelation": lambda: figure_autocorrelation(out=out),
        "figure_4_squared_residual_shares": lambda: figure_shares(out=out),
        "figure_5_operating_points": lambda: figure_operating_points(out=out),
        "figure_6_alarms_by_holiday_distance": lambda: figure_alarms_by_holiday_distance(out=out),
        "figure_7_found_by_band": lambda: figure_recall_by_band(out=out),
        "figure_8_contrasts": lambda: figure_contrasts(out=out),
        "figure_9_median_error": lambda: figure_median_error(out=out),
    }
    for name, make in figures.items():
        try:
            chart = make()
        except FileNotFoundError as error:
            print("skipped", name, "-", error, flush=True)
            continue
        chart.save(args.figures / f"{name}.svg")
        print("wrote", name, flush=True)
    if args.optimise:
        svgs = sorted(str(p) for p in args.figures.glob("figure_*.svg"))
        subprocess.run(
            ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", *svgs], check=True
        )


if __name__ == "__main__":
    main()
