# ruff: noqa: E501, PERF401
"""Draw experiment 210's figures as SVG from the stored tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
No chart has a calendar date on an axis: the outage timeline is by month, and the page states each
farm's outage months in text only as year and month.

    uv run --no-sync python -I ml-experiments/00210-wind-idle-noise-mva-artefact/e00210_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DATA_DIR: Final[Path] = Path("data/ml-experiments/00210-wind-idle-noise-mva-artefact")
OUT_DIR: Final[Path] = Path("docs/ml-experiments/00210-wind-idle-noise-mva-artefact")
WIDTH: Final[int] = 560
SITE_COLOURS: Final[dict[str, str]] = {
    "W1": ocf.BRAND_ORANGE,
    "W2": ocf.DATA_BLUE,
    "W3": ocf.DATA_GREEN,
}
UNITS: Final[dict[str, str]] = {"W1": "MVA", "W2": "MW", "W3": "MVA"}
WIND_AXIS: Final[alt.Axis] = alt.Axis(values=[0, 5, 10, 15, 20, 25])


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 90),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def density_panel(
    *, joined: pl.DataFrame, bins: pl.DataFrame, site: str, cut_in: float
) -> alt.Chart:
    """Draw one farm's density panel with the median and the 5% to 95% band."""
    frame = joined.filter(pl.col("site") == site)
    capacity = float(frame["capacity_mw"][0])
    n_power_bins = 50
    step = capacity * 1.1 / n_power_bins
    density = (
        frame.with_columns(
            x=(pl.col("wind_speed") / 0.5).floor() * 0.5,
            y=((pl.col("power") / step).floor() * step).clip(-step, capacity * 1.1),
        )
        .group_by("x", "y")
        .len()
        .with_columns(x2=pl.col("x") + 0.5, y2=pl.col("y") + step)
    )
    site_bins = bins.filter(pl.col("site") == site).with_columns(centre=pl.col("bin_lo") + 0.25)
    unit = UNITS[site]
    base_x = alt.X(
        "x:Q", title="ERA5 100 m wind speed (m/s)", axis=WIND_AXIS, scale=alt.Scale(domain=[0, 25])
    )
    heat = (
        alt.Chart(density)  # ty: ignore[unresolved-attribute]
        .mark_rect(aria=False)
        .encode(
            x=base_x,
            x2="x2:Q",
            y=alt.Y("y:Q", title=f"Power ({unit})"),
            y2="y2:Q",
            color=alt.Color(
                "len:Q",
                scale=alt.Scale(type="log", range=["#E8EEFB", ocf.DATA_BLUE, "#0B2A7A"]),
                legend=alt.Legend(title="Half-hours", orient="top", direction="horizontal"),
            ),
        )
    )
    line = (
        alt.Chart(site_bins)  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=2, aria=False)
        .encode(x="centre:Q", y="median:Q")
    )
    band_lo = (
        alt.Chart(site_bins)  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1, strokeDash=[4, 3], aria=False)
        .encode(x="centre:Q", y="p05:Q")
    )
    band_hi = (
        alt.Chart(site_bins)  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1, strokeDash=[4, 3], aria=False)
        .encode(x="centre:Q", y="p95:Q")
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [cut_in]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#444444", strokeDash=[2, 2], aria=False)
        .encode(x="x:Q")
    )
    return alt.layer(heat, band_lo, band_hi, line, rule).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH, height=190, title=f"{site} ({unit}, capacity {capacity:.1f})"
    )


def figure_1(*, joined: pl.DataFrame, bins: pl.DataFrame, report: dict) -> None:
    """Draw one figure (see the module docstring)."""
    panels = [
        density_panel(joined=joined, bins=bins, site=s, cut_in=report[s]["cut_in_used"])
        for s in ("W1", "W2", "W3")
    ]
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=_title(
                text="Power against ERA5 wind speed: low-power rows persist well above the turbines' cut-in speed",
                line="Shade: half-hours per 0.5 m/s by power bin (log scale), 2019-09 to 2026-09. Solid line: median per 0.5 m/s wind bin. Dashed lines: 5% and 95%. Vertical dotted line: cut-in speed used for 1.5 times cut-in (see page).",
            )
        )
    )
    chart.save(OUT_DIR / "figure_1_power_against_wind.svg")


def figure_2(*, by_year: pl.DataFrame) -> None:
    """Draw one figure (see the module docstring)."""
    rows = []
    for site in ("W1", "W2", "W3"):
        data = by_year.filter((pl.col("site") == site) & (pl.col("bin_lo") < 16)).with_columns(
            centre=pl.col("bin_lo") + 0.25
        )
        base = alt.Chart(data)
        band = base.mark_area(color=ocf.DATA_BLUE_LIGHT, opacity=0.7, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "centre:Q",
                title="Wind (m/s)",
                axis=alt.Axis(values=[0, 8, 16]),
                scale=alt.Scale(domain=[0, 16]),
            ),
            y=alt.Y("p05:Q", title=f"{site} power ({UNITS[site]})"),
            y2="p95:Q",
        )
        med = base.mark_line(color=ocf.BRAND_ORANGE, aria=False).encode(x="centre:Q", y="median:Q")  # ty: ignore[unresolved-attribute]
        rows.append(
            alt.layer(band, med)
            .properties(width=54, height=90)
            .facet(column=alt.Column("year:O", title=None, header=alt.Header(labelFontSize=10)))
        )
    alt.vconcat(*rows).properties(
        title=_title(
            text="The wind-power relationship is similar in every year, so the low-power rows are not one year's fault",
            line="Median (line) and 5% to 95% band (shade) of power per 0.5 m/s ERA5 wind bin, one column per calendar year (2019 and 2026 are part-years).",
        )
    ).save(OUT_DIR / "figure_2_by_year.svg")


def figure_3(*, report: dict) -> None:
    """Draw one figure (see the module docstring)."""
    rows = []
    for site, entry in report.items():
        for b in entry["h0"]["idle_share_by_bin"]:
            rows.append({"site": site, "wind": b["bin_lo"] + 0.25, "idle_share": b["idle_share"]})
    data = pl.DataFrame(rows).filter(pl.col("wind") < 20)
    line = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X(
                "wind:Q",
                title="ERA5 100 m wind speed (m/s)",
                axis=alt.Axis(values=[0, 5, 10, 15, 20]),
            ),
            y=alt.Y(
                "idle_share:Q",
                title="Share of half-hours in the idle band",
                scale=alt.Scale(type="symlog", constant=0.01, domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.01, 0.1, 0.5, 1], format="%"),
            ),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(domain=list(SITE_COLOURS), range=list(SITE_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=WIDTH, height=230)
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.01]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#444444", strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")
    )
    (line + rule).properties(
        title=_title(
            text="The idle share falls to 1% only at about 9 to 10 m/s for W1 and W2, and at about 16 m/s for W3",
            line="Share of half-hours whose power is at or below the 95% value of the idle floor (power at wind under 2 m/s), per 0.5 m/s ERA5 wind bin. Dashed line: the 1% limit of the pre-registered prediction. Log-like vertical scale.",
        )
    ).save(OUT_DIR / "figure_3_idle_share_by_wind.svg")


def figure_4(*, runs: pl.DataFrame, joined: pl.DataFrame) -> None:
    """Draw one figure (see the module docstring)."""
    months = (
        runs.with_columns(
            month=pl.col("start").dt.truncate("1mo").dt.replace_time_zone(None),
            hours=pl.col("n_half_hours") / 2,
        )
        .group_by("site", "month")
        .agg(pl.col("hours").sum())
    )
    full = (
        joined.with_columns(
            month=pl.col("valid_time").dt.truncate("1mo").dt.replace_time_zone(None)
        )
        .select("site", "month")
        .unique()
    )
    data = full.join(months, on=["site", "month"], how="left").with_columns(
        pl.col("hours").fill_null(0)
    )
    panels = []
    for site in ("W1", "W2", "W3"):
        panels.append(
            alt.Chart(data.filter(pl.col("site") == site))  # ty: ignore[unresolved-attribute]
            .mark_bar(color=SITE_COLOURS[site], aria=False)
            .encode(
                x=alt.X(
                    "yearmonth(month):T", title=None, axis=alt.Axis(format="%Y", tickCount="year")
                ),
                y=alt.Y("hours:Q", title=f"{site}: outage hours"),
            )
            .properties(width=WIDTH, height=70)
        )
    alt.vconcat(*panels).properties(
        title=_title(
            text="Suspected outages are rare for W1, regular for W2, and frequent for W3",
            line="Hours per month in runs of at least 12 hours at high ERA5 wind with power in the idle band, by the month in which the run starts. Each panel has its own vertical scale.",
        )
    ).save(OUT_DIR / "figure_4_outage_timeline.svg")


def figure_5(*, monthly: pl.DataFrame) -> None:
    """Draw one figure (see the module docstring)."""
    (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X(
                "month:Q",
                title="Calendar month",
                axis=alt.Axis(values=[1, 4, 7, 10]),
                scale=alt.Scale(domain=[1, 12]),
            ),
            y=alt.Y(
                "share_dropped:Q", title="Share of half-hours dropped", axis=alt.Axis(format="%")
            ),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(domain=list(SITE_COLOURS), range=list(SITE_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The dropped share is highest in spring and summer, when the wind is lightest",
                line="Share of half-hours dropped by the wind-speed rule plus the suspected-outage rows, by calendar month, all years.",
            ),
        )
        .save(OUT_DIR / "figure_5_dropped_by_month.svg")
    )


def main() -> None:
    """Draw one figure (see the module docstring)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((DATA_DIR / "report.json").read_text())
    joined = pl.read_parquet(DATA_DIR / "joined.parquet")
    figure_1(joined=joined, bins=pl.read_parquet(DATA_DIR / "bins.parquet"), report=report)
    figure_2(by_year=pl.read_parquet(DATA_DIR / "bins_by_year.parquet"))
    figure_3(report=report)
    figure_4(runs=pl.read_parquet(DATA_DIR / "outage_runs.parquet"), joined=joined)
    figure_5(monthly=pl.read_parquet(DATA_DIR / "dropped_by_month.parquet"))


if __name__ == "__main__":
    main()
