"""Draw the figures of experiment 174's write-up from the stored tables.

uv run --no-sync python ml-experiments/00174-*/e00174_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00174_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 100
GREY: Final[str] = "#808080"


def lines(*, text: str) -> list[str]:
    """Wrap text to the plot width."""
    return textwrap.wrap(text, WRAP)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG into the docs folder and PNG into the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def split_chart() -> alt.TopLevelMixin:
    """Return the three-way split of the unlabelled days' excess loss, with intervals."""
    table = pl.read_parquet(common.WORK_DIR / "split.parquet")
    order = list(table["part"])
    colours = [ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE]
    base = alt.Chart(table).encode(
        y=alt.Y("part:N", sort=order, title=None),
        color=alt.Color("part:N", scale=alt.Scale(domain=order, range=colours), legend=None),
    )
    bars = base.mark_bar(size=26).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "share:Q",
            title="Share of the unlabelled days' excess loss",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 0.7]),
        )
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="low:Q", x2="high:Q", color=alt.value("#111111")
    )
    return (bars + rule).properties(
        width=WIDTH,
        height=150,
        title=alt.TitleParams(
            "Weather explains about 3 in 10 of the loss on the unlabelled worst days",
            subtitle=lines(
                text="37 worst days of five substations that experiment 172 could not label. "
                "Bar: share of their excess loss. Black line: 95% interval from resampling whole "
                "calendar months. Weather: the part that disappears when ERA5 and CAMS replace "
                "the ensemble mean. Shared: the rest, on days when another Primary is also in its "
                "own worst 5%. Neither: the rest, on days no other Primary shares."
            ),
        ),
    )


def day_chart() -> alt.TopLevelMixin:
    """Return each unlabelled day's excess loss with ensemble weather and with past weather."""
    days = pl.read_parquet(common.WORK_DIR / "worst_days_split.parquet").filter(~pl.col("labelled"))
    days = days.sort("excess_ens", descending=True).with_columns(
        day=pl.concat_str("site", pl.lit(" "), pl.col("local_date").dt.strftime("%b %d")),
        shared=pl.when(pl.col("shared")).then(pl.lit("Shared day")).otherwise(pl.lit("Not shared")),
    )
    order = list(days["day"])
    domain = ["Shared day", "Not shared"]
    y = alt.Y("day:N", sort=order, title=None, axis=alt.Axis(labelFontSize=9))
    x = alt.X("excess_ens:Q", title="Excess loss (points of capacity; smaller is better)")
    link = (
        alt.Chart(days)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY)
        .encode(y=y, x=x, x2="excess_era5:Q")
    )
    ens = (
        alt.Chart(days)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45, shape="circle")
        .encode(
            y=y,
            x=x,
            color=alt.Color(
                "shared:N",
                scale=alt.Scale(domain=domain, range=[ocf.DATA_PURPLE, ocf.DATA_BLUE]),
                legend=alt.Legend(title="Ensemble-weather dot", orient="top"),
            ),
        )
    )
    era5 = (
        alt.Chart(days)  # ty: ignore[unresolved-attribute]
        .mark_point(shape="diamond", size=45, color="#333333", filled=False)
        .encode(y=y, x="excess_era5:Q")
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=GREY, strokeDash=[3, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + link + ens + era5).properties(
        width=WIDTH,
        height=560,
        title=alt.TitleParams(
            "On most unlabelled days the loss stays when past weather is used",
            subtitle=lines(
                text="Coloured dot: a day's excess loss with the ensemble-mean weather. Open "
                "diamond: the same day with ERA5 and CAMS. Grey line joins them; the dashed rule "
                "marks the substation's median day. Left of the dot is better."
            ),
        ),
    )


def chance_chart() -> alt.TopLevelMixin:
    """Return the circular-shift control for the shared-day share."""
    draws = np.load(common.WORK_DIR / "chance_any_other.npy")
    frame = pl.DataFrame({"share": draws})
    observed = common.mean_of(
        pl.read_parquet(common.WORK_DIR / "worst_days_split.parquet").filter(~pl.col("labelled"))[
            "shared"
        ]
    )
    hist = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_SKY)
        .encode(
            x=alt.X(
                "share:Q",
                bin=alt.Bin(step=0.027),
                title="Share of the 37 days with another Primary also in its worst 5%",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("count()", title="Draws of 1,000"),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"observed": [observed]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeWidth=3)
        .encode(x="observed:Q")
    )
    return (hist + rule).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            "Other Primaries share these days about as often as shifted copies of their days do",
            subtitle=lines(
                text="Blue bars: the share when each other Primary's worst days are moved by a "
                "random number of days, 1,000 times. Orange line: the real share. Moving all "
                "the other Primaries together gives a lower average (35.5%)."
            ),
        ),
    )


def main() -> None:
    """Draw and save the three figures."""
    save(chart=split_chart(), name="fig1_split")
    save(chart=day_chart(), name="fig2_days")
    save(chart=chance_chart(), name="fig3_chance")


if __name__ == "__main__":
    main()
