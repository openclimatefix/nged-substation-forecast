"""Draw the figures of experiment 180's write-up from `work/scores_*.json` and `work/stage0.json`.

uv run --no-sync python ml-experiments/00180-*/e00180_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00180_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600
ARM_LABELS: Final[dict[str, str]] = {
    "c": "Control: fleet shape",
    "p": "P: chosen tilt and azimuth",
    "k60": "K: export clip at 60%",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "c": ocf.DATA_BLUE,
    "p": ocf.DATA_SKY,
    "k60": ocf.BRAND_ORANGE,
}


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG beside the page and as PNG in the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def interval_chart(
    *,
    frame: pl.DataFrame,
    y: str,
    colour: str,
    x_title: str,
    title: list[str],
    zero_label: str,
    height: int,
) -> alt.LayerChart:
    """Return dots with interval rules, one row per `y`, and a labelled zero rule."""
    base = alt.Chart(frame).encode(
        y=alt.Y(f"{y}:N", sort=None, title=None),
        color=alt.Color(f"{colour}:N", scale=None, legend=None),
    )
    rules = base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title=x_title), x2="high:Q")  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=70, opacity=1.0).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0], "t": [zero_label]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")
    )
    label = (
        alt.Chart(pl.DataFrame({"x": [0.0], "t": [zero_label]}))  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=4, dy=-6, baseline="bottom", fontSize=10, color="#444444")
        .encode(x="x:Q", y=alt.value(0), text="t:N")
    )
    return (rules + dots + zero + label).properties(width=WIDTH, height=height, title=title)


def headline(*, scored: dict) -> alt.LayerChart:
    """Figure 1: D5's sunny-daylight gap to BG2 by arm."""
    rows = []
    for arm in ("c", "p", "k60"):
        mean, low, high = scored[f"{arm}|D5|sunny"]["gap_to_bg2"]
        rows.append(
            {
                "arm": ARM_LABELS[arm],
                "colour": ARM_COLOURS[arm],
                "mean": mean,
                "low": low,
                "high": high,
            }
        )
    return interval_chart(
        frame=pl.DataFrame(rows),
        y="arm",
        colour="colour",
        x_title="Mean absolute error minus BG2's (MW; smaller is better)",
        title=[
            "D5, sunniest third of daylight half-hours, all leads to 168 hours, scored year.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
        zero_label="same as BG2",
        height=130,
    )


def no_harm(*, scored: dict) -> alt.LayerChart:
    """Figure 2: each arm's change against the control, by substation."""
    rows = []
    for arm, sites in (("p", common.SCORED_SITES), ("k60", ("D5", "D10", "D16"))):
        for site in sites:
            mean, low, high = scored[f"{arm}|{site}|sunny"]["mae_minus_control"]
            rows.append(
                {
                    "row": f"{ARM_LABELS[arm].split(':')[0]} at {site}",
                    "colour": ARM_COLOURS[arm],
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    return interval_chart(
        frame=pl.DataFrame(rows),
        y="row",
        colour="colour",
        x_title="Mean absolute error minus the control's (MW; more negative is better)",
        title=[
            "Sunniest third of daylight half-hours, all leads, scored year, by substation.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
        zero_label="same as the control",
        height=300,
    )


def clip_sweep(*, tuning: dict, replicate: dict) -> alt.LayerChart:
    """Figure 3: the clip's share of held capacity against the change in D5's sunny error."""
    rows = []
    for period, scores in (("Tuning months", tuning), ("Replicate months", replicate)):
        for share in (50, 60, 70, 80, 90):
            mean, low, high = scores[f"k{share}|D5|sunny"]["mae_minus_control"]
            rows.append({"period": period, "share": share, "mean": mean, "low": low, "high": high})
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "share:Q",
            title="Clip ceiling as a share of D5's held capacity (%)",
            scale=alt.Scale(domain=[45, 95]),
            axis=alt.Axis(values=[50, 60, 70, 80, 90]),
        ),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(
                domain=["Tuning months", "Replicate months"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        xOffset="period:N",
        y=alt.Y("low:Q", title="D5 sunny error minus the control's (MW; more negative is better)"),
        y2="high:Q",
    )
    dots = base.mark_point(filled=True, size=70, opacity=1.0).encode(xOffset="period:N", y="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return (rules + dots + zero).properties(
        width=WIDTH,
        height=240,
        title=[
            "Clip ceiling chosen on the tuning months (60%), before the scored year was read.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
    )


def stage0(*, stored: dict) -> alt.LayerChart:
    """Figure 4: net power lost per unit of fleet shape, by decile of the shape, per substation."""
    rows = [
        {"site": site, "band": i + 1, "points": value}
        for site, values in stored.items()
        for i, value in enumerate(values)
        if site != "D9"
    ]
    colours = {
        "D5": ocf.BRAND_ORANGE,
        "D10": ocf.DATA_BLUE,
        "D11": ocf.DATA_SKY,
        "D14": ocf.DATA_PURPLE,
        "D16": ocf.DATA_GREEN,
    }
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, clip=True)
        .encode(
            x=alt.X(
                "band:Q",
                title="Band of the fleet solar shape (1 lowest, 8 highest)",
                scale=alt.Scale(domain=[2.5, 8.5]),
                axis=alt.Axis(values=[3, 4, 5, 6, 7, 8]),
            ),
            y=alt.Y(
                "points:Q",
                title="Fall below night-time power per unit shape (points)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    return chart.properties(
        width=WIDTH,
        height=240,
        title=[
            "Clear half-hours (clear-sky index 0.9 or more), May to August, 2019 to 2026.",
            "Bands 1 and 2 (lowest sun) are off the axis; the baseline is least reliable there.",
        ],
    )


def main() -> None:
    """Draw the figures."""
    scored = json.loads((common.WORK_DIR / "scores_scored.json").read_text())
    tuning = json.loads((common.WORK_DIR / "scores_tuning.json").read_text())
    replicate = json.loads((common.WORK_DIR / "scores_replicate.json").read_text())
    stored = json.loads((common.WORK_DIR / "stage0.json").read_text())
    save(chart=headline(scored=scored), name="figure_01_headline")
    save(chart=no_harm(scored=scored), name="figure_02_no_harm")
    save(chart=clip_sweep(tuning=tuning, replicate=replicate), name="figure_03_clip_share")
    save(chart=stage0(stored=stored), name="figure_04_implied_solar_by_shape_decile")


if __name__ == "__main__":
    main()
