"""Draw the figures of experiment 161's write-up from `work/report.json`.

uv run --no-sync python ml-experiments/00161-*/e00161_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00161_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
REPORT_PATH: Final[Path] = common.WORK_DIR / "report.json"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
BSP_COLOURS: Final[dict[str, str]] = {"D13": ocf.DATA_BLUE, "D17": ocf.BRAND_ORANGE}
ARM_LABELS: Final[dict[str, str]] = {
    "td": "Top-down",
    "bu_r": "Bottom-up, learned weights",
    "bu_s": "Bottom-up, weights fixed at 1",
    "bu_f": "Bottom-up, unbounded weights",
    "bu_r_tdspread": "Bottom-up, top-down spread table",
    "bu_oracle": "Oracle: observed children",
}
BANDS: Final[tuple[str, ...]] = ("0.5 to 24 h", "24 to 72 h", "72 to 168 h")


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """Bottom-up minus top-down quantile score, with 97.5% intervals, at each BSP."""
    names = {
        "C1": "Learned weights (headline)",
        "C2": "Weights fixed at 1",
        "C5": "Learned weights, top-down spread table",
        "C7": "Oracle: observed children",
    }
    rows = []
    for period, label in (("replicate", "even months (replicate)"), ("scored", "scored year")):
        for key, name in names.items():
            for bsp in common.BSPS:
                c = data["periods"][period][key][bsp]
                rows.append(
                    {
                        "arm": name,
                        "bsp": bsp,
                        "period": label,
                        "estimate": c["estimate"],
                        "low": c["lower"],
                        "high": c["upper"],
                    }
                )
    frame = pl.DataFrame(rows).filter(pl.col("period") == "scored year")
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("bsp:N", sort=list(BSP_COLOURS)),
        color=alt.Color(
            "bsp:N",
            scale=alt.Scale(domain=list(BSP_COLOURS), range=list(BSP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Quantile score minus top-down (points; below zero favours bottom-up)",
        )
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 260,
        height=220,
        title=_title(
            text="Bottom-up matches top-down at both BSPs; only observed children help",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours in the scored "
                "year, outside switching events, in points (percent of the BSP's effective "
                "capacity). Line: 97.5% interval from resampling whole calendar months. Zero marks "
                "the same score as the top-down forecast."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Quantile score by lead band for the top-down, bottom-up, and oracle arms."""
    arms = ("td", "bu_r", "bu_oracle")
    labels = [ARM_LABELS[a] for a in arms]
    rows = [
        {
            "bsp": bsp,
            "band": band,
            "arm": ARM_LABELS[arm],
            "score": data["periods"]["scored"]["arms"][bsp][arm][f"qs {band}"],
        }
        for bsp in common.BSPS
        for band in BANDS
        for arm in arms
    ]
    arm_colours = [GREY, ocf.DATA_BLUE, ocf.DATA_GREEN]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("score:Q", title="Quantile score (points; lower is better)"),
            color=alt.Color(
                "arm:N", scale=alt.Scale(domain=labels, range=arm_colours), legend=None
            ),
            row=alt.Row("band:N", sort=list(BANDS), title="Lead band"),
            column=alt.Column("bsp:N", sort=list(common.BSPS), title=None),
        )
        .properties(
            width=(WIDTH - 280) // 2,
            height=70,
            title=_title(
                text="Bottom-up equals top-down at every lead; the oracle gains at every lead",
                subtitle=(
                    "Scored year, outside switching events. Points are percentages of the "
                    "BSP's effective capacity."
                ),
            ),
        )
    )


def figure_decomposition(*, data: dict) -> alt.TopLevelMixin:
    """Standard deviation of the forecast error of each bottom-up part and of top-down."""
    names = {
        "sd_top_down_error": "Top-down BSP error",
        "sd_bottom_up_error": "Bottom-up BSP error",
        "sd_remainder_error": "Remainder error",
        "sd_children_error": "Summed children error",
    }
    rows = [
        {"bsp": bsp, "part": name, "sd": data["error_decomposition"][bsp][key]}
        for bsp in common.BSPS
        for key, name in names.items()
    ]
    labels = list(names.values())
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("part:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("sd:Q", title="Standard deviation of forecast error (points; lower is better)"),
            color=alt.Color(
                "bsp:N",
                scale=alt.Scale(domain=list(BSP_COLOURS), range=list(BSP_COLOURS.values())),
                legend=None,
            ),
            row=alt.Row("bsp:N", sort=list(common.BSPS), title=None),
        )
        .properties(
            width=WIDTH - 260,
            height=110,
            title=_title(
                text="Most of each BSP's error sits in the load the children do not cover",
                subtitle=(
                    "Scored year, outside switching events. Points are percentages of the BSP's "
                    "effective capacity. The remainder is the BSP's reading minus the weighted "
                    "sum of the covered children's readings."
                ),
            ),
        )
    )


def main() -> None:
    """Write every figure."""
    data = json.loads(REPORT_PATH.read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_by_lead(data=data), name="figure_2_by_lead")
    _save(figure_decomposition(data=data), name="figure_3_decomposition")


if __name__ == "__main__":
    main()
