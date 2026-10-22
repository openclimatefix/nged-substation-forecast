"""Draw the figures of experiment 204's write-up from `work/report.json`.

uv run --no-sync python ml-experiments/00161-*/e00204_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00204_common as common
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
    "a_td": "(a) Top-down",
    "b_partial_tdspread": "(b) Partial bottom-up, MW children only",
    "c": "(c) Full, apparent power read as signed",
    "d": "(d) Full, fold-aware signed flow",
    "dp50": "(d-p50) Full, signed flow at probability 0.5",
    "oracle": "Oracle: observed children",
}
EXTRA_PATH: Final[Path] = common.WORK_DIR / "extra.json"
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
    """Each full arm minus top-down, with 97.5% intervals, at each BSP."""
    names = {
        "b_tdspread_minus_a": ARM_LABELS["b_partial_tdspread"],
        "C1_c_minus_a": ARM_LABELS["c"],
        "C1_d_minus_a": ARM_LABELS["d"],
        "C7_oracle_minus_a": ARM_LABELS["oracle"],
    }
    rows = [
        {
            "arm": name,
            "bsp": bsp,
            "estimate": data["scored"][bsp]["contrasts"][key]["estimate"],
            "low": data["scored"][bsp]["contrasts"][key]["lower"],
            "high": data["scored"][bsp]["contrasts"][key]["upper"],
        }
        for key, name in names.items()
        for bsp in common.BSPS
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
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
            text="Adding every child to the sum does not beat top-down; only observed children do",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours in the scored "
                "year, in points (percent of the BSP's effective capacity). Line: 97.5% interval "
                "from resampling whole calendar months. All bottom-up arms use the top-down "
                "spread table."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Quantile score by lead band for the top-down, full, and oracle arms."""
    arms = ("a_td", "b_partial_tdspread", "d", "oracle")
    labels = [ARM_LABELS[a] for a in arms]
    rows = [
        {
            "bsp": bsp,
            "band": band,
            "arm": ARM_LABELS[arm],
            "score": data["scored"][bsp]["arms"][arm][f"qs {band}"],
        }
        for bsp in common.BSPS
        for band in BANDS
        for arm in arms
    ]
    arm_colours = [GREY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
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
                text="Full bottom-up equals top-down at every lead; the oracle gains at every lead",
                subtitle="Scored year, outside switching events. Points are percentages of the "
                "BSP's effective capacity.",
            ),
        )
    )


def figure_decomposition(*, extra: dict) -> alt.TopLevelMixin:
    """Standard deviation of the forecast error of each bottom-up part and of top-down."""
    names = {
        "sd_top_down_error": "Top-down BSP error",
        "sd_bottom_up_error": "Full bottom-up BSP error",
        "sd_remainder_error": "Remainder error",
        "sd_children_error": "Summed children error",
    }
    rows = [
        {"bsp": bsp, "part": name, "sd": extra["decomposition"][bsp][key]}
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
                text="With every child in the sum, the remainder still holds most error",
                subtitle=(
                    "Scored year, outside switching events. Points are percentages of the BSP's "
                    "effective capacity. The remainder is the BSP's reading minus the weighted "
                    "sum of the children's readings (arm (d))."
                ),
            ),
        )
    )


def figure_known_answer(*, extra: dict) -> alt.TopLevelMixin:
    """Mean absolute error of the summed signed flow of five Primaries with injected solar."""
    rows = []
    for key, label in (
        ("L0.0_q0", "No injection"),
        ("L1.0_pf97", "Farm at 1.0 x midday median"),
        ("L2.0_pf97", "Farm at 2.0 x midday median"),
    ):
        for arm, name in (("mae_c_mw", "(c) Read as signed"), ("mae_d_mw", "(d) Fold-aware flow")):
            rows.append({"case": label, "arm": name, "mae": extra["known_answer"][key][arm]})
    labels = ["(c) Read as signed", "(d) Fold-aware flow"]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=labels, title=None),
            x=alt.X(
                "mae:Q",
                title="Mean absolute error of the summed signed flow (MW)",
                axis=alt.Axis(tickCount=7),
            ),
            color=alt.Color(
                "arm:N", scale=alt.Scale(domain=labels, range=[GREY, ocf.BRAND_ORANGE]), legend=None
            ),
            row=alt.Row(
                "case:N",
                sort=[r["case"] for r in rows[::2]],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH - 260,
            height=50,
            title=_title(
                text="The fold-aware flow recovers the signed sum when reverse flow is large",
                subtitle=(
                    "Five Primaries with a metered solar farm subtracted, folded to apparent power "
                    "at power factor 0.97, scored year, experiment 152's stored forecasts."
                ),
            ),
        )
    )


def main() -> None:
    """Write every figure."""
    data = json.loads(REPORT_PATH.read_text())
    extra = json.loads(EXTRA_PATH.read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_by_lead(data=data), name="figure_2_by_lead")
    _save(figure_decomposition(extra=extra), name="figure_3_decomposition")
    _save(figure_known_answer(extra=extra), name="figure_4_known_answer")


if __name__ == "__main__":
    main()
