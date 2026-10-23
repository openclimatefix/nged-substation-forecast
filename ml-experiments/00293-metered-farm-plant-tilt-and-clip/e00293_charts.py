# ruff: noqa: E501
"""Draw experiment 293's figures from the score tables (farms are pooled; no farm series are drawn)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00293_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 330
ARM_LABELS: Final[dict[str, str]] = {
    "p2": "P2: tilt 35, hard cap at 0.995 quantile",
    "a_tilt20": "(a) tilt 20",
    "b_soft_clip": "(b) soft cap at 0.999 quantile",
    "c_tilt20_soft_clip": "(c) tilt 20 and soft cap",
    "d_pooled_shape": "(d) pooled leave-one-farm-out shape",
    "e_hard_cap_999": "(e) hard cap at 0.999 quantile",
    "f_tilt20_hard_cap_999": "(f) tilt 20, hard cap at 0.999",
    "ref_gam": "Experiment 287 GAM",
    "ref_mdn": "Experiment 287 MDN",
    "ref_blend": "Experiment 287 blend",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
MONTHS: Final[list[str]] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _title(lines: list[str]) -> alt.TitleParams:
    lines = [part for line in lines for part in textwrap.wrap(line, 100)]
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _dots(
    *, data: pl.DataFrame, x_title: str, sort: list[str], height: int, title: list[str], rule: bool
) -> alt.Chart:
    base: Any = alt.Chart(data)
    colour = alt.Color(
        "setting_label:N",
        scale=alt.Scale(domain=list(SETTING_LABELS.values()), range=SETTING_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("label:N", sort=sort, title=None, axis=alt.Axis(labelLimit=300))
    chart = base.mark_rule(size=2).encode(
        x=alt.X("lo:Q", title=x_title), x2="hi:Q", y=y, color=colour, yOffset="setting_label:N"
    ) + base.mark_point(filled=True, size=50).encode(
        x="value:Q", y=y, color=colour, yOffset="setting_label:N"
    )
    if rule:
        zero: Any = alt.Chart(pl.DataFrame({"zero": [0.0]}))
        chart += zero.mark_rule(strokeDash=[4, 3], color="#444").encode(x="zero:Q")
    return chart.properties(width=PLOT_WIDTH, height=height, title=_title(title))


def _label(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS, default=pl.col("arm")),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )


def main() -> None:
    """Write the four figures."""
    scores = pl.read_csv(common.DATA_DIR / "scores.csv")
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    order = list(ARM_LABELS.values())
    data = _label(
        scores.filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
    )
    _dots(
        data=data,
        x_title="Quantile score (points of capacity; lower is better)",
        sort=order,
        height=360,
        rule=False,
        title=[
            "Figure 1: quantile score of every arm, farms A to F, 12 to 36 h, daylight, 2025-09 to 2026-08.",
            "Dot: score. Line: 98.75% interval from resampling the 12 scored months.",
        ],
    ).save(DOCS_DIR / "figure_1_scores.svg")
    arms = [a for a in ARM_LABELS if not a.startswith("ref_") and a != "p2"]
    for metric, name, text in (
        ("score", "figure_2_score_contrasts.svg", "Figure 2: change in quantile score against P2"),
        ("pin99", "figure_3_q99_contrasts.svg", "Figure 3: change in 0.99 pinball loss against P2"),
    ):
        rows = _label(
            contrasts.filter(
                (pl.col("slice") == "all")
                & (pl.col("metric") == metric)
                & (pl.col("reference") == "p2")
                & pl.col("arm").is_in(arms)
            ).with_columns(value=pl.col("percent_change"))
        )
        _dots(
            data=rows,
            x_title=f"Change in {'quantile score' if metric == 'score' else '0.99 pinball loss'} against P2 (%; more negative is better)",
            sort=[ARM_LABELS[a] for a in arms],
            height=300,
            rule=True,
            title=[
                f"{text}. Dashed line: same as P2. Dot: change. Line: 98.75% interval from resampling the 12 scored months.",
            ],
        ).save(DOCS_DIR / name)
    monthly = pl.read_csv(common.DATA_DIR / "bias_by_month.csv").filter(
        pl.col("arm").is_in(["p2", "a_tilt20", "b_soft_clip", "c_tilt20_soft_clip"])
    )
    pooled = (
        monthly.group_by("arm", "month")
        .agg(actual=pl.col("actual").sum(), forecast=pl.col("forecast").sum())
        .with_columns(
            bias=100.0 * (pl.col("actual") / pl.col("forecast") - 1.0),
            label=pl.col("arm").replace_strict(ARM_LABELS),
            month_name=pl.col("month").map_elements(
                lambda m: MONTHS[m - 1], return_dtype=pl.String
            ),
        )
    )
    base: Any = alt.Chart(pooled)
    chart: Any = base.mark_line(point=True).encode(
        x=alt.X("month_name:N", sort=MONTHS, title="Calendar month of the scored year"),
        y=alt.Y("bias:Q", title="Actual over mean forecast minus 1 (%; 0 is unbiased)"),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=[
                    ARM_LABELS[a] for a in ("p2", "a_tilt20", "b_soft_clip", "c_tilt20_soft_clip")
                ],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_PURPLE],
            ),
            legend=alt.Legend(
                title=None, orient="top", columns=2, labelLimit=400, symbolType="stroke"
            ),
        ),
    )
    zero_chart: Any = alt.Chart(pl.DataFrame({"zero": [0.0]}))
    chart += zero_chart.mark_rule(strokeDash=[4, 3], color="#444").encode(y="zero:Q")
    chart.properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            [
                "Figure 4: bias of the mean forecast by calendar month, own history, clear half-hours (clear-sky index 0.9 or more), farms A to F pooled.",
                "Positive: the plant under-predicts. Dashed line: unbiased.",
            ]
        ),
    ).save(DOCS_DIR / "figure_4_monthly_bias.svg")


if __name__ == "__main__":
    main()
