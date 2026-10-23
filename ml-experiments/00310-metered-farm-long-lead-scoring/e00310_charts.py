# ruff: noqa: E501
"""Draw experiment 310's figures from the score tables (farm series are never drawn; farms pooled)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00310_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 120
ARM_LABELS: Final[dict[str, str]] = {
    "floor_climatology": "climatology",
    "floor_heuristic": "manual_heuristic",
    "p2": "Plant alone (P2)",
    "gam": "GAM alone",
    "mdn_ens": "MDN alone",
    "xgb": "XGBoost alone",
    "blend": "Blend of GAM, MDN, XGBoost",
}
ARM_COLOURS: Final[list[str]] = [
    "#888888",
    "#444444",
    ocf.DATA_SKY,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    ocf.BRAND_ORANGE,
]
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}


def _title(lines: list[str]) -> alt.TitleParams:
    lines = [part for line in lines for part in textwrap.wrap(line, 110)]
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _by_day(*, scores: pl.DataFrame, metric: str) -> pl.DataFrame:
    return scores.filter(
        (pl.col("slice_kind") == "day")
        & (pl.col("metric") == metric)
        & pl.col("arm").is_in(list(ARM_LABELS))
    ).with_columns(
        day=pl.col("slice").cast(pl.Int64),
        label=pl.col("arm").replace_strict(ARM_LABELS),
    )


def figure_1(*, scores: pl.DataFrame) -> None:
    """Quantile score against lead (day), every arm, both settings."""
    panels = []
    for setting, label in SETTING_LABELS.items():
        data = _by_day(scores=scores, metric="score").filter(pl.col("setting") == setting)
        base: Any = alt.Chart(data)
        colour = alt.Color(
            "label:N",
            sort=list(ARM_LABELS.values()),
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=4),
        )
        x = alt.X(
            "day:Q",
            title="Lead (days ahead of the 00 UTC run)",
            axis=alt.Axis(values=list(range(1, 11))),
        )
        line = base.mark_line(point=True).encode(
            x=x,
            y=alt.Y(
                "value:Q",
                title="Quantile score (points of capacity; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=colour,
        )
        panels.append(line.properties(width=PLOT_WIDTH, height=230, title=_title([label])))
    alt.vconcat(*panels).properties(
        title=_title(
            [
                "Figure 1: quantile score against lead, farms A to F pooled, daylight, runs from 2025-09 to 2026-08, quarterly refits.",
                "Lower is better. The arms share the same rows, so the intervals of their differences (Figure 2) are far narrower than the spread between months.",
            ]
        )
    ).save(DOCS_DIR / "figure_1_score_by_lead.svg")


def figure_2(*, contrasts: pl.DataFrame) -> None:
    """The blend's and each single family's change against climatology, by day."""
    wanted = ["blend", "gam", "mdn_ens", "xgb"]
    data = contrasts.filter(
        (pl.col("slice_kind") == "day")
        & (pl.col("metric") == "score")
        & (pl.col("reference") == "floor_climatology")
        & pl.col("arm").is_in(wanted)
    ).with_columns(
        day=pl.col("slice").cast(pl.Int64),
        label=pl.col("arm").replace_strict(ARM_LABELS),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )
    base: Any = alt.Chart(data)
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in wanted],
            range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
        ),
        legend=alt.Legend(title=None, orient="top", columns=4),
    )
    x = alt.X(
        "day:Q",
        title="Lead (days ahead of the 00 UTC run)",
        axis=alt.Axis(values=list(range(1, 11))),
    )
    zero: Any = base.mark_rule(strokeDash=[4, 3], color="#444").encode(y=alt.datum(0))
    layer = (
        base.mark_area(opacity=0.10).encode(x=x, y="lo:Q", y2="hi:Q", color=colour)
        + base.mark_line(point=True).encode(
            x=x,
            y=alt.Y("percent_change:Q", title="Change vs climatology (%)"),
            color=colour,
        )
        + zero
    )
    layer.properties(width=PLOT_WIDTH, height=230).facet(
        row=alt.Row("setting_label:N", title=None),
    ).properties(
        title=_title(
            [
                "Figure 2: change in quantile score against climatology by lead. The dashed line is climatology (no change).",
                "More negative is better. Line: change. Shade: 98.75% interval from resampling the 12 months of the run. Farms A to F pooled.",
            ]
        )
    ).save(DOCS_DIR / "figure_2_margin_over_climatology.svg")


def figure_3(*, scores: pl.DataFrame) -> None:
    """Coverage at the four tail levels against lead, the blend and XGBoost."""
    rows = []
    for level, nominal in (("01", 0.01), ("05", 0.05), ("95", 0.95), ("99", 0.99)):
        part = _by_day(scores=scores, metric=f"cover{level}").filter(
            pl.col("arm").is_in(["blend", "xgb", "floor_climatology"])
        )
        rows.append(part.with_columns(level=pl.lit(f"{nominal}"), nominal=pl.lit(nominal)))
    data = pl.concat(rows).with_columns(
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS)
    )
    base: Any = alt.Chart(data)
    x = alt.X("day:Q", title="Lead (days ahead)", axis=alt.Axis(values=list(range(1, 11))))
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in ("blend", "xgb", "floor_climatology")],
            range=[ocf.BRAND_ORANGE, ocf.DATA_GREEN, "#888888"],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    rule: Any = base.mark_rule(strokeDash=[4, 3], color="#444").encode(y="nominal:Q")
    line = base.mark_line(point=True).encode(
        x=x,
        y=alt.Y("value:Q", title="Share at or below", scale=alt.Scale(zero=False)),
        color=colour,
    )
    (rule + line).properties(width=PLOT_WIDTH // 4 - 45, height=170).facet(
        column=alt.Column("level:N", title="Nominal quantile (dashed line)"),
        row=alt.Row("setting_label:N", title=None),
    ).resolve_scale(y="independent").properties(
        title=_title(
            ["Figure 3: tail coverage against lead. A line on the dashed rule is calibrated."]
        )
    ).save(DOCS_DIR / "figure_3_tail_coverage.svg")


def figure_4(*, contrasts: pl.DataFrame) -> None:
    """The blend's change against climatology per farm and band."""
    data = contrasts.filter(
        (pl.col("slice_kind") == "farm")
        & (pl.col("metric") == "score")
        & (pl.col("arm") == "blend")
        & (pl.col("reference") == "floor_climatology")
    ).with_columns(setting_label=pl.col("setting").replace_strict(SETTING_LABELS))
    base: Any = alt.Chart(data)
    y = alt.Y("band:N", sort=list(common.BAND_ORDER), title="Day band")
    zero: Any = base.mark_rule(strokeDash=[4, 3], color="#444").encode(x=alt.datum(0))
    layer = (
        base.mark_point(filled=True, size=60).encode(
            x=alt.X("percent_change:Q", title="Change vs climatology (%)"),
            y=y,
            color=alt.Color("slice:N", title="Farm", legend=alt.Legend(orient="top", columns=6)),
        )
        + zero
    )
    layer.properties(width=PLOT_WIDTH, height=180).facet(
        row=alt.Row("setting_label:N", title=None)
    ).properties(
        title=_title(
            [
                "Figure 4: the blend's change in quantile score against climatology, by farm and day band."
            ]
        )
    ).save(DOCS_DIR / "figure_4_per_farm.svg")


def main() -> None:
    """Draw every figure."""
    scores = pl.read_csv(common.DATA_DIR / "scores.csv")
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_1(scores=scores)
    figure_2(contrasts=contrasts)
    figure_3(scores=scores)
    figure_4(contrasts=contrasts)


if __name__ == "__main__":
    main()
