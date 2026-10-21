"""Draw the figures of experiment 121 as SVG into the docs folder.

Only Primary substations appear, under their D-labels. The numbers come from `work/numbers.json`
(written by `e00121_report.py`), `work/nightstep.json` (`e00121_nightstep.py`), and the gates'
stored files, so a figure and the page quote the same numbers. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00121-*/e00121_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00121_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 480
WRAP: Final[int] = 90
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]


def _numbers() -> dict[str, list[dict[str, Any]]]:
    return json.loads((common.WORK_DIR / "numbers.json").read_text())["tables"]


def _table(*, start: str) -> pl.DataFrame:
    numbers = _numbers()
    key = next(k for k in numbers if k.startswith(start))
    return pl.DataFrame(numbers[key], infer_schema_length=None)


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
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


def figure_1_headline() -> alt.TopLevelMixin:
    """The heating contrasts and the night contrasts, with intervals and the replicate year."""
    mh = _table(start="MH contrasts")
    night = _table(start="Night contrasts on winter nights")
    evolving = _table(start="Evolving against fixed")
    mh_labels = {
        "C1: MH-K - B0-K": "Heating: five hinges (MH), all rows",
        "MH-5-K - B0-K": "Heating: five hinges only (MH-5)",
        "MH-sh-K - B0-K": "Heating: five hinges, shrunk (MH-sh)",
    }
    night_labels = {
        "N1: NS-L-K - B0-K": "Night: local-clock levels (NS-L, null control)",
        "N2: NS-U-K - B0-K": "Night: UTC-clock levels (NS-U)",
        "N3: NS-H-K - B0-K": "Night: cold hinge, one window (NS-H)",
        "N4: NS-HW-K - B0-K": "Night: cold hinge by half-hour (NS-HW)",
    }
    parts = [
        mh.filter(pl.col("contrast").is_in(list(mh_labels)))
        .with_columns(
            label=pl.col("contrast").replace_strict(mh_labels),
            level=pl.when(pl.col("contrast").str.starts_with("C1"))
            .then(pl.lit("97.5% interval, all rows"))
            .otherwise(pl.lit("95% interval, all rows, exploratory")),
        )
        .select("label", "level", "scored QS", "low", "high", "replicate QS"),
        night.with_columns(
            label=pl.col("contrast").replace_strict(night_labels),
            level=pl.lit("99.375% interval, winter nights"),
        ).select("label", "level", "scored QS", "low", "high", "replicate QS"),
        evolving.with_columns(
            label=pl.lit("Night: evolving against fixed UTC step"),
            level=pl.lit("95% interval, winter nights, exploratory"),
        ).select("label", "level", "scored QS", "low", "high", "replicate QS"),
    ]
    rows = pl.concat(parts)
    order = rows["label"].to_list()
    domain = [
        "97.5% interval, all rows",
        "95% interval, all rows, exploratory",
        "99.375% interval, winter nights",
        "95% interval, winter nights, exploratory",
    ]
    base = alt.Chart(rows)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    rules = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "low:Q",
            title="Quantile score minus B0-K's (points of capacity; more negative is better)",
            scale=alt.Scale(domain=[-0.3, 0.35]),
        ),
        x2="high:Q",
        color=alt.Color(
            "level:N",
            scale=alt.Scale(
                domain=domain,
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE],
            ),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
        ),
    )
    dots = base.mark_point(filled=True, size=70, color=DARK).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="scored QS:Q"
    )
    replicate = base.mark_point(filled=False, size=70, color=DARK, shape="diamond").encode(  # ty: ignore[unresolved-attribute]
        y=y, x="replicate QS:Q"
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (
        (rules + dots + replicate + zero)
        .properties(
            width=WIDTH,
            height=34 * len(order),
            title=_title(
                text="A UTC-clock night term lowers winter-night error; five hinges do not",
                subtitle=(
                    "Scored year 2025-07 to 2026-06, day ahead, after the filter, five decision "
                    "substations, past weather. Dot: estimate. Line: interval from resampling "
                    "whole calendar months. Diamond: replicate year 2024-25. Solid rule: no "
                    "difference from B0-K. Night labels are 'not powered' (figure 5)."
                ),
            ),
        )
        .configure_axis(labelFontSize=11)
    )


def figure_2_cold_nights() -> alt.TopLevelMixin:
    """Median forecast and observed power on cold winter nights, B0-K against two night arms."""
    data = json.loads((common.WORK_DIR / "nightstep.json").read_text())
    arms = {
        "b0_k": "B0-K (no night term)",
        "ns_u_k": "NS-U-K (UTC-clock levels)",
        "ns_hw_k": "NS-HW-K (cold hinge)",
    }
    rows = []
    for arm, name in arms.items():
        for site, profile in data[f"scored_{arm}"]["profiles"].items():
            for position, row in enumerate(profile):
                rows.append(
                    {
                        "site": site,
                        "series": name,
                        "hour": 22.0 + 0.5 * position,
                        "forecast": row["q50"],
                        "low": row["q10"],
                        "high": row["q90"],
                        "observed": row["actual"],
                    }
                )
    frame = pl.DataFrame(rows)
    colours = alt.Scale(
        domain=list(arms.values()),
        range=[GREY, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
    )
    x = alt.X("hour:Q", title="Hour (local; 24 is midnight)")
    lines = (
        alt.Chart()
        .mark_line(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "forecast:Q",
                title="Median power (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "series:N",
                scale=colours,
                legend=alt.Legend(title=None, orient="top", labelLimit=320),
            ),
        )
    )
    truth = (
        alt.Chart()
        .transform_filter(alt.datum.series == "B0-K (no night term)")
        .mark_line(strokeWidth=2.5, color=DARK, strokeDash=[5, 3])
        .encode(x=x, y=alt.Y("observed:Q", scale=alt.Scale(zero=False)))  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(lines, truth)
        .properties(width=170, height=130)
        .facet(facet=alt.Facet("site:N", sort=SITE_ORDER, title=None), data=frame, columns=3)
        .properties(
            title=_title(
                text="The UTC-clock night term reproduces the 01:00 step on cold winter nights",
                subtitle=(
                    "Scored winter 2025-26, weekday nights 22:00 to 08:00 with a daily mean "
                    "temperature below 8 C. Dashed black line: observed power. Coloured lines: "
                    "median day-ahead forecast after the filter. Past weather. The page's table "
                    "gives the step in points and MW."
                ),
            )
        )
        .resolve_scale(y="independent")
    )


def figure_3_winter_breakdown() -> alt.TopLevelMixin:
    """Each contrast's scored-year difference on every row subset."""
    frame = _table(start="Winter breakdown: scored")
    subsets = [
        "all rows",
        "Nov-Mar",
        "Apr-Oct",
        "winter nights",
        "other than winter nights",
        "cold snaps",
        "winter evening peaks",
        "BST nights",
        "cold BST nights",
    ]
    long = frame.select("contrast", *subsets).unpivot(
        index="contrast", variable_name="rows", value_name="difference"
    )
    long = long.with_columns(
        label=pl.col("contrast").str.slice(0, 34),
        text=pl.col("difference").round(3).cast(pl.String),
    )
    base = alt.Chart(long).encode(
        x=alt.X("rows:N", sort=subsets, title=None, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("label:N", title=None),
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "difference:Q",
            scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
            legend=alt.Legend(title="Score minus B0-K"),
        )
    )
    labels = base.mark_text(fontSize=10).encode(text="text:N")  # ty: ignore[unresolved-attribute]
    return (heat + labels).properties(
        width=WIDTH,
        height=30 * frame.height,
        title=_title(
            text="The night arms' gains sit on winter nights, with no loss elsewhere",
            subtitle=(
                "Scored year, day ahead, after the filter, mean quantile-score difference from "
                "B0-K in points of capacity on each set of rows (blue is better). Cold snaps are "
                "the coldest 10% of days of the winter."
            ),
        ),
    )


def figure_4_gate2() -> alt.TopLevelMixin:
    """MH-K minus B0-K on synthetic substations under two truths."""
    gate = json.loads((common.WORK_DIR / "gate2.json").read_text())
    rows = pl.DataFrame(
        [
            {
                "truth": "Five-hinge truth (amplitude set for a 0.03-point oracle gain)",
                "mean": gate["five"]["mean_mh_k_minus_b0_k"],
                "low": gate["five"]["interval_80"][0],
                "high": gate["five"]["interval_80"][1],
            },
            {
                "truth": "Single-hinge truth (B0 is exactly right)",
                "mean": gate["single"]["mean_mh_k_minus_b0_k"],
                "low": gate["single"]["interval_80"][0],
                "high": gate["single"]["interval_80"][1],
            },
        ]
    )
    base = alt.Chart(rows)
    y = alt.Y("truth:N", title=None, axis=alt.Axis(labelLimit=330))
    rule = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "low:Q",
            title="MH-K minus B0-K on 100 synthetic substations (points; negative is better)",
        ),
        x2="high:Q",
    )
    dot = base.mark_point(filled=True, size=80, color=DARK).encode(y=y, x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (rule + dot + zero).properties(
        width=WIDTH,
        height=90,
        title=_title(
            text="The five-hinge model finds a five-hinge curve and costs little when none exists",
            subtitle=(
                "Gate 2: 100 synthetic substations (20 per decision substation) with real noise, "
                "through the filter. Dot: mean over draws. Line: 80% interval over draws."
            ),
        ),
    )


def figure_5_gate4() -> alt.TopLevelMixin:
    """Detection rates of a 4-point and a 7-point night step at a 5% false-positive rate."""
    gate = json.loads((common.WORK_DIR / "gate4b.json").read_text())
    names = {"ns_u": "NS-U (UTC step)", "ns_hw": "NS-HW (cold step)", "ns_h": "NS-H (cold step)"}
    keys = {"ns_u": "utc", "ns_hw": "cold", "ns_h": "cold"}
    rows = []
    for arm, name in names.items():
        for size in (4, 7):
            for rows_name, nice in (
                ("label_rows", "Night rows the label uses"),
                ("injected_half_hour_rows", "Rows of the step's half-hour only"),
            ):
                rows.append(
                    {
                        "arm": name,
                        "step": f"{size}-point step",
                        "rows": nice,
                        "detected": gate["arms"][arm][rows_name][f"detects_{keys[arm]}{size}"],
                    }
                )
    frame = pl.DataFrame(rows)
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("step:N", title=None),
            y=alt.Y(
                "detected:Q",
                title="Share of 100 draws that detect the step",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "rows:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            xOffset="rows:N",
        )
    )
    target = (
        alt.Chart(pl.DataFrame({"y": [0.8]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(bars, target)
        .properties(width=130, height=150)
        .facet(column=alt.Column("arm:N", sort=list(names.values()), title=None), data=frame)
        .properties(
            title=_title(
                text="The screening design cannot detect a 4-point night step in night-row scores",
                subtitle=(
                    "Gate 4'': synthetic substations with real noise, 5% false positives. Dashed "
                    "rule: the 80% detection target. A 4-point step is 0.2 to 0.5 MW."
                ),
            )
        )
    )


def main() -> None:
    """Draw every figure."""
    for draw, name in (
        (figure_1_headline, "figure_1_headline"),
        (figure_2_cold_nights, "figure_2_cold_nights"),
        (figure_3_winter_breakdown, "figure_3_winter_breakdown"),
        (figure_4_gate2, "figure_4_gate2"),
        (figure_5_gate4, "figure_5_gate4"),
    ):
        _save(draw(), name=name)


if __name__ == "__main__":
    main()
