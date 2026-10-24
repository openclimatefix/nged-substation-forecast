# ruff: noqa: E501
"""Draw the figures of experiment 334 from `work/report/report_scored.json`.

data/ml-experiments/00334-*/run.sh ml-experiments/00334-*/e00334_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00334_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600
GREY: Final[str] = "#6B6B6B"
LABELS: Final[dict[str, str]] = {
    "default": "Default 3 (GAM + Kalman)",
    "M1": "M1: MDN",
    "M2": "M2: MDN + solar signal",
    "R": "R: GRU",
    "T": "T: TFT",
    "STD": "STD (experiment 330)",
    "BG2": "BG2",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    "default": GREY,
    "M1": ocf.BRAND_ORANGE,
    "M2": ocf.DATA_BLUE,
    "R": ocf.DATA_GREEN,
    "T": ocf.DATA_PURPLE,
    "STD": ocf.DATA_SKY,
    "BG2": "#B8860B",
    "manual_heuristic": "#222222",
    "climatology": "#AAAAAA",
}
ARMS: Final[tuple[str, ...]] = ("M1", "M2", "R", "T")
BANDS: Final[tuple[str, ...]] = tuple(f"lead {24 * d}-{24 * (d + 1)} h" for d in range(10))
GROUPS: Final[tuple[str, ...]] = (
    "all rows",
    "ordinary rows",
    "cold winter nights",
    "sunny daylight",
    "Economy 7 night (00:30-07:30)",
    "168-240 h",
)


def chart(data: pl.DataFrame) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def save(figure: alt.TopLevelMixin, *, name: str) -> None:
    """Write the SVG beside the page and a PNG for inspection."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    figure.save(DOCS_DIR / f"{name}.svg")
    figure.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a title with a wrapped subtitle."""
    return alt.TitleParams(
        text=text,
        subtitle=lines,
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
        subtitleColor="#444",
    )


def forest(*, rows: list[dict], name: str, head: str, lines: list[str]) -> None:
    """Draw an estimate-and-interval chart, one row per label, arms in colour."""
    data = pl.DataFrame(rows).with_columns(arm_name=pl.col("arm").replace(LABELS))
    order = list(dict.fromkeys(data["label"].to_list()))
    arms = [a for a in ARMS if a in data["arm"].to_list()]
    colour = alt.Color(
        "arm_name:N",
        scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms]),
        legend=alt.Legend(orient="top", title=None, columns=2),
    )
    layer = chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("arm_name:N"),
    )
    rule = (
        chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222", strokeDash=[4, 3]).encode(x="x:Q")
    )
    bars = layer.mark_rule(strokeWidth=2).encode(
        x=alt.X(
            "lo:Q",
            title="Quantile score minus the default (points of capacity; negative is better)",
        ),
        x2="hi:Q",
        color=colour,
    )
    dots = layer.mark_point(filled=True, size=60).encode(x="est:Q", color=colour)
    save(
        alt.layer(rule, bars, dots).properties(
            width=WIDTH - 230,
            height=26 * len(order) * len(arms),
            title=title(text=head, lines=lines),
        ),
        name=name,
    )


def main() -> None:
    """Draw every figure."""
    rep = json.loads((common.REPORT_DIR / "report_scored.json").read_text())
    c = rep["contrasts"]
    shown = ("default", "M1", "M2", "R", "T", "STD", "BG2", "manual_heuristic", "climatology")
    long = pl.DataFrame(
        [
            {
                "band": b.removeprefix("lead ").removesuffix(" h"),
                "arm": LABELS[a],
                "score": rep["by_mask"][a][b]["qs"],
            }
            for b in BANDS
            for a in shown
            if b in rep["by_mask"][a]
        ]
    )
    long = long.filter(pl.col("score").is_not_nan())
    order = [b.removeprefix("lead ").removesuffix(" h") for b in BANDS]
    lines = (
        chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=order, title="Lead time (hours)"),
            y=alt.Y(
                "score:Q",
                title="Quantile score (points of capacity; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[LABELS[a] for a in shown], range=[COLOURS[a] for a in shown]
                ),
                legend=alt.Legend(orient="top", title=None, columns=2),
            ),
        )
    )
    save(
        lines.properties(
            width=WIDTH - 60,
            height=320,
            title=title(
                text="Score against lead time, 0.5 to 240 hours",
                lines=[
                    "Quantile score by 24-hour lead band, six Primaries, scored year 2025-07-02 to 2026-06-30. Lower is better.",
                    "STD and BG2 stop at 168 hours.",
                ],
            ),
        ),
        name="figure_1_score_by_lead",
    )
    rows = [
        {
            "label": w + (" (without D10)" if without else ""),
            "arm": a,
            **{
                k: (c[a]["window_without_d10"] if without else c[a]["window"])[w][k]
                for k in ("est", "lo", "hi")
            },
        }
        for w in ("0.5-240 h", "0.5-168 h", "168-240 h")
        for without in (False, True)
        for a in ARMS
    ]
    forest(
        rows=rows,
        name="figure_2_gain_over_default",
        head="Gain of each network over Primary demand research default 3",
        lines=[
            "Quantile score minus the default, six Primaries, scored year. Dot: estimate. Line: 95% interval from resampling whole months."
        ],
    )
    rows = [
        {"label": g, "arm": a, **{k: c[a]["groups"][g][k] for k in ("est", "lo", "hi")}}
        for g in GROUPS
        for a in ARMS
    ]
    forest(
        rows=rows,
        name="figure_3_gain_by_group",
        head="Gain over the default by group of half-hours",
        lines=[
            "Quantile score minus the default, leads 0.5 to 240 hours unless the label says otherwise.",
            "Dot: estimate. Line: 95% interval from resampling whole months.",
        ],
    )
    rows = [
        {
            "label": f"all rows, leaving out {s}" if s != "none" else "all rows",
            "arm": a,
            **{
                k: (c[a]["window"]["0.5-240 h"] if s == "none" else c[a]["leave_one_site_out"][s])[
                    k
                ]
                for k in ("est", "lo", "hi")
            },
        }
        for s in ("none", *common.SITES)
        for a in ARMS
    ]
    forest(
        rows=rows,
        name="figure_4_leave_one_primary_out",
        head="Does one Primary carry the result?",
        lines=[
            "Quantile score minus the default with one Primary left out, 0.5 to 240 hours.",
            "Dot: estimate. Line: 95% interval. Left of zero is better.",
        ],
    )


if __name__ == "__main__":
    main()
