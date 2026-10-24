# ruff: noqa: E501
"""Draw the figures of experiment 333 from `work/report/report.json`.

Put experiments 278, 311, 317, 319, 320, 322, and 333 on `PYTHONPATH` (see `e00333_common`), then run
`uv run --no-sync python ml-experiments/00333-*/e00333_charts.py`.
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00311_charts as base  # ty: ignore[unresolved-import]
import e00333_common as own
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{own.EXPERIMENT_ID:05d}-{own.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = own.WORK_DIR / "png"
WIDTH: Final[int] = base.WIDTH
LABELS: Final[dict[str, str]] = {"default": "Research default 3", "z0": "Z0: history only", "z1": "Z1: with covariates", "manual_heuristic": "Manual heuristic", "climatology": "Climatology"}  # fmt: skip
COLOURS: Final[dict[str, str]] = {"default": base.GREY, "z0": ocf.DATA_BLUE, "z1": ocf.BRAND_ORANGE, "manual_heuristic": ocf.DATA_GREEN, "climatology": ocf.DATA_PURPLE}  # fmt: skip
ARMS: Final[tuple[str, ...]] = ("z0", "z1")
GROUPS: Final[tuple[str, ...]] = (
    "all rows",
    "ordinary rows",
    "cold winter nights",
    "sunny daylight",
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


def forest(*, rows: list[dict], name: str, head: str, lines: list[str]) -> None:
    """Draw an estimate-and-interval chart, one row per label, the two arms in colour."""
    data = pl.DataFrame(rows).with_columns(arm_name=pl.col("arm").replace(LABELS))
    order = list(dict.fromkeys(data["label"].to_list()))
    colour = alt.Color("arm_name:N", scale=alt.Scale(domain=[LABELS[a] for a in ARMS], range=[COLOURS[a] for a in ARMS]), legend=alt.Legend(orient="top", title=None))  # fmt: skip
    layer = chart(data).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), yOffset=alt.YOffset("arm_name:N"))  # fmt: skip
    rule = (
        chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222", strokeDash=[4, 3]).encode(x="x:Q")
    )
    bars = layer.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="Quantile score minus research default 3 (points of capacity; below zero is better)"), x2="hi:Q", color=colour)  # fmt: skip
    dots = layer.mark_point(filled=True, size=60).encode(x="est:Q", color=colour)
    save(alt.layer(rule, bars, dots).properties(width=WIDTH - 230, height=26 * len(order) * len(ARMS), title=base.title(text=head, lines=lines)), name=name)  # fmt: skip


def main() -> None:
    """Draw every figure."""
    rep = json.loads((own.REPORT_DIR / "report.json").read_text())
    shown = ("default", "z0", "z1", "manual_heuristic", "climatology")
    bands = sorted(rep["by_band"], key=lambda b: int(b.split("-")[0]))
    long = pl.DataFrame([{"band": b.removesuffix(" h"), "arm": LABELS[a], "score": rep["by_band"][b][a]["est"]} for b in bands for a in shown])  # fmt: skip
    order = [b.removesuffix(" h") for b in bands]
    lines = chart(long).mark_line(point=True).encode(x=alt.X("band:N", sort=order, title="Lead time (hours)"), y=alt.Y("score:Q", title="Quantile score (points of capacity; lower is better)", scale=alt.Scale(zero=False)), color=alt.Color("arm:N", scale=alt.Scale(domain=[LABELS[a] for a in shown], range=[COLOURS[a] for a in shown]), legend=alt.Legend(orient="top", title=None, columns=3)))  # fmt: skip
    save(lines.properties(width=WIDTH - 60, height=300, title=base.title(text="Chronos-2 without training trails research default 3 at every lead from 0.5 to 240 hours", lines=["Quantile score by 24-hour lead band, six Primaries, scored year 2025-07-02 to 2026-06-30. Lower is better."])), name="figure_1_score_by_lead")  # fmt: skip
    rows = [{"label": b.removesuffix(" h"), "arm": a, "est": rep["by_band"][b][f"{a}_vs_default"]["diff"], "lo": rep["by_band"][b][f"{a}_vs_default"]["diff_lo_95"], "hi": rep["by_band"][b][f"{a}_vs_default"]["diff_hi_95"]} for b in bands for a in ARMS]  # fmt: skip
    forest(rows=rows, name="figure_2_gap_by_lead", head="The gap to research default 3 is 0.4 to 0.8 points at every lead", lines=["Quantile score minus Primary demand research default 3 by 24-hour lead band (hours), six Primaries.", "Dot: estimate. Line: 95% interval from resampling whole months. Left of zero would be better than the default."])  # fmt: skip
    rows = [
        {"label": f"{g}, {v}", "arm": a, "est": e[f"{a}_vs_default"]["diff"], "lo": e[f"{a}_vs_default"]["diff_lo_95"], "hi": e[f"{a}_vs_default"]["diff_hi_95"]}
        for g in GROUPS for v in ("with D10", "without D10") for a in ARMS
        for e in [rep["by_group"][v][g]["0-240 h"]]
    ]  # fmt: skip
    forest(rows=rows, name="figure_3_gap_by_group", head="Covariates help on cold winter nights and hurt on sunny daylight", lines=["Quantile score minus research default 3, leads 0.5 to 240 hours, six Primaries, with and without D10.", "Dot: estimate. Line: 95% interval from resampling whole months. Left of zero would be better than the default."])  # fmt: skip


if __name__ == "__main__":
    main()
