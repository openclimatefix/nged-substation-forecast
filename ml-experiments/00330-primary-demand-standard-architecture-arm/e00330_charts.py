# ruff: noqa: E501
"""Draw the figures of experiment 330 from `work/report/report_scored.json`.

export PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-*:ml-experiments/00330-*
uv run --no-sync python ml-experiments/00330-*/e00330_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt

import e00311_charts as base  # isort: skip  # ty: ignore[unresolved-import]
import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00330_common as own  # isort: skip
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{own.EXPERIMENT_ID:05d}-{own.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = own.WORK_DIR / "png"
WIDTH: Final[int] = base.WIDTH
LABELS: Final[dict[str, str]] = {"default": "Default", "G": "G: gated head (314)", "STD": "STD: standard", "STDA": "STD-A: median only", "bg2": "BG2", "manual_heuristic": "Manual heuristic"}  # fmt: skip
COLOURS: Final[dict[str, str]] = {"default": base.GREY, "G": ocf.DATA_SKY, "STD": ocf.BRAND_ORANGE, "STDA": ocf.DATA_BLUE, "bg2": ocf.DATA_PURPLE, "manual_heuristic": ocf.DATA_GREEN}  # fmt: skip
GROUPS: Final[tuple[str, ...]] = ("all rows", "ordinary rows", "cold winter nights", "sunny daylight", "Economy 7 night (00:30-07:30)", "Christmas", "bank holidays", "leads 12-36 h")  # fmt: skip
ARMS: Final[tuple[str, ...]] = ("STD", "STDA", "G")
BANDS: Final[tuple[str, ...]] = tuple(f"lead {24 * d}-{24 * (d + 1)} h" for d in range(7))


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
    """Draw an estimate-and-interval chart, one row per label, arms in colour."""
    data = pl.DataFrame(rows).with_columns(arm_name=pl.col("arm").replace(LABELS))
    order = list(dict.fromkeys(data["label"].to_list()))
    colour = alt.Color("arm_name:N", scale=alt.Scale(domain=[LABELS[a] for a in ARMS], range=[COLOURS[a] for a in ARMS]), legend=alt.Legend(orient="top", title=None, columns=1))  # fmt: skip
    layer = chart(data).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), yOffset=alt.YOffset("arm_name:N"))  # fmt: skip
    rule = chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222", strokeDash=[4, 3]).encode(x="x:Q")  # fmt: skip
    bars = layer.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="Quantile score minus the default (points of capacity; more negative is better)"), x2="hi:Q", color=colour)  # fmt: skip
    dots = layer.mark_point(filled=True, size=60).encode(x="est:Q", color=colour)
    save(alt.layer(rule, bars, dots).properties(width=WIDTH - 230, height=26 * len(order) * len(ARMS), title=base.title(text=head, lines=lines)), name=name)  # fmt: skip


def main() -> None:
    """Draw every figure."""
    rep = json.loads((own.REPORT_DIR / "report_scored.json").read_text())
    c = rep["contrasts"]
    shown = ("default", "G", "STD", "STDA", "bg2", "manual_heuristic")
    long = pl.DataFrame([{"band": b.removeprefix("lead ").removesuffix(" h"), "arm": LABELS[a], "score": rep["by_mask"][a][b]["qs"]} for b in BANDS for a in shown])  # fmt: skip
    order = [b.removeprefix("lead ").removesuffix(" h") for b in BANDS]
    lines = chart(long).mark_line(point=True).encode(x=alt.X("band:N", sort=order, title="Lead time (hours)"), y=alt.Y("score:Q", title="Quantile score (points of capacity; lower is better)", scale=alt.Scale(zero=False)), color=alt.Color("arm:N", scale=alt.Scale(domain=[LABELS[a] for a in shown], range=[COLOURS[a] for a in shown]), legend=alt.Legend(orient="top", title=None, columns=2)))  # fmt: skip
    save(lines.properties(width=WIDTH - 60, height=300, title=base.title(text="The standard architecture scores below the default at every lead, 0.5 to 168 hours", lines=["Quantile score by 24-hour lead band, six Primaries, scored year 2025-07-02 to 2026-06-30. Lower is better. Leads above 168 hours were not run."])), name="figure_1_score_by_lead")  # fmt: skip
    rows = [{"label": g, "arm": a, **{k: c[a]["groups"][g][k] for k in ("est", "lo", "hi")}} for g in GROUPS for a in ARMS]  # fmt: skip
    forest(rows=rows, name="figure_2_gain_by_group", head="The gain of the standard architecture is spread over groups, with wide intervals on the hard ones", lines=["Quantile score minus Primary demand research default 3, six Primaries, leads 0.5 to 168 hours.", "Dot: estimate. Line: 95% interval from resampling whole months. Left of zero is better."])  # fmt: skip
    rows = [{"label": f"all rows, leaving out {s}" if s != "none" else "all rows", "arm": a, **{k: (c[a]["gain_95"] if s == "none" else c[a]["leave_one_site_out"][s]["ci99"])[k] for k in ("est", "lo", "hi")}} for s in ("none", *common.SCORED_SITES) for a in ARMS]  # fmt: skip
    forest(rows=rows, name="figure_3_leave_one_primary_out", head="Leaving out D10 shrinks the gain of the standard architecture to almost nothing", lines=["Quantile score minus the default with one Primary left out, scored year. Dot: estimate.", "Line: 99% interval (the top row, with nothing left out, shows 95%). Left of zero is better."])  # fmt: skip


if __name__ == "__main__":
    main()
