# ruff: noqa: E501
"""Draw the figures of experiment 314 from `work/report/report_scored.json` and the stored arms.

export PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-*
uv run --no-sync python ml-experiments/00314-*/e00314_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt

import e00311_charts as base  # isort: skip  # ty: ignore[unresolved-import]
import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00314_common as own  # isort: skip
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{own.EXPERIMENT_ID:05d}-{own.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = own.WORK_DIR / "png"
WIDTH: Final[int] = base.WIDTH
GREY: Final[str] = base.GREY
LABELS: Final[dict[str, str]] = {"default": "Default", "M": "Experiment 311 head (M)", "G": "G: gated mixture head", "P": "P: pinball head", "GP": "GP: gated pinball head", "bg2": "BG2"}  # fmt: skip
COLOURS: Final[dict[str, str]] = {"default": GREY, "M": ocf.DATA_SKY, "G": ocf.BRAND_ORANGE, "P": ocf.DATA_BLUE, "GP": ocf.DATA_GREEN, "bg2": ocf.DATA_PURPLE}  # fmt: skip
GROUPS: Final[tuple[str, ...]] = ("all rows", "ordinary rows", "cold winter nights", "sunny daylight", "Economy 7 night (00:30-07:30)", "Christmas", "bank holidays", "leads 12-36 h")  # fmt: skip
ARMS: Final[tuple[str, ...]] = ("G", "P", "GP")


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


def forest(*, rows: list[dict], name: str, head: str, lines: list[str], x_title: str) -> None:
    """Draw an estimate-and-interval chart, one row per label, arms in colour."""
    data = pl.DataFrame(rows).with_columns(arm_name=pl.col("arm").replace(LABELS))
    order = list(dict.fromkeys(data["label"].to_list()))
    arms = [a for a in ARMS if a in data["arm"].to_list()]
    colour = alt.Color("arm_name:N", scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms]), legend=alt.Legend(orient="top", title=None))  # fmt: skip
    layer = chart(data).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), yOffset=alt.YOffset("arm_name:N"))  # fmt: skip
    rule = chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222", strokeDash=[4, 3]).encode(x="x:Q")  # fmt: skip
    bars = layer.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title=x_title), x2="hi:Q", color=colour)  # fmt: skip
    dots = layer.mark_point(filled=True, size=60).encode(x="est:Q", color=colour)
    save(alt.layer(rule, bars, dots).properties(width=WIDTH - 230, height=26 * len(order) * len(arms), title=base.title(text=head, lines=lines)), name=name)  # fmt: skip


def main() -> None:
    """Draw every figure."""
    rep = json.loads((own.REPORT_DIR / "report_scored.json").read_text())
    c = rep["contrasts"]
    rows = [{"label": g, "arm": a, **{k: c[a]["groups"][g][k] for k in ("est", "lo", "hi")}} for g in GROUPS for a in ARMS]  # fmt: skip
    forest(rows=rows, name="figure_1_gain_by_group", head="Gating removes the harm on sunny daylight half-hours and keeps the gain elsewhere", lines=["Quantile score of each arm minus Primary demand research default 1, six Primaries, 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours.", "Dot: estimate. Line: 95% interval from resampling whole months. Left of zero is better."], x_title="Quantile score minus the default (points of capacity; more negative is better)")  # fmt: skip
    rows = [{"label": f"all rows, leaving out {s}" if s != "none" else "all rows", "arm": a, **{k: (c[a]["gain_95"] if s == "none" else c[a]["leave_one_site_out"][s]["ci99"])[k] for k in ("est", "lo", "hi")}} for s in ("none", *common.SCORED_SITES) for a in ARMS]  # fmt: skip
    forest(rows=rows, name="figure_2_leave_one_primary_out", head="Leaving out D10 removes most of the gain of every arm", lines=["Quantile score minus the default with one Primary left out, scored year. Dot: estimate.", "Line: 99% interval from resampling whole months (the top row, with nothing left out, shows 95%). Left of zero is better."], x_title="Quantile score minus the default (points of capacity; more negative is better)")  # fmt: skip
    arms_cov = ("default", "M", "G", "P", "bg2")
    for tag, group, text in (
        ("sunny", "sunny daylight", "sunny daylight half-hours"),
        ("cold", "cold winter nights", "cold winter nights"),
    ):
        cov = pl.DataFrame([{"arm": LABELS[a], "level": k, "coverage": rep["by_mask"][a][group][f"cov_{k}"]} for a in arms_cov for k in ("q01", "q05", "q95", "q99")])  # fmt: skip
        order = ["q01", "q05", "q95", "q99"]
        nominal = chart(pl.DataFrame({"level": order, "nominal": [0.01, 0.05, 0.95, 0.99]})).mark_tick(color="#222", thickness=2, size=40).encode(x=alt.X("level:N", sort=order), y="nominal:Q")  # fmt: skip
        pts = chart(cov).mark_point(filled=True, size=70).encode(x=alt.X("level:N", sort=order, title="Quantile level"), y=alt.Y("coverage:Q", title="Share of actuals at or below the quantile"), color=alt.Color("arm:N", scale=alt.Scale(domain=[LABELS[a] for a in arms_cov], range=[COLOURS[a] for a in arms_cov]), legend=alt.Legend(orient="top", title=None, columns=2)), xOffset="arm:N")  # fmt: skip
        save(alt.layer(nominal, pts).properties(width=WIDTH - 60, height=230, title=base.title(text=f"Tail coverage on {text}", lines=["Share of half-hours at or below each quantile, six Primaries, scored year. Black tick: the level it should hit."])), name=f"figure_{3 if tag == 'sunny' else 4}_tail_coverage_{tag}")  # fmt: skip
    arms_dir = own.WORK_DIR / "arms" / "scored"
    frame = pl.read_parquet(common.E251_SCORED / "rows.parquet")
    for a in ("default", "G", "P"):
        q = np.load(arms_dir / f"{a}.npy")
        frame = frame.with_columns(pl.Series(f"lo_{a}", q[:, 1]), pl.Series(f"hi_{a}", q[:, 9]))
    day_ahead = frame.filter((pl.col("lead_hours") >= 15.0) & (pl.col("lead_hours") < 39.0))
    for number, name, site, month, mask, text, note in (
        (5, "sunny_week", "D5", 6, pl.col("sunny") & pl.col("daylight"), "A June week at D5: the gated head matches the default on sunny half-hours only", "A week in June 2026"),
        (6, "cold_night_week", "D10", 1, pl.col("cold2"), "A cold January week at D10: the heads' bands against the default's", "A week in January 2026"),
    ):  # fmt: skip
        pick = day_ahead.filter(pl.col("site") == site)
        cand = pick.filter(
            (pl.col("valid_time").dt.month() == month)
            & (pl.col("valid_time").dt.year() == 2026)
            & mask
        )
        start = (
            cand["valid_time"]
            .dt.date()
            .value_counts()
            .sort("count", descending=True)["valid_time"][0]
        )
        start = pl.Series([start]).cast(pl.Date)[0]
        window = pick.filter((pl.col("valid_time").dt.date() >= start) & (pl.col("valid_time").dt.date() < start + pl.duration(days=7))).sort("valid_time")  # fmt: skip
        window = window.with_columns(hours=(pl.col("valid_time") - pl.col("valid_time").min()).dt.total_seconds() / 86400.0 + 1.0)  # fmt: skip
        window = window.with_columns(
            segment=(pl.col("hours").diff().fill_null(0.0) > 0.1).cast(pl.Int32).cum_sum()
        )
        shown = ("default", "G", "P")
        long = pl.concat([window.select("segment", "hours", pl.col(f"lo_{a}").alias("lo"), pl.col(f"hi_{a}").alias("hi"), pl.lit(LABELS[a]).alias("arm")) for a in shown])  # fmt: skip
        band = chart(long).mark_area(opacity=0.3).encode(x=alt.X("hours:Q", title="Day of the week (1 to 7)", axis=alt.Axis(values=[1, 2, 3, 4, 5, 6, 7], format="d")), y=alt.Y("lo:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)), y2="hi:Q", detail="segment:N", color=alt.Color("arm:N", scale=alt.Scale(domain=[LABELS[a] for a in shown], range=[COLOURS[a] for a in shown]), legend=alt.Legend(orient="top", title=None)))  # fmt: skip
        line = (
            chart(window)
            .mark_line(color="#111", strokeWidth=1.4)
            .encode(x="hours:Q", y="actual:Q", detail="segment:N")
        )
        save(alt.layer(band, line).properties(width=WIDTH - 60, height=230, title=base.title(text=text, lines=[f"{note}. Shaded: 5% to 95% band at day-ahead leads (15 to 39 hours). Black line: observed power. Substation {site}."])), name=f"figure_{number}_{name}")  # fmt: skip


if __name__ == "__main__":
    main()
