# ruff: noqa: E501
"""Draw the figures of experiment 220's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00220-*/e00220_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00220_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
SCOPES: Final[dict[str, str]] = {
    "five sites, all rows": "all rows, leads 0.5-168 h",
    "five sites, leads 15.5-39 h": "leads 15.5-39 h",
    "five sites, cold nights": "cold nights",
    "five sites, daylight": "daylight",
}
STACK_ARMS: Final[dict[str, tuple[str, str]]] = {
    "C4b A+S - A": ("A+S: solar summary", ocf.DATA_SKY),
    "C4a A+H - A": ("A+H: network correction", ocf.DATA_BLUE),
    "C1 A+H+S - A": ("A+H+S: both", ocf.BRAND_ORANGE),
    "H-own - A": ("H-own: network's own quantiles", ocf.DATA_PURPLE),
}
ARM_NAMES: Final[dict[str, str]] = {
    "a": "A",
    "a_s": "A+S",
    "a_h": "A+H",
    "a_hs": "A+H+S",
    "h_own": "H-own",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85),
        subtitle=textwrap.wrap(subtitle, 110),
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


def _forest(
    *, rows: list[dict], order: list[str], domain: list[str], colours: list[str], **kw: Any
) -> alt.TopLevelMixin:
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("group:N", sort=order, title=None),
        yOffset=alt.YOffset("arm:N", sort=domain),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=colours),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (
        zero
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=kw["xtitle"]), x2="high:Q"
        )
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=kw["height"],
        title=_title(text=kw["title"], subtitle=kw["subtitle"]),
    )


def figure_headline(*, report: dict) -> None:
    """Draw each stack arm's quantile score minus arm A's, by row set."""
    rows = [
        {
            "group": label,
            "arm": STACK_ARMS[key][0],
            "estimate": report["contrasts"][scope][key]["level95"]["estimate"],
            "low": report["contrasts"][scope][key]["level95"]["lower"],
            "high": report["contrasts"][scope][key]["level95"]["upper"],
        }
        for scope, label in SCOPES.items()
        for key in STACK_ARMS
    ]
    chart = _forest(
        rows=rows,
        order=list(SCOPES.values()),
        domain=[v[0] for v in STACK_ARMS.values()],
        colours=[v[1] for v in STACK_ARMS.values()],
        xtitle="Quantile score minus arm A (points of capacity; below zero is better)",
        height=260,
        title="The network and the solar summary each lower the score, and their gains add",
        subtitle=(
            "Dot: estimate. Line: 95% interval from resampling whole months. Scored year "
            "2025-07 to 2026-06, five Primary substations, outside the hand-filtered events."
        ),
    )
    _save(chart, name="figure_1_headline")


def figure_drop_one(*, report: dict) -> None:
    """Draw the cost of removing each part from A+H+S."""
    scope = "five sites, all rows"
    keys = {
        "cost of removing H (A+S - A+H+S)": "network (H)",
        "cost of removing S (A+H - A+H+S)": "solar summary (S)",
        "cost of removing widening (drop_widen - A+H+S)": "cold-day widening (experiment 164)",
        "cost of removing R, frozen network (drop_r - A+H+S)": "recipe R, network kept",
        "cost of removing offset, frozen network (drop_offset - A+H+S)": "night offset, network kept",
        "cost of removing R, no network (A+S drop R - A+S)": "recipe R, no network",
        "cost of removing offset, no network (A+S drop offset - A+S)": "night offset, no network",
    }
    rows = [
        {
            "group": label,
            "arm": "cost of removal",
            "estimate": report["contrasts"][scope][key]["level95"]["estimate"],
            "low": report["contrasts"][scope][key]["level95"]["lower"],
            "high": report["contrasts"][scope][key]["level95"]["upper"],
        }
        for key, label in keys.items()
    ]
    chart = _forest(
        rows=rows,
        order=list(keys.values()),
        domain=["cost of removal"],
        colours=[ocf.BRAND_ORANGE],
        xtitle="Quantile score without the part minus with it (points; above zero: the part helps)",
        height=250,
        title="Removing the network costs most; removing recipe R costs a third as much",
        subtitle=(
            "Dot: estimate. Line: 95% interval from resampling whole months. Five Primary "
            "substations, all leads, scored year. The network is frozen where it is kept."
        ),
    )
    _save(chart, name="figure_2_drop_one")


def figure_cold_coverage(*, report: dict) -> None:
    """Draw q95 and q05 coverage on cold nights by arm."""
    rows = [
        {
            "arm": ARM_NAMES[arm],
            "quantile": f"q{level}",
            "coverage": report["tails"]["cold nights"][arm][f"cov_q{level}"],
            "nominal": int(level) / 100,
        }
        for arm in ARM_NAMES
        for level in ("05", "95")
    ]
    frame = pl.DataFrame(rows)
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("arm:N", sort=list(ARM_NAMES.values()), title=None),
            y=alt.Y("coverage:Q", title="Share of readings at or under the quantile"),
            color=alt.Color(
                "quantile:N",
                scale=alt.Scale(domain=["q05", "q95"], range=[ocf.DATA_SKY, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            xOffset="quantile:N",
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [0.05, 0.95]}))
        .mark_rule(color="#333333", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (bars + rules).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="On cold nights the network lifts q95 coverage from 0.90 to 0.96, past the nominal 0.95",
            subtitle=(
                "Bars: coverage of the 5% and 95% quantiles on 5,395 cold-night rows (winter evenings "
                "and nights with ERA5 daily mean below 2 °C). Dashed lines: nominal 0.05 and 0.95."
            ),
        ),
    )
    _save(chart, name="figure_3_cold_night_coverage")


def figure_by_site(*, report: dict) -> None:
    """Draw A+H+S minus A at each substation."""
    sites = list(common.SCORED_SITES)
    frame = pl.DataFrame([{"site": s, "change": report["by_site"][s]["A+H+S - A"]} for s in sites])
    chart = (
        alt.Chart(frame)
        .mark_bar(color=ocf.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", sort=sites, title=None),
            y=alt.Y("change:Q", title="A+H+S minus A (points; below zero is better)"),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The gain is largest at D16, and D9 gets slightly worse",
                subtitle="Quantile score change by substation, all leads, scored year (D9 is not in the headline).",
            ),
        )
    )
    _save(chart, name="figure_4_by_site")


def main() -> None:
    """Draw all figures."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    figure_headline(report=report)
    figure_drop_one(report=report)
    figure_cold_coverage(report=report)
    figure_by_site(report=report)


if __name__ == "__main__":
    main()
