# ruff: noqa: E501
"""Draw the figures of experiment 251 from `work/report/*.json`.

uv run --no-sync python ml-experiments/00251-*/e00251_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00251_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "xgboost": "Weak production-style XGBoost",
    "manual_heuristic": "Manual heuristic",
    "bg2": "BG2 (XGBoost best guess)",
    "d": "Arm D (convex model with Kalman filter)",
    "dh": "D + network head",
    "arm1": "Arm 1: D + head + solar summary",
    "arm2c": "Arm 2c: arm 1, D5 clip without head (post hoc)",
}
COLOURS: Final[dict[str, str]] = {
    "climatology": ocf.GREY_3 if hasattr(ocf, "GREY_3") else "#9A9A9A",
    "xgboost": "#555555",
    "manual_heuristic": "#222222",
    "bg2": ocf.BRAND_ORANGE,
    "d": ocf.DATA_SKY,
    "dh": ocf.DATA_BLUE,
    "arm1": ocf.DATA_PURPLE,
    "arm2c": ocf.DATA_GREEN,
}


def _chart(data: pl.DataFrame) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def _scale(arms: list[str]) -> alt.Scale:
    return alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms])


def headline(*, report: dict) -> None:
    """MAE of the median against lead time, every stage and every comparison."""
    arms = list(LABELS)
    rows = [
        {
            "arm": LABELS[a],
            "lead": int(k) + 12,
            "est": v["mae"]["est"],
            "lo": v["mae"]["lo"],
            "hi": v["mae"]["hi"],
        }
        for a in arms
        for k, v in report["by_lead_bin"][a].items()
    ]
    base = _chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "lead:Q",
            title="Lead time (hours; centre of 24-hour bin)",
            axis=alt.Axis(values=[0, 24, 48, 72, 96, 120, 144, 168]),
        ),
        color=alt.Color(
            "arm:N",
            scale=_scale(arms),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    chart = (
        base.mark_errorband(opacity=0.12).encode(y=alt.Y("lo:Q", title="Mean absolute error of the median (points of capacity; smaller is better)"), y2="hi:Q")
        + base.mark_line(strokeWidth=2).encode(y="est:Q")
    ).properties(width=WIDTH, height=300, title=_title(
        title="Mean absolute error by lead time for the stack, BG2, and the baselines",
        subtitle="Mean absolute error of the median forecast, six substations, 2025-07 to 2026-06. Line: estimate. Band: 95% interval from resampling months.",
    ))  # fmt: skip
    _save(chart, name="figure_1_mae_by_lead")


def forest(
    *,
    report: dict,
    treats: list[str],
    ref: str,
    key: str,
    order: list[str],
    title: str,
    subtitle: str,
    name: str,
) -> None:
    """Quantile score of each treatment minus a reference, by group, with its interval."""
    rows = []
    for t in treats:
        c = report["contrasts"][f"{t}_minus_{ref}"]
        source = c["by_month"] if key == "by_month" else c["by_cut"]
        for g in order:
            v = source[g]
            rows.append(
                {
                    "group": g,
                    "arm": LABELS[t],
                    "estimate": v["est"],
                    "low": v["lo"],
                    "high": v["hi"],
                }
            )
    domain = [LABELS[t] for t in treats]
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("group:N", sort=order, title=None),
        yOffset=alt.YOffset("arm:N", sort=domain),
        color=alt.Color(
            "arm:N",
            scale=_scale(treats),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    zero = _chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(
            x=alt.X(
                "low:Q",
                title=f"Quantile score minus {LABELS[ref].split(' (')[0]}'s (points of capacity; below zero is better)",
            ),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")
    ).properties(
        width=WIDTH,
        height=max(120, 24 * len(order) * len(treats)),
        title=_title(title=title, subtitle=subtitle),
    )
    _save(chart, name=name)


def d5_clip(*, report: dict) -> None:
    """D5 sunny-daylight and daylight quantile score of each clip arm minus its reference."""
    names = {
        "arm2_minus_arm1": "Arm 2 (clip + head) minus arm 1",
        "arm2b_minus_arm1": "Arm 2b (clip, no head) minus arm 1 (post hoc)",
        "arm2b_minus_a_s": "Arm 2b minus A+S, the same arm without the clip (post hoc)",
    }
    rows = [
        {
            "cut": cut,
            "arm": names[k],
            "estimate": v["qs"][cut]["est"],
            "low": v["qs"][cut]["lo"],
            "high": v["qs"][cut]["hi"],
        }
        for k, v in report["clip"].items()
        for cut in ("D5 sunny daylight", "D5 daylight", "site D5")
    ]
    domain = list(names.values())
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("cut:N", sort=["D5 sunny daylight", "D5 daylight", "site D5"], title=None),
        yOffset=alt.YOffset("arm:N", sort=domain),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=[ocf.DATA_PURPLE, ocf.DATA_GREEN, ocf.DATA_BLUE]),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    zero = _chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title="Quantile score change (points of capacity; below zero is better)"), x2="high:Q")
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")
    ).properties(width=WIDTH, height=200, title=_title(
        title="At D5 the 60% clip helps only when the network head is absent",
        subtitle="Quantile score change at D5 when the clip is added, scored year. Dot: estimate. Line: 95% interval from resampling months. Sunny daylight is the sunniest third of daylight half-hours.",
    ))  # fmt: skip
    _save(chart, name="figure_4_d5_clip")


def tails(*, report: dict) -> None:
    """Coverage of q95 and q99 by UTC hour band for the main arms."""
    arms = ["bg2", "d", "arm1", "arm2c", "manual_heuristic"]
    panels = []
    for q, nominal in (("q95", 0.95), ("q99", 0.99)):
        rows = [
            {"arm": LABELS[a], "band": b, "coverage": v[q]["cov"]}
            for a in arms
            for b, v in report["tails_by_hour"][a].items()
        ]
        base = _chart(pl.DataFrame(rows)).encode(
            x=alt.X(
                "band:N",
                title="Valid time (UTC hour band)",
                sort=["00-06 UTC", "06-12 UTC", "12-18 UTC", "18-24 UTC"],
            ),
            y=alt.Y(
                "coverage:Q",
                title=f"Share of observations below {q}",
                scale=alt.Scale(domain=[0.85, 1.0]),
            ),
            color=alt.Color(
                "arm:N",
                scale=_scale(arms),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
            ),
        )
        rule = (
            _chart(pl.DataFrame({"nominal": [nominal]}))
            .mark_rule(color="#333333", strokeDash=[4, 3])
            .encode(y="nominal:Q")
        )
        panels.append(
            (base.mark_line(point=True, strokeWidth=2) + rule).properties(
                width=WIDTH // 2 - 60, height=240
            )
        )
    chart = alt.hconcat(*panels).resolve_scale(color="shared").properties(
        title=_title(
            title="Tail coverage by hour band: the stack and BG2 are near nominal, the manual heuristic is not",
            subtitle="Coverage of the 95% and 99% quantiles by UTC hour band, six substations, scored year. Dashed line: nominal level; the closer a line is to it, the better.",
        )
    )  # fmt: skip
    _save(chart, name="figure_5_tail_coverage_by_hour")


def curtailment() -> None:
    """Saving against the manual heuristic at the headline scaled limit, per arm."""
    data = json.loads((common.REPORT_DIR / "curtailment_scored.json").read_text())
    arms = [a for a in ("climatology", "xgboost", "bg2", "d", "arm1", "arm2c") if a in LABELS]
    rows = []
    for a in arms:
        v = data["0.9"]["savings"].get(a)
        if v:
            rows.append(
                {
                    "arm": LABELS[a],
                    "saving": v["saving_vs_manual_gbp"] / 1e6,
                    "low": v["low_gbp"] / 1e6,
                    "high": v["high_gbp"] / 1e6,
                }
            )
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=[LABELS[a] for a in arms], title=None),
        color=alt.Color("arm:N", scale=_scale(arms), legend=None),
    )
    zero = _chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title="Saving against the manual heuristic (GBP million per year, five substations; above zero is better)"), x2="high:Q")
        + base.mark_point(filled=True, size=50).encode(x="saving:Q")
    ).properties(width=WIDTH, height=180, title=_title(
        title="Curtailment cost saved against the manual heuristic, by forecast",
        subtitle="Node curtailment cost at a limit of 0.9 times each substation's largest earlier flow, GBP 2,000 per MWh-equivalent. Dot: estimate. Line: 95% interval from resampling months.",
    ))  # fmt: skip
    _save(chart, name="figure_6_curtailment_saving")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report_scored.json").read_text())
    headline(report=report)
    cuts = [
        "all rows",
        "cold nights (below 2 C)",
        "Christmas",
        "bank holidays",
        "sunny daylight",
        "winter (Dec-Feb)",
        "spring (Mar-May)",
        "summer (Jun-Aug)",
        "autumn (Sep-Nov)",
        "under 24 h",
        "24-72 h",
        "72-168 h",
    ]
    forest(report=report, treats=["arm1", "arm2c"], ref="bg2", key="by_cut", order=cuts,
           title="The stack's gain over BG2 sits in winter and on cold nights; summer is level",
           subtitle="Quantile score minus BG2's, by kind of row, six substations. Intervals resample months.",
           name="figure_2_by_cut_against_bg2")  # fmt: skip
    months = list(report["contrasts"]["arm1_minus_bg2"]["by_month"])
    forest(report=report, treats=["arm1", "arm2c"], ref="bg2", key="by_month", order=months,
           title="Quantile score minus BG2's, month by month",
           subtitle="Quantile score minus BG2's by month of forecast run, six substations. Intervals resample initialisation days.",
           name="figure_3_by_month_against_bg2")  # fmt: skip
    d5_clip(report=report)
    tails(report=report)
    curtailment()


if __name__ == "__main__":
    main()
