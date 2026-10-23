# ruff: noqa: E501
"""Draw the figures of experiment 306 from `work/report.json`.

uv run --no-sync python ml-experiments/00306-*/e00306_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00306_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
ARM_LABELS: Final[dict[str, str]] = {
    "s_both": "Solar x heating and weekend",
    "s_heat": "Solar x heating only",
    "s_wknd": "Solar x weekend only",
    "s_placebo": "Placebo: solar x day parity",
}
ARM_COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE]


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def headline(*, r: dict) -> None:
    """Reduction in quantile score against the control, by scope and arm, with intervals."""
    scopes = [
        ("amj_D5", "headline", "D5, April to June"),
        ("amj_D9", "headline", "D9, April to June"),
        ("amj_D5_D9", "headline", "D5 and D9 averaged, April to June"),
        ("amj_placebo_D10_D16", "headline", "Placebo D10 and D16, April to June"),
        ("jas_D5_D9", "headline", "D5 and D9 averaged, July to September"),
    ]
    rows = []
    for key, section, label in scopes:
        for arm, arm_label in ARM_LABELS.items():
            c = r[section][key][arm]["reduction"]
            rows.append(
                {"scope": label, "arm": arm_label, "est": c["est"], "lo": c["lo"], "hi": c["hi"]}
            )
    clear = r["amj_clear_midday_D5"]
    for arm in ("s_both", "s_heat", "s_wknd"):
        c = clear[arm]["reduction"]
        rows.append(
            {
                "scope": "D5 clear midday, April to June",
                "arm": ARM_LABELS[arm],
                "est": c["est"],
                "lo": c["lo"],
                "hi": c["hi"],
            }
        )
    order = [s[2] for s in scopes] + ["D5 clear midday, April to June"]
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "arm:N",
        title=None,
        scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
        legend=alt.Legend(orient="top", columns=2),
    )
    y = alt.Y("scope:N", sort=order, title=None)
    offset = alt.YOffset("arm:N", sort=list(ARM_LABELS.values()))
    base = alt.Chart(data)
    layers = [
        alt.Chart(pl.DataFrame({"z": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(x="z:Q"),
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X(
                "lo:Q",
                title="Score of the control minus score of the arm (points; positive means the arm is better)",
            ),
            x2="hi:Q",
            color=colour,
        ),
        base.mark_circle(size=40, opacity=1).encode(y=y, yOffset=offset, x="est:Q", color=colour),  # ty: ignore[unresolved-attribute]
    ]
    chart = alt.layer(*layers).properties(
        width=WIDTH,
        height=230,
        title=_title(
            "No solar interaction column lowers the April-to-June error at D5 and D9 by a useful amount",
            f"Quantile score (points of effective capacity). Dashed line: no change from the control. Dot: estimate. Line: 95% interval over ISO weeks of forecast initialisation time. Daylight half-hours up to 36 hours ahead. The control's April-to-June excess over BG2 is {r['headline']['amj_D5']['excess_control']['est']:.2f} points at D5 and {r['headline']['amj_D9']['excess_control']['est']:.2f} at D9.",
        ),
    )
    _save(chart.configure_view(stroke=None), name="figure_1_reduction")


def monthly(*, r: dict) -> None:
    """Monthly excess over BG2 of the control and two arms at D5 and D9."""
    domain = ["Control (arm D engine)", ARM_LABELS["s_both"], ARM_LABELS["s_wknd"]]
    colour = alt.Color(
        "arm:N",
        title=None,
        scale=alt.Scale(domain=domain, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
        legend=alt.Legend(orient="top"),
    )
    panels = []
    for site in ("D5", "D9"):
        rows = [
            {"month": month, "arm": label, "excess": values[arm]}
            for month, values in r["monthly_excess"][site].items()
            for arm, label in (("a2", domain[0]), ("s_both", domain[1]), ("s_wknd", domain[2]))
        ]
        base = alt.Chart(pl.DataFrame(rows)).encode(
            x=alt.X(
                "month:O",
                title="Month of the half-hour being forecast (valid time)",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("excess:Q", title="Score minus BG2's (points)"),
            color=colour,
        )
        zero = (
            alt.Chart(pl.DataFrame({"z": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#444444", strokeDash=[4, 3])
            .encode(y="z:Q")
        )
        panels.append(
            (zero + base.mark_line(point=True)).properties(width=WIDTH, height=150, title=site)
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            "The interaction columns leave the build-up of the excess from April untouched",
            "Quantile score minus BG2's (points of effective capacity; lower is better; dashed line: same as BG2), daylight half-hours up to 36 hours ahead, by month.",
        )
    )
    _save(chart.configure_view(stroke=None), name="figure_2_monthly_excess")


def main() -> None:
    """Draw every figure."""
    r = json.loads((common.WORK_DIR / "report.json").read_text())
    headline(r=r)
    monthly(r=r)


if __name__ == "__main__":
    main()
