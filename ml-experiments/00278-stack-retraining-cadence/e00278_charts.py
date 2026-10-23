# ruff: noqa: E501
"""Draw the figures of experiment 278 from `work/report/*.json`.

uv run --no-sync python ml-experiments/00278-*/e00278_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00278_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00278-stack-retraining-cadence"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600


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


def headline(*, gam: dict, head: dict, solar: dict) -> None:
    """Dot-and-interval chart of the five planned contrasts against their thresholds."""
    c3 = head["contrasts"]["stack: C3 once minus quarterly"]
    c4 = head["contrasts"]["stack: C4 monthly minus quarterly"]
    rows = [
        (
            "C1 GAM refit: quarterly minus monthly",
            gam["contrasts"]["C1 quarterly minus monthly"]["qs_99"],
            0.02,
            "all rows",
        ),
        (
            "C2 GAM refit: weekly minus monthly",
            gam["contrasts"]["C2 weekly minus monthly"]["qs_99"],
            -0.02,
            "all rows",
        ),
        (
            "C3 head: trained once minus quarterly (all rows)",
            c3["all"]["month_blocks_99"],
            0.03,
            "all rows",
        ),
        (
            "C3 head: trained once minus quarterly (Apr to Jun)",
            c3["last_quarter"]["week_blocks_99"]
            | {"est": c3["last_quarter"]["month_blocks_99"]["est"]},
            0.03,
            "last quarter (week blocks)",
        ),
        ("C4 head: monthly minus quarterly", c4["all"]["month_blocks_99"], -0.01, "all rows"),
        (
            "C5 solar factors: quarterly minus monthly (daylight)",
            solar["contrasts"]["C5 hs quarterly minus monthly"]["daylight"]["99"],
            0.01,
            "daylight rows",
        ),
    ]
    frame = pl.DataFrame(
        [
            {
                "contrast": r[0],
                "estimate": r[1]["est"],
                "low": r[1]["lo"],
                "high": r[1]["hi"],
                "threshold": r[2],
            }
            for r in rows
        ]
    )
    order = [r[0] for r in rows]
    base = _chart(frame).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=420))
    )
    zero = _chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")
    chart = (
        zero
        + base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(
            x=alt.X(
                "low:Q",
                title="Quantile score difference (points of capacity; negative favours the first-named cadence)",
            ),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=50, color=ocf.DATA_BLUE).encode(x="estimate:Q")
        + base.mark_tick(color=ocf.BRAND_ORANGE, thickness=2, size=18).encode(x="threshold:Q")
    ).properties(
        width=WIDTH,
        height=220,
        title=_title(
            title="Quarterly refits and a head trained once cost little; weekly refits gain nothing",
            subtitle="Dot: estimate. Line: 99% interval from resampling whole months (weeks for Apr to Jun). Orange tick: pre-registered threshold. Six substations, 2025-07 to 2026-06.",
        ),
    )
    _save(chart, name="figure_1_planned_contrasts")


def by_month(*, gam: dict, head: dict, solar: dict) -> None:
    """Monthly line chart of each contrast."""
    months = sorted(gam["contrasts"]["C1 quarterly minus monthly"]["by_month"])
    rows = []
    for m in months:
        rows += [
            {
                "month": m,
                "contrast": "GAM refit: quarterly minus monthly",
                "value": gam["contrasts"]["C1 quarterly minus monthly"]["by_month"][m],
            },
            {
                "month": m,
                "contrast": "Head: trained once minus quarterly",
                "value": head["contrasts"]["stack: C3 once minus quarterly"]["by_origin_month"][m],
            },
            {
                "month": m,
                "contrast": "Solar factors: quarterly minus monthly (daylight)",
                "value": solar["contrasts"]["C5 hs quarterly minus monthly"]["by_origin_month"][m],
            },
        ]
    domain = [
        "GAM refit: quarterly minus monthly",
        "Head: trained once minus quarterly",
        "Solar factors: quarterly minus monthly (daylight)",
    ]
    chart = (
        _chart(pl.DataFrame(rows))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("month:N", sort=months, title="Month of the forecast run's start"),
            y=alt.Y(
                "value:Q",
                title="Quantile score difference (points; below zero favours the first-named)",
            ),
            color=alt.Color(
                "contrast:N",
                scale=alt.Scale(
                    domain=domain, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                title="The cadence effects change sign across the year",
                subtitle="Each point is one month of the scored year, six substations pooled.",
            ),
        )
    )
    _save(chart, name="figure_2_by_month")


def absolute(*, gam: dict, head: dict) -> None:
    """Bar chart of absolute quantile scores."""
    a, h = gam["absolute"], head["absolute"]
    items = [
        ("BG2 refitted weekly", h["bg2_weekly"]["qs"], ocf.BRAND_ORANGE),
        ("BG2 refitted monthly", h["bg2_monthly"]["qs"], ocf.BRAND_ORANGE_LIGHT),
        ("Arm D, GAM refit monthly", a["monthly"]["qs"], ocf.DATA_SKY),
        ("Arm D, GAM refit weekly", a["weekly"]["qs"], ocf.DATA_SKY),
        ("Arm D, GAM refit quarterly", a["quarterly"]["qs"], ocf.DATA_SKY),
        ("Stack, head trained once", h["stack_once"]["qs"], ocf.DATA_BLUE),
        ("Stack, head retrained quarterly", h["stack_quarterly"]["qs"], ocf.DATA_BLUE),
        ("Stack, head retrained monthly", h["stack_monthly"]["qs"], ocf.DATA_BLUE),
    ]
    frame = pl.DataFrame([{"arm": n, "qs": v, "colour": c} for n, v, c in items])
    order = [n for n, _, _ in items]
    base = _chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None),
        x=alt.X(
            "qs:Q",
            scale=alt.Scale(domain=[2.4, 2.8]),
            title="Quantile score (points of capacity; smaller is better)",
        ),
    )
    chart = (
        base.mark_bar().encode(color=alt.Color("colour:N", scale=None))
        + base.mark_text(align="left", dx=3).encode(text=alt.Text("qs:Q", format=".3f"))
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            title="Absolute scores: every stack variant beats BG2, however it is retrained",
            subtitle="Six substations, 561,363 half-hours, 2025-07 to 2026-06. Bars start at 2.4 to show differences.",
        ),
    )
    _save(chart, name="figure_3_absolute")


def since(*, gam: dict, head: dict) -> None:
    """Skill against months since the last refit."""
    rows = []
    for k, v in gam["contrasts"]["C1 quarterly minus monthly"]["by_months_since_refit"].items():
        for season in ("winter", "summer"):
            if v[season] is not None:
                rows.append(  # noqa: PERF401
                    {
                        "months": int(k),
                        "series": f"GAM refit quarterly minus monthly, {season}",
                        "value": v[season],
                    }
                )
    for k, v in head["by_months_since"]["once_minus_monthly_by_k"].items():
        rows.append(
            {
                "months": int(k),
                "series": "Head trained once minus retrained monthly (months since 2025-07)",
                "value": v,
            }
        )
    domain = sorted({r["series"] for r in rows})
    chart = (
        _chart(pl.DataFrame(rows))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "months:Q",
                title="Months since the last refit (GAM) or since 2025-07 (head trained once)",
                axis=alt.Axis(values=list(range(12))),
            ),
            y=alt.Y(
                "value:Q",
                title="Quantile score difference (points; below zero favours the first-named)",
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=domain, range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                title="Skill does not fall steadily with months since the refit",
                subtitle="Six substations pooled. Months since the last GAM refit run from 0 to 2 under quarterly refits.",
            ),
        )
    )
    _save(chart, name="figure_4_months_since_refit")


def main() -> None:
    """Draw every figure."""
    gam = json.loads((common.REPORT_DIR / "gam_report.json").read_text())
    head = json.loads((common.REPORT_DIR / "head_report.json").read_text())
    solar = json.loads((common.REPORT_DIR / "solar_report.json").read_text())
    headline(gam=gam, head=head, solar=solar)
    by_month(gam=gam, head=head, solar=solar)
    absolute(gam=gam, head=head)
    since(gam=gam, head=head)


if __name__ == "__main__":
    main()
