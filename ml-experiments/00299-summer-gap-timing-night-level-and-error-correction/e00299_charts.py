# ruff: noqa: E501
"""Draw the figures of experiment 299 from `work/*.json` and `work/night_nightly.parquet`.

uv run --no-sync python ml-experiments/00299-*/e00299_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00299_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
SITE_COLOURS: Final[dict[str, str]] = {
    "D5": ocf.BRAND_ORANGE,
    "D9": ocf.DATA_BLUE,
    "D10": ocf.DATA_PURPLE,
}


def _save(chart: Any, *, name: str) -> None:
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


def weekly_gap(*, timing: dict) -> None:
    """Figure 1: weekly daylight gap of arm D over BG2 at D5, D9, and D10."""
    data = pl.DataFrame(timing["weekly"]).filter(
        pl.col("day") & pl.col("site").is_in(list(SITE_COLOURS))
    )
    data = data.with_columns(week=pl.col("valid_week").str.to_date())
    lines = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, clip=True)
        .encode(
            x=alt.X(
                "week:T", title="Week of the valid time (Monday)", axis=alt.Axis(format="%b %Y")
            ),
            y=alt.Y("gap:Q", title="Arm D minus BG2 (points; positive: arm D is worse)"),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(domain=list(SITE_COLOURS), range=list(SITE_COLOURS.values())),
                legend=alt.Legend(title="Substation", orient="top"),
            ),
        )
    )
    rules = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            pl.DataFrame({"day": ["2025-10-26", "2026-03-29"]}).with_columns(
                pl.col("day").str.to_date()
            )
        )
        .mark_rule(strokeDash=[4, 3], color="#666666")
        .encode(x="day:T")
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#999999").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + lines + rules).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="The daylight excess of arm D over BG2 builds from April to June and is not concentrated in the four weeks after the clock change",
            subtitle="Mean quantile-score difference on daylight half-hours, by week. Dashed lines: clock changes (26 October 2025, 29 March 2026). Anonymised Primary substations.",
        ),
    )
    _save(chart, name="fig1_weekly_gap")


def lead_hour(*, timing: dict) -> None:
    """Figure 2: April-to-June daylight gap by lead band at D5 and D9."""
    rows = []
    for site in ("D5", "D9"):
        for band in ("0-6", "6-24", "24-48", "48-168"):
            r = timing["by_site"][site][f"amj_daylight_lead_{band}"]
            rows.append({"site": site, "band": band, **{k: r[k] for k in ("est", "lo", "hi")}})
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("band:N", sort=["0-6", "6-24", "24-48", "48-168"], title="Lead (hours)"),
        color=alt.Color(
            "site:N",
            scale=alt.Scale(domain=["D5", "D9"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=alt.Legend(title="Substation", orient="top"),
        ),
        yOffset="site:N",
    )
    chart = (
        base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "est:Q",
                title="Arm D minus BG2, April to June daylight (points; positive: arm D is worse)",
            )
        )
    ).properties(
        width=WIDTH,
        height=170,
        title=_title(
            title="The April-to-June daylight excess is spread over all leads, not concentrated at short ones",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole weeks of forecast initialisation time.",
        ),
    )
    _save(chart, name="fig2_lead_band")


def night_residual() -> None:
    """Figure 3: D10 nightly mean residual of arm D, with month means."""
    nightly = pl.read_parquet(common.WORK_DIR / "night_nightly.parquet")
    nightly = nightly.with_columns(month_start=pl.col("night_date").dt.truncate("1mo"))
    monthly = (
        nightly.group_by("month_start")
        .agg(pl.col("resid_mean").mean().alias("m"))
        .sort("month_start")
    )
    dots = (
        alt.Chart(nightly)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=12, opacity=0.35, color=ocf.DATA_PURPLE)
        .encode(
            x=alt.X("night_date:T", title="Night (evening date)", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("resid_mean:Q", title="Actual minus arm D (points; negative: over-forecast)"),
        )
    )
    line = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.BRAND_ORANGE)
        .encode(x="month_start:T", y="m:Q")
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#999999").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + dots + line).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="D10's night-time residual wanders from month to month; it has no clean step",
            subtitle="Dots: nightly mean residual, 21:00 to 07:30 UTC, leads 12 to 36 hours. Orange: monthly mean.",
        ),
    )
    _save(chart, name="fig3_d10_night")


def layers(*, correction: dict) -> None:
    """Figure 4: share of the April-to-June daylight excess that removing each layer removes."""
    labels = {
        "filter_removal_share_AMJ": "Kalman filter (GAM-K)",
        "arm1_removal_share_AMJ": "Head, all inputs (arm 1)",
        "arm5_removal_share_AMJ": "Head, no filter state (arm 5)",
        "arm6_removal_share_AMJ": "Head, forecast only (arm 6)",
    }
    rows = []
    for site in ("D5", "D9"):
        for key, label in labels.items():
            r = correction["reduction"][site][key]
            rows.append({"site": site, "layer": label, **{k: r[k] for k in ("est", "lo", "hi")}})
    data = pl.DataFrame(rows).with_columns(
        lo=pl.col("lo").clip(-2.5, 1.0), hi=pl.col("hi").clip(-2.5, 1.0)
    )
    base = alt.Chart(data).encode(
        y=alt.Y("layer:N", sort=list(labels.values()), title=None),
        yOffset="site:N",
        color=alt.Color(
            "site:N",
            scale=alt.Scale(domain=["D5", "D9"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=alt.Legend(title="Substation", orient="top"),
        ),
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0, 0.3]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#666666")
        .encode(x="x:Q")
    )
    chart = (
        rule
        + base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "est:Q",
                title="Share of the excess that removing the layer removes (30% was the bar; negative: the layer helps)",
                scale=alt.Scale(domain=[-2.5, 1.0]),
            )
        )
    ).properties(
        width=WIDTH,
        height=230,
        title=_title(
            title="Removing the Kalman filter changes the spring excess by little, and removing a correction head raises it",
            subtitle="April-to-June daylight half-hours. Dot: estimate. Line: 95% interval from resampling whole weeks (clipped at -2.5). Dashed: zero and the 30% bar.",
        ),
    )
    _save(chart, name="fig4_layers")


def main() -> None:
    """Draw every figure."""
    timing = json.loads((common.WORK_DIR / "timing.json").read_text())
    correction = json.loads((common.WORK_DIR / "correction.json").read_text())
    weekly_gap(timing=timing)
    lead_hour(timing=timing)
    night_residual()
    layers(correction=correction)


if __name__ == "__main__":
    main()
