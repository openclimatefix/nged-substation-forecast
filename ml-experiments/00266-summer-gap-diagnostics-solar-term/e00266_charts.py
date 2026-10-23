# ruff: noqa: E501
"""Draw the figures of experiment 266 from `work/report.json`.

uv run --no-sync python ml-experiments/00266-*/e00266_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00266_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WINDOW_COLOURS: Final[dict[str, str]] = {
    "April to June": ocf.BRAND_ORANGE,
    "July to September": ocf.DATA_BLUE,
    "November to March": ocf.DATA_SKY,
}
WINDOW_KEYS: Final[dict[str, str]] = {
    "AMJ": "April to June",
    "JAS": "July to September",
    "NDJFM": "November to March",
}


def _chart(data: pl.DataFrame) -> Any:
    return alt.Chart(data)


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


def dots(
    *,
    rows: list[dict],
    order: list[str],
    title: str,
    subtitle: str,
    xtitle: str,
    name: str,
    colour_title: str = "Period",
    height_per_row: int = 14,
) -> None:
    """Dot (estimate) and line (95% interval) per label, coloured by `group`, with a zero rule."""
    data = pl.DataFrame(rows)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    colour = alt.Color(
        "group:N",
        title=colour_title,
        scale=alt.Scale(
            domain=groups,
            range=[
                WINDOW_COLOURS.get(g, c)
                for g, c in zip(
                    groups,
                    (
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                    ),
                    strict=False,
                )
            ],
        ),
        legend=alt.Legend(orient="top"),
    )
    y = alt.Y("label:N", sort=order, title=None)
    offset = alt.YOffset("group:N", sort=groups) if len(groups) > 1 else alt.value(0)
    base = _chart(data)
    interval = base.mark_rule(strokeWidth=2).encode(
        y=y, yOffset=offset, x=alt.X("lo:Q", title=xtitle), x2="hi:Q", color=colour
    )
    point = base.mark_circle(size=40, opacity=1).encode(
        y=y, yOffset=offset, x="est:Q", color=colour
    )
    zero = (
        _chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(x="z:Q")
    )
    chart = (zero + interval + point).properties(
        width=WIDTH,
        height=height_per_row * len(order) * max(1, len(groups)) * 0.8 + 40,
        title=_title(title=title, subtitle=subtitle),
    )
    _save(chart.configure_view(stroke=None), name=name)


def months(*, report: dict) -> None:
    """Figure 1: the gap to BG2 by month."""
    labels = list(report["gate_monthly"])
    rows = [
        {"label": mo, "est": v["est"], "lo": v["lo"], "hi": v["hi"], "group": "Arm 1 minus BG2"}
        for mo, v in report["gate_monthly"].items()
    ]
    dots(
        rows=rows,
        order=labels,
        title="The best-guess stack trails BG2 only in April, May, and June",
        subtitle="Quantile score of arm 1 minus BG2 by month the forecast run started (points of effective capacity; negative means arm 1 is better). Dot: estimate. Line: 95% interval from resampling whole weeks. Six substations, July 2025 to June 2026.",
        xtitle="Arm 1 minus BG2 (points; more negative is better)",
        name="figure_1_gap_by_month",
        colour_title="",
    )


def where(*, report: dict) -> None:
    """Figure 2: where the April-to-June gap sits, by substation and half-hour kind."""
    rows, order = [], []
    for s in common.SITES:
        for kind in ("daylight", "night"):
            v = report["post_hoc"]["daylight_night_by_site_AMJ"][s][kind]
            rows.append(
                {
                    "label": s,
                    "est": v["est"],
                    "lo": v["lo"],
                    "hi": v["hi"],
                    "group": "Daylight half-hours" if kind == "daylight" else "Night half-hours",
                }
            )
        order.append(s)
    dots(
        rows=rows,
        order=order,
        title="In April to June the gap sits at D5 and D9 by day and at D10 by night",
        subtitle="Quantile score of arm 1 minus BG2, April to June 2026, by substation (anonymised) and by daylight (clear-sky irradiance above 20 W/m2) or night. Dot: estimate. Line: 95% interval from resampling whole weeks.",
        xtitle="Arm 1 minus BG2 (points; more negative is better)",
        name="figure_2_gap_by_site_and_daylight",
        colour_title="Half-hours",
        height_per_row=22,
    )


def by_hour(*, report: dict) -> None:
    """Figure 3: the gap by UTC hour for the three periods."""
    rows = []
    for w, label in WINDOW_KEYS.items():
        for k, v in report["d1"][w].items():
            if k.startswith("utc hour") and v["est"] == v["est"]:
                rows.append(
                    {
                        "hour": int(k[-2:]),
                        "est": v["est"],
                        "lo": v["lo"],
                        "hi": v["hi"],
                        "group": label,
                    }
                )
    data = pl.DataFrame(rows)
    domain = list(WINDOW_COLOURS)
    colour = alt.Color(
        "group:N",
        title="Period",
        scale=alt.Scale(domain=domain, range=list(WINDOW_COLOURS.values())),
        legend=alt.Legend(orient="top"),
    )
    base = _chart(data)
    band = base.mark_area(opacity=0.15).encode(
        x=alt.X(
            "hour:Q",
            title="Hour of day, UTC (daylight half-hours only)",
            axis=alt.Axis(values=[6, 8, 10, 12, 14, 16, 18]),
            scale=alt.Scale(domain=[6, 19]),
        ),
        y=alt.Y("lo:Q", title="Arm 1 minus BG2 (points; more negative is better)"),
        y2="hi:Q",
        color=colour,
    )
    line = base.mark_line(point=True).encode(x="hour:Q", y="est:Q", color=colour)
    zero = (
        _chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(y="z:Q")
    )
    chart = (zero + band + line).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="The April to June gap is largest from 09:00 to 15:00 UTC",
            subtitle="Quantile score of arm 1 minus BG2 by hour of the valid time, daylight half-hours only, six substations. Line: estimate. Shade: 95% interval from resampling whole weeks. Zero is level with BG2.",
        ),
    )
    _save(chart.configure_view(stroke=None), name="figure_3_gap_by_hour")


def ablations(*, report: dict) -> None:
    """Figure 4: ablations in April to June."""
    d6 = report["d6"]
    labels = [
        ("arm1_minus_bg2", "Arm 1 minus BG2"),
        ("d_minus_bg2", "Arm D minus BG2 (no head, no summary)"),
        ("a_s_minus_bg2", "A+S minus BG2 (no head)"),
        ("dh_minus_bg2", "A+H minus BG2 (no summary)"),
        ("arm1_minus_a_s (no head)", "Arm 1 minus A+S: the head's effect"),
        ("arm1_minus_dh (no summary)", "Arm 1 minus A+H: the summary's effect"),
        ("arm1_minus_d (neither)", "Arm 1 minus arm D: both together"),
    ]
    rows = [
        {
            "label": text,
            "est": d6["AMJ"][k]["est"],
            "lo": d6["AMJ"][k]["lo"],
            "hi": d6["AMJ"][k]["hi"],
            "group": "April to June",
        }
        for k, text in labels
    ]
    dots(
        rows=rows,
        order=[t for _, t in labels],
        title="The gap exists before the head and the summary are added, and both reduce it",
        subtitle="Quantile score differences, April to June 2026, six substations. Negative means the first arm is better. Dot: estimate. Line: 95% interval from resampling whole weeks.",
        xtitle="Difference in quantile score (points; more negative is better)",
        name="figure_4_ablations_april_to_june",
        colour_title="",
        height_per_row=26,
    )


def solar_term(*, report: dict) -> None:
    """Figure 5: the older model with and without the solar term, against BG2, by period."""
    rows, order = [], []
    for key, text in (
        ("00139_nopv_minus_bg2", "No solar term minus BG2"),
        ("00139_pin_minus_bg2", "Pinned solar capacity minus BG2"),
        ("00139_asof_minus_bg2", "As-of capacity minus BG2"),
        ("00139_hindsight_minus_bg2", "Hindsight capacity minus BG2"),
        ("00111_gam0_minus_bg2", "No Kalman filter minus BG2"),
        ("00111_gamk_minus_bg2", "With Kalman filter minus BG2"),
    ):
        order.append(text)
        for w, label in WINDOW_KEYS.items():
            v = report["d2"][w][key]
            rows.append(
                {"label": text, "est": v["est"], "lo": v["lo"], "hi": v["hi"], "group": label}
            )
    dots(
        rows=rows,
        order=order,
        title="The older model with the solar term has the same seasonal gap to BG2 as the stack",
        subtitle="Quantile score of the older convex model (experiments 139 and 111) minus BG2, by period, six substations. Positive means worse than BG2. Dot: estimate. Line: 95% interval from resampling whole weeks.",
        xtitle="Older model minus BG2 (points; more negative is better)",
        name="figure_5_older_model_solar_term",
        height_per_row=30,
    )


def capacity(*, report: dict) -> None:
    """Figure 6: the capacity correction the residual implies."""
    rows = []
    for s in common.SITES:
        for w, label in WINDOW_KEYS.items():
            v = report["d3"]["by_site_window"][s][w]["arm_d"]
            rows.append(
                {
                    "label": s,
                    "est": v["correction_pct"],
                    "lo": v["pct_lo"],
                    "hi": v["pct_hi"],
                    "group": label,
                }
            )
    dots(
        rows=rows,
        order=list(common.SITES),
        title="At D5, D9, and D16 the pinned solar capacity looks too high in April to June",
        subtitle="Capacity correction implied by arm D's residual on clear midday half-hours 12 to 36 hours ahead, in percent of the pinned capacity. Positive: capacity too low. Negative: too high. Dot: estimate. Line: 95% interval from resampling whole weeks.",
        xtitle="Implied capacity correction (% of pinned capacity)",
        name="figure_6_capacity_correction",
        height_per_row=34,
    )


def shape(*, report: dict) -> None:
    """Figure 7: the clear-midday residual, morning against afternoon."""
    rows = []
    for s in common.SITES:
        for part in ("morning", "afternoon"):
            v = report["d5"]["by_site_window"][s]["AMJ"][part]
            rows.append(
                {
                    "label": s,
                    "est": v["est"],
                    "lo": v["lo"],
                    "hi": v["hi"],
                    "group": "Before 12:00 UTC" if part == "morning" else "After 12:00 UTC",
                }
            )
    dots(
        rows=rows,
        order=list(common.SITES),
        title="On clear mornings in April to June arm 1 forecasts too much demand at five of six substations",
        subtitle="Mean of actual minus median forecast of arm 1, in points, on clear midday half-hours (unit solar shape at least 0.7, both clear-sky indices above 0.8), April to June 2026. Negative: forecast too high. Dot: estimate. Line: 95% interval from resampling whole weeks.",
        xtitle="Actual minus median forecast (points)",
        name="figure_7_clear_midday_residual",
        colour_title="Half-hours",
        height_per_row=30,
    )


def weather(*, report: dict) -> None:
    """Figure 8: the gap by whether the ensemble and realised clear-sky indices agree."""
    rows = []
    for w, label in WINDOW_KEYS.items():
        for k, text in (
            ("agree", "Agree (within 0.1)"),
            ("middle", "Between"),
            ("disagree", "Disagree (by more than 0.2)"),
        ):
            v = report["d4"][w][k]
            rows.append(
                {"label": text, "est": v["est"], "lo": v["lo"], "hi": v["hi"], "group": label}
            )
    dots(
        rows=rows,
        order=["Agree (within 0.1)", "Between", "Disagree (by more than 0.2)"],
        title="The April to June gap is the same where the ensemble sunshine forecast was right and where it was wrong",
        subtitle="Quantile score of arm 1 minus BG2 on daylight half-hours, split by the gap between the ensemble-mean and the realised clear-sky index. Dot: estimate. Line: 95% interval from resampling whole weeks.",
        xtitle="Arm 1 minus BG2 (points; more negative is better)",
        name="figure_8_ensemble_agreement",
        height_per_row=34,
    )


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.WORK_DIR / "report.json").read_text())
    for fn in (months, where, by_hour, ablations, solar_term, capacity, shape, weather):
        fn(report=report)


if __name__ == "__main__":
    main()
