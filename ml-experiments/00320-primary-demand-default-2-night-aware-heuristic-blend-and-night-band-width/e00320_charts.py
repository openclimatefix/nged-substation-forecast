# ruff: noqa: E501
"""Draw the figures of experiment 320 from `work/report/*.json` and `work/arms/fit_log.json`.

uv run --no-sync python ml-experiments/00320-*/e00320_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00320_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
BANDS: Final[tuple[str, ...]] = tuple(f"{a}-{a + 24} h" for a in range(0, 240, 24))
CENTRES: Final[dict[str, int]] = {b: i * 24 + 12 for i, b in enumerate(BANDS)}
NAMES: Final[dict[str, str]] = {
    "default": "Default (Primary demand research default 2)",
    "b1": "Default plus B1 (blend off on forecast cold nights)",
    "b2": "Default plus B2 (temperature-aware blend weight)",
    "n": "Default plus N (night band-width correction)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    NAMES["default"]: ocf.DATA_BLUE,
    NAMES["b1"]: ocf.DATA_GREEN,
    NAMES["b2"]: ocf.DATA_PURPLE,
    NAMES["n"]: ocf.DATA_SKY,
    NAMES["manual_heuristic"]: ocf.BRAND_ORANGE,
    NAMES["climatology"]: "#7A7A7A",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _colour(*, keys: tuple[str, ...] | None = None) -> Any:
    names = [NAMES[k] for k in keys] if keys else list(COLOURS)
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=names, range=[COLOURS[n] for n in names]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420, symbolType="stroke"),
    )


def _lead_axis() -> Any:
    return alt.X(
        "lead:Q",
        title="Lead time (hours; each point is a 24-hour band, plotted at its centre)",
        scale=alt.Scale(domain=[0, 240]),
        axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
    )


def headline(*, report: dict) -> None:
    """Quantile score against lead for every forecast, with 95% intervals."""
    data = pl.DataFrame(
        [
            {
                "arm": NAMES[key],
                "lead": CENTRES[b],
                **{k: report["by_band"][b][key][k] for k in ("est", "lo", "hi")},
            }
            for b in BANDS
            for key in NAMES
        ]
    )
    yt = "Mean quantile score (points of capacity; smaller is better)"
    base = _chart(data)
    x = _lead_axis()
    chart = (
        base.mark_errorbar(thickness=1.5).encode(
            x=x, y=alt.Y("lo:Q", title=yt, scale=alt.Scale(zero=False)), y2="hi:Q", color=_colour()
        )
        + base.mark_line(strokeWidth=2).encode(x=x, y=alt.Y("est:Q", title=yt), color=_colour())
    ).properties(
        width=WIDTH,
        height=300,
        title=_title(
            title="Both blends lower the quantile score from 168 to 240 hours; the band-width correction barely moves it",
            subtitle="Line: mean over six Primary substations, scored year 2025-07-02 to 2026-06-30. Bar: 95% interval from resampling the 12 months of forecast date. Below 168 hours B1 and B2 equal the default.",
        ),
    )
    _save(chart, name="headline_score_by_lead")


def gain_by_group(*, report: dict) -> None:
    """Each arm's score minus the default's by protected group, at 168 to 240 hours, with 99.5% intervals."""
    arms = (("b1", NAMES["b1"]), ("b2", NAMES["b2"]), ("n", NAMES["n"]))
    rows = []
    for g, windows in report["by_group"].items():
        e = windows.get("168-240 h")
        if e is None:
            continue
        for key, label in arms:
            c = e[f"{key}_vs_default"]
            rows.append(
                {
                    "group": g,
                    "arm": label,
                    "diff": c["diff"],
                    "lo": c["diff_lo_995"],
                    "hi": c["diff_hi_995"],
                }
            )
    data = pl.DataFrame(rows)
    yt = "Score minus the default's (points of capacity; negative is better)"
    base = _chart(data)
    y = alt.Y(
        "group:N",
        title=None,
        sort=[
            "all rows",
            "sunny daylight",
            "April to October",
            "winter (Nov-Mar)",
            "cold winter nights",
        ],
    )
    chart = (
        base.mark_rule(color="#999999", strokeDash=[3, 3])
        .encode(x=alt.X("zero:Q", title=yt))
        .transform_calculate(zero="0")
        + base.mark_errorbar(thickness=1.5).encode(
            y=y,
            x=alt.X("lo:Q", title=yt),
            x2="hi:Q",
            color=_colour(keys=("b1", "b2", "n")),
            yOffset="arm:N",
        )
        + base.mark_point(filled=True, size=50).encode(
            y=y, x=alt.X("diff:Q", title=yt), color=_colour(keys=("b1", "b2", "n")), yOffset="arm:N"
        )
    ).properties(
        width=WIDTH,
        height=260,
        title=_title(
            title="The blends help every group except cold winter nights, where the interval is wide and reaches above zero",
            subtitle="Dot: difference at 168 to 240 hours. Bar: 99.5% interval from resampling forecast months. Dashed line: same score as the default.",
        ),
    )
    _save(chart, name="gain_by_group_168_240")


def band_width_by_site(*, report: dict) -> None:
    """Band-width correction's score change by Primary, night half-hours only, with 95% intervals."""
    rows = []
    for site, v in report["n_by_site_night"].items():
        c = v["night"]
        rows.append({"site": site, "diff": c["diff"], "lo": c["diff_lo_95"], "hi": c["diff_hi_95"]})
    data = pl.DataFrame(rows)
    yt = (
        "Score minus the default's, night half-hours, 0.5 to 240 hours (points; negative is better)"
    )
    base = _chart(data)
    y = alt.Y("site:N", title="Primary substation (anonymised)")
    chart = (
        base.mark_rule(color="#999999", strokeDash=[3, 3])
        .encode(x=alt.X("zero:Q", title=yt))
        .transform_calculate(zero="0")
        + base.mark_errorbar(thickness=1.5, color=ocf.DATA_SKY).encode(
            y=y, x=alt.X("lo:Q", title=yt), x2="hi:Q"
        )
        + base.mark_point(filled=True, size=60, color=ocf.DATA_SKY).encode(
            y=y, x=alt.X("diff:Q", title=yt)
        )
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            title="Narrowing the night bands helps D10 and leaves the other five Primaries unchanged",
            subtitle="Night is 21:00 to 07:30 UTC. Dot: difference. Bar: 95% interval from resampling forecast months. Dashed line: same score as the default.",
        ),
    )
    _save(chart, name="band_width_gain_by_primary")


def forecast_flag_share(*, diagnostics: dict) -> None:
    """Share of realised cold winter nights that the forecast temperature also flags, by lead."""
    data = pl.DataFrame(
        [
            {"lead": CENTRES[b], "share": v["share_flagged_by_forecast"]}
            for b, v in diagnostics["by_band"].items()
        ]
    )
    chart = (
        _chart(data)
        .mark_line(point=True, color=ocf.DATA_GREEN, strokeWidth=2)
        .encode(
            x=_lead_axis(),
            y=alt.Y(
                "share:Q",
                title="Share flagged by the forecast rule",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                title="The forecast temperature finds 75% to 90% of cold winter nights within 4 days but 31% at 9 to 10 days",
                subtitle="Share of the 2,384 + 5,728 half-hours in the realised cold-winter-night group whose ensemble-mean 24-hour temperature at the origin is below 2 C.",
            ),
        )
    )
    _save(chart, name="forecast_flag_share_by_lead")


def blend_weight(*, report: dict, log: dict) -> None:
    """Chain weight of the logistic blend against temperature, and the best table weight by lead."""
    params = np.array([v["logistic"] for k, v in log.items() if k != "replicate"]).mean(axis=0)
    temps = np.arange(-4.0, 24.1, 1.0)
    rows = []
    for b, label in enumerate(("168-192 h", "192-216 h", "216-240 h")):
        for d, kind in enumerate(("day", "night")):
            z = (temps - common.TEMPERATURE_CENTRE_C) / common.TEMPERATURE_SCALE_C
            w = 1.0 / (1.0 + np.exp(-(params[b, d, 0] + params[b, d, 1] * z)))
            rows += [
                {"temp": t, "w": float(v), "line": f"{label}, {kind}"}
                for t, v in zip(temps, w, strict=True)
            ]
    chart = (
        _chart(pl.DataFrame(rows))
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("temp:Q", title="Ensemble-mean 24-hour temperature at the origin (C)"),
            y=alt.Y(
                "w:Q",
                title="Weight on the convex chain (1 = chain alone)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("line:N", legend=alt.Legend(title=None, orient="top", columns=3)),
            strokeDash="line:N",
        )
        .properties(
            width=WIDTH,
            height=260,
            title=_title(
                title="B2 gives the chain more weight on cold days and the heuristic more on mild days",
                subtitle="Mean over the 12 monthly refits of the logistic chain weight, by lead band and by daylight or night.",
            ),
        )
    )
    _save(chart, name="blend_weight_by_temperature")
    expl = report["exploratory_weight_by_lead"]
    data = pl.DataFrame(
        [
            {"lead": CENTRES[b], "w": expl[f"{b} {s}"]["best_w"], "season": s}
            for b in BANDS
            for s in ("Nov-Mar", "Apr-Oct")
        ]
    )
    chart2 = (
        _chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=_lead_axis(),
            y=alt.Y(
                "w:Q", title="Best weight on the convex chain", scale=alt.Scale(domain=[0.4, 1.0])
            ),
            color=alt.Color(
                "season:N",
                scale=alt.Scale(
                    domain=["Nov-Mar", "Apr-Oct"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                title="The heuristic starts to help from the first day and helps most in April to October at 7 to 10 days",
                subtitle="Exploratory: weight on the convex chain that minimises the scored year's quantile score, per 24-hour band and season (fitted on the scored year itself, not as-of).",
            ),
        )
    )
    _save(chart2, name="best_weight_by_lead")


def tails(*, report: dict) -> None:
    """Share of outcomes below the q01 and q99 forecasts, by lead."""
    rows = [
        {
            "lead": CENTRES[b],
            "arm": NAMES[a],
            "level": lvl,
            "rate": report["tails_by_band"][b][a][key],
        }
        for b in BANDS
        for a in ("default", "b1", "n")
        for lvl, key in (("q01", "q01_below"), ("q99", "q99_below"))
    ]
    data = pl.DataFrame(rows)
    top = (
        _chart(data.filter(pl.col("level") == "q01"))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=_lead_axis(),
            y=alt.Y(
                "rate:Q",
                title="Share of outcomes below q01 (target 1%)",
                axis=alt.Axis(format=".1%"),
            ),
            color=_colour(),
        )
        .properties(width=WIDTH, height=170)
    )
    bottom = (
        _chart(data.filter(pl.col("level") == "q99"))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=_lead_axis(),
            y=alt.Y(
                "rate:Q",
                title="Share below q99 (target 99%)",
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(zero=False),
            ),
            color=_colour(),
        )
        .properties(width=WIDTH, height=170)
    )
    chart = alt.vconcat(top, bottom).properties(
        title=_title(
            title="Tail coverage stays close to target at every lead; narrowing the bands lets more outcomes fall below q01",
            subtitle="Six Primary substations, scored year. A perfect forecast has 1% of outcomes below q01 and 99% below q99.",
        )
    )
    _save(chart, name="tail_coverage_by_lead")


def main() -> None:
    """Draw every figure."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    diagnostics = json.loads((common.REPORT_DIR / "diagnostics.json").read_text())
    log = json.loads((common.ARMS_DIR / "fit_log.json").read_text())
    headline(report=report)
    gain_by_group(report=report)
    band_width_by_site(report=report)
    forecast_flag_share(diagnostics=diagnostics)
    blend_weight(report=report, log=log)
    tails(report=report)


if __name__ == "__main__":
    main()
