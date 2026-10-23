# ruff: noqa: E501, RUF028
"""Draw the figures of experiment 316 from the stage 0 and stage 1 reports.

uv run --no-sync python ml-experiments/00316-*/e00316_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00316_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / "00316-primary-demand-night-level-and-single-input-terms"
)  # fmt: skip
WIDTH: Final[int] = 640
HEAD_LABELS: Final[dict[str, str]] = {
    "P": "Pinball band head (experiment 314)",
    "M": "Mixture band head (experiment 311)",
    "P_width_only": "Pinball head, width change only",
    "P_centre_only": "Pinball head, centre shift only",
    "offset": "As-of night offset, no head",
}


def _chart(data: pl.DataFrame) -> Any:
    return alt.Chart(data)


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(common.WORK_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def figure_d10_gain(*, report: dict) -> None:
    """Draw D10's gain per arm on all, night, and day half-hours."""
    rows = [
        {"arm": HEAD_LABELS[a], "part": part, **report["d10_gain"][a][key]}
        for a in HEAD_LABELS
        for part, key in (
            ("All half-hours", "all"),
            ("Night (21:00-07:30 UTC)", "night"),
            ("Day", "day"),
        )
    ]
    data = pl.DataFrame(rows).select("arm", "part", "est", "lo", "hi")
    order = list(HEAD_LABELS.values())
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
    base = _chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color(
            "part:N",
            scale=alt.Scale(
                domain=["All half-hours", "Night (21:00-07:30 UTC)", "Day"], range=colours
            ),
            title=None,
            legend=alt.Legend(orient="top"),
        ),  # fmt: skip
        yOffset="part:N",
    )
    chart = (
        alt.layer(
            base.mark_rule().encode(x=alt.X("lo:Q", title="Gain at D10 (points of quantile score; larger is better)"), x2="hi:Q"),  # fmt: skip
            base.mark_point(filled=True, size=50).encode(x="est:Q"),
            _chart(pl.DataFrame({"z": [0.0]})).mark_rule(color="#888", strokeDash=[3, 3]).encode(x="z:Q"),
        )
        .properties(width=WIDTH, height=230, title=_title(
            title="Narrower night-time bands, not a level shift, give D10 its gain",
            subtitle="Gain = default's quantile score minus the arm's, on D10's 96,344 half-hours of the scored year, leads 0.5 to 168 hours. Dot: estimate. Line: 95% interval from resampling 12 calendar months. Dashed rule: no change.",  # fmt: skip
        ))
    )  # fmt: skip
    _save(chart, name="figure_1_d10_gain_by_arm")


def figure_d10_hours(*, report: dict) -> None:
    """Draw D10's default residual by UTC hour and the heads' gain by local hour."""
    resid = pl.DataFrame(
        {
            "hour": list(range(24)),
            "value": [report["d10_default_residual_by_utc_hour"][str(h)] for h in range(24)],
        }
    )
    top = (
        _chart(resid).mark_line(color=ocf.BRAND_ORANGE, point=True)
        .encode(x=alt.X("hour:Q", title="Hour of day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
                y=alt.Y("value:Q", title="Mean actual minus median (points)"))
        .properties(width=WIDTH, height=140, title=_title(
            title="The default forecast sits above the actual demand at D10 at every hour, by 0.2 to 2.0 points",
            subtitle="Mean of actual minus Primary demand research default 1's median at D10, by UTC hour, scored year.",  # fmt: skip
        ))
    )  # fmt: skip
    gain = pl.DataFrame(
        [
            {"hour": int(h), "arm": HEAD_LABELS[a], "value": v}
            for a in ("P", "M")
            for h, v in report["by_local_hour"][a].items()
        ]
    )
    bottom = (
        _chart(gain).mark_line(point=True)
        .encode(x=alt.X("hour:Q", title="Hour of day (UK local time)", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
                y=alt.Y("value:Q", title="Gain (points; larger is better)"),
                color=alt.Color("arm:N", scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.DATA_PURPLE]), title=None, legend=alt.Legend(orient="top")))  # fmt: skip
        .properties(width=WIDTH, height=160, title=_title(
            title="The band heads gain from the evening to the early morning",
            subtitle="Mean gain at D10 by UK local hour, scored year.",
        ))
    )  # fmt: skip
    _save(alt.vconcat(top, bottom), name="figure_2_d10_hours")


def figure_grafts(*, report: dict) -> None:
    """Draw the share of the refit's advantage that each graft recovers."""
    rows = [
        {"group": g, "graft": k, **v}
        for g, entry in report.items()
        for k, v in entry["S_level_matched"].items()
        if g in ("D5+D9 worst quarter", "D5+D9 all daylight", "D10 worst quarter")
    ]
    data = (
        pl.DataFrame(rows)
        .select("group", "graft", "est", "lo", "hi")
        .with_columns(pl.col("lo", "hi", "est").clip(-2.0, 2.0))
    )
    order = list(next(iter(report.values()))["S_level_matched"])
    base = _chart(data).encode(
        y=alt.Y("graft:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]),
            title=None,
            legend=alt.Legend(orient="top"),
        ),  # fmt: skip
        yOffset="group:N",
    )
    chart = (
        alt.layer(
            base.mark_rule().encode(x=alt.X("lo:Q", title="Share of the refit's advantage recovered (1 = all)", scale=alt.Scale(domain=[-2, 2]), axis=alt.Axis(labelOverlap=True)), x2="hi:Q"),  # fmt: skip
            base.mark_point(filled=True, size=50).encode(x="est:Q"),
            _chart(pl.DataFrame({"z": [0.0, 1.0]})).mark_rule(color="#888", strokeDash=[3, 3]).encode(x="z:Q"),
        )
        .properties(width=WIDTH, height=260, title=_title(
            title="On the worst April-to-June half-hours, grafting the sunshine response recovers the most of any family",
            subtitle="Share S of the depth-1 refit's advantage over arm D recovered by shifting arm D's quantiles by the refit's family contribution minus a surrogate's, each averaged to zero. Line: 95% interval from resampling ISO weeks, clipped at -2 and 2.",  # fmt: skip
        ))
    )  # fmt: skip
    _save(chart, name="figure_3_graft_shares")


def figure_curves() -> None:
    """Draw the refit's family contributions against the surrogate of arm D."""
    frame = pl.read_parquet(common.WORK_DIR / "stage1_frame.parquet").filter(
        pl.col("site").is_in(["D5", "D9"]) & pl.col("daylight")
    )
    elevation = 90.0 - frame["solar_zenith"]
    frame = frame.with_columns(elevation=elevation)
    bands = [(10, 20), (25, 35), (40, 50), (55, 65)]
    pieces = []
    for lo, hi in bands:
        sub = frame.filter((pl.col("elevation") >= lo) & (pl.col("elevation") < hi)).with_columns(
            bin=(pl.col("shortwave") // 50 * 50), band=pl.lit(f"Sun {lo} to {hi} degrees")
        )
        pieces.append(sub)
    solar = (
        pl.concat(pieces)
        .group_by(["band", "bin"])
        .agg(refit=pl.col("cr_solar").mean(), arm_d_surrogate=pl.col("cs_solar").mean(), n=pl.len())
        .filter(pl.col("n") >= 30)
    )
    long = solar.unpivot(index=["band", "bin", "n"], on=["refit", "arm_d_surrogate"], variable_name="curve", value_name="value")  # fmt: skip
    chart1 = (
        _chart(long).mark_line(point=True)
        .encode(
            x=alt.X("bin:Q", title="Ensemble shortwave radiation (W/m2, 50 W/m2 bins)"),
            y=alt.Y("value:Q", title="Solar-family contribution (points)"),
            color=alt.Color("band:N", scale=alt.Scale(range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE]), title=None, legend=alt.Legend(orient="top")),  # fmt: skip
            strokeDash=alt.StrokeDash("curve:N", scale=alt.Scale(domain=["refit", "arm_d_surrogate"], range=[[1, 0], [4, 3]]), title=None),  # fmt: skip
        )
        .properties(width=WIDTH, height=240, title=_title(
            title="The sunshine response of the refit and of the arm D surrogate, by sun height",
            subtitle="Mean contribution to the median at D5 and D9, April to June 2026 daylight half-hours. Solid: depth-1 refit. Dashed: depth-1 surrogate of arm D's median.",  # fmt: skip
        ))
    )  # fmt: skip
    hours = frame.group_by("hour_local").agg(refit=pl.col("cr_calendar").mean(), arm_d_surrogate=pl.col("cs_calendar").mean()).sort("hour_local")  # fmt: skip
    long_h = hours.unpivot(index="hour_local", variable_name="curve", value_name="value")
    chart2 = (
        _chart(long_h).mark_line(point=True)
        .encode(
            x=alt.X("hour_local:Q", title="Hour of day (UK local time)", scale=alt.Scale(domain=[4, 22]), axis=alt.Axis(values=[4, 8, 12, 16, 20])),  # fmt: skip
            y=alt.Y("value:Q", title="Calendar-family contribution (points)"),
            color=alt.Color("curve:N", scale=alt.Scale(domain=["refit", "arm_d_surrogate"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), title=None, legend=alt.Legend(orient="top")),  # fmt: skip
        )
        .properties(width=WIDTH, height=160, title=_title(
            title="The calendar response of the refit and of the arm D surrogate",
            subtitle="Mean contribution to the median at D5 and D9, daylight half-hours only are drawn.",
        ))
    )  # fmt: skip
    frame = frame.filter(pl.col("lag_same_time_latest_day").is_not_nan())
    edges = np.unique(
        np.quantile(frame["lag_same_time_latest_day"].to_numpy(), np.linspace(0, 1, 21))
    )
    lag = frame.with_columns(
        bin=pl.Series(np.digitize(frame["lag_same_time_latest_day"].to_numpy(), edges[1:-1]))
    ).group_by("bin").agg(
        lag_value=pl.col("lag_same_time_latest_day").mean(), refit=pl.col("cr_lags").mean(), arm_d_surrogate=pl.col("cs_lags").mean()
    ).sort("lag_value")  # fmt: skip
    long_l = lag.with_columns(
        pl.col("lag_value", "refit", "arm_d_surrogate").cast(pl.Float64)
    ).unpivot(index=["bin", "lag_value"], variable_name="curve", value_name="value")
    chart3 = (
        _chart(long_l).mark_line(point=True)
        .encode(
            x=alt.X("lag_value:Q", title="Power at the same time on the latest day (points; 20 equal-count bins)"),
            y=alt.Y("value:Q", title="Lag-family contribution (points)"),
            color=alt.Color("curve:N", scale=alt.Scale(domain=["refit", "arm_d_surrogate"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), title=None, legend=alt.Legend(orient="top")),  # fmt: skip
        )
        .properties(width=WIDTH, height=200, title=_title(
            title="The recent-power response of the refit and of the arm D surrogate",
            subtitle="Mean contribution of the three lag inputs to the median at D5 and D9, binned by the day lag.",
        ))
    )  # fmt: skip
    _save(
        alt.vconcat(chart1, chart3, chart2).resolve_scale(
            color="independent", strokeDash="independent"
        ),
        name="figure_4_term_curves",
    )


def main() -> None:
    """Draw every figure."""
    stage0 = json.loads((common.REPORT_DIR / "stage0.json").read_text())
    stage1 = json.loads((common.REPORT_DIR / "stage1.json").read_text())
    figure_d10_gain(report=stage0)
    figure_d10_hours(report=stage0)
    figure_grafts(report=stage1)
    figure_curves()


if __name__ == "__main__":
    main()
