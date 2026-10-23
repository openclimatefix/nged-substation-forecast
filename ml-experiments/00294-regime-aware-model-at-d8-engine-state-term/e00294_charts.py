# ruff: noqa: E501
"""Draw the figures of experiment 294 from the stored predictions and `score.json`.

uv run --no-sync python ml-experiments/00294-*/e00294_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00294_common as common
import e00294_score as sc
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.DATA_DIR / "png"
WIDTH: Final[int] = 560
LABELS: Final[dict[str, str]] = {
    "arm2": "Arm 2: experiment 191's rows",
    "arm3_indicator": "Break indicator only",
    "arm3": "Arm 3: engine state, last day",
    "arm3_matched": "Post hoc: trained on lagged state",
    "arm3_hindsight": "Engine state at the target (leaky)",
    "arm4_emp": "Arm 4: bands from past residuals",
    "arm4_2b": "Arm 4b: both tails widened",
}
COLOURS: Final[dict[str, str]] = {
    "arm2": "#7A7A7A",
    "arm3_indicator": ocf.DATA_SKY,
    "arm3": ocf.DATA_BLUE,
    "arm3_matched": ocf.DATA_PURPLE,
    "arm3_hindsight": ocf.BRAND_ORANGE,
    "arm4_emp": "#222222",
    "arm4_2b": "#B5B5B5",
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
        text=textwrap.wrap(title, 80),
        subtitle=textwrap.wrap(subtitle, 105),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _scale(keys: list[str]) -> alt.Scale:
    return alt.Scale(domain=[LABELS[k] for k in keys], range=[COLOURS[k] for k in keys])


def d8_losses() -> tuple[dict, np.ndarray]:
    """Return D8's per-row losses of every arm and the month of each row."""
    frames = {name: sc.load_arm(config=c, kind=k, site="D8") for name, (c, k) in sc.D8_ARMS.items()}
    frames["arm4_emp"], frames["arm4_2b"] = sc.load_e264(arm="arm_emp"), sc.load_e264(arm="arm2b")
    frames = sc.align(frames=frames)
    base = frames["arm1"]
    actual = base["actual"].to_numpy().astype(np.float64)
    thr, low = float(np.quantile(actual, 0.10)), float(np.quantile(actual, 0.02))
    return {
        a: sc.row_losses(frame=f, threshold=thr, low=low) for a, f in frames.items()
    }, sc.month_of(frame=base)


def headline() -> None:
    """Relative quantile score and twCRPS of each D8 arm against arm 1."""
    losses, months = d8_losses()
    rows = []
    for arm in LABELS:
        for metric, label in (
            ("qs", "Quantile score (CRPS stand-in)"),
            ("tw", "Lower-tail twCRPS"),
        ):
            r = sc.relative_contrast(
                losses=losses, arm=arm, reference="arm1", metric=metric, months=months
            )
            rows.append(
                {
                    "arm": LABELS[arm],
                    "metric": label,
                    "est": r["relative"] * 100,
                    "lo": r["low"] * 100,
                    "hi": r["high"] * 100,
                }
            )
    data = pl.DataFrame(rows)
    order = [LABELS[k] for k in LABELS]
    base = _chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color("arm:N", scale=_scale(list(LABELS)), legend=None),
    )
    rule = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X("lo:Q", title="Relative to arm 1, % (negative is better)"), x2="hi:Q"
            )
            + base.mark_point(filled=True, size=70).encode(x="est:Q")
            + rule
        )
        .properties(width=WIDTH - 250, height=170)
        .facet(
            row=alt.Row(
                "metric:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            )
        )
        .properties(
            title=_title(
                title="Only the arm that sees the engine's state at the target beats arm 1; the last-day state makes the forecast worse",
                subtitle="D8, scored year, leads 0.5 to 168 hours. Dot: estimate. Line: 98.75% interval from resampling whole months. Dashed line: same as arm 1 (the single model, whole history).",
            )
        )
    )
    _save(chart, name="figure_1_relative_score")


def coefficients(*, score: dict) -> None:
    """The engine coefficient of each monthly refit, in MW, against the recorded unit."""
    rows = []
    for r in score["checks"]["coefficients"]["D8"]:
        if "regF" not in r:
            continue
        rows += [
            {
                "cut": r["cut"],
                "series": "Per half-hour with 1+ units running",
                "mw": r["regF"][1] * 0.1,
            },
            {
                "cut": r["cut"],
                "series": "Per running unit (expected units)",
                "mw": r["regU"][1] * 0.1,
            },
            {
                "cut": r["cut"],
                "series": "On last-day state (trained lagged)",
                "mw": r["regP"][1] * 0.1,
            },
        ]
    data = pl.DataFrame(rows).with_columns(pl.col("cut").str.to_date())
    domain = [
        "Per half-hour with 1+ units running",
        "Per running unit (expected units)",
        "On last-day state (trained lagged)",
    ]
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]),
        legend=alt.Legend(orient="top", title=None, columns=1),
    )
    lines = (
        _chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("cut:T", title="Monthly refit date"),
            y=alt.Y("mw:Q", title="Fitted offset, MW", scale=alt.Scale(domain=[0, 4])),
            color=colour,
        )
    )
    unit = (
        _chart(pl.DataFrame({"y": [1.97]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    chart = (lines + unit).properties(
        width=WIDTH,
        height=220,
        title=_title(
            title="The fitted offset is steady across refits, and is smaller when the state is a day old",
            subtitle="D8, arm 3's engine column. Each point is one monthly refit on the whole history. Dashed line: one engine unit, 1.97 MW, from the engine's recorded output.",
        ),
    )
    _save(chart, name="figure_2_engine_offset")


def shortfall(*, score: dict) -> None:
    """Mean error (reading minus forecast) on engine-running and other half-hours."""
    rows = []
    for subset, label in (
        ("running", "Engine running (filtered probability above 0.5)"),
        ("ordinary", "Other half-hours"),
    ):
        for arm, row in score["scores"][f"D8/{subset}"].items():
            if arm in {"arm1", *LABELS}:
                rows.append(
                    {
                        "subset": label,
                        "arm": LABELS.get(arm, "Arm 1: single model"),
                        "err": row["mean_error"] * 0.1,
                    }
                )
    labels = {**LABELS, "arm1": "Arm 1: single model"}
    colours = {**COLOURS, "arm1": ocf.DATA_GREEN}
    scale = alt.Scale(domain=[labels[k] for k in labels], range=[colours[k] for k in labels])
    data = pl.DataFrame(rows)
    chart = (
        _chart(data)
        .mark_bar()
        .encode(
            y=alt.Y(
                "arm:N",
                sort=[labels[k] for k in ("arm1", *LABELS)],
                title=None,
                axis=alt.Axis(labelLimit=240),
            ),
            x=alt.X("err:Q", title="Reading minus forecast mean, MW"),
            color=alt.Color("arm:N", scale=scale, legend=None),
        )
        .properties(width=WIDTH - 250, height=170)
        .facet(
            row=alt.Row(
                "subset:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            )
        )
        .properties(
            title=_title(
                title="Arm 1 over-forecasts by 1.1 MW while the engine runs and under-forecasts by 0.8 MW otherwise",
                subtitle="D8, scored year. Mean of the reading minus the forecast mean. Zero means no bias.",
            )
        )
    )
    _save(chart, name="figure_3_shortfall_by_state")


def example_weeks() -> None:
    """Three weeks of D8 readings, forecasts and the engine state (one lead band)."""
    arms = {
        "arm1": "Arm 1",
        "arm3_matched": "Post hoc matched arm",
        "arm3_hindsight": "State at the target",
    }
    frames = {}
    for key, (c, k) in ((a, sc.D8_ARMS[a]) for a in arms):
        f = sc.load_arm(config=c, kind=k, site="D8").filter(
            (pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36)
        )
        frames[key] = f
    base = frames["arm1"].filter(
        (pl.col("valid_time") >= pl.datetime(2025, 12, 1, time_zone="UTC"))
        & (pl.col("valid_time") < pl.datetime(2025, 12, 22, time_zone="UTC"))
    )
    rows = []
    for key, label in arms.items():
        f = frames[key].join(base.select("run", "valid_time"), on=["run", "valid_time"])
        rows += [
            {"valid_time": t, "mw": m * 0.1, "series": label, "run": int(r)}
            for t, m, r in zip(f["valid_time"], f["mean"], f["run"], strict=True)
        ]
    rows += [
        {"valid_time": t, "mw": a * 0.1, "series": "Reading", "run": int(r)}
        for t, a, r in zip(base["valid_time"], base["actual"], base["run"], strict=True)
    ]
    shade = (
        base.select("valid_time", "p_engine_filtered")
        .with_columns((pl.col("p_engine_filtered") > 0.5).alias("on"))
        .filter("on")
        .select("valid_time")
    )
    domain = ["Reading", "Arm 1", "Post hoc matched arm", "State at the target"]
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=domain, range=["#222222", ocf.DATA_GREEN, ocf.DATA_PURPLE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", title=None, columns=2),
    )
    lines = (
        _chart(pl.DataFrame(rows))
        .mark_line(clip=True, strokeWidth=1.2)
        .encode(
            detail="run:N",
            x=alt.X("valid_time:T", title="Valid time, December 2025", axis=alt.Axis(format="%d")),
            y=alt.Y("mw:Q", title="D8 power, MW (forecast 12 to 36 hours ahead)"),
            color=colour,
        )
    )
    bands = (
        _chart(shade.with_columns((pl.col("valid_time") + pl.duration(minutes=30)).alias("end")))
        .mark_rect(opacity=0.15, color=ocf.BRAND_ORANGE)
        .encode(x="valid_time:T", x2="end:T", y=alt.value(0), y2=alt.value(240))
    )
    chart = (bands + lines).properties(
        width=WIDTH,
        height=240,
        title=_title(
            title="Only a forecast that sees the engine state at the target is pulled down during the shaded spells",
            subtitle="D8, 1 to 21 December 2025. Shading: half-hours when the detector's filtered probability of a running engine exceeds 0.5.",
        ),
    )
    _save(chart, name="figure_4_example_weeks")


def controls() -> None:
    """Arm 3 and the post hoc arm against arm 1 at the three Primaries without an engine."""
    rows = []
    for site in common.CONTROL_SITES:
        frames = sc.align(
            frames={
                n: sc.load_arm(config=c, kind=k, site=site) for n, (c, k) in sc.CONTROL_ARMS.items()
            }
        )
        actual = frames["arm1"]["actual"].to_numpy().astype(np.float64)
        losses = {
            a: sc.row_losses(
                frame=f,
                threshold=float(np.quantile(actual, 0.1)),
                low=float(np.quantile(actual, 0.02)),
            )
            for a, f in frames.items()
        }
        months = sc.month_of(frame=frames["arm1"])
        for arm in ("arm3_indicator", "arm3", "arm3_matched"):
            r = sc.relative_contrast(
                losses=losses, arm=arm, reference="arm1", metric="qs", months=months
            )
            rows.append(
                {
                    "site": site,
                    "arm": LABELS[arm],
                    "est": r["relative"] * 100,
                    "lo": r["low"] * 100,
                    "hi": r["high"] * 100,
                }
            )
    keys = ["arm3_indicator", "arm3", "arm3_matched"]
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=[LABELS[k] for k in keys], title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color("arm:N", scale=_scale(keys), legend=None),
    )
    rule = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X("lo:Q", title="Quantile score relative to arm 1, % (negative is better)"),
                x2="hi:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="est:Q")
            + rule
        )
        .properties(width=WIDTH - 250, height=90)
        .facet(
            row=alt.Row(
                "site:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            )
        )
        .properties(
            title=_title(
                title="At the three Primaries without an engine, the engine term moves the score by under 2%",
                subtitle="D5, D10, D11, scored year. Dot: estimate. Line: 98.75% interval from resampling whole months. Dashed line: same as arm 1 at that Primary.",
            )
        )
    )
    _save(chart, name="figure_5_controls")


def main() -> None:
    """Draw every figure."""
    score = json.loads(sc.OUT_PATH.read_text())
    headline()
    coefficients(score=score)
    shortfall(score=score)
    example_weeks()
    controls()


if __name__ == "__main__":
    main()
