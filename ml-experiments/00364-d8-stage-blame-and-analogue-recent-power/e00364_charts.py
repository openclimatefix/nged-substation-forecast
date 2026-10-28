# ruff: noqa: E501
"""Draw the figures of experiment 364 from `work/scores.json` and the `work/wide_*.parquet` frames.

Time series use a day count (1 to 7) on the axis, never calendar dates.
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00364-d8-stage-blame-and-analogue-recent-power"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
STAGE_LABELS: Final[dict[str, str]] = {
    "S0": "S0 plain GAM",
    "S1": "S1 + Kalman level",
    "S2": "S2 + Kalman transient",
    "S3": "S3 + night offset",
    "S4": "S4 + recipe R (GAM stack)",
}
REFERENCE_LABELS: Final[dict[str, str]] = {
    "AN": "Analogue member",
    "W": "Weekly XGBoost member",
    "BG2_reblend": "BG2 (blend)",
}
STAGE_COLOUR: Final[str] = ocf.BRAND_ORANGE
REFERENCE_COLOUR: Final[str] = ocf.DATA_BLUE
WIDTH: Final[int] = CONTENT_WIDTH_PX - 200
LEAD_BANDS: Final[tuple[str, ...]] = ("0-24h", "24-72h", "72-168h")


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a figure as SVG next to the page."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def interval_chart(
    *, rows: pl.DataFrame, order: list[str], x_title: str, reference: float | None
) -> alt.Chart:
    """Return a dot-and-interval panel with one row per label and a colour per kind."""
    base = alt.Chart(rows).encode(y=alt.Y("label:N", sort=order, title=None))
    colour = alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=["GAM stack stage", "BG2 reference"], range=[STAGE_COLOUR, REFERENCE_COLOUR]
        ),
        legend=None,
    )
    layers = [
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=x_title, scale=alt.Scale(zero=False)), x2="high:Q", color=colour
        ),
        base.mark_point(filled=True, size=70).encode(x="value:Q", color=colour),  # ty: ignore[unresolved-attribute]
    ]
    if reference is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"x": [reference]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(width=WIDTH, height=22 * len(order))  # ty: ignore[invalid-return-type]


def headline(*, scores: dict) -> None:
    """Figure 1: score and correlation per stage, with the members as references."""
    arms = scores["D8_weekly"]["arms"]["all"]
    names = [*STAGE_LABELS, *REFERENCE_LABELS]
    labels = {**STAGE_LABELS, **REFERENCE_LABELS}
    panels = []
    for metric, title in (
        ("score", "Quantile score (points of capacity; smaller is better)"),
        ("corr", "Correlation of median forecast with actual demand (larger is better)"),
    ):
        rows = pl.DataFrame(
            [
                {
                    "label": labels[n],
                    "value": arms[n][metric][0],
                    "low": arms[n][metric][1],
                    "high": arms[n][metric][2],
                    "kind": "GAM stack stage" if n in STAGE_LABELS else "BG2 reference",
                }
                for n in names
            ]
        )
        panels.append(
            interval_chart(
                rows=rows, order=[labels[n] for n in names], x_title=title, reference=None
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Each stage of the GAM stack adds 0 to 0.1 points at D8;",
                "BG2's members sit well below all of them",
            ],
            subtitle=[
                "Dot: estimate. Line: 95% interval from resampling whole valid months.",
                "96,986 rows, weekly refits. Orange: stack stage; blue: BG2 member.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_1_stages_score_and_correlation")


def lead_and_steps(*, scores: dict) -> None:
    """Figure 2: score and correlation by lead band, and on days with and without detected steps."""
    slices = [*LEAD_BANDS, "step_days", "no_step_days"]
    pretty = {
        "step_days": "Days with detected steps",
        "no_step_days": "Days without",
        **{b: f"Lead {b}" for b in LEAD_BANDS},
    }
    rows = []
    for sl in slices:
        for n in [*STAGE_LABELS, "AN"]:
            a = scores["D8_weekly"]["arms"][sl][n]
            rows.append(
                {
                    "slice": pretty[sl],
                    "arm": STAGE_LABELS.get(n, "Analogue member"),
                    "score": a["score"][0],
                    "corr": a["corr"][0],
                }
            )
    frame = pl.DataFrame(rows)
    order = [STAGE_LABELS[s] for s in STAGE_LABELS] + ["Analogue member"]
    domain = order
    colours = [
        ocf.BRAND_ORANGE_LIGHT,
        ocf.BRAND_ORANGE_LIGHT,
        ocf.BRAND_ORANGE_LIGHT,
        ocf.BRAND_ORANGE_LIGHT,
        ocf.BRAND_ORANGE,
        ocf.DATA_BLUE,
    ]
    panels = []
    for metric, title in (
        ("score", "Quantile score (points; smaller is better)"),
        ("corr", "Correlation (larger is better)"),
    ):
        panels.append(
            alt.Chart(frame)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=60)
            .encode(
                y=alt.Y("slice:N", sort=[pretty[s] for s in slices], title=None),
                x=alt.X(f"{metric}:Q", title=title, scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    sort=order,
                    scale=alt.Scale(domain=domain, range=colours),
                    legend=alt.Legend(orient="top", columns=3, title=None),
                ),
                shape=alt.Shape(
                    "arm:N",
                    sort=order,
                    scale=alt.Scale(
                        domain=domain,
                        range=["circle", "square", "diamond", "triangle-up", "cross", "circle"],
                    ),
                    legend=None,
                ),
            )
            .properties(width=WIDTH, height=130)
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            ["Stage scores by lead band and by days with detected steps", "(D8, weekly refits)"],
            subtitle=["Detected steps come from experiment 361's detector."],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_2_lead_and_step_days")


def monthly(*, scores: dict) -> None:
    """Figure 3: monthly score of S1, S4, and the analogue member."""
    rows = []
    for month, v in scores["D8_weekly"]["monthly"].items():
        for n, label in (
            ("S1", STAGE_LABELS["S1"]),
            ("S4", STAGE_LABELS["S4"]),
            ("AN", "Analogue member"),
        ):
            rows.append(
                {"month": month, "arm": label, "score": v[n]["score"], "corr": v[n]["corr"]}
            )
    frame = pl.DataFrame(rows)
    domain = [STAGE_LABELS["S1"], STAGE_LABELS["S4"], "Analogue member"]
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=domain, range=[ocf.BRAND_ORANGE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_BLUE]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    panels = [
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{m}:Q", title=t, scale=alt.Scale(zero=False)),
            color=colour,
        )
        .properties(width=WIDTH, height=150)
        for m, t in (
            ("score", "Quantile score (smaller is better)"),
            ("corr", "Correlation (larger is better)"),
        )
    ]
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            ["Monthly score and correlation at D8 (weekly refits)"], anchor="start"
        )
    )
    save(chart=chart, name="figure_3_monthly")


def placebos(*, scores: dict) -> None:
    """Figure 4: the stage split at the placebo Primaries D2 and D4, as score minus S0."""
    rows = []
    for key, label in (("D2_monthly", "D2"), ("D4_monthly", "D4"), ("D8_monthly", "D8")):
        base = scores[key]["arms"]["all"]["S0"]["score"][0]
        for n in STAGE_LABELS:
            v = scores[key]["arms"]["all"][n]["score"][0] - base
            rows.append({"site": label, "stage": STAGE_LABELS[n], "delta": v})
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "stage:N",
                sort=list(STAGE_LABELS.values()),
                title=None,
                axis=alt.Axis(labelAngle=-30),
            ),
            y=alt.Y("delta:Q", title="Score minus plain GAM (points; below zero is better)"),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(
                    domain=["D2", "D4", "D8"], range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                [
                    "Stage by stage at the placebo Primaries D2 and D4,",
                    "and at D8 (monthly refits)",
                ],
                subtitle=[
                    "D2 and D4 use the stored spread offsets; D8 uses the own-residual band."
                ],
                anchor="start",
            ),
        )
    )
    save(chart=chart, name="figure_4_placebos")


def analogue(*, scores: dict) -> None:
    """Figure 5: the analogue member with and without recent power."""
    arms = scores["D8_weekly"]["arms"]["all"]
    labels = {
        "AN": "Analogue member (stored)",
        "AN_control": "Analogue refit, unchanged",
        "AN_nolag": "Analogue without origin lags",
        "AN_norecent": "Analogue without recent power",
        "M": "Monthly XGBoost (no analogue columns)",
        "S4": "GAM stack",
    }
    panels = []
    for metric, title in (
        ("score", "Quantile score (points; smaller is better)"),
        ("corr", "Correlation (larger is better)"),
    ):
        rows = pl.DataFrame(
            [
                {
                    "label": lab,
                    "value": arms[n][metric][0],
                    "low": arms[n][metric][1],
                    "high": arms[n][metric][2],
                    "kind": "GAM stack stage" if n == "S4" else "BG2 reference",
                }
                for n, lab in labels.items()
            ]
        )
        panels.append(
            interval_chart(rows=rows, order=list(labels.values()), x_title=title, reference=None)
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            ["The analogue member with its recent-power inputs removed, at D8"],
            subtitle=[
                "Dot: estimate. Line: 95% interval from resampling whole valid months.",
                "Weekly GAM stack, 96,986 rows. Orange: GAM stack; blue: BG2 member.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_5_analogue_recent_power")


def pick_weeks(*, wide: pl.DataFrame) -> dict[str, pl.Datetime]:
    """Return the start of one week per case: steps, calm, cold winter nights, sunny April to June."""
    daily = (
        wide.filter((pl.col("lead_hours") > 12) & (pl.col("lead_hours") <= 36))
        .with_columns(week=pl.col("valid_time").dt.truncate("1w"))
        .group_by("week")
        .agg(
            rows=pl.len(),
            steps=pl.col("step_day").sum(),
            cold=pl.col("cold_night").sum(),
            sunny=pl.col("sunny").sum(),
            absdev=(pl.col("S4__q50") - pl.col("actual")).abs().mean(),
            month=pl.col("month").first(),
        )
        .filter(pl.col("rows") >= 280)
        .sort("week")
    )
    steps = daily.sort("steps", descending=True)["week"][0]
    calm = daily.filter(pl.col("steps") == 0).sort("absdev")["week"][0]
    cold = daily.filter(pl.col("month").is_in(["2025-12", "2026-01", "2026-02"])).sort(
        "cold", descending=True
    )["week"][0]
    sunny = daily.filter(pl.col("month").is_in(["2026-04", "2026-05", "2026-06"])).sort(
        "sunny", descending=True
    )["week"][0]
    return {
        "A week with detected steps": steps,
        "A calm week": calm,
        "A cold winter-night week": cold,
        "A sunny April-to-June week": sunny,
    }


def series_figures() -> None:
    """Figures 6 to 9: each stage against the actual demand over four chosen weeks; and the analogue arms."""
    wide = pl.read_parquet(WORK / "wide_D8_weekly.parquet")
    wide = wide.with_columns(
        step_day=pl.Series(
            np.asarray(json.loads((WORK / "step_day_flags.json").read_text()), dtype=bool)
        )
    )
    weeks = pick_weeks(wide=wide)
    (WORK / "weeks.json").write_text(json.dumps({k: str(v) for k, v in weeks.items()}))
    stage_arms = list(STAGE_LABELS)
    for i, (case, start) in enumerate(weeks.items(), start=6):
        week = (
            wide.filter(
                (pl.col("lead_hours") > 12)
                & (pl.col("lead_hours") <= 36)
                & (pl.col("valid_time") >= start)  # ty: ignore[unsupported-operator]
                & (pl.col("valid_time") < start + pl.duration(days=7))  # ty: ignore[unsupported-operator]
            )
            .sort("valid_time")
            .with_columns(day=(pl.col("valid_time") - start).dt.total_seconds() / 86_400 + 1)  # ty: ignore[unsupported-operator]
        )
        long = pl.concat(
            [
                week.select(
                    "day",
                    pl.lit(STAGE_LABELS[s]).alias("panel"),
                    pl.col(f"{s}__q50").alias("forecast"),
                    "actual",
                )
                for s in stage_arms
            ]
            + (
                [
                    week.select(
                        "day",
                        pl.lit("Analogue member").alias("panel"),
                        pl.col("AN__q50").alias("forecast"),
                        "actual",
                    )
                ]
            )
            + (
                [
                    week.select(
                        "day",
                        pl.lit("Analogue without recent power").alias("panel"),
                        pl.col("AN_norecent__q50").alias("forecast"),
                        "actual",
                    )
                ]
                if "AN_norecent__q50" in week.columns
                else []
            )
        )
        order = list(long["panel"].unique(maintain_order=True))
        base = alt.Chart(long).encode(
            x=alt.X(
                "day:Q",
                title="Day of the week (1 to 7)",
                scale=alt.Scale(domain=[1, 8]),
                axis=alt.Axis(values=[1, 2, 3, 4, 5, 6, 7]),
            )
        )
        actual = base.mark_line(color="#555555", strokeWidth=1.2, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("actual:Q", title=None)
        )
        forecast = base.mark_line(  # ty: ignore[unresolved-attribute]
            color=ocf.BRAND_ORANGE, strokeWidth=1.4, clip=True, aria=False
        ).encode(y="forecast:Q")
        chart = (
            alt.layer(actual, forecast)
            .properties(width=WIDTH, height=90)
            .facet(
                row=alt.Row(
                    "panel:N",
                    sort=order,
                    title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
                )
            )
            .properties(
                title=alt.TitleParams(
                    [
                        f"{case}: forecast (orange) made 12 to 36 hours ahead",
                        "against actual demand (grey)",
                    ],
                    subtitle=[
                        "Demand in points of capacity (0 to 100). One panel per stage of the GAM stack,",
                        "then the analogue member. Gaps are hours outside the 12 to 36 hour window.",
                    ],
                    anchor="start",
                )
            )
        )
        save(chart=chart, name=f"figure_{i}_{['steps', 'calm', 'cold', 'sunny'][i - 6]}_week")


def main() -> None:
    """Draw every figure."""
    scores = json.loads((WORK / "scores.json").read_text())
    headline(scores=scores)
    lead_and_steps(scores=scores)
    monthly(scores=scores)
    placebos(scores=scores)
    analogue(scores=scores)
    series_figures()


if __name__ == "__main__":
    main()
