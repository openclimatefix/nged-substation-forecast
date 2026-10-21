"""Draw the figures of experiment 160's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00160-*/e00160_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00160_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GREY: Final[str] = "#808080"
ARM_COLOURS: Final[dict[str, str]] = {
    "Reference (stored)": GREY,
    "I: interpolated": ocf.DATA_BLUE,
    "M: non-decreasing": ocf.BRAND_ORANGE,
    "IM: both": ocf.DATA_GREEN,
}
ARM_ORDER: Final[list[str]] = list(ARM_COLOURS)
PERIOD_COLOURS: Final[dict[str, str]] = {
    "replicate months": ocf.DATA_SKY,
    "scored year": ocf.BRAND_ORANGE,
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _colour() -> alt.Color:
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=ARM_ORDER, range=list(ARM_COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The planned contrasts over all leads, with 95% intervals."""
    names = {
        "C1": "C1 I minus reference (interpolated)",
        "C2": "C2 M minus reference (non-decreasing)",
        "C3": "C3 IM minus reference (both)",
    }
    rows = []
    for period, label in (("replicate", "replicate months"), ("scored", "scored year")):
        for key, name in names.items():
            c = data[period]["contrasts"][key]["all"]["qs"]
            rows.append(
                {
                    "contrast": name,
                    "period": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    marks = alt.Chart(pl.DataFrame({"x": [0.0, 0.01], "kind": ["same score", "pass limit"]}))
    zero = marks.mark_rule(color=GREY, strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Quantile score minus the reference's (points; below zero is better)",
            scale=alt.Scale(domain=[-0.012, 0.012]),
        )
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 260,
        height=160,
        title=_title(
            text="Smooth reading of the spread table costs under 0.01 points of quantile score",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours, five "
                "decision substations (D5, D10, D11, D14, D16). Line: 95% interval from "
                "resampling whole calendar months. Dashed rules: no change (0) and the pre-"
                "registered pass limit (+0.01)."
            ),
        ),
    )


def figure_width(*, data: dict) -> alt.TopLevelMixin:
    """Median q05-q95 width by lead day, one panel per decision substation, scored year."""
    rows = []
    for arm, per_site in data["scored"]["widths"].items():
        for site, values in per_site.items():
            for day, width in zip(values["lead_day"], values["median_width"], strict=True):
                rows.append(
                    {"arm": common.ARM_LABELS[arm], "site": site, "day": day, "width": width}
                )
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True, strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Lead day (0 is up to 24 hours ahead; 6 is 144 to 168 hours)",
                axis=alt.Axis(values=list(range(7))),
            ),
            y=alt.Y("width:Q", title="Median q05-q95 width (points)", scale=alt.Scale(zero=False)),
            color=_colour(),
        )
        .properties(width=WIDTH - 90, height=90)
        .facet(row=alt.Row("site:N", title=None, sort=list(common.DECISION_SITES)))
        .resolve_scale(y="independent")
    )
    return chart.properties(
        title=_title(
            text="Interpolating removes the flat days 2 to 5",
            subtitle=(
                "Median width of the 90% band (q05 to q95) by lead day, scored year, in points of "
                "effective capacity, one panel per decision substation. The reference is flat "
                "from day 2 to day 5 at every substation."
            ),
        )
    )


def figure_coverage(*, data: dict) -> alt.TopLevelMixin:
    """Coverage of the q05-q95 band and of q95 by lead day, mean over substations, scored year."""
    metrics = {
        "Share of actuals inside the q05-q95 band (nominal 0.90)": ("inside", 0.90),
        "Share of actuals at or below q95 (nominal 0.95)": ("cov_q95", 0.95),
    }
    panels = []
    for title, (field, nominal) in metrics.items():
        rows = [
            {"arm": common.ARM_LABELS[arm], "site": site, "day": day, "value": value}
            for arm, per_site in data["scored"]["widths"].items()
            for site, values in per_site.items()
            for day, value in zip(values["lead_day"], values[field], strict=True)
        ]
        frame = pl.DataFrame(rows).group_by(["arm", "day"]).agg(pl.col("value").mean())
        lines = (
            alt.Chart(frame)
            .mark_line(point=True, strokeWidth=2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:Q", title="Lead day", axis=alt.Axis(values=list(range(7)))),
                y=alt.Y("value:Q", title=title.split(" (")[0], scale=alt.Scale(zero=False)),
                color=_colour(),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"value": [nominal]}))
            .mark_rule(color="#333333", strokeDash=[4, 3])
            .encode(y="value:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((lines + rule).properties(width=WIDTH - 90, height=130, title=title))
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Coverage by lead day is similar in every variant",
            subtitle=(
                "Mean over the five decision substations, scored year. Dashed rule: the share a "
                "calibrated forecast would give."
            ),
        )
    )


def figure_tails(*, data: dict) -> alt.TopLevelMixin:
    """Tail coverage at four quantiles over all leads and on cold nights, scored year."""
    arm_of = {"ref": "Reference (stored)", "i": "I: interpolated", "m": "M: non-decreasing"}
    arm_of["im"] = "IM: both"
    order = ["q01", "q05", "q95", "q99"]
    nominal = pl.DataFrame({"quantile": order, "nominal": [0.01, 0.05, 0.95, 0.99]})
    panels = []
    for subset, title in (
        ("all leads", "All rows"),
        ("cold nights (day mean below 2 C)", "Cold winter nights (day mean below 2 C)"),
    ):
        rows = [
            {"arm": arm_of[row["arm"]], "quantile": q, "coverage": row[f"cov_{q}"]}
            for row in data["scored"]["tables"]["tail"][subset]
            if row["arm"] in arm_of
            for q in order
        ]
        points = (
            alt.Chart(pl.DataFrame(rows))
            .mark_point(filled=True, size=60)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("quantile:N", title="Quantile", sort=order),
                xOffset=alt.XOffset("arm:N", sort=ARM_ORDER),
                y=alt.Y("coverage:Q", title="Share of actuals at or below"),
                color=_colour(),
            )
        )
        ticks = (
            alt.Chart(nominal)
            .mark_tick(color="#333333", thickness=2, size=70)
            .encode(x=alt.X("quantile:N", sort=order), y="nominal:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((points + ticks).properties(width=WIDTH - 90, height=140, title=title))
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The cold-night upper tail stays under-covered in every variant",
            subtitle=(
                "Share of actuals at or below each quantile, scored year, five decision "
                "substations. Black tick: the share a calibrated forecast would give."
            ),
        )
    )


def main() -> None:
    """Draw and save the figures."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_width(data=data), name="figure_2_width_by_lead_day")
    _save(figure_coverage(data=data), name="figure_3_coverage_by_lead_day")
    _save(figure_tails(data=data), name="figure_4_tails")


if __name__ == "__main__":
    main()
