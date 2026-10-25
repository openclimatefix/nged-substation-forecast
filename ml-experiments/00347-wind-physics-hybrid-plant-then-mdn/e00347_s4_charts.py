# ruff: noqa: E501
"""Draw experiment 00347's Stage 4 figures (figure_10 onward) as SVG from the stored scores.

Reads `s4/rows.parquet`, `s4/score.json`, and `s4/median_bins.parquet` written by
`e00347_s4_score.py`. Optimise afterwards with `npx svgo@4 --multipass --precision=1
--final-newline f.svg`. Every farm is W1 to W3 and no chart shows a calendar date.

    PYTHONPATH=ml-experiments/00347-wind-physics-hybrid-plant-then-mdn:\
        ml-experiments/00230-wind-physics-core-part-1 \
    uv run --no-sync python ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_s4_charts.py
"""

import json
import os
import textwrap
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import altair as alt  # noqa: E402
import e00347_common as common  # noqa: E402
import numpy as np  # noqa: E402
import plotting.ocf_theme as ocf  # noqa: E402
import polars as pl  # noqa: E402

PLOT_WIDTH: Final[int] = 560
WRAP: Final[int] = 100
GREY: Final[str] = "#999999"
SCOPE: Final[str] = (
    "Three metered wind farms W1 to W3; 53 Tuesday runs, kept half-hours, scored at 0.5 hours to 15 days ahead, as in experiments 324 and 339."
)
SCOPE_LABELS: Final[dict[str, str]] = {
    "kept": "Every lead in the row set (0.5 hours to 15 days)",
    "true_1_10": "True days 1 to 10 (lead 24 to 240 hours)",
}
S4: Final = common.DATA_DIR / "s4"
LEVEL: Final[float] = 1 - 0.01 / 4
SCOPES: Final[tuple[str, ...]] = ("kept", "true_1_10")
GROUP_COLOURS: Final[dict[str, str]] = {
    "plant + MDN head (this experiment)": ocf.BRAND_ORANGE,
    "plant + offset table, no head": ocf.DATA_BLUE,
    "XGBoost and other families": ocf.DATA_PURPLE,
    "control": GREY,
}
ARM_LABELS: Final[dict[str, str]] = {
    "y4": "y4 pass-2 plant + MDN",
    "y3": "y3 Stage 3 plant + MDN, mean and sd only",
    "y2": "y2 Stage 3 plant + MDN",
    "y1": "y1 Stage 2 plant + MDN",
    "s3b_table": "pass-2 plant + offset table",
    "s3_table": "Stage 3 plant + offset table",
    "s2_table": "Stage 2 plant + offset table",
    "default_notable": "XGBoost alone",
    "c": "XGBoost + adapter column + table",
    "W4": "physics core + adapter + MDN",
    "P3": "physics core + adapter + table",
    "W3": "weather-only MDN",
    "W0": "climatology",
    "base": "plant median only, no farm columns (unequal control)",
    "basefarm": "plant median and farm only (control)",
    "y2_placebo": "y2 with shuffled summaries",
    "y2_placebo2": "y2 with shuffled spread summaries",
    "y2_lofo": "y2 leaving the farm out",
    "y4_lofo": "y4 leaving the farm out",
    "default": "default: XGBoost + offset table",
}
GROUP_OF: Final[dict[str, str]] = {
    **dict.fromkeys(("y1", "y2", "y3", "y4"), "plant + MDN head (this experiment)"),
    **dict.fromkeys(("s2_table", "s3_table", "s3b_table"), "plant + offset table, no head"),
    **dict.fromkeys(
        ("default_notable", "c", "W4", "P3", "W3", "W0", "default"), "XGBoost and other families"
    ),
    **dict.fromkeys(
        ("base", "basefarm", "y2_placebo", "y2_placebo2", "y2_lofo", "y4_lofo"), "control"
    ),
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _colour(*, legend: bool = True) -> alt.Color:
    return alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", labelLimit=500, columns=2) if legend else None,
    )


def interval_rows(
    *, rows: pl.DataFrame, arms: list[str], versus: str, scopes: tuple[str, ...] = ("kept",)
) -> pl.DataFrame:
    """Return, per scope and arm, the mean of (arm minus `versus`) with 95% and 99.75% intervals."""
    out = []
    for scope in scopes:
        out.extend(_interval_scope(rows=rows, arms=arms, versus=versus, scope=scope))
    return pl.DataFrame(out)


def _interval_scope(*, rows: pl.DataFrame, arms: list[str], versus: str, scope: str) -> list[dict]:
    kept = rows.filter(~pl.col("masked"))
    if scope == "true_1_10":
        kept = kept.filter(pl.col("lead_hours").is_between(24, 240))
    months = kept["month"].to_numpy()
    out = []
    for arm in arms:
        values = (kept[f"qs__{arm}"] - kept[f"qs__{versus}"]).to_numpy()
        labels = np.unique(months)
        sums = np.array([values[months == m].sum() for m in labels])
        counts = np.array([(months == m).sum() for m in labels])
        draws = np.random.default_rng(0).integers(0, labels.size, size=(20_000, labels.size))
        means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        low95, high95 = np.quantile(means, [0.025, 0.975])
        low99, high99 = np.quantile(means, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
        out.append(
            {
                "scope": SCOPE_LABELS[scope],
                "arm": ARM_LABELS[arm],
                "group": GROUP_OF[arm],
                "mean": float(values.mean()),
                "low95": float(low95),
                "high95": float(high95),
                "low99": float(low99),
                "high99": float(high99),
            }
        )
    return out


def figure_10(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """Headline: kept-row score of every arm minus the default, with intervals."""
    arms = [
        "y4",
        "y3",
        "y2",
        "y1",
        "s3b_table",
        "s3_table",
        "s2_table",
        "default_notable",
        "c",
        "W4",
    ]
    table = interval_rows(rows=rows, arms=arms, versus="default", scopes=SCOPES)
    order = [ARM_LABELS[a] for a in arms]
    y2 = {r["scope"]: r["mean"] for r in table.filter(pl.col("arm") == ARM_LABELS["y2"]).to_dicts()}
    base = alt.Chart(table).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)), color=_colour()
    )
    wide = base.mark_rule(strokeWidth=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low99:Q",
            title="Quantile score minus the default's (points of capacity; negative is better)",
        ),
        x2="high99:Q",
    )
    narrow = base.mark_rule(strokeWidth=4, aria=False).encode(x="low95:Q", x2="high95:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70, aria=False).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    layered = alt.layer(zero, wide, narrow, dot).properties(width=PLOT_WIDTH, height=230)
    faceted = layered.facet(
        row=alt.Row("scope:N", title=None, sort=list(SCOPE_LABELS.values())), data=table
    )
    return faceted.properties(
        title=_title(
            text=(
                f"The plant + head arm y2 beats the default by {-y2[SCOPE_LABELS['kept']]:.2f} points over every lead "
                f"and {-y2[SCOPE_LABELS['true_1_10']]:.2f} points over true days 1 to 10"
            ),
            lines=[
                "Dot: estimate. Thick line: 95% interval from resampling whole months of run date. "
                "Thin line: 99.75% interval (the adoption test's level for four arms). Dashed rule: "
                "the default (Wind farms research default 1), zero by construction. " + SCOPE
            ],
        ),
    )


def figure_11(*, rows: pl.DataFrame, scores: dict) -> alt.TopLevelMixin:
    """Absolute scores of every arm on the kept rows at both scopes."""
    arms = [
        "y4",
        "y3",
        "y2",
        "y1",
        "s3b_table",
        "s3_table",
        "s2_table",
        "default",
        "default_notable",
        "c",
        "W4",
        "P3",
        "W3",
        "W0",
    ]
    data = []
    for arm in arms:
        for subset, label in (
            ("kept", SCOPE_LABELS["kept"]),
            ("true_1_10", SCOPE_LABELS["true_1_10"]),
        ):
            data.append(
                {
                    "arm": ARM_LABELS[arm],
                    "group": GROUP_OF[arm],
                    "subset": label,
                    "score": scores[arm][subset],
                }
            )
    order = [ARM_LABELS[a] for a in arms]
    chart = (
        alt.Chart(pl.DataFrame(data))
        .mark_point(filled=True, size=70, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=_colour(),
            row=alt.Row("subset:N", title=None, sort=list(SCOPE_LABELS.values())),
        )
        .properties(
            width=PLOT_WIDTH,
            height=200,
            title=_title(
                text="Absolute scores: the plant + head arms and the physics core + adapter + MDN score lowest at both scopes",
                lines=["Each dot is the quantile score of one arm. " + SCOPE],
            ),
        )
    )
    return chart  # noqa: RET504


def figure_12(*, by_lead: dict, rows_per_day: list[int]) -> alt.TopLevelMixin:
    """Score by lead day on the kept rows."""
    names = {
        "default": ("default", ocf.DATA_PURPLE),
        "c": ("XGBoost + adapter column", ocf.DATA_SKY),
        "W4": ("physics core + adapter + MDN", ocf.DATA_GREEN),
        "y2": ("y2 plant + MDN", ocf.BRAND_ORANGE),
        "y4": ("y4 pass-2 plant + MDN", ocf.BRAND_ORANGE_LIGHT),
        "s3_table": ("Stage 3 plant + offset table", ocf.DATA_BLUE),
    }
    default = by_lead["default"]
    data = [
        {"lead_day": d, "difference": v - default[d], "arm": names[name][0]}
        for name, values in by_lead.items()
        if name in names and name != "default"
        for d, v in enumerate(values)
        if v is not None
    ]
    chart = (
        alt.Chart(pl.DataFrame(data))
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lead_day:O",
                title="Lead (calendar days from the run's date to the valid time's date)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("difference:Q", title="Score minus the default's (points; negative is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[v[0] for k, v in names.items() if k != "default"],
                    range=[v[1] for k, v in names.items() if k != "default"],
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=500, columns=2),
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = alt.layer(zero, chart)
    wins = sum(v < default[d] for d, v in enumerate(by_lead["y2"]) if v is not None)
    days_scored = sum(v is not None for v in default)
    return chart.properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text=f"The Stage 3 plant + MDN beats the default at {wins} of {days_scored} lead days (days 0 to 15)",
            lines=[
                (
                    "Dashed rule: the default. Kept half-hours of the 53 Tuesday runs; three metered wind farms W1 to W3. "
                    f"Day 0 holds {rows_per_day[0]:,} kept rows and day 15 holds {rows_per_day[15]:,}; each other day holds {min(rows_per_day[1:15]):,} to {max(rows_per_day[1:15]):,}."
                )
            ],
        ),
    )


def figure_13(*, tails: dict) -> alt.TopLevelMixin:
    """Tail calibration: the share of observations beyond each quantile against its nominal share."""
    arms = ["default", "default_notable", "c", "W4", "y1", "y2", "y3", "y4", "s3_table"]
    labels = ["below q01", "below q05", "above q95", "above q99"]
    nominal = {"below q01": 1.0, "below q05": 5.0, "above q95": 5.0, "above q99": 1.0}
    data = []
    for arm in arms:
        for label, value in zip(labels, tails[arm][:4], strict=True):
            data.append(
                {
                    "arm": ARM_LABELS[arm],
                    "group": GROUP_OF[arm],
                    "tail": label,
                    "share": value,
                    "nominal": nominal[label],
                }
            )
    table = pl.DataFrame(data)
    order = [ARM_LABELS[a] for a in arms]
    dots = (
        alt.Chart(table)
        .mark_point(filled=True, size=60, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("share:Q", title=None),
            color=_colour(),
        )
    )
    rule = (
        alt.Chart(table)
        .mark_rule(color="black", strokeDash=[4, 3], aria=False)
        .encode(x="nominal:Q")  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(rule, dots)
        .properties(width=130, height=190)
        .facet(
            column=alt.Column("tail:N", title=None, sort=labels),
        )
        .properties(
            title=_title(
                text="The head fixes the lower tail but leaves the upper tail over-confident: 2.3% of observations exceed q99",
                lines=[
                    "Horizontal axis: the share of kept half-hours beyond the quantile (%). Dashed rule: the nominal share. Dot: observed share. "
                    + SCOPE
                ],
            )
        )
    )


def figure_14(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """Controls: contrasts of the control arms with the default and with y2."""
    arms = ["y2", "y4", "basefarm", "base", "y2_placebo2", "y2_lofo", "y4_lofo"]
    table = interval_rows(rows=rows, arms=arms, versus="default")
    mean = {r["arm"]: r["mean"] for r in table.to_dicts()}
    gap = mean[ARM_LABELS["basefarm"]] - mean[ARM_LABELS["y2"]]
    spread = mean[ARM_LABELS["y2_placebo2"]] - mean[ARM_LABELS["y2"]]
    order = [ARM_LABELS[a] for a in arms]
    base = alt.Chart(table).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=_colour(),
    )
    narrow = base.mark_rule(strokeWidth=4, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low95:Q",
            title="Quantile score minus the default's (points of capacity; negative is better)",
        ),
        x2="high95:Q",
    )
    dot = base.mark_point(filled=True, size=70, aria=False).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, narrow, dot).properties(
        width=PLOT_WIDTH,
        height=190,
        title=_title(
            text=f"Controls: a head fed only the plant median and the farm scores {gap:.2f} worse than y2, and shuffling the spread summaries costs {spread:.1f} point",
            lines=[
                "Dot: estimate. Line: 95% interval from resampling whole months. Dashed rule: the "
                "default. Leave-out arms train on the other two farms without the farm columns. "
                "Shuffling every summary column costs +4.9 points (off the scale). " + SCOPE
            ],
        ),
    )


def figure_15(*, replicate: dict) -> alt.TopLevelMixin:
    """Replicate: monthly kept-row differences from the default for each head arm."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    data = []
    for arm in ("y1", "y2", "y3", "y4"):
        for month, value in zip(months, replicate[arm]["monthly"], strict=False):
            data.append({"month": month, "difference": value, "arm": ARM_LABELS[arm]})
    colours = {
        ARM_LABELS["y1"]: ocf.DATA_BLUE,
        ARM_LABELS["y2"]: ocf.BRAND_ORANGE,
        ARM_LABELS["y3"]: ocf.DATA_GREEN,
        ARM_LABELS["y4"]: ocf.DATA_PURPLE,
    }
    chart = (
        alt.Chart(pl.DataFrame(data))
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month:N",
                sort=months,
                title="Month of 2025 (run date)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("difference:Q", title="Score minus the default's (points; negative is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", labelLimit=500, columns=2),
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, chart).properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            text="Replicate: every plant + head arm beats the default in each of the six months, January to June 2025",
            lines=[
                "Kept half-hours of the daily runs; three metered wind farms W1 to W3; three seeds per arm."
            ],
        ),
    )


def figure_16(*, bins: pl.DataFrame) -> alt.TopLevelMixin:
    """Observed mean power against the forecast median, with the mean 5% to 95% band."""
    labels = {
        "default": ("default", ocf.DATA_PURPLE),
        "y2": ("y2 plant + MDN", ocf.BRAND_ORANGE),
    }
    table = bins.filter(pl.col("arm").is_in(list(labels)))
    table = table.with_columns(
        arm=pl.col("arm").replace_strict({k: v[0] for k, v in labels.items()}),
        forecast=pl.col("forecast_median") * 100,
        actual=pl.col("actual_mean") * 100,
        low=pl.col("q05_mean") * 100,
        high=pl.col("q95_mean") * 100,
    )
    scale = alt.Scale(domain=[v[0] for v in labels.values()], range=[v[1] for v in labels.values()])
    band = (
        alt.Chart(table)
        .mark_area(opacity=0.12, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "forecast:Q", title="Forecast median (% of capacity), in 20 bins of equal size"
            ),
            y=alt.Y(
                "low:Q",
                title="Mean observed power (line) and mean q05 to q95 band (shade), % of capacity",
            ),
            y2="high:Q",
            color=alt.Color(
                "arm:N",
                scale=scale,
                legend=alt.Legend(title=None, orient="top", labelLimit=500, columns=2),
            ),
        )
    )
    line = (
        alt.Chart(table)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x="forecast:Q", y="actual:Q", color=alt.Color("arm:N", scale=scale, legend=None)
        )
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 100.0], "y": [0.0, 100.0]}))
        .mark_line(color="black", strokeDash=[4, 3], aria=False)
        .encode(x="x:Q", y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(diagonal, band, line).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="At low forecast medians, the plant + MDN is closer to the observed mean than the default",
            lines=["Dashed line: observed equals forecast. " + SCOPE],
        ),
    )


def main() -> None:
    """Draw every Stage 4 figure."""
    rows = pl.read_parquet(S4 / "rows.parquet")
    report = json.loads((S4 / "score.json").read_text())
    bins = pl.read_parquet(S4 / "median_bins.parquet")
    figures = {
        "figure_10_score_against_default": figure_10(rows=rows),
        "figure_11_absolute_scores": figure_11(rows=rows, scores=report["scores"]),
        "figure_12_score_by_lead_day": figure_12(
            by_lead=report["by_lead_day"], rows_per_day=report["kept_rows_per_lead_day"]
        ),
        "figure_13_tail_calibration": figure_13(tails=report["tails"]),
        "figure_14_controls": figure_14(rows=rows),
        "figure_15_replicate_by_month": figure_15(replicate=report["replicate"]),
        "figure_16_observed_against_forecast": figure_16(bins=bins),
    }
    common.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in figures.items():
        chart.save(common.DOCS_DIR / f"{name}.svg")
        preview = os.environ.get("E00347_PREVIEW_DIR")
        if preview:
            chart.save(f"{preview}/{name}.png", scale_factor=1.0)
        print(name)


if __name__ == "__main__":
    main()
