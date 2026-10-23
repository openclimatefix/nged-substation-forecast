"""Draw experiment 00303's figures as SVG and print the numbers the write-up quotes.

Reads `work/score.json` and `work/rows.parquet`. Afterwards optimise with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python e00303_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00303_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00303-wind-adapter-in-production-setting"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
SCOPE: Final[str] = (
    "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026, rows kept by the mask."
)
LABELS: Final[dict[str, str]] = {
    "P0": "P0 XGBoost on members",
    "P2": "P2 adapter fitted to ERA5",
    "P3": "P3 adapter and offset table",
    "P4": "P4 fitted on hindcast",
    "P5": "P5 fine-tuned on hindcast",
    "P7": "P7 blind, fine-tuned",
    "P8": "P8 XGBoost with effective speed, and table",
    "P6": "P6 P2 plus neural head",
    "P5m": "P5m (post hoc) case-mean loss, member mixture",
    "P4c": "P4c (post hoc) case-mean loss and quantiles",
    "P5c": "P5c (post hoc) case-mean loss and quantiles",
    "P7c": "P7c (post hoc) case-mean loss and quantiles",
    "P3_LOFO": "P3 for a farm with no history",
    "P5_LOFO": "P5 for a farm with no history",
}
COLOURS: Final[dict[str, str]] = {
    "P0": GREY,
    "P2": ocf.DATA_BLUE,
    "P3": ocf.DATA_BLUE,
    "P4": ocf.BRAND_ORANGE,
    "P5": ocf.BRAND_ORANGE,
    "P7": ocf.BRAND_ORANGE,
    "P8": ocf.DATA_GREEN,
    "P6": ocf.DATA_PURPLE,
    "P5m": ocf.BRAND_ORANGE,
    "P4c": ocf.BRAND_ORANGE,
    "P5c": ocf.BRAND_ORANGE,
    "P7c": ocf.BRAND_ORANGE,
    "P3_LOFO": ocf.DATA_SKY,
    "P5_LOFO": ocf.DATA_SKY,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Every arm's score minus P1's, kept rows, 95% intervals."""
    rows = []
    for key in LABELS:
        entry = score["contrasts"].get(f"vsP1: {key} - P1")
        if entry is None:
            continue
        found = entry["kept | qs"]
        low, high = found["0.95000"]
        rows.append(
            {"arm": LABELS[key], "key": key, "estimate": found["mean"], "lower": low, "upper": high}
        )
    data = pl.DataFrame(rows)
    order = [LABELS[k] for k in LABELS if k in set(data["key"])]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "key:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title=("Arm's score minus P1's (points of capacity; more negative is better)"),
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=26 * len(order),
        title=_title(
            text="Fine-tuning the adapter on the ensemble hindcast scored worse than P1",
            lines=[
                "Quantile score minus P1's (XGBoost on members plus the offset table). Grey line: "
                "same as P1. Dot: estimate. Line: 95% interval from re-drawing whole months and "
                "seeds. " + SCOPE
            ],
        ),
    )


def figure_2(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """Score minus P1's by lead day for the main arms."""
    arms = [a for a in ("P2", "P3", "P5", "P5c", "P8") if f"qs__{a}" in rows.columns]
    line_colours = {
        "P2": ocf.DATA_SKY,
        "P3": ocf.DATA_BLUE,
        "P5": ocf.BRAND_ORANGE,
        "P5c": ocf.DATA_PURPLE,
        "P8": ocf.DATA_GREEN,
    }
    kept = rows.filter(~pl.col("masked")).with_columns(day=pl.col("lead_day").clip(0, 8))
    pieces = [
        kept.group_by("day")
        .agg(n=pl.len(), gain=(pl.col(f"qs__{a}") - pl.col("qs__P1")).mean())
        .with_columns(key=pl.lit(a), arm=pl.lit(LABELS[a]))
        .filter(pl.col("n") >= 200)
        for a in arms
    ]
    data = pl.concat(pieces)
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Lead day (calendar days from the run to the valid time; 8 means 8 or more)",
                axis=alt.Axis(values=list(range(9))),
            ),
            y=alt.Y("gain:Q", title="Score minus P1's (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[LABELS[a] for a in arms], range=[line_colours[a] for a in arms]
                ),
                legend=alt.Legend(
                    title=None, orient="top", columns=2, labelLimit=400, symbolType="stroke"
                ),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, chart).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="The fine-tuned arms lose most at short leads",
            lines=[
                "Mean quantile score minus P1's by lead day; below the grey line is better. "
                + SCOPE
            ],
        ),
    )


def figure_3(*, score: dict) -> alt.TopLevelMixin:
    """Share of observed power below the median forecast, per arm (50% is unbiased)."""
    rows = [
        {"arm": LABELS[k], "key": k, "share": 100 * v["kept"]["below_median"]}
        for k, v in score["scores"].items()
        if k in LABELS
    ]
    data = pl.DataFrame(rows)
    order = [LABELS[k] for k in LABELS if k in set(data["key"])]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "key:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
    )
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "share:Q",
            title="Observed power below the median forecast (% of rows; 50 is unbiased)",
            scale=alt.Scale(domain=[0, 60]),
        )
    )
    rule = alt.Chart(pl.DataFrame({"x": [50.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, rule).properties(
        width=WIDTH,
        height=24 * len(order),
        title=_title(
            text="Every arm still forecasts too low a median",
            lines=[
                "Share of kept rows where observed power fell below the median forecast. " + SCOPE
            ],
        ),
    )


def print_tables(*, score: dict) -> None:
    """Print the score table, the tail and bias checks, and the farm-by-farm median checks."""
    print(
        "| Arm | Kept | All rows | Lead 12 to 36 h | q01 pinball | q99 pinball "
        "| 5-95% coverage | Below median |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for key, label in {"P1": "P1", **LABELS}.items():
        value = score["scores"].get(key)
        if value is None:
            continue
        k = value["kept"]
        print(
            f"| {label} | {k['qs']:.3f} | {value['all']['qs']:.3f} "
            f"| {value['lead_12_36']['qs']:.3f} "
            f"| {k['p01']:.3f} | {k['p99']:.3f} | {100 * k['covered']:.1f}% "
            f"| {100 * k['below_median']:.1f}% |"
        )
    print("checks: median minus mean forecast, points of capacity, kept rows")
    for key in ("P0", "P1", "P2", "P3", "P5c", "P6", "P8"):
        value = score["scores"].get(key)
        if value:
            print(
                key,
                round(value["kept"]["median_minus_mean"], 3),
                "mean-forecast absolute error",
                round(value["kept"]["mean_abs_error"], 3),
                "median absolute error",
                round(value["kept"]["median_abs_error"], 3),
            )
    print("by farm, kept rows, score")
    for key in ("P1", "P3", "P5c", "P6", "P8"):
        print(key, {f: round(x, 3) for f, x in score["scores"][key]["by_farm_kept"].items()})


def main() -> None:
    """Write the figures and print the numbers the write-up quotes."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    rows = pl.read_parquet(common.WORK_DIR / "rows.parquet")
    print_tables(score=score)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(score=score)),
        ("figure_2", figure_2(rows=rows)),
        ("figure_3", figure_3(score=score)),
    ):
        chart.configure_view(strokeWidth=0).save(DOCS_DIR / f"{name}.svg")
    print("figures written")


if __name__ == "__main__":
    main()
