"""Draw experiment 143's figures from `work/results.json` and `work/daily.parquet`.

uv run --no-sync python ml-experiments/00143-*/e00143_charts.py
"""

import json
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00143_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00143-differentiable-physics-holiday-aware-updating-of-the-gam-k-filter"
)
WIDTH: Final[int] = 640
PERIOD_COLOURS: Final[dict[str, str]] = {
    "2024-25 months": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}
PERIOD_KEYS: Final[dict[str, str]] = {
    "development": "2024-25 months",
    "scored": "scored year",
}
CONTRASTS: Final[dict[str, str]] = {
    "C1 K-skip - K0, leakage days": "C1 K-skip minus K0, days 1-5 after a holiday",
    "C2 K-skip - K0, holiday days": "C2 K-skip minus K0, holiday days",
    "C3 K-skip - K0, other days": "C3 K-skip minus K0, other days",
    "C6 K-skip - K0, all rows": "C6 K-skip minus K0, all rows",
    "C4 K-skip+1 - K-skip, leakage days": "C4 K-skip+1 minus K-skip, days 1-5 after",
    "C5 K-zero - K-skip, holiday days": "C5 K-zero minus K-skip, holiday days",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "K0 (reads every reading)": ocf.DATA_BLUE,
    "K-skip (ignores 24 Dec-2 Jan)": ocf.BRAND_ORANGE,
    "K-skip-core (ignores 24-26 Dec)": ocf.DATA_SKY,
    "K-zero (also zero on holidays)": ocf.DATA_PURPLE,
}
ARM_COLUMNS: Final[dict[str, str]] = {
    "K0 (reads every reading)": "qs_k0",
    "K-skip (ignores 24 Dec-2 Jan)": "qs_ks",
    "K-skip-core (ignores 24-26 Dec)": "qs_kc",
    "K-zero (also zero on holidays)": "qs_kz",
}


def save(*, chart: alt.Chart | alt.LayerChart | alt.VConcatChart, name: str) -> None:
    """Write a chart as SVG beside the write-up."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")


def forest(*, results: dict) -> alt.Chart:
    """Draw the planned contrasts with their 99% intervals, both periods."""
    rows = []
    for key, period in PERIOD_KEYS.items():
        for label, name in CONTRASTS.items():
            q = results[key][label]["qs"]
            rows.append(
                {
                    "contrast": name,
                    "period": period,
                    "estimate": q["estimate"],
                    "lower": q["lower"],
                    "upper": q["upper"],
                }
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y(
            "contrast:N", sort=list(CONTRASTS.values()), title=None, axis=alt.Axis(labelLimit=300)
        ),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
            legend=alt.Legend(orient="top", title=None, labelLimit=300),
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Quantile score difference (points; more negative favours the first arm)",
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=60, aria=False).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (
        (zero + bars + dots)
        .properties(
            width=WIDTH - 300,
            height=300,
            title=alt.TitleParams(
                "Skipping holiday readings helped after Christmas 2025 and hurt in 2024-25",
                subtitle=[
                    (
                        "Dot: estimate. Line: 99% interval from resampling holiday clusters "
                        "(months for C3 and C6)."
                    ),
                    (
                        "Day D+1, five Primary substations. Zero: same as K0. Each period "
                        "holds 5 holiday clusters."
                    ),
                ],
            ),
        )
        .configure_view(stroke=None)
    )


def christmas(*, daily: pl.DataFrame) -> alt.Chart:
    """Draw the daily mean D+1 score of each arm around each Christmas."""
    pieces = []
    for year, label in ((2024, "Christmas 2024"), (2025, "Christmas 2025")):
        window = daily.filter(
            (pl.col("local_date") >= date(year, 12, 20))
            & (pl.col("local_date") <= date(year + 1, 1, 5))
        )
        for arm, column in ARM_COLUMNS.items():
            pieces.append(
                window.select(
                    pl.col("local_date").dt.strftime("%d %b").alias("day"),
                    (pl.col("local_date") - pl.date(year, 12, 20)).dt.total_days().alias("offset"),
                    pl.col(column).alias("score"),
                    pl.lit(arm).alias("arm"),
                    pl.lit(label).alias("christmas"),
                )
            )
    frame = pl.concat(pieces)
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X(
                "offset:Q",
                title="Days since 20 December (25 December is day 5, 3 January is day 14)",
                axis=alt.Axis(values=[0, 5, 10, 15]),
            ),
            y=alt.Y("score:Q", title="Score (points; smaller is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(
                    orient="top", title=None, columns=2, labelLimit=300, symbolLimit=8
                ),
            ),
        )
        .properties(width=WIDTH - 80, height=170)
    )
    return (
        alt.layer(lines)
        .facet(row=alt.Row("christmas:N", title=None))
        .properties(
            title=alt.TitleParams(
                "Around each Christmas, day D+1",
                subtitle=[
                    (
                        "Daily mean over five Primary substations; the arms differ only once "
                        "a holiday reading has been ignored."
                    )
                ],
            )
        )
        .configure_view(stroke=None)
    )


def clusters(*, results: dict) -> alt.LayerChart:
    """Draw K-skip minus K0 for each holiday cluster, on its holiday days and the days after."""
    rows = []
    for key, period in PERIOD_KEYS.items():
        for name, entry in results[key].items():
            if not name.startswith("cluster "):
                continue
            day = date.fromisoformat(name.removeprefix("cluster "))
            label = day.strftime("%b %Y") + (" (Christmas)" if day.month == 12 else "")
            for field, which in (
                ("holiday_diff", "holiday days"),
                ("after_diff", "days 1-5 after"),
            ):
                if entry[field] is not None:
                    rows.append(
                        {
                            "cluster": label,
                            "order": day.toordinal(),
                            "days": which,
                            "difference": entry[field],
                            "period": period,
                        }
                    )
    frame = pl.DataFrame(rows)
    order = frame.sort("order")["cluster"].unique(maintain_order=True).to_list()
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            y=alt.Y("cluster:N", sort=order, title=None),
            x=alt.X(
                "difference:Q",
                title="K-skip minus K0 (points of capacity; more negative favours K-skip)",
            ),
            color=alt.Color(
                "days:N",
                scale=alt.Scale(
                    domain=["holiday days", "days 1-5 after"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH - 200,
            height=260,
            title=alt.TitleParams(
                "The effect of skipping holiday readings differs from one holiday to the next",
                subtitle=[
                    (
                        "Mean quantile score difference at day D+1, five Primary substations, "
                        "one dot per holiday cluster and kind of day."
                    )
                ],
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return alt.layer(zero, chart).configure_view(stroke=None)


def main() -> None:
    """Draw every figure."""
    results = json.loads((common.WORK_DIR / "results.json").read_text())
    daily = pl.read_parquet(common.WORK_DIR / "daily.parquet").filter(
        pl.col("period") != "development"
    )
    save(chart=forest(results=results), name="figure_01_headline")
    save(chart=christmas(daily=daily), name="figure_02_christmas")
    save(chart=clusters(results=results), name="figure_03_clusters")


if __name__ == "__main__":
    main()
