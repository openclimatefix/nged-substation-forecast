"""Draw experiment 10's figures from the results tables, as SVG.

Reads the tables `e00010_report.py` wrote and writes the figures to
`docs/ml-experiments/00010-elexon-price-lags-stage1a-demand/`. Series carry the anonymous labels
experiment 00006 gave them (P1 to P16, B1, B2, G1, G2).

    uv run python ml-experiments/00010-elexon-price-lags-stage1a-demand/e00010_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00010_inputs as inputs
import e00010_report as report
import plotting.ocf_theme as ocf
import polars as pl
from contracts.settings import PROJECT_ROOT

OUT_DIR: Final[Path] = (
    PROJECT_ROOT / "docs" / "ml-experiments" / "00010-elexon-price-lags-stage1a-demand"
)
WIDTH: Final[int] = 600
BLUE: Final[str] = "#306BFF"
ORANGE: Final[str] = "#FF4901"
GREY: Final[str] = "#9AA0A6"
FOLD_LABELS: Final[dict[str, str]] = {
    "select_2024": "Selection year (2024-07 to 2025-06)",
    "confirm_2025": "Confirmation year (2025-07 to 2026-06)",
}


def per_series_chart() -> alt.VConcatChart:
    """Return stacked dot charts of each series' PRICE - BASE change in error, one per fold."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        data = pl.read_parquet(
            inputs.EXPERIMENT_DIR / "results" / f"{fold}_per_series.parquet"
        ).with_columns(
            group=pl.when(pl.col("site").is_in(report.LISTED))
            .then(pl.lit("Listed (B2, P4)"))
            .otherwise(pl.lit("Other"))
        )
        order = data.sort("price_gain")["site"].to_list()
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")
        dots = (
            alt.Chart(data)
            .mark_circle(size=70)
            .encode(
                x=alt.X(
                    "price_gain:Q",
                    title="Change in error from adding the price (points of capacity)",
                ),
                y=alt.Y("site:N", sort=order, title=None),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(domain=["Listed (B2, P4)", "Other"], range=[ORANGE, BLUE]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        panels.append(alt.layer(zero, dots).properties(width=WIDTH, height=230, title=label))
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Adding the price lowered the error at 7 of 17 series in the selection year, "
            "and at 5 of 18 in the confirmation year",
            anchor="start",
            fontSize=13,
        )
    )


def contrast_chart() -> alt.VConcatChart:
    """Return stacked interval charts of the planned contrasts, one per fold."""
    labels = {
        ("B2", "PRICE - BASE"): "B2: price against base",
        ("B2", "PRICE - CONTROL"): "B2: price against demand and wind",
        ("B2", "PRICE - SHUFFLED"): "B2: price against shuffled price",
        ("other series", "PRICE - BASE"): "Others: price against base",
        ("other series", "PRICE - CONTROL"): "Others: price against demand and wind",
        ("other series", "PRICE - SHUFFLED"): "Others: price against shuffled price",
    }
    panels = []
    for fold, label in FOLD_LABELS.items():
        raw = pl.read_parquet(inputs.EXPERIMENT_DIR / "results" / f"{fold}_contrasts.parquet")
        rows = [
            {**row, "label": labels[(row["scope"], row["contrast"])]}
            for row in raw.iter_rows(named=True)
            if (row["scope"], row["contrast"]) in labels
        ]
        rows += [
            {**row, "label": "C4: gain at B2 and P4 minus gain at others"}
            for row in raw.iter_rows(named=True)
            if row["contrast"].startswith("C4")
        ]
        data = pl.DataFrame(rows)
        order = data["label"].to_list()
        rule = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")
        bars = (
            alt.Chart(data)
            .mark_rule(strokeWidth=3, color=BLUE)
            .encode(
                x=alt.X(
                    "lower:Q",
                    title="Difference in error (points of capacity), 98.3% interval",
                ),
                x2="upper:Q",
                y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            )
        )
        points = (
            alt.Chart(data)
            .mark_circle(size=70, color=ORANGE)
            .encode(
                x="difference:Q",
                y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            )
        )
        panels.append(
            alt.layer(rule, bars, points).properties(width=WIDTH - 270, height=170, title=label)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Every interval for the price columns includes zero in the selection year",
            anchor="start",
            fontSize=13,
        )
    )


def main() -> None:
    """Write both figures as SVG."""
    alt.theme.enable("ocf") if "ocf" in alt.theme.names() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_series_chart().save(OUT_DIR / "figure_1.svg")
    contrast_chart().save(OUT_DIR / "figure_2.svg")
    print(f"wrote figures to {OUT_DIR} (theme module: {ocf.__name__})")


if __name__ == "__main__":
    main()
