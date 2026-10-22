# ruff: noqa: D103,E501,PERF401
"""Draw experiment 225's figures from the stored day table and results.

uv run --no-sync python ml-experiments/00225-*/e00225_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00225_run as run
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{run.EXPERIMENT_ID:05d}-{run.SLUG}"
)
WIDTH: Final[int] = 440


def save(chart: alt.TopLevelMixin, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / name)
    print("wrote", name)


def figure_temperature(days: pl.DataFrame, results: dict) -> None:
    unlabelled = days.filter(pl.col("worst") & pl.col("unlabelled")).select(
        pl.col("percentile") * 100
    )
    dots = (
        alt.Chart(unlabelled)  # ty: ignore[unresolved-attribute]
        .mark_tick(thickness=3, size=40, color=ocf.BRAND_ORANGE)
        .encode(
            x=alt.X(
                "percentile:Q",
                title="Temperature percentile of the day among the site's earlier days (0 coldest, 100 hottest)",
                scale=alt.Scale(domain=[0, 100]),
            )
        )
        .properties(width=WIDTH, height=50, title="The 37 unlabelled worst days")
    )
    shares = pl.DataFrame({"share": np.array(results["random_control"]["draws_outer_share"]) * 100})
    real = pl.DataFrame(
        {"x": [results["temperature"]["unlabelled_worst"]["share_outer_10pct"] * 100]}
    )
    hist = (
        alt.Chart(shares)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE_LIGHT)
        .encode(
            x=alt.X(
                "share:Q",
                bin=alt.Bin(step=2.5),
                title="Share of 37 days in the outer 10% of temperature (%)",
            ),
            y=alt.Y("count()", title="Draws out of 1,000"),
        )
    )
    rule = alt.Chart(real).mark_rule(color=ocf.BRAND_ORANGE, strokeWidth=3).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    controls = (hist + rule).properties(
        width=WIDTH,
        height=150,
        title="1,000 draws of 37 ordinary days (orange: the unlabelled days)",
    )
    chart = alt.vconcat(dots, controls).properties(
        title=alt.TitleParams(
            [
                "The unlabelled worst days are not hot or cold: 4 of 37 lie in the outer",
                "10% of temperature, fewer than the 7 or 8 expected by chance",
            ],
            anchor="start",
        )
    )
    save(chart, "figure_1_temperature_of_the_unlabelled_days.svg")


def figure_removed(results: dict) -> None:
    labels = {
        "n_arm2": "Network (arm 2)",
        "n_outputs": "Network on arm D's outputs only",
        "n_lagged": "Network with recent power",
        "n_blind": "Network with no forecast input",
    }
    rows = []
    for name, label in labels.items():
        for cls, cls_label in (
            ("unlabelled_worst", "37 unlabelled worst days"),
            ("labelled_worst", "35 labelled worst days"),
        ):
            seeds = results["networks"][name]["by_seed"][cls]
            rows.append(
                {
                    "network": label,
                    "class": cls_label,
                    "share": results["networks"][name]["by_class"][cls]["share_removed"] * 100,
                    "lo": min(seeds) * 100,
                    "hi": max(seeds) * 100,
                }
            )
    data = pl.DataFrame(rows)
    classes = ["37 unlabelled worst days", "35 labelled worst days"]
    base = alt.Chart(data).encode(
        y=alt.Y("network:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("class:N", sort=classes),
        color=alt.Color(
            "class:N",
            sort=classes,
            scale=alt.Scale(domain=classes, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore
        x=alt.X("lo:Q", title="Share of arm D's excess loss removed (%); line: range over 3 seeds"),
        x2="hi:Q",
    )
    dots = base.mark_point(filled=True, size=70).encode(x="share:Q")  # ty: ignore[unresolved-attribute]
    chart = (bars + dots).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            [
                "The network removes about 23% of the unlabelled days' excess loss",
                "and 12% of the labelled days'; without a forecast input it removes 4%",
            ],
            anchor="start",
        ),
    )
    save(chart, "figure_2_excess_removed_by_class.svg")


def figure_examples() -> None:
    ex = pl.read_parquet(run.WORK_DIR / "examples.parquet")
    local = pl.col("valid_time").dt.convert_time_zone(run.LOCAL_TZ)
    ex = ex.with_columns(
        hour=(local.dt.hour() + local.dt.minute() / 60.0),
        label=pl.col("site") + ", excess " + pl.col("excess").round(1).cast(pl.String) + " points",
    )
    ex = (
        ex.group_by("label", "valid_time", "hour")
        .agg(pl.col("actual", "q50", "n_arm2_q50").mean())
        .sort("label", "valid_time")
    )
    long = pl.concat(
        [
            ex.select("label", "hour", value=pl.col("actual"), series=pl.lit("Observed")),
            ex.select("label", "hour", value=pl.col("q50"), series=pl.lit("Arm D")),
            ex.select(
                "label", "hour", value=pl.col("n_arm2_q50"), series=pl.lit("Arm D plus network")
            ),
        ]
    )
    order = ["Observed", "Arm D", "Arm D plus network"]
    panels = []
    for label in long["label"].unique(maintain_order=True):
        panels.append(
            alt.Chart(long.filter(pl.col("label") == label))  # ty: ignore[unresolved-attribute]
            .mark_line()
            .encode(
                x=alt.X(
                    "hour:Q",
                    title="Local hour of day",
                    axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                    scale=alt.Scale(domain=[0, 24]),
                ),
                y=alt.Y("value:Q", title="Power, % of capacity", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    sort=order,
                    scale=alt.Scale(
                        domain=order, range=["#404040", ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
            .properties(width=WIDTH, height=110, title=label)
        )
    chart = alt.vconcat(*panels)
    save(
        chart.properties(
            title=alt.TitleParams(
                [
                    "Five unlabelled worst days (largest excess loss): median forecasts",
                    "of arm D and of arm D plus the network, against the observed power",
                ],
                anchor="start",
            )
        ),
        "figure_3_five_unlabelled_days.svg",
    )


def main() -> None:
    days = pl.read_parquet(run.WORK_DIR / "days.parquet")
    results = json.loads((run.WORK_DIR / "results.json").read_text())
    figure_temperature(days, results)
    figure_removed(results)
    figure_examples()


if __name__ == "__main__":
    main()
