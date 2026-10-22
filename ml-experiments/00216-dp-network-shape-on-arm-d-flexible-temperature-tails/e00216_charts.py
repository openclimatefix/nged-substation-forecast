# ruff: noqa: D103,E501
"""Draw experiment 216's figures from the stored scores and rows.

uv run --no-sync python ml-experiments/00216-*/e00216_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00216_common as common
import e00216_score as score
import e00216_temperature as temperature
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 440
COLOURS: Final = {
    "Network (stored)": ocf.BRAND_ORANGE,
    "F1 (tail columns)": ocf.DATA_BLUE,
    "F2 (with 72 h terms)": ocf.DATA_GREEN,
}
ORDER: Final = list(COLOURS)
ZERO = (
    alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
    .mark_rule(color="#808080", strokeDash=[4, 3])
    .encode(x="x:Q")
)  # ty: ignore[unresolved-attribute]


def save(chart: alt.TopLevelMixin, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / name)
    print("wrote", name)


def figure_headline(results: dict) -> None:
    rows = []
    n = results["scored"]["network"]
    rows.append(
        {
            "panel": "Scored year",
            "arm": "Network (stored)",
            "d": float(np.mean([v["difference"] for v in n.values()])),
            "lo": min(v["lower_97.5"] for v in n.values()),
            "hi": max(v["upper_97.5"] for v in n.values()),
        }
    )
    for period, label in (("scored", "Scored year"), ("replicate", "Replicate year")):
        for key, arm in (("f1_1", "F1 (tail columns)"), ("f2_1", "F2 (with 72 h terms)")):
            c = results[period]["contrasts"][f"{key}_vs_r0"]
            rows.append(
                {
                    "panel": label,
                    "arm": arm,
                    "d": c["difference"],
                    "lo": c["lower_97.5"],
                    "hi": c["upper_97.5"],
                }
            )
    full = pl.DataFrame(rows)
    panels = []
    for label in ("Scored year", "Replicate year"):
        data = full.filter(pl.col("panel") == label)
        base = alt.Chart(data).encode(
            y=alt.Y("arm:N", sort=ORDER, title=None),
            color=alt.Color(
                "arm:N", legend=None, scale=alt.Scale(domain=ORDER, range=list(COLOURS.values()))
            ),
        )
        bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score minus arm D's, points (negative is better); 97.5% interval",
            ),
            x2="hi:Q",
        )
        dots = base.mark_point(filled=True, size=90).encode(x="d:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (ZERO + bars + dots).properties(
                width=WIDTH, height=100 if label == "Replicate year" else 120, title=label
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Tail columns recover a fifth of the network's gain in the scored",
                "year and none in the replicate year. Network: mean of 3 seeds.",
            ],
            anchor="start",
        )
    )
    save(chart, "figure_1_gain_against_arm_d.svg")


def figure_deciles(deciles: dict) -> None:
    rows = []
    for decile, v in deciles["by_decile"].items():
        for key, arm in (
            ("n", "Network (stored)"),
            ("f1", "F1 (tail columns)"),
            ("f2", "F2 (with 72 h terms)"),
        ):
            rows.append({"decile": int(decile), "arm": arm, "gain": v[f"gain_{key}"]})
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "decile:O",
                title="Decile of the 24-hour average temperature (1 coldest, 10 hottest), scored year",
            ),
            y=alt.Y("gain:Q", title="Quantile score gain over arm D, points (higher is better)"),
            color=alt.Color(
                "arm:N",
                sort=ORDER,
                scale=alt.Scale(domain=ORDER, range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=alt.TitleParams(
                [
                    "The network's gain is spread over most deciles, not the extremes;",
                    "the tail columns gain mostly in deciles 8 and 9",
                ],
                anchor="start",
            ),
        )
    )
    save(chart, "figure_2_gain_by_decile.svg")


def figure_shape() -> None:
    """Mean added forecast by temperature bin for the network and the tail columns."""
    base = score.add_losses(frame=score.read_arm(arm="r0", period="scored")).filter(
        pl.col("site").is_in(common.DECISION_SITES)
    )
    keys = list(score.KEYS)
    out = base.select(*keys, mean_r0="mean", q50_r0="q50")
    for name in ("f1_1", "f2_1"):
        f = score.read_arm(arm=name, period="scored").select(
            *keys, pl.col("q50").alias(f"q50_{name}")
        )
        out = out.join(f, on=keys, how="inner")
    net = (
        pl.concat(
            [
                pl.read_parquet(score.E201_PREDS / f"arm2_seed{s}.parquet").select(
                    *keys, q50_n="q50"
                )
                for s in score.N_SEEDS
            ]
        )
        .group_by(*keys)
        .agg(pl.col("q50_n").mean())
    )
    out = out.join(net, on=keys, how="inner").join(
        temperature.all_frames(), on=["site", "valid_time"], how="inner"
    )
    out = out.with_columns(bin=(pl.col("temp24_c") / 2).floor() * 2 + 1)
    rows = []
    for key, arm in (
        ("q50_n", "Network (stored)"),
        ("q50_f1_1", "F1 (tail columns)"),
        ("q50_f2_1", "F2 (with 72 h terms)"),
    ):
        g = (
            out.group_by("bin")
            .agg(rows=pl.len(), shift=(pl.col(key) - pl.col("q50_r0")).mean())
            .filter(pl.col("rows") >= 500)
            .with_columns(arm=pl.lit(arm))
        )
        rows.append(g)
    data = pl.concat(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "bin:Q",
                title="24-hour average temperature, degrees Celsius (bins of 2; bins with 500 rows or more)",
            ),
            y=alt.Y("shift:Q", title="Mean median forecast minus arm D's, points of capacity"),
            color=alt.Color(
                "arm:N",
                sort=ORDER,
                scale=alt.Scale(domain=ORDER, range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=alt.TitleParams(
                [
                    "Both raise the forecast by about 2 points below 0 degrees Celsius; above",
                    "21 degrees the tail columns raise it by up to 7 points, the network by 2",
                ],
                anchor="start",
            ),
        )
    )
    save(chart, "figure_3_shape_of_the_correction.svg")


def figure_surface(surface: dict) -> None:
    rows = []
    for name in ("S", "S_filter", "S_season", "S_both"):
        v = surface[f"{name}_smooth1"]
        rows += [
            {"variant": name, "kind": "In sample", "r2": v["in_sample"]},
            {"variant": name, "kind": "Held-out month", "r2": v["leave_one_month_out"]},
        ]
    labels = {
        "S": "Temperature surface",
        "S_filter": "+ filter state",
        "S_season": "+ day of year",
        "S_both": "+ both",
    }
    data = pl.DataFrame(rows).with_columns(variant=pl.col("variant").replace(labels))
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("variant:N", sort=list(labels.values()), title=None),
            x=alt.X(
                "r2:Q",
                title="Share of the network's correction explained (higher is better)",
                scale=alt.Scale(domain=[0, 0.3]),
            ),
            yOffset="kind:N",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["In sample", "Held-out month"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=190,
            title=alt.TitleParams(
                [
                    "A convex surface in temperature explains under 1% of the network's",
                    "correction on held-out months; day of year raises that to about 10%",
                ],
                anchor="start",
            ),
        )
    )
    save(chart, "figure_4_convex_surface_explains_the_correction.svg")


def main() -> None:
    results = json.loads((common.WORK_DIR / "score.json").read_text())
    deciles = json.loads((common.WORK_DIR / "deciles.json").read_text())
    surface = json.loads((common.WORK_DIR / "surface.json").read_text())
    figure_headline(results)
    figure_deciles(deciles)
    figure_shape()
    figure_surface(surface)


if __name__ == "__main__":
    main()
