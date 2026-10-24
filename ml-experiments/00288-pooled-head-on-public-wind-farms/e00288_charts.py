"""Draw experiment 00288's four figures from `work/score.json` and `work/pool_score*`."""

# ruff: noqa: E501
import json
from pathlib import Path
from typing import Final

import altair as alt
import e00288_common as common
import polars as pl
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN, DATA_PURPLE

DOCS_DIR: Final = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00288-pooled-head-on-public-wind-farms"
)
WIDTH: Final = 640
SCORES: Final = "kept rows | days 1-2"
ARMS: Final = (
    "P0",
    "A0r-230",
    "T1",
    "P1",
    "P2",
    "P3",
    "P2_ns",
    "P3_ns",
    "X2",
    "X2_ns",
    "X3",
    "BL2",
    "CLIM-230",
)
ARM_NAMES: Final = {
    "P0": "P0: R0 alone",
    "A0r-230": "Experiment 230's A0r",
    "T1": "T1: offset table on W1-W3",
    "P1": "P1: head on W1-W3",
    "P2": "P2: pooled head, zero shot",
    "P3": "P3: pooled head, fine-tuned",
    "X2": "X2: XGBoost on pool",
    "X3": "X3: X2 plus W1-W3 rows",
    "BL2": "BL2: blind pooled head",
    "P2_ns": "P2 without static inputs",
    "P3_ns": "P3 without static inputs",
    "X2_ns": "X2 without static inputs",
    "CLIM-230": "Climatology",
}
CONTRASTS: Final = {
    "C1": "C1: P2 minus P1 (pooling vs own history)",
    "C2": "C2: P3 minus P0 (fine-tuned pooled head vs R0)",
    "C3": "C3: P2 minus X2 (network vs XGBoost, zero shot)",
    "D2": "D2: P3 minus T1 (pooled head vs offset table)",
    "D6": "D6: P1 minus P0 (head on W1-W3 vs R0)",
    "D14": "D14: P2 without static inputs minus P0",
    "D15": "D15: P3 without static inputs minus P0",
    "D16": "D16: P2 minus X2, both without static inputs",
}


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG into the experiment's docs folder."""
    chart.save(DOCS_DIR / f"{name}.svg")


def figure_absolute(*, scores: dict) -> None:
    """Dot chart of each arm's quantile score on W1 to W3."""
    rows = pl.DataFrame(
        {"arm": [ARM_NAMES[a] for a in ARMS], "score": [scores[a][SCORES]["qs"] for a in ARMS]}
    )
    order = rows["arm"].to_list()
    dots = (
        alt.Chart(rows)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=90, color=DATA_BLUE, aria=False)
        .encode(
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
        )
    )
    save(
        chart=dots.properties(
            width=WIDTH - 230,
            title=alt.TitleParams(
                [
                    "The pooled head with no fine-tune is worse",
                    "than R0 alone, most of all with static inputs",
                ],
                subtitle=[
                    "Three wind farms (W1 to W3), 53 Tuesday runs from July 2025,",
                    "rows kept by the curtailment mask, lead days 1 and 2,",
                    "mean of 3 seeds. Dot: estimate.",
                ],
            ),
        ),
        name="figure_1",
    )


def figure_contrasts(*, contrasts: dict) -> None:
    """Interval chart of the decisive and diagnostic contrasts."""
    rows = pl.DataFrame(
        [
            {
                "contrast": CONTRASTS[key],
                "point": contrasts[key][f"{SCORES} | qs"][0],
                "low": contrasts[key][f"{SCORES} | qs"][1],
                "high": contrasts[key][f"{SCORES} | qs"][2],
                "decisive": key.startswith("C"),
            }
            for key in CONTRASTS
        ]
    )
    order = rows["contrast"].to_list()
    y = alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=360))
    colour = alt.Color(
        "decisive:N",
        scale=alt.Scale(domain=[True, False], range=[BRAND_ORANGE, DATA_BLUE]),
        legend=None,
    )
    interval = (
        alt.Chart(rows)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=3, aria=False)
        .encode(
            y=y,
            x=alt.X(
                "low:Q",
                title=["Quantile score difference", "(points of capacity; negative is better)"],
            ),
            x2="high:Q",
            color=colour,
        )
    )
    dot = alt.Chart(rows).mark_circle(size=90, aria=False).encode(y=y, x="point:Q", color=colour)  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    save(
        chart=(interval + dot + zero).properties(
            width=WIDTH - 360,
            title=alt.TitleParams(
                ["Fine-tuning helps, but no more than an offset", "table fitted on W1 to W3"],
                subtitle=[
                    "Orange: the three decisive contrasts, 98.33% interval. Blue: descriptive contrasts.",
                    "Dot: estimate. Line: interval from resampling whole months and seeds.",
                    "Dashed line: no difference.",
                ],
            ),
        ),
        name="figure_2",
    )


def figure_pool(*, per_farm: pl.DataFrame) -> None:
    """Per-farm difference between the held-out head and R0, with and without static inputs."""
    rows = per_farm.select(
        "farm",
        pl.col("fold_pool").sub(pl.col("R0")).alias("With static inputs"),
        pl.col("fold_ns_pool").sub(pl.col("R0")).alias("Without static inputs"),
    ).sort("Without static inputs")
    order = rows["farm"].to_list()
    long = rows.unpivot(index="farm", variable_name="head", value_name="difference")
    dots = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=60, aria=False)
        .encode(
            y=alt.Y("farm:N", sort=order, title=None),
            x=alt.X(
                "difference:Q",
                title="Held-out head minus R0 (points of capacity; negative is better)",
            ),
            color=alt.Color(
                "head:N",
                scale=alt.Scale(
                    domain=["With static inputs", "Without static inputs"],
                    range=[DATA_PURPLE, DATA_GREEN],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    save(
        chart=(dots + zero).properties(
            width=WIDTH - 130,
            height=520,
            padding={"left": 24, "right": 10, "top": 5, "bottom": 5},
            title=alt.TitleParams(
                [
                    "A head that has not seen a farm beats R0 on 11 of 40",
                    "public farms with static inputs, 24 without",
                ],
                subtitle=[
                    "Public Scottish wind farms, 53 Tuesday runs from July 2025, lead days 1 and 2,",
                    "head trained with the farm's fifth of the pool held out (seed 0). Dashed line: no difference.",
                ],
            ),
        ),
        name="figure_3",
    )


def figure_curve(*, pool: dict) -> None:
    """R0's score on the public farms against months of training history."""
    rows = pl.DataFrame(
        {
            "months": [12, 24, 36, 48, 54],
            "score": [pool[f"R0 last {m} months"]["days 1-2"] for m in (12, 24, 36, 48)]
            + [pool["R0"]["days 1-2"]],
        }
    )
    line = (
        alt.Chart(rows)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=DATA_GREEN, aria=False)
        .encode(
            x=alt.X(
                "months:Q",
                title="Months of ERA5 and output history before April 2024 (last point: all available history)",
            ),
            y=alt.Y(
                "score:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[9.0, 9.5]),
            ),
        )
    )
    save(
        chart=line.properties(
            width=WIDTH - 120,
            title=alt.TitleParams(
                "More history does not improve R0 on the public farms",
                subtitle=[
                    "40 public Scottish wind farms, XGBoost R0 refitted on the last N months before April 2024,",
                    "53 Tuesday runs from July 2025, lead days 1 and 2.",
                ],
            ),
        ),
        name="figure_4",
    )


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    pool = json.loads((common.WORK_DIR / "pool_score.json").read_text())
    per_farm = pl.read_parquet(common.WORK_DIR / "pool_score_per_farm.parquet")
    figure_absolute(scores=score["scores"])
    figure_contrasts(contrasts=score["contrasts"])
    figure_pool(per_farm=per_farm)
    figure_curve(pool=pool)


if __name__ == "__main__":
    main()
