"""Draw experiment 00091's figures from the saved results, as SVG.

1. Change in Brier score against the binary flag, for three arms on three row sets.
2. Area under the receiver operating characteristic curve by duration of the injected transfer.
3. Reliability of the probability of an abnormal running arrangement.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00091-*/e00091_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00091_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS: Final[Path] = common.WORK_DIR / "results"
OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00091-sw2-26-calibrated-probability-of-abnormal-arrangement"
)
PLOT_WIDTH: Final[int] = 540
WRAP: Final[int] = 92
ARM_LABELS: Final[dict[str, str]] = {
    "flag": "Arm 1: yes-or-no flag",
    "latest": "Arm 2: latest candidate",
    "latest_x1": "Arm 3: latest candidate and step size",
    "best": "Post hoc: highest-scoring candidate",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "flag": ocf.DATA_PURPLE,
    "latest": ocf.DATA_BLUE,
    "latest_x1": ocf.DATA_SKY,
    "best": ocf.BRAND_ORANGE,
}
VARIANT_LABELS: Final[dict[str, str]] = {
    "all": "All rows (pre-registered)",
    "clean": "Without rows the uninjected detector flags",
    "clean_large": "Also without transfers too small to see",
}
SCOPE: Final[str] = (
    "16 Primary substations, 292 daily forecast origins, July 2025 to June 2026, "
    "injected transfers that start in even months."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _colour(*, arms: list[str]) -> alt.Color:
    return alt.Color(
        "arm_label:N",
        scale=alt.Scale(domain=[ARM_LABELS[a] for a in arms], range=[ARM_COLOURS[a] for a in arms]),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420),
    )


def figure_1() -> alt.TopLevelMixin:
    """The planned contrasts: arms 2 and 3 and the post hoc arm against the flag."""
    data = pl.concat(
        [pl.read_parquet(RESULTS / f"contrasts_{v}.parquet") for v in VARIANT_LABELS]
    ).with_columns(
        variant_label=pl.col("variant").replace_strict(VARIANT_LABELS),
        arm_label=pl.col("arm").replace_strict(ARM_LABELS),
        change=pl.col("change") * 100,
        low=pl.col("low") * 100,
        high=pl.col("high") * 100,
    )
    arms = ["latest", "latest_x1", "best"]
    base = alt.Chart(data).encode(
        y=alt.Y(
            "variant_label:N",
            sort=list(VARIANT_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=380),
        ),
        yOffset=alt.YOffset("arm_label:N", sort=[ARM_LABELS[a] for a in arms]),
        color=_colour(arms=arms),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "change:Q", title="Brier score minus arm 1's, as % of arm 1's (below zero is better)"
        )
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    rules = alt.Chart(
        pl.DataFrame({"x": [0.0, -20.0], "kind": ["Same as arm 1", "Target: 20% lower"]})
    )
    rule_marks = rules.mark_rule(strokeDash=[4, 3], color="#777777").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(rule_marks, bars, points).properties(
        width=PLOT_WIDTH,
        height=190,
        title=_title(
            text="A calibrated score did not beat the yes-or-no flag on the pre-registered rows",
            lines=[
                SCOPE + " Dot: estimate. Line: 97.5% interval from re-drawing whole injection "
                "batches. Left dashed rule: 20% lower than the flag, the target. Right dashed "
                "rule: same as the flag."
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Area under the curve by duration of the newest injected transfer."""
    bins = ("under 24 h", "1 to 3 d", "3 to 14 d", "over 14 d")
    data = pl.read_parquet(RESULTS / "duration_bins.parquet").with_columns(
        arm_label=pl.col("arm").replace_strict(ARM_LABELS),
        label=pl.col("bin") + " (" + pl.col("positives").cast(pl.String) + " positive rows)",
    )
    order = [data.filter(pl.col("bin") == b)["label"][0] for b in bins]
    arms = ["flag", "latest", "best"]
    chart = (
        alt.Chart(data.filter(pl.col("arm").is_in(arms)))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "auc:Q",
                scale=alt.Scale(domain=[0.5, 1.0]),
                title="Area under the receiver operating characteristic curve (higher is better)",
            ),
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=380)),
            yOffset=alt.YOffset("arm_label:N", sort=[ARM_LABELS[a] for a in arms]),
            color=_colour(arms=arms),
        )
    )
    return chart.properties(
        width=PLOT_WIDTH,
        height=190,
        title=_title(
            text="The probability separates well for transfers under 14 days and poorly beyond",
            lines=[
                SCOPE + " Rows of the clean set: both draws of injected transfers, without rows "
                "the uninjected detector flags. Bins hold the duration of the newest injected "
                "transfer active at each origin. 0.5 means no separation."
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Reliability curves on the pre-registered rows."""
    data = (
        pl.read_parquet(RESULTS / "reliability_all.parquet")
        .filter(pl.col("n") >= 200)
        .with_columns(arm_label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    arms = ["flag", "latest", "latest_x1"]
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#777777", strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_p:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted probability"),
            y=alt.Y(
                "observed:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Share of rows inside an injected transfer",
            ),
            color=_colour(arms=arms),
            order="mean_p:Q",
        )
    )
    return alt.layer(diagonal, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="Probabilities above 0.5 are too high",
            lines=[
                SCOPE + " Pre-registered rows, shapes S2, S3, S4, and S6. One point per tenth of "
                "predicted probability that holds at least 200 rows. Dashed line: perfectly "
                "calibrated. 97% of rows fall in the lowest bin."
            ],
        ),
    )


def main() -> None:
    """Write the three figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, draw in enumerate((figure_1, figure_2, figure_3), start=1):
        draw().save(OUT_DIR / f"figure_{index}.svg")


if __name__ == "__main__":
    main()
