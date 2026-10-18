"""Draw experiment 00088's figures from the saved results, as SVG.

1. Error against the true unmodified power by time since the transfer began, for the arms.
2. The planned contrasts and the post hoc variants, with their intervals.
3. The share of the gap between the plain booster and the ceiling that each arm closes.
4. Error after a transfer has ended, and how often the detector still marks it.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00088-*/e00088_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00088_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / (f"{common.EXPERIMENT_ID:05d}-{common.SLUG}")
)
RESULTS: Final[Path] = common.RESULTS_DIR
SCOPE: Final[str] = (
    "Injected transfers at 16 Primary substations, starting in October 2025 and February, "
    "April, and June 2026; shapes S2, S3, S4, and S6."
)
ARM_LABELS: Final[dict[str, str]] = {
    "arm1": "Plain booster, contaminated lags",
    "arm5": "Ceiling: plain booster, unmodified lags",
    "arm2_caus": "Lag correction, RO-NC MW as built",
    "arm3_inj": "Oracle-trained NRA, injected MW",
    "arm4_caus": "RO-NC-trained NRA, RO-NC MW as built",
    "arm4_cmax": "RO-NC-trained NRA, RO-NC MW largest step (post hoc)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "arm1": ocf.BRAND_ORANGE,
    "arm5": ocf.DATA_GREEN,
    "arm2_caus": ocf.DATA_SKY,
    "arm3_inj": ocf.DATA_BLUE,
    "arm4_caus": ocf.DATA_PURPLE,
    "arm4_cmax": ocf.DATA_PURPLE_LIGHT,
}
ARM_SHAPES: Final[dict[str, str]] = {
    "arm1": "circle",
    "arm5": "square",
    "arm2_caus": "triangle-up",
    "arm3_inj": "diamond",
    "arm4_caus": "cross",
    "arm4_cmax": "triangle-down",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _arm_scale() -> tuple[alt.Scale, alt.Legend]:
    arms = list(ARM_LABELS)
    scale = alt.Scale(domain=[ARM_LABELS[a] for a in arms], range=[ARM_COLOURS[a] for a in arms])
    return scale, alt.Legend(orient="top", columns=1, title=None, labelLimit=420)


def _by_bin(*, name: str, order: list[str], title: alt.TitleParams) -> alt.TopLevelMixin:
    data = (
        pl.read_parquet(RESULTS / f"{name}.parquet")
        .filter(pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    scale, legend = _arm_scale()
    shapes = alt.Scale(
        domain=[ARM_LABELS[a] for a in ARM_LABELS], range=[ARM_SHAPES[a] for a in ARM_LABELS]
    )
    base = alt.Chart(data).encode(
        x=alt.X("bin:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=[ARM_LABELS[a] for a in ARM_LABELS]),
        color=alt.Color("label:N", scale=scale, legend=legend),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "value:Q",
            scale=alt.Scale(zero=False),
            title="Mean absolute error of the median (points of capacity; smaller is better)",
        ),
        shape=alt.Shape("label:N", scale=shapes, legend=None),
    )
    bars = base.mark_rule().encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(width=PLOT_WIDTH, height=260, title=title)


def figure_1() -> alt.TopLevelMixin:
    """Error against the truth by time since onset, family A."""
    return _by_bin(
        name="arm_by_onset",
        order=["0-24 h", "1-3 d", "over 3 d"],
        title=_title(
            text="With the true MW the NRA forecast recovers the gap; with the detector's, not yet",
            lines=[
                SCOPE,
                (
                    "Forecast origins inside the transfer, by time since it began. Error is "
                    "measured against the true unmodified power. Dot: estimate, weighted by "
                    "the MW moved. Line: 95% interval from re-drawing whole months. RO-NC: "
                    "refit-offset neighbour-count detector, run only on data before the origin."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts and the post hoc variants."""
    names = {
        "C1 arm4_caus - arm1": "C1 RO-NC-trained NRA, RO-NC MW, minus plain",
        "C2 arm3_inj - arm1": "C2 Oracle-trained NRA, injected MW, minus plain",
        "C3 arm4_caus - arm2_caus": "C3 NRA minus lag correction alone (both RO-NC MW)",
        "C4 arm2_caus - arm1": "C4 Lag correction alone, RO-NC MW, minus plain",
        "post hoc arm4_cmax - arm1": "Post hoc: NRA, largest-step MW, minus plain",
        "post hoc arm2_cmax - arm1": "Post hoc: lag correction, largest-step MW, minus plain",
    }
    data = (
        pl.read_parquet(RESULTS / "planned_contrasts.parquet")
        .filter(pl.col("contrast").is_in(list(names)))
        .with_columns(
            label=pl.col("contrast").replace_strict(names),
            kind=pl.when(pl.col("contrast").str.starts_with("post hoc"))
            .then(pl.lit("Post hoc"))
            .otherwise(pl.lit("Planned (98.75% interval)")),
        )
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=380)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["Planned (98.75% interval)", "Post hoc"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "point:Q",
            title=(
                "Mean absolute error, first arm minus second "
                "(points of capacity; below zero is better)"
            ),
        ),
        shape=alt.Shape("kind:N", legend=None),
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=190,
        title=_title(
            text="The detector's MW as built did not help; counting the transfer once did",
            lines=[
                SCOPE,
                (
                    "Forecast origins inside the transfer. Dot: estimate. Line: interval from "
                    "re-drawing the four onset months. Zero: same error as the contrast's "
                    "second arm."
                ),
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """The share of the gap each arm closes, by time since onset."""
    names = {
        "arm2_inj": "Lag correction, injected MW",
        "arm4_inj": "RO-NC-trained NRA, injected MW",
        "arm3_inj": "Oracle-trained NRA, injected MW",
        "arm2_caus": "Lag correction, RO-NC MW as built",
        "arm4_caus": "RO-NC-trained NRA, RO-NC MW as built",
        "arm4_cmax": "RO-NC-trained NRA, largest-step MW (post hoc)",
    }
    data = (
        pl.read_parquet(RESULTS / "gap_shares_headline.parquet")
        .filter(
            pl.col("arm").is_in(list(names)) & pl.col("bin").is_in(["all", "1-3 d", "over 3 d"])
        )
        .with_columns(label=pl.col("arm").replace_strict(names))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=380)),
        color=alt.Color(
            "bin:N",
            sort=["all", "1-3 d", "over 3 d"],
            scale=alt.Scale(
                domain=["all", "1-3 d", "over 3 d"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(orient="top", title="Time since onset"),
        ),
        yOffset=alt.YOffset("bin:N", sort=["all", "1-3 d", "over 3 d"]),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "share:Q",
            scale=alt.Scale(domain=[-0.5, 1.0]),
            title="Share of the gap to the ceiling closed (0 = plain booster, 1 = ceiling)",
        ),
        shape=alt.Shape("bin:N", legend=None),
    )
    bars = base.mark_rule().encode(x="share_low:Q", x2="share_high:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="With the true MW the boosters close 50% to 83% of the gap",
            lines=[
                SCOPE,
                (
                    "Forecast origins inside the transfer. Dot: estimate. Line: 95% interval "
                    "from re-drawing whole months. Shares are only meaningful where the gap's "
                    "own interval excludes zero (it does in every bin shown)."
                ),
            ],
        ),
    )


def figure_4() -> alt.TopLevelMixin:
    """Error after a transfer has ended, family B."""
    return _by_bin(
        name="arm_by_since_end",
        order=["0-24 h", "1-3 d", "3-7 d"],
        title=_title(
            text="After a transfer ends, correcting with the detector's MW raises the error",
            lines=[
                SCOPE,
                (
                    "Forecast origins after the transfer has ended, by time since its end, "
                    "while a lag still lies inside it. Error is measured against the true "
                    "unmodified power. Dot: estimate. Line: 95% interval from re-drawing "
                    "whole months."
                ),
            ],
        ),
    )


def main() -> None:
    """Write the four figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number, figure in enumerate((figure_1(), figure_2(), figure_3(), figure_4()), start=1):
        figure.save(OUT_DIR / f"figure_{number}.svg")
        print(f"wrote figure_{number}.svg")


if __name__ == "__main__":
    main()
