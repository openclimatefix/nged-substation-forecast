"""Draw experiment 00032's figures from the saved results, as SVG.

1. The eight planned contrasts and the post hoc cold-start contrasts, with intervals.
2. The quantile score of the cold-start arms on the 8 held-out substations.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00032-substation-fingerprints-pooled-demand/e00032_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00032_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "73 forecast runs, one every 5 days, July 2025 to June 2026, ENS weather."
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: ID and location, minus ID",
    "C2": "C2: ID and training-window statistics, minus ID",
    "C3": "C3: ID and recent-window statistics, minus ID",
    "C4": "C4: ID and both, minus ID",
    "C5": "C5: training-window statistics without ID, minus ID",
    "C6": "C6: training-window statistics without ID, minus no ID",
    "C7": "C7: held-out, training-window statistics, minus none",
    "C8": "C8: held-out, training-window and recent, minus none",
}
ARM_LABELS: Final[dict[str, str]] = {
    "type_id_stack": "Pool with the ID (the series was in training)",
    "type_noid_stack": "Pool without the ID (series in training)",
    "cold_recent_stack": "Held out, recent-window statistics (post hoc)",
    "cold_none_stack": "Held out, no fingerprint",
    "cold_train_recent_fullwindow_stack": "Held out, both, whole-window (post hoc)",
    "cold_static_stack": "Held out, location",
    "cold_train_fullwindow_stack": "Held out, training, whole window (post hoc)",
    "cold_train_recent_stack": "Held out, both",
    "cold_train_stack": "Held out, training-window statistics",
    "manual_heuristic": "manual_heuristic",
    "climatology": "Climatology",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The eight planned contrasts on the quantile score over all leads."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("planned")
        & (pl.col("metric") == "quantile_score")
        & (pl.col("scope") == "all leads")
    )
    data = data.with_columns(label=pl.col("label").str.slice(0, 2).replace_strict(CONTRAST_LABELS))
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N",
            sort=list(CONTRAST_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=330),
        )
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Fingerprints add nothing to the series ID and hurt a held-out substation",
            lines=[
                (
                    "Each row is the first arm minus the second on the quantile score over all "
                    "leads; below zero favours the first arm. Grey line: no difference. C1 to C6 "
                    "use the 16 Primary substations and three seeds; C7 and C8 use 8 held-out "
                    "Primary substations and one seed. " + SCOPE
                ),
                "Dot: estimate. Line: 99.375% interval (Bonferroni across eight contrasts).",
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The quantile score of each cold-start arm on the held-out substations."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("scope") == "all leads")
            & (pl.col("table") == "cold")
            & (pl.col("group") == "Held-out (8)")
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N",
            sort=data.sort("value")["label"].to_list(),
            title=None,
            axis=alt.Axis(labelLimit=330),
        )
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title="Quantile score (points of capacity)", scale=alt.Scale(zero=False))
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH - 200,
        height=260,
        title=_title(
            text="Recent power statistics help a new substation; training statistics hurt it",
            lines=[
                (
                    "Quantile score over all leads (smaller is better) on the 8 held-out Primary "
                    "substations, after the 14-day error correction and spread factor of 1.15. "
                    "One seed. Dot: estimate. Line: 95% interval from re-drawing whole months. "
                    + SCOPE
                )
            ],
        ),
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
