"""Draw experiment 00090's figures from the saved results, as SVG.

1. The three planned contrasts on the rows whose lag sits inside an oracle event, 98.33% intervals.
2. The no-harm check and related contrasts on all rows outside switching.
3. Each arm's absolute quantile score on the rows with a lag inside an oracle event.
4. How many scored rows each label source puts a lag inside an event for.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00090-*/e00090_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00090_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
OUT_DIR: Final[Path] = common.CHART_DIR
SCOPE: Final[str] = "16 Primary substations, 292 daily forecast runs, July 2025 to June 2026."
LABELS: Final[dict[str, str]] = {
    "C1": "Detector labels (XL) minus plain",
    "C2": "Oracle labels minus plain",
    "C3": "Detector labels minus oracle labels",
}
OUTSIDE_LABELS: Final[dict[str, str]] = {
    "N1": "Detector labels minus plain",
    "D1 C2 on all rows outside switching": "Oracle labels minus plain",
    "D9 C3 on all rows outside switching": "Detector labels minus oracle labels",
    "D7 mask_xl minus plain, outside switching": "Detector-masked training minus plain",
}
ARM_LABELS: Final[dict[str, str]] = {
    "plain": "Plain (no label input)",
    "lagmw_oracle": "Corrected lags + MW, oracle labels",
    "lagmw_xl": "Corrected lags + MW, detector labels",
    "mask_xl": "Detector-masked training",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
ARM_KIND: Final[dict[str, str]] = {
    "plain": "This experiment",
    "lagmw_oracle": "This experiment",
    "lagmw_xl": "This experiment",
    "mask_xl": "This experiment",
    "manual_heuristic": "Baseline",
    "climatology": "Baseline",
}
KIND_COLOURS: Final[dict[str, str]] = {
    "This experiment": ocf.DATA_BLUE,
    "Baseline": ocf.DATA_PURPLE,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _rule(*, value: float, colour: str, dash: bool = False) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [value]}))
        .mark_rule(color=colour, strokeDash=[4, 3] if dash else [])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )


def _forest(
    *, data: pl.DataFrame, order: list[str], title: alt.TitleParams, extra: list
) -> alt.TopLevelMixin:
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=380))
    )
    points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title=(
                "Quantile score, first arm minus second (points of capacity; below zero is better)"
            ),
        )
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_rule(value=0.0, colour="#999999"), *extra, bars, points).properties(
        width=PLOT_WIDTH, height=130, title=title
    )


def _contrasts(*, folder: Path, labels: dict[str, str], level: str) -> pl.DataFrame:
    return (
        pl.read_parquet(folder / "contrasts.parquet")
        .filter(
            pl.col("id").is_in(list(labels))
            & (pl.col("level_label") == level)
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
        )
        .with_columns(label=pl.col("id").replace_strict(labels))
    )


def figure_1(*, folder: Path) -> alt.TopLevelMixin:
    """The planned contrasts on the rows whose lag sits inside an oracle event."""
    return _forest(
        data=_contrasts(folder=folder, labels=LABELS, level="planned"),
        order=list(LABELS.values()),
        title=_title(
            text="Correcting lags with the detector's labels did not lower the error",
            lines=[
                (
                    SCOPE + " Only the 37,394 half-hours (1.3%) outside switching whose lag "
                    "sits inside an event NGED's logs label, in 8 months."
                ),
                (
                    "Dot: estimate. Line: 98.33% interval from re-drawing whole months. "
                    "Zero: same score."
                ),
            ],
        ),
        extra=[],
    )


def figure_2(*, folder: Path) -> alt.TopLevelMixin:
    """The no-harm check on all rows outside switching."""
    return _forest(
        data=pl.concat(
            [
                _contrasts(folder=folder, labels={"N1": OUTSIDE_LABELS["N1"]}, level="planned"),
                _contrasts(
                    folder=folder,
                    labels={k: v for k, v in OUTSIDE_LABELS.items() if k != "N1"},
                    level="95",
                ),
            ]
        ),
        order=list(OUTSIDE_LABELS.values()),
        title=_title(
            text="Outside switching, the detector's labels made the forecast slightly worse",
            lines=[
                SCOPE + " All half-hours outside labelled switching and logged closures.",
                (
                    "Dot: estimate. Line: 98.33% interval for the top row, 95% for the others. "
                    "Dashed line: +0.02, the harm the plan allows."
                ),
            ],
        ),
        extra=[_rule(value=0.02, colour=ocf.BRAND_ORANGE, dash=True)],
    )


def figure_3(*, folder: Path) -> alt.TopLevelMixin:
    """Each arm's absolute quantile score on the rows with a lag inside an oracle event."""
    data = (
        pl.read_parquet(folder / "absolute.parquet")
        .filter((pl.col("subset") == "lag_oracle") & pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            kind=pl.col("arm").replace_strict(ARM_KIND),
        )
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
        ),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(KIND_COLOURS), range=list(KIND_COLOURS.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    return (
        base.mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            )
        )
        .properties(
            width=PLOT_WIDTH,
            height=200,
            title=_title(
                text="On rows with a lag inside an event, no arm beats the plain arm clearly",
                lines=[
                    SCOPE + " Mean over the 37,394 half-hours with a lag inside an event "
                    "(oracle labels), outside switching.",
                    "Dot: mean score. Intervals are in figure 1.",
                ],
            ),
        )
    )


def figure_4(*, folder: Path) -> alt.TopLevelMixin:
    """How many scored rows each label source puts a lag inside an event for."""
    counts = pl.read_parquet(folder / "label_counts.parquet").row(0, named=True)
    data = pl.DataFrame(
        {
            "label": [
                "Oracle labels (NGED's logs)",
                "Detector labels (XL)",
                "Both",
                "Detector only",
            ],
            "rows": [
                counts["lag_oracle"],
                counts["lag_xl"],
                counts["lag_both"],
                counts["lag_xl_only"],
            ],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=data["label"].to_list(), title=None),
            x=alt.X("rows:Q", title="Scored half-hours with a lag inside an event"),
        )
        .properties(
            width=PLOT_WIDTH,
            height=140,
            title=_title(
                text="The detector marks about eight times as many rows as NGED's logs",
                lines=[SCOPE + " Rows outside switching."],
            ),
        )
    )


def main() -> None:
    """Write the figures."""
    folder = common.RESULTS_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, draw in enumerate((figure_1, figure_2, figure_3, figure_4), start=1):
        draw(folder=folder).save(OUT_DIR / f"figure_{index}.svg")
        print(f"figure {index} written")


if __name__ == "__main__":
    main()
