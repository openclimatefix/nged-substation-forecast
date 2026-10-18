"""Draw experiment 00075's figures from the saved results, as SVG.

1. The three planned contrasts with their 98.33% intervals, and the non-inferiority margin.
2. The same three contrasts on the half-hours whose lag falls inside a labelled event.
3. Each arm's absolute quantile score on the NRA rows, beside experiment 00048's benchmarks.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00075-*/e00075_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00075_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
OUT_DIR: Final[Path] = common.CHART_DIR
SCOPE: Final[str] = "16 Primary substations, 292 daily forecast runs, July 2025 to June 2026."
LABELS: Final[dict[str, str]] = {
    "C1": "Masked training targets minus plain",
    "C2": "Masked + signed MW columns minus masked",
    "C3": "Masked + lags minus MW minus masked",
}
SUBSET_IDS: Final[dict[str, str]] = {
    "D7": "Masked training targets minus plain",
    "D8": "Masked + signed MW columns minus masked",
    "D9": "Masked + lags minus MW minus masked",
}
ARM_LABELS: Final[dict[str, str]] = {
    "plain": "Plain weekly (nothing removed)",
    "nra": "Masked training targets",
    "nra_mw": "Masked + signed MW columns",
    "nra_lag": "Masked + lags minus MW",
    "experiment 00048 weekly + 14-day correction": "Weekly + 14-day correction",
    "experiment 00048 stack": "Stack: frozen + 14-day correction",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
ARM_KIND: Final[dict[str, str]] = {
    "plain": "This experiment",
    "nra": "This experiment",
    "nra_mw": "This experiment",
    "nra_lag": "This experiment",
    "experiment 00048 weekly + 14-day correction": "Earlier benchmark",
    "experiment 00048 stack": "Earlier benchmark",
    "manual_heuristic": "Baseline",
    "climatology": "Baseline",
}
KIND_COLOURS: Final[dict[str, str]] = {
    "This experiment": ocf.DATA_BLUE,
    "Earlier benchmark": ocf.BRAND_ORANGE,
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


def figure_1(*, folder: Path) -> alt.TopLevelMixin:
    """The planned contrasts with their Bonferroni-level intervals."""
    data = (
        pl.read_parquet(folder / "contrasts.parquet")
        .filter(
            pl.col("id").is_in(list(LABELS))
            & (pl.col("level_label") == "planned")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
        )
        .with_columns(label=pl.col("id").replace_strict(LABELS))
    )
    return _forest(
        data=data,
        order=list(LABELS.values()),
        title=_title(
            text="Masking switching and adding its size changed the score by under 0.06 points",
            lines=[
                SCOPE + " Scored on half-hours outside labelled switching and logged closures.",
                (
                    "Dot: estimate. Line: 98.33% interval from re-drawing whole months. Zero: "
                    "same score. Dashed line: +0.02, the harm the masked arm may show and still "
                    "be adopted."
                ),
            ],
        ),
        extra=[_rule(value=0.02, colour=ocf.BRAND_ORANGE, dash=True)],
    )


def figure_2(*, folder: Path) -> alt.TopLevelMixin:
    """The same contrasts on the half-hours whose lag falls inside a labelled event."""
    data = (
        pl.read_parquet(folder / "contrasts.parquet")
        .filter(
            pl.col("id").is_in(list(SUBSET_IDS))
            & (pl.col("level_label") == "95")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
        )
        .with_columns(label=pl.col("id").replace_strict(SUBSET_IDS))
    )
    return _forest(
        data=data,
        order=list(SUBSET_IDS.values()),
        title=_title(
            text="Where a lag sits inside a labelled event, correcting the lag helped most",
            lines=[
                (
                    SCOPE + " Only the 37,394 half-hours (1.3%) whose newest, same-day, or "
                    "same-week lag falls in a labelled event, in 8 months."
                ),
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months. Zero: same "
                    "score. Read as description: the subset is small and the intervals are not "
                    "corrected for several comparisons."
                ),
            ],
        ),
        extra=[],
    )


def figure_3(*, folder: Path) -> alt.TopLevelMixin:
    """Each arm's absolute quantile score on the NRA rows."""
    data = (
        pl.read_parquet(folder / "absolute.parquet")
        .filter((pl.col("subset") == "nra") & pl.col("arm").is_in(list(ARM_LABELS)))
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
                text="All four weekly arms score within 0.02 points of each other",
                lines=[
                    (
                        SCOPE + " Mean over the half-hours outside labelled switching and logged "
                        "closures that every arm holds (2,928,842 cases)."
                    ),
                    (
                        "Dot: mean score. The paired contrasts in figure 1 are much tighter than "
                        "the uncertainty of any one dot, so no interval is drawn here."
                    ),
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    folder = common.RESULTS_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, draw in enumerate((figure_1, figure_2, figure_3), start=1):
        draw(folder=folder).save(OUT_DIR / f"figure_{index}.svg")
        print(f"figure {index} written")


if __name__ == "__main__":
    main()
