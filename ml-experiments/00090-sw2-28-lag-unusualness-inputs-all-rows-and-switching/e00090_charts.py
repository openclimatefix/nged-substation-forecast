"""Draw experiment 00090's figures from the saved results, as SVG.

1. The two planned contrasts, `clim_pit` minus `plain`, 97.5% intervals.
2. The all-row contrast by lead band, 95% intervals.
3. Coverage of the 5%-to-95% band for each arm, on all rows outside switching and on the rows with
   a lag inside a logged event.

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
    "C1": "All rows outside switching",
    "C2": "Rows with a lag inside a logged event",
}
ARM_LABELS: Final[dict[str, str]] = {
    "plain": "Plain (no lag PIT)",
    "clim_pit": "With lag PIT against climatology",
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
    """The two planned contrasts."""
    return _forest(
        data=_contrasts(folder=folder, labels=LABELS, level="planned"),
        order=list(LABELS.values()),
        title=_title(
            text="Giving the model each lag's PIT raised the error on all rows",
            lines=[
                SCOPE + " Both arms train on rows outside switching only.",
                "Dot: estimate. Line: 97.5% interval (whole months re-drawn). Zero: no change.",
            ],
        ),
        extra=[],
    )


def figure_2(*, folder: Path) -> alt.TopLevelMixin:
    """The all-row contrast by lead band."""
    data = (
        pl.read_parquet(folder / "contrasts.parquet")
        .filter(
            (pl.col("id") == "C1") & (pl.col("level_label") == "95") & (pl.col("site") == "pooled")
        )
        .with_columns(label=pl.col("scope"))
    )
    return _forest(
        data=data,
        order=["all leads", *(name for name, _, _ in common.LEAD_BANDS)],
        title=_title(
            text="The harm is similar at every lead time",
            lines=[
                SCOPE + " All half-hours outside switching.",
                "Dot: estimate. Line: 95% interval from re-drawing whole months.",
            ],
        ),
        extra=[],
    ).properties(height=170)


def figure_3(*, folder: Path) -> alt.TopLevelMixin:
    """Coverage of the 5%-to-95% band by arm and row set."""
    data = (
        pl.read_parquet(folder / "absolute.parquet")
        .filter(pl.col("subset").is_in(["nra", "lag_oracle"]) & (pl.col("scope") == "all leads"))
        .with_columns(
            arm_label=pl.col("arm").replace_strict(ARM_LABELS),
            rows=pl.col("subset").replace_strict({"nra": LABELS["C1"], "lag_oracle": LABELS["C2"]}),
        )
    )
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "rows:N", sort=list(LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
            ),
            yOffset="arm_label:N",
            x=alt.X(
                "cov90:Q",
                title="Share of observations inside the q05-to-q95 band",
                scale=alt.Scale(domain=[0.6, 1.0]),
            ),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(
                    domain=list(ARM_LABELS.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    return alt.layer(bars, _rule(value=0.9, colour="#999999", dash=True)).properties(
        width=PLOT_WIDTH,
        height=170,
        title=_title(
            text="The PIT arm's quantile band is narrower than it should be on both row sets",
            lines=[SCOPE + " Dashed line: the nominal 0.90."],
        ),
    )


def main() -> None:
    """Write the figures."""
    folder = common.RESULTS_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, draw in enumerate((figure_1, figure_2, figure_3), start=1):
        draw(folder=folder).save(OUT_DIR / f"figure_{index}.svg")
        print(f"figure {index} written")


if __name__ == "__main__":
    main()
