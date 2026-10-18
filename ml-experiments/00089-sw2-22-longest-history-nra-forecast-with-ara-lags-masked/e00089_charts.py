"""Draw experiment 00089's figures from the saved results, as SVG.

1. The three planned contrasts, 98.33% intervals.
2. The no-harm check and the descriptive contrasts on all rows outside switching.
3. Each arm's absolute quantile score on all rows outside switching and on the rows with a lag
   inside an abnormal running arrangement (ARA).
4. The share of target half-hours inside an ARA, per year and label source.
5. How often the 0.05 and 0.95 quantiles are crossed, per arm.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00089-*/e00089_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00089_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
OUT_DIR: Final[Path] = common.CHART_DIR
SCOPE: Final[str] = "16 Primary substations, 292 daily forecast runs, July 2025 to June 2026."
LABELS: Final[dict[str, str]] = {
    "C1": "C1: lags in an ARA masked (LM) minus not masked (L), rows with a lag in an ARA",
    "C2": "C2: longest history (L) minus 12 months (R), all rows outside switching",
    "C3": "C3: longest history, masked (LM) minus 12 months (R), rows with a lag in an ARA",
}
OUTSIDE_LABELS: Final[dict[str, str]] = {
    "N1": "LM minus L (98.33% interval)",
    "D4 M minus R on all rows outside switching": "M minus R (12 months, masked minus not)",
    "D6 LM_ens minus L_ens on all rows outside switching": "LM_ens minus L_ens (from 2024-04)",
    "D7 L_ens minus R on all rows outside switching": "L_ens minus R (from 2024-04 vs 12 months)",
    "D8 L minus L_ens on all rows outside switching": "L minus L_ens (2021-03 vs 2024-04 start)",
}
ARM_LABELS: Final[dict[str, str]] = {
    "r": "R: 12 months",
    "m": "M: 12 months, masked",
    "l": "L: longest history",
    "lm": "LM: longest history, masked",
    "l_ens": "L_ens: from 2024-04",
    "lm_ens": "LM_ens: from 2024-04, masked",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
}
SUBSET_LABELS: Final[dict[str, str]] = {
    "nra": "All rows outside switching",
    "lag_ara": "Rows with a lag in an ARA",
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
        width=PLOT_WIDTH, height=150, title=title
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
    """The planned contrasts."""
    return _forest(
        data=_contrasts(folder=folder, labels=LABELS, level="planned"),
        order=list(LABELS.values()),
        title=_title(
            text="Masking lags inside an ARA raised the error; the longer history changed nothing",
            lines=[
                SCOPE + " Rows outside switching.",
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
            text="Outside switching, masking raised the error and the start date did not matter",
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
    """Each arm's absolute quantile score on two row subsets."""
    data = (
        pl.read_parquet(folder / "absolute.parquet")
        .filter(pl.col("subset").is_in(list(SUBSET_LABELS)) & pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            rows=pl.col("subset").replace_strict(SUBSET_LABELS),
        )
    )
    return (
        alt.Chart(data)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None),
            x=alt.X(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "rows:N",
                scale=alt.Scale(
                    domain=list(SUBSET_LABELS.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=220,
            title=_title(
                text="Absolute scores: the arms sit within 0.1 points outside switching",
                lines=[
                    SCOPE + " manual_heuristic and climatology are scored on the same rows.",
                    "Dot: mean score. Intervals are in figures 1 and 2.",
                ],
            ),
        )
    )


def figure_4(*, folder: Path) -> alt.TopLevelMixin:
    """The share of half-hours inside an ARA per year."""
    data = (
        pl.read_parquet(folder / "events_per_year.parquet")
        .with_columns(
            ara=100.0 * pl.col("ara_halfhours") / pl.col("all_halfhours"),
            ro_nc=100.0 * pl.col("xl_halfhours") / pl.col("all_halfhours"),
            logged_event=100.0 * pl.col("oracle_event_halfhours") / pl.col("all_halfhours"),
            logged_closure=100.0 * pl.col("oracle_closed_halfhours") / pl.col("all_halfhours"),
        )
        .unpivot(
            ["ara", "ro_nc", "logged_event", "logged_closure"],
            index="year",
            variable_name="source",
            value_name="share",
        )
        .with_columns(
            source=pl.col("source").replace_strict(
                {
                    "ara": "Union (the ARA)",
                    "ro_nc": "RO-NC detector",
                    "logged_event": "NGED logged events",
                    "logged_closure": "NGED logged closures",
                }
            )
        )
    )
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("year:O", title="Calendar year (2026 is January to June)"),
            y=alt.Y("share:Q", title="Share of half-hours inside an ARA (%)"),
            color=alt.Color("source:N", legend=alt.Legend(orient="top", title=None)),
        )
        .properties(
            width=PLOT_WIDTH,
            height=200,
            title=_title(
                text="Neither label source covers 2021, so that year trains on unmasked switching",
                lines=[
                    "16 Primary substations, all half-hours of each year; 2021 starts in March."
                ],
            ),
        )
    )


def figure_5(*, folder: Path) -> alt.TopLevelMixin:
    """How often the 0.05 and 0.95 quantiles are crossed, per arm."""
    data = (
        pl.read_parquet(folder / "absolute.parquet")
        .filter((pl.col("subset") == "nra") & pl.col("arm").is_in(list(ARM_LABELS)[:6]))
        .select("arm", ex05="ex05", ex95="ex95")
        .unpivot(["ex05", "ex95"], index="arm", variable_name="tail", value_name="rate")
        .with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            tail=pl.col("tail").replace_strict(
                {"ex05": "Actual below the 0.05 quantile", "ex95": "Actual above the 0.95 quantile"}
            ),
        )
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None),
            x=alt.X("rate:Q", title="Share of half-hours (nominal 0.05)"),
            color=alt.Color("tail:N", legend=alt.Legend(orient="top", title=None)),
        )
    )
    return alt.layer(_rule(value=0.05, colour="#999999", dash=True), points).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text="The longer history crosses the 0.95 quantile less often, 0.064 against 0.078",
            lines=[SCOPE + " Rows outside switching. Dashed line: the nominal rate, 0.05."],
        ),
    )


def main() -> None:
    """Write the figures."""
    folder = common.RESULTS_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, draw in enumerate((figure_1, figure_2, figure_3, figure_4, figure_5), start=1):
        draw(folder=folder).save(OUT_DIR / f"figure_{index}.svg")
        print(f"figure {index} written")


if __name__ == "__main__":
    main()
