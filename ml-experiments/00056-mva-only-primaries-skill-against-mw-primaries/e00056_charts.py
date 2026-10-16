"""Draw experiment 00056's figures and print the leave-one-series-out gaps.

The figures come from the tables `e00056_score.py` wrote; the gaps are the ones the write-up quotes.

    uv run python ml-experiments/00056-*/e00056_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00056_score import WORK_DIR

CHART_DIR: Final[Path] = Path(
    "docs/ml-experiments/00056-mva-only-primaries-skill-against-mw-primaries"
)
WIDTH_PX: Final[int] = 640
GROUP_COLOURS: Final[dict[str, str]] = {"MW": ocf.DATA_BLUE, "MVA": ocf.BRAND_ORANGE}
GROUP_LABELS: Final[dict[str, str]] = {"MW": "Metered in MW", "MVA": "Metered in MVA"}
CUT_LABELS: Final[dict[str, str]] = {
    "all": "All hours",
    "sunny_midday": "Sunny midday",
    "other_hours": "Other hours",
}


def leave_one_out() -> pl.DataFrame:
    """Return the MW-minus-MVA skill gap with each series left out in turn (all hours and sunny)."""
    per_series = pl.read_parquet(WORK_DIR / "per_series.parquet")
    rows = []
    for cut in ("all", "sunny_midday"):
        cut_table = per_series.filter(pl.col("cut") == cut)
        for left_out in [None, *cut_table["time_series_id"].to_list()]:
            kept = cut_table.filter(pl.col("time_series_id") != (left_out or -1))
            skill = {}
            for group in GROUP_COLOURS:
                mine = kept.filter(pl.col("units") == group)
                best = float(mine["sum_best"].sum())
                benchmark = float(mine["sum_benchmark"].sum())
                skill[group] = 100 * (1 - best / benchmark)
            rows.append(
                {"cut": cut, "left_out": left_out, "gap_MW_minus_MVA": skill["MW"] - skill["MVA"]}
            )
    return pl.DataFrame(rows, schema_overrides={"left_out": pl.Int64})


def figure_skill_by_group(*, results: pl.DataFrame) -> alt.Chart:
    """Return the skill of each unit group, with month-resampled intervals, by hours cut."""
    frame = results.filter(
        (pl.col("resampled") == "months")
        & pl.col("cut").is_in(list(CUT_LABELS))
        & pl.col("statistic").is_in(["skill_MW", "skill_MVA"])
    ).with_columns(
        group=pl.col("statistic").str.replace("skill_", "").replace(GROUP_LABELS),
        hours=pl.col("cut").replace(CUT_LABELS),
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("hours:N", sort=list(CUT_LABELS.values()), title=None),
        yOffset="group:N",
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_LABELS.values()), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    rule = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (
        rule
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lower_95:Q",
                title="Skill against manual_heuristic (% lower quantile score; larger is better)",
            ),
            x2="upper_95:Q",
        )
        + base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH_PX - 120,
        height=150,
        title=alt.TitleParams(
            [
                "The best guess beats manual_heuristic by less at MVA Primaries,",
                "and the gap is mostly outside sunny middays",
            ],
            subtitle=[
                (
                    "Dot: pooled skill of 8 Primaries per group. Line: 95% interval from "
                    "resampling whole months."
                ),
                "Dashed rule: no better than manual_heuristic. July 2025 to June 2026.",
            ],
        ),
    )


def figure_per_series(*, per_series: pl.DataFrame) -> alt.Chart:
    """Return each Primary's skill at sunny midday, one dot per series, by unit group."""
    frame = per_series.filter(pl.col("cut") == "sunny_midday").with_columns(
        group=pl.col("units").replace(GROUP_LABELS)
    )
    rule = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    dots = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=0.9)
        .encode(
            x=alt.X(
                "skill:Q",
                axis=alt.Axis(values=list(range(-10, 35, 5))),
                title="Skill against manual_heuristic at sunny midday "
                "(% lower quantile score; larger is better)",
            ),
            y=alt.Y("group:N", sort=list(GROUP_LABELS.values()), title=None),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=list(GROUP_LABELS.values()), range=list(GROUP_COLOURS.values())
                ),
                legend=None,
            ),
        )
    )
    return (rule + dots).properties(
        width=WIDTH_PX - 120,
        height=110,
        title=alt.TitleParams(
            ["The groups overlap: the best MVA Primary", "beats most MW Primaries"],
            subtitle=[
                (
                    "Dot: one Primary, April to September, valid times 10:00 to 13:30 UTC, "
                    "July 2025 to June 2026."
                ),
                "Dashed rule: no better than manual_heuristic.",
            ],
        ),
    )


def figure_low_magnitude(*, counts: pl.DataFrame) -> alt.Chart:
    """Return each Primary's share of sunny-midday half-hours with magnitude under 10% of P99."""
    frame = counts.with_columns(
        group=pl.col("units").replace(GROUP_LABELS),
        series=pl.col("time_series_id").cast(pl.String),
    )
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "series:N",
                sort=alt.EncodingSortField(field="share_low_year", order="descending"),
                title="Primary (series id)",
            ),
            y=alt.Y(
                "share_low_year:Q",
                title="% of sunny-midday half-hours",
            ),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=list(GROUP_LABELS.values()), range=list(GROUP_COLOURS.values())
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH_PX - 120,
            height=170,
            title=alt.TitleParams(
                ["Midday near-zero flow is rare at MVA Primaries", "and common at 2 MW Primaries"],
                subtitle=[
                    "Bar: share of one Primary's half-hours (April to September, 10:00 to",
                    "13:30 UTC, July 2025 to June 2026) with |power| under 10% of its P99.",
                ],
            ),
        )
    )


def main() -> None:
    """Draw the figures and print the leave-one-out table."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    alt.theme.enable("ocf")
    results = pl.read_parquet(WORK_DIR / "results.parquet")
    per_series = pl.read_parquet(WORK_DIR / "per_series.parquet")
    counts = pl.read_parquet(WORK_DIR / "low_magnitude_counts.parquet")
    figures = {
        "figure_1_skill_by_unit": figure_skill_by_group(results=results),
        "figure_2_skill_per_series": figure_per_series(per_series=per_series),
        "figure_3_near_zero_midday": figure_low_magnitude(counts=counts),
    }
    for name, figure in figures.items():
        figure.save(CHART_DIR / f"{name}.svg")
    with pl.Config(tbl_rows=60):
        print(leave_one_out())


if __name__ == "__main__":
    main()
