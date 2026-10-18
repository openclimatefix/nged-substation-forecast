"""Draw experiment 00085's figures from the saved results, as SVG.

1. The cadence and starting-value contrasts under every schedule, with 95% intervals (headline).
2. Each arm's absolute quantile score with its 95% interval.
3. The compute minutes of a year of refits under each schedule.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00085-*/e00085_charts.py
"""

from typing import Final

import altair as alt
import e00085_common as common
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 540
SCOPE: Final[str] = (
    "16 Primary substations, 158 daily forecast runs (every second week of July 2025 to June 2026)."
)
CONTRAST_ORDER: Final[list[str]] = [
    "R1",
    "K1 P-roll",
    "K1 whole-history 150 rounds",
    "K1 P-roll quarterly (descriptive)",
    "R2",
    "K2 P-roll monthly",
    "K2 P-roll quarterly",
    "K2 P-roll weekly",
    "K2 full monthly",
    "K2 weekly 150 rounds",
]
LABELS: Final[dict[str, str]] = {
    "R1": "Monthly minus weekly, full setting (reference)",
    "K1 P-roll": "Monthly minus weekly, 12-month window",
    "K1 whole-history 150 rounds": "Monthly minus weekly, 150 rounds",
    "K1 P-roll quarterly (descriptive)": "Quarterly minus weekly, 12-month window",
    "R2": "Zero start minus default start, full weekly (reference)",
    "K2 P-roll monthly": "Zero minus default start, monthly, 12-month window",
    "K2 P-roll quarterly": "Zero minus default start, quarterly, 12-month window",
    "K2 P-roll weekly": "Zero minus default start, weekly, 12-month window",
    "K2 full monthly": "Zero minus default start, monthly, full setting",
    "K2 weekly 150 rounds": "Zero minus default start, weekly, 150 rounds",
}
ARM_LABELS: Final[dict[str, str]] = {
    "w": "Weekly, full setting",
    "fw": "Weekly, 150 rounds",
    "pw": "Weekly, 12-month window",
    "m": "Monthly, full setting",
    "fm": "Monthly, 150 rounds",
    "pm": "Monthly, 12-month window",
    "pq": "Quarterly, 12-month window",
}


def figure_1() -> alt.TopLevelMixin:
    """Contrasts with intervals: the reference first in each block."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet")
        .filter(pl.col("contrast").is_in(CONTRAST_ORDER))
        .with_columns(
            label=pl.col("contrast").replace_strict(LABELS),
            kind=pl.when(pl.col("schedule") == "reference")
            .then(pl.lit("Full weekly reference"))
            .otherwise(pl.lit("Cheaper schedule")),
        )
    )
    order = [LABELS[c] for c in CONTRAST_ORDER]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=380)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["Full weekly reference", "Cheaper schedule"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title="Quantile score, first arm minus second (points of capacity)")
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "Cheaper schedules reproduce the cadence gap; the start gap is too small to read",
            subtitle=[
                SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months.",
                "Rule at zero: same score. Below zero: the first arm is better.",
            ],
            anchor="start",
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Absolute quantile score of the estimated-start arms."""
    data = (
        pl.read_parquet(common.RESULTS_DIR / "absolute.parquet")
        .filter(pl.col("arm").is_in(list(ARM_LABELS)))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
        )
    )
    points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "qs:Q",
            title="Quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="qs_low:Q", x2="qs_high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            "Absolute score of each retraining schedule",
            subtitle=[SCOPE, "Dot: mean. Line: 95% interval from re-drawing whole months."],
            anchor="start",
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Compute minutes of a year of refits for 16 series."""
    data = pl.read_parquet(common.RESULTS_DIR / "cost_schedules.parquet")
    base = alt.Chart(data).encode(
        y=alt.Y(
            "schedule:N",
            sort=data["schedule"].to_list(),
            title=None,
            axis=alt.Axis(labelLimit=330),
        )
    )
    bars = base.mark_bar(color=ocf.DATA_SKY).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("minutes_high:Q", title="Compute minutes for 16 series and one year of refits")
    )
    ticks = base.mark_tick(color=ocf.DATA_BLUE, thickness=3, size=16).encode(  # ty: ignore[unresolved-attribute]
        x="minutes_low:Q"
    )
    return alt.layer(bars, ticks).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            "Cost of each schedule",
            subtitle=[
                "Bar: busiest measured fit time per refit. Dark tick: quietest.",
                "Fits shared the machine with other jobs, so fit times varied by up to 4 times.",
            ],
            anchor="start",
        ),
    )


def main() -> None:
    """Write the figures."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
    ):
        chart.save(common.CHART_DIR / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
