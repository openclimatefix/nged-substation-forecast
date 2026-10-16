"""Draw experiment 00034's figures from the saved results, as SVG.

1. The difference from the best-guess stack for each retraining cadence, without and with the
   14-day correction.
2. The quantile score by lead band for the stack, the frozen model, monthly, and weekly retraining.
3. The rank-agreement test: each proxy's difference on the correction contrast and the two
   feature-set contrasts, against full monthly retraining.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00034-*/e00034_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00034_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
CADENCES: Final[dict[str, str]] = {
    "yearly": "Frozen (1 refit)",
    "half_yearly": "Half-yearly (2)",
    "quarterly": "Quarterly (4)",
    "monthly": "Monthly (12)",
    "weekly": "Weekly (52)",
}
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def figure_1() -> alt.TopLevelMixin:
    """Each cadence's difference from the stack, with the 95% interval."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts_exploratory.parquet").filter(
        pl.col("id").str.starts_with("X_stack_")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "95")
    )
    rows = []
    for arm, label in CADENCES.items():
        for form, corrected in (("s", "no correction"), ("cs", "14-day correction")):
            row = contrasts.filter(pl.col("id") == f"X_stack_{arm}_{form}")
            if not row.is_empty():
                rows.append({"label": label, "correction": corrected, **row.row(0, named=True)})
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(CADENCES.values()), title=None),
        yOffset="correction:N",
        color=alt.Color(
            "correction:N",
            scale=alt.Scale(
                domain=["no correction", "14-day correction"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Quantile score minus the stack's (points of capacity; below zero is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(_zero_rule(), bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Monthly and weekly retraining beat the stack; the correction adds nothing then",
            lines=[
                "Stack: frozen model with the 14-day correction and spread factor 1.15 (score "
                "5.32). Rows give the number of refits in the year. " + SCOPE,
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months. Zero: same "
                    "score as the stack."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Quantile score by lead band for four arms."""
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
        (pl.col("group") == "Primary") & (pl.col("scope") != "all leads")
    )
    arms = {
        "n1_cs": "Stack (frozen, correction)",
        "yearly_s": "Frozen, no correction",
        "monthly_s": "Monthly retraining",
        "weekly_s": "Weekly retraining",
    }
    data = absolute.filter(pl.col("arm").is_in(list(arms))).with_columns(
        label=pl.col("arm").replace_strict(arms)
    )
    bands = [name for name, _, _ in common.LEAD_BANDS]
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("scope:N", sort=bands, title="Lead time", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                sort=list(arms.values()),
                scale=alt.Scale(
                    domain=list(arms.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(orient="top", title=None, columns=2),
            ),
        )
    )
    return chart.properties(
        width=PLOT_WIDTH,
        height=240,
        title=_title(
            text="Retraining helps at every lead time, most beyond two days",
            lines=["Mean quantile score by lead band. " + SCOPE],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Each proxy's difference on the three rank-test contrasts, against full monthly."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts_rank.parquet").filter(
        (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "95")
    )
    names = {
        "monthly": "Full monthly retraining",
        "roll12": "P-roll: 12-month window",
        "three_a_year": "P-anchor: 3 refits a year",
        "interp": "P-interp (uses later data)",
        "warm20_monthly": "P-warm, 20 rounds",
        "warm50_monthly": "P-warm, 50 rounds",
    }
    titles = {
        "R1": "Correction on minus off",
        "R2": "Weekly analogues minus N1",
        "R3": "One model per type minus per series",
    }
    rows = []
    for contrast_id, regime in ((c, r) for c in titles for r in names):
        row = contrasts.filter(pl.col("id") == f"{contrast_id}_{regime}")
        if not row.is_empty():
            rows.append(
                {"panel": titles[contrast_id], "proxy": names[regime], **row.row(0, named=True)}
            )
    panels = []
    for panel in titles.values():
        data = pl.DataFrame([r for r in rows if r["panel"] == panel])
        base = alt.Chart(data).encode(y=alt.Y("proxy:N", sort=list(names.values()), title=None))
        points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
        )
        bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(_zero_rule(), bars, points).properties(
                width=PLOT_WIDTH, height=130, title=alt.TitleParams(panel, anchor="start")
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Only the rolling window agrees with monthly retraining on every contrast",
            lines=[
                "A proxy agrees when its dot is near the dot of full monthly "
                "retraining (top row of each panel). " + SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months. Seed 0.",
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
    ):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
