"""Draw the anonymised figures for experiment 229 from the tables `e00229_score.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import e00229_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 270
ARM_LABELS: Final[dict[str, str]] = {
    "a5_tree_ukv_sky": "A5 tree, ensemble + UKV sky",
    "a4_tree_all_ukv": "A4 tree, ensemble + 22 UKV variables",
    "a1x_tree_ens": "A1x tree, ensemble only",
    "a3_blend": "A3 plant on blended irradiance",
    "a1_ens_two_number": "A1 plant, ensemble two numbers",
    "p4_tree_shuffle": "Placebo: A4 with shuffled UKV",
    "p3_blend_shuffle": "Placebo: A3 with shuffled UKV",
    "a2_ukv_alone": "A2 plant, UKV alone",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}


def title(lines: list[str] | str) -> alt.TitleParams:
    """Return a left-aligned, regular-weight title."""
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


SCOPE = "Metered solar farms A to F, 12 to 36 hour leads, March to December 2025."


def dot_and_line(
    *,
    data: pl.DataFrame,
    x: str,
    low: str,
    high: str,
    y: str,
    order: list[str],
    title: str,
    domain: list[float],
    colour: str,
) -> alt.LayerChart:
    """Return dots with interval lines, one row per `y` label."""
    base = alt.Chart(data)
    yenc = alt.Y(f"{y}:N", sort=order, title=None, axis=alt.Axis(labelLimit=250))
    return (
        base.mark_rule(color=colour, size=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{low}:Q", title=title, scale=alt.Scale(domain=domain)), x2=f"{high}:Q", y=yenc
        )
        + base.mark_point(color=colour, filled=True, size=80).encode(x=f"{x}:Q", y=yenc)  # ty: ignore[unresolved-attribute]
    )


def headline() -> None:
    """Figure 1: quantile score per arm at 12 to 36 hour leads."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter((pl.col("band") == 0) & (pl.col("n_months") == 10))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    chart = dot_and_line(
        data=scores,
        x="score",
        low="lo",
        high="hi",
        y="label",
        order=list(ARM_LABELS.values()),
        title="Quantile score (points of capacity; smaller is better)",
        domain=[4, 12],
        colour=ocf.DATA_BLUE,
    ).properties(
        width=PLOT_WIDTH,
        height=260,
        title=title([SCOPE, "Dot: mean over farms. Line: 95% interval from resampling months."]),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def per_farm() -> None:
    """Figure 2: the three blend contrasts and the placebo contrast, per farm and as the median."""
    contrasts = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter((pl.col("band") == 0) & pl.col("contrast").str.contains(r"^(C1a|C1b|C1c|C4) "))
        .with_columns(
            label=pl.col("contrast")
            .str.slice(0, 3)
            .replace_strict(
                {
                    "C1a": "A3 blend vs A1",
                    "C1b": "A4 tree vs A1x",
                    "C1c": "A5 tree vs A1x",
                    "C4 ": "A4 vs its placebo",
                }
            ),
            farm=pl.col("farm").replace({"median": "Median of farms"}),
        )
    )
    order = ["Median of farms", *common.FARMS]
    rows = ["A3 blend vs A1", "A4 tree vs A1x", "A5 tree vs A1x", "A4 vs its placebo"]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-3.0]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    panels = []
    for label in rows:
        sub = contrasts.filter(pl.col("label") == label)
        panels.append(
            (
                dot_and_line(
                    data=sub,
                    x="percent_change",
                    low="lo",
                    high="hi",
                    y="farm",
                    order=order,
                    title="Change in quantile score (%; more negative is better)",
                    domain=[-12, 6],
                    colour=ocf.DATA_BLUE if "placebo" not in label else ocf.BRAND_ORANGE,
                )
                + zero
                + threshold
            ).properties(width=PLOT_WIDTH, height=120, title=title(label))
        )
    chart = alt.vconcat(*panels).properties(
        title=title(
            [
                SCOPE,
                "Dot: change. Line: 99.17% interval (six planned contrasts). Grey: no change.",
                "Orange dashed: the smallest gain worth porting (3%).",
            ]
        )
    )
    chart.save(DOCS_DIR / "figure_2_per_farm.svg")


def sunny() -> None:
    """Figure 3: fraction of output fitted on sunny half-hours, per arm."""
    data = (
        pl.read_csv(common.DATA_DIR / "sunny.csv")
        .filter(pl.col("band") == 0)
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(data)
    y = alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=250))
    dots = base.mark_point(color=ocf.DATA_BLUE, filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "fraction_fitted_median:Q",
            title="Fraction of output fitted (higher is better)",
            scale=alt.Scale(domain=[0.84, 0.98]),
        ),
        y=y,
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.973]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (dots + rule).properties(
        width=PLOT_WIDTH,
        height=260,
        title=title(
            [
                "Fraction of output fitted on sunny half-hours (clear-sky irradiance > 650 W/m²).",
                "Farms A to F, 12 to 36 hour leads, March to December 2025; dot: farm median.",
                "Orange dashed: experiment 218's fit with observed weather (0.973, other rows).",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_3_sunny_fit.svg")


if __name__ == "__main__":
    headline()
    per_farm()
    sunny()
