"""Draw the anonymised figures for experiment 246 from the tables `e00246_score.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import e00246_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 270
ARM_LABELS: Final[dict[str, str]] = {
    "b4_tree_all_three": "B4 tree, ensemble + UKV sky + ICON-D2",
    "b4x_tree_ens_icon": "B4x tree, ensemble + ICON-D2",
    "r5_tree_ens_ukv_sky": "R5 tree, ensemble + UKV sky",
    "b3_blend_all": "B3 blend, ensemble + UKV + ICON-D2",
    "b2_blend_ens_icon": "B2 blend, ensemble + ICON-D2",
    "r3_blend_ens_ukv": "R3 blend, ensemble + UKV",
    "a1_ens_two_number": "A1 plant, ensemble two numbers",
    "p4_b4_shuffle": "Placebo: B4 with shuffled ICON-D2",
    "p3_b3_shuffle": "Placebo: B3 with shuffled ICON-D2",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}


def title(lines: list[str] | str) -> alt.TitleParams:
    """Return a left-aligned, regular-weight title."""
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


SCOPE = "Metered solar farms A to F, 12 to 29 hour leads, scored months of both folds."


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
        .filter(pl.col("band") == 0)
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
        height=300,
        title=title([SCOPE, "Dot: mean over farms. Line: 95% interval from resampling months."]),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def per_farm() -> None:
    """Figure 2: the five planned contrasts, per farm and as the median."""
    contrasts = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter((pl.col("band") == 0) & pl.col("contrast").str.contains(r"^(D1|D2|D3|D4|D5) "))
        .with_columns(
            label=pl.col("contrast")
            .str.slice(0, 2)
            .replace_strict(
                {
                    "D1": "D1 B3 blend vs R3 blend",
                    "D2": "D2 B4 tree vs R5 tree",
                    "D3": "D3 B2 blend vs A1 plant",
                    "D4": "D4 B3 vs its placebo",
                    "D5": "D5 B4 vs its placebo",
                }
            ),
            farm=pl.col("farm").replace({"median": "Median of farms"}),
        )
    )
    order = ["Median of farms", *common.FARMS]
    rows = [
        "D1 B3 blend vs R3 blend",
        "D2 B4 tree vs R5 tree",
        "D3 B2 blend vs A1 plant",
        "D4 B3 vs its placebo",
        "D5 B4 vs its placebo",
    ]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-1.5]}))
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
                    domain=[-8, 6],
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
                "Dot: change. Line: 99% interval (five planned contrasts). Grey: no change.",
                "Orange dashed: the smallest gain worth porting (1.5%).",
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
        height=300,
        title=title(
            [
                "Fraction of output fitted on sunny half-hours (clear-sky irradiance > 650 W/m²).",
                "Farms A to F, 12 to 29 hour leads, scored months of both folds; dot: farm median.",
                "Orange dashed: experiment 218's fit with observed weather (0.973, other rows).",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_3_sunny_fit.svg")


if __name__ == "__main__":
    headline()
    per_farm()
    sunny()
