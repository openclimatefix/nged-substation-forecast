"""Draw experiment 00078's figures from the saved results, as SVG.

- Figure 1: the planned contrasts and the reads on XGBoost's starting point, with their intervals.
- Figure 2: the capacity of each series in each vintage, beside the held capacity.
- Figure 3: the change in score from the season-year vintage, by lead band and by valid month.

Each SVG is afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00078-*/e00078_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00078_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 80
GROUP_COLOURS: Final[dict[str, str]] = {
    "Capacity vintage (98.75% interval)": ocf.DATA_BLUE,
    "Starting point of XGBoost (98.33% interval)": ocf.DATA_GREEN,
    "Exploratory (95% interval)": ocf.BRAND_ORANGE,
}


def labelled(
    *, table: pl.DataFrame, group: str, labels: dict[tuple[str, str, str], str]
) -> pl.DataFrame:
    """Return the rows whose (treatment, reference, scope) are in `labels`, with label and group."""
    keyed = table.with_columns(
        key=pl.struct("treatment", "reference", "scope").map_elements(
            lambda r: labels.get((r["treatment"], r["reference"], r["scope"]), ""),
            return_dtype=pl.String,
        )
    ).filter(pl.col("key") != "")
    return keyed.with_columns(label=pl.col("key"), group=pl.lit(group)).drop("key")


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    path = CHART_DIR / f"{name}.svg"
    chart.save(path)
    print(f"written {path}")


def title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a title with a subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP_CHARACTERS),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_1() -> None:
    """Draw the contrasts and their intervals."""
    decision = pl.read_parquet(common.RESULTS_DIR / "decision.parquet")
    start = pl.read_parquet(common.RESULTS_DIR / "start.parquet")
    explore = pl.read_parquet(common.RESULTS_DIR / "exploratory.parquet")
    vintage = labelled(
        table=decision,
        group="Capacity vintage (98.75% interval)",
        labels={
            ("am_sy", "am", "all leads"): "C1  season-year vintage minus held, all leads",
            ("am_sy", "am", "under 48 h"): "C2  season-year vintage minus held, under 48 h",
            ("am_sy", "am0", "all leads"): "C3  vintage minus no solar term (am0)",
            ("am_sy", "am_sy_twin", "all leads"): "C4  vintage minus wrong-date twin",
        },
    )
    starting = labelled(
        table=start,
        group="Starting point of XGBoost (98.33% interval)",
        labels={
            ("d1l_r_raw", "am0_raw", "10 d+, no correction"): "D1  d1l refit minus am0, 10 d+",
            (
                "d0_r_raw",
                "d0_m0_raw",
                "10 d+, no correction",
            ): "D2  d0 refit minus d0 from zero, 10 d+",
            ("d1l_r", "am0", "all leads"): "D3  d1l refit minus am0, all leads",
        },
    )
    other = labelled(
        table=explore,
        group="Exploratory (95% interval)",
        labels={
            ("am_wk", "am", "all leads"): "Weekly vintage minus held, all leads",
            ("am_wk", "am_sy", "all leads"): "Weekly minus season-year vintage",
            ("am_sy", "d0_m0", "all leads (post hoc)"): "Vintage minus d0 from zero (post hoc)",
        },
    )
    table = pl.concat([vintage, starting, other], how="diagonal_relaxed")
    order = table["label"].to_list()
    base = alt.Chart(table.to_pandas()).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=400),
        ),
    )
    rules = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Difference in quantile score (points of capacity; negative = first arm better)",
            scale=alt.Scale(domain=[-0.55, 0.2]),
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=90, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="difference:Q"
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (rules + dots + zero).properties(
        width=PLOT_WIDTH,
        height=300,
        title=title(
            text="A current capacity helps a little; starting XGBoost from zero helps more",
            subtitle=(
                "Five Primary substations, daylight half-hours of April to September, July 2025 "
                "to June 2026. Dot: difference in mean quantile score. Line: interval from "
                "resampling whole weeks of forecast start. Dashed rule: no difference."
            ),
        ),
    )
    save(chart=chart, name="figure_1_contrasts")


def figure_2() -> None:
    """Draw the capacity of each series in each vintage."""
    vintages = pl.read_parquet(common.VINTAGES_FILE).with_columns(
        month=pl.col("run_date").dt.truncate("1mo")
    )
    long = (
        vintages.group_by("site", "month")
        .agg(
            pl.col("held_mw").first().alias("Held (experiment 64)"),
            pl.col("season_year_mw").mean().alias("Season-year vintage"),
            pl.col("weekly_mw").mean().alias("Weekly vintage"),
        )
        .unpivot(index=["site", "month"], variable_name="capacity", value_name="mw")
    )
    colours = {
        "Held (experiment 64)": "#8A8A8A",
        "Season-year vintage": ocf.DATA_BLUE,
        "Weekly vintage": ocf.BRAND_ORANGE,
    }
    chart = (
        alt.Chart(long.to_pandas())
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:T", title="Month of forecast run", axis=alt.Axis(format="%b %y")),
            y=alt.Y("mw:Q", title="Embedded-solar capacity (MW)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "capacity:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=PLOT_WIDTH // 2 - 30, height=150)
        .facet(
            facet=alt.Facet("site:N", sort=["D9", "D10", "D11", "D14", "D16"], title=None),
            columns=2,
        )
        .properties(
            title=title(
                text="The vintages sit 13% to 31% above the held capacity by May 2026",
                subtitle=(
                    "Mean capacity of the forecast runs in each month, at each of the five Primary "
                    "substations. D14 uses its held value until its 2025/26 season-year has 8 "
                    "observed weeks."
                ),
            )
        )
    )
    save(chart=chart, name="figure_2_vintages")


def figure_3() -> None:
    """Draw the change in score by lead band and by valid month."""
    explore = pl.read_parquet(common.RESULTS_DIR / "exploratory.parquet").filter(
        (pl.col("treatment") == "am_sy") & (pl.col("reference") == "am")
    )
    bands = explore.filter(pl.col("scope").str.starts_with("band ")).with_columns(
        label=pl.col("scope").str.replace("band ", ""), panel=pl.lit("By lead band")
    )
    months = explore.filter(pl.col("scope").str.starts_with("valid month ")).with_columns(
        label=pl.col("scope").str.replace("valid month ", ""), panel=pl.lit("By valid month")
    )
    panels = []
    for table, name in ((bands, "By lead band"), (months, "By valid month")):
        order = table["label"].to_list()
        base = alt.Chart(table.to_pandas()).encode(
            y=alt.Y("label:N", sort=order, title=None),
        )
        rules = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lower:Q",
                title="Season-year vintage minus held capacity (points of capacity)",
                scale=alt.Scale(domain=[-0.12, 0.05]),
                axis=alt.Axis(values=[-0.1, -0.05, 0.0, 0.05]),
            ),
            x2="upper:Q",
        )
        dots = base.mark_point(filled=True, size=90, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x="difference:Q"
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((rules + dots + zero).properties(width=PLOT_WIDTH, height=120, title=name))
    chart = alt.vconcat(*panels).properties(
        title=title(
            text="The gain is concentrated in April and May 2026, at leads under 6 days",
            subtitle=(
                "Mean quantile score of the season-year vintage minus the held capacity (negative "
                "favours the vintage), with a 95% interval from resampling whole weeks of forecast "
                "start. Five Primary substations, daylight half-hours of April to September."
            ),
        )
    )
    save(chart=chart, name="figure_3_by_lead_and_month")


def main() -> None:
    """Draw the three figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()


if __name__ == "__main__":
    main()
