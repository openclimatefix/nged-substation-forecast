"""Draw experiment 00080's figures from the saved results, as SVG.

- Figure 1: the planned contrasts C1 to C4 and the wind reads, with their intervals.
- Figure 2: the zero-start gap with 500, 1000, and 2000 rounds and with early stopping, by class.
- Figure 3: the validation loss of the two starts round by round, by class.

Each SVG is afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00080-*/e00080_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00080_common as common
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 440
WRAP_CHARACTERS: Final[int] = 80
CLASS_LABELS: Final[dict[str, str]] = {
    "primary": "Primary substations (8)",
    "bspgsp": "Bulk and grid supply points (4)",
    "solar": "Solar farms (6)",
    "wind": "Wind farms, no margin (3)",
}


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
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


def pooled(*, ids: dict[str, str], level_label: str) -> pl.DataFrame:
    """Return the pooled all-leads quantile-score rows of some contrasts, with a display label."""
    table = pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet").filter(
        (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == level_label)
        & pl.col("id").is_in(list(ids))
    )
    return table.with_columns(label=pl.col("id").replace(ids))


def figure_1() -> None:
    """Draw the planned contrasts and the wind reads."""
    planned = pooled(
        ids={
            "C1": "C1  Primary substations, weekly refits",
            "C2": "C2  Bulk and grid supply points, weekly refits",
            "C3": "C3  Primary substations, frozen, 14-day correction",
            "C4": "C4  Solar farms, monthly refits",
        },
        level_label="planned",
    ).with_columns(group=pl.lit("Planned (98.75% interval)"))
    wind = pooled(
        ids={
            "X_wind_no_margin_zero_minus_estimated": "Wind farms without the power-curve margin",
        },
        level_label="95",
    ).with_columns(group=pl.lit("Exploratory (95% interval)"))
    table = pl.concat([planned, wind], how="diagonal_relaxed")
    order = table["label"].to_list()
    colours = {
        "Planned (98.75% interval)": ocf.DATA_BLUE,
        "Exploratory (95% interval)": ocf.BRAND_ORANGE,
    }
    base = alt.Chart(table.to_pandas()).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", direction="vertical"),
        ),
    )
    rules = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Zero start minus estimated start, quantile score (points of capacity)",
            scale=alt.Scale(domain=[-0.15, 0.15]),
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
        height=200,
        title=title(
            text="A zero starting value does not lower the error of the production models",
            subtitle=(
                "July 2025 to June 2026, daily runs of 15 days. Dot: difference in mean quantile "
                "score (negative: zero start better). Line: interval from resampling whole "
                "months of forecast start. Dashed rule: no difference."
            ),
        ),
    )
    save(chart=chart, name="figure_1_contrasts")


def figure_2() -> None:
    """Draw the zero-start gap against the number of rounds, by class."""
    table = pl.read_parquet(common.RESULTS_DIR / "contrasts.parquet").filter(
        (pl.col("site") == "pooled")
        & (pl.col("scope") == "all leads")
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "95")
    )
    rows = []
    for cls, label in CLASS_LABELS.items():
        for rounds, name in (
            ("500", f"R_{cls}_zero_minus_estimated_r500"),
            ("1000", f"R_{cls}_zero_minus_estimated_r1000"),
            ("2000", f"R_{cls}_zero_minus_estimated_r2000"),
            ("early stop", f"E_{cls}_zero_minus_estimated_early_stopping_best"),
        ):
            part = table.filter(pl.col("id") == name)
            if part.is_empty():
                continue
            rows.append(part.with_columns(klass=pl.lit(label), rounds=pl.lit(rounds)))
    order = ["500", "1000", "2000", "early stop"]
    panels = []
    for position, label in enumerate(CLASS_LABELS.values()):
        part = pl.concat(rows, how="diagonal_relaxed").filter(pl.col("klass") == label)
        base = alt.Chart(part.to_pandas()).encode(
            x=alt.X("rounds:N", sort=order, title="Boosting rounds", axis=alt.Axis(labelAngle=-45)),
        )
        rules = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "lower:Q",
                title="Zero start minus estimated start (points)" if position == 0 else None,
                scale=alt.Scale(domain=[-0.15, 0.7]),
                axis=alt.Axis() if position == 0 else alt.Axis(labels=False, ticks=False),
            ),
            y2="upper:Q",
        )
        dots = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y="difference:Q"
        )
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((rules + dots + zero).properties(width=100, height=220, title=label))
    chart = alt.hconcat(*panels).properties(
        title=title(
            text="More rounds do not close the gap between the two starts",
            subtitle=(
                "Frozen models, uncorrected, 292 daily runs. Dot: difference in mean quantile "
                "score. Line: 95% interval from resampling whole months of forecast start."
            ),
        )
    )
    save(chart=chart, name="figure_2_rounds")


def figure_3() -> None:
    """Draw the validation loss round by round for both starts, by class."""
    curves = pl.concat(
        [pl.read_parquet(path) for path in sorted((common.WORK_DIR / "curves").glob("*.parquet"))],
        how="diagonal_relaxed",
    ).with_columns(
        klass=pl.col("group").replace(
            {
                "Primary": "Primary",
                "BSP": "bspgsp",
                "GSP": "bspgsp",
                "solar": "solar",
                "wind": "wind",
            }
        )
    )
    shortest = curves.group_by("klass", "start", "site").agg(last=pl.col("round").max())
    shortest = shortest.group_by("klass", "start").agg(limit=pl.col("last").min())
    curves = curves.join(shortest, on=["klass", "start"]).filter(pl.col("round") <= pl.col("limit"))
    mean = (
        curves.group_by("klass", "start", "round")
        .agg(valid=pl.col("valid").mean(), n=pl.col("site").n_unique())
        .sort("klass", "start", "round")
    )
    mean = mean.with_columns(
        klass=pl.col("klass")
        .replace({"Primary": CLASS_LABELS["primary"]})
        .replace({k: v for k, v in CLASS_LABELS.items() if k != "primary"}),
        start=pl.col("start").replace({"def": "Estimated start", "m0": "Zero start"}),
    )
    chart = (
        alt.Chart(mean.to_pandas())
        .mark_line(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("round:Q", title="Boosting round"),
            y=alt.Y("valid:Q", title="Held-out pinball loss", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "start:N",
                scale=alt.Scale(
                    domain=["Estimated start", "Zero start"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=170, height=170)
        .facet(column=alt.Column("klass:N", title=None, sort=list(CLASS_LABELS.values())))
        .resolve_scale(y="independent")
        .properties(
            title=title(
                text="The zero start begins worse and has nearly caught up by 250 rounds",
                subtitle=(
                    "Mean over series of the pinball loss on the last 90 days before the training "
                    "end, for the model trained without those days, up to the round at which "
                    "the first series stopped early."
                ),
            )
        )
    )
    save(chart=chart, name="figure_3_learning_curves")


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
