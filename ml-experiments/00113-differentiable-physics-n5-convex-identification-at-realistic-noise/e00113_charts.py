"""Draw the figures of experiment 113 as SVG into the docs folder, from the stored summaries.

Only Primary substations appear, under their D-labels. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00113-*/e00113_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00113_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = Path("/tmp/e00113_png")
WIDTH: Final[int] = 480
WRAP: Final[int] = 95
FAMILY_LABEL: Final[dict[str, str]] = {
    "band": "Heating slope (per hour band)",
    "capacity": "Latest PV capacity",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 75),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_coverage(*, summary: pl.DataFrame) -> None:
    """Figure 1: I1 and I2 coverage of the truth in the decision cell and the iid5 cell."""
    rows = []
    series = (
        ("i1", "g_full", "I1 month-block bootstrap (decision)"),
        ("i1", "g_full_30", "I1 with drift step 30 (post hoc)"),
        ("i2", "g_full", "I2 posterior"),
    )
    for method, arm, method_label in series:
        for noise, noise_label in (
            ("real", "real residuals (decision noise)"),
            ("iid5", "independent noise, sd 5"),
        ):
            for family in ("band", "capacity"):
                pick = summary.filter(
                    (pl.col("window") == "all")
                    & (pl.col("truth") == "hd")
                    & (pl.col("arm") == arm)
                    & (pl.col("method") == method)
                    & (pl.col("noise") == noise)
                    & (pl.col("family") == family)
                ).row(0, named=True)
                rows.append(
                    {
                        "method": method_label,
                        "noise": noise_label,
                        "family": FAMILY_LABEL[family],
                        "coverage": 100 * pick["coverage_truth"],
                        "low": 100 * pick["coverage_truth_low"],
                        "high": 100 * pick["coverage_truth_high"],
                    }
                )
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "method:N",
        sort=None,
        scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top", labelLimit=400),
    )
    panels = []
    for family in FAMILY_LABEL.values():
        part = frame.filter(pl.col("family") == family)
        band = (
            alt.Chart(pl.DataFrame({"a": [70.0], "b": [90.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.15, color=ocf.DATA_GREEN)
            .encode(x=alt.X("a:Q", scale=alt.Scale(domain=[0, 100])), x2="b:Q")
        )
        base = alt.Chart(part).encode(
            y=alt.Y("noise:N", title=None, sort=None, axis=alt.Axis(labelLimit=260))
        )
        rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="Share of intervals that contain the true value (%); target 80",
            ),
            x2="high:Q",
            color=colour,
            yOffset="method:N",
        )
        dots = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x="coverage:Q", color=colour, yOffset="method:N"
        )
        panels.append(alt.layer(band, rule, dots).properties(width=WIDTH, height=90, title=family))
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text=(
                "The month-block bootstrap covers the true value 84% of the time at real noise; "
                "the posterior interval 20%"
            ),
            subtitle=(
                "Convex model with the same form as the synthetic truth, 100 synthetic substations "
                "(20 on each of 5 real substations). Dot: share of 80% intervals containing the "
                "truth. Line: 95% interval from resampling substations, then draws. Green band: 70 "
                "to 90%, the pre-registered pass range."
            ),
        )
    )
    _save(chart, name="figure_1_coverage")


def figure_recovery(*, frame: pl.DataFrame) -> None:
    """Figure 2: the point estimate against the true value in the decision cell."""
    pick = frame.filter(
        (pl.col("window") == "all")
        & (pl.col("truth") == "hd")
        & (pl.col("noise") == "real")
        & (pl.col("arm") == "g_full")
    ).with_columns(
        family=pl.when(pl.col("functional") == "capacity")
        .then(pl.lit(FAMILY_LABEL["capacity"]))
        .otherwise(pl.lit(FAMILY_LABEL["band"])),
        lo=pl.col("i1_lo"),
        hi=pl.col("i1_hi"),
    )
    top = float(max(pick["truth_value"].max(), pick["point"].max()))  # ty: ignore[invalid-argument-type]
    diag = (
        alt.Chart(pl.DataFrame({"x": [0.0, top], "y": [0.0, top]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(pick)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=22, opacity=0.7)
        .encode(
            x=alt.X("truth_value:Q", title="True value (points of effective capacity)"),
            y=alt.Y("point:Q", title="Estimate (points)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
        )
    )
    chart = alt.layer(diag, points).properties(
        width=WIDTH,
        height=300,
        title=_title(
            text="Estimates scatter around the truth with no systematic bias at real noise",
            subtitle=(
                "Convex model on synthetic substations with real noise, one dot per substation, "
                "draw, and functional "
                "(400 band slopes, 100 capacities). Dashed line: estimate equals truth. Median "
                "signed relative error: "
                "-1.4% for band slopes, -1.3% for capacity."
            ),
        ),
    )
    _save(chart, name="figure_2_recovery")


def figure_widths(*, summary: pl.DataFrame, analytic: pl.DataFrame) -> None:
    """Figure 3: the 80% interval's relative width by arm, beside the exact variance ratio."""
    wide = summary.filter(
        (pl.col("window") == "all")
        & (pl.col("truth") == "t0")
        & (pl.col("method") == "i1")
        & (pl.col("noise") == "real")
        & pl.col("arm").is_in(["g_full", "g_0", "g_h", "g_d"])
    ).select("arm", "family", "median_relative_width")
    labels = {
        "g_full": "Harmonics and drift (G-full)",
        "g_h": "Harmonics only (G-H)",
        "g_d": "Drift only (G-D)",
        "g_0": "Neither (G-0)",
    }
    frame = wide.with_columns(
        arm_label=pl.col("arm").replace(labels), family_label=pl.col("family").replace(FAMILY_LABEL)
    )
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y(
                "arm_label:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=240)
            ),
            x=alt.X(
                "median_relative_width:Q",
                title=(
                    "Median width of the 80% interval, as a share of the true value "
                    "(smaller is narrower)"
                ),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "family_label:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
            yOffset="family_label:N",
        )
        .properties(
            width=WIDTH,
            height=150,
            title=_title(
                text=(
                    "The day-of-year harmonics make the heating-slope intervals about 2 times wider"
                ),
                subtitle=(
                    "I1 intervals on synthetic substations whose truth has no harmonics or "
                    "drift, real noise, 100 substations. "
                    "Each fit is given the columns named on the axis; all four contain the truth."
                ),
            ),
        )
    )
    _save(chart, name="figure_3_widths")


def figure_absorption(*, analytic: pl.DataFrame) -> None:
    """Figure 4: the exact variance ratio r for each removed term, at both drift penalties."""
    pick = (
        analytic.filter(pl.col("r").is_not_nan() & (pl.col("window") == "all"))
        .group_by("drift_step", "removed", "functional")
        .agg(pl.col("r").median())
    )
    removed = {"g_0": "Both removed", "g_d": "Harmonics removed", "g_h": "Drift removed"}
    frame = pick.with_columns(
        removed_label=pl.col("removed").replace(removed),
        penalty=pl.when(pl.col("drift_step") > 100)
        .then(pl.lit("Drift step 3,000"))
        .otherwise(pl.lit("Drift step 30")),
        quantity=pl.when(pl.col("functional") == "capacity")
        .then(pl.lit("Latest capacity"))
        .otherwise(pl.lit("Band heating slope")),
    )
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "r:Q",
                scale=alt.Scale(domain=[0, 1.05]),
                title=(
                    "Variance of the functional with the term removed, as a share of the "
                    "full fit's variance"
                ),
            ),
            y=alt.Y("removed_label:N", sort=list(removed.values()), title=None),
            color=alt.Color(
                "quantity:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
            shape=alt.Shape(
                "penalty:N", legend=alt.Legend(title=None, orient="top", labelLimit=400)
            ),
        )
        .properties(
            width=WIDTH,
            height=150,
            title=_title(
                text=(
                    "Harmonics absorb most of the heating slopes' information; drift absorbs "
                    "part of the capacity's"
                ),
                subtitle=(
                    "Exact variance ratio r, median over 5 substations, all years; one dot per "
                    "band (4) and for capacity. "
                    "A ratio of 0.2 means the fit without the term would have a standard "
                    "deviation 0.45 times as large."
                ),
            ),
        )
    )
    _save(chart, name="figure_4_absorption")


def figure_real(*, real: pl.DataFrame) -> None:
    """Figure 5: the latest season-year's fitted capacity beside the vintage mean, per site."""
    pick = real.filter(
        (pl.col("cut") == "scored")
        & (pl.col("arm") == "g_full")
        & (pl.col("functional") == "capacity_2024")
    ).with_columns(vintage=pl.col("vintage_season_mean"))
    long = pl.concat(
        [
            pick.select(
                "site", "point", "vintage", lo=pl.col("i1_lo"), hi=pl.col("i1_hi")
            ).with_columns(interval=pl.lit("I1 month-block bootstrap")),
            pick.select(
                "site", "point", "vintage", lo=pl.col("i2_lo"), hi=pl.col("i2_hi")
            ).with_columns(interval=pl.lit("I2 posterior")),
        ]
    )
    colours = alt.Scale(
        domain=["I1 month-block bootstrap", "I2 posterior"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
    )
    base = alt.Chart(long).encode(
        y=alt.Y("site:N", title=None, sort=common.DECISION_SITES), yOffset="interval:N"
    )
    i1 = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="Latest season-year's PV capacity (points of effective capacity)"),
        x2="hi:Q",
        color=alt.Color(
            "interval:N", scale=colours, legend=alt.Legend(title=None, orient="top", labelLimit=400)
        ),
    )
    i2 = base.mark_point(filled=True, color="#222222", size=40).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    vintage = (
        alt.Chart(pick)  # ty: ignore[unresolved-attribute]
        .mark_tick(color=ocf.DATA_GREEN, thickness=3, size=40)
        .encode(x="vintage:Q", y=alt.Y("site:N", sort=common.DECISION_SITES))
    )
    chart = alt.layer(i1, i2, vintage).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text=(
                "The fitted latest capacity is 0 to 20% below the weekly vintage at the 4 "
                "substations that have one, and the posterior interval is far narrower than the "
                "bootstrap"
            ),
            subtitle=(
                "Real data, fit up to 2025-07-01, convex model with drift. Orange line: 80% "
                "month-block bootstrap interval (I1). "
                "Blue line: 80% posterior interval (I2), not calibrated at real noise. Black "
                "dot: estimate. Green tick: "
                "weekly vintage capacity averaged over the season-year (not available at D14: "
                "its vintage stops in 2023)."
            ),
        ),
    )
    _save(chart, name="figure_5_real_capacity")


def main() -> None:
    """Draw every figure."""
    work = common.WORK_DIR
    import e00113_summary as summary_module

    summary = pl.read_parquet(work / "summary.parquet")
    analytic = pl.read_parquet(work / "analytic.parquet")
    real = pl.read_parquet(work / "real_intervals.parquet")
    frame = summary_module.with_intervals(frame=summary_module.read_cells(tag="main"))
    figure_coverage(summary=summary)
    figure_recovery(frame=frame)
    figure_widths(summary=summary, analytic=analytic)
    figure_absorption(analytic=analytic)
    figure_real(real=real)
    json.dumps({})


if __name__ == "__main__":
    main()
