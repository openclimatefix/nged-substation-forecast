"""Draw experiment 00079's figures from the saved results, as SVG (raw data first).

No chart pairs a solar-farm label with a name. Substations are named. The SVGs are afterwards
optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=4 uv run --with holidays python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_charts.py
"""

import textwrap
from typing import Final

import altair as alt
import e00079_common as common
import e00079_fit as fit
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 600
WRAP: Final[int] = 92


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def raw_profile() -> alt.Chart:
    """Figure 1: median reading by local hour at the 8 MVA-only Primaries, sunny and cloudy days."""
    prepared = fit.prepare(frame=pl.read_parquet(common.FRAME_PATH))
    rows = prepared.filter(
        pl.col("site").is_in(common.MVA_PRIMARIES)
        & pl.col("month").is_in(common.SUMMER_MONTHS)
        & pl.col("power").is_not_null()
    ).with_columns(
        sky=pl.when(pl.col("day_cloud") < 0.15)
        .then(pl.lit("Sunny days (cloud under 0.15)"))
        .when(pl.col("day_cloud") > 0.5)
        .then(pl.lit("Cloudy days (cloud over 0.5)")),
        hour=pl.col("half_hour") / 2.0,
        name=pl.col("site").replace_strict(common.PRIMARY_NAMES),
    )
    scale = rows.group_by("site").agg(pl.col("power").median().alias("typical"))
    profile = (
        rows.drop_nulls("sky")
        .join(scale, on="site")
        .group_by("name", "sky", "hour")
        .agg((pl.col("power") / pl.col("typical")).median().alias("relative"))
        .sort("name", "sky", "hour")
    )
    return (
        alt.Chart(profile)
        .mark_line(clip=True)
        .encode(
            x=alt.X(
                "hour:Q",
                title="Local hour of day",
                scale=alt.Scale(domain=[6, 18]),
                axis=alt.Axis(values=[6, 9, 12, 15, 18]),
            ),
            y=alt.Y("relative:Q", title="Median reading / the Primary's median reading"),
            color=alt.Color(
                "sky:N",
                title=None,
                scale=alt.Scale(
                    domain=["Sunny days (cloud under 0.15)", "Cloudy days (cloud over 0.5)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH // 2 - 60, height=110)
        .facet(facet=alt.Facet("name:N", title=None), columns=4)
        .properties(
            title=_title(
                text="Raw readings at the 8 Primaries metered only in MVA",
                lines=[
                    (
                        "April to September, daylight hours only (6 to 18). A reading that bounced "
                        "off zero would show a midday rise on sunny days."
                    )
                ],
            )
        )
    )


def capacity_recovery(*, known: pl.DataFrame) -> alt.Chart:
    """Figure 2: fitted capacity against the signed fit's capacity, for both fits."""
    rows = (
        known.filter(
            pl.col("level").is_in([1.0, 1.5])
            & (pl.col("q_share") == 0.0)
            & (pl.col("site") != "D15")
        )
        .select(
            pl.col("level").cast(pl.String).alias("level"),
            "c_signed",
            pl.col("c_unfold").alias("Unfolding fit"),
            pl.col("c_naive").alias("Naive fit"),
        )
        .unpivot(index=["level", "c_signed"], variable_name="fit", value_name="c_fitted")
    )
    top = float(max(rows["c_signed"].max(), rows["c_fitted"].max()))  # ty: ignore[invalid-argument-type]
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, top], "y": [0.0, top]}))
        .mark_line(color="#666666", strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )
    panels = []
    for level in ("1.0", "1.5"):
        sub = rows.filter(pl.col("level") == level)
        panels.append(
            alt.layer(
                alt.Chart(sub)
                .mark_circle(size=22, opacity=0.5)
                .encode(
                    x=alt.X(
                        "c_signed:Q", title="Capacity of the fit to the unhidden signed power (MW)"
                    ),
                    y=alt.Y("c_fitted:Q", title="Capacity fitted from the magnitude (MW)"),
                    color=alt.Color(
                        "fit:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["Unfolding fit", "Naive fit"],
                            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                        ),
                        legend=alt.Legend(orient="top"),
                    ),
                ),
                diagonal,
            ).properties(width=WIDTH - 100, height=170, title=f"Injected solar: level {level}")
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="The unfolding fit recovers the capacity; the naive fit misses it",
            lines=[
                (
                    "One dot per Primary, solar farm, and season-year, with the farm's output "
                    "subtracted from 7 MW Primaries' power and the sign hidden. Dashed line: "
                    "exact recovery. Closer to the line is better."
                )
            ],
        )
    )


def direction(*, known: pl.DataFrame) -> alt.Chart:
    """Figure 3: balanced accuracy of the fitted flow direction, per Primary and scenario."""
    scenarios = {
        "level 1.0": (1.0, 0.0), "level 1.5": (1.5, 0.0), "level 1.0, q 10%": (1.0, 0.1),
        "level 1.0, q 20%": (1.0, 0.2), "level 1.0, q 40%": (1.0, 0.4),
    }  # fmt: skip
    rows = []
    for label, (level, q) in scenarios.items():
        d = known.filter(
            (pl.col("level") == level)
            & (pl.col("q_share") == q)
            & (pl.col("truth_negative") >= common.MIN_CLASS_ROWS)
            & (pl.col("truth_positive") >= common.MIN_CLASS_ROWS)
        )
        for site, g in d.group_by("site"):
            rows.append(
                {
                    "scenario": label,
                    "Primary": common.PRIMARY_NAMES[str(site[0])],
                    "balanced_accuracy": 0.5
                    * (
                        g["tp_unfold"].sum() / g["truth_negative"].sum()
                        + g["tn_unfold"].sum() / g["truth_positive"].sum()
                    ),
                }
            )
    frame = pl.DataFrame(rows)
    order = list(scenarios)
    dots = (
        alt.Chart(frame)
        .mark_circle(size=60)
        .encode(
            x=alt.X("scenario:N", sort=order, title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(
                "balanced_accuracy:Q",
                title="Balanced accuracy of flow direction",
                scale=alt.Scale(domain=[0.4, 1.0]),
            ),
            color=alt.Color(
                "Primary:N",
                scale=alt.Scale(
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.BRAND_ORANGE_LIGHT,
                        "#555555",
                        "#999999",
                    ]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    rules = (
        alt.Chart(
            pl.DataFrame(
                {"y": [0.5, 0.8], "label": ["0.5: all-positive guess", "0.8: pre-registered limit"]}
            )
        )
        .mark_rule(strokeDash=[4, 3], color="#666666")
        .encode(y="y:Q")
    )
    return alt.layer(rules, dots).properties(
        width=WIDTH - 80,
        height=230,
        title=_title(
            text="Direction is found at most Primaries, but not with a reactive-power floor",
            lines=[
                (
                    "One dot per MW Primary: the mean of the share of reverse-flow and "
                    "forward-flow "
                    "half-hours given the right sign, pooled over windows. Higher is better. "
                    "q is the "
                    "reactive-power floor as a share of midday power."
                )
            ],
        ),
    )


def real_mva() -> alt.Chart:
    """Figure 4: fitted capacity and fitted reverse-flow share at the 8 MVA-only Primaries."""
    point = pl.read_parquet(common.REAL_PATH).filter(pl.col("units") == "MVA")
    boot = pl.read_parquet(common.REAL_BOOT_PATH).filter(pl.col("units") == "MVA")
    summary = (
        point.group_by("site")
        .agg(pl.col("c_unfold").median().alias("capacity"), pl.col("negative_share").median())
        .join(
            boot.group_by("site").agg(
                pl.col("c_unfold").quantile(0.05).alias("low"),
                pl.col("c_unfold").quantile(0.95).alias("high"),
                pl.col("negative_share").quantile(0.95).alias("negative_p95"),
            ),
            on="site",
        )
        .with_columns(
            Primary=pl.col("site").replace_strict(common.PRIMARY_NAMES),
            share_label=(pl.col("negative_share") * 100).round(1).cast(pl.String) + "% negative",
        )
    )
    base = alt.Chart(summary).encode(y=alt.Y("Primary:N", sort="-x", title=None))
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(
        x=alt.X("low:Q", title="Fitted solar capacity (MW), median over season-years"), x2="high:Q"
    )
    dots = base.mark_circle(size=70, color=ocf.DATA_BLUE).encode(x="capacity:Q")
    text = base.mark_text(align="left", dx=8, color="#444444").encode(
        x="high:Q", text="share_label:N"
    )
    return alt.layer(bars, dots, text).properties(
        width=WIDTH - 160,
        height=220,
        title=_title(
            text="No reverse flow is found at any Primary metered only in MVA",
            lines=[
                (
                    "Dot: median over season-years. Line: 5th to 95th percentile of fits that "
                    "resample "
                    "whole weeks. Text: median share of half-hours with fitted negative flow. "
                    "Unfolding and naive fits agree except at Stickney, whose capacity is 0."
                )
            ],
        ),
    )


def main() -> None:
    """Write the four figures."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    known = pl.read_parquet(common.KNOWN_PATH)
    for name, chart in (
        ("figure_1_raw_profile", raw_profile()),
        ("figure_2_capacity_recovery", capacity_recovery(known=known)),
        ("figure_3_direction", direction(known=known)),
        ("figure_4_mva_primaries", real_mva()),
    ):
        chart.save(common.CHART_DIR / f"{name}.svg")
    print(sorted(p.name for p in common.CHART_DIR.iterdir()))


if __name__ == "__main__":
    main()
