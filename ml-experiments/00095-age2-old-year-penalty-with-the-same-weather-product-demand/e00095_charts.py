"""Draw experiment 00095's figures from `results/report.json`, as SVG.

1. The old-minus-recent gap on the control member and on ERA5, with intervals (headline).
2. The absolute quantile score by training year, on each weather product.
3. The gap by series, on each weather product.
4. The largest-gap series' monthly mean demand.
5. Tail calibration: how often the actual value fell below the 0.01, 0.05, 0.95, and 0.99 quantiles.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00095-*/e00095_charts.py
"""

import json
from typing import Final

import altair as alt
import e00095_common as common
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 540
SCOPE: Final[list[str]] = [
    "13 primary substations, 365 daily forecast runs, July 2025 to June 2026.",
    "Each model was trained once, on one year of data.",
]
CONTROL_COLOUR: Final[str] = ocf.BRAND_ORANGE
ERA5_COLOUR: Final[str] = ocf.DATA_BLUE
PRODUCTS: Final[list[str]] = ["Control member forecast", "ERA5"]
YEARS: Final[tuple[tuple[str, str], ...]] = (
    ("old1", "2021-07"),
    ("old2", "2022-07"),
    ("mid", "2023-07"),
    ("recent", "2024-07"),
)
"""Each training-year key and its label."""
TAIL_ARMS: Final[dict[str, str]] = {
    "old1_nwp": "2021-07 year, control member",
    "old1_era5": "2021-07 year, ERA5",
    "recent_nwp": "2024-07 year, control member",
    "recent_era5": "2024-07 year, ERA5",
}


def _report() -> dict:
    """Return the saved report."""
    return json.loads((common.RESULTS_DIR / "report.json").read_text())


def _colour(*, field: str = "product") -> alt.Color:
    """Return the colour encoding that separates the two weather products."""
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=PRODUCTS, range=[CONTROL_COLOUR, ERA5_COLOUR]),
        legend=alt.Legend(orient="top", title=None),
    )


def figure_1() -> alt.TopLevelMixin:
    """The old-minus-recent gap on the control member and on ERA5."""
    report = _report()
    planned, without = report["all_95"], report["without_d8"]
    c1, c2 = "C1 (control member, 2021-07 minus 2024-07)", "C2 (ERA5, 2021-07 minus 2024-07)"
    rows = [
        ("2021-07 year minus 2024-07 year (13 series)", PRODUCTS[0], planned[c1]),
        ("2021-07 year minus 2024-07 year (13 series)", PRODUCTS[1], planned[c2]),
        ("Same, without the largest-gap substation", PRODUCTS[0], without[c1]),
        ("Same, without the largest-gap substation", PRODUCTS[1], without[c2]),
        (
            "2023-07 year minus 2024-07 year (13 series)",
            PRODUCTS[0],
            planned["C4 (9 km control member, 2023-07 minus 2024-07)"],
        ),
        (
            "2023-07 year minus 2024-07 year (13 series)",
            PRODUCTS[1],
            report["all_95"]["ERA5, 2023-07 minus 2024-07"],
        ),
    ]
    data = pl.DataFrame(
        [
            {"group": g, "product": p, "estimate": r["estimate"], "low": r["lo"], "high": r["hi"]}
            for g, p, r in rows
        ]
    )
    order = list(dict.fromkeys(g for g, _, _ in rows))
    base = alt.Chart(data).encode(
        y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset=alt.YOffset("product:N", sort=PRODUCTS),
        color=_colour(),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Older-year minus 2024-07-year quantile score (points of capacity)",
            scale=alt.Scale(domain=[0, 2.4]),
        )
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=230,
        title=alt.TitleParams(
            "The old-year penalty is as large when both years use ERA5, so it is age, not weather",
            subtitle=[
                *SCOPE,
                "Dot: mean difference. Line: interval from resampling whole calendar months",
                "(95%). Each pair compares the two weather products.",
                "Above zero: the older training year is worse.",
            ],
            anchor="start",
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Absolute quantile score by training year, on each weather product."""
    absolute = _report()["absolute"]
    rows = []
    for key, label in YEARS:
        for product, suffix in zip(PRODUCTS, ("nwp", "era5"), strict=True):
            arm = f"{key}_{suffix}"
            if arm in absolute:
                v = absolute[arm]
                rows.append(
                    {
                        "year": label,
                        "product": product,
                        "qs": v["score11"],
                        "low": v["lo"],
                        "high": v["hi"],
                    }
                )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("year:N", title="Training year starts", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("product:N", sort=PRODUCTS),
        color=_colour(),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "qs:Q",
            title="Quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule().encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "Each older training year scores worse, on both weather products",
            subtitle=[
                *SCOPE,
                "Dot: mean. Line: 95% interval from re-drawing whole months. The control member",
                "arms use all 51 ensemble members; the ERA5 arms have one weather value per time.",
            ],
            anchor="start",
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """The 2021-07 minus 2024-07 gap by series, on each weather product."""
    by_series = _report()["by_series"]
    rows = []
    for site, v in by_series.items():
        rows.append({"site": site, "product": PRODUCTS[0], "gap": v["c1"]})
        rows.append({"site": site, "product": PRODUCTS[1], "gap": v["c2"]})
    order = sorted(by_series, key=lambda s: -by_series[s]["c1"])
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("site:N", sort=order, title="Primary substation (anonymised label)"),
        color=_colour(),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "gap:Q",
            title="2021-07 year minus 2024-07 year quantile score (points of capacity)",
            scale=alt.Scale(domain=[-0.5, 14]),
        )
    )
    return alt.layer(zero, points).properties(
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            "Both weather products give a similar gap per substation",
            subtitle=[*SCOPE, "Each dot is one substation. Above zero: the 2021-07 year is worse."],
            anchor="start",
        ),
    )


def figure_4() -> alt.TopLevelMixin:
    """Monthly mean demand of the largest-gap series."""
    months = pl.DataFrame(_report()["d8"]["months"]).with_columns(
        month=pl.col("month").str.to_date("%Y-%m")
    )
    line = (
        alt.Chart(months)
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:T", title="Month"),
            y=alt.Y("mean:Q", title="Mean demand over effective capacity"),
        )
    )
    return line.properties(
        width=WIDTH,
        height=230,
        title=alt.TitleParams(
            "Substation D8's demand fell steadily, with no step",
            subtitle=[
                "Monthly mean demand of the Primary substation labelled D8,",
                "over its effective capacity. Demand swings with the seasons.",
                "At least 99.3% of the half-hours of every month are observed.",
            ],
            anchor="start",
        ),
    )


def figure_5() -> alt.TopLevelMixin:
    """Share of actual values at or below each tail quantile, per arm."""
    tails = _report()["tails"]
    rows = []
    for arm, label in TAIL_ARMS.items():
        for column, v in tails[arm].items():
            rows.append({"arm": label, "level": column, "coverage": v["coverage"]})
    levels = ["q01", "q05", "q95", "q99"]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "level:N",
            sort=levels,
            title="Quantile level",
            axis=alt.Axis(labelAngle=0, labelExpr="slice(datum.label, 1) + '%'"),
        ),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=list(TAIL_ARMS.values()),
                range=[CONTROL_COLOUR, ERA5_COLOUR, ocf.DATA_PURPLE, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=300),
        ),
        xOffset=alt.XOffset("arm:N"),
    )
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("coverage:Q", title="Share of actual values at or below the quantile")
    )
    target = (
        alt.Chart(pl.DataFrame({"level": levels, "t": [0.01, 0.05, 0.95, 0.99]}))
        .mark_tick(color="#222222", thickness=2, size=70)
        .encode(x=alt.X("level:N", sort=levels), y="t:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(bars, target).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "The 2021-07 year's low quantiles sit too high",
            subtitle=[
                *SCOPE,
                "Bar: share of actual values at or below the quantile. Tick: the ideal share.",
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
        ("figure_4", figure_4()),
        ("figure_5", figure_5()),
    ):
        chart.save(common.CHART_DIR / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
