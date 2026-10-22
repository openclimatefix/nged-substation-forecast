# ruff: noqa: E501
"""Draw the Primaries' figures of experiment 198 from `results.json` and the stored forecasts.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00198-*/e00198_charts.py
"""

import json
from typing import Final

import altair as alt
import e00198_common as common
import e00198_fit as fit
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf

WIDTH: Final[int] = 590


def headline_chart(*, results: dict) -> None:
    """Draw the contrasts with intervals: the real cooling term, and the injected known answers."""
    known = results["known_answer"]
    post = results["post_hoc_known_answer"]
    rows = [
        ("Real: all summer rows", *results["c1"], "real"),
        ("Real: leads 0.5 to 48 h", *results["c1_by_lead"]["0.5-48 h"], "real"),
        ("Real: leads 48 to 168 h", *results["c1_by_lead"]["48-168 h"], "real"),
        ("Real: without the hottest spell", *results["c1_without_hottest_spell"], "real"),
        ("Real: hot days only", *results["c1_hot"], "real") if results["c1_hot"] else None,
        ("Injected 0.1 MW per degree at D10", *known["c1"], "injected"),
        ("Injected, ERA5 temperature (post hoc)", *post["oracle_era5_tau_20"]["c1"], "injected"),
    ]
    data = [
        {"label": label, "mean": mean, "low": low, "high": high, "kind": kind}
        for label, mean, low, high, kind in (r for r in rows if r)
    ]
    order = [d["label"] for d in data]
    colour = alt.Color(
        "kind:N", title=None, legend=alt.Legend(orient="top"),
        scale=alt.Scale(domain=["real", "injected"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
    )  # fmt: skip
    base = alt.Chart(alt.Data(values=data)).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), color=colour
    )
    layer = (
        base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title="Cooling arm minus arm D (quantile-score points; below 0 is better)"), x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=90).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(alt.InlineData(values=[{"x": 0}])).mark_rule(color="#888", strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(alt.InlineData(values=[{"x": fit.SUPPORTED_BAR}])).mark_rule(color="#888", strokeDash=[1, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # fmt: skip
    layer.properties(
        width=WIDTH - 250,
        height=220,
        title=alt.TitleParams(
            "A cooling term does not lower the summer score, and an injected one would have shown",
            subtitle=[
                "Dot: estimate. Line: 99% interval from resampling 14-day blocks. Dashed rule: no change.",
                "Dotted rule: the pre-registered bar of -0.01. Five decision Primaries, June to September.",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        ),
    ).save(common.FIGURE_DIR / "figure_1_contrasts.svg")


def residual_chart(*, sites: list[fit.SiteData]) -> None:
    """Draw the afternoon residual against ensemble-mean temperature, per Primary."""
    pieces = []
    for data in sites:
        rows = (
            (data.lead <= 24) & np.isin(data.valid_month, [6, 7, 8, 9])
            & (data.hour >= 12) & (data.hour < 18)
        )  # fmt: skip
        frame = (
            pl.DataFrame(
                {
                    "temp_bin": np.floor(data.ens_mean_t[rows] / 2) * 2 + 1,
                    "resid": (data.actual - data.mean)[rows],
                }
            )
            .group_by("temp_bin")
            .agg(pl.col("resid").mean().alias("mean_resid"), pl.len().alias("n"))
            .filter(pl.col("n") >= 40)
        )
        pieces.append(frame.with_columns(site=pl.lit(data.site)))
    data_all = pl.concat(pieces)
    chart = (
        alt.Chart(alt.Data(values=data_all.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X(
                "temp_bin:Q", title="Ensemble-mean temperature (degrees Celsius, 2-degree bins)"
            ),
            y=alt.Y("mean_resid:Q", title="Mean actual minus arm D (points of capacity)"),
            color=alt.Color("site:N", title=None, legend=alt.Legend(orient="top", columns=6)),
        )
        .properties(
            width=WIDTH - 90,
            height=260,
            title=alt.TitleParams(
                "Arm D's afternoon error turns positive above 25 degrees, on few days",
                subtitle=[
                    "Valid 12:00 to 18:00 UTC, June to September, leads up to 24 h, bins with 40 or more rows. A cooling load would rise to the right."
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            ),
        )
    )
    (
        chart
        + alt.Chart(alt.InlineData(values=[{"y": 0}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888", strokeDash=[4, 3])
        .encode(y="y:Q")
    ).save(common.FIGURE_DIR / "figure_2_residual_vs_temperature.svg")


def hot_end_summary(*, sites: list[fit.SiteData]) -> None:
    """Print the afternoon residual at ensemble-mean temperatures of 25 degrees or more, per Primary."""
    print(
        "Hot-end afternoon residual (valid 12:00-18:00 UTC, June-September, leads up to 24 h, ensemble-mean T >= 25):"
    )
    for data in sites:
        rows = (
            (data.lead <= 24) & np.isin(data.valid_month, [6, 7, 8, 9])
            & (data.hour >= 12) & (data.hour < 18) & (data.ens_mean_t >= common.HOT_DAY_C)
        )  # fmt: skip
        mild = (
            (data.lead <= 24) & np.isin(data.valid_month, [6, 7, 8, 9])
            & (data.hour >= 12) & (data.hour < 18) & (data.ens_mean_t >= 19) & (data.ens_mean_t < 25)
        )  # fmt: skip
        residual = data.actual - data.mean
        print(
            f"{data.site}: {int(rows.sum())} rows on {len(np.unique(data.date_index[rows]))} days, "
            f"mean residual {residual[rows].mean():+.2f} points; at 19 to 25 degrees {residual[mild].mean():+.2f} points"
        )


def main() -> None:
    """Draw both figures."""
    results = json.loads((common.WORK_DIR / "results.json").read_text())
    common.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    headline_chart(results=results)
    cache = common.load_cache()
    sites = [fit.load_site(site=s, cache=cache) for s in common.STORED_SITES]
    residual_chart(sites=sites)
    hot_end_summary(sites=sites)


if __name__ == "__main__":
    main()
