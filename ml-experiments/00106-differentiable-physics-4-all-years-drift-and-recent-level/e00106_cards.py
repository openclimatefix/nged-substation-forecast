"""Draw the substation cards of experiment 106: what the fitted model says about each substation.

One card per decision substation (D-labels only), from arm A3's frozen fit at 1 July 2025: the
pinned solar capacity beside the capacity the convex model fits; the daily profile by day type; the
heating response by hour band and by the time over which temperature is averaged; the Christmas
effect and the lighting ramps; the fitted drift; and the Kalman filter's level and transient over
the scored year. One more figure compares the key properties of the five substations.

**Provisional: fitted to past weather on a screening run.** The pooled model's capacity (the plan's
third capacity) is not drawn, because this run used the convex model; the pinned and the fitted
capacity are shown, and neither is a measurement.

    uv run python ml-experiments/00106-*/e00106_cards.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00106_common as common
import e00106_g as g
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_calendar import BAND_NAMES, DAY_TYPES, SMOOTHING_HOURS
from studies.dp_gam import N_BANDS
from studies.dp_pinned_capacity import read_vintages

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PANEL_WIDTH: Final[int] = 250
PANEL_HEIGHT: Final[int] = 120
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]
BAND_COLOURS: Final[list[str]] = [
    "#808080",
    ocf.DATA_SKY,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
]
DAY_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE, ocf.BRAND_ORANGE]


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def site_tables(*, site: str, cache: object, vintages: pl.DataFrame, tuned: dict) -> dict:
    """Fit A1 at the scored cut and return the tables a card draws.

    Args:
        site: The substation's label.
        cache: The feature cache.
        vintages: The real-time capacity vintages.
        tuned: The tuned settings.

    Returns:
        A dict of Polars tables: `profile`, `heating`, `lighting`, `tent`, `drift`, `capacity`,
        `filter`, and a dict `summary` of numbers for the comparison figure.
    """
    definition = g.arm_definitions(tuned=tuned)["a1"]
    run = g.run_site(
        cache=cache,  # ty: ignore[invalid-argument-type]
        vintages=vintages,
        site=site,
        cut=common.SCORED_CUT,
        definition=definition,
        with_replica=False,
    )
    layout, theta = run.fit.layout, run.fit.theta[0]
    profile = theta[layout.slices["profile"]].reshape(len(DAY_TYPES), 48)
    heating = theta[layout.slices["heating"]].reshape(len(SMOOTHING_HOURS), N_BANDS)
    tent = theta[layout.slices["tent"]].reshape(2, N_BANDS)
    n_knots = len(layout.knot_days)
    drift = theta[layout.slices["drift"]].reshape(N_BANDS, n_knots)
    assert run.pinned is not None
    pinned_rows = [
        {"date": str(week), "points": float(points), "kind": "Pinned (real-time vintage)"}
        for week, points in zip(run.pinned.weeks, run.pinned.points, strict=True)
    ]
    fitted = pl.read_parquet(common.WORK_DIR / "grid_capacities.parquet").filter(
        pl.col("site") == site,
        pl.col("cut") == "scored",
        pl.col("arm") == "a1_fc",
    )
    fitted_rows = [
        {
            "date": f"{int(row['season_year'])}-07-01",
            "points": row["capacity_points"],
            "kind": "Fitted (convex model)",
        }
        for row in fitted.iter_rows(named=True)
        if row["season_year"] >= 2019
    ]
    level = pl.read_parquet(common.DATA_DIR / "predictions" / "scored_a3" / "predictions.parquet")
    level = level.filter(pl.col("time_series_id") == cache.sites[site].table.time_series_id)  # ty: ignore[unresolved-attribute]
    daily = (
        level.group_by(pl.col("valid_time").dt.truncate("1d").alias("day"))
        .agg(pl.col("level").mean(), pl.col("anomaly").mean())
        .sort("day")
    )
    return {
        "profile": pl.DataFrame(
            [
                {"half_hour": h / 2.0, "points": float(profile[d, h]), "day_type": name}
                for d, name in enumerate(DAY_TYPES)
                for h in range(48)
            ]
        ),
        "heating": pl.DataFrame(
            [
                {
                    "band": BAND_NAMES[b].capitalize(),
                    "points": float(heating[k, b]),
                    "window": f"{SMOOTHING_HOURS[k]} h",
                }
                for k in range(len(SMOOTHING_HOURS))
                for b in range(N_BANDS)
            ]
        ),
        "lighting": pl.DataFrame(
            [
                {"ramp": f"ramp {i + 1}", "points": float(v)}
                for i, v in enumerate(theta[layout.slices["lighting"]])
            ]
        ),
        "tent": pl.DataFrame(
            [
                {"band": BAND_NAMES[b].capitalize(), "points": float(tent[0, b])}
                for b in range(N_BANDS)
            ]
        ),
        "drift": pl.DataFrame(
            [
                {
                    "date": str(np.datetime64(int(layout.knot_days[k]), "D")),
                    "points": float(drift[b, k] - drift[b].mean()),
                    "band": BAND_NAMES[b].capitalize(),
                }
                for b in range(N_BANDS)
                for k in range(n_knots)
            ]
        ),
        "capacity": pl.DataFrame([*pinned_rows, *fitted_rows]),
        "filter": pl.concat(
            [
                daily.select("day", pl.col("level").alias("points")).with_columns(
                    state=pl.lit("Level")
                ),
                daily.select("day", pl.col("anomaly").alias("points")).with_columns(
                    state=pl.lit("Transient")
                ),
            ]
        ),
        "summary": {
            "site": site,
            "pinned capacity (points)": run.pinned.held_points,
            "heating slope, all bands (points per 10 C)": float(heating.sum()),
            "drift range, midday (points)": float(np.ptp(drift[2])),
            "level standard deviation (points)": float(np.std(daily["level"].to_numpy())),
        },
    }


def card(*, site: str, tables: dict) -> alt.TopLevelMixin:
    """Return one substation's card."""
    capacity = tables["capacity"].with_columns(date=pl.col("date").str.to_date())
    cap_chart = (
        alt.Chart(capacity)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y")),
            y=alt.Y("points:Q", title="Points of capacity"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Pinned (real-time vintage)", "Fitted (convex model)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=200),
            ),
        )
        .properties(
            title=alt.TitleParams("Solar capacity", fontSize=12),
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
    )
    profile = (
        alt.Chart(tables["profile"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X(
                "half_hour:Q",
                title="Local hour of day",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y("points:Q", title="Points", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "day_type:N",
                scale=alt.Scale(domain=list(DAY_TYPES), range=DAY_COLOURS),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(
            title=alt.TitleParams("Daily profile", fontSize=12),
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
    )
    heating = (
        alt.Chart(tables["heating"])  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=[b.capitalize() for b in BAND_NAMES], title=None),
            y=alt.Y("points:Q", title="Points per 10 C below 15.5 C"),
            color=alt.Color(
                "window:N",
                scale=alt.Scale(
                    domain=[f"{h} h" for h in SMOOTHING_HOURS],
                    range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(
                    title="Temperature averaged over", orient="top", direction="horizontal"
                ),
            ),
        )
        .properties(
            title=alt.TitleParams("Heating response", fontSize=12),
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
    )
    christmas = (
        alt.Chart(tables["tent"])  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_GREEN)
        .encode(
            x=alt.X("band:N", sort=[b.capitalize() for b in BAND_NAMES], title=None),
            y=alt.Y("points:Q", title="Points on Christmas Day"),
        )
        .properties(
            title=alt.TitleParams("Christmas Day effect", fontSize=12),
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
    )
    drift_table = tables["drift"].with_columns(date=pl.col("date").str.to_date())
    drift = (
        alt.Chart(drift_table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y")),
            y=alt.Y("points:Q", title="Points (centred)"),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(domain=[b.capitalize() for b in BAND_NAMES], range=BAND_COLOURS),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(
            title=alt.TitleParams("Drift (held at its last value)", fontSize=12),
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
    )
    filter_chart = (
        alt.Chart(tables["filter"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=["Level", "Transient"], range=[ocf.DATA_BLUE, ocf.DATA_GREEN]
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(
            title=alt.TitleParams("Filter state, day-ahead", fontSize=12),
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
        )
    )
    grid = alt.vconcat(
        alt.hconcat(cap_chart, profile).resolve_scale(color="independent"),
        alt.hconcat(heating, christmas).resolve_scale(color="independent"),
        alt.hconcat(drift, filter_chart).resolve_scale(color="independent"),
    ).resolve_scale(color="independent")
    return grid.properties(
        title=_title(
            text=f"Substation card: {site}",
            subtitle=(
                "Provisional: fitted to past weather on a screening run. Arm A1's frozen fit at "
                "1 July 2025 (convex model, pinned capacity, drift) and arm A3's Kalman filter "
                "over the scored year. Every value is in points of the substation's effective "
                "capacity. The capacities are not measurements: the pinned one is a real-time "
                "estimate and the fitted one depends on the other terms."
            ),
        )
    )


def comparison(*, summaries: list[dict]) -> alt.TopLevelMixin:
    """Return the figure that compares the five substations' key properties."""
    table = pl.DataFrame(summaries).unpivot(
        index="site", variable_name="property", value_name="value"
    )
    properties = [c for c in summaries[0] if c != "site"]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("site:N", sort=SITE_ORDER, title=None),
            y=alt.Y("value:Q", title=None),
        )
        .properties(width=PANEL_WIDTH // 2 + 20, height=PANEL_HEIGHT + 20)
        .facet(
            column=alt.Column(
                "property:N",
                sort=properties,
                title=None,
                header=alt.Header(labelLimit=150, labelFontSize=10),
            )
        )
        .resolve_scale(y="independent")
    )
    return chart.properties(
        title=_title(
            text="Five substations side by side",
            subtitle=(
                "Provisional: fitted to past weather on a screening run. Left to right: the "
                "pinned solar capacity held after 30 June 2025, the heating response summed over "
                "hour bands and averaging windows, the range of the midday drift since 2019, and "
                "the day-to-day spread of the Kalman filter's level in the scored year. All in "
                "points of effective capacity."
            ),
        )
    )


def main() -> None:
    """Draw the five cards and the comparison."""
    cache = common.load_cache()
    vintages = read_vintages()
    tuned = g.read_tuned()
    summaries = []
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for site in SITE_ORDER:
        tables = site_tables(site=site, cache=cache, vintages=vintages, tuned=tuned)
        summaries.append(tables["summary"])
        chart = card(site=site, tables=tables)
        name = f"figure_card_{site.lower()}"
        chart.save(DOCS_DIR / f"{name}.svg")
        chart.save(Path("/tmp/e00106_png") / f"{name}.png", scale_factor=1.5)
        print("wrote", name)
    chart = comparison(summaries=summaries)
    chart.save(DOCS_DIR / "figure_card_comparison.svg")
    chart.save(Path("/tmp/e00106_png") / "figure_card_comparison.png", scale_factor=1.5)
    print("wrote figure_card_comparison")


if __name__ == "__main__":
    main()
