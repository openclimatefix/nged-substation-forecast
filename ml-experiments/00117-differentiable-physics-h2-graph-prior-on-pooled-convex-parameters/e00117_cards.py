"""Draw the substation cards (provisional): graph-prior parameters, example weeks, cold snaps.

One figure per decision site, under its D-label: (1) the heating slopes, the profile scale, and the
holiday level of G, N, H, and NH with their 80% month-block intervals (capacity only if its own
coverage reading passed in the smooth cell); (2) example forecasts for four weeks fixed before
scoring (median, 80%, and 90% bands of G and NH against the observed series); (3) how G, N, H, and
NH do on the site's cold-snap days. The two excluded series never appear.

    uv run python ml-experiments/00117-*/e00117_cards.py [--capacity-passed]
"""

import argparse
import json
import textwrap
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00117_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = Path.home() / ".claude" / "scratch" / "e00117_png"
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
WEEKS: Final[dict[str, date]] = {
    "Summer": date(2025, 7, 14),
    "Autumn": date(2025, 10, 13),
    "Winter": date(2026, 1, 12),
    "Spring": date(2026, 4, 13),
}
"""The Mondays that start the four example weeks, fixed in the plan before any score was read."""
ARM_COLOURS: Final[dict[str, str]] = {
    "G": ocf.BRAND_ORANGE,
    "N": ocf.DATA_BLUE,
    "H": ocf.DATA_PURPLE,
    "NH": ocf.DATA_GREEN,
}
ARM_FILES: Final[dict[str, str]] = {"G": "g", "N": "n", "H": "h_lo", "NH": "nh_lo"}
FUNCTION_LABELS: Final[dict[str, str]] = {
    "slope_0": "Heating slope, night",
    "slope_1": "Heating slope, morning",
    "slope_2": "Heating slope, midday",
    "slope_3": "Heating slope, evening",
    "scale": "Profile scale",
    "capacity": "PV capacity",
    "holiday": "Holiday level",
}
WARNING: Final[str] = (
    "Provisional: fitted to past weather on a screening run. Heating slopes are known to about "
    "38% of their size at real noise (experiment 113); read them as weakly identified."
)


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def load_set(*, arm: str) -> pl.DataFrame:
    """Read the scored year's predictions of an arm."""
    path = common.DATA_DIR / "predictions" / f"scored_{ARM_FILES[arm]}" / "predictions.parquet"
    return pl.read_parquet(path)


def clean_weeks(*, frame: pl.DataFrame, series_ids: list[int]) -> dict[str, tuple[date, bool]]:
    """Return each season's Monday, moved to the next clean Monday if the week holds an event.

    A week is not clean if any decision site has a holiday-period day or a hand-filtered event in
    it. The rule was fixed before scoring.

    Args:
        frame: One arm's scored-year predictions.
        series_ids: The decision substations' series identifiers.

    Returns:
        For each season, the Monday used and whether it moved from the planned one.
    """
    rows = frame.filter(pl.col("time_series_id").is_in(series_ids))
    out = {}
    for season, monday in WEEKS.items():
        start = monday
        while True:
            stop = start + timedelta(days=7)
            week = rows.filter(
                (pl.col("valid_time").dt.date() >= start) & (pl.col("valid_time").dt.date() < stop)
            )
            if not (week["event_day"].any() or week["switching_mask"].any()):
                break
            start += timedelta(days=7)
        out[season] = (start, start != monday)
    return out


def parameter_panel(*, intervals: pl.DataFrame, site: str, capacity: bool) -> alt.TopLevelMixin:
    """Return the parameter panel of one site: four arms' estimates with 80% intervals."""
    keep = [f for f in FUNCTION_LABELS if capacity or f != "capacity"]
    frame = intervals.filter(
        (pl.col("node") == site)
        & pl.col("functional").is_in(keep)
        & pl.col("arm").is_in(["g", "n", "h_lo", "nh_lo"])
    ).with_columns(
        label=pl.col("functional").replace_strict(FUNCTION_LABELS),
        arm_label=pl.col("arm").replace_strict({"g": "G", "n": "N", "h_lo": "H", "nh_lo": "NH"}),
    )
    colour = alt.Color(
        "arm_label:N",
        scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
        legend=alt.Legend(orient="top", title=None),
    )
    y = alt.Y("label:N", sort=[FUNCTION_LABELS[f] for f in keep], title=None)
    offset = alt.YOffset("arm_label:N", sort=list(ARM_COLOURS))
    base = alt.Chart(frame)
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="Estimate (points of effective capacity; line: 80% interval)"),
        x2="hi:Q",
        y=y,
        yOffset=offset,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y, yOffset=offset, color=colour
    )
    return (bars + dots).properties(width=WIDTH, height=30 * len(keep))


def forecast_panel(
    *, sets: dict[str, pl.DataFrame], series_id: int, weeks: dict[str, tuple[date, bool]]
) -> alt.TopLevelMixin:
    """Return the example weeks of G and NH for one site."""
    parts = []
    order = [f"{season}, week of {monday:%d %b %Y}" for season, (monday, _) in weeks.items()]
    for arm in ("G", "NH"):
        frame = sets[arm].filter(pl.col("time_series_id") == series_id)
        for season, (monday, _) in weeks.items():
            stop = monday + timedelta(days=7)
            week = frame.filter(
                (pl.col("valid_time").dt.date() >= monday) & (pl.col("valid_time").dt.date() < stop)
            ).with_columns(
                arm=pl.lit(arm),
                season=pl.lit(f"{season}, week of {monday:%d %b %Y}"),
                day=(
                    pl.col("valid_time") - pl.lit(monday).cast(pl.Datetime("us", "UTC"))
                ).dt.total_minutes()
                / 1440.0,
            )
            parts.append(week)
    data = pl.concat(parts).select(
        "arm", "season", "day", "q05", "q10", "q50", "q90", "q95", "actual"
    )
    arm_colour = alt.Color(
        "arm:N",
        legend=None,
        scale=alt.Scale(domain=["G", "NH"], range=[ARM_COLOURS["G"], ARM_COLOURS["NH"]]),
    )
    base = alt.Chart(data)
    x = alt.X("day:Q", title="Days since Monday 00:00 UTC", axis=alt.Axis(values=list(range(8))))
    band90 = base.mark_area(opacity=0.18).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q05:Q", title="Points"), y2="q95:Q", color=arm_colour
    )
    band80 = base.mark_area(opacity=0.3).encode(  # ty: ignore[unresolved-attribute]
        x=x, y="q10:Q", y2="q90:Q", color=arm_colour
    )
    median = base.mark_line(strokeWidth=1.3).encode(  # ty: ignore[unresolved-attribute]
        x=x, y="q50:Q", color=arm_colour
    )
    observed = base.mark_line(strokeWidth=1.0, color="#222222").encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(band90, band80, median, observed)
        .properties(width=250, height=70)
        .facet(
            row=alt.Row(
                "season:N",
                title=None,
                sort=order,
                header=alt.Header(
                    labelAngle=0, labelAlign="left", labelOrient="top", labelPadding=2
                ),
            ),
            column=alt.Column("arm:N", title=None, sort=["G", "NH"]),
        )
    )


def cold_snap_table(*, sets: dict[str, pl.DataFrame], series_id: int) -> pl.DataFrame:
    """Return each arm's cold-snap score and interval coverage for one site."""
    rows = []
    for arm, frame in sets.items():
        part = frame.filter(
            (pl.col("time_series_id") == series_id)
            & pl.col("cold_snap")
            & ~pl.col("switching_mask")
        )
        days = part.select(day=(pl.col("valid_time") - pl.duration(minutes=30)).dt.date())[
            "day"
        ].n_unique()
        if days < common.MIN_COLD_SNAP_DAYS:
            rows.append({"arm": arm, "days": days, "score": None, "cover80": None, "cover90": None})
            continue
        quantiles = part.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float64)
        actual = part["actual"].to_numpy().astype(np.float64)
        score = common.quantile_score(actual=actual, quantiles=quantiles).mean()
        rows.append(
            {
                "arm": arm,
                "days": days,
                "score": float(score),
                "cover80": float(
                    np.mean((actual >= part["q10"].to_numpy()) & (actual <= part["q90"].to_numpy()))
                ),
                "cover90": float(
                    np.mean((actual >= part["q05"].to_numpy()) & (actual <= part["q95"].to_numpy()))
                ),
            }
        )
    return pl.DataFrame(rows)


def cross_site_figure(*, intervals: pl.DataFrame) -> alt.TopLevelMixin:
    """Compare the five sites' G estimates of each parameter side by side, with 80% intervals."""
    keep = ["slope_0", "slope_1", "slope_2", "slope_3", "scale", "holiday"]
    frame = intervals.filter(
        pl.col("node").is_in(list(common.DECISION_SITES))
        & (pl.col("arm") == "g")
        & pl.col("functional").is_in(keep)
    ).with_columns(
        label=pl.col("functional").replace_strict(FUNCTION_LABELS),
        one=pl.lit(0.0),
    )
    y = alt.Y("node:N", sort=list(common.DECISION_SITES), title=None)
    bars = (
        alt.Chart()
        .mark_rule(strokeWidth=2, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Estimate (points of effective capacity; line: 80% interval)"),
            x2="hi:Q",
            y=y,
        )
    )
    dots = (
        alt.Chart()
        .mark_point(filled=True, size=50, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x="point:Q", y=y
        )
    )
    return (
        alt.layer(bars, dots, data=frame)
        .properties(width=WIDTH // 2, height=110)
        .facet(
            facet=alt.Facet("label:N", sort=[FUNCTION_LABELS[f] for f in keep], title=None),
            columns=2,
        )
        .properties(
            title=alt.TitleParams(
                text="The five decision substations side by side, each fitted alone (G)",
                subtitle=textwrap.wrap(WARNING, WRAP),
                anchor="start",
                fontSize=14,
                subtitleFontSize=11,
            )
        )
    )


def main() -> None:
    """Draw one card per decision site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-passed", action="store_true")
    arguments = parser.parse_args()
    cache = common.load_cache()
    intervals = pl.read_parquet(common.WORK_DIR / "real_intervals_scored.parquet")
    _save(cross_site_figure(intervals=intervals), name="card_cross_site")
    sets = {arm: load_set(arm=arm) for arm in ARM_COLOURS}
    ids = {s: cache.sites[s].table.time_series_id for s in common.DECISION_SITES}
    weeks = clean_weeks(frame=sets["G"], series_ids=list(ids.values()))
    tables = {}
    for site, series_id in ids.items():
        panel = parameter_panel(intervals=intervals, site=site, capacity=arguments.capacity_passed)
        _save(
            panel.properties(
                title=alt.TitleParams(
                    text=f"{site}: fitted parameters, graph prior against none",
                    subtitle=textwrap.wrap(WARNING, WRAP),
                    anchor="start",
                    fontSize=14,
                    subtitleFontSize=11,
                )
            ),
            name=f"card_{site}_parameters",
        )
        _save(
            forecast_panel(sets=sets, series_id=series_id, weeks=weeks).properties(
                title=alt.TitleParams(
                    text=f"{site}: example weeks, scored year",
                    subtitle=textwrap.wrap(
                        "Black: observed. Colour: median, 80% band (dark), and 90% band (light) "
                        "of G (left) and NH (right), in points of effective capacity.",
                        WRAP,
                    ),
                    anchor="start",
                    fontSize=14,
                    subtitleFontSize=11,
                )
            ),
            name=f"card_{site}_weeks",
        )
        tables[site] = cold_snap_table(sets=sets, series_id=series_id).to_dicts()
    (common.WORK_DIR / "cards.json").write_text(
        json.dumps(
            {"weeks": {k: [str(v[0]), v[1]] for k, v in weeks.items()}, "cold_snaps": tables},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
