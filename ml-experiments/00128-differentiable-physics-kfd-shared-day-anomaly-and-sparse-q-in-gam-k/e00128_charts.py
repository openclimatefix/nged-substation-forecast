"""Draw the figures of experiment 128's write-up from the stored results and prediction files.

Every figure carries its own scope, unit, and direction in its title and subtitle. The titles
describe what a figure shows and are rewritten to state the finding once the results are in. SVGs go
to the write-up's folder, PNGs to a scratch folder for the render check. Only D-labels appear.

    uv run python ml-experiments/00128-*/e00128_charts.py [--only headline gate ...]
"""

import argparse
import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00128_common as common
import e00128_filter as joint
import e00128_fit as fit
import e00128_score as score
import e00128_streams as streams
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
ARM_LABELS: Final[dict[str, str]] = {
    "GK": "GK (level and anomaly filter, experiment 111)",
    "KD": "KD (GK plus a day-level state)",
    "KS": "KS (GK with sparse level variances)",
    "KDS": "KDS (day-level state and sparse level variances)",
    "KEZ": "KEZ (experiment 118)",
    "bg2": "BG2 (production recipe)",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "GK": ocf.DATA_BLUE,
    "KD": ocf.DATA_SKY,
    "KS": ocf.DATA_GREEN,
    "KDS": ocf.BRAND_ORANGE,
    "KEZ": ocf.DATA_PURPLE,
    "bg2": DARK,
    "climatology": "#C8C8C8",
    "manual_heuristic": GREY,
}
SUBSET_LABELS: Final[dict[str, str]] = {
    "all rows": "All rows",
    "winter": "Winter (Nov to Mar)",
    "evening_peak": "Winter evening peak (16:00 to 20:00)",
    "winter_night": "Winter nights (00:00 to 06:00)",
    "cold_snap": "Cold-snap days",
    "cold_winter_night": "Cold winter nights (22:00 to 08:00)",
    "holiday_and_next": "Holiday days and the day after",
    "other_days": "Other days",
}
EXAMPLE_WEEKS: Final[dict[str, datetime]] = {
    "July": datetime(2025, 7, 14, tzinfo=UTC),
    "October": datetime(2025, 10, 13, tzinfo=UTC),
    "January": datetime(2026, 1, 12, tzinfo=UTC),
    "April": datetime(2026, 4, 13, tzinfo=UTC),
}
SITE_ORDER: Final[tuple[str, ...]] = (*common.DECISION_SITES, "D9")
ONE_WEEK: Final[timedelta] = timedelta(days=7)


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def results(*, period: str) -> dict:
    """Return the stored results of a period."""
    return json.loads((common.WORK_DIR / f"results_{period}.json").read_text())


def _zero_rule() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )


def figure_headline(*, scored: dict, replicate: dict) -> alt.TopLevelMixin:
    """The five planned contrasts and the d-removal checks at D+1, in both periods."""
    entries = (
        ("C1 KDS minus GK", "planned"),
        ("C2 KD minus GK", "planned"),
        ("C3 KS minus GK", "planned"),
        ("C4 KDS minus KD", "planned"),
        ("C5 KDS minus KS", "planned"),
        ("KDS minus GK-sp", "exploratory"),
        ("KDS-nod minus KDS", "exploratory"),
        ("KDS-shuf minus KDS", "exploratory"),
        ("KDS-da minus KDS", "exploratory"),
    )
    rows = []
    for period, data in (("scored year", scored), ("replicate months", replicate)):
        for name, group in entries:
            entry = data.get(group, {}).get(name, {}).get("all rows", {}).get("D+1")
            if entry is None or entry["lower"] is None:
                continue
            rows.append(
                {
                    "contrast": name,
                    "period": period,
                    "estimate": entry["estimate"],
                    "low": entry["lower"],
                    "high": entry["upper"],
                }
            )
    table = pl.DataFrame(rows)
    order = list(dict.fromkeys(table["contrast"].to_list()))
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset=alt.YOffset("period:N", sort=["scored year", "replicate months"]),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(
                domain=["scored year", "replicate months"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    chart = (
        _zero_rule()
        + base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "estimate:Q",
                title="Difference in quantile score (points; below zero favours the first arm)",
            )
        )
    ).properties(width=WIDTH - 340, height=28 * len(order))
    return chart.properties(
        title=_title(
            text="Planned contrasts of the day-level state and the sparse level variances at D+1",
            subtitle=(
                "Dot: difference in mean quantile score at day D+1 (leads 15 to 39 hours) over the "
                "five decision substations, rows outside the switching events. Line: interval from "
                "resampling whole calendar months, paired across arms (C1 to C5: 99%; the rest: "
                "95%). Dashed line: no difference."
            ),
        )
    )


def figure_by_lead(*, scored: dict) -> alt.TopLevelMixin:
    """Absolute mean quantile score by lead band for the four planned arms and the references."""
    leads = ["0-24 h", "24-48 h", "48-144 h", "144-168 h"]
    rows = []
    for arm in ("GK", "KD", "KS", "KDS"):
        by_subset = scored["absolute"].get(arm)
        if by_subset is not None:
            rows.extend(
                {"arm": ARM_LABELS[arm], "lead": lead, "qs": by_subset["all rows"][lead]["qs"]}
                for lead in leads
            )
    for name in ("kez", "bg2", "climatology", "manual_heuristic"):
        shared = scored["reference_scores"].get(name, {}).get(name)
        if shared is not None:
            label = ARM_LABELS["KEZ" if name == "kez" else name]
            rows.extend({"arm": label, "lead": lead, "qs": shared[lead]} for lead in leads)
    table = pl.DataFrame(rows)
    keys = [k for k, v in ARM_LABELS.items() if v in set(table["arm"].to_list())]
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "lead:N", sort=leads, title="Lead band (hours after the 09:00 UTC forecast origin)"
            ),
            y=alt.Y(
                "qs:Q",
                title="Mean quantile score (points; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[k] for k in keys], range=[ARM_COLOURS[k] for k in keys]
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=280,
            title=_title(
                text="Mean quantile score by lead band for the planned arms and the references",
                subtitle=(
                    "Mean quantile score by lead band over the five decision substations, scored "
                    "year, rows outside the switching events. The planned arms are on shared "
                    "rows. The references are on the rows they share with KDS."
                ),
            ),
        )
    )


def figure_subsets(*, scored: dict) -> alt.TopLevelMixin:
    """C1, C2, and C3 on the winter and event subsets at D+1."""
    rows = []
    for name, label in (
        ("C1 KDS minus GK", "KDS minus GK"),
        ("C2 KD minus GK", "KD minus GK"),
        ("C3 KS minus GK", "KS minus GK"),
    ):
        for subset, subset_label in SUBSET_LABELS.items():
            entry = scored["planned"][name][subset].get("D+1")
            if entry is None:
                continue
            few = entry["too_few_days"]
            rows.append(
                {
                    "contrast": label,
                    "subset": subset_label,
                    "estimate": entry["estimate"],
                    "low": entry["lower"] if entry["lower"] is not None else entry["estimate"],
                    "high": entry["upper"] if entry["upper"] is not None else entry["estimate"],
                    "note": f"too few days ({entry['n_days']})" if few else "",
                    "interval": entry["lower"] is not None,
                }
            )
    table = pl.DataFrame(rows)
    order = list(SUBSET_LABELS.values())
    base = alt.Chart(table).encode(
        y=alt.Y("subset:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("contrast:N"),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(
                domain=["KDS minus GK", "KD minus GK", "KS minus GK"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    chart = (
        _zero_rule()
        + base.transform_filter("datum.interval")
        .mark_rule(strokeWidth=2)
        .encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "estimate:Q",
                title="Difference in quantile score (points; below zero is better than GK)",
            )
        )
        + base.mark_text(align="left", dx=8, fontSize=9).encode(x="estimate:Q", text="note:N")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH - 320, height=52 * len(order))
    return chart.properties(
        title=_title(
            text="Winter, cold-day, and holiday contrasts at D+1",
            subtitle=(
                "Difference in mean quantile score at D+1 against GK, scored year, five decision "
                "substations. Line: 95% interval (whole months, or whole days for cold-snap, "
                "cold-night, and holiday days); none where fewer than 15 days. Cold snap: a "
                "November to March day colder than the 10th percentile of such days from 2019-09 "
                "to 2024-06."
            ),
        )
    )


def figure_gate() -> alt.TopLevelMixin:
    """Gate pass rates by reading, arm, and noise form at the decision amplitude."""
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    scenario = gate["decision_scenario"]
    names = {"1": "1 tracking", "2": "2 no cross-talk", "4": "4 forecast error follows"}
    rows = [
        {
            "reading": names[r["reading"]],
            "arm": r["arm"],
            "form": (
                "filter's own noise model" if r["form"] == "ar1" else "real residuals resampled"
            ),
            "rate": r["rate"],
            "low": max(r["rate"] - r["half_width"], 0.0),
            "high": min(r["rate"] + r["half_width"], 1.0),
        }
        for r in gate["rows"]
        if r["scenario"] == scenario
    ]
    table = pl.DataFrame(rows).with_columns(group=pl.col("arm") + ", " + pl.col("form"))
    order = list(names.values())
    base = alt.Chart(table).encode(
        y=alt.Y("reading:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("group:N"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=["GK", "KD", "KS", "KDS"],
                range=[ARM_COLOURS[a] for a in ("GK", "KD", "KS", "KDS")],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        shape=alt.Shape(
            "form:N",
            scale=alt.Scale(
                domain=["filter's own noise model", "real residuals resampled"],
                range=["diamond", "circle"],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    chart = (
        base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1])), x2="high:Q"
        )
        + base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("rate:Q", title="Share of draws that pass the reading")
        )
    ).properties(width=WIDTH - 300, height=90 * len(order))
    return chart.properties(
        title=_title(
            text=f"Known-answer gate: share of draws passing each reading, scenario {scenario}",
            subtitle=(
                "Synthetic streams built on the real PV regressor and real residuals of the five "
                "decision substations over 2024-07 to 2025-06. Dot: share of draws passing. Line: "
                "95% interval. Circle: real residuals resampled by month. Diamond: the filter's "
                "own noise model."
            ),
        )
    )


def figure_tails(*, scored: dict) -> alt.TopLevelMixin:
    """Coverage at the four tail quantiles at D+1, per arm and row set."""
    rows = []
    sets = {
        "all rows": "All rows",
        "midday_apr_sep": "Midday, April to September",
        "winter_evening_peak": "Winter evening peaks",
    }
    for arm in ("GK", "KD", "KS", "KDS"):
        for which, which_label in sets.items():
            for column, level in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
                entry = scored["tails"][arm][which]["D+1"][column]
                rows.append(
                    {
                        "arm": arm,
                        "rows": which_label,
                        "level": f"{level:.2f}",
                        "gap": 100.0 * (entry["coverage"] - level),
                    }
                )
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("level:N", sort=["0.01", "0.05", "0.95", "0.99"], title="Quantile level"),
            x=alt.X(
                "gap:Q", title="Coverage minus nominal level (percentage points; 0 is calibrated)"
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["GK", "KD", "KS", "KDS"],
                    range=[ARM_COLOURS[a] for a in ("GK", "KD", "KS", "KDS")],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            row=alt.Row("rows:N", title=None, sort=list(sets.values())),
        )
        .properties(
            width=WIDTH - 150,
            height=110,
            title=_title(
                text="Tail coverage at D+1 for the four planned arms",
                subtitle=(
                    "Share of observed values at or below each forecast quantile at D+1, minus the "
                    "level itself, in percentage points, scored year, five decision substations. "
                    "The reverse-flow side is the lower tail at midday from April to September; "
                    "the demand side is the upper tail on winter evening peaks."
                ),
            ),
        )
    )


def _site_trace(*, site: str, arm: str) -> tuple[streams.SiteStreams, joint.JointTrace]:
    cache = common.load_cache()
    setup = common.load_setup(cache=cache, site=site)
    data = streams.build_site_streams(cache=cache, setup=setup, fits=common.load_fits(site=site))
    params = (
        joint.gamk_params(hyper=common.gamk_hyper(site=site))
        if arm == "GK"
        else fit.read_params(arm=arm, site=site)
    )
    trace = joint.run_joint_filter(
        residual=data.residual, band=data.band, day=data.day, params=params
    )
    return data, trace


def figure_card(*, site: str) -> alt.TopLevelMixin:
    """One substation's card: the filtered d, and KDS's levels, over the scored year."""
    cache = common.load_cache()
    data, trace = _site_trace(site=site, arm="KDS")
    start = common.local_day_index(cache=cache)[
        int(np.searchsorted(cache.end_times, np.datetime64("2025-07-01T00:00:00", "us")))
    ]
    n_days = int(data.day[np.isfinite(data.residual)].max()) + 1
    frames = []
    names = {joint.D_INDEX: "d (day-level state)", 0: "level, night", 2: "level, midday"}
    for index, label in names.items():
        values = joint.end_of_day_series(trace=trace, values=trace.state[:, index], n_days=n_days)
        frames.append(
            pl.DataFrame(
                {
                    "day": np.arange(start, n_days) - start,
                    "value": values[start:],
                    "state": label,
                }
            ).drop_nulls()
        )
    table = pl.concat(frames)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.3)
        .encode(
            x=alt.X("day:Q", title="Days since 2025-07-01"),
            y=alt.Y("value:Q", title="Filtered state (points of capacity)"),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=list(names.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=180,
            title=_title(
                text=f"Substation {site}: KDS's filtered day-level state and levels",
                subtitle=(
                    "State at the end of each local day over the scored year, in points of "
                    "effective capacity. The fitted constants are in the substation table."
                ),
            ),
        )
    )


def figure_week(*, month: str) -> alt.TopLevelMixin:
    """D+1 forecasts chained over one example week, GK and KDS, at each site."""
    first = EXAMPLE_WEEKS[month]
    panels = []
    for site in SITE_ORDER:
        pieces = []
        for arm in ("GK", "KDS"):
            frame = pl.read_parquet(score.PREDS_DIR / f"{arm}_scored_{site}.parquet")
            frame = frame.filter(
                (pl.col("lead_hours") > 15.0)
                & (pl.col("lead_hours") <= 39.0)
                & (pl.col("valid_time") > first)
                & (pl.col("valid_time") <= first + ONE_WEEK)
            ).with_columns(arm=pl.lit(arm))
            pieces.append(frame)
        table = (
            pl.concat(pieces)
            .with_columns(hours=(pl.col("valid_time") - first).dt.total_minutes() / 60.0 / 24.0)
            .with_columns(day=pl.col("hours").floor().cast(pl.Int32))
            .select("arm", "hours", "day", "q05", "q10", "q50", "q90", "q95", "actual")
        )
        kds = table.filter(pl.col("arm") == "KDS")
        gk = table.filter(pl.col("arm") == "GK")
        band90 = (
            alt.Chart(kds)
            .mark_area(opacity=0.18, color=ocf.BRAND_ORANGE)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hours:Q",
                    title="Days since the start of the week",
                    axis=alt.Axis(values=[0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                y=alt.Y("q05:Q", title="Net power (% of capacity)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                detail="day:N",
            )
        )
        band80 = (
            alt.Chart(kds)
            .mark_area(opacity=0.28, color=ocf.BRAND_ORANGE)
            .encode(x="hours:Q", y="q10:Q", y2="q90:Q", detail="day:N")  # ty: ignore[unresolved-attribute]
        )
        median = (
            alt.Chart(kds)
            .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.6)
            .encode(x="hours:Q", y="q50:Q", detail="day:N")  # ty: ignore[unresolved-attribute]
        )
        gk_line = (
            alt.Chart(gk)
            .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.2, strokeDash=[4, 2])
            .encode(x="hours:Q", y="q50:Q", detail="day:N")  # ty: ignore[unresolved-attribute]
        )
        actual = (
            alt.Chart(kds)  # ty: ignore[unresolved-attribute]
            .mark_line(color=DARK, strokeWidth=1.0)
            .encode(x="hours:Q", y="actual:Q", detail="day:N")
        )
        panels.append(
            (band90 + band80 + gk_line + median + actual).properties(
                width=WIDTH - 90, height=100, title=site
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text=f"Example week in {month}: KDS and GK day-ahead forecasts against observation",
            subtitle=(
                "Day-ahead (D+1) forecasts chained from each 09:00 UTC origin over the week "
                f"starting {first:%d %B %Y}, fixed in advance as the second Monday of the month. "
                "Black: observed. Orange: KDS median, with 80% and 90% bands. "
                "Dashed blue: GK median. A gap is a day with no scored row (no reading, or inside "
                "a switching event)."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()
    scored, replicate = results(period="scored"), results(period="replicate")
    figures = {
        "headline": lambda: _save(
            figure_headline(scored=scored, replicate=replicate), name="figure_01_headline"
        ),
        "by_lead": lambda: _save(figure_by_lead(scored=scored), name="figure_02_by_lead"),
        "subsets": lambda: _save(figure_subsets(scored=scored), name="figure_03_winter_and_events"),
        "gate": lambda: _save(figure_gate(), name="figure_04_gate"),
        "tails": lambda: _save(figure_tails(scored=scored), name="figure_05_tails"),
        **{
            f"card_{site}": (
                lambda site=site: _save(figure_card(site=site), name=f"figure_06_card_{site}")
            )
            for site in common.DECISION_SITES
        },
        **{
            f"week_{month.lower()}": (
                lambda month=month: _save(
                    figure_week(month=month), name=f"figure_07_week_{month.lower()}"
                )
            )
            for month in EXAMPLE_WEEKS
        },
    }
    for name, draw in figures.items():
        if args.only is None or name in args.only:
            draw()


if __name__ == "__main__":
    main()
