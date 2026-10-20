"""Draw the figures of experiment 118's write-up from the stored results and prediction files.

Every figure carries its own scope, unit, and direction in its title and subtitle. SVGs go to the
write-up's folder, PNGs to a scratch folder for the render check. Only D-labels appear.

    uv run python ml-experiments/00118-*/e00118_charts.py [--only headline gate ...]
"""

import argparse
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00118_common as common
import e00118_forecast as fc
import e00118_noise as noise_io
import e00118_score as score
import e00118_streams as streams
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_effects_filter import HEATING, HOLIDAY_INDEX, PROFILE, PV_INDEX

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
    "KEZ": "KEZ (14 weights plus anomaly)",
    "KE": "KE (14 weights, no anomaly)",
    "KEZ-Q0": "KEZ-Q0 (only the level moves)",
    "KEZ-QI": "KEZ-QI (one shared step variance)",
    "KEZ-DT": "KEZ-DT (holiday days treated apart)",
    "KEZ-1s": "KEZ-1s (first run, one-step likelihood)",
    "KEZ-drop-pv": "KEZ without the PV weight",
    "bg2": "BG2 (production recipe)",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "GK": ocf.DATA_BLUE,
    "KEZ": ocf.BRAND_ORANGE,
    "KE": ocf.DATA_SKY,
    "KEZ-Q0": ocf.DATA_PURPLE,
    "KEZ-QI": ocf.DATA_GREEN,
    "KEZ-DT": ocf.BRAND_ORANGE_LIGHT,
    "KEZ-1s": ocf.DATA_PURPLE_LIGHT,
    "KEZ-drop-pv": ocf.DATA_GREEN_LIGHT,
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
    "holiday": "Holiday days",
    "non_holiday": "Other days",
}
WEIGHT_NAMES: Final[tuple[str, ...]] = (
    "level, night",
    "level, morning",
    "level, midday",
    "level, evening",
    "profile, weekday",
    "profile, Saturday",
    "profile, Sunday",
    "profile, bank holiday",
    "heating, night",
    "heating, morning",
    "heating, midday",
    "heating, evening",
    "PV capacity",
    "holiday effects",
)
EXAMPLE_WEEKS: Final[dict[str, datetime]] = {
    "July": datetime(2025, 7, 14, tzinfo=UTC),
    "October": datetime(2025, 10, 13, tzinfo=UTC),
    "January": datetime(2026, 1, 12, tzinfo=UTC),
    "April": datetime(2026, 4, 13, tzinfo=UTC),
}
SITE_ORDER: Final[tuple[str, ...]] = (*common.DECISION_SITES, "D9")


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
    """The five planned contrasts and the holiday-day contrasts at D+1, in both periods."""
    rows = []
    entries = (
        ("C1 KEZ minus GK", "planned", "all rows"),
        ("C2 KE minus GK", "planned", "all rows"),
        ("C3 KEZ minus KE", "planned", "all rows"),
        ("C4 KEZ minus KEZ-Q0", "planned", "all rows"),
        ("C5 KEZ minus KEZ-QI", "planned", "all rows"),
        ("KEZ-DT minus KEZ", "exploratory", "holiday"),
        ("KEZ-DT minus KEZ", "exploratory", "non_holiday"),
        ("KEZ-drop-pv minus KEZ", "exploratory", "all rows"),
    )
    for period, data in (("scored year", scored), ("replicate months", replicate)):
        for name, group, subset in entries:
            entry = data[group][name][subset].get("D+1")
            if entry is None or entry["lower"] is None:
                continue
            label = name if subset == "all rows" else f"{name} ({SUBSET_LABELS[subset].lower()})"
            rows.append(
                {
                    "contrast": label,
                    "period": period,
                    "estimate": entry["estimate"],
                    "low": entry["lower"],
                    "high": entry["upper"],
                    "level": "99% interval" if group == "planned" else "95% interval",
                }
            )
    table = pl.DataFrame(rows)
    order = list(dict.fromkeys(table["contrast"].to_list()))
    base = alt.Chart(table).encode(  # ty: ignore[unresolved-attribute]
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
            text="The effects filter does not beat GAM-K, and holding the level on holidays hurts",
            subtitle=(
                "Dot: difference in mean quantile score at day D+1 (leads 15 to 39 hours) over "
                "the five decision substations, rows outside the switching events. Line: interval "
                "from resampling whole calendar months (C1 to C5, 99%) or whole days (holiday "
                "days, 95%), paired across arms. Dashed line: no difference."
            ),
        )
    )


def figure_by_lead(*, scored: dict) -> alt.TopLevelMixin:
    """Absolute mean quantile score by lead band, headline arms and references."""
    leads = ["0-24 h", "24-48 h", "48-144 h", "144-168 h"]
    rows = []
    for arm, label in ARM_LABELS.items():
        by_subset = scored["absolute"].get(arm)
        if by_subset is None:
            continue
        rows.extend(
            {"arm": label, "lead": lead, "qs": by_subset["all rows"][lead]["qs"]} for lead in leads
        )
    for name in ("bg2", "climatology", "manual_heuristic"):
        shared = scored["reference_scores"][name][name]
        rows.extend({"arm": ARM_LABELS[name], "lead": lead, "qs": shared[lead]} for lead in leads)
    table = pl.DataFrame(rows)
    domain = [ARM_LABELS[a] for a in ARM_LABELS if ARM_LABELS[a] in set(table["arm"].to_list())]
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
                    domain=domain,
                    range=[ARM_COLOURS[a] for a in ARM_LABELS if ARM_LABELS[a] in domain],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=280,
            title=_title(
                text="BG2 matches the arms at D+1 and is ahead beyond 24 hours",
                subtitle=(
                    "Mean quantile score by lead band over the five decision substations, scored "
                    "year, rows outside the switching events. The arms of this experiment are on "
                    "shared rows. BG2, climatology, and the manual heuristic are on the rows they "
                    "share with KEZ (the climatology and manual heuristic are far above the rest "
                    "at every lead; BG2 is within about 0.1)."
                ),
            ),
        )
    )


def figure_subsets(*, scored: dict) -> alt.TopLevelMixin:
    """KEZ minus GK, and KEZ without PV minus GK, on the winter and event subsets at D+1."""
    rows = []
    for name, group, label in (
        ("C1 KEZ minus GK", "planned", "KEZ minus GK"),
        ("KEZ-drop-pv minus GK", "exploratory", "KEZ without the PV weight minus GK"),
        ("KEZ-DT minus GK", "exploratory", "KEZ-DT minus GK"),
    ):
        for subset, subset_label in SUBSET_LABELS.items():
            entry = scored[group][name][subset].get("D+1")
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
    base = alt.Chart(table).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("subset:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("contrast:N"),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(
                domain=[
                    "KEZ minus GK",
                    "KEZ without the PV weight minus GK",
                    "KEZ-DT minus GK",
                ],
                range=[ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE],
            ),
            legend=alt.Legend(title=None, orient="top", labelLimit=320, columns=2),
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
    ).properties(width=WIDTH - 320, height=44 * len(order))
    return chart.properties(
        title=_title(
            text="No arm beats GAM-K in winter or on cold days, and KEZ-DT loses on holidays",
            subtitle=(
                "Difference in mean quantile score at D+1 against GK, scored year, five decision "
                "substations. Line: 95% interval (whole months, or whole days for cold-snap and "
                "holiday days); none where fewer than 15 days. Cold snap: a November to March day "
                "colder than the 10th percentile of such days from 2019-09 to 2024-06."
            ),
        )
    )


def figure_gate() -> alt.TopLevelMixin:
    """Gate pass rates by reading, scenario, noise form, and regressor condition."""
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    names = {
        "1": "1 PV tracking",
        "2": "2 heating tracking",
        "3": "3 no false movement",
        "4": "4 no cross-talk",
        "5": "5 skill follows",
    }
    rows = [
        {
            "reading": f"{names[r['reading']]} ({r['scenario']})",
            "form": (
                "filter's own noise model" if r["form"] == "ar1" else "real residuals resampled"
            ),
            "condition": r["condition"],
            "rate": r["rate"],
            "low": max(r["rate"] - r["half_width"], 0.0),
            "high": min(r["rate"] + r["half_width"], 1.0),
        }
        for r in gate["rows"]
        if r["scenario"] in ("S0", "S1", "S2", "S3")
    ]
    table = pl.DataFrame(rows).with_columns(group=pl.col("form") + ", " + pl.col("condition"))
    order = sorted(set(table["reading"].to_list()))
    base = alt.Chart(table).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("reading:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("group:N"),
        color=alt.Color(
            "form:N",
            scale=alt.Scale(
                domain=["filter's own noise model", "real residuals resampled"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        shape=alt.Shape(
            "condition:N",
            scale=alt.Scale(domain=["exact", "eiv"], range=["circle", "diamond"]),
            legend=alt.Legend(title="regressors (eiv: weather error)", orient="top"),
        ),
    )
    bar = (
        alt.Chart(pl.DataFrame({"x": [0.8]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=DARK, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        bar
        + base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1])), x2="high:Q"
        )
        + base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("rate:Q", title="Share of 100 draws that pass the reading (bar: 0.80)")
        )
    ).properties(width=WIDTH - 300, height=26 * len(order))
    return chart.properties(
        title=_title(
            text="The filter finds injected changes in its own noise, but not in real residuals",
            subtitle=(
                "Known-answer gate on synthetic streams built on the real regressors of the five "
                "decision substations over 2024-07 to 2025-06: 20 draws per substation. Dot: share "
                "of draws passing. Line: 95% interval. Dashed line: the pass bar."
            ),
        )
    )


def figure_noise() -> alt.TopLevelMixin:
    """The fitted relative step per year of each weight, per site."""
    stored = noise_io.read_noise(arm="KEZ")
    rows = []
    for site, fit in stored["sites"].items():
        for index, name in enumerate(WEIGHT_NAMES):
            rows.append(
                {
                    "weight": name,
                    "site": site,
                    "step": max(fit["relative_step_per_year"][index], 1e-3),
                    "by_prior": fit["set_by_prior"][index],
                }
            )
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=55)
        .encode(
            y=alt.Y("weight:N", sort=list(WEIGHT_NAMES), title=None),
            x=alt.X(
                "step:Q",
                scale=alt.Scale(type="log"),
                title="Fitted step over a year (share of the effect; points for a level)",
            ),
            color=alt.Color(
                "site:N",
                sort=list(SITE_ORDER),
                scale=alt.Scale(
                    domain=list(SITE_ORDER),
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        GREY,
                    ],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            shape=alt.Shape(
                "by_prior:N",
                scale=alt.Scale(domain=[False, True], range=["circle", "square"]),
                legend=alt.Legend(title="q set by the prior", orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 200,
            height=360,
            title=_title(
                text="Fitted weight steps differ by 100 times between weights and substations",
                subtitle=(
                    "Noise matrix of KEZ fitted on the tuning months (the odd months of 2024-07 to "
                    "2025-06) by the day-ahead likelihood: square root of 365 times each weight's "
                    "daily variance, as a share of the effect (a level is in points). Log scale."
                ),
            ),
        )
    )


def figure_tails(*, scored: dict) -> alt.TopLevelMixin:
    """Coverage at the four tail quantiles at D+1, per arm."""
    rows = []
    for arm in ("GK", "KEZ", "KE", "KEZ-Q0", "KEZ-QI", "KEZ-DT"):
        for column, level in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
            entry = scored["tails"][arm]["D+1"][column]
            rows.append(
                {
                    "arm": ARM_LABELS[arm],
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
                    domain=[
                        ARM_LABELS[a] for a in ("GK", "KEZ", "KE", "KEZ-Q0", "KEZ-QI", "KEZ-DT")
                    ],
                    range=[
                        ARM_COLOURS[a] for a in ("GK", "KEZ", "KE", "KEZ-Q0", "KEZ-QI", "KEZ-DT")
                    ],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH - 150,
            height=170,
            title=_title(
                text="The tails are about as well calibrated as GAM-K's",
                subtitle=(
                    "Share of observed values at or below each forecast quantile at D+1, minus the "
                    "level itself, in percentage points, scored year, five decision substations."
                ),
            ),
        )
    )


def _kez_traces() -> dict[str, tuple[streams.SiteStreams, fc.ArmFilter]]:
    cache = common.load_cache()
    out = {}
    for site in SITE_ORDER:
        setup = common.load_setup(cache=cache, site=site)
        data = streams.build_site_streams(
            cache=cache, setup=setup, fits=common.load_fits(site=site)
        )
        out[site] = (data, fc.build_filters(cache=cache, data=data)["KEZ"])
    return out


def figure_weights(*, traces: dict) -> alt.TopLevelMixin:
    """The filtered PV, heating, profile, and holiday weights over the scored year, per site."""
    cache = common.load_cache()
    rows = []
    start = np.datetime64("2025-07-01T00:00", "us")
    for site, (data, filtered) in traces.items():
        trace = filtered.trace
        times = cache.end_times[trace.positions]
        keep = times > start
        days = (times[keep] - start) / np.timedelta64(1, "D")
        for label, index in (
            ("PV capacity weight", PV_INDEX),
            ("heating weight (night)", HEATING.start),
            ("heating weight (midday)", HEATING.start + 2),
            ("weekday profile weight", PROFILE.start),
            ("holiday-effects weight", HOLIDAY_INDEX),
        ):
            weight = 1.0 + trace.state[keep, index] / data.scale[index]
            sd = trace.sd[keep, index] / data.scale[index]
            step = max(len(days) // 365, 1)
            rows.extend(
                {
                    "site": site,
                    "weight": label,
                    "day": float(days[k]),
                    "value": float(weight[k]),
                    "low": float(weight[k] - sd[k]),
                    "high": float(weight[k] + sd[k]),
                }
                for k in range(0, len(days), step)
            )
    table = pl.DataFrame(rows).with_columns(
        pl.col("value").clip(-1.0, 4.0),
        pl.col("low").clip(-1.0, 4.0),
        pl.col("high").clip(-1.0, 4.0),
    )
    base = alt.Chart(table).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "day:Q", title="Days since 1 July 2025", axis=alt.Axis(values=[0, 92, 184, 275, 365])
        ),
        color=alt.Color(
            "weight:N",
            scale=alt.Scale(
                domain=[
                    "PV capacity weight",
                    "heating weight (night)",
                    "heating weight (midday)",
                    "weekday profile weight",
                    "holiday-effects weight",
                ],
                range=[
                    ocf.BRAND_ORANGE,
                    ocf.DATA_BLUE,
                    ocf.DATA_SKY,
                    ocf.DATA_PURPLE,
                    ocf.DATA_GREEN,
                ],
            ),
            legend=alt.Legend(title=None, orient="top", columns=3),
        ),
    )
    chart = (
        base.mark_line(strokeWidth=1.2).encode(
            y=alt.Y(
                "value:Q",
                title="Filtered weight (1 is the fitted effect)",
                scale=alt.Scale(domain=[-1, 4]),
            )
        )
    ).properties(width=WIDTH - 80, height=110)
    return alt.vconcat(
        *[
            chart.transform_filter(alt.datum.site == site).properties(title=site)
            for site in SITE_ORDER
        ]
    ).properties(
        title=_title(
            text="Most weights stay near 1; midday heating and holiday weights swing at some sites",
            subtitle=(
                "KEZ's filtered weights at each substation over the scored year, from the state "
                "after every half-hourly reading, clipped to -1 to 4. A weight of 1 is the fitted "
                "effect. Weights are not interpretable (see the gate)."
            ),
        )
    )


def figure_week(*, month: str) -> alt.TopLevelMixin:
    """D+1 forecasts chained over one example week, GK and KEZ, at each site."""
    first = EXAMPLE_WEEKS[month]
    panels = []
    for site in SITE_ORDER:
        pieces = []
        for arm in ("GK", "KEZ"):
            frame = pl.read_parquet(score.PREDS_DIR / f"{arm}_scored_{site}.parquet")
            frame = frame.filter(
                (pl.col("lead_hours") > 15.0)
                & (pl.col("lead_hours") <= 39.0)
                & (pl.col("valid_time") > first)
                & (
                    pl.col("valid_time")
                    <= first + (datetime(2025, 1, 8, tzinfo=UTC) - datetime(2025, 1, 1, tzinfo=UTC))
                )
            ).with_columns(arm=pl.lit(arm))
            pieces.append(frame)
        table = (
            pl.concat(pieces)
            .with_columns(hours=(pl.col("valid_time") - first).dt.total_minutes() / 60.0 / 24.0)
            .with_columns(day=pl.col("hours").floor().cast(pl.Int32))
            .select("arm", "hours", "day", "q05", "q10", "q50", "q90", "q95", "actual")
        )
        kez = table.filter(pl.col("arm") == "KEZ")
        gk = table.filter(pl.col("arm") == "GK")
        band90 = (
            alt.Chart(kez)
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
            alt.Chart(kez)
            .mark_area(opacity=0.28, color=ocf.BRAND_ORANGE)
            .encode(  # ty: ignore[unresolved-attribute]
                x="hours:Q", y="q10:Q", y2="q90:Q", detail="day:N"
            )
        )
        median = (
            alt.Chart(kez)
            .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.6)
            .encode(  # ty: ignore[unresolved-attribute]
                x="hours:Q", y="q50:Q", detail="day:N"
            )
        )
        gk_line = (
            alt.Chart(gk)
            .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.2, strokeDash=[4, 2])
            .encode(  # ty: ignore[unresolved-attribute]
                x="hours:Q", y="q50:Q", detail="day:N"
            )
        )
        actual = (
            alt.Chart(kez)
            .mark_line(color=DARK, strokeWidth=1.0)
            .encode(x="hours:Q", y="actual:Q", detail="day:N")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            (band90 + band80 + gk_line + median + actual).properties(
                width=WIDTH - 90, height=100, title=site
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text=f"Example week in {month}: KEZ and GK day-ahead forecasts against observation",
            subtitle=(
                "Day-ahead (D+1) forecasts chained from each 09:00 UTC origin over the week "
                f"starting {first:%d %B %Y}, fixed in advance as the second Monday of the month. "
                "Black: observed. Orange: KEZ median, with 80% and 90% bands. "
                "Dashed blue: GK median."
            ),
        )
    )


def figure_cold(*, scored: dict) -> alt.TopLevelMixin:
    """Mean quantile score of GK and KEZ on the cold-snap days of the scored winter, per site."""
    rows = []
    for site in SITE_ORDER:
        cold_days = scored["cold_snap_days_in_rows"][site]
        for arm in ("GK", "KEZ"):
            frame = pl.read_parquet(score.PREDS_DIR / f"{arm}_scored_{site}.parquet")
            frame = frame.filter(
                ~pl.col("switching_mask")
                & (pl.col("lead_hours") > 15.0)
                & (pl.col("lead_hours") <= 39.0)
            )
            cold = score.cold_snap_dates(site=site)[1]
            local = pl.col("valid_time").dt.convert_time_zone(score.LOCAL_ZONE) - pl.duration(
                minutes=30
            )
            frame = frame.with_columns(date=local.dt.date().cast(pl.String)).filter(
                pl.col("date").is_in(list(cold))
            )
            if frame.is_empty():
                continue
            rows.append(
                {
                    "site": f"{site} ({cold_days} cold days)",
                    "arm": ARM_LABELS[arm],
                    "qs": float(score.quantile_score(frame=frame).mean()),
                }
            )
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("site:N", title=None),
            x=alt.X(
                "qs:Q",
                title="Mean quantile score at D+1 on cold-snap days (points; smaller is better)",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS["GK"], ARM_LABELS["KEZ"]],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=330),
            ),
        )
        .properties(
            width=WIDTH - 220,
            height=170,
            title=_title(
                text="On cold-snap days there are too few days to tell the arms apart",
                subtitle=(
                    "Mean quantile score of GK and KEZ at D+1 on the cold-snap days of the scored "
                    "winter at each substation. Every substation has under 15 such days, so no "
                    "interval is given. Rule: a November to March day colder than the 10th "
                    "percentile of such days from 2019-09 to 2024-06."
                ),
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
        "noise": lambda: _save(figure_noise(), name="figure_05_noise"),
        "tails": lambda: _save(figure_tails(scored=scored), name="figure_06_tails"),
        "weights": lambda: _save(figure_weights(traces=_kez_traces()), name="figure_07_weights"),
        "cold": lambda: _save(figure_cold(scored=scored), name="figure_08_cold_snaps"),
        **{
            f"week_{month.lower()}": (
                lambda month=month: _save(
                    figure_week(month=month), name=f"figure_09_week_{month.lower()}"
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
