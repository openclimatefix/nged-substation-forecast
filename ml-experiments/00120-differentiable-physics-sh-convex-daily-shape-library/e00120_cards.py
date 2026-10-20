"""Draw the provisional substation cards of experiment 120 into the docs folder.

**Provisional: fitted to past weather on a screening run.** One card per decision substation
(D5, D10, D11, D14, D16; D-labels, so dates may be shown), from the scored-year headline fit, and
one cross-substation figure. D15 and D18 appear on no card: D18 sits in the fit but is a generator
series, and its figures belong under the neutral label G-c in a separate figure, never beside a
D-label.

Each card holds, as far as the model has the piece:

- the daily profile per day type, with the library and the deviation drawn separately;
- the substation's weights on the shared shapes and the size of its deviation;
- the heating response per hour band, and the holiday, Christmas, and lighting effects;
- the pinned PV capacity;
- example forecasts on four weeks fixed by a calendar rule before any score was read (the first
  Monday on or after 15 July 2025, 15 October 2025, 15 January 2026, and 15 April 2026), with the
  80% and 90% bands of SH and the median of N against the observed series;
- a cold-winter-night panel: the median over cold winter nights of the actual power and of the
  forecast medians (rule: a night from 23:00 to 07:00 local time, November to March, whose mean
  6-hour-smoothed ERA5 temperature at the site is below 2 °C; "too few nights" under 5).

    uv run python ml-experiments/00120-*/e00120_cards.py
"""

import textwrap
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00120_common as common
import e00120_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = Path("/tmp/e00120_png")
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
SHAPE_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    "#808080",
]
DAY_TYPE_NAMES: Final[list[str]] = ["Weekday", "Saturday", "Sunday", "Bank holiday"]
FORECAST_WEEK_STARTS: Final[tuple[date, ...]] = (
    date(2025, 7, 15),
    date(2025, 10, 15),
    date(2026, 1, 15),
    date(2026, 4, 15),
)
"""The first Monday on or after each date is the week's start (the rule fixed in the plan)."""
COLD_NIGHT_C: Final[float] = 2.0
MIN_COLD_NIGHTS: Final[int] = 5
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
PROVISIONAL: Final[str] = "Provisional: fitted to past weather on a screening run."


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=13,
        subtitleFontSize=10,
    )


def first_monday_on_or_after(*, day: date) -> date:
    """Return the first Monday on or after a date."""
    return day + timedelta(days=(7 - day.weekday()) % 7)


def _headline() -> tuple[dict[str, np.ndarray], list[str]]:
    fits = report.fit_files(label="scored", arm="sh")
    arrays = min(fits, key=lambda fit: fit[2]["objective"])[1]
    return arrays, [str(label) for label in arrays["labels"]]


def profile_panel(*, arrays: dict[str, np.ndarray], position: int) -> alt.ChartType:
    """The daily profile per day type: library, deviation, and their sum, centred in each band."""
    library = arrays["weights"][position] @ arrays["shapes"]
    deviation = arrays["deviations"][position]
    rows = [
        {
            "day type": name,
            "hour": half_hour / 2.0,
            "part": part,
            "value": float(values[d * 48 + half_hour]),
        }
        for part, values in (
            ("Library", library),
            ("Deviation", deviation),
            ("Library + deviation", library + deviation),
        )
        for d, name in enumerate(DAY_TYPE_NAMES)
        for half_hour in range(48)
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X(
                "hour:Q",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
                title="Local half-hour (hour of day)",
            ),
            y=alt.Y("value:Q", title="Points of capacity"),
            color=alt.Color(
                "part:N",
                scale=alt.Scale(
                    domain=["Library", "Deviation", "Library + deviation"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#404040"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH // 2 - 40, height=90)
    )
    return chart.facet(column=alt.Column("day type:N", sort=DAY_TYPE_NAMES, title=None), columns=2)


def mix_panel(*, arrays: dict[str, np.ndarray], position: int) -> alt.ChartType:
    """The substation's share of weight on each shape, and the deviation's share of the profile."""
    weights = arrays["weights"][position]
    shares = weights / weights.sum()
    data = pl.DataFrame(
        {"shape": [f"Shape {k + 1}" for k in range(len(shares))], "share": shares.tolist()}
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("share:Q", stack="zero", title="Share of the substation's weight"),
            color=alt.Color(
                "shape:N",
                scale=alt.Scale(domain=data["shape"].to_list(), range=SHAPE_COLOURS[: len(shares)]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=28)
    )


def heating_panel(*, arrays: dict[str, np.ndarray], label: str) -> alt.ChartType:
    """The heating slope per hour band, summed over the four smoothing windows."""
    coefficients = arrays[f"{label}__heating"].reshape(-1, 4)
    totals = coefficients.sum(axis=0)
    data = pl.DataFrame({"band": [f"Band {i + 1}" for i in range(len(totals))], "slope": totals})
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.BRAND_ORANGE)
        .encode(
            x=alt.X("band:N", title="Hour band (4 bands per day)", sort=data["band"].to_list()),
            y=alt.Y("slope:Q", title="Points per 10 °C below the threshold"),
        )
        .properties(width=WIDTH // 2 - 40, height=90)
    )


def calendar_panel(*, arrays: dict[str, np.ndarray], label: str) -> alt.ChartType:
    """Holiday effects per period and band, as a heat map of coefficients in points of capacity."""
    holiday = arrays[f"{label}__holiday"].reshape(len(HOLIDAY_PERIODS), -1)
    rows = [
        {"period": str(name), "band": f"Band {b + 1}", "effect": float(holiday[p, b])}
        for p, name in enumerate(HOLIDAY_PERIODS)
        for b in range(holiday.shape[1])
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("band:N", title=None),
            y=alt.Y("period:N", sort=[str(name) for name in HOLIDAY_PERIODS], title=None),
            color=alt.Color(
                "effect:Q",
                scale=alt.Scale(scheme="blueorange", domainMid=0),
                legend=alt.Legend(orient="right", title="Points"),
            ),
        )
        .properties(width=WIDTH // 2 - 40, height=max(90, 7 * len(HOLIDAY_PERIODS)))
    )


def weight_by_winter_panel(*, site: str) -> alt.ChartType:
    """The substation's weight share on the step shape at each cut (one per fit window)."""
    table = report.weight_by_year_table().filter(pl.col("site") == site)
    names = {"tuning": "2023-24", "replicate": "2024-25", "scored": "2025-26"}
    data = table.with_columns(
        winter=pl.col("cut").replace_strict(names), share=pl.col("weight share")
    ).filter(pl.col("share").is_not_null())
    if data.height == 0:
        return (
            alt.Chart(pl.DataFrame({"text": ["Weight by winter: cannot tell (no fit)"]}))  # ty: ignore[unresolved-attribute]
            .mark_text(align="left")
            .encode(text="text:N")
            .properties(width=WIDTH, height=20)
        )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("winter:N", sort=list(names.values()), title="Winter of the fit window"),
            y=alt.Y("share:Q", title="Share of weight on the step shape"),
        )
        .properties(width=WIDTH - 20, height=80)
    )


def forecast_weeks(*, site: str) -> alt.ChartType:
    """Four fixed weeks: SH's median and 80% and 90% bands, N's median, and the observed series."""
    sh = report.seeded(label="scored", arm="sh", pooled=False).filter(pl.col("site_name") == site)
    n = report.seeded(label="scored", arm="n", pooled=False).filter(pl.col("site_name") == site)
    starts = [first_monday_on_or_after(day=day) for day in FORECAST_WEEK_STARTS]
    frames = []
    for start in starts:
        stop = start + timedelta(days=7)
        for name, frame in (("SH", sh), ("N", n)):
            part = frame.filter(
                (pl.col("valid_time").dt.date() >= start) & (pl.col("valid_time").dt.date() < stop)
            ).with_columns(
                week=pl.lit(f"Week of {start:%d %b %Y}"),
                model=pl.lit(name),
                hours=(
                    pl.col("valid_time") - pl.lit(start).cast(pl.Datetime("us", "UTC"))
                ).dt.total_minutes()
                / 60.0,
            )
            frames.append(
                part.select("week", "model", "hours", "q05", "q10", "q50", "q90", "q95", "actual")
            )
    x = alt.X(
        "hours:Q",
        axis=alt.Axis(values=[0, 48, 96, 144], title="Hours from Monday 00:00 UTC"),
        scale=alt.Scale(domain=[0, 168]),
    )
    panels = []
    for label in [f"Week of {start:%d %b %Y}" for start in starts]:
        part = pl.concat(frames).filter(pl.col("week") == label)
        sh_part = part.filter(pl.col("model") == "SH")
        n_part = part.filter(pl.col("model") == "N")
        band90 = (
            alt.Chart(sh_part)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.18, color=ocf.DATA_BLUE)
            .encode(x=x, y=alt.Y("q05:Q", title=label), y2="q95:Q")
        )
        band80 = (
            alt.Chart(sh_part)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.28, color=ocf.DATA_BLUE)
            .encode(x=x, y="q10:Q", y2="q90:Q")
        )
        median = (
            alt.Chart(sh_part)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.5)
            .encode(x=x, y="q50:Q")
        )
        fleet = (
            alt.Chart(n_part)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.DATA_PURPLE, strokeDash=[4, 3], strokeWidth=1.2)
            .encode(x=x, y="q50:Q")
        )
        actual = (
            alt.Chart(sh_part)  # ty: ignore[unresolved-attribute]
            .mark_line(color="#202020", strokeWidth=1.2)
            .encode(x=x, y="actual:Q")
        )
        panels.append(
            (band90 + band80 + median + fleet + actual).properties(width=WIDTH - 20, height=80)
        )
    return alt.vconcat(*panels)


def cold_night_data(*, site: str) -> tuple[pl.DataFrame, int]:
    """Return the cold-winter-night rows of a site with the hours since 23:00, and the night count.

    Args:
        site: A decision substation.

    Returns:
        The rows of SH, N, and G on the cold nights, and the number of cold nights.
    """
    cache = common.load_cache()
    features = cache.sites[site]
    temperature = pl.DataFrame(
        {
            "valid_time": pl.Series(cache.end_times).dt.replace_time_zone("UTC"),
            "temperature": features.smoothed_temperature_c[:, 0],
        }
    )
    frames = []
    for arm, name in (("sh", "SH"), ("n", "N"), ("g", "G")):
        frame = (
            report.seeded(label="scored", arm=arm, pooled=False)
            .filter(pl.col("site_name") == site)
            .join(temperature, on="valid_time", how="left")
        )
        local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
        minutes = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute()
        is_night = (minutes >= 23 * 60) | (minutes <= 7 * 60)
        night = (local - pl.duration(hours=7, minutes=30)).dt.date()
        frame = frame.filter(is_night & local.dt.month().is_in(WINTER_MONTHS)).with_columns(
            night=night,
            hours=((minutes - 23 * 60) % (24 * 60)) / 60.0,
            model=pl.lit(name),
        )
        frames.append(frame)
    data = pl.concat(
        [
            f.select("night", "hours", "model", "q10", "q50", "q90", "actual", "temperature")
            for f in frames
        ]
    )
    cold = (
        data.filter(pl.col("model") == "SH")
        .group_by("night")
        .agg(pl.col("temperature").mean())
        .filter(pl.col("temperature") < COLD_NIGHT_C)
    )
    return data.join(cold.select("night"), on="night", how="inner"), cold.height


def cold_night_panel(*, site: str) -> alt.ChartType:
    """The median over cold winter nights of the actual power and the forecast medians."""
    data, n_nights = cold_night_data(site=site)
    if n_nights < MIN_COLD_NIGHTS:
        return (
            alt.Chart(pl.DataFrame({"text": [f"Cold winter nights: too few nights ({n_nights})"]}))  # ty: ignore[unresolved-attribute]
            .mark_text(align="left")
            .encode(text="text:N")
            .properties(width=WIDTH, height=20)
        )
    curves = (
        data.group_by("hours", "model")
        .agg(
            pl.col("actual").median(),
            pl.col("q50").median(),
            pl.col("q10").median(),
            pl.col("q90").median(),
        )
        .sort("hours")
    )
    x = alt.X(
        "hours:Q",
        axis=alt.Axis(values=[0, 2, 4, 6, 8], title="Hours after 23:00 local time"),
        scale=alt.Scale(domain=[0, 8]),
    )
    sh = curves.filter(pl.col("model") == "SH")
    band = (
        alt.Chart(sh)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.25, color=ocf.DATA_BLUE)
        .encode(x=x, y="q10:Q", y2="q90:Q")
    )
    forecasts = (
        alt.Chart(curves)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.6)
        .encode(
            x=x,
            y=alt.Y("q50:Q", title="Points of capacity"),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=["SH", "N", "G"],
                    range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title="Forecast median"),
            ),
        )
    )
    actual = (
        alt.Chart(sh)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#202020", strokeWidth=2)
        .encode(x=x, y="actual:Q")
    )
    return (band + forecasts + actual).properties(width=WIDTH - 20, height=130)


def card(*, site: str, arrays: dict[str, np.ndarray], labels: list[str]) -> alt.ChartType:
    """Return one substation's card."""
    position = labels.index(site)
    capacity = float(arrays["pinned_points"][position])
    deviation_share = float(
        np.linalg.norm(arrays["deviations"][position]) ** 2
        / (
            np.linalg.norm(arrays["weights"][position] @ arrays["shapes"]) ** 2
            + np.linalg.norm(arrays["deviations"][position]) ** 2
        )
    )
    n_nights = cold_night_data(site=site)[1]
    panels = alt.vconcat(
        profile_panel(arrays=arrays, position=position),
        mix_panel(arrays=arrays, position=position),
        alt.hconcat(
            heating_panel(arrays=arrays, label=site), calendar_panel(arrays=arrays, label=site)
        ),
        weight_by_winter_panel(site=site),
        forecast_weeks(site=site),
        cold_night_panel(site=site),
    )
    return panels.properties(
        title=_title(
            text=f"{site}: what the fitted library says",
            subtitle=(
                f"{PROVISIONAL} Pinned PV capacity {capacity:.1f} points of capacity (the "
                f"real-time vintage, not a measurement). Deviation holds {deviation_share:.0%} of "
                f"the profile's energy. Weights and shapes are identified only up to the rotation "
                f"the data fix. Forecast panels: blue band 80% and 90%, dashed purple N's median, "
                f"black observed. Cold winter nights in the scored year: {n_nights}."
            ),
        )
    )


def cross_figure(*, arrays: dict[str, np.ndarray], labels: list[str]) -> alt.ChartType:
    """Compare capacity, heating slope, shape share, and deviation share across the substations."""
    rows = []
    for site in SITES:
        position = labels.index(site)
        weights = arrays["weights"][position]
        shares = weights / weights.sum()
        library_energy = float(np.linalg.norm(weights @ arrays["shapes"]) ** 2)
        deviation_energy = float(np.linalg.norm(arrays["deviations"][position]) ** 2)
        rows += [
            {
                "site": site,
                "property": "Pinned PV capacity (points)",
                "value": float(arrays["pinned_points"][position]),
            },
            {
                "site": site,
                "property": "Heating slope (points per 10 °C, bands summed)",
                "value": float(arrays[f"{site}__heating"].sum()),
            },
            {
                "site": site,
                "property": "Share of weight on the largest shape",
                "value": float(shares.max()),
            },
            {
                "site": site,
                "property": "Deviation share of profile energy",
                "value": deviation_energy / (library_energy + deviation_energy),
            },
        ]
    data = pl.DataFrame(rows)
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("value:Q", title=None, scale=alt.Scale(zero=False)),
            y=alt.Y("site:N", sort=list(SITES), title=None),
        )
        .properties(width=WIDTH, height=110)
        .facet(row=alt.Row("property:N", title=None))
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Key properties of the five decision substations side by side",
                subtitle=(
                    f"{PROVISIONAL} The capacity is the pinned real-time vintage, and the heating "
                    "slope has no interval at this tier, so neither is a measurement."
                ),
            )
        )
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.0)
    print("wrote", name)


def main() -> None:
    """Draw the cards and the cross-substation figure."""
    arrays, labels = _headline()
    for site in SITES:
        _save(card(site=site, arrays=arrays, labels=labels), name=f"card_{site}")
    _save(cross_figure(arrays=arrays, labels=labels), name="figure_cards_cross_substation")


if __name__ == "__main__":
    main()
