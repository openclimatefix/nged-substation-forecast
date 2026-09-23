"""Draw the anonymised charts for the write-up on ENS forecast error by horizon.

One-off throwaway script for the charts in the first step of
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, reading what
`ens_forecast_horizons.py` wrote.

**Every error and every interval a chart shares with the page comes from the tables the report was
printed from**, `leaderboard.parquet` and `intervals.parquet`, and the script stops unless each
number it draws appears in `report.md` as printed. The example days, the weeks, and the
per-generator intervals draw numbers the report does not print, from the saved inputs, predictions,
and losses.

Generators appear only as `A` to `F` and `W1` to `W3`, every output is a fraction of the
generator's own capacity, the time axes count hours or days rather than dates, and no data mark
writes its values into the SVG's accessibility text. The example days' weather is drawn with no
generator label.

Run it with `uv run python studies/beam_diffuse_split/ens_forecast_charts.py`, after
`ens_forecast_horizons.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import logging
import math
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final, Literal

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from ens_forecast_horizons import (
    BAND_DAYS,
    CALENDAR_ONLY_MONTH,
    EMULATED_DAY,
    METRIC,
    OUTPUT_DIR,
    PERCENTAGE_POINTS,
    PLANNED,
    baseline_arm,
    ens_arm,
    month_ens_arm,
    upsampling_arm,
)
from studies.bootstrap import bootstrap_absolute
from studies.charts import (
    CONTENT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
    wrapped,
)
from weather_product_charts import ASSETS_DIR

_LOG: Final[logging.Logger] = logging.getLogger("ens_forecast_charts")

DomainType = Literal["solar", "wind"]
DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

SITES: Final[dict[DomainType, tuple[str, ...]]] = {
    "solar": ("A", "B", "C", "D", "E", "F"),
    "wind": ("W1", "W2", "W3"),
}

SCOPES: Final[dict[DomainType, str]] = {
    "solar": "Six solar farms in Lincolnshire, every daylight hour, April 2024 to September 2026.",
    "wind": "Three wind farms in Lincolnshire, every hour, August 2024 to September 2026.",
}

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole months and a fitting seed."
)
X_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"
MAE_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"

WAY_NAMES: Final[dict[str, str]] = {
    "control": "Control member",
    "mean": "Ensemble mean",
    "members": "Member by member",
}
"""How the charts name the three ways of using ENS."""

BASELINE_NAMES: Final[dict[str, str]] = {
    "persistence": "Persistence",
    "diurnal_persistence": "Diurnal persistence",
    "smart_persistence": "Smart persistence",
    "climatology": "Climatology",
}

REFERENCE_NAMES: Final[dict[str, str]] = {
    "era5": "ERA5 (not a forecast)",
    "live_gb_rich_xgb": "UKV and ICON-EU (not a forecast)",
}

METHOD_NAMES: Final[dict[str, str]] = {
    "native": "Native steps",
    "linear": "Linear",
    "clear_sky": "Clear-sky index",
    "clear_sky_conserving": "Clear-sky, step mean kept",
    "linear_pchip": "Linear, shape-preserving temperature",
    "clear_sky_pchip": "Clear-sky index, shape-preserving temperature",
    "clear_sky_conserving_pchip": "Clear-sky index, step mean kept, shape-preserving temperature",
    "direction_components": "Direction from components",
    "speed_components": "Speed from components",
    "components": "Both from components",
}
"""Each upsampling combination's name on a chart, as `ens_forecast_horizons.COMBINATIONS` keys
them."""

COMPARED: Final[tuple[str, str]] = (
    "The combination it was judged against",
    "Linear interpolation",
)
"""The two comparisons each upsampling candidate is drawn against, the same in every panel, so the
colour each takes is the same in every panel of the figure's shared colour scale."""

EXAMPLE_COLOURS: Final[dict[str, str]] = {
    "native": ocf.BLACK_1,
    "linear": ocf.BRAND_ORANGE,
    "clear_sky": ocf.DATA_BLUE,
    "clear_sky_conserving": ocf.DATA_SKY,
    "components": ocf.DATA_BLUE,
}
"""Each drawn combination's colour. A figure shares one colour scale across its panels, so every
combination has one colour whichever technology it is drawn for."""

EXAMPLE_METHODS: Final[dict[DomainType, tuple[str, ...]]] = {
    "solar": ("native", "linear", "clear_sky", "clear_sky_conserving"),
    "wind": ("native", "linear", "components"),
}
"""The combinations the example days draw: for solar each radiation technique, for wind the two
ways of taking the 100 m speed."""

SHAPES: Final[tuple[str, ...]] = ("circle", "diamond", "square", "triangle-up")
DASHES: Final[tuple[tuple[int, ...], ...]] = ((1, 0), (6, 3), (2, 2), (8, 2, 2, 2))
ENS_COLOUR: Final[str] = ocf.DATA_BLUE
BASELINE_COLOUR: Final[str] = ocf.BRAND_ORANGE
REFERENCE_COLOUR: Final[str] = ocf.DATA_SKY
MEASURED_COLOUR: Final[str] = ocf.BLACK_1
"""Data Blue, Brand Orange, and Data Sky, each a main data colour of OCF's brand guidelines, colour
the three groups; within a group, shape and dash tell the series apart."""

TIME_PANEL_HEIGHT_PX: Final[int] = 70


def _band_label(day: int) -> str:
    """Return a band's row label.

    Args:
        day: The band's day.

    Returns:
        Such as `Day 1`.
    """
    return f"Day {day}"


def _check_printed(*, report: str, texts: list[str]) -> None:
    """Stop unless every formatted number appears in the report as printed.

    Args:
        report: The report's text.
        texts: The table-cell fragments to look for.

    Raises:
        ValueError: If any fragment is missing.
    """
    missing = [text for text in texts if text not in report]
    if missing:
        msg = f"{len(missing)} numbers are not in report.md as printed, such as {missing[:3]}"
        raise ValueError(msg)


def _board(*, domain: DomainType, report: str) -> dict[str, dict[str, float]]:
    """Read one technology's leaderboard at the primary setting, checked against the report.

    Args:
        domain: `solar` or `wind`.
        report: The report's text.

    Returns:
        Arm to its `value`, `lower_95`, and `upper_95`, in percentage points.
    """
    rows = pl.read_parquet(OUTPUT_DIR / "leaderboard.parquet").filter(
        (pl.col("domain") == domain) & (pl.col("setting") == "pooled")
    )
    _check_printed(
        report=report,
        texts=[
            f"| {row['arm']} | {row['mae_pp']:.3f} "
            f"| [{row['lower_95_pp']:.3f}, {row['upper_95_pp']:.3f}] |"
            for row in rows.iter_rows(named=True)
        ],
    )
    return {
        row["arm"]: {
            "value": row["mae_pp"],
            "lower_95": row["lower_95_pp"],
            "upper_95": row["upper_95_pp"],
        }
        for row in rows.iter_rows(named=True)
    }


def _contrasts(*, report: str) -> pl.DataFrame:
    """Read every interval, checked against the report.

    Args:
        report: The report's text.

    Returns:
        One row per interval.
    """
    rows = pl.read_parquet(OUTPUT_DIR / "intervals.parquet")
    _check_printed(
        report=report,
        texts=[
            f"| {row['treatment']} − {row['reference']} | {row['difference_pp']:+.3f} "
            f"| [{row['lower_95_pp']:+.3f}, {row['upper_95_pp']:+.3f}] |"
            for row in rows.iter_rows(named=True)
        ],
    )
    return rows


def _pick(
    *,
    contrasts: pl.DataFrame,
    domain: DomainType,
    treatment: str,
    reference: str,
    setting: str = "pooled",
) -> dict[str, float]:
    """Return one contrast's estimate and interval.

    Args:
        contrasts: Every interval.
        domain: `solar` or `wind`.
        treatment: The treatment arm.
        reference: The reference arm.
        setting: The hyperparameter setting.

    Returns:
        `difference`, `lower_95`, and `upper_95`, in points.

    Raises:
        ValueError: Unless exactly one interval matches.
    """
    rows = contrasts.filter(
        (pl.col("domain") == domain)
        & (pl.col("treatment") == treatment)
        & (pl.col("reference") == reference)
        & (pl.col("setting") == setting)
    )
    if rows.height != 1:
        msg = f"{rows.height} intervals for {domain} {treatment} − {reference} ({setting})"
        raise ValueError(msg)
    row = rows.row(0, named=True)
    return {
        "difference": row["difference_pp"],
        "lower_95": row["lower_95_pp"],
        "upper_95": row["upper_95_pp"],
    }


def _domain_around(*, low: float, high: float, step: float) -> tuple[float, float]:
    """Return an axis range around `low` and `high`, padded out to multiples of `step`.

    Args:
        low: The smallest value to show.
        high: The largest value to show.
        step: The rounding step.

    Returns:
        The range.
    """
    return (math.floor(low / step) * step - step / 2, math.ceil(high / step) * step + step / 2)


# --- Figure 1: the leaderboard ------------------------------------------------------------------


def _series_rows(*, board: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Return the leaderboard's points: each way and each baseline at each band.

    Args:
        board: The technology's leaderboard.

    Returns:
        One row per series and band, with `series`, `group`, `day`, `value`, `lower_95`, and
        `upper_95`.
    """
    rows = []
    for day in BAND_DAYS:
        rows += [
            {
                "series": f"ENS: {name}",
                "group": "ens",
                "day": day,
                **board[ens_arm(way=way, day=day)],
            }
            for way, name in WAY_NAMES.items()
        ]
        for baseline in ("persistence", "diurnal_persistence", "smart_persistence"):
            rows.append(
                {
                    "series": BASELINE_NAMES[baseline],
                    "group": "baseline",
                    "day": day,
                    **board[baseline_arm(name=baseline, day=day)],
                }
            )
        rows.append(
            {"series": "Climatology", "group": "baseline", "day": day, **board["climatology"]}
        )
    return pl.DataFrame(rows)


def _leaderboard_panel(
    *, domain: DomainType, board: dict[str, dict[str, float]], keys: bool
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one technology's error at each horizon: ENS three ways, the baselines, the references.

    Args:
        domain: `solar` or `wind`.
        board: The technology's leaderboard.
        keys: Whether to draw the key above the panel.

    Returns:
        The panel.
    """
    rows = _series_rows(board=board)
    series = rows["series"].unique(maintain_order=True).to_list()
    colours = [ENS_COLOUR if s.startswith("ENS") else BASELINE_COLOUR for s in series]
    ens_series = [s for s in series if s.startswith("ENS")]
    baseline_series = [s for s in series if not s.startswith("ENS")]
    shapes = [
        SHAPES[ens_series.index(s)] if s in ens_series else SHAPES[baseline_series.index(s)]
        for s in series
    ]
    dashes = [
        list(DASHES[ens_series.index(s)] if s in ens_series else DASHES[baseline_series.index(s)])
        for s in series
    ]
    offsets = {s: (index - (len(series) - 1) / 2) * 0.12 for index, s in enumerate(series)}
    data = rows.with_columns(
        x=pl.col("day") + pl.col("series").replace_strict(offsets, return_dtype=pl.Float64)
    ).with_columns(pl.col("value", "lower_95", "upper_95").round(3))
    references = pl.DataFrame(
        [{"name": REFERENCE_NAMES[arm], **board[arm]} for arm in REFERENCE_NAMES]
    )
    values = [
        *data["lower_95"].to_list(),
        *data["upper_95"].to_list(),
        *references["value"].to_list(),
    ]
    y_domain = _domain_around(low=min(values), high=max(values), step=1.0)
    x_scale = alt.Scale(domain=[-0.8, 15.8], nice=False)
    y_scale = alt.Scale(domain=list(y_domain), nice=False, zero=False)
    width = CONTENT_WIDTH_PX - 70
    x = alt.X(
        "x:Q",
        scale=x_scale,
        title="Horizon: days after the ENS run's own day",
        axis=alt.Axis(values=list(BAND_DAYS), format="d", grid=False),
    )
    y = alt.Y("value:Q", scale=y_scale, title=wrapped(text=MAE_TITLE, width=40))
    colour = alt.Color("series:N", scale=alt.Scale(domain=series, range=colours), legend=None)
    shape = alt.Shape("series:N", scale=alt.Scale(domain=series, range=shapes), legend=None)
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=series, range=dashes), legend=None)
    reference_rules = (
        alt.Chart(references)
        .mark_rule(color=REFERENCE_COLOUR, strokeWidth=2, aria=False)
        .encode(y=alt.Y("value:Q", scale=y_scale))  # ty: ignore[unresolved-attribute]
    )
    reference_text = (
        alt.Chart(references.with_columns(x=pl.lit(15.7)))
        .mark_text(align="right", baseline="bottom", dy=-3, color=ocf.BLACK_1, aria=False)
        .encode(x=alt.X("x:Q", scale=x_scale), y=alt.Y("value:Q", scale=y_scale), text="name:N")  # ty: ignore[unresolved-attribute]
    )
    lines = (
        alt.Chart(data.filter(pl.col("series") != "Climatology"))
        .mark_line(strokeWidth=1.5, aria=False)
        .encode(x=x, y=y, color=colour, strokeDash=dash, detail="series:N")  # ty: ignore[unresolved-attribute]
    )
    intervals = (
        alt.Chart(data)
        .mark_rule(strokeWidth=1.5, aria=False)
        .encode(x=x, y=alt.Y("lower_95:Q", scale=y_scale), y2="upper_95:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    tooltip = [
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("day:Q", title="Day"),
        alt.Tooltip("value:Q", title="Mean absolute error"),
        alt.Tooltip("lower_95:Q", title="Lower 95%"),
        alt.Tooltip("upper_95:Q", title="Upper 95%"),
    ]
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=55, opacity=1, aria=False)
        .encode(x=x, y=y, color=colour, shape=shape, tooltip=tooltip)  # ty: ignore[unresolved-attribute]
    )
    panel = alt.LayerChart(
        layer=[reference_rules, reference_text, lines, intervals, points],
        width=width,
        height=260,
        title=alt.TitleParams(domain.capitalize(), anchor="start", frame="group", fontSize=14),
    )
    if not keys:
        return panel
    return alt.vconcat(
        _key(
            title="ENS forecasts", labels=ens_series, colour=ENS_COLOUR, shapes=SHAPES, width=width
        ),
        _key(
            title="No-weather baselines",
            labels=baseline_series,
            colour=BASELINE_COLOUR,
            shapes=SHAPES,
            width=width,
        ),
        panel,
        spacing=6,
    )


def _key(
    *, title: str, labels: list[str], colour: str, shapes: tuple[str, ...], width: int
) -> alt.LayerChart:
    """Draw one row of a key: a coloured point of each shape and its label.

    Args:
        title: The row's title.
        labels: Each entry's label.
        colour: The entries' colour.
        shapes: Each entry's shape, in order.
        width: The row's width.

    Returns:
        A one-row chart.
    """
    slot = width // len(labels)
    data = pl.DataFrame(
        {
            "label": labels,
            "shape": list(shapes[: len(labels)]),
            "x": [6 + index * slot for index in range(len(labels))],
        }
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=55, opacity=1, color=colour)
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), shape=alt.Shape("shape:N", scale=None))  # ty: ignore[unresolved-attribute]
    )
    text = (
        alt.Chart(data)
        .mark_text(align="left", dx=10, color=ocf.BLACK_1, limit=slot - 20)
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[points, text],
        width=width,
        height=16,
        title=alt.TitleParams(title, anchor="start", fontSize=11),
    )


def leaderboard(
    *,
    boards: dict[DomainType, dict[str, dict[str, float]]],
    chosen: dict[DomainType, str],
    title: str,
) -> alt.VConcatChart:
    """Draw Figure 1.

    Args:
        boards: Each technology's leaderboard.
        chosen: Each technology's chosen upsampling combination, as the chart names it.
        title: The figure's title, stating the finding.

    Returns:
        The figure.
    """
    return figure(
        panels=[_leaderboard_panel(domain=d, board=boards[d], keys=d == "solar") for d in DOMAINS],
        number=1,
        title=title,
        subtitle=[
            (
                "Every ENS row is an XGBoost model per generator given ENS at that horizon, three "
                "ways: the control member; the mean of the 51 members; each member in turn, the 51 "
                "forecasts averaged. Each is trained on the input it is scored with. ENS is "
                "upsampled to hourly by the combination the rule in Figures 7 and 8 chose: "
                f"for solar, {chosen['solar']}; for wind, {chosen['wind']}."
            ),
            (
                "No-weather baselines read the telemetry up to 09:00 UTC on the run's own day, "
                "when the live service can first read the run, or up to 00 UTC for day 0. "
                "Climatology does not depend on the horizon. The two light blue rules are not "
                "forecasts: ERA5, and the best past-weather input a live service can read, scored "
                "on the same hours."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
            (
                "The intervals are wide mainly because every row's error rises and falls together "
                "from month to month; Figures 2, 9, and 10 compare rows on the same months."
            ),
        ],
        figure_planning=None,
    )


# --- Contrast figures ---------------------------------------------------------------------------


def _contrast_panel(
    *,
    contrasts: pl.DataFrame,
    domain: DomainType,
    pairs: list[tuple[str, list[tuple[str, str]]]],
    days: tuple[int, ...],
    x_domain: tuple[float, float],
    zero_label: str,
    better_label: str,
    condition_title: str,
    keys: bool,
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one technology's contrasts, one row per band and one mark per condition.

    Args:
        contrasts: Every interval.
        domain: `solar` or `wind`.
        pairs: Each condition's name and, per band in `days`, its (treatment, reference).
        days: The bands, one row each.
        x_domain: The x axis's range.
        zero_label: What zero means.
        better_label: What the negative direction means.
        condition_title: The key's title.
        keys: Whether to label the zero rule and the better direction.

    Returns:
        The panel.
    """
    rows = []
    for condition, per_day in pairs:
        for day, (treatment, reference) in zip(days, per_day, strict=True):
            rows.append(
                {
                    "label": _band_label(day),
                    "family": "weather model",
                    "condition": condition,
                    **_pick(
                        contrasts=contrasts, domain=domain, treatment=treatment, reference=reference
                    ),
                }
            )
    return interval_panel(
        rows=pl.DataFrame(rows),
        x_domain=x_domain,
        x_title=X_TITLE,
        zero_label=zero_label,
        better_label=better_label,
        conditions=[condition for condition, _ in pairs],
        condition_title=condition_title,
        panel_title=domain.capitalize(),
        reference_labels=keys,
        family_key=False,
        figure_planning="exploratory",
    )


def _x_domain(
    *, contrasts: pl.DataFrame, pairs: list[tuple[DomainType, str, str]]
) -> tuple[float, float]:
    """Return a range covering every named interval and zero.

    Args:
        contrasts: Every interval.
        pairs: (domain, treatment, reference) triples.

    Returns:
        The padded range.
    """
    values = [0.0]
    for domain, treatment, reference in pairs:
        picked = _pick(contrasts=contrasts, domain=domain, treatment=treatment, reference=reference)
        values += [picked["lower_95"], picked["upper_95"]]
    span = max(values) - min(values)
    step = 0.1 if span < 1.5 else 0.5 if span < 6 else 1.0
    return _domain_around(low=min(values), high=max(values), step=step)


def against_day0(*, contrasts: pl.DataFrame, title: str) -> alt.VConcatChart:
    """Draw Figure 2: each horizon against day 0, each way of using ENS.

    Args:
        contrasts: Every interval.
        title: The figure's title.

    Returns:
        The figure.
    """
    days = BAND_DAYS[1:]
    # The ensemble mean is the planned arm, so it takes the first, fully-shaded condition; the
    # control member and member-by-member are exploratory here and take the light shade.
    emphasis_order = ("mean", "control", "members")
    panels = []
    for domain in DOMAINS:
        pairs = [
            (WAY_NAMES[way], [(ens_arm(way=way, day=day), ens_arm(way=way, day=0)) for day in days])
            for way in emphasis_order
        ]
        triples = [(domain, t, r) for _, per_day in pairs for t, r in per_day]
        panels.append(
            _contrast_panel(
                contrasts=contrasts,
                domain=domain,
                pairs=pairs,
                days=days,
                x_domain=_x_domain(contrasts=contrasts, pairs=triples),
                zero_label="same as day 0",
                better_label="better than day 0",
                condition_title="Way of using ENS",
                keys=domain == "solar",
            )
        )
    return figure(
        panels=panels,
        number=2,
        title=title,
        subtitle=[
            (
                "Mean absolute error at each horizon minus the same way's error at day 0, on the "
                "same generator-hours. Planned: the ensemble mean at days 1 and 7; every other "
                "mark "
                "is exploratory."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning=None,
    )


def ways(*, contrasts: pl.DataFrame, title: str) -> alt.VConcatChart:
    """Draw Figure 3: the three ways of using ENS against each other at each horizon.

    Args:
        contrasts: Every interval.
        title: The figure's title.

    Returns:
        The figure.
    """
    panels = []
    for domain in DOMAINS:
        pairs = [
            (
                "Member by member minus ensemble mean",
                [(ens_arm(way="members", day=d), ens_arm(way="mean", day=d)) for d in BAND_DAYS],
            ),
            (
                "Ensemble mean minus control member",
                [(ens_arm(way="mean", day=d), ens_arm(way="control", day=d)) for d in BAND_DAYS],
            ),
        ]
        triples = [(domain, t, r) for _, per_day in pairs for t, r in per_day]
        panels.append(
            _contrast_panel(
                contrasts=contrasts,
                domain=domain,
                pairs=pairs,
                days=BAND_DAYS,
                x_domain=_x_domain(contrasts=contrasts, pairs=triples),
                zero_label="no difference",
                better_label="first named better",
                condition_title="Contrast",
                keys=domain == "solar",
            )
        )
    return figure(
        panels=panels,
        number=9,
        title=title,
        subtitle=[
            (
                "Difference in mean absolute error between two ways of using ENS at the same "
                "horizon, on the same generator-hours. Planned: both contrasts at day 1; every "
                "other "
                "mark is exploratory."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning=None,
    )


def against_baselines(
    *, contrasts: pl.DataFrame, best: dict[DomainType, dict[int, str]], title: str
) -> alt.VConcatChart:
    """Draw Figure 4: the ensemble mean and the member-by-member forecast against the best baseline.

    Args:
        contrasts: Every interval.
        best: Each technology's best baseline at each band.
        title: The figure's title.

    Returns:
        The figure.
    """
    panels = []
    for domain in DOMAINS:
        pairs = [
            (
                f"{name} minus the best baseline",
                [(ens_arm(way=way, day=d), best[domain][d]) for d in BAND_DAYS],
            )
            for way, name in (("mean", "Ensemble mean"), ("members", "Member by member"))
        ]
        triples = [(domain, t, r) for _, per_day in pairs for t, r in per_day]
        panels.append(
            _contrast_panel(
                contrasts=contrasts,
                domain=domain,
                pairs=pairs,
                days=BAND_DAYS,
                x_domain=_x_domain(contrasts=contrasts, pairs=triples),
                zero_label="same as the best baseline",
                better_label="ENS better",
                condition_title="Contrast",
                keys=domain == "solar",
            )
        )
    named = "; ".join(
        f"{domain}: "
        + ", ".join(
            f"day {day} {BASELINE_NAMES[arm.split('_day')[0]].lower()}"
            for day, arm in best[domain].items()
        )
        for domain in DOMAINS
    )
    # The best baseline at day 7 is chosen after the run, but the day-7 ensemble-mean-against-
    # climatology contrast is itself one of `PLANNED`. Where that baseline turns out to be
    # climatology, the mark it draws here is planned, not exploratory, and the subtitle says so.
    planned_domains = [
        domain for domain in DOMAINS if (ens_arm(way="mean", day=7), best[domain][7]) in PLANNED
    ]
    plan_note = (
        "All marks are exploratory."
        if not planned_domains
        else (
            "Planned: the ensemble mean against climatology at day 7, for "
            + " and ".join(planned_domains)
            + "; every other mark is exploratory."
        )
    )
    return figure(
        panels=panels,
        number=10,
        title=title,
        subtitle=[
            (
                "Mean absolute error of an XGBoost model given ENS minus that of the no-weather "
                "baseline with the lowest error at the same horizon, chosen after the run: "
                f"{named}. {plan_note}"
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning=None,
    )


def calendar_contrast(*, contrasts: pl.DataFrame, title: str) -> alt.VConcatChart:
    """Draw Figure 11: the ensemble mean against the calendar-only arm, at every horizon.

    `calendar_only` is a post hoc arm added after the first science review: the same XGBoost model
    given the ENS arms' non-weather columns and no weather at all, so a contrast against it tells
    apart ENS running out of skill from the model overfitting the calendar columns.

    Args:
        contrasts: Every interval.
        title: The figure's title.

    Returns:
        The figure.
    """
    pairs = [
        (
            "Day of year",
            [(ens_arm(way="mean", day=day), "calendar_only") for day in BAND_DAYS],
        ),
        (
            "Calendar month",
            [(month_ens_arm(day=day), CALENDAR_ONLY_MONTH) for day in BAND_DAYS],
        ),
    ]
    panels = []
    for domain in DOMAINS:
        triples = [(domain, t, r) for _, per_day in pairs for t, r in per_day]
        panels.append(
            _contrast_panel(
                contrasts=contrasts,
                domain=domain,
                pairs=pairs,
                days=BAND_DAYS,
                x_domain=_x_domain(contrasts=contrasts, pairs=triples),
                zero_label="same as calendar-only",
                better_label="ENS better",
                condition_title="Calendar column both arms use",
                keys=domain == "solar",
            )
        )
    return figure(
        panels=panels,
        number=11,
        title=title,
        subtitle=[
            (
                "Mean absolute error of an XGBoost model given ENS's ensemble mean minus the same "
                "model given no weather at all, only the ENS arms' calendar and sun-geometry "
                "columns, with the day-of-year column and with calendar month instead. All marks "
                "are post hoc: added after the first and second science reviews."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning=None,
    )


def upsampling_contrasts(
    *, contrasts: pl.DataFrame, domain: DomainType, number: int, title: str
) -> alt.VConcatChart:
    """Draw one technology's upsampling contrasts: each candidate against its comparator and linear.

    The candidates and the combination each was judged against are read from the report's
    upsampling section, so the figure shows whatever the choosing rule tried.

    Args:
        contrasts: Every interval.
        domain: `solar` or `wind`.
        number: The figure's number.
        title: The figure's title.

    Returns:
        The figure, one panel per candidate.
    """
    rows = contrasts.filter((pl.col("domain") == domain) & (pl.col("section") == "upsampling"))
    candidates = dict.fromkeys(
        treatment.split("_day")[0].removeprefix("up_") for treatment in rows["treatment"].to_list()
    )
    shared_domain = _x_domain(
        contrasts=contrasts,
        pairs=[(domain, t, r) for t, r in rows.select("treatment", "reference").iter_rows()],
    )
    panels = []
    for candidate in candidates:
        compared = rows.filter(
            pl.col("treatment") == upsampling_arm(method=candidate, day=BAND_DAYS[0])
        )["reference"].to_list()
        references = [reference.split("_day")[0].removeprefix("up_") for reference in compared]
        judged_against = references[0]
        pairs = [
            (
                COMPARED[index],
                [
                    (
                        upsampling_arm(method=candidate, day=d),
                        upsampling_arm(method=reference, day=d),
                    )
                    for d in BAND_DAYS
                ],
            )
            for index, reference in enumerate(references)
        ]
        panels.append(
            interval_panel(
                rows=pl.DataFrame(
                    [
                        {
                            "label": _band_label(day),
                            "family": "weather model",
                            "condition": condition,
                            **_pick(
                                contrasts=contrasts,
                                domain=domain,
                                treatment=treatment,
                                reference=reference,
                            ),
                        }
                        for condition, per_day in pairs
                        for day, (treatment, reference) in zip(BAND_DAYS, per_day, strict=True)
                    ]
                ),
                x_domain=shared_domain,
                x_title=X_TITLE,
                zero_label="no difference",
                better_label=f"{METHOD_NAMES[candidate].lower()} better",
                conditions=list(COMPARED),
                condition_title="Compared with",
                panel_title=(
                    f"{METHOD_NAMES[candidate]}, judged against "
                    f"{METHOD_NAMES[judged_against].lower()}"
                ),
                family_key=False,
                figure_planning="exploratory",
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[
            (
                "Mean absolute error of an XGBoost model given the ensemble mean upsampled one "
                "way, minus the same model given it upsampled another way, on the same "
                "generator-hours. Days 0 to 5 sit on ENS's 3-hour steps, days 7 to 14 on its "
                "6-hour steps. All marks are exploratory."
            ),
            f"{DOTS} {CAPACITY}",
            SCOPES[domain],
        ],
        figure_planning=None,
    )


# --- Time series --------------------------------------------------------------------------------


def _example_day(*, inputs: pl.DataFrame, domain: DomainType) -> tuple[str, datetime]:
    """Choose the example day by rule, from measured output alone.

    For solar, the day from April to September with the largest mean hour-to-hour change in
    measured output; for wind, the day with the largest range of measured output. Only days on
    which every hour of the day is in the scored rows for solar's daylight or wind's whole day
    count, and only at the generator with the most scored hours.

    Args:
        inputs: The saved inputs, with `power_mw` on the scored rows.
        domain: `solar` or `wind`.

    Returns:
        The generator and the day's midnight.
    """
    measured = (
        inputs.filter((pl.col("method") == "linear") & (pl.col("day") == 1))
        .drop_nulls("power_mw")
        .select("site", "time", output=pl.col("power_mw") / pl.col("effective_capacity_mw"))
        .unique()
        .sort("site", "time")
    )
    site = measured.group_by("site").len().sort("len", descending=True)["site"][0]
    at_site = measured.filter(pl.col("site") == site).with_columns(
        date=(pl.col("time") - pl.duration(minutes=30 if domain == "solar" else 0)).dt.truncate(
            "1d"
        ),
        step=(pl.col("output") - pl.col("output").shift(1)).abs(),
    )
    if domain == "solar":
        days = (
            at_site.filter(pl.col("date").dt.month().is_between(4, 9))
            .group_by("date")
            .agg(score=pl.col("step").mean(), hours=pl.len())
            .filter(pl.col("hours") >= 12)
        )
    else:
        days = (
            at_site.group_by("date")
            .agg(score=pl.col("output").max() - pl.col("output").min(), hours=pl.len())
            .filter(pl.col("hours") == 24)
        )
    return site, days.sort("score", descending=True)["date"][0]


def _day_panels(
    *, inputs: pl.DataFrame, domain: DomainType, site: str, date: datetime, day: int, first: bool
) -> alt.VConcatChart:
    """Draw one horizon's example day: the upsampled input by technique, and the measured output.

    Args:
        inputs: The saved inputs.
        domain: `solar` or `wind`.
        site: The generator, never drawn.
        date: The day's midnight.
        day: The band.
        first: Whether to draw the key.

    Returns:
        The two stacked panels.
    """
    field = "ghi" if domain == "solar" else "speed_100m"
    unit = "Global irradiance (W m⁻²)" if domain == "solar" else "100 m wind speed (m s⁻¹)"
    start, end = date, date + timedelta(days=1)
    rows = inputs.filter(
        (pl.col("site") == site)
        & (pl.col("day") == day)
        & pl.col("time").is_between(start, end, closed="right" if domain == "solar" else "left")
    ).with_columns(hour=(pl.col("time") - start).dt.total_minutes() / 60.0)
    methods = list(EXAMPLE_METHODS[domain])
    names = [METHOD_NAMES[m] for m in methods]
    every = list(EXAMPLE_COLOURS)
    series = rows.filter(pl.col("method").is_in(methods)).select(
        "hour",
        value=pl.col(field),
        name=pl.col("method").replace_strict(dict(zip(methods, names, strict=True))),
    )
    width = (CONTENT_WIDTH_PX - 90) // 2
    x = alt.X(
        "hour:Q",
        scale=alt.Scale(domain=[0, 24], nice=False),
        axis=alt.Axis(values=list(range(0, 25, 6)), format="d"),
        title="Hour of the day (UTC)",
    )
    colour = alt.Color(
        "name:N",
        scale=alt.Scale(
            domain=[METHOD_NAMES[m] for m in every], range=[EXAMPLE_COLOURS[m] for m in every]
        ),
        legend=None,
    )
    weather = (
        alt.Chart(series.filter(pl.col("name") != names[0]))
        .mark_line(strokeWidth=1.8, aria=False)
        .encode(x=x, y=alt.Y("value:Q", title=wrapped(text=unit, width=26)), color=colour)  # ty: ignore[unresolved-attribute]
    )
    native_points = (
        alt.Chart(series.filter(pl.col("name") == names[0]))
        .mark_point(filled=True, size=45, shape="square", aria=False, color=ocf.BLACK_1)
        .encode(x=x, y="value:Q")  # ty: ignore[unresolved-attribute]
    )
    top = alt.LayerChart(
        layer=[weather, native_points],
        width=width,
        height=110,
        title=alt.TitleParams(
            f"Day {day}: {'3-hour' if day * 24 + 24 <= 144 else '6-hour'} steps",
            anchor="start",
            fontSize=12,
        ),
    )
    measured = (
        rows.filter(pl.col("method") == "linear")
        .drop_nulls("power_mw")
        .select(
            "hour", output=pl.col("power_mw") / pl.col("effective_capacity_mw") * PERCENTAGE_POINTS
        )
    )
    bottom = (
        alt.Chart(measured)
        .mark_line(
            strokeWidth=1.8,
            point=alt.OverlayMarkDef(size=20, filled=True, aria=False),
            color=MEASURED_COLOUR,
            aria=False,
        )
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "output:Q",
                title=wrapped(text="Measured output (% of capacity)", width=26),
                scale=alt.Scale(domain=[0, 110]),
            ),
        )
        .properties(width=width, height=80)
    )
    return alt.vconcat(top, bottom, spacing=6)


def example_days(
    *, inputs: dict[DomainType, pl.DataFrame], title: str
) -> tuple[alt.VConcatChart, dict[DomainType, str]]:
    """Draw the upsampling example days, one row per technology, day 1 beside day 7.

    Args:
        inputs: Each technology's saved inputs.
        title: The figure's title.

    Returns:
        The figure, and each technology's example month and year for the page.
    """
    rows = []
    months: dict[DomainType, str] = {}
    for domain in DOMAINS:
        site, date = _example_day(inputs=inputs[domain], domain=domain)
        months[domain] = f"{date:%B %Y}"
        methods = list(EXAMPLE_METHODS[domain])
        colours = [EXAMPLE_COLOURS[m] for m in methods]
        key = _line_key(labels=[METHOD_NAMES[m] for m in methods], colours=colours)
        pair = alt.hconcat(
            *(
                _day_panels(
                    inputs=inputs[domain], domain=domain, site=site, date=date, day=day, first=True
                )
                for day in (1, 7)
            ),
            spacing=20,
        )
        rows.append(
            alt.vconcat(
                key,
                pair,
                spacing=6,
                title=alt.TitleParams(domain.capitalize(), anchor="start", fontSize=14),
            )
        )
    return (
        figure(
            panels=rows,
            number=6,
            title=title,
            subtitle=[
                (
                    "Top of each pair: the ensemble mean of one day at one generator, upsampled to "
                    "hourly each way; squares mark ENS's own steps, each radiation value the mean "
                    "over the step ending at the square. Bottom: that generator's measured hourly "
                    "output. The same calendar day at day 1 and day 7."
                ),
                (
                    "Days chosen by rule from measured output alone: for solar the April-to-"
                    "September day with the largest mean hour-to-hour change, for wind the day "
                    "with the largest range. Generator not named."
                ),
                CAPACITY,
            ],
            figure_planning=None,
        ),
        months,
    )


def _line_key(*, labels: Sequence[str], colours: Sequence[str]) -> alt.LayerChart:
    """Draw a one-row key of short line segments.

    Args:
        labels: Each entry's label.
        colours: Each entry's colour.

    Returns:
        A one-row chart.
    """
    width = CONTENT_WIDTH_PX - 60
    slot = width // len(labels)
    data = pl.DataFrame(
        {
            "label": labels,
            "colour": colours,
            "x": [index * slot for index in range(len(labels))],
            "x2": [index * slot + 18 for index in range(len(labels))],
        }
    )
    segments = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=None),
            x2="x2:Q",
            y=alt.value(8),
            color=alt.Color("colour:N", scale=None),
        )
    )
    text = (
        alt.Chart(data)
        .mark_text(align="left", dx=24, color=ocf.BLACK_1, limit=slot - 30)
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(layer=[segments, text], width=width, height=16)


def _seed_mean(*, domain: DomainType, arms: list[str]) -> pl.DataFrame:
    """Return each arm's out-of-fold prediction averaged over its seeds, as a fraction of capacity.

    Args:
        domain: `solar` or `wind`.
        arms: The arms.

    Returns:
        One row per (site, time, arm), with `measured` and `predicted`.
    """
    capacity = pl.read_parquet(OUTPUT_DIR / f"{domain}_rows.parquet").select(
        "site", "time", "effective_capacity_mw"
    )
    return (
        pl.scan_parquet(OUTPUT_DIR / f"{domain}_predictions.parquet")
        .filter(pl.col("setting") == "pooled", pl.col("arm").is_in(arms))
        .group_by("site", "time", "arm")
        .agg(pl.col("power_mw").first(), pl.col("prediction_mw").mean())
        .collect()
        .join(capacity, on=["site", "time"])
        .select(
            "site",
            "time",
            "arm",
            measured=(pl.col("power_mw") / pl.col("effective_capacity_mw")).cast(pl.Float64),
            predicted=(pl.col("prediction_mw") / pl.col("effective_capacity_mw")).cast(pl.Float64),
        )
    )


def _choose_week(*, measured: pl.DataFrame, domain: DomainType) -> datetime:
    """Choose one week by rule, from the weeks every generator covers on all seven days.

    Solar takes the week whose daily mean output varies most from day to day; wind the week with
    the largest mean hour-to-hour change in output. A generator covers a day with at least 4 scored
    hours for solar and 12 for wind.

    Args:
        measured: One row per (site, time) with `measured`.
        domain: `solar` or `wind`.

    Returns:
        The week's Monday.
    """
    minimum = 4 if domain == "solar" else 12
    frame = measured.sort("site", "time").with_columns(
        week=pl.col("time").dt.truncate("1w"),
        date=pl.col("time").dt.date(),
        step=(pl.col("measured") - pl.col("measured").shift(1).over("site")).abs(),
    )
    covered = (
        frame.group_by("week", "site", "date")
        .len()
        .filter(pl.col("len") >= minimum)
        .group_by("week")
        .len()
        .filter(pl.col("len") == len(SITES[domain]) * 7)
        .select("week")
    )
    daily = frame.group_by("week", "date").agg(daily=pl.col("measured").mean())
    scores = (
        daily.group_by("week").agg(score=pl.col("daily").std())
        if domain == "solar"
        else frame.group_by("week").agg(score=pl.col("step").mean())
    )
    return scores.join(covered, on="week").sort("score", descending=True)["week"][0]


def models_work(*, domain: DomainType, number: int, title: str) -> tuple[alt.VConcatChart, str]:
    """Draw one technology's week: measured output and the day-1 and day-7 ensemble-mean forecasts.

    Args:
        domain: `solar` or `wind`.
        number: The figure's number.
        title: The figure's title.

    Returns:
        The figure, and the week's month and year for the page.
    """
    arms = [ens_arm(way="mean", day=1), ens_arm(way="mean", day=7)]
    predictions = _seed_mean(domain=domain, arms=arms)
    measured = predictions.filter(pl.col("arm") == arms[0]).select("site", "time", "measured")
    week = _choose_week(measured=measured, domain=domain)
    names = ["Measured output", "Day-1 forecast", "Day-7 forecast"]
    long = pl.concat(
        [
            measured.select("site", "time", series=pl.lit(names[0]), value=pl.col("measured")),
            *(
                predictions.filter(pl.col("arm") == arm).select(
                    "site", "time", series=pl.lit(name), value=pl.col("predicted")
                )
                for arm, name in zip(arms, names[1:], strict=True)
            ),
        ]
    ).with_columns(pl.col("value") * PERCENTAGE_POINTS)
    hours = pl.datetime_range(
        week, week + timedelta(days=7), interval="1h", closed="left", eager=True
    )
    grid = pl.DataFrame({"time": hours}).join(pl.DataFrame({"series": names}), how="cross")
    colours = [MEASURED_COLOUR, ocf.DATA_BLUE, ocf.BRAND_ORANGE]
    panels = []
    for index, site in enumerate(SITES[domain]):
        series = (
            grid.join(
                long.filter(pl.col("site") == site).drop("site"), on=["time", "series"], how="left"
            )
            .with_columns(elapsed=(pl.col("time") - week).dt.total_minutes() / (60 * 24))
            .sort("series", "time")
        )
        last = index == len(SITES[domain]) - 1
        panels.append(
            alt.Chart(series)
            .mark_line(strokeWidth=1.3, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "elapsed:Q",
                    scale=alt.Scale(domain=[0, 7], nice=False),
                    axis=alt.Axis(
                        values=[0.5 + d for d in range(7)],
                        labelExpr="'Day ' + (datum.value + 0.5)",
                        labels=last,
                        ticks=False,
                        grid=False,
                    ),
                    title=None,
                ),
                y=alt.Y(
                    "value:Q",
                    title=site,
                    scale=alt.Scale(domain=[0, 110], nice=False),
                    axis=alt.Axis(values=[0, 50, 100], titleAngle=0),
                ),
                color=alt.Color(
                    "series:N", scale=alt.Scale(domain=names, range=colours), legend=None
                ),
            )
            .properties(width=CONTENT_WIDTH_PX - 80, height=TIME_PANEL_HEIGHT_PX)
        )
    rule = (
        "the week whose daily mean output varies most from day to day"
        if domain == "solar"
        else "the week with the largest mean hour-to-hour change in output"
    )
    return (
        figure(
            panels=[_line_key(labels=names, colours=colours), alt.vconcat(*panels, spacing=4)],
            number=number,
            title=title,
            subtitle=[
                (
                    "Hourly output as a percentage of capacity, measured and as forecast out of "
                    "fold "
                    "by the XGBoost model given the ENS ensemble mean at day 1 and at day 7, "
                    "averaged over its three fitting seeds. Gaps are hours outside the scored rows."
                ),
                f"The week is chosen by rule from measured output alone: {rule}. {CAPACITY}",
            ],
            figure_planning=None,
        ),
        f"{week:%B %Y}",
    )


def per_generator(*, title: str, number: int) -> alt.VConcatChart:
    """Draw each generator's error given the day-1 ensemble mean and given the day-7 one.

    Args:
        title: The figure's title.
        number: The figure's number.

    Returns:
        The figure.
    """
    panels = []
    conditions = ("Day 1", "Day 7")
    for domain in DOMAINS:
        losses = pl.read_parquet(OUTPUT_DIR / f"{domain}_losses.parquet").filter(
            pl.col("setting") == "pooled"
        )
        rows = []
        for site in SITES[domain]:
            for condition, day in zip(conditions, (1, 7), strict=True):
                arm = ens_arm(way="mean", day=day)
                interval = bootstrap_absolute(
                    losses=losses.filter((pl.col("site") == site) & (pl.col("arm") == arm)),
                    arm=arm,
                    metric=METRIC,
                )
                rows.append(
                    {
                        "label": site,
                        "family": "weather model",
                        "condition": condition,
                        "value": interval["value"] * PERCENTAGE_POINTS,
                        "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                        "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
                    }
                )
        frame = pl.DataFrame(rows)
        panels.append(
            leaderboard_panel(
                rows=frame,
                x_domain=_domain_around(
                    low=float(frame.select(pl.col("lower_95").min()).item()),
                    high=float(frame.select(pl.col("upper_95").max()).item()),
                    step=1.0,
                ),
                x_title=MAE_TITLE,
                conditions=conditions,
                condition_title="Horizon",
                panel_title=domain.capitalize(),
                keys=domain == "solar",
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[
            (
                "Each generator's mean absolute error, given the ENS ensemble mean at day 1 and at "
                "day 7. Exploratory."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning=None,
    )


def _chosen(*, report: str) -> dict[DomainType, str]:
    """Read each technology's chosen upsampling combination from the report.

    Args:
        report: The report's text.

    Returns:
        Each technology's combination, as the charts name it, in lower case.
    """
    return {
        domain: METHOD_NAMES[
            re.search(
                rf"{domain.capitalize()}: the upsampling technique chosen: (\w+)", report
            ).group(1)  # ty: ignore[unresolved-attribute]
        ].lower()
        for domain in DOMAINS
    }


# --- Main ---------------------------------------------------------------------------------------


def main() -> int:
    """Draw every chart and print the example periods' months for the page."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = (OUTPUT_DIR / "report.md").read_text()
    boards = {domain: _board(domain=domain, report=report) for domain in DOMAINS}
    contrasts = _contrasts(report=report)
    best = {
        domain: {
            day: min(
                (
                    "climatology",
                    *(
                        baseline_arm(name=n, day=day)
                        for n in ("persistence", "diurnal_persistence", "smart_persistence")
                    ),
                ),
                key=lambda arm, d=domain: boards[d][arm]["value"],
            )
            for day in BAND_DAYS
        }
        for domain in DOMAINS
    }
    inputs = {
        domain: pl.read_parquet(OUTPUT_DIR / f"{domain}_inputs.parquet") for domain in DOMAINS
    }
    days_chart, day_months = example_days(inputs=inputs, title=TITLES["example_days"])
    solar_week, solar_month = models_work(domain="solar", number=3, title=TITLES["solar_week"])
    wind_week, wind_month = models_work(domain="wind", number=4, title=TITLES["wind_week"])
    charts = {
        "ens_horizons_leaderboard": leaderboard(
            boards=boards, chosen=_chosen(report=report), title=TITLES["leaderboard"]
        ),
        "ens_horizons_against_day0": against_day0(contrasts=contrasts, title=TITLES["day0"]),
        "ens_horizons_ways": ways(contrasts=contrasts, title=TITLES["ways"]),
        "ens_horizons_against_baselines": against_baselines(
            contrasts=contrasts, best=best, title=TITLES["baselines"]
        ),
        "ens_horizons_against_calendar": calendar_contrast(
            contrasts=contrasts, title=TITLES["calendar"]
        ),
        "ens_upsampling_days": days_chart,
        "ens_upsampling_solar": upsampling_contrasts(
            contrasts=contrasts, domain="solar", number=7, title=TITLES["upsampling_solar"]
        ),
        "ens_upsampling_wind": upsampling_contrasts(
            contrasts=contrasts, domain="wind", number=8, title=TITLES["upsampling_wind"]
        ),
        "ens_horizons_solar_week": solar_week,
        "ens_horizons_wind_week": wind_week,
        "ens_horizons_per_generator": per_generator(title=TITLES["per_generator"], number=5),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    sys.stdout.write(
        f"Example days: solar {day_months['solar']}, wind {day_months['wind']}. "
        f"Weeks: solar {solar_month}, wind {wind_month}. Emulated band: day {EMULATED_DAY}.\n"
    )
    return 0


TITLES: Final[dict[str, str]] = {
    "leaderboard": (
        "At these farms, an XGBoost model given the ENS ensemble mean beats every no-weather "
        "baseline to day 5 for solar and to day 7 for wind"
    ),
    "day0": "Forecast error rises with every day of horizon, fastest over the first week",
    "ways": (
        "The ensemble mean beats the control member, and beats feeding each member through the "
        "model, at almost every horizon"
    ),
    "baselines": (
        "The ENS ensemble mean beats the best no-weather baseline by several points to day 5, "
        "and by day 14 climatology is ahead"
    ),
    "calendar": (
        "The ensemble mean beats the same model given no weather to day 7; after that it does "
        "not, whichever calendar column both use"
    ),
    "example_days": (
        "The clear-sky index keeps the solar day's shape, where linear interpolation shifts it late"
    ),
    "upsampling_solar": (
        "Rebuilding solar radiation through the clear-sky index lowers the error at every horizon "
        "to day 7"
    ),
    "upsampling_wind": (
        "No way of interpolating ENS's wind moves the wind error by a tenth of a point"
    ),
    "solar_week": (
        "Given the day-1 ensemble mean, the XGBoost model follows the day-to-day swings at every "
        "solar farm; the day-7 forecast stays close to an average day"
    ),
    "wind_week": (
        "Given the day-1 ensemble mean, the XGBoost model follows the wind farms' swings; the "
        "day-7 forecast mostly does not"
    ),
    "per_generator": "Every generator's error rises between day 1 and day 7",
}
"""Each figure's title, which states its finding and matches the heading it sits under."""


if __name__ == "__main__":
    sys.exit(main())
