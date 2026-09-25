"""Draw the anonymised charts for the write-up on weather forecasts compared at matched leads.

One-off throwaway script for the charts of
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, reading what
`nwp_forecast_comparison.py` saved: `<domain>_losses.parquet` and `<domain>_predictions.parquet`.
Every interval is computed here from those per-row losses, with the same functions the report
uses (`difference`, `leaderboard`, `bracket` in `nwp_forecast_comparison.py`, which call
`studies.bootstrap`), so no refit is needed and a chart cannot disagree with the report.

Five charts per technology, each with its own title, subtitle, axis titles and key:

1. `headline`: the seven planned contrasts P1a to P4b with 95% intervals, at both settings.
2. `leaderboard`: every arm's own mean absolute error on the shared rows, best first.
3. `models_work`: one week of out-of-fold day-1 ENS-mean forecasts against measured output.
4. `by_lead_day`: error by lead day, with ENS's day-0 and day-1 intervals shaded.
5. `blends`: the two blends against ENS alone and against their permutation controls.

Generators appear only as `A` to `F` and `W1` to `W3`, every error is a fraction of the
generator's own capacity, the time axes count days of the week rather than dates, and no data mark
writes its values into the SVG's accessibility text. The script stops if a site label is not one
of the anonymised labels.

Run it with `uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py --input-dir
DIR --output-dir DIR`. Each SVG is optimised with `npx svgo@4 --multipass --precision=1
--final-newline` unless `--no-svgo` is given. Charts belong under `docs/studies/assets/` only once a
real report exists.
"""

import argparse
import logging
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from build_forecast_inputs import PRODUCT_SLUGS
from nwp_forecast_comparison import (
    BLEND_ARMS,
    GENERATOR_CONTRASTS,
    PERCENTAGE_POINTS,
    SETTINGS,
    DomainType,
    arms_present,
    difference,
    is_baseline_arm,
    leaderboard,
    losses_path,
    predictions_path,
)
from studies.anonymise import SITE_LABELS, WIND_SITE_LABELS
from studies.charts import (
    CONTENT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
    planning,
)

_LOG: Final[logging.Logger] = logging.getLogger("nwp_forecast_charts")

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

SITES: Final[dict[DomainType, tuple[str, ...]]] = {
    "solar": SITE_LABELS,
    "wind": WIND_SITE_LABELS,
}
"""Each technology's anonymised site labels; a chart refuses any other label."""

TECHNOLOGY_NAMES: Final[dict[DomainType, str]] = {
    "solar": "the six solar farms",
    "wind": "the three wind farms",
}

CAPACITY_NOTE: Final[str] = "Every error is a fraction of the generator's own capacity."
SHARED_ROWS_NOTE: Final[str] = "Every product is scored on exactly the same hours."
DOTS_NOTE: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole months and a fitting seed."
)
MAE_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
DIFFERENCE_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"

SETTING_NAMES: Final[dict[str, str]] = {
    "primary": "Primary XGBoost setting",
    "sensitivity": "Second XGBoost setting",
}
"""Each hyperparameter setting's name in a chart key."""

SETTING_COLOURS: Final[tuple[str, str]] = (ocf.BRAND_ORANGE, ocf.DATA_BLUE)
"""The colour of the primary and the second setting, in that order."""

FORECAST_CONDITION: Final[str] = "Forecast product"
BASELINE_CONDITION: Final[str] = "No-weather baseline"

TIME_PANEL_HEIGHT_PX: Final[int] = 70
LEADERBOARD_ROW_PX: Final[int] = 28
LEAD_PANEL_HEIGHT_PX: Final[int] = 300

PRODUCT_NAMES: Final[dict[str, str]] = {
    **{slug: name for name, slug in PRODUCT_SLUGS.items()},
    "ens_mean": "ENS mean",
    "ens_control": "ENS control member",
    "gefs_mean": "GEFS mean",
}
"""Each column prefix's product name, up to its `_day<N>` suffix."""

BASELINE_NAMES: Final[dict[str, str]] = {
    "persistence": "Persistence",
    "diurnal_persistence": "Diurnal persistence",
    "smart_persistence": "Smart persistence",
}


# --- Labels -------------------------------------------------------------------------------------


def prefix_label(*, prefix: str) -> str:
    """Name one weather-column prefix such as `icon_eu_day2`, as `ICON-EU day 2`.

    Args:
        prefix: A product's column prefix, ending `_day<N>`.

    Returns:
        The product's name and its day.

    Raises:
        ValueError: If the prefix does not end `_day<N>` or names no known product.
    """
    slug, _, day = prefix.rpartition("_day")
    if not day.isdigit() or slug not in PRODUCT_NAMES:
        msg = f"prefix_label: {prefix!r} is not a known product prefix"
        raise ValueError(msg)
    return f"{PRODUCT_NAMES[slug]} day {day}"


def blend_label(*, arm: str) -> str:
    """Name a blend or its permutation control by listing its products, leads included.

    Args:
        arm: `blend_p4a`, `blend_p4b`, or either with `_control`.

    Returns:
        For example `ENS mean day 1 + ICON-EU day 1 + IFS 0.25° day 1`; a control shuffles the two
        products after the first, which its label says.

    Raises:
        KeyError: If `arm` names no blend.
    """
    control = arm.endswith("_control")
    ens, first, second = BLEND_ARMS[arm.removesuffix("_control")]
    names = [
        prefix_label(prefix=ens),
        *(
            f"{'shuffled ' if control else ''}{prefix_label(prefix=other)}"
            for other in (first, second)
        ),
    ]
    return " + ".join(names)


def arm_label(*, arm: str) -> str:
    """Name any arm for a chart's row label.

    Args:
        arm: An arm in the saved losses.

    Returns:
        The label: a blend lists its products, a baseline says it uses no weather.

    Raises:
        ValueError: If `arm` is none of the arm kinds the study fits or scores.
    """
    if arm.startswith("blend_"):
        return blend_label(arm=arm)
    if arm == "climatology":
        return "Climatology (no weather)"
    for slug, name in BASELINE_NAMES.items():
        if arm.startswith(f"{slug}_day"):
            return f"{name} day {arm.removeprefix(f'{slug}_day')} (no weather)"
    return prefix_label(prefix=arm)


def short_blend_label(*, arm: str) -> str:
    """Name a blend compactly for the leaderboard, whose rows are one or two text lines tall.

    Args:
        arm: `blend_p4a`, `blend_p4b`, or either with `_control`.

    Returns:
        For example `P4a blend: ENS + ICON-EU + IFS 0.25°`, or `P4a control: ENS + shuffled
        ICON-EU + IFS 0.25°`.
    """
    name = arm.removesuffix("_control").removeprefix("blend_").upper().replace("P4A", "P4a")
    name = name.replace("P4B", "P4b")
    _, first, second = BLEND_ARMS[arm.removesuffix("_control")]
    control = arm.endswith("_control")
    first_name = PRODUCT_NAMES[first.rpartition("_day")[0]]
    second_name = PRODUCT_NAMES[second.rpartition("_day")[0]]
    if control:
        return f"{name} control: ENS + shuffled {first_name} + {second_name}"
    return f"{name} blend: ENS + {first_name} + {second_name}"


# --- Loading ------------------------------------------------------------------------------------


def check_anonymised(*, frame: pl.DataFrame, domain: DomainType) -> None:
    """Raise unless every site label in `frame` is one of the technology's anonymised labels.

    Args:
        frame: Saved losses or predictions, carrying `site`.
        domain: `solar` or `wind`.

    Raises:
        ValueError: If any site is not one of `SITES[domain]`.
    """
    unexpected = set(frame["site"].unique().to_list()) - set(SITES[domain])
    if unexpected:
        msg = f"{domain}: {len(unexpected)} site label(s) are not anonymised labels"
        raise ValueError(msg)


def load(*, input_dir: Path, domain: DomainType) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read one technology's saved losses and predictions.

    Args:
        input_dir: The directory `nwp_forecast_comparison.py` wrote to.
        domain: `solar` or `wind`.

    Returns:
        The per-row losses and the per-row predictions, after the anonymisation check.
    """
    losses = pl.read_parquet(losses_path(output_dir=input_dir, domain=domain))
    predictions = pl.read_parquet(predictions_path(output_dir=input_dir, domain=domain))
    check_anonymised(frame=losses, domain=domain)
    check_anonymised(frame=predictions, domain=domain)
    return losses, predictions


def by_setting(*, losses: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split saved losses by hyperparameter setting.

    Args:
        losses: Saved per-row losses carrying `setting`.

    Returns:
        Each setting's losses, keyed by setting name, in `SETTINGS` order.
    """
    return {name: losses.filter(pl.col("setting") == name) for name in SETTINGS}


def scope_text(*, losses: pl.DataFrame, domain: DomainType) -> str:
    """Return the sentence naming a chart's technology and period, months only, no dates.

    Args:
        losses: Saved per-row losses carrying `time`.
        domain: `solar` or `wind`.

    Returns:
        For example `Six solar farms ... scored hours from December 2024 to May 2026.`
    """
    first, last = losses["time"].min(), losses["time"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        msg = "scope_text: `time` holds no datetimes"
        raise TypeError(msg)
    return (
        f"Scored hours of {TECHNOLOGY_NAMES[domain]} in Lincolnshire, "
        f"{first:%B %Y} to {last:%B %Y}."
    )


def padded_domain(*, low: float, high: float, include_zero: bool) -> tuple[float, float]:
    """Return an x range around `low` and `high`, padded and rounded out to a multiple of 0.1.

    Args:
        low: The smallest value to show.
        high: The largest value to show.
        include_zero: Whether the range must contain zero.

    Returns:
        The range.
    """
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    pad = max(0.1 * (high - low), 0.05)
    return (math.floor((low - pad) * 10) / 10, math.ceil((high + pad) * 10) / 10)


# --- Chart 1: headline contrasts ------------------------------------------------------------------


class ContrastSpec(NamedTuple):
    """One planned contrast: its ID and the two arms it subtracts."""

    identifier: str
    treatment: str
    reference: str


PLANNED_CONTRASTS: Final[tuple[ContrastSpec, ...]] = (
    ContrastSpec("P1a", "ukv_day1", "ens_mean_day1"),
    ContrastSpec("P1b", "ukv_day1", "ens_mean_day0"),
    ContrastSpec("P2a", "icon_eu_day1", "ens_mean_day1"),
    ContrastSpec("P2b", "icon_eu_day1", "ens_mean_day0"),
    ContrastSpec("P3", "gefs_mean_day1", "ens_mean_day1"),
    ContrastSpec("P4a", "blend_p4a", "ens_mean_day1"),
    ContrastSpec("P4b", "blend_p4b", "ens_mean_day1"),
)
"""The planned contrasts of the plan, in the report's order. P1a and P2a are the upper side of
each bracket (product minus ENS day 1), P1b and P2b the lower side (product minus ENS day 0)."""

EXPLORATORY_CONTRASTS: Final[tuple[ContrastSpec, ...]] = (
    ContrastSpec("X-p4a-control", "blend_p4a_control", "ens_mean_day1"),
    ContrastSpec("X-p4b-control", "blend_p4b_control", "ens_mean_day1"),
)
"""The blends' permutation controls against ENS alone, which the report labels exploratory."""

GUARD_CONTRASTS: Final[tuple[ContrastSpec, ...]] = (
    ContrastSpec("P4a guard", "blend_p4a", "blend_p4a_control"),
    ContrastSpec("P4b guard", "blend_p4b", "blend_p4b_control"),
)
"""Each blend against its own permutation control, planned."""


def contrast_label(*, spec: ContrastSpec, compact: bool = False) -> str:
    """Name a contrast in a row label: the first arm minus the second, with the ID where it helps.

    Args:
        spec: The contrast.
        compact: Whether to name a blend by its short label, for a panel whose rows share a figure
            with the blends' full product lists; a blend against its own control reads `P4a blend
            minus its own control`.

    Returns:
        For example `P1a: UKV day 1 minus ENS mean day 1`.
    """
    if not compact:
        return (
            f"{spec.identifier}: {arm_label(arm=spec.treatment)} minus "
            f"{arm_label(arm=spec.reference)}"
        )
    if spec.reference == f"{spec.treatment}_control":
        return f"{spec.identifier[:3]} blend minus its own control"
    return f"{short_blend_label(arm=spec.treatment)} minus {arm_label(arm=spec.reference)}"


def contrast_rows(
    *,
    losses_by_setting: dict[str, pl.DataFrame],
    specs: Sequence[ContrastSpec],
    planned: bool,
    compact: bool = False,
) -> pl.DataFrame:
    """Compute each contrast's interval at each setting, skipping any whose arms are absent.

    Args:
        losses_by_setting: Saved losses split by setting.
        specs: The contrasts.
        planned: Whether every contrast in `specs` is planned.
        compact: Passed to `contrast_label`.

    Returns:
        One row per (contrast, setting) with `label`, `family`, `difference`, `lower_95`,
        `upper_95` in points of capacity, `condition` (the setting's display name) and `planned`.
    """
    records = []
    for spec in specs:
        for setting, losses in losses_by_setting.items():
            if not arms_present(losses=losses, arms=(spec.treatment, spec.reference)):
                continue
            interval = difference(losses=losses, treatment=spec.treatment, reference=spec.reference)
            records.append(
                {
                    "label": contrast_label(spec=spec, compact=compact),
                    "family": "weather model",
                    "difference": interval["difference"] * PERCENTAGE_POINTS,
                    "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                    "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
                    "condition": SETTING_NAMES[setting],
                    "planned": planned,
                }
            )
    return pl.DataFrame(
        records,
        schema={
            "label": pl.String,
            "family": pl.String,
            "difference": pl.Float64,
            "lower_95": pl.Float64,
            "upper_95": pl.Float64,
            "condition": pl.String,
            "planned": pl.Boolean,
        },
    )


def _contrast_panel(
    *, rows: pl.DataFrame, panel_title: str, x_title: str = DIFFERENCE_TITLE
) -> alt.LayerChart | alt.VConcatChart:
    """Draw contrast rows as dots and intervals with both settings in their own colours."""
    x_domain = padded_domain(
        low=float(rows["lower_95"].min()),  # ty: ignore[invalid-argument-type]
        high=float(rows["upper_95"].max()),  # ty: ignore[invalid-argument-type]
        include_zero=True,
    )
    return interval_panel(
        rows=rows,
        x_domain=x_domain,
        x_title=x_title,
        zero_label="same error",
        better_label="first product better",
        conditions=list(SETTING_NAMES.values()),
        condition_colours=SETTING_COLOURS,
        condition_title="XGBoost hyperparameter setting",
        panel_title=panel_title,
        family_key=False,
        figure_planning=planning(rows=[rows]),
    )


def headline(*, losses: pl.DataFrame, domain: DomainType, title: str) -> alt.VConcatChart | None:
    """Draw the planned contrasts P1a to P4b with 95% intervals at both settings.

    Args:
        losses: Saved per-row losses.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure, or None where no planned contrast has its arms in the losses.
    """
    rows = contrast_rows(
        losses_by_setting=by_setting(losses=losses), specs=PLANNED_CONTRASTS, planned=True
    )
    if rows.is_empty():
        return None
    return figure(
        panels=[_contrast_panel(rows=rows, panel_title="Planned contrasts")],
        number=FIGURE_NUMBERS[(domain, "headline")],
        title=title,
        subtitle=[
            (
                "Difference in mean absolute error between two XGBoost models, each given one "
                "forecast product, first product minus second, in points of capacity. "
                "Negative means the first product forecasts better. "
                f"{SHARED_ROWS_NOTE} {CAPACITY_NOTE}"
            ),
            (
                "P1a, P2a, P3, P4a and P4b compare at the ENS mean's day-1 lead; P1b and P2b "
                "compare with the ENS mean at day 0, so a product's day-1 error lies between the "
                "two ENS errors when it is bracketed. "
                f"{DOTS_NOTE}"
            ),
            scope_text(losses=losses, domain=domain),
        ],
        figure_planning=planning(rows=[rows]),
    )


# --- Chart 2: leaderboard -------------------------------------------------------------------------


def leaderboard_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return every arm's absolute error and interval at the primary setting, best first.

    Args:
        losses: Saved per-row losses.

    Returns:
        One row per arm with `label`, `family`, `condition`, `value`, `lower_95`, `upper_95` in
        percent of capacity, sorted by `value`.
    """
    primary = by_setting(losses=losses)["primary"]
    arms = sorted(arm for arm in primary["arm"].unique().to_list() if on_leaderboard(arm=arm))
    board = leaderboard(losses=primary, arms=arms)
    return (
        board.with_columns(
            pl.col("value", "lower_95", "upper_95") * PERCENTAGE_POINTS,
            label=pl.col("arm").map_elements(
                lambda arm: (
                    short_blend_label(arm=arm)
                    if arm.startswith("blend_")
                    else "ENS mean day 0 (a bracket side, not a product)"
                    if arm == "ens_mean_day0"
                    else arm_label(arm=arm)
                ),
                return_dtype=pl.String,
            ),
            family=pl.lit("weather model"),
            condition=pl.when(pl.col("arm").map_elements(_is_baseline, return_dtype=pl.Boolean))
            .then(pl.lit(BASELINE_CONDITION))
            .otherwise(pl.lit(FORECAST_CONDITION)),
        )
        .sort("value")
        .select("label", "family", "condition", "value", "lower_95", "upper_95")
    )


def on_leaderboard(*, arm: str) -> bool:
    """Whether the leaderboard draws `arm`: the day-1 forecasts, ENS at day 0, and two baselines.

    Days 2 and 3 are on the lead-day chart, and the other no-weather baselines are far above the
    products and would stretch the axis.

    Args:
        arm: An arm in the saved losses.

    Returns:
        True for a blend, a day-1 arm, `ens_mean_day0`, `climatology` and `smart_persistence_day1`.
    """
    if is_baseline_arm(arm=arm):
        return arm in ("climatology", "smart_persistence_day1")
    return arm.startswith("blend_") or arm == "ens_mean_day0" or arm.endswith("_day1")


def _is_baseline(arm: str) -> bool:
    """Whether `arm` names a no-weather baseline; `map_elements` passes the value positionally."""
    return is_baseline_arm(arm=arm)


def leaderboard_figure(*, losses: pl.DataFrame, domain: DomainType, title: str) -> alt.VConcatChart:
    """Draw every arm's own mean absolute error with its 95% interval, best first.

    Args:
        losses: Saved per-row losses.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure.
    """
    rows = leaderboard_rows(losses=losses)
    x_domain = padded_domain(
        low=float(rows["lower_95"].min()),  # ty: ignore[invalid-argument-type]
        high=float(rows["upper_95"].max()),  # ty: ignore[invalid-argument-type]
        include_zero=False,
    )
    panel = leaderboard_panel(
        rows=rows,
        x_domain=x_domain,
        x_title=MAE_TITLE,
        conditions=[FORECAST_CONDITION, BASELINE_CONDITION],
        condition_title="Kind of forecast",
        solid=True,
        row_step_px=LEADERBOARD_ROW_PX,
    )
    return figure(
        panels=[panel],
        number=FIGURE_NUMBERS[(domain, "leaderboard")],
        title=title,
        subtitle=[
            (
                "Each row is one forecast's own mean absolute error, as a percentage of "
                "capacity, on the hours every forecast is scored on, with no weather forecast "
                "given for the baselines. Primary XGBoost setting. Smaller is better."
            ),
            (
                "Only day-1 forecasts, ENS at day 0, climatology and smart persistence at day 1 "
                "are shown; the other days are in the lead-day figure. P4a blends ENS with the "
                "day-1 ICON-EU and IFS 0.25°; P4b uses their day-2 forecasts. A control gives "
                "an XGBoost model the same columns with the shuffled products' values moved "
                "among matched hours. Overlapping intervals here can still hide a significant "
                f"paired difference (Figure {FIGURE_NUMBERS[(domain, 'headline')]}). {DOTS_NOTE}"
            ),
            f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE}",
        ],
        figure_planning=None,
    )


# --- Chart 3: the models work ---------------------------------------------------------------------

EXAMPLE_ARM: Final[str] = "ens_mean_day1"
"""The arm whose out-of-fold forecasts the "models work" chart draws."""

MEASURED_COLOUR: Final[str] = ocf.DATA_GREEN
FORECAST_COLOUR: Final[str] = ocf.DATA_BLUE


def measured_and_forecast(
    *, losses: pl.DataFrame, predictions: pl.DataFrame, arm: str
) -> pl.DataFrame:
    """Return one arm's seed-averaged out-of-fold forecast and the measured output, by capacity.

    The measured output is the capped forecast minus the capped signed error, so the chart reads
    only the two saved files.

    Args:
        losses: Saved per-row losses.
        predictions: Saved per-row predictions.
        arm: The arm, at the primary setting.

    Returns:
        One row per (site, time) with `measured` and `forecast`, as fractions of capacity.
    """
    keys = ["arm", "setting", "site", "time", "seed"]
    selected = (
        predictions.filter(pl.col("arm") == arm, pl.col("setting") == "primary")
        .join(
            losses.select(*keys, "signed_error_capped_mw", "effective_capacity_mw"),
            on=keys,
        )
        .with_columns(measured_mw=pl.col("prediction_capped_mw") - pl.col("signed_error_capped_mw"))
    )
    return (
        selected.group_by("site", "time")
        .agg(
            pl.col("prediction_capped_mw").mean(),
            pl.col("measured_mw").first(),
            pl.col("effective_capacity_mw").first(),
        )
        .select(
            "site",
            "time",
            measured=pl.col("measured_mw") / pl.col("effective_capacity_mw"),
            forecast=pl.col("prediction_capped_mw") / pl.col("effective_capacity_mw"),
        )
        .sort("site", "time")
    )


def choose_week(*, series: pl.DataFrame, domain: DomainType) -> datetime:
    """Choose one week by rule from measured output alone, among weeks with full coverage.

    Solar takes the week whose daily mean output varies most from day to day; wind the week with
    the largest mean hour-to-hour change in output. A generator covers a day with at least 4
    scored hours for solar and 12 for wind, and a week qualifies when every generator covers all
    seven of its days.

    Args:
        series: One row per (site, time) with `measured`.
        domain: `solar` or `wind`.

    Returns:
        The week's Monday, at midnight.

    Raises:
        ValueError: If no week is covered on all seven days by every generator.
    """
    minimum = 4 if domain == "solar" else 12
    frame = series.sort("site", "time").with_columns(
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
    if covered.is_empty():
        msg = f"choose_week: no week has every {domain} generator covered on all seven days"
        raise ValueError(msg)
    scores = (
        frame.group_by("week", "date")
        .agg(daily=pl.col("measured").mean())
        .group_by("week")
        .agg(score=pl.col("daily").std())
        if domain == "solar"
        else frame.group_by("week").agg(score=pl.col("step").mean())
    )
    chosen = scores.join(covered, on="week").sort("score", descending=True)["week"][0]
    if not isinstance(chosen, datetime):
        msg = "choose_week: `time` holds no datetimes"
        raise TypeError(msg)
    return chosen


def line_key(*, labels: Sequence[str], colours: Sequence[str]) -> alt.LayerChart:
    """Draw a one-row key of short line segments above a chart.

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
            "label": list(labels),
            "colour": list(colours),
            "x": [index * slot for index in range(len(labels))],
            "x2": [index * slot + 18 for index in range(len(labels))],
        }
    )
    segments = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2.5, aria=False)
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


def models_work(
    *, losses: pl.DataFrame, predictions: pl.DataFrame, domain: DomainType, title: str
) -> tuple[alt.VConcatChart, str]:
    """Draw one chosen week: measured output and the day-1 ENS-mean forecast, per generator.

    Args:
        losses: Saved per-row losses.
        predictions: Saved per-row predictions.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure, and the week's month and year for the page's text.
    """
    series = measured_and_forecast(losses=losses, predictions=predictions, arm=EXAMPLE_ARM)
    week = choose_week(series=series, domain=domain)
    names = ["Measured output", "Day-1 forecast"]
    colours = [MEASURED_COLOUR, FORECAST_COLOUR]
    long = pl.concat(
        [
            series.select(
                "site", "time", series=pl.lit(names[0]), value=pl.col("measured") * 100.0
            ),
            series.select(
                "site", "time", series=pl.lit(names[1]), value=pl.col("forecast") * 100.0
            ),
        ]
    )
    hours = pl.datetime_range(
        week, week + timedelta(days=7), interval="1h", closed="left", eager=True
    )
    grid = pl.DataFrame({"time": hours}).join(pl.DataFrame({"series": names}), how="cross")
    panels = []
    for index, site in enumerate(SITES[domain]):
        drawn = (
            grid.join(
                long.filter(pl.col("site") == site).drop("site"), on=["time", "series"], how="left"
            )
            .with_columns(elapsed=(pl.col("time") - week).dt.total_minutes() / (60 * 24))
            .sort("series", "time")
        )
        last = index == len(SITES[domain]) - 1
        panels.append(
            alt.Chart(drawn)
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
                        title="Day of the chosen week" if last else None,
                    ),
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
            .properties(width=CONTENT_WIDTH_PX - 120, height=TIME_PANEL_HEIGHT_PX)
        )
    rule = (
        "the week whose daily mean output varies most from day to day"
        if domain == "solar"
        else "the week with the largest mean hour-to-hour change in output"
    )
    chart = figure(
        panels=[line_key(labels=names, colours=colours), alt.vconcat(*panels, spacing=4)],
        number=FIGURE_NUMBERS[(domain, "models_work")],
        title=title,
        subtitle=[
            (
                "Hourly output as a percentage of capacity, measured and as forecast out of "
                "fold by the XGBoost model given the ENS ensemble mean at day 1, averaged over "
                "its fitting seeds. Primary XGBoost setting. A gap in a line is an hour outside "
                "the scored rows."
            ),
            (
                f"The week is chosen by rule from measured output alone: {rule}. "
                "Each vertical axis runs from 0 to 110% of the generator's own capacity."
            ),
        ],
        figure_planning=None,
    )
    return chart, f"{week:%B %Y}"


# --- Chart 4: error by lead day -------------------------------------------------------------------

ENSEMBLE_SERIES: Final[str] = "Ensemble mean (ENS or GEFS)"
PREVIOUS_RUNS_SERIES: Final[str] = "Previous Runs product (single run)"

SERIES_COLOURS: Final[dict[str, str]] = {
    ENSEMBLE_SERIES: ocf.DATA_BLUE,
    PREVIOUS_RUNS_SERIES: ocf.BRAND_ORANGE,
}
"""Each group of products' colour. Nine single products need more colours than the brand palette
holds apart under colour-vision deficiency, so colour marks the group and every product's name is
written beside its last point."""

LEAD_BAND_COLOURS: Final[tuple[str, str]] = (ocf.DATA_BLUE_LIGHT, ocf.DATA_SKY_LIGHT)
"""The shading of ENS's day-0 and day-1 intervals."""

X_MAX_DAYS: Final[float] = 4.3
"""The lead-day chart's right edge, leaving room for the product names beside day 3."""

DODGE_DAYS: Final[float] = 0.03
"""Horizontal spacing between series at one lead day, in days, so intervals do not overprint."""


def spread_labels(*, values: Sequence[float], min_gap: float) -> list[float]:
    """Move label positions apart so none sit closer than `min_gap`, each moving little.

    Neighbouring labels that are too close push each other apart by half the shortfall each,
    repeatedly, so a cluster spreads evenly around where its labels wanted to be.

    Args:
        values: Each label's wanted position.
        min_gap: The smallest distance between two labels' positions.

    Returns:
        Each label's position, in the input order.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    placed = [float(values[index]) for index in order]
    for _ in range(200):
        moved = False
        for low, high in zip(range(len(placed) - 1), range(1, len(placed)), strict=True):
            shortfall = min_gap - (placed[high] - placed[low])
            if shortfall > 1e-9:
                placed[low] -= shortfall / 2
                placed[high] += shortfall / 2
                moved = True
        if not moved:
            break
    result = [0.0] * len(values)
    for position, index in enumerate(order):
        result[index] = placed[position]
    return result


def lead_series_name(*, arm: str) -> str | None:
    """Name the series an arm belongs in on the lead-day chart.

    Args:
        arm: An arm in the saved losses.

    Returns:
        The series' key in `SERIES_COLOURS` for a product read at a whole day, or None for a
        baseline, a blend, a control member and any arm the chart leaves out.
    """
    slug, _, day = arm.rpartition("_day")
    if not day.isdigit() or slug == "ens_control":
        return None
    if slug in ("ens_mean", "gefs_mean"):
        return ENSEMBLE_SERIES
    return PREVIOUS_RUNS_SERIES if slug in PRODUCT_NAMES else None


def lead_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each product's error and interval at each lead day, at the primary setting.

    Args:
        losses: Saved per-row losses.

    Returns:
        One row per arm with `series`, `product` (the name written beside its last point), `day`,
        `value`, `lower_95` and `upper_95` in percent of capacity.
    """
    primary = by_setting(losses=losses)["primary"]
    arms = [arm for arm in sorted(primary["arm"].unique().to_list()) if lead_series_name(arm=arm)]
    days_held = Counter(arm.rpartition("_day")[0] for arm in arms)
    arms = [arm for arm in arms if days_held[arm.rpartition("_day")[0]] >= 3]
    board = leaderboard(losses=primary, arms=arms)
    return (
        board.with_columns(
            pl.col("value", "lower_95", "upper_95") * PERCENTAGE_POINTS,
            series=pl.col("arm").map_elements(
                lambda arm: lead_series_name(arm=arm), return_dtype=pl.String
            ),
            product=pl.col("arm").map_elements(
                lambda arm: PRODUCT_NAMES[arm.rpartition("_day")[0]], return_dtype=pl.String
            ),
            day=pl.col("arm").str.extract(r"_day(\d+)$", 1).cast(pl.Int64),
        )
        .sort("series", "product", "day")
        .select("arm", "series", "product", "day", "value", "lower_95", "upper_95")
    )


def by_lead_day(*, losses: pl.DataFrame, domain: DomainType, title: str) -> alt.VConcatChart | None:
    """Draw error against lead day, with ENS's day-0 and day-1 intervals shaded.

    Args:
        losses: Saved per-row losses.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure, or None where the ENS mean's day-0 and day-1 arms are absent.
    """
    rows = lead_rows(losses=losses)
    ens = rows.filter(pl.col("product") == PRODUCT_NAMES["ens_mean"])
    if not {0, 1} <= set(ens["day"].to_list()):
        return None
    names = [name for name in SERIES_COLOURS if name in set(rows["series"].to_list())]
    products = rows["product"].unique(maintain_order=True).to_list()
    offsets = {
        product: (index - (len(products) - 1) / 2) * DODGE_DAYS
        for index, product in enumerate(products)
    }
    # Each product is drawn as its own line, coloured by its group; the key names the group.
    drawn = rows.with_columns(
        x=pl.col("day") + pl.col("product").replace_strict(offsets, return_dtype=pl.Float64),
        line=pl.col("product"),
    )
    low = float(rows["lower_95"].min())  # ty: ignore[invalid-argument-type]
    high = float(rows["upper_95"].max())  # ty: ignore[invalid-argument-type]
    y_domain = padded_domain(low=low, high=high, include_zero=False)
    x_scale = alt.Scale(domain=[-0.5, X_MAX_DAYS], nice=False)
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=[SERIES_COLOURS[name] for name in names]),
        legend=None,
    )
    x_axis = alt.Axis(
        values=[0, 1, 2, 3],
        labelExpr="'Day ' + datum.value",
        grid=False,
        title="Lead day (ENS: 24-hour band; other products: whole-day lead)",
    )
    y = alt.Y(
        "value:Q",
        scale=alt.Scale(domain=list(y_domain), nice=False),
        axis=alt.Axis(title=MAE_TITLE),
    )
    bands = pl.DataFrame(
        {
            "day": [0, 1],
            "lower": ens.sort("day")["lower_95"].to_list()[:2],
            "upper": ens.sort("day")["upper_95"].to_list()[:2],
            "colour": list(LEAD_BAND_COLOURS),
            "label": ["ENS mean day-0 interval", "ENS mean day-1 interval"],
            "x0": [-0.5, -0.5],
            "x1": [X_MAX_DAYS, X_MAX_DAYS],
            "y_text": [float(ens.filter(pl.col("day") == day)["lower_95"][0]) for day in (0, 1)],
        }
    )
    shading = (
        alt.Chart(bands)
        .mark_rect(opacity=0.45, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x0:Q", scale=x_scale, axis=x_axis),
            x2="x1:Q",
            y=alt.Y("lower:Q", scale=alt.Scale(domain=list(y_domain), nice=False)),
            y2="upper:Q",
            color=alt.Color("colour:N", scale=None),
        )
    )
    band_text = (
        alt.Chart(bands)
        .mark_text(align="left", baseline="top", dx=4, dy=2, color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x0:Q", scale=x_scale),
            y=alt.Y("y_text:Q", scale=alt.Scale(domain=list(y_domain), nice=False)),
            text="label:N",
        )
    )
    lines = (
        alt.Chart(drawn)
        .mark_line(strokeWidth=1.5, aria=False)
        .encode(x=alt.X("x:Q", scale=x_scale, axis=x_axis), y=y, color=colour, detail="line:N")  # ty: ignore[unresolved-attribute]
    )
    rules = (
        alt.Chart(drawn)
        .mark_rule(strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("lower_95:Q", scale=alt.Scale(domain=list(y_domain), nice=False)),
            y2="upper_95:Q",
            color=colour,
        )
    )
    points = (
        alt.Chart(drawn)
        .mark_point(filled=True, size=50, opacity=1, aria=False)
        .encode(x=alt.X("x:Q", scale=x_scale, axis=x_axis), y=y, color=colour)  # ty: ignore[unresolved-attribute]
    )
    ends = drawn.sort("day").group_by("product", maintain_order=True).last()
    ends = ends.with_columns(
        label_y=pl.Series(
            spread_labels(values=ends["value"].to_list(), min_gap=(y_domain[1] - y_domain[0]) / 24)
        )
    )
    end_text = (
        alt.Chart(ends)
        .mark_text(align="left", dx=12, fontSize=10, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("label_y:Q", scale=alt.Scale(domain=list(y_domain), nice=False)),
            text="product:N",
            color=colour,
        )
    )
    panel = alt.LayerChart(
        layer=[shading, band_text, lines, rules, points, end_text],
        width=CONTENT_WIDTH_PX - 100,
        height=LEAD_PANEL_HEIGHT_PX,
    )
    return figure(
        panels=[
            line_key(labels=names, colours=[SERIES_COLOURS[name] for name in names]),
            panel,
        ],
        number=FIGURE_NUMBERS[(domain, "by_lead_day")],
        title=title,
        subtitle=[
            (
                "Mean absolute error of an XGBoost model given each forecast product at each "
                "lead day, as a percentage of capacity. Primary XGBoost setting. Smaller is "
                "better. Shaded bands: the 95% intervals of the ENS mean's day-0 and day-1 "
                "errors; a whole-day product at day 1 lies between them when it is bracketed."
            ),
            (
                "Dot: estimate. Line: 95% interval from resampling whole months and a fitting "
                "seed. Products at one day are drawn side by side, and the name of each is "
                "written beside its last point. Products with a day-1 forecast only are left "
                f"out; the day-1 leaderboard shows them. {SHARED_ROWS_NOTE}"
            ),
            f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE}",
        ],
        figure_planning=None,
    )


# --- Chart 5: blends -----------------------------------------------------------------------------


def blend_absolute_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return ENS alone, both blends and both controls, with absolute errors at the primary setting.

    Args:
        losses: Saved per-row losses.

    Returns:
        One row per arm with `label`, `family`, `condition`, `value`, `lower_95` and `upper_95` in
        percent of capacity, in the order ENS alone, then each blend followed by its control.
    """
    primary = by_setting(losses=losses)["primary"]
    arms = ["ens_mean_day1", "blend_p4a", "blend_p4a_control", "blend_p4b", "blend_p4b_control"]
    board = leaderboard(losses=primary, arms=arms)
    labels = {
        "ens_mean_day1": arm_label(arm="ens_mean_day1"),
        **{arm: f"{blend_label(arm=arm)} ({_blend_tag(arm=arm)})" for arm in arms[1:]},
    }
    order = {arm: index for index, arm in enumerate(arms)}
    return (
        board.with_columns(
            pl.col("value", "lower_95", "upper_95") * PERCENTAGE_POINTS,
            label=pl.col("arm").replace_strict(labels, return_dtype=pl.String),
            family=pl.lit("weather model"),
            condition=pl.lit(SETTING_NAMES["primary"]),
            order=pl.col("arm").replace_strict(order, return_dtype=pl.Int64),
        )
        .sort("order")
        .select("label", "family", "condition", "value", "lower_95", "upper_95")
    )


def _blend_tag(*, arm: str) -> str:
    """Return `P4a blend`, `P4a control`, `P4b blend` or `P4b control` for a blend arm."""
    plan = "P4a" if "p4a" in arm else "P4b"
    return f"{plan} control" if arm.endswith("_control") else f"{plan} blend"


def blends(*, losses: pl.DataFrame, domain: DomainType, title: str) -> alt.VConcatChart | None:
    """Draw ENS alone, each blend and each control, then the paired contrasts among them.

    Args:
        losses: Saved per-row losses.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure, or None where the blends are absent from the losses.
    """
    primary = by_setting(losses=losses)["primary"]
    if not arms_present(losses=primary, arms=("blend_p4a", "blend_p4b", "ens_mean_day1")):
        return None
    absolute = blend_absolute_rows(losses=losses)
    x_domain = padded_domain(
        low=float(absolute["lower_95"].min()),  # ty: ignore[invalid-argument-type]
        high=float(absolute["upper_95"].max()),  # ty: ignore[invalid-argument-type]
        include_zero=False,
    )
    level_panel = leaderboard_panel(
        rows=absolute,
        x_domain=x_domain,
        x_title=MAE_TITLE,
        conditions=list(SETTING_NAMES.values()),
        condition_title="XGBoost hyperparameter setting",
        solid=True,
        keys=False,
        panel_title="Each forecast's own error (primary XGBoost setting)",
        row_step_px=44,
    )
    losses_by_setting = by_setting(losses=losses)
    contrasts = pl.concat(
        [
            contrast_rows(
                losses_by_setting=losses_by_setting,
                specs=[spec for spec in PLANNED_CONTRASTS if spec.identifier in ("P4a", "P4b")],
                planned=True,
                compact=True,
            ),
            contrast_rows(
                losses_by_setting=losses_by_setting,
                specs=GUARD_CONTRASTS,
                planned=True,
                compact=True,
            ),
            contrast_rows(
                losses_by_setting=losses_by_setting,
                specs=EXPLORATORY_CONTRASTS,
                planned=False,
                compact=True,
            ),
        ]
    )
    contrast_panel = _contrast_panel(
        rows=contrasts,
        panel_title="Paired differences: blend minus ENS alone, blend minus its own control",
    )
    return figure(
        panels=[level_panel, contrast_panel],
        number=FIGURE_NUMBERS[(domain, "blends")],
        title=title,
        subtitle=[
            (
                "XGBoost models given the ENS mean alone, or ENS plus two more products. "
                "A control has the same columns as its blend, with the two other products' "
                "weather shuffled among hours of the same site, month and hour of day, so it "
                "carries no real information from them. Points of capacity; negative means the "
                "first forecast in a row is better."
            ),
            (
                "P4a uses the day-1 forecasts of ICON-EU and IFS 0.25°, P4b their day-2 "
                f"forecasts. {SHARED_ROWS_NOTE} {DOTS_NOTE}"
            ),
            f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE}",
        ],
        figure_planning=planning(rows=[contrasts]),
    )


# --- Chart 6: one generator at a time ---------------------------------------------------------

GENERATOR_CONDITIONS: Final[dict[str, str]] = {
    "P1a": "UKV day 1 minus ENS day 1",
    "P2a": "ICON-EU day 1 minus ENS day 1",
    "P4b": "P4b blend minus ENS day 1",
}
"""Each per-generator contrast's key text; the contrasts are the report's `GENERATOR_CONTRASTS`."""

GENERATOR_COLOURS: Final[tuple[str, str, str]] = (ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE)


def per_generator(
    *, losses: pl.DataFrame, domain: DomainType, title: str
) -> alt.VConcatChart | None:
    """Draw P1a, P2a and P4b one generator at a time, each with its 95% interval.

    Args:
        losses: Saved per-row losses.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure, or None where the contrasts' arms are absent from the losses.
    """
    primary = by_setting(losses=losses)["primary"]
    records = []
    for site in SITES[domain]:
        site_losses = primary.filter(pl.col("site") == site)
        for identifier, (treatment, reference) in GENERATOR_CONTRASTS.items():
            if not arms_present(losses=site_losses, arms=(treatment, reference)):
                continue
            interval = difference(losses=site_losses, treatment=treatment, reference=reference)
            records.append(
                {
                    "label": f"Generator {site}",
                    "family": "weather model",
                    "difference": interval["difference"] * PERCENTAGE_POINTS,
                    "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                    "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
                    "condition": GENERATOR_CONDITIONS[identifier],
                }
            )
    if not records:
        return None
    rows = pl.DataFrame(records)
    x_domain = padded_domain(
        low=float(rows["lower_95"].min()),  # ty: ignore[invalid-argument-type]
        high=float(rows["upper_95"].max()),  # ty: ignore[invalid-argument-type]
        include_zero=True,
    )
    panel = interval_panel(
        rows=rows,
        x_domain=x_domain,
        x_title=DIFFERENCE_TITLE,
        zero_label="same error",
        better_label="first product better",
        conditions=list(GENERATOR_CONDITIONS.values()),
        condition_colours=GENERATOR_COLOURS,
        condition_title="Contrast at one generator",
        family_key=False,
    )
    return figure(
        panels=[panel],
        number=FIGURE_NUMBERS[(domain, "per_generator")],
        title=title,
        subtitle=[
            (
                "Difference in mean absolute error between two XGBoost models, first product "
                "minus second, in points of capacity, at each generator alone. Negative means "
                "the first product forecasts better. Primary XGBoost setting. The 95% interval "
                "resamples whole months and a fitting seed within one generator, so it does not "
                "cover differences between generators. All rows are exploratory."
            ),
            f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE} {SHARED_ROWS_NOTE}",
        ],
        figure_planning=None,
    )


# --- Output -------------------------------------------------------------------------------------

FIGURE_NUMBERS: Final[dict[tuple[DomainType, str], int]] = {
    ("solar", "headline"): 1,
    ("wind", "headline"): 2,
    ("solar", "models_work"): 3,
    ("wind", "models_work"): 4,
    ("solar", "per_generator"): 5,
    ("wind", "per_generator"): 6,
    ("solar", "leaderboard"): 7,
    ("wind", "leaderboard"): 8,
    ("solar", "by_lead_day"): 9,
    ("wind", "by_lead_day"): 10,
    ("solar", "blends"): 11,
    ("wind", "blends"): 12,
}
"""Each chart's figure number on the page, in the page's order: the headline pair opens the page."""

TITLES: Final[dict[tuple[DomainType, str], str]] = {
    ("solar", "headline"): (
        "For solar power, ENS beats UKV and ICON-EU at matched lead and GEFS at equal lead; "
        "a blend gains 0.35 points at an optimistic lead and none at a conservative one"
    ),
    ("wind", "headline"): (
        "For wind power, ENS beats UKV, ICON-EU is unresolved against ENS, and a blend "
        "lowers the error by 0.18 points even at a conservative lead"
    ),
    ("solar", "leaderboard"): (
        "At day 1 every forecast beats climatology (14.5%); ENS and IFS 0.25° have the lowest "
        "error of the single solar forecasts"
    ),
    ("wind", "leaderboard"): (
        "At day 1 every forecast beats climatology (18.5%) by more than 9 points; ENS, IFS 0.25° "
        "and ICON-EU have the lowest error of the single wind forecasts"
    ),
    ("solar", "models_work"): (
        "Out-of-fold day-1 ENS-mean forecasts follow the measured output at all six solar farms"
    ),
    ("wind", "models_work"): (
        "Out-of-fold day-1 ENS-mean forecasts follow the measured output at all three wind farms"
    ),
    ("solar", "per_generator"): (
        "At each of the six solar farms UKV and ICON-EU have a higher error than ENS at day 1"
    ),
    ("wind", "per_generator"): (
        "At two of the three wind farms UKV has a higher error than ENS at day 1, "
        "and at two the P4b blend has a lower error"
    ),
    ("solar", "by_lead_day"): (
        "Solar error rises with lead day for every forecast, and no product beats the ENS mean "
        "at matched lead"
    ),
    ("wind", "by_lead_day"): (
        "Wind error rises with lead day for every forecast, and no product beats the ENS mean "
        "at matched lead"
    ),
    ("solar", "blends"): (
        "For solar power a blend of ENS, ICON-EU and IFS 0.25° gains 0.35 points at an "
        "optimistic lead, but its control is itself worse than ENS alone"
    ),
    ("wind", "blends"): (
        "For wind power a blend of ENS, ICON-EU and IFS 0.25° lowers the error by 0.66 points at "
        "an optimistic lead and 0.18 points at a conservative lead, and its control does not"
    ),
}
"""Each chart's title, stating the finding for the products tested. Every number is in
`report.md`."""


def optimise(*, path: Path) -> None:
    """Optimise one SVG in place with `svgo`.

    Args:
        path: The SVG.

    Raises:
        subprocess.CalledProcessError: If `svgo` fails.
    """
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )


def draw_domain(
    *, input_dir: Path, domain: DomainType
) -> tuple[dict[str, alt.VConcatChart], str | None]:
    """Draw every chart of one technology that its saved losses can support.

    Args:
        input_dir: The directory `nwp_forecast_comparison.py` wrote to.
        domain: `solar` or `wind`.

    Returns:
        Each chart keyed by its name, and the chosen week's month and year.
    """
    losses, predictions = load(input_dir=input_dir, domain=domain)
    week_month: str | None = None
    charts: dict[str, alt.VConcatChart | None] = {
        "headline": headline(losses=losses, domain=domain, title=TITLES[(domain, "headline")]),
        "leaderboard": leaderboard_figure(
            losses=losses, domain=domain, title=TITLES[(domain, "leaderboard")]
        ),
        "by_lead_day": by_lead_day(
            losses=losses, domain=domain, title=TITLES[(domain, "by_lead_day")]
        ),
        "blends": blends(losses=losses, domain=domain, title=TITLES[(domain, "blends")]),
        "per_generator": per_generator(
            losses=losses, domain=domain, title=TITLES[(domain, "per_generator")]
        ),
    }
    work, week_month = models_work(
        losses=losses, predictions=predictions, domain=domain, title=TITLES[(domain, "models_work")]
    )
    charts["models_work"] = work
    return {name: chart for name, chart in charts.items() if chart is not None}, week_month


def main() -> int:
    """Draw every chart for both technologies and write them as SVG.

    Returns:
        The exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Saved losses' directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where SVGs are written.")
    parser.add_argument("--no-svgo", action="store_true", help="Skip the svgo optimisation.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        charts, week_month = draw_domain(input_dir=args.input_dir, domain=domain)
        for name, chart in charts.items():
            path = args.output_dir / f"nwp_forecast_{domain}_{name}.svg"
            chart.save(path)
            if not args.no_svgo:
                optimise(path=path)
            _LOG.info("wrote %s", path)
        sys.stdout.write(f"{domain}: example week in {week_month}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
