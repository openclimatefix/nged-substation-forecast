"""Draw the anonymised charts for the write-up on weather forecasts compared at matched leads.

One-off throwaway script for the charts of
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, reading what
`nwp_forecast_comparison.py` saved: `<domain>_losses.parquet` and `<domain>_predictions.parquet`.
Every interval is computed here from those per-row losses, with the same functions the report
uses (`difference`, `leaderboard`, `bracket` in `nwp_forecast_comparison.py`, which call
`studies.bootstrap`), so no refit is needed and a chart cannot disagree with the report.

Six charts per technology, each with its own title, subtitle, axis titles and key:

1. `leaderboard`: each product's own mean absolute error at every fitted lead day, one product per
   row, best day-1 error first, with climatology and day-1 smart persistence as dashed lines.
2. `headline`: the seven planned contrasts P1a to P4b with 95% intervals, at both settings.
3. `models_work`: one week of out-of-fold day-1 ENS-mean forecasts against measured output.
4. `by_lead_day`: error by lead day, with ENS's day-0 and day-1 intervals shaded.
5. `blends`: the two blends against ENS alone and against their permutation controls.
6. `per_generator`: the P1a, P2a and P4b contrasts at each generator.

Generators appear only as `A` to `F` and `W1` to `W3`, every error is a fraction of the
generator's own capacity, the time axes count days of the week rather than dates, and no data mark
writes its values into the SVG's accessibility text. The script stops if a site label is not one
of the anonymised labels.

Run it with `uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py --input-dir
DIR --extra-dir DIR --extra-dir DIR --output-dir DIR`, with one `--extra-dir` per extra-lead fit
folder. Each SVG is optimised with `npx svgo@4 --multipass --precision=1 --final-newline` unless
`--no-svgo` is given. Charts belong under `docs/studies/assets/` only once a
real report exists.
"""

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
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
    leaderboard,
    losses_path,
    predictions_path,
)
from studies.anonymise import SITE_LABELS, WIND_SITE_LABELS
from studies.charts import (
    CONTENT_WIDTH_PX,
    LABEL_WIDTH_PX,
    PLOT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
    planning,
    ticks,
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

CAPACITY_NOTE: Final[str] = (
    "Every error is a fraction of the generator's own capacity, taken as its 99th percentile of "
    "metered output."
)
SHARED_ROWS_NOTE: Final[str] = "Every product is scored on exactly the same hours."
DOTS_NOTE: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole months and a fitting seed."
)
MAE_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
DIFFERENCE_TITLE: Final[str] = "Difference in mean absolute error (points of capacity)"

SETTING_NAMES: Final[dict[str, str]] = {
    "primary": "Primary XGBoost setting",
    "sensitivity": "Sensitivity XGBoost setting",
}
"""Each hyperparameter setting's name in a chart key."""

SETTING_COLOURS: Final[tuple[str, str]] = (ocf.BRAND_ORANGE, ocf.DATA_BLUE)
"""The colour of the primary and the second setting, in that order."""

TIME_PANEL_HEIGHT_PX: Final[int] = 70
LEAD_PANEL_HEIGHT_PX: Final[int] = 300

PRODUCT_NAMES: Final[dict[str, str]] = {
    **{slug: name for name, slug in PRODUCT_SLUGS.items()},
    "gfs": "GFS (Open-Meteo)",
    "gfs_native": "GFS (native)",
    "ifs_single": "IFS HRES (9 km, Open-Meteo)",
    "ens_mean": "ENS mean",
    "ens_control": "ENS control member",
    "gefs_mean": "GEFS mean",
}
"""Each column prefix's product name, up to its `_day<N>` suffix. Two rows are NOAA GFS: the one
Open-Meteo serves (`gfs`, its GFS-SEAMLESS archive) and the one read from Dynamical.org's native
store (`gfs_native`); each name says its source, because no row mixes the two. `ifs_single` is
ECMWF IFS HRES read from Open-Meteo's Single Runs archive, a row of its own and never merged with
`ifs025` ("IFS 0.25°", Open-Meteo's Previous Runs), a coarser product and not another version of
it. The name states the source and the grid, as the two GFS names state their sources."""

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


PUBLISHED_DEVICE: Final[str] = "cpu"
"""The device of the published fits, which `nwp_forecast_comparison.py` ran on the CPU."""

EXTRA_DEVICE: Final[str] = "cuda"
"""The device of an extra-lead fit that carries no `device` column: `fit_extra_leads.py` always ran
on the GPU."""


def extra_arm_devices(*, extras: Sequence[pl.DataFrame]) -> dict[str, str]:
    """Map each arm the extra-lead folders hold to the device that fitted it.

    Args:
        extras: One losses frame per extra-lead folder, each with an `arm` column and optionally a
            `device` column (`EXTRA_DEVICE` where absent).

    Returns:
        Each arm's device.

    Raises:
        ValueError: If an arm appears in two folders, or on two devices within one, which would
            put two fits of one arm, possibly on different devices, into one figure.
    """
    devices: dict[str, str] = {}
    for index, extra in enumerate(extras):
        column = "device" if "device" in extra.columns else None
        pairs = (
            extra.select("arm", device=pl.col(column) if column else pl.lit(EXTRA_DEVICE))
            .unique()
            .iter_rows()
        )
        for arm, device in pairs:
            if arm in devices:
                msg = (
                    f"arm {arm} is fitted twice: on {devices[arm]} before extra folder {index}, "
                    f"and on {device} in it"
                )
                raise ValueError(msg)
            devices[arm] = device
    return devices


def combine_losses(
    *, published: pl.DataFrame, extras: Sequence[pl.DataFrame], prefer_extras: bool
) -> pl.DataFrame:
    """Stack the published losses and the extra-lead folders' losses.

    Args:
        published: The published CPU fits' losses.
        extras: One losses frame per extra-lead folder.
        prefer_extras: Whether an arm that both the published losses and an extra folder hold is
            taken from the extra folder (its GPU refit) rather than from `published`.

    Returns:
        The stacked losses, without any `device` column.
    """
    extra = pl.concat(
        [frame.drop("device", strict=False) for frame in extras], how="diagonal_relaxed"
    )
    held = extra["arm"].unique().to_list()
    if prefer_extras:
        published = published.filter(~pl.col("arm").is_in(held))
    else:
        extra = extra.filter(~pl.col("arm").is_in(published["arm"].unique().to_list()))
    return pl.concat([published, extra], how="diagonal_relaxed")


def check_single_device(
    *, arms: Sequence[str], extra_devices: Mapping[str, str], published_arms: Collection[str]
) -> None:
    """Raise unless every arm drawn as a mark was fitted on one device.

    Args:
        arms: The arms the figure draws.
        extra_devices: `extra_arm_devices`'s result.
        published_arms: The arms of the published (CPU) fits.

    Raises:
        ValueError: If the arms were fitted on more than one device, or an arm has no fit.
    """
    by_device: dict[str, list[str]] = {}
    for arm in arms:
        device = extra_devices.get(arm, PUBLISHED_DEVICE if arm in published_arms else None)
        by_device.setdefault(device or "unknown", []).append(arm)
    if len(by_device) > 1 or "unknown" in by_device:
        summary = {device: sorted(names) for device, names in sorted(by_device.items())}
        msg = f"the leaderboard's marks mix devices: {summary}"
        raise ValueError(msg)


class Loaded(NamedTuple):
    """One technology's saved outputs, as `load` returns them."""

    losses: pl.DataFrame
    predictions: pl.DataFrame
    leaderboard_losses: pl.DataFrame
    extra_devices: dict[str, str]
    published_arms: frozenset[str]


def load(*, input_dir: Path, domain: DomainType, extra_dirs: Sequence[Path] = ()) -> Loaded:
    """Read one technology's saved losses and predictions.

    Args:
        input_dir: The directory `nwp_forecast_comparison.py` wrote to.
        domain: `solar` or `wind`.
        extra_dirs: The directories `fit_extra_leads.py` wrote to. Their arms that the published
            losses do not hold (the extra lead days, fitted later on a GPU) are appended to
            `losses`, and an arm both hold is taken from `input_dir`, so no contrast mixes devices.
            `leaderboard_losses` instead takes an arm both hold from the extra folder, so the
            leaderboard's marks all come from GPU refits.

    Returns:
        The per-row losses, the per-row predictions, the losses for the leaderboard, each extra
        arm's device, and the published arms, after the anonymisation check.

    Raises:
        ValueError: If an arm appears in two extra folders.
    """
    published = pl.read_parquet(losses_path(output_dir=input_dir, domain=domain))
    extras = [pl.read_parquet(losses_path(output_dir=path, domain=domain)) for path in extra_dirs]
    devices = extra_arm_devices(extras=extras)
    if extras:
        losses = combine_losses(published=published, extras=extras, prefer_extras=False)
        board = combine_losses(published=published, extras=extras, prefer_extras=True)
    else:
        losses = board = published
    predictions = pl.read_parquet(predictions_path(output_dir=input_dir, domain=domain))
    check_anonymised(frame=losses, domain=domain)
    check_anonymised(frame=predictions, domain=domain)
    return Loaded(
        losses=losses,
        predictions=predictions,
        leaderboard_losses=board,
        extra_devices=devices,
        published_arms=frozenset(published["arm"].unique().to_list()),
    )


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


# --- Chart 2: headline contrasts ------------------------------------------------------------------


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
                "P1a, P2a, P3, P4a, and P4b compare at the ENS mean's day-1 lead; P1b and P2b "
                "compare with the ENS mean at day 0, so a product's day-1 error lies between the "
                "two ENS errors when it is bracketed. "
                f"{DOTS_NOTE}"
            ),
            scope_text(losses=losses, domain=domain),
        ],
        figure_planning=planning(rows=[rows]),
    )


# --- Chart 1: leaderboard -------------------------------------------------------------------------


LEAD_COLOURS: Final[dict[int, str]] = {
    0: ocf.BLACK_1,
    1: ocf.DATA_BLUE,
    2: ocf.DATA_SKY,
    3: ocf.DATA_DEEP_TEAL,
    5: ocf.DATA_GREEN,
    7: ocf.ENSEMBLE_LINE,
    10: ocf.DATA_AMBER,
    14: ocf.DATA_BURNT_ORANGE,
}
"""Each lead day's mark colour on the leaderboard. The six chromatic colours, Data Blue, Data Sky,
Data Deep Teal, Data Green, Data Amber, and Data Burnt Orange for days 1, 2, 3, 5, 10, and 14, pass
the bundled `validate_palette.py` all-pairs separation checks in light mode on the page background
(worst colour-blind distance 10.5, worst normal-vision distance 19.5). Data Green fails the script's
lightness band (L 0.81 against a ceiling of 0.77), and Data Sky, Data Green, and Data Amber are
below 3:1 contrast against the background, so each lead's fixed slot within its row and the page's
tables of every number carry the reading as well. Day 0 is black. Day 7 is a neutral grey, the
brand palette's mid grey, and a diamond like day 0, because no chromatic colour is left for it.
Data Amber, Data Deep Teal, and Data Burnt Orange are internal-use colours, approved for the
lead-day charts by the maintainer."""

MAX_LINE_DAY: Final[int] = 3
"""The last lead day the lead-day lines draw: every product plotted there is fitted at every day up
to it, so no line spans a lead that was not fitted."""

LEAD_LABEL_ROWS: Final[float] = 1.0
"""How many rows of room the leaderboard leaves above its first product for the names of the
baseline lines."""

LEAD_LABEL_ROOM: Final[float] = 1.5
"""How far right of the baselines the leaderboard's x axis runs, in percentage points of capacity,
leaving room for the name written beside the smart-persistence line."""

DEVICE_NOTES: Final[dict[DomainType, str]] = {
    "solar": "a GPU fit of an arm differs from its CPU fit by at most 0.02 points",
    "wind": (
        "a GPU fit of an arm has an error 0.04 to 0.09 points lower than its CPU fit, in every "
        "estimate"
    ),
}
"""How far a GPU refit of an arm lies from its published CPU fit (the extra-lead report's device
noise floor), for the leaderboard's subtitle. For wind every estimate is lower, though no single
interval excludes zero."""

DIAMOND_DAYS: Final[frozenset[int]] = frozenset({0, 7})
"""The lead days drawn as diamonds, a second encoding beside the colour: day 0 (black) and day 7
(grey), the two days that are not one of the chromatic circles."""

LEAD_POINT_SIZE: Final[int] = 45
"""The area of one lead-day mark, in square pixels."""

LEAD_ROW_PX: Final[int] = 60
"""The height of one product's row on the leaderboard, which holds one mark per fitted lead."""

LEAD_DODGE_ROWS: Final[float] = 0.135
"""The vertical spacing between the marks of one product's lead days, in rows."""

LEAD_ARM: Final[re.Pattern[str]] = re.compile(r"^(?P<slug>.+)_day(?P<day>\d+)$")


def parsed_lead_arms(*, losses: pl.DataFrame) -> dict[str, tuple[str, int]]:
    """Map each arm the leaderboard draws to its product slug and lead day.

    Args:
        losses: Saved per-row losses.

    Returns:
        `{arm: (slug, day)}` for every arm named `<slug>_day<N>` whose slug is a product.
    """
    return {
        arm: (match["slug"], int(match["day"]))
        for arm in sorted(losses["arm"].unique().to_list())
        if (match := LEAD_ARM.match(arm)) and match["slug"] in PRODUCT_NAMES
    }


def lead_board_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return every product's error and interval at every lead day it was fitted at.

    Args:
        losses: Saved per-row losses.

    Returns:
        One row per fitted product and lead day, at the primary setting, with `product`, `day`,
        `value`, `lower_95` and `upper_95` in percent of capacity. A lead day a product was not
        fitted at has no row, so it is left blank rather than filled.
    """
    primary = by_setting(losses=losses)["primary"]
    parsed = parsed_lead_arms(losses=primary)
    board = leaderboard(losses=primary, arms=list(parsed))
    return (
        board.with_columns(
            pl.col("value", "lower_95", "upper_95") * PERCENTAGE_POINTS,
            product=pl.col("arm").replace_strict(
                {arm: PRODUCT_NAMES[slug] for arm, (slug, _) in parsed.items()},
                return_dtype=pl.String,
            ),
            day=pl.col("arm").replace_strict(
                {arm: day for arm, (_, day) in parsed.items()}, return_dtype=pl.Int64
            ),
        )
        .select("product", "day", "value", "lower_95", "upper_95")
        .sort("product", "day")
    )


def lead_board_products(*, rows: pl.DataFrame) -> list[str]:
    """Order the leaderboard's products by their day-1 error, best first.

    Args:
        rows: What `lead_board_rows` returns.

    Returns:
        Every product's name; each has a day-1 row (a product without one makes the figure raise).
    """
    return rows.filter(pl.col("day") == 1).sort("value")["product"].to_list()


def leaderboard_figure(*, loaded: Loaded, domain: DomainType, title: str) -> alt.VConcatChart:
    """Draw each product's mean absolute error at each fitted lead day, one product per row.

    Args:
        loaded: `load`'s result; the marks come from `loaded.leaderboard_losses`.
        domain: `solar` or `wind`.
        title: The figure's title.

    Returns:
        The figure.

    Raises:
        ValueError: If the marks were fitted on more than one device.
    """
    losses = loaded.leaderboard_losses
    check_single_device(
        arms=list(parsed_lead_arms(losses=by_setting(losses=losses)["primary"])),
        extra_devices=loaded.extra_devices,
        published_arms=loaded.published_arms,
    )
    rows = lead_board_rows(losses=losses)
    products = lead_board_products(rows=rows)
    baselines = leaderboard(
        losses=by_setting(losses=losses)["primary"], arms=["climatology", "smart_persistence_day1"]
    ).with_columns(
        pl.col("value") * PERCENTAGE_POINTS,
        label=pl.col("arm").replace_strict(
            {"climatology": "Climatology", "smart_persistence_day1": "Smart persistence, day 1"},
            return_dtype=pl.String,
        ),
    )
    days = sorted(set(rows["day"].to_list()))
    offsets = {day: (rank - (len(days) - 1) / 2) * LEAD_DODGE_ROWS for rank, day in enumerate(days)}
    data = rows.with_columns(
        y=pl.col("product").replace_strict(
            {name: float(index) for index, name in enumerate(products)}, return_dtype=pl.Float64
        )
        + pl.col("day").replace_strict(offsets, return_dtype=pl.Float64),
        lead=pl.format("Day {}", pl.col("day")),
    )
    lead_names = [f"Day {day}" for day in days]
    x_domain = padded_domain(
        low=float(rows["lower_95"].min()),  # ty: ignore[invalid-argument-type]
        high=max(
            float(rows["upper_95"].max()),  # ty: ignore[invalid-argument-type]
            float(baselines["value"].max()),  # ty: ignore[invalid-argument-type]
        )
        + LEAD_LABEL_ROOM,
        include_zero=False,
    )
    x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
    x_axis = alt.Axis(values=ticks(x_domain=x_domain), format=".2~f", grid=False)
    y_scale = alt.Scale(domain=[len(products) - 0.5, -LEAD_LABEL_ROWS], nice=False)
    labels = {str(index): name for index, name in enumerate(products)}
    y_axis = alt.Axis(
        values=list(range(len(products))),
        labelExpr=f"{json.dumps(labels)}[datum.value]",
        labelLimit=LABEL_WIDTH_PX,
        minExtent=LABEL_WIDTH_PX,
        maxExtent=LABEL_WIDTH_PX,
        labelPadding=6,
        ticks=False,
        domain=False,
        title=None,
    )
    colour = alt.Color(
        "lead:N",
        scale=alt.Scale(domain=lead_names, range=[LEAD_COLOURS[day] for day in days]),
        legend=None,
    )
    x_title = MAE_TITLE
    separators = pl.DataFrame({"y": [index + 0.5 for index in range(len(products) - 1)]})
    rules = (
        alt.Chart(separators)
        .mark_rule(color=ocf.GRID, strokeWidth=1, aria=False)
        .encode(y=alt.Y("y:Q", scale=y_scale, axis=y_axis))  # ty: ignore[unresolved-attribute]
    )
    intervals = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower_95:Q", scale=x_scale, axis=x_axis, title=x_title),
            x2="upper_95:Q",
            y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
            color=colour,
        )
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=LEAD_POINT_SIZE, opacity=1, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", scale=x_scale, axis=x_axis, title=x_title),
            y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
            color=colour,
            shape=alt.Shape(
                "lead:N",
                scale=alt.Scale(
                    domain=lead_names,
                    range=["diamond" if day in DIAMOND_DAYS else "circle" for day in days],
                ),
                legend=None,
            ),
        )
    )
    reference = baselines.with_columns(
        y=pl.lit(-LEAD_LABEL_ROWS + 0.25, dtype=pl.Float64),
        text=pl.col("label"),
    )
    reference_rules = (
        alt.Chart(reference)
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.5, color=ocf.BLACK_1, aria=False)
        .encode(x=alt.X("value:Q", scale=x_scale, axis=x_axis, title=x_title))  # ty: ignore[unresolved-attribute]
    )
    low_arm, high_arm = baselines.sort("value")["arm"].to_list()
    reference_text = [
        alt.Chart(reference.filter(pl.col("arm") == arm))
        .mark_text(align=align, dx=dx, baseline="middle", fontSize=10, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", scale=x_scale, axis=x_axis, title=x_title),
            y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
            text="text:N",
            color=alt.value(ocf.BLACK_1),
        )
        for arm, align, dx in ((low_arm, "right", -5), (high_arm, "left", 5))
    ]
    panel = alt.LayerChart(
        layer=[rules, reference_rules, intervals, points, *reference_text],
        width=PLOT_WIDTH_PX,
        height=LEAD_ROW_PX * (len(products) + LEAD_LABEL_ROWS - 0.5),
    )
    return figure(
        panels=[
            line_key(
                labels=lead_names,
                colours=[LEAD_COLOURS[day] for day in days],
            ),
            panel,
        ],
        number=FIGURE_NUMBERS[(domain, "leaderboard")],
        title=title,
        subtitle=[
            (
                "Each row is one forecast product. Each mark is an XGBoost model's mean absolute "
                "error, as a percentage of capacity, given that product's forecast at one lead "
                "day, on the hours every forecast is scored on. The XGBoost model uses the "
                "primary setting. Smaller is better. A lead day with no mark was not fitted for "
                "that product; nothing is filled in. Dashed lines mark the no-weather baselines."
            ),
            (
                "Within each row, marks run from day 0 at the top to day 14 at the bottom; day 7 "
                "is a grey diamond and day 0 is a black diamond. Day 0 is not a day-ahead "
                "forecast a service could read: ENS's and GEFS's day 0 comes from the 00 UTC run, "
                "0 to 23 hours before the hour it describes; GFS (native)'s from the freshest of "
                "its four runs a day, 0 to 5 hours before; and every other product's from "
                "Open-Meteo's freshest run, 0 to 3 hours before for ICON-EU and ICON-D2 and at a "
                "lead set by each product's own run cycle for the rest. GFS (native) at days 1 "
                "and above reads the 00 UTC run that many days before, from Dynamical.org's "
                "native store, and its radiation is a mean since the last 6-hourly reset, "
                "converted to the mean over each hour or 3 hours before its label; GFS "
                "(Open-Meteo) is NOAA GFS as Open-Meteo's archive serves it. IFS HRES (9 km, "
                "Open-Meteo) is ECMWF's IFS HRES as Open-Meteo's Single Runs archive serves it on "
                "its 9 km grid, a finer product than IFS 0.25° and not another version of it. "
                "It holds 00 UTC runs only, so its day 0 is the 00 UTC run of the hour's own "
                "day, and it is fitted at days 0 to 7 (its runs end at lead 240 hours, so it has "
                "no day 10). IFS HRES publishes 3-hourly steps after lead 90 hours and 6-hourly "
                "steps after lead 144 hours, and Open-Meteo interpolates them to hourly, so "
                "hourly values at day 5 and above, and at the last hours of day 3, are "
                "interpolated. Open-Meteo's processing of HRES has not been checked against a "
                "native archive. Every mark is an XGBoost model fitted "
                "on a graphics processing unit (GPU) at the primary setting, so no mark mixes "
                f"devices; {DEVICE_NOTES[domain]}. "
                f"Overlapping intervals here can still hide a significant paired difference "
                f"(Figure {FIGURE_NUMBERS[(domain, 'headline')]}). {DOTS_NOTE}"
            ),
            (
                "Leads are not equal: a forecast from Open-Meteo's Previous Runs archive comes "
                "from the freshest run made at least a day before the hour it describes, so its "
                "day-1 lead is shorter than ENS's on most hours, which favours that product. "
                "At days 1 and above, only GEFS, GFS (native), IFS HRES (9 km, Open-Meteo), and "
                "the ENS control member share ENS's lead; at day 0, GFS (native) reads the "
                "freshest of its four runs a day. IFS HRES (9 km, Open-Meteo) is scored on the "
                "shared hours minus the target days whose 00 UTC run the archive lacks (seven run "
                "days), which the other rows include. That difference alone moves the ENS mean's "
                f"day-1 error by {ROW_SET_SHIFT_TEXT[domain]}. Contrasts with IFS HRES are "
                "computed on the hours both score."
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


KEY_ROW_PX: Final[int] = 18
"""The height of each extra row of a wrapped `line_key`."""


def line_key(
    *,
    labels: Sequence[str],
    colours: Sequence[str],
    width: int = CONTENT_WIDTH_PX - 60,
    dashed: Sequence[bool] | None = None,
    columns: int | None = None,
) -> alt.LayerChart:
    """Draw a key of short line segments above a chart, in one row unless `columns` wraps it.

    Args:
        labels: Each entry's label.
        colours: Each entry's colour.
        width: The key's width in pixels.
        dashed: Whether each entry's segment is dashed; None draws every segment solid.
        columns: How many entries a row holds before the key wraps to a new row; None puts every
            entry in one row. A wider slot leaves room for a longer label.

    Returns:
        A chart one row tall, or `KEY_ROW_PX` taller for each extra row.
    """
    per_row = columns or len(labels)
    slot = width // per_row
    rows = -(-len(labels) // per_row)
    # A one-row key keeps a constant y, so its SVG is byte-identical to a key drawn without rows.
    y_encoding = alt.Y("y:Q", scale=None) if rows > 1 else alt.value(8)
    data = pl.DataFrame(
        {
            "label": list(labels),
            "colour": list(colours),
            "x": [index % per_row * slot for index in range(len(labels))],
            "x2": [index % per_row * slot + 18 for index in range(len(labels))],
            "y": [8 + KEY_ROW_PX * (index // per_row) for index in range(len(labels))],
            "dashed": [str(flag).lower() for flag in (dashed or [False] * len(labels))],
        }
    )
    segments = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=None),
            x2="x2:Q",
            y=y_encoding,
            color=alt.Color("colour:N", scale=None),
            strokeDash=alt.StrokeDash(
                "dashed:N",
                scale=alt.Scale(domain=["false", "true"], range=[[1, 0], [5, 3]]),
                legend=None,
            ),
        )
    )
    text = (
        alt.Chart(data)
        .mark_text(align="left", dx=22, color=ocf.BLACK_1, limit=slot - 24)
        .encode(x=alt.X("x:Q", scale=None), y=y_encoding, text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(layer=[segments, text], width=width, height=16 + KEY_ROW_PX * (rows - 1))


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
                "its fitting seeds. The XGBoost model uses the primary setting. A gap in a line is "
                "an hour outside the scored rows."
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

PRODUCT_COLOURS: Final[dict[str, str]] = {
    "ENS mean": ocf.DATA_BLUE,
    "GEFS mean": ocf.DATA_SKY,
    "IFS 0.25°": ocf.DATA_BURNT_ORANGE,
    "ICON-EU": ocf.DATA_DEEP_TEAL,
    "ICON global": ocf.DATA_GREEN,
    "GFS (Open-Meteo)": ocf.DATA_AMBER,
    "GFS (native)": ocf.DATA_AMBER,
    "IFS HRES (9 km, Open-Meteo)": ocf.DATA_BURNT_ORANGE,
    "ARPEGE Europe": ocf.BLACK_1,
}
"""Each product's colour on the lead-day chart. The six coloured products pass the bundled
`validate_palette.py` all-pairs checks except two. Data Green misses the lightness band (L 0.81
against a ceiling of 0.77), and Data Sky, Data Green and Data Amber have a contrast warning against
the page background, so every product's name is also written beside its last point. The worst
colour-blind distance is 10.5 and the worst normal-vision distance is 19.5, above the script's
targets of 8 and 15. No seventh chromatic colour passes, so ARPEGE, which only the solar chart
holds, is black and dashed (black passes both separation checks). Data Amber, Data Deep Teal and
Data Burnt Orange are internal-use colours, approved for these charts by the maintainer; the
`dataviz` skill records that the maintainer once swapped Burnt Orange for Magenta beside Amber.

The two GFS rows are the same weather model from two sources, so they share Data Amber, and the
native one is dashed. A seventh chromatic colour does not exist: adding Data Purple or Data Magenta
to the six chromatic colours fails the colour-blind all-pairs check (worst ΔE 2.0 for Purple against
Data Blue, 2.3 for Magenta against Data Blue, against a floor of 8), so the dash and the name beside
each line's last point carry the difference.

`IFS HRES (9 km, Open-Meteo)` reuses IFS 0.25°'s Data Burnt Orange and is dashed, for the same
reason. No colour was added, so the palette check above is unchanged."""

KEY_LABELS: Final[dict[str, str]] = {
    "ARPEGE Europe": "ARPEGE",
    "IFS HRES (9 km, Open-Meteo)": "IFS HRES 9 km",
}
"""Shorter names for the key above the lead-day chart and for the names beside each line's last
point, whose room is limited: the key wraps to `KEY_COLUMNS` entries a row, and the label beside a
line has about 115 pixels."""

ROW_SET_SHIFT_TEXT: Final[dict[DomainType, str]] = {
    "solar": (
        "+0.022 points (8.771% on all 35,263 shared rows, 8.792% on the 34,771 rows without "
        "the gap days)"
    ),
    "wind": (
        "+0.029 points (8.350% on all 37,407 shared rows, 8.379% on the 37,001 rows without "
        "the gap days)"
    ),
}
"""The row-set diagnostic's result for the leaderboard caption, by technology: the ENS mean's day-1
error on all shared rows and without IFS HRES (9 km, Open-Meteo)'s day-1 gap days, as the fourth
batch's fit report states them."""

KEY_COLUMNS: Final[int] = 5
"""How many entries a row of the lead-day chart's key holds. The chart can hold nine products, and
nine slots would leave 44 pixels for a label."""

DASHED_PRODUCTS: Final[frozenset[str]] = frozenset(
    {"ARPEGE Europe", "GFS (native)", "IFS HRES (9 km, Open-Meteo)"}
)
"""Products drawn with a dashed line, because no seventh distinguishable colour exists: ARPEGE is
black, and the native GFS and IFS HRES (9 km, Open-Meteo) rows share the amber of Open-Meteo's GFS
row and the burnt orange of IFS 0.25°."""

LEAD_BAND_COLOURS: Final[tuple[str, str]] = (ocf.DATA_BLUE_LIGHT, ocf.DATA_SKY_LIGHT)
"""The shading of ENS's day-0 and day-1 intervals."""

X_MAX_DAYS: Final[float] = 4.3
"""The lead-day chart's right edge, leaving room for the product names beside day 3."""

DODGE_DAYS: Final[float] = 0.06
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
    """Name the product an arm belongs to on the lead-day chart.

    Args:
        arm: An arm in the saved losses.

    Returns:
        The product's key in `PRODUCT_COLOURS` for a product read at a whole day, or None for a
        baseline, a blend, a control member and any arm the chart leaves out.
    """
    slug, _, day = arm.rpartition("_day")
    if not day.isdigit() or slug == "ens_control":
        return None
    name = PRODUCT_NAMES.get(slug)
    return name if name in PRODUCT_COLOURS else None


def lead_rows(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each product's error and interval at each lead day, at the primary setting.

    Args:
        losses: Saved per-row losses.

    Returns:
        One row per arm with `series`, `product` (the name written beside its last point), `day`,
        `value`, `lower_95` and `upper_95` in percent of capacity.
    """
    primary = by_setting(losses=losses)["primary"]
    arms = [
        arm
        for arm in sorted(primary["arm"].unique().to_list())
        if lead_series_name(arm=arm)
        and (arm == "ens_mean_day0" or 1 <= int(arm.rpartition("_day")[2]) <= MAX_LINE_DAY)
    ]
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
    names = [name for name in PRODUCT_COLOURS if name in set(rows["series"].to_list())]
    products = rows["product"].unique(maintain_order=True).to_list()
    offsets = {
        product: (index - (len(products) - 1) / 2) * DODGE_DAYS
        for index, product in enumerate(products)
    }
    # Each product is its own line in its own colour; ARPEGE, native GFS and IFS HRES (9 km,
    # Open-Meteo) are dashed.
    drawn = rows.with_columns(
        x=pl.col("day") + pl.col("product").replace_strict(offsets, return_dtype=pl.Float64),
        line=pl.col("product"),
        dashed=pl.col("product").is_in(list(DASHED_PRODUCTS)).cast(pl.String),
    )
    low = float(rows["lower_95"].min())  # ty: ignore[invalid-argument-type]
    high = float(rows["upper_95"].max())  # ty: ignore[invalid-argument-type]
    y_domain = padded_domain(low=low, high=high, include_zero=False)
    x_scale = alt.Scale(domain=[-0.5, X_MAX_DAYS], nice=False)
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=[PRODUCT_COLOURS[name] for name in names]),
        legend=None,
    )
    x_axis = alt.Axis(
        values=[0, 1, 2, 3],
        labelExpr="'Day ' + datum.value",
        grid=False,
        title=(
            "Lead day (ENS, native GFS, and IFS HRES (9 km, Open-Meteo): a 24-hour band of leads "
            "of the 00 UTC run that many days earlier; other products: the freshest run at "
            "least as many days old as the lead day)"
        ),
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
    band_text = [
        alt.Chart(bands.filter(pl.col("day") == day))
        .mark_text(align="right", baseline=baseline, dx=-4, dy=dy, color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x1:Q", scale=x_scale),
            y=alt.Y(f"{edge}:Q", scale=alt.Scale(domain=list(y_domain), nice=False)),
            text="label:N",
        )
        for day, edge, baseline, dy in ((0, "lower", "bottom", -2), (1, "upper", "top", 2))
    ]
    lines = (
        alt.Chart(drawn)
        .mark_line(strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=x_scale, axis=x_axis),
            y=y,
            color=colour,
            detail="line:N",
            strokeDash=alt.StrokeDash(
                "dashed:N",
                scale=alt.Scale(domain=["false", "true"], range=[[1, 0], [5, 3]]),
                legend=None,
            ),
        )
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
        end_label=pl.col("product").replace(KEY_LABELS),
        label_x=pl.lit(MAX_LINE_DAY + max(offsets.values()), dtype=pl.Float64),
        label_y=pl.Series(
            spread_labels(values=ends["value"].to_list(), min_gap=(y_domain[1] - y_domain[0]) / 16)
        ),
    )
    end_text = (
        alt.Chart(ends)
        .mark_text(align="left", dx=12, fontSize=10, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("label_x:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("label_y:Q", scale=alt.Scale(domain=list(y_domain), nice=False)),
            text="end_label:N",
            color=colour,
        )
    )
    panel = alt.LayerChart(
        layer=[shading, *band_text, lines, rules, points, end_text],
        width=CONTENT_WIDTH_PX - 100,
        height=LEAD_PANEL_HEIGHT_PX,
    )
    return figure(
        panels=[
            line_key(
                labels=[KEY_LABELS.get(name, name) for name in names],
                colours=[PRODUCT_COLOURS[name] for name in names],
                dashed=[name in DASHED_PRODUCTS for name in names],
                columns=KEY_COLUMNS,
            ),
            panel,
        ],
        number=FIGURE_NUMBERS[(domain, "by_lead_day")],
        title=title,
        subtitle=[
            (
                "Mean absolute error of an XGBoost model given each forecast product at each "
                "lead day, as a percentage of capacity. The XGBoost model uses the primary "
                "setting. Smaller is better. Shaded bands: the 95% intervals of the ENS mean's "
                "day-0 and day-1 errors; the two ENS leads between which a Previous Runs "
                "product's day-1 lead falls."
            ),
            (
                f"{DOTS_NOTE} Products at one day are drawn side by side, and each product's "
                "name is written beside its last point. Two of the dashed lines each sit beside a "
                "solid line of the same colour; on the solar chart, the dashed black line is "
                "ARPEGE Europe, shown as ARPEGE. The dashed burnt-orange line, shown as IFS HRES "
                "9 km, is "
                "IFS HRES (9 km, Open-Meteo): 00 UTC runs only, scored without the target days "
                "whose run the archive lacks, not checked against a native archive, and its "
                "day-3 values after lead 90 hours are interpolated from 3-hourly steps. The "
                "solid burnt-orange line is IFS 0.25°, a coarser product. The two GFS lines are "
                "one weather model from two sources: the solid line is Open-Meteo's "
                "GFS-SEAMLESS archive, and the dashed line is Dynamical.org's native GFS "
                "store, whose radiation is a mean since "
                "the last 6-hourly reset and is converted to the mean over each hour before its "
                "label. ICON-EU's day 0 is not drawn here: it is "
                "Open-Meteo's freshest ICON-EU run, a lead of at most 3 hours, fitted later on a "
                "GPU. IFS HRES 9 km is drawn at days 1 to 3 only: its day 0 and its day 7 are "
                "left out, as day 7 is for every product. Products fitted at fewer than three of "
                "days 0 to 3, and days 5, 10, and 14, "
                f"are left out; Figure {FIGURE_NUMBERS[(domain, 'leaderboard')]} shows them. "
                f"{SHARED_ROWS_NOTE}"
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
        panel_title="Paired differences: blend minus ENS, blend minus control",
    )
    return figure(
        panels=[level_panel, contrast_panel],
        number=FIGURE_NUMBERS[(domain, "blends")],
        title=title,
        subtitle=[
            (
                "XGBoost models given the ENS mean alone, or ENS plus two more products. "
                "A control has the same columns as its blend, with the two other products' "
                "weather shuffled among hours of the same site, year-month, and hour of day, so it "
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
                "the first product forecasts better. The XGBoost model uses the primary setting. "
                "The 95% interval resamples whole months and a fitting seed within one generator, "
                "so it does not cover differences between generators. All rows are exploratory."
            ),
            f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE} {SHARED_ROWS_NOTE}",
        ],
        figure_planning=None,
    )


# --- Output -------------------------------------------------------------------------------------

FIGURE_NUMBERS: Final[dict[tuple[DomainType, str], int]] = {
    ("solar", "leaderboard"): 1,
    ("wind", "leaderboard"): 2,
    ("solar", "headline"): 3,
    ("wind", "headline"): 4,
    ("solar", "models_work"): 5,
    ("wind", "models_work"): 6,
    ("solar", "per_generator"): 7,
    ("wind", "per_generator"): 8,
    ("solar", "by_lead_day"): 11,
    ("wind", "by_lead_day"): 12,
    ("solar", "blends"): 9,
    ("wind", "blends"): 10,
}
"""Each chart's figure number on the page, in the page's order: the leaderboard pair opens the page,
then the planned contrasts."""

TITLES: Final[dict[tuple[DomainType, str], str]] = {
    ("solar", "headline"): (
        "For solar power, ENS beats UKV and ICON-EU at matched lead and GEFS at equal lead. "
        "A blend lowers the error by 0.35 points at an optimistic lead and shows no "
        "detectable gain at a conservative lead"
    ),
    ("wind", "headline"): (
        "For wind power, ENS beats UKV and GEFS, ICON-EU is unresolved against ENS, and a blend "
        "lowers the error by 0.18 points even at a conservative lead"
    ),
    ("solar", "leaderboard"): (
        "For solar power, error rises with lead to day 10: at day 1 every weather forecast shown "
        "has a lower error than climatology (14.5%), but at day 14 neither the ENS mean nor the "
        "GEFS mean does; the ENS mean and IFS 0.25° have the lowest day-1 errors"
    ),
    ("wind", "leaderboard"): (
        "For wind power, error rises with lead: at day 1 every weather forecast shown has a lower "
        "error than climatology (18.5%), but at day 14 neither the ENS mean nor the GEFS mean "
        "does; the ENS mean and IFS 0.25° have the lowest day-1 errors"
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
        "and at two of the three wind farms the P4b blend has a lower error"
    ),
    ("solar", "by_lead_day"): (
        "Solar error rises with lead day for every forecast; IFS 0.25° cannot be told apart from "
        "the ENS mean at days 1 to 3, and every other product has a higher error than ENS"
    ),
    ("wind", "by_lead_day"): (
        "Wind error rises with lead day for every forecast; IFS 0.25° has a lower error than the "
        "ENS mean at days 2 and 3, at a lead shorter than ENS's on most hours"
    ),
    ("solar", "blends"): (
        "For solar power a blend of ENS, ICON-EU, and IFS 0.25° lowers the error by 0.35 "
        "percentage points at an optimistic lead, but the blend's control is itself worse than "
        "ENS alone"
    ),
    ("wind", "blends"): (
        "For wind power a blend of ENS, ICON-EU, and IFS 0.25° lowers the error by 0.66 "
        "percentage points at an optimistic lead and 0.18 percentage points at a conservative "
        "lead, and the blend's control does not"
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
    *, input_dir: Path, domain: DomainType, extra_dirs: Sequence[Path] = ()
) -> tuple[dict[str, alt.VConcatChart], str | None]:
    """Draw every chart of one technology that its saved losses can support.

    Args:
        input_dir: The directory `nwp_forecast_comparison.py` wrote to.
        domain: `solar` or `wind`.
        extra_dirs: The directories `fit_extra_leads.py` wrote to.

    Returns:
        Each chart keyed by its name, and the chosen week's month and year.
    """
    loaded = load(input_dir=input_dir, domain=domain, extra_dirs=extra_dirs)
    losses, predictions = loaded.losses, loaded.predictions
    week_month: str | None = None
    charts: dict[str, alt.VConcatChart | None] = {
        "headline": headline(losses=losses, domain=domain, title=TITLES[(domain, "headline")]),
        "leaderboard": leaderboard_figure(
            loaded=loaded, domain=domain, title=TITLES[(domain, "leaderboard")]
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
    parser.add_argument(
        "--extra-dir",
        type=Path,
        action="append",
        default=[],
        help="An extra lead days' losses directory; repeat it for each fit batch.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where SVGs are written.")
    parser.add_argument("--no-svgo", action="store_true", help="Skip the svgo optimisation.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        charts, week_month = draw_domain(
            input_dir=args.input_dir, domain=domain, extra_dirs=args.extra_dir
        )
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
