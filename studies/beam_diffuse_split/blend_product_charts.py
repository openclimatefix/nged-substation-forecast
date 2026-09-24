"""Draw the anonymised charts for the write-up on blending weather products.

One-off throwaway script for the charts in
<https://github.com/openclimatefix/nged-substation-forecast/issues/836>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/blending-weather-products/>.

**Every interval and every error a chart shares with the page is read from the report
`blend_products.py` wrote**, so a chart cannot disagree with the page. Before drawing, the script
checks every contrast row it parsed against `intervals.parquet`, and recomputes two printed rows
through `blend_products.py`'s own functions. Three charts also draw numbers the report does not
print: the weekly time series and the per-generator errors, from `predictions.parquet` and
`losses.parquet`, and the cumulative share of the wind gain. Each of those checks a printed number
first: every whole-record error against the report's tables, every per-generator difference against
the report's per-site rows, and the 5% point and the share of hours improving against the report's
lines.

Generators appear only as `A` to `F` and `W1` to `W3`, and every output is a fraction of the
generator's own capacity.

Run it with `uv run python studies/beam_diffuse_split/blend_product_charts.py`, after
`blend_products.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final, Literal, NamedTuple

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from blend_products import (
    METRIC,
    MOST_IMPROVED_SHARE,
    OUTPUT_DIR,
    PERCENTAGE_POINTS,
    SOLAR,
    WIND,
    _headline_line,
    _interval,
    _line,
)
from studies.charts import (
    CONTENT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
    wrapped,
)
from weather_product_charts import (
    ASSETS_DIR,
    CAPACITY,
    LEADERBOARD_X_TITLE,
    NAMES,
    X_TITLE,
    _two_places,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DomainType = Literal["solar", "wind"]
SettingType = Literal["pooled", "sensitivity"]

SECTION_DECIDING: Final[str] = (
    "Deciding contrasts: each named set's enriched blend against its enriched best single"
)
SECTION_PLAIN: Final[str] = "Secondary: the plain contrasts named before the first run"
SECTION_SYNTHETIC: Final[str] = (
    "Sensitivity positive control: a synthetic product carrying part of the target"
)
SECTION_EXPLORATORY: Final[str] = "Exploratory contrasts"
SECTION_SPLITS: Final[str] = "The named enriched blends by site, season and era (exploratory)"
SECTION_BANDS: Final[str] = (
    "Wind: where the gain comes from, by measured output (main XGBoost settings)"
)

SITES: Final[dict[DomainType, tuple[str, ...]]] = {
    "solar": ("A", "B", "C", "D", "E", "F"),
    "wind": ("W1", "W2", "W3"),
}
"""The anonymous generator labels."""

DOMAIN_NAMES: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

SCOPES: Final[dict[DomainType, str]] = {
    "solar": "Six solar farms in Lincolnshire, December 2022 to September 2026.",
    "wind": "Three wind farms in Lincolnshire, August 2024 to September 2026.",
}

SEASONS: Final[tuple[tuple[str, str], ...]] = (
    ("season winter", "December to February"),
    ("season spring", "March to May"),
    ("season summer", "June to August"),
    ("season autumn", "September to November"),
)
"""The report's season scopes, and the months each covers."""

ERAS: Final[tuple[tuple[str, str], ...]] = (
    ("era pre", "Before UKV's upgrade on 21 January 2026"),
    ("era post", "After UKV's upgrade"),
)
"""The report's era scopes, either side of the Met Office's upgrade of UKV."""

SINGLE_NAMES: Final[dict[str, str]] = {
    "cams_rich": "CAMS",
    "icon_d2_rich": "ICON-D2",
    "ukv_rich": "UKV",
}
"""The enriched best single products, as the charts name them."""

_DOMAIN_SETS: Final[dict[DomainType, dict[str, tuple[str, ...]]]] = {
    "solar": {blend_set.name: blend_set.products for blend_set in SOLAR.sets},
    "wind": {blend_set.name: blend_set.products for blend_set in WIND.sets},
}
"""Each domain's blend sets' products, in the order `blend_products.py` lists them."""


def _blend_label(*, domain: DomainType, blend: str) -> str:
    """Name a blend by listing its products, never by their count.

    The maintainer's rule for every chart on this page: a label that names a blend lists the
    blend's products, so a reader never has to take "all six products" on faith.

    Args:
        domain: `solar` or `wind`.
        blend: The blend's key in `blend_products.py`.

    Returns:
        Each product's display name, in the blend's own order, joined with " + ".
    """
    return " + ".join(NAMES[product] for product in _DOMAIN_SETS[domain][blend])


class NamedSet(NamedTuple):
    """One named blend set, and the enriched best single product it is judged against."""

    domain: DomainType
    blend: str
    name: str
    best: str


NAMED_SETS: Final[tuple[NamedSet, ...]] = (
    NamedSet("solar", "everything", _blend_label(domain="solar", blend="everything"), "cams_rich"),
    NamedSet("solar", "cams_icon_eu", "CAMS and ICON-EU", "cams_rich"),
    NamedSet("solar", "live_all", _blend_label(domain="solar", blend="live_all"), "icon_d2_rich"),
    NamedSet("wind", "everything", _blend_label(domain="wind", blend="everything"), "ukv_rich"),
    NamedSet("wind", "live_gb", "UKV and ICON-EU", "ukv_rich"),
)
"""The sets behind the deciding contrasts, in the order the page gives them."""

SET_NAMES: Final[dict[DomainType, dict[str, str]]] = {
    "solar": {
        "cams_icon_d2": "CAMS and ICON-D2",
        "cams_icon_eu": "CAMS and ICON-EU",
        "cams_era5": "CAMS and ERA5",
        "live_gb": "UKV and ICON-EU",
        "live_all": _blend_label(domain="solar", blend="live_all"),
        "everything": _blend_label(domain="solar", blend="everything"),
    },
    "wind": {
        "best_pair": "ICON-D2 and UKV",
        "live_gb": "UKV and ICON-EU",
        "live_all": _blend_label(domain="wind", blend="live_all"),
        "everything": _blend_label(domain="wind", blend="everything"),
    },
}
"""Every set's name as the "What to use" table gives it, one entry per `Domain.sets` in
`blend_products.py`."""

LEADERBOARD_TITLE: Final[str] = (
    "leaderboard, every single product and blend, plain and enriched (main XGBoost settings)"
)
LEADERBOARD_KINDS: Final[tuple[str, str]] = ("Blend", "Single product")
LEADERBOARD_LIVE: Final[tuple[str, str]] = ("Yes", "No: history only, or ICON-D2's area only")
"""Whether a live service anywhere in Great Britain can read every product a row reads."""

DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole months and a fitting seed."
)
LEADERBOARD_DOMAIN: Final[dict[DomainType, tuple[float, float]]] = {
    "solar": (4.0, 9.5),
    "wind": (4.5, 9.0),
}
"""Each panel's x range, padded around the widest interval `blend_products.py` printed."""

HEADLINE_DOMAIN: Final[tuple[float, float]] = (-0.7, 0.1)
SETTING_CONDITIONS: Final[tuple[str, str]] = (
    "Main XGBoost settings",
    "Shallower XGBoost settings (a check)",
)
POST_HOC_SUFFIX: Final[str] = " (post hoc)"
"""Ends the label of a deciding-contrast row drawn beside exploratory rows in the same figure."""

ALL_POST_HOC: Final[str] = (
    "Every row is post hoc: chosen after the first run's results, and fixed before the re-run "
    "that produced the numbers on this page."
)
"""The subtitle line of a figure whose rows are all post hoc."""

SOME_POST_HOC: Final[str] = (
    "Post hoc: chosen after the first run's results, and fixed before the re-run that produced "
    "the numbers on this page. Every other row is exploratory."
)
"""The subtitle line of a figure whose rows ending in `POST_HOC_SUFFIX` are post hoc."""

MEASURED_COLOUR: Final[str] = ocf.BLACK_1
SINGLE_COLOUR: Final[str] = ocf.DATA_BLUE
BLEND_COLOUR: Final[str] = ocf.BRAND_ORANGE
"""Data Blue and Brand Orange pass `validate_palette.js` against the page background: 32.5 ΔE apart
under protanopia, and both above 3:1 contrast. The measured output is the text colour."""

MODEL_KEYS: Final[tuple[str, str]] = ("Best single product, enriched", "Blend, enriched")
"""The key entries of the per-generator error chart."""

MIN_HOURS_PER_DAY: Final[dict[DomainType, int]] = {"solar": 4, "wind": 12}
"""The scored hours a generator needs on each day of a week for the week to be chosen."""

WEEK_PANEL_HEIGHT_PX: Final[int] = 58
WEEK_SPACING_PX: Final[int] = 8
WEEK_ROW_LABEL_PX: Final[int] = 112

_GENERATOR_PANEL_TITLE_CHARACTERS: Final[int] = 60
"""Figure 6's panel-title wrap width: `CONTENT_WIDTH_PX - 100` wide, drawn outside
`interval_panel`. A full blend's product list can push a title past one line."""

_INTERVAL_PANEL_TITLE_CHARACTERS: Final[int] = 48
"""Figure 10's panel-title wrap width: `interval_panel`'s narrower `PLOT_WIDTH_PX`."""


def _signed(value: object) -> str:
    """Write a report number to two places with its sign, as the page writes an interval bound."""
    number = float(str(value))
    return f"{'−' if number < 0 else '+'}{_two_places(abs(number))}"


def _cells(line: str) -> list[str]:
    """Return a markdown table row's cells, stripped of padding."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(*, report_text: str) -> dict[str, list[dict[str, str]]]:
    """Read every table in the report, keyed by the heading above it.

    Args:
        report_text: The report.

    Returns:
        For each heading with a table under it, one mapping of header to cell per row.
    """
    tables: dict[str, list[dict[str, str]]] = {}
    heading = ""
    header: list[str] = []
    for line in report_text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            header = []
        elif line.startswith("|"):
            cells = _cells(line)
            if not header:
                header = cells
            elif not set(line) <= set("|-"):
                tables.setdefault(heading, []).append(dict(zip(header, cells, strict=True)))
        else:
            header = []
    return tables


def _bounds(*, text: str) -> tuple[float, float]:
    """Read `[low, high]` or `low to high` as two numbers."""
    low, high = re.split(r", | to ", text.strip("[]"))
    return float(low), float(high)


def _contrasts(*, tables: dict[str, list[dict[str, str]]]) -> pl.DataFrame:
    """Gather every contrast row in the report into one frame.

    Args:
        tables: The output of `_tables`.

    Returns:
        One row per contrast row, with `section`, `domain`, `setting`, `scope`, `treatment`,
        `reference`, `difference`, `lower_95` and `upper_95` in points of capacity, and, where the
        report prints them, `fold_lower_95`, `fold_upper_95`, `site_lowest` and `site_highest`.
    """
    rows: list[dict[str, object]] = []
    for section, table in tables.items():
        for row in table:
            if "ΔMAE (pp of capacity)" not in row:
                continue
            treatment, reference = row["Contrast"].split(" − ")
            lower, upper = _bounds(text=row["95% interval, months and seed"])
            fold = row.get("95% t-interval across the 5 folds")
            fold_lower, fold_upper = _bounds(text=fold) if fold else (None, None)
            sites = row.get("Generators, lowest to highest")
            site_lowest, site_highest = _bounds(text=sites) if sites else (None, None)
            rows.append(
                {
                    "section": section,
                    "domain": row["Domain"],
                    "setting": row["Setting"],
                    "scope": row.get("Scope", "all"),
                    "treatment": treatment,
                    "reference": reference,
                    "difference": float(row["ΔMAE (pp of capacity)"]),
                    "lower_95": lower,
                    "upper_95": upper,
                    "fold_lower_95": fold_lower,
                    "fold_upper_95": fold_upper,
                    "site_lowest": site_lowest,
                    "site_highest": site_highest,
                }
            )
    return pl.DataFrame(rows)


def _check_against_intervals(*, contrasts: pl.DataFrame) -> None:
    """Check every parsed contrast row against `intervals.parquet`, to the printed digit.

    Args:
        contrasts: The output of `_contrasts`.

    Raises:
        ValueError: If any row has no match, or differs from its match once rounded.
    """
    saved = pl.read_parquet(OUTPUT_DIR / "intervals.parquet").select(
        "domain",
        "setting",
        "scope",
        "treatment",
        "reference",
        saved_difference=pl.col("difference_pp").round(3),
        saved_lower=pl.col("lower_95_pp").round(3),
        saved_upper=pl.col("upper_95_pp").round(3),
    )
    joined = contrasts.join(
        saved.unique(), on=["domain", "setting", "scope", "treatment", "reference"], how="left"
    )
    mismatched = joined.filter(
        pl.col("saved_difference").is_null()
        | ((pl.col("difference") - pl.col("saved_difference")).abs() > 1e-9)
        | ((pl.col("lower_95") - pl.col("saved_lower")).abs() > 1e-9)
        | ((pl.col("upper_95") - pl.col("saved_upper")).abs() > 1e-9)
    )
    if mismatched.height or joined.height != contrasts.height:
        msg = f"report rows disagree with intervals.parquet: {mismatched.head(5)}"
        raise ValueError(msg)


def _losses(*, domain: DomainType, arms: list[str]) -> pl.DataFrame:
    """Return the primary setting's per-row losses for some arms of one domain.

    Args:
        domain: `solar` or `wind`.
        arms: The arms to keep.

    Returns:
        The rows of `losses.parquet` for those arms.
    """
    return (
        pl.scan_parquet(OUTPUT_DIR / "losses.parquet")
        .filter(
            pl.col("domain") == domain,
            pl.col("setting") == "pooled",
            pl.col("arm").is_in(arms),
        )
        .collect()
    )


def _reproduce(*, report_text: str) -> None:
    """Recompute two printed rows through `blend_products.py`'s own functions, and match both.

    Args:
        report_text: The report.

    Raises:
        ValueError: If either recomputed row differs from the report's.
    """
    losses = _losses(domain="wind", arms=["everything_rich_xgb", "ukv_rich"])
    lines = report_text.splitlines()
    for line in (
        _headline_line(
            _interval(
                losses=losses,
                contrast=("everything_rich_xgb", "ukv_rich"),
                domain=WIND,
                setting="pooled",
                section="deciding",
            )
        ),
        _line(
            _interval(
                losses=losses.filter(pl.col("site") == "W2"),
                contrast=("everything_rich_xgb", "ukv_rich"),
                domain=WIND,
                setting="pooled",
                section="split",
                scope="site W2",
            )
        ),
    ):
        if line not in lines:
            msg = f"the recomputed row does not match the report: {line}"
            raise ValueError(msg)


def _pick(
    *,
    contrasts: pl.DataFrame,
    section: str,
    domain: DomainType,
    treatment: str,
    reference: str,
    setting: SettingType = "pooled",
    scope: str = "all",
) -> dict[str, object]:
    """Return the one report row matching every field.

    Args:
        contrasts: The output of `_contrasts`.
        section: The heading of the row's table.
        domain: `solar` or `wind`.
        treatment: The treatment arm.
        reference: The reference arm.
        setting: The hyperparameter setting.
        scope: The row's scope.

    Returns:
        The row.

    Raises:
        ValueError: If no row or more than one matches.
    """
    matches = contrasts.filter(
        pl.col("section") == section,
        pl.col("domain") == domain,
        pl.col("setting") == setting,
        pl.col("scope") == scope,
        pl.col("treatment") == treatment,
        pl.col("reference") == reference,
    )
    if matches.height != 1:
        msg = f"{section}/{domain}/{setting}/{scope}/{treatment} − {reference}: {matches.height}"
        raise ValueError(msg)
    return matches.row(0, named=True)


def _panel_rows(*, rows: list[tuple[str, dict[str, object], str]]) -> pl.DataFrame:
    """Turn picked report rows into the rows `interval_panel` draws.

    Args:
        rows: For each mark, its label, its report row, and its condition (empty for none).

    Returns:
        One row per mark, all in the weather-model family.
    """
    return pl.DataFrame(
        {
            "label": [label for label, _, _ in rows],
            "difference": [row["difference"] for _, row, _ in rows],
            "lower_95": [row["lower_95"] for _, row, _ in rows],
            "upper_95": [row["upper_95"] for _, row, _ in rows],
            "condition": [condition for _, _, condition in rows],
            "family": ["weather model"] * len(rows),
        }
    )


def _blend_errors(*, tables: dict[str, list[dict[str, str]]], domain: DomainType) -> pl.DataFrame:
    """Read the enriched blends' mean absolute errors at the primary setting.

    Args:
        tables: The output of `_tables`.
        domain: `solar` or `wind`.

    Returns:
        One row per set, with `set`, `best`, `best_mae`, and each method's error.
    """
    title = (
        f"{domain.capitalize()}: blends of enriched columns, primary setting (second in brackets)"
    )
    rows = []
    for row in tables[title]:
        best, best_mae = row["Best single, primary"].split(" ")
        rows.append(
            {
                "set": row["Set"],
                "best": best,
                "best_mae": float(best_mae),
                **{
                    method: float(row[f"`{method}`"].split(" ")[0])
                    for method in ("xgb", "control", "mean", "stack", "equal")
                },
            }
        )
    return pl.DataFrame(rows)


_LEADERBOARD_ROW_STEP_PX: Final[int] = 36
"""Each leaderboard row's height: room for a two-line label, with the one three-line label
overhanging into the gap between rows."""


def _leaderboard_rows(
    *, tables: dict[str, list[dict[str, str]]], domain: DomainType
) -> pl.DataFrame:
    """Read one domain's leaderboard table, labelled as the page labels every row.

    Args:
        tables: The output of `_tables`.
        domain: `solar` or `wind`.

    Returns:
        One row per arm, sorted with the lowest error first, with `label`, `family`, `kind`
        (`Blend` or `Single product`), `condition` (whether a live service anywhere in Great
        Britain can read it), `value` (the arm's own mean absolute error), `lower_95` and
        `upper_95`.
    """
    rows = []
    for row in tables[f"{domain.capitalize()}: {LEADERBOARD_TITLE}"]:
        lower, upper = _bounds(text=row["95% interval, months and seed"])
        single = row["Kind"] == "single"
        name = NAMES[row["Set or product"]] if single else SET_NAMES[domain][row["Set or product"]]
        # A set name opening with a common word is lower-cased mid-label; a product name is not.
        label = name if single or name.split()[0].isupper() else name[0].lower() + name[1:]
        if row["Variant"] == "rich":
            label += ", enriched"
        rows.append(
            {
                "label": label,
                "family": "weather model",
                "kind": LEADERBOARD_KINDS[1] if single else LEADERBOARD_KINDS[0],
                "condition": LEADERBOARD_LIVE[
                    0 if row["Live, Great-Britain-wide?"] == "yes" else 1
                ],
                "value": float(row["MAE (pp of capacity)"]),
                "lower_95": lower,
                "upper_95": upper,
            }
        )
    return pl.DataFrame(rows).sort("value")


def _leaderboard(*, tables: dict[str, list[dict[str, str]]]) -> alt.VConcatChart:
    """Draw every single product's and every blend's absolute error, ranked best first.

    Args:
        tables: The output of `_tables`.

    Returns:
        Figure 1.
    """
    rows = {domain: _leaderboard_rows(tables=tables, domain=domain) for domain in DOMAIN_NAMES}
    best_live = {
        domain: rows[domain].filter(pl.col("condition") == LEADERBOARD_LIVE[0]).row(0, named=True)
        for domain in DOMAIN_NAMES
    }
    panels = [
        leaderboard_panel(
            rows=rows[domain],
            x_domain=LEADERBOARD_DOMAIN[domain],
            x_title=LEADERBOARD_X_TITLE,
            conditions=LEADERBOARD_LIVE,
            condition_title="Can a live service anywhere in Great Britain read it?",
            kinds=LEADERBOARD_KINDS,
            kind_title="Blend or single product",
            panel_title=domain.capitalize(),
            keys=domain == "solar",
            row_step_px=_LEADERBOARD_ROW_STEP_PX,
        )
        for domain in DOMAIN_NAMES
    ]
    return figure(
        panels=panels,
        number=1,
        title=(
            "An XGBoost model given several weather products has the lowest error of all the "
            "single products and blends tested"
        ),
        subtitle=[
            (
                "Each row is an XGBoost model given the named product or blend of products, at "
                "the main XGBoost settings, ranked best first. Enriched: each product is also "
                "given the hours either side, and CAMS its beam split."
            ),
            (
                "The best row a live service anywhere in Great Britain can read: "
                + "; ".join(
                    f"{domain}, {best_live[domain]['label']} "
                    f"({_two_places(best_live[domain]['value'])}%)"
                    for domain in DOMAIN_NAMES
                )
                + "."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
            (
                "The intervals are wide mainly because every row's error rises and falls together "
                "from month to month. Figure 2 compares two XGBoost models on the same months, "
                "which cancels that shared swing, so two rows can overlap here and still differ "
                "significantly there."
            ),
        ],
        figure_planning=None,
    )


def _headline(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw each named set's enriched blend against its enriched best single, at both settings.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 2.
    """
    marks = []
    for named in NAMED_SETS:
        label = f"{named.domain.capitalize()}: {named.name}, against {SINGLE_NAMES[named.best]}"
        marks += [
            (
                label,
                _pick(
                    contrasts=contrasts,
                    section=SECTION_DECIDING,
                    domain=named.domain,
                    treatment=f"{named.blend}_rich_xgb",
                    reference=named.best,
                    setting=setting,
                ),
                condition,
            )
            for setting, condition in zip(
                ("pooled", "sensitivity"), SETTING_CONDITIONS, strict=True
            )
        ]
    panel = interval_panel(
        rows=_panel_rows(rows=marks),
        x_domain=HEADLINE_DOMAIN,
        x_title=X_TITLE,
        zero_label="same as the best single product",
        better_label="blend better",
        conditions=SETTING_CONDITIONS,
        condition_title="XGBoost hyperparameters",
    )
    return figure(
        panels=[panel],
        number=2,
        title=(
            "At these nine farms, an XGBoost model given several weather products beats an XGBoost "
            "model given the best single product and its neighbouring hours"
        ),
        subtitle=[
            (
                "Blend's mean absolute error minus the best single product's. Each best single "
                "product is given its neighbouring hours, and CAMS its beam split, as the blend's "
                "products are."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
            ALL_POST_HOC,
        ],
        figure_planning=None,
    )


def _decomposition(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Split each named blend's gain into the other products' weather and the extra columns alone.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 7.
    """
    conditions = ("Blend − control", "Control − best single product")
    marks = []
    for named in NAMED_SETS:
        blend = f"{named.blend}_rich_xgb"
        control = f"{named.blend}_rich_control"
        label = f"{named.domain.capitalize()}: {named.name}, against {SINGLE_NAMES[named.best]}"
        for (treatment, reference), condition in zip(
            ((blend, control), (control, named.best)), conditions, strict=True
        ):
            row = _pick(
                contrasts=contrasts,
                section=SECTION_DECIDING,
                domain=named.domain,
                treatment=treatment,
                reference=reference,
            )
            marks.append((label, row, condition))
    panel = interval_panel(
        rows=_panel_rows(rows=marks),
        x_domain=(-0.7, 0.2),
        x_title=X_TITLE,
        zero_label="no difference",
        better_label="first-named XGBoost model better",
        conditions=conditions,
        condition_title="Each blend's gain, split in two",
    )
    return figure(
        panels=[panel],
        number=7,
        title="The gain comes from the other products' weather, not from the extra columns",
        subtitle=[
            (
                "Orange: the blend minus its control, which measures the other products' weather. "
                "Blue: the control minus the best single product, which measures the extra columns "
                "alone. The control is the blend with every product but the best single one "
                "replaced by that product's values from other days of the same month, at the same "
                "hour."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
            ALL_POST_HOC,
        ],
        figure_planning=None,
    )


def _splits(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw each named enriched blend against its best single, per generator and per season.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 8.
    """
    panels = []
    for domain in ("solar", "wind"):
        named = [named for named in NAMED_SETS if named.domain == domain]
        conditions = tuple(n.name for n in named)
        scopes = [(f"site {site}", f"Generator {site}") for site in SITES[domain]]
        marks = [
            (
                label,
                _pick(
                    contrasts=contrasts,
                    section=SECTION_SPLITS,
                    domain=domain,
                    treatment=f"{n.blend}_rich_xgb",
                    reference=n.best,
                    scope=scope,
                ),
                condition,
            )
            for scope, label in [*scopes, *SEASONS, *ERAS]
            for n, condition in zip(named, conditions, strict=True)
        ]
        panels.append(
            interval_panel(
                rows=_panel_rows(rows=marks),
                x_domain=(-0.8, 0.2),
                x_title=X_TITLE if domain == "wind" else "",
                zero_label="same as the best single product",
                better_label="blend better",
                conditions=conditions,
                condition_title=f"{domain.capitalize()} blend",
                panel_title=(
                    f"{domain.capitalize()}: each generator, each season, and each side of UKV's "
                    "upgrade"
                ),
                reference_labels=domain == "solar",
                family_key=False,
            )
        )
    return figure(
        panels=panels,
        number=8,
        title=(
            "Each named blend beats its best single product at every generator, in every season, "
            "and on each side of UKV's upgrade"
        ),
        subtitle=[
            (
                "Enriched blend's mean absolute error minus the enriched best single product's, "
                "main XGBoost settings. The best single product is CAMS for the solar blends "
                "holding CAMS, ICON-D2 for the "
                f"{_blend_label(domain='solar', blend='live_all')} blend, and UKV for wind. "
                "The splits share hours and XGBoost models, so they are not independent tests."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning="exploratory",
    ).resolve_scale(color="independent", shape="independent")


def _methods(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the four ways of blending each named set, each against the set's best single.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 10.
    """
    methods = (
        ("xgb", "XGBoost given every product's columns"),
        ("stack", "Linear stack of the single-product predictions"),
        ("mean", "XGBoost given the products' mean"),
        ("equal", "Equal-weight mean of the predictions"),
    )
    close = _pick(
        contrasts=contrasts,
        section=SECTION_EXPLORATORY,
        domain="wind",
        treatment="live_gb_rich_stack",
        reference="live_gb_rich_xgb",
    )
    lead = _two_places(abs(float(str(close["difference"]))))
    panels = []
    for index, named in enumerate(NAMED_SETS):
        marks = []
        for method, label in methods:
            row = _pick(
                contrasts=contrasts,
                section=SECTION_DECIDING if method == "xgb" else SECTION_EXPLORATORY,
                domain=named.domain,
                treatment=f"{named.blend}_rich_{method}",
                reference=named.best,
            )
            marks.append((label + (POST_HOC_SUFFIX if method == "xgb" else ""), row, ""))
        panels.append(
            interval_panel(
                rows=_panel_rows(rows=marks),
                x_domain=(-0.6, 2.0),
                x_title=X_TITLE if index == len(NAMED_SETS) - 1 else "",
                zero_label="same as the best single product",
                better_label="blend better",
                panel_title=wrapped(
                    text=(
                        f"{named.domain.capitalize()}: {named.name}, "
                        f"against {SINGLE_NAMES[named.best]}"
                    ),
                    width=_INTERVAL_PANEL_TITLE_CHARACTERS,
                ),
                reference_labels=index == 0,
            )
        )
    return figure(
        panels=panels,
        number=10,
        title=(
            "Given every product's columns, an XGBoost model has the lowest error of the four "
            "blends in every named set"
        ),
        subtitle=[
            (
                "Each enriched blend's mean absolute error minus the enriched best single "
                "product's, main XGBoost settings. A linear stack weights the single-product "
                "XGBoost models' predictions, fitted per generator on the other folds."
            ),
            (
                "For UKV with ICON-EU, the XGBoost blend leads the linear stack by "
                f"{lead} points [{_signed(close['lower_95'])}, "
                f"{_signed(close['upper_95'])}], not statistically significant at the 5% level "
                "(exploratory)."
            ),
            f"{DOTS} {CAPACITY}",
            SOME_POST_HOC,
        ],
        figure_planning=None,
    )


def _synthetic(*, contrasts: pl.DataFrame) -> alt.VConcatChart:
    """Draw the synthetic product's gain beside each domain's headline blend.

    Args:
        contrasts: Every contrast row in the report.

    Returns:
        Figure 11.
    """
    panels = []
    pairs: tuple[tuple[DomainType, str], ...] = (("solar", "cams_rich"), ("wind", "ukv_rich"))
    for index, (domain, best) in enumerate(pairs):
        name = SINGLE_NAMES[best]
        everything = next(n for n in NAMED_SETS if n.domain == domain and n.blend == "everything")
        marks = [
            (
                f"{name} plus the synthetic product, against {name}",
                _pick(
                    contrasts=contrasts,
                    section=SECTION_SYNTHETIC,
                    domain=domain,
                    treatment="synthetic_xgb",
                    reference=best,
                ),
                "",
            ),
            (
                f"{name} plus the synthetic product, against its control",
                _pick(
                    contrasts=contrasts,
                    section=SECTION_SYNTHETIC,
                    domain=domain,
                    treatment="synthetic_xgb",
                    reference="synthetic_control",
                ),
                "",
            ),
            (
                f"For scale: {everything.name}, against {name}{POST_HOC_SUFFIX}",
                _pick(
                    contrasts=contrasts,
                    section=SECTION_DECIDING,
                    domain=domain,
                    treatment="everything_rich_xgb",
                    reference=best,
                ),
                "",
            ),
        ]
        panels.append(
            interval_panel(
                rows=_panel_rows(rows=marks),
                x_domain=HEADLINE_DOMAIN,
                x_title=X_TITLE if index == 1 else "",
                zero_label="no difference",
                better_label="first-named XGBoost model better",
                panel_title=domain.capitalize(),
                reference_labels=index == 0,
            )
        )
    return figure(
        panels=panels,
        number=11,
        title="A known small signal is recovered in full",
        subtitle=[
            (
                "The synthetic product is each hour's measured output plus random noise, sized so "
                "the synthetic gain would be about the size of the solar headline gain. The "
                "comparison recovers that known gain: 0.13 points for solar and 0.14 for wind."
            ),
            f"{DOTS} {CAPACITY}",
            f"{SCOPES['solar']} {SCOPES['wind']}",
            SOME_POST_HOC,
        ],
        figure_planning=None,
    )


def _seed_mean_predictions(*, domain: DomainType, arms: list[str]) -> pl.DataFrame:
    """Return each arm's out-of-fold prediction averaged over the seeds, beside the measured output.

    Args:
        domain: `solar` or `wind`.
        arms: The arms to read.

    Returns:
        One row per (site, time, arm), with `measured` and `predicted` as fractions of capacity.
    """
    capacity = (
        _losses(domain=domain, arms=arms[:1])
        .group_by("site")
        .agg(
            capacity=pl.col("effective_capacity_mw").first(),
            n_capacities=pl.col("effective_capacity_mw").n_unique(),
        )
    )
    if (capacity["n_capacities"] > 1).any():
        msg = f"a {domain} generator has more than one capacity"
        raise ValueError(msg)
    return (
        pl.scan_parquet(OUTPUT_DIR / "predictions.parquet")
        .filter(
            pl.col("domain") == domain,
            pl.col("setting") == "pooled",
            pl.col("arm").is_in(arms),
        )
        .group_by("site", "time", "arm")
        .agg(pl.col("power_mw").first(), pl.col("prediction_mw").mean())
        .collect()
        .join(capacity, on="site")
        .select(
            "site",
            "time",
            "arm",
            measured=pl.col("power_mw") / pl.col("capacity"),
            predicted=pl.col("prediction_mw") / pl.col("capacity"),
        )
    )


class Week(NamedTuple):
    """A week chosen by a stated rule, and the rule's name."""

    start: datetime
    rule: str


def _choose_weeks(*, measured: pl.DataFrame, domain: DomainType) -> list[Week]:
    """Choose three weeks by rule, from the weeks every generator covers well on all seven days.

    A generator covers a day well when the day holds at least `MIN_HOURS_PER_DAY` scored hours.

    Solar takes the week with the highest mean output, the week whose daily mean output varies most
    from day to day, and the week with the lowest mean output. Wind takes the week with the highest
    mean output, the week with the largest mean hour-to-hour change in output, and the week with
    the lowest mean output. Each rule picks from the weeks the rules before it left.

    Args:
        measured: One row per (site, time) with `measured` as a fraction of capacity.
        domain: `solar` or `wind`.

    Returns:
        The three weeks, each starting on a Monday.
    """
    frame = measured.sort("site", "time").with_columns(
        week=pl.col("time").dt.truncate("1w"),
        day=pl.col("time").dt.date(),
        step=(pl.col("measured") - pl.col("measured").shift(1).over("site")).abs(),
    )
    daily = frame.group_by("week", "day").agg(daily=pl.col("measured").mean())
    covered = (
        frame.group_by("week", "site", "day")
        .len()
        .filter(pl.col("len") >= MIN_HOURS_PER_DAY[domain])
        .group_by("week")
        .len()
        .filter(pl.col("len") == len(SITES[domain]) * 7)
    )
    weeks = (
        frame.group_by("week")
        .agg(mean=pl.col("measured").mean(), step=pl.col("step").mean())
        .join(daily.group_by("week").agg(day_to_day=pl.col("daily").std()), on="week")
        .join(covered.select("week"), on="week")
    )
    variable = (
        ("day_to_day", "Most variable from day to day")
        if domain == "solar"
        else (
            "step",
            "Most variable from hour to hour",
        )
    )
    rules = [
        ("mean", "Highest output", True),
        (variable[0], variable[1], True),
        ("mean", "Lowest output", False),
    ]
    chosen: list[Week] = []
    for column, rule, descending in rules:
        left = weeks.filter(~pl.col("week").is_in([week.start for week in chosen]))
        start = left.sort(column, descending=descending)["week"][0]
        chosen.append(Week(start=start, rule=rule))
    return chosen


def _line_key(
    *, labels: Sequence[str], colours: Sequence[str], width: int = CONTENT_WIDTH_PX
) -> alt.LayerChart:
    """Draw a one-row key of short line segments, above a figure's panels.

    A label too long for its slot wraps onto more lines instead of being cut off with an
    ellipsis mid-word, the same estimate `_key` in `studies.charts` rests on: about 7 px a
    character at this text mark's font size.

    Args:
        labels: Each line's label.
        colours: Each line's colour.
        width: The key's width, which a figure whose panels carry row labels narrows.

    Returns:
        A one-row chart, taller where a label needs more than one line.
    """
    slot = width // len(labels)
    chars_per_line = max(10, (slot - 30) // 7)
    wrapped_labels = [wrapped(text=label, width=chars_per_line) for label in labels]
    lines = max(len(label_lines) for label_lines in wrapped_labels)
    data = pl.DataFrame(
        {
            "label": ["\n".join(label_lines) for label_lines in wrapped_labels],
            "colour": list(colours),
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
        .mark_text(
            align="left",
            baseline="middle",
            dx=24,
            dy=8,
            color=ocf.BLACK_1,
            lineHeight=13,
            lineBreak="\n",
        )
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[segments, text], width=width, height=16 if lines == 1 else 16 + 13 * (lines - 1)
    )


def _point_key(
    *, labels: list[str], colours: list[str], width: int = CONTENT_WIDTH_PX
) -> alt.LayerChart:
    """Draw a one-row key of a filled circle and a filled diamond, above a figure's panels.

    Args:
        labels: The two entries' labels.
        colours: The two entries' colours.
        width: The key's width, which a figure whose panels carry row labels narrows.

    Returns:
        A one-row chart.
    """
    slot = width // len(labels)
    data = pl.DataFrame(
        {
            "label": labels,
            "colour": colours,
            "shape": ["circle", "diamond"][: len(labels)],
            "x": [8 + index * slot for index in range(len(labels))],
        }
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=70, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=None),
            y=alt.value(8),
            color=alt.Color("colour:N", scale=None),
            shape=alt.Shape("shape:N", scale=None),
        )
    )
    text = (
        alt.Chart(data)
        .mark_text(align="left", dx=12, color=ocf.BLACK_1, limit=slot - 30)
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(layer=[points, text], width=width, height=16)


def _week_panel(
    *,
    series: pl.DataFrame,
    week: Week,
    site: str,
    first_row: bool,
    last_row: bool,
    first_column: bool,
) -> alt.LayerChart:
    """Draw one generator's week: the measured output and both XGBoost models' predictions.

    Args:
        series: Long rows for this site and week: `time`, `series`, `value`, with a null at every
            hour the common rows lack, so the lines break there.
        week: The week.
        site: The generator's label.
        first_row: Whether to title the panel with the week.
        last_row: Whether to draw the date axis.
        first_column: Whether to draw the output axis and the generator's label.

    Returns:
        The panel.
    """
    width = (CONTENT_WIDTH_PX - WEEK_ROW_LABEL_PX - 2 * WEEK_SPACING_PX) // 3
    names = series["series"].unique(maintain_order=True).to_list()
    x = alt.X(
        "time:T",
        title=None,
        scale=alt.Scale(
            domain=[week.start.isoformat(), (week.start + timedelta(days=7)).isoformat()]
        ),
        axis=alt.Axis(
            format="%a",
            values=[int((week.start + timedelta(days=day)).timestamp() * 1000) for day in range(7)],
            labelAlign="left",
            labels=last_row,
            ticks=last_row,
            grid=True,
        ),
    )
    y = alt.Y(
        "value:Q",
        title=site if first_column else None,
        scale=alt.Scale(domain=[0, 110], nice=False),
        axis=alt.Axis(values=[0, 50, 100], labels=first_column, ticks=first_column, titleAngle=0),
    )
    line = (
        alt.Chart(series)
        .mark_line(strokeWidth=1.3, interpolate="linear")
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=y,
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=names, range=[MEASURED_COLOUR, SINGLE_COLOUR, BLEND_COLOUR]),
                legend=None,
            ),
        )
    )
    title = (
        alt.TitleParams([week.rule, f"week of {week.start:%-d %B %Y}"], anchor="start", fontSize=12)
        if first_row
        else alt.TitleParams("")
    )
    return alt.LayerChart(layer=[line], width=width, height=WEEK_PANEL_HEIGHT_PX, title=title)


def _weeks_figure(
    *,
    domain: DomainType,
    single: str,
    blend: str,
    labels: tuple[str, str],
    number: int,
    title: str,
    errors: pl.DataFrame,
) -> alt.VConcatChart:
    """Draw three weeks at every generator: measured output, best single, and best blend.

    Args:
        domain: `solar` or `wind`.
        single: The enriched best single product's arm.
        blend: The enriched blend's arm.
        labels: The key labels of the single product and the blend.
        number: The figure's number.
        title: The figure's title.
        errors: The domain's enriched blend errors, to check the predictions against.

    Returns:
        The figure.

    Raises:
        ValueError: If either arm's whole-record error differs from the report's.
    """
    everything = errors.filter(pl.col("set") == "everything").row(0, named=True)
    losses = _losses(domain=domain, arms=[single, blend])
    for arm, printed in ((single, everything["best_mae"]), (blend, everything["xgb"])):
        computed = losses.filter(pl.col("arm") == arm).select(pl.col(METRIC).mean()).item()
        if round(computed * PERCENTAGE_POINTS, 3) != printed:
            msg = f"{arm}'s error {computed} does not reproduce the report's {printed}"
            raise ValueError(msg)
    predictions = _seed_mean_predictions(domain=domain, arms=[single, blend])
    measured = predictions.filter(pl.col("arm") == single).select("site", "time", "measured")
    weeks = _choose_weeks(measured=measured, domain=domain)
    names = ["Measured output", *labels]
    long = pl.concat(
        [
            measured.select("site", "time", series=pl.lit(names[0]), value=pl.col("measured")),
            *(
                predictions.filter(pl.col("arm") == arm).select(
                    "site", "time", series=pl.lit(name), value=pl.col("predicted")
                )
                for arm, name in ((single, labels[0]), (blend, labels[1]))
            ),
        ]
    ).with_columns(pl.col("value") * PERCENTAGE_POINTS)
    rows = []
    for row_index, site in enumerate(SITES[domain]):
        cells = []
        for column_index, week in enumerate(weeks):
            hours = pl.datetime_range(
                week.start, week.start + timedelta(days=7), interval="1h", closed="left", eager=True
            ).alias("time")
            grid = pl.DataFrame(hours).join(pl.DataFrame({"series": names}), how="cross")
            series = grid.join(
                long.filter(pl.col("site") == site).drop("site"), on=["time", "series"], how="left"
            ).sort(pl.col("series").replace_strict(names, list(range(3))), "time")
            cells.append(
                _week_panel(
                    series=series,
                    week=week,
                    site=site,
                    first_row=row_index == 0,
                    last_row=row_index == len(SITES[domain]) - 1,
                    first_column=column_index == 0,
                )
            )
        rows.append(alt.hconcat(*cells, spacing=WEEK_SPACING_PX))
    grid_chart = alt.vconcat(*rows, spacing=4)
    key = _line_key(
        labels=names,
        colours=[MEASURED_COLOUR, SINGLE_COLOUR, BLEND_COLOUR],
        width=CONTENT_WIDTH_PX - 60,
    )
    return figure(
        panels=[key, grid_chart],
        number=number,
        title=title,
        subtitle=[
            (
                "Hourly output as a percentage of capacity, measured and as predicted out of fold "
                "by each XGBoost model, averaged over its three fitting seeds. One row per "
                "generator; gaps are hours not covered by every product."
            ),
            (
                "Weeks chosen by rule from the weeks in which every generator has at least "
                f"{MIN_HOURS_PER_DAY[domain]} scored hours on each of the seven days: "
                + (
                    "the highest mean output, the largest day-to-day spread in daily mean output, "
                    "and the lowest mean output."
                    if domain == "solar"
                    else "the highest mean output, the largest mean hour-to-hour change, and the "
                    "lowest mean output."
                )
            ),
            f"{CAPACITY} {SCOPES[domain]}",
        ],
        figure_planning=None,
    )


def _per_generator_errors(
    *, contrasts: pl.DataFrame, errors: dict[DomainType, pl.DataFrame]
) -> alt.VConcatChart:
    """Draw each named set's enriched blend and best single's mean absolute error per generator.

    Args:
        contrasts: Every contrast row in the report, to check each generator's difference.
        errors: Each domain's enriched blend errors.

    Returns:
        Figure 6.

    Raises:
        ValueError: If a generator's difference does not reproduce the report's per-site row.
    """
    pairs = NAMED_SETS
    x_scale = alt.Scale(domain=[2, 10], nice=False)
    panels = []
    for index, named in enumerate(pairs):
        blend = f"{named.blend}_rich_xgb"
        by_site = (
            _losses(domain=named.domain, arms=[blend, named.best])
            .group_by("site", "arm")
            .agg(mae=pl.col(METRIC).mean() * PERCENTAGE_POINTS)
            .pivot("arm", index="site", values="mae")
            .sort("site")
        )
        for site, single_mae, blend_mae in by_site.select("site", named.best, blend).iter_rows():
            printed = _pick(
                contrasts=contrasts,
                section=SECTION_SPLITS,
                domain=named.domain,
                treatment=blend,
                reference=named.best,
                scope=f"site {site}",
            )["difference"]
            if round(blend_mae - single_mae, 3) != printed:
                msg = f"{named}, generator {site}: {blend_mae - single_mae} against {printed}"
                raise ValueError(msg)
        error = errors[named.domain].filter(pl.col("set") == named.blend).row(0, named=True)
        long = pl.concat(
            [
                by_site.select(
                    "site",
                    model=pl.lit(MODEL_KEYS[0]),
                    mae=pl.col(named.best),
                ),
                by_site.select(
                    "site",
                    model=pl.lit(MODEL_KEYS[1]),
                    mae=pl.col(blend),
                ),
            ]
        )
        y = alt.Y(
            "site:N",
            title=None,
            sort=list(SITES[named.domain]),
            axis=alt.Axis(labelExpr="'Generator ' + datum.value", ticks=False, domain=False),
        )
        x = alt.X(
            "mae:Q",
            title=(
                ["Mean absolute error (% of capacity; smaller is better)"]
                if index == len(pairs) - 1
                else None
            ),
            scale=x_scale,
            axis=alt.Axis(values=list(range(2, 11))),
        )
        colour = alt.Color(
            "model:N",
            scale=alt.Scale(domain=list(MODEL_KEYS), range=[SINGLE_COLOUR, BLEND_COLOUR]),
            legend=None,
        )
        link = (
            alt.Chart(by_site.select("site", low=pl.col(named.best), high=pl.col(blend)))
            .mark_rule(color=ocf.ENSEMBLE_LINE, strokeWidth=1.5)
            .encode(x=alt.X("low:Q", scale=x_scale), x2="high:Q", y=y)  # ty: ignore[unresolved-attribute]
        )
        points = (
            alt.Chart(long)
            .mark_point(filled=True, size=70, opacity=1)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x,
                y=y,
                color=colour,
                shape=alt.Shape(
                    "model:N",
                    scale=alt.Scale(domain=list(MODEL_KEYS), range=["circle", "diamond"]),
                    legend=None,
                ),
            )
        )
        panels.append(
            alt.LayerChart(
                layer=[link, points],
                width=CONTENT_WIDTH_PX - 100,
                height=alt.Step(20),
                title=alt.TitleParams(
                    wrapped(
                        text=(
                            f"{named.domain.capitalize()}: {named.name} "
                            f"({_two_places(error['xgb'])}%), against {SINGLE_NAMES[named.best]} "
                            f"({_two_places(error['best_mae'])}%)"
                        ),
                        width=_GENERATOR_PANEL_TITLE_CHARACTERS,
                    ),
                    anchor="start",
                    fontSize=14,
                ),
            )
        )
    key = _point_key(
        labels=list(MODEL_KEYS), colours=[SINGLE_COLOUR, BLEND_COLOUR], width=CONTENT_WIDTH_PX - 100
    )
    return figure(
        panels=[key, *panels],
        number=6,
        title="Each blend's error is lower than its best single product's at every generator",
        subtitle=[
            (
                "Mean absolute error per generator, main XGBoost settings, averaged over the "
                "three fitting seeds; each panel title gives the whole-record errors. Computed "
                "for this chart from the saved per-hour errors, and checked against the report's "
                "per-generator differences."
            ),
            f"{CAPACITY} {SCOPES['solar']} {SCOPES['wind']}",
        ],
        figure_planning=None,
    )


def _band_rows(*, tables: dict[str, list[dict[str, str]]]) -> pl.DataFrame:
    """Read the wind gain by measured-output band, for the two deciding wind blends.

    Args:
        tables: The output of `_tables`.

    Returns:
        One row per (contrast, band), with `contrast`, `band`, `share_of_rows` and `difference`.
    """
    return pl.DataFrame(
        [
            {
                "contrast": row["Contrast"],
                "band": row["Measured output, fraction of capacity"],
                "share_of_rows": float(row["Share of rows"].rstrip("%")),
                "difference": float(row["ΔMAE within the band"]),
            }
            for row in tables[SECTION_BANDS]
            if "_rich_" in row["Contrast"]
        ]
    )


def _cumulative_share(*, treatment: str, reference: str, report_text: str) -> pl.DataFrame:
    """Return the cumulative share of a wind contrast's net gain, the most-improved hours first.

    Args:
        treatment: The blend.
        reference: The best single product.
        report_text: The report, whose most-improved-rows line this reproduces.

    Returns:
        One row per percentile of hours, with `hours` and `share` in percent.

    Raises:
        ValueError: If the share at `MOST_IMPROVED_SHARE`, or the share of hours improving, does
            not reproduce the report's line.
    """
    losses = _losses(domain="wind", arms=[treatment, reference])
    rows = (
        losses.pivot("arm", index=["site", "time", "seed"], values=METRIC)
        .group_by("site", "time")
        .agg(pl.col(treatment).mean(), pl.col(reference).mean())
    )
    ordered = np.sort((rows[treatment] - rows[reference]).to_numpy())
    total = ordered.sum()
    top = ordered[: int(len(ordered) * MOST_IMPROVED_SHARE)].sum() / total
    expected = (
        f"{treatment} − {reference}: the most-improved {MOST_IMPROVED_SHARE:.0%} of rows carry "
        f"{top:.0%} of the whole gain, and {(ordered < 0).mean():.0%} of rows improve."
    )
    if expected not in report_text:
        msg = f"the recomputed concentration does not match the report: {expected}"
        raise ValueError(msg)
    cumulative = np.cumsum(ordered) / total
    positions = np.linspace(0, len(ordered), 101).astype(int)
    return pl.DataFrame(
        {
            "hours": np.linspace(0, 100, 101),
            "share": np.concatenate([[0.0], cumulative[positions[1:] - 1]]) * PERCENTAGE_POINTS,
        }
    )


def _wind_bands(*, tables: dict[str, list[dict[str, str]]], report_text: str) -> alt.VConcatChart:
    """Draw the wind gain by measured-output band, and the cumulative share of the net gain.

    Args:
        tables: The output of `_tables`.
        report_text: The report.

    Returns:
        Figure 9.
    """
    contrasts = {
        "everything_rich_xgb − ukv_rich": _blend_label(domain="wind", blend="everything"),
        "live_gb_rich_xgb − ukv_rich": "UKV and ICON-EU",
    }
    names = list(contrasts.values())
    colour = alt.Color(
        "blend:N", scale=alt.Scale(domain=names, range=[BLEND_COLOUR, SINGLE_COLOUR]), legend=None
    )
    bands = _band_rows(tables=tables).with_columns(
        blend=pl.col("contrast").replace_strict(contrasts),
        band_label=pl.col("band")
        + " ("
        + pl.col("share_of_rows").round(0).cast(pl.Int32).cast(pl.Utf8)
        + "% of hours)",
    )
    band_order = bands["band_label"].unique(maintain_order=True).to_list()
    bars = (
        alt.Chart(bands)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "band_label:N",
                sort=band_order,
                title=None,
                axis=alt.Axis(minExtent=210, maxExtent=210),
            ),
            yOffset=alt.YOffset("blend:N", sort=names),
            x=alt.X(
                "difference:Q",
                title=wrapped(
                    text=(
                        "Mean absolute error minus UKV's, within the band (points of capacity; "
                        "more negative means blend better)"
                    ),
                    width=62,
                ),
                scale=alt.Scale(domain=[-0.9, 0.0], nice=False),
            ),
            color=colour,
        )
        .properties(
            width=CONTENT_WIDTH_PX - 210,
            height=alt.Step(12),
            title=alt.TitleParams(
                "The gain within each band of measured output, as a fraction of capacity",
                anchor="start",
                fontSize=14,
            ),
        )
    )
    curves = pl.concat(
        _cumulative_share(
            treatment=treatment, reference="ukv_rich", report_text=report_text
        ).with_columns(blend=pl.lit(name))
        for (treatment, name) in (
            (contrast.split(" − ")[0], name) for contrast, name in contrasts.items()
        )
    )
    x = alt.X(
        "hours:Q",
        title="Share of hours, the most improved first (%)",
        scale=alt.Scale(domain=[0, 100], nice=False),
    )
    curve = (
        alt.Chart(curves)
        .mark_line(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "share:Q", title=wrapped(text="Cumulative share of the net gain (%)", width=30)
            ),
            color=colour,
        )
    )
    references = (
        alt.Chart(pl.DataFrame({"share": [100.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
        .encode(y="share:Q")  # ty: ignore[unresolved-attribute]
    )
    five = (
        alt.Chart(pl.DataFrame({"hours": [MOST_IMPROVED_SHARE * PERCENTAGE_POINTS]}))
        .mark_rule(color=ocf.ENSEMBLE_LINE)
        .encode(x="hours:Q")  # ty: ignore[unresolved-attribute]
    )
    labels = [
        alt.Chart(pl.DataFrame({"hours": [hours], "share": [share], "text": [text]}))
        .mark_text(align=align, baseline="bottom", dx=dx, dy=-3, color=ocf.BLACK_1)
        .encode(x="hours:Q", y="share:Q", text="text:N")  # ty: ignore[unresolved-attribute]
        for hours, share, text, align, dx in (
            (100.0, 100.0, "the net gain, 100%", "right", -4),
            (MOST_IMPROVED_SHARE * PERCENTAGE_POINTS, 20.0, "5% of hours", "left", 4),
        )
    ]
    lower = alt.LayerChart(
        layer=[references, five, curve, *labels],
        width=CONTENT_WIDTH_PX - 210,
        height=200,
        title=alt.TitleParams(
            "Hours sorted from most improved to most worsened", anchor="start", fontSize=14
        ),
    )
    return figure(
        panels=[
            _line_key(
                labels=names, colours=[BLEND_COLOUR, SINGLE_COLOUR], width=CONTENT_WIDTH_PX - 210
            ),
            bars,
            lower,
        ],
        number=9,
        title="The wind blends' gain is spread across every level of output",
        subtitle=[
            (
                "Top, from the report: each blend's mean absolute error minus enriched UKV's, "
                "within each band of measured output. Bottom, computed for this chart and checked "
                "against the report: the running sum of each hour's improvement, as a share of the "
                "net gain."
            ),
            (
                "The bottom curves rise above 100% because the hours that improve gain more than "
                "the net; the hours that worsen then bring the sum back down to 100%."
            ),
            f"{CAPACITY} {SCOPES['wind']}",
        ],
        figure_planning="exploratory",
    )


def main() -> int:
    """Read the report, check it against the saved results, and write the SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_text = (OUTPUT_DIR / "report.md").read_text()
    tables = _tables(report_text=report_text)
    contrasts = _contrasts(tables=tables)
    _check_against_intervals(contrasts=contrasts)
    _reproduce(report_text=report_text)
    errors: dict[DomainType, pl.DataFrame] = {
        domain: _blend_errors(tables=tables, domain=domain) for domain in ("solar", "wind")
    }
    charts = {
        "blend_leaderboard": _leaderboard(tables=tables),
        "blend_headline": _headline(contrasts=contrasts),
        "blend_solar_weeks": _weeks_figure(
            domain="solar",
            single="cams_rich",
            blend="everything_rich_xgb",
            labels=("CAMS, enriched", _blend_label(domain="solar", blend="everything")),
            number=4,
            title=(
                "XGBoost models given CAMS, or "
                f"{_blend_label(domain='solar', blend='everything')}, track measured solar output"
            ),
            errors=errors["solar"],
        ),
        "blend_wind_weeks": _weeks_figure(
            domain="wind",
            single="ukv_rich",
            blend="everything_rich_xgb",
            labels=("UKV, enriched", _blend_label(domain="wind", blend="everything")),
            number=5,
            title=(
                "XGBoost models given UKV, or "
                f"{_blend_label(domain='wind', blend='everything')}, track measured wind output"
            ),
            errors=errors["wind"],
        ),
        "blend_per_generator": _per_generator_errors(contrasts=contrasts, errors=errors),
        "blend_decomposition": _decomposition(contrasts=contrasts),
        "blend_splits": _splits(contrasts=contrasts),
        "blend_wind_bands": _wind_bands(tables=tables, report_text=report_text),
        "blend_methods": _methods(contrasts=contrasts),
        "blend_synthetic": _synthetic(contrasts=contrasts),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
