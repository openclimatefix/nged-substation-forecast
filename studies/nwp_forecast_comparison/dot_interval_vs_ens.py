"""Draw each weather product's error minus the ECMWF ENS mean's, as dots with 95% intervals.

One figure per technology (solar, wind), one panel per lead day (0, 1, 2, 3, 4, 5, 7, 10, and 14,
the days of the page's leaderboards), one row per weather product the leaderboards carry. A dot is
the product's mean absolute error minus the ENS mean's at the same lead day, in percentage points
of capacity, so a negative dot means the product forecasts better. The zero line is the ENS mean.
The interval is the 95% interval from resampling whole calendar months and a fitting seed
(`studies.bootstrap.bootstrap_difference`). A row that rests on fewer than
`MIN_MONTHS_FOR_INTERVAL` months gets a hollow dot and no interval.

It reads the saved per-row losses of the fits under `data/studies/nwp_forecast_comparison_*` and
fits nothing. **Each row uses the arms the leaderboards draw**: an arm that a published CPU fit and
a later GPU refit both hold is read from the GPU refit, as the leaderboards do, so a product and
its reference were fitted on one device. The reference is the leaderboard's ENS-mean arm of the
same lead day:

- The products in `_leads_day10`, `_leads_day10b`, `_leads_day10c`, and `_leads_day10d` are
  subtracted from the ENS mean of `LEADERBOARD_ENS_SOURCES`: `_leads_day10b` at days 2 and 7 and
  `_leads_day10` at every other day. The folders hold the same `(site, time, seed)` keys, so the
  pair joins across folders.
- AIFS Single, the AIFS ENS mean, and WeatherNext 3 (WN3) each sit in a folder with an ENS mean
  fitted on the same rows, which is the reference. For wind, WN3's reference is `ens_meanvec`, the
  ENS mean built from the mean-vector speed, which matches how WN3's speed is built.

WN3 appears only as the pooled row, which covers 7 months (February to April and June to September
2026). The script writes `report.md`, `intervals.parquet`, and `README.md` to a new output folder,
and one SVG per technology. It refuses to overwrite any of them.

Run it with `uv run python studies/nwp_forecast_comparison/dot_interval_vs_ens.py`.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple

import altair as alt
import polars as pl
from nwp_forecast_charts import (
    CAPACITY_NOTE,
    DOTS_NOTE,
    PRODUCT_NAMES,
    TECHNOLOGY_NAMES,
    check_anonymised,
    padded_domain,
)
from nwp_forecast_comparison import METRIC, PERCENTAGE_POINTS, DomainType
from studies.bootstrap import MIN_MONTHS_FOR_INTERVAL, bootstrap_difference
from studies.charts import figure, interval_panel

_LOG: Final[logging.Logger] = logging.getLogger("dot_interval_vs_ens")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 7, 10, 14)
"""The lead days of the page's leaderboards, one panel each."""

PRIMARY_SETTING: Final[str] = "primary"
"""The hyperparameter setting every row is read at."""

LOSS_COLUMNS: Final[tuple[str, ...]] = ("arm", "site", "time", "seed", "month", "setting", METRIC)
"""The columns read from each saved losses file."""

KEY_COLUMNS: Final[tuple[str, ...]] = ("site", "time", "seed")
"""The columns that identify one row of one arm; a product and its reference pair on them."""

SourceType = Literal[
    "leads_day10",
    "leads_day10b",
    "leads_day10c",
    "leads_day10d",
    "aifs_single_blends",
    "aifs_ens_blends",
    "aifs_single_extra",
    "aifs_ens_extra",
    "wn3_blends",
    "wn3_extra",
]
"""The folders a row's losses come from."""


class Source(NamedTuple):
    """Where one fit batch saved its per-row losses."""

    folder: str
    pattern: str


_AIFS_BLENDS: Final[str] = "nwp_forecast_comparison_aifs_blends"
_AIFS_EXTRA: Final[str] = "nwp_forecast_comparison_aifs_extra_days"
_PER_DAY: Final[str] = "{{domain}}_{name}_day{{day}}_losses.parquet"

SOURCES: Final[dict[SourceType, Source]] = {
    "leads_day10": Source("nwp_forecast_comparison_leads_day10", "{domain}_losses.parquet"),
    "leads_day10b": Source("nwp_forecast_comparison_leads_day10b", "{domain}_losses.parquet"),
    "leads_day10c": Source("nwp_forecast_comparison_leads_day10c", "{domain}_losses.parquet"),
    "leads_day10d": Source("nwp_forecast_comparison_leads_day10d", "{domain}_losses.parquet"),
    "aifs_single_blends": Source(_AIFS_BLENDS, _PER_DAY.format(name="single")),
    "aifs_ens_blends": Source(_AIFS_BLENDS, _PER_DAY.format(name="ens")),
    "aifs_single_extra": Source(_AIFS_EXTRA, _PER_DAY.format(name="single")),
    "aifs_ens_extra": Source(_AIFS_EXTRA, _PER_DAY.format(name="ens")),
    "wn3_blends": Source("nwp_forecast_comparison_wn3", _PER_DAY.format(name="wn3")),
    "wn3_extra": Source("nwp_forecast_comparison_wn3_extra_days", _PER_DAY.format(name="wn3")),
}
"""Each source's folder under `data/studies/`, and its losses file's name. A `{day}` in the name
means one file per lead day. The `_blends` folders hold days 1, 2, 7, and 14 and the `_extra`
folders days 0, 3, 4, and 10. The superseded `nwp_forecast_comparison_leads` folder is not a
source, because the leaderboards do not draw it."""

REFERENCE_NAMES: Final[dict[str, str]] = {
    "ens_mean": "ENS mean",
    "ens_meanvec": "ENS mean (mean-vector speed)",
}
"""Each reference arm prefix's name in a report."""

WN3_LABEL: Final[str] = "WeatherNext 3 mean (7 months)"
"""WN3's row label; its rows cover February to April and June to September 2026 only."""


def _sources(
    days: tuple[int, ...], default: SourceType, overrides: Mapping[int, SourceType] | None = None
) -> dict[int, SourceType]:
    """Map each lead day to its source: `default`, except for the days in `overrides`."""
    return {day: (overrides or {}).get(day, default) for day in days}


class ProductRow(NamedTuple):
    """One product's arm prefix, the source of each lead day it is fitted at, and its reference."""

    prefix: str
    sources: Mapping[int, SourceType]
    reference_sources: Mapping[int, SourceType] | None = None
    domains: tuple[DomainType, ...] = DOMAINS
    wind_reference: str = "ens_mean"
    label: str | None = None


LEADERBOARD_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 7, 10, 14)
"""The lead days of the products fitted in the `leads_day10*` folders (and ENS mean and GEFS)."""

AIFS_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 7, 10, 14)
"""The lead days of AIFS Single, the AIFS ENS mean, and WN3: every day but 5."""

LEADERBOARD_ENS_SOURCES: Final[dict[int, SourceType]] = _sources(
    LEADERBOARD_DAYS, "leads_day10", {2: "leads_day10b", 7: "leads_day10b"}
)
"""The folder of the ENS-mean arm the leaderboards draw at each lead day. The published folder
holds ENS-mean arms at days 0 to 3 too, but the leaderboards read the GPU refit that each of these
folders holds, which `load`'s `prefer_extras` picks."""

EXTRA_DAYS: Final[tuple[int, ...]] = (0, 3, 4, 10)
"""The AIFS and WN3 lead days that sit in the `_extra` folders."""

_B: Final[SourceType] = "leads_day10b"
_LEADS_TO_3: Final[tuple[int, ...]] = (0, 1, 2, 3)
_LEADS_TO_7: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 7)


def _leaderboard_row(
    prefix: str,
    sources: Mapping[int, SourceType],
    domains: tuple[DomainType, ...] = DOMAINS,
) -> ProductRow:
    """Return a product whose reference is the leaderboard's ENS mean at each of its days."""
    return ProductRow(prefix, sources, reference_sources=LEADERBOARD_ENS_SOURCES, domains=domains)


PRODUCTS: Final[tuple[ProductRow, ...]] = (
    _leaderboard_row("ukv", _sources((0, 1), "leads_day10")),
    _leaderboard_row("icon_d2", _sources((0, 1), "leads_day10")),
    _leaderboard_row("icon_eu", _sources(_LEADS_TO_3, "leads_day10", {2: _B, 3: _B})),
    _leaderboard_row("icon_global", _sources((0, 1, 2, 3, 5), "leads_day10", {2: _B})),
    _leaderboard_row("gfs", _sources(_LEADS_TO_7, "leads_day10", {2: _B})),
    _leaderboard_row("gfs_native", _sources(LEADERBOARD_DAYS, "leads_day10c")),
    _leaderboard_row("ifs025", _sources(_LEADS_TO_7, "leads_day10", {2: _B})),
    _leaderboard_row("ifs_single", _sources(_LEADS_TO_7, "leads_day10d")),
    _leaderboard_row("arpege", _sources(_LEADS_TO_3, "leads_day10", {2: _B, 3: _B}), ("solar",)),
    _leaderboard_row("arome", _sources((0, 1), "leads_day10"), ("solar",)),
    _leaderboard_row("dmi_harmonie", _sources((0, 1), "leads_day10")),
    _leaderboard_row("knmi_harmonie", _sources((0, 1), "leads_day10")),
    _leaderboard_row("gefs_mean", _sources(LEADERBOARD_DAYS, "leads_day10", {2: _B, 7: _B})),
    _leaderboard_row(
        "ens_control",
        _sources(LEADERBOARD_DAYS, _B, {0: "leads_day10", 1: "leads_day10"}),
    ),
    ProductRow(
        "aifs_single",
        _sources(AIFS_DAYS, "aifs_single_blends", dict.fromkeys(EXTRA_DAYS, "aifs_single_extra")),
    ),
    ProductRow(
        "aifs_ens_mean",
        _sources(AIFS_DAYS, "aifs_ens_blends", dict.fromkeys(EXTRA_DAYS, "aifs_ens_extra")),
    ),
    ProductRow(
        "wn3_mean",
        _sources(AIFS_DAYS, "wn3_blends", dict.fromkeys(EXTRA_DAYS, "wn3_extra")),
        wind_reference="ens_meanvec",
        label=WN3_LABEL,
    ),
)
"""Every row the figures draw: each product the leaderboards carry, except the ENS mean, which is
the reference. A product at a lead day no row names is not in the archive or not fitted. Day 4
holds only AIFS and WN3, and day 5 holds neither."""


class Comparison(NamedTuple):
    """One dot: a product's arm at a lead day, and the ENS-mean arm it is subtracted from."""

    domain: DomainType
    day: int
    label: str
    treatment: str
    treatment_source: SourceType
    reference: str
    reference_source: SourceType
    reference_label: str


def comparisons(*, domain: DomainType) -> list[Comparison]:
    """List every dot of one technology's figure.

    Args:
        domain: `solar` or `wind`.

    Returns:
        One comparison per product and lead day, in `PRODUCTS` order.
    """
    output = []
    for product in PRODUCTS:
        if domain not in product.domains:
            continue
        reference_prefix = product.wind_reference if domain == "wind" else "ens_mean"
        reference_sources = product.reference_sources or product.sources
        output.extend(
            Comparison(
                domain=domain,
                day=day,
                label=product.label or PRODUCT_NAMES[product.prefix],
                treatment=f"{product.prefix}_day{day}",
                treatment_source=source,
                reference=f"{reference_prefix}_day{day}",
                reference_source=reference_sources[day],
                reference_label=REFERENCE_NAMES[reference_prefix],
            )
            for day, source in product.sources.items()
        )
    return output


_DAY_SUFFIX: Final[re.Pattern[str]] = re.compile(r"_day(?P<day>\d+)$")


def losses_path(*, data_dir: Path, source: SourceType, domain: DomainType, arm: str) -> Path:
    """Return the saved losses file that holds an arm.

    Args:
        data_dir: The `data/studies` directory.
        source: The fit batch the arm came from.
        domain: `solar` or `wind`.
        arm: The arm, ending in `_day<N>`.

    Returns:
        The file's path.

    Raises:
        ValueError: If the arm's name does not end in `_day<N>`.
    """
    match = _DAY_SUFFIX.search(arm)
    if match is None:
        msg = f"arm {arm!r} does not end in _day<N>"
        raise ValueError(msg)
    spec = SOURCES[source]
    return data_dir / spec.folder / spec.pattern.format(domain=domain, day=match["day"])


def load_arms(
    *, data_dir: Path, domain: DomainType, wanted: Collection[tuple[SourceType, str]]
) -> dict[tuple[SourceType, str], pl.DataFrame]:
    """Read the primary-setting losses of each wanted (source, arm), one file read per file.

    Args:
        data_dir: The `data/studies` directory.
        domain: `solar` or `wind`.
        wanted: The (source, arm) pairs to read.

    Returns:
        Each pair's rows, with `LOSS_COLUMNS`.

    Raises:
        ValueError: If an arm has no rows at the primary setting, holds a site label that is not
            an anonymised label, or holds one `(site, time, seed)` key twice, which would make a
            paired difference join one row to two.
    """
    by_file: dict[Path, list[tuple[SourceType, str]]] = {}
    for source, arm in sorted(set(wanted)):
        path = losses_path(data_dir=data_dir, source=source, domain=domain, arm=arm)
        by_file.setdefault(path, []).append((source, arm))
    output: dict[tuple[SourceType, str], pl.DataFrame] = {}
    for path, pairs in by_file.items():
        frame = (
            pl.scan_parquet(path)
            .select(LOSS_COLUMNS)
            .filter(
                pl.col("setting") == PRIMARY_SETTING,
                pl.col("arm").is_in([arm for _, arm in pairs]),
            )
            .collect()
        )
        for source, arm in pairs:
            rows = frame.filter(pl.col("arm") == arm)
            if rows.is_empty():
                msg = f"{path} holds no primary-setting rows for arm {arm!r}"
                raise ValueError(msg)
            if rows.select(KEY_COLUMNS).is_duplicated().any():
                msg = f"{path}: arm {arm!r} holds a (site, time, seed) key more than once"
                raise ValueError(msg)
            check_anonymised(frame=rows, domain=domain)
            output[(source, arm)] = rows
    return output


def contrast_rows(
    *, arms: Mapping[tuple[SourceType, str], pl.DataFrame], plan: Sequence[Comparison]
) -> pl.DataFrame:
    """Bootstrap each comparison's difference from its reference, in percentage points.

    Args:
        arms: `load_arms`'s result, holding every arm the plan names.
        plan: The comparisons.

    Returns:
        One row per comparison: its fields, `value`, `lower`, `upper`, `seed_spread` (points),
        `n_rows` (rows per seed), `n_months`, and `has_interval`, whether the row rests on at
        least `MIN_MONTHS_FOR_INTERVAL` months.
    """
    records = []
    for comparison in plan:
        treatment = arms[(comparison.treatment_source, comparison.treatment)]
        reference = arms[(comparison.reference_source, comparison.reference)]
        result = bootstrap_difference(
            losses=pl.concat([treatment, reference]),
            treatment=comparison.treatment,
            reference=comparison.reference,
            metric=METRIC,
        )
        records.append(
            {
                **comparison._asdict(),
                "value": result["difference"] * PERCENTAGE_POINTS,
                "lower": result["lower_95"] * PERCENTAGE_POINTS,
                "upper": result["upper_95"] * PERCENTAGE_POINTS,
                "seed_spread": result["seed_spread"] * PERCENTAGE_POINTS,
                "n_rows": result["n_rows"],
                "n_months": result["n_months"],
                "has_interval": result["n_months"] >= MIN_MONTHS_FOR_INTERVAL,
            }
        )
    return pl.DataFrame(records)


def compute(*, data_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Read the saved losses and bootstrap every dot of one technology.

    Args:
        data_dir: The `data/studies` directory.
        domain: `solar` or `wind`.

    Returns:
        `contrast_rows`'s result.
    """
    plan = comparisons(domain=domain)
    wanted = {(c.treatment_source, c.treatment) for c in plan} | {
        (c.reference_source, c.reference) for c in plan
    }
    return contrast_rows(arms=load_arms(data_dir=data_dir, domain=domain, wanted=wanted), plan=plan)


# --- Chart --------------------------------------------------------------------------------------

FAMILY: Final[str] = "weather model"
"""Every product here is a weather model, so every mark takes that family's colour."""

INTERVAL_CONDITION: Final[str] = f"{MIN_MONTHS_FOR_INTERVAL} or more months: interval shown"
NO_INTERVAL_CONDITION: Final[str] = f"Fewer than {MIN_MONTHS_FOR_INTERVAL} months: no interval"

ROW_STEP_PX: Final[int] = 22
"""The height of one row: 9 panels of up to 17 rows need them closer than the default."""

X_TITLE: Final[str] = "Error minus the ENS mean's (points of capacity)"


def chart_rows(*, rows: pl.DataFrame, day: int, with_conditions: bool) -> pl.DataFrame:
    """Shape one lead day's rows for `interval_panel`, best product first.

    Args:
        rows: `contrast_rows`'s result for one technology.
        day: The lead day.
        with_conditions: Whether to add the `condition` column, which marks a row with too few
            months for an interval.

    Returns:
        Rows sorted by ascending difference, carrying `label`, `family`, `difference`, `lower_95`
        and `upper_95` (null where the row has no interval), and `condition` where asked for.
    """
    shaped = (
        rows.filter(pl.col("day") == day)
        .sort("value")
        .select(
            "label",
            family=pl.lit(FAMILY),
            difference=pl.col("value"),
            lower_95=pl.when(pl.col("has_interval")).then(pl.col("lower")),
            upper_95=pl.when(pl.col("has_interval")).then(pl.col("upper")),
            condition=pl.when(pl.col("has_interval"))
            .then(pl.lit(INTERVAL_CONDITION))
            .otherwise(pl.lit(NO_INTERVAL_CONDITION)),
        )
    )
    return shaped if with_conditions else shaped.drop("condition")


def finding_title(*, rows: pl.DataFrame, domain: DomainType) -> str:
    """State how many rows are better than, worse than, or indistinguishable from the ENS mean.

    Args:
        rows: `contrast_rows`'s result for one technology.
        domain: `solar` or `wind`.

    Returns:
        The figure's title, every number of which is counted from `rows`.
    """
    with_interval = rows.filter(pl.col("has_interval"))
    better = with_interval.filter(pl.col("upper") < 0).height
    worse = with_interval.filter(pl.col("lower") > 0).height
    undetected = with_interval.height - better - worse
    short = rows.height - with_interval.height
    text = (
        f"For {domain} power, {worse} of {rows.height} product-and-lead rows have a higher "
        f"error than the ENS mean, {better} have a lower error, and {undetected} cannot be "
        "told apart"
    )
    return text + (f"; {short} have too few months for an interval" if short else "")


def subtitle_lines(*, domain: DomainType, has_short_rows: bool) -> list[str]:
    """Write the figure's subtitle lines, which say what a dot, a line, and zero mean.

    Args:
        domain: `solar` or `wind`.
        has_short_rows: Whether any row is drawn as a hollow dot with no interval.

    Returns:
        The lines, before wrapping.
    """
    lines = [
        (
            "Each row is an XGBoost model given one product's weather for "
            f"{TECHNOLOGY_NAMES[domain]}. Dot: the product's error minus the ENS mean's at the "
            "same lead day, in percentage points of capacity; negative means the product "
            "forecasts better. Zero is the ENS mean."
        ),
        DOTS_NOTE + (" A hollow dot has too few months for an interval." if has_short_rows else ""),
        (
            "Each row is scored on the hours its product and its ENS mean share. "
            "Day 0 is not a lead a live service could use, and it favours the Previous Runs "
            "products: they read the freshest run, a lead of a few hours, while ENS reads the "
            "00 UTC run of the day, a lead of 0 to 23 hours."
        ),
        CAPACITY_NOTE,
    ]
    if domain == "wind":
        lines.append(
            "WeatherNext 3's reference is the ENS mean built from the mean-vector speed, as "
            "WeatherNext 3's own speed is."
        )
    return lines


def draw(*, rows: pl.DataFrame, domain: DomainType, number: int) -> alt.VConcatChart:
    """Draw one technology's figure: one panel per lead day.

    Args:
        rows: `contrast_rows`'s result for one technology.
        domain: `solar` or `wind`.
        number: The figure's number on the page.

    Returns:
        The figure.
    """
    has_short_rows = not rows["has_interval"].all()
    low = min(rows["value"].to_list() + rows["lower"].to_list())
    high = max(rows["value"].to_list() + rows["upper"].to_list())
    x_domain = padded_domain(low=low, high=high, include_zero=True)
    days = [day for day in DAYS if day in set(rows["day"].to_list())]
    panels = [
        interval_panel(
            rows=chart_rows(rows=rows, day=day, with_conditions=has_short_rows),
            x_domain=x_domain,
            x_title=X_TITLE if index == len(days) - 1 else "",
            zero_label="same as the ENS mean",
            better_label="better than the ENS mean",
            conditions=(INTERVAL_CONDITION, NO_INTERVAL_CONDITION) if has_short_rows else (),
            panel_title=f"Lead day {day}",
            reference_labels=index == 0,
            family_key=False,
            condition_key=index == 0,
            figure_planning="exploratory",
            colour_by_family=True,
            row_step_px=ROW_STEP_PX,
        )
        for index, day in enumerate(days)
    ]
    return figure(
        panels=panels,
        number=number,
        title=finding_title(rows=rows, domain=domain),
        subtitle=subtitle_lines(domain=domain, has_short_rows=has_short_rows),
        figure_planning="exploratory",
    )


# --- Report and README ---------------------------------------------------------------------------


def _cell(*, row: Mapping[str, object]) -> str:
    """Format one row's estimate, with its interval where it has one."""
    value = f"{row['value']:+.3f}"
    if not row["has_interval"]:
        return f"{value} (no interval)"
    return f"{value} [{row['lower']:+.3f}, {row['upper']:+.3f}]"


def report_text(*, rows: Mapping[DomainType, pl.DataFrame]) -> str:
    """Print every dot's estimate and interval, so the page and the charts quote the report.

    Args:
        rows: Each technology's `contrast_rows` result.

    Returns:
        The report, in Markdown.
    """
    lines = [
        "# Each weather product minus the ENS mean, by lead day",
        "",
        (
            "Product minus reference, in points of capacity at the primary XGBoost setting. "
            "Each interval is the 95% interval from resampling whole months and a fitting seed. "
            f"A row with fewer than {MIN_MONTHS_FOR_INTERVAL} months prints no interval. "
            "`Rows` is the rows per fitting seed that both arms score."
        ),
        "",
    ]
    for domain, frame in rows.items():
        lines += [
            f"## {domain}",
            "",
            "| Day | Product | Reference arm | Difference (points) | Months | Rows |",
            "|---|---|---|---|---|---|",
        ]
        ordered = frame.sort("day", "value")
        lines += [
            f"| {row['day']} | {row['label']} | {row['reference']} ({row['reference_source']}) | "
            f"{_cell(row=row)} | {row['n_months']} | {row['n_rows']} |"
            for row in ordered.iter_rows(named=True)
        ]
        lines.append("")
    return "\n".join(lines)


def readme_text(*, rows: Mapping[DomainType, pl.DataFrame]) -> str:
    """Describe the output folder's files and the reference arm each product used.

    Args:
        rows: Each technology's `contrast_rows` result.

    Returns:
        The README, in Markdown.
    """
    groups: dict[tuple[str, str, str, str], tuple[set[str], set[int]]] = {}
    for domain, frame in rows.items():
        for row in frame.iter_rows(named=True):
            key = (
                row["label"],
                row["treatment_source"],
                re.sub(_DAY_SUFFIX, "", row["reference"]),
                row["reference_source"],
            )
            domains, days = groups.setdefault(key, (set(), set()))
            domains.add(domain)
            days.add(row["day"])
    lines = [
        "# Each weather product minus the ECMWF ENS mean, dots and intervals",
        "",
        (
            "Written once by `studies/nwp_forecast_comparison/dot_interval_vs_ens.py` from the "
            "saved per-row losses of earlier fits. It fits nothing, and it is never overwritten."
        ),
        "",
        (
            "- `report.md` prints every dot: the product, its reference arm, the difference "
            "with its 95% interval, and the months and rows it rests on."
        ),
        (
            "- `intervals.parquet` holds one row per dot: `domain`, `day`, `label`, `treatment`, "
            "`reference`, the two sources, `value`, `lower`, and `upper` in points of capacity, "
            "`seed_spread`, `n_rows` (per fitting seed), `n_months`, and `has_interval`."
        ),
        "",
        (
            "Each row uses the arms the leaderboards draw: an arm held by both a published CPU "
            "fit and a later GPU refit is read from the GPU refit. The reference is the "
            "leaderboard's ENS-mean arm of the same lead day. The `leads_day10*` folders hold "
            "the same `(site, time, seed)` keys, so a product in one folder pairs with the ENS "
            "mean of another. AIFS Single, the AIFS ENS mean, and WeatherNext 3 use the ENS mean "
            "fitted on their own rows, in their own file. Wind WeatherNext 3 rows use the "
            "`ens_meanvec` arm. Each line below names a product's source folder and its "
            "reference's, for the days listed:"
        ),
        "",
        "| Product | Technologies | Days | Product source | Reference | Reference source |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {label} | {', '.join(sorted(domains))} | {', '.join(map(str, sorted(days)))} | "
        f"{source} | {prefix} | {reference_source} |"
        for (label, source, prefix, reference_source), (domains, days) in sorted(groups.items())
    ]
    lines += [
        "",
        (
            "Not included, because each was not fitted on the comparison's rows and folds, so "
            "its error is not comparable with these rows: the Open-Meteo ensemble means, "
            "UKV from the CEDA archive, and NORA3. The climatology and persistence baselines "
            "are not weather products, and the blends are not products, so neither has a row. "
            "The superseded `nwp_forecast_comparison_leads` folder is not read, because the "
            "leaderboards do not draw it."
        ),
    ]
    return "\n".join([*lines, ""])


# --- Output -------------------------------------------------------------------------------------


def write_once(*, path: Path, write: str | pl.DataFrame | alt.VConcatChart) -> None:
    """Write a file that must not already exist.

    Args:
        path: Where to write.
        write: Text, a frame (as parquet), or a chart (as SVG).

    Raises:
        FileExistsError: If `path` exists.
    """
    if path.exists():
        msg = f"{path} exists; this script writes each output once"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(write, str):
        path.write_text(write)
    elif isinstance(write, pl.DataFrame):
        write.write_parquet(path)
    else:
        write.save(path)


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


def repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    Duplicated from the other study scripts, because study scripts cannot import one another's
    private helpers.

    Returns:
        The directory holding `studies/`.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        git_dir = Path(marker.read_text().removeprefix("gitdir:").strip())
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def main() -> int:
    """Compute every dot, then write the report, the parquet, the README, and the SVGs.

    Returns:
        The exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    studies_dir = repo_data_dir() / "studies"
    parser.add_argument("--data-dir", type=Path, default=studies_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=studies_dir / "nwp_forecast_comparison_vs_ens_dots_all_days",
    )
    parser.add_argument(
        "--svg-dir", type=Path, default=PROJECT_ROOT / "docs" / "studies" / "assets"
    )
    parser.add_argument(
        "--first-figure-number",
        type=int,
        default=19,
        help="The solar figure's number on the page; the wind figure follows it.",
    )
    parser.add_argument("--no-svgo", action="store_true", help="Skip the svgo optimisation.")
    args = parser.parse_args()
    rows = {domain: compute(data_dir=args.data_dir, domain=domain) for domain in DOMAINS}
    files = ("report.md", "intervals.parquet", "README.md")
    svgs = {
        domain: args.svg_dir / f"nwp_forecast_{domain}_dots_vs_ens_all_days.svg"
        for domain in DOMAINS
    }
    taken = [args.output_dir, *(args.output_dir / name for name in files), *svgs.values()]
    existing = [path for path in taken if path.exists()]
    if existing:
        msg = f"{existing} exist; this script writes each output once"
        raise FileExistsError(msg)
    charts = {
        domain: draw(rows=rows[domain], domain=domain, number=args.first_figure_number + index)
        for index, domain in enumerate(DOMAINS)
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_once(path=args.output_dir / "report.md", write=report_text(rows=rows))
    write_once(path=args.output_dir / "intervals.parquet", write=pl.concat(rows.values()))
    write_once(path=args.output_dir / "README.md", write=readme_text(rows=rows))
    for domain, path in svgs.items():
        write_once(path=path, write=charts[domain])
        if not args.no_svgo:
            optimise(path=path)
        _LOG.info("wrote %s", path)
        sys.stdout.write(f"{domain} caption: {finding_title(rows=rows[domain], domain=domain)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
