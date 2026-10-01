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

- The products in `_leads_day10`, `_leads_day10b`, `_leads_day10c`, `_leads_day10d`, and
  `_day4_shared` are subtracted from the ENS mean of `LEADERBOARD_ENS_SOURCES`: `_leads_day10b` at
  days 2 and 7, `_day4_shared` at day 4, and `_leads_day10` at every other day. The folders hold
  the same `(site, time, seed)` keys, so the pair joins across folders. Two arms are scored without
  the target days where their own weather is missing, and their paired difference drops the ENS
  mean's rows on those days (`GAPPED_ARMS`): IFS HRES 9 km at every day (1,197 to 1,536 rows fewer
  at each day, in 2025-08 and 2026-06, where its archive has no run, and about 1.4% and 1.2% at day
  4), and ICON global at day 4 for solar (288 rows fewer). `contrast_rows` raises for any other arm
  whose keys differ from its reference's.
- AIFS Single, the AIFS ENS mean, and WeatherNext 3 (WN3) each sit in a folder with an ENS mean
  fitted on the same rows, which is the reference (at day 5, in the `_day5_aifs_wn3` folder). For
  wind, WN3's reference is `ens_meanvec`, the
  ENS mean built from the mean-vector speed, which matches how WN3's speed is built.

**AIFS Single, the AIFS ENS mean, and WN3 have two marks per row.** Each is fitted on fewer months
than the leaderboards' ENS mean covers (16, 11, and 7 against 21), so its own reference is an ENS
mean refitted on those months, which scores worse than the 21-month ENS mean on the same keys. The
filled dot is against that refitted reference. The hollow diamond is against the leaderboard's
ENS-mean arm of the same day (`LEADERBOARD_ENS_SOURCES`) on the `(site, time, seed)` keys the two
share, each mark with its own interval. An interval from fewer than 12 months is dashed.

WN3 appears only as the pooled row, which covers 7 months (February to April and June to September
2026). The script writes `report.md`, `intervals.parquet`, and `README.md` to a new output folder,
and one SVG per technology. It refuses to overwrite any of them, except that `--replace-svgs`
redraws only the SVGs, with a light grey band behind every second product row, and leaves the
output folder untouched. Each replacement is drawn and optimised beside the old SVG, which is
swapped out only once the new one is ready.

**`--blends` draws ENS plus one product instead.** Each row is an XGBoost model given the ENS
mean's columns plus one product's columns (AIFS Single, ICON-EU at the optimistic and at the
conservative lead, UKV, and WeatherNext 3), and its dot is the blend's error minus the ENS mean's
alone on the same rows, at days 1, 2, 7, and 14 where the blend exists. The blends are read from
`nwp_forecast_comparison_aifs_blends` and `nwp_forecast_comparison_product_blends`. The report adds
the comparison C5, the AIFS Single blend minus each ICON-EU blend, which reuses the keys check. The
dots are at the primary setting only; `fit_product_blends.py`'s `report.md` holds both settings.

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
from studies.charts import figure, interval_panel, planning

_LOG: Final[logging.Logger] = logging.getLogger("dot_interval_vs_ens")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

FIRST_FIGURE_NUMBER: Final[int] = 1
"""The solar figure's number on the page; the wind figure follows it."""

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
    "day4_shared",
    "aifs_single_day5",
    "aifs_ens_day5",
    "wn3_day5",
    "product_blends_single",
    "product_blends_wn3",
]
"""The folders a row's losses come from."""


class Source(NamedTuple):
    """Where one fit batch saved its per-row losses."""

    folder: str
    pattern: str


_AIFS_BLENDS: Final[str] = "nwp_forecast_comparison_aifs_blends"
_AIFS_EXTRA: Final[str] = "nwp_forecast_comparison_aifs_extra_days"
_DAY5: Final[str] = "nwp_forecast_comparison_day5_aifs_wn3"
_PRODUCT_BLENDS: Final[str] = "nwp_forecast_comparison_product_blends"
_PER_DAY: Final[str] = "{{domain}}_{name}_day{{day}}_losses.parquet"

SOURCES: Final[dict[SourceType, Source]] = {
    "leads_day10": Source(
        folder="nwp_forecast_comparison_leads_day10", pattern="{domain}_losses.parquet"
    ),
    "leads_day10b": Source(
        folder="nwp_forecast_comparison_leads_day10b", pattern="{domain}_losses.parquet"
    ),
    "leads_day10c": Source(
        folder="nwp_forecast_comparison_leads_day10c", pattern="{domain}_losses.parquet"
    ),
    "leads_day10d": Source(
        folder="nwp_forecast_comparison_leads_day10d", pattern="{domain}_losses.parquet"
    ),
    "aifs_single_blends": Source(folder=_AIFS_BLENDS, pattern=_PER_DAY.format(name="single")),
    "aifs_ens_blends": Source(folder=_AIFS_BLENDS, pattern=_PER_DAY.format(name="ens")),
    "aifs_single_extra": Source(folder=_AIFS_EXTRA, pattern=_PER_DAY.format(name="single")),
    "aifs_ens_extra": Source(folder=_AIFS_EXTRA, pattern=_PER_DAY.format(name="ens")),
    "wn3_blends": Source(folder="nwp_forecast_comparison_wn3", pattern=_PER_DAY.format(name="wn3")),
    "wn3_extra": Source(
        folder="nwp_forecast_comparison_wn3_extra_days", pattern=_PER_DAY.format(name="wn3")
    ),
    "day4_shared": Source(
        folder="nwp_forecast_comparison_day4_shared", pattern="{domain}_losses.parquet"
    ),
    "aifs_single_day5": Source(folder=_DAY5, pattern=_PER_DAY.format(name="single")),
    "aifs_ens_day5": Source(folder=_DAY5, pattern=_PER_DAY.format(name="ens")),
    "wn3_day5": Source(folder=_DAY5, pattern=_PER_DAY.format(name="wn3")),
    "product_blends_single": Source(folder=_PRODUCT_BLENDS, pattern=_PER_DAY.format(name="single")),
    "product_blends_wn3": Source(folder=_PRODUCT_BLENDS, pattern=_PER_DAY.format(name="wn3")),
}
"""Each source's folder under `data/studies/`, and its losses file's name. A `{day}` in the name
means one file per lead day. The `_blends` folders hold days 1, 2, 7, and 14 and the `_extra`
folders days 0, 3, 4, and 10. The `_product_blends` folder holds the ICON-EU and UKV blends on the
`single` rows, and the WeatherNext 3 blend on the `wn3` rows. The superseded
`nwp_forecast_comparison_leads` folder is not a source, because the leaderboards do not draw it."""

REFERENCE_NAMES: Final[dict[str, str]] = {
    "ens_mean": "ENS mean",
    "ens_meanvec": "ENS mean (mean-vector speed)",
}
"""Each reference arm prefix's name in a report."""

WN3_LABEL: Final[str] = "WeatherNext 3 mean (7 months)"
"""WN3's row label; its rows cover February to April and June to September 2026 only."""


def _sources(
    *, days: tuple[int, ...], default: SourceType, overrides: Mapping[int, SourceType] | None = None
) -> dict[int, SourceType]:
    """Map each lead day to its source: `default`, except for the days in `overrides`."""
    return {day: (overrides or {}).get(day, default) for day in days}


class ProductRow(NamedTuple):
    """One product's arm prefix, the source of each lead day it is fitted at, and its reference.

    `on_fewer_months` marks a product fitted on fewer months than the 21 the leaderboards' ENS
    mean covers. Its ENS-mean reference is refitted on its own rows, and the row also gets a
    second mark against the leaderboard's ENS mean on the same `(site, time, seed)` keys.
    """

    prefix: str
    sources: Mapping[int, SourceType]
    reference_sources: Mapping[int, SourceType] | None = None
    domains: tuple[DomainType, ...] = DOMAINS
    wind_reference: str = "ens_mean"
    label: str | None = None
    on_fewer_months: bool = False


LEADERBOARD_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 7, 10, 14)
"""The lead days of the ENS control member, GEFS, and native GFS."""

AIFS_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 7, 10, 14)
"""The lead days of AIFS Single, the AIFS ENS mean, and WN3: every panel's day."""

LEADERBOARD_ENS_SOURCES: Final[dict[int, SourceType]] = _sources(
    days=LEADERBOARD_DAYS,
    default="leads_day10",
    overrides={2: "leads_day10b", 4: "day4_shared", 7: "leads_day10b"},
)
"""The folder of the ENS-mean arm the leaderboards draw at each lead day (day 4: the day-4 shared
rows' folder). The published folder
holds ENS-mean arms at days 0 to 3 too, but the leaderboards read the GPU refit that the
`leads_day10` and `leads_day10b` folders hold, because `nwp_forecast_charts.load` prefers an arm
from an extra-lead folder to the published folder's copy."""

EXTRA_DAYS: Final[tuple[int, ...]] = (0, 3, 4, 10)
"""The AIFS and WN3 lead days that sit in the `_extra` folders."""

_D10B: Final[SourceType] = "leads_day10b"
_D4: Final[SourceType] = "day4_shared"
_LEADS_TO_3: Final[tuple[int, ...]] = (0, 1, 2, 3)
_LEADS_TO_4: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
_LEADS_TO_5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
_LEADS_TO_7: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 7)


def _leaderboard_row(
    *,
    prefix: str,
    sources: Mapping[int, SourceType],
    domains: tuple[DomainType, ...] = DOMAINS,
) -> ProductRow:
    """Return a product whose reference is the leaderboard's ENS mean at each of its days."""
    return ProductRow(
        prefix=prefix,
        sources=sources,
        reference_sources=LEADERBOARD_ENS_SOURCES,
        domains=domains,
    )


PRODUCTS: Final[tuple[ProductRow, ...]] = (
    _leaderboard_row(prefix="ukv", sources=_sources(days=(0, 1), default="leads_day10")),
    _leaderboard_row(prefix="icon_d2", sources=_sources(days=(0, 1), default="leads_day10")),
    _leaderboard_row(
        prefix="icon_eu",
        sources=_sources(
            days=_LEADS_TO_4, default="leads_day10", overrides={2: _D10B, 3: _D10B, 4: _D4}
        ),
    ),
    _leaderboard_row(
        prefix="icon_global",
        sources=_sources(days=_LEADS_TO_5, default="leads_day10", overrides={2: _D10B, 4: _D4}),
    ),
    _leaderboard_row(
        prefix="gfs",
        sources=_sources(days=_LEADS_TO_7, default="leads_day10", overrides={2: _D10B, 4: _D4}),
    ),
    _leaderboard_row(
        prefix="gfs_native",
        sources=_sources(days=LEADERBOARD_DAYS, default="leads_day10c", overrides={4: _D4}),
    ),
    _leaderboard_row(
        prefix="ifs025",
        sources=_sources(days=_LEADS_TO_7, default="leads_day10", overrides={2: _D10B, 4: _D4}),
    ),
    _leaderboard_row(
        prefix="ifs_single",
        sources=_sources(days=_LEADS_TO_7, default="leads_day10d", overrides={4: _D4}),
    ),
    _leaderboard_row(
        prefix="arpege",
        sources=_sources(days=_LEADS_TO_3, default="leads_day10", overrides={2: _D10B, 3: _D10B}),
        domains=("solar",),
    ),
    _leaderboard_row(
        prefix="arome", sources=_sources(days=(0, 1), default="leads_day10"), domains=("solar",)
    ),
    _leaderboard_row(prefix="dmi_harmonie", sources=_sources(days=(0, 1), default="leads_day10")),
    _leaderboard_row(prefix="knmi_harmonie", sources=_sources(days=(0, 1), default="leads_day10")),
    _leaderboard_row(
        prefix="gefs_mean",
        sources=_sources(
            days=LEADERBOARD_DAYS,
            default="leads_day10",
            overrides={2: _D10B, 4: _D4, 7: _D10B},
        ),
    ),
    _leaderboard_row(
        prefix="ens_control",
        sources=_sources(
            days=LEADERBOARD_DAYS,
            default=_D10B,
            overrides={0: "leads_day10", 1: "leads_day10", 4: _D4},
        ),
    ),
    ProductRow(
        prefix="aifs_single",
        sources=_sources(
            days=AIFS_DAYS,
            default="aifs_single_blends",
            overrides={**dict.fromkeys(EXTRA_DAYS, "aifs_single_extra"), 5: "aifs_single_day5"},
        ),
        on_fewer_months=True,
    ),
    ProductRow(
        prefix="aifs_ens_mean",
        sources=_sources(
            days=AIFS_DAYS,
            default="aifs_ens_blends",
            overrides={**dict.fromkeys(EXTRA_DAYS, "aifs_ens_extra"), 5: "aifs_ens_day5"},
        ),
        on_fewer_months=True,
    ),
    ProductRow(
        prefix="wn3_mean",
        sources=_sources(
            days=AIFS_DAYS,
            default="wn3_blends",
            overrides={**dict.fromkeys(EXTRA_DAYS, "wn3_extra"), 5: "wn3_day5"},
        ),
        wind_reference="ens_meanvec",
        label=WN3_LABEL,
        on_fewer_months=True,
    ),
)
"""Every row the figures draw: each product the leaderboards carry, except the ENS mean, which is
the reference. A product at a lead day no row names is not in the archive or not fitted. Day 4
holds the products of the day-4 shared folder, AIFS, and WN3. Day 5 holds AIFS, WN3, and the ICON
global, Open-Meteo GFS, IFS 0.25°, IFS HRES 9 km, ENS control, GEFS, and native GFS rows of the
`leads_day10*` folders."""


BLEND_DAYS: Final[tuple[int, ...]] = (1, 2, 7, 14)
"""The lead days of the blend figures: each blend's panels."""

_AIFS_SINGLE_BLENDS: Final[SourceType] = "aifs_single_blends"
_PRODUCT_SINGLE: Final[SourceType] = "product_blends_single"
_SINGLE_ENS: Final[dict[int, SourceType]] = _sources(days=BLEND_DAYS, default=_AIFS_SINGLE_BLENDS)
_WN3_OWN: Final[dict[int, SourceType]] = _sources(days=BLEND_DAYS, default="product_blends_wn3")

AIFS_BLEND_LABEL: Final[str] = "ENS mean + AIFS Single"
ICON_EU_OPTIMISTIC_LABEL: Final[str] = "ENS mean + ICON-EU (optimistic lead)"
ICON_EU_CONSERVATIVE_LABEL: Final[str] = "ENS mean + ICON-EU (conservative lead)"
UKV_LABEL: Final[str] = "ENS mean + UKV (optimistic upper bound)"
WN3_BLEND_LABEL: Final[str] = "ENS mean + WeatherNext 3 (7 months)"

BLEND_PLANNED_DAYS: Final[dict[str, tuple[int, ...]]] = {
    AIFS_BLEND_LABEL: (1, 2, 7),
    ICON_EU_CONSERVATIVE_LABEL: (1, 2),
    UKV_LABEL: (1,),
}
"""The blend rows written into the study plan before any fit, by label, and the days at which each
is planned (contrasts C1 to C3). Every other row is exploratory: the optimistic ICON-EU blend, the
WeatherNext 3 blend, and AIFS Single at day 14."""

BLEND_AXIS_GROUPS: Final[tuple[tuple[int, ...], ...]] = ((1, 2), (7, 14))
"""The lead days that share an x axis in the blend figures. Days 7 and 14 hold wide intervals, so
they get an axis of their own and days 1 and 2 stay readable."""

BLEND_PRODUCTS: Final[tuple[ProductRow, ...]] = (
    ProductRow(
        prefix="blend_aifs_single",
        sources=_sources(days=BLEND_DAYS, default=_AIFS_SINGLE_BLENDS),
        reference_sources=_SINGLE_ENS,
        label=AIFS_BLEND_LABEL,
    ),
    ProductRow(
        prefix="blend_icon_eu",
        sources=_sources(days=(1, 2), default=_PRODUCT_SINGLE),
        reference_sources=_SINGLE_ENS,
        label=ICON_EU_OPTIMISTIC_LABEL,
    ),
    ProductRow(
        prefix="blend_icon_eu_conservative",
        sources=_sources(days=(1, 2), default=_PRODUCT_SINGLE),
        reference_sources=_SINGLE_ENS,
        label=ICON_EU_CONSERVATIVE_LABEL,
    ),
    ProductRow(
        prefix="blend_ukv",
        sources=_sources(days=(1,), default=_PRODUCT_SINGLE),
        reference_sources=_SINGLE_ENS,
        label=UKV_LABEL,
    ),
    ProductRow(
        prefix="blend_wn3",
        sources=_sources(days=BLEND_DAYS, default="product_blends_wn3"),
        reference_sources=_WN3_OWN,
        label=WN3_BLEND_LABEL,
        on_fewer_months=True,
    ),
)
"""The rows of the `--blends` figures. Each blend is read at the primary setting and subtracted
from the ENS mean that sits beside it in the file that holds its reference: AIFS Single's file
for the four blends on the `single` rows, and the WeatherNext 3 file for WeatherNext 3's blend,
whose 7 months also get a second mark against the leaderboard's 21-month ENS mean. ICON-EU reaches
days 1 and 2 in the published inputs, and UKV live day 1."""


class Comparison(NamedTuple):
    """One dot: a product's arm at a lead day, and the ENS-mean arm it is subtracted from.

    A product fitted on fewer months also names the leaderboard's 21-month ENS-mean arm
    (`leaderboard_reference`), for its second mark.
    """

    domain: DomainType
    day: int
    label: str
    treatment: str
    treatment_source: SourceType
    reference: str
    reference_source: SourceType
    reference_label: str
    leaderboard_reference: str | None = None
    leaderboard_reference_source: SourceType | None = None


def comparisons(
    *, domain: DomainType, products: Sequence[ProductRow] = PRODUCTS
) -> list[Comparison]:
    """List every dot of one technology's figure.

    Args:
        domain: `solar` or `wind`.
        products: The rows to draw: `PRODUCTS`, or `BLEND_PRODUCTS` for the blend figures.

    Returns:
        One comparison per product and lead day, in the order of `products`.
    """
    output = []
    for product in products:
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
                leaderboard_reference=f"ens_mean_day{day}" if product.on_fewer_months else None,
                leaderboard_reference_source=(
                    LEADERBOARD_ENS_SOURCES[day] if product.on_fewer_months else None
                ),
            )
            for day, source in product.sources.items()
        )
    return output


C5_DAYS: Final[tuple[int, ...]] = (1, 2)
"""The days at which ICON-EU has a blend, so the AIFS Single blend has one to be compared with."""


def blend_rankings(*, domain: DomainType) -> list[Comparison]:
    """List comparison C5: the AIFS Single blend minus each ICON-EU blend, at each day both exist.

    The conservative ICON-EU blend reads the freshest run at least 48 hours before the valid
    hour, which is older than the AIFS Single run, so the comparison is read against the
    optimistic ICON-EU blend too. Both blends sit on the `single` rows, so `contrast_rows` raises
    if their keys differ.

    Args:
        domain: `solar` or `wind`.

    Returns:
        One comparison per (day, ICON-EU lead), with the AIFS Single blend as the treatment.
    """
    return [
        Comparison(
            domain=domain,
            day=day,
            label=f"{AIFS_BLEND_LABEL} minus {reference_label}",
            treatment=f"blend_aifs_single_day{day}",
            treatment_source=_AIFS_SINGLE_BLENDS,
            reference=f"{prefix}_day{day}",
            reference_source=_PRODUCT_SINGLE,
            reference_label=reference_label,
        )
        for day in C5_DAYS
        for prefix, reference_label in (
            ("blend_icon_eu_conservative", ICON_EU_CONSERVATIVE_LABEL),
            ("blend_icon_eu", ICON_EU_OPTIMISTIC_LABEL),
        )
    ]


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


MIN_MONTHS_FOR_SOLID_INTERVAL: Final[int] = 12
"""Fewer months than this draws an interval dashed: a month-block bootstrap over fewer than a year
of months tends to give intervals that are too narrow."""

SECOND_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"second_{name}"
    for name in (
        "value",
        "lower",
        "upper",
        "seed_spread",
        "n_rows",
        "treatment_rows",
        "reference_rows",
        "n_months",
        "has_interval",
        "dashed",
    )
)
"""The columns of a row's second mark, against the leaderboard's ENS mean; null for a product
that has none."""

GAP_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"reference_gap_{name}"
    for name in (
        "value",
        "lower",
        "upper",
        "seed_spread",
        "n_rows",
        "treatment_rows",
        "reference_rows",
        "n_months",
        "has_interval",
        "dashed",
    )
)
"""The columns of the gap between a short-history product's two references: its own-file ENS mean
minus the leaderboard's ENS mean, on the `(site, time, seed)` keys the two share, in points; null
for a product with one reference."""

GAPPED_ARMS: Final[dict[DomainType, re.Pattern[str]]] = {
    "solar": re.compile(r"ifs_single_day\d+|icon_global_day4"),
    "wind": re.compile(r"ifs_single_day\d+"),
}
"""The arms scored without the target days where their own weather is missing, so the ENS mean holds
rows they do not: IFS HRES 9 km at every day, and ICON global at day 4."""


def _check_same_keys(
    *,
    comparison: Comparison,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    reference_may_hold_more: bool = False,
) -> None:
    """Raise unless a product and its reference score the same `(site, time, seed)` keys.

    A paired difference inner-joins the two arms, so keys in only one of them drop out silently.
    The arms in `GAPPED_ARMS` may lack keys the reference holds, and nothing else may. A product
    fitted on fewer months than the leaderboard's ENS mean may lack keys that ENS mean holds, and
    `reference_may_hold_more` allows that for the second mark.

    Args:
        comparison: The dot being computed.
        treatment: The product's rows.
        reference: The reference's rows.
        reference_may_hold_more: Whether the reference may hold keys the product lacks.

    Raises:
        ValueError: If the product holds a key the reference lacks, or the reference holds keys
            the product lacks beyond what `GAPPED_ARMS` and `reference_may_hold_more` allow.
    """
    only_treatment = treatment.select(KEY_COLUMNS).join(
        reference.select(KEY_COLUMNS), on=list(KEY_COLUMNS), how="anti"
    )
    only_reference = reference.select(KEY_COLUMNS).join(
        treatment.select(KEY_COLUMNS), on=list(KEY_COLUMNS), how="anti"
    )
    gapped = reference_may_hold_more or (
        GAPPED_ARMS[comparison.domain].fullmatch(comparison.treatment) is not None
    )
    if only_treatment.height or (only_reference.height and not gapped):
        msg = (
            f"{comparison.domain} day {comparison.day} {comparison.label}: "
            f"{only_treatment.height} keys only in {comparison.treatment} and "
            f"{only_reference.height} only in {comparison.reference}"
        )
        raise ValueError(msg)


def _bootstrap_record(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, treatment_arm: str, reference_arm: str
) -> dict[str, float | int | bool]:
    """Bootstrap one product-and-reference pair, in percentage points.

    Args:
        treatment: The product's rows.
        reference: The reference's rows.
        treatment_arm: The product's arm name.
        reference_arm: The reference's arm name.

    Returns:
        `value`, `lower`, `upper`, `seed_spread`, `n_rows` (per seed), `treatment_rows`,
        `reference_rows`, `n_months`, `has_interval`, and `dashed`, whether the interval rests on
        fewer than `MIN_MONTHS_FOR_SOLID_INTERVAL` months.
    """
    result = bootstrap_difference(
        losses=pl.concat([treatment, reference]),
        treatment=treatment_arm,
        reference=reference_arm,
        metric=METRIC,
    )
    return {
        "value": result["difference"] * PERCENTAGE_POINTS,
        "lower": result["lower_95"] * PERCENTAGE_POINTS,
        "upper": result["upper_95"] * PERCENTAGE_POINTS,
        "seed_spread": result["seed_spread"] * PERCENTAGE_POINTS,
        "n_rows": result["n_rows"],
        "treatment_rows": treatment.height,
        "reference_rows": reference.height,
        "n_months": result["n_months"],
        "has_interval": result["n_months"] >= MIN_MONTHS_FOR_INTERVAL,
        "dashed": result["n_months"] < MIN_MONTHS_FOR_SOLID_INTERVAL,
    }


def contrast_rows(
    *, arms: Mapping[tuple[SourceType, str], pl.DataFrame], plan: Sequence[Comparison]
) -> pl.DataFrame:
    """Bootstrap each comparison's difference from its reference, in percentage points.

    Args:
        arms: `load_arms`'s result, holding every arm the plan names.
        plan: The comparisons.

    Returns:
        One row per comparison: its fields, `value`, `lower`, `upper`, `seed_spread` (points),
        `n_rows` (rows per seed), `treatment_rows` and `reference_rows` (every row of each arm),
        `n_months`, `has_interval` (at least `MIN_MONTHS_FOR_INTERVAL` months), and `dashed` (fewer
        than `MIN_MONTHS_FOR_SOLID_INTERVAL`). A comparison with a `leaderboard_reference` also
        carries the same columns prefixed `second_`, for the product against that ENS mean on the
        `(site, time, seed)` keys the two share, and the `reference_gap_` columns (`GAP_COLUMNS`),
        and null otherwise.
    """
    records = []
    for comparison in plan:
        treatment = arms[(comparison.treatment_source, comparison.treatment)]
        reference = arms[(comparison.reference_source, comparison.reference)]
        _check_same_keys(comparison=comparison, treatment=treatment, reference=reference)
        record = {
            **comparison._asdict(),
            **_bootstrap_record(
                treatment=treatment,
                reference=reference,
                treatment_arm=comparison.treatment,
                reference_arm=comparison.reference,
            ),
        }
        second: dict[str, float | int | bool | None] = dict.fromkeys(SECOND_COLUMNS + GAP_COLUMNS)
        if (
            comparison.leaderboard_reference is not None
            and comparison.leaderboard_reference_source is not None
        ):
            leaderboard = arms[
                (comparison.leaderboard_reference_source, comparison.leaderboard_reference)
            ]
            for product in (treatment, reference):
                _check_same_keys(
                    comparison=comparison,
                    treatment=product,
                    reference=leaderboard,
                    reference_may_hold_more=True,
                )
            second |= {
                f"reference_gap_{name}": value
                for name, value in _bootstrap_record(
                    treatment=reference.with_columns(arm=pl.lit("own_file_ens_mean")),
                    reference=leaderboard.with_columns(arm=pl.lit("leaderboard_ens_mean")),
                    treatment_arm="own_file_ens_mean",
                    reference_arm="leaderboard_ens_mean",
                ).items()
            }
            second |= {
                f"second_{name}": value
                for name, value in _bootstrap_record(
                    treatment=treatment,
                    reference=leaderboard,
                    treatment_arm=comparison.treatment,
                    reference_arm=comparison.leaderboard_reference,
                ).items()
            }
        records.append({**record, **second})
    return pl.DataFrame(records, infer_schema_length=None)


def compute_plan(*, data_dir: Path, domain: DomainType, plan: Sequence[Comparison]) -> pl.DataFrame:
    """Read the saved losses the plan names and bootstrap each comparison.

    Args:
        data_dir: The `data/studies` directory.
        domain: `solar` or `wind`.
        plan: The comparisons.

    Returns:
        `contrast_rows`'s result.
    """
    wanted = (
        {(c.treatment_source, c.treatment) for c in plan}
        | {(c.reference_source, c.reference) for c in plan}
        | {
            (c.leaderboard_reference_source, c.leaderboard_reference)
            for c in plan
            if c.leaderboard_reference is not None and c.leaderboard_reference_source is not None
        }
    )
    return contrast_rows(arms=load_arms(data_dir=data_dir, domain=domain, wanted=wanted), plan=plan)


def compute(*, data_dir: Path, domain: DomainType, blends: bool = False) -> pl.DataFrame:
    """Read the saved losses and bootstrap every dot of one technology.

    Args:
        data_dir: The `data/studies` directory.
        domain: `solar` or `wind`.
        blends: Whether to draw the ENS-plus-one-product blends instead of the single products.

    Returns:
        `contrast_rows`'s result.
    """
    products = BLEND_PRODUCTS if blends else PRODUCTS
    plan = comparisons(domain=domain, products=products)
    return compute_plan(data_dir=data_dir, domain=domain, plan=plan)


# --- Chart --------------------------------------------------------------------------------------

FAMILY: Final[str] = "weather model"
"""Every product here is a weather model, so every mark takes that family's colour."""

ROW_STEP_PX: Final[int] = 30
"""The height of one row: 9 panels of up to 17 rows, each row holding up to two marks."""

X_TITLE: Final[str] = "Error minus the ENS mean's (points of capacity)"


class Footnote(NamedTuple):
    """A caveat on one row: the row's dot is drawn hollow, and the text joins the subtitle."""

    domain: DomainType
    label: str
    day: int
    text: str


FOOTNOTES: Final[tuple[Footnote, ...]] = (
    Footnote(
        domain="wind",
        label=WN3_LABEL,
        day=10,
        text=(
            "Caveat (hollow circle in place of the filled dot): WeatherNext 3's day-10 weather "
            "scored no better than shuffled weather in this fit (19.00% against 18.80% and "
            "18.94%); the cause is not established and this is not evidence that WeatherNext 3 "
            "loses skill."
        ),
    ),
)
"""The caveats the figures carry, each from the WN3 fit's own report (`report.md` of
`nwp_forecast_comparison_wn3_extra_days`, wind day 10, pooled months)."""


def footnotes_for(*, domain: DomainType, rows: pl.DataFrame) -> list[Footnote]:
    """List the footnotes whose row is in a technology's rows.

    Args:
        domain: `solar` or `wind`.
        rows: `contrast_rows`'s result for the technology.

    Returns:
        The matching footnotes, in `FOOTNOTES` order.
    """
    present = {(row["label"], row["day"]) for row in rows.iter_rows(named=True)}
    return [
        note for note in FOOTNOTES if note.domain == domain and (note.label, note.day) in present
    ]


def chart_rows(
    *, rows: pl.DataFrame, day: int, domain: DomainType, blends: bool = False
) -> pl.DataFrame:
    """Shape one lead day's rows for `interval_panel`, best product first.

    Args:
        rows: `contrast_rows`'s result for one technology.
        day: The lead day.
        domain: `solar` or `wind`, which selects the footnoted rows.
        blends: Whether the rows are blends, which adds the Boolean `planned` column
            (`BLEND_PLANNED_DAYS`).

    Returns:
        Rows sorted by ascending difference, carrying `label`, `family`, `difference`, `lower_95`
        and `upper_95` (null where the row has no interval), `hollow` (no interval, or a
        footnoted row), `dashed`, and `other_difference`, `other_lower_95`, `other_upper_95`, and
        `other_dashed` for the second mark, null for a row without one.
    """
    noted = [(note.label, note.day) for note in FOOTNOTES if note.domain == domain]
    is_noted = pl.struct("label", "day").is_in(
        [{"label": label, "day": note_day} for label, note_day in noted]
    )
    planned = (
        [pl.col("label").is_in([k for k, days in BLEND_PLANNED_DAYS.items() if day in days])]
        if blends
        else []
    )
    return (
        rows.filter(pl.col("day") == day)
        .sort("value")
        .select(
            "label",
            *[expr.alias("planned") for expr in planned],
            family=pl.lit(FAMILY),
            difference=pl.col("value"),
            lower_95=pl.when(pl.col("has_interval")).then(pl.col("lower")),
            upper_95=pl.when(pl.col("has_interval")).then(pl.col("upper")),
            hollow=~pl.col("has_interval") | is_noted,
            dashed=pl.col("dashed"),
            other_difference=pl.col("second_value"),
            other_lower_95=pl.when(pl.col("second_has_interval")).then(pl.col("second_lower")),
            other_upper_95=pl.when(pl.col("second_has_interval")).then(pl.col("second_upper")),
            other_dashed=pl.col("second_dashed"),
        )
    )


def figure_title(*, domain: DomainType, blends: bool = False) -> str:
    """Name what a figure shows, without counting rows.

    Rows are not independent tests, and the ENS control member is not a competing product, so a
    count of rows with an interval above or below zero would mislead.

    Args:
        domain: `solar` or `wind`.
        blends: Whether the figure draws ENS plus one product instead of single products.

    Returns:
        The figure's title.
    """
    if blends:
        return (
            f"For {domain} power, the error of the ENS mean plus one weather product, minus the "
            "ENS mean alone's error, by lead day"
        )
    return (
        f"For {domain} power, each weather product's error minus the ENS mean's error, by lead day"
    )


def blend_subtitle_lines(*, domain: DomainType) -> list[str]:
    """Write the blend figure's subtitle lines, which say what each row, mark, and zero mean."""
    return [
        (
            "Each row is an XGBoost model given the ENS mean's weather plus one product's weather "
            f"for {TECHNOLOGY_NAMES[domain]}. Dot: its error minus that of an XGBoost model given "
            "the ENS mean's weather alone, on the same hours, in percentage points of capacity; "
            "negative means adding the product helps. Zero is the ENS mean alone. Where a 95% "
            "interval includes zero, adding the product is not distinguished from not adding it."
        ),
        DOTS_NOTE,
        (
            "Dashed line: an interval from fewer than 12 months, which a month-block bootstrap "
            "tends to draw too narrow."
        ),
        (
            "Lead: ENS mean plus AIFS Single and ENS mean plus WeatherNext 3 read the 00 UTC "
            "run N days before the valid day. ENS mean plus ICON-EU reads the freshest ICON-EU "
            "run at least N days before the valid hour at the optimistic lead, and at least N+1 "
            "days before at the conservative lead. ENS mean plus UKV reads UKV's day-1 value, so "
            "it is an optimistic upper bound; no blend with UKV is ranked."
        ),
        (
            "Rows: ENS mean plus AIFS Single, ICON-EU, or UKV shares 16 months. ENS mean plus "
            "WeatherNext 3 rests on 7 months, against an ENS mean refitted on those months "
            "(filled dot) and against the 21-month ENS mean (hollow diamond)."
        ),
        CAPACITY_NOTE,
    ]


def subtitle_lines(*, domain: DomainType, rows: pl.DataFrame, blends: bool = False) -> list[str]:
    """Write the figure's subtitle lines, which say what each mark, line, and zero mean.

    Args:
        domain: `solar` or `wind`.
        rows: `contrast_rows`'s result for the technology.
        blends: Whether the figure draws ENS plus one product instead of single products.

    Returns:
        The lines, before wrapping.
    """
    if blends:
        lines = blend_subtitle_lines(domain=domain)
        if not rows["has_interval"].all():
            lines.append(
                f"Hollow circle: fewer than {MIN_MONTHS_FOR_INTERVAL} months, so no interval."
            )
        return lines
    lines = [
        (
            "Each row is an XGBoost model given one product's weather for "
            f"{TECHNOLOGY_NAMES[domain]}. Dot: the product's error minus the ENS mean's at the "
            "same lead day, in percentage points of capacity; negative means the product "
            "forecasts better. Zero is the ENS mean. Where a 95% interval includes zero, the "
            "product and the ENS mean are not distinguished."
        ),
        DOTS_NOTE,
        (
            "Dashed line: an interval from fewer than 12 months, which a month-block bootstrap "
            "tends to draw too narrow."
        ),
        (
            "AIFS Single, AIFS ENS mean, and WeatherNext 3 have two marks. Filled dot: against "
            "an ENS mean refitted on the same 16, 11, or 7 months. Hollow diamond: against the "
            "leaderboard's 21-month ENS mean, on the same hours. For wind, WeatherNext 3's "
            "filled dot is against the ENS mean-vector reference."
        ),
        (
            "Each row is scored on the hours its product and its reference share. Day 0 is not "
            "a lead a live service could use."
        ),
        (
            "Lead mismatch: at days 0 to 7, UKV, ICON-D2, ICON-EU, ICON global, GFS "
            "(Open-Meteo), IFS 0.25°, ARPEGE, AROME, and both HARMONIE-AROME products read the "
            "freshest run at least N days old, a shorter lead than ENS's on most hours, and "
            "native GFS does at day 0. That favours those products, so a higher error than ENS "
            "is conservative for them, and a lower error is not evidence of skill at equal lead."
        ),
        CAPACITY_NOTE,
    ]
    if not rows["has_interval"].all():
        lines.append(f"Hollow circle: fewer than {MIN_MONTHS_FOR_INTERVAL} months, so no interval.")
    lines.extend(note.text for note in footnotes_for(domain=domain, rows=rows))
    return lines


def draw(
    *,
    rows: pl.DataFrame,
    domain: DomainType,
    number: int,
    blends: bool = False,
    row_bands: bool = False,
) -> alt.VConcatChart:
    """Draw one technology's figure: one panel per lead day.

    Args:
        rows: `contrast_rows`'s result for one technology.
        domain: `solar` or `wind`.
        number: The figure's number on the page.
        blends: Whether the figure draws ENS plus one product instead of single products.
        row_bands: Whether every second product row gets a light grey band behind it.

    Returns:
        The figure.
    """
    present = set(rows["day"].to_list())
    days = [day for day in (BLEND_DAYS if blends else DAYS) if day in present]
    groups = (
        [tuple(d for d in group if d in days) for group in BLEND_AXIS_GROUPS]
        if blends
        else [tuple(days)]
    )
    domains = {
        day: _x_domain(rows=rows.filter(pl.col("day").is_in(group)))
        for group in groups
        if group
        for day in group
    }
    panel_rows = {day: chart_rows(rows=rows, day=day, domain=domain, blends=blends) for day in days}
    figure_planning = planning(rows=list(panel_rows.values()))
    last_of_group = {group[-1] for group in groups if group}
    panels = [
        interval_panel(
            rows=panel_rows[day],
            x_domain=domains[day],
            x_title=X_TITLE if day in last_of_group else "",
            zero_label="same as the ENS mean",
            better_label="better than the ENS mean",
            panel_title=_panel_title(day=day, blends=blends, domains=domains),
            reference_labels=index == 0,
            family_key=False,
            figure_planning=figure_planning,
            colour_by_family=True,
            row_step_px=ROW_STEP_PX,
            row_bands=row_bands,
        )
        for index, day in enumerate(days)
    ]
    return figure(
        panels=panels,
        number=number,
        title=figure_title(domain=domain, blends=blends),
        subtitle=subtitle_lines(domain=domain, rows=rows, blends=blends),
        figure_planning=figure_planning,
    )


def _x_domain(*, rows: pl.DataFrame) -> tuple[float, float]:
    """Return the padded x range, including zero, that holds every mark and interval of `rows`."""
    shown = pl.concat(
        [
            rows.select("value", "lower", "upper"),
            rows.select(
                value=pl.col("second_value"),
                lower=pl.col("second_lower"),
                upper=pl.col("second_upper"),
            ).drop_nulls("value"),
        ]
    )
    low = min(shown["value"].to_list() + shown["lower"].drop_nulls().to_list())
    high = max(shown["value"].to_list() + shown["upper"].drop_nulls().to_list())
    return padded_domain(low=low, high=high, include_zero=True)


def _panel_title(*, day: int, blends: bool, domains: Mapping[int, tuple[float, float]]) -> str:
    """Name a panel's lead day, and say where its axis is wider than the first group's."""
    title = f"Lead day {day}"
    if blends and domains[day] != domains[min(domains)]:
        title += " (wider axis than days 1 and 2)"
    return title


BLENDS_README_TEXT: Final[
    str
] = """# ENS plus one weather product minus the ENS mean alone, dots and intervals

Written once by `studies/nwp_forecast_comparison/dot_interval_vs_ens.py --blends` from the saved
per-row losses of `nwp_forecast_comparison_aifs_blends` (the AIFS Single blend and ENS's mean) and
`nwp_forecast_comparison_product_blends` (the ICON-EU, UKV, and WeatherNext 3 blends). It fits
nothing, and it is never overwritten.

- `report.md` prints every dot at the primary setting, and comparison C5, the AIFS Single blend
  minus each ICON-EU blend.
- `intervals.parquet` holds one row per dot, with the columns of the single-product figures'
  `intervals.parquet`. The WeatherNext 3 blend also carries the second mark against the
  leaderboard's 21-month ENS mean.
- `rankings.parquet` holds one row per C5 comparison, with the same statistics.

Both hyperparameter settings, and the verdicts, are in `report.md` of
`nwp_forecast_comparison_product_blends`.
"""


# --- Report and README ---------------------------------------------------------------------------


def _cell(*, row: Mapping[str, object], prefix: str = "") -> str:
    """Format one mark's estimate, with its interval where it has one, or a dash where no mark.

    Args:
        row: One `contrast_rows` row.
        prefix: `second_` for the second mark, empty for the first.

    Returns:
        The cell's text.
    """
    if row[f"{prefix}value"] is None:
        return "-"
    value = f"{row[f'{prefix}value']:+.3f}"
    if not row[f"{prefix}has_interval"]:
        return f"{value} (no interval)"
    dashed = " (dashed)" if row[f"{prefix}dashed"] else ""
    return f"{value} [{row[f'{prefix}lower']:+.3f}, {row[f'{prefix}upper']:+.3f}]{dashed}"


def ranking_lines(*, rankings: Mapping[DomainType, pl.DataFrame]) -> list[str]:
    """Print comparison C5 (AIFS Single blend minus each ICON-EU blend) at the primary setting."""
    lines = [
        "## C5: the AIFS Single blend minus each ICON-EU blend",
        "",
        (
            "AIFS Single ranks above ICON-EU only if the difference against the optimistic "
            "ICON-EU blend has an upper bound below zero; ICON-EU ranks above AIFS Single only "
            "if the difference against the conservative ICON-EU blend has a lower bound above "
            "zero. `fit_product_blends.py`'s `report.md` applies the rule at both settings."
        ),
        "",
        "| Technology | Day | Comparison | Difference (points) | Months | Rows |",
        "|---|---|---|---|---|---|",
    ]
    for domain, frame in rankings.items():
        lines += [
            f"| {domain} | {row['day']} | {row['label']} | {_cell(row=row)} | {row['n_months']} | "
            f"{row['n_rows']} |"
            for row in frame.sort("day", "label").iter_rows(named=True)
        ]
    lines.append("")
    return lines


def report_text(
    *,
    rows: Mapping[DomainType, pl.DataFrame],
    rankings: Mapping[DomainType, pl.DataFrame] | None = None,
) -> str:
    """Print every dot's estimate and interval, so the page and the charts quote the report.

    Args:
        rows: Each technology's `contrast_rows` result.
        rankings: Each technology's comparison C5, for the blend figures, or `None` for the single
            products.

    Returns:
        The report, in Markdown.
    """
    lines = [
        (
            "# Each weather product minus the ENS mean, by lead day"
            if rankings is None
            else "# ENS plus one weather product, minus the ENS mean alone, by lead day"
        ),
        "",
        (
            "Product minus reference, in points of capacity at the primary XGBoost setting. "
            "Each interval is the 95% interval from resampling whole months and a fitting seed. "
            f"A row with fewer than {MIN_MONTHS_FOR_INTERVAL} months prints no interval. "
            "`Rows` is the rows per fitting seed that both arms score. AIFS Single, the AIFS ENS "
            "mean, and WeatherNext 3 have a second column: the same product against the "
            "leaderboard's 21-month ENS mean, on the `(site, time, seed)` keys the two share, "
            "and a last column gives the gap between the two references, the product's "
            "own-file ENS mean minus the leaderboard's. A dashed interval rests on fewer than "
            "12 months."
        ),
        "",
    ]
    for domain, frame in rows.items():
        lines += [
            f"## {domain}",
            "",
            (
                "| Day | Product | Reference arm | Difference (points) | Months | Rows | "
                "Against the 21-month ENS mean (points) | Months | Rows | "
                "Own-file ENS mean minus the 21-month ENS mean (points) |"
            ),
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        ordered = frame.sort("day", "value")
        lines += [
            f"| {row['day']} | {row['label']} | {row['reference']} ({row['reference_source']}) | "
            f"{_cell(row=row)} | {row['n_months']} | {row['n_rows']} | "
            f"{_cell(row=row, prefix='second_')} | {row['second_n_months'] or '-'} | "
            f"{row['second_n_rows'] or '-'} | "
            f"{_cell(row=row, prefix='reference_gap_')} |"
            for row in ordered.iter_rows(named=True)
        ]
        lines.append("")
        lines += [
            f"{note.label}, day {note.day}: {note.text}"
            for note in footnotes_for(domain=domain, rows=frame)
        ]
        lines.append("")
    if rankings is not None:
        lines += ranking_lines(rankings=rankings)
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
            "`seed_spread`, `n_rows` (per fitting seed, rows both arms score), "
            "`treatment_rows` and `reference_rows` (every row of each arm), `n_months`, "
            "`has_interval`, and `dashed` (fewer than 12 months). AIFS Single, the AIFS ENS "
            "mean, and WeatherNext 3 also carry `leaderboard_reference` and its source, and the "
            "same statistics for the second mark, prefixed `second_`, and for the gap between "
            "the two references (own-file ENS mean minus leaderboard ENS mean, on the shared "
            "keys), prefixed `reference_gap_`."
        ),
        "",
        (
            "Each row uses the arms the leaderboards draw: an arm held by both a published CPU "
            "fit and a later GPU refit is read from the GPU refit. The reference is the "
            "leaderboard's ENS-mean arm of the same lead day. The `leads_day10*` folders hold "
            "the same `(site, time, seed)` keys, so a product in one folder pairs with the ENS "
            "mean of another; day 4 uses the ENS mean of `nwp_forecast_comparison_day4_shared`. "
            "The exceptions are IFS HRES 9 km, which lacks 1,197 to 1,536 of the ENS mean's "
            "rows at each day (in 2025-08 and 2026-06, where its archive has no run, and about "
            "1.4% and 1.2% at day 4), and ICON global at day 4 (288 rows fewer for solar); their "
            "paired differences drop those rows, so compare `treatment_rows` with "
            "`reference_rows`. AIFS Single, the AIFS ENS mean, and WeatherNext 3 use the ENS mean "
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
    second_sources = sorted(
        {
            (row["day"], row["leaderboard_reference_source"])
            for frame in rows.values()
            for row in frame.iter_rows(named=True)
            if row["leaderboard_reference"] is not None
        }
    )
    lines += [
        "",
        (
            "AIFS Single, the AIFS ENS mean, and WeatherNext 3 are fitted on 16, 11, and 7 months, "
            "and their ENS mean is refitted on those months, which scores worse than the "
            "leaderboard's 21-month ENS mean on the same keys. Each of their rows therefore has "
            "a second mark, a hollow diamond, against the leaderboard's `ens_mean_day<N>` on the "
            "`(site, time, seed)` keys the two share (for wind too, where the filled dot is "
            "against `ens_meanvec`). The second mark's reference source by day:"
        ),
        "",
        "| Day | Reference source |",
        "|---|---|",
        *(f"| {day} | {source} |" for day, source in second_sources),
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


def write_svg(*, path: Path, chart: alt.VConcatChart, replace: bool, svgo: bool) -> None:
    """Write one figure's SVG, optimised, replacing an existing SVG only once the new one is ready.

    Args:
        path: The SVG's path.
        chart: The figure.
        replace: Whether `path` may already exist. The new SVG is then drawn and optimised
            beside it, and swapped in last, so a failure leaves the old SVG in place.
        svgo: Whether to optimise the SVG.
    """
    target = path.with_name(f"{path.stem}.draft.svg") if replace else path
    if replace:
        target.unlink(missing_ok=True)
    write_once(path=target, write=chart)
    if svgo:
        optimise(path=target)
    if replace:
        target.replace(path)


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
    default_output_dir = studies_dir / "nwp_forecast_comparison_vs_ens_dots_final"
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--svg-dir", type=Path, default=PROJECT_ROOT / "docs" / "studies" / "assets"
    )
    parser.add_argument(
        "--first-figure-number",
        type=int,
        default=FIRST_FIGURE_NUMBER,
        help="The solar figure's number on the page; the wind figure follows it.",
    )
    parser.add_argument("--no-svgo", action="store_true", help="Skip the svgo optimisation.")
    parser.add_argument(
        "--replace-svgs",
        action="store_true",
        help=(
            "Redraw only the SVGs, replacing any that exist, and write nothing to --output-dir, "
            "which must already hold the first run's files."
        ),
    )
    parser.add_argument(
        "--blends",
        action="store_true",
        help="Draw ENS plus one product instead of the single products. Needs --output-dir.",
    )
    args = parser.parse_args()
    if args.blends and args.output_dir == default_output_dir:
        parser.error("--blends needs its own --output-dir, because the default folder is taken")
    rows = {
        domain: compute(data_dir=args.data_dir, domain=domain, blends=args.blends)
        for domain in DOMAINS
    }
    rankings = (
        {
            domain: compute_plan(
                data_dir=args.data_dir, domain=domain, plan=blend_rankings(domain=domain)
            )
            for domain in DOMAINS
        }
        if args.blends
        else None
    )
    files = ("report.md", "intervals.parquet", "README.md", *(("rankings.parquet",) * args.blends))
    stem = "blends_vs_ens" if args.blends else "dots_vs_ens"
    svgs = {domain: args.svg_dir / f"nwp_forecast_{domain}_{stem}.svg" for domain in DOMAINS}
    if args.replace_svgs:
        if not args.output_dir.is_dir():
            msg = f"{args.output_dir} is missing; --replace-svgs redraws the SVGs of a finished run"
            raise FileNotFoundError(msg)
    else:
        taken = [args.output_dir, *(args.output_dir / name for name in files), *svgs.values()]
        existing = [path for path in taken if path.exists()]
        if existing:
            msg = f"{existing} exist; this script writes each output once"
            raise FileExistsError(msg)
    charts = {
        domain: draw(
            rows=rows[domain],
            domain=domain,
            number=args.first_figure_number + index,
            blends=args.blends,
            row_bands=True,
        )
        for index, domain in enumerate(DOMAINS)
    }
    if not args.replace_svgs:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_once(
            path=args.output_dir / "report.md", write=report_text(rows=rows, rankings=rankings)
        )
        write_once(path=args.output_dir / "intervals.parquet", write=pl.concat(rows.values()))
        readme = BLENDS_README_TEXT if args.blends else readme_text(rows=rows)
        write_once(path=args.output_dir / "README.md", write=readme)
        if rankings is not None:
            write_once(
                path=args.output_dir / "rankings.parquet",
                write=pl.concat(rankings.values(), how="vertical_relaxed"),
            )
    for domain, path in svgs.items():
        write_svg(path=path, chart=charts[domain], replace=args.replace_svgs, svgo=not args.no_svgo)
        _LOG.info("wrote %s", path)
        sys.stdout.write(f"{domain} caption: {figure_title(domain=domain, blends=args.blends)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
