"""Score weather products as descriptions of past sunshine, each panel on one common row set.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/809>. The write-up is
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-solar/>.

**Every product is shown to the same booster, on the same rows, with the same temperature.** Each
arm differs from the others only in its irradiance columns, so a contrast between two arms is a
contrast between two products' irradiance. The products are CAMS (a satellite retrieval), ERA5 (a
reanalysis), UKV, ICON-D2, ICON-EU, and ICON global (weather models).

**The rows are chosen by nothing any product says.** Each per-source build drops an hour with a
zero half-hour only where its own irradiance reads bright, so a plain inner join would drop an hour
whenever any one product read it as bright. This script instead drops every hour holding a zero
half-hour, recomputed from the power table, and reads CAMS from the build that keeps every hour
rather than only those CAMS itself rates as reliable.

**The folds are cut inside each era of the UKV record, and every arm is told the era.** The Met
Office upgraded UKV on 21 January 2026. Folds cut from each site's whole span would put every
post-upgrade row in the last fold, scored by a model that never saw post-upgrade UKV. Cutting inside
each era puts both eras in every fold, and the `era_code` feature lets one pooled model learn that
a product's mapping may have changed. A separate fit on the post-upgrade rows alone is run as a
sensitivity check.

**Three instruments answer three questions.** The global arms (`<product>_global`) rank the
products on how well their global irradiance predicts power after per-site recalibration. The split
arms (`<product>_split` against `<product>_erbs`) ask whether a product's published beam/diffuse
split adds anything beyond what a separation model derives from its own global irradiance, which
cancels the gain a model gets from re-encoding alone. The leave-one-site-out arms train on five
sites and score the sixth, which is the situation of a generator with no metered history.

**The forecast lead differs between products and is part of what a consumer receives.** UKV's
archive holds the T+0 analysis; ICON-D2 and ICON-EU hold 1-to-3-hour forecasts; ICON global holds
1-to-6-hour forecasts; ERA5's radiation comes from its own forecasts at 1 to 12 hours. The lead
tables compare ICON-D2 with ICON-EU at matched served leads, and split ICON global against ICON-EU
by ICON global's lead.

**The products are scored in panels, and a panel never mixes periods.** A panel is a set of
products scored on the site-hours every one of them covers, so adding a product with a shorter
record shortens the whole panel rather than giving that product easier or harder months
(`PANELS`):

- `published`: the six products above, December 2022 to September 2026, as the first write-up
  reported them. Its outputs are the first round's, which other studies read, so it is never re-run
  over them.
- `long`: the six plus SARAH-3 (a second satellite retrieval) and ICON-DREAM-EU (a second
  reanalysis), over the same span cut to ICON-DREAM's last month. It runs every analysis the
  `published` panel runs. The panel spans Open-Meteo's change of UKV source on 12 August 2024 and
  treats it as the first round did, with the `ukv_live` scope.
- `all`: all twelve products, adding the four fetched per site from Open-Meteo (ECMWF-IFS-HRES,
  ARPEGE Europe, and the two HARMONIE-AROME models), from September 2024.
- `record`: the four products with records from 2021, which is SARAH-3's start: ERA5, CAMS,
  SARAH-3 and ICON-DREAM-EU. It exists for the year-by-year table of ERA5's error against every
  other product, which every panel prints for its own years.

Run it with `uv run python studies/beam_diffuse_split/weather_products.py`, after
`build_dataset.py` has been run for `open-meteo`, `ukv`, `icon-d2`, `icon-eu`, `icon-global`, the
six added products, and for `cams` with `--min-cams-reliability 0 --suffix _allhours`. `--panel`
chooses the panels, `long`, `all` and `record` by default. With `--report-only` it skips the fits
and rebuilds the report from the losses a full run saved. Neither mode overwrites a file: move an
existing output to a `superseded/` subfolder first.
"""

import argparse
import calendar
import concurrent.futures
import itertools
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from build_dataset import CAMS_PATH, _hourly_power, _pv_sites, nearest_era5_cell, read_era5
from commissioning import drop_commissioning_ramp
from export_cap import with_export_cap
from physics_model import (
    CELL_TEMPERATURE_RISE_K,
    MIN_COS_ZENITH,
    REFERENCE_CELL_TEMPERATURE_C,
    Geometry,
    plane_of_array,
)
from run_experiment import (
    MAX_CONCURRENT_FITS,
    SHARED_FEATURES,
    Job,
    _add_time_features,
    dataset_path_for,
    run_all,
)
from sources import (
    OPEN_METEO_MODELS,
    STUDY_DATA_DIR,
    UNSCORED_EXTRACTED_SPLITS,
    UPDATE_OUTPUT_DIR,
    SourceType,
    point_output_path_for,
)
from studies.bootstrap import (
    BOOTSTRAP_SEED,
    MIN_MONTHS_FOR_INTERVAL,
    N_BOOTSTRAP_RESAMPLES,
    YearInterval,
    bootstrap_difference,
    bootstrap_difference_by_year,
    per_fold_differences,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    assign_folds,
    clamp_to_cap,
    fit_one_fold,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.neighbouring_hours import with_neighbouring_hours
from studies.raw_comparison import raw_column_comparison
from studies.solar import extraterrestrial_horizontal, zenith

_LOG = logging.getLogger(__name__)

PERCENTAGE_POINTS: Final[float] = 100.0

OUTPUT_DIR_NAME: Final[str] = "beam_diffuse_weather_products"
"""The first round's results directory under `STUDY_DATA_DIR`, which the `published` panel names.

The blending study and the chart script read the first round's losses and report from here.
"""

PRODUCTS: Final[dict[str, str]] = {
    "cams": "cams_allhours",
    "era5": "open-meteo",
    "ukv": "ukv",
    "icon_d2": "icon-d2",
    "icon_eu": "icon-eu",
    "icon_global": "icon-global",
}
"""Arm prefix to the build `build_dataset.py` wrote, for every product compared."""

BASE_PRODUCT: Final[str] = "era5"
"""The product whose shared columns - power, capacity, geometry, temperature - the frame keeps.

ERA5 covers every hour the others do, and every per-site build already takes its temperature from
ERA5, so the base supplies nothing a contestant does not already share.
"""

SERVED_LEAD: Final[dict[str, str]] = {
    "cams": "no forecast step (satellite retrieval)",
    "era5": "1 to 12 hours (its own forecasts from 06 and 18 UTC)",
    "ukv": "T+0 (the analysis)",
    "icon_d2": "1 to 3 hours",
    "icon_eu": "1 to 3 hours",
    "icon_global": "1 to 6 hours",
    "sarah3": "no forecast step (satellite retrieval)",
    "icon_dream": "1 to 3 hours (its own forecasts from 3-hourly analyses)",
    "ifs_hres": "not yet measured (see product_checks.md)",
    "arpege": "not yet measured (see product_checks.md)",
    "dmi_harmonie": "not yet measured (see product_checks.md)",
    "knmi_harmonie": "not yet measured (see product_checks.md)",
}
"""How far ahead each product's served hourly value was forecast, as the archive holds it.

The four forecast models the second round adds are fetched from the same archive as UKV and the
ICON products, so their leads follow their run cycles; `check_new_products.py` prints where each
one's hour-to-hour jumps fall, which is where the cycle is read from.
"""

RUN_INTERVAL_HOURS: Final[dict[str, int]] = {
    "icon_d2": 3,
    "icon_eu": 3,
    "icon_global": 6,
    "icon_dream": 3,
}
"""The run cadence of each ICON product, which fixes its served lead at each label hour.

The evidence for the mapping is on the write-up page.
"""

SARAH_SATELLITE_ERAS: Final[tuple[tuple[str, datetime, datetime], ...]] = (
    (
        "Meteosat-11, January 2022 (includes a fortnight from Meteosat-9)",
        datetime(2022, 1, 1, tzinfo=UTC),
        datetime(2022, 2, 1, tzinfo=UTC),
    ),
    (
        "Meteosat-11, 2021",
        datetime(2021, 1, 1, tzinfo=UTC),
        datetime(2022, 1, 1, tzinfo=UTC),
    ),
    (
        "Meteosat-11, February 2022 to 20 March 2023",
        datetime(2022, 2, 1, tzinfo=UTC),
        datetime(2023, 3, 21, tzinfo=UTC),
    ),
    (
        "Meteosat-10, from 21 March 2023",
        datetime(2023, 3, 21, tzinfo=UTC),
        datetime(2027, 1, 1, tzinfo=UTC),
    ),
)
"""The satellite behind SARAH-3's retrieval over each span, as (label, start, end before).

SARAH-3's European disc moved from Meteosat-11 to Meteosat-10 on 21 March 2023, and Meteosat-9
stood in for a fortnight, 17 to 31 January 2022, which is therefore a span of its own, left out of
both Meteosat-11 spans. The report prints SARAH-3's error against CAMS in each span; no era feature
enters the fit.

**Each span still holds a few days served by a different satellite than its label names**, read
from the SIS variable's `platform` attribute in SARAH-3's own files: Meteosat-9 also stands in for
19 to 20 December 2021, inside the "Meteosat-11, 2021" span, and Meteosat-11 stands in for 17 to 20
August 2024, 11 to 12 November 2025, and 24 to 28 April 2026, inside the "Meteosat-10, from 21 March
2023" span. No arm reads which satellite an hour comes from, so these few-day mismatches are not
expected to move the per-span contrasts by much, and the write-up says so.
"""

LEAD_TABLE_HOURS: Final[tuple[int, int]] = (7, 19)
"""The first and last UTC label hours the lead tables use.

Outside these hours the sun is too low for a difference between products to carry signal.
"""

UPGRADE_MONTH: Final[str] = "2026-02"
"""The first whole month after the Met Office made the PS47 upgrade operational on 2026-01-21."""

UPGRADE_DAY: Final[datetime] = datetime(2026, 1, 21, tzinfo=UTC)
"""The upgrade instant.

The rows from here to the end of January carry the pre-upgrade month label but post-upgrade UKV,
so they are dropped from every arm.
"""

UKV_LIVE_INGEST: Final[datetime] = datetime(2024, 8, 12, tzinfo=UTC)
"""When Open-Meteo's own UKV downloader started.

Earlier UKV in the archive is a backfill from a source Open-Meteo does not name.
"""

ICON_EU_CORRUPT_BLOCK: Final[tuple[datetime, datetime]] = (
    datetime(2023, 6, 21, 1, tzinfo=UTC),
    datetime(2023, 6, 21, 6, tzinfo=UTC),
)
"""ICON-EU's one known corrupt block, dropped from every arm (see `sources.OPEN_METEO_MODELS`)."""

DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("cams_global", "icon_d2_global"),
    ("icon_eu_global", "icon_d2_global"),
    ("icon_eu_global", "ukv_global"),
    ("icon_global_global", "icon_eu_global"),
)
"""The four contrasts the recommendations rest on, named before the run.

Whether a satellite retrieval beats the best weather model; what the Great-Britain-wide ICON costs
against the regional one; which Great-Britain-wide weather model is better; and what the global
ICON costs against the European one. Every other contrast in the report is exploratory.
"""

NEW_PRODUCTS: Final[dict[str, str]] = {
    "sarah3": "sarah-3",
    "icon_dream": "icon-dream-eu",
    "ifs_hres": "ecmwf-ifs-hres",
    "arpege": "arpege-europe",
    "dmi_harmonie": "dmi-harmonie-arome",
    "knmi_harmonie": "knmi-harmonie-arome",
}
"""The six products the second round adds, by arm prefix, as `PRODUCTS` lists the first six."""

ALL_PRODUCTS: Final[dict[str, str]] = PRODUCTS | NEW_PRODUCTS
"""Every product either round scores, by arm prefix."""

UNUSABLE_SPLITS: Final[frozenset[str]] = frozenset(
    prefix
    for prefix, source in NEW_PRODUCTS.items()
    if source in UNSCORED_EXTRACTED_SPLITS
    or (source in OPEN_METEO_MODELS and not OPEN_METEO_MODELS[source].split_scored)
)
"""Products whose own split no arm reads, so they get neither a split arm nor an Erbs arm.

Read from `sources.OPEN_METEO_MODELS` and `sources.UNSCORED_EXTRACTED_SPLITS`, which say why for
each: DMI's served direct flux is zero in 48% of daytime hours, ARPEGE's and KNMI's is a separation
model's output, and SARAH-3's is modelled from its own global flux. A split arm would
measure the defect or the separation model, not the weather model, and an Erbs arm exists only as
the split arm's reference.
"""

NEW_PLANNED_CONTRASTS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "long": (
        ("sarah3_global", "cams_global"),
        ("icon_dream_global", "era5_global"),
    ),
    "all": (
        ("ifs_hres_global", "icon_eu_global"),
        ("knmi_harmonie_global", "icon_eu_global"),
        ("dmi_harmonie_global", "icon_d2_global"),
    ),
}
"""The five contrasts the second round names before its run, by the panel each is measured on.

Whether the second satellite retrieval matches CAMS; whether the second reanalysis beats ERA5;
whether ECMWF's global model beats the Great-Britain-wide ICON, measured on the `all` panel because
ECMWF's model is one of the four fetched from Open-Meteo; whether KNMI's 5.5 km
HARMONIE-AROME model over Europe beats the Great-Britain-wide ICON; and whether the Danish 2 km
model matches the German one. Every other contrast involving a new product is exploratory.

**How an unequal served lead biases three of them, written down before any lead is measured.** An
archive value served at a longer lead is a worse description of the hour, so the product with the
longer lead is handicapped, and a contrast between two leads is part model, part lead:

- ECMWF-IFS-HRES − ICON-EU: ECMWF runs every 6 hours, or every 12 for its longest forecasts, where
  ICON-EU runs every 3, so IFS-HRES's served lead is expected to be the longer. The bias is against
  IFS-HRES: a win for it survives equal leads, and a loss may be the lead's.
- KNMI HARMONIE-AROME − ICON-EU and DMI HARMONIE-AROME − ICON-D2: HARMONIE-AROME configurations
  typically run every 1 to 3 hours, so their served lead is expected to be as short as ICON's or
  shorter. The bias, if any, is in their favour: a loss for them survives equal leads, and a win may
  be the lead's.

Once `check_new_products.py` has measured each model's run interval, the interval goes into
`RUN_INTERVAL_HOURS`, and the report prints each planned contrast within the rows where both
products sit at the same served lead, beside the rows where they do not.
"""

PanelType = Literal["published", "long", "all", "record"]
"""Which set of products is scored together, on the site-hours all of them cover."""


class Panel(NamedTuple):
    """One set of products scored together, and what is run and reported on it.

    Attributes:
        products: The arm prefixes scored, keys of `ALL_PRODUCTS`.
        output_dir: Where the panel's losses and report are written.
        full_analysis: Whether to run the analyses the first write-up reports beyond the
            leaderboard: UKV rebuilt from its snapshots, the post-upgrade fit, leave one site out,
            the lead tables, and every scope. Each needs UKV and the ICON products.
        planned: The contrasts named before the run, as (treatment, reference) arm pairs. Each is
            also fitted at the second hyperparameter setting, except on the `published` panel,
            which reproduces the first round as it ran.
        first_time: The first hour the panel scores, or `None` to start where its products'
            records first overlap.
    """

    products: tuple[str, ...]
    output_dir: Path
    full_analysis: bool
    planned: tuple[tuple[str, str], ...]
    first_time: datetime | None = None


PANELS: Final[dict[PanelType, Panel]] = {
    "published": Panel(
        products=tuple(PRODUCTS),
        output_dir=STUDY_DATA_DIR / OUTPUT_DIR_NAME,
        full_analysis=True,
        planned=DECIDING_CONTRASTS,
    ),
    "long": Panel(
        products=(*PRODUCTS, "sarah3", "icon_dream"),
        output_dir=UPDATE_OUTPUT_DIR / "solar_long",
        full_analysis=True,
        planned=(*DECIDING_CONTRASTS, *NEW_PLANNED_CONTRASTS["long"]),
    ),
    "all": Panel(
        products=tuple(ALL_PRODUCTS),
        output_dir=UPDATE_OUTPUT_DIR / "solar_all",
        full_analysis=False,
        planned=NEW_PLANNED_CONTRASTS["all"],
        first_time=datetime(2024, 9, 1, tzinfo=UTC),
    ),
    "record": Panel(
        products=("era5", "cams", "sarah3", "icon_dream"),
        output_dir=UPDATE_OUTPUT_DIR / "solar_record",
        full_analysis=False,
        planned=(),
    ),
}
"""Every panel. The `published` panel's directory is the first round's; every other panel's sits
under `sources.UPDATE_OUTPUT_DIR`.

The `all` panel starts on 1 September 2024 rather than when HARMONIE-AROME's archive does, in July
2024, because Open-Meteo's UKV before 12 August 2024 is a backfill from a source it does not name:
a change of source is an era boundary, and a panel of 26 months has no room to cut folds on both
sides of one. The first whole month after the change is the start.
"""

DEFAULT_PANELS: Final[tuple[PanelType, ...]] = ("long", "all", "record")
"""The panels a run fits unless told otherwise: the second round's."""

UKV_SNAPSHOT_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "ukv_trap_global": ("ghi_trap_ukv",),
    "ukv_pair_global": ("ghi_instant_previous_ukv", "ghi_instant_ukv"),
    "ukv_trap_ctx_global": ("ghi_trap_previous_ukv", "ghi_trap_ukv", "ghi_trap_next_ukv"),
    "icon_eu_ctx_global": ("ghi_previous_icon_eu", "ghi_icon_eu", "ghi_next_icon_eu"),
}
"""Four post hoc arms: two build UKV's hour from its own snapshots, and two give UKV's snapshot mean
and ICON-EU's hourly mean the same context.

UKV publishes an instantaneous field each hour, and Open-Meteo's served hourly value is the
snapshot at the hour's end rescaled by a ratio of cosines, where every ICON product serves a true
mean over the hour. `ukv_trap_global` is the mean of the snapshots at both ends of the hour, and
`ukv_pair_global` shows the model both snapshots. The two `_ctx` arms add the hour before and the
hour after. Every arm here is post hoc.
"""

LEAVE_ONE_SITE_OUT_SEED: Final[int] = SEEDS[0]
"""The one seed the leave-one-site-out fits use, which keeps them to one fit per site and fold."""

METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
"""The loss every table reports: each row's clamped error over its own generator's capacity."""

SEASONS: Final[dict[int, str]] = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
"""Calendar month to meteorological season, for the breakdowns."""

SCOPES: Final[tuple[str, ...]] = ("all", "pre", "pre_matched", "post", "ukv_live", "cams_reliable")
"""Every scope the pooled losses are bootstrapped over."""

CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
        "| Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|",
)


ICON_D2_WESTERN_EDGE_DEG: Final[tuple[float, float]] = (-2.5, -2.7)
"""Two longitudes bracketing ICON-D2's western edge at the generators' latitude.

Open-Meteo's ICON-D2 serves data at 2.5°W but not at 2.7°W near 53°N, so the edge lies between.
"""

ERA5_GRID_DEG: Final[float] = 0.25
"""The spacing of the grid ERA5 is published on, which sets which generators share a cell."""

EARTH_RADIUS_KM: Final[float] = 6371.0


def _km(*, latitudes: tuple[float, float], longitudes: tuple[float, float]) -> float:
    """Return the great-circle distance between two points, by the haversine formula.

    Args:
        latitudes: The two points' latitudes in degrees.
        longitudes: The two points' longitudes in degrees.

    Returns:
        The distance in kilometres.
    """
    first, second = (math.radians(latitude) for latitude in latitudes)
    half_chord = (
        math.sin((second - first) / 2) ** 2
        + math.cos(first)
        * math.cos(second)
        * math.sin(math.radians(longitudes[1] - longitudes[0]) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(half_chord))


def geometry_lines(*, sites: pl.DataFrame, noun: str) -> list[str]:
    """Report how the generators sit relative to each other, to ICON-D2's edge, and to ERA5's grid.

    Only distances and counts are printed, never a coordinate.

    Args:
        sites: The roster, with `latitude` and `longitude`.
        noun: What the generators are called in the heading, such as `solar farms`.

    Returns:
        Markdown lines.
    """
    latitudes = sites["latitude"].to_list()
    longitudes = sites["longitude"].to_list()
    points = list(zip(latitudes, longitudes, strict=True))
    pairs = [
        _km(latitudes=(a[0], b[0]), longitudes=(a[1], b[1]))
        for a, b in itertools.combinations(points, 2)
    ]
    middle = sum(latitudes) / len(latitudes)
    north_south = _km(latitudes=(min(latitudes), max(latitudes)), longitudes=(0.0, 0.0))
    east_west = _km(latitudes=(middle, middle), longitudes=(min(longitudes), max(longitudes)))
    cells = {
        (round(latitude / ERA5_GRID_DEG), round(longitude / ERA5_GRID_DEG))
        for latitude, longitude in points
    }
    lines = [
        f"#### Where the {noun} sit",
        "",
        f"- Pairwise distance: {min(pairs):.1f} km to {max(pairs):.1f} km.",
        f"- Bounding box: {north_south:.1f} km north to south by {east_west:.1f} km east to west.",
    ]
    for edge in ICON_D2_WESTERN_EDGE_DEG:
        west = [_km(latitudes=(lat, lat), longitudes=(lon, edge)) for lat, lon in points]
        lines.append(
            f"- Due-west distance to {abs(edge)}°W: {min(west):.0f} km to {max(west):.0f} km."
        )
    lines.append(
        f"- ERA5 {ERA5_GRID_DEG}° grid cells holding them: {len(cells)} (nearest grid point)."
    )
    return lines


def _named(column: str, product: str) -> str:
    """Return a product's copy of an irradiance column, such as `bhi_icon_eu`.

    Args:
        column: The build's column name, such as `bhi_w_m2`.
        product: The product prefix.

    Returns:
        The joined frame's column name.
    """
    return f"{column.removesuffix('_w_m2')}_{product}"


def joined(*, products: tuple[str, ...] = tuple(PRODUCTS)) -> pl.DataFrame:
    """Inner-join every product on the site-hours all of them cover.

    Args:
        products: The arm prefixes to join, keys of `ALL_PRODUCTS`, which must include
            `BASE_PRODUCT`. The default is the first round's six.

    Returns:
        One row per common site-hour, carrying `ghi_<p>`, `bhi_<p>`, `dhi_<p>`, `erbs_bhi_<p>`, and
        `erbs_dhi_<p>` for every product `p`, and the base product's power, geometry, and
        temperature. Where the products include UKV, the rows are those on which UKV's snapshots
        also exist, and where they include ICON-EU, those on which ICON-EU's neighbouring hours do.

    Raises:
        ValueError: If `products` leaves out `BASE_PRODUCT`, whose power every row takes.
    """
    if BASE_PRODUCT not in products:
        msg = f"every panel needs {BASE_PRODUCT}, whose power and temperature the rows take"
        raise ValueError(msg)
    irradiance = ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2", "erbs_bhi_w_m2", "erbs_dhi_w_m2")
    base = pl.read_parquet(dataset_path_for(source=ALL_PRODUCTS[BASE_PRODUCT]))
    frame = base.with_columns(
        pl.col(column).alias(_named(column, BASE_PRODUCT)) for column in irradiance
    ).drop(*irradiance)
    for product in products:
        if product == BASE_PRODUCT:
            continue
        other = pl.read_parquet(dataset_path_for(source=ALL_PRODUCTS[product])).select(
            "site",
            "time",
            *(pl.col(column).alias(_named(column, product)) for column in irradiance),
        )
        frame = frame.join(other, on=["site", "time"], how="inner")
    if "ukv" in products:
        frame = frame.join(_ukv_snapshots(), on=["site", "time"], how="inner")
    if "icon_eu" in products:
        frame = frame.join(_icon_eu_context(), on=["site", "time"], how="inner")
    return frame.sort("site", "time")


def _neighbours(*, frame: pl.DataFrame, column: str, prefix: str, product: str) -> pl.DataFrame:
    """Return one column's value in the hour before and the hour after each row.

    Args:
        frame: One row per (site, time), carrying `column`.
        column: The column to shift.
        prefix: The joined frame's name stem, such as `ghi` or `ghi_trap`.
        product: The product suffix.

    Returns:
        One row per (site, time) with `<prefix>_previous_<product>` and `<prefix>_next_<product>`.
    """
    shifted = {
        "previous": frame.select("site", pl.col("time").dt.offset_by("1h"), pl.col(column)),
        "next": frame.select("site", pl.col("time").dt.offset_by("-1h"), pl.col(column)),
    }
    previous, following = (
        shifted[side].rename({column: f"{prefix}_{side}_{product}"})
        for side in ("previous", "next")
    )
    return previous.join(following, on=["site", "time"], how="inner")


MAX_INSTANT_OVER_TOA: Final[float] = 1.1
"""How far `ghi_instant_w_m2` may exceed the top-of-atmosphere flux at its own instant before the
row is dropped as a sunrise spike.

Open-Meteo serves the `_instant` columns by multiplying its stored hourly mean by the
instantaneous-over-hour-mean cosine of the solar zenith angle, computed without refraction
(`verify_ukv_lineage.GEOMETRY_FACTOR_RANGE` names and bounds the same factor, for a different,
stricter purpose — picking instants clean enough to compare against the Met Office's own files,
not identifying which served values are unusable). Near sunrise that factor departs from one on
most hours without producing an implausible value, so filtering on the factor's range would drop
more than half of every product's rows. What actually makes a row unusable is the factor's size:
in the first hours after sunrise it can reach the hundreds, and the stored mean at those hours is
still small (1 to 28 W/m2 at the rows this rule drops), so multiplying the two produces a value
with no physical meaning — measured at up to 16,537 W/m2 in this archive, against a real GHI
ceiling of about 1,400 W/m2. The stored mean's own 1 W/m2 rounding is a minor contributor,
explaining the excess over the ceiling in only about one row in ten. The row is instead flagged
directly against the physical ceiling: the flux a horizontal surface receives with no atmosphere
at all, from `studies.solar.extraterrestrial_horizontal`, evaluated with pvlib's apparent
(refracted) solar zenith angle. Near the horizon that refracted angle allows more flux than
Open-Meteo's refraction-free geometry does, and the 10% margin is a tolerance for that mismatch,
not for a physical process such as cloud enhancement: every row this rule drops has the sun below
4.5 degrees of apparent elevation.
"""


MIN_TOP_OF_ATMOSPHERE_W_M2: Final[float] = 1e-6
"""Floors the top-of-atmosphere flux so a true night-side instant gets a ceiling of (near) zero
rather than exactly zero, which any served value above zero then correctly fails."""


def _without_sunrise_spikes(*, frame: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows where `ghi_instant_w_m2` exceeds what is physically possible at that instant.

    Args:
        frame: Rows carrying `site`, `time`, and `ghi_instant_ukv`, one row per (site, time).
        sites: The roster, carrying `site`, `latitude`, and `longitude`.

    Returns:
        `frame`, with every row whose `ghi_instant_ukv` exceeds `MAX_INSTANT_OVER_TOA` times the
        top-of-atmosphere flux at that instant removed.
    """
    coordinates = {
        str(row["site"]): (float(row["latitude"]), float(row["longitude"]))
        for row in sites.to_dicts()
    }
    kept: list[pl.DataFrame] = []
    for (site,), rows in frame.sort("site", "time").group_by(["site"], maintain_order=True):
        latitude, longitude = coordinates[str(site)]
        stamps = rows["time"]
        angle = zenith(stamps=stamps, latitude=latitude, longitude=longitude)
        top_of_atmosphere = extraterrestrial_horizontal(stamps=stamps, zenith_deg=angle)
        ceiling = MAX_INSTANT_OVER_TOA * np.maximum(top_of_atmosphere, MIN_TOP_OF_ATMOSPHERE_W_M2)
        plausible = rows["ghi_instant_ukv"].to_numpy() <= ceiling
        kept.append(rows.filter(pl.Series(plausible)))
    return pl.concat(kept)


def _ukv_snapshots() -> pl.DataFrame:
    """Return UKV's instantaneous global irradiance at both ends of each hour, and their mean.

    Every row whose `ghi_instant_ukv` exceeds what solar geometry allows at that instant (see
    `_without_sunrise_spikes`) is dropped before the trapezoid mean and the neighbouring-hour
    context are built from it, so a dropped instant also drops the hour that would have averaged
    it in and the neighbouring hour that would have used it as context.

    Returns:
        One row per (site, time) with `ghi_instant_previous_ukv`, `ghi_instant_ukv` and
        `ghi_trap_ukv`, for every hour whose both snapshots were served and neither was a sunrise
        spike.
    """
    download = pl.read_parquet(point_output_path_for(source="ukv")).select(
        "site", "time", ghi_instant_ukv=pl.col("ghi_instant_w_m2")
    )
    clean = _without_sunrise_spikes(frame=download, sites=_pv_sites())
    previous = clean.select(
        "site",
        time=pl.col("time").dt.offset_by("1h"),
        ghi_instant_previous_ukv=pl.col("ghi_instant_ukv"),
    )
    snapshots = clean.join(previous, on=["site", "time"], how="inner").with_columns(
        ghi_trap_ukv=(pl.col("ghi_instant_previous_ukv") + pl.col("ghi_instant_ukv")) / 2.0
    )
    context = _neighbours(frame=snapshots, column="ghi_trap_ukv", prefix="ghi_trap", product="ukv")
    return snapshots.join(context, on=["site", "time"], how="inner")


def _icon_eu_context() -> pl.DataFrame:
    """Return ICON-EU's hourly mean in the hour before and the hour after each hour.

    The neighbours are read from the raw download, so on 2023-06-21 the 07:00 UTC row carries the
    corrupt 06:00 value as its previous hour in `icon_eu_ctx_global`, the one exception to that
    block being dropped from every arm. It touches five rows of one post hoc arm.

    Returns:
        One row per (site, time) with `ghi_previous_icon_eu` and `ghi_next_icon_eu`.
    """
    download = pl.read_parquet(point_output_path_for(source="icon-eu")).select(
        "site", "time", "ghi_w_m2"
    )
    return _neighbours(frame=download, column="ghi_w_m2", prefix="ghi", product="icon_eu")


CONTEXT_PRODUCTS: Final[tuple[str, ...]] = ("cams", "era5", "icon_d2", "icon_global")
"""The products `with_irradiance_context` adds neighbouring hours for.

UKV's and ICON-EU's neighbouring hours are already in `joined`'s frame, as `ghi_trap_previous_ukv`,
`ghi_trap_next_ukv`, `ghi_previous_icon_eu`, and `ghi_next_icon_eu`.
"""


CONTEXT_SOURCES: Final[dict[str, SourceType]] = {"icon_d2": "icon-d2", "icon_global": "icon-global"}
"""The downloads the per-site weather models' neighbouring hours are read from."""


def _irradiance_download(*, product: str) -> pl.DataFrame:
    """Return one product's served global irradiance at every site and hour it was downloaded.

    ERA5 is gridded, so each site reads its nearest cell, as `build_dataset.py` does.

    Args:
        product: A key of `CONTEXT_PRODUCTS`.

    Returns:
        One row per (site, time) with `ghi_w_m2`.
    """
    if product == "era5":
        gridded = read_era5(source="open-meteo")
        cells = nearest_era5_cell(sites=_pv_sites(), era5=gridded)
        return cells.join(
            gridded,
            left_on=["cell_latitude", "cell_longitude"],
            right_on=["latitude", "longitude"],
        ).select("site", "time", "ghi_w_m2")
    path = (
        CAMS_PATH if product == "cams" else point_output_path_for(source=CONTEXT_SOURCES[product])
    )
    return pl.read_parquet(path).select("site", "time", "ghi_w_m2")


def with_irradiance_context(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the hour before and the hour after each row for every product in `CONTEXT_PRODUCTS`.

    The neighbours are read from each product's own download, not from the scored rows, which
    exclude hours by the target. Each download is checked to reproduce the frame's own column at
    offset zero first, so a neighbour cannot come from a series labelled differently.

    Args:
        frame: The common rows, carrying `ghi_<product>` for every product.

    Returns:
        `frame`, in its own row order, with `ghi_previous_<product>` and `ghi_next_<product>`.

    Raises:
        ValueError: If a download does not reproduce the frame's column at offset zero.
    """
    for product in CONTEXT_PRODUCTS:
        own = f"ghi_{product}"
        frame = with_neighbouring_hours(
            frame=frame,
            source=_irradiance_download(product=product),
            columns={
                f"{own}_at_zero": ("ghi_w_m2", 0),
                f"ghi_previous_{product}": ("ghi_w_m2", -1),
                f"ghi_next_{product}": ("ghi_w_m2", 1),
            },
        )
        if not frame[f"{own}_at_zero"].cast(pl.Float64).equals(frame[own].cast(pl.Float64)):
            msg = f"the {product} download does not reproduce {own} at offset zero"
            raise ValueError(msg)
        frame = frame.drop(f"{own}_at_zero")
    return frame


def common_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows no product should be scored on, by rules no product's values decide.

    Args:
        frame: The joined frame.

    Returns:
        The frame without zero-half-hour hours, ICON-EU's corrupt block, the post-upgrade tail of
        January 2026, and the commissioning ramp.
    """
    zero_hours = (
        _hourly_power(sites=_pv_sites()).filter(pl.col("has_zero_half_hour")).select("site", "time")
    )
    start, end = ICON_EU_CORRUPT_BLOCK
    february = datetime(2026, 2, 1, tzinfo=UTC)
    return drop_commissioning_ramp(
        dataset=frame.join(zero_hours, on=["site", "time"], how="anti")
        .filter(
            ~pl.col("time").is_between(start, end),
            ~pl.col("time").is_between(UPGRADE_DAY, february, closed="left"),
        )
        .sort("site", "time")
    )


def with_eras(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Label each row's era, add the era feature, and cut folds inside each era.

    Args:
        frame: The common rows, carrying `month`.

    Returns:
        The frame with `era`, `era_code` and `fold`.
    """
    post = pl.col("month") >= UPGRADE_MONTH
    labelled = frame.with_columns(
        era=pl.when(post).then(pl.lit("post")).otherwise(pl.lit("pre")),
        era_code=post.cast(pl.Int8),
    )
    return assign_folds(dataset=labelled, by=("site", "era"))


def _arm_columns(
    *, products: tuple[str, ...], with_snapshot_arms: bool
) -> dict[str, tuple[str, ...]]:
    """Return every pooled arm's irradiance columns: three per product, and the UKV snapshot arms.

    Args:
        products: The arm prefixes, keys of `ALL_PRODUCTS`.
        with_snapshot_arms: Whether to add `UKV_SNAPSHOT_ARMS`, which need UKV and ICON-EU.

    Returns:
        Each arm's irradiance columns, in the order the arms are fitted.
    """
    arms: dict[str, tuple[str, ...]] = {}
    for product in products:
        ghi = _named("ghi_w_m2", product)
        arms[f"{product}_global"] = (ghi,)
        if product in UNUSABLE_SPLITS:
            continue
        arms[f"{product}_split"] = (ghi, _named("bhi_w_m2", product), _named("dhi_w_m2", product))
        arms[f"{product}_erbs"] = (
            ghi,
            _named("erbs_bhi_w_m2", product),
            _named("erbs_dhi_w_m2", product),
        )
    if with_snapshot_arms:
        arms |= UKV_SNAPSHOT_ARMS
    return arms


def jobs(
    *, products: tuple[str, ...] = tuple(PRODUCTS), with_snapshot_arms: bool = True
) -> list[Job]:
    """Return the three arms per product: global only, its own split, and Erbs on its own global.

    A product in `UNUSABLE_SPLITS` gets its global arm alone.

    Args:
        products: The arm prefixes, keys of `ALL_PRODUCTS`. The default is the first round's six.
        with_snapshot_arms: Whether to add `UKV_SNAPSHOT_ARMS`, which need UKV and ICON-EU.

    Returns:
        One job per arm, every arm shown the shared features and the era.
    """
    shared = (*SHARED_FEATURES, "era_code")
    return [
        (arm, "pooled", "power_mw", (*shared, *columns), PRIMARY_HYPER_PARAMETERS, False)
        for arm, columns in _arm_columns(
            products=products, with_snapshot_arms=with_snapshot_arms
        ).items()
    ]


def sensitivity_jobs(*, panel: Panel) -> list[Job]:
    """Return every arm in a panel's planned contrasts, at the second hyperparameter setting.

    A second setting shows whether an ordering belongs to the features or to the settings.

    Args:
        panel: The panel whose planned contrasts are refitted.

    Returns:
        One job per arm, named as in `jobs`, under the setting `sensitivity`.
    """
    columns = _arm_columns(products=panel.products, with_snapshot_arms=panel.full_analysis)
    shared = (*SHARED_FEATURES, "era_code")
    arms = dict.fromkeys(arm for contrast in panel.planned for arm in contrast)
    return [
        (
            arm,
            "sensitivity",
            "power_mw",
            (*shared, *columns[arm]),
            SENSITIVITY_HYPER_PARAMETERS,
            False,
        )
        for arm in arms
    ]


def _post_only_losses(
    *, frame: pl.DataFrame, products: tuple[str, ...], max_workers: int
) -> pl.DataFrame:
    """Fit every global arm on the post-upgrade rows alone, as the separate-era sensitivity check.

    Args:
        frame: The common rows with `era`.
        products: The arm prefixes to fit.
        max_workers: How many fits run at once.

    Returns:
        Losses for every global arm, scored on the post-upgrade rows by models trained on them only.
    """
    post = assign_folds(dataset=frame.filter(pl.col("era") == "post").drop("fold"))
    jobs: list[Job] = [
        (
            f"{product}_global",
            "post_only",
            "power_mw",
            (*SHARED_FEATURES, _named("ghi_w_m2", product)),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for product in products
    ]
    return run_all(dataset=post, jobs=jobs, max_workers=max_workers)


def _leave_one_site_out_losses(
    *, frame: pl.DataFrame, products: tuple[str, ...], max_workers: int
) -> pl.DataFrame:
    """Train on five sites' capacity-normalised power and score the sixth, one fold at a time.

    The scored site is never trained on, and neither are the scored fold's calendar months at any
    other site: the six sites share their weather, so a model trained on a neighbour's power in the
    same hours would learn each day's outcome rather than transfer. This is the withholding
    `run_experiment._add_learned_split` applies for the same reason. The target is a fraction of
    capacity, so five sites can share one model, and the site's own capacity is assumed known.

    Args:
        frame: The common rows, carrying `fold`, `month`, `constrained` and `cap_mw`.
        products: The arm prefixes to fit.
        max_workers: How many fits run at once.

    Returns:
        One row per (site, time, arm) with the capped error as a fraction of capacity.
    """
    frame = frame.with_columns(power_fraction=pl.col("power_mw") / pl.col("effective_capacity_mw"))
    folds = frame.select("site", "fold").unique().sort("site", "fold").rows()

    def _one(product: str, site: str, fold: int) -> pl.DataFrame:
        test = frame.filter((pl.col("site") == site) & (pl.col("fold") == fold))
        train = frame.filter(
            (pl.col("site") != site)
            & ~pl.col("constrained")
            & ~pl.col("month").is_in(test["month"].unique().to_list())
        )
        point, _ = fit_one_fold(
            train=train,
            test=test,
            features=[*SHARED_FEATURES, "era_code", _named("ghi_w_m2", product)],
            target="power_fraction",
            hyper_parameters=PRIMARY_HYPER_PARAMETERS,
            seed=LEAVE_ONE_SITE_OUT_SEED,
            with_quantiles=False,
        )
        capacity = test["effective_capacity_mw"].cast(pl.Float64).to_numpy()
        capped = clamp_to_cap(prediction=point * capacity, cap_mw=test["cap_mw"]) / capacity
        error = np.abs(test["power_fraction"].cast(pl.Float64).to_numpy() - capped)
        return test.select("site", "time", "month", "fold", "era").with_columns(
            pl.Series(METRIC, error),
            arm=pl.lit(f"{product}_global"),
            seed=pl.lit(LEAVE_ONE_SITE_OUT_SEED, dtype=pl.Int32),
        )

    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_one, product=product, site=site, fold=fold)
            for product in products
            for site, fold in folds
        ]
        outputs.extend(future.result() for future in concurrent.futures.as_completed(futures))
    return pl.concat(outputs)


def _log_capacity_by_month(
    *, frame: pl.DataFrame, products: tuple[str, ...]
) -> dict[str, pl.DataFrame]:
    """Return each product's log implied capacity per site and month, split into season and noise.

    A month's implied capacity is the metered output divided by what a fixed panel model predicts
    per megawatt from the product: a south-facing panel at 30° tilt, the Erbs split of the
    product's own global irradiance, and a -0.4 %/K temperature derate. Only unconstrained hours
    with the sun above 10° are used. The logarithm is taken per site and month. `seasonal` is the
    site's mean for that calendar month minus the site's overall mean, and `residual` is what is
    left once the calendar month's mean is subtracted.

    Args:
        frame: The common rows.
        products: The arm prefixes to measure, which must include `cams`, the reference.

    Returns:
        Per product, one row per (site, month) with `log_capacity`, `calendar`, `seasonal`, and
        `residual`.
    """
    daylight = frame.filter(~pl.col("constrained") & (pl.col("solar_elevation_deg") > 10.0))
    zenith = np.radians(daylight["solar_zenith_deg"].to_numpy())
    log_by_product: dict[str, pl.DataFrame] = {}
    for product in products:
        geometry = Geometry(
            cos_zenith=np.maximum(np.cos(zenith), MIN_COS_ZENITH),
            sin_zenith=np.sin(zenith),
            solar_azimuth_rad=np.radians(daylight["solar_azimuth_deg"].to_numpy()),
            global_horizontal=daylight[_named("ghi_w_m2", product)].to_numpy(),
            beam_horizontal=daylight[_named("erbs_bhi_w_m2", product)].to_numpy(),
            diffuse_horizontal=daylight[_named("erbs_dhi_w_m2", product)].to_numpy(),
            air_temperature_c=daylight["temp_c"].to_numpy(),
        )
        irradiance = plane_of_array(
            geometry=geometry, tilt_rad=np.radians(30.0), azimuth_rad=np.radians(180.0)
        )
        cell = geometry.air_temperature_c + CELL_TEMPERATURE_RISE_K * irradiance / 1000.0
        per_mw = irradiance / 1000.0 * (1.0 - 0.004 * (cell - REFERENCE_CELL_TEMPERATURE_C))
        log_by_product[product] = (
            daylight.select("site", "month", power=pl.col("power_mw"))
            .with_columns(per_mw=pl.Series(per_mw))
            .group_by("site", "month")
            .agg(log_capacity=(pl.col("power").sum() / pl.col("per_mw").sum()).log())
            .with_columns(calendar=pl.col("month").str.slice(-2))
            .with_columns(
                seasonal=pl.col("log_capacity").mean().over("site", "calendar")
                - pl.col("log_capacity").mean().over("site"),
                residual=pl.col("log_capacity")
                - pl.col("log_capacity").mean().over("site", "calendar"),
            )
        )
    return log_by_product


def _implied_capacity(*, log_by_product: dict[str, pl.DataFrame]) -> list[str]:
    """Measure how steady each product's implied capacity is from month to month, and by season.

    Capacity estimation reads a product's irradiance with no model fitted to the generator, so the
    question is how far one month's implied capacity strays. The spread of
    `_log_capacity_by_month`'s residual is the month-to-month noise with the seasonal cycle
    removed. The calendar-month means give the seasonal swing, reported as the departures of
    November, December and January from the annual mean.

    Args:
        log_by_product: The output of `_log_capacity_by_month`.

    Returns:
        Markdown lines: a table of spread, interval against CAMS, and each winter month's departure.
    """
    months = sorted(log_by_product["cams"]["month"].unique().to_list())
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(months), size=(N_BOOTSTRAP_RESAMPLES, len(months)))
    lines = [
        (
            "| Product | Month-to-month spread, seasonal cycle removed | Spread minus CAMS's "
            "| November, December, January against the annual mean |"
        ),
        "|---|---|---|---|",
    ]
    cams_spread = _spread_by_draw(frame=log_by_product["cams"], months=months, draws=draws)
    for product, frame_log in log_by_product.items():
        spread = _spread_by_draw(frame=frame_log, months=months, draws=draws)
        difference = (spread - cams_spread) * PERCENTAGE_POINTS
        plug_in = (
            float(frame_log.select(pl.col("residual").std()).item())
            - float(log_by_product["cams"].select(pl.col("residual").std()).item())
        ) * PERCENTAGE_POINTS
        winter = " / ".join(
            f"{np.expm1(_seasonal(frame=frame_log, calendar=month)) * PERCENTAGE_POINTS:+.0f}%"
            for month in ("11", "12", "01")
        )
        residual_spread = float(frame_log.select(pl.col("residual").std()).item())
        lower, upper = np.percentile(difference, (2.5, 97.5))
        lines.append(
            f"| {product} | {residual_spread * PERCENTAGE_POINTS:.1f}% "
            f"| {plug_in:+.1f} [{lower:+.1f}, {upper:+.1f}] "
            f"| {winter} |"
        )
    return lines


def _seasonal(*, frame: pl.DataFrame, calendar: str) -> float:
    """Return a calendar month's mean seasonal term of the log implied capacity.

    Args:
        frame: One product's output of `_log_capacity_by_month`.
        calendar: The two-digit calendar month, such as `12`.

    Returns:
        The mean over that month's site-months of `seasonal`.
    """
    return float(
        frame.filter(pl.col("calendar") == calendar).select(pl.col("seasonal").mean()).item()
    )


def _implied_capacity_by_month(*, log_by_product: dict[str, pl.DataFrame]) -> list[str]:
    """Report each product's implied capacity in every calendar month, against its annual mean.

    Each value is the exponential, minus one, of the mean over that month's site-months of
    `_log_capacity_by_month`'s `seasonal` term, as a percentage.

    Args:
        log_by_product: The output of `_log_capacity_by_month`.

    Returns:
        Markdown lines: one row per product, one column per calendar month.
    """
    months = [f"{month:02d}" for month in range(1, 13)]
    lines = [
        "| Product | " + " | ".join(calendar.month_abbr[int(m)] for m in months) + " |",
        "|---" * 13 + "|",
    ]
    for product, frame_log in log_by_product.items():
        cells = [
            f"{np.expm1(_seasonal(frame=frame_log, calendar=month)) * PERCENTAGE_POINTS:+.1f}"
            for month in months
        ]
        lines.append(f"| {product} | " + " | ".join(cells) + " |")
    december = log_by_product["cams"].filter(pl.col("calendar") == "12")
    lines += [
        "",
        (
            f"December's figure rests on {december['month'].n_unique()} Decembers and "
            f"{december.height} generator-months."
        ),
    ]
    return lines


def _spread_by_draw(*, frame: pl.DataFrame, months: list[str], draws: np.ndarray) -> np.ndarray:
    """Return the residual spread for each bootstrap draw of whole months.

    Args:
        frame: One row per (site, month) with `residual`.
        months: Every month label, in the order `draws` indexes.
        draws: Month indices, one row per resample.

    Returns:
        One standard deviation per resample.
    """
    by_month = frame.group_by("month").agg(pl.col("residual")).sort("month")
    residuals = dict(zip(by_month["month"].to_list(), by_month["residual"].to_list(), strict=True))
    return np.array(
        [
            np.std(np.concatenate([residuals.get(months[index], []) for index in draw]), ddof=1)
            for draw in draws
        ]
    )


def _scope(*, losses: pl.DataFrame, scope: str) -> pl.DataFrame:
    """Restrict the losses to one scope of the pooled run.

    Args:
        losses: Per-row losses carrying `month` and `time`.
        scope: One of `SCOPES`.

    Returns:
        The rows belonging to that scope.

    Raises:
        ValueError: If `scope` is not one of `SCOPES`.
    """
    losses = losses.with_columns(
        era=pl.when(pl.col("month") >= UPGRADE_MONTH).then(pl.lit("post")).otherwise(pl.lit("pre"))
    )
    if scope == "all":
        return losses
    if scope in ("pre", "post"):
        return losses.filter(pl.col("era") == scope)
    if scope == "pre_matched":
        post_months = {label[-2:] for label in losses.filter(pl.col("era") == "post")["month"]}
        return losses.filter(
            (pl.col("era") == "pre") & pl.col("month").str.slice(-2).is_in(sorted(post_months))
        )
    if scope == "ukv_live":
        return losses.filter(pl.col("time") >= UKV_LIVE_INGEST)
    if scope == "cams_reliable":
        reliable = pl.read_parquet(dataset_path_for(source="cams")).select("site", "time")
        return losses.join(reliable, on=["site", "time"], how="semi")
    msg = f"unknown scope {scope}"
    raise ValueError(msg)


def _mae(*, losses: pl.DataFrame, arm: str) -> float:
    """Return one arm's mean error, in percentage points of capacity.

    Args:
        losses: Per-row losses.
        arm: The arm to score.

    Returns:
        The mean of each row's error over its generator's capacity, in percentage points.
    """
    rows = losses.filter(pl.col("arm") == arm)
    return float(rows.select(pl.col(METRIC).mean()).item()) * PERCENTAGE_POINTS


def _contrast_line(*, losses: pl.DataFrame, treatment: str, reference: str, label: str) -> str:
    """Return one markdown row: the difference, its interval, and the folds agreeing in sign.

    Args:
        losses: Per-row losses holding both arms.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.
        label: The scope label for the first column.

    Returns:
        The table row.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    same_sign = sum(np.sign(value) == np.sign(interval["difference"]) for value in folds)
    difference, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    )
    excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
    return (
        f"| {label} | {treatment} − {reference} | {difference:+.3f} | "
        f"[{lower:+.3f}, {upper:+.3f}] | {'**yes**' if excludes else 'no'} | "
        f"{same_sign} of {len(folds)} | {interval['n_rows']:,} |"
    )


def _served_lead(*, product: str) -> pl.Expr:
    """Return the served lead in hours of a product's value at each row's label hour.

    A served hourly value is a backward mean over the hour ending at its label, so the run that
    supplies label hour `T` is the latest one at or before `T - 1`, and the lead is `T` minus that
    run: 1, 2 or 3 hours for a 3-hourly model, 1 to 6 for a 6-hourly one.

    Args:
        product: A key of `RUN_INTERVAL_HOURS`.

    Returns:
        The lead, as an integer expression.
    """
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    return ((hour - 1) % RUN_INTERVAL_HOURS[product]) + 1


def _matched_lead_lines(*, panel: Panel, losses: pl.DataFrame) -> list[str]:
    """Split each planned contrast by whether its two products sit at the same served lead.

    Only a contrast whose two products both have an entry in `RUN_INTERVAL_HOURS` can be split;
    the others are listed as waiting for a measured run interval.

    Args:
        panel: The panel reported.
        losses: The pooled losses.

    Returns:
        Markdown lines.
    """
    first, last = LEAD_TABLE_HOURS
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    daytime = losses.filter(hour.is_between(first, last))
    lines = ["#### Planned contrasts at matched and unmatched served leads", "", *CONTRAST_HEADER]
    waiting: list[str] = []
    for treatment, reference in panel.planned:
        products = [arm.removesuffix("_global") for arm in (treatment, reference)]
        if not all(product in RUN_INTERVAL_HOURS for product in products):
            waiting.append(f"{treatment} − {reference}")
            continue
        treatment_lead, reference_lead = (_served_lead(product=product) for product in products)
        for label, condition in (
            ("same lead", treatment_lead == reference_lead),
            (f"{products[0]} shorter", treatment_lead < reference_lead),
            (f"{products[0]} longer", treatment_lead > reference_lead),
        ):
            rows = daytime.filter(condition)
            if rows.height:
                lines.append(
                    _contrast_line(
                        losses=rows,
                        treatment=treatment,
                        reference=reference,
                        label=f"{label}, {first:02d}–{last:02d} UTC",
                    )
                )
    if waiting:
        lines += [
            "",
            (
                "Not split, because a product has no entry in RUN_INTERVAL_HOURS (a retrieval, a "
                "reanalysis, UKV's analysis, or a run interval not yet measured): "
                f"{', '.join(waiting)}."
            ),
        ]
    return lines


def _sarah_era_lines(*, losses: pl.DataFrame) -> list[str]:
    """Report SARAH-3's error against CAMS in each span of `SARAH_SATELLITE_ERAS`.

    A span of fewer than `studies.bootstrap.MIN_MONTHS_FOR_INTERVAL` months is listed under the
    table with its estimate and no interval, so the table holds only rows with an interval.

    Args:
        losses: The pooled losses, holding `sarah3_global` and `cams_global`.

    Returns:
        Markdown lines: the table, then any span too short for an interval.
    """
    lines = [
        "#### SARAH-3 against CAMS, by the satellite behind SARAH-3 (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    short: list[str] = []
    for label, start, end in SARAH_SATELLITE_ERAS:
        rows = losses.filter(pl.col("time").is_between(start, end, closed="left"))
        if not rows.height:
            continue
        months = rows["month"].n_unique()
        if months >= MIN_MONTHS_FOR_INTERVAL:
            lines.append(
                _contrast_line(
                    losses=rows, treatment="sarah3_global", reference="cams_global", label=label
                )
            )
            continue
        difference = _mae(losses=rows, arm="sarah3_global") - _mae(losses=rows, arm="cams_global")
        short.append(
            f"- {label}: sarah3_global − cams_global {difference:+.3f} on {months} months, too "
            "few for an interval."
        )
    if short:
        lines += ["", *short]
    return lines


RAW_IRRADIANCE_PRODUCTS: Final[tuple[tuple[str, str], ...]] = (
    ("sarah3", "SARAH-3"),
    ("icon_d2", "ICON-D2"),
    ("icon_eu", "ICON-EU"),
    ("icon_global", "ICON global"),
    ("icon_dream", "ICON-DREAM-EU"),
    ("ukv", "UKV"),
    ("era5", "ERA5"),
)
"""The products `_raw_irradiance_vs_cams_lines` compares against CAMS, as (arm prefix, display
name). UKV is its Open-Meteo hourly value as scored elsewhere on this page, not rebuilt from its
snapshots.
"""


def _raw_irradiance_vs_cams_lines(*, frame: pl.DataFrame) -> list[str]:
    """Report each product's raw global irradiance against CAMS's, with no power model involved.

    Every forecast contrast on this page passes each product's irradiance through an XGBoost model
    fitted per generator, which recalibrates a steady bias away. This table instead compares the
    served irradiance values directly, row for row, on the daylight hours `frame` holds (`frame` is
    already the `long` panel's common rows, so every listed product's column is present on every
    row). It exists to check whether the raw irradiance already ranks the products the way the
    power-model contrasts do, or whether the model is doing the ranking.

    Args:
        frame: The panel's common rows, holding `ghi_cams` and each product's own `ghi_<product>`.

    Returns:
        Markdown lines: one row per product.
    """
    daylight = frame.filter(pl.col("solar_elevation_deg") > 0.0)
    lines = [
        "#### Raw global irradiance against CAMS's, before any power model (exploratory)",
        "",
        (
            f"On the {daylight.height:,} daylight generator-hours every listed product shares "
            "with CAMS. Positive bias: the product reads higher than CAMS."
        ),
        "",
        "| Product | Bias (W/m²) | Mean absolute difference (W/m²) | Correlation with CAMS |",
        "|---|---|---|---|",
    ]
    for product, name in RAW_IRRADIANCE_PRODUCTS:
        comparison = raw_column_comparison(
            frame=daylight, treatment=_named("ghi_w_m2", product), reference="ghi_cams"
        )
        lines.append(
            f"| {name} | {comparison['bias']:+.2f} | {comparison['mad']:.2f} "
            f"| {comparison['correlation']:.3f} |"
        )
    return lines


CLEARNESS_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("overcast kt<0.3", 0.0, 0.3),
    ("broken 0.3-0.6", 0.3, 0.6),
    ("clear kt>=0.6", 0.6, 2.0),
)
"""CAMS's clearness index `kt` (global irradiance over extraterrestrial), bucketed into overcast,
broken-cloud, and clear skies, for `_sarah_cams_breakdown_lines`.
"""


def _sarah_cams_breakdown_lines(*, frame: pl.DataFrame, losses: pl.DataFrame) -> list[str]:
    """Report SARAH-3's error against CAMS, per generator and by CAMS's clearness index.

    Neither breakdown is a planned contrast, so both are exploratory. The clearness index `kt` is
    CAMS's own global irradiance over the extraterrestrial irradiance at the same site and time,
    which does not depend on the weather product read for the metric.

    Args:
        frame: The panel's common rows, holding `ghi_cams` and `extraterrestrial_horizontal_w_m2`.
        losses: The pooled losses, holding `sarah3_global` and `cams_global`.

    Returns:
        Markdown lines: one table per generator, then one per clearness band.
    """
    kt = frame.select(
        "site",
        "time",
        kt=pl.when(pl.col("extraterrestrial_horizontal_w_m2") > 0)
        .then(pl.col("ghi_cams") / pl.col("extraterrestrial_horizontal_w_m2"))
        .otherwise(None),
    )
    keyed = losses.join(kt, on=["site", "time"], how="left")
    lines = ["#### SARAH-3 against CAMS, by generator (exploratory)", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=keyed.filter(pl.col("site") == site),
            treatment="sarah3_global",
            reference="cams_global",
            label=f"site {site}",
        )
        for site in sorted(keyed["site"].unique().to_list())
    ]
    clearness_heading = (
        "#### SARAH-3 against CAMS, by CAMS's clearness index (exploratory, chosen after the "
        "results were seen)"
    )
    lines += ["", clearness_heading, "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=keyed.filter(pl.col("kt").is_between(low, high, closed="left")),
            treatment="sarah3_global",
            reference="cams_global",
            label=name,
        )
        for name, low, high in CLEARNESS_BANDS
    ]
    return lines


def _icon_dream_icon_eu_by_year_lines(*, losses: pl.DataFrame) -> list[str]:
    """Report ICON-DREAM-EU's error minus ICON-EU's, in each calendar year (exploratory).

    Shows whether ICON-DREAM-EU's gap to ICON-EU drifts from year to year, so a reader who sees
    every product's lead over ERA5 shrink in 2025 and 2026 can check whether ICON-DREAM-EU is
    getting worse relative to a weather model rather than only relative to ERA5's reanalysis.

    Args:
        losses: The pooled losses, holding `icon_dream_global` and `icon_eu_global`.

    Returns:
        Markdown lines: one row per year. Positive means ICON-DREAM-EU's error is the larger.
    """
    by_year = bootstrap_difference_by_year(
        losses=losses,
        treatment="icon_dream_global",
        references=("icon_eu_global",),
        metric=METRIC,
    )
    lines = [
        "#### ICON-DREAM-EU against ICON-EU, by year (exploratory)",
        "",
        (
            "Positive: ICON-DREAM-EU's error is the larger. A year of fewer than "
            f"{MIN_MONTHS_FOR_INTERVAL} months gets no interval."
        ),
        "",
        "| Year | ICON-DREAM-EU − ICON-EU (pp of capacity) | 95% interval | Months | Rows |",
        "|---|---|---|---|---|",
    ]
    for row in by_year:
        difference, lower, upper = (
            row[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
        )
        interval = f"[{lower:+.3f}, {upper:+.3f}]" if row["enough_months"] else "too few months"
        lines.append(
            f"| {row['year']} | {difference:+.3f} | {interval} | {row['n_months']} "
            f"| {row['n_rows']:,} |"
        )
    return lines


def _icon_dream_icon_eu_lead_lines(*, losses: pl.DataFrame) -> list[str]:
    """Compare ICON-DREAM-EU against ICON-EU at matched served leads (exploratory).

    ICON-DREAM-EU and ICON-EU run on the same 3-hourly cycle, so every label hour holds both at the
    same lead, and the contrast within a lead bucket is lead-matched. This isolates whether the
    hourly-mean conversion, which needs more de-averaging at a longer lead, explains part of
    ICON-DREAM-EU's gap to ICON-EU.

    Args:
        losses: The pooled losses, holding `icon_dream_global` and `icon_eu_global`.

    Returns:
        Markdown lines.
    """
    first, last = LEAD_TABLE_HOURS
    daytime = losses.filter(
        pl.col("time").dt.hour().cast(pl.Int32).is_between(first, last)
    ).with_columns(lead_3h=_served_lead(product="icon_dream"))
    label = f"{first:02d}–{last:02d} UTC"
    lines = [
        "#### ICON-DREAM-EU against ICON-EU at matched served leads (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=daytime.filter(pl.col("lead_3h") == lead),
            treatment="icon_dream_global",
            reference="icon_eu_global",
            label=f"both at lead {lead} h, {label}",
        )
        for lead in (1, 2, 3)
    ]
    return lines


def _lead_tables(*, losses: pl.DataFrame) -> list[str]:
    """Compare ICON products at matched served leads and hour by hour, and break CAMS down.

    ICON-D2 and ICON-EU run on the same 3-hourly cycle, so every label hour holds them at the same
    lead and their contrast within a lead bucket is lead-matched. ICON global's lead equals
    ICON-EU's wherever its own lead is 3 hours or less. The per-hour table carries the time of day
    as well as the lead, which is why it is listed rather than bucketed.

    Args:
        losses: The pooled losses.

    Returns:
        Markdown lines.
    """
    first, last = LEAD_TABLE_HOURS
    hour = pl.col("time").dt.hour().cast(pl.Int32)
    daytime = losses.filter(hour.is_between(first, last)).with_columns(
        lead_3h=_served_lead(product="icon_eu"), lead_6h=_served_lead(product="icon_global")
    )
    label = f"{first:02d}–{last:02d} UTC"
    lines = ["#### ICON-D2 against ICON-EU at matched served leads", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=daytime.filter(pl.col("lead_3h") == lead),
            treatment="icon_d2_global",
            reference="icon_eu_global",
            label=f"both at lead {lead} h, {label}",
        )
        for lead in (1, 2, 3)
    ]
    lines += [
        "",
        "#### ICON-EU against UKV rebuilt from its snapshots, by ICON-EU's served lead (post hoc)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=daytime.filter(pl.col("lead_3h") == lead),
            treatment="icon_eu_global",
            reference="ukv_trap_global",
            label=f"ICON-EU at lead {lead} h, {label}",
        )
        for lead in (1, 2, 3)
    ]
    lines += ["", "#### ICON global against ICON-EU, split by ICON global's lead", ""]
    lines += [*CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=daytime.filter(condition),
            treatment="icon_global_global",
            reference="icon_eu_global",
            label=f"{name}, {label}",
        )
        for name, condition in (
            ("ICON global lead 1 to 3 h, equal to ICON-EU's", pl.col("lead_6h") <= 3),
            ("ICON global lead 4 to 6 h", pl.col("lead_6h") > 3),
        )
    ]
    by_hour = (
        daytime.filter(pl.col("arm").is_in(["icon_d2_global", "icon_eu_global"]))
        .pivot(on="arm", index=["time", "site", "seed"], values=METRIC)
        .group_by(hour.alias("hour"))
        .agg(difference=(pl.col("icon_d2_global") - pl.col("icon_eu_global")).mean())
        .sort("hour")
    )
    lines += [
        "",
        "#### ICON-D2 against ICON-EU, hour by hour (pp of capacity; point estimates)",
        "",
        "| Hour (UTC) | Served lead of both | ICON-D2 − ICON-EU |",
        "|---|---|---|",
    ]
    lines += [
        f"| {row['hour']:02d} | {((row['hour'] - 1) % 3) + 1} h "
        f"| {row['difference'] * PERCENTAGE_POINTS:+.3f} |"
        for row in by_hour.iter_rows(named=True)
    ]
    lines += [
        "",
        "#### ICON-D2 against ICON-EU at each hour, with intervals (post hoc)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=daytime.filter(hour == label_hour),
            treatment="icon_d2_global",
            reference="icon_eu_global",
            label=f"hour {label_hour:02d} UTC",
        )
        for label_hour in range(first, last + 1)
    ]
    lines += ["", "#### CAMS against ICON-D2, broken down", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=daytime.filter(pl.col("lead_3h") == 1),
            treatment="cams_global",
            reference="icon_d2_global",
            label=f"ICON-D2 at lead 1 h, {label}",
        )
    ]
    breakdowns = {
        "site": pl.col("site"),
        "season": pl.col("time").dt.month().replace_strict(SEASONS, return_dtype=pl.Utf8),
        "year": pl.col("time").dt.year().cast(pl.Utf8),
    }
    for name, key in breakdowns.items():
        keyed = losses.with_columns(group=key)
        lines += [
            _contrast_line(
                losses=keyed.filter(pl.col("group") == group),
                treatment="cams_global",
                reference="icon_d2_global",
                label=f"{name} {group}",
            )
            for group in sorted(keyed["group"].unique().to_list())
        ]
    return lines


def era5_difference_by_year(
    *,
    losses: pl.DataFrame,
    era5_arm: str,
    other_arms: tuple[str, ...],
    months: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Return ERA5's error minus each other arm's, in each calendar year, with its interval.

    Each year is bootstrapped on its own months alone, by
    `studies.bootstrap.bootstrap_difference_by_year`, which also flags a year holding too few
    months for its interval to mean anything.

    Args:
        losses: Per-row losses at one setting, carrying `time`, `month`, `site`, `seed` and `arm`.
        era5_arm: ERA5's arm, the treatment in every difference.
        other_arms: The arms ERA5 is compared against.
        months: When given, restricts every year to these calendar months (1-12) before
            resampling, so a partial year and a full year compare on the same months. `None` keeps
            every month a year holds.

    Returns:
        One row per (arm, year) with `arm`, `year`, the interval in fractions of capacity, and
        `enough_months`. A positive difference means ERA5's error is the larger.
    """
    intervals: list[YearInterval] = bootstrap_difference_by_year(
        losses=losses, treatment=era5_arm, references=other_arms, metric=METRIC, months=months
    )
    return pl.DataFrame(intervals).rename({"reference": "arm"})


def era5_by_year_lines(*, by_year: pl.DataFrame, months_note: str | None = None) -> list[str]:
    """Render `era5_difference_by_year`'s table as markdown.

    A year holding fewer than `studies.bootstrap.MIN_MONTHS_FOR_INTERVAL` months shows its estimate
    and no interval: with the month as the resampling unit, its interval would reflect little more
    than the fitting seed.

    Args:
        by_year: The output of `era5_difference_by_year`.
        months_note: When `by_year` was built with a `months` restriction, a short clause naming
            it, appended to the caption (e.g. "on January to September of each year"). `None` for
            an unrestricted, full-calendar-year table.

    Returns:
        Markdown lines: one row per (arm, year).
    """
    caption = "Positive: ERA5's error is the larger. A year of fewer than "
    caption += f"{MIN_MONTHS_FOR_INTERVAL} months gets no interval."
    if months_note is not None:
        caption += f" Every year is restricted to the same months, {months_note}."
    lines = [
        "#### ERA5 against every other product, year by year (exploratory)",
        "",
        caption,
        "",
        (
            "| Against | Year | ERA5 − product (pp of capacity) | 95% interval | Excludes zero? "
            "| Months | Rows |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in by_year.iter_rows(named=True):
        difference, lower, upper = (
            row[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
        )
        if row["enough_months"]:
            interval = f"[{lower:+.3f}, {upper:+.3f}]"
            excludes = row["lower_95"] > 0.0 or row["upper_95"] < 0.0
            verdict = "**yes**" if excludes else "no"
        else:
            interval, verdict = "too few months", "—"
        lines.append(
            f"| {row['arm']} | {row['year']} | {difference:+.3f} | {interval} | {verdict} "
            f"| {row['n_months']} | {row['n_rows']:,} |"
        )
    return lines


class PanelLosses(NamedTuple):
    """Every set of losses one panel's report reads.

    Attributes:
        pooled: Every arm at the main hyperparameter setting, on the common rows.
        sensitivity: The planned contrasts' arms at the second setting; empty where none ran.
        post_only: The global arms fitted on the post-upgrade rows alone, or `None` where the panel
            runs no full analysis.
        transfer: The leave-one-site-out losses, or `None` likewise.
    """

    pooled: pl.DataFrame
    sensitivity: pl.DataFrame
    post_only: pl.DataFrame | None
    transfer: pl.DataFrame | None


def _leaderboard_lines(*, panel: Panel, losses: PanelLosses) -> list[str]:
    """Return every product's error in each arm, and every arm's feature columns.

    Args:
        panel: The panel reported.
        losses: The panel's losses.

    Returns:
        Markdown lines.
    """
    transfer = losses.transfer
    header = "| Product | Served lead | Global only | Own split | Erbs on own global |"
    rule = "|---|---|---|---|---|"
    if transfer is not None:
        header += " Leave one site out |"
        rule += "---|"
    lines = [header, rule]
    for product in panel.products:
        cells = [f"{_mae(losses=losses.pooled, arm=f'{product}_global'):.3f}"]
        cells += [
            "—" if product in UNUSABLE_SPLITS else f"{_mae(losses=losses.pooled, arm=arm):.3f}"
            for arm in (f"{product}_split", f"{product}_erbs")
        ]
        if transfer is not None:
            cells.append(f"{_mae(losses=transfer, arm=f'{product}_global'):.3f}")
        lines.append(f"| {product} | {SERVED_LEAD[product]} | " + " | ".join(cells) + " |")
    lines += ["", "Mean absolute error as a percentage of each site's P99 output.", ""]
    lines += ["#### Every arm's feature columns, beyond the shared features and the era", ""]
    lines += [
        f"- `{arm}`: {', '.join(f'`{column}`' for column in columns)}"
        for arm, columns in _arm_columns(
            products=panel.products, with_snapshot_arms=panel.full_analysis
        ).items()
    ]
    return lines


def _full_analysis_lines(*, panel: Panel, losses: PanelLosses) -> list[str]:
    """Return the first write-up's analyses beyond the leaderboard, which need UKV and ICON.

    Args:
        panel: The panel reported, one with `full_analysis`.
        losses: The panel's losses, with `post_only` and `transfer`.

    Returns:
        Markdown lines.

    Raises:
        ValueError: If the post-upgrade or the leave-one-site-out losses are missing.
    """
    if losses.post_only is None or losses.transfer is None:
        msg = "a full analysis needs the post-upgrade and the leave-one-site-out losses"
        raise ValueError(msg)
    pooled = losses.pooled
    lines = [
        "#### The post scope, fitted on post-upgrade rows alone",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=losses.post_only, treatment=treatment, reference=reference, label="post"
        )
        for treatment, reference in (*panel.planned, ("ukv_global", "era5_global"))
    ]
    lines += [
        "",
        "#### Leave one site out, the scored months withheld everywhere: the planned contrasts",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(
            losses=losses.transfer, treatment=treatment, reference=reference, label="all"
        )
        for treatment, reference in panel.planned
    ]
    lines += [
        "",
        "#### UKV's served hour against the mean of its own two snapshots (post hoc)",
        "",
        *CONTRAST_HEADER,
    ]
    lines += [
        _contrast_line(losses=scoped, treatment=treatment, reference=reference, label=scope)
        for scope in ("all", "ukv_live", "post")
        for scoped in (_scope(losses=pooled, scope=scope),)
        for treatment, reference in (
            ("ukv_trap_global", "ukv_global"),
            ("icon_eu_global", "ukv_trap_global"),
            ("icon_eu_global", "ukv_pair_global"),
            ("icon_eu_ctx_global", "icon_eu_global"),
            ("ukv_trap_ctx_global", "ukv_trap_global"),
            ("icon_eu_ctx_global", "ukv_pair_global"),
            ("icon_eu_ctx_global", "ukv_trap_ctx_global"),
            ("ukv_trap_global", "era5_global"),
        )
    ]
    lines += [
        "",
        (
            f"MAE: ukv_trap_global {_mae(losses=pooled, arm='ukv_trap_global'):.3f}, "
            f"ukv_pair_global {_mae(losses=pooled, arm='ukv_pair_global'):.3f}."
        ),
    ]
    return lines


def _report(
    *,
    name: PanelType,
    panel: Panel,
    frame: pl.DataFrame,
    losses: PanelLosses,
    by_year: pl.DataFrame,
) -> str:
    """Assemble one panel's markdown report.

    Args:
        name: The panel's name.
        panel: The panel reported.
        frame: The panel's common rows.
        losses: The panel's losses.
        by_year: The output of `era5_difference_by_year`.

    Returns:
        The report.
    """
    pooled = losses.pooled
    lines = [
        (
            f"### The {name} panel: {len(panel.products)} weather products on {frame.height:,} "
            f"common site-hours ({frame['time'].min():%Y-%m-%d} to "
            f"{frame['time'].max():%Y-%m-%d})"
        ),
        "",
        *_leaderboard_lines(panel=panel, losses=losses),
    ]
    if panel.planned:
        lines += ["", "#### Planned contrasts, named before the run", "", *CONTRAST_HEADER]
        lines += [
            _contrast_line(losses=pooled, treatment=treatment, reference=reference, label="all")
            for treatment, reference in panel.planned
        ]
    if losses.sensitivity.height > 0:
        lines += [
            "",
            "#### Planned contrasts at the second hyperparameter setting",
            "",
            *CONTRAST_HEADER,
        ]
        lines += [
            _contrast_line(
                losses=losses.sensitivity,
                treatment=treatment,
                reference=reference,
                label="sensitivity",
            )
            for treatment, reference in panel.planned
        ]
    scopes = SCOPES if panel.full_analysis else ("all",)
    lines += ["", "#### Every product against ERA5, by scope (exploratory)", "", *CONTRAST_HEADER]
    for scope in scopes:
        scoped = _scope(losses=pooled, scope=scope)
        lines += [
            _contrast_line(
                losses=scoped, treatment=f"{product}_global", reference="era5_global", label=scope
            )
            for product in panel.products
            if product != BASE_PRODUCT
        ]
    if panel.full_analysis:
        post_months = _scope(losses=pooled, scope="post")["month"].n_unique()
        lines += [
            "",
            (
                f"The post scope holds {post_months} months, so its intervals rest on "
                f"{post_months} clusters and under-cover; read its fold-sign counts alongside them."
            ),
        ]
    lines += ["", "#### A product's own split against Erbs on its own global", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(
            losses=pooled, treatment=f"{product}_split", reference=f"{product}_erbs", label="all"
        )
        for product in panel.products
        if product not in UNUSABLE_SPLITS
    ]
    if panel.full_analysis:
        lines += ["", *_full_analysis_lines(panel=panel, losses=losses)]
    log_by_product = _log_capacity_by_month(frame=frame, products=panel.products)
    lines += [
        "",
        "#### Implied capacity: month-to-month spread and seasonal swing",
        "",
        *_implied_capacity(log_by_product=log_by_product),
        "",
        "#### Implied capacity by calendar month against the annual mean (%)",
        "",
        *_implied_capacity_by_month(log_by_product=log_by_product),
    ]
    if panel.full_analysis:
        lines += ["", *_lead_tables(losses=pooled)]
    if panel.planned:
        lines += ["", *_matched_lead_lines(panel=panel, losses=pooled)]
    if {"sarah3", "cams"} <= set(panel.products):
        lines += ["", *_sarah_era_lines(losses=pooled)]
        lines += ["", *_sarah_cams_breakdown_lines(frame=frame, losses=pooled)]
    if {"icon_dream", "icon_eu"} <= set(panel.products):
        lines += ["", *_icon_dream_icon_eu_by_year_lines(losses=pooled)]
        lines += ["", *_icon_dream_icon_eu_lead_lines(losses=pooled)]
    if {product for product, _ in RAW_IRRADIANCE_PRODUCTS} <= set(panel.products):
        lines += ["", *_raw_irradiance_vs_cams_lines(frame=frame)]
    lines += ["", *era5_by_year_lines(by_year=by_year)]
    lines += ["", *geometry_lines(sites=_pv_sites(), noun="solar farms")]
    return "\n".join(lines) + "\n"


def _panel_jobs(*, name: PanelType, panel: Panel) -> list[Job]:
    """Return every pooled fit a panel runs: its arms, and its planned contrasts' second setting.

    Args:
        name: The panel's name; the `published` panel reproduces the first round, which ran no
            second setting.
        panel: The panel.

    Returns:
        The jobs.

    Raises:
        ValueError: If a panel other than `published` names planned contrasts but no second-setting
            job, which would leave its report without the sensitivity table.
    """
    extra = sensitivity_jobs(panel=panel) if name != "published" else []
    if name != "published" and panel.planned and not extra:
        msg = f"the {name} panel plans contrasts but fits none at the second setting"
        raise ValueError(msg)
    return jobs(products=panel.products, with_snapshot_arms=panel.full_analysis) + extra


def _panel_frame(*, panel: Panel, panel_jobs: list[Job]) -> pl.DataFrame:
    """Build a panel's common rows, with the era, the folds, the time features and the export cap.

    Every column any job is shown is checked for a missing value, because the inner joins that make
    the rows common are what keep every arm's input present on every row.

    Args:
        panel: The panel.
        panel_jobs: Every job the panel fits, whose feature columns are checked.

    Returns:
        The rows every arm of the panel is fitted and scored on.
    """
    rows = common_rows(frame=joined(products=panel.products))
    if panel.first_time is not None:
        rows = rows.filter(pl.col("time") >= panel.first_time)
    frame = with_export_cap(dataset=with_eras(frame=_add_time_features(dataset=rows)))
    check_no_missing(frame=frame, columns=[column for job in panel_jobs for column in job[3]])
    return frame


def _fit_panel(
    *, panel: Panel, panel_jobs: list[Job], frame: pl.DataFrame, max_workers: int
) -> PanelLosses:
    """Fit every arm a panel reports.

    Args:
        panel: The panel.
        panel_jobs: The pooled fits, from `_panel_jobs`.
        frame: The panel's common rows.
        max_workers: How many fits run at once.

    Returns:
        The panel's losses.
    """
    losses = run_all(dataset=frame, jobs=panel_jobs, max_workers=max_workers)
    return PanelLosses(
        pooled=losses.filter(pl.col("setting") == "pooled"),
        sensitivity=losses.filter(pl.col("setting") == "sensitivity"),
        post_only=(
            _post_only_losses(frame=frame, products=panel.products, max_workers=max_workers)
            if panel.full_analysis
            else None
        ),
        transfer=(
            _leave_one_site_out_losses(
                frame=frame, products=panel.products, max_workers=max_workers
            )
            if panel.full_analysis
            else None
        ),
    )


def run_panel(
    *, name: PanelType, report_only: bool, max_workers: int = MAX_CONCURRENT_FITS
) -> None:
    """Fit one panel, or read its saved losses, and write its report and year-by-year table.

    Args:
        name: The panel to run.
        report_only: Whether to read the losses a full run saved instead of fitting.
        max_workers: How many fits run at once.
    """
    panel = PANELS[name]
    panel_jobs = _panel_jobs(name=name, panel=panel)
    frame = _panel_frame(panel=panel, panel_jobs=panel_jobs)
    by_site = frame.group_by("site", "era").agg(pl.len(), pl.col("month").n_unique()).sort("site")
    _LOG.info("%s panel common rows: %d\n%s", name, frame.height, by_site)

    output_dir = panel.output_dir
    loss_paths = {
        key: output_dir / f"{key}.parquet"
        for key in ("losses", "post_only_losses", "leave_one_site_out_losses")
    }
    written = [output_dir / "report.md", output_dir / "era5_by_year.parquet"]
    if report_only:
        refuse_to_overwrite(paths=written)
        saved = pl.read_parquet(loss_paths["losses"])
        losses = PanelLosses(
            pooled=saved.filter(pl.col("setting") == "pooled"),
            sensitivity=saved.filter(pl.col("setting") == "sensitivity"),
            post_only=(
                pl.read_parquet(loss_paths["post_only_losses"]) if panel.full_analysis else None
            ),
            transfer=(
                pl.read_parquet(loss_paths["leave_one_site_out_losses"])
                if panel.full_analysis
                else None
            ),
        )
    else:
        refuse_to_overwrite(paths=[*loss_paths.values(), *written])
        output_dir.mkdir(parents=True, exist_ok=True)
        losses = _fit_panel(
            panel=panel, panel_jobs=panel_jobs, frame=frame, max_workers=max_workers
        )
        pl.concat([losses.pooled, losses.sensitivity]).write_parquet(loss_paths["losses"])
        if losses.post_only is not None:
            losses.post_only.write_parquet(loss_paths["post_only_losses"])
        if losses.transfer is not None:
            losses.transfer.write_parquet(loss_paths["leave_one_site_out_losses"])

    by_year = era5_difference_by_year(
        losses=losses.pooled,
        era5_arm=f"{BASE_PRODUCT}_global",
        other_arms=tuple(f"{p}_global" for p in panel.products if p != BASE_PRODUCT),
    )
    report = _report(name=name, panel=panel, frame=frame, losses=losses, by_year=by_year)
    by_year.write_parquet(output_dir / "era5_by_year.parquet")
    (output_dir / "report.md").write_text(report)
    sys.stdout.write(report)


def main() -> int:
    """Run every panel named on the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        nargs="+",
        choices=tuple(PANELS),
        default=list(DEFAULT_PANELS),
        help="The panels to run.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from the losses already on disk instead of refitting.",
    )
    parser.add_argument(
        "--concurrent-fits",
        type=int,
        default=MAX_CONCURRENT_FITS,
        help="How many fits run at once, each on 4 cores; lower it to share the machine.",
    )
    arguments = parser.parse_args()
    for name in arguments.panel:
        run_panel(
            name=name, report_only=arguments.report_only, max_workers=arguments.concurrent_fits
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
