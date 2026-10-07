"""Does a reanalysis describe past wind better than the five products on the wind leaderboard?

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/968>. The plan is
`plans/wind-cerra-nora3.md`. One script fits either reanalysis: `--product cerra` for CERRA, a
3-hourly analysis, or `--product nora3` for NORA3, an hourly one. Each product has its own row
set, its own output folder and its own report.

**Row set.** The main wind study's common rows (`wind_product_frames.common_rows`, from 12 August
2024), inner-joined to the reanalysis's wind at each farm's nearest cell, up to 30 June 2026 for
CERRA and 31 August 2026 for NORA3. CERRA has one analysis every 3 hours, so its row set holds
one main-study hour in three, and `hour_of_day` takes 8 values. Every arm is scored on exactly
these rows.

**Arms.** Each block refits the five main-study products (ERA5, UKV, ICON-D2, ICON-EU, and ICON
global) on the block's own rows, beside the new product at its 100 m level. Every arm's inputs
are `wind_product_frames.wind_columns` column for column: the speed at the arm's hub height, that
height's direction as a sine and a cosine, the 10 m speed, and the shared hour of day, day of
year and `era_code`. There are no columns for neighbouring hours. Every fit uses
`colsample_bytree=1`, the CPU, and `studies.arm_runner.MAX_CONCURRENT_FITS` fits at once.

**Whether the arms carry direction is decided by a constant, never by which files exist.**
`CERRA_WITH_DIRECTION` is set before any fit. When True, the CERRA block needs the 100 m
direction file and raises if it is missing. An exploratory height (75 m or 150 m) whose direction
file is missing is omitted, and the report names it, so the planned arms' inputs never change.
When False, the block drops the direction pair from every arm and every reference, so the widths
stay equal. No arm borrows ERA5's direction. NORA3 needs its 10 m file, and raises if the file is
absent.

**Folds.** `cerra_past_solar.with_covering_folds` picks the fold rotation that leaves no calendar
month without a training row, and cuts the folds inside each era. Intervals resample whole
calendar months and one of three fitting seeds.

**Planned contrasts** (`PLANNED_CONTRASTS`, written before any fit, each also at the second
hyperparameter setting): the new product at 100 m against ERA5 at 100 m, and the new product
against the leading product of the main leaderboard block (`LEADING_MAIN_PRODUCT`). Every other
contrast is exploratory and labelled so in the report: the new product against each other
main-study product, ICON-D2 against ERA5 on these rows, and the exploratory heights (CERRA at 75
m and 150 m, NORA3 at 50 m) against the new product at 100 m. Every arm is fitted at both
settings, so each exploratory contrast has a second setting too. Comparing the heights of one
product is the question of issue
957. NORA3 against CERRA on their shared site-hours needs the losses of both blocks, and is not
     computed here.

Run it with `uv run python studies/past_weather/reanalysis_past_wind.py --product cerra`.
`--check-only` builds the rows and checks the inputs, the folds and the column widths, then
prints counts and stops without fitting or writing. `--report-only` rebuilds `report.md` from the
saved `losses.parquet`, and still checks the saved fingerprint. A fresh run stops
(`refuse_to_overwrite`) while `losses.parquet`, `losses.fingerprint` or `report.md` exists, until
they are moved to a `superseded/` subfolder. Only one agent may run it at a time, because every
worktree shares one data folder.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple

import polars as pl
from cerra_past_solar import check_column_counts, uncovered_share, with_covering_folds
from ens_past_solar import _absolute_table_lines, _arm_columns_lines, _fingerprint
from studies.arm_runner import MAX_CONCURRENT_FITS, Job, add_time_features, run_all
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    calendar_month_coverage,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.pv_dataset import wind_sites
from studies.reanalysis_wind import (
    CERRA_DIRECTION_FILES,
    NORA3_SURFACE_HEIGHT_M,
    derive_nearest_cells,
    derive_nearest_nora3_cells,
    read_cerra_direction,
    read_cerra_wind,
    read_nora3_wind,
)
from studies.solar_product_frames import with_eras
from studies.sources import (
    CERRA_PRODUCT_DIR,
    NORA3_10M_PRODUCT_DIR,
    NORA3_PRODUCT_DIR,
    STUDY_DATA_DIR,
)
from studies.wind_product_frames import SHARED_FEATURES, common_rows, joined, wind_columns
from weather_products import CONTRAST_HEADER, _contrast_line
from wind_products import geometry_lines

CERRA_DIR: Final[Path] = CERRA_PRODUCT_DIR
CERRA_GRID_PATH: Final[Path] = CERRA_DIR / "cerra_grid.parquet"
NORA3_PATH: Final[Path] = NORA3_PRODUCT_DIR / "NORA3_wind.parquet"
NORA3_10M_PATH: Final[Path] = NORA3_10M_PRODUCT_DIR / "NORA3_wind.parquet"

MAIN_PRODUCTS: Final[tuple[str, ...]] = ("era5", "ukv", "icon_d2", "icon_eu", "icon_global")
"""The five main-study products, each refitted on this script's rows."""

LEADING_MAIN_PRODUCT: Final[str] = "icon_d2"
"""The main leaderboard block's product with the lowest error, the second planned reference."""

HUB_HEIGHT_M: Final[int] = 100
"""The height at which each reanalysis is scored."""

CERRA_WITH_DIRECTION: Final[bool] = True
"""Whether the CERRA block's arms carry the wind direction as a sine and a cosine.

Decided before any fit, because the direction files are being downloaded. When False, every arm
and reference in the CERRA block drops the direction pair. NORA3 always carries direction.
"""


class ProductSpec(NamedTuple):
    """What differs between the two reanalyses.

    Attributes:
        key: The product's name in column names and arm names: `cerra` or `nora3`.
        label: The name the report writes.
        end: The first instant after the product's last hour, in UTC.
        step_hours: The gap between the product's hours: 3 for CERRA and 1 for NORA3.
        extra_heights_m: The heights of the exploratory arms.
    """

    key: str
    label: str
    end: datetime
    step_hours: int
    extra_heights_m: tuple[int, ...]


SPECS: Final[Mapping[str, ProductSpec]] = MappingProxyType(
    {
        "cerra": ProductSpec(
            key="cerra",
            label="CERRA",
            end=datetime(2026, 7, 1, tzinfo=UTC),
            step_hours=3,
            extra_heights_m=(75, 150),
        ),
        "nora3": ProductSpec(
            key="nora3",
            label="NORA3",
            end=datetime(2026, 9, 1, tzinfo=UTC),
            step_hours=1,
            extra_heights_m=(50,),
        ),
    }
)
"""Each product's spec, keyed by the `--product` value."""


def arm_name(*, key: str) -> str:
    """Return the arm that scores one product key.

    Args:
        key: A main product, a new product's `key`, or an exploratory key from `height_key`.

    Returns:
        The arm name.
    """
    return f"{key}_wind"


def height_key(*, spec: ProductSpec, height_m: int) -> str:
    """Return the product key of an exploratory height.

    Args:
        spec: The product.
        height_m: One of the spec's `extra_heights_m`.

    Returns:
        The key, such as `cerra_75m`.
    """
    return f"{spec.key}_{height_m}m"


PLANNED_CONTRASTS: Final[Mapping[str, tuple[tuple[str, str], ...]]] = MappingProxyType(
    {
        key: (
            (arm_name(key=key), arm_name(key="era5")),
            (arm_name(key=key), arm_name(key=LEADING_MAIN_PRODUCT)),
        )
        for key in SPECS
    }
)
"""Each product's two planned contrasts, written before any fit, as (treatment, reference)."""


def exploratory_contrasts(
    *, spec: ProductSpec, extra_heights_m: Sequence[int]
) -> tuple[tuple[str, str], ...]:
    """Return every contrast that is not planned, as (treatment, reference) pairs.

    Args:
        spec: The product.
        extra_heights_m: The exploratory heights the block holds.

    Returns:
        The new product against each main product it is not planned against, ICON-D2 against ERA5,
        and each exploratory height against the new product at 100 m.
    """
    new = arm_name(key=spec.key)
    planned = {reference for _, reference in PLANNED_CONTRASTS[spec.key]}
    pairs = [
        (new, arm_name(key=product))
        for product in MAIN_PRODUCTS
        if arm_name(key=product) not in planned
    ]
    pairs.append((arm_name(key=LEADING_MAIN_PRODUCT), arm_name(key="era5")))
    pairs += [
        (arm_name(key=height_key(spec=spec, height_m=height)), new) for height in extra_heights_m
    ]
    return tuple(pairs)


def with_direction_for(*, spec: ProductSpec) -> bool:
    """Say whether the product's arms carry direction, as decided by `CERRA_WITH_DIRECTION`.

    Args:
        spec: The product.

    Returns:
        `CERRA_WITH_DIRECTION` for CERRA, and True for NORA3, whose direction is in its main file.
    """
    return CERRA_WITH_DIRECTION if spec.key == "cerra" else True


def plan_extra_heights(
    *, spec: ProductSpec, directory: Path, with_direction: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the exploratory heights into those the block can score and those it must omit.

    Only CERRA's direction files are separate downloads. The planned height's direction file is
    required, and an exploratory height whose file is missing is omitted, so the planned arms'
    inputs never depend on the exploratory files.

    Args:
        spec: The product.
        directory: The folder holding CERRA's parquet files.
        with_direction: Whether the block's arms carry direction.

    Returns:
        The exploratory heights to include and the exploratory heights omitted, in metres.

    Raises:
        FileNotFoundError: If `with_direction` is True and CERRA's 100 m direction file is missing.
    """
    if spec.key != "cerra" or not with_direction:
        return spec.extra_heights_m, ()
    planned = directory / CERRA_DIRECTION_FILES[HUB_HEIGHT_M]
    if not planned.exists():
        msg = (
            f"CERRA_WITH_DIRECTION is True but the {HUB_HEIGHT_M} m direction file {planned.name} "
            "has not been downloaded; download it, or set the constant to False before any fit"
        )
        raise FileNotFoundError(msg)
    present = tuple(
        h for h in spec.extra_heights_m if (directory / CERRA_DIRECTION_FILES[h]).exists()
    )
    omitted = tuple(h for h in spec.extra_heights_m if h not in present)
    return present, omitted


def arm_features(
    *, spec: ProductSpec, with_direction: bool, extra_heights_m: Sequence[int]
) -> dict[str, tuple[str, ...]]:
    """Return every arm's feature columns, from one place, in the order the model sees them.

    Args:
        spec: The product.
        with_direction: Whether the arms carry the direction sine and cosine. When False, every arm
            in the block drops the pair, so the widths stay equal.
        extra_heights_m: The exploratory heights to add an arm for.

    Returns:
        Arm name to feature columns.
    """
    keys = [*MAIN_PRODUCTS, spec.key]
    keys += [height_key(spec=spec, height_m=height) for height in extra_heights_m]
    features: dict[str, tuple[str, ...]] = {}
    for key in keys:
        speed, sine, cosine, surface = wind_columns(product=key)
        wind = (speed, sine, cosine, surface) if with_direction else (speed, surface)
        features[arm_name(key=key)] = (*SHARED_FEATURES, *wind)
    return features


def check_block_widths(*, features: Mapping[str, Sequence[str]]) -> None:
    """Raise unless every arm in the block has the same number of distinct columns.

    Args:
        features: Arm name to feature columns.

    Raises:
        ValueError: If an arm repeats a column or two arms differ in width.
    """
    arms = list(features)
    check_column_counts(features=features, contrasts=[(arm, arms[0]) for arm in arms[1:]])


def jobs(*, features: Mapping[str, tuple[str, ...]]) -> list[Job]:
    """Return every arm at both hyperparameter settings.

    Args:
        features: Arm name to feature columns, from `arm_features`.

    Returns:
        One job per arm and setting, all with column subsampling absent from the settings and no
        quantile scoring.
    """
    check_block_widths(features=features)
    return [
        (arm, setting, "power_mw", columns, hyper_parameters, False)
        for setting, hyper_parameters in (
            ("pooled", PRIMARY_HYPER_PARAMETERS),
            ("sensitivity", SENSITIVITY_HYPER_PARAMETERS),
        )
        for arm, columns in features.items()
    ]


def product_wind_columns(
    *, raw: pl.DataFrame, spec: ProductSpec, with_direction: bool, extra_heights_m: Sequence[int]
) -> pl.DataFrame:
    """Rename a reanalysis's wind to the main study's column names, one column set per arm.

    Args:
        raw: `site`, `time`, `wind_speed_10m`, and for each scored height `h` its `wind_speed_{h}m`
            and, when `with_direction`, `wind_direction_{h}m`.
        spec: The product.
        with_direction: Whether `raw` holds direction columns.
        extra_heights_m: The exploratory heights to build columns for.

    Returns:
        `site`, `time`, and the four `wind_product_frames.wind_columns` per arm (two without
        direction), with the direction as a sine and a cosine.
    """
    heights = {spec.key: HUB_HEIGHT_M}
    heights |= {height_key(spec=spec, height_m=h): h for h in extra_heights_m}
    columns = [pl.col("site"), pl.col("time")]
    for key, height in heights.items():
        speed, sine, cosine, surface = wind_columns(product=key)
        columns += [
            pl.col(f"wind_speed_{height}m").alias(speed),
            pl.col("wind_speed_10m").alias(surface),
        ]
        if with_direction:
            direction = pl.col(f"wind_direction_{height}m").radians()
            columns += [direction.sin().alias(sine), direction.cos().alias(cosine)]
    return raw.select(columns)


def select_product_hours(
    *, base: pl.DataFrame, wind: pl.DataFrame, spec: ProductSpec
) -> pl.DataFrame:
    """Keep the main rows that the product has a value for, up to the product's last hour.

    CERRA has one value in 3 hours, so this keeps one main-study hour in three.

    Args:
        base: The main study's common rows.
        wind: `site`, `time` and the product's columns.
        spec: The product.

    Returns:
        The main rows joined to the product's columns on `(site, time)`.

    Raises:
        ValueError: If the product holds a value off its own step, such as a CERRA hour that is
            not a multiple of 3 UTC.
    """
    within = wind.filter(pl.col("time") < spec.end)
    off_step = within.filter(
        (pl.col("time").dt.hour() % spec.step_hours != 0)
        | (pl.col("time").dt.minute() != 0)
        | (pl.col("time").dt.second() != 0)
    ).height
    if off_step:
        msg = f"{off_step} {spec.label} values are not on its {spec.step_hours}-hourly step"
        raise ValueError(msg)
    return base.join(within, on=["site", "time"], how="inner")


class AssembledRows(NamedTuple):
    """The row set `assemble_rows` returns, and the counts the report prints about how it was cut.

    Attributes:
        frame: One row per (site, time), with every arm's columns, `era_code` and `fold`.
        main_rows: The main study's common rows before the join.
        fold_offsets: The rotation of each era's folds.
        uncovered_main_folds: The share of rows in calendar months with no training row under the
            main study's fold recipe applied to these rows.
        uncovered_fitted: The share of rows in a calendar month seen in two or more years that has
            no training row, under the folds the block is fitted on (the wind page's avoidable
            share).
        one_year_only_fitted: The share of rows in a calendar month seen in one year only, which no
            fold design can cover, under the folds the block is fitted on.
    """

    frame: pl.DataFrame
    main_rows: int
    fold_offsets: Mapping[int, int]
    uncovered_main_folds: float
    uncovered_fitted: float
    one_year_only_fitted: float


def assemble_rows(
    *,
    base: pl.DataFrame,
    wind: pl.DataFrame,
    spec: ProductSpec,
    features: Mapping[str, tuple[str, ...]],
) -> AssembledRows:
    """Join the product onto the main rows, cut covering folds, and check no arm has a gap.

    Args:
        base: The main study's common rows with `hour_of_day`, `day_of_year` and `month`.
        wind: `site`, `time` and the product's columns from `product_wind_columns`.
        spec: The product.
        features: Arm name to feature columns.

    Returns:
        The row set and the counts the report prints.

    Raises:
        ValueError: If a product value is off its step, no fold design covers every calendar month,
            or an arm's column holds a missing value.
    """
    kept = select_product_hours(base=base, wind=wind, spec=spec)
    uncovered_main = uncovered_share(frame=with_eras(frame=kept))
    frame, offsets = with_covering_folds(frame=kept)
    check_no_missing(
        frame=frame,
        columns=["power_mw", *(column for columns in features.values() for column in columns)],
    )
    coverage = calendar_month_coverage(frame=frame)
    one_year_rows = coverage.filter(pl.col("n_years") == 1)["n_scored"].sum()
    return AssembledRows(
        frame=frame,
        main_rows=base.height,
        fold_offsets=offsets,
        uncovered_main_folds=uncovered_main,
        uncovered_fitted=uncovered_share(frame=frame),
        one_year_only_fitted=float(one_year_rows) / frame.height,
    )


def read_product(
    *, spec: ProductSpec, sites: pl.DataFrame
) -> tuple[pl.DataFrame, bool, pl.DataFrame, tuple[int, ...], tuple[int, ...]]:
    """Read the product's wind at each site's nearest cell.

    Args:
        spec: The product.
        sites: The wind site list.

    Returns:
        The raw wind, whether it carries direction, the nearest cells' `distance_km`, the
        exploratory heights read, and the exploratory heights omitted.

    Raises:
        FileNotFoundError: If NORA3's 10 m file, or CERRA's 100 m direction file while
            `CERRA_WITH_DIRECTION` is True, has not been downloaded.
    """
    with_direction = with_direction_for(spec=spec)
    extra, omitted = plan_extra_heights(
        spec=spec, directory=CERRA_DIR, with_direction=with_direction
    )
    heights = [HUB_HEIGHT_M, *extra]
    if spec.key == "cerra":
        cells = derive_nearest_cells(grid=pl.read_parquet(CERRA_GRID_PATH), sites=sites)
        raw = read_cerra_wind(directory=CERRA_DIR, cells=cells)
        if with_direction:
            direction = read_cerra_direction(directory=CERRA_DIR, cells=cells, heights=heights)
            raw = raw.join(direction, on=["site", "time"], how="left")
        return raw, with_direction, cells, extra, omitted
    cells = derive_nearest_nora3_cells(sites=sites)
    return read_nora3(cells=cells, heights=heights), with_direction, cells, extra, omitted


def read_nora3(
    *,
    cells: pl.DataFrame,
    heights: Sequence[int],
    path: Path = NORA3_PATH,
    path_10m: Path = NORA3_10M_PATH,
) -> pl.DataFrame:
    """Read NORA3's wind at the scored heights and its 10 m speed.

    Args:
        cells: One row per site, from `derive_nearest_nora3_cells`.
        heights: The scored heights, in metres.
        path: NORA3's main file, holding 50 m and 100 m.
        path_10m: NORA3's 10 m file.

    Returns:
        `site`, `time`, and `wind_speed_{h}m` and `wind_direction_{h}m` for each height, and
        `wind_speed_10m`, on the main file's hours. An hour the 10 m file lacks holds a null, which
        `assemble_rows` rejects.

    Raises:
        FileNotFoundError: If the 10 m file does not exist, because every arm is shown the 10 m
            speed and NORA3 has no other source for it.
    """
    if not path_10m.exists():
        msg = (
            f"NORA3's 10 m file {path_10m.parent.name}/{path_10m.name} has not been downloaded; "
            "every arm needs the 10 m speed, so the NORA3 block cannot run without it"
        )
        raise FileNotFoundError(msg)
    main = read_nora3_wind(path=path, cells=cells, heights=heights)
    surface = read_nora3_wind(path=path_10m, cells=cells, heights=[NORA3_SURFACE_HEIGHT_M]).select(
        "site", "time", "wind_speed_10m"
    )
    return main.join(surface, on=["site", "time"], how="left")


class Built(NamedTuple):
    """The row set and everything the report prints about how it was built.

    Attributes:
        assembled: `assemble_rows`'s result.
        features: Arm name to feature columns.
        with_direction: Whether the arms carry the direction pair.
        distance_range_km: The pooled range of the nearest cells' distances.
        extra_heights_m: The exploratory heights the block holds.
        omitted_arms: The exploratory arms left out because their direction file is missing.
    """

    assembled: AssembledRows
    features: dict[str, tuple[str, ...]]
    with_direction: bool
    distance_range_km: tuple[float, float]
    extra_heights_m: tuple[int, ...]
    omitted_arms: tuple[str, ...]


def build_rows(*, spec: ProductSpec, sites: pl.DataFrame) -> Built:
    """Build the block's row set from the main rows and the reanalysis files.

    Args:
        spec: The product.
        sites: The wind site list.

    Returns:
        The rows, the arms' columns, and the counts the report prints.
    """
    raw, with_direction, cells, extra, omitted = read_product(spec=spec, sites=sites)
    features = arm_features(spec=spec, with_direction=with_direction, extra_heights_m=extra)
    wind = product_wind_columns(
        raw=raw, spec=spec, with_direction=with_direction, extra_heights_m=extra
    )
    base = add_time_features(dataset=common_rows(frame=joined(sites=sites)))
    assembled = assemble_rows(base=base, wind=wind, spec=spec, features=features)
    return Built(
        assembled=assembled,
        features=features,
        with_direction=with_direction,
        distance_range_km=(
            float(cells.select(pl.col("distance_km").min()).item()),
            float(cells.select(pl.col("distance_km").max()).item()),
        ),
        extra_heights_m=extra,
        omitted_arms=tuple(
            arm_name(key=height_key(spec=spec, height_m=height)) for height in omitted
        ),
    )


class OutputPaths(NamedTuple):
    """The three files a fresh run writes.

    Attributes:
        losses: The per-row losses.
        fingerprint: The hash of the rows, jobs, seeds and hyperparameters.
        report: The markdown report.
    """

    losses: Path
    fingerprint: Path
    report: Path


def output_paths(*, spec: ProductSpec) -> OutputPaths:
    """Return one product's write-once output files.

    Args:
        spec: The product.

    Returns:
        The paths, in a folder of the product's own.
    """
    directory = STUDY_DATA_DIR / "past_weather_v2" / f"wind_{spec.key}"
    return OutputPaths(
        losses=directory / "losses.parquet",
        fingerprint=directory / "losses.fingerprint",
        report=directory / "report.md",
    )


def fit_and_save(
    *,
    frame: pl.DataFrame,
    job_list: list[Job],
    fingerprint: str,
    paths: OutputPaths,
    fit: Callable[..., pl.DataFrame] = run_all,
) -> pl.DataFrame:
    """Refuse to overwrite an output, fit every job, and save the losses and the fingerprint.

    Args:
        frame: The row set.
        job_list: Every job to fit.
        fingerprint: The hash `_fingerprint` returned.
        paths: The files to write.
        fit: The fit loop; a test passes a stand-in.

    Returns:
        The losses.

    Raises:
        FileExistsError: If any output file exists, before anything is fitted.
    """
    refuse_to_overwrite(paths=paths)
    paths.losses.parent.mkdir(parents=True, exist_ok=True)
    losses = fit(dataset=frame, jobs=job_list, max_workers=MAX_CONCURRENT_FITS)
    losses.write_parquet(paths.losses)
    paths.fingerprint.write_text(fingerprint)
    return losses


def _row_lines(*, spec: ProductSpec, built: Built) -> list[str]:
    """Render the row counts, pooled ranges and input set as markdown.

    Args:
        spec: The product.
        built: `build_rows`'s result.

    Returns:
        Markdown lines with counts and pooled ranges only.
    """
    assembled = built.assembled
    frame = assembled.frame
    per_site = sorted(frame.group_by("site").len().iter_rows())
    low, high = built.distance_range_km
    direction = (
        "each arm's hub-level direction as a sine and a cosine"
        if built.with_direction
        else (
            "**no direction columns in any arm of this block**, because "
            f"`CERRA_WITH_DIRECTION` is False; the direction pair is absent from every "
            f"{spec.label} arm and reference, and no arm borrows ERA5's direction"
        )
    )
    return [
        "#### Rows and inputs",
        "",
        f"- Main-study common rows before the join: {assembled.main_rows:,}.",
        (
            f"- Rows scored here: {frame.height:,} ({frame['time'].min():%Y-%m-%d} to "
            f"{frame['time'].max():%Y-%m-%d}), on {len(per_site)} sites; per site "
            + ", ".join(f"{site} {count:,}" for site, count in per_site)
            + "."
        ),
        f"- Distinct values of `hour_of_day`: {frame['hour_of_day'].n_unique()}.",
        f"- Nearest-cell distance, pooled: {low:.1f} km to {high:.1f} km.",
        f"- Input set: {spec.label} at {HUB_HEIGHT_M} m, with {direction}.",
        (f"- Fold rotation by era: {dict(assembled.fold_offsets)}."),
        (
            "- Under the folds this block is fitted on, the share of rows in a calendar month seen "
            "in two or more years that has no training row (avoidable): "
            f"{assembled.uncovered_fitted:.1%}; the share in a calendar month seen in one year "
            f"only, which no fold design can cover: {assembled.one_year_only_fitted:.1%}."
        ),
        (
            "- Extra, for comparison only: the share of rows in a calendar month with no training "
            "row under the main study's published fold recipe applied to these rows: "
            f"{assembled.uncovered_main_folds:.1%}."
        ),
        (
            "- Exploratory arms omitted because their direction file is missing: "
            + (", ".join(f"`{arm}`" for arm in built.omitted_arms) or "none")
            + "."
        ),
        f"- Fits: CPU, `colsample_bytree` absent (1), {MAX_CONCURRENT_FITS} fits at once.",
    ]


def _contrast_table(
    *, losses: pl.DataFrame, pairs: Sequence[tuple[str, str]], label: str
) -> list[str]:
    """Render contrasts as one markdown table.

    Args:
        losses: The losses at one setting.
        pairs: (treatment, reference) pairs.
        label: The scope label of every row.

    Returns:
        Markdown lines.
    """
    return [
        *CONTRAST_HEADER,
        *(
            _contrast_line(losses=losses, treatment=treatment, reference=reference, label=label)
            for treatment, reference in pairs
        ),
    ]


def _report(
    *,
    spec: ProductSpec,
    built: Built,
    losses: pl.DataFrame,
    sites: pl.DataFrame,
    job_list: list[Job],
) -> str:
    """Assemble the markdown report.

    Args:
        spec: The product.
        built: `build_rows`'s result.
        losses: Every arm's losses at both settings.
        sites: The wind site list, for the geometry lines.
        job_list: Every job, for the feature-column section.

    Returns:
        The report.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    planned = PLANNED_CONTRASTS[spec.key]
    exploratory = exploratory_contrasts(spec=spec, extra_heights_m=built.extra_heights_m)
    lines = [
        (
            f"### {spec.label} at {HUB_HEIGHT_M} m against the main-study products, on "
            f"{built.assembled.frame.height:,} site-hours of wind"
        ),
        "",
        *_absolute_table_lines(pooled=pooled, arms=tuple(built.features)),
        "",
        (
            "Mean absolute error as a percentage of each site's capacity. The interval is a 95% "
            "bound from resampling whole calendar months and a fitting seed. The two contrasts "
            "under `Planned contrasts` were written into the plan before the first fit; every "
            "other contrast is exploratory."
        ),
        "",
        *_row_lines(spec=spec, built=built),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Planned contrasts",
        "",
        *_contrast_table(losses=pooled, pairs=planned, label="all"),
        "",
        "#### Absolute error at the second hyperparameter setting",
        "",
        *_absolute_table_lines(pooled=sensitivity, arms=tuple(built.features)),
        "",
        "#### Planned contrasts at the second hyperparameter setting",
        "",
        *_contrast_table(losses=sensitivity, pairs=planned, label="sensitivity"),
        "",
        "#### Exploratory contrasts",
        "",
        *_contrast_table(losses=pooled, pairs=exploratory, label="all (exploratory)"),
        "",
        "#### Exploratory contrasts at the second hyperparameter setting",
        "",
        *_contrast_table(losses=sensitivity, pairs=exploratory, label="sensitivity (exploratory)"),
        "",
        *geometry_lines(sites=sites, noun="wind farms"),
    ]
    return "\n".join(lines) + "\n"


def _check_only_lines(
    *, spec: ProductSpec, built: Built, job_list: list[Job], paths: OutputPaths
) -> list[str]:
    """Render what `--check-only` prints: counts and pooled ranges, never a value or a site detail.

    Args:
        spec: The product.
        built: `build_rows`'s result.
        job_list: Every job that would be fitted.
        paths: The output files, only tested for existence.

    Returns:
        Text lines.
    """
    widths = sorted({len(columns) for columns in built.features.values()})
    return [
        (
            f"--check-only: {spec.label} inputs, folds and column widths are valid; nothing was "
            "fitted or written."
        ),
        *_row_lines(spec=spec, built=built),
        f"- Arms: {len(built.features)}; jobs: {len(job_list)}; column widths: {widths}.",
        f"- Output files already present: {sum(path.exists() for path in paths)} of {len(paths)}.",
    ]


def main() -> int:
    """Build the row set, and either check it, fit every arm, or rebuild the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=sorted(SPECS), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs, rows, folds and column widths; fit and write nothing.",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild report.md from the saved losses.parquet alone.",
    )
    arguments = parser.parse_args()
    spec = SPECS[arguments.product]

    sites = wind_sites()
    built = build_rows(spec=spec, sites=sites)
    frame = built.assembled.frame
    paths = output_paths(spec=spec)
    job_list = jobs(features=built.features)

    if arguments.check_only:
        sys.stdout.write(
            "\n".join(_check_only_lines(spec=spec, built=built, job_list=job_list, paths=paths))
            + "\n"
        )
        return 0

    fingerprint = _fingerprint(frame=frame, job_list=job_list)
    if arguments.report_only:
        saved = paths.fingerprint.read_text().strip() if paths.fingerprint.exists() else None
        if saved != fingerprint:
            msg = (
                f"--report-only: {paths.losses} was fitted on a different row set, column set, "
                "seed set, feature values or hyperparameter setting than this code now produces; "
                "re-run without --report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(paths.losses)
    else:
        losses = fit_and_save(frame=frame, job_list=job_list, fingerprint=fingerprint, paths=paths)

    report = _report(spec=spec, built=built, losses=losses, sites=sites, job_list=job_list)
    paths.report.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
