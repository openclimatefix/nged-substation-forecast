"""Does CERRA's wind direction help predict wind power beyond CERRA's wind speed?

One-off throwaway script for issue 994 of `openclimatefix/nged-substation-forecast`.
The plan is `plans/study-994-cerra-wind-direction.md` and was committed before the first fit.
`cerra_wind_levels.py` (issue 957) is the prerequisite study: this script reuses its row set, folds,
seeds, bootstrap and gates, and adds the direction columns it lacked.

**Data.** `data/studies/weather/CERRA/`: CERRA's wind speed and wind direction at 10, 50, 75, 100,
and 150 m, at the 190 grid cells around the trial area, every 3 hours from 2019-09-01 to 2026-06-30.
Direction is in degrees clockwise from north, the direction the wind blows from. The files' circular
mean at 100 m is about 228 degrees, the south-westerly of the UK's prevailing wind, which confirms
that convention. An arm gets a direction as its sine and cosine, which is the same under either
convention. `studies.wind_direction` holds the encoding, the veer and the month shuffle.

**Row set.** Exactly `cerra_wind_levels.py`'s: every 3-hourly wind-farm hour with a centred power
hour, minus every hour holding an exactly-zero half-hour. The rule reads the target only, so every
arm scores the same rows. The run stops unless the row set's (farm, time) keys equal the saved keys
of `data/studies/cerra_wind_levels/rows.parquet`, which also settles that the power-hour offset and
the zero rule are inherited from that study.

**Arms.** Every arm carries `hour_of_day`, `day_of_year`, and wind columns, with column subsampling
off. Arms sit in two families, and every arm of a family has the same number of columns, so no
contrast compares two widths. An arm with fewer real wind columns than its family's width is padded
with monotone transforms of the 100 m speed, which a tree cannot use.

- Core family, 5 wind columns (the width of the prior study's arms):
  `speed_10m`, `speed_100m` (identical to the prior study's arms), `speed_10m_dir`,
  `speed_100m_dir` (speed, sine and cosine of direction at that height), and `speed_100m_dir_noise`
  (the 100 m speed with another month's 100 m direction).
- Veer family, 12 wind columns, every arm holding the 10 m and 100 m speeds: `veer_speed_10_100`
  (no direction), `veer_dir_100`, `veer_dir_10_100`, `veer_angle_10_100` (the 100 m direction and
  the sine and cosine of the veer from 10 m to 100 m), `veer_dir_all5` (direction at all five
  heights), and `veer_dir_100_noise` (`veer_dir_100` plus another month's 10 m direction).

**Planned contrasts** (`PLANNED_CONTRASTS`, written into the plan before any fit, each also run at
the second hyperparameter setting and adjusted for three comparisons):

1. `speed_100m_dir` minus `speed_100m`.
2. `speed_10m_dir` minus `speed_10m`.
3. `speed_100m_dir` minus `speed_10m_dir`.

Every other contrast is exploratory, and the veer family's contrasts are exploratory by design.

**Controls.** Two negative controls shuffle direction by month: `speed_100m_dir_noise` against
`speed_100m` (width-matched noise against padding) and `veer_dir_100_noise` against `veer_dir_100`.
Two positive controls use synthetic targets, a fixed power curve of the 100 m speed cut by 40% in
a direction sector (`direction_sector`) or at a veer of 10 degrees or more (`veer`), with noise. The
run writes every output first and then raises unless `speed_100m_dir` beats `speed_100m` on the
sector target and `veer_angle_10_100` beats `veer_dir_100` on the veer target, each by an interval
that excludes zero.

**Checks that need no fit.** `--dry-run` prints the arms, the number of fits, the output files and
which direction files are missing, reads no data and exits 0. `--check` reads the direction files
that exist and the private power table, builds the row set from the heights that exist, runs every
check below on them, and exits non-zero on a violation. Neither fits a model or writes an output.
The full run stops while any direction file is missing.

Run it with `uv run python studies/beam_diffuse_split/cerra_wind_direction.py`. `--report-only`
rebuilds `report.md` and the interval tables from the saved losses, still checking the saved
fingerprint. A re-run stops while any output exists, until it is moved to a `superseded/`
subfolder. Only one agent may run it at a time, because every worktree shares one data folder.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_dataset import _wind_sites
from cerra_wind_levels import (
    CERRA_DIR,
    CUT_IN_M_S,
    GRID_PATH,
    MAX_CORES,
    MAX_WORKERS,
    NOISE_SEED,
    PRIMARY_SETTING,
    RATED_M_S,
    SENSITIVITY_SETTING,
    SHARED_FEATURES,
    SYNTHETIC_NOISE_FRACTION,
    Contrast,
    check_column_counts,
    check_same_rows,
    check_settings,
    read_half_hourly_power,
)
from ens_past_solar import _arm_columns_lines, _fingerprint
from run_experiment import Job, _add_time_features, run_all
from sources import STUDIES_DATA_DIR
from studies.bootstrap import (
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
    per_fold_differences,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    THREADS_PER_FIT,
    assign_folds,
    calendar_month_coverage,
    raise_on_uncovered_months,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.reanalysis_wind import (
    CERRA_DIRECTION_FILES,
    CERRA_FILES,
    derive_nearest_cells,
    join_centred_power,
    read_cerra_direction,
    read_cerra_wind,
)
from studies.wind_direction import shuffled_by_month, sine_cosine, veer_degrees
from weather_products import METRIC, PERCENTAGE_POINTS, _mae

_LOG: Final[logging.Logger] = logging.getLogger("cerra_wind_direction")

OUTPUT_DIR: Final[Path] = STUDIES_DATA_DIR / "cerra_wind_direction"
"""Where the script writes its outputs, and a `superseded/` folder for re-runs. The folder must not
exist before the first run's outputs: `refuse_to_overwrite` stops on any file."""

PRIOR_ROWS_PATH: Final[Path] = STUDIES_DATA_DIR / "cerra_wind_levels" / "rows.parquet"
PRIOR_LOSSES_PATH: Final[Path] = STUDIES_DATA_DIR / "cerra_wind_levels" / "losses.parquet"
"""The prerequisite study's saved row set and per-row losses, read-only."""

HEIGHTS_M: Final[tuple[int, ...]] = (10, 50, 75, 100, 150)
"""Every height whose direction the full run reads."""

OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "losses.parquet",
    "losses.fingerprint",
    "rows.parquet",
    "intervals.parquet",
    "absolute.parquet",
    "report.md",
)

CORE_WIND_COLUMNS: Final[int] = 5
VEER_WIND_COLUMNS: Final[int] = 12
"""The wind columns every arm of a family carries: the family's widest arm's real columns."""

TRANSFORMS: Final[dict[str, Callable[[pl.Expr], pl.Expr]]] = {
    "squared": lambda column: column.pow(2),
    "sqrt": lambda column: column.sqrt(),
    "log1p": lambda column: column.log1p(),
    "cubed": lambda column: column.pow(3),
    "pow_1_5": lambda column: column.pow(1.5),
    "pow_2_5": lambda column: column.pow(2.5),
    "fourth_root": lambda column: column.pow(0.25),
    "pow_4": lambda column: column.pow(4),
    "exp_tenth": lambda column: (column / 10.0).exp(),
    "reciprocal_1p": lambda column: 1.0 / (1.0 + column),
}
"""The strictly monotone transforms of a speed that pad an arm, in padding order. The first four
are `cerra_wind_levels.py`'s, so the two speed-only arms equal that study's arms column for
column."""

SECTOR_CENTRE_DEG: Final[float] = 255.0
SECTOR_HALF_WIDTH_DEG: Final[float] = 30.0
VEER_THRESHOLD_DEG: Final[float] = 10.0
SYNTHETIC_LOSS: Final[float] = 0.4
"""The share of power the synthetic targets remove inside the direction sector, or at a veer of
`VEER_THRESHOLD_DEG` or more from 10 m to 100 m."""

SYNTHETIC_SHARE_RANGE: Final[tuple[float, float]] = (0.10, 0.50)
"""The share of rows a synthetic target's rule may affect. Outside it the control is vacuous or the
whole target."""

SECTOR_TARGET: Final[str] = "synthetic_sector_mw"
VEER_TARGET: Final[str] = "synthetic_veer_mw"

SECTOR_SETTING: Final[str] = "positive_control_sector"
VEER_SETTING: Final[str] = "positive_control_veer"
CONTROL_SETTINGS: Final[tuple[str, ...]] = (SECTOR_SETTING, VEER_SETTING)

PLANNED_COUNT: Final[int] = 3
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / PLANNED_COUNT)
"""The coverage in percent of the interval adjusted for the three planned contrasts: 98.33."""

CORE_ARMS: Final[tuple[str, ...]] = (
    "speed_10m",
    "speed_100m",
    "speed_10m_dir",
    "speed_100m_dir",
    "speed_100m_dir_noise",
)
VEER_ARMS: Final[tuple[str, ...]] = (
    "veer_speed_10_100",
    "veer_dir_100",
    "veer_dir_10_100",
    "veer_angle_10_100",
    "veer_dir_all5",
    "veer_dir_100_noise",
)
REAL_ARMS: Final[tuple[str, ...]] = (*CORE_ARMS, *VEER_ARMS)
"""The arms fitted on the real target, at both settings."""

SECTOR_CONTROL_ARMS: Final[tuple[str, ...]] = ("speed_100m", "speed_100m_dir")
VEER_CONTROL_ARMS: Final[tuple[str, ...]] = ("veer_dir_100", "veer_dir_10_100", "veer_angle_10_100")

PLANNED_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("speed_100m_dir", "speed_100m", "what 100 m direction adds to 100 m speed", True),
    Contrast("speed_10m_dir", "speed_10m", "what 10 m direction adds to 10 m speed", True),
    Contrast("speed_100m_dir", "speed_10m_dir", "100 m against 10 m, each with direction", True),
)
"""The three contrasts the plan names before any fit."""

EXPLORATORY_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast(
        "speed_100m_dir", "speed_100m_dir_noise", "direction against shuffled direction", False
    ),
    Contrast("veer_dir_100", "veer_speed_10_100", "100 m direction on top of two speeds", False),
    Contrast("veer_dir_10_100", "veer_speed_10_100", "two directions on top of two speeds", False),
    Contrast("veer_dir_10_100", "veer_dir_100", "the 10 m direction beyond the 100 m one", False),
    Contrast(
        "veer_angle_10_100", "veer_dir_100", "the veer angle beyond the 100 m direction", False
    ),
    Contrast("veer_dir_all5", "veer_dir_100", "directions at five heights against one", False),
    Contrast("veer_dir_all5", "veer_dir_10_100", "five directions against two", False),
    Contrast(
        "veer_dir_10_100", "veer_dir_100_noise", "the 10 m direction against a shuffled one", False
    ),
)
"""Reported for context. The veer rows are exploratory by design."""

NEGATIVE_CONTROLS: Final[tuple[Contrast, ...]] = (
    Contrast("speed_100m_dir_noise", "speed_100m", "shuffled direction against padding", False),
    Contrast("veer_dir_100_noise", "veer_dir_100", "a shuffled 10 m direction against none", False),
)
POSITIVE_CONTROLS: Final[tuple[tuple[Contrast, str], ...]] = (
    (
        Contrast("speed_100m_dir", "speed_100m", "direction where the target has a sector", False),
        SECTOR_SETTING,
    ),
    (
        Contrast(
            "veer_angle_10_100", "veer_dir_100", "the veer angle where the target has veer", False
        ),
        VEER_SETTING,
    ),
    (
        Contrast(
            "veer_dir_10_100", "veer_dir_100", "raw directions where the target has veer", False
        ),
        VEER_SETTING,
    ),
)
"""The first two gate the run. The third reports whether two raw directions can recover a veer rule,
which bounds what a null `veer_dir_10_100` against `veer_dir_100` can be read as."""


def _sin(*, height_m: int) -> str:
    return f"wind_dir_sin_{height_m}m"


def _cos(*, height_m: int) -> str:
    return f"wind_dir_cos_{height_m}m"


def _speed(*, height_m: int) -> str:
    return f"wind_speed_{height_m}m"


def _noise_pair(*, height_m: int) -> tuple[str, str]:
    return (f"{_sin(height_m=height_m)}_other_month", f"{_cos(height_m=height_m)}_other_month")


VEER_SIN: Final[str] = "veer_sin_10_100m"
VEER_COS: Final[str] = "veer_cos_10_100m"


def padding_columns(*, base: str, count: int) -> tuple[str, ...]:
    """Name the monotone transforms of one column that pad an arm to its family's width.

    Args:
        base: The column to transform.
        count: How many transforms to name, at most `len(TRANSFORMS)`.

    Returns:
        The columns `with_padding` adds, in `TRANSFORMS` order.

    Raises:
        ValueError: If `count` is negative or exceeds `len(TRANSFORMS)`.
    """
    if not 0 <= count <= len(TRANSFORMS):
        msg = f"count must be between 0 and {len(TRANSFORMS)}, got {count}"
        raise ValueError(msg)
    return tuple(f"{base}_{name}" for name in list(TRANSFORMS)[:count])


def _wind_columns(*, real: tuple[str, ...], pad_from: str, width: int) -> tuple[str, ...]:
    """Return one arm's wind columns: its real columns, padded to its family's width.

    Args:
        real: The arm's real wind columns.
        pad_from: The speed column the padding transforms.
        width: The family's number of wind columns.

    Returns:
        A tuple of exactly `width` names.
    """
    return (*real, *padding_columns(base=pad_from, count=width - len(real)))


def arm_columns() -> dict[str, tuple[str, ...]]:
    """Return every arm's feature columns, from one function so no arm can silently lose one.

    Returns:
        Each arm's name to its shared features then its wind columns.
    """
    s10, s100 = _speed(height_m=10), _speed(height_m=100)
    dir100 = (_sin(height_m=100), _cos(height_m=100))
    dir10 = (_sin(height_m=10), _cos(height_m=10))
    core = {
        "speed_10m": _wind_columns(real=(s10,), pad_from=s10, width=CORE_WIND_COLUMNS),
        "speed_100m": _wind_columns(real=(s100,), pad_from=s100, width=CORE_WIND_COLUMNS),
        "speed_10m_dir": _wind_columns(real=(s10, *dir10), pad_from=s10, width=CORE_WIND_COLUMNS),
        "speed_100m_dir": _wind_columns(
            real=(s100, *dir100), pad_from=s100, width=CORE_WIND_COLUMNS
        ),
        "speed_100m_dir_noise": _wind_columns(
            real=(s100, *_noise_pair(height_m=100)), pad_from=s100, width=CORE_WIND_COLUMNS
        ),
    }
    all_five = tuple(
        name for height in HEIGHTS_M for name in (_sin(height_m=height), _cos(height_m=height))
    )
    speeds = (s10, s100)
    veer_real = {
        "veer_speed_10_100": speeds,
        "veer_dir_100": (*speeds, *dir100),
        "veer_dir_10_100": (*speeds, *dir10, *dir100),
        "veer_angle_10_100": (*speeds, *dir100, VEER_SIN, VEER_COS),
        "veer_dir_all5": (*speeds, *all_five),
        "veer_dir_100_noise": (*speeds, *dir100, *_noise_pair(height_m=10)),
    }
    veer = {
        arm: _wind_columns(real=real, pad_from=s100, width=VEER_WIND_COLUMNS)
        for arm, real in veer_real.items()
    }
    return {arm: (*SHARED_FEATURES, *columns) for arm, columns in {**core, **veer}.items()}


def family_of(*, arm: str) -> str:
    """Name the family an arm belongs to.

    Args:
        arm: An arm name.

    Returns:
        `core` or `veer`.

    Raises:
        KeyError: If the arm is in neither family.
    """
    if arm in CORE_ARMS:
        return "core"
    if arm in VEER_ARMS:
        return "veer"
    raise KeyError(arm)


def check_family_widths(*, arms: dict[str, tuple[str, ...]]) -> None:
    """Stop unless every arm of a family has the same number of distinct columns.

    The families differ in width on purpose, so `check_column_counts` runs on each family alone. A
    contrast across families is refused as well.

    Args:
        arms: Each arm's name to its feature columns.

    Raises:
        ValueError: Naming the family whose arms differ in width, or a contrast across families.
    """
    for family in ("core", "veer"):
        check_column_counts(
            arms={arm: columns for arm, columns in arms.items() if family_of(arm=arm) == family}
        )
    contrasts = (
        *PLANNED_CONTRASTS,
        *EXPLORATORY_CONTRASTS,
        *NEGATIVE_CONTROLS,
        *(pair[0] for pair in POSITIVE_CONTROLS),
    )
    crossing = [
        (c.treatment, c.reference)
        for c in contrasts
        if family_of(arm=c.treatment) != family_of(arm=c.reference)
    ]
    if crossing:
        msg = f"contrasts across families would compare two widths: {crossing}"
        raise ValueError(msg)


def heights_needed(*, arms: tuple[str, ...] = REAL_ARMS) -> tuple[int, ...]:
    """List the direction heights the named arms read.

    Args:
        arms: The arms to inspect.

    Returns:
        The heights in `HEIGHTS_M` whose sine columns appear in an arm's columns.
    """
    columns = arm_columns()
    used = {name for arm in arms for name in columns[arm]}
    return tuple(height for height in HEIGHTS_M if _sin(height_m=height) in used)


def missing_direction_files(*, directory: Path = CERRA_DIR) -> list[str]:
    """List the direction files the full run needs that are not on disk.

    Args:
        directory: The folder holding the CERRA files.

    Returns:
        The missing file names, in height order.
    """
    return [
        CERRA_DIRECTION_FILES[height]
        for height in HEIGHTS_M
        if not (directory / CERRA_DIRECTION_FILES[height]).exists()
    ]


def with_direction_columns(*, frame: pl.DataFrame, heights: tuple[int, ...]) -> pl.DataFrame:
    """Add each present height's sine and cosine, the veer and the monotone padding.

    Args:
        frame: Rows carrying the speed columns, `wind_direction_{h}m` for each height in `heights`,
            `site` and `month`, sorted by site then time.
        heights: The heights whose direction is present.

    Returns:
        The frame with `wind_dir_sin_{h}m` and `wind_dir_cos_{h}m`, the shuffled-month pair for
        10 m and 100 m, `veer_sin_10_100m`, `veer_cos_10_100m` and `veer_deg_10_100m` where both
        heights are present, and the `TRANSFORMS` of the 10 m and 100 m speeds.
    """
    additions: list[pl.Expr] = []
    for height in heights:
        sin, cos = sine_cosine(direction_deg=pl.col(f"wind_direction_{height}m"))
        additions += [sin.alias(_sin(height_m=height)), cos.alias(_cos(height_m=height))]
    if 10 in heights and 100 in heights:
        veer = veer_degrees(
            upper_deg=pl.col("wind_direction_100m"), lower_deg=pl.col("wind_direction_10m")
        )
        sin, cos = sine_cosine(direction_deg=veer)
        additions += [veer.alias("veer_deg_10_100m"), sin.alias(VEER_SIN), cos.alias(VEER_COS)]
    additions += [
        transform(pl.col(_speed(height_m=height))).alias(f"{_speed(height_m=height)}_{name}")
        for height in (10, 100)
        for name, transform in TRANSFORMS.items()
    ]
    return frame.with_columns(additions)


def with_shuffled_direction(*, frame: pl.DataFrame, heights: tuple[int, ...]) -> pl.DataFrame:
    """Add the negative controls' columns: another month's direction at 10 m and at 100 m.

    The direction in degrees is shuffled, then encoded, so each row's sine and cosine stay a
    consistent pair.

    Args:
        frame: Rows sorted by site then time, carrying `month`, `site` and `wind_direction_{h}m`.
        heights: The heights whose direction is present.

    Returns:
        The frame with `_noise_pair` columns for each of 10 m and 100 m that is in `heights`, each
        shuffled separately within each farm.
    """
    columns: list[pl.Series] = []
    for height in (10, 100):
        if height not in heights:
            continue
        values = frame[f"wind_direction_{height}m"].to_numpy()
        months = frame["month"].to_numpy()
        sites = frame["site"].to_numpy()
        shuffled = np.empty_like(values)
        for site_index, site in enumerate(sorted(set(sites))):
            rows = sites == site
            generator = np.random.default_rng((NOISE_SEED, height, site_index))
            shuffled[rows] = shuffled_by_month(
                values=values[rows], months=months[rows], generator=generator
            )
        radians = np.radians(shuffled.astype(np.float64))
        sin_name, cos_name = _noise_pair(height_m=height)
        columns += [pl.Series(sin_name, np.sin(radians)), pl.Series(cos_name, np.cos(radians))]
    return frame.with_columns(columns)


def _power_curve_fraction(*, speed: pl.Expr) -> pl.Expr:
    """Return a fixed power curve as a fraction of capacity: cut-in, a cubic, then rated power.

    Args:
        speed: The wind speed expression in metres per second.

    Returns:
        The fraction in [0, 1].
    """
    return ((speed - CUT_IN_M_S) / (RATED_M_S - CUT_IN_M_S)).clip(0.0, 1.0).pow(3)


def synthetic_rule_masks(*, frame: pl.DataFrame) -> dict[str, pl.Series]:
    """Return which rows each synthetic target cuts.

    Args:
        frame: Rows carrying `wind_direction_100m` and `veer_deg_10_100m`.

    Returns:
        `SECTOR_TARGET` to the rows whose 100 m direction lies within `SECTOR_HALF_WIDTH_DEG` of
        `SECTOR_CENTRE_DEG`, and `VEER_TARGET` to the rows whose veer from 10 m to 100 m is at least
        `VEER_THRESHOLD_DEG`.
    """
    distance = ((pl.col("wind_direction_100m") - SECTOR_CENTRE_DEG + 180.0) % 360.0 - 180.0).abs()
    masks = frame.select(
        sector=distance < SECTOR_HALF_WIDTH_DEG,
        veer=pl.col("veer_deg_10_100m") >= VEER_THRESHOLD_DEG,
    )
    return {SECTOR_TARGET: masks["sector"], VEER_TARGET: masks["veer"]}


def with_synthetic_targets(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the two positive controls' targets.

    Each target is the fixed power curve of the 100 m speed, multiplied by `1 - SYNTHETIC_LOSS`
    inside its rule's rows, plus Gaussian noise, in megawatts. Only an arm that can read the rule's
    direction columns can reproduce the cut.

    Args:
        frame: Rows carrying the 100 m speed, `wind_direction_100m`, `veer_deg_10_100m` and
            `effective_capacity_mw`.

    Returns:
        The frame with `SECTOR_TARGET` and `VEER_TARGET`.
    """
    masks = synthetic_rule_masks(frame=frame)
    fraction = _power_curve_fraction(speed=pl.col(_speed(height_m=100)))
    additions = []
    for index, target in enumerate((SECTOR_TARGET, VEER_TARGET)):
        noise = np.random.default_rng((NOISE_SEED, index)).normal(
            0.0, SYNTHETIC_NOISE_FRACTION, frame.height
        )
        cut = pl.when(masks[target]).then(1.0 - SYNTHETIC_LOSS).otherwise(1.0)
        additions.append(
            (
                (fraction * cut + pl.Series(noise)).clip(0.0, 1.0) * pl.col("effective_capacity_mw")
            ).alias(target)
        )
    return frame.with_columns(additions)


def check_synthetic_shares(*, frame: pl.DataFrame) -> dict[str, float]:
    """Stop unless each synthetic rule cuts a share of rows inside `SYNTHETIC_SHARE_RANGE`.

    Args:
        frame: Rows carrying the columns `synthetic_rule_masks` reads.

    Returns:
        Each target's share of affected rows.

    Raises:
        ValueError: Naming each target whose share is outside the range.
    """
    shares = {
        target: float(mask.to_numpy().mean())
        for target, mask in synthetic_rule_masks(frame=frame).items()
    }
    low, high = SYNTHETIC_SHARE_RANGE
    bad = {target: share for target, share in shares.items() if not low <= share <= high}
    if bad:
        msg = f"a synthetic rule affects a share of rows outside [{low}, {high}]: {bad}"
        raise ValueError(msg)
    return shares


def build_rows(
    *,
    wind: pl.DataFrame,
    direction: pl.DataFrame,
    half_hourly: pl.DataFrame,
    sites: pl.DataFrame,
    heights: tuple[int, ...],
) -> pl.DataFrame:
    """Build the frame the fit loop reads.

    Args:
        wind: `read_cerra_wind`'s frame.
        direction: `read_cerra_direction`'s frame for `heights`.
        half_hourly: `read_half_hourly_power`'s frame.
        sites: The wind roster, with `site` and `effective_capacity_mw`.
        heights: The heights whose direction is present.

    Returns:
        One row per farm-hour with a wind value and a centred power hour, minus every hour holding
        an exactly-zero half-hour, sorted by site then time, with the speeds, directions, every
        arm's derived columns that `heights` allows, `power_mw`, `effective_capacity_mw`, the
        calendar features, `month`, `fold`, the controls' columns, and the `constrained` and
        `cap_mw` columns the fit loop reads. NGED has confirmed that no wind farm in the trial area
        is under active network management.

    Raises:
        ValueError: If the 10 m or 100 m direction is absent, a column an arm reads holds a missing
            value, a calendar month has no training row in a fold, or a synthetic rule affects
            too few or too many rows.
    """
    if not {10, 100} <= set(heights):
        msg = "the controls and the veer arms need the 10 m and 100 m directions"
        raise ValueError(msg)
    joined = (
        join_centred_power(wind=wind, half_hourly=half_hourly)
        .join(direction, on=["site", "time"], how="left")
        .join(sites.select("site", "effective_capacity_mw"), on="site")
        .filter(~pl.col("has_zero_half_hour"))
        .with_columns(constrained=pl.lit(value=False), cap_mw=pl.lit(None, dtype=pl.Float64))
        .pipe(lambda rows: _add_time_features(dataset=rows))
        .sort("site", "time")
    )
    frame = with_shuffled_direction(
        frame=with_direction_columns(frame=joined, heights=heights), heights=heights
    )
    frame = with_synthetic_targets(frame=frame)
    frame = assign_folds(dataset=frame)
    arms = arm_columns()
    present = {
        name
        for arm in REAL_ARMS
        if set(heights_needed(arms=(arm,))) <= set(heights)
        for name in arms[arm]
    }
    check_no_missing(
        frame=frame,
        columns=("power_mw", "effective_capacity_mw", *sorted(present)),
    )
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=frame))
    check_synthetic_shares(frame=frame)
    return frame


def jobs() -> list[Job]:
    """Return every fit: the arms at both settings, and the two positive controls.

    Returns:
        `REAL_ARMS` at `PRIMARY_SETTING` and at `SENSITIVITY_SETTING`, `SECTOR_CONTROL_ARMS` on
        `SECTOR_TARGET` at `SECTOR_SETTING`, and `VEER_CONTROL_ARMS` on `VEER_TARGET` at
        `VEER_SETTING`, all at the primary hyperparameters.
    """
    columns = arm_columns()
    fits: list[Job] = []
    for setting, hyper_parameters in (
        (PRIMARY_SETTING, PRIMARY_HYPER_PARAMETERS),
        (SENSITIVITY_SETTING, SENSITIVITY_HYPER_PARAMETERS),
    ):
        fits += [
            (arm, setting, "power_mw", columns[arm], hyper_parameters, False) for arm in REAL_ARMS
        ]
    for setting, target, arms in (
        (SECTOR_SETTING, SECTOR_TARGET, SECTOR_CONTROL_ARMS),
        (VEER_SETTING, VEER_TARGET, VEER_CONTROL_ARMS),
    ):
        fits += [
            (arm, setting, target, columns[arm], PRIMARY_HYPER_PARAMETERS, False) for arm in arms
        ]
    return fits


def check_same_rows_as_prior(*, frame: pl.DataFrame, prior_path: Path = PRIOR_ROWS_PATH) -> int:
    """Stop unless the row set's (farm, time) keys equal the prerequisite study's.

    Args:
        frame: The row set.
        prior_path: The prerequisite study's saved `rows.parquet`.

    Returns:
        The number of rows both sets share.

    Raises:
        ValueError: If the keys differ, naming the counts only.
    """
    prior = pl.read_parquet(prior_path).select("site", "time")
    ours = frame.select("site", "time")
    only_new = ours.join(prior, on=["site", "time"], how="anti").height
    only_prior = prior.join(ours, on=["site", "time"], how="anti").height
    if only_new or only_prior:
        msg = (
            f"the row set differs from the prerequisite study's: {only_new} rows only here, "
            f"{only_prior} only there"
        )
        raise ValueError(msg)
    return ours.height


def check_direction_files(*, directory: Path = CERRA_DIR, heights: tuple[int, ...]) -> list[str]:
    """Check each present direction file against the speed file of the same height.

    The check reads `lineage`-free facts from the files themselves: the file holds the same
    (time, cell) keys as the speed file, has no nulls or NaNs, stays within [0, 360], and has a
    circular mean in the south-westerly half of the compass (180 to 300 degrees), which is what the
    meteorological from-direction convention gives over the UK.

    Args:
        directory: The CERRA folder.
        heights: The heights to check, each of which must be on disk.

    Returns:
        One text line per height, for the report.

    Raises:
        ValueError: Naming each height that fails.
    """
    lines: list[str] = []
    failures: list[str] = []
    for height in heights:
        direction = pl.scan_parquet(directory / CERRA_DIRECTION_FILES[height])
        speed = pl.scan_parquet(directory / CERRA_FILES[height])
        keys = ["valid_time", "y_index", "x_index"]
        stats = direction.select(
            rows=pl.len(),
            bad=(
                pl.col("wind_direction_deg").is_null()
                | pl.col("wind_direction_deg").is_nan()
                | ~pl.col("wind_direction_deg").is_between(0.0, 360.0)
            ).sum(),
            mean_sin=pl.col("wind_direction_deg").radians().sin().mean(),
            mean_cos=pl.col("wind_direction_deg").radians().cos().mean(),
        ).collect()
        mismatch = (
            direction.select(keys).join(speed.select(keys), on=keys, how="anti").collect().height
            + speed.select(keys).join(direction.select(keys), on=keys, how="anti").collect().height
        )
        row = stats.row(0, named=True)
        mean_deg = float(np.degrees(np.arctan2(row["mean_sin"], row["mean_cos"])) % 360.0)
        ok = row["bad"] == 0 and mismatch == 0 and 180.0 <= mean_deg <= 300.0
        lines.append(
            f"{height} m: {row['rows']:,} rows, {row['bad']} bad values, {mismatch} keys unmatched "
            f"to the speed file, circular mean {mean_deg:.0f} degrees"
        )
        if not ok:
            failures.append(f"{height} m")
    if failures:
        msg = f"direction files failing their checks: {failures}; see {lines}"
        raise ValueError(msg)
    return lines


def check_settings_and_arms() -> None:
    """Run the checks on the arm list and the fit settings that need no data.

    Raises:
        ValueError: If an arm family differs in width, a contrast crosses families or names an
            unknown arm, or a fit would run off the CPU, with column subsampling, or on too many
            cores.
    """
    arms = arm_columns()
    check_family_widths(arms=arms)
    unknown = {
        arm
        for contrast in (
            *PLANNED_CONTRASTS,
            *EXPLORATORY_CONTRASTS,
            *NEGATIVE_CONTROLS,
            *(pair[0] for pair in POSITIVE_CONTROLS),
        )
        for arm in (contrast.treatment, contrast.reference)
        if arm not in arms
    }
    if unknown:
        msg = f"contrasts name arms with no columns: {sorted(unknown)}"
        raise ValueError(msg)
    check_settings(job_list=jobs())


def fit_count(*, job_list: list[Job], n_sites: int = 3, n_folds: int = 5, n_seeds: int = 3) -> int:
    """Count the XGBoost fits a run makes.

    Args:
        job_list: Every (arm, setting) fit.
        n_sites: The number of wind farms.
        n_folds: The number of folds each (arm, farm) is cut into.
        n_seeds: The number of fitting seeds.

    Returns:
        The number of single XGBoost fits.
    """
    return len(job_list) * n_sites * n_folds * n_seeds


def interval_record(
    *, losses: pl.DataFrame, contrast: Contrast, setting: str, scope: str
) -> dict[str, object]:
    """Compute one contrast's interval and both arms' errors on one scope of rows.

    Args:
        losses: One setting's per-row losses for both arms, restricted to the scope.
        contrast: The pairing.
        setting: The setting the losses came from.
        scope: A label for the rows: `all`, a farm label, or a half of the year.

    Returns:
        A record with the differences in percentage points of capacity, the 95% interval, the
        three-comparison interval for a planned contrast, the fold signs, and both arms' errors.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=contrast.treatment, reference=contrast.reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=contrast.treatment, reference=contrast.reference, metric=METRIC
    )
    agreeing = sum(np.sign(value) == np.sign(interval["difference"]) for value in folds)
    wide: tuple[float, float] | tuple[None, None] = (None, None)
    if contrast.planned:
        wide = bootstrap_difference_at_level(
            losses=losses,
            treatment=contrast.treatment,
            reference=contrast.reference,
            metric=METRIC,
            level=BONFERRONI_LEVEL,
        )
    return {
        "setting": setting,
        "scope": scope,
        "treatment": contrast.treatment,
        "reference": contrast.reference,
        "planned": contrast.planned,
        "treatment_mae_pp": _mae(losses=losses, arm=contrast.treatment),
        "reference_mae_pp": _mae(losses=losses, arm=contrast.reference),
        "difference_pp": interval["difference"] * PERCENTAGE_POINTS,
        "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
        "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
        "significant": interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0,
        "lower_bonferroni_pp": None if wide[0] is None else wide[0] * PERCENTAGE_POINTS,
        "upper_bonferroni_pp": None if wide[1] is None else wide[1] * PERCENTAGE_POINTS,
        "folds_agreeing": int(agreeing),
        "n_folds": len(folds),
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
        "seed_spread_pp": interval["seed_spread"] * PERCENTAGE_POINTS,
    }


def _half_of_year(*, losses: pl.DataFrame, half: str) -> pl.DataFrame:
    """Restrict losses to October to March (`winter`) or April to September (`summer`).

    Args:
        losses: Per-row losses carrying `time`.
        half: `winter` or `summer`.

    Returns:
        The rows in that half.
    """
    month = pl.col("time").dt.month()
    return losses.filter(
        (month >= 10) | (month <= 3) if half == "winter" else month.is_between(4, 9)
    )


def contrast_records(*, losses: pl.DataFrame, sites: list[str]) -> pl.DataFrame:
    """Compute every interval the report and the charts quote.

    Args:
        losses: Every arm's losses, every setting.
        sites: The farm labels.

    Returns:
        One row per (setting, scope, contrast). The planned contrasts, the exploratory contrasts and
        the negative controls run on all rows at both real-target settings. The planned contrasts
        also run per farm and per half of the year at the primary setting (exploratory splits).
        Each positive control runs on its own synthetic target.
    """
    by_setting = {
        setting: losses.filter(pl.col("setting") == setting)
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING, *CONTROL_SETTINGS)
    }
    records = []
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
        records += [
            interval_record(
                losses=by_setting[setting], contrast=contrast, setting=setting, scope="all"
            )
            for contrast in (*PLANNED_CONTRASTS, *EXPLORATORY_CONTRASTS, *NEGATIVE_CONTROLS)
        ]
    primary = by_setting[PRIMARY_SETTING]
    splits = [(site, primary.filter(pl.col("site") == site)) for site in sites]
    splits += [(half, _half_of_year(losses=primary, half=half)) for half in ("winter", "summer")]
    for contrast in PLANNED_CONTRASTS:
        records += [
            interval_record(
                losses=scoped,
                contrast=contrast._replace(planned=False),
                setting=PRIMARY_SETTING,
                scope=scope,
            )
            for scope, scoped in splits
        ]
    records += [
        interval_record(losses=by_setting[setting], contrast=contrast, setting=setting, scope="all")
        for contrast, setting in POSITIVE_CONTROLS
    ]
    return pl.DataFrame(records, infer_schema_length=None)


def absolute_records(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Compute every arm's absolute error and interval, at each setting and on each farm.

    Args:
        losses: Every arm's losses, every setting.

    Returns:
        One row per (setting, scope, arm) with the error and its 95% interval in percentage points
        of capacity.
    """
    records = []
    sites = sorted(losses["site"].unique().to_list())
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING, *CONTROL_SETTINGS):
        in_setting = losses.filter(pl.col("setting") == setting)
        for scope, scoped in (
            ("all", in_setting),
            *((site, in_setting.filter(pl.col("site") == site)) for site in sites),
        ):
            for arm in sorted(scoped["arm"].unique().to_list()):
                interval = bootstrap_absolute(losses=scoped, arm=arm, metric=METRIC)
                records.append(
                    {
                        "setting": setting,
                        "scope": scope,
                        "arm": arm,
                        "mae_pp": interval["value"] * PERCENTAGE_POINTS,
                        "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
                        "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
                        "n_rows": interval["n_rows"],
                        "n_months": interval["n_months"],
                    }
                )
    return pl.DataFrame(records)


def prior_agreement(*, losses: pl.DataFrame, prior_path: Path = PRIOR_LOSSES_PATH) -> list[str]:
    """Compare the two speed-only arms' per-row losses with the prerequisite study's.

    The two arms have the same columns, rows, folds, seeds and settings as in the prerequisite
    study, so the losses should be identical on the same device.

    Args:
        losses: This run's per-row losses.
        prior_path: The prerequisite study's saved `losses.parquet`.

    Returns:
        One report line per arm with the largest absolute difference and the number of rows
        compared, or a line saying the prior file is absent.
    """
    if not prior_path.exists():
        return [f"{prior_path.name} is absent, so no comparison was made."]
    prior = pl.read_parquet(prior_path).filter(pl.col("setting") == PRIMARY_SETTING)
    lines = []
    for arm in ("speed_10m", "speed_100m"):
        ours = losses.filter((pl.col("setting") == PRIMARY_SETTING) & (pl.col("arm") == arm))
        theirs = prior.filter(pl.col("arm") == arm)
        joined = ours.join(theirs, on=["site", "time", "seed"], suffix="_prior")
        gap = float(np.abs(joined[METRIC].to_numpy() - joined[f"{METRIC}_prior"].to_numpy()).max())
        lines.append(
            f"- `{arm}`: largest absolute difference from the prerequisite study's per-row loss "
            f"{gap:.3g} over {joined.height:,} rows "
            f"({ours.height:,} here, {theirs.height:,} there)."
        )
    return lines


CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Setting, scope | Contrast | Treatment error | Reference error "
        "| Difference (pp of capacity) "
        "| Unadjusted 95% interval | Significant at the 5% level, unadjusted? "
        f"| {BONFERRONI_LEVEL:.2f}% interval (three planned contrasts) "
        "| Significant after adjustment? | Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|---|---|---|---|",
)


def _contrast_lines(*, records: pl.DataFrame) -> list[str]:
    """Render contrast records as a markdown table.

    Args:
        records: Rows of `contrast_records`.

    Returns:
        Markdown lines, header included.
    """
    lines = [*CONTRAST_HEADER]
    for row in records.iter_rows(named=True):
        lower, upper = row["lower_bonferroni_pp"], row["upper_bonferroni_pp"]
        adjusted = "n/a" if lower is None else f"[{lower:+.3f}, {upper:+.3f}]"
        verdict = "n/a" if lower is None else ("**yes**" if lower > 0.0 or upper < 0.0 else "no")
        lines.append(
            f"| {row['setting']}, {row['scope']} | {row['treatment']} − {row['reference']} "
            f"| {row['treatment_mae_pp']:.3f} | {row['reference_mae_pp']:.3f} "
            f"| {row['difference_pp']:+.3f} "
            f"| [{row['lower_95_pp']:+.3f}, {row['upper_95_pp']:+.3f}] "
            f"| {'**yes**' if row['significant'] else 'no'} | {adjusted} | {verdict} "
            f"| {row['folds_agreeing']} of {row['n_folds']} | {row['n_rows']:,} |"
        )
    return lines


def _absolute_lines(*, absolute: pl.DataFrame, setting: str, arms: tuple[str, ...]) -> list[str]:
    """Render each arm's pooled absolute error and 95% interval as a markdown table.

    Args:
        absolute: `absolute_records`' frame.
        setting: The setting.
        arms: The arms to list, in order.

    Returns:
        Markdown lines, header included.
    """
    pooled = absolute.filter((pl.col("setting") == setting) & (pl.col("scope") == "all"))
    lines = ["| Arm | Error (pp of capacity) | 95% interval | Rows |", "|---|---|---|---|"]
    for arm in arms:
        row = pooled.filter(pl.col("arm") == arm).row(0, named=True)
        lines.append(
            f"| {arm} | {row['mae_pp']:.3f} | [{row['lower_95_pp']:.3f}, {row['upper_95_pp']:.3f}] "
            f"| {row['n_rows']:,} |"
        )
    return lines


def report_lines(
    *,
    frame: pl.DataFrame,
    intervals: pl.DataFrame,
    absolute: pl.DataFrame,
    job_list: list[Job],
    file_lines: list[str],
    prior_lines: list[str],
    shares: dict[str, float],
) -> list[str]:
    """Assemble the report the page quotes.

    Args:
        frame: The main row set.
        intervals: `contrast_records`' frame.
        absolute: `absolute_records`' frame.
        job_list: Every fit.
        file_lines: `check_direction_files`' lines.
        prior_lines: `prior_agreement`'s lines.
        shares: `check_synthetic_shares`' result.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            f"### CERRA wind direction on {frame.height:,} farm-hours of wind "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        (
            f"Every fit ran on the CPU with no column subsampling, on {MAX_WORKERS} fits of "
            f"{THREADS_PER_FIT} threads at a time, {fit_count(job_list=job_list):,} XGBoost fits "
            "in all. Errors are means of each row's absolute error over its farm's capacity, in "
            "percentage points, on 3-hourly rows. The rows equal the prerequisite study's."
        ),
        "",
        "#### Direction files (checked before any fit)",
        "",
        *(f"- {line}" for line in file_lines),
        "",
        "#### Agreement of the speed-only arms with the prerequisite study",
        "",
        *prior_lines,
        "",
        "#### Synthetic targets",
        "",
        *(f"- `{target}`: the rule cuts {share:.1%} of rows." for target, share in shares.items()),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Every arm's absolute error, primary setting",
        "",
        *_absolute_lines(absolute=absolute, setting=PRIMARY_SETTING, arms=REAL_ARMS),
        "",
        "#### Every arm's absolute error, second setting",
        "",
        *_absolute_lines(absolute=absolute, setting=SENSITIVITY_SETTING, arms=REAL_ARMS),
        "",
    ]
    pooled = intervals.filter(pl.col("scope") == "all")
    for title, selection in (
        (
            "Planned contrasts, primary setting",
            pooled.filter(pl.col("planned") & (pl.col("setting") == PRIMARY_SETTING)),
        ),
        (
            "Planned contrasts, second setting",
            pooled.filter(pl.col("planned") & (pl.col("setting") == SENSITIVITY_SETTING)),
        ),
        (
            "Exploratory contrasts (veer arms included), both settings",
            pooled.filter(
                ~pl.col("planned")
                & pl.col("setting").is_in([PRIMARY_SETTING, SENSITIVITY_SETTING])
                & ~pl.col("treatment").is_in([c.treatment for c in NEGATIVE_CONTROLS])
            ),
        ),
        (
            "Planned contrasts by farm and half of the year (exploratory splits)",
            intervals.filter(pl.col("scope") != "all"),
        ),
        (
            "Negative controls, both settings",
            pooled.filter(pl.col("treatment").is_in([c.treatment for c in NEGATIVE_CONTROLS])),
        ),
        (
            "Positive controls (synthetic targets)",
            pooled.filter(pl.col("setting").is_in(CONTROL_SETTINGS)),
        ),
    ):
        lines += [f"#### {title}", "", *_contrast_lines(records=selection), ""]
    for setting, arms in ((SECTOR_SETTING, SECTOR_CONTROL_ARMS), (VEER_SETTING, VEER_CONTROL_ARMS)):
        lines += [
            f"#### Absolute error on the synthetic target, `{setting}`",
            "",
            *_absolute_lines(absolute=absolute, setting=setting, arms=arms),
            "",
        ]
    return lines


def check_positive_controls(*, intervals: pl.DataFrame) -> None:
    """Stop unless the two gating positive controls recover their synthetic effects.

    Args:
        intervals: `contrast_records`' frame.

    Raises:
        ValueError: Naming each gating control whose upper 95% bound is not below zero.
    """
    failed = []
    for contrast, setting in POSITIVE_CONTROLS[:2]:
        row = intervals.filter(
            (pl.col("setting") == setting)
            & (pl.col("treatment") == contrast.treatment)
            & (pl.col("reference") == contrast.reference)
        ).row(0, named=True)
        if not row["upper_95_pp"] < 0.0:
            failed.append(
                f"{contrast.treatment} minus {contrast.reference} on {setting}: "
                f"{row['difference_pp']:+.3f} pp "
                f"[{row['lower_95_pp']:+.3f}, {row['upper_95_pp']:+.3f}]"
            )
    if failed:
        msg = (
            f"a positive control failed, so a null on the real target is not 'no effect': {failed}"
        )
        raise ValueError(msg)


def dry_run() -> int:
    """Print the plan of the run, reading no data, and exit 0 even when files are missing.

    Returns:
        0.
    """
    job_list = jobs()
    missing = missing_direction_files()
    lines = [
        f"output folder: {OUTPUT_DIR} (exists: {OUTPUT_DIR.exists()})",
        f"outputs: {', '.join(OUTPUT_NAMES)}",
        (
            f"arms: {len(REAL_ARMS)} real, fitted at both settings; {len(job_list)} "
            f"(arm, setting) fit groups; {fit_count(job_list=job_list):,} XGBoost fits"
        ),
        f"planned contrasts: {len(PLANNED_CONTRASTS)}; exploratory: {len(EXPLORATORY_CONTRASTS)}",
        f"cores: at most {MAX_WORKERS * THREADS_PER_FIT} of {MAX_CORES}",
    ]
    for arm, columns in arm_columns().items():
        lines.append(f"{arm} ({family_of(arm=arm)}, {len(columns)} columns): {', '.join(columns)}")
    lines.append("missing direction files: " + (", ".join(missing) if missing else "none"))
    lines.append("heights the arms read: " + ", ".join(str(h) for h in heights_needed()))
    lines.append("a full run needs every direction file; --check works with those present")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def check() -> int:
    """Run every check that needs no fit, on the direction files that exist.

    Returns:
        0 if every check passes.

    Raises:
        ValueError: If a check fails.
    """
    check_settings_and_arms()
    present = tuple(
        height for height in HEIGHTS_M if (CERRA_DIR / CERRA_DIRECTION_FILES[height]).exists()
    )
    lines = check_direction_files(heights=present)
    sites = _wind_sites()
    cells = derive_nearest_cells(grid=pl.read_parquet(GRID_PATH), sites=sites)
    wind = read_cerra_wind(directory=CERRA_DIR, cells=cells)
    direction = read_cerra_direction(directory=CERRA_DIR, cells=cells, heights=list(present))
    half_hourly = read_half_hourly_power(sites=sites)
    frame = build_rows(
        wind=wind, direction=direction, half_hourly=half_hourly, sites=sites, heights=present
    )
    shared = check_same_rows_as_prior(frame=frame)
    shares = check_synthetic_shares(frame=frame)
    refuse_to_overwrite(paths=[OUTPUT_DIR / name for name in OUTPUT_NAMES])
    out = [
        "arm and setting checks passed",
        *lines,
        f"row set: {frame.height:,} rows, {shared:,} keys equal to the prerequisite study's",
        f"synthetic rule shares: {shares}",
        f"heights present: {present}; missing files: {missing_direction_files() or 'none'}",
        "outputs: none exist",
    ]
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def main() -> int:
    """Run the checks, fit every arm, bootstrap every contrast, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; read no data.")
    parser.add_argument("--check", action="store_true", help="Run every check; fit nothing.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild the report and intervals from the saved losses.",
    )
    arguments = parser.parse_args()
    if arguments.dry_run:
        return dry_run()
    if arguments.check:
        return check()

    missing = missing_direction_files()
    if missing:
        msg = f"direction files not downloaded yet: {missing}"
        raise FileNotFoundError(msg)
    job_list = jobs()
    check_settings_and_arms()
    sites = _wind_sites()
    cells = derive_nearest_cells(grid=pl.read_parquet(GRID_PATH), sites=sites)
    wind = read_cerra_wind(directory=CERRA_DIR, cells=cells)
    direction = read_cerra_direction(directory=CERRA_DIR, cells=cells, heights=list(HEIGHTS_M))
    half_hourly = read_half_hourly_power(sites=sites)
    file_lines = check_direction_files(heights=HEIGHTS_M)
    frame = build_rows(
        wind=wind, direction=direction, half_hourly=half_hourly, sites=sites, heights=HEIGHTS_M
    )
    check_same_rows_as_prior(frame=frame)
    shares = check_synthetic_shares(frame=frame)
    fingerprint = _fingerprint(frame=frame, job_list=job_list)
    paths = {name: OUTPUT_DIR / name for name in OUTPUT_NAMES}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if arguments.report_only:
        if paths["losses.fingerprint"].read_text().strip() != fingerprint:
            msg = (
                "--report-only: the saved losses were fitted on a different row set, column set, "
                "seed set, or hyperparameter setting than this code now produces"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(paths["losses.parquet"])
        refuse_to_overwrite(
            paths=[paths["report.md"], paths["intervals.parquet"], paths["absolute.parquet"]]
        )
    else:
        refuse_to_overwrite(paths=list(paths.values()))
        losses = run_all(dataset=frame, jobs=job_list, max_workers=MAX_WORKERS)
        check_same_rows(losses=losses)
        losses.write_parquet(paths["losses.parquet"])
        paths["losses.fingerprint"].write_text(fingerprint)
        frame.write_parquet(paths["rows.parquet"])

    farms = sorted(frame["site"].unique().to_list())
    intervals = contrast_records(losses=losses, sites=farms)
    absolute = absolute_records(losses=losses)
    intervals.write_parquet(paths["intervals.parquet"])
    absolute.write_parquet(paths["absolute.parquet"])
    report = "\n".join(
        report_lines(
            frame=frame,
            intervals=intervals,
            absolute=absolute,
            job_list=job_list,
            file_lines=file_lines,
            prior_lines=prior_agreement(losses=losses),
            shares=shares,
        )
    )
    paths["report.md"].write_text(report)
    sys.stdout.write(report)
    check_positive_controls(intervals=intervals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
