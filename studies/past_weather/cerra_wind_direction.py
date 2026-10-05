"""Does CERRA's wind direction help predict wind power beyond CERRA's wind speed?

One-off throwaway script for issue 994 of `openclimatefix/nged-substation-forecast`.
The plan is `plans/study-994-cerra-wind-direction.md` and was committed before the first fit.
`cerra_wind_levels.py` (issue 957) is the prerequisite study: this script reuses its row set, folds,
seeds, bootstrap and gates, and adds the direction columns it lacked.

**Data.** `data/studies/weather/CERRA/`: CERRA's wind speed and wind direction at 10, 50, 75, 100,
and 150 m, at the 190 grid cells around the trial area, every 3 hours from 2019-09-01 to 2026-06-30.
Direction is in degrees clockwise from north, the direction the wind blows from. The files' circular
mean at 100 m is about 227 degrees, the south-westerly of the UK's prevailing wind, which confirms
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
Four positive controls inject a cut into the real power, `power_mw` times one minus the loss on a
rule's rows: a direction sector (`sector`, within 30 degrees of 255 degrees) or a signed veer of
at least 10 degrees, clockwise with height from 10 m to 100 m (`veer`), each at 40% and at 10%. The
report prints each rule's share of rows and mean injected effect in percentage points of capacity.
The run writes every output and then raises unless, on the 40% targets, `speed_100m_dir` beats
`speed_100m` on the sector target, and `veer_angle_10_100` and `veer_dir_10_100` each beat
`veer_dir_100` on the veer target, each by an interval below zero. The 10% targets are reported
only.

**Checks that need no fit.** `--dry-run` prints the arms, the number of fits, the output files and
which direction files are missing, reads no data and exits 0. `--check` reads the direction files
that exist and the private power table, builds the row set from the heights that exist, runs every
check below on them, and exits non-zero on a violation. Neither fits a model or writes an output.
The full run stops while any direction file is missing.

Run it with `uv run python studies/past_weather/cerra_wind_direction.py`. `--report-only`
rebuilds `report.md` and the interval tables from the saved losses, still checking the saved
fingerprint, and refuses only if one of those three files exists. A full run stops before any fit
while any output exists, until it is moved to a
`superseded/` subfolder. Only one agent may run it at a time, because every worktree shares one
data folder.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from cerra_wind_levels import (
    CERRA_DIR,
    GRID_PATH,
    MAX_CORES,
    MAX_WORKERS,
    NOISE_SEED,
    PRIMARY_SETTING,
    SENSITIVITY_SETTING,
    SHARED_FEATURES,
    Contrast,
    check_column_counts,
    check_same_rows,
    check_settings,
    read_half_hourly_power,
)
from ens_past_solar import _arm_columns_lines, _fingerprint
from studies.arm_runner import Job, add_time_features, run_all
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
from studies.pv_dataset import wind_sites
from studies.reanalysis_wind import (
    CERRA_DIRECTION_FILES,
    CERRA_FILES,
    derive_nearest_cells,
    join_centred_power,
    read_cerra_direction,
    read_cerra_wind,
)
from studies.sources import CERRA_WIND_DIRECTION_DIR, CERRA_WIND_LEVELS_DIR
from studies.wind_direction import shuffled_by_month, sine_cosine, veer_degrees
from weather_products import METRIC, PERCENTAGE_POINTS, _mae

_LOG: Final[logging.Logger] = logging.getLogger("cerra_wind_direction")

OUTPUT_DIR: Final[Path] = CERRA_WIND_DIRECTION_DIR
"""Where the script writes its outputs, and a `superseded/` folder for re-runs. The folder must not
exist before the first run's outputs: `refuse_to_overwrite` stops on any file."""

PRIOR_ROWS_PATH: Final[Path] = CERRA_WIND_LEVELS_DIR / "rows.parquet"
PRIOR_LOSSES_PATH: Final[Path] = CERRA_WIND_LEVELS_DIR / "losses.parquet"
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
"""A veer is `veer_degrees` of the 100 m direction over the 10 m direction, positive when the wind
turns clockwise with height (veering). The veer rule cuts only a signed veer of at least
`VEER_THRESHOLD_DEG`, not a backing (anticlockwise) turn of the same size."""

SYNTHETIC_SHARE_RANGE: Final[tuple[float, float]] = (0.10, 0.50)
"""The share of rows an injected rule may affect. Outside it the control is vacuous or the whole
target."""


class Injection(NamedTuple):
    """One positive control: a rule, and the share of real power it removes on the rule's rows."""

    target: str
    rule: str
    loss: float


INJECTIONS: Final[tuple[Injection, ...]] = (
    Injection("injected_sector_40", "sector", 0.40),
    Injection("injected_sector_10", "sector", 0.10),
    Injection("injected_veer_40", "veer", 0.40),
    Injection("injected_veer_10", "veer", 0.10),
)
"""Each target is the real `power_mw` times `1 - loss` on the rule's rows and unchanged elsewhere,
so every real feature of the target stays. Each target name is also its setting name."""

CONTROL_SETTINGS: Final[tuple[str, ...]] = tuple(i.target for i in INJECTIONS)

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
VEER_CONTROL_ARMS: Final[tuple[str, ...]] = (
    "veer_dir_100",
    "veer_dir_10_100",
    "veer_angle_10_100",
    "veer_dir_all5",
)

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
POSITIVE_CONTROLS: Final[tuple[tuple[Contrast, str], ...]] = tuple(
    (contrast, injection.target)
    for injection in INJECTIONS
    for contrast in (
        (Contrast("speed_100m_dir", "speed_100m", "direction where power has a sector cut", False),)
        if injection.rule == "sector"
        else (
            Contrast(
                "veer_angle_10_100",
                "veer_dir_100",
                "the veer angle where power has a veer cut",
                False,
            ),
            Contrast(
                "veer_dir_10_100",
                "veer_dir_100",
                "two raw directions where power has a veer cut",
                False,
            ),
            Contrast(
                "veer_dir_all5",
                "veer_dir_100",
                "five raw directions where power has a veer cut",
                False,
            ),
        )
    )
)
"""Every positive control's contrast on every injected target."""

GATING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("speed_100m_dir", "injected_sector_40"),
    ("veer_angle_10_100", "injected_veer_40"),
    ("veer_dir_10_100", "injected_veer_40"),
)
"""The (treatment, target) pairs whose upper 95% bound must lie below zero. The 10% cuts, and
`veer_dir_all5` against `veer_dir_100`, are reported and do not gate the run. If a gate fails, the
run has already written every output, and the page reports the instrument's bound on the real
target instead of reading a null."""


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
    columns: list[pl.Expr] = []
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
        degrees = pl.Series(f"shuffled_{height}m", shuffled)
        sin, cos = sine_cosine(direction_deg=pl.lit(degrees))
        sin_name, cos_name = _noise_pair(height_m=height)
        columns += [sin.alias(sin_name), cos.alias(cos_name)]
    return frame.with_columns(columns)


def rule_masks(*, frame: pl.DataFrame) -> dict[str, pl.Series]:
    """Return which rows each injection rule cuts.

    Args:
        frame: Rows carrying `wind_direction_100m` and `veer_deg_10_100m`.

    Returns:
        `sector` to the rows whose 100 m direction lies within `SECTOR_HALF_WIDTH_DEG` of
        `SECTOR_CENTRE_DEG`, and `veer` to the rows whose signed veer from 10 m to 100 m is at least
        `VEER_THRESHOLD_DEG` (clockwise with height).
    """
    distance = ((pl.col("wind_direction_100m") - SECTOR_CENTRE_DEG + 180.0) % 360.0 - 180.0).abs()
    masks = frame.select(
        sector=distance < SECTOR_HALF_WIDTH_DEG,
        veer=pl.col("veer_deg_10_100m") >= VEER_THRESHOLD_DEG,
    )
    return {"sector": masks["sector"], "veer": masks["veer"]}


def with_injected_targets(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the positive controls' targets, each the real power cut on a rule's rows.

    Args:
        frame: Rows carrying `power_mw`, `wind_direction_100m` and `veer_deg_10_100m`.

    Returns:
        The frame with one column per `INJECTIONS` target: `power_mw * (1 - loss)` on the rule's
        rows and `power_mw` elsewhere. Only an arm that can read the rule's direction columns can
        reproduce the cut.
    """
    masks = rule_masks(frame=frame)
    return frame.with_columns(
        pl.when(masks[injection.rule])
        .then(pl.col("power_mw") * (1.0 - injection.loss))
        .otherwise(pl.col("power_mw"))
        .alias(injection.target)
        for injection in INJECTIONS
    )


def injected_effects(*, frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Measure each injection's size.

    Args:
        frame: Rows carrying `power_mw`, `effective_capacity_mw` and the `INJECTIONS` targets.

    Returns:
        Each target to the share of rows its rule cuts and the mean injected effect in percentage
        points of capacity, the mean over all rows of (power minus target) over capacity times 100.
    """
    masks = rule_masks(frame=frame)
    return {
        injection.target: (
            float(masks[injection.rule].to_numpy().mean()),
            float(
                ((frame["power_mw"] - frame[injection.target]) / frame["effective_capacity_mw"])
                .to_numpy()
                .mean()
            )
            * PERCENTAGE_POINTS,
        )
        for injection in INJECTIONS
    }


def check_synthetic_shares(*, frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Stop unless each injection rule cuts a share of rows inside `SYNTHETIC_SHARE_RANGE`.

    Args:
        frame: Rows carrying the columns `injected_effects` reads.

    Returns:
        `injected_effects`' result.

    Raises:
        ValueError: Naming each target whose share is outside the range.
    """
    effects = injected_effects(frame=frame)
    low, high = SYNTHETIC_SHARE_RANGE
    bad = {target: share for target, (share, _) in effects.items() if not low <= share <= high}
    if bad:
        msg = f"an injection rule affects a share of rows outside [{low}, {high}]: {bad}"
        raise ValueError(msg)
    return effects


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
        .pipe(lambda rows: add_time_features(dataset=rows))
        .sort("site", "time")
    )
    frame = with_shuffled_direction(
        frame=with_direction_columns(frame=joined, heights=heights), heights=heights
    )
    frame = with_injected_targets(frame=frame)
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
    """Return every fit: the arms at both settings, and the four positive controls.

    Returns:
        `REAL_ARMS` at `PRIMARY_SETTING` and at `SENSITIVITY_SETTING`, and for each injection its
        control arms (`SECTOR_CONTROL_ARMS` or `VEER_CONTROL_ARMS`) on its target, at the primary
        hyperparameters, with the injection's target name as the setting.
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
    for injection in INJECTIONS:
        arms = SECTOR_CONTROL_ARMS if injection.rule == "sector" else VEER_CONTROL_ARMS
        fits += [
            (
                arm,
                injection.target,
                injection.target,
                columns[arm],
                PRIMARY_HYPER_PARAMETERS,
                False,
            )
            for arm in arms
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
        scope: A label for the rows: `all`, a farm label, or a calendar year.

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


def _year(*, losses: pl.DataFrame, year: int) -> pl.DataFrame:
    """Restrict losses to one calendar year.

    Args:
        losses: Per-row losses carrying `time`.
        year: The calendar year.

    Returns:
        The rows in that year.
    """
    return losses.filter(pl.col("time").dt.year() == year)


def contrast_records(*, losses: pl.DataFrame, sites: list[str]) -> pl.DataFrame:
    """Compute every interval the report and the charts quote.

    Args:
        losses: Every arm's losses, every setting.
        sites: The farm labels.

    Returns:
        One row per (setting, scope, contrast). The planned contrasts, the exploratory contrasts and
        the negative controls run on all rows at both real-target settings. The planned contrasts
        also run per farm and per full calendar year at the primary setting (exploratory splits).
        Each positive control runs on its own injected target.
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
    years = full_years(times=primary["time"])
    splits += [(f"year {year}", _year(losses=primary, year=year)) for year in years]
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


PRIOR_TOLERANCE: Final[float] = 1e-6
"""The largest absolute difference in a per-row loss, as a fraction of capacity, that
`prior_agreement` accepts between this run's speed-only arms and the prerequisite study's."""


def prior_agreement(*, losses: pl.DataFrame, prior_path: Path = PRIOR_LOSSES_PATH) -> list[str]:
    """Stop unless the two speed-only arms reproduce the prerequisite study's per-row losses.

    The two arms have the same columns, rows, folds, seeds and settings as in the prerequisite
    study, so the losses should be identical on the same device.

    Args:
        losses: This run's per-row losses.
        prior_path: The prerequisite study's saved `losses.parquet`.

    Returns:
        One report line per arm with the largest absolute difference and the number of rows
        compared.

    Raises:
        FileNotFoundError: If the prerequisite study's losses are absent.
        ValueError: If an arm's rows do not all match, or the largest difference exceeds
            `PRIOR_TOLERANCE`.
    """
    if not prior_path.exists():
        msg = f"{prior_path} is absent, so the speed-only arms cannot be checked against it"
        raise FileNotFoundError(msg)
    prior = pl.read_parquet(prior_path).filter(pl.col("setting") == PRIMARY_SETTING)
    lines = []
    for arm in ("speed_10m", "speed_100m"):
        ours = losses.filter((pl.col("setting") == PRIMARY_SETTING) & (pl.col("arm") == arm))
        theirs = prior.filter(pl.col("arm") == arm)
        joined = ours.join(theirs, on=["site", "time", "seed"], suffix="_prior")
        if not ours.height or joined.height != ours.height or joined.height != theirs.height:
            msg = (
                f"{arm} does not reproduce the prerequisite study: {joined.height:,} rows joined "
                f"of {ours.height:,} here and {theirs.height:,} there"
            )
            raise ValueError(msg)
        gap = float(np.abs(joined[METRIC].to_numpy() - joined[f"{METRIC}_prior"].to_numpy()).max())
        if not gap <= PRIOR_TOLERANCE:
            msg = (
                f"{arm} does not reproduce the prerequisite study: largest difference {gap:.3g} "
                f"(NaN counts as a failure) against a tolerance of {PRIOR_TOLERANCE}"
            )
            raise ValueError(msg)
        lines.append(
            f"- `{arm}`: largest absolute difference from the prerequisite study's per-row loss "
            f"{gap:.3g} over {joined.height:,} rows."
        )
    return lines


ERA_MEAN_DEG: Final[float] = 30.0
ERA_VEER_P95_DEG: Final[float] = 10.0
"""A full year `differs from its neighbours` when its 100 m circular mean is more than
`ERA_MEAN_DEG` degrees from the circular mean of the other full years' circular means, or its
veer 95th percentile is more than `ERA_VEER_P95_DEG` degrees from the median of theirs. The full
years' own circular means span 23 degrees, so 30 degrees sits outside that spread."""


def full_years(*, times: pl.Series) -> list[int]:
    """List the calendar years with rows in all 12 months.

    Args:
        times: A datetime series.

    Returns:
        The years in which every month appears. A year with fewer months (the record starts in
        September 2019 and ends in June 2026) is partial and is left out.
    """
    counts = (
        times.to_frame("time")
        .group_by(year=pl.col("time").dt.year())
        .agg(months=pl.col("time").dt.month().n_unique())
    )
    return sorted(counts.filter(pl.col("months") == MONTHS_PER_YEAR)["year"].to_list())


MONTHS_PER_YEAR: Final[int] = 12


def direction_by_year(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Summarise the 100 m direction and the veer for each calendar year.

    The prerequisite study's era check scans CERRA's speed for a step at any month and names no
    production-stream boundary, so this script compares full years. A change of production stream
    inside the record would show as a full year that `differs` from the other full years under the
    rule at `ERA_MEAN_DEG`. A partial year is listed and never flagged, because its season mix
    differs from a full year's.

    Args:
        frame: Rows carrying `time`, `wind_direction_100m` and `veer_deg_10_100m`.

    Returns:
        One row per year with `rows`, `months`, `circular_mean_deg` of the 100 m direction, the
        `veer_median_deg`, `veer_p95_deg` and `veer_share_ge_threshold` of the signed veer, and
        `differs` (null for a partial year).
    """
    sin, cos = sine_cosine(direction_deg=pl.col("wind_direction_100m"))
    table = (
        frame.group_by(year=pl.col("time").dt.year())
        .agg(
            rows=pl.len(),
            months=pl.col("time").dt.month().n_unique(),
            mean_sin=sin.mean(),
            mean_cos=cos.mean(),
            veer_median_deg=pl.col("veer_deg_10_100m").median(),
            veer_p95_deg=pl.col("veer_deg_10_100m").quantile(0.95),
            veer_share_ge_threshold=(pl.col("veer_deg_10_100m") >= VEER_THRESHOLD_DEG).mean(),
        )
        .with_columns(
            circular_mean_deg=pl.arctan2(pl.col("mean_sin"), pl.col("mean_cos")).degrees() % 360.0
        )
        .sort("year")
    )
    rows = table.to_dicts()
    full = [row for row in rows if row["months"] == MONTHS_PER_YEAR]
    for row in rows:
        others = [other for other in full if other["year"] != row["year"]]
        if row["months"] != MONTHS_PER_YEAR or not others:
            row["differs"] = None
            continue
        others_radians = np.radians([other["circular_mean_deg"] for other in others])
        others_mean_deg = float(
            np.degrees(np.arctan2(np.sin(others_radians).mean(), np.cos(others_radians).mean()))
        )
        mean_gap = abs((row["circular_mean_deg"] - others_mean_deg + 180.0) % 360.0 - 180.0)
        veer_gap = abs(
            row["veer_p95_deg"] - float(np.median([other["veer_p95_deg"] for other in others]))
        )
        row["differs"] = mean_gap > ERA_MEAN_DEG or veer_gap > ERA_VEER_P95_DEG
    return pl.DataFrame(rows, schema_overrides={"differs": pl.Boolean}).select(
        "year",
        "rows",
        "months",
        "circular_mean_deg",
        "veer_median_deg",
        "veer_p95_deg",
        "veer_share_ge_threshold",
        "differs",
    )


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
    effects: dict[str, tuple[float, float]],
) -> list[str]:
    """Assemble the report the page quotes.

    Args:
        frame: The main row set.
        intervals: `contrast_records`' frame.
        absolute: `absolute_records`' frame.
        job_list: Every fit.
        file_lines: `check_direction_files`' lines.
        prior_lines: `prior_agreement`'s lines.
        effects: `check_synthetic_shares`' result.

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
        "#### Injected targets (positive controls)",
        "",
        (
            "Each target is the real power times one minus the loss on the rule's rows. The "
            "veer rule cuts a signed veer of at least "
            f"{VEER_THRESHOLD_DEG:.0f} degrees, clockwise with height."
        ),
        "",
        *(
            f"- `{target}`: the rule cuts {share:.1%} of rows, a mean injected effect of "
            f"{effect:.3f} pp of capacity over all rows."
            for target, (share, effect) in effects.items()
        ),
        "",
        "#### Direction and veer by calendar year (era check)",
        "",
        (
            "A year with fewer than 12 months is partial, is never flagged, and is left out of the "
            f"per-year contrast splits. A full year differs when its 100 m circular mean is over "
            f"{ERA_MEAN_DEG:.0f} degrees, or its veer 95th percentile over {ERA_VEER_P95_DEG:.0f} "
            "degrees, from the other full years (circular mean for direction, median for veer)."
        ),
        "",
        (
            "| Year | Rows | Months | 100 m circular mean (degrees) | Veer median "
            "| Veer 95th percentile | Share at or above threshold "
            "| Differs from the other full years? |"
        ),
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {row['year']}{'' if row['differs'] is not None else ' (partial)'} "
            f"| {row['rows']:,} | {row['months']} | {row['circular_mean_deg']:.1f} "
            f"| {row['veer_median_deg']:.2f} | {row['veer_p95_deg']:.2f} "
            f"| {row['veer_share_ge_threshold']:.1%} "
            f"| {'n/a' if row['differs'] is None else ('**yes**' if row['differs'] else 'no')} |"
            for row in direction_by_year(frame=frame).iter_rows(named=True)
        ),
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
            "Planned contrasts by farm and calendar year (exploratory splits)",
            intervals.filter(pl.col("scope") != "all"),
        ),
        (
            "Negative controls, both settings",
            pooled.filter(pl.col("treatment").is_in([c.treatment for c in NEGATIVE_CONTROLS])),
        ),
        (
            "Positive controls (injected targets)",
            pooled.filter(pl.col("setting").is_in(CONTROL_SETTINGS)),
        ),
    ):
        lines += [f"#### {title}", "", *_contrast_lines(records=selection), ""]
    for injection in INJECTIONS:
        setting = injection.target
        arms = SECTOR_CONTROL_ARMS if injection.rule == "sector" else VEER_CONTROL_ARMS
        lines += [
            f"#### Absolute error on the injected target, `{setting}`",
            "",
            *_absolute_lines(absolute=absolute, setting=setting, arms=arms),
            "",
        ]
    return lines


def check_positive_controls(*, intervals: pl.DataFrame) -> None:
    """Stop unless the gating positive controls recover their injected effects.

    Args:
        intervals: `contrast_records`' frame.

    Raises:
        ValueError: If the pairs checked differ from `GATING_CONTRASTS`, or naming each gating
            control whose upper 95% bound is not below zero.
    """
    failed = []
    checked: set[tuple[str, str]] = set()
    for contrast, setting in POSITIVE_CONTROLS:
        if (contrast.treatment, setting) not in GATING_CONTRASTS:
            continue
        checked.add((contrast.treatment, setting))
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
    if checked != set(GATING_CONTRASTS):
        msg = f"the gate checked {sorted(checked)} but must check {sorted(GATING_CONTRASTS)}"
        raise ValueError(msg)
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
    sites = wind_sites()
    cells = derive_nearest_cells(grid=pl.read_parquet(GRID_PATH), sites=sites)
    wind = read_cerra_wind(directory=CERRA_DIR, cells=cells)
    direction = read_cerra_direction(directory=CERRA_DIR, cells=cells, heights=list(present))
    half_hourly = read_half_hourly_power(sites=sites)
    frame = build_rows(
        wind=wind, direction=direction, half_hourly=half_hourly, sites=sites, heights=present
    )
    shared = check_same_rows_as_prior(frame=frame)
    effects = check_synthetic_shares(frame=frame)
    if not PRIOR_LOSSES_PATH.exists():
        msg = f"{PRIOR_LOSSES_PATH} is absent, so the run's agreement check could not pass"
        raise FileNotFoundError(msg)
    refuse_to_overwrite(paths=[OUTPUT_DIR / name for name in OUTPUT_NAMES])
    out = [
        "arm and setting checks passed",
        *lines,
        f"row set: {frame.height:,} rows, {shared:,} keys equal to the prerequisite study's",
        *(
            f"{target}: rule cuts {share:.1%} of rows, mean injected effect {effect:.3f} pp"
            for target, (share, effect) in effects.items()
        ),
        *(
            f"{row['year']} ({row['months']} months): {row['rows']:,} rows, 100 m circular mean "
            f"{row['circular_mean_deg']:.0f} degrees, veer 95th percentile "
            f"{row['veer_p95_deg']:.1f} degrees"
            for row in direction_by_year(frame=frame).iter_rows(named=True)
        ),
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
    sites = wind_sites()
    cells = derive_nearest_cells(grid=pl.read_parquet(GRID_PATH), sites=sites)
    wind = read_cerra_wind(directory=CERRA_DIR, cells=cells)
    direction = read_cerra_direction(directory=CERRA_DIR, cells=cells, heights=list(HEIGHTS_M))
    half_hourly = read_half_hourly_power(sites=sites)
    file_lines = check_direction_files(heights=HEIGHTS_M)
    frame = build_rows(
        wind=wind, direction=direction, half_hourly=half_hourly, sites=sites, heights=HEIGHTS_M
    )
    check_same_rows_as_prior(frame=frame)
    effects = check_synthetic_shares(frame=frame)
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
        refuse_to_overwrite(
            paths=[paths["report.md"], paths["intervals.parquet"], paths["absolute.parquet"]]
        )
        losses = pl.read_parquet(paths["losses.parquet"])
    else:
        refuse_to_overwrite(paths=list(paths.values()))
        losses = run_all(dataset=frame, jobs=job_list, max_workers=MAX_WORKERS)
        losses.write_parquet(paths["losses.parquet"])
        frame.write_parquet(paths["rows.parquet"])
        check_same_rows(losses=losses)
        paths["losses.fingerprint"].write_text(fingerprint)

    farms = sorted(frame["site"].unique().to_list())
    prior_lines = prior_agreement(losses=losses)
    intervals = contrast_records(losses=losses, sites=farms)
    absolute = absolute_records(losses=losses)
    report = "\n".join(
        report_lines(
            frame=frame,
            intervals=intervals,
            absolute=absolute,
            job_list=job_list,
            file_lines=file_lines,
            prior_lines=prior_lines,
            effects=effects,
        )
    )
    intervals.write_parquet(paths["intervals.parquet"])
    absolute.write_parquet(paths["absolute.parquet"])
    paths["report.md"].write_text(report)
    sys.stdout.write(report)
    check_positive_controls(intervals=intervals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
