"""The known-answer injection builder of experiment 00045.

Every unit is one Primary, one solar farm, one path seed, and one of 14 cases. The unit's power is a
semi-synthetic baseline (the Primary's real power, minus its smooth fitted PV, plus a known
constant PV path) plus a known change: scaled farm output for ramps, steps, falls, and the
constant path, a share of another Primary's power for transfers, and a daytime load for the
seasonal-load cases. The true weekly capacity path of every unit is known by construction, and
each fit of baseline plus injection is scored against it. A variant that adds the same change to
the real power instead of the semi-synthetic baseline is also built; its truth is the paired
difference of two fits.

Farms A to C and path seeds 0 to 3 are the tuning units; farms D to F and seeds 1000 to 1003 are
the scoring units. This module only builds units and runs fits on the units it is told to run; the
caller decides which split to fit.

A farm-week with output missing on more than 2 days is dropped from the farm's units: the
Primary's rows of that week leave the fit, so the week is unobserved, and the week takes no part
in any score. Primary half-hours without farm output (meter dropouts) leave the fit too.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python \
        ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_tune.py
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import e00045_common as common
import e00045_metrics as metrics
import e00045_model as model
import numpy as np
import polars as pl

CaseName = Literal[
    "ramp_25",
    "ramp_50",
    "step_05",
    "step_10",
    "step_20",
    "step_30",
    "fall_30_8",
    "fall_30_26",
    "fall_15_26",
    "transfer_inside_mask",
    "transfer_outside_mask",
    "seasonal_load_independent",
    "seasonal_load_sunshine",
    "constant_path",
    "ratio_constant_farm",
]
CASES: Final[tuple[CaseName, ...]] = (
    "ramp_25",
    "ramp_50",
    "step_05",
    "step_10",
    "step_20",
    "step_30",
    "fall_30_8",
    "fall_30_26",
    "fall_15_26",
    "transfer_inside_mask",
    "transfer_outside_mask",
    "seasonal_load_independent",
    "seasonal_load_sunshine",
    "constant_path",
)
"""The 14 scored cases: 2 ramps, 4 steps, 3 temporary falls, 2 transfers, 2 seasonal loads, and 1
constant path. The constant path is built on the placebo sky: the PV shape is permuted within each
month and the known constant capacity times the permuted shape is added to the real power, so the
baseline holds no real growth and the known path is exactly constant."""
RATIO_CASES: Final[tuple[CaseName, ...]] = ("ratio_constant_farm",)
"""Auxiliary units, not scored: a constant capacity injected from a farm on the semi-synthetic
baseline, which measures each Primary's recovery ratio (the farm's shape differs from `u`)."""
RAMP_SHARES: Final[dict[str, float]] = {"ramp_25": 0.25, "ramp_50": 0.50}
STEP_SHARES: Final[dict[str, float]] = {
    "step_05": 0.05,
    "step_10": 0.10,
    "step_20": 0.20,
    "step_30": 0.30,
}
FALLS: Final[dict[str, tuple[float, int]]] = {
    "fall_30_8": (0.30, 8),
    "fall_30_26": (0.30, 26),
    "fall_15_26": (0.15, 26),
}
"""Each temporary fall's depth (a share of the baseline capacity) and duration in weeks."""
RAMP_WEEKS: Final[int] = 52
TRANSFER_SHARE: Final[float] = 0.3
TRANSFER_DAYS: Final[int] = 30
SEASONAL_LOAD_SHARE: Final[float] = 0.10
"""Peak of the seasonal daytime load, as a share of the baseline capacity."""
LOAD_FIRST_HOUR: Final[int] = 8
LOAD_LAST_HOUR: Final[int] = 20
"""The load runs on local half-hours from 08:00 up to 20:00."""
BASELINE_CAPACITY_SEASON_YEAR: Final[int] = 2025
"""The season-year (2025/26) of experiment 00040's capacity that the baseline PV path equals."""
EXPERIMENT_00040_SCORES_PATH: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR / "00040-pv-capacity-error-bars" / "work" / "real_scores.parquet"
)

SplitType = Literal["tuning", "scoring"]
VariantType = Literal["semi_synthetic", "real_plus_injection"]
TUNING_FARMS: Final[tuple[str, ...]] = ("A", "B", "C")
SCORING_FARMS: Final[tuple[str, ...]] = ("D", "E", "F")
TUNING_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3)
SCORING_SEEDS: Final[tuple[int, ...]] = (1000, 1001, 1002, 1003)
TUNING_PLACEBO_SEEDS: Final[range] = range(100)
SCORING_PLACEBO_SEEDS: Final[range] = range(1000, 1100)
FARMS_BY_SPLIT: Final[dict[SplitType, tuple[str, ...]]] = {
    "tuning": TUNING_FARMS,
    "scoring": SCORING_FARMS,
}
SEEDS_BY_SPLIT: Final[dict[SplitType, tuple[int, ...]]] = {
    "tuning": TUNING_SEEDS,
    "scoring": SCORING_SEEDS,
}
FitSpec = tuple[model.FitName, float, float]
"""A fit's name and its `a` and `r` (ignored by the fits that do not use them)."""

MIN_USABLE_DAYS: Final[int] = common.MIN_USABLE_DAYS_PER_WEEK
MARGIN_START_WEEKS: Final[int] = 12
MARGIN_END_WEEKS: Final[int] = 26
"""A step or fall starts at least this many weeks after the first observed week, and a step at
least this many weeks before the last."""
DONOR_GROUPS: Final[tuple[frozenset[str], ...]] = (frozenset({"D15", "D18"}),)
"""Primaries that share a polygon (Wrangle T1 and T2) and so are never each other's donor."""


@dataclass(frozen=True)
class Unit:
    """One known-answer unit.

    Attributes:
        primary: Frame site code of the Primary.
        farm: Solar farm label (A to F) whose output is injected.
        seed: Path seed.
        case: Which of the 14 cases.
    """

    primary: str
    farm: str
    seed: int
    case: CaseName


def units_for(*, split: SplitType) -> list[Unit]:
    """Return every unit of a split: farms by seeds by Primaries by cases.

    Args:
        split: `"tuning"` (farms A to C, seeds 0 to 3) or `"scoring"` (farms D to F, seeds 1000
            to 1003).

    Returns:
        The units, in a fixed order.
    """
    return [
        Unit(primary=primary, farm=farm, seed=seed, case=case)
        for farm in FARMS_BY_SPLIT[split]
        for seed in SEEDS_BY_SPLIT[split]
        for primary in common.PRIMARIES
        for case in CASES
    ]


def assert_split(*, farm: str, seeds: tuple[int, ...], split: SplitType) -> None:
    """Raise when a farm or a seed does not belong to `split`, so tuning never touches scoring."""
    if farm not in FARMS_BY_SPLIT[split] or not set(seeds) <= set(SEEDS_BY_SPLIT[split]):
        raise ValueError(f"farm {farm} / seeds {seeds} are not {split} units")


@dataclass(frozen=True)
class Inputs:
    """The U1 tables and the Primaries' baseline capacities.

    Attributes:
        half_hourly: `inputs_half_hourly.parquet` (every Primary).
        weeks: `inputs_primary_weeks.parquet`.
        farm_half_hourly: Each farm's normalised output (`farm`, `time`, `farm_norm`).
        farm_weeks: One row per farm-week with its `dropped` flag.
        base_capacity: Each Primary's 2025/26 capacity from experiment 00040, MW.
    """

    half_hourly: pl.DataFrame
    weeks: pl.DataFrame
    farm_half_hourly: pl.DataFrame
    farm_weeks: pl.DataFrame
    base_capacity: dict[str, float]


def load_inputs() -> Inputs:
    """Read the U1 tables and experiment 00040's capacities."""
    scores = pl.read_parquet(EXPERIMENT_00040_SCORES_PATH).filter(
        pl.col("year") == BASELINE_CAPACITY_SEASON_YEAR
    )
    return Inputs(
        half_hourly=pl.read_parquet(common.HALF_HOURLY_PATH),
        weeks=pl.read_parquet(common.INPUTS_PATH),
        farm_half_hourly=pl.read_parquet(common.FARM_HALF_HOURLY_PATH).select(
            "farm", "time", "farm_norm"
        ),
        farm_weeks=pl.read_parquet(common.FARM_WEEKS_PATH),
        base_capacity=dict(zip(scores["site"], scores["C"], strict=True)),
    )


def donor_of(*, primary: str) -> str:
    """Return the Primary whose power a transfer moves: the next Primary outside its own group."""
    start = common.PRIMARIES.index(primary)
    for step in range(1, len(common.PRIMARIES)):
        candidate = common.PRIMARIES[(start + step) % len(common.PRIMARIES)]
        if not any(primary in group and candidate in group for group in DONOR_GROUPS):
            return candidate
    raise ValueError(primary)


@dataclass(frozen=True)
class PrimaryBase:
    """One Primary's usable half-hours with the semi-synthetic baseline power.

    Attributes:
        site: Frame site code.
        capacity: The baseline PV capacity, MW (experiment 00040's 2025/26 value).
        table: `model.usable_rows_table` plus `baseline_power` (MW): the real power plus the
            full-record free weekly fit's PV, minus the baseline capacity times `u`.
    """

    site: str
    capacity: float
    table: pl.DataFrame


def unconstrained_weekly_fit(*, design: model.Design) -> np.ndarray:
    """Return the least-squares weekly capacity of every observed week, with no sign bound.

    Args:
        design: A Primary's design.

    Returns:
        Capacity of every week, MW (NaN in weeks without rows).
    """
    problem = design.problem()
    observed = np.nonzero(problem.observed)[0]
    gram = problem.gram[np.ix_(observed, observed)] + model.RIDGE * np.eye(len(observed))
    fitted = np.full(design.n_weeks, np.nan)
    fitted[problem.span[0] + observed] = np.linalg.solve(gram, problem.rhs[observed])
    return fitted


def primary_base(*, inputs: Inputs, site: str) -> PrimaryBase:
    """Build the semi-synthetic baseline of one Primary.

    The baseline power is the real power minus the PV of the full-record season-year constant
    fit (`smooth * u`) plus the known baseline PV (`capacity * u`), in this model's sign
    convention where PV lowers power. Only a smooth PV path is removed, so the week-to-week noise
    of the real series that is correlated with `u` stays in the baseline: a free weekly fit of
    the baseline scatters around the known capacity by the weekly standard error. The known
    baseline path is the constant `capacity`.

    Args:
        inputs: The loaded tables.
        site: Frame site code.

    Returns:
        The Primary's baseline.
    """
    table = model.usable_rows_table(half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=site)
    fitted = model.fit_season_year_constant(design=model.design_from_rows_table(table=table))
    capacity = inputs.base_capacity[site]
    weeks = table["week"].to_numpy()
    baseline = table["power"].to_numpy() + (fitted[weeks] - capacity) * table["u"].to_numpy()
    return PrimaryBase(
        site=site, capacity=capacity, table=table.with_columns(baseline_power=pl.Series(baseline))
    )


def _usable_week_filter(*, table: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows of weeks with fewer than 3 usable days (a day needs 10 usable half-hours)."""
    days = table.with_columns(day=common.day_expr(time=pl.col("time")))
    usable_days = (
        days.group_by("week", "day")
        .agg(n=pl.len())
        .filter(pl.col("n") >= common.MIN_USABLE_HALF_HOURS_PER_DAY)
        .group_by("week")
        .agg(n_days=pl.len())
        .filter(pl.col("n_days") >= MIN_USABLE_DAYS)
    )
    return table.join(usable_days.select("week"), on="week", how="semi")


def unit_rows(
    *,
    base: PrimaryBase,
    inputs: Inputs,
    farm: str,
    masked_window: tuple[datetime, datetime] | None = None,
) -> pl.DataFrame:
    """Return the Primary's rows that survive a farm's gaps (and a mask window, when given).

    Rows are kept only where the farm has output, in weeks that the farm-week table does not drop,
    outside `masked_window` padded by 24 hours on each side, and in weeks that still hold 3 usable
    days. The farm's normalised output is added as `farm_norm`.

    Args:
        base: The Primary's baseline.
        inputs: The loaded tables.
        farm: Solar farm label.
        masked_window: A closed interval to remove, or `None`.

    Returns:
        The rows sorted by time, with the columns of `PrimaryBase.table` and `farm_norm`.
    """
    dropped = inputs.farm_weeks.filter((pl.col("farm") == farm) & pl.col("dropped"))["week"]
    rows = (
        base.table.join(
            inputs.farm_half_hourly.filter(pl.col("farm") == farm).select("time", "farm_norm"),
            on="time",
            how="inner",
        )
        .filter(~pl.col("week").is_in(dropped.implode()))
        .with_columns(pl.col("farm_norm").cast(pl.Float64))
    )
    if masked_window is not None:
        pad = timedelta(hours=common.MASK_PAD_HOURS)
        start, end = masked_window
        rows = rows.filter(
            ~(
                (pl.col("time") > start - pad)
                & (pl.col("time") <= end + pad + timedelta(minutes=30))
            )
        )
    return _usable_week_filter(table=rows).sort("time")


def observed_weeks(*, table: pl.DataFrame) -> np.ndarray:
    """Return a boolean mask over the weeks that hold rows."""
    mask = np.zeros(common.N_WEEKS, dtype=bool)
    mask[table["week"].unique().to_numpy()] = True
    return mask


@dataclass(frozen=True)
class Injection:
    """A known change, ready to apply to any table of the same Primary and farm.

    Attributes:
        increment: True change in capacity of every week over the baseline, MW.
        delta_for: Function from a table of rows to the MW to add to each row's power.
        masked_window: The interval whose rows leave the fit (padded by 24 hours), or `None`.
        fall: First week and number of weeks of a temporary fall, or `None`.
    """

    increment: np.ndarray
    delta_for: Callable[[pl.DataFrame], np.ndarray]
    masked_window: tuple[datetime, datetime] | None = None
    fall: tuple[int, int] | None = None


def _rng(*, unit: Unit) -> np.random.Generator:
    """Return the unit's own random generator, fixed by its Primary, farm, seed, and case."""
    return np.random.default_rng(
        [
            unit.seed,
            common.FARMS.index(unit.farm),
            common.PRIMARIES.index(unit.primary),
            (*CASES, *RATIO_CASES).index(unit.case),
        ]
    )


class UnbuildableUnitError(ValueError):
    """No week of the farm's record can hold the unit's injection."""


def _pick(*, rng: np.random.Generator, candidates: np.ndarray, fallback: np.ndarray) -> int:
    """Return a random candidate week, or a random fallback week when there is no candidate."""
    pool = candidates if candidates.size else fallback
    if not pool.size:
        raise UnbuildableUnitError("no week can hold the injection")
    return int(rng.choice(pool))


def _week_is_summer(*, week: int) -> bool:
    return common.week_start(week=week).month in common.SUMMER_MONTHS


def _first_week_of_month(*, week: int) -> bool:
    return common.week_start(week=week).day <= 7


def _farm_delta(*, weekly_change: np.ndarray) -> Callable[[pl.DataFrame], np.ndarray]:
    """Return the power change of a farm-shaped injection: PV rises when the weekly change is +."""

    def delta(table: pl.DataFrame) -> np.ndarray:
        return -weekly_change[table["week"].to_numpy()] * table["farm_norm"].to_numpy()

    return delta


def _transfer_weekly_fraction(*, start: datetime) -> np.ndarray:
    """Return, for each week, the share of its 7 days that fall in the 30-day transfer window."""
    fraction = np.zeros(common.N_WEEKS)
    for day in range(TRANSFER_DAYS):
        week = (start + timedelta(days=day) - common.FIRST_WEEK_START).days // 7
        if 0 <= week < common.N_WEEKS:
            fraction[week] += 1 / 7
    return fraction


def donor_power(*, inputs: Inputs, donor: str) -> pl.DataFrame:
    """Return a donor Primary's real power at every half-hour it reports (`time`, `donor_power`)."""
    return inputs.half_hourly.filter(
        (pl.col("site") == donor) & pl.col("power").is_not_null()
    ).select("time", donor_power=pl.col("power"))


def _seasonal_load_delta(
    *, capacity: float, sunshine: bool
) -> Callable[[pl.DataFrame], np.ndarray]:
    """Return the power change of a daytime load that peaks on 21 June at 10% of capacity."""

    def delta(table: pl.DataFrame) -> np.ndarray:
        local = table.with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )["local"]
        daytime = (
            (local.dt.hour() >= LOAD_FIRST_HOUR) & (local.dt.hour() < LOAD_LAST_HOUR)
        ).to_numpy()
        day_of_year = local.dt.ordinal_day().to_numpy()
        seasonal = 0.5 * (1 + np.cos(2 * np.pi * (day_of_year - 172) / 365.25))
        load = SEASONAL_LOAD_SHARE * capacity * seasonal * daytime
        if sunshine:
            clear = table["ghi_u_clear"].to_numpy()
            index = np.divide(
                table["ghi_u"].to_numpy(), clear, out=np.zeros(len(clear)), where=clear > 0
            )
            load = load * np.clip(index, 0.0, 1.2)
        return load

    return delta


def build_injection(
    *,
    unit: Unit,
    base: PrimaryBase,
    farm_rows: pl.DataFrame,
    inputs: Inputs,
) -> Injection:
    """Build a unit's known change.

    The path seed, the farm, the Primary, and the case fix every random choice. Steps and falls
    start only in weeks that the unit's rows observe. Half the steps start from April to September
    and half from October to March (seeds alternate); in each step case one seed in four starts in
    the first week of a month, so that the monthly demand cells meet the step.

    Args:
        unit: The unit.
        base: The Primary's baseline.
        farm_rows: `unit_rows(...)` without a mask window, which says which weeks are observed.
        inputs: The loaded tables.

    Returns:
        The change: its true weekly increment and how to turn it into a power change on any rows.
    """
    rng = _rng(unit=unit)
    capacity = base.capacity
    observed = observed_weeks(table=farm_rows)
    weeks = np.nonzero(observed)[0]
    first, last = int(weeks[0]), int(weeks[-1])
    grid = np.arange(common.N_WEEKS)
    increment = np.zeros(common.N_WEEKS)
    seed_index = unit.seed % 1000
    case = unit.case
    if case in RAMP_SHARES:
        starts = weeks[(weeks >= first + 6) & (weeks <= last - RAMP_WEEKS - 6)]
        start = _pick(rng=rng, candidates=starts, fallback=weeks[weeks <= last - RAMP_WEEKS])
        increment = RAMP_SHARES[case] * capacity * np.clip((grid - start) / RAMP_WEEKS, 0.0, 1.0)
    elif case in STEP_SHARES:
        size_index = list(STEP_SHARES).index(case)
        summer = seed_index % 2 == 0
        on_boundary = seed_index == (2 if size_index % 2 == 0 else 3)
        eligible = weeks[(weeks >= first + MARGIN_START_WEEKS) & (weeks <= last - MARGIN_END_WEEKS)]
        in_season = np.array([_week_is_summer(week=int(w)) == summer for w in eligible], dtype=bool)
        candidates = eligible[in_season]
        if on_boundary:
            first_of_month = np.array(
                [_first_week_of_month(week=int(w)) for w in candidates], dtype=bool
            )
            candidates = candidates[first_of_month]
        start = _pick(rng=rng, candidates=candidates, fallback=eligible)
        increment = STEP_SHARES[case] * capacity * (grid >= start)
    elif case in FALLS:
        depth, duration = FALLS[case]
        starts = np.array(
            [
                w
                for w in weeks
                if first + MARGIN_START_WEEKS <= w <= last - duration - MARGIN_START_WEEKS
                and observed[w : w + duration].mean() >= 0.75
                and observed[max(0, w - 8) : w].sum() >= 4
            ],
            dtype=int,
        )
        start = _pick(rng=rng, candidates=starts, fallback=weeks[weeks <= last - duration])
        increment = -depth * capacity * ((grid >= start) & (grid < start + duration))
        return Injection(
            increment=increment,
            delta_for=_farm_delta(weekly_change=increment),
            fall=(start, duration),
        )
    elif case == "ratio_constant_farm":
        increment = np.full(common.N_WEEKS, capacity)
    elif case.startswith("transfer"):
        return _transfer_injection(unit=unit, base=base, rng=rng, weeks=weeks, inputs=inputs)
    else:
        return Injection(
            increment=increment,
            delta_for=_seasonal_load_delta(
                capacity=capacity, sunshine=case == "seasonal_load_sunshine"
            ),
        )
    return Injection(increment=increment, delta_for=_farm_delta(weekly_change=increment))


def _transfer_injection(
    *,
    unit: Unit,
    base: PrimaryBase,
    rng: np.random.Generator,
    weeks: np.ndarray,
    inputs: Inputs,
) -> Injection:
    """Build a 30-day transfer of 0.3 of the donor's net power into the Primary.

    A transfer outside the mask is a change of capacity that the fit sees: its true increment is
    0.3 of the donor's baseline capacity, scaled by the share of each week the window covers. A
    transfer inside the mask is removed from the fit together with 24 hours on either side, so its
    true increment is zero.
    """
    donor = donor_of(primary=unit.primary)
    starts = weeks[(weeks >= weeks[0] + 8) & (weeks <= weeks[-1] - 8)]
    start_week = _pick(rng=rng, candidates=starts, fallback=weeks)
    start = common.week_start(week=start_week) + timedelta(days=int(rng.integers(0, 7)))
    window = (start, start + timedelta(days=TRANSFER_DAYS))
    donor_table = donor_power(inputs=inputs, donor=donor)

    def delta(table: pl.DataFrame) -> np.ndarray:
        joined = table.select("time").join(donor_table, on="time", how="left")
        inside = ((joined["time"] > window[0]) & (joined["time"] <= window[1])).to_numpy()
        return TRANSFER_SHARE * np.nan_to_num(joined["donor_power"].to_numpy(), nan=0.0) * inside

    inside_mask = unit.case == "transfer_inside_mask"
    increment = (
        np.zeros(common.N_WEEKS)
        if inside_mask
        else TRANSFER_SHARE
        * inputs.base_capacity[donor]
        * _transfer_weekly_fraction(start=window[0])
    )
    return Injection(
        increment=increment, delta_for=delta, masked_window=window if inside_mask else None
    )


@dataclass(frozen=True)
class UnitFit:
    """The fits of one unit: the design, its power, and the fitted paths.

    Attributes:
        increment: True weekly increment, MW.
        observed: Mask of the weeks that hold rows in the fitted design.
        paths: Fitted path of every requested fit, MW (raw: no recovery-ratio division).
        baseline_paths: The same fits on the real power alone (no injection), same design; only
            for the real-plus-injection variant, whose truth is the paired difference. Empty for
            the semi-synthetic variant, which is scored as the fit of baseline plus injection
            against the known path.
        fall: First week and duration of a temporary fall, or `None`.
    """

    increment: np.ndarray
    observed: np.ndarray
    paths: dict[FitSpec, np.ndarray]
    baseline_paths: dict[FitSpec, np.ndarray]
    fall: tuple[int, int] | None


def _fit_all(*, design: model.Design, fits: tuple[FitSpec, ...]) -> dict[FitSpec, np.ndarray]:
    return {spec: model.FITS[spec[0]](design=design, a=spec[1], r=spec[2]) for spec in fits}


def fit_unit(
    *,
    unit: Unit,
    base: PrimaryBase,
    inputs: Inputs,
    farm_rows: pl.DataFrame,
    fits: tuple[FitSpec, ...],
    variant: VariantType,
    shared_baseline: dict[FitSpec, np.ndarray] | None = None,
) -> UnitFit:
    """Inject a unit's change into its baseline, and run the fits with and without the injection.

    Args:
        unit: The unit.
        base: The Primary's baseline.
        inputs: The loaded tables.
        farm_rows: `unit_rows(...)` without a mask window.
        fits: The fits to run.
        variant: `"semi_synthetic"` adds the change to the baseline power, `"real_plus_injection"`
            to the real power.
        shared_baseline: The fits of the baseline on `farm_rows`, which every unit without a mask
            window shares; they are computed when it is `None`.

    Returns:
        The fitted paths with and without the injection.
    """
    if unit.case == "constant_path":
        return _fit_constant_path_on_placebo_sky(
            unit=unit, base=base, farm_rows=farm_rows, fits=fits
        )
    injection = build_injection(unit=unit, base=base, farm_rows=farm_rows, inputs=inputs)
    rows = (
        farm_rows
        if injection.masked_window is None
        else unit_rows(
            base=base, inputs=inputs, farm=unit.farm, masked_window=injection.masked_window
        )
    )
    column = "baseline_power" if variant == "semi_synthetic" else "power"
    power0 = rows[column].to_numpy().astype(float)
    power1 = power0 + injection.delta_for(rows)
    if variant == "semi_synthetic":
        baseline_paths: dict[FitSpec, np.ndarray] = {}
    elif shared_baseline is not None and injection.masked_window is None:
        baseline_paths = shared_baseline
    else:
        baseline_paths = _fit_all(design=_design(rows=rows, power=power0), fits=fits)
    return UnitFit(
        increment=injection.increment,
        observed=observed_weeks(table=rows),
        paths=_fit_all(design=_design(rows=rows, power=power1), fits=fits),
        baseline_paths=baseline_paths,
        fall=injection.fall,
    )


def _fit_constant_path_on_placebo_sky(
    *, unit: Unit, base: PrimaryBase, farm_rows: pl.DataFrame, fits: tuple[FitSpec, ...]
) -> UnitFit:
    """Fit the constant-path unit: permuted PV shape, with the known capacity added back.

    The real power is kept, the PV shape `u` is permuted among the days of each month, and the
    known baseline capacity times the permuted shape is subtracted from the power (PV lowers
    power). The real series' own growth is uncorrelated with the permuted shape, so the known
    path is constant.

    Args:
        unit: The constant-path unit.
        base: The Primary's baseline.
        farm_rows: `unit_rows(...)` without a mask window (the rows that the farm's gaps leave).
        fits: The fits to run.

    Returns:
        The fits, with a zero true increment and no baseline fits.
    """
    rows = permute_u(table=farm_rows, rng=_rng(unit=unit))
    power = rows["power"].to_numpy() - base.capacity * rows["u"].to_numpy()
    return UnitFit(
        increment=np.zeros(common.N_WEEKS),
        observed=observed_weeks(table=rows),
        paths=_fit_all(design=_design(rows=rows, power=power), fits=fits),
        baseline_paths={},
        fall=None,
    )


def _design(*, rows: pl.DataFrame, power: np.ndarray) -> model.Design:
    return model.design_from_rows_table(table=rows.with_columns(power=pl.Series(power)))


def _atomic_write(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file through a temporary name so that a crash leaves no partial file."""
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    frame.write_parquet(temporary)
    temporary.replace(path)


def run_unit_job(
    *,
    primary: str,
    farm: str,
    split: SplitType,
    fits: tuple[FitSpec, ...],
    out_dir: Path,
    variant: VariantType = "semi_synthetic",
    skip_unbuildable: bool = False,
) -> Path:
    """Fit every unit of one Primary and one farm, and write the units and paths atomically.

    Skips the work when the output exists, so a crashed run resumes.

    Args:
        primary: Frame site code.
        farm: Solar farm label; it must belong to `split`.
        split: Which split the job belongs to.
        fits: The fits to run on every unit.
        out_dir: Directory for the two output files.
        variant: Baseline variant.
        skip_unbuildable: Leave out a unit whose injection no week of the farm's short record can
            hold (`UnbuildableUnitError`) instead of raising; the caller counts the missing units.

    Returns:
        The path of the paths file.
    """
    assert_split(farm=farm, seeds=SEEDS_BY_SPLIT[split], split=split)
    stem = f"{split}_{variant}_{primary}_{farm}"
    paths_file, units_file = out_dir / f"{stem}_paths.parquet", out_dir / f"{stem}_units.parquet"
    if paths_file.exists() and units_file.exists():
        return paths_file
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    base = primary_base(inputs=inputs, site=primary)
    farm_rows = unit_rows(base=base, inputs=inputs, farm=farm)
    shared_baseline = (
        _fit_all(
            design=_design(rows=farm_rows, power=farm_rows["power"].to_numpy().astype(float)),
            fits=fits,
        )
        if variant == "real_plus_injection"
        else None
    )
    path_rows, unit_rows_out = [], []
    for seed in SEEDS_BY_SPLIT[split]:
        for case in (*CASES, *RATIO_CASES):
            unit = Unit(primary=primary, farm=farm, seed=seed, case=case)
            try:
                result = fit_unit(
                    unit=unit,
                    base=base,
                    inputs=inputs,
                    farm_rows=farm_rows,
                    fits=fits,
                    variant=variant,
                    shared_baseline=shared_baseline,
                )
            except UnbuildableUnitError:
                if skip_unbuildable:
                    continue
                raise
            unit_rows_out.append(
                {
                    "primary": primary,
                    "farm": farm,
                    "seed": seed,
                    "case": case,
                    "variant": variant,
                    "base_capacity": base.capacity,
                    "increment": result.increment.astype(np.float32),
                    "observed": result.observed,
                    "fall_start": result.fall[0] if result.fall else None,
                    "fall_weeks": result.fall[1] if result.fall else None,
                }
            )
            path_rows.extend(
                {
                    "primary": primary,
                    "farm": farm,
                    "seed": seed,
                    "case": case,
                    "fit": spec[0],
                    "a": spec[1],
                    "r": spec[2],
                    "with_injection": result.paths[spec].astype(np.float32),
                    "baseline": (
                        result.baseline_paths[spec].astype(np.float32)
                        if spec in result.baseline_paths
                        else None
                    ),
                }
                for spec in fits
            )
    unit_schema = {
        "primary": pl.String,
        "farm": pl.String,
        "seed": pl.Int32,
        "case": pl.String,
        "variant": pl.String,
        "base_capacity": pl.Float64,
        "increment": pl.List(pl.Float32),
        "observed": pl.List(pl.Boolean),
        "fall_start": pl.Int32,
        "fall_weeks": pl.Int32,
    }
    path_schema = {
        "primary": pl.String,
        "farm": pl.String,
        "seed": pl.Int32,
        "case": pl.String,
        "fit": pl.String,
        "a": pl.Float64,
        "r": pl.Float64,
        "with_injection": pl.List(pl.Float32),
        "baseline": pl.List(pl.Float32),
    }
    _atomic_write(frame=pl.DataFrame(unit_rows_out, schema=unit_schema), path=units_file)
    _atomic_write(frame=pl.DataFrame(path_rows, schema=path_schema), path=paths_file)
    return paths_file


def estimate_path(
    *,
    with_injection: np.ndarray,
    baseline: np.ndarray | None,
    ratio: float,
    variant: VariantType,
    base_capacity: float,
) -> np.ndarray:
    """Return the recovery-ratio-corrected estimate of a unit's capacity path.

    The semi-synthetic estimate comes from one fit, of the baseline plus the injection, and is
    compared with the known baseline capacity plus the injected path. Only the part above the
    known baseline capacity is divided by the Primary's recovery ratio, so the baseline PV (which
    the fit recovers at a ratio near 1) is not rescaled. The real-plus-injection estimate is the
    paired difference of two fits, divided by the ratio; it is compared with the true increment
    and measures additivity, because the real series' own capacity is not known.

    Args:
        with_injection: Raw fitted path with the injection, MW.
        baseline: Raw fitted path of the real power without the injection, same design, MW
            (real-plus-injection variant only).
        ratio: The Primary's recovery ratio.
        variant: Baseline variant.
        base_capacity: The known baseline capacity, MW.

    Returns:
        The estimated path, MW (NaN where the fit has no path).
    """
    if variant == "semi_synthetic":
        return base_capacity + (with_injection - base_capacity) / ratio
    assert baseline is not None
    return (with_injection - baseline) / ratio


def true_path(*, increment: np.ndarray, base_capacity: float, variant: VariantType) -> np.ndarray:
    """Return the true path that `estimate_path` is compared with."""
    return base_capacity + increment if variant == "semi_synthetic" else increment


def recovery_ratio(*, units: pl.DataFrame, paths: pl.DataFrame, spec: FitSpec) -> float:
    """Return a Primary's recovery ratio of one fit from its constant-path units.

    The ratio of a unit is the mean over observed weeks of the fitted increment divided by the
    injected constant capacity. The increment is the fit minus the known baseline capacity
    (semi-synthetic variant), or the fit minus the fit without the injection (real-plus-injection
    variant). The ratio of the Primary is the median over its farms and seeds.

    Args:
        units: `units` table rows of one Primary (any cases).
        paths: `paths` table rows of the same Primary.
        spec: The fit.

    Returns:
        The median recovery ratio.
    """
    key = ["primary", "farm", "seed", "case"]
    joined = (
        paths.filter(
            (pl.col("case") == "ratio_constant_farm")
            & (pl.col("fit") == spec[0])
            & (pl.col("a") == spec[1])
            & (pl.col("r") == spec[2])
        )
        .join(units.select(*key, "observed", "base_capacity"), on=key)
        .iter_rows(named=True)
    )
    ratios = []
    for row in joined:
        observed = np.array(row["observed"], dtype=bool)
        fitted = np.array(row["with_injection"], dtype=float) - (
            row["base_capacity"]
            if row["baseline"] is None
            else np.array(row["baseline"], dtype=float)
        )
        ratios.append(float(np.nanmean(fitted[observed]) / row["base_capacity"]))
    return float(np.median(ratios))


# ---------------------------------------------------------------------------------------------
# The placebo sky
# ---------------------------------------------------------------------------------------------


def permute_u(*, table: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Replace each day's PV shape `u` by that of another day of the same calendar month.

    Power stays in place, so any link between cloud and power is broken. A row whose donor day has
    no row at that half-hour is dropped.

    Args:
        table: `model.usable_rows_table` rows of one Primary.
        rng: The random generator.

    Returns:
        The rows with a permuted `u`; the other columns are unchanged.
    """
    shifted = table["time"].dt.offset_by("-15m")
    day = (shifted.dt.date().cast(pl.Datetime("us")).dt.epoch("d")).to_numpy()
    half_hour = (shifted.dt.hour().cast(pl.Int64) * 2 + shifted.dt.minute() // 30).to_numpy()
    month_key = (shifted.dt.year().cast(pl.Int64) * 100 + shifted.dt.month()).to_numpy()
    unique_days, day_index = np.unique(day, return_inverse=True)
    grid = np.full((len(unique_days), 48), np.nan)
    grid[day_index, half_hour] = table["u"].to_numpy()
    day_month = np.zeros(len(unique_days), dtype=np.int64)
    day_month[day_index] = month_key
    donor = np.arange(len(unique_days))
    for month in np.unique(day_month):
        members = np.nonzero(day_month == month)[0]
        donor[members] = rng.permutation(members)
    permuted = grid[donor[day_index], half_hour]
    return table.with_columns(u=pl.Series(permuted)).filter(pl.col("u").is_not_nan())


def run_placebo_job(
    *,
    primary: str,
    split: SplitType,
    fits: tuple[FitSpec, ...],
    out_dir: Path,
) -> Path:
    """Fit the placebo sky of one Primary and write its false-step counts atomically.

    For every fit the job also writes the unpermuted real series' mean capacity and the 10% step
    threshold, and the placebo counts of steps of 10% and of 5% of that mean. A placebo step in a
    week within 4 weeks of a real rise of 10% or more is not counted.

    Args:
        primary: Frame site code.
        split: Which split's permutation seeds to use.
        fits: The fits to run.
        out_dir: Directory for the output.

    Returns:
        The path of the output file.
    """
    out_file = out_dir / f"{split}_placebo_{primary}.parquet"
    if out_file.exists():
        return out_file
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    table = model.usable_rows_table(
        half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=primary
    )
    real = _fit_all(design=model.design_from_rows_table(table=table), fits=fits)
    stats = {}
    for spec, path in real.items():
        mean = float(np.nanmean(path))
        stats[spec] = (mean, metrics.rise_weeks(path=path, threshold=0.10 * mean))
    seeds = TUNING_PLACEBO_SEEDS if split == "tuning" else SCORING_PLACEBO_SEEDS
    out = []
    for seed in seeds:
        permuted = permute_u(table=table, rng=np.random.default_rng([seed, 45]))
        found = _fit_all(design=model.design_from_rows_table(table=permuted), fits=fits)
        for spec, path in found.items():
            mean, exempt = stats[spec]
            n_weeks = int(np.isfinite(path).sum())
            out.append(
                {
                    "primary": primary,
                    "perm_seed": seed,
                    "fit": spec[0],
                    "a": spec[1],
                    "r": spec[2],
                    "real_mean_capacity": mean,
                    "n_weeks": n_weeks,
                    "steps_10": metrics.count_steps(
                        path=path, threshold=0.10 * mean, exempt=exempt
                    ),
                    "steps_05": metrics.count_steps(
                        path=path, threshold=0.05 * mean, exempt=exempt
                    ),
                    "drift_per_5y": metrics.ratchet_drift_per_5_years(path=path, reference=mean),
                }
            )
    _atomic_write(frame=pl.DataFrame(out), path=out_file)
    return out_file
