"""Score the frozen penalty pair and four comparison fits on the scoring units (experiment 00045).

The stages, each resumable because every job writes its output atomically and skips work whose
output exists:

1. `jobs`: fit the five fits on every scoring unit (both baseline variants) and on the placebo sky.
2. `ratios`: measure each Primary's recovery ratio of every fit on the tuning farms.
3. `score`: score every unit against its known path (`unit_scores.parquet`).
4. `vintages`: refit at origins every 4 weeks over the last 3 years, truncated at the origin.
5. `report`: the decision conditions (i) to (vi), the contrasts, the sub-breakdowns, and the
   verdict, printed and written to `score_report.txt`. Condition (vi) reads the coverage tables
   that `e00045_bootstrap.py` writes.

The frozen file is read, never written. `--smoke` runs every stage on tuning farm A, two
Primaries, and three origins, into `scores/smoke/`.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_score.py
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00045_common as common
import e00045_inject as inject
import e00045_metrics as metrics
import e00045_model as model
import numpy as np
import polars as pl

FOLDER_PATH: Final[Path] = Path(__file__).resolve().parent
FROZEN_PATH: Final[Path] = FOLDER_PATH / "frozen_config.json"
SCORES_DIR: Final[Path] = common.DATA_DIR / "scores"
SMOKE_DIR: Final[Path] = SCORES_DIR / "smoke"
SMOKE_PRIMARIES: Final[tuple[str, ...]] = ("D10", "D15")
SMOKE_ORIGINS: Final[int] = 3
SELFTEST_SCRIPTS: Final[tuple[str, ...]] = ("e00045_selftest.py", "e00045_inject_selftest.py")
STAGES: Final[tuple[str, ...]] = ("jobs", "ratios", "score", "vintages", "report")

FIT_NAMES: Final[tuple[model.FitName, ...]] = (
    "asymmetric",
    "free",
    "symmetric",
    "non_decreasing",
    "season_year",
)
"""The frozen fit first, then the four comparison fits."""
FIT_LABELS: Final[dict[str, str]] = {
    "asymmetric": "asymmetric",
    "free": "free weekly",
    "symmetric": "symmetric",
    "non_decreasing": "non-decreasing",
    "season_year": "season-year constant",
}
RAMPS: Final[tuple[str, ...]] = ("ramp_25", "ramp_50")
STEPS: Final[tuple[str, ...]] = ("step_05", "step_10", "step_20", "step_30")
STEPS_10: Final[tuple[str, ...]] = ("step_10", "step_20", "step_30")
FALL_CASES: Final[tuple[str, ...]] = tuple(inject.FALLS)
TRANSFERS: Final[tuple[str, ...]] = ("transfer_inside_mask", "transfer_outside_mask")
LOADS: Final[tuple[str, ...]] = ("seasonal_load_independent", "seasonal_load_sunshine")
CASE_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "ramps": RAMPS,
    "steps": STEPS,
    "falls": FALL_CASES,
    "transfers": TRANSFERS,
    "seasonal loads": LOADS,
    "constant path": ("constant_path",),
}
CONDITION_I_CASES: Final[tuple[str, ...]] = (*RAMPS, *STEPS)
C2_CASES: Final[tuple[str, ...]] = (*RAMPS, *STEPS_10)
"""Condition (i) is scored on ramps and all four steps; contrast C2 on ramps and steps of 10% or
more, as the plan words them."""
VARIANTS: Final[tuple[inject.VariantType, ...]] = ("semi_synthetic", "real_plus_injection")
DETECTION_WEEKS: Final[int] = 3
DETECTION_SHARE: Final[float] = 0.5
"""A step is detected when the estimate rises by at least this share of the step within
`DETECTION_WEEKS` weeks of the true date, above the median of the 4 weeks before it."""
DETECTION_LIMIT_SHARE: Final[float] = 0.8
STEP_SIZES: Final[tuple[float, ...]] = tuple(inject.STEP_SHARES.values())
VINTAGE_CASES: Final[tuple[inject.CaseName, ...]] = ("ramp_50", "step_10", "step_20")
VINTAGE_EVERY_WEEKS: Final[int] = 4
VINTAGE_YEARS_WEEKS: Final[int] = 156
MIN_VINTAGE_WEEKS: Final[int] = 52
"""A vintage is fitted only when the Primary, the unit, and each ratio unit hold at least this many
observed weeks before the origin (a season-year, enough to estimate the demand cells)."""
VINTAGE_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "primary": pl.String,
    "farm": pl.String,
    "seed": pl.Int64,
    "case": pl.String,
    "origin": pl.Int64,
    "age": pl.Int64,
    "error_vintage": pl.Float64,
    "error_final": pl.Float64,
}
VINTAGE_AGES: Final[tuple[int, ...]] = (1, 2, 4, 8, 13, 26)
LIMIT_I: Final[float] = 0.15
LIMIT_II: Final[float] = -0.20
LIMIT_FALSE_MEDIAN: Final[float] = 0.5
LIMIT_FALSE_WORST: Final[float] = 1.0
LIMIT_REFUTED_FALSE: Final[float] = 1.5
LIMIT_DRIFT: Final[float] = 0.05
LIMIT_FALL_26: Final[float] = 0.5
LIMIT_FALL_8: Final[float] = 0.5
LIMIT_COVERAGE_NON_STEP: Final[tuple[float, float]] = (0.85, 0.95)
LIMIT_COVERAGE_STEP: Final[float] = 0.60
LIMIT_C5: Final[float] = 2.0
LIMIT_DETECTION: Final[float] = 0.25
N_DRAWS: Final[int] = 2000
"""Cluster-bootstrap draws of every interval in the report."""
FLAGGED_PRIMARIES: Final[tuple[str, ...]] = ("D5", "D15")
"""Kirton and Wrangle T1, which the report flags in every per-Primary table."""


@dataclass(frozen=True)
class Frozen:
    """The frozen penalty pair.

    Attributes:
        status: `"frozen"` or `"refuted-fallback"`.
        a: Rise penalty, in weekly standard errors.
        r: Ratio of the fall penalty to the rise penalty.
    """

    status: str
    a: float
    r: float

    def fits(self) -> tuple[inject.FitSpec, ...]:
        """Return the five fits: the frozen pair, then free, symmetric, non-decreasing, constant."""
        return (
            ("asymmetric", self.a, self.r),
            ("free", 0.0, 0.0),
            ("symmetric", self.a, 1.0),
            ("non_decreasing", self.a, math.inf),
            ("season_year", 0.0, 0.0),
        )


def load_frozen() -> Frozen:
    """Read the frozen penalty pair."""
    raw = json.loads(FROZEN_PATH.read_text())
    return Frozen(status=raw["status"], a=float(raw["a"]), r=float(raw["r"]))


@dataclass(frozen=True)
class Run:
    """What one invocation scores.

    Attributes:
        split: `"scoring"`, or `"tuning"` for the smoke test.
        primaries: Frame site codes to fit.
        farms: Solar farms of the split to fit.
        out_dir: Directory of every output.
        origins: Vintage origin weeks.
        workers: Number of single-threaded worker processes.
    """

    split: inject.SplitType
    primaries: tuple[str, ...]
    farms: tuple[str, ...]
    out_dir: Path
    origins: tuple[int, ...]
    workers: int


def vintage_origins() -> tuple[int, ...]:
    """Return the origin weeks: every 4 weeks over the last 3 years of the record."""
    return tuple(range(common.N_WEEKS - VINTAGE_YEARS_WEEKS, common.N_WEEKS, VINTAGE_EVERY_WEEKS))


def atomic_write(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file through a temporary name so that a crash leaves no partial file."""
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    frame.write_parquet(temporary)
    temporary.replace(path)


def design_for(*, rows: pl.DataFrame, power: np.ndarray) -> model.Design:
    """Build a design from a rows table whose `power` is replaced by `power`."""
    return model.design_from_rows_table(table=rows.with_columns(power=pl.Series(power)))


# ---------------------------------------------------------------------------------------------
# Stage 1 and 2: fits
# ---------------------------------------------------------------------------------------------


def _unit_job(args: tuple[str, str, inject.SplitType, inject.VariantType, Path, Frozen]) -> str:
    primary, farm, split, variant, out_dir, frozen = args
    inject.run_unit_job(
        primary=primary,
        farm=farm,
        split=split,
        fits=frozen.fits(),
        out_dir=out_dir,
        variant=variant,
    )
    return f"units {variant} {primary} {farm}"


def _placebo_job(args: tuple[str, inject.SplitType, Path, Frozen]) -> str:
    primary, split, out_dir, frozen = args
    inject.run_placebo_job(primary=primary, split=split, fits=frozen.fits(), out_dir=out_dir)
    return f"placebo {primary}"


def ratio_job(*, primary: str, farm: str, frozen: Frozen, out_dir: Path) -> Path:
    """Measure one Primary's recovery ratio of every fit from one tuning farm's constant injection.

    The injection has the farm's shape at the Primary's baseline capacity, so the ratio is the mean
    over observed weeks of the fitted increment divided by that capacity. The injection holds no
    random choice, so one path seed stands for all four.

    Args:
        primary: Frame site code.
        farm: Tuning farm label.
        frozen: The frozen penalty pair.
        out_dir: Directory of the output.

    Returns:
        The path of the output file.
    """
    out_file = out_dir / f"ratio_{primary}_{farm}.parquet"
    if out_file.exists():
        return out_file
    inputs = inject.load_inputs()
    base = inject.primary_base(inputs=inputs, site=primary)
    design, observed, _ = ratio_design(base=base, inputs=inputs, farm=farm)
    rows = []
    for spec in frozen.fits():
        path = model.FITS[spec[0]](design=design, a=spec[1], r=spec[2])
        rows.append(
            {
                "primary": primary,
                "farm": farm,
                "fit": spec[0],
                "ratio": float(np.nanmean((path - base.capacity)[observed]) / base.capacity),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(frame=pl.DataFrame(rows), path=out_file)
    return out_file


def ratio_design(
    *, base: inject.PrimaryBase, inputs: inject.Inputs, farm: str
) -> tuple[model.Design, np.ndarray, pl.DataFrame]:
    """Build the design of a Primary's constant farm injection (the recovery-ratio unit).

    Args:
        base: The Primary's baseline.
        inputs: The loaded tables.
        farm: Tuning farm label.

    Returns:
        The design, the mask of observed weeks, and the rows it was built from.
    """
    rows = inject.unit_rows(base=base, inputs=inputs, farm=farm)
    unit = inject.Unit(
        primary=base.site, farm=farm, seed=inject.TUNING_SEEDS[0], case="ratio_constant_farm"
    )
    injection = inject.build_injection(unit=unit, base=base, farm_rows=rows, inputs=inputs)
    power = rows["baseline_power"].to_numpy().astype(float) + injection.delta_for(rows)
    return design_for(rows=rows, power=power), inject.observed_weeks(table=rows), rows


def _ratio_job(args: tuple[str, str, Frozen, Path]) -> str:
    primary, farm, frozen, out_dir = args
    ratio_job(primary=primary, farm=farm, frozen=frozen, out_dir=out_dir)
    return f"ratio {primary} {farm}"


def run_fit_jobs(*, run: Run, frozen: Frozen) -> None:
    """Run the unit, placebo, and ratio jobs on `run.workers` single-threaded processes."""
    for farm in run.farms:
        inject.assert_split(farm=farm, seeds=inject.SEEDS_BY_SPLIT[run.split], split=run.split)
    with ProcessPoolExecutor(max_workers=run.workers) as pool:
        placebo = [(p, run.split, run.out_dir, frozen) for p in run.primaries]
        units = [
            (p, f, run.split, v, run.out_dir, frozen)
            for v in VARIANTS
            for f in run.farms
            for p in run.primaries
        ]
        for line in pool.map(_placebo_job, placebo):
            print(line, flush=True)
        for line in pool.map(_unit_job, units):
            print(line, flush=True)


def run_ratio_jobs(*, run: Run, frozen: Frozen) -> None:
    """Measure the recovery ratios on the tuning farms."""
    with ProcessPoolExecutor(max_workers=run.workers) as pool:
        args = [(p, f, frozen, run.out_dir) for f in inject.TUNING_FARMS for p in run.primaries]
        for line in pool.map(_ratio_job, args):
            print(line, flush=True)


def read_ratios(*, out_dir: Path) -> dict[tuple[str, str], float]:
    """Return each Primary's recovery ratio of each fit: the median over the tuning farms."""
    table = pl.concat([pl.read_parquet(f) for f in sorted(out_dir.glob("ratio_*.parquet"))])
    grouped = table.group_by("primary", "fit").agg(ratio=pl.col("ratio").median())
    return {(r["primary"], r["fit"]): r["ratio"] for r in grouped.iter_rows(named=True)}


# ---------------------------------------------------------------------------------------------
# Stage 3: scoring each unit
# ---------------------------------------------------------------------------------------------


def _step_features(*, estimate: np.ndarray, increment: np.ndarray) -> dict[str, object]:
    """Return a step unit's start week, size, season, and whether the estimate shows the step."""
    start = int(np.argmax(increment > 0))
    size = float(increment[-1])
    before = estimate[max(0, start - 4) : start]
    after = estimate[start : start + DETECTION_WEEKS + 1]
    if not np.isfinite(before).any() or not np.isfinite(after).any():
        detected = None
    else:
        detected = bool(np.nanmax(after) - np.nanmedian(before) >= DETECTION_SHARE * size)
    return {
        "step_start": start,
        "step_size": size,
        "step_summer": common.week_start(week=start).month in common.SUMMER_MONTHS,
        "detected": detected,
    }


def score_row(
    *,
    row: dict[str, Any],
    variant: inject.VariantType,
    ratio: float,
    real_mean: float,
) -> dict[str, object]:
    """Score one unit and fit against its known path.

    Args:
        row: One joined row of the units and paths tables.
        variant: Baseline variant.
        ratio: The Primary's recovery ratio of this fit (1 for the constant-path unit, which is
            built from the PV shape itself).
        real_mean: Mean capacity of the unpermuted real series under this fit, MW.

    Returns:
        The unit's scores. Scores that do not apply to the case are `None`.
    """
    observed = np.array(row["observed"], dtype=bool)
    increment = np.array(row["increment"], dtype=float)
    base_capacity = float(row["base_capacity"])
    case = str(row["case"])
    estimate = inject.estimate_path(
        with_injection=np.array(row["with_injection"], dtype=float),
        baseline=None if row["baseline"] is None else np.array(row["baseline"], dtype=float),
        ratio=ratio,
        variant=variant,
        base_capacity=base_capacity,
    )
    truth = inject.true_path(increment=increment, base_capacity=base_capacity, variant=variant)
    scale = float(np.mean((base_capacity + increment)[observed]))
    out: dict[str, object] = {
        "variant": variant,
        "primary": row["primary"],
        "farm": row["farm"],
        "seed": row["seed"],
        "case": case,
        "fit": row["fit"],
        "a": row["a"],
        "r": row["r"],
        "ratio": ratio,
        "path_error": metrics.path_error(
            estimate=estimate, truth=truth, weeks=observed, scale=scale
        ),
        "peak_error": float(np.nanmax(np.abs(estimate - truth)[observed]) / scale),
        "n_missing_observed": int((~np.isfinite(estimate[observed])).sum()),
        "fall_recovered": None,
        "steps_10": None,
        "steps_05": None,
        "drift_per_5y": None,
        "swing": None,
        "step_start": None,
        "step_size": None,
        "step_summer": None,
        "detected": None,
    }
    if row["fall_start"] is not None:
        share = metrics.fall_recovered(
            estimate=estimate,
            truth=truth,
            weeks=observed,
            start=int(row["fall_start"]),
            duration=int(row["fall_weeks"]),
        )
        out["fall_recovered"] = share if np.isfinite(share) else None
    if case == "constant_path" and variant == "semi_synthetic":
        n_weeks = int(np.isfinite(estimate).sum())
        for label, share_of_mean in (("steps_10", 0.10), ("steps_05", 0.05)):
            found = metrics.count_steps(path=estimate, threshold=share_of_mean * real_mean)
            out[label] = metrics.steps_per_year(steps=found, n_weeks=n_weeks)
        out["drift_per_5y"] = metrics.ratchet_drift_per_5_years(
            path=estimate, reference=float(np.nanmean(truth))
        )
    if case in LOADS and variant == "semi_synthetic":
        out["swing"] = metrics.seasonal_swing(estimate=estimate, truth=truth, weeks=observed)
    if case in inject.STEP_SHARES:
        out.update(_step_features(estimate=estimate, increment=increment))
    return out


def unit_scores(
    *,
    run: Run,
    variant: inject.VariantType,
    ratios: dict[tuple[str, str], float],
    real_mean: dict[tuple[str, str], float],
) -> pl.DataFrame:
    """Score every unit and fit of one variant.

    The constant-path unit is skipped in the real-plus-injection variant: it has no fit of the real
    power alone to difference against.

    Args:
        run: What is scored.
        variant: Baseline variant.
        ratios: Recovery ratio of each (Primary, fit).
        real_mean: Mean capacity of the unpermuted real series of each (Primary, fit).

    Returns:
        One row per unit and fit.
    """
    stem = f"{run.split}_{variant}"
    units = pl.concat(
        [pl.read_parquet(f) for f in sorted(run.out_dir.glob(f"{stem}_*_units.parquet"))]
    )
    paths = pl.concat(
        [pl.read_parquet(f) for f in sorted(run.out_dir.glob(f"{stem}_*_paths.parquet"))]
    )
    key = ["primary", "farm", "seed", "case"]
    joined = paths.join(units.drop("variant"), on=key).filter(
        ~pl.col("case").is_in(list(inject.RATIO_CASES))
    )
    if variant == "real_plus_injection":
        joined = joined.filter(pl.col("case") != "constant_path")
    rows = []
    for row in joined.iter_rows(named=True):
        fit = row["fit"]
        ratio = 1.0 if row["case"] == "constant_path" else ratios[(row["primary"], fit)]
        rows.append(
            score_row(
                row=row,
                variant=variant,
                ratio=ratio,
                real_mean=real_mean.get((row["primary"], fit), math.nan),
            )
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def read_placebo(*, run: Run) -> pl.DataFrame:
    """Read the placebo jobs' counts, with the false-step rate per year of each permutation."""
    files = sorted(run.out_dir.glob(f"{run.split}_placebo_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]).with_columns(
        rate_10=pl.col("steps_10") / (pl.col("n_weeks") / metrics.WEEKS_PER_YEAR),
        rate_05=pl.col("steps_05") / (pl.col("n_weeks") / metrics.WEEKS_PER_YEAR),
    )


def run_scoring(*, run: Run) -> pl.DataFrame:
    """Score every unit of both variants and write `unit_scores.parquet`."""
    ratios = read_ratios(out_dir=run.out_dir)
    placebo = read_placebo(run=run)
    real_mean = {
        (r["primary"], r["fit"]): r["real_mean_capacity"]
        for r in placebo.unique(subset=["primary", "fit"]).iter_rows(named=True)
    }
    scores = pl.concat(
        [unit_scores(run=run, variant=v, ratios=ratios, real_mean=real_mean) for v in VARIANTS],
        how="vertical_relaxed",
    )
    atomic_write(frame=scores, path=run.out_dir / "unit_scores.parquet")
    return scores


# ---------------------------------------------------------------------------------------------
# Stage 4: real-time vintages on injections
# ---------------------------------------------------------------------------------------------


def _ratio_at(
    *,
    designs: list[tuple[model.Design, np.ndarray]],
    spec: inject.FitSpec,
    capacity: float,
    origin: int | None,
) -> float:
    """Return the recovery ratio from data before `origin` (all data when `None`)."""
    values = []
    grid = np.arange(common.N_WEEKS)
    for design, observed in designs:
        path = model.FITS[spec[0]](design=design, a=spec[1], r=spec[2], origin_week=origin)
        use = observed & np.isfinite(path)
        if origin is not None:
            use &= grid < origin
        values.append(float(np.mean((path - capacity)[use]) / capacity))
    return float(np.median(values))


def vintage_job(*, primary: str, run: Run, frozen: Frozen) -> Path:
    """Refit the frozen fit at every origin on a Primary's injection units, truncated at the origin.

    The Primary scale and the recovery ratio are recomputed from data before each origin, the
    ratio from the tuning farms' constant injections. The week `age` weeks before the origin is
    scored against the truth, and so is the same week in the final fit.

    Args:
        primary: Frame site code.
        run: What is scored.
        frozen: The frozen penalty pair.

    Returns:
        The path of the output file.
    """
    out_file = run.out_dir / f"vintage_{primary}.parquet"
    if out_file.exists():
        return out_file
    spec: inject.FitSpec = frozen.fits()[0]
    inputs = inject.load_inputs()
    base = inject.primary_base(inputs=inputs, site=primary)
    ratio_designs = [
        ratio_design(base=base, inputs=inputs, farm=f)[:2] for f in inject.TUNING_FARMS
    ]
    usable_origins = [
        origin
        for origin in run.origins
        if min(int(observed[:origin].sum()) for _, observed in ratio_designs) >= MIN_VINTAGE_WEEKS
    ]
    ratio_by_origin = {
        origin: _ratio_at(designs=ratio_designs, spec=spec, capacity=base.capacity, origin=origin)
        for origin in (None, *usable_origins)
    }
    out = []
    for farm in run.farms:
        rows = inject.unit_rows(base=base, inputs=inputs, farm=farm)
        observed = inject.observed_weeks(table=rows)
        for seed in inject.SEEDS_BY_SPLIT[run.split]:
            for case in VINTAGE_CASES:
                unit = inject.Unit(primary=primary, farm=farm, seed=seed, case=case)
                injection = inject.build_injection(
                    unit=unit, base=base, farm_rows=rows, inputs=inputs
                )
                power = rows["baseline_power"].to_numpy().astype(float) + injection.delta_for(rows)
                design = design_for(rows=rows, power=power)
                truth = base.capacity + injection.increment
                scale = float(np.mean(truth[observed]))
                final = _corrected(
                    path=model.FITS[spec[0]](design=design, a=spec[1], r=spec[2]),
                    ratio=ratio_by_origin[None],
                    capacity=base.capacity,
                )
                for origin in usable_origins:
                    if observed[:origin].sum() < MIN_VINTAGE_WEEKS:
                        continue
                    vintage = _corrected(
                        path=model.FITS[spec[0]](
                            design=design, a=spec[1], r=spec[2], origin_week=origin
                        ),
                        ratio=ratio_by_origin[origin],
                        capacity=base.capacity,
                    )
                    for age in VINTAGE_AGES:
                        week = origin - age
                        if not observed[week] or not np.isfinite(vintage[week]):
                            continue
                        out.append(
                            {
                                "primary": primary,
                                "farm": farm,
                                "seed": seed,
                                "case": case,
                                "origin": origin,
                                "age": age,
                                "error_vintage": abs(vintage[week] - truth[week]) / scale,
                                "error_final": abs(final[week] - truth[week]) / scale,
                            }
                        )
    run.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(frame=pl.DataFrame(out, schema=VINTAGE_SCHEMA), path=out_file)
    return out_file


def _corrected(*, path: np.ndarray, ratio: float, capacity: float) -> np.ndarray:
    """Return a semi-synthetic path with only the part above the baseline capacity rescaled."""
    return capacity + (path - capacity) / ratio


def _vintage_job(args: tuple[str, Run, Frozen]) -> str:
    primary, run, frozen = args
    vintage_job(primary=primary, run=run, frozen=frozen)
    return f"vintage {primary}"


def run_vintage_jobs(*, run: Run, frozen: Frozen) -> None:
    """Run one vintage job per Primary."""
    with ProcessPoolExecutor(max_workers=run.workers) as pool:
        for line in pool.map(_vintage_job, [(p, run, frozen) for p in run.primaries]):
            print(line, flush=True)


# ---------------------------------------------------------------------------------------------
# Cluster bootstrap over Primaries, then farm-seeds
# ---------------------------------------------------------------------------------------------


def tensor_of(
    *,
    frame: pl.DataFrame,
    fit: str,
    cases: tuple[str, ...],
    column: str,
    variant: str = "semi_synthetic",
) -> np.ndarray:
    """Return a column as an array indexed by (Primary, farm-seed, case), NaN where absent."""
    rows = frame.filter(
        (pl.col("fit") == fit) & (pl.col("variant") == variant) & pl.col("case").is_in(list(cases))
    )
    primaries = [p for p in common.PRIMARIES if p in set(rows["primary"])]
    keys = sorted(
        {(r["farm"], r["seed"]) for r in rows.select("farm", "seed").iter_rows(named=True)}
    )
    out = np.full((len(primaries), len(keys), len(cases)), np.nan)
    for r in rows.select("primary", "farm", "seed", "case", column).iter_rows(named=True):
        value = r[column]
        if value is None:
            continue
        out[
            primaries.index(r["primary"]),
            keys.index((r["farm"], r["seed"])),
            cases.index(r["case"]),
        ] = float(value)
    return out


def _safe(func: Callable[[np.ndarray], float], values: np.ndarray) -> float:
    """Return `func(values)` over the finite values, or NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(func(finite)) if finite.size else math.nan


def median_of_medians(values: np.ndarray) -> float:
    """Return the median over Primaries of each Primary's median over its other axes."""
    return _safe(
        np.median, np.array([_safe(np.median, values[p].ravel()) for p in range(len(values))])
    )


def median_of_means(values: np.ndarray) -> float:
    """Return the median over Primaries of each Primary's mean over its other axes."""
    return _safe(
        np.median, np.array([_safe(np.mean, values[p].ravel()) for p in range(len(values))])
    )


def worst_of_means(values: np.ndarray) -> float:
    """Return the largest over Primaries of each Primary's mean over its other axes."""
    return _safe(np.max, np.array([_safe(np.mean, values[p].ravel()) for p in range(len(values))]))


@dataclass(frozen=True)
class Interval:
    """A statistic and its 95% cluster-bootstrap interval."""

    point: float
    low: float
    high: float

    def __str__(self) -> str:
        """Format as `point [low, high]`."""
        return f"{self.point:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"


class Draws:
    """Cluster-bootstrap indices: Primaries with replacement, then farm-seeds within each.

    Every statistic given the same number of farm-seed columns reuses the same indices, so a
    contrast between two fits is paired.
    """

    def __init__(self, *, n_primaries: int, n_draws: int = N_DRAWS, seed: int = 45) -> None:  # noqa: D107
        self.n_draws = n_draws
        self._seed = seed
        self.primaries = np.random.default_rng(seed).integers(
            0, n_primaries, (n_draws, n_primaries)
        )
        self._within: dict[int, np.ndarray] = {}

    def within(self, *, n_columns: int) -> np.ndarray:
        """Return the (draw, drawn Primary, column) indices of `n_columns` farm-seed columns."""
        if n_columns not in self._within:
            rng = np.random.default_rng([self._seed, n_columns])
            self._within[n_columns] = rng.integers(
                0, n_columns, (self.n_draws, self.primaries.shape[1], n_columns)
            )
        return self._within[n_columns]


def boot(
    *,
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    draws: Draws,
    resample_columns: bool = True,
) -> np.ndarray:
    """Return the statistic of every bootstrap draw of a (Primary, farm-seed, case) array."""
    within = draws.within(n_columns=values.shape[1])
    out = np.empty(draws.n_draws)
    for d in range(draws.n_draws):
        drawn = np.empty((values.shape[0], *values.shape[1:]))
        for j in range(values.shape[0]):
            primary = values[draws.primaries[d, j] % values.shape[0]]
            drawn[j] = primary[within[d, j]] if resample_columns else primary
        out[d] = statistic(drawn)
    return out


def interval(*, point: float, samples: np.ndarray) -> Interval:
    """Return the point and the 2.5th and 97.5th percentiles of the finite samples."""
    finite = samples[np.isfinite(samples)]
    low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (math.nan, math.nan)
    return Interval(point=point, low=float(low), high=float(high))


def contrast(
    *,
    left: np.ndarray,
    right: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    draws: Draws,
    relative: bool = False,
    resample_columns: bool = True,
) -> Interval:
    """Return `left - right` (or `left / right - 1`) of a statistic, with a paired interval."""

    def combine(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
        return a / b - 1 if relative else a - b

    point = float(combine(statistic(left), statistic(right)))
    samples = combine(
        boot(values=left, statistic=statistic, draws=draws, resample_columns=resample_columns),
        boot(values=right, statistic=statistic, draws=draws, resample_columns=resample_columns),
    )
    return interval(point=point, samples=np.asarray(samples))


# ---------------------------------------------------------------------------------------------
# Stage 5: the decision report
# ---------------------------------------------------------------------------------------------


def table_text(frame: pl.DataFrame) -> str:
    """Format a frame for the report."""
    with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=220, float_precision=3):
        return str(frame)


class Report:
    """Collects the report's lines, prints them, and keeps every table as a parquet file."""

    def __init__(self, *, out_dir: Path) -> None:  # noqa: D107
        self.out_dir = out_dir
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        """Add a line."""
        self.lines.append(text)

    def section(self, title: str) -> None:
        """Add a section heading."""
        self.line()
        self.line(f"== {title} ==")

    def table(
        self, name: str, frame: pl.DataFrame, *, note: str = "", transpose: bool = False
    ) -> None:
        """Add a table and write it to `<name>.parquet`; `transpose` only changes the printout."""
        if note:
            self.line(note)
        self.line(f"[{name}]")
        if transpose:
            labels = frame.columns[0]
            shown = frame.drop(labels).transpose(
                include_header=True, column_names=frame[labels].to_list()
            )
            self.line(table_text(shown))
        else:
            self.line(table_text(frame))
        atomic_write(frame=frame, path=self.out_dir / f"{name}.parquet")

    def write(self) -> None:
        """Print the report and write `score_report.txt`."""
        text = "\n".join(self.lines) + "\n"
        print(text)
        (self.out_dir / "score_report.txt").write_text(text)


@dataclass
class Condition:
    """One decision condition with its number and verdict.

    Attributes:
        label: `"(i)"` to `"(vi)"`.
        text: What the condition requires.
        number: The measured value, formatted.
        passed: `True`, `False`, or `None` when the condition could not be scored.
    """

    label: str
    text: str
    number: str
    passed: bool | None

    def __str__(self) -> str:
        """Format as one report line."""
        mark = {True: "PASS", False: "FAIL", None: "NOT SCORED"}[self.passed]
        return f"{self.label} {self.text}: {self.number} -> {mark}"


def _by_primary(
    frame: pl.DataFrame, *, fit: str, cases: tuple[str, ...], column: str
) -> pl.DataFrame:
    """Return the median of `column` per Primary for one fit on the semi-synthetic variant."""
    return (
        frame.filter(
            (pl.col("fit") == fit)
            & (pl.col("variant") == "semi_synthetic")
            & pl.col("case").is_in(list(cases))
        )
        .group_by("primary")
        .agg(pl.col(column).median())
        .sort("primary")
    )


def false_step_tensors(
    *, scores: pl.DataFrame, placebo: pl.DataFrame, fit: str, column: str = "steps_10"
) -> tuple[np.ndarray, np.ndarray]:
    """Return false-step rates per year as (Primary, replicate, 1) arrays on the two skies.

    Args:
        scores: `unit_scores`.
        placebo: The placebo counts with their rates.
        fit: The fit.
        column: `"steps_10"` or `"steps_05"`.

    Returns:
        The constant-path sky (replicates are farm-seeds) and the placebo sky (permutations).
    """
    constant = tensor_of(frame=scores, fit=fit, cases=("constant_path",), column=column)
    rate = "rate_10" if column == "steps_10" else "rate_05"
    rows = placebo.filter(pl.col("fit") == fit)
    primaries = [p for p in common.PRIMARIES if p in set(rows["primary"])]
    seeds = sorted(set(rows["perm_seed"]))
    sky = np.full((len(primaries), len(seeds), 1), np.nan)
    for r in rows.iter_rows(named=True):
        sky[primaries.index(r["primary"]), seeds.index(r["perm_seed"]), 0] = r[rate]
    return constant, sky


def _drift_tensors(
    *, scores: pl.DataFrame, placebo: pl.DataFrame, fit: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ratchet drift as (Primary, replicate, 1) arrays on the two skies."""
    constant = tensor_of(frame=scores, fit=fit, cases=("constant_path",), column="drift_per_5y")
    rows = placebo.filter(pl.col("fit") == fit)
    primaries = [p for p in common.PRIMARIES if p in set(rows["primary"])]
    seeds = sorted(set(rows["perm_seed"]))
    sky = np.full((len(primaries), len(seeds), 1), np.nan)
    for r in rows.iter_rows(named=True):
        sky[primaries.index(r["primary"]), seeds.index(r["perm_seed"]), 0] = r["drift_per_5y"]
    return constant, sky


def detection_limits(*, scores: pl.DataFrame, fit: str) -> pl.DataFrame:
    """Return, per Primary and half of the year, the smallest step size detected in 80% of units.

    Args:
        scores: `unit_scores`.
        fit: The fit.

    Returns:
        One row per Primary and season with the share detected at each of the four step sizes and
        the detection limit (infinity when no size reaches 80%).
    """
    steps = scores.filter(
        (pl.col("fit") == fit)
        & (pl.col("variant") == "semi_synthetic")
        & pl.col("case").is_in(list(STEPS))
        & pl.col("detected").is_not_null()
    ).with_columns(
        season=pl.when(pl.col("step_summer")).then(pl.lit("Apr-Sep")).otherwise(pl.lit("Oct-Mar"))
    )
    shares = (
        steps.group_by("primary", "season", "case")
        .agg(share=pl.col("detected").cast(pl.Float64).mean())
        .sort("primary", "season", "case")
    )
    rows = []
    for (primary, season), group in shares.group_by(["primary", "season"], maintain_order=True):
        by_case = dict(zip(group["case"], group["share"], strict=True))
        limit = math.inf
        for case, size in zip(STEPS, STEP_SIZES, strict=True):
            if by_case.get(case, 0.0) >= DETECTION_LIMIT_SHARE:
                limit = size * 100
                break
        rows.append(
            {
                "primary": primary,
                "season": season,
                **{f"share_{case}": by_case.get(case) for case in STEPS},
                "limit_percent": limit,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def build_report(*, run: Run, frozen: Frozen, scores: pl.DataFrame) -> Report:  # noqa: PLR0915
    """Compute every condition, contrast, and breakdown, and return the report."""
    report = Report(out_dir=run.out_dir)
    n_units = (
        scores.filter(pl.col("variant") == "semi_synthetic")
        .select("primary", "farm", "seed", "case")
        .unique()
        .height
    )
    placebo = read_placebo(run=run)
    n_primaries = scores["primary"].n_unique()
    draws = Draws(n_primaries=n_primaries)
    report.line(
        f"Experiment 45 scoring report. Split {run.split}; Primaries {n_primaries}; units "
        f"{n_units}"
        f"; frozen status {frozen.status}; a={frozen.a}, r={frozen.r}."
    )
    report.line(
        "Statistics are medians over farm-seeds and cases within a Primary, then medians over "
        f"Primaries. Intervals: 95% cluster bootstrap, {N_DRAWS} draws, Primaries resampled and "
        "then farm-seeds within each drawn Primary."
    )
    missing = int(scores["n_missing_observed"].sum())
    report.line(f"Observed weeks with no fitted value (should be 0): {missing}.")

    conditions: list[Condition] = []
    asym = "asymmetric"

    # (i) path error on ramps and steps, worse of the two baselines.
    errors = {
        v: median_of_medians(
            tensor_of(
                frame=scores, fit=asym, cases=CONDITION_I_CASES, column="path_error", variant=v
            )
        )
        for v in VARIANTS
    }
    worst_i = max(errors.values())
    passed_i = worst_i <= LIMIT_I
    conditions.append(
        Condition(
            "(i)",
            f"median path error of the asymmetric fit on ramps and steps <= {LIMIT_I}",
            f"semi-synthetic {errors['semi_synthetic']:.3f}, real plus injection "
            f"{errors['real_plus_injection']:.3f}, worse {worst_i:.3f}",
            worst_i <= LIMIT_I,
        )
    )

    # (ii) C2: asymmetric / season-year constant - 1 on ramps and steps of 10% or more.
    left = tensor_of(frame=scores, fit=asym, cases=C2_CASES, column="path_error")
    right = tensor_of(frame=scores, fit=FIT_NAMES[4], cases=C2_CASES, column="path_error")
    c2 = contrast(left=left, right=right, statistic=median_of_medians, draws=draws, relative=True)
    c2_primaries_only = contrast(
        left=left,
        right=right,
        statistic=median_of_medians,
        draws=draws,
        relative=True,
        resample_columns=False,
    )
    conditions.append(
        Condition(
            "(ii)",
            f"upper limit of C2 (path error relative to the season-year constant) < {LIMIT_II}",
            f"C2 {c2}; Primaries-only interval [{c2_primaries_only.low:+.3f}, "
            f"{c2_primaries_only.high:+.3f}]",
            c2.high < LIMIT_II,
        )
    )

    # (iii) false steps on both skies.
    constant_rate, sky_rate = false_step_tensors(scores=scores, placebo=placebo, fit=asym)
    stats = {
        "constant path": (median_of_means(constant_rate), worst_of_means(constant_rate)),
        "placebo sky": (median_of_means(sky_rate), worst_of_means(sky_rate)),
    }
    passed_iii = all(
        median <= LIMIT_FALSE_MEDIAN and worst <= LIMIT_FALSE_WORST
        for median, worst in stats.values()
    )
    conditions.append(
        Condition(
            "(iii)",
            f"false steps of 10% per Primary-year, median <= {LIMIT_FALSE_MEDIAN} and worst <= "
            f"{LIMIT_FALSE_WORST}, on both skies",
            "; ".join(f"{sky} median {m:.2f}, worst {w:.2f}" for sky, (m, w) in stats.items()),
            passed_iii,
        )
    )
    median_false = max(median for median, _ in stats.values())

    # (iv) ratchet drift.
    constant_drift, sky_drift = _drift_tensors(scores=scores, placebo=placebo, fit=asym)
    drift = {
        "constant path": median_of_medians(constant_drift),
        "placebo sky": median_of_medians(sky_drift),
    }
    conditions.append(
        Condition(
            "(iv)",
            f"ratchet drift <= {LIMIT_DRIFT} of capacity per 5 years, on both skies",
            "; ".join(f"{sky} {value:+.3f}" for sky, value in drift.items()),
            all(value <= LIMIT_DRIFT for value in drift.values()),
        )
    )

    # (v) temporary falls.
    fall_26 = min(
        median_of_medians(
            tensor_of(
                frame=scores, fit=asym, cases=("fall_30_26",), column="fall_recovered", variant=v
            )
        )
        for v in VARIANTS
    )
    fall_8 = max(
        median_of_medians(
            tensor_of(
                frame=scores, fit=asym, cases=("fall_30_8",), column="fall_recovered", variant=v
            )
        )
        for v in VARIANTS
    )
    conditions.append(
        Condition(
            "(v)",
            f"30% fall of 26 weeks recovered >= {LIMIT_FALL_26} and of 8 weeks < {LIMIT_FALL_8}",
            f"26 weeks {fall_26:.3f} (worse baseline), 8 weeks {fall_8:.3f} (worse baseline)",
            fall_26 >= LIMIT_FALL_26 and fall_8 < LIMIT_FALL_8,
        )
    )

    # (vi) coverage, from the bootstrap script's tables.
    coverage_path = run.out_dir / "coverage_summary.parquet"
    if coverage_path.exists():
        coverage = pl.read_parquet(coverage_path).filter(pl.col("group") == "all")
        by_kind = {r["weeks"]: r["coverage_90"] for r in coverage.iter_rows(named=True)}
        non_step, step = by_kind.get("non-step"), by_kind.get("step")
        passed_vi = (
            None
            if non_step is None or step is None
            else LIMIT_COVERAGE_NON_STEP[0] <= non_step <= LIMIT_COVERAGE_NON_STEP[1]
            and step >= LIMIT_COVERAGE_STEP
        )
        number = f"non-step weeks {non_step}, step weeks {step}"
    else:
        passed_vi, number = None, "coverage bootstrap has not run"
    conditions.append(
        Condition(
            "(vi)",
            f"90% interval covers non-step weeks in {LIMIT_COVERAGE_NON_STEP} and step weeks >= "
            f"{LIMIT_COVERAGE_STEP}",
            number,
            passed_vi,
        )
    )

    detection = detection_limits(scores=scores, fit=asym)
    exceeds = {
        season: int(
            detection.filter(
                (pl.col("season") == season) & (pl.col("limit_percent") > LIMIT_DETECTION * 100)
            ).height
        )
        for season in ("Apr-Sep", "Oct-Mar")
    }
    selftests = _read_selftests(out_dir=run.out_dir)
    verdict, reason = verdict_of(
        frozen=frozen,
        selftests_failed=selftests is False,
        detection_exceeds=exceeds,
        n_primaries=n_primaries,
        passed={c.label: c.passed for c in conditions},
        median_false=median_false,
        c2=c2,
    )

    report.section("Decision conditions (i) to (vi)")
    for condition in conditions:
        report.line(str(condition))
    report.line(
        "Self-tests: "
        + {True: "pass", False: "FAIL", None: "not run in this session"}[selftests]
        + "."
    )
    report.line(
        f"Primaries whose detection limit exceeds {LIMIT_DETECTION * 100:.0f}%: {exceeds} "
        f"(of {n_primaries})."
    )
    report.line(f"Median false-step rate (worse sky): {median_false:.2f} per Primary-year.")
    report.line(f"VERDICT (the plan's clauses tested in their order): {verdict}. {reason}")
    holding = [
        name
        for name, holds in (
            ("a self-test failed", selftests is False),
            (
                "detection limit above 25% at 5 or more Primaries in every season",
                all(count >= min(5, n_primaries) for count in exceeds.values()),
            ),
            ("no pair on the tuning grid met the tuning limits", frozen.status != "frozen"),
            ("(i) fails", passed_i is False),
            (
                f"median false-step rate above {LIMIT_REFUTED_FALSE}",
                median_false > LIMIT_REFUTED_FALSE,
            ),
            ("the lower limit of C2 is above 0", c2.low > 0),
        )
        if holds
    ]
    report.line("Clauses that hold: " + ("; ".join(holding) or "none") + ".")

    _add_comparison_tables(report=report, scores=scores, placebo=placebo, draws=draws)
    _add_breakdowns(report=report, scores=scores, placebo=placebo, detection=detection)
    _add_vintages(report=report, run=run)
    _add_coverage(report=report, run=run)
    return report


def verdict_of(  # noqa: PLR0911
    *,
    frozen: Frozen,
    selftests_failed: bool,
    detection_exceeds: dict[str, int],
    n_primaries: int,
    passed: dict[str, bool | None],
    median_false: float,
    c2: Interval,
) -> tuple[str, str]:
    """Apply the plan's verdict logic, in its order, and name the clause that decided.

    Args:
        frozen: The frozen penalty pair and its status.
        selftests_failed: Whether a self-test failed.
        detection_exceeds: Per season, the number of Primaries whose detection limit exceeds 25%.
        n_primaries: Number of Primaries scored.
        passed: Condition label to pass, fail, or `None` (not scored).
        median_false: Larger of the two skies' median false-step rates per Primary-year.
        c2: The interval of contrast C2.

    Returns:
        The verdict and a sentence naming the decisive clause.
    """
    if selftests_failed:
        return "inconclusive", "a self-test failed."
    if all(count >= min(5, n_primaries) for count in detection_exceeds.values()):
        return "inconclusive", "the smallest detectable step exceeds 25% at 5 or more Primaries."
    if frozen.status != "frozen":
        return "refuted", "no pair on the tuning grid met the tuning limits (frozen_config.json)."
    if passed["(i)"] is False:
        return "refuted", "condition (i) failed."
    if median_false > LIMIT_REFUTED_FALSE:
        return "refuted", f"the median false-step rate exceeds {LIMIT_REFUTED_FALSE}."
    if c2.low > 0:
        return "refuted", "the lower limit of C2 is above 0."
    if all(value is True for value in passed.values()):
        return "supported", "all six conditions hold."
    if passed["(i)"] and c2.high < 0:
        return "partly supported", "(i) holds, C2 is below 0, and one of (ii) to (vi) fails."
    return "inconclusive", "no clause of the plan's logic applies."


def _read_selftests(*, out_dir: Path) -> bool | None:
    """Return whether the recorded self-tests all passed, or `None` when none were recorded."""
    path = out_dir / "selftests.json"
    if not path.exists():
        return None
    return all(code == 0 for code in json.loads(path.read_text()).values())


def run_selftests(*, out_dir: Path) -> None:
    """Run both self-test scripts and record their exit codes."""
    codes = {
        script: subprocess.run(
            [sys.executable, str(FOLDER_PATH / script)], cwd=FOLDER_PATH, check=False
        ).returncode
        for script in SELFTEST_SCRIPTS
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selftests.json").write_text(json.dumps(codes, indent=2) + "\n")


def _add_comparison_tables(
    *, report: Report, scores: pl.DataFrame, placebo: pl.DataFrame, draws: Draws
) -> None:
    """Add the comparison-fit table and the contrasts C1, C3, and C4."""
    rows = []
    for fit in FIT_NAMES:
        constant, sky = false_step_tensors(scores=scores, placebo=placebo, fit=fit)
        constant_drift, sky_drift = _drift_tensors(scores=scores, placebo=placebo, fit=fit)
        row: dict[str, object] = {"fit": FIT_LABELS[fit]}
        for variant in VARIANTS:
            label = "" if variant == "semi_synthetic" else "_real_plus_injection"
            row[f"path_error_ramps_steps{label}"] = median_of_medians(
                tensor_of(
                    frame=scores,
                    fit=fit,
                    cases=CONDITION_I_CASES,
                    column="path_error",
                    variant=variant,
                )
            )
        row["path_error_falls"] = median_of_medians(
            tensor_of(frame=scores, fit=fit, cases=FALL_CASES, column="path_error")
        )
        row["path_error_constant"] = median_of_medians(
            tensor_of(frame=scores, fit=fit, cases=("constant_path",), column="path_error")
        )
        row["false_steps_constant_median"] = median_of_means(constant)
        row["false_steps_constant_worst"] = worst_of_means(constant)
        row["false_steps_placebo_median"] = median_of_means(sky)
        row["false_steps_placebo_worst"] = worst_of_means(sky)
        row["drift_constant"] = median_of_medians(constant_drift)
        row["drift_placebo"] = median_of_medians(sky_drift)
        for case in ("fall_30_26", "fall_30_8", "fall_15_26"):
            row[f"{case}_recovered"] = median_of_medians(
                tensor_of(frame=scores, fit=fit, cases=(case,), column="fall_recovered")
            )
        rows.append(row)
    report.section("Comparison fits (semi-synthetic unless named; medians over Primaries)")
    report.table("comparison_fits", pl.DataFrame(rows), transpose=True)

    def path_error(fit: str, cases: tuple[str, ...]) -> np.ndarray:
        return tensor_of(frame=scores, fit=fit, cases=cases, column="path_error")

    contrasts = []
    asym = "asymmetric"
    for label, other, cases in (
        ("C1 ramps", "free", RAMPS),
        ("C1 steps of 10% or more", "free", STEPS_10),
        ("C1 constant path", "free", ("constant_path",)),
        ("C4 temporary falls", "non_decreasing", FALL_CASES),
        ("C4 transfers outside the mask", "non_decreasing", ("transfer_outside_mask",)),
    ):
        result = contrast(
            left=path_error(asym, cases),
            right=path_error(other, cases),
            statistic=median_of_medians,
            draws=draws,
        )
        contrasts.append(
            _contrast_row(label, "path error, asymmetric minus " + FIT_LABELS[other], result)
        )
    sym_constant, sym_sky = false_step_tensors(scores=scores, placebo=placebo, fit="symmetric")
    asym_constant, asym_sky = false_step_tensors(scores=scores, placebo=placebo, fit=asym)
    contrasts.append(
        _contrast_row(
            "C3 false steps, placebo sky",
            "steps per Primary-year, asymmetric minus symmetric",
            contrast(left=asym_sky, right=sym_sky, statistic=median_of_means, draws=draws),
        )
    )
    contrasts.append(
        _contrast_row(
            "C3 false steps, constant path",
            "steps per Primary-year, asymmetric minus symmetric",
            contrast(
                left=asym_constant, right=sym_constant, statistic=median_of_means, draws=draws
            ),
        )
    )
    contrasts.append(
        _contrast_row(
            "C3 26-week fall recovered",
            "share of depth, asymmetric minus symmetric",
            contrast(
                left=tensor_of(
                    frame=scores, fit=asym, cases=("fall_30_26",), column="fall_recovered"
                ),
                right=tensor_of(
                    frame=scores, fit="symmetric", cases=("fall_30_26",), column="fall_recovered"
                ),
                statistic=median_of_medians,
                draws=draws,
            ),
        )
    )
    report.section("Contrasts C1, C3, C4 (descriptive; 95% intervals, no correction)")
    report.table("contrasts", pl.DataFrame(contrasts))


def _contrast_row(label: str, what: str, result: Interval) -> dict[str, object]:
    return {
        "contrast": label,
        "quantity": what,
        "point": result.point,
        "low": result.low,
        "high": result.high,
    }


def _add_breakdowns(
    *, report: Report, scores: pl.DataFrame, placebo: pl.DataFrame, detection: pl.DataFrame
) -> None:
    """Add the sub-breakdowns by case, season, size, fall, transfer, load, and Primary."""
    by_case = (
        scores.group_by("variant", "fit", "case", "primary")
        .agg(
            path_error=pl.col("path_error").median(),
            peak_error=pl.col("peak_error").median(),
            fall_recovered=pl.col("fall_recovered").median(),
            swing=pl.col("swing").median(),
        )
        .group_by("variant", "fit", "case")
        .agg(
            path_error=pl.col("path_error").median(),
            path_error_worst_primary=pl.col("path_error").max(),
            peak_error=pl.col("peak_error").median(),
            fall_recovered=pl.col("fall_recovered").median(),
            swing=pl.col("swing").median(),
        )
        .sort("variant", "case", "fit")
    )
    report.section("By case (median over Primaries of each Primary's median over farm-seeds)")
    report.table("by_case", by_case)

    family_rows = []
    for family, cases in CASE_FAMILIES.items():
        for fit in FIT_NAMES:
            for variant in VARIANTS:
                values = tensor_of(
                    frame=scores, fit=fit, cases=cases, column="path_error", variant=variant
                )
                if np.isfinite(values).any():
                    family_rows.append(
                        {
                            "family": family,
                            "fit": FIT_LABELS[fit],
                            "variant": variant,
                            "path_error": median_of_medians(values),
                        }
                    )
    report.section("By case family")
    report.table("by_family", pl.DataFrame(family_rows))

    steps = scores.filter(
        pl.col("case").is_in(list(STEPS)) & (pl.col("variant") == "semi_synthetic")
    ).with_columns(
        season=pl.when(pl.col("step_summer")).then(pl.lit("Apr-Sep")).otherwise(pl.lit("Oct-Mar"))
    )
    by_season = (
        steps.group_by("fit", "season", "primary")
        .agg(
            path_error=pl.col("path_error").median(),
            detected=pl.col("detected").cast(pl.Float64).mean(),
        )
        .group_by("fit", "season")
        .agg(path_error=pl.col("path_error").median(), detected=pl.col("detected").median())
        .sort("fit", "season")
    )
    report.section("Steps by the season of the step (Apr-Sep against Oct-Mar)")
    report.table("by_season", by_season)

    curve = (
        steps.group_by("fit", "season", "case")
        .agg(share_detected=pl.col("detected").cast(pl.Float64).mean(), n=pl.len())
        .sort("fit", "season", "case")
    )
    report.section("Detection curve: share of steps detected, by size and season")
    report.table("detection_curve", curve)
    report.section(
        "Detection limit of the asymmetric fit per Primary (smallest size detected in 80%)"
    )
    report.table("detection_limits", detection)

    falls = (
        scores.filter(pl.col("case").is_in(list(FALL_CASES)))
        .group_by("variant", "fit", "case", "primary")
        .agg(pl.col("fall_recovered").median())
        .group_by("variant", "fit", "case")
        .agg(
            fall_recovered=pl.col("fall_recovered").median(),
            worst_primary=pl.col("fall_recovered").min(),
        )
        .sort("variant", "case", "fit")
    )
    report.section("Fall recovery by depth and duration (share of the fall's depth shown)")
    report.table("fall_recovery", falls)

    transfers = by_case.filter(pl.col("case").is_in(list(TRANSFERS)))
    report.section("Transfers inside and outside the mask")
    report.table(
        "transfers", transfers.select("variant", "fit", "case", "path_error", "peak_error")
    )

    loads = by_case.filter(pl.col("case").is_in(list(LOADS)))
    report.section("Seasonal-load creep (seasonal swing of the error, share of capacity)")
    report.table("seasonal_load", loads.select("variant", "fit", "case", "swing", "path_error"))

    flagged = []
    for fit in FIT_NAMES:
        for sky, (constant, sky_rate) in (
            ("10%", false_step_tensors(scores=scores, placebo=placebo, fit=fit)),
            ("5%", false_step_tensors(scores=scores, placebo=placebo, fit=fit, column="steps_05")),
        ):
            primaries = [p for p in common.PRIMARIES if p in set(scores["primary"])]
            for index, primary in enumerate(primaries):
                flagged.append(
                    {
                        "fit": FIT_LABELS[fit],
                        "step_threshold": sky,
                        "primary": primary,
                        "name": common.PRIMARY_NAMES[primary],
                        "flag": common.PRIMARY_STATUS[primary],
                        "constant_path": _safe(np.mean, constant[index].ravel()),
                        "placebo_sky": _safe(np.mean, sky_rate[index].ravel()),
                        "path_error_ramps_steps": _safe(
                            np.median,
                            tensor_of(
                                frame=scores,
                                fit=fit,
                                cases=CONDITION_I_CASES,
                                column="path_error",
                            )[index].ravel(),
                        ),
                    }
                )
    report.section(
        "False steps per Primary-year and path error per Primary "
        f"(Kirton D5 and Wrangle T1 D15 are the flagged Primaries: {FLAGGED_PRIMARIES})"
    )
    report.table("per_primary", pl.DataFrame(flagged))


def _add_vintages(*, report: Report, run: Run) -> None:
    """Add the real-time vintage tables and contrast C5."""
    files = sorted(run.out_dir.glob("vintage_*.parquet"))
    report.section("Real-time vintages on injections (C5)")
    if not files:
        report.line("The vintage stage has not run.")
        return
    vintages = pl.concat([pl.read_parquet(f) for f in files])
    by_age = (
        vintages.group_by("age", "primary")
        .agg(
            error_vintage=pl.col("error_vintage").mean(),
            error_final=pl.col("error_final").mean(),
            n=pl.len(),
        )
        .with_columns(ratio=pl.col("error_vintage") / pl.col("error_final"))
        .group_by("age")
        .agg(
            error_vintage=pl.col("error_vintage").median(),
            error_final=pl.col("error_final").median(),
            ratio_median_primary=pl.col("ratio").median(),
            ratio_worst_primary=pl.col("ratio").max(),
            n_origin_units=pl.col("n").sum(),
        )
        .sort("age")
    )
    latest = by_age.filter(pl.col("age") == 1).row(0, named=True)
    report.line(
        f"C5 (latest completed week, age 1): vintage error {latest['error_vintage']:.3f}, final "
        f"error {latest['error_final']:.3f}, difference "
        f"{latest['error_vintage'] - latest['error_final']:+.3f}, ratio "
        f"{latest['ratio_median_primary']:.2f} (limit {LIMIT_C5}; reported, never alters the "
        "verdict)."
    )
    report.table("vintages_by_age", by_age)
    by_case = (
        vintages.filter(pl.col("age") == 1)
        .group_by("case", "primary")
        .agg(pl.col("error_vintage").mean(), pl.col("error_final").mean())
        .group_by("case")
        .agg(pl.col("error_vintage").median(), pl.col("error_final").median())
    )
    report.table("vintages_by_case", by_case)


def _add_coverage(*, report: Report, run: Run) -> None:
    """Add the coverage tables written by `e00045_bootstrap.py`."""
    report.section("Bootstrap coverage (residual block bootstrap, whole path refitted)")
    path = run.out_dir / "coverage_summary.parquet"
    if not path.exists():
        report.line("The coverage bootstrap has not run.")
        return
    report.table("coverage_summary_shown", pl.read_parquet(path))


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=[*STAGES, "selftests"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke", action="store_true", help="tuning farm A, two Primaries, three origins"
    )
    args = parser.parse_args()
    frozen = load_frozen()
    if args.smoke:
        run = Run(
            split="tuning",
            primaries=SMOKE_PRIMARIES,
            farms=("A",),
            out_dir=SMOKE_DIR,
            origins=vintage_origins()[:SMOKE_ORIGINS],
            workers=min(args.workers, len(SMOKE_PRIMARIES)),
        )
    else:
        run = Run(
            split="scoring",
            primaries=common.PRIMARIES,
            farms=inject.SCORING_FARMS,
            out_dir=SCORES_DIR,
            origins=vintage_origins(),
            workers=args.workers,
        )
    run.out_dir.mkdir(parents=True, exist_ok=True)
    if "selftests" in args.stages:
        run_selftests(out_dir=run.out_dir)
    if "jobs" in args.stages:
        run_fit_jobs(run=run, frozen=frozen)
    if "ratios" in args.stages:
        run_ratio_jobs(run=run, frozen=frozen)
    if "vintages" in args.stages:
        run_vintage_jobs(run=run, frozen=frozen)
    if "score" in args.stages or "report" in args.stages:
        scores_path = run.out_dir / "unit_scores.parquet"
        scores = (
            run_scoring(run=run)
            if "score" in args.stages or not scores_path.exists()
            else pl.read_parquet(scores_path)
        )
    if "report" in args.stages:
        build_report(run=run, frozen=frozen, scores=scores).write()


if __name__ == "__main__":
    main()
