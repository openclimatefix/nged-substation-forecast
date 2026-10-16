"""Residual block bootstrap of the frozen fit: coverage of the 90% interval (experiment 00045).

For 96 scoring units (one path seed, the 3 scoring farms, the 8 Primaries, and 4 cases: a ramp of
50%, a step of 10%, a 30% fall lasting 26 weeks, and the constant path on the placebo sky) the
script fits the frozen asymmetric pair, takes the half-hourly residuals, and draws new series from
them. Each draw adds residuals to the fitted values and refits the whole weekly path with the same
penalties, so the penalty's bias at a step enters the interval. The residuals come in 7-day blocks:
the block of a target week is a random week of the same calendar month, from any year of the
Primary's record, and keeps its half-hour of the week. A half-hour with no residual in the source
block takes zero. The 5th and 95th percentiles of the draws give the 90% interval of every week;
the 25th and 75th and the 10th and 90th give a calibration curve. Coverage of the known path is
scored separately at weeks within 2 weeks of a true step (a weekly change of at least 5% of
capacity) and at the other weeks.

Each job (one Primary and one farm) writes its own file and is skipped when the file exists.
`--smoke` runs one tuning farm on two Primaries with 5 draws into `scores/smoke/`.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_bootstrap.py
"""

import argparse
import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00045_common as common
import e00045_inject as inject
import e00045_model as model
import e00045_score as score
import numpy as np
import polars as pl

COVERAGE_CASES: Final[tuple[inject.CaseName, ...]] = (
    "ramp_50",
    "step_10",
    "fall_30_26",
    "constant_path",
)
"""The four case families: a ramp, a step, a fall, and the constant path."""
DEFAULT_DRAWS: Final[int] = 200
STEP_THRESHOLD: Final[float] = 0.05
"""A weekly change in the true path of at least this share of capacity is a step."""
STEP_NEIGHBOURHOOD_WEEKS: Final[int] = 2
QUANTILES: Final[tuple[float, ...]] = (5, 10, 25, 75, 90, 95)


def residuals(*, rows: model.Rows, path: np.ndarray) -> np.ndarray:
    """Return the half-hourly residuals of the fit: the series minus its fitted values.

    The model is `power = cell + b * HDD - c(w) * u + error`. Given the weekly capacity `c`, the
    demand cells and `b` are the least-squares fit of `power + c * u` on the cell means and the HDD.

    Args:
        rows: The usable half-hours.
        path: Fitted weekly capacity of every week (NaN outside the fitted span), MW.

    Returns:
        The residual of every row, MW.
    """
    target = rows.power + np.nan_to_num(path)[rows.week] * rows.u
    _, cell_index = np.unique(rows.cells, return_inverse=True)
    counts = np.bincount(cell_index).astype(float)

    def demean(values: np.ndarray) -> np.ndarray:
        return values - (np.bincount(cell_index, weights=values) / counts)[cell_index]

    target_centred, hdd_centred = demean(target), demean(rows.hdd)
    slope = float(hdd_centred @ target_centred / max(hdd_centred @ hdd_centred, 1e-12))
    return target_centred - slope * hdd_centred


def residual_grid(*, table: pl.DataFrame, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals laid out by week, day of week, and half-hour, with each row's slot.

    Args:
        table: The rows (with `time` and `week`) that `residual` belongs to.
        residual: The residual of every row, MW.

    Returns:
        A zero-filled array of shape (weeks, 7, 48), and the (day of week, half-hour) slot of each
        row as an array of shape (rows, 2).
    """
    shifted = table["time"].dt.offset_by("-15m")
    day_of_week = (shifted.dt.weekday().cast(pl.Int64) - 1).to_numpy()
    half_hour = (shifted.dt.hour().cast(pl.Int64) * 2 + shifted.dt.minute() // 30).to_numpy()
    week = table["week"].to_numpy().astype(np.int64)
    grid = np.zeros((common.N_WEEKS, 7, 48))
    grid[week, day_of_week, half_hour] = residual
    return grid, np.stack([day_of_week, half_hour], axis=1)


def source_weeks(*, observed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw, for every week, a week of the same calendar month from the observed weeks.

    Args:
        observed: Mask of the weeks that hold rows.
        rng: The random generator.

    Returns:
        The source week of every week (itself for a week with no candidate).
    """
    months = np.array([common.week_start(week=w).month for w in range(common.N_WEEKS)])
    chosen = np.arange(common.N_WEEKS)
    for month in np.unique(months):
        pool = np.nonzero(observed & (months == month))[0]
        targets = np.nonzero(months == month)[0]
        if pool.size:
            chosen[targets] = rng.choice(pool, size=len(targets))
    return chosen


def step_week_mask(*, truth: np.ndarray, capacity: float) -> np.ndarray:
    """Return the weeks within 2 weeks of a true step (a weekly change of 5% of capacity)."""
    mask = np.zeros(len(truth), dtype=bool)
    for week in np.nonzero(np.abs(np.diff(truth)) >= STEP_THRESHOLD * capacity)[0] + 1:
        mask[max(0, week - STEP_NEIGHBOURHOOD_WEEKS) : week + STEP_NEIGHBOURHOOD_WEEKS + 1] = True
    return mask


def unit_series(
    *,
    unit: inject.Unit,
    base: inject.PrimaryBase,
    inputs: inject.Inputs,
    farm_rows: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return a unit's rows, the power that is fitted, and the true path (semi-synthetic baseline).

    Args:
        unit: The unit.
        base: The Primary's baseline.
        inputs: The loaded tables.
        farm_rows: `inject.unit_rows(...)` without a mask window.

    Returns:
        The rows, the power of every row, MW, and the true capacity of every week, MW.
    """
    if unit.case == "constant_path":
        rows = inject.permute_u(table=farm_rows, rng=inject._rng(unit=unit))
        power = rows["power"].to_numpy() - base.capacity * rows["u"].to_numpy()
        return rows, power, np.full(common.N_WEEKS, base.capacity)
    injection = inject.build_injection(unit=unit, base=base, farm_rows=farm_rows, inputs=inputs)
    power = farm_rows["baseline_power"].to_numpy().astype(float) + injection.delta_for(farm_rows)
    return farm_rows, power, base.capacity + injection.increment


def bootstrap_unit(
    *,
    unit: inject.Unit,
    base: inject.PrimaryBase,
    inputs: inject.Inputs,
    farm_rows: pl.DataFrame,
    frozen: score.Frozen,
    ratio: float,
    n_draws: int,
) -> pl.DataFrame:
    """Bootstrap one unit and return its per-week interval and truth.

    Args:
        unit: The unit.
        base: The Primary's baseline.
        inputs: The loaded tables.
        farm_rows: `inject.unit_rows(...)` without a mask window.
        frozen: The frozen penalty pair.
        ratio: The Primary's recovery ratio of the asymmetric fit (1 for the constant path).
        n_draws: Number of bootstrap draws.

    Returns:
        One row per observed week with the truth, the estimate, and the six percentiles.
    """
    rows, power, truth = unit_series(unit=unit, base=base, inputs=inputs, farm_rows=farm_rows)
    design = score.design_for(rows=rows, power=power)
    spec = frozen.fits()[0]
    fit = model.FITS[spec[0]]
    path = fit(design=design, a=spec[1], r=spec[2])
    resid = residuals(rows=design.rows, path=path)
    fitted = design.rows.power - resid
    grid, slot = residual_grid(table=rows, residual=resid)
    observed = inject.observed_weeks(table=rows)
    rng = np.random.default_rng([unit.seed, common.PRIMARIES.index(unit.primary), 45])
    estimates = np.full((n_draws, common.N_WEEKS), np.nan)
    for draw in range(n_draws):
        source = source_weeks(observed=observed, rng=rng)
        new_power = fitted + grid[source[design.rows.week], slot[:, 0], slot[:, 1]]
        redrawn = model.Design(
            rows=dataclasses.replace(design.rows, power=new_power),
            n_weeks=design.n_weeks,
            week_season_year=design.week_season_year,
        )
        estimates[draw] = (
            base.capacity + (fit(design=redrawn, a=spec[1], r=spec[2]) - base.capacity) / ratio
        )
    weeks = np.nonzero(observed)[0]
    percentiles = np.nanpercentile(estimates[:, weeks], QUANTILES, axis=0)
    point = base.capacity + (path - base.capacity) / ratio
    steps = step_week_mask(truth=truth, capacity=base.capacity)
    return pl.DataFrame(
        {
            "primary": unit.primary,
            "farm": unit.farm,
            "seed": unit.seed,
            "case": unit.case,
            "week": weeks,
            "truth": truth[weeks],
            "estimate": point[weeks],
            "step_week": steps[weeks],
            **{f"q{int(q):02d}": percentiles[index] for index, q in enumerate(QUANTILES)},
            "n_draws": n_draws,
        }
    )


def bootstrap_job(
    *,
    primary: str,
    farm: str,
    split: inject.SplitType,
    out_dir: Path,
    n_draws: int,
    pair: score.Frozen | None = None,
) -> Path:
    """Bootstrap the 4 units of one Primary and one farm, and write the table atomically.

    Args:
        primary: Frame site code.
        farm: Farm label of `split`.
        split: Which split's farms and path seed to use (the first path seed).
        out_dir: Directory of the scores (the ratios are read from it).
        n_draws: Number of bootstrap draws per unit.
        pair: The penalty pair to bootstrap; the frozen pair when `None`.

    Returns:
        The path of the output file.
    """
    out_file = out_dir / f"coverage_{split}_{primary}_{farm}.parquet"
    if out_file.exists():
        return out_file
    inject.assert_split(farm=farm, seeds=inject.SEEDS_BY_SPLIT[split], split=split)
    frozen = score.load_frozen() if pair is None else pair
    ratios = score.read_ratios(out_dir=out_dir)
    inputs = inject.load_inputs()
    base = inject.primary_base(inputs=inputs, site=primary)
    farm_rows = inject.unit_rows(base=base, inputs=inputs, farm=farm)
    seed = inject.SEEDS_BY_SPLIT[split][0]
    tables = []
    for case in COVERAGE_CASES:
        started = time.monotonic()
        unit = inject.Unit(primary=primary, farm=farm, seed=seed, case=case)
        try:
            tables.append(
                bootstrap_unit(
                    unit=unit,
                    base=base,
                    inputs=inputs,
                    farm_rows=farm_rows,
                    frozen=frozen,
                    ratio=1.0 if case == "constant_path" else ratios[(primary, "asymmetric")],
                    n_draws=n_draws,
                )
            )
        except inject.UnbuildableUnitError:
            print(f"{primary} {farm} {case}: skipped, the record is too short", flush=True)
            continue
        print(f"{primary} {farm} {case}: {time.monotonic() - started:.0f} s", flush=True)
    score.atomic_write(frame=pl.concat(tables), path=out_file)
    return out_file


def _job(args: tuple[str, str, inject.SplitType, Path, int]) -> str:
    primary, farm, split, out_dir, n_draws = args
    bootstrap_job(primary=primary, farm=farm, split=split, out_dir=out_dir, n_draws=n_draws)
    return f"coverage {primary} {farm}"


def coverage_summary(*, tables: pl.DataFrame) -> pl.DataFrame:
    """Return the share of weeks whose truth lies inside the 50%, 80%, and 90% intervals.

    Args:
        tables: Every job's per-week table.

    Returns:
        One row per group (all units, each case, each Primary) and kind of week (step or not).
    """
    flagged = tables.with_columns(
        weeks=pl.when(pl.col("step_week")).then(pl.lit("step")).otherwise(pl.lit("non-step")),
        inside_90=(pl.col("truth") >= pl.col("q05")) & (pl.col("truth") <= pl.col("q95")),
        inside_80=(pl.col("truth") >= pl.col("q10")) & (pl.col("truth") <= pl.col("q90")),
        inside_50=(pl.col("truth") >= pl.col("q25")) & (pl.col("truth") <= pl.col("q75")),
    )
    groups = [
        flagged.with_columns(group=pl.lit("all")),
        flagged.with_columns(group="case:" + pl.col("case")),
        flagged.with_columns(group="primary:" + pl.col("primary")),
    ]
    return (
        pl.concat(groups)
        .group_by("group", "weeks")
        .agg(
            coverage_90=pl.col("inside_90").mean(),
            coverage_80=pl.col("inside_80").mean(),
            coverage_50=pl.col("inside_50").mean(),
            n_weeks=pl.len(),
            n_units=pl.struct("primary", "farm", "case").n_unique(),
            mean_half_width_90=((pl.col("q95") - pl.col("q05")) / 2).mean(),
        )
        .sort("group", "weeks")
    )


def main() -> None:
    """Run the jobs and write `coverage_summary.parquet` and `coverage_report.txt`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        split: inject.SplitType = "tuning"
        out_dir, primaries, farms, draws = (
            score.SMOKE_DIR,
            score.SMOKE_PRIMARIES,
            ("A",),
            5,
        )
    else:
        split = "scoring"
        out_dir, primaries, farms, draws = (
            score.SCORES_DIR,
            common.PRIMARIES,
            inject.SCORING_FARMS,
            args.draws,
        )
    jobs = [(p, f, split, out_dir, draws) for f in farms for p in primaries]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)
    tables = pl.concat(
        [pl.read_parquet(f) for f in sorted(out_dir.glob(f"coverage_{split}_*.parquet"))]
    )
    summary = coverage_summary(tables=tables)
    score.atomic_write(frame=summary, path=out_dir / "coverage_summary.parquet")
    text = (
        f"Coverage of the 90% residual-block-bootstrap interval ({draws} draws per unit; "
        f"{tables.select('primary', 'farm', 'case').unique().height} units).\n"
        + score.table_text(summary)
        + "\n"
    )
    (out_dir / "coverage_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
