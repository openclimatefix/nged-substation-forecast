"""Tune the asymmetric L1 penalty pair on the tuning units of experiment 00045 and freeze it.

The grid is `a` in {0.5, 1, 2, 4} times `r` in {10, 30}. Every pair is fitted on the tuning units
only (Solar farms A to C, path seeds 0 to 3, all 8 Primaries, 14 cases) and on the placebo sky
(permutation seeds 0 to 99). A pair is feasible when, at the median Primary, false steps of 10% of
capacity are at most 0.5 per Primary-year on both skies (and at most 1.0 at the worst Primary), and
a 30% fall lasting 26 weeks is recovered at 0.5 or more of its depth. The feasible pair with the
smallest median path error over ramps, steps, and the constant path is written to
`frozen_config.json` in this folder. When no pair is feasible the file says `refuted-fallback`.
The scoring units are never fitted here.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_tune.py
"""

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypedDict

import e00045_common as common
import e00045_inject as inject
import e00045_metrics as metrics
import numpy as np
import polars as pl

GRID_A: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 4.0)
GRID_R: Final[tuple[float, ...]] = (10.0, 30.0)
PAIRS: Final[tuple[inject.FitSpec, ...]] = tuple(
    ("asymmetric", a, r) for a in GRID_A for r in GRID_R
)
VARIANTS: Final[tuple[inject.VariantType, ...]] = ("semi_synthetic",)
"""The tuning scores use the semi-synthetic variant, whose truth is known. The paired difference of
the real-plus-injection variant is only a secondary additivity measure, taken at scoring."""
MAX_STEPS_MEDIAN: Final[float] = 0.5
MAX_STEPS_WORST: Final[float] = 1.0
MIN_FALL_RECOVERED: Final[float] = 0.5
"""The tuning limits: false steps of 10% per Primary-year at the median and worst Primary, and the
share of a 30% fall lasting 26 weeks that is recovered."""
TIE_TOLERANCE: Final[float] = 0.001
"""Pairs whose median path error is within this of the best are tied; the larger `a` wins."""
MIXTURE_CASES: Final[tuple[str, ...]] = (
    "ramp_25",
    "ramp_50",
    "step_05",
    "step_10",
    "step_20",
    "step_30",
    "constant_path",
)
SCRIPT_FILES: Final[tuple[str, ...]] = (
    "e00045_tune.py",
    "e00045_inject.py",
    "e00045_metrics.py",
    "e00045_model.py",
)
FROZEN_PATH: Final[Path] = Path(__file__).resolve().parent / "frozen_config.json"
TUNING_DIR: Final[Path] = common.DATA_DIR / "tuning"
FOLDER_PATH: Final[Path] = Path(__file__).resolve().parent


class PairRow(TypedDict):
    """One row of the tuning table: the scores of one `(a, r)` pair."""

    a: float
    r: float
    path_error_mixture: float
    path_error_ramps: float
    path_error_steps: float
    path_error_constant: float
    steps_placebo_median: float
    steps_placebo_worst: float
    steps_constant_median: float
    steps_constant_worst: float
    fall_26_recovered: float
    fall_8_recovered: float
    drift_per_5y_constant: float
    drift_per_5y_placebo: float
    feasible: bool
    violation: float


def git_state() -> tuple[str, bool]:
    """Return the last commit that touched the tuning scripts, and whether they are modified."""
    paths = [str(FOLDER_PATH / name) for name in SCRIPT_FILES]
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *paths],
        capture_output=True,
        text=True,
        check=True,
        cwd=FOLDER_PATH,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        check=True,
        cwd=FOLDER_PATH,
    ).stdout.strip()
    return commit, bool(dirty)


def _unit_job(args: tuple[str, str, inject.VariantType, Path]) -> str:
    primary, farm, variant, out_dir = args
    inject.run_unit_job(
        primary=primary, farm=farm, split="tuning", fits=PAIRS, out_dir=out_dir, variant=variant
    )
    return f"units {variant} {primary} {farm}"


def _placebo_job(args: tuple[str, Path]) -> str:
    primary, out_dir = args
    inject.run_placebo_job(primary=primary, split="tuning", fits=PAIRS, out_dir=out_dir)
    return f"placebo {primary}"


def run_jobs(
    *, primaries: tuple[str, ...], farms: tuple[str, ...], workers: int, out_dir: Path
) -> None:
    """Run the unit and placebo jobs on `workers` single-threaded processes."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        unit_args = [(p, f, v, out_dir) for v in VARIANTS for f in farms for p in primaries]
        placebo_args = [(p, out_dir) for p in primaries]
        for line in pool.map(_placebo_job, placebo_args):
            print(line, flush=True)
        for line in pool.map(_unit_job, unit_args):
            print(line, flush=True)


def _read_variant(
    *, out_dir: Path, variant: inject.VariantType
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the units and paths tables of every job of one variant."""
    units = pl.concat(
        [pl.read_parquet(f) for f in sorted(out_dir.glob(f"tuning_{variant}_*_units.parquet"))]
    )
    paths = pl.concat(
        [pl.read_parquet(f) for f in sorted(out_dir.glob(f"tuning_{variant}_*_paths.parquet"))]
    )
    return units, paths


def unit_scores(
    *,
    out_dir: Path,
    variant: inject.VariantType,
    ratios: dict[tuple[str, inject.FitSpec], float],
    real_mean: dict[tuple[str, inject.FitSpec], float],
) -> pl.DataFrame:
    """Score every tuning unit and fit of one variant: path error, steps, drift, fall recovery.

    The path error is the mean absolute difference between the estimate and the truth over the
    observed weeks, divided by the mean of the true total capacity (the baseline capacity plus
    the injected increment) over those weeks. The steps and the drift are scored on the constant
    path of the semi-synthetic variant only, because only that baseline holds no real growth.

    Args:
        out_dir: Directory of the unit jobs' output.
        variant: Baseline variant.
        ratios: Recovery ratio of each (Primary, fit) in this variant.
        real_mean: Mean capacity of the unpermuted real series under each (Primary, fit).

    Returns:
        One row per unit and fit.
    """
    units, paths = _read_variant(out_dir=out_dir, variant=variant)
    key = ["primary", "farm", "seed", "case"]
    joined = paths.join(units.drop("variant"), on=key)
    rows = []
    for row in joined.iter_rows(named=True):
        spec: inject.FitSpec = (row["fit"], row["a"], row["r"])
        ratio = ratios[(row["primary"], spec)]
        observed = np.array(row["observed"], dtype=bool)
        increment = np.array(row["increment"], dtype=float)
        estimate = inject.estimate_path(
            with_injection=np.array(row["with_injection"], dtype=float),
            baseline=None if row["baseline"] is None else np.array(row["baseline"], dtype=float),
            ratio=ratio,
            variant=variant,
            base_capacity=row["base_capacity"],
        )
        truth = inject.true_path(
            increment=increment, base_capacity=row["base_capacity"], variant=variant
        )
        scale = float(np.mean((row["base_capacity"] + increment)[observed]))
        out: dict[str, object] = {
            "variant": variant,
            "primary": row["primary"],
            "farm": row["farm"],
            "seed": row["seed"],
            "case": row["case"],
            "a": row["a"],
            "r": row["r"],
            "ratio": ratio,
            "path_error": metrics.path_error(
                estimate=estimate, truth=truth, weeks=observed, scale=scale
            ),
            "fall_recovered": None,
            "steps_10": None,
            "drift_per_5y": None,
        }
        if row["fall_start"] is not None:
            share = metrics.fall_recovered(
                estimate=estimate,
                truth=truth,
                weeks=observed,
                start=row["fall_start"],
                duration=row["fall_weeks"],
            )
            out["fall_recovered"] = share if np.isfinite(share) else None
        if row["case"] == "constant_path" and variant == "semi_synthetic":
            threshold = 0.10 * real_mean[(row["primary"], spec)]
            steps = metrics.count_steps(path=estimate, threshold=threshold)
            out["steps_10"] = metrics.steps_per_year(
                steps=steps, n_weeks=int(np.isfinite(estimate).sum())
            )
            out["drift_per_5y"] = metrics.ratchet_drift_per_5_years(
                path=estimate, reference=float(np.nanmean(truth))
            )
        rows.append(out)
    return pl.DataFrame(rows, infer_schema_length=None)


def recovery_ratios(
    *, out_dir: Path, variant: inject.VariantType
) -> dict[tuple[str, inject.FitSpec], float]:
    """Return the recovery ratio of each Primary and fit from the variant's constant-path units."""
    units, paths = _read_variant(out_dir=out_dir, variant=variant)
    return {
        (primary, spec): inject.recovery_ratio(
            units=units.filter(pl.col("primary") == primary),
            paths=paths.filter(pl.col("primary") == primary),
            spec=spec,
        )
        for primary in units["primary"].unique().sort()
        for spec in PAIRS
    }


def _median(series: pl.Series) -> float:
    """Return the median of a series as a float."""
    return float(np.median(series.to_numpy()))


def _worst(series: pl.Series) -> float:
    """Return the largest value of a series as a float."""
    return float(np.max(series.to_numpy()))


def _by_primary(mine: pl.DataFrame, cases: list[str], column: str = "path_error") -> pl.Series:
    """Return the median of `column` over the units of `cases`, one value per Primary."""
    return (
        mine.filter(pl.col("case").is_in(cases))
        .group_by("primary")
        .agg(pl.col(column).median())[column]
    )


def _worse_of_variants(
    mine: pl.DataFrame,
    cases: list[str],
    column: str = "path_error",
    *,
    larger_is_worse: bool = True,
) -> float:
    """Return the median over Primaries in the worse of the two variants."""
    medians = [
        _median(_by_primary(mine.filter(pl.col("variant") == variant), cases, column))
        for variant in VARIANTS
    ]
    return max(medians) if larger_is_worse else min(medians)


def tuning_table(*, scores: pl.DataFrame, placebo: pl.DataFrame) -> list[PairRow]:
    """Aggregate unit scores into one row per pair: medians over units, then over Primaries.

    A path error or a fall recovery is taken in the worse of the two baseline variants, as the
    plan's decision rule scores condition (i). False steps and drift come from the semi-synthetic
    constant path and the placebo sky.

    Args:
        scores: `unit_scores` of both variants, concatenated.
        placebo: The placebo jobs' counts.

    Returns:
        One row per `(a, r)` pair.
    """
    table: list[PairRow] = []
    for _, a, r in PAIRS:
        mine = scores.filter((pl.col("a") == a) & (pl.col("r") == r))
        steps_placebo = (
            placebo.filter((pl.col("a") == a) & (pl.col("r") == r))
            .with_columns(rate=pl.col("steps_10") / (pl.col("n_weeks") / metrics.WEEKS_PER_YEAR))
            .group_by("primary")
            .agg(pl.col("rate").mean(), pl.col("drift_per_5y").median())
        )
        constant = (
            mine.filter((pl.col("case") == "constant_path") & (pl.col("variant") == VARIANTS[0]))
            .group_by("primary")
            .agg(pl.col("steps_10").mean(), pl.col("drift_per_5y").median())
        )
        mixture = list(MIXTURE_CASES)
        row: PairRow = {
            "a": a,
            "r": r,
            "path_error_mixture": _worse_of_variants(mine, mixture),
            "path_error_ramps": _worse_of_variants(mine, ["ramp_25", "ramp_50"]),
            "path_error_steps": _worse_of_variants(
                mine, ["step_05", "step_10", "step_20", "step_30"]
            ),
            "path_error_constant": _worse_of_variants(mine, ["constant_path"]),
            "steps_placebo_median": _median(steps_placebo["rate"]),
            "steps_placebo_worst": _worst(steps_placebo["rate"]),
            "steps_constant_median": _median(constant["steps_10"]),
            "steps_constant_worst": _worst(constant["steps_10"]),
            "fall_26_recovered": _worse_of_variants(
                mine, ["fall_30_26"], "fall_recovered", larger_is_worse=False
            ),
            "fall_8_recovered": _worse_of_variants(
                mine, ["fall_30_8"], "fall_recovered", larger_is_worse=False
            ),
            "drift_per_5y_constant": _median(constant["drift_per_5y"]),
            "drift_per_5y_placebo": _median(steps_placebo["drift_per_5y"]),
            "feasible": False,
            "violation": 0.0,
        }
        row["violation"] = max(
            0.0,
            row["steps_placebo_median"] - MAX_STEPS_MEDIAN,
            row["steps_constant_median"] - MAX_STEPS_MEDIAN,
            row["steps_placebo_worst"] - MAX_STEPS_WORST,
            row["steps_constant_worst"] - MAX_STEPS_WORST,
            MIN_FALL_RECOVERED - row["fall_26_recovered"],
        )
        row["feasible"] = row["violation"] == 0.0
        table.append(row)
    return table


def choose(*, table: list[PairRow]) -> PairRow | None:
    """Return the feasible pair with the smallest mixture error, ties to the larger `a`, or None."""
    feasible = [row for row in table if row["feasible"]]
    if not feasible:
        return None
    best = min(row["path_error_mixture"] for row in feasible)
    tied = [row for row in feasible if row["path_error_mixture"] <= best + TIE_TOLERANCE]
    return max(tied, key=lambda row: (row["a"], -row["r"]))


def main() -> None:
    """Run the tuning jobs, aggregate, print the table, and write the frozen file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--primaries", nargs="*", default=list(common.PRIMARIES))
    parser.add_argument("--farms", nargs="*", default=list(inject.TUNING_FARMS))
    parser.add_argument("--out-dir", type=Path, default=TUNING_DIR)
    parser.add_argument("--no-freeze", action="store_true", help="print the table, write nothing")
    args = parser.parse_args()
    commit, dirty = git_state()
    if dirty and not args.no_freeze:
        raise SystemExit("the tuning scripts have uncommitted changes: commit them before freezing")
    for farm in args.farms:
        inject.assert_split(farm=farm, seeds=inject.TUNING_SEEDS, split="tuning")
    print(
        f"tuning units: {len(inject.TUNING_FARMS)} farms x {len(inject.TUNING_SEEDS)} seeds x "
        f"{len(common.PRIMARIES)} Primaries x {len(inject.CASES)} cases = "
        f"{len(inject.units_for(split='tuning'))}; scoring units (not fitted): "
        f"{len(inject.units_for(split='scoring'))}"
    )
    run_jobs(
        primaries=tuple(args.primaries),
        farms=tuple(args.farms),
        workers=args.workers,
        out_dir=args.out_dir,
    )
    placebo = pl.concat(
        [pl.read_parquet(f) for f in sorted(args.out_dir.glob("tuning_placebo_*.parquet"))]
    )
    real_mean: dict[tuple[str, inject.FitSpec], float] = {}
    for row in placebo.unique(subset=["primary", "a", "r"]).iter_rows(named=True):
        spec: inject.FitSpec = ("asymmetric", row["a"], row["r"])
        real_mean[(row["primary"], spec)] = row["real_mean_capacity"]
    ratios_by_variant = {v: recovery_ratios(out_dir=args.out_dir, variant=v) for v in VARIANTS}
    ratios = ratios_by_variant["semi_synthetic"]
    scores = pl.concat(
        [
            unit_scores(
                out_dir=args.out_dir, variant=v, ratios=ratios_by_variant[v], real_mean=real_mean
            )
            for v in VARIANTS
        ],
        how="vertical_relaxed",
    )
    scores.write_parquet(args.out_dir / "tuning_unit_scores.parquet")
    table = tuning_table(scores=scores, placebo=placebo)
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, float_precision=3):
        print(pl.DataFrame(table))
    chosen = choose(table=table)
    print("chosen:", chosen and (chosen["a"], chosen["r"]))
    if args.no_freeze:
        return
    fallback = min(table, key=lambda row: (row["violation"], row["path_error_mixture"]))
    pick = chosen or fallback
    config = {
        "status": "frozen" if chosen else "refuted-fallback",
        "experiment": common.EXPERIMENT_ID,
        "a": pick["a"],
        "r": pick["r"],
        "note": None
        if chosen
        else (
            "No pair on the grid met the tuning limits. The pair shown has the smallest violation "
            "and is kept only so that later units have a fit to run. The plan's verdict is refuted."
        ),
        "limits": {
            "steps_median": MAX_STEPS_MEDIAN,
            "steps_worst": MAX_STEPS_WORST,
            "fall_26_recovered_min": MIN_FALL_RECOVERED,
        },
        "recovery_ratio_by_primary": {
            primary: ratios[(primary, ("asymmetric", pick["a"], pick["r"]))]
            for primary in common.PRIMARIES
        },
        "tuning_table": table,
        "tuning_units": len(inject.units_for(split="tuning")),
        "tuning_farms": list(inject.TUNING_FARMS),
        "tuning_seeds": list(inject.TUNING_SEEDS),
        "placebo_seeds": [inject.TUNING_PLACEBO_SEEDS.start, inject.TUNING_PLACEBO_SEEDS.stop - 1],
        "supersedes": (
            "The earlier frozen config (a=2, r=10, commit 75005575) was superseded: its baseline "
            "subtracted the full-record free weekly fit, which removed the weekly noise correlated "
            "with the PV shape, and it scored paired differences of two fits sharing one noise "
            "draw. This config subtracts the season-year constant fit and scores the fit of "
            "baseline plus injection against the known path."
        ),
        "tuning_script_commit": commit,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    FROZEN_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {FROZEN_PATH}")


if __name__ == "__main__":
    main()
