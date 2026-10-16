"""Placebo sky on the real series: false steps of the four variants (experiment 00045).

For each Primary the PV shape `u` is replaced, day by day, by that of another day of the same
calendar month (`inject.permute_u`), 200 permutations per Primary (seeds 2000 to 2199, none used
by tuning or scoring). Power stays in place, so any link between cloud and power is broken and
every rise of the fitted path is false by construction, except where the real series holds real
growth. A rise starting in a week within 4 weeks of a rise of 10% or more of the variant's mean
capacity in the unpermuted real path is therefore not counted (`metrics.rise_weeks`). A false step
is a rise of at least 10% (also 5%, reported) of the mean unpermuted capacity over any 4
consecutive weeks. Rises are relative to the path's own mean, so the recovery-ratio division, a
uniform scale, does not change a count.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_placebo.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00045_common as common
import e00045_inject as inject
import e00045_metrics as metrics
import e00045_model as model
import e00045_real as real
import e00045_score as score
import numpy as np
import polars as pl

PERMUTATION_SEEDS: Final[range] = range(2000, 2200)
"""200 permutations per Primary."""
PLACEBO_DIR: Final[Path] = real.REAL_DIR


def placebo_job(*, site: str, seeds: range) -> Path:
    """Fit every variant on the permuted-shape series of one Primary and count false steps.

    Args:
        site: Frame site code.
        seeds: Permutation seeds.

    Returns:
        The path of the output file (one row per permutation and variant).
    """
    out_file = PLACEBO_DIR / f"placebo_{site}.parquet"
    if out_file.exists():
        return out_file
    PLACEBO_DIR.mkdir(parents=True, exist_ok=True)
    inputs = inject.load_inputs()
    table = model.usable_rows_table(half_hourly=inputs.half_hourly, weeks=inputs.weeks, site=site)
    real_design = model.design_from_rows_table(table=table)
    stats = {}
    for variant, spec in real.VARIANTS.items():
        path = model.FITS[spec[0]](design=real_design, a=spec[1], r=spec[2])
        mean = float(np.nanmean(path))
        exempt = metrics.rise_weeks(path=path, threshold=0.10 * mean)
        stats[variant] = (mean, exempt)
    rows = []
    for seed in seeds:
        permuted = inject.permute_u(table=table, rng=np.random.default_rng([seed, 45, 2]))
        design = model.design_from_rows_table(table=permuted)
        for variant, spec in real.VARIANTS.items():
            path = model.FITS[spec[0]](design=design, a=spec[1], r=spec[2])
            mean, exempt = stats[variant]
            rows.append(
                {
                    "primary": site,
                    "perm_seed": seed,
                    "variant": variant,
                    "real_mean_capacity": mean,
                    "exempt_weeks": int(exempt.sum()),
                    "n_weeks": int(np.isfinite(path).sum()),
                    "steps_10": metrics.count_steps(
                        path=path, threshold=0.10 * mean, exempt=exempt
                    ),
                    "steps_05": metrics.count_steps(
                        path=path, threshold=0.05 * mean, exempt=exempt
                    ),
                    "drift_per_5y": metrics.ratchet_drift_per_5_years(path=path, reference=mean),
                }
            )
    score.atomic_write(frame=pl.DataFrame(rows), path=out_file)
    return out_file


def _job(args: tuple[str, range]) -> str:
    placebo_job(site=args[0], seeds=args[1])
    return f"placebo {args[0]}"


def summarise(*, tables: pl.DataFrame) -> pl.DataFrame:
    """Return false steps per Primary-year and the share of permutations with any false step.

    Args:
        tables: Every Primary's placebo rows.

    Returns:
        One row per Primary and variant: mean and median false steps (10% and 5%) per year of data,
        the share of permutations with at least one 10% false step, the mean drift per 5 years as a
        share of capacity, and the number of weeks exempt for real growth.
    """
    per_year = metrics.WEEKS_PER_YEAR
    return (
        tables.with_columns(
            steps_10_py=pl.col("steps_10") / (pl.col("n_weeks") / per_year),
            steps_05_py=pl.col("steps_05") / (pl.col("n_weeks") / per_year),
        )
        .group_by("primary", "variant")
        .agg(
            steps_10_per_year_mean=pl.col("steps_10_py").mean(),
            steps_10_per_year_median=pl.col("steps_10_py").median(),
            steps_05_per_year_mean=pl.col("steps_05_py").mean(),
            share_with_a_10pct_step=(pl.col("steps_10") > 0).mean(),
            drift_per_5y_mean=pl.col("drift_per_5y").mean(),
            exempt_weeks=pl.col("exempt_weeks").first(),
            n_weeks=pl.col("n_weeks").median(),
            n_permutations=pl.len(),
        )
        .sort("primary", "variant")
    )


def main() -> None:
    """Run one placebo job per Primary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, [(s, PERMUTATION_SEEDS) for s in common.PRIMARIES]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
