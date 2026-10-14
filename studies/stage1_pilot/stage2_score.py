"""Pool each arm's members into a mixture, apply the learned blend, and score every case.

For each seed and each arm in `ARM_DIRS`, the 51 members' 9 quantiles are pooled into a mixture
(`stage2_common.mixture_quantiles`), giving one quantile function per scored case. The learned blend
is the beta transform of the mixture's cumulative distribution
(`stage2_common.blended_levels`): for each technology, lead band, and validation month, the two
shape parameters are chosen on `stage2_common.BLEND_GRID` to minimise the mean CRPS estimate on the
earlier validation months whose valid times end 14 days before the month starts. A month with fewer
than `MIN_FIT_CASES` earlier cases uses both parameters equal to 1 (the plain mixture). Every arm is
scored with and without the blend (`<arm>_blp`). `climatology_q` is a reference with no members.

Scored cases are study 2a's (the member-0 rows with an observed power), so every arm is scored on
the same rows. Output: `data/studies/per_study/stage2_pilot/scores/rows_seed<s>.parquet` with one
row per arm and case: the median, the absolute error of the median and the CRPS estimate (points of
capacity), the 10 to 90 and 5 to 95 interval coverage (100 for covered, else 0), and
`blend_params.parquet` with the chosen parameters.

    uv run python studies/stage1_pilot/stage2_score.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
from stage1_fit_stage1a import observed_training_power
from stage2_common import (
    BLEND_GRID,
    ENS_FEATURES_DIR,
    LEVEL_COLUMNS,
    LEVELS,
    MEDIAN_INDEX,
    MEMBERS,
    STAGE2_ROOT,
    band_name,
    blended_levels,
    crps_estimate,
    mixture_quantiles,
    read_levels,
    scoring_cases,
)

SCORE_DIR: Final[Path] = STAGE2_ROOT / "scores"
ARM_DIRS: Final[dict[str, Path]] = {
    "a_ens_quantile": STAGE2_ROOT / "arm_a",
    "b_mixture_full": STAGE2_ROOT / "stage1a_q" / "full" / "val",
    "b_mixture_short": STAGE2_ROOT / "stage1a_q" / "short" / "val",
    "c_stage2_full": STAGE2_ROOT / "calibrated" / "full" / "val",
    "c_stage2_short": STAGE2_ROOT / "calibrated" / "short" / "val",
}
CASE_KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
MIN_FIT_CASES: Final[int] = 5_000
MAX_FIT_CASES: Final[int] = 30_000
BLEND_SEED: Final[int] = 20261013


def member_fine_quantiles(*, directory: Path, seed: int, cases: pl.DataFrame) -> np.ndarray:
    """Return the mixture quantile function (199 levels) of every case, in `cases` row order.

    Args:
        directory: Holds `seed<s>/init=<date>.parquet` files of per-member quantile columns.
        seed: The fitting seed.
        cases: The scored cases sorted by initialisation time, series, then valid time.
    """
    keys = cases.select(CASE_KEYS)
    pieces = []
    for path in sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet")):
        members = (
            pl.read_parquet(directory / f"seed{seed}" / path.name)
            .join(keys, on=CASE_KEYS, how="inner")
            .sort("power_fcst_init_time", "time_series_id", "valid_time", "ensemble_member")
        )
        if members.height % MEMBERS:
            msg = f"{directory}: {path.name} has a case without all {MEMBERS} members"
            raise ValueError(msg)
        matrix = members.select(LEVEL_COLUMNS).to_numpy().reshape(-1, MEMBERS, len(LEVEL_COLUMNS))
        pieces.append(mixture_quantiles(member_quantiles=matrix.astype(np.float32)))
    fine = np.concatenate(pieces)
    if fine.shape[0] != cases.height:
        msg = f"{directory}: {fine.shape[0]} cases scored, expected {cases.height}"
        raise ValueError(msg)
    return fine


def tuned_blend(
    *, fine: np.ndarray, actual: np.ndarray, cases: pl.DataFrame
) -> tuple[np.ndarray, pl.DataFrame]:
    """Return the blended 9 quantiles of every case and the chosen parameters.

    Args:
        fine: Shape (cases, 199) mixture quantile functions.
        actual: Observed fraction of capacity per case.
        cases: The cases, with `technology`, `band`, `month`, and `valid_time`.
    """
    generator = np.random.default_rng(BLEND_SEED)
    out = np.empty((fine.shape[0], LEVELS.size), dtype=np.float32)
    chosen = []
    valid = cases["valid_time"].to_numpy()
    technology = cases["technology"].to_numpy()
    band = cases["band"].to_numpy()
    month = cases["month"].to_numpy()
    for month_label in sorted(set(month)):
        first = np.datetime64(f"{month_label}-01T00:00:00")
        cutoff = first - np.timedelta64(inputs.PURGE_DAYS, "D")
        for tech in ("solar", "wind"):
            for name, _, _ in inputs.LEAD_BANDS:
                group = (month == month_label) & (technology == tech) & (band == name)
                if not group.any():
                    continue
                fit_index = np.flatnonzero((valid < cutoff) & (technology == tech) & (band == name))
                alpha = beta = 1.0
                if fit_index.size >= MIN_FIT_CASES:
                    if fit_index.size > MAX_FIT_CASES:
                        fit_index = generator.choice(fit_index, MAX_FIT_CASES, replace=False)
                    best = np.inf
                    for a in BLEND_GRID:
                        for b in BLEND_GRID:
                            score = crps_estimate(
                                quantiles=blended_levels(fine=fine[fit_index], alpha=a, beta=b),
                                actual=actual[fit_index],
                            ).mean()
                            if score < best:
                                best, alpha, beta = score, a, b
                out[group] = blended_levels(fine=fine[group], alpha=alpha, beta=beta)
                chosen.append(
                    {
                        "month": month_label,
                        "technology": tech,
                        "band": name,
                        "alpha": alpha,
                        "beta": beta,
                        "fit_cases": int(fit_index.size),
                    }
                )
    return out, pl.DataFrame(chosen)


def score_rows(
    *, arm: str, seed: int, cases: pl.DataFrame, quantiles: np.ndarray, actual: np.ndarray
) -> pl.DataFrame:
    """Return one scored row per case (errors in points of capacity, coverage 0 or 100)."""
    median = quantiles[:, MEDIAN_INDEX]
    return cases.select(
        pl.lit(arm).alias("arm"),
        pl.lit(seed).alias("seed"),
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "actual_norm",
    ).with_columns(
        point_norm=pl.Series(median, dtype=pl.Float64),
        abs_error=pl.Series(100.0 * np.abs(median - actual), dtype=pl.Float64),
        crps=pl.Series(100.0 * crps_estimate(quantiles=quantiles, actual=actual), dtype=pl.Float64),
        cover_80=pl.Series(
            100.0 * ((actual >= quantiles[:, 1]) & (actual <= quantiles[:, 7])), dtype=pl.Float64
        ),
        cover_90=pl.Series(
            100.0 * ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 8])), dtype=pl.Float64
        ),
    )


def climatology_quantiles(*, cases: pl.DataFrame, sites: pl.DataFrame) -> np.ndarray:
    """Return each case's 9 climatological quantiles by series, local month, and half-hour slot."""

    def key(column: str) -> list[pl.Expr]:
        local = pl.col(column).dt.convert_time_zone("Europe/London")
        return [
            local.dt.month().alias("month_of_year"),
            (local.dt.hour() * 2 + local.dt.minute() // 30).alias("slot"),
        ]

    observed = observed_training_power(sites=sites).with_columns(*key("valid_time"))
    table = observed.group_by("time_series_id", "month_of_year", "slot").agg(
        *[
            pl.col("target").quantile(float(level)).alias(name)
            for level, name in zip(LEVELS, LEVEL_COLUMNS, strict=True)
        ]
    )
    joined = cases.with_columns(*key("valid_time")).join(
        table, on=["time_series_id", "month_of_year", "slot"], how="left", validate="m:1"
    )
    return joined.select(LEVEL_COLUMNS).to_numpy()


def main() -> None:
    """Score every arm for every seed."""
    sites = inputs.site_table()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    all_params = []
    for seed in inputs.SEEDS:
        target = SCORE_DIR / f"rows_seed{seed}.parquet"
        if target.exists():
            continue
        cases = scoring_cases(seed=seed).join(
            sites.select("time_series_id", "technology"), on="time_series_id"
        )
        cases = cases.sort("power_fcst_init_time", "time_series_id", "valid_time").with_columns(
            band=pl.Series(band_name(lead_hours=cases["lead_hours"].to_numpy())),
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
        )
        actual = cases["actual_norm"].to_numpy().astype(np.float64)
        pieces = []
        for arm, directory in ARM_DIRS.items():
            fine = member_fine_quantiles(directory=directory, seed=seed, cases=cases)
            raw = read_levels(fine=fine, levels=LEVELS)
            blended, params = tuned_blend(fine=fine, actual=actual, cases=cases)
            pieces.append(score_rows(arm=arm, seed=seed, cases=cases, quantiles=raw, actual=actual))
            pieces.append(
                score_rows(
                    arm=f"{arm}_blp", seed=seed, cases=cases, quantiles=blended, actual=actual
                )
            )
            all_params.append(params.with_columns(arm=pl.lit(arm), seed=pl.lit(seed)))
            print(f"seed {seed}: {arm} scored", flush=True)
        pieces.append(
            score_rows(
                arm="climatology_q",
                seed=seed,
                cases=cases,
                quantiles=np.sort(climatology_quantiles(cases=cases, sites=sites), axis=1),
                actual=actual,
            )
        )
        pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
    if all_params:
        pl.concat(all_params).write_parquet(SCORE_DIR / "blend_params.parquet")


if __name__ == "__main__":
    main()
