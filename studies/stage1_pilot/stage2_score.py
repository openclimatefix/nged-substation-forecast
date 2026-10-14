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
from scipy.stats import beta as beta_distribution
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

    Returns:
        Array of shape (cases, 199).
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """Return the blended 9 quantiles of every case, each case's two parameters, and the table.

    Args:
        fine: Shape (cases, 199) mixture quantile functions.
        actual: Observed fraction of capacity per case.
        cases: The cases, with `technology`, `band`, `month`, and `valid_time`.

    Returns:
        The blended quantiles, the two parameter arrays, and the parameter table.
    """
    generator = np.random.default_rng(BLEND_SEED)
    out = np.empty((fine.shape[0], LEVELS.size), dtype=np.float32)
    alphas = np.ones(fine.shape[0])
    betas = np.ones(fine.shape[0])
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
                alphas[group] = alpha
                betas[group] = beta
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
    return out, alphas, betas, pl.DataFrame(chosen)


def exceedance_probability(
    *, fine: np.ndarray, threshold: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Return P(power > threshold) per case from the mixture quantile function and the blend.

    The mixture's cumulative probability at the threshold is the share of the 199 grid levels whose
    quantile is at or below it; the beta transform then maps it to the blended cumulative
    probability. Resolution is 1/199, and probabilities beyond 0.995 or below 0.005 are not
    resolved (the tail rule).
    """
    cumulative = (fine <= threshold[:, None]).mean(axis=1)
    return 1.0 - beta_distribution.cdf(cumulative, alpha, beta)


def pinball(*, quantile: np.ndarray, level: float, actual: np.ndarray) -> np.ndarray:
    """Return the pinball loss of one quantile forecast per case."""
    difference = actual - quantile
    return np.where(difference >= 0, level * difference, (level - 1) * difference)


def exceedance_columns(
    *,
    fine: np.ndarray,
    thresholds: dict[str, np.ndarray],
    actual: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the exceedance probability and outcome (0 or 1) for each threshold."""
    columns = {}
    for name, threshold in thresholds.items():
        columns[f"p_{name}"] = exceedance_probability(
            fine=fine, threshold=threshold, alpha=alpha, beta=beta
        )
        columns[f"o_{name}"] = (actual > threshold).astype(np.float64)
    return columns


def exceedance_thresholds(*, cases: pl.DataFrame, sites: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each case's series' 90th and 95th percentile of cleaned power in the training period.

    The percentiles are of power over effective capacity, from 2019-09 to before 2025-06-17.
    """
    table = (
        observed_training_power(sites=sites)
        .group_by("time_series_id")
        .agg(
            exceed_90=pl.col("target").quantile(0.9),
            exceed_95=pl.col("target").quantile(0.95),
        )
    )
    joined = cases.join(table, on="time_series_id", how="left", validate="m:1")
    return {
        "exceed_90": joined["exceed_90"].to_numpy(),
        "exceed_95": joined["exceed_95"].to_numpy(),
    }


def score_rows(
    *,
    arm: str,
    seed: int,
    cases: pl.DataFrame,
    quantiles: np.ndarray,
    actual: np.ndarray,
    exceedance: dict[str, np.ndarray],
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
        pinball_90=pl.Series(
            100.0 * pinball(quantile=quantiles[:, 7], level=0.9, actual=actual), dtype=pl.Float64
        ),
        pinball_95=pl.Series(
            100.0 * pinball(quantile=quantiles[:, 8], level=0.95, actual=actual), dtype=pl.Float64
        ),
        **{name: pl.Series(values, dtype=pl.Float64) for name, values in exceedance.items()},
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
        thresholds = exceedance_thresholds(cases=cases, sites=sites)
        ones = np.ones(cases.height)
        pieces = []
        for arm, directory in ARM_DIRS.items():
            fine = member_fine_quantiles(directory=directory, seed=seed, cases=cases)
            blended, alphas, betas, params = tuned_blend(fine=fine, actual=actual, cases=cases)
            for name, quantiles, alpha, beta in (
                (arm, read_levels(fine=fine, levels=LEVELS), ones, ones),
                (f"{arm}_blp", blended, alphas, betas),
            ):
                pieces.append(
                    score_rows(
                        arm=name,
                        seed=seed,
                        cases=cases,
                        quantiles=quantiles,
                        actual=actual,
                        exceedance=exceedance_columns(
                            fine=fine, thresholds=thresholds, actual=actual, alpha=alpha, beta=beta
                        ),
                    )
                )
            all_params.append(params.with_columns(arm=pl.lit(arm), seed=pl.lit(seed)))
            print(f"seed {seed}: {arm} scored", flush=True)
        climate = np.sort(climatology_quantiles(cases=cases, sites=sites), axis=1)
        pieces.append(
            score_rows(
                arm="climatology_q",
                seed=seed,
                cases=cases,
                quantiles=climate,
                actual=actual,
                exceedance=exceedance_columns(
                    fine=mixture_quantiles(member_quantiles=climate[:, None, :].astype(np.float32)),
                    thresholds=thresholds,
                    actual=actual,
                    alpha=ones,
                    beta=ones,
                ),
            )
        )
        pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
    if all_params:
        pl.concat(all_params).write_parquet(SCORE_DIR / "blend_params.parquet")


if __name__ == "__main__":
    main()
