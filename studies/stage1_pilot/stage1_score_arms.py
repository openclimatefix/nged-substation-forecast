"""Score every arm on the same rows and save the per-row losses.

For each fitting seed this writes one table of rows, one per arm and ensemble-mean forecast, with
the observed power, the point forecast, and the capacity-normalised absolute error. For an arm
that predicts all 51 ENS members it also holds the fair CRPS of the 51 member point predictions.
Rows are the validation runs' half-hours with an observed power, so every arm is scored on the
cases the stored control was scored on.

Arms (see the README):

- `c0`: the stored fixed-pipeline control (`production_cleaned`), seeds 0, 1, 2.
- `c0_refit`, `c0_plus_s1`: the controls refitted here (`stage1_fit_enstrained.py`).
- `s1_control`: stage 1a on ENS member 0's weather.
- `s1_memberavg`: stage 1a on each of the 51 members' weather, predictions averaged.
- `s1_cams` (solar farms): `s1_memberavg` with stage 1a trained on CAMS irradiance.
- `s1_recipe_on_ens`: stage 1a's columns and fitting loop trained on the ENS control-member
  training rows instead of ERA5 (applied to all 51 members, predictions averaged).
- `era5_ceiling`, `cams_ceiling`: stage 1a on ERA5 (or CAMS) weather at the valid time.

Output: `data/studies/per_study/stage1_pilot/scores/rows_seed<s>.parquet`.

    uv run python studies/stage1_pilot/stage1_score_arms.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs

SCORE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "scores"
STAGE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "stage1a"
ENS_TRAINED_DIR: Final[Path] = inputs.OUTPUT_ROOT / "ens_trained"
MEMBERS: Final[int] = 51
"""The ENS member count; the control member is member 0."""

MEMBER_ARMS: Final[tuple[str, ...]] = (
    "c0",
    "c0_refit",
    "c0_plus_s1",
    "s1_memberavg",
    "s1_cams",
    "s1_recipe_on_ens",
)
"""Arms that predict all 51 members, and so have a CRPS."""

ARMS: Final[tuple[str, ...]] = (
    *MEMBER_ARMS,
    "s1_control",
    "era5_ceiling",
    "cams_ceiling",
)
"""Every arm, in the order tables list them."""


def fair_crps(*, members: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the fair CRPS of each row's members, `mean|x - y| - sum_{i<j}|x_i - x_j| / (m(m-1))`.

    Uses the sorted-member identity of `ml_core.metrics._fair_crps_expr` in float64.

    Args:
        members: Shape (rows, m) member forecasts.
        actual: Shape (rows,) observations.
    """
    m = members.shape[1]
    sorted_members = np.sort(members.astype(np.float64), axis=1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    pairwise = (sorted_members * weights).sum(axis=1) / (m * (m - 1))
    return np.abs(sorted_members - actual[:, None]).mean(axis=1) - pairwise


def _member_matrix(*, frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Reduce per-member rows to one row per case with the member predictions as a list."""
    return (
        frame.sort("ensemble_member")
        .group_by("time_series_id", "power_fcst_init_time", "valid_time", maintain_order=True)
        .agg(members=pl.col(column), count=pl.len())
    )


def _score_rows(
    *,
    cases: pl.DataFrame,
    members_by_case: pl.DataFrame,
    arm: str,
    seed: int,
    sites: pl.DataFrame,
) -> pl.DataFrame:
    joined = cases.join(
        members_by_case, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="left"
    )
    if joined["members"].null_count() or (joined["count"] != MEMBERS).any():
        msg = f"arm {arm}: a case does not have all {MEMBERS} members"
        raise ValueError(msg)
    capacity = joined["effective_capacity_mw"].to_numpy().astype(np.float64)
    matrix = np.array(joined["members"].to_list(), dtype=np.float64)
    actual = joined["actual"].to_numpy().astype(np.float64)
    point = matrix.mean(axis=1)
    return joined.select(
        pl.lit(arm).alias("arm"),
        pl.lit(seed).alias("seed"),
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        pl.Series("actual_norm", actual / capacity),
        pl.Series("point_norm", point / capacity),
        pl.Series("abs_error", 100.0 * np.abs(point - actual) / capacity),
        pl.Series(
            "abs_error_clipped",
            100.0 * np.abs(np.clip(point, 0.0, capacity) - actual) / capacity,
        ),
        pl.Series("crps", 100.0 * fair_crps(members=matrix, actual=actual) / capacity),
    )


def score_seed(*, seed: int, sites: pl.DataFrame) -> pl.DataFrame:
    """Score every arm for one seed over all 53 validation runs."""
    ids = sites["time_series_id"].to_list()
    capacity = sites.select("time_series_id", "effective_capacity_mw")
    ceiling = pl.read_parquet(STAGE_DIR / "ceiling" / f"seed{seed}.parquet")
    pieces: list[pl.DataFrame] = []
    for path in sorted((inputs.FRAMES_DIR / "val").glob("init=*.parquet")):
        frame = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(ids))
        cases = (
            frame.filter((pl.col("ensemble_member") == 0) & pl.col("power").is_finite())
            .select(
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                pl.col("power").alias("actual"),
                (
                    (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                    / 60.0
                ).alias("lead_hours"),
            )
            .join(capacity, on="time_series_id")
        )
        stage1 = pl.read_parquet(STAGE_DIR / "val" / f"seed{seed}" / path.name)
        stage1 = stage1.join(capacity, on="time_series_id").with_columns(
            era5_power=pl.col("s1_era5") * pl.col("effective_capacity_mw"),
            cams_power=pl.col("s1_cams") * pl.col("effective_capacity_mw"),
        )
        stored = pl.read_parquet(
            inputs.CONTROL_FITS_DIR / f"cuda_seed{seed}_none" / "forecasts" / path.name
        ).filter(pl.col("time_series_id").is_in(ids))
        arms: dict[str, pl.DataFrame] = {
            "c0": _member_matrix(frame=stored, column="power_fcst"),
            "c0_refit": _member_matrix(
                frame=pl.read_parquet(ENS_TRAINED_DIR / "c0_refit" / f"seed{seed}" / path.name),
                column="power_fcst",
            ),
            "c0_plus_s1": _member_matrix(
                frame=pl.read_parquet(ENS_TRAINED_DIR / "c0_plus_s1" / f"seed{seed}" / path.name),
                column="power_fcst",
            ),
            "s1_memberavg": _member_matrix(frame=stage1, column="era5_power"),
            "s1_cams": _member_matrix(
                frame=stage1.filter(pl.col("s1_cams").is_not_nan()), column="cams_power"
            ),
            "s1_recipe_on_ens": _member_matrix(
                frame=pl.read_parquet(
                    ENS_TRAINED_DIR / "s1_recipe_on_ens" / f"seed{seed}" / path.name
                ),
                column="power_fcst",
            ),
        }
        for arm, members_by_case in arms.items():
            subset = cases
            if arm == "s1_cams":
                subset = cases.join(
                    sites.filter(pl.col("technology") == "solar").select("time_series_id"),
                    on="time_series_id",
                )
            pieces.append(
                _score_rows(
                    cases=subset, members_by_case=members_by_case, arm=arm, seed=seed, sites=sites
                )
            )
        control = stage1.filter(pl.col("ensemble_member") == 0).select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            pl.col("era5_power").alias("point"),
        )
        pieces.append(_point_rows(cases=cases, point=control, arm="s1_control", seed=seed))
        for arm, column in (("era5_ceiling", "s1_era5"), ("cams_ceiling", "s1_cams")):
            scoped = (
                cases
                if arm == "era5_ceiling"
                else cases.join(
                    sites.filter(pl.col("technology") == "solar").select("time_series_id"),
                    on="time_series_id",
                )
            )
            analysis = ceiling.join(capacity, on="time_series_id").select(
                "time_series_id",
                "valid_time",
                point=pl.col(column) * pl.col("effective_capacity_mw"),
            )
            joined = scoped.join(analysis, on=["time_series_id", "valid_time"], how="left")
            if joined["point"].null_count() or joined["point"].is_nan().any():
                msg = f"arm {arm}: some cases have no ceiling value"
                raise ValueError(msg)
            pieces.append(
                _point_rows(
                    cases=scoped,
                    point=joined.select(
                        "time_series_id", "power_fcst_init_time", "valid_time", "point"
                    ),
                    arm=arm,
                    seed=seed,
                )
            )
    floats = ("lead_hours", "actual_norm", "point_norm", "abs_error", "abs_error_clipped", "crps")
    return pl.concat([piece.cast(dict.fromkeys(floats, pl.Float64)) for piece in pieces])


def _point_rows(*, cases: pl.DataFrame, point: pl.DataFrame, arm: str, seed: int) -> pl.DataFrame:
    """Score an arm that has a single point forecast per case (no members, so no CRPS)."""
    joined = cases.join(
        point, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner"
    )
    if joined.height != cases.height:
        msg = f"arm {arm}: {cases.height - joined.height} cases have no point forecast"
        raise ValueError(msg)
    return joined.select(
        pl.lit(arm).alias("arm"),
        pl.lit(seed).alias("seed"),
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        (pl.col("actual") / pl.col("effective_capacity_mw")).alias("actual_norm"),
        (pl.col("point") / pl.col("effective_capacity_mw")).alias("point_norm"),
        (
            100.0 * (pl.col("point") - pl.col("actual")).abs() / pl.col("effective_capacity_mw")
        ).alias("abs_error"),
        (
            100.0
            * (pl.col("point").clip(0.0, pl.col("effective_capacity_mw")) - pl.col("actual")).abs()
            / pl.col("effective_capacity_mw")
        ).alias("abs_error_clipped"),
        pl.lit(None, dtype=pl.Float64).alias("crps"),
    )


def main() -> None:
    """Score each seed, skipping seeds already written."""
    sites = inputs.site_table()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in inputs.SEEDS:
        target = SCORE_DIR / f"rows_seed{seed}.parquet"
        if target.exists():
            continue
        rows = score_seed(seed=seed, sites=sites)
        rows.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"seed {seed}: {rows.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
