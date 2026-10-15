"""Fit experiment 00023's calibrators and build each arm's pooled mixture.

For each seed and technology this fits four multi-quantile XGBoost calibrators (`cal0`, `cal_lags`,
`cal_sums`, `cal_both`) on the ENS control member's out-of-fold stage 1a quantiles from 2024-07 to
2025-06 (experiment 00002's calibration rows), with the observed power over effective capacity as
the target. Inputs: the member's 9 stage 1a quantiles, the lead in hours, and the arm's recent-power
columns (`e00023_common.recent_power_columns`). Each calibrator is applied to every member of the 53
validation runs, and the 51 members' calibrated quantiles are pooled into a mixture. The `mixture`
arm pools the raw stage 1a quantiles.

Output: `work/mixture/<arm>_<technology>_seed<s>.parquet` with the key columns and the 9 mixture
quantiles of every case, skipped for a seed whose files all exist.

    uv run python ml-experiments/00023-*/e00023_fit.py [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from e00023_common import (
    ARMS,
    CALIBRATOR_ARMS,
    CALIBRATOR_PARAMETERS,
    CALIBRATOR_ROUNDS,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    LEVELS,
    MEMBERS,
    SEEDS,
    STAGE1A_DIR,
    TECHNOLOGIES,
    TRAINING_VALID_END,
    WORK_DIR,
    TechnologyType,
    day_summaries,
    grid_index,
    mixture_quantiles,
    power_grids,
    read_levels,
    recent_power_columns,
    site_table,
    take,
)

MIXTURE_DIR: Final[Path] = WORK_DIR / "mixture"
USES_LAGS: Final[dict[str, bool]] = {
    "cal0": False,
    "cal_lags": True,
    "cal_sums": False,
    "cal_both": True,
}
USES_SUMMARIES: Final[dict[str, bool]] = {
    "cal0": False,
    "cal_lags": False,
    "cal_sums": True,
    "cal_both": True,
}
LAG_SLICE: Final[slice] = slice(0, 3)
SUMMARY_SLICE: Final[slice] = slice(3, 9)


def design_matrix(
    *,
    frame: pl.DataFrame,
    recent: np.ndarray,
    arm: str,
) -> np.ndarray:
    """Return an arm's calibrator inputs: 9 quantiles, the lead, then its recent-power columns."""
    lead = (
        frame["valid_time"] - frame["power_fcst_init_time"]
    ).dt.total_minutes().to_numpy() / 60.0
    columns = [frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float32), lead[:, None]]
    if USES_LAGS[arm]:
        columns.append(recent[:, LAG_SLICE])
    if USES_SUMMARIES[arm]:
        columns.append(recent[:, SUMMARY_SLICE])
    return np.concatenate(columns, axis=1).astype(np.float32)


def recent_for_frame(
    *, frame: pl.DataFrame, grids: dict[int, np.ndarray], summaries: dict[int, np.ndarray]
) -> np.ndarray:
    """Return the recent-power columns of every row of a frame, series by series."""
    out = np.full((frame.height, 9), np.nan, dtype=np.float32)
    series = frame["time_series_id"].to_numpy()
    target = grid_index(times=frame["valid_time"])
    origin = grid_index(times=frame["power_fcst_init_time"])
    for series_id in np.unique(series):
        mask = series == series_id
        out[mask] = recent_power_columns(
            power=grids[int(series_id)],
            summaries=summaries[int(series_id)],
            target=target[mask],
            origin=origin[mask],
        )
    return out


def training_rows(
    *, technology_ids: list[int], seed: int, grids: dict[int, np.ndarray]
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the control member's calibration rows of a technology and their targets.

    Rows with a valid time from `TRAINING_VALID_END` on, or without an observed target, are
    dropped.
    """
    rows = pl.read_parquet(STAGE1A_DIR / "train_oof" / f"seed{seed}.parquet").filter(
        pl.col("time_series_id").is_in(technology_ids), pl.col("valid_time") < TRAINING_VALID_END
    )
    if rows["ensemble_member"].n_unique() != 1:
        msg = "the calibration rows should hold the control member only"
        raise ValueError(msg)
    target = np.full(rows.height, np.nan, dtype=np.float32)
    series = rows["time_series_id"].to_numpy()
    index = grid_index(times=rows["valid_time"])
    for series_id in technology_ids:
        mask = series == series_id
        target[mask] = take(values=grids[series_id], index=index[mask])
    keep = np.isfinite(target)
    return rows.filter(pl.Series(keep)), target[keep]


def pool(*, frame: pl.DataFrame, quantiles: np.ndarray) -> pl.DataFrame:
    """Pool the members of each case into a mixture of 9 quantiles.

    Args:
        frame: Member rows with the key columns and `ensemble_member`.
        quantiles: Shape (rows, 9), each row's quantiles, sorted along the levels.

    Returns:
        One row per case with the key columns and the 9 mixture quantiles.

    Raises:
        ValueError: If a case does not have exactly `MEMBERS` members.
    """
    order = np.lexsort(
        (
            frame["ensemble_member"].to_numpy(),
            frame["valid_time"].to_numpy(),
            frame["power_fcst_init_time"].to_numpy(),
            frame["time_series_id"].to_numpy(),
        )
    )
    ordered = frame[order.tolist()]
    if ordered.height % MEMBERS:
        msg = "member rows do not divide into whole cases"
        raise ValueError(msg)
    cases = ordered.height // MEMBERS
    keys = ordered.select(KEY_COLUMNS)
    first = keys[::MEMBERS]
    if not keys[MEMBERS - 1 :: MEMBERS].equals(first):
        msg = "a case does not hold exactly one run of members"
        raise ValueError(msg)
    fine = mixture_quantiles(member_quantiles=quantiles[order].reshape(cases, MEMBERS, 9))
    return first.hstack(
        pl.DataFrame(read_levels(fine=fine).astype(np.float32), schema=list(LEVEL_COLUMNS))
    )


def main() -> None:
    """Fit and apply every calibrator for every seed, skipping seeds whose files exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one seed, three runs, no writes")
    smoke = parser.parse_args().smoke
    sites = site_table()
    grids = power_grids(sites=sites)
    summaries = {series_id: day_summaries(power=grid) for series_id, grid in grids.items()}
    ids: dict[TechnologyType, list[int]] = {
        technology: sites.filter(pl.col("technology") == technology)["time_series_id"].to_list()
        for technology in TECHNOLOGIES
    }
    MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS[:1] if smoke else SEEDS:
        paths = {
            (arm, technology): MIXTURE_DIR / f"{arm}_{technology}_seed{seed}.parquet"
            for arm in ARMS
            if arm in ("mixture", *CALIBRATOR_ARMS)
            for technology in TECHNOLOGIES
        }
        if not smoke and all(path.exists() for path in paths.values()):
            continue
        started = time.monotonic()
        models: dict[tuple[str, TechnologyType], xgb.Booster] = {}
        for technology in TECHNOLOGIES:
            rows, target = training_rows(technology_ids=ids[technology], seed=seed, grids=grids)
            recent = recent_for_frame(frame=rows, grids=grids, summaries=summaries)
            for arm in CALIBRATOR_ARMS:
                models[arm, technology] = xgb.train(
                    {**CALIBRATOR_PARAMETERS, "quantile_alpha": LEVELS, "seed": seed},
                    xgb.DMatrix(design_matrix(frame=rows, recent=recent, arm=arm), label=target),
                    num_boost_round=CALIBRATOR_ROUNDS,
                )
            print(
                f"seed {seed} {technology}: {rows.height} calibration rows, "
                f"{time.monotonic() - started:.0f} s",
                flush=True,
            )
        files = sorted((STAGE1A_DIR / "val" / f"seed{seed}").glob("init=*.parquet"))
        pieces: dict[tuple[str, TechnologyType], list[pl.DataFrame]] = {key: [] for key in paths}
        for file in files[:3] if smoke else files:
            member_rows = pl.read_parquet(file)
            for technology in TECHNOLOGIES:
                frame = member_rows.filter(pl.col("time_series_id").is_in(ids[technology]))
                raw = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
                pieces["mixture", technology].append(pool(frame=frame, quantiles=raw))
                recent = recent_for_frame(frame=frame, grids=grids, summaries=summaries)
                for arm in CALIBRATOR_ARMS:
                    predicted = models[arm, technology].predict(
                        xgb.DMatrix(design_matrix(frame=frame, recent=recent, arm=arm))
                    )
                    calibrated = np.sort(np.clip(predicted, 0.0, 1.0), axis=1)
                    pieces[arm, technology].append(pool(frame=frame, quantiles=calibrated))
            print(f"seed {seed}: {file.name} done", flush=True)
        if smoke:
            for (arm, technology), parts in pieces.items():
                print(arm, technology, pl.concat(parts).select(LEVEL_COLUMNS).mean().row(0))
            return
        for key, path in paths.items():
            pl.concat(pieces[key]).write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
        print(f"seed {seed}: finished in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
