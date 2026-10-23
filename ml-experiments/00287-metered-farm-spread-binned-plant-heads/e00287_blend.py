# ruff: noqa: E501
"""Blend GAM, the network head (seed mean), and XGBoost per quantile level, rolling-origin.

Copied from experiment 260's method: per-level simplex weights minimising the farm-equal pinball
loss on rows whose valid time precedes the month's first run; the first three scored months use
equal weights. XGBoost is experiment 275's `x2` (past weather plus the era flag), read from its
stored predictions.

    uv run --no-sync python ml-experiments/00287-*/e00287_blend.py
"""

import os
from typing import Final

import e00287_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

POSTHOC: Final[bool] = os.environ.get("BLEND_POSTHOC") == "1"
"""Set `BLEND_POSTHOC=1` for the post hoc blend that swaps the network head for the mixture head."""
COMPONENTS: Final[tuple[str, ...]] = (
    ("gam", "mdn", "xgb_x2") if POSTHOC else ("gam", "nn", "xgb_x2")
)
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
GRID_STEPS: Final[tuple[float, ...]] = (0.05, 0.01, 0.002)
REFINE_STEPS: Final[int] = 5
BURN_IN_MONTHS: Final[int] = 3


def load_component(*, setting: str, arm: str) -> pl.DataFrame:
    """Return an arm's seed-averaged quantiles per (farm, time)."""
    if arm == "xgb_x2" and setting == "own_history":
        files = sorted((common.E275_DIR / "predictions" / setting).glob("x2_s*_*.parquet"))
    elif arm == "xgb_x2":
        # Experiment 275 stored no zero-history XGBoost fits; experiment 270's l6 is the nearest
        # arm (XGBoost on ensemble rows, all inputs).
        files = sorted((common.E270_DIR / "predictions" / setting).glob("l6_s*_*.parquet"))
    else:
        files = sorted((common.PRED_DIR / setting).glob(f"{arm}_s*_*.parquet"))
    frame = pl.concat(
        [
            pl.read_parquet(f).select(
                "time_series_id", "time", "init_time", "lead_hours", "actual", *QCOLS
            )
            for f in files
        ]
    )
    return frame.group_by("time_series_id", "time").agg(
        pl.col("init_time").first(),
        pl.col("lead_hours").first(),
        pl.col("actual").first(),
        pl.col(QCOLS).mean(),
    )


def common_rows(*, setting: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the rows every component holds, and the components' quantiles (arm, row, level)."""
    frames = {a: load_component(setting=setting, arm=a) for a in COMPONENTS}
    keys = frames[COMPONENTS[0]].select("time_series_id", "time")
    for f in frames.values():
        keys = keys.join(f.select("time_series_id", "time"), on=["time_series_id", "time"])
    keys = keys.sort("time_series_id", "time")
    base = (
        keys.join(frames[COMPONENTS[0]], on=["time_series_id", "time"])
        .sort("time_series_id", "time")
        .select("time_series_id", "time", "init_time", "lead_hours", "actual")
        .with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
    )
    stack = np.stack(
        [
            keys.join(frames[a], on=["time_series_id", "time"])
            .sort("time_series_id", "time")
            .select(QCOLS)
            .to_numpy()
            for a in COMPONENTS
        ]
    )
    return base, stack


def simplex_points(*, n_components: int, step: float) -> np.ndarray:
    """Return every point of the simplex whose weights are multiples of `step`."""
    ticks = np.arange(0.0, 1.0 + 1e-9, step)
    grid = np.stack(np.meshgrid(*([ticks] * (n_components - 1)), indexing="ij"), axis=-1)
    grid = grid.reshape(-1, n_components - 1)
    grid = grid[grid.sum(axis=1) <= 1.0 + 1e-9]
    return np.column_stack([grid, 1.0 - grid.sum(axis=1)])


def pinball_weights(
    *, y: np.ndarray, q: np.ndarray, level: float, row_weight: np.ndarray
) -> np.ndarray:
    """Return simplex weights minimising the weighted pinball loss of the weighted sum.

    The loss is convex in the weights. A coarse grid of the simplex, then finer grids around the best
    point, finds the minimum to the finest step. A linear programme took tens of seconds per fit.

    Args:
        y: Actual output per row.
        q: Component quantiles, shape (rows, components).
        level: The quantile level.
        row_weight: Each row's weight in the loss.

    Returns:
        One weight per component, at least 0, summing to 1.
    """
    n_components = q.shape[1]
    best = np.full(n_components, 1.0 / n_components)
    for stage, step in enumerate(GRID_STEPS):
        if stage == 0:
            candidates = simplex_points(n_components=n_components, step=step)
        else:
            reach = np.arange(-REFINE_STEPS, REFINE_STEPS + 1) * step
            offsets = np.stack(np.meshgrid(*([reach] * n_components), indexing="ij"), axis=-1)
            candidates = np.clip(best + offsets.reshape(-1, n_components), 0.0, None)
            candidates = candidates / candidates.sum(axis=1, keepdims=True)
        error = y[:, None] - q @ candidates.T
        loss = (row_weight[:, None] * np.maximum(level * error, (level - 1.0) * error)).sum(axis=0)
        best = candidates[int(np.argmin(loss))]
    return best


def farm_row_weight(*, farms: np.ndarray) -> np.ndarray:
    """Return weights that give each farm equal total weight."""
    _, inverse, counts = np.unique(farms, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def fit_weights(*, base: pl.DataFrame, stack: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return weights of shape (levels, components) fitted on the given row mask."""
    y = base["actual"].to_numpy()[rows]
    farms = base["time_series_id"].to_numpy()[rows]
    row_weight = farm_row_weight(farms=farms)
    row_weight = row_weight / row_weight.sum()
    out = np.empty((len(common.LEVELS), stack.shape[0]))
    for k, level in enumerate(common.LEVELS):
        out[k] = pinball_weights(
            y=y, q=stack[:, rows, k].T, level=float(level), row_weight=row_weight
        )
    return out


def combine(*, stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted quantiles (rows, levels), sorted across levels."""
    combined = (
        np.einsum("ak,ark->rk", weights.T, stack)
        if weights.ndim == 2
        else np.einsum("a,ark->rk", weights, stack)
    )
    return np.sort(combined, axis=1)


def rolling(*, base: pl.DataFrame, stack: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return the rolling-origin blend's quantiles and the weight records, fitted month by month."""
    months = sorted(base["month"].unique().to_list())
    month_col = base["month"].to_numpy()
    time = base["time"].dt.epoch("us").to_numpy()
    init = base["init_time"].dt.epoch("us").to_numpy()
    blend = np.empty((base.height, len(common.LEVELS)))
    records: list[dict[str, object]] = []
    equal = np.full((len(common.LEVELS), stack.shape[0]), 1.0 / stack.shape[0])
    for i, month in enumerate(months):
        in_month = month_col == month
        earlier = time < init[in_month].min()
        weights = equal if i < BURN_IN_MONTHS else fit_weights(base=base, stack=stack, rows=earlier)
        blend[in_month] = combine(stack=stack[:, in_month, :], weights=weights)
        for k, level in enumerate(common.LEVELS):
            records.append(
                {
                    "month": month,
                    "level": float(level),
                    **{a: float(weights[k, j]) for j, a in enumerate(COMPONENTS)},
                }
            )
    return blend, records


def write(*, base: pl.DataFrame, setting: str, arm: str, quantiles: np.ndarray) -> None:
    """Write an arm's rows per farm, atomically."""
    out_dir = common.PRED_DIR / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = (
        base.select("time", "init_time", "lead_hours", "time_series_id", "actual")
        .with_columns(
            arm=pl.lit(arm),
            seed=pl.lit(0),
            setting=pl.lit(setting),
            **{c: pl.Series(quantiles[:, i]) for i, c in enumerate(QCOLS)},
        )
        .select(
            "time",
            "init_time",
            "lead_hours",
            "time_series_id",
            "arm",
            "seed",
            "setting",
            "actual",
            *QCOLS,
        )
    )
    for farm in common.FARMS:
        part = frame.filter(pl.col("time_series_id") == farm)
        path = out_dir / f"{arm}_s0_{farm}.parquet"
        tmp = path.with_suffix(".tmp")
        part.write_parquet(tmp)
        tmp.replace(path)


def main() -> None:
    """Build the blend and write XGBoost's rows beside it, for both settings."""
    records: list[dict[str, object]] = []
    for setting in common.SETTINGS:
        base, stack = common_rows(setting=setting)
        blend, rec = rolling(base=base, stack=stack)
        write(base=base, setting=setting, arm="blend_mdn" if POSTHOC else "blend", quantiles=blend)
        if POSTHOC:
            continue
        for arm in ("nn", "mdn"):
            ens = load_component(setting=setting, arm=arm)
            ens = base.select("time_series_id", "time").join(
                ens, on=["time_series_id", "time"], how="left"
            )
            write(
                base=base, setting=setting, arm=f"{arm}_ens", quantiles=ens.select(QCOLS).to_numpy()
            )
        write(base=base, setting=setting, arm="xgb_x2", quantiles=stack[COMPONENTS.index("xgb_x2")])
        records += [{"setting": setting, **r} for r in rec]
        print(setting, base.height, flush=True)
    pl.DataFrame(records).write_csv(
        common.DATA_DIR / ("weights_posthoc.csv" if POSTHOC else "weights.csv")
    )


if __name__ == "__main__":
    main()
