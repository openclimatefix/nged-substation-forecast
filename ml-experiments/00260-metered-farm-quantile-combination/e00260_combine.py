# ruff: noqa: E501
"""Combine experiment 243's stored forecasters per quantile level, and adjust the outer levels.

Writes arms c1 (equal weights), c2 (rolling-origin pinball weights), c2_lomo (leave-one-month-out
weights), and c3 (c2 plus the adaptive conformal adjustment) for each setting and farm, and the fitted
weights.

    uv run --no-sync python ml-experiments/00260-*/e00260_combine.py
"""

from typing import Final

import e00260_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

COMPONENTS: Final[tuple[str, ...]] = ("h5_quantile", "nn_pure", "xgboost")
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
GRID_STEPS: Final[tuple[float, ...]] = (0.05, 0.01, 0.002)
"""Grid step per stage of the weight search."""
REFINE_STEPS: Final[int] = 5
"""Grid points each side of the best weights in a refining stage."""
BURN_IN_MONTHS: Final[int] = 3
"""The first scored months are scored with equal weights."""
STEP: Final[float] = 0.5
"""The adaptive conformal step size, fixed in the plan before scoring."""
ADJUSTED: Final[dict[str, tuple[int, bool]]] = {
    "q01": (0, False),
    "q05": (1, False),
    "q95": (9, True),
    "q99": (10, True),
}
"""Adjusted levels: column, (index, whether it is an upper level)."""
LEAD_EDGES: Final[tuple[float, ...]] = (18.0, 24.0, 30.0)
CSI_EDGES: Final[tuple[float, ...]] = (0.3, 0.6, 0.9)


def load_component(*, setting: str, arm: str) -> pl.DataFrame:
    """Return an arm's seed-averaged quantiles per (farm, time)."""
    files = sorted((common.E243_DIR / "predictions" / setting).glob(f"{arm}_s*_*.parquet"))
    frame = pl.concat([pl.read_parquet(f) for f in files])
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
    csi = pl.concat(
        [
            common.load_farm(s).frame.select("time", "csi", time_series_id=pl.lit(s))
            for s in common.FARMS
        ]
    )
    base = (
        base.join(csi, on=["time_series_id", "time"], how="left")
        .sort("time_series_id", "time")
        .with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
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


def rolling_and_lomo(
    *, base: pl.DataFrame, stack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Return c2 and c2_lomo quantiles and the weight records, fitted month by month."""
    months = sorted(base["month"].unique().to_list())
    month_col = base["month"].to_numpy()
    time = base["time"].dt.epoch("us").to_numpy()
    init = base["init_time"].dt.epoch("us").to_numpy()
    n_rows = base.height
    c2 = np.empty((n_rows, len(common.LEVELS)))
    lomo = np.empty_like(c2)
    records: list[dict[str, object]] = []
    equal = np.full((len(common.LEVELS), stack.shape[0]), 1.0 / stack.shape[0])
    for i, month in enumerate(months):
        in_month = month_col == month
        first_run = init[in_month].min()
        earlier = time < first_run
        weights = equal if i < BURN_IN_MONTHS else fit_weights(base=base, stack=stack, rows=earlier)
        c2[in_month] = combine(stack=stack[:, in_month, :], weights=weights)
        lomo_weights = fit_weights(base=base, stack=stack, rows=~in_month)
        lomo[in_month] = combine(stack=stack[:, in_month, :], weights=lomo_weights)
        for kind, w in (("c2", weights), ("c2_lomo", lomo_weights)):
            for k, level in enumerate(common.LEVELS):
                records.append(
                    {
                        "kind": kind,
                        "month": month,
                        "level": float(level),
                        **{a: float(w[k, j]) for j, a in enumerate(COMPONENTS)},
                    }
                )
    return c2, lomo, records


def band(*, x: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Return the index of the band each value falls in (missing values go to the last band)."""
    return np.digitize(np.nan_to_num(x, nan=1.0), edges)


def adaptive_conformal(*, base: pl.DataFrame, quantiles: np.ndarray) -> np.ndarray:
    """Adjust the outer levels with one offset per lead and clear-sky band, from observed misses only.

    Before each run, each group's offset moves by `STEP` times the group's mean miss rate minus the
    nominal rate, over rows whose valid time is before the run's initialisation time and that earlier
    runs forecast. Upper levels move up when they miss too often, lower levels move down.
    """
    adjusted = quantiles.copy()
    actual = base["actual"].to_numpy()
    time = base["time"].dt.epoch("us").to_numpy()
    init = base["init_time"].dt.epoch("us").to_numpy()
    group = band(x=base["lead_hours"].to_numpy(), edges=LEAD_EDGES) * (len(CSI_EDGES) + 1) + band(
        x=base["csi"].to_numpy(), edges=CSI_EDGES
    )
    n_groups = (len(LEAD_EDGES) + 1) * (len(CSI_EDGES) + 1)
    offsets = {c: np.zeros(n_groups) for c in ADJUSTED}
    done = np.zeros(base.height, dtype=bool)
    issued_offset = {c: np.zeros(base.height) for c in ADJUSTED}
    for run in np.unique(init):
        observed = (time < run) & ~done & (init < run)
        for c, (k, upper) in ADJUSTED.items():
            level = float(common.LEVELS[k])
            nominal = 1.0 - level if upper else level
            for g in range(n_groups):
                rows = observed & (group == g)
                if not rows.any():
                    continue
                q = quantiles[rows, k] + issued_offset[c][rows]
                miss = (actual[rows] > q) if upper else (actual[rows] < q)
                shift = STEP * (miss.mean() - nominal)
                offsets[c][g] += shift if upper else -shift
        done |= observed
        now = init == run
        for c, (k, _upper) in ADJUSTED.items():
            issued_offset[c][now] = offsets[c][group[now]]
            adjusted[now, k] = quantiles[now, k] + issued_offset[c][now]
    return np.sort(np.clip(adjusted, 0.0, None), axis=1)


def write(*, base: pl.DataFrame, setting: str, arm: str, quantiles: np.ndarray) -> None:
    """Write an arm's rows per farm, atomically."""
    out_dir = common.PRED_DIR / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = base.select("time", "init_time", "lead_hours", "time_series_id", "actual").with_columns(
        arm=pl.lit(arm),
        seed=pl.lit(0),
        setting=pl.lit(setting),
        **{c: pl.Series(quantiles[:, i]) for i, c in enumerate(QCOLS)},
    )
    for farm in common.FARMS:
        part = frame.filter(pl.col("time_series_id") == farm)
        path = out_dir / f"{arm}_s0_{farm}.parquet"
        tmp = path.with_suffix(".tmp")
        part.write_parquet(tmp)
        tmp.replace(path)


def main() -> None:
    """Build every arm for both settings."""
    all_records: list[dict[str, object]] = []
    for setting in common.SETTINGS:
        base, stack = common_rows(setting=setting)
        write(base=base, setting=setting, arm="c1", quantiles=np.sort(stack.mean(axis=0), axis=1))
        for j, name in enumerate(("h5_mean", "nn_mean", "xgb_mean")):
            write(base=base, setting=setting, arm=name, quantiles=stack[j])
        c2, lomo, records = rolling_and_lomo(base=base, stack=stack)
        write(base=base, setting=setting, arm="c2", quantiles=c2)
        write(base=base, setting=setting, arm="c2_lomo", quantiles=lomo)
        if setting == "own_history":
            write(
                base=base,
                setting=setting,
                arm="c3",
                quantiles=adaptive_conformal(base=base, quantiles=c2),
            )
        all_records += [{"setting": setting, **r} for r in records]
        print(setting, base.height, flush=True)
    pl.DataFrame(all_records).write_csv(common.DATA_DIR / "weights.csv")


if __name__ == "__main__":
    main()
