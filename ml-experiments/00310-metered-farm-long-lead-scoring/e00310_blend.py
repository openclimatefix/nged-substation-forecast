# ruff: noqa: E501
"""Blend GAM, the mixture head (seed mean), and XGBoost per quantile level, rolling-origin, per day band.

Copied from experiment 300's blend: per-level simplex weights minimising the farm-equal pinball
loss on rows whose valid time precedes the month's first run; the first three scored months use
equal weights. A month is the month of the run (`init_time`). `blend_nolag` mixes the three
models fitted without lagged power (own history only).

    uv run --no-sync python ml-experiments/00310-*/e00310_blend.py
"""

from typing import Final

import e00310_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

BLENDS: Final[dict[str, tuple[str, ...]]] = {
    "blend": ("gam", "mdn", "xgb"),
    "blend_nolag": ("gam_nolag", "mdn_nolag", "xgb_nolag"),
}
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
GRID_STEPS: Final[tuple[float, ...]] = (0.05, 0.01, 0.002)
REFINE_STEPS: Final[int] = 5
BURN_IN_MONTHS: Final[int] = 3
KEYS: Final[list[str]] = ["time_series_id", "time", "init_time"]


def load_component(*, band: str, setting: str, arm: str) -> pl.DataFrame:
    """Return an arm's seed-averaged quantiles per (farm, time, run)."""
    files = sorted((common.PRED_DIR / band / setting).glob(f"{arm}_s*_*_*.parquet"))
    if not files:
        return pl.DataFrame()
    frame = pl.concat(
        [pl.read_parquet(f).select(*KEYS, "lead_hours", "actual", *QCOLS) for f in files]
    )
    return frame.group_by(KEYS).agg(
        pl.col("lead_hours").first(), pl.col("actual").first(), pl.col(QCOLS).mean()
    )


def common_rows(
    *, band: str, setting: str, components: tuple[str, ...]
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the rows every component holds, and the components' quantiles (arm, row, level)."""
    frames = {a: load_component(band=band, setting=setting, arm=a) for a in components}
    keys = frames[components[0]].select(KEYS)
    for f in frames.values():
        keys = keys.join(f.select(KEYS), on=KEYS)
    keys = keys.sort(KEYS)
    base = (
        keys.join(frames[components[0]], on=KEYS)
        .sort(KEYS)
        .select(*KEYS, "lead_hours", "actual")
        .with_columns(month=pl.col("init_time").dt.strftime("%Y-%m"))
    )
    stack = np.stack(
        [keys.join(frames[a], on=KEYS).sort(KEYS).select(QCOLS).to_numpy() for a in components]
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


def rolling(
    *, base: pl.DataFrame, stack: np.ndarray, components: tuple[str, ...]
) -> tuple[np.ndarray, list[dict[str, object]]]:
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
                    **{a: float(weights[k, j]) for j, a in enumerate(components)},
                }
            )
    return blend, records


def write(*, base: pl.DataFrame, band: str, setting: str, arm: str, quantiles: np.ndarray) -> None:
    """Write an arm's rows per farm, atomically."""
    out_dir = common.PRED_DIR / band / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = base.select(*KEYS, "lead_hours", "actual").with_columns(
        arm=pl.lit(arm),
        seed=pl.lit(0),
        setting=pl.lit(setting),
        band=pl.lit(band),
        **{c: pl.Series(quantiles[:, i]) for i, c in enumerate(QCOLS)},
    )
    for farm in common.FARMS:
        part = frame.filter(pl.col("time_series_id") == farm)
        path = out_dir / f"{arm}_s0_{farm}.parquet"
        tmp = path.with_suffix(".tmp")
        part.write_parquet(tmp)
        tmp.replace(path)


def main() -> None:
    """Build each blend and the seed-averaged mixture head, for every band and both settings."""
    records: list[dict[str, object]] = []
    for band in common.BAND_ORDER:
        for setting in common.SETTINGS:
            for name, components in BLENDS.items():
                if name == "blend_nolag" and setting != "own_history":
                    continue
                base, stack = common_rows(band=band, setting=setting, components=components)
                blend, rec = rolling(base=base, stack=stack, components=components)
                write(base=base, band=band, setting=setting, arm=name, quantiles=blend)
                records += [{"band": band, "setting": setting, "blend": name, **r} for r in rec]
                print(band, setting, name, base.height, flush=True)
            for arm, out in (("mdn", "mdn_ens"), ("mdn_nolag", "mdn_nolag_ens")):
                if arm == "mdn_nolag" and setting != "own_history":
                    continue
                ens = load_component(band=band, setting=setting, arm=arm)
                base, _ = common_rows(band=band, setting=setting, components=("gam", arm))
                ens = base.select(KEYS).join(ens, on=KEYS, how="left")
                write(
                    base=base,
                    band=band,
                    setting=setting,
                    arm=out,
                    quantiles=ens.select(QCOLS).to_numpy(),
                )
    pl.DataFrame(records).write_csv(common.DATA_DIR / "weights.csv")


if __name__ == "__main__":
    main()
