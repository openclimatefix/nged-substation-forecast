"""Fit the three cold-night widening fixes on the tuning months and write every arm's forecasts.

| Arm | Change to the reference's quantile offsets (quantile minus mean) |
|---|---|
| `ref` | none: experiment 160's arm I |
| `t` | one multiplier per quantile level and ensemble-temperature class, pooled over substations |
| `s` | on cold-forecast rows, times one plus a coefficient times the ensemble temperature spread |
| `o` | on cold-forecast rows, multipliers on the upper offsets (levels 0.75 to 0.99) only |

The mean forecast is never changed. The multipliers and the coefficient are fitted on the tuning
months only. The gate uses the experiment-111 ensemble's temperature, which exists at forecast time.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00164-*/e00164_fit.py
"""

import os
from typing import Final

import e00164_common as common
import numpy as np
import polars as pl

SPREAD_GRID: Final[np.ndarray] = np.arange(0.0, 2.01, 0.1)
MULTIPLIER_RANGE: Final[tuple[float, float]] = (0.5, 4.0)
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
MINIMUM_OFFSET: Final[float] = 1e-6
T_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.9, 0.95, 0.99)
"""Levels fix T rescales; the offsets of q25 to q75 are too small to fit by a ratio."""


def ensemble_temperature(*, frame: pl.DataFrame, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's ensemble-mean 24-hour temperature and its spread across members.

    The window is 24 half-hours either side of the row's valid time, clipped to the run's window.
    The spread is the standard deviation across the 51 members of each member's window mean.
    """
    with np.load(common.E111_STORE_DIR / f"{site}.npz") as stored:
        origins = stored["origins"]
        temperature = stored["temperature"].astype(np.float64)
    finite = np.isfinite(temperature)
    cumulative = np.zeros((*temperature.shape[:2], temperature.shape[2] + 1))
    count = np.zeros_like(cumulative)
    np.cumsum(np.where(finite, temperature, 0.0), axis=2, out=cumulative[:, :, 1:])
    np.cumsum(finite, axis=2, out=count[:, :, 1:])
    run = frame["run"].to_numpy().astype(np.int64)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    if not (origins[run] == origin).all():
        msg = f"{site}: the run column does not index experiment 111's store"
        raise ValueError(msg)
    step = np.timedelta64(30, "m")
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    position = ((valid - origin) // step).astype(np.int64) - 1
    last = temperature.shape[2] - 1
    low = np.clip(position - common.WINDOW_HALF_HOURS, 0, last)
    high = np.clip(position + common.WINDOW_HALF_HOURS, 0, last)
    total = cumulative[run, :, high + 1] - cumulative[run, :, low]
    number = count[run, :, high + 1] - count[run, :, low]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_member = np.where(number > 0, total / number, np.nan)
    return np.nanmean(per_member, axis=1), np.nanstd(per_member, axis=1)


def load(*, period: str, site: str) -> pl.DataFrame:
    """Return the reference's forecasts with the ensemble temperature columns."""
    frame = pl.read_parquet(common.reference_path(period=period, site=site)).select(
        list(common.BASE_COLUMNS)
    )
    mean_c, std_c = ensemble_temperature(frame=frame, site=site)
    return frame.with_columns(ens_mean_c=pl.Series(mean_c), ens_std_c=pl.Series(std_c))


def fit_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows a fit may use: a reading, outside the switching events, lead of 168 hours."""
    return frame.filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= common.HORIZON_HOURS)
        & pl.col("ens_mean_c").is_finite()
    )


def temperature_class(*, ens_mean_c: np.ndarray) -> np.ndarray:
    """Return the class (0 to 3) of each ensemble-mean temperature; `NaN` falls in class 3."""
    return np.searchsorted(np.array(common.TEMPERATURE_CLASS_EDGES_C), ens_mean_c, side="right")


def offsets_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the reference's quantile offsets from the mean, in points, shape (rows, 11)."""
    quantiles = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    return quantiles - frame["mean"].to_numpy().astype(np.float64)[:, None]


def fit_multipliers(
    *, residual: np.ndarray, offsets: np.ndarray, levels: tuple[float, ...]
) -> dict[float, float]:
    """Return, per level, the multiplier that makes the fitted rows' coverage nominal.

    For an upper offset (positive), `actual <= mean + k * offset` holds when the ratio of residual
    to offset is at most `k`, so `k` is the ratio's quantile at the level. For a lower offset
    (negative) the inequality flips, so `k` is the ratio's quantile at one minus the level.
    """
    out = {}
    for level in levels:
        position = common.QUANTILES.index(f"q{round(level * 100):02d}")
        offset = offsets[:, position]
        usable = np.abs(offset) > MINIMUM_OFFSET
        ratio = residual[usable] / offset[usable]
        sign_positive = np.median(offset[usable]) > 0
        quantile = level if sign_positive else 1.0 - level
        k = float(np.quantile(ratio, quantile))
        out[level] = float(np.clip(k, *MULTIPLIER_RANGE))
    return out


def apply_scale(*, frame: pl.DataFrame, scale: np.ndarray) -> pl.DataFrame:
    """Return the frame with quantiles rebuilt as mean plus scaled offsets, sorted per row."""
    mean = frame["mean"].to_numpy().astype(np.float64)
    quantiles = np.sort(mean[:, None] + offsets_of(frame=frame) * scale, axis=1)
    return frame.with_columns(
        pl.Series(q, quantiles[:, position].astype(np.float32))
        for position, q in enumerate(common.QUANTILES)
    )


def tail_pinball(*, residual: np.ndarray, offsets: np.ndarray, scale: np.ndarray) -> float:
    """Return the mean pinball loss over the tail levels for scaled offsets."""
    total = 0.0
    for level in TAIL_LEVELS:
        position = common.QUANTILES.index(f"q{round(level * 100):02d}")
        error = residual - offsets[:, position] * scale
        total += float(np.mean(np.maximum(level * error, (level - 1.0) * error)))
    return total / len(TAIL_LEVELS)


def main() -> None:
    """Fit on the tuning months, build the arms for every period and substation, write them."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (period, site): load(period=period, site=site)
        for period in common.PERIODS
        for site in common.SCORED_SITES
    }
    tuning = pl.concat([fit_rows(frame=frames["tuning", s]) for s in common.SCORED_SITES])
    residual = (tuning["actual"] - tuning["mean"]).to_numpy().astype(np.float64)
    offsets = offsets_of(frame=tuning)
    classes = temperature_class(ens_mean_c=tuning["ens_mean_c"].to_numpy())
    cold = tuning["ens_mean_c"].to_numpy() < common.COLD_GATE_C
    all_levels = T_LEVELS

    t_fit: dict[int, dict[float, float]] = {}
    for cls in range(len(common.TEMPERATURE_CLASS_EDGES_C) + 1):
        chosen = classes == cls
        n_rows = int(chosen.sum())
        if n_rows < common.MIN_CLASS_ROWS:
            t_fit[cls] = dict.fromkeys(all_levels, 1.0)
        else:
            t_fit[cls] = fit_multipliers(
                residual=residual[chosen], offsets=offsets[chosen], levels=all_levels
            )
    o_fit = fit_multipliers(
        residual=residual[cold], offsets=offsets[cold], levels=common.UPPER_FIX_LEVELS
    )
    spread = tuning["ens_std_c"].to_numpy()
    losses = [
        tail_pinball(
            residual=residual[cold],
            offsets=offsets[cold],
            scale=1.0 + beta * spread[cold],
        )
        for beta in SPREAD_GRID
    ]
    beta = float(SPREAD_GRID[int(np.argmin(losses))])
    parameters = {
        "t": {str(c): {str(k): v for k, v in m.items()} for c, m in t_fit.items()},
        "o": {str(k): v for k, v in o_fit.items()},
        "s_beta": beta,
        "s_losses": dict(zip(map(str, SPREAD_GRID.round(2)), losses, strict=True)),
        "tuning_rows": tuning.height,
        "tuning_cold_rows": int(cold.sum()),
        "tuning_class_rows": [int((classes == c).sum()) for c in range(4)],
    }
    common.write_json_atomic(path=common.WORK_DIR / "fit_parameters.json", content=parameters)
    print(parameters, flush=True)

    for (period, site), frame in frames.items():
        mean_c = frame["ens_mean_c"].to_numpy()
        std_c = np.nan_to_num(frame["ens_std_c"].to_numpy(), nan=0.0)
        is_cold = np.nan_to_num(mean_c, nan=99.0) < common.COLD_GATE_C
        cls = temperature_class(ens_mean_c=np.nan_to_num(mean_c, nan=99.0))
        n_rows = frame.height
        scale_t = np.ones((n_rows, len(common.QUANTILES)))
        scale_o = np.ones((n_rows, len(common.QUANTILES)))
        for position, level in enumerate(common.LEVELS.tolist()):
            if level in o_fit:
                scale_o[:, position] = np.where(is_cold, o_fit[level], 1.0)
            if level in T_LEVELS:
                scale_t[:, position] = np.array([t_fit[c][level] for c in range(4)])[cls]
        scale_s = np.where(is_cold, 1.0 + beta * std_c, 1.0)[:, None] * np.ones(
            (1, len(common.QUANTILES))
        )
        scale_s[:, common.MEDIAN_POSITION] = 1.0
        scales = {"ref": np.ones_like(scale_t), "t": scale_t, "s": scale_s, "o": scale_o}
        for arm, scale in scales.items():
            built = apply_scale(frame=frame, scale=scale).with_columns(arm=pl.lit(arm))
            target = common.out_path(arm=arm, period=period, site=site)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            built.select(list(common.OUTPUT_COLUMNS)).write_parquet(temporary)
            temporary.rename(target)
        print(period, site, "written", flush=True)


if __name__ == "__main__":
    main()
