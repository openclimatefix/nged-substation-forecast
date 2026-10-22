"""Adapt arm A's quantile offsets online from recent coverage misses; write every arm's forecasts.

| Arm | Quantiles |
|---|---|
| `a` | experiment 169's arm A, as stored |
| `cw` | `a` with the q90, q95, and q99 offsets times 1.06, 1.06, and 1.11 (a control) |
| `aci_g<gamma>` | `a` with each offset times the learned scale of its hour band and level |

For a series, hour band b, and level q, a log scale `u[b, q]` starts at 0. After each reading at
valid time v in band b (in valid-time order), `u[b, q] += gamma * (miss - alpha_q)`, where `miss`
is 1 when the reading falls outside arm A's quantile (above an upper level, below a lower level)
as issued by the latest run before v, and `alpha_q` is the nominal miss rate. This is a step of
stochastic gradient descent on the pinball loss of the scale, the update of adaptive conformal
inference (Gibbs and Candes, 2021). `u` is clamped to `[ln 0.7, ln 2.0]`. A forecast issued at T
uses the state after every reading at least 3.5 hours old. The mean is never changed.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00213-*/e00213_adapt.py
"""

import e00213_common as common
import numpy as np
import polars as pl

LOG_FLOOR: float = float(np.log(common.SCALE_FLOOR))
LOG_CAP: float = float(np.log(common.SCALE_CAP))
POSITIONS: np.ndarray = np.array(common.ADAPTED_POSITIONS)
IS_UPPER: np.ndarray = common.QUANTILES.index("q75") <= POSITIONS
ALPHA: np.ndarray = np.where(IS_UPPER, 1.0 - common.LEVELS[POSITIONS], common.LEVELS[POSITIONS])


def readings_of(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return one usable reading per valid time, with arm A's quantiles from the latest run.

    A reading is unusable when it is missing or inside a hand-filtered switching event.
    """
    return (
        frame.filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
        .sort(["valid_time", "lead_hours"])
        .unique(subset="valid_time", keep="first", maintain_order=True)
        .sort("valid_time")
    )


def state_history(*, readings: pl.DataFrame, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the valid times (ns) and the log scales (readings, bands, levels) after each reading.

    The state restarts at 0 after a gap of more than `BLOCK_GAP_HOURS` between readings.
    """
    times = readings["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[ns]")
    actual = readings["actual"].to_numpy().astype(np.float64)
    band = readings["hour_band"].to_numpy().astype(np.int64)
    quantiles = readings.select(list(common.QUANTILES)).to_numpy().astype(np.float64)[:, POSITIONS]
    miss = np.where(IS_UPPER[None, :], actual[:, None] > quantiles, actual[:, None] < quantiles)
    gap_ns = np.timedelta64(int(common.BLOCK_GAP_HOURS * 3600), "s").astype("timedelta64[ns]")
    history = np.zeros((len(times), common.N_BANDS, len(POSITIONS)))
    state = np.zeros((common.N_BANDS, len(POSITIONS)))
    for i in range(len(times)):
        if i > 0 and times[i] - times[i - 1] > gap_ns:
            state[:] = 0.0
        state[band[i]] = np.clip(state[band[i]] + gamma * (miss[i] - ALPHA), LOG_FLOOR, LOG_CAP)
        history[i] = state
    return times, history


def adapted(*, frame: pl.DataFrame, times: np.ndarray, history: np.ndarray) -> pl.DataFrame:
    """Return the frame with each offset times the scale learned from readings 3.5 hours old."""
    issue = (
        frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[ns]")
    )
    cutoff = issue - np.timedelta64(int(common.DELAY_HOURS * 3600), "s").astype("timedelta64[ns]")
    index = np.searchsorted(times, cutoff, side="right") - 1
    gap_ns = np.timedelta64(int(common.BLOCK_GAP_HOURS * 3600), "s").astype("timedelta64[ns]")
    usable = index >= 0
    safe = np.where(usable, index, 0)
    usable &= (cutoff - times[safe]) <= gap_ns
    band = frame["hour_band"].to_numpy().astype(np.int64)
    log_scale = np.where(usable[:, None], history[safe, band, :], 0.0)
    scale = np.ones((frame.height, len(common.QUANTILES)))
    scale[:, POSITIONS] = np.exp(log_scale)
    return apply_scale(frame=frame, scale=scale)


def apply_scale(*, frame: pl.DataFrame, scale: np.ndarray) -> pl.DataFrame:
    """Return the frame with quantiles rebuilt as mean plus scaled offsets, sorted per row."""
    mean = frame["mean"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    rebuilt = np.sort(mean[:, None] + (quantiles - mean[:, None]) * scale, axis=1)
    return frame.with_columns(
        pl.Series(q, rebuilt[:, position].astype(np.float32))
        for position, q in enumerate(common.QUANTILES)
    )


def write_arm(*, frame: pl.DataFrame, arm: str, period: str, site: str) -> None:
    """Write one arm's forecasts atomically."""
    common.write_parquet_atomic(
        frame=frame.with_columns(arm=pl.lit(arm)).select(list(common.OUTPUT_COLUMNS)),
        target=common.out_path(arm=arm, period=period, site=site),
    )


def main() -> None:
    """Build every arm for every period and substation and write the forecasts."""
    for period in common.PERIODS:
        for site in common.SCORED_SITES:
            frame = pl.read_parquet(common.stored_path(period=period, site=site)).select(
                list(common.BASE_COLUMNS)
            )
            readings = readings_of(frame=frame)
            write_arm(frame=frame, arm="a", period=period, site=site)
            cw_scale = np.ones((frame.height, len(common.QUANTILES)))
            for position, multiplier in common.CW_MULTIPLIERS.items():
                cw_scale[:, position] = multiplier
            write_arm(
                frame=apply_scale(frame=frame, scale=cw_scale), arm="cw", period=period, site=site
            )
            for gamma in common.GAMMAS:
                times, history = state_history(readings=readings, gamma=gamma)
                write_arm(
                    frame=adapted(frame=frame, times=times, history=history),
                    arm=common.gamma_arm(gamma=gamma),
                    period=period,
                    site=site,
                )
            # Reproduction control: gamma = 0 must give arm A exactly.
            times, history = state_history(readings=readings, gamma=0.0)
            same = adapted(frame=frame, times=times, history=history)
            if not same.select(list(common.QUANTILES)).equals(
                apply_scale(
                    frame=frame, scale=np.ones((frame.height, len(common.QUANTILES)))
                ).select(list(common.QUANTILES))
            ):
                msg = f"{period} {site}: gamma = 0 differs from arm A"
                raise RuntimeError(msg)
            print(period, site, "written", readings.height, "readings", flush=True)


if __name__ == "__main__":
    main()
