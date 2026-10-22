# ruff: noqa: E501
"""Fit the band factors on the tuning months and build every arm of experiment 192 for all rows.

| Arm | Quantiles |
|---|---|
| `a` | experiment 164's fix O (experiment 169's arm A), as stored |
| `m` | 51 members through the solar term (experiment 184's `delta_m`), on every row |
| `t` | 51 members through the temperature terms only |
| `st` | 51 members through every input (experiment 184's `delta_f`) |
| `mr` | placebo: arm A's offsets times a factor per lead band and kind, with every response zero |
| `tp` | placebo: `t` with each row's 51 responses taken from a random row of the same substation, month, hour band, lead day, and temperature band |
| `tg` | two numbers: Gaussian points with each row's own mean and spread of the `t` responses |
| `stg` | two numbers: the same for the `st` responses |

Each arm is a mixture of 51 members, each member's forecast being arm A's mean plus its response,
with arm A's 11 quantile offsets times one factor per (lead band, row kind) around it. Every factor
is fitted on the tuning months.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00192-*/e00192_arms.py
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00192_common as common
import e00192_data as data
import e00192_fit as fit
import numpy as np
import polars as pl
from scipy.stats import norm
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours

ARMS: Final[tuple[str, ...]] = ("a", "m", "t", "st", "mr", "tp", "tg", "stg")
KINDS: Final[tuple[str, ...]] = ("daylight", "twilight", "night")
ARM_DIR = common.WORK_DIR / "arms"
TEMPERATURE_EDGES: Final[tuple[float, ...]] = (-2.0, 2.0, 6.0, 10.0, 14.0)
GAUSS_POINTS: Final[np.ndarray] = norm.ppf((np.arange(51) + 0.5) / 51)
SHUFFLE_SEED: Final[int] = 192
SOURCE: Final[dict[str, str | None]] = {
    "m": "delta_m",
    "t": "delta_t",
    "st": "delta_st",
    "mr": None,
    "tp": "delta_t",
    "tg": "delta_t",
    "stg": "delta_st",
}


def responses(*, arm: str, frame: pl.DataFrame, deltas: dict[str, np.ndarray]) -> np.ndarray:
    """Return an arm's 51 member responses per row, with the placebo or summary transform applied."""
    source = SOURCE[arm]
    if source is None:
        return np.zeros_like(deltas["delta_t"], dtype=np.float64)
    delta = deltas[source].astype(np.float64)
    if arm in ("tg", "stg"):
        mean = np.nanmean(delta, axis=1, keepdims=True)
        return mean + np.nanstd(delta, axis=1, keepdims=True) * GAUSS_POINTS[None, :]
    if arm == "tp":
        return strong_shuffle(frame=frame, delta=delta)
    return delta


def strong_shuffle(*, frame: pl.DataFrame, delta: np.ndarray) -> np.ndarray:
    """Shuffle whole rows of responses within substation, month, hour band, lead day, and temperature band."""
    lead_day = np.minimum((frame["lead_hours"].to_numpy() // 24).astype(int), 6)
    t_band = np.digitize(np.nan_to_num(frame["t_mean"].to_numpy(), nan=99.0), TEMPERATURE_EDGES)
    group = (
        frame["site"]
        + "|"
        + frame["valid_time"].dt.month().cast(pl.String)
        + "|"
        + frame["hour_band"].cast(pl.String)
        + "|"
        + pl.Series(lead_day.astype(str))
        + "|"
        + pl.Series(t_band.astype(str))
    ).to_numpy()
    rng = np.random.default_rng(SHUFFLE_SEED)
    out = delta.copy()
    for key in np.unique(group):
        rows = np.flatnonzero(group == key)
        out[rows] = delta[rng.permutation(rows)]
    return out


def fit_cell_factors(*, frame: pl.DataFrame, delta: np.ndarray) -> list[list[float]]:
    """Return the factor of each (kind, lead band) cell that minimises the tuning pinball loss."""
    usable = data.usable_rows(frame=frame)
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64) - mean[:, None]
    actual = frame["actual"].to_numpy().astype(np.float64)
    kind = frame["kind"].to_numpy()
    rng = np.random.default_rng(fit.SEED)
    out = []
    for name in KINDS:
        factors = []
        for band in range(N_LEAD_BANDS):
            rows = np.flatnonzero(usable & (kind == name) & (lead_band == band))
            rows = np.sort(
                rng.choice(rows, size=min(fit.FACTOR_FIT_ROWS, len(rows)), replace=False)
            )
            filled = fit.fill_missing(delta=delta[rows])
            losses = [
                float(
                    fit.pinball_rows(
                        actual=actual[rows],
                        quantiles=fit.mixture(
                            mean=mean[rows],
                            delta=filled,
                            offsets=offsets[rows],
                            factor=np.full(len(rows), f),
                        ),
                    ).mean()
                )
                for f in fit.FACTOR_GRID
            ]
            factors.append(float(fit.FACTOR_GRID[int(np.argmin(losses))]))
        out.append(factors)
    return out


def apply_arm(*, frame: pl.DataFrame, delta: np.ndarray, factors: list[list[float]]) -> np.ndarray:
    """Return an arm's 11 quantiles: the mixture on rows with members, arm A's elsewhere."""
    out = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)
    has_members = np.isfinite(delta).any(axis=1)
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = out - mean[:, None]
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    kind_index = np.array([KINDS.index(k) for k in frame["kind"].to_numpy()])
    table = np.array(factors)
    for k in range(len(KINDS)):
        rows = np.flatnonzero(has_members & (kind_index == k))
        factor = table[k][lead_band[rows]]
        filled = fit.fill_missing(delta=delta[rows])
        out[rows] = fit.mixture(mean=mean[rows], delta=filled, offsets=offsets[rows], factor=factor)
    return out


def _fit_job(arm: str) -> tuple[str, list[list[float]]]:
    cache = common.load_cache()
    frame, deltas = data.load_period(period="tuning", cache=cache)
    factors = fit_cell_factors(frame=frame, delta=responses(arm=arm, frame=frame, deltas=deltas))
    print(arm, "factors", factors, flush=True)
    return arm, factors


def _period_job(args: tuple[str, dict[str, list[list[float]]]]) -> str:
    period, factors = args
    cache = common.load_cache()
    frame, deltas = data.load_period(period=period, cache=cache)
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(ARM_DIR / f"{period}_frame.parquet")
    base = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float32)
    np.save(ARM_DIR / f"{period}_a.npy", base)
    for arm in ARMS[1:]:
        quantiles = apply_arm(
            frame=frame,
            delta=responses(arm=arm, frame=frame, deltas=deltas),
            factors=factors[arm],
        )
        np.save(ARM_DIR / f"{period}_{arm}.npy", np.sort(quantiles, axis=1).astype(np.float32))
    return f"{period} written, {frame.height} rows"


def main() -> None:
    """Fit factors for every arm on the tuning months, then write every arm for every period."""
    with ProcessPoolExecutor(max_workers=4) as pool:
        factors = dict(pool.map(_fit_job, ARMS[1:]))
        common.write_json_atomic(path=common.WORK_DIR / "factors.json", content=factors)
        for line in pool.map(
            _period_job, [(p, factors) for p in ("tuning", "replicate", "scored")]
        ):
            print(line, flush=True)


if __name__ == "__main__":
    main()
