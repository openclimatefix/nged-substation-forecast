# ruff: noqa: E501
"""Helpers copied from experiment 193 (two-number arms); `e00278_s_replay.py` calls them.

Original docstring: build the two-number arms and their placebos from experiment 184's stored
member responses.

Each daylight row of arm A has 51 member responses (`delta_m`, the change in the mean forecast when
one member's unit PV shape replaces the ensemble mean's). This script summarises them and rebuilds
the forecast's 11 quantiles from the summary alone:

| Arm | Summary used | Quantiles |
|---|---|---|
| `q2` | mean `mu` and standard deviation `sigma` | 51 Gaussian points `mu + s * sigma * z`, each with arm A's offsets times `f`, pooled |
| `q2s1` | the same, spread factor fixed at 1 | nested check: `q2` with `s = 1` |
| `q3` | `mu`, `sigma`, 10th and 90th percentiles | a two-piece normal shape from the percentiles, standardised to `sigma`, in place of the Gaussian |
| `mr` | none | placebo: every response zero, arm A's offsets times `f` |
| `mp` | `q2`'s, from a random row of the same substation, month, and hour band | placebo |
| `mp_strong` | `q2`'s, from a random row of the same substation, month, hour band, lead day, and clear-sky band | placebo |
| `qn` | `q2`'s, from a random daylight row of the same substation, month, and lead day, given to night rows | night placebo: arm A in daylight |

The offset factor `f` and the spread factor `s` are fitted per lead band on the tuning months, by
the daily-loss grid search of experiment 187 (`f` first with `s = 1`, then `s`, then `f` again).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00202-*/e00278_s_arms.py
"""

from collections.abc import Callable
from typing import Final

import e00278_s_common as common
import e00278_s_data as data
import e00278_s_score as score
import numpy as np
import polars as pl
from scipy.stats import norm
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.ens_members import mixture_quantiles

N_MEMBERS: Final[int] = 51
Z_POINTS: Final[np.ndarray] = norm.ppf((np.arange(N_MEMBERS) + 0.5) / N_MEMBERS)
F_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
S_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.5001, 0.1), 2)
FIT_ROWS: Final[int] = 20_000
CHUNK: Final[int] = 4000
SEED: Final[int] = 193
Z_10_90: Final[float] = float(norm.ppf(0.9))
EPSILON: Final[float] = 1e-9
ARM_DIR = common.WORK_DIR / "arms"
E184_ARM_DIR = common.E184_WORK_DIR / "arms"


def summaries(*, delta: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four summary numbers of each row's member responses (NaN for no member)."""
    with np.errstate(all="ignore"):
        return {
            "mu": np.nanmean(delta, axis=1),
            "sigma": np.nanstd(delta, axis=1),
            "p10": np.nanpercentile(delta, 10.0, axis=1),
            "p90": np.nanpercentile(delta, 90.0, axis=1),
        }


def gaussian_points(*, stats: dict[str, np.ndarray], spread: float) -> np.ndarray:
    """Return 51 points per row: `mu + spread * sigma * z`."""
    return stats["mu"][:, None] + spread * stats["sigma"][:, None] * Z_POINTS[None, :]


def skewed_points(*, stats: dict[str, np.ndarray], spread: float) -> np.ndarray:
    """Return 51 points per row of a two-piece normal shape with mean `mu` and std `spread * sigma`.

    The lower and upper scales come from the 10th and 90th percentiles; the points are then
    standardised so that their mean is `mu` and their standard deviation `spread * sigma`.
    """
    lower = np.maximum(stats["mu"] - stats["p10"], 0.0) / Z_10_90
    upper = np.maximum(stats["p90"] - stats["mu"], 0.0) / Z_10_90
    raw = np.where(Z_POINTS[None, :] < 0, lower[:, None], upper[:, None]) * Z_POINTS[None, :]
    centred = raw - raw.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    shape = np.where(scale > EPSILON, centred / np.maximum(scale, EPSILON), Z_POINTS[None, :])
    return stats["mu"][:, None] + spread * stats["sigma"][:, None] * shape


def mixture(*, mean: np.ndarray, delta: np.ndarray, offsets: np.ndarray, f: float) -> np.ndarray:
    """Return the 11 mixture quantiles of 51 centres `mean + delta`, each with offsets times `f`."""
    out = np.empty((len(mean), len(data.QUANTILES)))
    for start in range(0, len(mean), CHUNK):
        part = slice(start, start + CHUNK)
        centres = (mean[part, None] + delta[part])[:, :, None]
        out[part] = mixture_quantiles(member_quantiles=centres + f * offsets[part, None, :])
    return out


def pinball_rows(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error), axis=1)


def daylight_rows(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows an arm changes: daylight with at least one member."""
    return ((frame["kind"] == "daylight") & (frame["n_members"] > 0)).to_numpy()


def fit_factors(
    *, frame: pl.DataFrame, make_points: Callable[[float], np.ndarray], fit_spread: bool
) -> list[tuple[float, float]]:
    """Return `(f, s)` per lead band minimising the tuning daylight loss.

    Args:
        frame: The tuning rows.
        make_points: Called as `make_points(spread)`, returns the (rows, 51) centres' responses.
        fit_spread: When False, `s` stays 1.

    Returns:
        The pair `(f, s)` of each lead band.
    """
    usable = data.usable_rows(frame=frame) & daylight_rows(frame=frame)
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    mean = frame["mean"].to_numpy().astype(np.float64)
    offsets = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64) - mean[:, None]
    actual = frame["actual"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(SEED)
    out = []
    for band in range(N_LEAD_BANDS):
        rows = np.flatnonzero(usable & (lead_band == band))
        rows = np.sort(rng.choice(rows, size=min(FIT_ROWS, len(rows)), replace=False))

        def loss(f: float, s: float, rows: np.ndarray = rows) -> float:
            q = mixture(mean=mean[rows], delta=make_points(s)[rows], offsets=offsets[rows], f=f)
            return float(pinball_rows(actual=actual[rows], quantiles=q).mean())

        f, s = 1.0, 1.0
        f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
        if fit_spread:
            s = float(S_GRID[int(np.argmin([loss(f, g) for g in S_GRID]))])
            f = float(F_GRID[int(np.argmin([loss(g, s) for g in F_GRID]))])
        out.append((f, s))
        print("band", band, "f", f, "s", s, "loss", loss(f, s), flush=True)
    return out


def apply_arm(
    *,
    frame: pl.DataFrame,
    make_points: Callable[[float], np.ndarray],
    factors: list[tuple[float, float]],
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Return an arm's 11 quantiles: the mixture on the changed rows and arm A's elsewhere.

    Args:
        frame: The rows.
        make_points: Called as `make_points(spread)`, returns (rows, 51) responses.
        factors: `(f, s)` per lead band.
        rows: The row indices to change; daylight rows with a member when None.

    Returns:
        The 11 quantiles of every row, shape (rows, 11).
    """
    out = frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64)
    if rows is None:
        rows = np.flatnonzero(daylight_rows(frame=frame))
    mean = frame["mean"].to_numpy().astype(np.float64)
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    offsets = out - mean[:, None]
    for band, (f, s) in enumerate(factors):
        chosen = rows[lead_band[rows] == band]
        if len(chosen) == 0:
            continue
        points = make_points(s)
        out[chosen] = mixture(mean=mean[chosen], delta=points[chosen], offsets=offsets[chosen], f=f)
    return out


def group_keys(*, frame: pl.DataFrame, strong: bool) -> np.ndarray:
    """Return a shuffle-group label per row: substation, month, and hour band (and more if strong)."""
    key = (
        frame["site"]
        + "|"
        + frame["valid_time"].dt.month().cast(pl.String)
        + "|"
        + frame["hour_band"].cast(pl.String)
    )
    if strong:
        lead_day = np.minimum((frame["lead_hours"].to_numpy() // 24).astype(int), 6)
        key = key + "|" + pl.Series(lead_day.astype(str)) + "|" + frame["csi_band"].fill_null("na")
    return key.to_numpy()


def shuffled_stats(
    *, frame: pl.DataFrame, stats: dict[str, np.ndarray], strong: bool, seed: int
) -> dict[str, np.ndarray]:
    """Return the summaries with each daylight row's set taken from a random row of its group."""
    day = daylight_rows(frame=frame)
    group = group_keys(frame=frame, strong=strong)
    rng = np.random.default_rng(seed)
    source = np.arange(frame.height)
    for key in np.unique(group[day]):
        rows = np.flatnonzero(day & (group == key))
        source[rows] = rows[rng.permutation(len(rows))]
    return {name: values[source] for name, values in stats.items()}


def night_placebo_stats(
    *, frame: pl.DataFrame, stats: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the night rows and, for each, a daylight row's summaries (same substation, month, lead day)."""
    lead_day = np.minimum((frame["lead_hours"].to_numpy() // 24).astype(int), 6)
    group = (
        frame["site"]
        + "|"
        + frame["valid_time"].dt.month().cast(pl.String)
        + "|"
        + pl.Series(lead_day.astype(str))
    ).to_numpy()
    day = daylight_rows(frame=frame) & np.isfinite(stats["mu"])
    night = np.flatnonzero((frame["kind"] == "night").to_numpy())
    rng = np.random.default_rng(SEED + 2)
    source = np.full(frame.height, -1)
    for key in np.unique(group[night]):
        pool = np.flatnonzero(day & (group == key))
        rows = night[group[night] == key]
        if len(pool) > 0:
            source[rows] = pool[rng.integers(0, len(pool), size=len(rows))]
    night = night[source[night] >= 0]
    return night, {name: values[source] for name, values in stats.items()}


def labelled(*, period: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a period's labelled rows (experiment 184's frame) and its member responses."""
    frame = score.label_columns(frame=pl.read_parquet(E184_ARM_DIR / f"{period}_frame.parquet"))
    return frame, np.load(E184_ARM_DIR / f"{period}_delta_m.npy").astype(np.float64)
